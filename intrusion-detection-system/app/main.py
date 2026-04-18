from __future__ import annotations

import logging
import os

from fastapi import BackgroundTasks, FastAPI, HTTPException

from app.audit import AuditLogger
from app.config import Settings
from app.model_registry import load_artifacts, load_domain_router, load_model_file
from app.schemas import AnalyzeRequest, AnalyzeResponse, MetadataResponse, PredictRequest, PredictResponse
from app.service import InferenceService
from app.triage import TriageService, _default_triage_item
from clawdbot.threat_intel import ThreatCache

log = logging.getLogger(__name__)

settings = Settings.from_env()

try:
    model, pipeline, final_features, calibration = load_artifacts(
        artifact_dir=settings.artifact_dir,
        model_filename=settings.model_filename,
        pipeline_filename=settings.pipeline_filename,
        features_filename=settings.features_filename,
        calibration_filename=settings.calibration_filename,
    )
    domain_router = load_domain_router(artifact_dir=settings.artifact_dir)
    original_model = load_model_file(
        os.path.join(settings.base_artifact_dir, settings.base_model_filename)
    )
    inference_service = InferenceService(
        model=model,
        original_model=original_model,
        pipeline=pipeline,
        final_features=final_features,
        artifact_dir=settings.artifact_dir,
        calibration=calibration,
        domain_router=domain_router,
        unknown_confidence_threshold=settings.unknown_confidence_threshold,
    )
except Exception as exc:
    inference_service = None
    startup_error = str(exc)
else:
    startup_error = None

audit_logger = AuditLogger(log_path=settings.audit_log_path)

# Open STIX cache for read-only MITRE lookups in prompts (shared DB with agent)
threat_cache = ThreatCache(db_path=settings.threat_cache_db)

triage_service = TriageService(
    api_key=settings.gemini_api_key,
    model=settings.triage_model,
    timeout_seconds=settings.triage_timeout_seconds,
    ollama_base_url=settings.ollama_base_url,
    ollama_model_tier1=settings.ollama_model_tier1,
    ollama_model_tier2=settings.ollama_model_tier2,
    ollama_escalation_confidence=settings.ollama_escalation_confidence,
    triage_backend=settings.triage_backend,
    threat_cache=threat_cache,
)

app = FastAPI(title="TON IoT IDS Inference API", version="0.1.0")


@app.get("/")
def root() -> dict:
    return {
        "service": "TON IoT IDS Inference API",
        "routes": ["/health", "/metadata", "/predict", "/analyze", "/docs"],
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if inference_service is not None else "error",
        "artifact_dir": settings.artifact_dir,
        "startup_error": startup_error,
    }


@app.get("/metadata", response_model=MetadataResponse)
def metadata() -> MetadataResponse:
    if inference_service is None:
        raise HTTPException(status_code=500, detail=f"Model not loaded: {startup_error}")

    return MetadataResponse(**inference_service.metadata())


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    if inference_service is None:
        raise HTTPException(status_code=500, detail=f"Model not loaded: {startup_error}")

    try:
        predictions = inference_service.predict(payload.records)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    return PredictResponse(
        model_name=inference_service.model_name,
        class_names=inference_service.class_names,
        predictions=predictions,
    )


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest, background_tasks: BackgroundTasks) -> AnalyzeResponse:
    if inference_service is None:
        raise HTTPException(status_code=500, detail=f"Model not loaded: {startup_error}")

    try:
        predictions = inference_service.predict(payload.records)

        # Return FAST fallback triage immediately (has MITRE mappings already)
        fallback_triage = [_default_triage_item(p) for p in predictions]

        audit_id = audit_logger.log_analyze(
            model_name=inference_service.model_name,
            records=payload.records,
            predictions=predictions,
            triage=fallback_triage,
            llm_enabled=settings.background_triage_enabled and triage_service.enabled,
            llm_error=None,
        )

        # Background LLM enrichment is opt-in; ClawdBot handles incident-level
        # unknown classification to avoid per-flow Ollama storms.
        if settings.background_triage_enabled and triage_service.enabled:
            background_tasks.add_task(
                _background_triage,
                predictions=predictions,
                records=payload.records,
                context=payload.context,
                audit_id=audit_id,
            )

    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Analyze failed: {exc}") from exc

    return AnalyzeResponse(
        model_name=inference_service.model_name,
        class_names=inference_service.class_names,
        predictions=predictions,
        triage=fallback_triage,
        audit_id=audit_id,
        llm_enabled=settings.background_triage_enabled and triage_service.enabled,
        llm_error=None,
    )


def _background_triage(
    *, predictions: list[dict], records: list[dict],
    context: dict | None, audit_id: str,
) -> None:
    """Run Ollama triage in background and log enriched results."""
    try:
        triage, llm_error = triage_service.triage_predictions(
            predictions=predictions, records=records, context=context,
        )
        audit_logger.log_analyze(
            model_name=inference_service.model_name,
            records=records,
            predictions=predictions,
            triage=triage,
            llm_enabled=True,
            llm_error=llm_error,
        )
        non_fallback = sum(1 for t in triage if t.get("source", "").startswith("ollama"))
        log.info("Background triage done for audit_id=%s: %d LLM-enriched", audit_id, non_fallback)
    except Exception as exc:
        log.error("Background triage failed for audit_id=%s: %s", audit_id, exc)
