from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

MAX_RECORDS_PER_REQUEST = 2048
MAX_RECORD_KEYS = 512


class PredictRequest(BaseModel):
    """Request body for model prediction over one or more flow records."""
    records: list[dict[str, Any]] = Field(..., min_length=1, max_length=MAX_RECORDS_PER_REQUEST)

    @field_validator("records")
    @classmethod
    def validate_records(cls, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Validate record shape, size, and basic numeric field constraints."""
        for idx, record in enumerate(records):
            if not isinstance(record, dict) or not record:
                raise ValueError(f"records[{idx}] must be a non-empty object")

            if len(record) > MAX_RECORD_KEYS:
                raise ValueError(
                    f"records[{idx}] has too many fields ({len(record)}). Max allowed is {MAX_RECORD_KEYS}."
                )

            for field_name in ("duration", "src_bytes", "dst_bytes"):
                if field_name not in record:
                    continue
                value = record[field_name]
                if value in (None, ""):
                    continue
                try:
                    parsed = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"records[{idx}].{field_name} must be numeric") from exc
                if parsed < 0:
                    raise ValueError(f"records[{idx}].{field_name} must be >= 0")

            if "proto" in record and record["proto"] not in (None, "") and not isinstance(record["proto"], str):
                raise ValueError(f"records[{idx}].proto must be a string")

        return records


class PredictionItem(BaseModel):
    """Single model prediction with probability distribution and route metadata."""
    predicted_index: int
    predicted_label: str
    confidence: float
    probabilities: dict[str, float]
    route: str | None = None
    router_confidence: float | None = None


class PredictResponse(BaseModel):
    """Response body for the prediction endpoint."""
    model_name: str
    class_names: list[str]
    predictions: list[PredictionItem]


class MetadataResponse(BaseModel):
    """Response body describing loaded model artifacts and input requirements."""
    model_name: str
    artifact_dir: str
    class_names: list[str]
    feature_count: int
    input_dim: int
    required_fields: list[str]
    feature_signature_sha256: str
    unknown_confidence_threshold: float
    routing_enabled: bool = False
    route_fields: list[str] = []


class AnalyzeRequest(PredictRequest):
    """Analyze request with optional triage context."""
    context: dict[str, Any] | None = None


class MitreTechnique(BaseModel):
    """MITRE ATT&CK technique mapped to a detection."""
    id: str
    name: str
    confidence: str
    reason: str


class TriageItem(BaseModel):
    """SOC triage result attached to a model prediction."""
    label: str
    severity: str
    mitre_tactics: list[str]
    mitre_techniques: list[MitreTechnique]
    summary: str
    next_actions: list[str]
    confidence_note: str
    source: str


class AnalyzeResponse(BaseModel):
    """Response body for inference plus SOC triage."""
    model_name: str
    class_names: list[str]
    predictions: list[PredictionItem]
    triage: list[TriageItem]
    audit_id: str
    llm_enabled: bool
    llm_error: str | None = None
