import os
from dataclasses import dataclass


def _detect_project_root() -> str:
    """Detect the repository root used for default artifact paths."""
    env_root = os.environ.get("TON_IOT_PROJECT_ROOT")
    if env_root and os.path.isdir(env_root):
        return os.path.abspath(env_root)

    cwd = os.path.abspath(os.getcwd())
    if os.path.isdir(os.path.join(cwd, "data")) and os.path.isdir(os.path.join(cwd, "artifacts")):
        return cwd

    return cwd


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables."""
    project_root: str
    artifact_dir: str
    base_artifact_dir: str
    model_filename: str
    base_model_filename: str
    pipeline_filename: str
    features_filename: str
    calibration_filename: str
    audit_log_path: str
    triage_model: str
    triage_timeout_seconds: int
    gemini_api_key: str | None
    host: str
    port: int
    # Ollama (local LLM) settings
    ollama_base_url: str
    ollama_model_tier1: str
    ollama_model_tier2: str
    ollama_escalation_confidence: float
    triage_backend: str  # "ollama" | "gemini" | "fallback"
    background_triage_enabled: bool
    unknown_confidence_threshold: float  # predictions below this are labeled "unknown"
    threat_cache_db: str  # path to SQLite threat intel cache (STIX lookups)

    @staticmethod
    def from_env() -> "Settings":
        """Build settings from environment variables and defaults."""
        project_root = _detect_project_root()
        default_artifact_dir = os.path.join(project_root, "artifacts", "resnet_cic_public")
        default_base_artifact_dir = default_artifact_dir
        default_audit_path = os.path.join(project_root, "artifacts", "audit", "analyze_events.jsonl")
        default_threat_db = os.path.join(project_root, "data", "threat_cache.db")

        return Settings(
            project_root=project_root,
            artifact_dir=os.environ.get("TON_IOT_ARTIFACT_DIR", default_artifact_dir),
            base_artifact_dir=os.environ.get("TON_IOT_BASE_ARTIFACT_DIR", default_base_artifact_dir),
            model_filename=os.environ.get("TON_IOT_MODEL_FILENAME", "resnet_model.keras"),
            base_model_filename=os.environ.get("TON_IOT_BASE_MODEL_FILENAME", "resnet_model.keras"),
            pipeline_filename=os.environ.get("TON_IOT_PIPELINE_FILENAME", "preprocessing_pipeline.pkl"),
            features_filename=os.environ.get("TON_IOT_FEATURES_FILENAME", "final_features.txt"),
            calibration_filename=os.environ.get("TON_IOT_CALIBRATION_FILENAME", "calibration.pkl"),
            audit_log_path=os.environ.get("TON_IOT_AUDIT_LOG_PATH", default_audit_path),
            triage_model=os.environ.get("TON_IOT_TRIAGE_MODEL", "gemini-2.0-flash"),
            triage_timeout_seconds=int(os.environ.get("TON_IOT_TRIAGE_TIMEOUT_SECONDS", "30")),
            gemini_api_key=os.environ.get("GEMINI_API_KEY"),
            host=os.environ.get("TON_IOT_API_HOST", "0.0.0.0"),
            port=int(os.environ.get("TON_IOT_API_PORT", "8000")),
            ollama_base_url=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            ollama_model_tier1=os.environ.get("OLLAMA_MODEL_TIER1", "clawdbot-triage"),
            ollama_model_tier2=os.environ.get("OLLAMA_MODEL_TIER2", "llama3.1:70b-instruct-q8_0"),
            ollama_escalation_confidence=float(os.environ.get("OLLAMA_ESCALATION_CONFIDENCE", "0.75")),
            triage_backend=os.environ.get("TON_IOT_TRIAGE_BACKEND", "ollama"),
            background_triage_enabled=os.environ.get("TON_IOT_BACKGROUND_TRIAGE", "0").lower() in {"1", "true", "yes"},
            unknown_confidence_threshold=float(os.environ.get("TON_IOT_UNKNOWN_THRESHOLD", "0.45")),
            threat_cache_db=os.environ.get("THREAT_CACHE_DB", default_threat_db),
        )
