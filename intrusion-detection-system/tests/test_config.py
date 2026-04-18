"""Unit tests for config module."""

from __future__ import annotations

import os

from app.config import Settings, _detect_project_root


class TestSettings:
    def test_from_env_defaults(self, monkeypatch):
        monkeypatch.delenv("TON_IOT_PROJECT_ROOT", raising=False)
        monkeypatch.delenv("TON_IOT_ARTIFACT_DIR", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("TON_IOT_TRIAGE_BACKEND", raising=False)
        monkeypatch.delenv("TON_IOT_BACKGROUND_TRIAGE", raising=False)
        settings = Settings.from_env()
        assert settings.model_filename == "resnet_transfer_model_7class.keras"
        assert settings.pipeline_filename == "preprocessing_pipeline.pkl"
        assert settings.features_filename == "final_features.txt"
        assert settings.host == "0.0.0.0"
        assert settings.port == 8000
        assert settings.triage_timeout_seconds == 30
        assert settings.ollama_base_url == "http://127.0.0.1:11434"
        assert settings.ollama_model_tier1 == "clawdbot-triage"
        assert settings.ollama_model_tier2 == "llama3.1:70b-instruct-q8_0"
        assert settings.ollama_escalation_confidence == 0.75
        assert settings.triage_backend == "ollama"
        assert settings.background_triage_enabled is False
        assert settings.unknown_confidence_threshold == 0.45

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("TON_IOT_API_PORT", "9999")
        monkeypatch.setenv("TON_IOT_API_HOST", "127.0.0.1")
        monkeypatch.setenv("GEMINI_API_KEY", "test-key-123")
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://10.0.0.5:11434")
        monkeypatch.setenv("OLLAMA_MODEL_TIER1", "qwen2.5:32b")
        monkeypatch.setenv("TON_IOT_TRIAGE_BACKEND", "gemini")
        monkeypatch.setenv("TON_IOT_BACKGROUND_TRIAGE", "true")
        settings = Settings.from_env()
        assert settings.port == 9999
        assert settings.host == "127.0.0.1"
        assert settings.gemini_api_key == "test-key-123"
        assert settings.ollama_base_url == "http://10.0.0.5:11434"
        assert settings.ollama_model_tier1 == "qwen2.5:32b"
        assert settings.triage_backend == "gemini"
        assert settings.background_triage_enabled is True

    def test_gemini_key_none_by_default(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        settings = Settings.from_env()
        assert settings.gemini_api_key is None
