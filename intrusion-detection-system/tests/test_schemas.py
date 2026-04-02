"""Unit tests for pydantic request/response schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    MetadataResponse,
    PredictRequest,
    PredictResponse,
    PredictionItem,
    TriageItem,
)


# ── PredictRequest ──────────────────────────────────────────


class TestPredictRequest:
    def test_valid_single_record(self, normal_record):
        req = PredictRequest(records=[normal_record])
        assert len(req.records) == 1

    def test_valid_batch(self, batch_records):
        req = PredictRequest(records=batch_records)
        assert len(req.records) == 10

    def test_empty_records_rejected(self):
        with pytest.raises(ValidationError):
            PredictRequest(records=[])

    def test_non_dict_record_rejected(self):
        with pytest.raises(ValidationError):
            PredictRequest(records=["not_a_dict"])

    def test_empty_dict_record_rejected(self):
        with pytest.raises(ValidationError):
            PredictRequest(records=[{}])

    def test_negative_duration_rejected(self):
        with pytest.raises(ValidationError):
            PredictRequest(records=[{"duration": -1, "src_bytes": 0, "dst_bytes": 0, "proto": "tcp"}])

    def test_negative_src_bytes_rejected(self):
        with pytest.raises(ValidationError):
            PredictRequest(records=[{"duration": 0, "src_bytes": -5, "dst_bytes": 0, "proto": "tcp"}])

    def test_non_numeric_duration_rejected(self):
        with pytest.raises(ValidationError):
            PredictRequest(records=[{"duration": "abc", "src_bytes": 0, "dst_bytes": 0, "proto": "tcp"}])

    def test_proto_must_be_string(self):
        with pytest.raises(ValidationError):
            PredictRequest(records=[{"duration": 0, "src_bytes": 0, "dst_bytes": 0, "proto": 123}])

    def test_too_many_keys_rejected(self):
        big_record = {f"col_{i}": i for i in range(600)}
        with pytest.raises(ValidationError):
            PredictRequest(records=[big_record])

    def test_extra_fields_allowed(self, normal_record):
        normal_record["extra_col"] = 42
        req = PredictRequest(records=[normal_record])
        assert req.records[0]["extra_col"] == 42


# ── AnalyzeRequest ──────────────────────────────────────────


class TestAnalyzeRequest:
    def test_inherits_predict_validation(self, normal_record):
        req = AnalyzeRequest(records=[normal_record])
        assert req.context is None

    def test_accepts_context(self, normal_record):
        req = AnalyzeRequest(records=[normal_record], context={"source": "clawdbot"})
        assert req.context["source"] == "clawdbot"


# ── Response models (round-trip) ────────────────────────────


class TestResponseModels:
    def test_prediction_item(self):
        item = PredictionItem(
            predicted_index=0,
            predicted_label="normal",
            confidence=0.95,
            probabilities={"normal": 0.95, "ddos_dos": 0.05},
        )
        assert item.predicted_label == "normal"

    def test_predict_response(self):
        resp = PredictResponse(
            model_name="resnet_base",
            class_names=["normal", "ddos_dos"],
            predictions=[
                PredictionItem(
                    predicted_index=0,
                    predicted_label="normal",
                    confidence=0.99,
                    probabilities={"normal": 0.99, "ddos_dos": 0.01},
                )
            ],
        )
        assert resp.predictions[0].confidence == 0.99

    def test_metadata_response(self):
        resp = MetadataResponse(
            model_name="resnet_base",
            artifact_dir="/artifacts/resnet_base",
            class_names=["normal", "ddos_dos"],
            feature_count=10,
            input_dim=10,
            required_fields=["duration", "src_bytes", "dst_bytes", "proto"],
            feature_signature_sha256="abc123",
        )
        assert resp.feature_count == 10

    def test_triage_item(self):
        item = TriageItem(
            label="ddos_dos",
            severity="high",
            mitre_tactics=["Impact"],
            mitre_techniques=[
                {"id": "T1498", "name": "Network Denial of Service", "confidence": "high", "reason": "Heuristic"}
            ],
            summary="DDoS detected",
            next_actions=["Block source IP"],
            confidence_note="High confidence",
            source="fallback",
        )
        assert item.severity == "high"

    def test_analyze_response(self):
        resp = AnalyzeResponse(
            model_name="resnet_base",
            class_names=["normal", "ddos_dos"],
            predictions=[
                PredictionItem(
                    predicted_index=1,
                    predicted_label="ddos_dos",
                    confidence=0.92,
                    probabilities={"normal": 0.08, "ddos_dos": 0.92},
                )
            ],
            triage=[
                TriageItem(
                    label="ddos_dos",
                    severity="high",
                    mitre_tactics=["Impact"],
                    mitre_techniques=[],
                    summary="DDoS detected",
                    next_actions=[],
                    confidence_note="",
                    source="fallback",
                )
            ],
            audit_id="abc123",
            llm_enabled=False,
            llm_error=None,
        )
        assert resp.llm_enabled is False
