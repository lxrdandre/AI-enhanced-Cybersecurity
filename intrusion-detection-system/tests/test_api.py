"""Integration tests for FastAPI endpoints using TestClient.

These tests mock the model + pipeline so they run without GPU / artifacts.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tests.conftest import SAMPLE_ATTACK_RECORDS, SAMPLE_NORMAL_RECORD


# -- Helpers to build a fake InferenceService -----------------

CLASS_NAMES = ["normal", "backdoor", "ddos_dos", "injection", "password", "scanning", "xss"]


def _fake_predict(records):
    """Return a realistic prediction dict list without a real model."""
    results = []
    for _ in records:
        probs = np.random.dirichlet(np.ones(len(CLASS_NAMES)))
        idx = int(np.argmax(probs))
        results.append(
            {
                "predicted_index": idx,
                "predicted_label": CLASS_NAMES[idx],
                "confidence": float(probs[idx]),
                "probabilities": {n: float(p) for n, p in zip(CLASS_NAMES, probs)},
            }
        )
    return results


def _build_mock_service():
    """Build mock service."""
    svc = MagicMock()
    svc.model_name = "test_model"
    svc.class_names = CLASS_NAMES
    svc.predict.side_effect = _fake_predict
    svc.metadata.return_value = {
        "model_name": "test_model",
        "artifact_dir": "/fake/artifacts",
        "class_names": CLASS_NAMES,
        "feature_count": 42,
        "input_dim": 42,
        "required_fields": ["duration", "src_bytes", "dst_bytes", "proto"],
        "feature_signature_sha256": "deadbeef",
        "unknown_confidence_threshold": 0.3,
    }
    return svc


# -- Patch app-level singletons before importing the TestClient --


@pytest.fixture()
def client():
    """Provide a FastAPI test client with mocked services."""
    mock_service = _build_mock_service()

    with (
        patch("app.main.inference_service", mock_service),
        patch("app.main.startup_error", None),
    ):
        from fastapi.testclient import TestClient

        from app.main import app

        yield TestClient(app)


# -- Tests ----------------------------------------------------


class TestRootEndpoint:
    """Group tests covering root endpoint behavior."""
    def test_root_returns_routes(self, client):
        """Verify that root returns routes."""
        resp = client.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert "routes" in body


class TestHealthEndpoint:
    """Group tests covering health endpoint behavior."""
    def test_health_ok(self, client):
        """Verify that health ok."""
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestMetadataEndpoint:
    """Group tests covering metadata endpoint behavior."""
    def test_metadata_fields(self, client):
        """Verify that metadata fields."""
        resp = client.get("/metadata")
        assert resp.status_code == 200
        body = resp.json()
        for key in ("model_name", "class_names", "feature_count", "input_dim", "required_fields"):
            assert key in body


class TestPredictEndpoint:
    """Group tests covering predict endpoint behavior."""
    def test_single_record(self, client):
        """Verify that single record."""
        resp = client.post("/predict", json={"records": [SAMPLE_NORMAL_RECORD]})
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["predictions"]) == 1
        pred = body["predictions"][0]
        assert "predicted_label" in pred
        assert "confidence" in pred
        assert pred["predicted_label"] in CLASS_NAMES

    def test_batch_records(self, client, batch_records):
        """Verify that batch records."""
        resp = client.post("/predict", json={"records": batch_records})
        assert resp.status_code == 200
        assert len(resp.json()["predictions"]) == 10

    def test_empty_records_rejected(self, client):
        """Verify that empty records rejected."""
        resp = client.post("/predict", json={"records": []})
        assert resp.status_code == 422

    def test_invalid_payload_rejected(self, client):
        """Verify that invalid payload rejected."""
        resp = client.post("/predict", json={"records": "not_a_list"})
        assert resp.status_code == 422


class TestAnalyzeEndpoint:
    """Group tests covering analyze endpoint behavior."""
    def test_analyze_returns_triage(self, client):
        """Verify that analyze returns triage."""
        payload = {"records": [SAMPLE_NORMAL_RECORD]}
        resp = client.post("/analyze", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert "triage" in body
        assert "audit_id" in body
        assert isinstance(body["triage"], list)

    def test_analyze_with_context(self, client):
        """Verify that analyze with context."""
        payload = {
            "records": [SAMPLE_NORMAL_RECORD],
            "context": {"source": "clawdbot", "incident_id": "test-001"},
        }
        resp = client.post("/analyze", json=payload)
        assert resp.status_code == 200

    def test_analyze_attack_records(self, client):
        """Verify that analyze attack records."""
        payload = {"records": SAMPLE_ATTACK_RECORDS}
        resp = client.post("/analyze", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["predictions"]) == 2
        assert len(body["triage"]) == 2
