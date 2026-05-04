"""Unit tests for the audit logger."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from app.audit import AuditLogger, _records_hash


class TestRecordsHash:
    """Group tests covering records hash behavior."""
    def test_deterministic(self):
        """Verify that deterministic."""
        records = [{"a": 1}, {"b": 2}]
        assert _records_hash(records) == _records_hash(records)

    def test_different_records_different_hash(self):
        """Verify that different records different hash."""
        assert _records_hash([{"a": 1}]) != _records_hash([{"a": 2}])

    def test_order_matters(self):
        """Verify that order matters."""
        assert _records_hash([{"a": 1}, {"b": 2}]) != _records_hash([{"b": 2}, {"a": 1}])


class TestAuditLogger:
    """Group tests covering audit logger behavior."""
    @pytest.fixture()
    def tmp_log(self, tmp_path):
        """Provide a temporary audit log path."""
        return str(tmp_path / "audit" / "test_events.jsonl")

    def test_creates_log_directory(self, tmp_log):
        """Verify that creates log directory."""
        AuditLogger(log_path=tmp_log)
        assert os.path.isdir(os.path.dirname(tmp_log))

    def test_log_analyze_writes_line(self, tmp_log):
        """Verify that log analyze writes line."""
        logger = AuditLogger(log_path=tmp_log)
        audit_id = logger.log_analyze(
            model_name="test_model",
            records=[{"duration": 1}],
            predictions=[{"predicted_label": "normal", "confidence": 0.9}],
            triage=[{"label": "normal", "severity": "low"}],
            llm_enabled=False,
            llm_error=None,
        )
        assert isinstance(audit_id, str) and len(audit_id) == 32

        with open(tmp_log) as f:
            lines = f.readlines()
        assert len(lines) == 1

        row = json.loads(lines[0])
        assert row["audit_id"] == audit_id
        assert row["model_name"] == "test_model"
        assert row["record_count"] == 1
        assert row["llm_enabled"] is False

    def test_multiple_logs_append(self, tmp_log):
        """Verify that multiple logs append."""
        logger = AuditLogger(log_path=tmp_log)
        for _ in range(5):
            logger.log_analyze(
                model_name="m",
                records=[{}],
                predictions=[{}],
                triage=[{}],
                llm_enabled=False,
                llm_error=None,
            )
        with open(tmp_log) as f:
            assert len(f.readlines()) == 5
