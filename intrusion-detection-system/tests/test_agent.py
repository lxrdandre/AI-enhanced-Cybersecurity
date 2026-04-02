"""Tests for clawdbot.agent — orchestrator logic."""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from clawdbot.agent import EventLogger, _post_analyze


# ── _post_analyze ─────────────────────────────────────────

class TestPostAnalyze:
    @patch("clawdbot.agent.urllib.request.urlopen")
    def test_success(self, mock_urlopen):
        api_response = {
            "model_name": "resnet",
            "class_names": ["normal", "ddos_dos"],
            "predictions": [{"predicted_label": "ddos_dos", "confidence": 0.9}],
            "triage": [{"severity": "high", "label": "ddos_dos"}],
        }
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = json.dumps(api_response).encode()
        mock_urlopen.return_value = mock_resp

        records = [{"duration": 1, "src_bytes": 100, "dst_bytes": 50, "proto": "tcp", "_meta": {"src_ip": "1.1.1.1"}}]
        result = _post_analyze("http://localhost:8000", records)

        assert result is not None
        assert result["predictions"][0]["predicted_label"] == "ddos_dos"

        # Verify _meta was stripped from the POST body
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        sent_body = json.loads(req.data.decode())
        assert "_meta" not in sent_body["records"][0]

    @patch("clawdbot.agent.urllib.request.urlopen")
    def test_api_failure_returns_none(self, mock_urlopen):
        mock_urlopen.side_effect = TimeoutError("timeout")
        result = _post_analyze("http://localhost:8000", [{"duration": 1}])
        assert result is None

    @patch("clawdbot.agent.urllib.request.urlopen")
    def test_trailing_slash_handled(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = json.dumps({"predictions": [], "triage": []}).encode()
        mock_urlopen.return_value = mock_resp

        _post_analyze("http://localhost:8000/", [{"duration": 1}])
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "http://localhost:8000/analyze"


# ── EventLogger ───────────────────────────────────────────

class TestEventLogger:
    def test_creates_log_dir(self, tmp_path):
        log_dir = tmp_path / "sub" / "logs"
        EventLogger(str(log_dir))
        assert log_dir.is_dir()

    def test_log_attack_writes_jsonl(self, tmp_path):
        el = EventLogger(str(tmp_path))
        el.log_attack(
            prediction={"predicted_label": "ddos_dos", "confidence": 0.92, "probabilities": {}},
            triage={"severity": "high", "mitre_tactics": ["Impact"], "mitre_techniques": [],
                    "summary": "DDoS", "next_actions": ["block"], "source": "tier1"},
            flow_meta={"src_ip": "1.1.1.1", "dst_ip": "2.2.2.2", "src_port": 80, "dst_port": 443},
            telegram_sent=True,
            audit_id="abc123",
        )
        attacks = (tmp_path / "attacks.jsonl").read_text().strip().split("\n")
        assert len(attacks) == 1
        event = json.loads(attacks[0])
        assert event["event"] == "attack_detected"
        assert event["prediction"]["label"] == "ddos_dos"
        assert event["prediction"]["confidence"] == 0.92
        assert event["triage"]["severity"] == "high"
        assert event["triage"]["mitre_tactics"] == ["Impact"]
        assert event["flow"]["src_ip"] == "1.1.1.1"
        assert event["actions"]["telegram_sent"] is True
        assert event["audit_id"] == "abc123"
        assert "ts" in event
        assert "epoch" in event

    def test_log_attack_appends(self, tmp_path):
        el = EventLogger(str(tmp_path))
        for _ in range(3):
            el.log_attack(
                prediction={"predicted_label": "scanning", "confidence": 0.8},
                triage={"severity": "medium"},
                flow_meta=None,
                telegram_sent=False,
            )
        lines = (tmp_path / "attacks.jsonl").read_text().strip().split("\n")
        assert len(lines) == 3

    def test_log_action_writes_to_actions_file(self, tmp_path):
        el = EventLogger(str(tmp_path))
        el.log_action(action="agent_start", detail={"interface": "eth0"})
        el.log_action(action="agent_stop")

        lines = (tmp_path / "actions.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2
        start = json.loads(lines[0])
        assert start["event"] == "agent_start"
        assert start["detail"]["interface"] == "eth0"
        stop = json.loads(lines[1])
        assert stop["event"] == "agent_stop"

    def test_log_attack_none_flow_meta(self, tmp_path):
        el = EventLogger(str(tmp_path))
        el.log_attack(
            prediction={"predicted_label": "xss", "confidence": 0.7},
            triage={"severity": "high"},
            flow_meta=None,
            telegram_sent=False,
        )
        event = json.loads((tmp_path / "attacks.jsonl").read_text().strip())
        assert event["flow"] is None
