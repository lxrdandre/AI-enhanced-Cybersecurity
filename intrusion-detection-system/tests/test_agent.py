"""Tests for clawdbot.agent — orchestrator logic."""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from clawdbot.actuator import _parse_whitelist
from clawdbot.agent import (
    EventLogger,
    _apply_incident_rules,
    _classify_primary_unknown_incident,
    _incident_summary,
    _is_management_record,
    _parse_mgmt_ports,
    _post_analyze,
)


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


class TestManagementFiltering:
    def test_parse_mgmt_ports_keeps_dashboard_defaults(self):
        ports = _parse_mgmt_ports("22,64295")
        assert {22, 64295, 5000, 8000}.issubset(ports)

    def test_dashboard_flow_between_whitelisted_peers_is_management(self):
        whitelist = _parse_whitelist("100.111.76.168,100.111.77.70")
        record = {
            "_meta": {
                "src_ip": "100.111.76.168",
                "dst_ip": "100.111.77.70",
                "src_port": 49911,
                "dst_port": 5000,
            }
        }
        assert _is_management_record(record, whitelist, _parse_mgmt_ports(""))

    def test_non_management_port_is_not_filtered(self):
        whitelist = _parse_whitelist("100.111.76.168,100.111.77.70")
        record = {
            "_meta": {
                "src_ip": "100.111.76.168",
                "dst_ip": "100.111.77.70",
                "src_port": 49911,
                "dst_port": 8081,
            }
        }
        assert not _is_management_record(record, whitelist, _parse_mgmt_ports(""))


# ── EventLogger ───────────────────────────────────────────

class TestEventLogger:
    def test_creates_log_dir(self, tmp_path):
        log_dir = tmp_path / "sub" / "logs"
        EventLogger(str(log_dir))
        assert log_dir.is_dir()

    def test_log_attack_writes_jsonl(self, tmp_path):
        el = EventLogger(str(tmp_path))
        el.log_attack(
            prediction={
                "predicted_label": "ddos_dos",
                "confidence": 0.92,
                "probabilities": {},
                "route": "original",
                "router_confidence": 0.81,
            },
            triage={"severity": "high", "mitre_tactics": ["Impact"], "mitre_techniques": [],
                    "summary": "DDoS", "next_actions": ["block"], "source": "tier1",
                    "label": "ddos_dos", "incident_role": "primary"},
            flow_meta={"src_ip": "1.1.1.1", "dst_ip": "2.2.2.2", "src_port": 80, "dst_port": 443},
            telegram_sent=True,
            audit_id="abc123",
            incident_id="incident-1",
            incident_summary={"threat_count": 1, "possible_count": 2},
            block_result={"ip": "1.1.1.1", "applied": True, "ttl": 3600},
            reputation={"badge": "Known-bad", "hit_count": 2},
        )
        attacks = (tmp_path / "attacks.jsonl").read_text().strip().split("\n")
        assert len(attacks) == 1
        event = json.loads(attacks[0])
        assert event["event"] == "attack_detected"
        assert event["prediction"]["label"] == "ddos_dos"
        assert event["prediction"]["confidence"] == 0.92
        assert event["prediction"]["route"] == "original"
        assert event["prediction"]["router_confidence"] == 0.81
        assert event["triage"]["label"] == "ddos_dos"
        assert event["triage"]["severity"] == "high"
        assert event["triage"]["incident_role"] == "primary"
        assert event["triage"]["mitre_tactics"] == ["Impact"]
        assert event["flow"]["src_ip"] == "1.1.1.1"
        assert event["actions"]["telegram_sent"] is True
        assert event["actions"]["block_result"]["applied"] is True
        assert event["reputation"]["badge"] == "Known-bad"
        assert event["audit_id"] == "abc123"
        assert event["incident_id"] == "incident-1"
        assert event["incident_summary"]["possible_count"] == 2
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


def test_incident_summary_counts_primary_and_secondary_labels():
    detections = [
        {"prediction": {"predicted_label": "scanning"}, "triage": {"label": "scanning", "incident_role": "primary", "incident_primary_label": "scanning"}},
        {"prediction": {"predicted_label": "scanning"}, "triage": {"label": "scanning", "incident_role": "primary", "incident_primary_label": "scanning"}},
        {"prediction": {"predicted_label": "unknown"}, "triage": {"label": "unknown", "incident_role": "secondary", "incident_primary_label": "scanning"}},
        {"prediction": {"predicted_label": "password"}, "triage": {"label": "password", "incident_role": "secondary", "incident_primary_label": "scanning"}},
        {"prediction": {"predicted_label": "password"}, "triage": {"label": "password", "incident_role": "secondary", "incident_primary_label": "scanning"}},
    ]

    summary = _incident_summary(detections)

    assert summary["threat_count"] == 1
    assert summary["primary_label"] == "scanning"
    assert summary["possible_count"] == 2
    assert summary["secondary_labels"] == ["password", "unknown"]


class DummyTriageService:
    enabled = True

    def __init__(self, label: str = "scanning"):
        self.label = label
        self.calls = []

    def triage_predictions(self, *, predictions, records, context):
        self.calls.append({"predictions": predictions, "records": records, "context": context})
        return [{
            "label": self.label,
            "severity": "high",
            "mitre_tactics": ["Reconnaissance"],
            "mitre_techniques": [{"id": "T1595", "name": "Active Scanning"}],
            "summary": "Incident-level unknown classification.",
            "next_actions": ["Review the incident as scanning."],
            "source": "ollama:test",
        }], None


def test_primary_unknown_incident_classified_once():
    detections = [
        {
            "prediction": {"predicted_label": "unknown", "confidence": 0.41},
            "triage": {"label": "unknown", "incident_role": "primary", "incident_primary_label": "unknown"},
            "flow_meta": {"src_ip": "1.1.1.1", "dst_ip": "2.2.2.2", "dst_port": 8080, "proto": "tcp"},
        },
        {
            "prediction": {"predicted_label": "unknown", "confidence": 0.39},
            "triage": {"label": "unknown", "incident_role": "primary", "incident_primary_label": "unknown"},
            "flow_meta": {"src_ip": "1.1.1.1", "dst_ip": "2.2.2.3", "dst_port": 8081, "proto": "tcp"},
        },
    ]
    svc = DummyTriageService(label="scanning")

    _classify_primary_unknown_incident(detections, triage_service=svc)

    assert len(svc.calls) == 1
    assert svc.calls[0]["context"]["classification_scope"] == "incident"
    assert svc.calls[0]["context"]["unknown_priority"] == "primary"
    assert svc.calls[0]["records"][0]["incident_flow_count"] == 2
    assert all(det["triage"]["label"] == "scanning" for det in detections)
    assert all(det["triage"]["incident_primary_label"] == "scanning" for det in detections)
    assert all(det["triage"]["llm_reclassified"] is True for det in detections)


def test_secondary_unknown_incident_not_sent_to_llm():
    detections = [
        {
            "prediction": {"predicted_label": "scanning", "confidence": 0.91},
            "triage": {"label": "scanning", "incident_role": "primary", "incident_primary_label": "scanning"},
            "flow_meta": {"src_ip": "1.1.1.1", "dst_ip": "2.2.2.2", "dst_port": 80, "proto": "tcp"},
        },
        {
            "prediction": {"predicted_label": "unknown", "confidence": 0.32},
            "triage": {"label": "unknown", "incident_role": "secondary", "incident_primary_label": "scanning"},
            "flow_meta": {"src_ip": "1.1.1.1", "dst_ip": "2.2.2.2", "dst_port": 81, "proto": "tcp"},
        },
    ]
    svc = DummyTriageService(label="password")

    _classify_primary_unknown_incident(detections, triage_service=svc)

    assert svc.calls == []
    assert detections[1]["triage"]["label"] == "unknown"


class TestIncidentRules:
    def test_normal_fanout_promotes_to_scanning(self):
        detections = []
        for port in range(20, 35):
            detections.append({
                "prediction": {"predicted_label": "normal", "confidence": 0.98},
                "triage": {
                    "label": "normal",
                    "severity": "low",
                    "mitre_tactics": [],
                    "mitre_techniques": [],
                },
                "flow_meta": {"src_ip": "9.9.9.9", "dst_ip": "10.0.0.5", "dst_port": port},
            })

        out = _apply_incident_rules(detections)

        assert all(d["triage"]["label"] == "scanning" for d in out)
        assert all(d["triage"]["incident_role"] == "primary" for d in out)
        assert all(d["triage"]["source"] == "heuristic:scan-fanout" for d in out)
        assert any(t["id"] == "T1595" for t in out[0]["triage"]["mitre_techniques"])

    def test_scan_campaign_suppresses_secondary_labels(self):
        detections = []
        for port in range(20, 35):
            detections.append({
                "prediction": {"predicted_label": "scanning", "confidence": 0.95},
                "triage": {
                    "label": "scanning",
                    "severity": "high",
                    "mitre_tactics": ["Reconnaissance"],
                    "mitre_techniques": [{"id": "T1595", "name": "Active Scanning"}],
                },
                "flow_meta": {"src_ip": "9.9.9.9", "dst_ip": "10.0.0.5", "dst_port": port},
            })
        for label, tid in (("password", "T1110"), ("ddos_dos", "T1498"), ("unknown", "T0000")):
            detections.append({
                "prediction": {"predicted_label": label, "confidence": 0.95},
                "triage": {
                    "label": label,
                    "severity": "high",
                    "mitre_tactics": ["x"],
                    "mitre_techniques": [{"id": tid, "name": "x"}],
                },
                "flow_meta": {"src_ip": "9.9.9.9", "dst_ip": "10.0.0.5", "dst_port": 22},
            })

        out = _apply_incident_rules(detections)

        scanning = [d for d in out if d["triage"]["label"] == "scanning"]
        secondary = [d for d in out if d["triage"]["label"] != "scanning"]
        assert all(d["triage"]["incident_role"] == "primary" for d in scanning)
        assert all(d["triage"]["incident_role"] == "secondary" for d in secondary)
        assert all(d["triage"]["mitre_techniques"] == [] for d in secondary)
        assert all(d["triage"]["severity"] == "review" for d in secondary)
