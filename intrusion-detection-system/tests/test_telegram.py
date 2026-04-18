"""Tests for clawdbot.telegram — TelegramNotifier."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from clawdbot.telegram import SEVERITY_ORDER, TelegramNotifier, _format_block_result


# ── Fixtures ──────────────────────────────────────────────

@pytest.fixture()
def notifier():
    return TelegramNotifier(
        bot_token="123:FAKE",
        chat_id="-100999",
        severity_threshold="medium",
    )


@pytest.fixture()
def sample_prediction():
    return {
        "predicted_index": 2,
        "predicted_label": "ddos_dos",
        "confidence": 0.92,
        "probabilities": {"normal": 0.05, "ddos_dos": 0.92, "scanning": 0.03},
    }


@pytest.fixture()
def sample_triage():
    return {
        "label": "ddos_dos",
        "severity": "high",
        "mitre_tactics": ["Impact"],
        "mitre_techniques": [
            {"id": "T1498", "name": "Network Denial of Service", "confidence": "high", "reason": "flood"}
        ],
        "summary": "DDoS traffic detected",
        "next_actions": ["Block source IP", "Check upstream"],
        "confidence_note": "High confidence",
        "source": "ollama-tier1",
    }


@pytest.fixture()
def sample_meta():
    return {"src_ip": "192.168.1.100", "dst_ip": "10.0.0.1", "src_port": 54321, "dst_port": 443}


# ── should_alert / severity filtering ────────────────────

class TestShouldAlert:
    def test_above_threshold(self, notifier):
        assert notifier.should_alert("high") is True

    def test_at_threshold(self, notifier):
        assert notifier.should_alert("medium") is True

    def test_below_threshold(self, notifier):
        assert notifier.should_alert("low") is False

    def test_critical_always_alerts(self, notifier):
        assert notifier.should_alert("critical") is True

    def test_unknown_severity_below_threshold(self, notifier):
        assert notifier.should_alert("unknown") is False

    def test_all_severities_ordered(self):
        levels = sorted(SEVERITY_ORDER.items(), key=lambda x: x[1])
        names = [name for name, _ in levels]
        assert names == ["low", "medium", "review", "high", "critical"]


# ── enabled property ─────────────────────────────────────

class TestEnabled:
    def test_enabled_when_configured(self, notifier):
        assert notifier.enabled is True

    def test_disabled_without_token(self):
        n = TelegramNotifier(bot_token="", chat_id="-100999")
        assert n.enabled is False

    def test_disabled_without_chat_id(self):
        n = TelegramNotifier(bot_token="123:FAKE", chat_id="")
        assert n.enabled is False


# ── format_alert ──────────────────────────────────────────

class TestFormatAlert:
    def test_contains_severity(self, notifier, sample_prediction, sample_triage):
        msg = notifier.format_alert(sample_prediction, sample_triage)
        assert "HIGH" in msg

    def test_contains_mitre(self, notifier, sample_prediction, sample_triage):
        msg = notifier.format_alert(sample_prediction, sample_triage)
        assert "T1498" in msg
        assert "Impact" in msg

    def test_contains_confidence(self, notifier, sample_prediction, sample_triage):
        msg = notifier.format_alert(sample_prediction, sample_triage)
        assert "0.920" in msg

    def test_contains_flow_meta(self, notifier, sample_prediction, sample_triage, sample_meta):
        msg = notifier.format_alert(sample_prediction, sample_triage, sample_meta)
        assert "192.168.1.100" in msg
        assert "10.0.0.1" in msg

    def test_contains_next_actions(self, notifier, sample_prediction, sample_triage):
        msg = notifier.format_alert(sample_prediction, sample_triage)
        assert "Block source IP" in msg

    def test_html_escaping(self, notifier, sample_prediction):
        triage = {
            "label": "<script>xss</script>",
            "severity": "high",
            "mitre_tactics": [],
            "mitre_techniques": [],
            "summary": "test & verify",
            "next_actions": [],
            "source": "test",
        }
        msg = notifier.format_alert(sample_prediction, triage)
        assert "<script>" not in msg
        assert "&lt;script&gt;" in msg
        assert "&amp;" in msg


# ── send_message ──────────────────────────────────────────

class TestSendMessage:
    @patch("clawdbot.telegram.urllib.request.urlopen")
    def test_success(self, mock_urlopen, notifier):
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = json.dumps({"ok": True}).encode()
        mock_urlopen.return_value = mock_resp

        assert notifier.send_message("test") is True

    @patch("clawdbot.telegram.urllib.request.urlopen")
    def test_api_error(self, mock_urlopen, notifier):
        mock_urlopen.side_effect = TimeoutError("timeout")
        assert notifier.send_message("test") is False


# ── alert (integration of the above) ─────────────────────

class TestAlert:
    @patch("clawdbot.telegram.urllib.request.urlopen")
    def test_alert_sent_above_threshold(self, mock_urlopen, notifier, sample_prediction, sample_triage):
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = json.dumps({"ok": True}).encode()
        mock_urlopen.return_value = mock_resp

        assert notifier.alert(sample_prediction, sample_triage) is True
        mock_urlopen.assert_called_once()

    def test_alert_skipped_below_threshold(self, notifier, sample_prediction):
        triage = {"severity": "low", "label": "normal", "source": "test"}
        assert notifier.alert(sample_prediction, triage) is False

    def test_alert_skipped_when_disabled(self, sample_prediction, sample_triage):
        n = TelegramNotifier(bot_token="", chat_id="")
        assert n.alert(sample_prediction, sample_triage) is False


# ── Block result formatting ──────────────────────────────

class TestFormatBlockResult:
    def test_blocked(self):
        br = {"ip": "10.0.0.5", "applied": True, "ttl": 3600}
        result = _format_block_result(br)
        assert "Blocked" in result
        assert "10.0.0.5" in result
        assert "60min" in result

    def test_blocked_dry_run(self):
        br = {"ip": "10.0.0.5", "applied": True, "ttl": 1800, "dry_run": True}
        result = _format_block_result(br)
        assert "DRY-RUN" in result
        assert "30min" in result

    def test_whitelisted(self):
        br = {"ip": "10.0.0.1", "applied": False, "skipped_reason": "whitelisted"}
        result = _format_block_result(br)
        assert "whitelisted" in result

    def test_actuator_disabled(self):
        br = {"ip": "10.0.0.1", "applied": False, "skipped_reason": "actuator_disabled"}
        result = _format_block_result(br)
        assert "actuator disabled" in result


class TestFormatAlertWithBlockResult:
    def test_contains_firewall_action(self, notifier, sample_prediction, sample_triage, sample_meta):
        br = {"ip": "192.168.1.100", "applied": True, "ttl": 3600}
        msg = notifier.format_alert(sample_prediction, sample_triage, sample_meta, block_result=br)
        assert "Firewall action" in msg
        assert "Blocked" in msg
        assert "60min" in msg

    def test_contains_whitelisted(self, notifier, sample_prediction, sample_triage, sample_meta):
        br = {"ip": "192.168.1.100", "applied": False, "skipped_reason": "whitelisted"}
        msg = notifier.format_alert(sample_prediction, sample_triage, sample_meta, block_result=br)
        assert "whitelisted" in msg

    def test_contains_reputation(self, notifier, sample_prediction, sample_triage, sample_meta):
        rep = {"badge": "\U0001f534 Known-bad", "hit_count": 5}
        msg = notifier.format_alert(sample_prediction, sample_triage, sample_meta, reputation=rep)
        assert "Known-bad" in msg
        assert "5 hit(s)" in msg

    def test_no_block_section_when_none(self, notifier, sample_prediction, sample_triage):
        msg = notifier.format_alert(sample_prediction, sample_triage)
        assert "Firewall action" not in msg


class TestBatchSummaryWithBlockResults:
    def test_batch_shows_blocked_ips(self, notifier, sample_prediction, sample_triage, sample_meta):
        detections = [{
            "prediction": sample_prediction,
            "triage": sample_triage,
            "flow_meta": sample_meta,
            "block_result": {"ip": "192.168.1.100", "applied": True, "ttl": 3600},
            "reputation": None,
        }]
        msg = notifier.format_batch_summary(detections)
        assert "Firewall actions" in msg
        assert "Blocked" in msg
        assert "192.168.1.100" in msg

    def test_batch_shows_whitelisted(self, notifier, sample_prediction, sample_triage, sample_meta):
        detections = [{
            "prediction": sample_prediction,
            "triage": sample_triage,
            "flow_meta": sample_meta,
            "block_result": {"ip": "10.0.0.1", "applied": False, "skipped_reason": "whitelisted"},
            "reputation": None,
        }]
        msg = notifier.format_batch_summary(detections)
        assert "whitelisted" in msg

    def test_batch_no_firewall_section_without_blocks(self, notifier, sample_prediction, sample_triage, sample_meta):
        detections = [{
            "prediction": sample_prediction,
            "triage": sample_triage,
            "flow_meta": sample_meta,
            "block_result": None,
            "reputation": None,
        }]
        msg = notifier.format_batch_summary(detections)
        assert "Firewall actions" not in msg

    def test_batch_maps_mitre_for_primary_only(self, notifier, sample_meta):
        detections = [
            {
                "prediction": {"predicted_label": "scanning", "confidence": 0.95},
                "triage": {
                    "label": "scanning",
                    "severity": "high",
                    "mitre_techniques": [{"id": "T1595", "name": "Active Scanning"}],
                },
                "flow_meta": sample_meta,
            },
            {
                "prediction": {"predicted_label": "scanning", "confidence": 0.92},
                "triage": {
                    "label": "scanning",
                    "severity": "high",
                    "mitre_techniques": [{"id": "T1595", "name": "Active Scanning"}],
                },
                "flow_meta": sample_meta,
            },
            {
                "prediction": {"predicted_label": "password", "confidence": 0.91},
                "triage": {
                    "label": "password",
                    "severity": "high",
                    "mitre_techniques": [{"id": "T1110", "name": "Brute Force"}],
                },
                "flow_meta": sample_meta,
            },
            {
                "prediction": {"predicted_label": "ddos_dos", "confidence": 0.93},
                "triage": {
                    "label": "ddos_dos",
                    "severity": "high",
                    "mitre_techniques": [{"id": "T1498", "name": "Network Denial of Service"}],
                },
                "flow_meta": sample_meta,
            },
        ]

        msg = notifier.format_batch_summary(detections)

        assert "1 threat, 2 possible" in msg
        assert "Primary incident" in msg
        assert "scanning" in msg
        assert "Secondary signals" in msg
        assert "not MITRE-mapped" in msg
        assert "flow(s)" not in msg
        assert "T1595" in msg
        assert "T1110" not in msg
        assert "T1498" not in msg
