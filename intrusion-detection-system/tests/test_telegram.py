"""Tests for clawdbot.telegram — TelegramNotifier."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from clawdbot.telegram import SEVERITY_ORDER, TelegramNotifier


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
