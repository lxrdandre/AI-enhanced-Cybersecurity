"""Tests for clawdbot.actuator - nftables active response engine."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from clawdbot.actuator import (
    ACTIONABLE_SEVERITIES,
    DEFAULT_BLOCK_SECONDS,
    MAX_BLOCK_SECONDS,
    MIN_BLOCK_SECONDS,
    SEVERITY_DURATION,
    Actuator,
    _is_whitelisted,
    _parse_whitelist,
)


# -- Whitelist parsing ----------------------------------------

class TestParseWhitelist:
    """Group tests covering parse whitelist behavior."""
    def test_defaults_always_included(self):
        """Verify that defaults always included."""
        wl = _parse_whitelist("")
        # loopback and NetBird mesh must be present
        import ipaddress
        assert any(ipaddress.ip_address("127.0.0.1") in net for net in wl)
        assert any(ipaddress.ip_address("100.111.76.168") in net for net in wl)
        assert any(ipaddress.ip_address("100.111.77.70") in net for net in wl)

    def test_extra_entries_added(self):
        """Verify that extra entries added."""
        import ipaddress
        wl = _parse_whitelist("10.0.0.0/8, 192.168.1.100")
        assert any(ipaddress.ip_address("10.50.0.1") in net for net in wl)
        assert any(ipaddress.ip_address("192.168.1.100") in net for net in wl)

    def test_invalid_entries_skipped(self):
        """Verify that invalid entries skipped."""
        wl = _parse_whitelist("not-an-ip, 10.0.0.0/8")
        import ipaddress
        assert any(ipaddress.ip_address("10.1.1.1") in net for net in wl)

    def test_empty_entries_skipped(self):
        """Verify that empty entries skipped."""
        wl = _parse_whitelist(",,,")
        # Should still have defaults
        assert len(wl) >= 2


class TestIsWhitelisted:
    """Group tests covering is whitelisted behavior."""
    def test_loopback_whitelisted(self):
        """Verify that loopback whitelisted."""
        wl = _parse_whitelist("")
        assert _is_whitelisted("127.0.0.1", wl)

    def test_netbird_ips_whitelisted(self):
        """Verify that netbird ips whitelisted."""
        wl = _parse_whitelist("")
        assert _is_whitelisted("100.111.76.168", wl)
        assert _is_whitelisted("100.111.77.70", wl)

    def test_external_ip_not_whitelisted(self):
        """Verify that external ip not whitelisted."""
        wl = _parse_whitelist("")
        assert not _is_whitelisted("203.0.113.50", wl)

    def test_invalid_ip_treated_as_whitelisted(self):
        """Verify that invalid ip treated as whitelisted."""
        wl = _parse_whitelist("")
        assert _is_whitelisted("not-an-ip", wl)


# -- Actuator disabled ---------------------------------------

class TestActuatorDisabled:
    """Group tests covering actuator disabled behavior."""
    def test_block_noop_when_disabled(self):
        """Verify that block noop when disabled."""
        act = Actuator(enabled=False)
        result = act.block("1.2.3.4", reason="test")
        assert result["applied"] is False
        assert result["skipped_reason"] == "actuator_disabled"

    def test_unblock_noop_when_disabled(self):
        """Verify that unblock noop when disabled."""
        act = Actuator(enabled=False)
        result = act.unblock("1.2.3.4")
        assert result["applied"] is False
        assert result["skipped_reason"] == "actuator_disabled"

    def test_maybe_block_returns_none_when_disabled(self):
        """Verify that maybe block returns none when disabled."""
        act = Actuator(enabled=False)
        result = act.maybe_block_from_detection(
            src_ip="1.2.3.4", severity="high", label="dos_ddos", confidence=0.95,
        )
        assert result is None

    def test_setup_noop_when_disabled(self):
        """Verify that setup noop when disabled."""
        act = Actuator(enabled=False)
        act.setup()  # Should not raise


# -- Actuator dry-run -----------------------------------------

class TestActuatorDryRun:
    """Group tests covering actuator dry run behavior."""
    def test_block_dry_run_no_subprocess(self):
        """Verify that block dry run no subprocess."""
        act = Actuator(enabled=True, dry_run=True)
        with patch("clawdbot.actuator._run") as mock_run:
            result = act.block("203.0.113.50", ttl=600, reason="test")
        assert result["applied"] is True
        assert result["dry_run"] is True
        mock_run.assert_not_called()

    def test_block_dry_run_tracks_in_memory(self):
        """Verify that block dry run tracks in memory."""
        act = Actuator(enabled=True, dry_run=True)
        act.block("203.0.113.50", ttl=600, reason="test")
        assert act.active_block_count == 1

    def test_whitelist_still_enforced_in_dry_run(self):
        """Verify that whitelist still enforced in dry run."""
        act = Actuator(enabled=True, dry_run=True)
        result = act.block("127.0.0.1", reason="test")
        assert result["applied"] is False
        assert result["skipped_reason"] == "whitelisted"

    def test_netbird_ip_blocked_skipped_in_dry_run(self):
        """Verify that netbird ip blocked skipped in dry run."""
        act = Actuator(enabled=True, dry_run=True)
        result = act.block("100.111.76.168", reason="test")
        assert result["applied"] is False
        assert result["skipped_reason"] == "whitelisted"

    def test_setup_dry_run_no_subprocess(self):
        """Verify that setup dry run no subprocess."""
        act = Actuator(enabled=True, dry_run=True)
        with patch("clawdbot.actuator._run") as mock_run:
            act.setup()
        mock_run.assert_not_called()


# -- Actuator live (mocked nft) -------------------------------

class TestActuatorLive:
    """Group tests covering actuator live behavior."""
    @patch("clawdbot.actuator._run")
    def test_setup_creates_table_set_chain(self, mock_run):
        """Verify that setup creates table set chain."""
        mock_run.return_value = MagicMock(returncode=0)
        act = Actuator(enabled=True, dry_run=False)
        act.setup()
        assert mock_run.call_count == 4  # table + set + chain + rule

    @patch("clawdbot.actuator._run")
    def test_block_calls_nft_add_element(self, mock_run):
        """Verify that block calls nft add element."""
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        act = Actuator(enabled=True, dry_run=False)
        result = act.block("203.0.113.50", ttl=900, reason="scanning")
        assert result["applied"] is True
        assert result["ttl"] == 900
        # Check the nft command
        call_args = mock_run.call_args[0][0]
        assert "add" in call_args
        assert "element" in call_args
        assert "203.0.113.50" in " ".join(call_args)

    @patch("clawdbot.actuator._run")
    def test_block_invalid_ip_skipped(self, mock_run):
        """Verify that block invalid ip skipped."""
        act = Actuator(enabled=True, dry_run=False)
        result = act.block("not.valid", reason="test")
        assert result["applied"] is False
        assert result["skipped_reason"] == "invalid_ip"
        mock_run.assert_not_called()

    @patch("clawdbot.actuator._run")
    def test_unblock_calls_nft_delete_element(self, mock_run):
        """Verify that unblock calls nft delete element."""
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        act = Actuator(enabled=True, dry_run=False)
        result = act.unblock("203.0.113.50", reason="false_positive")
        assert result["applied"] is True

    @patch("clawdbot.actuator._run")
    def test_nft_failure_captured(self, mock_run):
        """Verify that nft failure captured."""
        mock_run.return_value = MagicMock(returncode=1, stderr="permission denied")
        act = Actuator(enabled=True, dry_run=False)
        result = act.block("203.0.113.50", reason="test")
        assert result["applied"] is False
        assert "permission denied" in result["skipped_reason"]


# -- TTL clamping ---------------------------------------------

class TestTTLClamping:
    """Group tests covering ttlclamping behavior."""
    @patch("clawdbot.actuator._run")
    def test_ttl_clamped_to_max(self, mock_run):
        """Verify that ttl clamped to max."""
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        act = Actuator(enabled=True, dry_run=False)
        result = act.block("203.0.113.50", ttl=999999, reason="test")
        assert result["ttl"] == MAX_BLOCK_SECONDS

    @patch("clawdbot.actuator._run")
    def test_ttl_clamped_to_min(self, mock_run):
        """Verify that ttl clamped to min."""
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        act = Actuator(enabled=True, dry_run=False)
        result = act.block("203.0.113.50", ttl=5, reason="test")
        assert result["ttl"] == MIN_BLOCK_SECONDS


# -- maybe_block_from_detection -------------------------------

class TestMaybeBlock:
    """Group tests covering maybe block behavior."""
    def test_no_src_ip_returns_none(self):
        """Verify that no src ip returns none."""
        act = Actuator(enabled=True, dry_run=True)
        result = act.maybe_block_from_detection(
            src_ip=None, severity="high", label="dos_ddos", confidence=0.9,
        )
        assert result is None

    def test_low_severity_returns_none(self):
        """Verify that low severity returns none."""
        act = Actuator(enabled=True, dry_run=True)
        result = act.maybe_block_from_detection(
            src_ip="203.0.113.50", severity="low", label="scanning", confidence=0.8,
        )
        assert result is None

    def test_review_severity_returns_none(self):
        """Verify that review severity returns none."""
        act = Actuator(enabled=True, dry_run=True)
        result = act.maybe_block_from_detection(
            src_ip="203.0.113.50", severity="review", label="unknown", confidence=0.5,
        )
        assert result is None

    def test_high_severity_blocks(self):
        """Verify that high severity blocks."""
        act = Actuator(enabled=True, dry_run=True)
        result = act.maybe_block_from_detection(
            src_ip="203.0.113.50", severity="high", label="dos_ddos", confidence=0.95,
        )
        assert result is not None
        assert result["applied"] is True
        assert result["ttl"] == SEVERITY_DURATION["high"]

    def test_critical_severity_blocks_with_longer_ttl(self):
        """Verify that critical severity blocks with longer ttl."""
        act = Actuator(enabled=True, dry_run=True)
        result = act.maybe_block_from_detection(
            src_ip="203.0.113.50", severity="critical", label="injection", confidence=0.99,
        )
        assert result is not None
        assert result["ttl"] == SEVERITY_DURATION["critical"]

    def test_duplicate_block_skipped(self):
        """Verify that duplicate block skipped."""
        act = Actuator(enabled=True, dry_run=True)
        r1 = act.maybe_block_from_detection(
            src_ip="203.0.113.50", severity="high", label="dos_ddos", confidence=0.9,
        )
        r2 = act.maybe_block_from_detection(
            src_ip="203.0.113.50", severity="high", label="dos_ddos", confidence=0.9,
        )
        assert r1 is not None
        assert r2 is None  # already blocked

    def test_whitelisted_ip_not_blocked(self):
        """Verify that whitelisted ip not blocked."""
        act = Actuator(enabled=True, dry_run=True)
        result = act.maybe_block_from_detection(
            src_ip="100.111.76.168", severity="critical", label="backdoor", confidence=0.99,
        )
        assert result is not None
        assert result["applied"] is False
        assert result["skipped_reason"] == "whitelisted"


# -- active_block_count ---------------------------------------

class TestActiveBlockCount:
    """Group tests covering active block count behavior."""
    def test_count_tracks_blocks(self):
        """Verify that count tracks blocks."""
        act = Actuator(enabled=True, dry_run=True)
        assert act.active_block_count == 0
        act.block("203.0.113.1", ttl=3600, reason="test")
        act.block("203.0.113.2", ttl=3600, reason="test")
        assert act.active_block_count == 2

    def test_count_prunes_expired(self):
        """Verify that count prunes expired."""
        act = Actuator(enabled=True, dry_run=True)
        act.block("203.0.113.1", ttl=60, reason="test")
        # Manually expire
        act._active_blocks["203.0.113.1"] = 0.0
        assert act.active_block_count == 0
