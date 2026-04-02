"""Tests for clawdbot.capture — FlowTable and TrafficCapture."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from clawdbot.capture import FlowKey, FlowStats, FlowTable, TrafficCapture


# ── FlowKey ───────────────────────────────────────────────

class TestFlowKey:
    def test_hash_equality(self):
        a = FlowKey("1.1.1.1", "2.2.2.2", "tcp", 80, 443)
        b = FlowKey("1.1.1.1", "2.2.2.2", "tcp", 80, 443)
        assert a == b
        assert hash(a) == hash(b)

    def test_hash_inequality(self):
        a = FlowKey("1.1.1.1", "2.2.2.2", "tcp", 80, 443)
        b = FlowKey("1.1.1.1", "3.3.3.3", "tcp", 80, 443)
        assert a != b

    def test_usable_as_dict_key(self):
        key = FlowKey("1.1.1.1", "2.2.2.2", "tcp", 80, 443)
        d = {key: "value"}
        assert d[key] == "value"


# ── FlowTable ─────────────────────────────────────────────

def _make_ip_packet(src="1.1.1.1", dst="2.2.2.2", sport=12345, dport=80, proto="tcp", size=100):
    """Build a mock scapy packet with IP/TCP/UDP layers."""
    pkt = MagicMock()
    pkt.__len__ = MagicMock(return_value=size)

    ip_layer = MagicMock()
    ip_layer.src = src
    ip_layer.dst = dst
    ip_layer.proto = 6

    tcp_layer = MagicMock()
    tcp_layer.sport = sport
    tcp_layer.dport = dport

    udp_layer = MagicMock()
    udp_layer.sport = sport
    udp_layer.dport = dport

    def has_layer(layer_cls):
        if layer_cls is None:
            return False
        name = getattr(layer_cls, "__name__", str(layer_cls))
        if proto == "tcp":
            return name in ("IP", "TCP")
        elif proto == "udp":
            return name in ("IP", "UDP")
        return name == "IP"

    pkt.haslayer = has_layer
    pkt.__getitem__ = lambda self, cls: {
        "IP": ip_layer, "TCP": tcp_layer, "UDP": udp_layer
    }.get(getattr(cls, "__name__", str(cls)), ip_layer)

    return pkt


class TestFlowTable:
    def test_process_packet_creates_flow(self):
        with patch("clawdbot.capture.IP") as mock_ip, \
             patch("clawdbot.capture.TCP") as mock_tcp, \
             patch("clawdbot.capture.UDP") as mock_udp, \
             patch("clawdbot.capture.ICMP") as mock_icmp:
            mock_ip.__name__ = "IP"
            mock_tcp.__name__ = "TCP"
            mock_udp.__name__ = "UDP"
            mock_icmp.__name__ = "ICMP"

            ft = FlowTable()
            pkt = _make_ip_packet(proto="tcp")
            ft.process_packet(pkt)
            assert ft.active_flows == 1

    def test_harvest_returns_records(self):
        with patch("clawdbot.capture.IP") as mock_ip, \
             patch("clawdbot.capture.TCP") as mock_tcp, \
             patch("clawdbot.capture.UDP") as mock_udp, \
             patch("clawdbot.capture.ICMP") as mock_icmp:
            mock_ip.__name__ = "IP"
            mock_tcp.__name__ = "TCP"
            mock_udp.__name__ = "UDP"
            mock_icmp.__name__ = "ICMP"

            ft = FlowTable()
            pkt = _make_ip_packet(proto="tcp", size=200)
            ft.process_packet(pkt)

            records = ft.harvest()
            assert len(records) == 1
            r = records[0]
            assert r["proto"] == "tcp"
            assert r["src_bytes"] == 200
            assert r["_meta"]["src_ip"] == "1.1.1.1"
            assert r["_meta"]["dst_ip"] == "2.2.2.2"

    def test_harvest_drains_table(self):
        with patch("clawdbot.capture.IP") as mock_ip, \
             patch("clawdbot.capture.TCP") as mock_tcp, \
             patch("clawdbot.capture.UDP") as mock_udp, \
             patch("clawdbot.capture.ICMP") as mock_icmp:
            mock_ip.__name__ = "IP"
            mock_tcp.__name__ = "TCP"
            mock_udp.__name__ = "UDP"
            mock_icmp.__name__ = "ICMP"

            ft = FlowTable()
            pkt = _make_ip_packet(proto="tcp")
            ft.process_packet(pkt)

            ft.harvest()
            assert ft.active_flows == 0

    def test_no_ip_packet_ignored(self):
        ft = FlowTable()
        pkt = MagicMock()
        pkt.haslayer = MagicMock(return_value=False)
        ft.process_packet(pkt)
        assert ft.active_flows == 0


# ── TrafficCapture ────────────────────────────────────────

class TestTrafficCapture:
    def test_requires_scapy(self):
        with patch("clawdbot.capture.scapy_sniff", None):
            with pytest.raises(RuntimeError, match="scapy"):
                TrafficCapture(interface="eth0")

    @patch("clawdbot.capture.scapy_sniff")
    def test_start_stop(self, mock_sniff):
        mock_sniff.return_value = None
        tc = TrafficCapture(interface="lo")
        tc.start()
        assert tc._thread is not None
        tc.stop()
        assert tc._thread is None

    @patch("clawdbot.capture.scapy_sniff")
    def test_harvest_delegates(self, mock_sniff):
        tc = TrafficCapture(interface="lo")
        result = tc.harvest()
        assert result == []
