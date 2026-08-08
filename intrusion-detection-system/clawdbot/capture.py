"""Live network traffic capture and CIC-style flow aggregation for ClawdBot IDS.

The live sniffer is passive: Scapy sniffs packets from the selected interface(s)
and aggregates them into bidirectional flows. The harvested record keeps the
legacy TON-style fields used elsewhere in the project and adds a broader
CICFlowMeter-like feature set so the public CIC model is not fed mostly zeros.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

try:
    from scapy.all import ICMP, IP, TCP, UDP, sniff as scapy_sniff
    from scapy.layers.dns import DNS, DNSQR
    from scapy.layers.http import HTTPRequest, HTTPResponse
except ImportError:
    scapy_sniff = None
    IP = TCP = UDP = ICMP = DNS = DNSQR = HTTPRequest = HTTPResponse = None
    log.warning("scapy not installed - live capture unavailable")


ACTIVE_IDLE_GAP_SECONDS = 1.0


@dataclass
class RunningStats:
    """Track online aggregate statistics for a flow metric."""
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    minimum: float = math.inf
    maximum: float = 0.0

    def update(self, value: float) -> None:
        """Add one non-negative observation to the aggregate."""
        value = max(float(value), 0.0)
        self.count += 1
        self.total += value
        self.total_sq += value * value
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    @property
    def mean(self) -> float:
        """Return the arithmetic mean, or zero when empty."""
        return self.total / self.count if self.count else 0.0

    @property
    def variance(self) -> float:
        """Return population variance for the observed values."""
        if self.count <= 1:
            return 0.0
        mean = self.mean
        return max((self.total_sq / self.count) - (mean * mean), 0.0)

    @property
    def std(self) -> float:
        """Return the population standard deviation."""
        return math.sqrt(self.variance)

    @property
    def min_value(self) -> float:
        """Return the minimum observed value, or zero when empty."""
        return 0.0 if self.count == 0 else self.minimum

    @property
    def max_value(self) -> float:
        """Return the maximum observed value, or zero when empty."""
        return 0.0 if self.count == 0 else self.maximum


# -- Service inference from well-known ports ------------------

_PORT_TO_SERVICE = {
    20: "ftp-data",
    21: "ftp",
    22: "ssh",
    23: "telnet",
    25: "smtp",
    53: "dns",
    80: "http",
    110: "pop3",
    143: "imap",
    443: "ssl",
    993: "imaps",
    995: "pop3s",
    8080: "http",
    3306: "mysql",
    5432: "postgres",
    6379: "redis",
    27017: "mongodb",
}


def _infer_service(proto: str, src_port: int, dst_port: int) -> str:
    """Infer a service name from protocol and well-known ports."""
    svc = _PORT_TO_SERVICE.get(dst_port) or _PORT_TO_SERVICE.get(src_port)
    return svc or "-"


def _infer_conn_state(
    syn_seen: bool,
    syn_ack_seen: bool,
    fin_seen: bool,
    rst_seen: bool,
    src_pkts: int,
    dst_pkts: int,
) -> str:
    """Infer a compact connection-state code from TCP flow evidence."""
    if rst_seen:
        if syn_seen and not syn_ack_seen:
            return "REJ"
        return "RSTR"
    if syn_seen and not syn_ack_seen and dst_pkts == 0:
        return "S0"
    if syn_seen and syn_ack_seen and fin_seen:
        return "SF"
    if syn_seen and syn_ack_seen:
        return "S1"
    if src_pkts > 0 and dst_pkts == 0:
        return "S0"
    return "OTH"


def _safe_int(value, default: int = 0) -> int:
    """Convert a value to int, returning the default on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_flag_string(tcp_layer) -> str:
    """Return a sanitized TCP flag string from a Scapy layer."""
    flags = getattr(tcp_layer, "flags", None)
    if flags is None:
        return ""
    flag_type = type(flags)
    if flag_type.__name__ in {"MagicMock", "Mock"} or flag_type.__module__.startswith("unittest.mock"):
        return ""
    text = str(flags).upper()
    if "MAGICMOCK" in text or text.startswith("<"):
        return ""
    return "".join(ch for ch in text if ch.isalpha())


def _safe_decode(value) -> str:
    """Decode bytes as UTF-8 and normalize missing values to empty text."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _rate(total: float, duration: float) -> float:
    """Return a per-second rate with zero-duration protection."""
    if duration <= 0:
        return 0.0
    return float(total) / float(duration)


def _mean_or_zero(total: float, count: int) -> float:
    """Return an average, or zero when the count is empty."""
    return float(total) / float(count) if count > 0 else 0.0


def _http_uri_depth(uri: str) -> int:
    """Return a compact URI path depth."""
    path = str(uri or "").split("?", 1)[0]
    return len([part for part in path.split("/") if part])


def _contains_any(text: str, needles: tuple[str, ...]) -> int:
    """Return 1 when any marker is present in text."""
    lowered = str(text or "").lower()
    return int(any(marker in lowered for marker in needles))


def _zeek_history(stats: "FlowStats") -> str:
    """Approximate Zeek history flags from Scapy-observed flow state."""
    flags = []
    if stats.syn_flag_count:
        flags.append("s")
    if stats.ack_flag_count:
        flags.append("a")
    if stats.rst_flag_count:
        flags.append("r")
    if stats.fin_flag_count:
        flags.append("f")
    if stats.tcp_payload_bytes > 0 or stats.http_request_body_len > 0 or stats.http_response_body_len > 0:
        flags.append("d")
    return "".join(flags) or "-"


def _zeek_state(conn_state: str) -> str:
    """Normalize connection state for context feature counters."""
    return str(conn_state or "-").strip().lower()


@dataclass(frozen=True)
class FlowKey:
    """Identify a bidirectional network flow."""
    src_ip: str
    dst_ip: str
    proto: str
    src_port: int = 0
    dst_port: int = 0


@dataclass
class FlowStats:
    """Accumulate packet, byte, timing, DNS, and HTTP metrics for a flow."""
    src_bytes: int = 0
    dst_bytes: int = 0
    src_pkts: int = 0
    dst_pkts: int = 0
    src_ip_bytes: int = 0
    dst_ip_bytes: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0
    syn_seen: bool = False
    syn_ack_seen: bool = False
    fin_seen: bool = False
    rst_seen: bool = False
    fin_flag_count: int = 0
    syn_flag_count: int = 0
    rst_flag_count: int = 0
    psh_flag_count: int = 0
    ack_flag_count: int = 0
    urg_flag_count: int = 0
    cwe_flag_count: int = 0
    ece_flag_count: int = 0
    fwd_psh_flags: int = 0
    bwd_psh_flags: int = 0
    fwd_urg_flags: int = 0
    bwd_urg_flags: int = 0
    src_header_bytes: int = 0
    dst_header_bytes: int = 0
    dns_query: str = "-"
    dns_qclass: str = "-"
    dns_qtype: str = "-"
    dns_rcode: str = "-"
    dns_AA: int = 0
    http_method: str = "-"
    http_host: str = "-"
    http_uri: str = "-"
    http_user_agent: str = "-"
    http_referrer: str = "-"
    http_version: str = "-"
    http_request_body_len: int = 0
    http_response_body_len: int = 0
    http_status_code: int = 0
    http_trans_depth: int = 0
    http_file_data: str = ""
    init_win_bytes_forward: int = 0
    init_win_bytes_backward: int = 0
    _init_win_forward_set: bool = False
    _init_win_backward_set: bool = False
    act_data_pkt_fwd: int = 0
    min_seg_size_forward: int = 0
    tcp_ack: int = 0
    tcp_seq: int = 0
    tcp_payload_bytes: int = 0
    tcp_flags_numeric: int = 0
    udp_time_delta: float = 0.0
    last_udp_ts: float = 0.0
    _min_seg_size_forward_set: bool = False
    last_packet_ts: float = 0.0
    last_src_ts: float = 0.0
    last_dst_ts: float = 0.0
    flow_iat: RunningStats = field(default_factory=RunningStats)
    src_iat: RunningStats = field(default_factory=RunningStats)
    dst_iat: RunningStats = field(default_factory=RunningStats)
    packet_lengths: RunningStats = field(default_factory=RunningStats)
    src_packet_lengths: RunningStats = field(default_factory=RunningStats)
    dst_packet_lengths: RunningStats = field(default_factory=RunningStats)
    active_periods: RunningStats = field(default_factory=RunningStats)
    idle_periods: RunningStats = field(default_factory=RunningStats)
    _active_start: float = 0.0
    _active_last: float = 0.0
    _active_finalized: bool = False

    def update_activity(self, now: float) -> None:
        """Update activity."""
        if self.first_seen == 0.0:
            self.first_seen = now
        if self.last_packet_ts > 0.0:
            self.flow_iat.update(now - self.last_packet_ts)
        self.last_packet_ts = now
        self.last_seen = now

        if self._active_start == 0.0:
            self._active_start = now
            self._active_last = now
            self._active_finalized = False
            return

        gap = now - self._active_last
        if gap > ACTIVE_IDLE_GAP_SECONDS:
            active_duration = self._active_last - self._active_start
            if active_duration > 0.0:
                self.active_periods.update(active_duration)
            self.idle_periods.update(gap)
            self._active_start = now
        self._active_last = now
        self._active_finalized = False

    def finalize(self) -> None:
        """Build the model-ready record for the completed flow."""
        if self._active_finalized or self._active_start == 0.0:
            return
        active_duration = self._active_last - self._active_start
        if active_duration > 0.0:
            self.active_periods.update(active_duration)
        self._active_finalized = True


class FlowTable:
    """Aggregate raw packets into bidirectional flow records."""

    def __init__(self):
        """Initialize the flow table instance."""
        self._flows: dict[FlowKey, FlowStats] = {}
        self._lock = threading.Lock()

    def process_packet(self, pkt) -> None:
        """Fold one packet into the appropriate flow aggregate."""
        if IP is None or not pkt.haslayer(IP):
            return

        ip = pkt[IP]
        now = time.time()
        packet_len = len(pkt)
        ip_len = _safe_int(getattr(ip, "len", None), packet_len) or packet_len
        ip_header_len = (_safe_int(getattr(ip, "ihl", None), 5) or 5) * 4

        src_port = dst_port = 0
        tcp_flags = ""
        tcp_flags_numeric = 0
        transport_header_len = 0
        payload_len = 0
        tcp_window = 0
        tcp_ack = 0
        tcp_seq = 0

        if pkt.haslayer(TCP):
            proto = "tcp"
            tcp = pkt[TCP]
            src_port = _safe_int(getattr(tcp, "sport", 0))
            dst_port = _safe_int(getattr(tcp, "dport", 0))
            tcp_flags = _safe_flag_string(tcp)
            tcp_flags_numeric = _safe_int(getattr(tcp, "flags", 0))
            tcp_ack = _safe_int(getattr(tcp, "ack", 0))
            tcp_seq = _safe_int(getattr(tcp, "seq", 0))
            transport_header_len = (_safe_int(getattr(tcp, "dataofs", None), 5) or 5) * 4
            tcp_window = max(_safe_int(getattr(tcp, "window", 0)), 0)
        elif pkt.haslayer(UDP):
            proto = "udp"
            udp = pkt[UDP]
            src_port = _safe_int(getattr(udp, "sport", 0))
            dst_port = _safe_int(getattr(udp, "dport", 0))
            transport_header_len = 8
        elif pkt.haslayer(ICMP):
            proto = "icmp"
            transport_header_len = 8
        else:
            proto = str(getattr(ip, "proto", "ip")).lower()

        total_header_len = ip_header_len + transport_header_len
        payload_len = max(packet_len - total_header_len, 0)

        key = FlowKey(ip.src, ip.dst, proto, src_port, dst_port)
        rev_key = FlowKey(ip.dst, ip.src, proto, dst_port, src_port)

        with self._lock:
            reverse = False
            flow = self._flows.get(key)
            if flow is None and rev_key in self._flows:
                flow = self._flows[rev_key]
                reverse = True
            if flow is None:
                flow = FlowStats(first_seen=now)
                self._flows[key] = flow

            flow.update_activity(now)
            if proto == "tcp":
                flow.tcp_ack = tcp_ack
                flow.tcp_seq = tcp_seq
                flow.tcp_flags_numeric = tcp_flags_numeric
                flow.tcp_payload_bytes += payload_len
            elif proto == "udp":
                if flow.last_udp_ts > 0.0:
                    flow.udp_time_delta = now - flow.last_udp_ts
                flow.last_udp_ts = now

            if reverse:
                flow.dst_bytes += packet_len
                flow.dst_pkts += 1
                flow.dst_ip_bytes += ip_len
                flow.dst_header_bytes += total_header_len
                flow.dst_packet_lengths.update(packet_len)
                if flow.last_dst_ts > 0.0:
                    flow.dst_iat.update(now - flow.last_dst_ts)
                flow.last_dst_ts = now
                if proto == "tcp" and not flow._init_win_backward_set and tcp_window > 0:
                    flow.init_win_bytes_backward = tcp_window
                    flow._init_win_backward_set = True
            else:
                flow.src_bytes += packet_len
                flow.src_pkts += 1
                flow.src_ip_bytes += ip_len
                flow.src_header_bytes += total_header_len
                flow.src_packet_lengths.update(packet_len)
                if flow.last_src_ts > 0.0:
                    flow.src_iat.update(now - flow.last_src_ts)
                flow.last_src_ts = now
                if proto == "tcp" and not flow._init_win_forward_set and tcp_window > 0:
                    flow.init_win_bytes_forward = tcp_window
                    flow._init_win_forward_set = True
                if payload_len > 0:
                    flow.act_data_pkt_fwd += 1
                    if not flow._min_seg_size_forward_set or payload_len < flow.min_seg_size_forward:
                        flow.min_seg_size_forward = payload_len
                        flow._min_seg_size_forward_set = True
            flow.packet_lengths.update(packet_len)

            if "S" in tcp_flags and "A" not in tcp_flags:
                flow.syn_seen = True
            if "S" in tcp_flags and "A" in tcp_flags:
                flow.syn_ack_seen = True
            if "F" in tcp_flags:
                flow.fin_seen = True
            if "R" in tcp_flags:
                flow.rst_seen = True
            if "F" in tcp_flags:
                flow.fin_flag_count += 1
            if "S" in tcp_flags:
                flow.syn_flag_count += 1
            if "R" in tcp_flags:
                flow.rst_flag_count += 1
            if "P" in tcp_flags:
                flow.psh_flag_count += 1
                if reverse:
                    flow.bwd_psh_flags += 1
                else:
                    flow.fwd_psh_flags += 1
            if "A" in tcp_flags:
                flow.ack_flag_count += 1
            if "U" in tcp_flags:
                flow.urg_flag_count += 1
                if reverse:
                    flow.bwd_urg_flags += 1
                else:
                    flow.fwd_urg_flags += 1
            if "C" in tcp_flags:
                flow.cwe_flag_count += 1
            if "E" in tcp_flags:
                flow.ece_flag_count += 1

            if DNS is not None and pkt.haslayer(DNS) and flow.dns_query == "-":
                dns = pkt[DNS]
                if DNSQR is not None and pkt.haslayer(DNSQR):
                    qr = pkt[DNSQR]
                    flow.dns_query = _safe_decode(getattr(qr, "qname", "")).rstrip(".") or "-"
                    flow.dns_qclass = str(_safe_int(getattr(qr, "qclass", "-"), 0) or "-")
                    flow.dns_qtype = str(_safe_int(getattr(qr, "qtype", "-"), 0) or "-")
                flow.dns_rcode = str(_safe_int(getattr(dns, "rcode", "-"), 0) or "-")
                flow.dns_AA = _safe_int(getattr(dns, "aa", 0), 0)

            if HTTPRequest is not None and pkt.haslayer(HTTPRequest) and flow.http_method == "-":
                request = pkt[HTTPRequest]
                flow.http_trans_depth += 1
                flow.http_method = _safe_decode(getattr(request, "Method", "")) or "-"
                flow.http_host = _safe_decode(getattr(request, "Host", "")) or "-"
                flow.http_uri = _safe_decode(getattr(request, "Path", "")) or "-"
                flow.http_user_agent = _safe_decode(getattr(request, "User_Agent", "")) or "-"
                flow.http_referrer = _safe_decode(getattr(request, "Referer", "")) or "-"
                flow.http_version = _safe_decode(getattr(request, "Http_Version", "")) or "-"
                content_length = _safe_int(getattr(request, "Content_Length", 0), 0)
                flow.http_request_body_len = max(content_length, payload_len)

            if HTTPResponse is not None and pkt.haslayer(HTTPResponse) and flow.http_status_code == 0:
                response = pkt[HTTPResponse]
                flow.http_status_code = _safe_int(getattr(response, "Status_Code", 0), 0)
                content_length = _safe_int(getattr(response, "Content_Length", 0), 0)
                flow.http_response_body_len = max(content_length, payload_len)

    def harvest(self) -> list[dict]:
        """Return finalized flow records and clear completed aggregates."""
        with self._lock:
            flows = self._flows
            self._flows = {}

        records = []
        for key, stats in flows.items():
            stats.finalize()
            duration = max(stats.last_seen - stats.first_seen, 0.0)
            total_pkts = stats.src_pkts + stats.dst_pkts
            total_bytes = stats.src_bytes + stats.dst_bytes

            service = _infer_service(key.proto, key.src_port, key.dst_port)
            conn_state = "-"
            if key.proto == "tcp":
                conn_state = _infer_conn_state(
                    stats.syn_seen,
                    stats.syn_ack_seen,
                    stats.fin_seen,
                    stats.rst_seen,
                    stats.src_pkts,
                    stats.dst_pkts,
                )

            history = _zeek_history(stats)
            http_uri = stats.http_uri if stats.http_uri not in {"", "-"} else ""
            http_count = stats.http_trans_depth if stats.http_trans_depth > 0 else int(stats.http_status_code > 0)
            dns_count = int(stats.dns_query not in {"", "-"})
            ssh_count = int(service == "ssh")
            ssl_count = int(service == "ssl")
            files_count = int(bool(stats.http_file_data))
            http_content_length = stats.http_request_body_len + stats.http_response_body_len
            http_status_text = str(stats.http_status_code) if stats.http_status_code else "-"
            orig_to_resp_bytes_ratio = (
                float(stats.src_bytes) / float(stats.dst_bytes) if stats.dst_bytes > 0 else float(stats.src_bytes)
            )

            record = {
                "ts": round(stats.first_seen or time.time(), 6),
                "src_port": key.src_port,
                "dst_port": key.dst_port,
                "id_orig_p": key.src_port,
                "id_resp_p": key.dst_port,
                "proto": key.proto,
                "service": service,
                "duration": round(duration, 6),
                "src_bytes": stats.src_bytes,
                "dst_bytes": stats.dst_bytes,
                "orig_bytes": stats.src_bytes,
                "resp_bytes": stats.dst_bytes,
                "conn_state": conn_state,
                "history": history,
                "missed_bytes": 0,
                "src_pkts": stats.src_pkts,
                "src_ip_bytes": stats.src_ip_bytes,
                "dst_pkts": stats.dst_pkts,
                "dst_ip_bytes": stats.dst_ip_bytes,
                "orig_pkts": stats.src_pkts,
                "orig_ip_bytes": stats.src_ip_bytes,
                "resp_pkts": stats.dst_pkts,
                "resp_ip_bytes": stats.dst_ip_bytes,
                "flow_total_bytes": total_bytes,
                "flow_total_pkts": total_pkts,
                "orig_to_resp_bytes_ratio": orig_to_resp_bytes_ratio,
                "dns_query": stats.dns_query,
                "dns_count": dns_count,
                "dns_qclass": stats.dns_qclass,
                "dns_qtype": stats.dns_qtype,
                "dns_rcode": stats.dns_rcode,
                "dns_AA": stats.dns_AA,
                "dns_qry_name": stats.dns_query,
                "dns_qry_name_len": len(stats.dns_query) if stats.dns_query not in {"", "-"} else 0,
                "dns_qry_qu": stats.dns_qclass,
                "dns_qry_type": stats.dns_qtype,
                "dns_retransmission": 0,
                "dns_retransmit_request": 0,
                "dns_retransmit_request_in": 0,
                "http_trans_depth": stats.http_trans_depth,
                "http_count": http_count,
                "http_method": stats.http_method,
                "http_host": stats.http_host,
                "http_uri": http_uri,
                "http_uri_len": len(http_uri),
                "http_uri_depth": _http_uri_depth(http_uri),
                "http_uri_has_query": int("?" in http_uri),
                "http_uri_has_sql": _contains_any(http_uri, ("select", "union", "sleep", " or ", "%27", "'")),
                "http_uri_has_xss": _contains_any(http_uri, ("<script", "%3cscript", "alert(", "onerror")),
                "http_uri_has_traversal": _contains_any(http_uri, ("../", "..%2f", "%2e%2e")),
                "http_uri_has_cmd": _contains_any(http_uri, (";id", "|id", "cmd=", "exec", "bash", "wget", "curl")),
                "http_uri_has_upload": _contains_any(http_uri, ("upload", "filename=", "multipart")),
                "http_referrer": stats.http_referrer,
                "http_referer": stats.http_referrer,
                "http_version": stats.http_version,
                "http_request_body_len": stats.http_request_body_len,
                "http_response_body_len": stats.http_response_body_len,
                "http_status_code": stats.http_status_code,
                "http_content_length": http_content_length,
                "http_file_data": stats.http_file_data,
                "http_request_method": stats.http_method,
                "http_request_full_uri": http_uri,
                "http_request_uri": http_uri,
                "http_request_uri_query": http_uri.split("?", 1)[1] if "?" in http_uri else "",
                "http_request_version": stats.http_version,
                "http_response": http_status_text,
                "http_resp_mime_types": "-",
                "http_tls_port": key.dst_port if key.dst_port in {443, 8443} or key.src_port in {443, 8443} else 0,
                "http_user_agent": stats.http_user_agent,
                "http_user_agent_len": len(stats.http_user_agent) if stats.http_user_agent not in {"", "-"} else 0,
                "files_count": files_count,
                "notice_count": 0,
                "weird_count": 0,
                "ssl_count": ssl_count,
                "ssh_count": ssh_count,
                "arp_hw_size": 0,
                "arp_opcode": 0,
                "icmp_checksum": 0,
                "icmp_seq_le": 0,
                "icmp_transmit_timestamp": 0,
                "icmp_unused": 0,
                "mbtcp_len": 0,
                "mbtcp_trans_id": 0,
                "mbtcp_unit_id": 0,
                "mqtt_conack_flags": 0,
                "mqtt_conflag_cleansess": 0,
                "mqtt_conflags": 0,
                "mqtt_hdrflags": 0,
                "mqtt_len": 0,
                "mqtt_msg": "",
                "mqtt_msg_decoded_as": "",
                "mqtt_msgtype": 0,
                "mqtt_proto_len": 0,
                "mqtt_protoname": "",
                "mqtt_topic": "",
                "mqtt_topic_len": 0,
                "mqtt_ver": 0,
                "tcp_ack": stats.tcp_ack,
                "tcp_ack_raw": stats.tcp_ack,
                "tcp_checksum": 0,
                "tcp_connection_fin": int(stats.fin_seen),
                "tcp_connection_rst": int(stats.rst_seen),
                "tcp_connection_syn": int(stats.syn_seen),
                "tcp_connection_synack": int(stats.syn_ack_seen),
                "tcp_dstport": key.dst_port if key.proto == "tcp" else 0,
                "tcp_flags": stats.tcp_flags_numeric,
                "tcp_flags_ack": int(stats.ack_flag_count > 0),
                "tcp_len": stats.tcp_payload_bytes,
                "tcp_options": "",
                "tcp_payload": stats.tcp_payload_bytes,
                "tcp_seq": stats.tcp_seq,
                "tcp_srcport": key.src_port if key.proto == "tcp" else 0,
                "udp_port": key.dst_port if key.proto == "udp" else 0,
                "udp_time_delta": stats.udp_time_delta,
                "flow_bytes_s": _rate(total_bytes, duration),
                "flow_byts_s": _rate(total_bytes, duration),
                "flow_pkts_s": _rate(total_pkts, duration),
                "flow_packets_s": _rate(total_pkts, duration),
                "flow_iat_mean": stats.flow_iat.mean,
                "flow_iat_std": stats.flow_iat.std,
                "flow_iat_max": stats.flow_iat.max_value,
                "flow_iat_min": stats.flow_iat.min_value,
                "fwd_iat_total": stats.src_iat.total,
                "fwd_iat_tot": stats.src_iat.total,
                "fwd_iat_mean": stats.src_iat.mean,
                "fwd_iat_std": stats.src_iat.std,
                "fwd_iat_max": stats.src_iat.max_value,
                "fwd_iat_min": stats.src_iat.min_value,
                "bwd_iat_total": stats.dst_iat.total,
                "bwd_iat_tot": stats.dst_iat.total,
                "bwd_iat_mean": stats.dst_iat.mean,
                "bwd_iat_std": stats.dst_iat.std,
                "bwd_iat_max": stats.dst_iat.max_value,
                "bwd_iat_min": stats.dst_iat.min_value,
                "fwd_psh_flags": stats.fwd_psh_flags,
                "bwd_psh_flags": stats.bwd_psh_flags,
                "fwd_urg_flags": stats.fwd_urg_flags,
                "bwd_urg_flags": stats.bwd_urg_flags,
                "fwd_header_length": stats.src_header_bytes,
                "fwd_header_len": stats.src_header_bytes,
                "bwd_header_length": stats.dst_header_bytes,
                "bwd_header_len": stats.dst_header_bytes,
                "fwd_header_length_1": stats.src_header_bytes,
                "fwd_packets_s": _rate(stats.src_pkts, duration),
                "fwd_pkts_s": _rate(stats.src_pkts, duration),
                "bwd_packets_s": _rate(stats.dst_pkts, duration),
                "bwd_pkts_s": _rate(stats.dst_pkts, duration),
                "min_packet_length": stats.packet_lengths.min_value,
                "max_packet_length": stats.packet_lengths.max_value,
                "packet_length_mean": stats.packet_lengths.mean,
                "packet_length_std": stats.packet_lengths.std,
                "packet_length_variance": stats.packet_lengths.variance,
                "pkt_len_min": stats.packet_lengths.min_value,
                "pkt_len_max": stats.packet_lengths.max_value,
                "pkt_len_mean": stats.packet_lengths.mean,
                "pkt_len_std": stats.packet_lengths.std,
                "pkt_len_var": stats.packet_lengths.variance,
                "fwd_packet_length_min": stats.src_packet_lengths.min_value,
                "fwd_packet_length_max": stats.src_packet_lengths.max_value,
                "fwd_packet_length_mean": stats.src_packet_lengths.mean,
                "fwd_packet_length_std": stats.src_packet_lengths.std,
                "fwd_pkt_len_min": stats.src_packet_lengths.min_value,
                "fwd_pkt_len_max": stats.src_packet_lengths.max_value,
                "fwd_pkt_len_mean": stats.src_packet_lengths.mean,
                "fwd_pkt_len_std": stats.src_packet_lengths.std,
                "bwd_packet_length_min": stats.dst_packet_lengths.min_value,
                "bwd_packet_length_max": stats.dst_packet_lengths.max_value,
                "bwd_packet_length_mean": stats.dst_packet_lengths.mean,
                "bwd_packet_length_std": stats.dst_packet_lengths.std,
                "bwd_pkt_len_min": stats.dst_packet_lengths.min_value,
                "bwd_pkt_len_max": stats.dst_packet_lengths.max_value,
                "bwd_pkt_len_mean": stats.dst_packet_lengths.mean,
                "bwd_pkt_len_std": stats.dst_packet_lengths.std,
                "fin_flag_count": stats.fin_flag_count,
                "syn_flag_count": stats.syn_flag_count,
                "rst_flag_count": stats.rst_flag_count,
                "psh_flag_count": stats.psh_flag_count,
                "ack_flag_count": stats.ack_flag_count,
                "urg_flag_count": stats.urg_flag_count,
                "cwe_flag_count": stats.cwe_flag_count,
                "ece_flag_count": stats.ece_flag_count,
                "fin_flag_cnt": stats.fin_flag_count,
                "syn_flag_cnt": stats.syn_flag_count,
                "rst_flag_cnt": stats.rst_flag_count,
                "psh_flag_cnt": stats.psh_flag_count,
                "ack_flag_cnt": stats.ack_flag_count,
                "urg_flag_cnt": stats.urg_flag_count,
                "ece_flag_cnt": stats.ece_flag_count,
                "down_up_ratio": float(stats.dst_pkts) / float(stats.src_pkts) if stats.src_pkts > 0 else 0.0,
                "average_packet_size": _mean_or_zero(total_bytes, total_pkts),
                "pkt_size_avg": _mean_or_zero(total_bytes, total_pkts),
                "avg_fwd_segment_size": _mean_or_zero(stats.src_bytes, stats.src_pkts),
                "avg_bwd_segment_size": _mean_or_zero(stats.dst_bytes, stats.dst_pkts),
                "fwd_seg_size_avg": _mean_or_zero(stats.src_bytes, stats.src_pkts),
                "bwd_seg_size_avg": _mean_or_zero(stats.dst_bytes, stats.dst_pkts),
                "fwd_avg_bytes_bulk": 0.0,
                "fwd_avg_packets_bulk": 0.0,
                "fwd_avg_bulk_rate": 0.0,
                "bwd_avg_bytes_bulk": 0.0,
                "bwd_avg_packets_bulk": 0.0,
                "bwd_avg_bulk_rate": 0.0,
                "subflow_fwd_packets": stats.src_pkts,
                "subflow_fwd_bytes": stats.src_bytes,
                "subflow_bwd_packets": stats.dst_pkts,
                "subflow_bwd_bytes": stats.dst_bytes,
                "subflow_fwd_pkts": stats.src_pkts,
                "subflow_fwd_byts": stats.src_bytes,
                "subflow_bwd_pkts": stats.dst_pkts,
                "subflow_bwd_byts": stats.dst_bytes,
                "init_win_bytes_forward": stats.init_win_bytes_forward,
                "init_win_bytes_backward": stats.init_win_bytes_backward,
                "init_fwd_win_byts": stats.init_win_bytes_forward,
                "init_bwd_win_byts": stats.init_win_bytes_backward,
                "act_data_pkt_fwd": stats.act_data_pkt_fwd,
                "min_seg_size_forward": stats.min_seg_size_forward,
                "fwd_seg_size_min": stats.min_seg_size_forward,
                "active_mean": stats.active_periods.mean,
                "active_std": stats.active_periods.std,
                "active_max": stats.active_periods.max_value,
                "active_min": stats.active_periods.min_value,
                "idle_mean": stats.idle_periods.mean,
                "idle_std": stats.idle_periods.std,
                "idle_max": stats.idle_periods.max_value,
                "idle_min": stats.idle_periods.min_value,
                "_meta": {
                    "src_ip": key.src_ip,
                    "dst_ip": key.dst_ip,
                    "proto": key.proto,
                    "src_port": key.src_port,
                    "dst_port": key.dst_port,
                },
            }
            records.append(record)

        _add_live_context_features(records)
        return records

    @property
    def active_flows(self) -> int:
        """Return the number of currently tracked flow aggregates."""
        with self._lock:
            return len(self._flows)


_CONTEXT_NUMERIC_SUM_COLS = (
    "duration",
    "orig_bytes",
    "resp_bytes",
    "flow_total_bytes",
    "flow_total_pkts",
    "orig_pkts",
    "resp_pkts",
    "missed_bytes",
    "http_count",
    "dns_count",
    "ssh_count",
    "ssl_count",
    "files_count",
    "notice_count",
    "weird_count",
    "http_uri_has_query",
    "http_uri_has_sql",
    "http_uri_has_xss",
    "http_uri_has_traversal",
    "http_uri_has_cmd",
    "http_uri_has_upload",
)
_CONTEXT_PROTO_VALUES = ("tcp", "udp", "icmp")
_CONTEXT_SERVICE_VALUES = ("http", "dns", "ssh", "ssl", "ftp")
_CONTEXT_STATE_VALUES = ("s0", "sf", "rej", "rstos0", "rstr", "rsto", "sh", "shr", "oth")
_CONTEXT_METHOD_VALUES = ("get", "post", "put", "head", "options")
_CONTEXT_STATUS_VALUES = ("200", "301", "302", "400", "401", "403", "404", "500")
_CONTEXT_PORTS = (20, 21, 22, 23, 53, 80, 123, 443, 502, 1883, 2000, 2323, 3306, 5432, 6379, 8000, 8080, 8443, 44818, 64295)
_CONTEXT_WINDOWS = (5.0, 15.0, 60.0)


def _context_window_name(seconds: float) -> str:
    """Match the Zeek dataset builder's context-window feature names."""
    text = f"{seconds:g}".replace(".", "p")
    return f"{text}s"


def _num(record: dict, key: str) -> float:
    """Read a numeric record field."""
    try:
        return float(record.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def _text(record: dict, key: str) -> str:
    """Read a normalized text record field."""
    return str(record.get(key, "") or "").strip().lower()


def _context_values(record: dict) -> dict[str, float]:
    """Build the per-row basis values used by rolling context features."""
    values = {"_one": 1.0}
    for col in _CONTEXT_NUMERIC_SUM_COLS:
        values[col] = _num(record, col)

    proto = _text(record, "proto")
    service = _text(record, "service")
    state = _zeek_state(str(record.get("conn_state", "")))
    method = _text(record, "http_method")
    status = str(record.get("http_status_code") or record.get("http_response") or "").strip().lower()
    history = _text(record, "history")
    resp_port = int(_num(record, "id_resp_p"))
    orig_port = int(_num(record, "id_orig_p"))

    for value in _CONTEXT_PROTO_VALUES:
        values[f"proto_{value}"] = float(proto == value)
    for value in _CONTEXT_SERVICE_VALUES:
        values[f"service_{value}"] = float(service == value)
    for value in _CONTEXT_STATE_VALUES:
        values[f"state_{value}"] = float(state == value)
    for value in _CONTEXT_METHOD_VALUES:
        values[f"method_{value}"] = float(method == value)
    for value in _CONTEXT_STATUS_VALUES:
        values[f"status_{value}"] = float(status == value)

    values["history_syn"] = float("s" in history)
    values["history_ack"] = float("a" in history)
    values["history_rst"] = float("r" in history)
    values["history_data"] = float("d" in history)
    values["resp_low_port"] = float(0 < resp_port < 1024)
    values["resp_high_port"] = float(resp_port >= 1024)
    values["orig_high_port"] = float(orig_port >= 1024)
    for port in _CONTEXT_PORTS:
        values[f"resp_port_{port}"] = float(resp_port == port)
    return values


def _add_live_context_features(records: list[dict]) -> None:
    """Add causal Zeek-style rolling context features to harvested live flows."""
    if not records:
        return

    ordered = sorted(enumerate(records), key=lambda item: _num(item[1], "ts"))
    row_values = {idx: _context_values(record) for idx, record in ordered}
    basis_keys = list(next(iter(row_values.values())).keys())

    for seconds in _CONTEXT_WINDOWS:
        prefix = f"ctx_{_context_window_name(seconds)}"
        sums = {key: 0.0 for key in basis_keys}
        active: deque[tuple[float, dict[str, float]]] = deque()

        for idx, record in ordered:
            ts_value = _num(record, "ts")
            values = row_values[idx]
            active.append((ts_value, values))
            for key, value in values.items():
                sums[key] += value

            min_ts = ts_value - seconds
            while active and active[0][0] < min_ts:
                _, expired = active.popleft()
                for key, value in expired.items():
                    sums[key] -= value

            flow_count = sums["_one"]
            record[f"{prefix}_flow_count"] = flow_count
            record[f"{prefix}_flow_rate"] = flow_count / seconds
            for col in _CONTEXT_NUMERIC_SUM_COLS:
                value = sums[col]
                record[f"{prefix}_{col}_sum"] = value
                if col in {"flow_total_bytes", "flow_total_pkts"}:
                    record[f"{prefix}_{col}_rate"] = value / seconds
            for key in basis_keys:
                if key == "_one" or key in _CONTEXT_NUMERIC_SUM_COLS:
                    continue
                record[f"{prefix}_{key}_count"] = sums[key]


class TrafficCapture:
    """Sniff packets on one or more interfaces and feed them into the flow table."""

    def __init__(self, *, interface: str, bpf_filter: str = "ip"):
        """Initialize the traffic capture instance."""
        if scapy_sniff is None:
            raise RuntimeError("scapy is required for live capture. Install with: pip install scapy")
        self.interfaces = [i.strip() for i in interface.split(",") if i.strip()]
        self.interface = interface
        self.bpf_filter = bpf_filter
        self.flow_table = FlowTable()
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []

    def _sniff_loop(self, iface: str) -> None:
        """Run the Scapy sniff loop until capture is stopped."""
        log.info("Starting capture on %s (filter: %s)", iface, self.bpf_filter)
        while not self._stop_event.is_set():
            try:
                scapy_sniff(
                    iface=iface,
                    filter=self.bpf_filter,
                    prn=self.flow_table.process_packet,
                    store=False,
                    timeout=1,
                    quiet=True,
                )
            except PermissionError:
                log.error("Permission denied for interface %s. Run with sudo or set CAP_NET_RAW.", iface)
                break
            except OSError as exc:
                log.error("Capture error on %s: %s", iface, exc)
                if not self._stop_event.is_set():
                    time.sleep(1)

    def start(self) -> None:
        """Start packet capture in a background thread."""
        if self._threads and any(t.is_alive() for t in self._threads):
            log.warning("Capture already running")
            return
        self._stop_event.clear()
        self._threads = []
        for iface in self.interfaces:
            thread = threading.Thread(target=self._sniff_loop, args=(iface,), daemon=True, name=f"capture-{iface}")
            thread.start()
            self._threads.append(thread)
            log.info("Capture thread started on %s", iface)

    def stop(self) -> None:
        """Stop packet capture and wait for the sniff thread."""
        self._stop_event.set()
        for thread in self._threads:
            thread.join(timeout=5)
        self._threads = []
        log.info("Capture stopped")

    def harvest(self) -> list[dict]:
        """Return finalized flow records and clear completed aggregates."""
        return self.flow_table.harvest()

    @property
    def running(self) -> bool:
        """Return True while the capture thread is active."""
        return bool(self._threads) and any(thread.is_alive() for thread in self._threads)
