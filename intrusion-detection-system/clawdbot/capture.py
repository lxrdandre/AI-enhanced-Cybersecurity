"""Live network traffic capture and flow aggregation for ClawdBot IDS.

Extracts the 25 features expected by the SE-DWNet model trained on TON-IoT:
  src_port, dst_port, proto, service, duration, src_bytes, dst_bytes,
  conn_state, missed_bytes, src_pkts, src_ip_bytes, dst_pkts, dst_ip_bytes,
  dns_query, dns_qclass, dns_qtype, dns_rcode, dns_AA,
  http_trans_depth, http_method, http_referrer, http_version,
  http_response_body_len, http_status_code, http_user_agent
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

try:
    from scapy.all import IP, TCP, UDP, ICMP, sniff as scapy_sniff
    from scapy.layers.dns import DNS, DNSQR
    from scapy.layers.http import HTTPRequest, HTTPResponse
except ImportError:
    scapy_sniff = None
    IP = TCP = UDP = ICMP = DNS = DNSQR = HTTPRequest = HTTPResponse = None
    log.warning("scapy not installed — live capture unavailable")


# ── Service inference from well-known ports ──────────────────

_PORT_TO_SERVICE = {
    20: "ftp-data", 21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp",
    53: "dns", 80: "http", 110: "pop3", 143: "imap", 443: "ssl",
    993: "imaps", 995: "pop3s", 8080: "http", 3306: "mysql",
    5432: "postgres", 6379: "redis", 27017: "mongodb",
}


def _infer_service(proto: str, src_port: int, dst_port: int) -> str:
    """Infer Zeek-style 'service' from well-known port numbers."""
    svc = _PORT_TO_SERVICE.get(dst_port) or _PORT_TO_SERVICE.get(src_port)
    if svc:
        return svc
    return "-"


# ── Conn-state inference from TCP flags ──────────────────────

def _infer_conn_state(
    syn_seen: bool, syn_ack_seen: bool, fin_seen: bool, rst_seen: bool,
    src_pkts: int, dst_pkts: int,
) -> str:
    """Approximate Zeek conn_state from observed TCP flags."""
    if rst_seen:
        if syn_seen and not syn_ack_seen:
            return "REJ"       # Connection rejected
        return "RSTR"          # Reset
    if syn_seen and not syn_ack_seen and dst_pkts == 0:
        return "S0"            # SYN only, no reply (scan / filtered)
    if syn_seen and syn_ack_seen and fin_seen:
        return "SF"            # Normal established + finished
    if syn_seen and syn_ack_seen:
        return "S1"            # Established, not finished
    if src_pkts > 0 and dst_pkts == 0:
        return "S0"
    return "OTH"


@dataclass
class FlowKey:
    src_ip: str
    dst_ip: str
    proto: str
    src_port: int = 0
    dst_port: int = 0

    def __hash__(self):
        return hash((self.src_ip, self.dst_ip, self.proto, self.src_port, self.dst_port))

    def __eq__(self, other):
        return (
            self.src_ip == other.src_ip
            and self.dst_ip == other.dst_ip
            and self.proto == other.proto
            and self.src_port == other.src_port
            and self.dst_port == other.dst_port
        )


@dataclass
class FlowStats:
    src_bytes: int = 0
    dst_bytes: int = 0
    src_pkts: int = 0
    dst_pkts: int = 0
    src_ip_bytes: int = 0
    dst_ip_bytes: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0
    # TCP flag tracking
    syn_seen: bool = False
    syn_ack_seen: bool = False
    fin_seen: bool = False
    rst_seen: bool = False
    # DNS fields (first query in the flow)
    dns_query: str = "-"
    dns_qclass: str = "-"
    dns_qtype: str = "-"
    dns_rcode: str = "-"
    dns_AA: int = 0
    # HTTP fields (first request/response in the flow)
    http_method: str = "-"
    http_user_agent: str = "-"
    http_referrer: str = "-"
    http_version: str = "-"
    http_response_body_len: int = 0
    http_status_code: int = 0
    http_trans_depth: int = 0


class FlowTable:
    """Aggregates raw packets into bidirectional flow records with full feature extraction."""

    def __init__(self):
        self._flows: dict[FlowKey, FlowStats] = {}
        self._lock = threading.Lock()

    def process_packet(self, pkt) -> None:
        if IP is None or not pkt.haslayer(IP):
            return

        ip = pkt[IP]
        now = time.time()
        ip_len = ip.len if hasattr(ip, "len") and ip.len else len(pkt)

        src_port = dst_port = 0
        tcp_flags = ""
        if pkt.haslayer(TCP):
            proto = "tcp"
            src_port = pkt[TCP].sport
            dst_port = pkt[TCP].dport
            tcp_flags = str(pkt[TCP].flags)
        elif pkt.haslayer(UDP):
            proto = "udp"
            src_port = pkt[UDP].sport
            dst_port = pkt[UDP].dport
        elif pkt.haslayer(ICMP):
            proto = "icmp"
        else:
            proto = str(ip.proto)

        key = FlowKey(
            src_ip=ip.src,
            dst_ip=ip.dst,
            proto=proto,
            src_port=src_port,
            dst_port=dst_port,
        )

        pkt_len = len(pkt)

        with self._lock:
            if key not in self._flows:
                self._flows[key] = FlowStats(first_seen=now)

            flow = self._flows[key]
            flow.src_bytes += pkt_len
            flow.src_pkts += 1
            flow.src_ip_bytes += ip_len
            flow.last_seen = now

            # Track TCP flags
            if "S" in tcp_flags and "A" not in tcp_flags:
                flow.syn_seen = True
            if "S" in tcp_flags and "A" in tcp_flags:
                flow.syn_ack_seen = True
            if "F" in tcp_flags:
                flow.fin_seen = True
            if "R" in tcp_flags:
                flow.rst_seen = True

            # Extract DNS fields
            if DNS is not None and pkt.haslayer(DNS) and flow.dns_query == "-":
                dns = pkt[DNS]
                if DNSQR is not None and pkt.haslayer(DNSQR):
                    qr = pkt[DNSQR]
                    raw_name = qr.qname
                    if isinstance(raw_name, bytes):
                        raw_name = raw_name.decode("utf-8", errors="replace").rstrip(".")
                    flow.dns_query = str(raw_name) if raw_name else "-"
                    flow.dns_qclass = str(int(qr.qclass)) if hasattr(qr, "qclass") else "-"
                    flow.dns_qtype = str(int(qr.qtype)) if hasattr(qr, "qtype") else "-"
                if hasattr(dns, "rcode"):
                    flow.dns_rcode = str(int(dns.rcode))
                if hasattr(dns, "aa"):
                    flow.dns_AA = int(dns.aa)

            # Extract HTTP request fields
            if HTTPRequest is not None and pkt.haslayer(HTTPRequest) and flow.http_method == "-":
                http_req = pkt[HTTPRequest]
                flow.http_trans_depth += 1
                method = getattr(http_req, "Method", None)
                if method:
                    flow.http_method = method.decode("utf-8", errors="replace") if isinstance(method, bytes) else str(method)
                ua = getattr(http_req, "User_Agent", None)
                if ua:
                    flow.http_user_agent = ua.decode("utf-8", errors="replace") if isinstance(ua, bytes) else str(ua)
                ref = getattr(http_req, "Referer", None)
                if ref:
                    flow.http_referrer = ref.decode("utf-8", errors="replace") if isinstance(ref, bytes) else str(ref)
                ver = getattr(http_req, "Http_Version", None)
                if ver:
                    flow.http_version = ver.decode("utf-8", errors="replace") if isinstance(ver, bytes) else str(ver)

            # Extract HTTP response fields
            if HTTPResponse is not None and pkt.haslayer(HTTPResponse) and flow.http_status_code == 0:
                http_resp = pkt[HTTPResponse]
                code = getattr(http_resp, "Status_Code", None)
                if code:
                    try:
                        flow.http_status_code = int(code)
                    except (ValueError, TypeError):
                        pass
                # Approximate response body length from Content-Length header
                cl = getattr(http_resp, "Content_Length", None)
                if cl:
                    try:
                        flow.http_response_body_len = int(cl)
                    except (ValueError, TypeError):
                        pass

            # Check reverse direction
            rev_key = FlowKey(
                src_ip=ip.dst,
                dst_ip=ip.src,
                proto=proto,
                src_port=dst_port,
                dst_port=src_port,
            )
            if rev_key in self._flows:
                self._flows[rev_key].dst_bytes += pkt_len
                self._flows[rev_key].dst_pkts += 1
                self._flows[rev_key].dst_ip_bytes += ip_len

    def harvest(self) -> list[dict]:
        """Drain the flow table and return records with all 25 model features."""
        with self._lock:
            flows = self._flows
            self._flows = {}

        records = []
        for key, stats in flows.items():
            duration = max(stats.last_seen - stats.first_seen, 0.0)

            # Infer Zeek-style fields
            service = _infer_service(key.proto, key.src_port, key.dst_port)
            conn_state = "-"
            if key.proto == "tcp":
                conn_state = _infer_conn_state(
                    stats.syn_seen, stats.syn_ack_seen,
                    stats.fin_seen, stats.rst_seen,
                    stats.src_pkts, stats.dst_pkts,
                )

            records.append({
                # ── The 25 model features ──
                "src_port": key.src_port,
                "dst_port": key.dst_port,
                "proto": key.proto,
                "service": service,
                "duration": round(duration, 6),
                "src_bytes": stats.src_bytes,
                "dst_bytes": stats.dst_bytes,
                "conn_state": conn_state,
                "missed_bytes": 0,   # not observable without Zeek
                "src_pkts": stats.src_pkts,
                "src_ip_bytes": stats.src_ip_bytes,
                "dst_pkts": stats.dst_pkts,
                "dst_ip_bytes": stats.dst_ip_bytes,
                "dns_query": stats.dns_query,
                "dns_qclass": stats.dns_qclass,
                "dns_qtype": stats.dns_qtype,
                "dns_rcode": stats.dns_rcode,
                "dns_AA": stats.dns_AA,
                "http_trans_depth": stats.http_trans_depth,
                "http_method": stats.http_method,
                "http_referrer": stats.http_referrer,
                "http_version": stats.http_version,
                "http_response_body_len": stats.http_response_body_len,
                "http_status_code": stats.http_status_code,
                "http_user_agent": stats.http_user_agent,
                # ── Metadata (not model features, used for alerting) ──
                "_meta": {
                    "src_ip": key.src_ip,
                    "dst_ip": key.dst_ip,
                    "src_port": key.src_port,
                    "dst_port": key.dst_port,
                },
            })

        return records

    @property
    def active_flows(self) -> int:
        with self._lock:
            return len(self._flows)


class TrafficCapture:
    """Sniffs packets on one or more network interfaces and feeds them into a FlowTable."""

    def __init__(self, *, interface: str, bpf_filter: str = "ip"):
        if scapy_sniff is None:
            raise RuntimeError(
                "scapy is required for live capture. Install with: pip install scapy"
            )
        # Support comma-separated interfaces, e.g. "ens18,wt0"
        self.interfaces = [i.strip() for i in interface.split(",") if i.strip()]
        self.interface = interface  # keep original for logging
        self.bpf_filter = bpf_filter
        self.flow_table = FlowTable()
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []

    def _sniff_loop(self, iface: str) -> None:
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
                log.error(
                    "Permission denied for interface %s. Run with sudo or set CAP_NET_RAW.",
                    iface,
                )
                break
            except OSError as exc:
                log.error("Capture error on %s: %s", iface, exc)
                if not self._stop_event.is_set():
                    time.sleep(1)

    def start(self) -> None:
        if self._threads and any(t.is_alive() for t in self._threads):
            log.warning("Capture already running")
            return
        self._stop_event.clear()
        self._threads = []
        for iface in self.interfaces:
            t = threading.Thread(target=self._sniff_loop, args=(iface,), daemon=True, name=f"capture-{iface}")
            t.start()
            self._threads.append(t)
            log.info("Capture thread started on %s", iface)

    def stop(self) -> None:
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=5)
        self._threads = []
        log.info("Capture stopped")

    def harvest(self) -> list[dict]:
        return self.flow_table.harvest()

    @property
    def running(self) -> bool:
        return bool(self._threads) and any(t.is_alive() for t in self._threads)
