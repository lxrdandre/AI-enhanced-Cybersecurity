"""Convert one labelled PCAP into a Zeek-flow CSV.

Zeek is preferred for the retraining dataset because it produces stable
connection/protocol records instead of sparse packet-level fields. The output is
one row per Zeek conn.log flow, enriched with protocol logs joined by uid.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TARGET_CLASSES = {"backdoor", "dos_ddos", "injection", "normal", "password", "scanning"}

BASE_COLUMNS = [
    "type",
    "source_label",
    "uid",
    "ts",
    "datetime",
    "id_orig_h",
    "id_orig_p",
    "id_resp_h",
    "id_resp_p",
    "proto",
    "service",
    "duration",
    "orig_bytes",
    "resp_bytes",
    "conn_state",
    "local_orig",
    "local_resp",
    "missed_bytes",
    "history",
    "orig_pkts",
    "orig_ip_bytes",
    "resp_pkts",
    "resp_ip_bytes",
    "tunnel_parents",
    "flow_total_bytes",
    "flow_total_pkts",
    "orig_to_resp_bytes_ratio",
    "src_is_kali",
    "dst_is_kali",
    "src_is_target",
    "dst_is_target",
]

ENRICH_COLUMNS = [
    "http_count",
    "http_method",
    "http_host",
    "http_uri",
    "http_uri_len",
    "http_uri_depth",
    "http_uri_has_query",
    "http_uri_has_sql",
    "http_uri_has_xss",
    "http_uri_has_traversal",
    "http_uri_has_cmd",
    "http_uri_has_upload",
    "http_referrer",
    "http_user_agent",
    "http_user_agent_len",
    "http_status_code",
    "http_request_body_len",
    "http_response_body_len",
    "http_orig_mime_types",
    "http_resp_mime_types",
    "dns_count",
    "dns_query",
    "dns_query_len",
    "dns_qtype_name",
    "dns_rcode_name",
    "dns_answers_count",
    "ssh_count",
    "ssh_version",
    "ssh_auth_success",
    "ssh_client",
    "ssh_server",
    "ssl_count",
    "ssl_version",
    "ssl_cipher",
    "ssl_server_name",
    "files_count",
    "notice_count",
    "weird_count",
]

OUTPUT_COLUMNS = BASE_COLUMNS + ENRICH_COLUMNS


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "|".join(as_text(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, "", "-"):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, "", "-"):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def ts_to_iso(ts: Any) -> str:
    value = as_float(ts, default=0.0)
    if value <= 0:
        return ""
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat(timespec="milliseconds")


def load_json_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def first_nonempty(rows: list[dict[str, Any]], *keys: str) -> str:
    for row in rows:
        for key in keys:
            value = as_text(row.get(key)).strip()
            if value and value != "-":
                return value
    return ""


def contains_any(text: str, patterns: tuple[str, ...]) -> int:
    lower = text.lower()
    return int(any(pattern in lower for pattern in patterns))


def regex_any(text: str, patterns: tuple[str, ...]) -> int:
    lower = text.lower()
    return int(any(re.search(pattern, lower) for pattern in patterns))


def uri_depth(uri: str) -> int:
    path = uri.split("?", 1)[0]
    return len([part for part in path.split("/") if part])


SQL_PATTERNS = (
    " union ",
    "union%20",
    " select ",
    "select%20",
    " or 1=1",
    "or%201%3d1",
    "' or ",
    "%27%20or",
    "\" or ",
    "%22%20or",
    "pg_sleep",
    "sleep(",
    "benchmark(",
    "sqlmap",
    "@@version",
)
XSS_PATTERNS = (
    "<script",
    "%3cscript",
    "onerror",
    "onload",
    "alert(",
    "javascript:",
    "<svg",
    "%3csvg",
)
TRAVERSAL_PATTERNS = (
    "../",
    "..\\",
    "%2e%2e",
    "/etc/passwd",
    "boot.ini",
    "win.ini",
)
CMD_LITERAL_PATTERNS = (
    "whoami",
    "uname",
    "/bin/sh",
    "cmd.exe",
    "powershell",
    "wget ",
    "curl ",
    "cat /etc/passwd",
)
CMD_REGEX_PATTERNS = (
    r"(^|[?&=;%0a])id($|[&;|])",
    r"(%3b|;|%7c|\||%26%26|&&)",
)
UPLOAD_PATTERNS = (
    "upload",
    "firmware",
    "multipart",
    ".php",
    "filename=",
)


def count_by_uid(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        uid = as_text(row.get("uid")).strip()
        if uid:
            counts[uid] += 1
    return counts


def group_by_uid(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        uid = as_text(row.get("uid")).strip()
        if uid:
            grouped[uid].append(row)
    return grouped


def run_zeek(pcap: Path, zeek_bin: str, workdir: Path) -> None:
    cmd = [
        zeek_bin,
        "-Cr",
        str(pcap),
        "LogAscii::use_json=T",
        "LogAscii::json_timestamps=JSON::TS_EPOCH",
    ]
    result = subprocess.run(cmd, cwd=workdir, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Zeek failed with exit="
            + str(result.returncode)
            + "\nSTDERR:\n"
            + result.stderr.strip()
            + "\nSTDOUT:\n"
            + result.stdout.strip()
        )


def build_rows(
    *,
    workdir: Path,
    type_label: str,
    source_label: str,
    target_ip: str,
    kali_ip: str,
    ssh_port: str,
    canonical_ssh_port: str,
) -> list[dict[str, Any]]:
    conn_rows = load_json_log(workdir / "conn.log")
    http_by_uid = group_by_uid(load_json_log(workdir / "http.log"))
    dns_by_uid = group_by_uid(load_json_log(workdir / "dns.log"))
    ssh_by_uid = group_by_uid(load_json_log(workdir / "ssh.log"))
    ssl_by_uid = group_by_uid(load_json_log(workdir / "ssl.log"))
    files_count = count_by_uid(load_json_log(workdir / "files.log"))
    notice_count = count_by_uid(load_json_log(workdir / "notice.log"))
    weird_count = count_by_uid(load_json_log(workdir / "weird.log"))

    rows: list[dict[str, Any]] = []
    for conn in conn_rows:
        uid = as_text(conn.get("uid"))
        orig_h = as_text(conn.get("id.orig_h"))
        resp_h = as_text(conn.get("id.resp_h"))
        orig_p = as_text(conn.get("id.orig_p"))
        resp_p = as_text(conn.get("id.resp_p"))
        if canonical_ssh_port and ssh_port and resp_h == target_ip and resp_p == ssh_port:
            resp_p = canonical_ssh_port
        if canonical_ssh_port and ssh_port and orig_h == target_ip and orig_p == ssh_port:
            orig_p = canonical_ssh_port
        orig_bytes = as_float(conn.get("orig_bytes"))
        resp_bytes = as_float(conn.get("resp_bytes"))
        orig_pkts = as_float(conn.get("orig_pkts"))
        resp_pkts = as_float(conn.get("resp_pkts"))

        http_rows = http_by_uid.get(uid, [])
        dns_rows = dns_by_uid.get(uid, [])
        ssh_rows = ssh_by_uid.get(uid, [])
        ssl_rows = ssl_by_uid.get(uid, [])

        dns_answers = 0
        for dns in dns_rows:
            answers = dns.get("answers")
            if isinstance(answers, list):
                dns_answers += len(answers)
            elif as_text(answers):
                dns_answers += 1

        http_uri = first_nonempty(http_rows, "uri")
        http_referrer = first_nonempty(http_rows, "referrer")
        http_user_agent = first_nonempty(http_rows, "user_agent")
        http_method = first_nonempty(http_rows, "method")
        dns_query = first_nonempty(dns_rows, "query")
        http_signal_text = " ".join([http_uri, http_referrer, http_user_agent, http_method])

        row = {
            "type": type_label,
            "source_label": source_label,
            "uid": uid,
            "ts": as_text(conn.get("ts")),
            "datetime": ts_to_iso(conn.get("ts")),
            "id_orig_h": orig_h,
            "id_orig_p": orig_p,
            "id_resp_h": resp_h,
            "id_resp_p": resp_p,
            "proto": as_text(conn.get("proto")),
            "service": as_text(conn.get("service")),
            "duration": as_text(conn.get("duration")),
            "orig_bytes": as_text(conn.get("orig_bytes")),
            "resp_bytes": as_text(conn.get("resp_bytes")),
            "conn_state": as_text(conn.get("conn_state")),
            "local_orig": as_text(conn.get("local_orig")),
            "local_resp": as_text(conn.get("local_resp")),
            "missed_bytes": as_text(conn.get("missed_bytes")),
            "history": as_text(conn.get("history")),
            "orig_pkts": as_text(conn.get("orig_pkts")),
            "orig_ip_bytes": as_text(conn.get("orig_ip_bytes")),
            "resp_pkts": as_text(conn.get("resp_pkts")),
            "resp_ip_bytes": as_text(conn.get("resp_ip_bytes")),
            "tunnel_parents": as_text(conn.get("tunnel_parents")),
            "flow_total_bytes": orig_bytes + resp_bytes,
            "flow_total_pkts": orig_pkts + resp_pkts,
            "orig_to_resp_bytes_ratio": orig_bytes / (resp_bytes + 1.0),
            "src_is_kali": int(bool(kali_ip) and orig_h == kali_ip),
            "dst_is_kali": int(bool(kali_ip) and resp_h == kali_ip),
            "src_is_target": int(bool(target_ip) and orig_h == target_ip),
            "dst_is_target": int(bool(target_ip) and resp_h == target_ip),
            "http_count": len(http_rows),
            "http_method": http_method,
            "http_host": first_nonempty(http_rows, "host"),
            "http_uri": http_uri,
            "http_uri_len": len(http_uri),
            "http_uri_depth": uri_depth(http_uri),
            "http_uri_has_query": int("?" in http_uri),
            "http_uri_has_sql": contains_any(http_signal_text, SQL_PATTERNS),
            "http_uri_has_xss": contains_any(http_signal_text, XSS_PATTERNS),
            "http_uri_has_traversal": contains_any(http_signal_text, TRAVERSAL_PATTERNS),
            "http_uri_has_cmd": int(contains_any(http_signal_text, CMD_LITERAL_PATTERNS) or regex_any(http_signal_text, CMD_REGEX_PATTERNS)),
            "http_uri_has_upload": contains_any(http_signal_text, UPLOAD_PATTERNS),
            "http_referrer": http_referrer,
            "http_user_agent": http_user_agent,
            "http_user_agent_len": len(http_user_agent),
            "http_status_code": first_nonempty(http_rows, "status_code"),
            "http_request_body_len": first_nonempty(http_rows, "request_body_len"),
            "http_response_body_len": first_nonempty(http_rows, "response_body_len"),
            "http_orig_mime_types": first_nonempty(http_rows, "orig_mime_types"),
            "http_resp_mime_types": first_nonempty(http_rows, "resp_mime_types"),
            "dns_count": len(dns_rows),
            "dns_query": dns_query,
            "dns_query_len": len(dns_query),
            "dns_qtype_name": first_nonempty(dns_rows, "qtype_name"),
            "dns_rcode_name": first_nonempty(dns_rows, "rcode_name"),
            "dns_answers_count": dns_answers,
            "ssh_count": len(ssh_rows),
            "ssh_version": first_nonempty(ssh_rows, "version"),
            "ssh_auth_success": first_nonempty(ssh_rows, "auth_success"),
            "ssh_client": first_nonempty(ssh_rows, "client"),
            "ssh_server": first_nonempty(ssh_rows, "server"),
            "ssl_count": len(ssl_rows),
            "ssl_version": first_nonempty(ssl_rows, "version"),
            "ssl_cipher": first_nonempty(ssl_rows, "cipher"),
            "ssl_server_name": first_nonempty(ssl_rows, "server_name"),
            "files_count": files_count.get(uid, 0),
            "notice_count": notice_count.get(uid, 0),
            "weird_count": weird_count.get(uid, 0),
        }
        rows.append(row)
    return rows


def extract_pcap(
    *,
    pcap: Path,
    output: Path,
    type_label: str,
    source_label: str,
    zeek_bin: str,
    target_ip: str,
    kali_ip: str,
    ssh_port: str = "",
    canonical_ssh_port: str = "",
    limit: int | None = None,
    keep_zeek_logs: Path | None = None,
) -> int:
    if type_label not in TARGET_CLASSES:
        raise ValueError(f"--type must be one of {sorted(TARGET_CLASSES)}, got {type_label!r}")
    if not pcap.exists():
        raise FileNotFoundError(f"PCAP not found: {pcap}")
    pcap = pcap.expanduser().resolve()

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="zeek_extract_") as tmp:
        workdir = Path(tmp)
        run_zeek(pcap, zeek_bin, workdir)
        rows = build_rows(
            workdir=workdir,
            type_label=type_label,
            source_label=source_label,
            target_ip=target_ip,
            kali_ip=kali_ip,
            ssh_port=ssh_port,
            canonical_ssh_port=canonical_ssh_port,
        )
        if limit is not None and limit > 0:
            rows = rows[:limit]
        if keep_zeek_logs is not None:
            dst = keep_zeek_logs / source_label
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(workdir, dst)

    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: as_text(row.get(col)) for col in OUTPUT_COLUMNS})
    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcap", required=True, type=Path)
    parser.add_argument("--type", required=True, dest="type_label", choices=sorted(TARGET_CLASSES))
    parser.add_argument("--source-label", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--zeek", default=shutil.which("zeek") or "zeek")
    parser.add_argument("--target-ip", default=os.environ.get("TARGET_IP", ""))
    parser.add_argument("--kali-ip", default=os.environ.get("KALI_IP", ""))
    parser.add_argument("--ssh-port", default=os.environ.get("SSH_PORT", "64295"))
    parser.add_argument("--canonical-ssh-port", default=os.environ.get("CANONICAL_SSH_PORT", "22"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--keep-zeek-logs", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = extract_pcap(
        pcap=args.pcap,
        output=args.output,
        type_label=args.type_label,
        source_label=args.source_label,
        zeek_bin=args.zeek,
        target_ip=args.target_ip,
        kali_ip=args.kali_ip,
        ssh_port=str(args.ssh_port),
        canonical_ssh_port=str(args.canonical_ssh_port),
        limit=args.limit,
        keep_zeek_logs=args.keep_zeek_logs,
    )
    print(f"Wrote {rows:,} Zeek flow rows to {args.output}")


if __name__ == "__main__":
    main()
