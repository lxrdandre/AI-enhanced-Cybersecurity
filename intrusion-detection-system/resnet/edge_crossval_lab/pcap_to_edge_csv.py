"""Extract Edge-IIoTset-style packet rows from a PCAP with TShark.

The output schema is intentionally aligned with the live SE-DWNet Edge
classifier and with ``validate_edge_iiotset_dataset.py``. It creates one row per
packet, adds ``type`` and ``source_label`` labels, and keeps ``split_time`` from
``frame.time_epoch`` so the file can also be used by temporal diagnostics.

Example:
    python3 resnet/edge_crossval_lab/pcap_to_edge_csv.py \
        --pcap data/edge_crossval/raw/ddos_tcp_syn.pcap \
        --type dos_ddos \
        --source-label ddos_tcp_syn \
        --output data/edge_crossval/csv/ddos_tcp_syn.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import shutil
import subprocess
import sys
from pathlib import Path


TARGET_CLASSES = {"backdoor", "dos_ddos", "injection", "normal", "password", "scanning"}

TSHARK_FIELDS: list[tuple[str, str]] = [
    ("frame.time_epoch", "frame_time_epoch"),
    ("ip.src", "src_ip"),
    ("ip.dst", "dst_ip"),
    ("arp.hw.size", "arp_hw_size"),
    ("arp.opcode", "arp_opcode"),
    ("dns.qry.name", "dns_qry_name"),
    ("dns.qry.name.len", "dns_qry_name_len"),
    ("dns.qry.qu", "dns_qry_qu"),
    ("dns.qry.type", "dns_qry_type"),
    ("dns.retransmission", "dns_retransmission"),
    ("dns.retransmit_request", "dns_retransmit_request"),
    ("dns.retransmit_request_in", "dns_retransmit_request_in"),
    ("http.content_length", "http_content_length"),
    ("http.file_data", "http_file_data"),
    ("http.referer", "http_referer"),
    ("http.request.full_uri", "http_request_full_uri"),
    ("http.request.method", "http_request_method"),
    ("http.request.uri.query", "http_request_uri_query"),
    ("http.request.version", "http_request_version"),
    ("http.response", "http_response"),
    ("http.tls_port", "http_tls_port"),
    ("icmp.checksum", "icmp_checksum"),
    ("icmp.seq_le", "icmp_seq_le"),
    ("icmp.transmit_timestamp", "icmp_transmit_timestamp"),
    ("icmp.unused", "icmp_unused"),
    ("mbtcp.len", "mbtcp_len"),
    ("mbtcp.trans_id", "mbtcp_trans_id"),
    ("mbtcp.unit_id", "mbtcp_unit_id"),
    ("mqtt.conack.flags", "mqtt_conack_flags"),
    ("mqtt.conflag.cleansess", "mqtt_conflag_cleansess"),
    ("mqtt.conflags", "mqtt_conflags"),
    ("mqtt.hdrflags", "mqtt_hdrflags"),
    ("mqtt.len", "mqtt_len"),
    ("mqtt.msg", "mqtt_msg"),
    ("mqtt.msg_decoded_as", "mqtt_msg_decoded_as"),
    ("mqtt.msgtype", "mqtt_msgtype"),
    ("mqtt.proto_len", "mqtt_proto_len"),
    ("mqtt.protoname", "mqtt_protoname"),
    ("mqtt.topic", "mqtt_topic"),
    ("mqtt.topic_len", "mqtt_topic_len"),
    ("mqtt.ver", "mqtt_ver"),
    ("tcp.ack", "tcp_ack"),
    ("tcp.ack_raw", "tcp_ack_raw"),
    ("tcp.checksum", "tcp_checksum"),
    ("tcp.connection.fin", "tcp_connection_fin"),
    ("tcp.connection.rst", "tcp_connection_rst"),
    ("tcp.connection.syn", "tcp_connection_syn"),
    ("tcp.connection.synack", "tcp_connection_synack"),
    ("tcp.dstport", "tcp_dstport"),
    ("tcp.flags", "tcp_flags"),
    ("tcp.flags.ack", "tcp_flags_ack"),
    ("tcp.len", "tcp_len"),
    ("tcp.options", "tcp_options"),
    ("tcp.payload", "tcp_payload"),
    ("tcp.seq", "tcp_seq"),
    ("tcp.srcport", "tcp_srcport"),
    ("udp.port", "udp_port"),
    ("udp.time_delta", "udp_time_delta"),
]

OUTPUT_COLUMNS = ["type", "source_label", "split_time"] + [col for _, col in TSHARK_FIELDS]
LARGE_TSHARK_FIELDS = {
    "http.file_data",
    "tcp.payload",
}
pd = None


def raise_csv_field_limit() -> None:
    """Allow large TShark fields if the user explicitly extracts them."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def require_pandas():
    global pd
    if pd is None:
        try:
            import pandas as pandas_module
        except ModuleNotFoundError as exc:
            raise SystemExit("This script requires pandas. Activate the project/server venv first.") from exc
        pd = pandas_module
    return pd


def supported_tshark_fields(tshark_path: str) -> set[str] | None:
    """Return the local TShark field registry, or None if it cannot be queried."""
    try:
        result = subprocess.run(
            [tshark_path, "-G", "fields"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        print(f"Warning: could not query TShark fields ({exc}); trying all configured fields.", file=sys.stderr)
        return None

    fields: set[str] = set()
    for line in result.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) >= 3 and parts[0] == "F":
            fields.add(parts[2])
    return fields


def select_fields(tshark_path: str, include_large_fields: bool = False) -> list[tuple[str, str]]:
    supported = supported_tshark_fields(tshark_path)
    configured = TSHARK_FIELDS if include_large_fields else [
        item for item in TSHARK_FIELDS if item[0] not in LARGE_TSHARK_FIELDS
    ]
    if not include_large_fields:
        skipped = ", ".join(sorted(LARGE_TSHARK_FIELDS))
        print(f"Skipping large payload fields by default: {skipped}")

    if supported is None:
        return configured
    selected = [(field, col) for field, col in configured if field in supported]
    missing = [field for field, _ in configured if field not in supported]
    if missing:
        print(f"TShark is missing {len(missing)} configured fields; output columns will be blank for them.")
        print("Missing fields: " + ", ".join(missing[:30]) + (" ..." if len(missing) > 30 else ""))
    if not selected:
        raise RuntimeError("No configured TShark fields are supported by this TShark build.")
    return selected


def run_tshark(pcap: Path, tshark_path: str, fields: list[tuple[str, str]]) -> pd.DataFrame:
    raise_csv_field_limit()
    pandas = require_pandas()
    cmd = [
        tshark_path,
        "-r",
        str(pcap),
        "-T",
        "fields",
        "-E",
        "header=y",
        "-E",
        "separator=,",
        "-E",
        "quote=d",
        "-E",
        "occurrence=f",
    ]
    for field, _ in fields:
        cmd.extend(["-e", field])

    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode not in (0, 1):
        raise RuntimeError(f"TShark failed with exit={result.returncode}: {result.stderr.strip()}")
    if not result.stdout.strip():
        return pandas.DataFrame(columns=[col for _, col in TSHARK_FIELDS])

    reader = csv.DictReader(io.StringIO(result.stdout))
    field_to_col = dict(fields)
    rows: list[dict[str, str]] = []
    for raw in reader:
        row = {col: "" for _, col in TSHARK_FIELDS}
        for field, value in raw.items():
            col = field_to_col.get(field)
            if col is not None:
                row[col] = "" if value is None else value
        rows.append(row)
    return pandas.DataFrame(rows, columns=[col for _, col in TSHARK_FIELDS])


def extract_pcap(
    *,
    pcap: Path,
    output: Path,
    type_label: str,
    source_label: str,
    tshark_path: str,
    limit: int | None = None,
    include_large_fields: bool = False,
) -> int:
    if type_label not in TARGET_CLASSES:
        raise ValueError(f"--type must be one of {sorted(TARGET_CLASSES)}, got {type_label!r}")
    if not pcap.exists():
        raise FileNotFoundError(f"PCAP not found: {pcap}")

    fields = select_fields(tshark_path, include_large_fields=include_large_fields)
    df = run_tshark(pcap, tshark_path, fields)
    if limit is not None and limit > 0:
        df = df.head(limit).copy()

    df.insert(0, "split_time", df.get("frame_time_epoch", "").astype(str))
    df.insert(0, "source_label", source_label)
    df.insert(0, "type", type_label)
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[OUTPUT_COLUMNS]

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return int(len(df))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcap", required=True, type=Path)
    parser.add_argument("--type", required=True, dest="type_label", choices=sorted(TARGET_CLASSES))
    parser.add_argument("--source-label", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tshark", default=shutil.which("tshark") or "tshark")
    parser.add_argument("--limit", type=int, default=None, help="Optional max rows to export from this PCAP.")
    parser.add_argument(
        "--include-large-fields",
        action="store_true",
        help="Extract huge payload fields such as http.file_data and tcp.payload. Disabled by default.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = extract_pcap(
        pcap=args.pcap,
        output=args.output,
        type_label=args.type_label,
        source_label=args.source_label,
        tshark_path=args.tshark,
        limit=args.limit,
        include_large_fields=args.include_large_fields,
    )
    print(f"Wrote {rows:,} rows to {args.output}")


if __name__ == "__main__":
    main()
