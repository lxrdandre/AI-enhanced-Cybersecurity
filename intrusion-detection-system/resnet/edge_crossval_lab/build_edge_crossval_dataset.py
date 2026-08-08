"""Build a capped Edge-IIoTset-style cross-validation CSV from labelled CSVs.

The default ``edge_like`` distribution mirrors the training build used for the
current SE-DWNet Edge model:

    backdoor: 24,862
    dos_ddos: 100,000
    injection: 100,000
    normal: 100,000
    password: 100,000
    scanning: 100,000

For a smaller validation set, set ``--cap-per-major-class``. With the default
``edge_like`` distribution, backdoor is capped at about 24.9% of that value.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


TARGET_CLASSES = ["backdoor", "dos_ddos", "injection", "normal", "password", "scanning"]
MAJOR_CLASSES = ["dos_ddos", "injection", "normal", "password", "scanning"]
EDGE_BACKDOOR_RATIO = 24862 / 100000
MAX_CAP_PER_CLASS = 60000

EDGE_COLUMNS = [
    "type",
    "source_label",
    "split_time",
    "frame_time_epoch",
    "src_ip",
    "dst_ip",
    "arp_hw_size",
    "arp_opcode",
    "dns_qry_name",
    "dns_qry_name_len",
    "dns_qry_qu",
    "dns_qry_type",
    "dns_retransmission",
    "dns_retransmit_request",
    "dns_retransmit_request_in",
    "http_content_length",
    "http_file_data",
    "http_referer",
    "http_request_full_uri",
    "http_request_method",
    "http_request_uri_query",
    "http_request_version",
    "http_response",
    "http_tls_port",
    "icmp_checksum",
    "icmp_seq_le",
    "icmp_transmit_timestamp",
    "icmp_unused",
    "mbtcp_len",
    "mbtcp_trans_id",
    "mbtcp_unit_id",
    "mqtt_conack_flags",
    "mqtt_conflag_cleansess",
    "mqtt_conflags",
    "mqtt_hdrflags",
    "mqtt_len",
    "mqtt_msg",
    "mqtt_msg_decoded_as",
    "mqtt_msgtype",
    "mqtt_proto_len",
    "mqtt_protoname",
    "mqtt_topic",
    "mqtt_topic_len",
    "mqtt_ver",
    "tcp_ack",
    "tcp_ack_raw",
    "tcp_checksum",
    "tcp_connection_fin",
    "tcp_connection_rst",
    "tcp_connection_syn",
    "tcp_connection_synack",
    "tcp_dstport",
    "tcp_flags",
    "tcp_flags_ack",
    "tcp_len",
    "tcp_options",
    "tcp_payload",
    "tcp_seq",
    "tcp_srcport",
    "udp_port",
    "udp_time_delta",
]

LABEL_ALIASES = {
    "benign": "normal",
    "normal_traffic": "normal",
    "ddos": "dos_ddos",
    "dos": "dos_ddos",
    "dos_ddos": "dos_ddos",
    "ddos_tcp_syn": "dos_ddos",
    "ddos_udp": "dos_ddos",
    "ddos_icmp": "dos_ddos",
    "ddos_http": "dos_ddos",
    "sql_injection": "injection",
    "sqli": "injection",
    "xss": "injection",
    "uploading": "injection",
    "upload": "injection",
    "port_scanning": "scanning",
    "port_scan": "scanning",
    "scan": "scanning",
    "scanner": "scanning",
    "vulnerability_scanner": "scanning",
    "os_fingerprinting": "scanning",
    "password_attack": "password",
    "bruteforce": "password",
    "brute_force": "password",
    "ssh_bruteforce": "password",
    "backdoor_http_c2": "backdoor",
}
pd = None


def require_pandas():
    global pd
    if pd is None:
        try:
            import pandas as pandas_module
        except ModuleNotFoundError as exc:
            raise SystemExit("This script requires pandas. Activate the project/server venv first.") from exc
        pd = pandas_module
    return pd


def normalize_token(value: object) -> str:
    token = str(value).strip().lower()
    token = re.sub(r"[^a-z0-9]+", "_", token).strip("_")
    return LABEL_ALIASES.get(token, token)


def read_input_csvs(paths: list[Path]) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    pandas = require_pandas()
    frames: list[pd.DataFrame] = []
    file_reports: list[dict[str, object]] = []
    for path in paths:
        df = pandas.read_csv(path, low_memory=False)
        if "type" not in df.columns:
            raise ValueError(f"{path} does not contain a 'type' column.")
        if "source_label" not in df.columns:
            df["source_label"] = path.stem
        df["type"] = df["type"].map(normalize_token)
        df["source_label"] = df["source_label"].fillna(path.stem).astype(str)
        before = len(df)
        df = df[df["type"].isin(TARGET_CLASSES)].copy()
        for col in EDGE_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        df = df[EDGE_COLUMNS]
        frames.append(df)
        file_reports.append(
            {
                "file": str(path),
                "rows_read": int(before),
                "rows_kept": int(len(df)),
                "class_counts": {k: int(v) for k, v in Counter(df["type"]).items()},
                "source_counts": {k: int(v) for k, v in Counter(df["source_label"]).items()},
            }
        )
    if not frames:
        raise ValueError("No input CSV files found.")
    return pandas.concat(frames, ignore_index=True), file_reports


def target_caps(args: argparse.Namespace) -> dict[str, int]:
    if args.distribution == "balanced":
        backdoor_cap = args.backdoor_cap or args.cap_per_major_class
    else:
        backdoor_cap = args.backdoor_cap or max(1, round(args.cap_per_major_class * EDGE_BACKDOOR_RATIO))
    caps = {cls: args.cap_per_major_class for cls in MAJOR_CLASSES}
    caps["backdoor"] = int(backdoor_cap)
    return {cls: int(caps[cls]) for cls in TARGET_CLASSES}


def normalize_cap(value: int, arg_name: str) -> int:
    if value < 1:
        raise SystemExit(f"{arg_name} must be at least 1, got: {value}")
    if value > MAX_CAP_PER_CLASS:
        print(f"{arg_name}={value} exceeds the hard cap {MAX_CAP_PER_CLASS}; using {MAX_CAP_PER_CLASS}.")
        return MAX_CAP_PER_CLASS
    return value


def sample_even_by_source(df: pd.DataFrame, target: int, seed: int) -> pd.DataFrame:
    pandas = require_pandas()
    if len(df) <= target:
        return df
    sources = [src for src in sorted(df["source_label"].dropna().astype(str).unique()) if src]
    if not sources:
        return df.sample(n=target, random_state=seed)

    base = target // len(sources)
    remainder = target % len(sources)
    selected_parts: list[pd.DataFrame] = []
    selected_set: set[int] = set()
    for idx, source in enumerate(sources):
        quota = base + (1 if idx < remainder else 0)
        source_df = df[df["source_label"].astype(str) == source]
        take = min(quota, len(source_df))
        if take > 0:
            part = source_df.sample(n=take, random_state=seed + idx)
            selected_parts.append(part)
            selected_set.update(int(i) for i in part.index)

    selected = pandas.concat(selected_parts, ignore_index=False) if selected_parts else df.iloc[0:0]
    shortfall = target - len(selected)
    if shortfall > 0:
        remaining = df.drop(index=list(selected_set), errors="ignore")
        if len(remaining) > 0:
            fill = remaining.sample(n=min(shortfall, len(remaining)), random_state=seed + 999)
            selected = pandas.concat([selected, fill], ignore_index=False)
    return selected


def cap_dataset(df: pd.DataFrame, caps: dict[str, int], quota_mode: str, seed: int) -> pd.DataFrame:
    pandas = require_pandas()
    parts: list[pd.DataFrame] = []
    for class_idx, cls in enumerate(TARGET_CLASSES):
        class_df = df[df["type"] == cls]
        cap = caps[cls]
        if quota_mode == "random":
            sampled = class_df if len(class_df) <= cap else class_df.sample(n=cap, random_state=seed + class_idx)
        else:
            sampled = sample_even_by_source(class_df, cap, seed + class_idx * 100)
        parts.append(sampled)
    return pandas.concat(parts, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=None, help="Directory containing labelled source CSVs.")
    parser.add_argument("--input-csv", type=Path, action="append", default=[], help="Explicit input CSV; can be repeated.")
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    parser.add_argument(
        "--distribution",
        choices=["edge_like", "balanced"],
        default="edge_like",
        help="edge_like keeps backdoor at 24.862%% of the major-class cap.",
    )
    parser.add_argument(
        "--cap-per-major-class",
        "--cap-per-class",
        dest="cap_per_major_class",
        type=int,
        default=MAX_CAP_PER_CLASS,
    )
    parser.add_argument("--backdoor-cap", type=int, default=None)
    parser.add_argument("--quota-mode", choices=["even", "random"], default="even")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-shuffle", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.cap_per_major_class = normalize_cap(args.cap_per_major_class, "--cap-per-major-class")
    if args.backdoor_cap is not None:
        args.backdoor_cap = normalize_cap(args.backdoor_cap, "--backdoor-cap")
    paths = list(args.input_csv)
    if args.input_dir is not None:
        paths.extend(sorted(args.input_dir.glob("*.csv")))
    paths = sorted({path.resolve() for path in paths})
    if not paths:
        raise SystemExit("No input CSV files provided.")

    df, file_reports = read_input_csvs(paths)
    raw_counts = {cls: int((df["type"] == cls).sum()) for cls in TARGET_CLASSES}
    raw_source_counts = {
        cls: {k: int(v) for k, v in Counter(df.loc[df["type"] == cls, "source_label"]).items()}
        for cls in TARGET_CLASSES
    }

    caps = target_caps(args)
    sampled = cap_dataset(df, caps, args.quota_mode, args.seed)
    if not args.no_shuffle:
        sampled = sampled.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(args.output_csv, index=False)

    sampled_counts = {cls: int((sampled["type"] == cls).sum()) for cls in TARGET_CLASSES}
    shortfalls = {
        cls: max(0, int(caps[cls]) - int(sampled_counts[cls]))
        for cls in TARGET_CLASSES
    }
    sampled_source_counts = {
        cls: {k: int(v) for k, v in Counter(sampled.loc[sampled["type"] == cls, "source_label"]).items()}
        for cls in TARGET_CLASSES
    }
    report = {
        "output_csv": str(args.output_csv),
        "rows": int(len(sampled)),
        "columns": int(sampled.shape[1]),
        "distribution": args.distribution,
        "target_caps": caps,
        "raw_class_counts": raw_counts,
        "sampled_class_counts": sampled_counts,
        "shortfalls": shortfalls,
        "raw_source_counts": raw_source_counts,
        "sampled_source_counts": sampled_source_counts,
        "input_files": file_reports,
    }
    args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote CSV: {args.output_csv}")
    print(f"Wrote report: {args.report_json}")
    print(f"Rows: {len(sampled):,}")
    print(f"Target caps: {caps}")
    print(f"Class counts: {sampled_counts}")
    if any(shortfalls.values()):
        print(f"Shortfalls: {shortfalls}")


if __name__ == "__main__":
    main()
