"""Aggregate labelled Zeek flow rows into short IDS windows.

The raw Zeek CSV has one row per conn.log flow. That is useful for inspection,
but it is a weak supervised-learning target because a capture window label
does not mean every individual flow row is malicious. This script keeps the
same six labels and turns each source PCAP into fixed-size time windows with
rate/count features that match what a live IDS can observe.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


TARGET_CLASSES = ["backdoor", "dos_ddos", "injection", "normal", "password", "scanning"]
DEFAULT_WINDOW_SECONDS = 5.0

pd = None
np = None


def require_deps():
    global pd, np
    if pd is None or np is None:
        try:
            import numpy as numpy_module
            import pandas as pandas_module
        except ModuleNotFoundError as exc:
            raise SystemExit("This script requires pandas and numpy. Activate the project/server venv first.") from exc
        pd = pandas_module
        np = numpy_module
    return pd, np


def read_inputs(paths: list[Path]):
    pandas, _ = require_deps()
    frames = []
    reports = []
    for path in paths:
        df = pandas.read_csv(path, low_memory=False)
        if "type" not in df.columns:
            raise ValueError(f"{path} does not contain a 'type' column.")
        if "source_label" not in df.columns:
            df["source_label"] = path.stem
        before = len(df)
        df["type"] = df["type"].astype(str).str.strip().str.lower()
        df = df[df["type"].isin(TARGET_CLASSES)].copy()
        frames.append(df)
        reports.append(
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
    return pandas.concat(frames, ignore_index=True, sort=False).fillna(""), reports


def to_num(frame, col: str):
    pandas, _ = require_deps()
    if col not in frame.columns:
        return pandas.Series([0.0] * len(frame), index=frame.index, dtype="float64")
    return pandas.to_numeric(frame[col], errors="coerce").fillna(0.0)


def as_text(frame, col: str):
    pandas, _ = require_deps()
    if col not in frame.columns:
        return pandas.Series([""] * len(frame), index=frame.index, dtype="object")
    return frame[col].fillna("").astype(str).str.strip().str.lower()


def add_numeric_stats(row: dict[str, float], frame, col: str) -> None:
    values = to_num(frame, col)
    row[f"{col}_sum"] = float(values.sum())
    row[f"{col}_mean"] = float(values.mean()) if len(values) else 0.0
    row[f"{col}_max"] = float(values.max()) if len(values) else 0.0


def add_value_counts(row: dict[str, float], frame, col: str, values: list[str]) -> None:
    series = as_text(frame, col)
    for value in values:
        row[f"{col}_{value}_count"] = int((series == value).sum())


def add_port_counts(row: dict[str, float], frame, col: str, ports: list[int], prefix: str) -> None:
    values = to_num(frame, col).astype("int64")
    row[f"{prefix}_unique_count"] = int(values[values > 0].nunique())
    row[f"{prefix}_low_port_count"] = int(((values > 0) & (values < 1024)).sum())
    row[f"{prefix}_high_port_count"] = int((values >= 1024).sum())
    for port in ports:
        row[f"{prefix}_{port}_count"] = int((values == port).sum())


def aggregate_window(frame, *, type_label: str, source_label: str, window_start: float, window_seconds: float) -> dict[str, float | str]:
    _, numpy = require_deps()
    flow_count = int(len(frame))
    row: dict[str, float | str] = {
        "type": type_label,
        "source_label": source_label,
        "ts": float(window_start),
        "window_seconds": float(window_seconds),
        "flow_count": flow_count,
        "flow_rate": float(flow_count) / max(window_seconds, 1e-6),
    }

    for col in [
        "duration",
        "orig_bytes",
        "resp_bytes",
        "missed_bytes",
        "orig_pkts",
        "resp_pkts",
        "orig_ip_bytes",
        "resp_ip_bytes",
        "flow_total_bytes",
        "flow_total_pkts",
        "orig_to_resp_bytes_ratio",
        "http_request_body_len",
        "http_response_body_len",
        "dns_answers_count",
        "http_uri_len",
        "http_uri_depth",
        "http_user_agent_len",
        "dns_query_len",
    ]:
        add_numeric_stats(row, frame, col)

    total_bytes = float(to_num(frame, "flow_total_bytes").sum())
    total_pkts = float(to_num(frame, "flow_total_pkts").sum())
    row["bytes_rate"] = total_bytes / max(window_seconds, 1e-6)
    row["pkts_rate"] = total_pkts / max(window_seconds, 1e-6)
    row["bytes_per_flow"] = total_bytes / max(flow_count, 1)
    row["pkts_per_flow"] = total_pkts / max(flow_count, 1)

    add_port_counts(row, frame, "id_orig_p", [20, 21, 22, 23, 53, 80, 123, 443, 502, 1883, 2000, 2323, 3306, 5432, 6379, 8080, 8443, 44818, 64295], "orig_port")
    add_port_counts(row, frame, "id_resp_p", [20, 21, 22, 23, 53, 80, 123, 443, 502, 1883, 2000, 2323, 3306, 5432, 6379, 8080, 8443, 44818, 64295], "resp_port")

    add_value_counts(row, frame, "proto", ["tcp", "udp", "icmp"])
    add_value_counts(row, frame, "service", ["http", "dns", "ssh", "ssl", "ftp", "smtp", "dhcp"])
    add_value_counts(row, frame, "conn_state", ["s0", "s1", "sf", "rej", "rstos0", "rstr", "rsto", "shr", "sh", "oth"])
    add_value_counts(row, frame, "http_method", ["get", "post", "put", "head", "options"])
    add_value_counts(row, frame, "http_status_code", ["200", "301", "302", "400", "401", "403", "404", "500"])
    add_value_counts(row, frame, "dns_qtype_name", ["a", "aaaa", "ptr", "txt", "mx", "srv"])
    add_value_counts(row, frame, "dns_rcode_name", ["noerror", "nxdomain", "servfail", "refused"])

    for col in [
        "src_is_target",
        "dst_is_target",
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
    ]:
        values = to_num(frame, col)
        row[f"{col}_sum"] = float(values.sum())
        row[f"{col}_rate"] = float(values.sum()) / max(window_seconds, 1e-6)

    history = as_text(frame, "history")
    row["history_syn_count"] = int(history.str.contains("s", regex=False).sum())
    row["history_ack_count"] = int(history.str.contains("a", regex=False).sum())
    row["history_data_count"] = int(history.str.contains("d", regex=False).sum())
    row["history_rst_count"] = int(history.str.contains("r", regex=False).sum())

    uri = as_text(frame, "http_uri")
    dns_query = as_text(frame, "dns_query")
    row["http_unique_uri_count"] = int(uri[uri != ""].nunique())
    row["dns_unique_query_count"] = int(dns_query[dns_query != ""].nunique())
    row["http_empty_uri_ratio"] = float((uri == "").mean()) if flow_count else 0.0
    row["dns_empty_query_ratio"] = float((dns_query == "").mean()) if flow_count else 0.0

    numeric_values = [value for key, value in row.items() if key not in {"type", "source_label"}]
    for value in numeric_values:
        if isinstance(value, float) and not numpy.isfinite(value):
            raise ValueError(f"Non-finite aggregate value in {source_label}")
    return row


def build_windows(df, window_seconds: float, min_flows: int):
    pandas, _ = require_deps()
    df = df.copy()
    df["ts_num"] = pandas.to_numeric(df.get("ts", ""), errors="coerce")
    df = df[df["ts_num"].notna()].copy()
    rows = []
    for source_label, source_df in df.groupby("source_label", sort=True):
        source_df = source_df.sort_values("ts_num", kind="mergesort")
        labels = source_df["type"].dropna().astype(str).unique().tolist()
        if len(labels) != 1:
            raise ValueError(f"Source {source_label} has multiple labels: {labels}")
        type_label = labels[0]
        first_ts = float(source_df["ts_num"].min())
        source_df["_window_index"] = ((source_df["ts_num"] - first_ts) // window_seconds).astype("int64")
        for _, window_df in source_df.groupby("_window_index", sort=True):
            if len(window_df) < min_flows:
                continue
            window_start = first_ts + int(window_df["_window_index"].iloc[0]) * window_seconds
            rows.append(
                aggregate_window(
                    window_df,
                    type_label=type_label,
                    source_label=str(source_label),
                    window_start=window_start,
                    window_seconds=window_seconds,
                )
            )
    return pandas.DataFrame(rows).fillna("")


def cap_by_class(df, cap_per_class: int, seed: int):
    pandas, _ = require_deps()
    if cap_per_class <= 0:
        return df
    parts = []
    for class_index, cls in enumerate(TARGET_CLASSES):
        class_df = df[df["type"] == cls]
        if len(class_df) > cap_per_class:
            class_df = class_df.sample(n=cap_per_class, random_state=seed + class_index)
        parts.append(class_df)
    return pandas.concat(parts, ignore_index=True, sort=False).fillna("")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, action="append", default=[])
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    parser.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument("--min-flows", type=int, default=1)
    parser.add_argument("--cap-per-class", type=int, default=0, help="Optional window cap per class. 0 keeps all windows.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-shuffle", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.window_seconds <= 0:
        raise SystemExit("--window-seconds must be positive.")
    paths = list(args.input_csv)
    if args.input_dir is not None:
        paths.extend(sorted(args.input_dir.glob("*.csv")))
    paths = sorted({path.resolve() for path in paths})
    if not paths:
        raise SystemExit("No input CSV files provided.")

    pandas, _ = require_deps()
    df, input_reports = read_inputs(paths)
    raw_counts = {cls: int((df["type"] == cls).sum()) for cls in TARGET_CLASSES}
    windows = build_windows(df, args.window_seconds, args.min_flows)
    windows = cap_by_class(windows, args.cap_per_class, args.seed)
    if not args.no_shuffle and not windows.empty:
        windows = windows.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    windows.to_csv(args.output_csv, index=False)

    window_counts = {cls: int((windows["type"] == cls).sum()) for cls in TARGET_CLASSES}
    report = {
        "output_csv": str(args.output_csv),
        "rows": int(len(windows)),
        "columns": int(windows.shape[1]),
        "window_seconds": float(args.window_seconds),
        "min_flows": int(args.min_flows),
        "cap_per_class": int(args.cap_per_class),
        "raw_flow_counts": raw_counts,
        "window_class_counts": window_counts,
        "source_counts": {
            cls: {k: int(v) for k, v in Counter(windows.loc[windows["type"] == cls, "source_label"]).items()}
            for cls in TARGET_CLASSES
        },
        "input_files": input_reports,
    }
    args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote window CSV: {args.output_csv}")
    print(f"Wrote report: {args.report_json}")
    print(f"Rows: {len(windows):,}")
    print(f"Columns: {windows.shape[1]:,}")
    print(f"Window seconds: {args.window_seconds:g}")
    print(f"Class counts: {window_counts}")


if __name__ == "__main__":
    main()
