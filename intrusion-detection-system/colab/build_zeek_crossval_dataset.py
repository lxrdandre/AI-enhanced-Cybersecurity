"""Build a capped Zeek-flow cross-validation/training CSV from source CSVs.

By default the builder adds causal rolling context features to every flow row.
That keeps the raw Zeek row count high while giving the classifier local IDS
context such as recent flow rates, protocol mix, connection states, and attack
payload flags.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


TARGET_CLASSES = ["backdoor", "dos_ddos", "injection", "normal", "password", "scanning"]
MAX_CAP_PER_CLASS = 60000
DEFAULT_CONTEXT_WINDOWS = "5,15,60"

LABEL_ALIASES = {
    "benign": "normal",
    "normal_traffic": "normal",
    "ddos": "dos_ddos",
    "dos": "dos_ddos",
    "sqli": "injection",
    "sql_injection": "injection",
    "xss": "injection",
    "upload": "injection",
    "uploading": "injection",
    "scan": "scanning",
    "scanner": "scanning",
    "port_scan": "scanning",
    "port_scanning": "scanning",
    "vulnerability_scanner": "scanning",
    "os_fingerprinting": "scanning",
    "bruteforce": "password",
    "brute_force": "password",
    "ssh_bruteforce": "password",
    "password_attack": "password",
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


def read_inputs(paths: list[Path]):
    pandas = require_pandas()
    frames = []
    reports = []
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


def parse_windows(raw: str) -> list[float]:
    windows: list[float] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if value <= 0:
            raise SystemExit(f"Context window seconds must be positive, got: {part}")
        windows.append(value)
    return windows


def window_name(seconds: float) -> str:
    text = f"{seconds:g}".replace(".", "p")
    return f"{text}s"


def numeric_series(df, col: str):
    pandas = require_pandas()
    if col not in df.columns:
        return pandas.Series([0.0] * len(df), index=df.index, dtype="float64")
    return pandas.to_numeric(df[col], errors="coerce").fillna(0.0).astype("float64")


def text_series(df, col: str):
    pandas = require_pandas()
    if col not in df.columns:
        return pandas.Series([""] * len(df), index=df.index, dtype="object")
    return df[col].fillna("").astype(str).str.strip().str.lower()


def rolling_sum(series, window_seconds: float):
    pandas = require_pandas()
    return series.rolling(pandas.Timedelta(seconds=window_seconds)).sum().fillna(0.0).to_numpy()


def add_rolling_context_features(df, windows: list[float]):
    pandas = require_pandas()
    if not windows or "source_label" not in df.columns or "ts" not in df.columns:
        return df

    df = df.copy()
    df["_ts_num"] = pandas.to_numeric(df["ts"], errors="coerce")
    valid_ts = df["_ts_num"].notna()
    if not valid_ts.any():
        df.drop(columns=["_ts_num"], inplace=True, errors="ignore")
        return df

    print(f"Adding causal rolling context features for windows: {', '.join(f'{w:g}s' for w in windows)}")
    df["_orig_order"] = range(len(df))
    parts = []
    numeric_sum_cols = [
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
    ]
    proto_values = ["tcp", "udp", "icmp"]
    service_values = ["http", "dns", "ssh", "ssl", "ftp"]
    state_values = ["s0", "sf", "rej", "rstos0", "rstr", "rsto", "sh", "shr", "oth"]
    method_values = ["get", "post", "put", "head", "options"]
    status_values = ["200", "301", "302", "400", "401", "403", "404", "500"]
    ports = [20, 21, 22, 23, 53, 80, 123, 443, 502, 1883, 2000, 2323, 3306, 5432, 6379, 8000, 8080, 8443, 44818, 64295]

    for source, group in df.groupby("source_label", sort=False):
        group = group.sort_values(["_ts_num", "_orig_order"], kind="mergesort").copy()
        invalid = group["_ts_num"].isna()
        if invalid.any():
            bad = group.loc[invalid].copy()
            for seconds in windows:
                prefix = f"ctx_{window_name(seconds)}"
                bad[f"{prefix}_flow_count"] = 1.0
                bad[f"{prefix}_flow_rate"] = 1.0 / seconds
            parts.append(bad)
            group = group.loc[~invalid].copy()
        if group.empty:
            continue

        time_index = pandas.to_datetime(group["_ts_num"], unit="s", utc=True)
        work = pandas.DataFrame(index=time_index)
        work["_one"] = 1.0
        for col in numeric_sum_cols:
            work[col] = numeric_series(group, col).to_numpy()

        proto = text_series(group, "proto")
        service = text_series(group, "service")
        state = text_series(group, "conn_state")
        method = text_series(group, "http_method")
        status = text_series(group, "http_status_code")
        history = text_series(group, "history")
        resp_p = numeric_series(group, "id_resp_p").astype("int64")
        orig_p = numeric_series(group, "id_orig_p").astype("int64")

        for value in proto_values:
            work[f"proto_{value}"] = (proto == value).astype("float64").to_numpy()
        for value in service_values:
            work[f"service_{value}"] = (service == value).astype("float64").to_numpy()
        for value in state_values:
            work[f"state_{value}"] = (state == value).astype("float64").to_numpy()
        for value in method_values:
            work[f"method_{value}"] = (method == value).astype("float64").to_numpy()
        for value in status_values:
            work[f"status_{value}"] = (status == value).astype("float64").to_numpy()
        work["history_syn"] = history.str.contains("s", regex=False).astype("float64").to_numpy()
        work["history_ack"] = history.str.contains("a", regex=False).astype("float64").to_numpy()
        work["history_rst"] = history.str.contains("r", regex=False).astype("float64").to_numpy()
        work["history_data"] = history.str.contains("d", regex=False).astype("float64").to_numpy()
        work["resp_low_port"] = ((resp_p > 0) & (resp_p < 1024)).astype("float64").to_numpy()
        work["resp_high_port"] = (resp_p >= 1024).astype("float64").to_numpy()
        work["orig_high_port"] = (orig_p >= 1024).astype("float64").to_numpy()
        for port in ports:
            work[f"resp_port_{port}"] = (resp_p == port).astype("float64").to_numpy()

        context_columns = {}
        for seconds in windows:
            prefix = f"ctx_{window_name(seconds)}"
            flow_count = rolling_sum(work["_one"], seconds)
            context_columns[f"{prefix}_flow_count"] = flow_count
            context_columns[f"{prefix}_flow_rate"] = flow_count / seconds
            for col in numeric_sum_cols:
                values = rolling_sum(work[col], seconds)
                context_columns[f"{prefix}_{col}_sum"] = values
                if col in {"flow_total_bytes", "flow_total_pkts"}:
                    context_columns[f"{prefix}_{col}_rate"] = values / seconds
            for col in work.columns:
                if col == "_one" or col in numeric_sum_cols:
                    continue
                context_columns[f"{prefix}_{col}_count"] = rolling_sum(work[col], seconds)

        if context_columns:
            context_df = pandas.DataFrame(context_columns, index=group.index)
            group = pandas.concat([group, context_df], axis=1).copy()

        parts.append(group)

    out = pandas.concat(parts, ignore_index=False, sort=False)
    out = out.sort_values("_orig_order", kind="mergesort").drop(columns=["_ts_num", "_orig_order"], errors="ignore")
    return out.reset_index(drop=True).fillna("")


def sample_even_by_source(df, target: int, seed: int):
    pandas = require_pandas()
    if len(df) <= target:
        return df
    sources = [src for src in sorted(df["source_label"].dropna().astype(str).unique()) if src]
    if not sources:
        return df.sample(n=target, random_state=seed)

    base = target // len(sources)
    remainder = target % len(sources)
    selected_parts = []
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


def cap_dataset(df, cap_per_class: int, quota_mode: str, seed: int):
    pandas = require_pandas()
    parts = []
    for class_idx, cls in enumerate(TARGET_CLASSES):
        class_df = df[df["type"] == cls]
        if quota_mode == "random":
            sampled = (
                class_df
                if len(class_df) <= cap_per_class
                else class_df.sample(n=cap_per_class, random_state=seed + class_idx)
            )
        else:
            sampled = sample_even_by_source(class_df, cap_per_class, seed + class_idx * 100)
        parts.append(sampled)
    return pandas.concat(parts, ignore_index=True, sort=False).fillna("")


def normalize_cap(value: int, arg_name: str) -> int:
    if value < 1:
        raise SystemExit(f"{arg_name} must be at least 1, got: {value}")
    if value != MAX_CAP_PER_CLASS:
        print(f"{arg_name}={value} differs from the required cap {MAX_CAP_PER_CLASS}; using {MAX_CAP_PER_CLASS}.")
        return MAX_CAP_PER_CLASS
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--input-csv", type=Path, action="append", default=[])
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    parser.add_argument(
        "--cap-per-class",
        "--cap-per-major-class",
        dest="cap_per_class",
        type=int,
        default=MAX_CAP_PER_CLASS,
    )
    parser.add_argument("--quota-mode", choices=["even", "random"], default="even")
    parser.add_argument(
        "--context-windows",
        default=DEFAULT_CONTEXT_WINDOWS,
        help="Comma-separated causal rolling windows in seconds. Use empty string with --no-context to disable.",
    )
    parser.add_argument("--no-context", action="store_true", help="Do not add rolling context features.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-shuffle", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.cap_per_class = normalize_cap(args.cap_per_class, "--cap-per-class")
    paths = list(args.input_csv)
    if args.input_dir is not None:
        paths.extend(sorted(args.input_dir.glob("*.csv")))
    paths = sorted({path.resolve() for path in paths})
    if not paths:
        raise SystemExit("No input CSV files provided.")

    df, file_reports = read_inputs(paths)
    context_windows = [] if args.no_context else parse_windows(args.context_windows)
    if context_windows:
        df = add_rolling_context_features(df, context_windows)
    raw_counts = {cls: int((df["type"] == cls).sum()) for cls in TARGET_CLASSES}
    raw_source_counts = {
        cls: {k: int(v) for k, v in Counter(df.loc[df["type"] == cls, "source_label"]).items()}
        for cls in TARGET_CLASSES
    }
    sampled = cap_dataset(df, args.cap_per_class, args.quota_mode, args.seed)
    if not args.no_shuffle:
        sampled = sampled.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(args.output_csv, index=False)

    sampled_counts = {cls: int((sampled["type"] == cls).sum()) for cls in TARGET_CLASSES}
    shortfalls = {
        cls: max(0, int(args.cap_per_class) - int(sampled_counts[cls]))
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
        "cap_per_class": int(args.cap_per_class),
        "context_windows_seconds": [float(value) for value in context_windows],
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
    print(f"Cap per class: {args.cap_per_class:,}")
    if context_windows:
        print(f"Context windows: {[f'{value:g}s' for value in context_windows]}")
    print(f"Class counts: {sampled_counts}")
    if any(shortfalls.values()):
        print(f"Shortfalls: {shortfalls}")


if __name__ == "__main__":
    main()
