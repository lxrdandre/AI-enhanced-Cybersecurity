"""Build the project 6-class Edge-IIoTset CSV.

The script consumes the raw Edge-IIoTset layout downloaded from Kaggle:

    Edge-IIoTset dataset/
      Attack traffic/*.csv
      Normal traffic/*/*.csv

It maps the original Edge-IIoTset attack files into the project taxonomy:

    backdoor, dos_ddos, injection, normal, password, scanning

MITM and ransomware are excluded by default. The per-class cap is applied after
mapping, with an even quota across source files inside each final class so that
merged classes keep subtype diversity.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_EDGE_ROOT = "/data/datasets/edge_iiotset/Edge-IIoTset dataset"
DEFAULT_PER_CLASS_CAP = 100_000
DEFAULT_CHUNK_SIZE = 100_000
DEFAULT_SEED = 42

TARGET_CLASSES = ["backdoor", "dos_ddos", "injection", "normal", "password", "scanning"]
EXCLUDED_SOURCE_LABELS = {"mitm", "ransomware"}
SPLIT_TIME_COL = "split_time"
SOURCE_LABEL_COL = "source_label"
TIME_CANDIDATES = (
    "frame_time_epoch",
    "timestamp",
    "ts",
    "datetime",
    "frame_time",
    "time",
    "date",
)

ATTACK_FILE_TO_SOURCE_LABEL = {
    "Backdoor_attack.csv": "backdoor",
    "DDoS_HTTP_Flood_attack.csv": "ddos_http",
    "DDoS_ICMP_Flood_attack.csv": "ddos_icmp",
    "DDoS_TCP_SYN_Flood_attack.csv": "ddos_tcp_syn",
    "DDoS_UDP_Flood_attack.csv": "ddos_udp",
    "MITM_attack.csv": "mitm",
    "OS_Fingerprinting_attack.csv": "os_fingerprinting",
    "Password_attack.csv": "password",
    "Port_Scanning_attack.csv": "port_scanning",
    "Ransomware_attack.csv": "ransomware",
    "SQL_injection_attack.csv": "sql_injection",
    "Uploading_attack.csv": "uploading",
    "Vulnerability_scanner_attack.csv": "vulnerability_scanner",
    "XSS_attack.csv": "xss",
}

SOURCE_TO_TARGET_LABEL = {
    "backdoor": "backdoor",
    "ddos_http": "dos_ddos",
    "ddos_icmp": "dos_ddos",
    "ddos_tcp_syn": "dos_ddos",
    "ddos_udp": "dos_ddos",
    "os_fingerprinting": "scanning",
    "password": "password",
    "port_scanning": "scanning",
    "sql_injection": "injection",
    "uploading": "injection",
    "vulnerability_scanner": "scanning",
    "xss": "injection",
}

LABEL_COLUMNS = {
    "attack_label",
    "attack_type",
    "binary_label",
    "category",
    "class",
    "label",
    "multiclass_label",
    "target",
    SPLIT_TIME_COL,
}

IDENTIFIER_COLUMNS = {
    "arp_dst_proto_ipv4",
    "arp_src_proto_ipv4",
    "date",
    "datetime",
    "eth_dst",
    "eth_dst_oui",
    "eth_src",
    "eth_src_oui",
    "frame_number",
    "frame_time",
    "frame_time_delta",
    "frame_time_delta_displayed",
    "frame_time_epoch",
    "frame_time_relative",
    "id_orig_h",
    "id_resp_h",
    "ip_dst",
    "ip_dst_host",
    "ip_src",
    "ip_src_host",
    "ipv6_dst",
    "ipv6_src",
    "src_ip",
    "srcip",
    "dst_ip",
    "dstip",
    "tcp_stream",
    "time",
    "timestamp",
    "ts",
    "udp_stream",
}

PREFERRED_FRONT = [
    "type",
    SPLIT_TIME_COL,
    SOURCE_LABEL_COL,
    "src_port",
    "dst_port",
    "proto",
    "protocol",
    "service",
    "duration",
    "frame_len",
]


@dataclass
class SourceFile:
    """Describe one raw Edge-IIoTSet CSV and its assigned sampling quota."""
    path: Path
    source_label: str
    target_label: str
    normalized_columns: list[str]
    source_order: int
    split_time_source: str = "unknown"
    rows: int = 0
    quota: int = 0


def normalize_text(value: object) -> str:
    """Return lowercase ASCII text with collapsed whitespace."""
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text.strip().lower())
    return text


def normalize_column(name: object) -> str:
    """Return a stable snake_case column name with common aliases applied."""
    normalized = normalize_text(name)
    normalized = normalized.replace("/", "_").replace("-", "_").replace(".", "_")
    normalized = re.sub(r"[^0-9a-z_]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    aliases = {
        "source_ip": "src_ip",
        "destination_ip": "dst_ip",
        "source_port": "src_port",
        "destination_port": "dst_port",
        "protocol_type": "protocol",
    }
    return aliases.get(normalized, normalized)


def should_drop_column(column: str, *, keep_identifiers: bool) -> bool:
    """Return True when a normalized column should be excluded from output."""
    if column in {"type", SPLIT_TIME_COL, SOURCE_LABEL_COL}:
        return False
    if column in LABEL_COLUMNS:
        return True
    if keep_identifiers:
        return False
    if column in IDENTIFIER_COLUMNS:
        return True
    if column.endswith("_ip") or column.endswith("_mac") or column.endswith("_addr"):
        return True
    if ("src" in column or "dst" in column) and column.endswith("_host"):
        return True
    return False


def coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Merge duplicate-named columns by taking the first non-null value."""
    if df.columns.is_unique:
        return df
    merged: dict[str, pd.Series] = {}
    for column in dict.fromkeys(df.columns):
        values = df.loc[:, df.columns == column]
        merged[column] = values.iloc[:, 0] if values.shape[1] == 1 else values.bfill(axis=1).iloc[:, 0]
    return pd.DataFrame(merged)


def derive_split_time_values(
    df: pd.DataFrame,
    *,
    fallback_time_values: pd.Series,
    source: SourceFile,
) -> pd.Series:
    """Return per-row split-time values from the best timestamp signal available."""
    fallback = pd.to_numeric(fallback_time_values, errors="coerce").fillna(0).astype("float64")

    for candidate in TIME_CANDIDATES:
        if candidate not in df.columns:
            continue

        raw = df[candidate].astype(str).str.strip()
        non_empty = raw[raw != ""]
        if len(non_empty) < 3 or non_empty.nunique(dropna=True) < 3:
            continue

        numeric = pd.to_numeric(raw, errors="coerce")
        numeric_valid = int(numeric.notna().sum())
        if numeric_valid >= 3 and numeric.nunique(dropna=True) >= 3:
            values = numeric.astype("float64")
            values = values.where(values.notna(), fallback)
            if source.split_time_source in {"unknown", "source_file_row_order"}:
                source.split_time_source = candidate
            return values.astype(str)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(raw, errors="coerce", utc=True)
        parsed_valid = int(parsed.notna().sum())
        if parsed_valid >= 3 and parsed.nunique(dropna=True) >= 3:
            values = pd.Series(np.nan, index=df.index, dtype="float64")
            values.loc[parsed.notna()] = parsed.loc[parsed.notna()].astype("int64") / 1_000_000_000
            values = values.where(values.notna(), fallback)
            if source.split_time_source in {"unknown", "source_file_row_order"}:
                source.split_time_source = candidate
            return values.astype(str)

    if source.split_time_source == "unknown":
        source.split_time_source = "source_file_row_order"
    return fallback.astype(str)


def resolve_edge_root(edge_root: str) -> Path:
    """Resolve the raw Edge-IIoTSet root directory from supported layouts."""
    root = Path(edge_root).expanduser().resolve()
    if (root / "Attack traffic").is_dir() and (root / "Normal traffic").is_dir():
        return root
    nested = root / "Edge-IIoTset dataset"
    if (nested / "Attack traffic").is_dir() and (nested / "Normal traffic").is_dir():
        return nested
    raise FileNotFoundError(
        "Could not find Edge-IIoTset raw folders. Expected 'Attack traffic' "
        f"and 'Normal traffic' under: {root}"
    )


def default_output_csv(edge_root: Path, cap: int) -> Path:
    """Return the default processed CSV path for the selected cap."""
    cap_name = "all" if cap <= 0 else f"cap{cap // 1000}k"
    return edge_root.parent / "processed" / f"edge_iiotset_6class_{cap_name}.csv"


def source_label_for_file(path: Path, edge_root: Path) -> str | None:
    """Map a raw CSV path to its Edge-IIoTSet source label."""
    try:
        rel_parts = path.relative_to(edge_root).parts
    except ValueError:
        rel_parts = path.parts
    if rel_parts and rel_parts[0] == "Normal traffic":
        if len(rel_parts) >= 2:
            sensor_name = normalize_column(rel_parts[1])
            return f"normal_{sensor_name}" if sensor_name else "normal"
        return "normal"
    return ATTACK_FILE_TO_SOURCE_LABEL.get(path.name)


def normalized_header(path: Path, *, keep_identifiers: bool) -> list[str]:
    """Read and normalize usable header columns without loading full data."""
    header = pd.read_csv(path, nrows=0, dtype=str, low_memory=False, on_bad_lines="skip")
    columns = []
    for raw_col in header.columns:
        col = normalize_column(raw_col)
        if not col or should_drop_column(col, keep_identifiers=keep_identifiers):
            continue
        columns.append(col)
    return list(dict.fromkeys(columns))


def discover_sources(edge_root: Path, *, keep_identifiers: bool) -> list[SourceFile]:
    """Discover usable attack and normal CSV sources under the dataset root."""
    attack_files = sorted((edge_root / "Attack traffic").glob("*.csv"))
    normal_files = sorted((edge_root / "Normal traffic").rglob("*.csv"))
    sources: list[SourceFile] = []

    for source_order, path in enumerate(attack_files + normal_files):
        source_label = source_label_for_file(path, edge_root)
        if source_label is None:
            print(f"Skipping unmapped CSV: {path}")
            continue
        if source_label in EXCLUDED_SOURCE_LABELS:
            print(f"Excluding {source_label}: {path.name}")
            continue
        target_label = "normal" if source_label.startswith("normal") else SOURCE_TO_TARGET_LABEL.get(source_label)
        if target_label not in TARGET_CLASSES:
            print(f"Skipping source label without target mapping: {source_label} ({path})")
            continue
        sources.append(
            SourceFile(
                path=path,
                source_label=source_label,
                target_label=target_label,
                normalized_columns=normalized_header(path, keep_identifiers=keep_identifiers),
                source_order=source_order,
            )
        )

    if not sources:
        raise RuntimeError(f"No usable Edge-IIoTset CSV files found under {edge_root}")
    return sources


def count_rows(path: Path, chunksize: int) -> int:
    """Count data rows in a CSV using chunked reads."""
    rows = 0
    for chunk in pd.read_csv(path, dtype=str, low_memory=False, chunksize=chunksize, on_bad_lines="skip"):
        rows += len(chunk)
    return int(rows)


def allocate_even_quota(counts: list[int], cap: int) -> list[int]:
    """Distribute a per-class cap evenly across source files."""
    quotas = [0 for _ in counts]
    remaining = int(cap)
    active = {idx for idx, count in enumerate(counts) if count > 0}

    while active and remaining > 0:
        share = max(1, remaining // len(active))
        for idx in list(sorted(active)):
            capacity = counts[idx] - quotas[idx]
            take = min(share, capacity, remaining)
            quotas[idx] += take
            remaining -= take
            if quotas[idx] >= counts[idx]:
                active.remove(idx)
            if remaining <= 0:
                break

    return quotas


def allocate_proportional_quota(counts: list[int], cap: int) -> list[int]:
    """Distribute a per-class cap in proportion to source row counts."""
    total = sum(counts)
    if total <= cap:
        return counts[:]

    raw = [count * cap / total for count in counts]
    quotas = [min(count, int(np.floor(value))) for count, value in zip(counts, raw)]
    remaining = cap - sum(quotas)
    order = sorted(
        range(len(counts)),
        key=lambda idx: (raw[idx] - np.floor(raw[idx]), counts[idx]),
        reverse=True,
    )
    while remaining > 0:
        progressed = False
        for idx in order:
            if quotas[idx] >= counts[idx]:
                continue
            quotas[idx] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break
    return quotas


def allocate_quotas(sources: list[SourceFile], *, cap: int, quota_mode: str) -> None:
    """Assign row quotas to every discovered source file."""
    grouped: dict[str, list[SourceFile]] = defaultdict(list)
    for source in sources:
        grouped[source.target_label].append(source)

    for target_label, class_sources in sorted(grouped.items()):
        counts = [source.rows for source in class_sources]
        total = sum(counts)
        if cap <= 0 or total <= cap:
            quotas = counts
        elif quota_mode == "proportional":
            quotas = allocate_proportional_quota(counts, cap)
        else:
            quotas = allocate_even_quota(counts, cap)

        for source, quota in zip(class_sources, quotas):
            source.quota = int(quota)


def output_columns(sources: list[SourceFile]) -> list[str]:
    """Return the deterministic output column order for the processed CSV."""
    columns = {"type", SPLIT_TIME_COL, SOURCE_LABEL_COL}
    for source in sources:
        columns.update(source.normalized_columns)
    front = [col for col in PREFERRED_FRONT if col in columns]
    rest = sorted(col for col in columns if col not in front)
    return front + rest


def clean_chunk(
    chunk: pd.DataFrame,
    *,
    source: SourceFile,
    columns: list[str],
    keep_identifiers: bool,
    fallback_time_values: pd.Series,
) -> pd.DataFrame:
    """Normalize one CSV chunk and add target/source metadata columns."""
    df = chunk.copy()
    df.columns = [normalize_column(col) for col in df.columns]
    df = coalesce_duplicate_columns(df)

    split_time = derive_split_time_values(
        df,
        fallback_time_values=fallback_time_values,
        source=source,
    )

    drop_cols = [col for col in df.columns if should_drop_column(col, keep_identifiers=keep_identifiers)]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")
    df["type"] = source.target_label
    df[SPLIT_TIME_COL] = split_time.to_numpy()
    df[SOURCE_LABEL_COL] = source.source_label

    for col in columns:
        if col not in df.columns:
            df[col] = ""
    df = df[columns]

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
    return df.fillna("")


def selected_positions(total_rows: int, quota: int, rng: np.random.Generator) -> set[int] | None:
    """Return sampled row positions for a source quota, or None for all rows."""
    if quota >= total_rows:
        return None
    positions = rng.choice(total_rows, size=quota, replace=False)
    return set(int(pos) for pos in positions)


def write_source(
    source: SourceFile,
    *,
    output_csv: Path,
    columns: list[str],
    chunksize: int,
    rng: np.random.Generator,
    keep_identifiers: bool,
    write_header: bool,
) -> tuple[int, bool]:
    """Append sampled and normalized rows from one source CSV to the output."""
    if source.quota <= 0:
        return 0, write_header

    keep_positions = selected_positions(source.rows, source.quota, rng)
    position_start = 0
    written = 0

    for chunk in pd.read_csv(source.path, dtype=str, low_memory=False, chunksize=chunksize, on_bad_lines="skip"):
        if keep_positions is None:
            selected = chunk
        else:
            positions = np.arange(position_start, position_start + len(chunk), dtype=np.int64)
            mask = np.array([int(pos) in keep_positions for pos in positions], dtype=bool)
            selected = chunk.loc[mask]
        position_start += len(chunk)

        if selected.empty:
            continue

        selected_row_positions = pd.to_numeric(
            pd.Series(selected.index, index=selected.index),
            errors="coerce",
        ).fillna(0).astype("int64")
        fallback_time_values = pd.Series(
            source.source_order * 1_000_000_000 + selected_row_positions.to_numpy(dtype=np.int64),
            index=selected.index,
            dtype="int64",
        )

        cleaned = clean_chunk(
            selected,
            source=source,
            columns=columns,
            keep_identifiers=keep_identifiers,
            fallback_time_values=fallback_time_values,
        )
        cleaned.to_csv(output_csv, mode="a", index=False, header=write_header)
        write_header = False
        written += len(cleaned)

    return written, write_header


def build_dataset(
    *,
    edge_root: Path,
    output_csv: Path,
    report_json: Path,
    per_class_cap: int,
    chunksize: int,
    seed: int,
    quota_mode: str,
    keep_identifiers: bool,
) -> None:
    """Build the capped six-class Edge-IIoTSet CSV and summary report."""
    sources = discover_sources(edge_root, keep_identifiers=keep_identifiers)

    print("Counting source rows...")
    for source in sources:
        source.rows = count_rows(source.path, chunksize)
        rel_path = source.path.relative_to(edge_root)
        print(f"  {rel_path} -> {source.target_label}/{source.source_label}: {source.rows:,}")

    allocate_quotas(sources, cap=per_class_cap, quota_mode=quota_mode)
    columns = output_columns(sources)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_csv.exists():
        output_csv.unlink()

    rng = np.random.default_rng(seed)
    write_header = True
    written_by_target: Counter = Counter()
    written_by_source: Counter = Counter()
    file_reports = []

    print("\nWriting capped dataset...")
    for source in sources:
        rel_path = str(source.path.relative_to(edge_root))
        written, write_header = write_source(
            source,
            output_csv=output_csv,
            columns=columns,
            chunksize=chunksize,
            rng=rng,
            keep_identifiers=keep_identifiers,
            write_header=write_header,
        )
        written_by_target[source.target_label] += written
        written_by_source[source.source_label] += written
        file_reports.append(
            {
                "file": rel_path,
                "source_label": source.source_label,
                "target_label": source.target_label,
                "rows": int(source.rows),
                "quota": int(source.quota),
                "written": int(written),
                "split_time_source": source.split_time_source,
            }
        )
        print(f"  {rel_path}: wrote {written:,}/{source.rows:,} as {source.target_label}")

    report = {
        "edge_root": str(edge_root),
        "output_csv": str(output_csv),
        "report_json": str(report_json),
        "per_class_cap": int(per_class_cap),
        "quota_mode": quota_mode,
        "seed": int(seed),
        "chunksize": int(chunksize),
        "keep_identifiers": bool(keep_identifiers),
        "target_classes": TARGET_CLASSES,
        "excluded_source_labels": sorted(EXCLUDED_SOURCE_LABELS),
        "source_to_target_label": SOURCE_TO_TARGET_LABEL,
        "columns": columns,
        "column_count": len(columns),
        "class_counts": {label: int(written_by_target.get(label, 0)) for label in TARGET_CLASSES},
        "source_counts": {label: int(count) for label, count in sorted(written_by_source.items())},
        "split_time_column": SPLIT_TIME_COL,
        "source_label_column": SOURCE_LABEL_COL,
        "split_time_note": (
            "split_time and source_label are metadata for source-aware temporal splitting. "
            "The Edge trainer drops both before feature preprocessing."
        ),
        "files": file_reports,
    }

    with report_json.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("\n=== Edge-IIoTset 6-Class Build ===")
    print(f"Output CSV:  {output_csv}")
    print(f"Report JSON: {report_json}")
    print(f"Rows:        {sum(written_by_target.values()):,}")
    print(f"Columns:     {len(columns)}")
    print(f"Classes:     {dict((label, int(written_by_target.get(label, 0))) for label in TARGET_CLASSES)}")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Build the capped 6-class Edge-IIoTset CSV for the project ResNet pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--edge-root",
        default=DEFAULT_EDGE_ROOT,
        help="Path to 'Edge-IIoTset dataset' or its parent directory.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Output CSV path. Defaults to <edge parent>/processed/edge_iiotset_6class_cap100k.csv.",
    )
    parser.add_argument(
        "--report-json",
        default=None,
        help="Report JSON path. Defaults to output CSV stem plus '_report.json'.",
    )
    parser.add_argument(
        "--per-class-cap",
        type=int,
        default=DEFAULT_PER_CLASS_CAP,
        help="Maximum rows per final class after mapping. Use 0 to keep all rows.",
    )
    parser.add_argument(
        "--quota-mode",
        choices=("even", "proportional"),
        default="even",
        help="How to divide a capped final class across its source files.",
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--keep-identifiers",
        action="store_true",
        help="Keep IP/MAC/time/stream identifier columns. By default they are dropped to reduce leakage.",
    )
    return parser


def main() -> None:
    """Run the command-line entry point."""
    args = build_parser().parse_args()
    edge_root = resolve_edge_root(args.edge_root)
    output_csv = Path(args.output_csv).expanduser().resolve() if args.output_csv else default_output_csv(edge_root, args.per_class_cap)
    report_json = (
        Path(args.report_json).expanduser().resolve()
        if args.report_json
        else output_csv.with_name(output_csv.stem + "_report.json")
    )

    build_dataset(
        edge_root=edge_root,
        output_csv=output_csv,
        report_json=report_json,
        per_class_cap=args.per_class_cap,
        chunksize=args.chunk_size,
        seed=args.seed,
        quota_mode=args.quota_mode,
        keep_identifiers=args.keep_identifiers,
    )


if __name__ == "__main__":
    main()
