from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.environ.get("TON_IOT_PROJECT_ROOT") or os.path.normpath(os.path.join(SCRIPT_DIR, ".."))

DEFAULT_CIC2017_DIR = os.path.join(PROJECT_ROOT, "data", "cicids2017")
DEFAULT_CIC2018_DIR = os.path.join(PROJECT_ROOT, "data", "cicids2018")
DEFAULT_OUTPUT_CSV = os.path.join(PROJECT_ROOT, "data", "cic_public_6class.csv")
DEFAULT_PER_CLASS_CAP = 800_000
DEFAULT_CHUNK_SIZE = 100_000

LABEL_CANDIDATES = ("Label", "label", "Attack", "attack", "type")
DROP_COLUMNS = {"flow_id", "raw_label"}
PREFERRED_FRONT = ["type", "ts", "src_ip", "dst_ip", "src_port", "dst_port", "proto", "duration"]
CANONICAL_CLASSES = ["backdoor", "dos_ddos", "infiltration", "normal", "password", "scanning"]
LOWERCASE_VALUE_COLS = {"proto"}

COLUMN_ALIASES = {
    "source_ip": "src_ip",
    "destination_ip": "dst_ip",
    "source_port": "src_port",
    "destination_port": "dst_port",
    "protocol": "proto",
    "timestamp": "ts",
    "flow_duration": "duration",
    "tot_fwd_pkts": "src_pkts",
    "total_fwd_packets": "src_pkts",
    "tot_bwd_pkts": "dst_pkts",
    "total_backward_packets": "dst_pkts",
    "totlen_fwd_pkts": "src_bytes",
    "total_length_of_fwd_packets": "src_bytes",
    "totlen_bwd_pkts": "dst_bytes",
    "total_length_of_bwd_packets": "dst_bytes",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a public CIC dataset with 6 classes tuned for this project.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cic2017-dir", default=DEFAULT_CIC2017_DIR, help="Root folder for CIC-IDS2017 CSV files.")
    parser.add_argument("--cic2018-dir", default=DEFAULT_CIC2018_DIR, help="Root folder for CSE-CIC-IDS2018 CSV files.")
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV, help="Merged output CSV path.")
    parser.add_argument("--per-class-cap", type=int, default=DEFAULT_PER_CLASS_CAP, help="Cap per final class while building. 0 keeps all rows.")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Rows per pandas chunk while streaming CSVs.")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    text = re.sub(r"\s+", " ", text.strip().lower())
    return text


def normalize_column(name: str) -> str:
    normalized = normalize_text(name)
    normalized = normalized.replace("/", "_").replace("-", "_")
    normalized = re.sub(r"[^0-9a-z_]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return COLUMN_ALIASES.get(normalized, normalized)


def map_public_label(raw_label: object) -> str | None:
    label = normalize_text(raw_label)

    if label in {"benign", "normal"}:
        return "normal"
    if "infiltration" in label or "infilteration" in label:
        return "infiltration"
    if "portscan" in label or "port scan" in label:
        return "scanning"
    if label == "bot" or "botnet" in label or "backdoor" in label:
        return "backdoor"
    if any(token in label for token in ("ftp-patator", "ssh-patator", "ftp-bruteforce", "ssh-bruteforce")):
        return "password"
    if any(token in label for token in ("xss", "sql injection", "command injection", "web attack brute force", "brute force -web")):
        return None
    if any(token in label for token in ("ddos", "loic", "hoic", "goldeneye", "slowloris", "slowhttptest", "hulk")):
        return "dos_ddos"
    if label.startswith("dos ") or label.startswith("dos-") or label == "dos":
        return "dos_ddos"
    if "heartbleed" in label or "heartleech" in label:
        return None
    return None


def find_csvs(root: str) -> list[Path]:
    base = Path(root)
    if not base.exists():
        return []
    return sorted(path for path in base.rglob("*.csv") if path.is_file())


def find_label_column(columns: list[str]) -> str | None:
    for candidate in LABEL_CANDIDATES:
        if candidate in columns:
            return candidate
    lowered = {str(col).lower(): col for col in columns}
    for candidate in LABEL_CANDIDATES:
        found = lowered.get(candidate.lower())
        if found:
            return found
    return None


def coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.columns.is_unique:
        return df
    merged: dict[str, pd.Series] = {}
    for column in dict.fromkeys(df.columns):
        values = df.loc[:, df.columns == column]
        merged[column] = values.iloc[:, 0] if values.shape[1] == 1 else values.bfill(axis=1).iloc[:, 0]
    return pd.DataFrame(merged)


def scan_file_schema(path: Path) -> dict | None:
    header = pd.read_csv(path, nrows=0, dtype=str, low_memory=False, on_bad_lines="skip")
    columns = [str(col).strip() for col in header.columns]
    label_col = find_label_column(columns)
    if label_col is None:
        return None
    normalized = [
        normalize_column(col)
        for col in columns
        if col != label_col and normalize_column(col) not in DROP_COLUMNS
    ]
    return {
        "path": str(path),
        "label_col": label_col,
        "normalized_cols": list(dict.fromkeys(normalized)),
    }


def discover_inputs(cic2017_dir: str, cic2018_dir: str) -> tuple[list[dict], dict]:
    report = {"sources": {}, "files": []}
    inputs: list[dict] = []
    for dataset_name, root in (("cicids2017", cic2017_dir), ("cicids2018", cic2018_dir)):
        paths = find_csvs(root)
        report["sources"][dataset_name] = {"files_found": len(paths), "rows_written": 0}
        for path in paths:
            info = scan_file_schema(path)
            entry = {"file": str(path), "dataset": dataset_name}
            if info is None:
                entry["skipped"] = "no_label_column"
                report["files"].append(entry)
                continue
            inputs.append({"dataset": dataset_name, **info})
            entry["label_col"] = info["label_col"]
            entry["schema_cols"] = info["normalized_cols"]
            report["files"].append(entry)
    if not inputs:
        raise RuntimeError("No input CSV with a label column was found in the provided dataset folders.")
    return inputs, report


def output_columns(inputs: list[dict]) -> list[str]:
    columns = set()
    for info in inputs:
        columns.update(info["normalized_cols"])
    columns.add("type")
    front = [col for col in PREFERRED_FRONT if col in columns]
    rest = sorted(col for col in columns if col not in front)
    return front + rest


def normalize_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if col == "type":
            df[col] = df[col].astype(str).map(normalize_text)
            continue
        if col in LOWERCASE_VALUE_COLS:
            df[col] = df[col].astype(str).map(normalize_text)
            continue
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
    return df


def process_chunk(chunk: pd.DataFrame, label_col: str) -> tuple[pd.DataFrame, dict]:
    chunk.columns = [str(col).strip() for col in chunk.columns]
    raw_labels = chunk[label_col].fillna("").astype(str)
    mapped = raw_labels.map(map_public_label)

    stats = {
        "rows_read": int(len(chunk)),
        "rows_kept_before_cap": 0,
        "raw_labels_seen": dict(Counter(raw_labels.map(normalize_text))),
        "mapped_labels_kept": {},
        "dropped_raw_labels": dict(Counter(raw_labels.loc[mapped.isna()].map(normalize_text))),
    }

    keep = mapped.notna()
    if not keep.any():
        return pd.DataFrame(), stats

    chunk = chunk.loc[keep].copy().reset_index(drop=True)
    mapped = mapped.loc[keep].reset_index(drop=True)
    chunk.drop(columns=[label_col], inplace=True, errors="ignore")
    chunk.columns = [normalize_column(col) for col in chunk.columns]
    chunk = coalesce_duplicate_columns(chunk)
    chunk["type"] = mapped
    chunk.drop(columns=[col for col in DROP_COLUMNS if col in chunk.columns], inplace=True, errors="ignore")
    chunk = normalize_string_columns(chunk)

    stats["rows_kept_before_cap"] = int(len(chunk))
    stats["mapped_labels_kept"] = dict(Counter(chunk["type"]))
    return chunk, stats


def apply_cap(frame: pd.DataFrame, counts: dict[str, int], cap: int, seed: int, sample_id: int) -> tuple[pd.DataFrame, dict[str, int], int]:
    if frame.empty or cap <= 0:
        if frame.empty:
            return frame, {}, sample_id
        written = dict(Counter(frame["type"]))
        for label, n in written.items():
            counts[label] = counts.get(label, 0) + int(n)
        return frame, written, sample_id

    parts = []
    written: dict[str, int] = {}
    for label, group in frame.groupby("type", sort=False):
        remaining = cap - counts.get(label, 0)
        if remaining <= 0:
            continue
        if len(group) > remaining:
            group = group.sample(n=remaining, random_state=seed + sample_id)
            sample_id += 1
        parts.append(group)
        counts[label] = counts.get(label, 0) + int(len(group))
        written[label] = int(len(group))
    if not parts:
        return pd.DataFrame(columns=frame.columns), written, sample_id
    return pd.concat(parts, ignore_index=True), written, sample_id


def stream_build(inputs: list[dict], out_csv: str, final_cols: list[str], cap: int, chunk_size: int, seed: int, report: dict) -> dict[str, int]:
    written_counts = {label: 0 for label in CANONICAL_CLASSES}
    first_write = True
    sample_id = 1

    for info in inputs:
        if cap > 0 and all(written_counts[label] >= cap for label in CANONICAL_CLASSES):
            break

        file_written = Counter()
        file_kept = Counter()
        file_seen = Counter()
        file_dropped = Counter()
        rows_read = 0
        rows_kept_before_cap = 0

        print(f"Processing {info['dataset']}: {info['path']}")
        for chunk in pd.read_csv(
            info["path"],
            dtype=str,
            low_memory=False,
            on_bad_lines="skip",
            chunksize=chunk_size,
        ):
            cleaned, stats = process_chunk(chunk, info["label_col"])
            rows_read += stats["rows_read"]
            rows_kept_before_cap += stats["rows_kept_before_cap"]
            file_seen.update(stats["raw_labels_seen"])
            file_dropped.update(stats["dropped_raw_labels"])
            file_kept.update(stats["mapped_labels_kept"])

            capped, written_now, sample_id = apply_cap(cleaned, written_counts, cap, seed, sample_id)
            if capped.empty:
                continue

            capped = capped.reindex(columns=final_cols, fill_value="")
            capped.to_csv(out_csv, mode="a", header=first_write, index=False)
            first_write = False
            file_written.update(written_now)
            report["sources"][info["dataset"]]["rows_written"] += int(len(capped))

            if cap > 0 and all(written_counts[label] >= cap for label in CANONICAL_CLASSES):
                break

        file_report = next(item for item in report["files"] if item["file"] == info["path"])
        file_report.update({
            "rows_read": rows_read,
            "rows_kept_before_cap": rows_kept_before_cap,
            "rows_written": int(sum(file_written.values())),
            "raw_labels_seen": dict(file_seen),
            "mapped_labels_kept": dict(file_kept),
            "mapped_labels_written": dict(file_written),
            "dropped_raw_labels": dict(file_dropped),
        })

    if first_write:
        pd.DataFrame(columns=final_cols).to_csv(out_csv, index=False)
    return written_counts


def main() -> None:
    args = build_parser().parse_args()
    output_csv = os.path.abspath(args.output_csv)
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    inputs, report = discover_inputs(args.cic2017_dir, args.cic2018_dir)
    final_cols = output_columns(inputs)
    if os.path.exists(output_csv):
        os.remove(output_csv)

    written_counts = stream_build(
        inputs=inputs,
        out_csv=output_csv,
        final_cols=final_cols,
        cap=args.per_class_cap,
        chunk_size=args.chunk_size,
        seed=args.seed,
        report=report,
    )

    report_path = os.path.splitext(output_csv)[0] + "_report.json"
    report["output_csv"] = output_csv
    report["class_cap"] = args.per_class_cap
    report["chunk_size"] = args.chunk_size
    report["columns"] = final_cols
    report["class_counts_after_cap"] = {label: int(written_counts.get(label, 0)) for label in CANONICAL_CLASSES}
    report["rows_after_cap"] = int(sum(written_counts.values()))

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("=== Public CIC 6-Class Build ===")
    print(f"CIC-IDS2017 dir: {args.cic2017_dir}")
    print(f"CSE-CIC-IDS2018 dir: {args.cic2018_dir}")
    print(f"Output CSV:      {output_csv}")
    print(f"Report JSON:     {report_path}")
    print(f"Chunk size:      {args.chunk_size:,}")
    print(f"Class cap:       {args.per_class_cap:,}" if args.per_class_cap > 0 else "Class cap:       unlimited")
    print(f"Rows written:    {sum(written_counts.values()):,}")
    print(f"Classes:         {report['class_counts_after_cap']}")
    print(f"Columns:         {len(final_cols)}")
    print("Taxonomy:        backdoor, dos_ddos, infiltration, normal, password, scanning")
    print("Dropped labels:  web attacks and heartbleed/heartleech are excluded by design.")


if __name__ == "__main__":
    main()
