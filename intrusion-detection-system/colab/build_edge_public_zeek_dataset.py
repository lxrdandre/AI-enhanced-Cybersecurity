"""Build a public PCAP dataset with the same Zeek features as the custom dataset.

This is a wrapper around the existing custom-dataset pipeline:

1. Recursively find public dataset ``.pcap``/``.pcapng`` files.
2. Infer or read a six-class label for each PCAP.
3. Convert every PCAP with ``pcap_to_zeek_csv.py``.
4. Merge/cap/add context features with ``build_zeek_crossval_dataset.py``.

It works with Edge-IIoTset PCAP names out of the box and can also be used for
CIC or other public datasets. For unusual directory names, pass ``--manifest``
with columns:

    pcap,type[,source_label,target_ip,kali_ip]

The generated final CSV has the same base and ``ctx_*`` feature schema used by
``zeek_crossval.csv``.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_zeek_crossval_dataset import (  # noqa: E402
    TARGET_CLASSES,
    add_rolling_context_features,
    cap_dataset,
    parse_windows,
    read_inputs,
    require_pandas,
)
from pcap_to_zeek_csv import extract_pcap  # noqa: E402


PCAP_SUFFIXES = {".pcap", ".pcapng", ".cap"}
DEFAULT_CONTEXT_WINDOWS = "5,15,60"
DEFAULT_CAP_PER_CLASS = 100_000

LABEL_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    (
        "normal",
        (
            r"\bbenign\b",
            r"\bnormal\b",
            r"\bbackground\b",
            r"\bbenigntraffic\b",
            r"\bbenign_traffic\b",
        ),
    ),
    (
        "dos_ddos",
        (
            r"\bddos\b",
            r"\bdos\b",
            r"\bhulk\b",
            r"\bgoldeneye\b",
            r"\bslowhttptest\b",
            r"\bslowloris\b",
            r"\bslow_http\b",
            r"\bsyn_flood\b",
            r"\budp_flood\b",
            r"\bicmp_flood\b",
            r"\bhoic\b",
            r"\bloic\b",
        ),
    ),
    (
        "password",
        (
            r"\bbrute[_ -]?force\b",
            r"\bbruteforce\b",
            r"\bpatator\b",
            r"\bftp[_ -]?patator\b",
            r"\bssh[_ -]?patator\b",
            r"\bdictionary\b",
            r"\bpassword\b",
            r"\bhydra\b",
            r"\bmedusa\b",
        ),
    ),
    (
        "scanning",
        (
            r"\bport[_ -]?scan\b",
            r"\bportscan\b",
            r"\bport[_ -]?scanning\b",
            r"\bscan\b",
            r"\bscanning\b",
            r"\bscanner\b",
            r"\brecon\b",
            r"\bnmap\b",
            r"\bvulnerability[_ -]?scan\b",
            r"\bvulnerability[_ -]?scanner\b",
            r"\bos[_ -]?fingerprint",
            r"\bos[_ -]?fingerprinting\b",
            r"\bservice[_ -]?scan\b",
        ),
    ),
    (
        "injection",
        (
            r"\bsql\b",
            r"\bsqli\b",
            r"\bsql[_ -]?injection\b",
            r"\bxss\b",
            r"\bweb[_ -]?attack\b",
            r"\bcommand[_ -]?injection\b",
            r"\bcmd[_ -]?injection\b",
            r"\bpath[_ -]?traversal\b",
            r"\bupload\b",
            r"\buploading\b",
            r"\bfile[_ -]?upload\b",
        ),
    ),
    (
        "backdoor",
        (
            r"\bbackdoor\b",
            r"\bbot\b",
            r"\bbotnet\b",
            r"\binfiltration\b",
            r"\bmalware\b",
        ),
    ),
]


def normalize_source_label(path: Path) -> str:
    text = "_".join(path.with_suffix("").parts[-4:])
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return text or path.stem.lower()


def label_from_text(text: str) -> str | None:
    haystack = re.sub(r"[^a-z0-9]+", " ", text.lower())
    haystack = f" {haystack} "
    for label, patterns in LABEL_PATTERNS:
        if any(re.search(pattern, haystack) for pattern in patterns):
            return label
    return None


def infer_label(path: Path) -> str | None:
    return label_from_text(" ".join(path.parts[-6:]))


def find_pcaps(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in PCAP_SUFFIXES
    )


def read_manifest(path: Path, root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"pcap", "type"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Manifest is missing required columns: {sorted(missing)}")
        for idx, row in enumerate(reader, start=2):
            pcap_raw = str(row.get("pcap", "")).strip()
            type_label = str(row.get("type", "")).strip().lower()
            if type_label not in TARGET_CLASSES:
                raise SystemExit(f"{path}:{idx}: invalid type {type_label!r}; expected one of {TARGET_CLASSES}")
            pcap = Path(pcap_raw)
            if not pcap.is_absolute():
                pcap = root / pcap
            source_label = str(row.get("source_label", "")).strip() or normalize_source_label(pcap)
            rows.append(
                {
                    "pcap": str(pcap),
                    "type": type_label,
                    "source_label": source_label,
                    "target_ip": str(row.get("target_ip", "")).strip(),
                    "kali_ip": str(row.get("kali_ip", "")).strip(),
                }
            )
    return rows


def discover_inputs(root: Path, manifest: Path | None) -> tuple[list[dict[str, str]], dict[str, object]]:
    if manifest is not None:
        inputs = read_manifest(manifest, root)
        return inputs, {"mode": "manifest", "manifest": str(manifest), "unlabeled_pcaps": []}

    inputs: list[dict[str, str]] = []
    unlabeled: list[str] = []
    for pcap in find_pcaps(root):
        type_label = infer_label(pcap)
        if type_label is None:
            unlabeled.append(str(pcap))
            continue
        inputs.append(
            {
                "pcap": str(pcap),
                "type": type_label,
                "source_label": normalize_source_label(pcap),
                "target_ip": "",
                "kali_ip": "",
            }
        )
    return inputs, {"mode": "infer_from_path", "manifest": None, "unlabeled_pcaps": unlabeled}


def uniquify_source_labels(inputs: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: Counter[str] = Counter()
    output: list[dict[str, str]] = []
    for item in inputs:
        copied = dict(item)
        base = copied["source_label"]
        seen[base] += 1
        if seen[base] > 1:
            copied["source_label"] = f"{base}_{seen[base]}"
        output.append(copied)
    return output


def write_extraction_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pcap",
        "type",
        "source_label",
        "csv",
        "rows",
        "status",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a public Zeek CSV using the same features as the custom Zeek dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pcap-root", required=True, type=Path, help="Directory containing public dataset PCAP/PCAPNG files.")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional CSV with pcap,type[,source_label,target_ip,kali_ip].")
    parser.add_argument("--work-dir", type=Path, default=Path("data/public_zeek_work"))
    parser.add_argument("--output-csv", type=Path, default=Path("data/public_zeek_same_features.csv"))
    parser.add_argument("--report-json", type=Path, default=Path("data/public_zeek_same_features_report.json"))
    parser.add_argument("--cap-per-class", type=int, default=DEFAULT_CAP_PER_CLASS)
    parser.add_argument("--quota-mode", choices=["even", "random"], default="even")
    parser.add_argument("--context-windows", default=DEFAULT_CONTEXT_WINDOWS)
    parser.add_argument("--no-context", action="store_true")
    parser.add_argument(
        "--context-before-cap",
        action="store_true",
        help="Compute rolling context on all extracted rows before class capping. More faithful but much heavier.",
    )
    parser.add_argument("--limit-per-pcap", type=int, default=0, help="Optional row limit during Zeek extraction, 0 means no limit.")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing per-PCAP extracted CSVs.")
    parser.add_argument("--keep-zeek-logs", action="store_true")
    parser.add_argument("--zeek", default=shutil.which("zeek") or "zeek")
    parser.add_argument("--target-ip", default=os.environ.get("TARGET_IP", ""))
    parser.add_argument("--kali-ip", default=os.environ.get("KALI_IP", ""))
    parser.add_argument("--ssh-port", default=os.environ.get("SSH_PORT", "64295"))
    parser.add_argument("--canonical-ssh-port", default=os.environ.get("CANONICAL_SSH_PORT", "22"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pcap_root = args.pcap_root.expanduser().resolve()
    if not pcap_root.exists():
        raise SystemExit(f"PCAP root does not exist: {pcap_root}")

    inputs, discovery_info = discover_inputs(pcap_root, args.manifest.expanduser().resolve() if args.manifest else None)
    inputs = uniquify_source_labels(inputs)
    if not inputs:
        raise SystemExit(
            "No labeled PCAPs found. Use --manifest, or rename/place PCAPs in directories containing labels like "
            "Benign, DDoS, DoS, PortScan, BruteForce, SQL, XSS, Bot, Backdoor."
        )

    args.work_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = args.work_dir / "zeek_csv"
    logs_dir = args.work_dir / "zeek_logs" if args.keep_zeek_logs else None
    csv_dir.mkdir(parents=True, exist_ok=True)
    if logs_dir is not None:
        logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"PCAP root: {pcap_root}")
    print(f"Labeled PCAPs: {len(inputs):,}")
    print(f"Initial inferred class counts: {dict(Counter(item['type'] for item in inputs))}")
    if discovery_info.get("unlabeled_pcaps"):
        print(f"Skipped unlabeled PCAPs: {len(discovery_info['unlabeled_pcaps'])}")

    extraction_rows: list[dict[str, object]] = []
    input_csvs: list[Path] = []
    for idx, item in enumerate(inputs, start=1):
        pcap = Path(item["pcap"]).expanduser().resolve()
        type_label = str(item["type"])
        source_label = str(item["source_label"])
        output = csv_dir / f"{source_label}.csv"
        if output.exists() and args.skip_existing:
            rows_written = max(0, sum(1 for _ in output.open("r", encoding="utf-8", errors="replace")) - 1)
            status = "skipped_existing"
            print(f"[{idx}/{len(inputs)}] SKIP {source_label}: {rows_written:,} rows")
        else:
            try:
                rows_written = extract_pcap(
                    pcap=pcap,
                    output=output,
                    type_label=type_label,
                    source_label=source_label,
                    zeek_bin=args.zeek,
                    target_ip=str(item.get("target_ip") or args.target_ip),
                    kali_ip=str(item.get("kali_ip") or args.kali_ip),
                    ssh_port=str(args.ssh_port),
                    canonical_ssh_port=str(args.canonical_ssh_port),
                    limit=args.limit_per_pcap if args.limit_per_pcap > 0 else None,
                    keep_zeek_logs=logs_dir,
                )
                status = "ok"
                print(f"[{idx}/{len(inputs)}] {source_label}/{type_label}: {rows_written:,} Zeek flow rows")
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                rows_written = 0
                status = "error"
                print(f"[{idx}/{len(inputs)}] ERROR {source_label}: {exc}")
                extraction_rows.append(
                    {
                        "pcap": str(pcap),
                        "type": type_label,
                        "source_label": source_label,
                        "csv": str(output),
                        "rows": rows_written,
                        "status": status,
                        "error": str(exc),
                    }
                )
                continue

        input_csvs.append(output)
        extraction_rows.append(
            {
                "pcap": str(pcap),
                "type": type_label,
                "source_label": source_label,
                "csv": str(output),
                "rows": rows_written,
                "status": status,
                "error": "",
            }
        )

    extraction_manifest = args.work_dir / "extraction_manifest.csv"
    write_extraction_manifest(extraction_manifest, extraction_rows)
    if not input_csvs:
        raise SystemExit("No per-PCAP Zeek CSVs were produced.")

    pandas = require_pandas()
    df, file_reports = read_inputs(input_csvs)
    raw_counts = {cls: int((df["type"] == cls).sum()) for cls in TARGET_CLASSES}
    context_windows = [] if args.no_context else parse_windows(args.context_windows)
    if context_windows and args.context_before_cap:
        df = add_rolling_context_features(df, context_windows)
        sampled = cap_dataset(df, args.cap_per_class, args.quota_mode, args.seed)
    else:
        sampled = cap_dataset(df, args.cap_per_class, args.quota_mode, args.seed)
        del df
        gc.collect()
        if context_windows:
            sampled = add_rolling_context_features(sampled, context_windows)

    if not args.no_shuffle:
        sampled = sampled.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(args.output_csv, index=False)

    sampled_counts = {cls: int((sampled["type"] == cls).sum()) for cls in TARGET_CLASSES}
    shortfalls = {cls: max(0, int(args.cap_per_class) - sampled_counts[cls]) for cls in TARGET_CLASSES}
    report = {
        "dataset": "public_zeek_same_features",
        "pcap_root": str(pcap_root),
        "output_csv": str(args.output_csv),
        "report_json": str(args.report_json),
        "work_dir": str(args.work_dir),
        "extraction_manifest": str(extraction_manifest),
        "rows": int(len(sampled)),
        "columns": int(sampled.shape[1]),
        "cap_per_class": int(args.cap_per_class),
        "quota_mode": args.quota_mode,
        "context_before_cap": bool(args.context_before_cap),
        "context_windows_seconds": [float(value) for value in context_windows],
        "raw_class_counts": raw_counts,
        "sampled_class_counts": sampled_counts,
        "shortfalls": shortfalls,
        "source_counts": {
            cls: {k: int(v) for k, v in Counter(sampled.loc[sampled["type"] == cls, "source_label"]).items()}
            for cls in TARGET_CLASSES
        },
        "discovery": discovery_info,
        "extractions": extraction_rows,
        "input_files": file_reports,
        "columns_list": sampled.columns.tolist(),
    }
    args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("\nDONE")
    print(f"Output CSV:  {args.output_csv}")
    print(f"Report JSON: {args.report_json}")
    print(f"Rows:        {len(sampled):,}")
    print(f"Columns:     {sampled.shape[1]:,}")
    print(f"Counts:      {sampled_counts}")
    if any(shortfalls.values()):
        print(f"Shortfalls:  {shortfalls}")
    print(f"Manifest:    {extraction_manifest}")


if __name__ == "__main__":
    main()
