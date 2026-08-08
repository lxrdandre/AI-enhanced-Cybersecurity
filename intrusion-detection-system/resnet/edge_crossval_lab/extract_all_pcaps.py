"""Extract all labelled PCAPs captured by ``capture_source.sh``.

The script reads ``*.meta.json`` files from the raw capture directory and calls
``pcap_to_edge_csv.py`` for each matching PCAP.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from pcap_to_edge_csv import extract_pcap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/edge_crossval/raw", type=Path)
    parser.add_argument("--csv-dir", default="data/edge_crossval/csv", type=Path)
    parser.add_argument("--tshark", default=shutil.which("tshark") or "tshark")
    parser.add_argument("--limit-per-pcap", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--include-large-fields",
        action="store_true",
        help="Extract huge payload fields such as http.file_data and tcp.payload. Disabled by default.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metas = sorted(args.raw_dir.glob("*.meta.json"))
    if not metas:
        raise SystemExit(f"No meta files found in {args.raw_dir}")

    args.csv_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for meta_path in metas:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        source_label = str(meta["source_label"])
        type_label = str(meta["type"])
        pcap = Path(meta["pcap"])
        if not pcap.is_absolute() and not pcap.exists():
            pcap = args.raw_dir / pcap.name
        output = args.csv_dir / f"{source_label}.csv"
        if output.exists() and not args.overwrite:
            print(f"Skipping existing {output}")
            continue
        rows = extract_pcap(
            pcap=pcap,
            output=output,
            type_label=type_label,
            source_label=source_label,
            tshark_path=args.tshark,
            limit=args.limit_per_pcap,
            include_large_fields=args.include_large_fields,
        )
        total += rows
        print(f"{source_label}: {rows:,} rows")
    print(f"Extracted {total:,} rows into {args.csv_dir}")


if __name__ == "__main__":
    main()
