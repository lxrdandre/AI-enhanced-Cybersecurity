"""Extract all labelled PCAPs into Zeek-flow CSVs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from pcap_to_zeek_csv import extract_pcap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/edge_crossval/raw", type=Path)
    parser.add_argument("--csv-dir", default="data/edge_crossval/zeek_csv", type=Path)
    parser.add_argument("--zeek", default=shutil.which("zeek") or "zeek")
    parser.add_argument("--limit-per-pcap", type=int, default=None)
    parser.add_argument("--target-ip", default=os.environ.get("TARGET_IP", ""))
    parser.add_argument("--kali-ip", default=os.environ.get("KALI_IP", ""))
    parser.add_argument("--ssh-port", default=os.environ.get("SSH_PORT", "22"))
    parser.add_argument("--canonical-ssh-port", default=os.environ.get("CANONICAL_SSH_PORT", "22"))
    parser.add_argument("--keep-zeek-logs-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
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
            zeek_bin=args.zeek,
            target_ip=args.target_ip or str(meta.get("target_ip", "")),
            kali_ip=args.kali_ip or str(meta.get("kali_ip", "")),
            ssh_port=str(args.ssh_port),
            canonical_ssh_port=str(args.canonical_ssh_port),
            limit=args.limit_per_pcap,
            keep_zeek_logs=args.keep_zeek_logs_dir,
        )
        total += rows
        print(f"{source_label}: {rows:,} Zeek flow rows")
    print(f"Extracted {total:,} Zeek flow rows into {args.csv_dir}")


if __name__ == "__main__":
    main()
