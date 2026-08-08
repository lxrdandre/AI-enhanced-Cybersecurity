"""Count rows by class/source in extracted Edge-style CSV files."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


TARGET_CLASSES = ["backdoor", "dos_ddos", "injection", "normal", "password", "scanning"]


def count_files(paths: list[Path]) -> dict[str, object]:
    class_counts: Counter[str] = Counter()
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    files: list[dict[str, object]] = []

    for path in paths:
        rows = 0
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "type" not in reader.fieldnames:
                continue
            for row in reader:
                label = str(row.get("type", "")).strip()
                source = str(row.get("source_label", path.stem)).strip() or path.stem
                if label not in TARGET_CLASSES:
                    continue
                rows += 1
                class_counts[label] += 1
                source_counts[label][source] += 1
        files.append({"file": str(path), "rows": rows})

    return {
        "class_counts": {cls: int(class_counts[cls]) for cls in TARGET_CLASSES},
        "source_counts": {
            cls: {source: int(count) for source, count in sorted(source_counts[cls].items())}
            for cls in TARGET_CLASSES
        },
        "files": files,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data/edge_crossval/csv", type=Path)
    parser.add_argument("--class-name", choices=TARGET_CLASSES, default=None)
    parser.add_argument("--value-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(args.input_dir.glob("*.csv"))
    report = count_files(paths)
    if args.class_name:
        value = int(report["class_counts"].get(args.class_name, 0))
        print(value if args.value_only else f"{args.class_name}: {value}")
        return
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    print(report["class_counts"])


if __name__ == "__main__":
    main()
