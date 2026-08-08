"""Audit Zeek SE-DWNet artifacts for obvious train/test leakage risks."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


HARD_LEAK_TOKENS = (
    "type",
    "label",
    "source_label",
    "uid",
    "id_orig_h",
    "id_resp_h",
    "src_is_kali",
    "dst_is_kali",
    "attack",
    "pcap",
    "meta",
)

WARN_TOKENS = (
    "source",
    "kali",
    "orig_h",
    "resp_h",
)


def load_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def find_token_hits(features: list[str], tokens: tuple[str, ...]) -> dict[str, list[str]]:
    hits: dict[str, list[str]] = {}
    for feature in features:
        lower = feature.lower()
        parts = [part for part in lower.replace("-", "_").split("_") if part]
        for token in tokens:
            if "_" in token:
                matched = token == lower or token in lower
            else:
                matched = token == lower or token in parts
            if matched:
                hits.setdefault(token, []).append(feature)
    return hits


def summarize_csv(csv_path: Path, max_rows: int) -> dict:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise SystemExit("This audit needs pandas for CSV checks. Run it in the training venv.") from exc

    if not csv_path.exists():
        return {"exists": False, "path": str(csv_path)}

    df = pd.read_csv(csv_path, nrows=max_rows, low_memory=False)
    result = {
        "exists": True,
        "path": str(csv_path),
        "sample_rows": int(len(df)),
        "sample_columns": int(df.shape[1]),
        "ctx_columns": int(sum(col.startswith("ctx_") for col in df.columns)),
        "metadata_columns_present": [col for col in ["type", "source_label", "uid", "id_orig_h", "id_resp_h", "src_is_kali", "dst_is_kali"] if col in df.columns],
    }
    if "type" in df.columns:
        result["sample_class_counts"] = {str(k): int(v) for k, v in Counter(df["type"]).items()}
    if "source_label" in df.columns and "type" in df.columns:
        source_label_types = df.groupby("source_label")["type"].nunique(dropna=False)
        result["source_labels_with_multiple_types"] = int((source_label_types > 1).sum())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--sample-rows", type=int, default=200_000)
    args = parser.parse_args()

    artifact_dir = args.artifact_dir
    metadata = load_json(artifact_dir / "training_metadata.json")
    features = load_lines(artifact_dir / "final_features.txt")
    if not features and metadata.get("selected_features"):
        features = [str(item) for item in metadata["selected_features"]]

    csv_path = args.csv
    if csv_path is None and metadata.get("data_csv"):
        csv_path = Path(str(metadata["data_csv"]))

    hard_hits = find_token_hits(features, HARD_LEAK_TOKENS)
    warn_hits = find_token_hits(features, WARN_TOKENS)
    ctx_features = [feature for feature in features if feature.startswith("ctx_")]

    print("=== Zeek Leakage Audit ===")
    print(f"Artifact dir: {artifact_dir}")
    print(f"Selected features: {len(features)}")
    print(f"Context features selected: {len(ctx_features)}")
    if ctx_features:
        print(f"Context feature sample: {ctx_features[:20]}")

    print("\nSplit metadata:")
    print(f"  split_mode: {metadata.get('split_mode')}")
    print(f"  source_group_mode: {metadata.get('source_group_mode')}")
    print(f"  final_holdout_mode: {metadata.get('final_holdout_mode') or metadata.get('final_holdout', {}).get('mode')}")
    print(f"  dedupe_enabled: {metadata.get('dedupe_enabled')}")
    print(f"  smote_enabled: {metadata.get('smote_enabled')}")

    print("\nHard leakage feature hits:")
    if hard_hits:
        for token, cols in sorted(hard_hits.items()):
            print(f"  FAIL {token}: {cols}")
    else:
        print("  none")

    print("\nSuspicious feature hits:")
    if warn_hits:
        for token, cols in sorted(warn_hits.items()):
            print(f"  WARN {token}: {cols}")
    else:
        print("  none")

    final_holdout_mode = metadata.get("final_holdout_mode") or metadata.get("final_holdout", {}).get("mode")
    if ctx_features and final_holdout_mode not in {"source", "temporal"}:
        print("\nFAIL: ctx_* features are selected but final holdout is not source/temporal.")
    if ctx_features and final_holdout_mode == "temporal":
        print("\nWARN: temporal holdout may still split an exact source_label PCAP; source holdout is stricter.")
    if metadata.get("split_mode") == "random":
        print("\nWARN: random split is optimistic for same-capture Zeek flows.")

    if csv_path is not None:
        print("\nCSV sample:")
        csv_info = summarize_csv(csv_path, args.sample_rows)
        print(json.dumps(csv_info, indent=2, sort_keys=True))

    if hard_hits:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
