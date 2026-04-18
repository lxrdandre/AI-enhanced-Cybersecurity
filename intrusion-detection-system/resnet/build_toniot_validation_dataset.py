"""
Build a never-seen TON-IoT holdout validation CSV from raw split files.

The raw input directory is expected to contain files named like:
    Network_dataset_1.csv ... Network_dataset_23.csv

Rows already present in data/Network_dataset_capped.csv are excluded after the
same model-visible cleanup used by resnet_base.py:
    - strip column names
    - drop ts/date/time/label
    - drop IP address columns
    - drop mitm/ransomware
    - merge dos and ddos into dos_ddos

The output is still raw model input, not scaled/encoded data. Use
validate_toniot_holdout.py to apply the saved preprocessing pipeline.
"""

from __future__ import annotations

import argparse
import glob
import heapq
import os
from collections import Counter

import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

TARGET_CLASSES = [
    "backdoor",
    "dos_ddos",
    "injection",
    "normal",
    "password",
    "scanning",
    "xss",
]

DROP_LABELS = {"mitm", "ransomware"}
DROP_COLS = ["ts", "date", "time", "label"]
IP_COLS = ["src_ip", "dst_ip", "srcip", "dstip"]


def _detect_project_root() -> str:
    env_root = os.environ.get("TON_IOT_PROJECT_ROOT")
    if env_root and os.path.isdir(env_root):
        return os.path.abspath(env_root)
    cwd = os.path.abspath(os.getcwd())
    if os.path.isdir(os.path.join(cwd, "data")) and os.path.isdir(os.path.join(cwd, "artifacts")):
        return cwd
    return os.path.normpath(os.path.join(SCRIPT_DIR, ".."))


def _clean_frame(df: pd.DataFrame, *, signature_cols: list[str] | None = None) -> pd.DataFrame:
    """Apply the training-time row cleanup but do not scale/encode features."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    df.drop(columns=DROP_COLS, errors="ignore", inplace=True)

    if "type" not in df.columns:
        raise RuntimeError("Missing required TON-IoT target column 'type'.")

    labels = df["type"].astype(str).str.strip().str.lower()
    df = df.loc[~labels.isin(DROP_LABELS)].copy()
    df["type"] = df["type"].astype(str).str.strip().str.lower()
    df.loc[df["type"].isin(["dos", "ddos"]), "type"] = "dos_ddos"
    df = df[df["type"].isin(TARGET_CLASSES)].copy()

    df.drop(columns=IP_COLS, errors="ignore", inplace=True)

    if signature_cols is not None:
        for col in signature_cols:
            if col not in df.columns:
                df[col] = ""
        df = df[signature_cols]

    return df.fillna("")


def _row_hashes(df: pd.DataFrame, signature_cols: list[str]) -> np.ndarray:
    """Return deterministic per-row hashes for duplicate exclusion."""
    canonical = df.loc[:, signature_cols].astype(str).apply(lambda col: col.str.strip())
    return pd.util.hash_pandas_object(canonical, index=False).to_numpy(dtype=np.uint64)


def _load_capped_hashes(capped_csv: str, chunksize: int) -> tuple[set[int], list[str]]:
    hashes: set[int] = set()
    signature_cols: list[str] | None = None
    total_rows = 0

    print(f"Indexing capped training data: {capped_csv}")
    for chunk_idx, chunk in enumerate(
        pd.read_csv(capped_csv, dtype=str, low_memory=False, chunksize=chunksize, on_bad_lines="skip"),
        start=1,
    ):
        cleaned = _clean_frame(chunk, signature_cols=signature_cols)
        if signature_cols is None:
            signature_cols = cleaned.columns.tolist()
            if "type" not in signature_cols:
                raise RuntimeError("Capped dataset cleanup removed 'type'; cannot build signatures.")
        row_hashes = _row_hashes(cleaned, signature_cols)
        hashes.update(int(h) for h in row_hashes)
        total_rows += len(cleaned)
        print(f"  capped chunk {chunk_idx}: rows={len(cleaned):>8}, unique_hashes={len(hashes):>8}")

    if not signature_cols:
        raise RuntimeError(f"No usable rows found in capped dataset: {capped_csv}")

    print(f"Capped usable rows: {total_rows}")
    print(f"Capped unique model-visible rows: {len(hashes)}")
    return hashes, signature_cols


def _find_raw_files(raw_dir: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(raw_dir, "Network_dataset_*.csv")))
    files = [path for path in files if not os.path.basename(path).lower().endswith("_capped.csv")]
    if not files:
        raise FileNotFoundError(f"No raw files found at {raw_dir}/Network_dataset_*.csv")
    return files


def _push_sample(
    reservoirs: dict[str, list[tuple[float, int, int, dict]]],
    selected_hashes: set[int],
    counters: Counter,
    cls: str,
    row_hash: int,
    row: dict,
    samples_per_class: int,
    rng: np.random.Generator,
) -> None:
    """Reservoir-sample unique rows per class."""
    if row_hash in selected_hashes:
        return

    counters[cls] += 1
    heap = reservoirs[cls]
    priority = float(rng.random())

    if len(heap) < samples_per_class:
        heapq.heappush(heap, (-priority, counters[cls], row_hash, row))
        selected_hashes.add(row_hash)
        return

    if priority >= -heap[0][0]:
        return

    _, _, old_hash, _ = heapq.heapreplace(heap, (-priority, counters[cls], row_hash, row))
    selected_hashes.discard(old_hash)
    selected_hashes.add(row_hash)


def build_validation_dataset(
    *,
    raw_dir: str,
    capped_csv: str,
    output_csv: str,
    samples_per_class: int,
    chunksize: int,
    seed: int,
) -> None:
    capped_hashes, signature_cols = _load_capped_hashes(capped_csv, chunksize)
    raw_files = _find_raw_files(raw_dir)

    print(f"Raw files: {len(raw_files)}")
    for path in raw_files:
        print(f"  {path}")
    print(f"Target samples per class: {samples_per_class}")

    rng = np.random.default_rng(seed)
    reservoirs = {cls: [] for cls in TARGET_CLASSES}
    selected_hashes: set[int] = set()
    eligible_seen: Counter = Counter()
    skipped_seen = 0

    for raw_path in raw_files:
        print(f"Scanning raw file: {raw_path}")
        for chunk_idx, chunk in enumerate(
            pd.read_csv(raw_path, dtype=str, low_memory=False, chunksize=chunksize, on_bad_lines="skip"),
            start=1,
        ):
            cleaned = _clean_frame(chunk, signature_cols=signature_cols)
            if cleaned.empty:
                continue

            row_hashes = _row_hashes(cleaned, signature_cols)
            not_seen_mask = np.array([int(h) not in capped_hashes for h in row_hashes], dtype=bool)
            skipped_seen += int((~not_seen_mask).sum())
            cleaned = cleaned.loc[not_seen_mask].copy()
            row_hashes = row_hashes[not_seen_mask]

            for row_hash, (_, row) in zip(row_hashes, cleaned.iterrows()):
                cls = str(row["type"])
                _push_sample(
                    reservoirs,
                    selected_hashes,
                    eligible_seen,
                    cls,
                    int(row_hash),
                    row.to_dict(),
                    samples_per_class,
                    rng,
                )

            counts = {cls: len(reservoirs[cls]) for cls in TARGET_CLASSES}
            print(f"  chunk {chunk_idx}: selected={counts}")

            if all(len(reservoirs[cls]) >= samples_per_class for cls in TARGET_CLASSES):
                # Keep scanning all files would make the reservoir more random, but this
                # script prioritizes practical runtime on the full TON-IoT raw dump.
                print("Target reached for every class; stopping early.")
                break
        else:
            continue
        break

    rows: list[dict] = []
    for cls in TARGET_CLASSES:
        rows.extend(item[3] for item in reservoirs[cls])

    if not rows:
        raise RuntimeError("No holdout rows selected. Check raw_dir and capped_csv inputs.")

    out_df = pd.DataFrame(rows)
    out_df = out_df[signature_cols]
    out_df = out_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    print("\n=== Holdout Dataset Built ===")
    print(f"Output: {output_csv}")
    print(f"Rows:   {len(out_df)}")
    print(f"Labels: {dict(Counter(out_df['type']))}")
    print(f"Skipped rows already in capped dataset: {skipped_seen}")
    print(f"Eligible never-seen rows scanned: {dict(eligible_seen)}")
    missing = [cls for cls in TARGET_CLASSES if len(reservoirs[cls]) < samples_per_class]
    if missing:
        print(f"WARNING: below target for classes: {missing}")


def main() -> None:
    project_root = _detect_project_root()
    parser = argparse.ArgumentParser(description="Build TON-IoT never-seen holdout validation CSV.")
    parser.add_argument(
        "--raw-dir",
        default=os.path.join(project_root, "data", "raw"),
        help="Directory containing Network_dataset_*.csv raw split files.",
    )
    parser.add_argument(
        "--capped-csv",
        default=os.path.join(project_root, "data", "Network_dataset_capped.csv"),
        help="Training/capped CSV whose model-visible rows must be excluded.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(project_root, "data", "toniot_holdout_validation.csv"),
        help="Output holdout CSV path.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=10_000,
        help="Target unique holdout rows per class.",
    )
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_validation_dataset(
        raw_dir=args.raw_dir,
        capped_csv=args.capped_csv,
        output_csv=args.output,
        samples_per_class=args.samples_per_class,
        chunksize=args.chunksize,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
