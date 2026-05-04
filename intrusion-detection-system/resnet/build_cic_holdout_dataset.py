"""Build the unseen CIC 6-class spare validation CSV from the saved artifact.

The default output is the exact reserved holdout set (`holdout_spare`) created
by the trainer: 1,000 unseen rows per class when available.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_PARENT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_PARENT not in sys.path:
    sys.path.insert(0, PROJECT_PARENT)

from resnet_cic_public import (  # noqa: E402
    CAT_COLS,
    IP_COLS,
    LABEL_CANDIDATES,
    LOG_COLS,
    TARGET_CLASSES,
    TIME_COLS,
    build_split_manifest,
    canon_label,
    dedupe_with_raw_reference,
    derive_time_order,
    label_column,
    model_row_hashes,
    optimize_dtypes,
    project_root,
    select_split_pool_and_spare_indices,
    split_frames,
)


def default_artifact_dir(root: str) -> str:
    return os.path.join(root, "artifacts", "resnet_cic_public")


def default_output_csv(root: str) -> str:
    return os.path.join(root, "data", "cic_public_6class_spare_validation.csv")


def load_training_metadata(artifact_dir: str) -> dict:
    path = os.path.join(artifact_dir, "training_metadata.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"training_metadata.json not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def prepare_clean_cic_frame(
    *,
    csv_path: str,
    label_col: str | None,
    split_mode: str,
    time_col: str | None,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict]:
    df = pd.read_csv(csv_path, low_memory=False, dtype=str, on_bad_lines="skip")
    df.columns = df.columns.str.strip()
    target_col = label_column(df, label_col)
    if target_col != "type":
        df["type"] = df[target_col]

    if split_mode == "temporal":
        df["_time_order"] = derive_time_order(df, explicit_col=time_col)
    else:
        df["_time_order"] = np.arange(len(df), dtype=np.float64)

    drop_metadata = [col for col in TIME_COLS if col in df.columns]
    drop_metadata += [col for col in LABEL_CANDIDATES if col in df.columns and col != "type"]
    df.drop(columns=drop_metadata, errors="ignore", inplace=True)

    labels_norm = df["type"].astype(str).str.strip().str.lower().map(canon_label)
    keep = labels_norm.isin(TARGET_CLASSES)
    df = df.loc[keep].copy()
    df["type"] = labels_norm.loc[keep].to_numpy()
    if df.empty:
        raise RuntimeError("No rows left after filtering to the CIC 6-class taxonomy.")
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    ip_cols = [c for c in IP_COLS if c in df.columns]
    df.drop(columns=ip_cols, inplace=True, errors="ignore")

    y_all = df["type"].astype(str).reset_index(drop=True)
    time_order_all = pd.to_numeric(df["_time_order"], errors="coerce").reset_index(drop=True)
    x_raw = df.drop(columns=["type", "_time_order"]).reset_index(drop=True)
    x_all = x_raw.copy()

    valid_cat_cols = [c for c in CAT_COLS if c in x_all.columns]
    num_cols = [c for c in x_all.columns if c not in valid_cat_cols]

    for col in valid_cat_cols:
        x_all[col] = x_all[col].fillna("missing").replace("-", "missing").astype(str)

    for col in num_cols:
        x_all[col] = pd.to_numeric(x_all[col], errors="coerce")

    x_all.replace([np.inf, -np.inf], 0, inplace=True)
    x_all = x_all.fillna(0)

    for col in LOG_COLS:
        if col in x_all.columns:
            x_all[col] = np.log1p(pd.to_numeric(x_all[col], errors="coerce").fillna(0).clip(lower=0))

    constant_cols = [col for col in x_all.columns if x_all[col].nunique(dropna=False) <= 1]
    if constant_cols:
        x_all.drop(columns=constant_cols, inplace=True)

    x_all = optimize_dtypes(x_all)
    x_raw, x_all, y_all, time_order_all, dedupe_info = dedupe_with_raw_reference(x_raw, x_all, y_all, time_order_all)
    info = {
        "target_col": target_col,
        "dropped_ip_cols": ip_cols,
        "constant_cols_dropped": len(constant_cols),
        "dedupe": dedupe_info,
        "class_counts": dict(Counter(y_all)),
    }
    return x_raw, x_all, y_all, time_order_all, info


def reconstruct_manifest(
    *,
    x_all: pd.DataFrame,
    y_all: pd.Series,
    time_order_all: pd.Series,
    metadata: dict,
) -> pd.DataFrame:
    split_mode = str(metadata.get("split_mode", "temporal"))
    val_size = float(metadata.get("val_size", 0.20))
    test_size = float(metadata.get("test_size", 0.20))
    seed = int(metadata.get("seed", 42))
    temporal_fallback = str(metadata.get("temporal_fallback", "random"))
    train_ratio = 1.0 - val_size - test_size
    split_pool_idx, spare_idx, _selection_info = select_split_pool_and_spare_indices(
        y_all,
        time_order_all,
        split_mode=split_mode,
        split_per_class_cap=int(metadata.get("split_per_class_cap", 150_000)),
        spare_per_class=int(metadata.get("spare_validation_per_class", 1_000)),
        seed=seed,
    )

    x_split_pool = x_all.loc[split_pool_idx].reset_index(drop=True)
    y_split_pool = y_all.loc[split_pool_idx].reset_index(drop=True)
    time_split_pool = time_order_all.loc[split_pool_idx].reset_index(drop=True)
    x_spare = x_all.loc[spare_idx].reset_index(drop=True)
    y_spare = y_all.loc[spare_idx].reset_index(drop=True)

    x_train_df, x_val_df, x_test_df, y_train_str, y_val_str, y_test_str, _split_info = split_frames(
        x_split_pool,
        y_split_pool,
        time_split_pool,
        split_mode=split_mode,
        train_ratio=train_ratio,
        val_ratio=val_size,
        test_ratio=test_size,
        seed=seed,
        temporal_fallback=temporal_fallback,
    )

    return build_split_manifest(
        x_train_fit=x_train_df,
        y_train_fit=y_train_str,
        x_val=x_val_df,
        y_val=y_val_str,
        x_test=x_test_df,
        y_test=y_test_str,
        x_spare=x_spare,
        y_spare=y_spare,
    )


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(
        description="Build the unseen CIC 6-class spare validation CSV from the saved artifact.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--artifact-dir", default=default_artifact_dir(root), help="CIC artifact directory.")
    parser.add_argument("--csv", default=None, help="Source CIC CSV. Defaults to training_metadata.json data_csv.")
    parser.add_argument("--label-col", default=None, help="Override label column. Auto-detected when omitted.")
    parser.add_argument("--output-csv", default=default_output_csv(root), help="Output holdout CSV.")
    parser.add_argument("--report-json", default=None, help="Optional JSON report path.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    artifact_dir = os.path.abspath(args.artifact_dir)
    metadata = load_training_metadata(artifact_dir)
    csv_path = os.path.abspath(args.csv or metadata.get("data_csv") or "")
    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(f"Source CSV not found: {csv_path}")

    split_mode = str(metadata.get("split_mode", "temporal"))
    time_col = metadata.get("time_col")
    x_raw_all, x_all, y_all, time_order_all, prep_info = prepare_clean_cic_frame(
        csv_path=csv_path,
        label_col=args.label_col,
        split_mode=split_mode,
        time_col=time_col,
        seed=int(metadata.get("seed", args.seed)),
    )

    manifest_path = os.path.join(artifact_dir, "split_membership.csv.gz")
    if os.path.exists(manifest_path):
        manifest = pd.read_csv(manifest_path, dtype=str)
        manifest_source = "artifact"
    else:
        manifest = reconstruct_manifest(
            x_all=x_all,
            y_all=y_all,
            time_order_all=time_order_all,
            metadata=metadata,
        )
        manifest_source = "reconstructed"
        manifest.to_csv(manifest_path, index=False, compression="gzip")

    eligible_hashes = set(
        manifest.loc[manifest["split"] == "holdout_spare", "row_hash"].astype(str).tolist()
    )
    if not eligible_hashes:
        raise RuntimeError("No holdout_spare rows found in split manifest.")

    row_hashes = model_row_hashes(x_all, y_all)
    eligible_mask = row_hashes.isin(eligible_hashes)
    holdout_x = x_raw_all.loc[eligible_mask].reset_index(drop=True)
    holdout_y = y_all.loc[eligible_mask].reset_index(drop=True)

    output_df = holdout_x.copy()
    output_df["type"] = holdout_y.to_numpy()
    output_path = os.path.abspath(args.output_csv)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_df.to_csv(output_path, index=False)

    report_path = os.path.abspath(args.report_json) if args.report_json else os.path.splitext(output_path)[0] + "_report.json"
    report = {
        "artifact_dir": artifact_dir,
        "source_csv": csv_path,
        "output_csv": output_path,
        "manifest_path": manifest_path,
        "manifest_source": manifest_source,
        "rows_written": int(len(output_df)),
        "class_counts": dict(Counter(holdout_y)),
        "prep_info": prep_info,
        "manifest_split_counts": {
            split: int(count)
            for split, count in manifest["split"].value_counts().sort_index().items()
        },
    }
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("=== CIC 6-Class Spare Validation Build ===")
    print(f"Artifact dir:    {artifact_dir}")
    print(f"Source CSV:      {csv_path}")
    print(f"Manifest source: {manifest_source}")
    print(f"Output CSV:      {output_path}")
    print(f"Report JSON:     {report_path}")
    print(f"Rows written:    {len(output_df):,}")
    print(f"Classes:         {dict(Counter(holdout_y))}")


if __name__ == "__main__":
    main()
