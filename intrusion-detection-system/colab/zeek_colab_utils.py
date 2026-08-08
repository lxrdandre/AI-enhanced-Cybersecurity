"""Shared preprocessing/evaluation helpers for Colab experiments."""

from __future__ import annotations

import gc
import json
import os
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.api.types import is_numeric_dtype
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
RESNET_DIR = REPO_ROOT / "resnet"


def add_repo_paths() -> None:
    for path in (str(REPO_ROOT), str(RESNET_DIR)):
        if path not in sys.path:
            sys.path.insert(0, path)


add_repo_paths()

import resnet_zeek_crossval as zeek_train  # noqa: E402

try:
    from imblearn.over_sampling import SMOTE
except ImportError:  # pragma: no cover
    SMOTE = None


TARGET_CLASSES = zeek_train.TARGET_CLASSES
SOURCE_LABEL_COL = zeek_train.SOURCE_LABEL_COL
SOURCE_GROUP_COL = zeek_train.SOURCE_GROUP_COL
LOG_COLS = zeek_train.LOG_COLS


def ensure_parent(path: str | os.PathLike[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def save_json(path: str | os.PathLike[str], data: dict[str, Any]) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(data), f, indent=2)


def save_pickle(path: str | os.PathLike[str], data: Any) -> None:
    ensure_parent(path)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(path: str | os.PathLike[str]) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def class_counts(values: pd.Series | np.ndarray | list[Any]) -> dict[str, int]:
    return {str(k): int(v) for k, v in sorted(Counter(values).items())}


def softmax_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    scores = scores - np.max(scores, axis=1, keepdims=True)
    exp = np.exp(scores)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def maybe_sample_training_rows(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_rows_per_class: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_rows_per_class <= 0:
        return x, y
    rng = np.random.default_rng(seed)
    keep_indices: list[np.ndarray] = []
    for cls in sorted(np.unique(y)):
        indices = np.flatnonzero(y == cls)
        if len(indices) > max_rows_per_class:
            indices = rng.choice(indices, size=max_rows_per_class, replace=False)
        keep_indices.append(indices)
    selected = np.concatenate(keep_indices)
    rng.shuffle(selected)
    return x[selected], y[selected]


def evaluate_predictions(
    *,
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    class_names: list[str],
    output_dir: str,
    prefix: str,
    probabilities: np.ndarray | None = None,
) -> dict[str, Any]:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)

    report = classification_report(
        y_true,
        y_pred,
        labels=class_names,
        target_names=class_names,
        zero_division=0,
        digits=4,
    )
    print(f"\nClassification Report ({prefix}):")
    print(report)

    with open(Path(output_dir) / f"{prefix}_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.title(prefix)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"{prefix}_confusion_matrix.png", dpi=180)
    plt.close()

    pred_data: dict[str, Any] = {
        "true_class": y_true,
        "predicted_class": y_pred,
        "correct": y_true == y_pred,
    }
    if probabilities is not None:
        probabilities = np.asarray(probabilities)
        pred_data["confidence"] = np.max(probabilities, axis=1)
        for index, cls in enumerate(class_names):
            if index < probabilities.shape[1]:
                pred_data[f"prob_{cls}"] = probabilities[:, index]
    pd.DataFrame(pred_data).to_csv(Path(output_dir) / f"{prefix}_predictions.csv", index=False)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=class_names, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=class_names, average="weighted", zero_division=0)),
        "support": int(len(y_true)),
    }


def _load_labelled_csv(csv_path: str, label_col: str | None, seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    print(f"Loading CSV: {csv_path}", flush=True)
    df = pd.read_csv(csv_path, low_memory=False, on_bad_lines="skip")
    df.columns = df.columns.str.strip()
    target_col = zeek_train.label_column(df, label_col)
    if target_col != "type":
        df["type"] = df[target_col]

    labels_norm = df["type"].astype(str).str.strip().str.lower().map(zeek_train.canon_label)
    keep = labels_norm.isin(TARGET_CLASSES)
    if int((~keep).sum()):
        print(f"Dropping {int((~keep).sum()):,} rows with unsupported labels.")
    df = df.loc[keep].copy()
    df["type"] = labels_norm.loc[keep].to_numpy()
    df = df.reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No usable six-class rows were found.")

    info = {
        "csv": os.path.abspath(csv_path),
        "rows": int(len(df)),
        "target_column": target_col,
        "class_counts": class_counts(df["type"]),
        "seed": int(seed),
    }
    return df, info


def build_custom_zeek_matrices(
    *,
    csv_path: str,
    label_col: str | None = None,
    split: str = "random",
    source_group_mode: str = "family",
    time_col: str = "ts",
    temporal_fallback: str = "random",
    val_size: float = 0.15,
    test_size: float = 0.15,
    final_holdout_size: float = 0.05,
    final_holdout_mode: str = "random",
    target_k: int = 192,
    smote_mode: str = "auto",
    smote_imbalance_ratio: float = 1.10,
    dedupe: bool = False,
    expected_per_class: int = 60_000,
    max_categorical_cardinality: int = 1024,
    numeric_valid_ratio: float = 0.98,
    feature_inference_sample_size: int = 120_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Build train/val/test/holdout matrices using the Zeek trainer's logic."""
    df, data_info = _load_labelled_csv(csv_path, label_col, seed)
    if split == "temporal":
        df["_time_order"] = zeek_train.derive_time_order(df, explicit_col=time_col)
    elif "ts" in df.columns:
        df["_time_order"] = pd.to_numeric(df["ts"], errors="coerce")
    else:
        df["_time_order"] = np.arange(len(df), dtype=np.float64)
    df[SOURCE_GROUP_COL] = zeek_train.build_source_groups(df, source_group_mode)

    if expected_per_class > 0:
        mismatch = {cls: count for cls, count in class_counts(df["type"]).items() if count != expected_per_class}
        if mismatch:
            print(f"WARNING: class counts differ from expected {expected_per_class:,}: {mismatch}")

    meta_cols = ["type", "_time_order", SOURCE_GROUP_COL]
    if SOURCE_LABEL_COL in df.columns:
        meta_cols.append(SOURCE_LABEL_COL)
    split_meta_all = df[meta_cols].copy().reset_index(drop=True)

    drop_metadata = sorted(
        (zeek_train.ZEEK_METADATA_COLUMNS | set(zeek_train.LABEL_CANDIDATES) | {"type", "_time_order", SOURCE_GROUP_COL})
        .intersection(df.columns)
    )
    x_all = df.drop(columns=drop_metadata, errors="ignore").reset_index(drop=True)
    print(f"Dropped metadata columns: {drop_metadata}")
    print(f"Raw feature matrix shape: {x_all.shape}", flush=True)

    if feature_inference_sample_size and len(x_all) > feature_inference_sample_size:
        x_role_sample = x_all.sample(n=feature_inference_sample_size, random_state=seed)
    else:
        x_role_sample = x_all
    print(f"Inferring feature roles from {len(x_role_sample):,}/{len(x_all):,} rows...", flush=True)
    cat_cols, num_cols, dropped_feature_cols, feature_role_details = zeek_train.infer_feature_roles(
        x_role_sample,
        max_categorical_cardinality=max_categorical_cardinality,
        numeric_valid_ratio=numeric_valid_ratio,
    )
    del x_role_sample

    if dropped_feature_cols:
        x_all.drop(columns=dropped_feature_cols, errors="ignore", inplace=True)
    valid_cat_cols = [col for col in cat_cols if col in x_all.columns]
    num_cols = [col for col in num_cols if col in x_all.columns]

    for col in valid_cat_cols:
        x_all[col] = x_all[col].fillna("missing").replace("-", "missing").astype(str)
    for col in num_cols:
        if is_numeric_dtype(x_all[col]):
            x_all[col] = x_all[col].astype("float32", copy=False)
        else:
            x_all[col] = pd.to_numeric(x_all[col], errors="coerce").astype("float32")

    x_all.replace([np.inf, -np.inf], 0, inplace=True)
    x_all = x_all.fillna(0)
    for col in LOG_COLS:
        if col in x_all.columns and is_numeric_dtype(x_all[col]):
            x_all[col] = np.log1p(x_all[col].fillna(0).clip(lower=0))

    constant_cols = [col for col in x_all.columns if x_all[col].nunique(dropna=False) <= 1]
    if constant_cols:
        x_all.drop(columns=constant_cols, inplace=True)
        valid_cat_cols = [col for col in valid_cat_cols if col not in constant_cols]
        num_cols = [col for col in num_cols if col not in constant_cols]
        print(f"Dropped constant columns after cleanup: {len(constant_cols)}")
    if not x_all.columns.tolist():
        raise RuntimeError("No usable features remain after preprocessing.")

    x_all = zeek_train.optimize_dtypes(x_all)
    print(f"Cleaned data shape: {x_all.shape}")
    print(f"Numeric columns:     {len(num_cols)}")
    print(f"Categorical columns: {len(valid_cat_cols)}")

    dedupe_info = {"before": int(len(x_all)), "after": int(len(x_all)), "dropped": 0}
    if dedupe:
        x_all, split_meta_all, dedupe_info = zeek_train.dedupe_rows_with_meta(x_all, split_meta_all)
    else:
        print("Skipping exact row deduplication.")
        x_all = x_all.reset_index(drop=True)
        split_meta_all = split_meta_all.reset_index(drop=True)

    split_meta_all["_row_id"] = np.arange(len(split_meta_all), dtype=np.int64)
    print("Reserving final holdout after cleanup/dedupe and before SMOTE...")
    if final_holdout_mode == "source":
        pool_meta, final_holdout_meta, final_holdout_info = zeek_train.split_final_holdout_by_source_label(
            split_meta_all,
            holdout_size=final_holdout_size,
            seed=seed,
        )
    elif final_holdout_mode == "temporal":
        pool_meta, final_holdout_meta, final_holdout_info = zeek_train.split_final_holdout_temporal(
            split_meta_all,
            holdout_size=final_holdout_size,
        )
    else:
        pool_meta, final_holdout_meta, final_holdout_info = zeek_train.split_final_holdout(
            split_meta_all,
            holdout_size=final_holdout_size,
            seed=seed,
            source_stratified=True,
        )
        final_holdout_info["mode"] = "random"

    pool_ids = pool_meta["_row_id"].astype(int).to_numpy()
    holdout_ids = final_holdout_meta["_row_id"].astype(int).to_numpy()
    x_final_holdout_df = x_all.iloc[holdout_ids].reset_index(drop=True) if len(holdout_ids) else pd.DataFrame(columns=x_all.columns)
    x_all = x_all.iloc[pool_ids].reset_index(drop=True)
    y_final_holdout = final_holdout_meta["type"].astype(str).reset_index(drop=True) if len(holdout_ids) else pd.Series(dtype=str)
    y_all = pool_meta["type"].astype(str).reset_index(drop=True)
    time_order_all = pd.to_numeric(pool_meta["_time_order"], errors="coerce").reset_index(drop=True)
    split_group_all = pool_meta[SOURCE_GROUP_COL].fillna(pool_meta["type"]).astype(str).reset_index(drop=True)
    print(f"Training pool rows: {len(x_all):,}")
    print(f"Final holdout rows: {len(x_final_holdout_df):,}")
    if len(y_final_holdout):
        print(f"Final holdout counts: {class_counts(y_final_holdout)}")

    train_ratio = 1.0 - val_size - test_size
    if not (0 < train_ratio < 1):
        raise ValueError("val_size + test_size must be less than 1.0")
    print(f"Splitting data ({split} {train_ratio:.2f}/{val_size:.2f}/{test_size:.2f})...")
    if split == "source":
        x_train_df, x_val_df, x_test_df, y_train_str, y_val_str, y_test_str, split_info = zeek_train.split_frames_by_source_group(
            x_all,
            y_all,
            split_group_all,
            train_ratio=train_ratio,
            val_ratio=val_size,
            test_ratio=test_size,
            seed=seed,
        )
    else:
        x_train_df, x_val_df, x_test_df, y_train_str, y_val_str, y_test_str, split_info = zeek_train.split_frames(
            x_all,
            y_all,
            time_order_all,
            split_mode=split,
            train_ratio=train_ratio,
            val_ratio=val_size,
            test_ratio=test_size,
            seed=seed,
            temporal_fallback=temporal_fallback,
            split_group=split_group_all,
        )

    split_group_counts = class_counts(split_group_all)
    del df, split_meta_all, pool_meta, final_holdout_meta, x_all, y_all, time_order_all, split_group_all
    gc.collect()

    le_target = LabelEncoder()
    le_target.fit(y_train_str)
    y_train = le_target.transform(y_train_str)
    y_val = le_target.transform(y_val_str)
    y_test = le_target.transform(y_test_str)
    class_names = le_target.classes_.tolist()
    print(f"Classes: {class_names}")

    encoders = {}
    x_train_df = x_train_df.reset_index(drop=True)
    x_val_df = x_val_df.reset_index(drop=True)
    x_test_df = x_test_df.reset_index(drop=True)
    x_final_holdout_df = x_final_holdout_df.reset_index(drop=True)
    for col in valid_cat_cols:
        encoder = zeek_train.SafeLabelEncoder()
        x_train_df[col] = encoder.fit(x_train_df[col]).transform(x_train_df[col])
        x_val_df[col] = encoder.transform(x_val_df[col])
        x_test_df[col] = encoder.transform(x_test_df[col])
        if not x_final_holdout_df.empty:
            x_final_holdout_df[col] = encoder.transform(x_final_holdout_df[col])
        encoders[col] = encoder

    scaler_num = MinMaxScaler()
    if num_cols:
        x_train_df[num_cols] = scaler_num.fit_transform(x_train_df[num_cols].values)
        x_val_df[num_cols] = scaler_num.transform(x_val_df[num_cols].values)
        x_test_df[num_cols] = scaler_num.transform(x_test_df[num_cols].values)
        if not x_final_holdout_df.empty:
            x_final_holdout_df[num_cols] = scaler_num.transform(x_final_holdout_df[num_cols].values)

    print(f"Selecting top {target_k} features...")
    feature_names = x_train_df.columns.tolist()
    discrete_mask = np.array([col in valid_cat_cols for col in feature_names], dtype=bool)
    selector = SelectKBest(
        score_func=zeek_train.make_mi_scorer(discrete_mask, seed),
        k=min(target_k, x_train_df.shape[1]),
    )
    selector.fit(x_train_df, y_train)
    x_train_sel = selector.transform(x_train_df).astype(np.float32)
    x_val_sel = selector.transform(x_val_df).astype(np.float32)
    x_test_sel = selector.transform(x_test_df).astype(np.float32)
    x_holdout_sel = (
        selector.transform(x_final_holdout_df).astype(np.float32)
        if not x_final_holdout_df.empty
        else np.empty((0, selector.get_support().sum()), dtype=np.float32)
    )
    final_features = x_train_df.columns[selector.get_support()].tolist()
    print(f"Selected features: {len(final_features)}")

    final_scaler = MinMaxScaler()
    x_train_sel = np.nan_to_num(final_scaler.fit_transform(x_train_sel)).astype(np.float32)
    x_val_sel = np.nan_to_num(final_scaler.transform(x_val_sel)).astype(np.float32)
    x_test_sel = np.nan_to_num(final_scaler.transform(x_test_sel)).astype(np.float32)
    if len(x_holdout_sel):
        x_holdout_sel = np.nan_to_num(final_scaler.transform(x_holdout_sel)).astype(np.float32)

    train_counts = Counter(y_train)
    max_class_count = max(train_counts.values())
    min_class_count = min(train_counts.values())
    imbalance_ratio = float(max_class_count) / max(float(min_class_count), 1.0)
    use_smote = smote_mode == "on" or (smote_mode == "auto" and imbalance_ratio >= smote_imbalance_ratio)
    smote_strategy = {}
    if use_smote:
        for cls, count in train_counts.items():
            if count < 2:
                continue
            target = min(int(count * zeek_train.SMOTE_MAX_MULTIPLIER), max_class_count)
            if target > count:
                smote_strategy[cls] = target
    if smote_strategy:
        if SMOTE is None:
            raise RuntimeError("SMOTE requested but imbalanced-learn is not installed.")
        min_count = min(train_counts[cls] for cls in smote_strategy)
        k_neighbors = max(1, min(5, min_count - 1))
        print(f"Applying SMOTE: {smote_strategy}")
        smote = SMOTE(sampling_strategy=smote_strategy, random_state=seed, k_neighbors=k_neighbors)
        x_train_fit, y_train_fit = smote.fit_resample(x_train_sel, y_train)
    else:
        print("SMOTE skipped.")
        x_train_fit, y_train_fit = x_train_sel, y_train

    preprocess_bundle = {
        "scaler_num": scaler_num,
        "selector": selector,
        "final_scaler": final_scaler,
        "encoders": encoders,
        "target_encoder": le_target,
        "features": final_features,
        "valid_cat_cols": valid_cat_cols,
        "num_cols": num_cols,
        "metadata_columns_dropped": drop_metadata,
        "dropped_feature_columns": dropped_feature_cols + constant_cols,
        "log_cols": LOG_COLS,
        "dataset_name": "custom_zeek_crossval",
        "data_csv": os.path.abspath(csv_path),
    }

    metadata = {
        "data": data_info,
        "split": split,
        "source_group_mode": source_group_mode,
        "split_info": split_info,
        "split_group_counts": split_group_counts,
        "final_holdout": final_holdout_info,
        "dedupe": dedupe_info,
        "dedupe_enabled": bool(dedupe),
        "smote_mode": smote_mode,
        "smote_enabled": bool(smote_strategy),
        "smote_imbalance_ratio": imbalance_ratio,
        "target_k": int(target_k),
        "class_names": class_names,
        "final_features": final_features,
        "train_counts": class_counts(y_train_str),
        "train_counts_after_smote": {class_names[int(k)]: int(v) for k, v in sorted(Counter(y_train_fit).items())},
        "val_counts": class_counts(y_val_str),
        "test_counts": class_counts(y_test_str),
        "holdout_counts": class_counts(y_final_holdout) if len(y_final_holdout) else {},
        "numeric_columns": num_cols,
        "categorical_columns": valid_cat_cols,
        "feature_role_details": feature_role_details,
    }

    return {
        "x_train": x_train_fit,
        "y_train": y_train_fit,
        "x_val": x_val_sel,
        "y_val": y_val,
        "x_test": x_test_sel,
        "y_test": y_test,
        "x_holdout": x_holdout_sel,
        "y_holdout": y_final_holdout.to_numpy().astype(str) if len(y_final_holdout) else np.array([], dtype=str),
        "class_names": class_names,
        "target_encoder": le_target,
        "preprocess_bundle": preprocess_bundle,
        "metadata": metadata,
    }


def transform_external_csv(
    *,
    csv_path: str,
    preprocess_bundle: dict[str, Any],
    label_col: str | None = None,
    sample_per_class: int = 0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Transform an external labelled CSV with a saved custom-Zeek pipeline."""
    df, info = _load_labelled_csv(csv_path, label_col, seed)
    if sample_per_class > 0:
        df = (
            df.groupby("type", group_keys=False)
            .apply(lambda g: g.sample(n=min(len(g), sample_per_class), random_state=seed))
            .reset_index(drop=True)
        )
        info["sample_per_class"] = int(sample_per_class)
        info["sampled_rows"] = int(len(df))
        info["sampled_class_counts"] = class_counts(df["type"])

    final_features = list(preprocess_bundle["features"])
    valid_cat_cols = list(preprocess_bundle["valid_cat_cols"])
    num_cols = list(preprocess_bundle["num_cols"])
    encoders = preprocess_bundle["encoders"]
    scaler_num = preprocess_bundle["scaler_num"]
    final_scaler = preprocess_bundle["final_scaler"]
    log_cols = preprocess_bundle.get("log_cols", LOG_COLS)

    required_cols = list(dict.fromkeys(valid_cat_cols + num_cols + final_features))
    present_required = [col for col in required_cols if col in df.columns]
    missing_required = [col for col in required_cols if col not in df.columns]
    for col in missing_required:
        df[col] = "missing" if col in valid_cat_cols else 0

    x = df[required_cols].copy()
    for col in valid_cat_cols:
        if col not in x.columns:
            continue
        x[col] = x[col].fillna("missing").replace("-", "missing").astype(str)
        encoder = encoders.get(col)
        x[col] = encoder.transform(x[col]) if encoder is not None else 0
    for col in num_cols:
        if col in x.columns:
            x[col] = pd.to_numeric(x[col], errors="coerce").astype("float32")

    x.replace([np.inf, -np.inf], 0, inplace=True)
    x = x.fillna(0)
    for col in log_cols:
        if col in x.columns and is_numeric_dtype(x[col]):
            x[col] = np.log1p(pd.to_numeric(x[col], errors="coerce").fillna(0).clip(lower=0))

    if num_cols:
        x[num_cols] = scaler_num.transform(x[num_cols].values)

    missing_final = [col for col in final_features if col not in x.columns]
    if missing_final:
        raise RuntimeError(f"Missing final features after external transform: {missing_final[:20]}")
    matrix = np.nan_to_num(final_scaler.transform(x[final_features].values)).astype(np.float32)

    coverage = {
        "csv": os.path.abspath(csv_path),
        "rows": int(len(df)),
        "class_counts": class_counts(df["type"]),
        "required_feature_count": int(len(required_cols)),
        "present_required_count": int(len(present_required)),
        "missing_required_count": int(len(missing_required)),
        "missing_required_sample": missing_required[:80],
        "final_feature_count": int(len(final_features)),
        "final_features_present_count": int(sum(col in present_required for col in final_features)),
        "final_features_missing": [col for col in final_features if col not in present_required],
    }
    info["feature_coverage"] = coverage
    return matrix, df["type"].to_numpy().astype(str), info
