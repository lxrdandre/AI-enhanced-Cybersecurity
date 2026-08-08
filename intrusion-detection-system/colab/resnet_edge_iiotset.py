"""SE-DWNet training for the capped Edge-IIoTset 6-class dataset.

Expected build output from resnet/build_edge_iiotset_dataset.py:

    === Edge-IIoTset 6-Class Build ===
    Output CSV:  /data/datasets/edge_iiotset/processed/edge_iiotset_6class_cap100k.csv
    Report JSON: /data/datasets/edge_iiotset/processed/edge_iiotset_6class_cap100k_report.json
    Rows:        524,862
    Columns:     56
    Classes:     {'backdoor': 24862, 'dos_ddos': 100000, 'injection': 100000,
                  'normal': 100000, 'password': 100000, 'scanning': 100000}

The dataset builder adds split_time and source_label metadata columns. By
default this trainer reserves a 5% source-stratified final holdout first, then
uses a random source-stratified train/validation/test split on the remaining
rows. Metadata columns are dropped before model feature processing.

Target taxonomy:
    backdoor, dos_ddos, injection, normal, password, scanning
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import sys
import warnings
from collections import Counter
from functools import partial

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from pandas.api.types import is_numeric_dtype
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    Multiply,
    Reshape,
    SeparableConv1D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_PARENT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_PARENT)

try:
    from app.preprocessing import SafeLabelEncoder  # noqa: E402
except ImportError:  # Colab flat-upload fallback.
    from sklearn.base import BaseEstimator, TransformerMixin  # noqa: E402

    class SafeLabelEncoder(BaseEstimator, TransformerMixin):
        """Deterministic label-to-id mapping with an unknown bucket at 0."""

        def __init__(self):
            self.mapper = {}
            self.unknown_token = 0

        def fit(self, y):
            y_series = pd.Series(y).astype(str)
            unique_labels = np.unique(y_series.values)
            unique_labels = np.sort(unique_labels)
            self.mapper = {label: idx + 1 for idx, label in enumerate(unique_labels)}
            return self

        def transform(self, y):
            return pd.Series(y).astype(str).map(self.mapper).fillna(self.unknown_token).astype(np.int32).values


TARGET_CLASSES = {"backdoor", "dos_ddos", "injection", "normal", "password", "scanning"}
LABEL_CANDIDATES = ("type", "attack", "attack_type", "category", "class", "label", "Label")
TIME_COLS = ("split_time", "ts", "timestamp", "datetime", "date", "time", "frame_time_epoch", "frame_time")
SOURCE_LABEL_COL = "source_label"

TARGET_K = 40
BATCH_SIZE = 1024
MAX_EPOCHS = 100
LEARNING_RATE = 5e-4
SEED = 42

USE_FOCAL_LOSS = True
FOCAL_GAMMA = 2.0
FOCAL_ALPHA_CLIP = (0.5, 5.0)
SMOTE_MAX_MULTIPLIER = 4

LOG_COLS = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "src_pkts",
    "dst_pkts",
    "http_request_body_len",
    "http_response_body_len",
    "missed_bytes",
]

FORCED_CATEGORICAL_HINTS = (
    "arp_opcode",
    "conn_state",
    "dns_qclass",
    "dns_qtype",
    "dns_rcode",
    "http_method",
    "http_request_method",
    "http_response_code",
    "http_user_agent",
    "http_version",
    "icmp_type",
    "mqtt_msgtype",
    "proto",
    "protocol",
    "service",
    "ssl_cipher",
    "ssl_version",
    "tcp_flags",
    "tcp_flag",
)

DROP_FEATURE_COLUMNS = {
    "attack_label",
    "attack_type",
    "binary_label",
    "category",
    "class",
    "label",
    "multiclass_label",
    "source_label",
    "target",
}

MISSING_TOKENS = {"", "-", "nan", "none", "null", "missing", "?", "<na>"}


def project_root() -> str:
    """Return the project root used for data and artifact paths."""
    env_root = os.environ.get("TON_IOT_PROJECT_ROOT")
    if env_root and os.path.isdir(env_root):
        return os.path.abspath(env_root)
    cwd = os.path.abspath(os.getcwd())
    if os.path.isdir(os.path.join(cwd, "data")) and os.path.isdir(os.path.join(cwd, "artifacts")):
        return cwd
    return PROJECT_PARENT


def default_csv() -> str:
    """Return the default processed Edge-IIoTSet training CSV path."""
    return "/data/datasets/edge_iiotset/processed/edge_iiotset_6class_cap100k.csv"


def default_output_dir(project_root_path: str) -> str:
    """Return the default artifact directory for Edge-IIoTSet training."""
    return os.path.join(project_root_path, "artifacts", "se_dwnet_edge_iiotset")


def canon_label(label: object) -> str:
    """Return the canonical six-class label for a raw dataset label."""
    value = str(label).strip().lower()
    return "dos_ddos" if value in {"dos", "ddos", "ddos_dos"} else value


def label_column(df: pd.DataFrame, explicit: str | None) -> str:
    """Find the target label column, honoring an explicit override."""
    if explicit:
        if explicit not in df.columns:
            raise RuntimeError(f"Label column '{explicit}' not found. Columns: {list(df.columns[:40])}")
        return explicit
    for col in LABEL_CANDIDATES:
        if col not in df.columns:
            continue
        values = set(df[col].dropna().astype(str).head(50_000).map(canon_label).unique())
        if values.intersection(TARGET_CLASSES):
            return col
    raise RuntimeError(f"Could not identify target label column. Columns: {list(df.columns[:40])}")


def sample_values(series: pd.Series, n: int = 5) -> list[str]:
    """Return representative non-empty values for diagnostics."""
    values = series.dropna().astype(str).str.strip()
    values = values[values != ""].head(n).tolist()
    return values


def numeric_time(series: pd.Series) -> tuple[pd.Series | None, str]:
    """Parse a candidate timestamp column as numeric epoch-like values."""
    cleaned = series.astype(str).str.strip().str.replace(",", "", regex=False)
    values = pd.to_numeric(cleaned, errors="coerce")
    valid = int(values.notna().sum())
    unique = int(values.nunique(dropna=True))
    detail = f"numeric valid={valid:,}/{len(series):,}, unique={unique:,}, sample={sample_values(series)}"
    if valid >= 3 and unique >= 3:
        return values.astype("float64"), detail
    return None, detail


def datetime_time(series: pd.Series) -> tuple[pd.Series | None, str]:
    """Parse a candidate timestamp column as datetimes in UTC seconds."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        values = pd.to_datetime(series, errors="coerce", utc=True)
    valid = int(values.notna().sum())
    unique = int(values.nunique(dropna=True))
    detail = f"datetime valid={valid:,}/{len(series):,}, unique={unique:,}, sample={sample_values(series)}"
    if valid < 3 or unique < 3:
        return None, detail
    seconds = pd.Series(np.nan, index=series.index, dtype="float64")
    seconds.loc[values.notna()] = values.loc[values.notna()].astype("int64") / 1_000_000_000
    return seconds, detail


def derive_time_order(df: pd.DataFrame, explicit_col: str | None) -> pd.Series:
    """Return the row ordering signal used by temporal splitting."""
    candidates = []
    if explicit_col:
        candidates.append(explicit_col)
    candidates.extend(["split_time", "ts", "timestamp", "datetime", "frame_time_epoch", "frame_time"])

    for col in candidates:
        if col not in df.columns:
            continue
        numeric, numeric_detail = numeric_time(df[col])
        if numeric is not None:
            print(f"Temporal split using numeric column '{col}' ({numeric_detail})")
            return numeric
        parsed, datetime_detail = datetime_time(df[col])
        if parsed is not None:
            print(f"Temporal split using datetime column '{col}' ({datetime_detail})")
            return parsed
        print(f"Column '{col}' is not usable for temporal split ({numeric_detail}; {datetime_detail})")

    if "date" in df.columns and "time" in df.columns:
        combined = df["date"].astype(str).str.strip() + " " + df["time"].astype(str).str.strip()
        parsed, datetime_detail = datetime_time(combined)
        if parsed is not None:
            print(f"Temporal split using combined 'date time' ({datetime_detail})")
            return parsed
        print(f"Combined 'date time' is not usable for temporal split ({datetime_detail})")

    raise RuntimeError(
        "Temporal split requires a usable timestamp column. "
        "Use --split random for the capped Edge-IIoTset CSV built without time columns."
    )


def ratio_counts(n: int, train_ratio: float, val_ratio: float) -> tuple[int, int]:
    """Convert split ratios into non-empty train and validation counts."""
    if n < 3:
        raise RuntimeError("Need at least 3 rows per class to split train/val/test.")
    train_n = max(1, int(round(n * train_ratio)))
    val_n = max(1, int(round(n * val_ratio)))
    if train_n + val_n >= n:
        val_n = max(1, n - train_n - 1)
    if train_n + val_n >= n:
        train_n = max(1, n - val_n - 1)
    return train_n, val_n


def boundary_without_time_leak(times: np.ndarray, target_cut: int) -> int:
    """Move a temporal split boundary away from identical timestamps."""
    n = len(times)
    cut = max(1, min(n - 1, target_cut))
    forward = cut
    while forward < n and times[forward - 1] == times[forward]:
        forward += 1
    if forward < n:
        return forward
    backward = cut
    while backward > 1 and times[backward - 1] == times[backward]:
        backward -= 1
    return backward


def dedupe_model_visible_rows(
    x: pd.DataFrame,
    y: pd.Series,
    time_order: pd.Series,
    split_group: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series | None, dict]:
    """Drop duplicate feature rows before splitting to reduce leakage."""
    feature_hash = pd.util.hash_pandas_object(x, index=False).astype("uint64")
    meta = pd.DataFrame(
        {
            "_target": y.to_numpy(),
            "_time_order": pd.to_numeric(time_order, errors="coerce").to_numpy(),
            "_row_order": np.arange(len(x), dtype=np.int64),
            "_feature_hash": feature_hash.to_numpy(),
        }
    )

    label_counts = meta.groupby("_feature_hash")["_target"].nunique()
    conflict_hashes = set(label_counts[label_counts > 1].index.tolist())
    if conflict_hashes:
        conflict_mask = meta["_feature_hash"].isin(conflict_hashes)
        print(f"Dropping {int(conflict_mask.sum()):,} duplicate feature rows with conflicting labels.")
        keep = ~conflict_mask
        x = x.loc[keep].copy()
        y = y.loc[keep].copy()
        time_order = time_order.loc[keep].copy()
        if split_group is not None:
            split_group = split_group.loc[keep].copy()
        meta = meta.loc[keep].copy()

    before = len(x)
    keep_idx = (
        meta.assign(_sort_time=meta["_time_order"].fillna(np.inf))
        .sort_values(["_sort_time", "_row_order"], kind="mergesort")
        .drop_duplicates("_feature_hash", keep="first")
        .index
    )
    x = x.loc[keep_idx].reset_index(drop=True)
    y = y.loc[keep_idx].reset_index(drop=True)
    time_order = time_order.loc[keep_idx].reset_index(drop=True)
    if split_group is not None:
        split_group = split_group.loc[keep_idx].reset_index(drop=True)
    info = {"before": int(before), "after": int(len(x)), "dropped": int(before - len(x))}
    print(f"Deduplicated model-visible rows: {before:,} -> {len(x):,}")
    return x, y, time_order, split_group, info


def split_frames(
    x: pd.DataFrame,
    y: pd.Series,
    time_order: pd.Series,
    *,
    split_mode: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    temporal_fallback: str,
    split_group: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, dict]:
    """Split features and labels into train, validation, and test frames."""
    temp_size = val_ratio + test_ratio
    test_ratio_of_temp = test_ratio / temp_size
    split_info = {"mode": split_mode, "class_splits": {}}

    if split_mode == "random":
        stratify_values = choose_stratify(split_group, y, context="Random train/val/test split") if split_group is not None else y
        x_train, x_temp, y_train, y_temp, strat_train, strat_temp = train_test_split(
            x,
            y,
            stratify_values,
            test_size=temp_size,
            stratify=stratify_values,
            random_state=seed,
        )
        x_val, x_test, y_val, y_test = train_test_split(
            x_temp,
            y_temp,
            test_size=test_ratio_of_temp,
            stratify=strat_temp,
            random_state=seed,
        )
        split_info["stratify"] = "split_group" if split_group is not None else "target"
        return (
            x_train.reset_index(drop=True),
            x_val.reset_index(drop=True),
            x_test.reset_index(drop=True),
            y_train.reset_index(drop=True),
            y_val.reset_index(drop=True),
            y_test.reset_index(drop=True),
            split_info,
        )

    frame = x.copy()
    frame["_target"] = y.to_numpy()
    frame["_time_order"] = pd.to_numeric(time_order, errors="coerce").to_numpy()
    frame["_split_group"] = split_group.astype(str).to_numpy() if split_group is not None else y.astype(str).to_numpy()
    frame["_row_order"] = np.arange(len(frame), dtype=np.int64)
    train_parts = []
    val_parts = []
    test_parts = []

    for (label, group_name), group in frame.groupby(["_target", "_split_group"], sort=True):
        valid_time = group["_time_order"].notna()
        usable_time = group.loc[valid_time, "_time_order"].nunique(dropna=True) >= 3
        if not usable_time:
            if temporal_fallback != "random":
                raise RuntimeError(
                    f"{label}: temporal split requires at least 3 usable unique timestamps. "
                    "Use --temporal-fallback random or --split random."
                )
            train_n, _ = ratio_counts(len(group), train_ratio, val_ratio)
            train, temp = train_test_split(group, train_size=train_n, shuffle=True, random_state=seed)
            val, test = train_test_split(
                temp,
                train_size=val_ratio / (val_ratio + test_ratio),
                shuffle=True,
                random_state=seed,
            )
            split_kind = "random_fallback"
        else:
            invalid_time_rows = group.loc[~valid_time]
            ordered = group.loc[valid_time].sort_values(["_time_order", "_row_order"], kind="mergesort").reset_index(drop=True)
            train_n, val_n = ratio_counts(len(ordered), train_ratio, val_ratio)
            cut1 = boundary_without_time_leak(ordered["_time_order"].to_numpy(), train_n)
            cut2 = boundary_without_time_leak(ordered["_time_order"].to_numpy(), train_n + val_n)
            if not (0 < cut1 < cut2 < len(ordered)):
                raise RuntimeError(f"{label}: temporal boundary collapsed. Use a finer timestamp column.")
            train = ordered.iloc[:cut1]
            val = ordered.iloc[cut1:cut2]
            test = ordered.iloc[cut2:]
            if not invalid_time_rows.empty:
                train = pd.concat([invalid_time_rows, train], ignore_index=True)
            split_kind = "temporal"

        train_parts.append(train)
        val_parts.append(val)
        test_parts.append(test)
        split_key = f"{label}/{group_name}"
        split_info["class_splits"][split_key] = {
            "kind": split_kind,
            "label": str(label),
            "split_group": str(group_name),
            "train": int(len(train)),
            "val": int(len(val)),
            "test": int(len(test)),
        }
        print(f"{split_kind} split {label}/{group_name}: train={len(train):,}, val={len(val):,}, test={len(test):,}")

    train = pd.concat(train_parts, ignore_index=True)
    val = pd.concat(val_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    drop_cols = ["_target", "_time_order", "_split_group", "_row_order"]
    return (
        train.drop(columns=drop_cols).reset_index(drop=True),
        val.drop(columns=drop_cols).reset_index(drop=True),
        test.drop(columns=drop_cols).reset_index(drop=True),
        train["_target"].reset_index(drop=True),
        val["_target"].reset_index(drop=True),
        test["_target"].reset_index(drop=True),
        split_info,
    )


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric dataframe columns to reduce memory use."""
    for col in df.columns:
        if df[col].dtype == "float64":
            df[col] = df[col].astype("float32")
        elif df[col].dtype == "int64":
            df[col] = df[col].astype("int32")
    return df


def counter_to_int_dict(values) -> dict[str, int]:
    """Convert a Counter-like object into JSON-safe integer counts."""
    return {str(label): int(count) for label, count in Counter(values).items()}


def choose_stratify(values: pd.Series, fallback: pd.Series, *, context: str) -> pd.Series:
    """Choose a viable stratification series, falling back when groups are too small."""
    counts = values.astype(str).value_counts()
    if not counts.empty and int(counts.min()) >= 2:
        return values.astype(str)
    print(f"{context}: source-level stratification has groups with <2 rows; falling back to final class labels.")
    return fallback.astype(str)


def split_final_holdout(
    df: pd.DataFrame,
    *,
    holdout_size: float,
    seed: int,
    source_stratified: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Reserve an optional final holdout before model selection splits."""
    if holdout_size <= 0:
        info = {"enabled": False, "holdout_size": float(holdout_size), "stratify": None}
        return df.reset_index(drop=True), df.iloc[0:0].copy().reset_index(drop=True), info
    if not 0 < holdout_size < 1:
        raise ValueError("--final-holdout-size must be 0 or a fraction between 0 and 1.")

    label_strata = df["type"].astype(str)
    if source_stratified and SOURCE_LABEL_COL in df.columns:
        source_strata = df[SOURCE_LABEL_COL].fillna(df["type"]).astype(str)
        strata = choose_stratify(source_strata, label_strata, context="Final holdout")
        stratify_name = SOURCE_LABEL_COL if strata.equals(source_strata) else "type"
    else:
        strata = label_strata
        stratify_name = "type"

    train_pool, final_holdout = train_test_split(
        df,
        test_size=holdout_size,
        stratify=strata,
        random_state=seed,
    )
    info = {
        "enabled": True,
        "holdout_size": float(holdout_size),
        "stratify": stratify_name,
        "pool_rows": int(len(train_pool)),
        "holdout_rows": int(len(final_holdout)),
        "holdout_counts": counter_to_int_dict(final_holdout["type"]),
    }
    if SOURCE_LABEL_COL in final_holdout.columns:
        info["holdout_source_counts"] = counter_to_int_dict(final_holdout[SOURCE_LABEL_COL].fillna(final_holdout["type"]))
    return train_pool.reset_index(drop=True), final_holdout.reset_index(drop=True), info


def looks_forced_categorical(column: str) -> bool:
    """Return True when a feature name should be treated as categorical."""
    lower = column.lower()
    return any(hint == lower or hint in lower for hint in FORCED_CATEGORICAL_HINTS)


def infer_feature_roles(
    x: pd.DataFrame,
    *,
    max_categorical_cardinality: int,
    numeric_valid_ratio: float,
) -> tuple[list[str], list[str], list[str], dict]:
    """Classify dataframe columns into categorical, numeric, and dropped features."""
    cat_cols: list[str] = []
    num_cols: list[str] = []
    dropped_cols: list[str] = []
    details: dict[str, dict] = {}

    for col in x.columns:
        if col in DROP_FEATURE_COLUMNS:
            dropped_cols.append(col)
            details[col] = {"role": "dropped_label_or_metadata"}
            continue

        if is_numeric_dtype(x[col]):
            non_missing = x[col].dropna()
            non_missing_n = int(len(non_missing))
            unique_n = int(non_missing.nunique(dropna=True))
            if non_missing_n == 0 or unique_n <= 1:
                dropped_cols.append(col)
                details[col] = {
                    "role": "dropped_constant_or_empty",
                    "non_missing": non_missing_n,
                    "unique": unique_n,
                }
                continue
            num_cols.append(col)
            details[col] = {
                "role": "numeric_native",
                "non_missing": non_missing_n,
                "unique": unique_n,
                "numeric_valid": non_missing_n,
                "numeric_ratio": 1.0,
                "sample": sample_values(x[col], n=5),
            }
            continue

        values = x[col].astype(str).str.strip()
        non_missing = values[~values.str.lower().isin(MISSING_TOKENS)]
        non_missing_n = int(len(non_missing))
        unique_n = int(non_missing.nunique(dropna=True))

        if non_missing_n == 0 or unique_n <= 1:
            dropped_cols.append(col)
            details[col] = {
                "role": "dropped_constant_or_empty",
                "non_missing": non_missing_n,
                "unique": unique_n,
            }
            continue

        if looks_forced_categorical(col):
            if unique_n > max_categorical_cardinality:
                dropped_cols.append(col)
                role = "dropped_high_cardinality_categorical"
            else:
                cat_cols.append(col)
                role = "categorical_forced"
        elif unique_n <= max_categorical_cardinality:
            cat_cols.append(col)
            role = "categorical"
        else:
            dropped_cols.append(col)
            role = "dropped_high_cardinality_text"

        details[col] = {
            "role": role,
            "non_missing": non_missing_n,
            "unique": unique_n,
            "numeric_valid": 0,
            "numeric_ratio": 0.0,
            "sample": sample_values(x[col], n=5),
        }

    return cat_cols, num_cols, dropped_cols, details


def se_block_1d(x, reduction: int = 16):
    """Apply a one-dimensional squeeze-and-excitation block."""
    channels = int(x.shape[-1])
    squeeze = GlobalAveragePooling1D()(x)
    squeeze = Reshape((1, channels))(squeeze)
    hidden = max(channels // reduction, 4)
    excite = Dense(hidden, activation="relu", kernel_initializer="he_normal", use_bias=False)(squeeze)
    excite = Dense(channels, activation="sigmoid", kernel_initializer="he_normal", use_bias=False)(excite)
    return Multiply()([x, excite])


def sedwnet_block(x, filters: int, stride: int = 1, se_reduction: int = 16, dropout: float = 0.0):
    """Apply one residual separable-convolution SE-DWNet block."""
    residual = x
    if stride != 1 or int(x.shape[-1]) != filters:
        residual = Conv1D(filters, 1, strides=stride, padding="same", kernel_initializer="he_normal")(residual)
        residual = BatchNormalization()(residual)

    x = SeparableConv1D(
        filters,
        3,
        strides=stride,
        padding="same",
        depthwise_initializer="he_normal",
        pointwise_initializer="he_normal",
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = SeparableConv1D(
        filters,
        3,
        strides=1,
        padding="same",
        depthwise_initializer="he_normal",
        pointwise_initializer="he_normal",
    )(x)
    x = BatchNormalization()(x)
    x = se_block_1d(x, reduction=se_reduction)
    if dropout > 0:
        x = Dropout(dropout)(x)
    x = Add()([x, residual])
    x = Activation("relu")(x)
    return x


def build_se_dwnet(input_dim: int, num_classes: int) -> Model:
    """Build the SE-DWNet classifier for tabular Edge-IIoTSet features."""
    inputs = Input(shape=(input_dim,), name="tabular_input")
    x = Reshape((input_dim, 1), name="feat_as_1d")(inputs)
    x = Conv1D(64, 3, padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = sedwnet_block(x, filters=64, stride=1)
    x = sedwnet_block(x, filters=128, stride=2)
    x = sedwnet_block(x, filters=256, stride=2)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation="softmax", dtype="float32")(x)
    return Model(inputs, outputs, name="SE_DWNet_Edge_IIoTset")


def build_parser(project_root_path: str) -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Train SE-DWNet on the capped 6-class Edge-IIoTset CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", default=default_csv(), help="Training CSV.")
    parser.add_argument("--label-col", default=None, help="Target label column. Auto-detected when omitted.")
    parser.add_argument("--output-dir", default=default_output_dir(project_root_path), help="Artifact directory.")
    parser.add_argument("--target-k", type=int, default=TARGET_K, help="Number of selected features.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--split", choices=("random", "temporal"), default="random", help="Validation split strategy.")
    parser.add_argument("--time-col", default=None, help="Timestamp column for temporal split if available.")
    parser.add_argument(
        "--temporal-fallback",
        choices=("error", "random"),
        default="random",
        help="What to do when a class cannot be split temporally.",
    )
    parser.add_argument("--test-size", type=float, default=0.15, help="Internal test fraction from the post-holdout training pool.")
    parser.add_argument("--val-size", type=float, default=0.15, help="Internal validation fraction from the post-holdout training pool.")
    parser.add_argument(
        "--final-holdout-size",
        type=float,
        default=0.05,
        help="Per-run final holdout fraction reserved before train/val/test splitting. Use 0 to disable.",
    )
    parser.add_argument(
        "--no-source-stratified-holdout",
        action="store_true",
        help="Reserve final holdout by final class only instead of source_label subtype.",
    )
    parser.add_argument("--smote", choices=("auto", "on", "off"), default="auto", help="SMOTE balancing mode.")
    parser.add_argument(
        "--smote-imbalance-ratio",
        type=float,
        default=1.25,
        help="Auto-SMOTE threshold: max train class count / min train class count.",
    )
    parser.add_argument("--no-smote", action="store_true", help="Deprecated alias for --smote off.")
    parser.add_argument("--no-dedupe", action="store_true", help="Disable exact row deduplication before splitting.")
    parser.add_argument(
        "--max-categorical-cardinality",
        type=int,
        default=512,
        help="Drop nonnumeric categorical/text columns with more unique values than this.",
    )
    parser.add_argument(
        "--numeric-valid-ratio",
        type=float,
        default=0.98,
        help="Minimum non-missing numeric parse ratio needed to treat a feature as numeric.",
    )
    parser.add_argument(
        "--feature-inference-sample-size",
        type=int,
        default=100_000,
        help="Rows sampled for numeric/categorical feature role inference. 0 uses the full dataset.",
    )
    return parser


def main() -> None:
    """Run the command-line entry point."""
    project_root_path = project_root()
    args = build_parser(project_root_path).parse_args()
    artifact_dir = os.path.abspath(args.output_dir)
    os.makedirs(artifact_dir, exist_ok=True)

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    print("=== SE-DWNet Edge-IIoTset Training ===")
    print(f"GPUs available: {len(tf.config.list_physical_devices('GPU'))}")
    print(f"Project root: {project_root_path}")
    print(f"CSV:          {args.csv}")
    print(f"Artifacts:    {artifact_dir}")
    print(f"Split mode:   {args.split}")
    print(f"SMOTE mode:   {'off' if args.no_smote else args.smote}")

    if not os.path.exists(args.csv):
        raise FileNotFoundError(args.csv)

    print("Loading CSV into dataframe...", flush=True)
    df = pd.read_csv(args.csv, low_memory=False, on_bad_lines="skip")
    print(f"Loaded dataframe shape: {df.shape}", flush=True)
    df.columns = df.columns.str.strip()
    target_col = label_column(df, args.label_col)
    if target_col != "type":
        df["type"] = df[target_col]

    if args.split == "temporal":
        df["_time_order"] = derive_time_order(df, explicit_col=args.time_col)
    else:
        df["_time_order"] = np.arange(len(df), dtype=np.float64)

    drop_metadata = [col for col in TIME_COLS if col in df.columns]
    drop_metadata += [col for col in LABEL_CANDIDATES if col in df.columns and col != "type"]
    df.drop(columns=drop_metadata, errors="ignore", inplace=True)
    print(f"Target column: {target_col}")
    print(f"Dropped metadata columns: {drop_metadata}")

    labels_norm = df["type"].astype(str).str.strip().str.lower().map(canon_label)
    keep = labels_norm.isin(TARGET_CLASSES)
    dropped = int((~keep).sum())
    if dropped:
        print(f"Dropping {dropped:,} rows with unsupported labels.")
    df = df.loc[keep].copy()
    df["type"] = labels_norm.loc[keep].to_numpy()
    if df.empty:
        raise RuntimeError("No rows left after filtering to the Edge-IIoTset 6-class taxonomy.")
    df = df.reset_index(drop=True)
    class_counts_after_cleanup = counter_to_int_dict(df["type"])
    print(f"Class counts after cleanup: {class_counts_after_cleanup}")

    df, final_holdout_df, final_holdout_info = split_final_holdout(
        df,
        holdout_size=args.final_holdout_size,
        seed=args.seed,
        source_stratified=not args.no_source_stratified_holdout,
    )
    print(f"Training pool rows after final holdout reserve: {len(df):,}")
    if not final_holdout_df.empty:
        print(f"Final holdout rows: {len(final_holdout_df):,}")
        print(f"Final holdout counts: {counter_to_int_dict(final_holdout_df['type'])}")

    print("Preparing feature matrix...", flush=True)
    y_all = df["type"].astype(str).reset_index(drop=True)
    time_order_all = pd.to_numeric(df["_time_order"], errors="coerce").reset_index(drop=True)
    if SOURCE_LABEL_COL in df.columns:
        split_group_all = df[SOURCE_LABEL_COL].fillna(df["type"]).astype(str).reset_index(drop=True)
        print(f"Source-aware temporal groups: {counter_to_int_dict(split_group_all)}", flush=True)
    else:
        split_group_all = df["type"].astype(str).reset_index(drop=True)
        print("No source_label column found; temporal split will group by final class only.", flush=True)
    x_all = df.drop(columns=["type", "_time_order"]).reset_index(drop=True)
    print(f"Raw feature matrix shape: {x_all.shape}", flush=True)

    y_final_holdout = final_holdout_df["type"].astype(str).reset_index(drop=True) if not final_holdout_df.empty else pd.Series(dtype=str)
    x_final_holdout = (
        final_holdout_df.drop(columns=["type", "_time_order"]).reset_index(drop=True)
        if not final_holdout_df.empty
        else pd.DataFrame(columns=x_all.columns)
    )

    if args.feature_inference_sample_size and len(x_all) > args.feature_inference_sample_size:
        x_role_sample = x_all.sample(n=args.feature_inference_sample_size, random_state=args.seed)
    else:
        x_role_sample = x_all
    print(f"Inferring feature roles from {len(x_role_sample):,}/{len(x_all):,} rows...", flush=True)
    cat_cols, num_cols, dropped_feature_cols, feature_role_details = infer_feature_roles(
        x_role_sample,
        max_categorical_cardinality=args.max_categorical_cardinality,
        numeric_valid_ratio=args.numeric_valid_ratio,
    )
    del x_role_sample
    if dropped_feature_cols:
        x_all.drop(columns=dropped_feature_cols, errors="ignore", inplace=True)
        x_final_holdout.drop(columns=dropped_feature_cols, errors="ignore", inplace=True)

    valid_cat_cols = [col for col in cat_cols if col in x_all.columns]
    num_cols = [col for col in num_cols if col in x_all.columns]

    for col in x_all.columns:
        if col not in x_final_holdout.columns:
            x_final_holdout[col] = ""
    x_final_holdout = x_final_holdout[x_all.columns]

    for col in valid_cat_cols:
        x_all[col] = x_all[col].fillna("missing").replace("-", "missing").astype(str)
        if col in x_final_holdout.columns:
            x_final_holdout[col] = x_final_holdout[col].fillna("missing").replace("-", "missing").astype(str)

    for col in num_cols:
        if is_numeric_dtype(x_all[col]):
            x_all[col] = x_all[col].astype("float32", copy=False)
        else:
            x_all[col] = np.nan
        if col in x_final_holdout.columns:
            if is_numeric_dtype(x_final_holdout[col]):
                x_final_holdout[col] = x_final_holdout[col].astype("float32", copy=False)
            else:
                x_final_holdout[col] = np.nan

    x_all.replace([np.inf, -np.inf], 0, inplace=True)
    x_all = x_all.fillna(0)
    x_final_holdout.replace([np.inf, -np.inf], 0, inplace=True)
    x_final_holdout = x_final_holdout.fillna(0)

    for col in LOG_COLS:
        if col in x_all.columns and is_numeric_dtype(x_all[col]):
            x_all[col] = np.log1p(x_all[col].fillna(0).clip(lower=0))
        if col in x_final_holdout.columns and is_numeric_dtype(x_final_holdout[col]):
            x_final_holdout[col] = np.log1p(x_final_holdout[col].fillna(0).clip(lower=0))

    constant_cols = [col for col in x_all.columns if x_all[col].nunique(dropna=False) <= 1]
    if constant_cols:
        x_all.drop(columns=constant_cols, inplace=True)
        x_final_holdout.drop(columns=constant_cols, errors="ignore", inplace=True)
        valid_cat_cols = [col for col in valid_cat_cols if col not in constant_cols]
        num_cols = [col for col in num_cols if col not in constant_cols]
        print(f"Dropped constant columns after cleanup: {len(constant_cols)}")

    for col in x_all.columns:
        if col not in x_final_holdout.columns:
            x_final_holdout[col] = 0
    x_final_holdout = x_final_holdout[x_all.columns]

    if not x_all.columns.tolist():
        raise RuntimeError("No usable feature columns remain after preprocessing.")

    x_all = optimize_dtypes(x_all)
    x_final_holdout = optimize_dtypes(x_final_holdout)
    print(f"Cleaned data shape: {x_all.shape}")
    print(f"Numeric columns:     {len(num_cols)}")
    print(f"Categorical columns: {len(valid_cat_cols)}")
    print(f"Dropped features:    {len(dropped_feature_cols) + len(constant_cols)}")

    dedupe_info = {"before": int(len(x_all)), "after": int(len(x_all)), "dropped": 0}
    if not args.no_dedupe:
        x_all, y_all, time_order_all, split_group_all, dedupe_info = dedupe_model_visible_rows(
            x_all,
            y_all,
            time_order_all,
            split_group_all,
        )
    else:
        x_all = x_all.reset_index(drop=True)
        y_all = y_all.reset_index(drop=True)
        time_order_all = time_order_all.reset_index(drop=True)
        split_group_all = split_group_all.reset_index(drop=True)

    if not (0 < args.val_size < 1 and 0 < args.test_size < 1 and args.val_size + args.test_size < 1):
        raise ValueError("--val-size and --test-size must be positive and sum to less than 1.0")
    train_ratio = 1.0 - args.val_size - args.test_size
    print(f"Splitting data ({args.split} {train_ratio:.2f}/{args.val_size:.2f}/{args.test_size:.2f})...")

    x_train_df, x_val_df, x_test_df, y_train_str, y_val_str, y_test_str, split_info = split_frames(
        x_all,
        y_all,
        time_order_all,
        split_mode=args.split,
        train_ratio=train_ratio,
        val_ratio=args.val_size,
        test_ratio=args.test_size,
        seed=args.seed,
        temporal_fallback=args.temporal_fallback,
        split_group=split_group_all,
    )

    split_group_counts = counter_to_int_dict(split_group_all)

    del df, final_holdout_df, x_all, y_all, time_order_all, split_group_all
    gc.collect()

    le_target = LabelEncoder()
    le_target.fit(y_train_str)
    y_train = le_target.transform(y_train_str)
    y_val = le_target.transform(y_val_str)
    y_test = le_target.transform(y_test_str)

    num_classes = len(le_target.classes_)
    class_names = le_target.classes_.tolist()
    print(f"Classes ({num_classes}): {class_names}")

    x_train_df = x_train_df.reset_index(drop=True)
    x_final_holdout_df = x_final_holdout.reset_index(drop=True)
    del x_final_holdout

    encoders = {}
    for col in valid_cat_cols:
        encoder = SafeLabelEncoder()
        x_train_df[col] = encoder.fit(x_train_df[col]).transform(x_train_df[col])
        x_val_df[col] = encoder.transform(x_val_df[col])
        x_test_df[col] = encoder.transform(x_test_df[col])
        if not x_final_holdout_df.empty:
            x_final_holdout_df[col] = encoder.transform(x_final_holdout_df[col])
        encoders[col] = encoder

    scaler_num = MinMaxScaler()
    x_train_df[num_cols] = scaler_num.fit_transform(x_train_df[num_cols].values)
    x_val_df[num_cols] = scaler_num.transform(x_val_df[num_cols].values)
    x_test_df[num_cols] = scaler_num.transform(x_test_df[num_cols].values)
    if not x_final_holdout_df.empty:
        x_final_holdout_df[num_cols] = scaler_num.transform(x_final_holdout_df[num_cols].values)

    print(f"Selecting top {args.target_k} features (mutual information)...")
    feature_names = x_train_df.columns.tolist()
    discrete_mask = np.array([col in valid_cat_cols for col in feature_names], dtype=bool)
    mi_scorer = partial(mutual_info_classif, discrete_features=discrete_mask, n_neighbors=3, random_state=args.seed, n_jobs=-1)
    selector = SelectKBest(score_func=mi_scorer, k=min(args.target_k, x_train_df.shape[1]))
    selector.fit(x_train_df, y_train)

    x_train_sel = selector.transform(x_train_df).astype(np.float32)
    x_val_sel = selector.transform(x_val_df).astype(np.float32)
    x_test_sel = selector.transform(x_test_df).astype(np.float32)
    x_final_holdout_sel = (
        selector.transform(x_final_holdout_df).astype(np.float32)
        if not x_final_holdout_df.empty
        else np.empty((0, selector.get_support().sum()), dtype=np.float32)
    )

    selected_mask = selector.get_support()
    final_features = x_train_df.columns[selected_mask].tolist()
    print(f"Selected features ({len(final_features)}): {final_features}")

    with open(os.path.join(artifact_dir, "final_features.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(final_features) + "\n")

    final_scaler = MinMaxScaler()
    x_train_sel = np.nan_to_num(final_scaler.fit_transform(x_train_sel)).astype(np.float32)
    x_val_sel = np.nan_to_num(final_scaler.transform(x_val_sel)).astype(np.float32)
    x_test_sel = np.nan_to_num(final_scaler.transform(x_test_sel)).astype(np.float32)
    if len(x_final_holdout_sel):
        x_final_holdout_sel = np.nan_to_num(final_scaler.transform(x_final_holdout_sel)).astype(np.float32)

    del x_train_df, x_val_df, x_test_df, x_final_holdout_df
    gc.collect()

    print("Applying SMOTE for class balancing...")
    train_counts = Counter(y_train)
    print(f"Pre-SMOTE:  {dict(train_counts)}")
    max_class_count = max(train_counts.values())
    min_class_count = min(train_counts.values())
    imbalance_ratio = float(max_class_count) / max(float(min_class_count), 1.0)
    smote_mode = "off" if args.no_smote else args.smote
    use_smote = smote_mode == "on" or (smote_mode == "auto" and imbalance_ratio >= args.smote_imbalance_ratio)
    print(
        f"SMOTE mode: {smote_mode} "
        f"(imbalance ratio={imbalance_ratio:.3f}, auto threshold={args.smote_imbalance_ratio:.3f})"
    )

    smote_strategy = {}
    if use_smote:
        for cls, count in train_counts.items():
            if count < 2:
                continue
            target = min(int(count * SMOTE_MAX_MULTIPLIER), max_class_count)
            if target > count:
                smote_strategy[cls] = target

    if smote_strategy:
        min_count = min(train_counts[cls] for cls in smote_strategy)
        k_neighbors = max(1, min(5, min_count - 1))
        smote = SMOTE(sampling_strategy=smote_strategy, random_state=args.seed, k_neighbors=k_neighbors)
        x_train_bal, y_train_bal = smote.fit_resample(x_train_sel, y_train)
        print(f"Post-SMOTE: {dict(Counter(y_train_bal))}")
    else:
        x_train_bal, y_train_bal = x_train_sel, y_train
        if smote_mode == "off":
            reason = "disabled"
        elif not use_smote:
            reason = "class balance is already close enough"
        else:
            reason = "no eligible classes"
        print(f"SMOTE skipped ({reason})")

    y_train_onehot = to_categorical(y_train_bal, num_classes=num_classes).astype(np.float32)
    y_val_onehot = to_categorical(y_val, num_classes=num_classes).astype(np.float32)

    model = build_se_dwnet(x_train_bal.shape[1], num_classes)
    optimizer = Adam(learning_rate=args.learning_rate, clipnorm=1.0)

    if USE_FOCAL_LOSS:
        counts = np.bincount(y_train_bal, minlength=num_classes).astype(np.float32)
        inv = 1.0 / np.maximum(counts, 1.0)
        alpha_vec = inv / inv.mean()
        alpha_vec = np.clip(alpha_vec, *FOCAL_ALPHA_CLIP).astype(np.float32)
        print(f"Focal alpha: min={alpha_vec.min():.4f}, mean={alpha_vec.mean():.4f}, max={alpha_vec.max():.4f}")
        loss_fn = tf.keras.losses.CategoricalFocalCrossentropy(alpha=alpha_vec.tolist(), gamma=FOCAL_GAMMA, from_logits=False)
        loss_info = {
            "name": "CategoricalFocalCrossentropy",
            "gamma": float(FOCAL_GAMMA),
            "alpha_clip": list(FOCAL_ALPHA_CLIP),
            "alpha_min": float(alpha_vec.min()),
            "alpha_mean": float(alpha_vec.mean()),
            "alpha_max": float(alpha_vec.max()),
        }
    else:
        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0.1)
        loss_info = {"name": "CategoricalCrossentropy", "label_smoothing": 0.1}

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
    ]

    print("Training...")
    history = model.fit(
        x_train_bal,
        y_train_onehot,
        validation_data=(x_val_sel, y_val_onehot),
        epochs=args.max_epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    print("Saving artifacts...")
    model_path = os.path.join(artifact_dir, "se_dwnet_model.keras")
    model.save(model_path)

    pipeline_bundle = {
        "scaler_num": scaler_num,
        "selector": selector,
        "final_scaler": final_scaler,
        "encoders": encoders,
        "target_encoder": le_target,
        "features": final_features,
        "valid_cat_cols": valid_cat_cols,
        "num_cols": num_cols,
        "seed": args.seed,
        "loss": loss_info,
        "dataset_name": "edge_iiotset_6class",
        "data_csv": os.path.abspath(args.csv),
        "split_mode": args.split,
    }

    pipeline_path = os.path.join(artifact_dir, "preprocessing_pipeline.pkl")
    with open(pipeline_path, "wb") as f:
        pickle.dump(pipeline_bundle, f)

    feature_roles_path = os.path.join(artifact_dir, "feature_roles.json")
    with open(feature_roles_path, "w", encoding="utf-8") as f:
        json.dump(feature_role_details, f, indent=2)

    metadata = {
        "script": "resnet_edge_iiotset.py",
        "dataset_name": "edge_iiotset_6class",
        "data_csv": os.path.abspath(args.csv),
        "artifact_dir": artifact_dir,
        "model_path": model_path,
        "pipeline_path": pipeline_path,
        "feature_roles_path": feature_roles_path,
        "classes": class_names,
        "selected_features": final_features,
        "target_k": args.target_k,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "epochs_ran": int(len(history.history.get("loss", []))),
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "split_mode": args.split,
        "split_info": split_info,
        "split_group_column": SOURCE_LABEL_COL,
        "split_group_counts": split_group_counts,
        "time_col": args.time_col,
        "temporal_fallback": args.temporal_fallback,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "final_holdout": final_holdout_info,
        "dedupe": dedupe_info,
        "smote_mode": smote_mode,
        "smote_enabled": bool(use_smote and smote_strategy),
        "smote_imbalance_ratio": imbalance_ratio,
        "smote_auto_threshold": args.smote_imbalance_ratio,
        "smote_max_multiplier": SMOTE_MAX_MULTIPLIER,
        "loss": loss_info,
        "class_counts_after_cleanup": class_counts_after_cleanup,
        "train_counts": counter_to_int_dict(y_train_str),
        "val_counts": counter_to_int_dict(y_val_str),
        "test_counts": counter_to_int_dict(y_test_str),
        "numeric_columns": num_cols,
        "categorical_columns": valid_cat_cols,
        "dropped_feature_columns": dropped_feature_cols + constant_cols,
        "max_categorical_cardinality": args.max_categorical_cardinality,
        "numeric_valid_ratio": args.numeric_valid_ratio,
        "feature_inference_sample_size": args.feature_inference_sample_size,
    }
    with open(os.path.join(artifact_dir, "training_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    with open(os.path.join(artifact_dir, "training_metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    history_path = os.path.join(artifact_dir, "training_history.json")
    history_data = {key: [float(value) for value in values] for key, values in history.history.items()}
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_data, f, indent=2)

    print("Evaluating on test set...")
    test_probs = model.predict(x_test_sel, batch_size=args.batch_size)
    test_pred = np.argmax(test_probs, axis=1)

    y_test_readable = le_target.inverse_transform(y_test)
    y_pred_readable = le_target.inverse_transform(test_pred)

    report_str = classification_report(
        y_test_readable,
        y_pred_readable,
        labels=class_names,
        target_names=class_names,
        zero_division=0,
        digits=4,
    )
    print("\nClassification Report (TEST):")
    print(report_str)

    with open(os.path.join(artifact_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write("=== SE-DWNet Edge-IIoTset Evaluation ===\n")
        f.write(f"CSV: {os.path.abspath(args.csv)}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Loss: {loss_info}\n\n")
        f.write(report_str)

    cm = confusion_matrix(y_test_readable, y_pred_readable, labels=le_target.classes_)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=le_target.classes_, yticklabels=le_target.classes_, cmap="Blues")
    plt.title("SE-DWNet Edge-IIoTset Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(artifact_dir, "se_dwnet_edge_iiotset_confusion_matrix.png"), dpi=200)
    plt.close()

    if len(x_final_holdout_sel):
        print("Evaluating on final untouched holdout...")
        holdout_probs = model.predict(x_final_holdout_sel, batch_size=args.batch_size)
        holdout_pred = np.argmax(holdout_probs, axis=1)
        holdout_pred_readable = le_target.inverse_transform(holdout_pred)
        holdout_true = y_final_holdout.to_numpy().astype(str)
        holdout_report = classification_report(
            holdout_true,
            holdout_pred_readable,
            labels=class_names,
            target_names=class_names,
            zero_division=0,
            digits=4,
        )
        print("\nClassification Report (FINAL HOLDOUT):")
        print(holdout_report)

        with open(os.path.join(artifact_dir, "final_holdout_classification_report.txt"), "w", encoding="utf-8") as f:
            f.write("=== SE-DWNet Edge-IIoTset Final Holdout Evaluation ===\n")
            f.write(f"CSV: {os.path.abspath(args.csv)}\n")
            f.write(f"Model: {model_path}\n")
            f.write(f"Loss: {loss_info}\n")
            f.write(f"Holdout: {final_holdout_info}\n\n")
            f.write(holdout_report)

        holdout_cm = confusion_matrix(holdout_true, holdout_pred_readable, labels=le_target.classes_)
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            holdout_cm,
            annot=True,
            fmt="d",
            xticklabels=le_target.classes_,
            yticklabels=le_target.classes_,
            cmap="Blues",
        )
        plt.title("SE-DWNet Edge-IIoTset Final Holdout Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(os.path.join(artifact_dir, "se_dwnet_edge_iiotset_final_holdout_confusion_matrix.png"), dpi=200)
        plt.close()

        holdout_pred_data = {
            "true_class": holdout_true,
            "predicted_class": holdout_pred_readable,
            "confidence": np.max(holdout_probs, axis=1),
            "correct": holdout_true == holdout_pred_readable,
        }
        for index, cls in enumerate(class_names):
            holdout_pred_data[f"prob_{cls}"] = holdout_probs[:, index]
        pd.DataFrame(holdout_pred_data).to_csv(
            os.path.join(artifact_dir, "final_holdout_predictions.csv"),
            index=False,
        )

    print(f"DONE. Artifacts saved to: {artifact_dir}")


if __name__ == "__main__":
    main()
