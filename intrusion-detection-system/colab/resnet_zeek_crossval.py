"""SE-DWNet training for the Zeek-flow Edge cross-validation dataset.

Expected dataset from resnet/edge_crossval_lab/build_dataset_from_pcaps.sh:

    CSV:      data/zeek_crossval.csv, or
              data/edge_crossval/zeek_crossval.csv
    Classes:  backdoor, dos_ddos, injection, normal, password, scanning
    Target:   roughly 60,000 rows per class after the cap. If one class is
              short, SMOTE is applied only to the training split by default.

This trainer is for the new Zeek-flow dataset, not the old Edge-IIoTset
TShark packet-feature dataset. It drops lab identity columns such as Zeek UID,
IP addresses, source_label, and Kali boolean flags before training. Target-side
direction flags are kept because an IDS normally knows which side is protected.

Default split is random and source-family stratified so all generated attack
families participate in train/validation/test and the full CSV is used. Use
``--split temporal`` or ``--split source`` for stricter drift tests. Final
holdout defaults to a 5% random untouched post-training check. SMOTE is enabled
in auto mode and only affects the training split when class imbalance remains.
Exact-row deduplication is disabled by default because repeated Zeek flows are
traffic volume signal, especially for DoS, scanning, and password attacks.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import pickle
import re
import shutil
import sys
from collections import Counter
from functools import partial

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
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

try:
    from imblearn.over_sampling import SMOTE
except ImportError:  # pragma: no cover - only needed when SMOTE is explicitly enabled.
    SMOTE = None


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from resnet_edge_iiotset import (  # noqa: E402
    FOCAL_ALPHA_CLIP,
    FOCAL_GAMMA,
    MISSING_TOKENS,
    SafeLabelEncoder,
    boundary_without_time_leak,
    canon_label,
    choose_stratify,
    counter_to_int_dict,
    dedupe_model_visible_rows,
    derive_time_order,
    infer_feature_roles,
    optimize_dtypes,
    ratio_counts,
    split_final_holdout,
    split_frames,
)


TARGET_CLASSES = {"backdoor", "dos_ddos", "injection", "normal", "password", "scanning"}
LABEL_CANDIDATES = ("type", "attack", "attack_type", "category", "class", "label", "Label")
SOURCE_LABEL_COL = "source_label"
SOURCE_GROUP_COL = "_source_group"

TARGET_K = 192
BATCH_SIZE = 1024
MAX_EPOCHS = 100
LEARNING_RATE = 5e-4
SEED = 42
SMOTE_MAX_MULTIPLIER = 2
MODEL_DROPOUT = 0.35

ZEEK_METADATA_COLUMNS = {
    "uid",
    "ts",
    "datetime",
    "timestamp",
    "frame_time",
    "frame_time_epoch",
    "split_time",
    "id_orig_h",
    "id_resp_h",
    "source_label",
    "src_is_kali",
    "dst_is_kali",
}

LOG_COLS = [
    "duration",
    "orig_bytes",
    "resp_bytes",
    "missed_bytes",
    "orig_pkts",
    "orig_ip_bytes",
    "resp_pkts",
    "resp_ip_bytes",
    "flow_total_bytes",
    "flow_total_pkts",
    "http_request_body_len",
    "http_response_body_len",
    "dns_answers_count",
    "http_count",
    "dns_count",
    "ssh_count",
    "ssl_count",
    "files_count",
    "notice_count",
    "weird_count",
]


def project_root() -> str:
    env_root = os.environ.get("TON_IOT_PROJECT_ROOT")
    if env_root and os.path.isdir(env_root):
        return os.path.abspath(env_root)
    cwd = os.path.abspath(os.getcwd())
    if os.path.isdir(os.path.join(cwd, "resnet")):
        return cwd
    return os.path.normpath(os.path.join(SCRIPT_DIR, ".."))


def default_csv(project_root_path: str) -> str:
    candidates = [
        os.path.join(project_root_path, "data", "zeek_crossval.csv"),
        os.path.join(project_root_path, "data", "edge_crossval", "zeek_crossval.csv"),
        os.path.join(project_root_path, "data", "edge_zeek_crossval.csv"),
        os.path.join(project_root_path, "data", "zeek_crossval_60k.csv"),
        os.path.join(project_root_path, "data", "zeek_crossval_windows_5s.csv"),
        os.path.join(project_root_path, "data", "edge_crossval", "zeek_crossval_windows_5s.csv"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def default_output_dir(project_root_path: str) -> str:
    return os.path.join(project_root_path, "artifacts", "se_dwnet_zeek_crossval_60k")


def label_column(df: pd.DataFrame, explicit: str | None) -> str:
    if explicit:
        if explicit not in df.columns:
            raise RuntimeError(f"Label column '{explicit}' not found. Columns: {list(df.columns[:60])}")
        return explicit
    for col in LABEL_CANDIDATES:
        if col not in df.columns:
            continue
        values = set(df[col].dropna().astype(str).head(50_000).map(canon_label).unique())
        if values.intersection(TARGET_CLASSES):
            return col
    raise RuntimeError(f"Could not identify target label column. Columns: {list(df.columns[:60])}")


def source_family(value: object) -> str:
    raw = str(value).strip()
    if not raw or raw.lower() in MISSING_TOKENS:
        return "unknown"
    return re.sub(r"_r\d+$", "", raw)


def build_source_groups(df: pd.DataFrame, mode: str) -> pd.Series:
    if SOURCE_LABEL_COL not in df.columns:
        return df["type"].astype(str)
    values = df[SOURCE_LABEL_COL].fillna(df["type"]).astype(str)
    if mode == "family":
        return values.map(source_family)
    return values


def split_frames_by_source_group(
    x: pd.DataFrame,
    y: pd.Series,
    source_group: pd.Series,
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, dict]:
    frame = x.copy()
    frame["_target"] = y.astype(str).to_numpy()
    frame["_source_group"] = source_group.astype(str).to_numpy()
    frame["_row_order"] = np.arange(len(frame), dtype=np.int64)

    rng = np.random.default_rng(seed)
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    split_info = {"mode": "source", "class_splits": {}}

    for label, class_frame in frame.groupby("_target", sort=True):
        groups = sorted(class_frame["_source_group"].dropna().astype(str).unique().tolist())
        if len(groups) < 3:
            train_n, _ = ratio_counts(len(class_frame), train_ratio, val_ratio)
            train, temp = train_test_split(class_frame, train_size=train_n, shuffle=True, random_state=seed)
            val, test = train_test_split(
                temp,
                train_size=val_ratio / (val_ratio + test_ratio),
                shuffle=True,
                random_state=seed,
            )
            split_kind = "random_fallback"
            split_info["class_splits"][str(label)] = {
                "kind": split_kind,
                "reason": "fewer_than_3_source_groups",
                "source_groups": groups,
                "train": int(len(train)),
                "val": int(len(val)),
                "test": int(len(test)),
            }
        else:
            shuffled_groups = np.array(groups, dtype=object)
            rng.shuffle(shuffled_groups)
            train_group_n, val_group_n = ratio_counts(len(shuffled_groups), train_ratio, val_ratio)
            train_groups = set(shuffled_groups[:train_group_n].tolist())
            val_groups = set(shuffled_groups[train_group_n : train_group_n + val_group_n].tolist())
            test_groups = set(shuffled_groups[train_group_n + val_group_n :].tolist())
            train = class_frame[class_frame["_source_group"].isin(train_groups)]
            val = class_frame[class_frame["_source_group"].isin(val_groups)]
            test = class_frame[class_frame["_source_group"].isin(test_groups)]
            split_kind = "source_group"
            split_info["class_splits"][str(label)] = {
                "kind": split_kind,
                "train_groups": sorted(train_groups),
                "val_groups": sorted(val_groups),
                "test_groups": sorted(test_groups),
                "train": int(len(train)),
                "val": int(len(val)),
                "test": int(len(test)),
            }

        print(f"{split_kind} split {label}: train={len(train):,}, val={len(val):,}, test={len(test):,}")
        train_parts.append(train)
        val_parts.append(val)
        test_parts.append(test)

    train = pd.concat(train_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val = pd.concat(val_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test = pd.concat(test_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    drop_cols = ["_target", "_source_group", "_row_order"]
    return (
        train.drop(columns=drop_cols).reset_index(drop=True),
        val.drop(columns=drop_cols).reset_index(drop=True),
        test.drop(columns=drop_cols).reset_index(drop=True),
        train["_target"].reset_index(drop=True),
        val["_target"].reset_index(drop=True),
        test["_target"].reset_index(drop=True),
        split_info,
    )


def dedupe_rows_with_meta(
    x: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    y_col: str = "type",
    time_col: str = "_time_order",
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Drop duplicate model-visible rows while keeping aligned split metadata."""
    feature_hash = pd.util.hash_pandas_object(x, index=False).astype("uint64")
    work = meta.reset_index(drop=True).copy()
    work["_row_pos"] = np.arange(len(work), dtype=np.int64)
    work["_feature_hash"] = feature_hash.to_numpy()
    work["_sort_time"] = pd.to_numeric(work[time_col], errors="coerce").fillna(np.inf).to_numpy()

    label_counts = work.groupby("_feature_hash")[y_col].nunique()
    conflict_hashes = set(label_counts[label_counts > 1].index.tolist())
    if conflict_hashes:
        conflict_mask = work["_feature_hash"].isin(conflict_hashes)
        print(f"Dropping {int(conflict_mask.sum()):,} duplicate feature rows with conflicting labels.")
        keep = ~conflict_mask.to_numpy()
        x = x.loc[keep].reset_index(drop=True)
        meta = meta.loc[keep].reset_index(drop=True)
        work = meta.copy()
        work["_row_pos"] = np.arange(len(work), dtype=np.int64)
        work["_feature_hash"] = pd.util.hash_pandas_object(x, index=False).astype("uint64").to_numpy()
        work["_sort_time"] = pd.to_numeric(work[time_col], errors="coerce").fillna(np.inf).to_numpy()

    before = len(x)
    keep_positions = (
        work.sort_values(["_sort_time", "_row_pos"], kind="mergesort")
        .drop_duplicates("_feature_hash", keep="first")["_row_pos"]
        .to_numpy()
    )
    x_out = x.iloc[keep_positions].reset_index(drop=True)
    meta_out = meta.iloc[keep_positions].reset_index(drop=True)
    info = {"before": int(before), "after": int(len(x_out)), "dropped": int(before - len(x_out))}
    print(f"Deduplicated model-visible rows: {before:,} -> {len(x_out):,}")
    return x_out, meta_out, info


def split_final_holdout_temporal(
    df: pd.DataFrame,
    *,
    holdout_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Reserve the latest rows per class/source group for a clean temporal holdout."""
    if holdout_size <= 0:
        info = {"enabled": False, "holdout_size": float(holdout_size), "mode": "off"}
        return df.reset_index(drop=True), df.iloc[0:0].copy().reset_index(drop=True), info
    if not 0 < holdout_size < 1:
        raise ValueError("--final-holdout-size must be 0 or a fraction between 0 and 1.")

    train_parts: list[pd.DataFrame] = []
    holdout_parts: list[pd.DataFrame] = []
    split_details: dict[str, dict[str, int | str]] = {}
    frame = df.copy()
    frame["_row_order_holdout"] = np.arange(len(frame), dtype=np.int64)

    for (label, group_name), group in frame.groupby(["type", SOURCE_GROUP_COL], sort=True):
        group = group.sort_values(["_time_order", "_row_order_holdout"], kind="mergesort")
        holdout_n = int(round(len(group) * holdout_size))
        holdout_n = max(1, holdout_n) if len(group) >= 10 else holdout_n
        if holdout_n <= 0 or holdout_n >= len(group):
            train = group
            holdout = group.iloc[0:0]
        else:
            train = group.iloc[:-holdout_n]
            holdout = group.iloc[-holdout_n:]
        train_parts.append(train)
        holdout_parts.append(holdout)
        key = f"{label}/{group_name}"
        split_details[key] = {
            "label": str(label),
            "split_group": str(group_name),
            "pool_rows": int(len(train)),
            "holdout_rows": int(len(holdout)),
        }

    train_pool = pd.concat(train_parts, ignore_index=True).drop(columns=["_row_order_holdout"], errors="ignore")
    final_holdout = pd.concat(holdout_parts, ignore_index=True).drop(columns=["_row_order_holdout"], errors="ignore")
    info = {
        "enabled": True,
        "holdout_size": float(holdout_size),
        "mode": "temporal",
        "stratify": SOURCE_GROUP_COL,
        "pool_rows": int(len(train_pool)),
        "holdout_rows": int(len(final_holdout)),
        "holdout_counts": counter_to_int_dict(final_holdout["type"]) if not final_holdout.empty else {},
        "group_splits": split_details,
    }
    if SOURCE_LABEL_COL in final_holdout.columns:
        info["holdout_source_counts"] = counter_to_int_dict(final_holdout[SOURCE_LABEL_COL].fillna(final_holdout["type"]))
    return train_pool.reset_index(drop=True), final_holdout.reset_index(drop=True), info


def split_final_holdout_by_source_label(
    df: pd.DataFrame,
    *,
    holdout_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Reserve source_label PCAPs when possible while bounding class holdout size.

    Some generated PCAPs are larger than the requested holdout fraction by
    themselves. A strict whole-PCAP split can silently turn a 10% holdout into
    a much larger and skewed set. This keeps the requested class fraction: it
    uses whole source labels when that stays near target, otherwise it takes the
    latest rows from the class and records that a source label was split.
    """
    if holdout_size <= 0:
        info = {"enabled": False, "holdout_size": float(holdout_size), "mode": "off"}
        return df.reset_index(drop=True), df.iloc[0:0].copy().reset_index(drop=True), info
    if not 0 < holdout_size < 1:
        raise ValueError("--final-holdout-size must be 0 or a fraction between 0 and 1.")
    if SOURCE_LABEL_COL not in df.columns:
        raise RuntimeError("--final-holdout-mode source requires a source_label column.")

    train_parts: list[pd.DataFrame] = []
    holdout_parts: list[pd.DataFrame] = []
    split_details: dict[str, dict[str, object]] = {}
    frame = df.copy()
    frame["_row_order_holdout"] = np.arange(len(frame), dtype=np.int64)
    whole_source_lower = 0.80
    whole_source_upper = 1.20

    for label, group in frame.groupby("type", sort=True):
        group = group.copy()
        target_rows = int(round(len(group) * holdout_size))
        target_rows = max(1, target_rows) if len(group) >= 10 else target_rows
        target_rows = min(target_rows, max(0, len(group) - 1))
        if target_rows <= 0:
            train = group
            holdout = group.iloc[0:0]
            holdout_sources: list[str] = []
            holdout_families: list[str] = []
            bounded_fallback = False
            source_overlap: list[str] = []
            train_parts.append(train)
            holdout_parts.append(holdout)
            split_details[str(label)] = {
                "label": str(label),
                "pool_rows": int(len(train)),
                "holdout_rows": 0,
                "target_holdout_rows": int(target_rows),
                "holdout_sources": holdout_sources,
                "holdout_families": holdout_families,
                "bounded_fallback": bounded_fallback,
                "source_overlap_count": 0,
            }
            continue

        source_summary = (
            group.groupby(SOURCE_LABEL_COL, sort=True)
            .agg(rows=(SOURCE_LABEL_COL, "size"), max_time=("_time_order", "max"), family=(SOURCE_GROUP_COL, "first"))
            .reset_index()
        )
        if len(source_summary) < 2:
            train = group
            holdout = group.iloc[0:0]
            holdout_sources: list[str] = []
            holdout_families: list[str] = []
            bounded_fallback = False
        else:
            min_reasonable_rows = max(1, int(round(target_rows * whole_source_lower)))
            max_reasonable_rows = max(target_rows, int(round(target_rows * whole_source_upper)))
            selected_indices: list[int] = []
            selected_rows = 0

            ranked = source_summary.sort_values(["max_time", "rows"], ascending=[False, True], kind="mergesort")
            for idx, row in ranked.iterrows():
                if len(selected_indices) >= len(source_summary) - 1:
                    break
                rows = int(row["rows"])
                if selected_rows + rows <= max_reasonable_rows:
                    selected_indices.append(int(idx))
                    selected_rows += rows
                if selected_rows >= target_rows:
                    break

            if selected_indices and min_reasonable_rows <= selected_rows <= max_reasonable_rows:
                bounded_fallback = False
                holdout_sources = source_summary.loc[selected_indices, SOURCE_LABEL_COL].astype(str).tolist()
                holdout_mask = group[SOURCE_LABEL_COL].astype(str).isin(holdout_sources)
                holdout = group.loc[holdout_mask]
                train = group.loc[~holdout_mask]
            else:
                bounded_fallback = True
                sort_cols = ["_time_order"]
                if "_row_id" in group.columns:
                    sort_cols.append("_row_id")
                sort_cols.append("_row_order_holdout")
                group_sorted = group.sort_values(sort_cols, kind="mergesort")
                holdout = group_sorted.iloc[-target_rows:]
                train = group.drop(index=holdout.index)
                holdout_sources = holdout[SOURCE_LABEL_COL].astype(str).unique().tolist()

            holdout_families = sorted(holdout[SOURCE_GROUP_COL].astype(str).unique().tolist())

        source_overlap = sorted(
            set(train[SOURCE_LABEL_COL].astype(str)).intersection(set(holdout[SOURCE_LABEL_COL].astype(str)))
        )

        train_parts.append(train)
        holdout_parts.append(holdout)
        split_details[str(label)] = {
            "label": str(label),
            "pool_rows": int(len(train)),
            "holdout_rows": int(len(holdout)),
            "target_holdout_rows": int(target_rows),
            "holdout_sources": holdout_sources,
            "holdout_families": holdout_families,
            "bounded_fallback": bounded_fallback,
            "source_overlap_count": int(len(source_overlap)),
            "source_overlap_sample": source_overlap[:20],
        }

    train_pool = pd.concat(train_parts, ignore_index=True).drop(columns=["_row_order_holdout"], errors="ignore")
    final_holdout = pd.concat(holdout_parts, ignore_index=True).drop(columns=["_row_order_holdout"], errors="ignore")
    overlap = sorted(set(train_pool[SOURCE_LABEL_COL].astype(str)).intersection(set(final_holdout[SOURCE_LABEL_COL].astype(str))))
    bounded_classes = [label for label, details in split_details.items() if details.get("bounded_fallback")]

    info = {
        "enabled": True,
        "holdout_size": float(holdout_size),
        "mode": "source_bounded",
        "stratify": SOURCE_LABEL_COL,
        "pool_rows": int(len(train_pool)),
        "holdout_rows": int(len(final_holdout)),
        "holdout_counts": counter_to_int_dict(final_holdout["type"]) if not final_holdout.empty else {},
        "holdout_source_counts": counter_to_int_dict(final_holdout[SOURCE_LABEL_COL]) if not final_holdout.empty else {},
        "group_splits": split_details,
        "source_overlap_count": int(len(overlap)),
        "source_overlap_sample": overlap[:20],
        "bounded_fallback_classes": bounded_classes,
        "note": "Whole source labels are used when they fit the requested holdout size; oversized classes are split by latest rows.",
    }
    return train_pool.reset_index(drop=True), final_holdout.reset_index(drop=True), info


def se_block_1d(x, reduction: int = 16):
    channels = int(x.shape[-1])
    squeeze = GlobalAveragePooling1D()(x)
    squeeze = Reshape((1, channels))(squeeze)
    hidden = max(channels // reduction, 4)
    excite = Dense(hidden, activation="relu", kernel_initializer="he_normal", use_bias=False)(squeeze)
    excite = Dense(channels, activation="sigmoid", kernel_initializer="he_normal", use_bias=False)(excite)
    return Multiply()([x, excite])


def sedwnet_block(x, filters: int, stride: int = 1, se_reduction: int = 16, dropout: float = 0.0):
    residual = x
    if stride != 1 or int(x.shape[-1]) != filters:
        residual = Conv1D(filters, 1, strides=stride, padding="same", kernel_initializer="he_normal")(residual)
        residual = BatchNormalization()(residual)

    x = SeparableConv1D(filters, 3, strides=stride, padding="same", depthwise_initializer="he_normal", pointwise_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = SeparableConv1D(filters, 3, strides=1, padding="same", depthwise_initializer="he_normal", pointwise_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = se_block_1d(x, reduction=se_reduction)
    if dropout > 0:
        x = Dropout(dropout)(x)
    x = Add()([x, residual])
    x = Activation("relu")(x)
    return x


def build_se_dwnet(input_dim: int, num_classes: int, *, dropout: float = MODEL_DROPOUT) -> Model:
    inputs = Input(shape=(input_dim,), name="zeek_flow_input")
    x = Reshape((input_dim, 1), name="feat_as_1d")(inputs)
    x = Conv1D(64, 3, padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = sedwnet_block(x, filters=64, stride=1)
    x = sedwnet_block(x, filters=128, stride=2)
    x = sedwnet_block(x, filters=256, stride=2)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout)(x)
    outputs = Dense(num_classes, activation="softmax", dtype="float32")(x)
    return Model(inputs, outputs, name="SE_DWNet_Zeek_Crossval")


def make_mi_scorer(discrete_mask: np.ndarray, seed: int):
    kwargs = {
        "discrete_features": discrete_mask,
        "n_neighbors": 3,
        "random_state": seed,
    }
    if "n_jobs" in inspect.signature(mutual_info_classif).parameters:
        kwargs["n_jobs"] = -1
    return partial(mutual_info_classif, **kwargs)


def build_parser(project_root_path: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train SE-DWNet on the balanced Zeek-flow Edge cross-validation dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", default=default_csv(project_root_path), help="Training CSV.")
    parser.add_argument("--label-col", default=None, help="Target label column. Auto-detected when omitted.")
    parser.add_argument("--output-dir", default=default_output_dir(project_root_path), help="Artifact directory.")
    parser.add_argument("--target-k", type=int, default=TARGET_K, help="Number of selected features.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument(
        "--split",
        choices=("source", "random", "temporal"),
        default="random",
        help="Default is source-family-stratified random for full-dataset training; temporal/source are stricter tests.",
    )
    parser.add_argument("--source-group-mode", choices=("family", "label"), default="family")
    parser.add_argument("--time-col", default="ts", help="Timestamp column for temporal split.")
    parser.add_argument("--temporal-fallback", choices=("error", "random"), default="random")
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument(
        "--final-holdout-size",
        type=float,
        default=0.05,
        help="Untouched post-training holdout fraction reserved before train/val/test splitting. Use 0 to disable.",
    )
    parser.add_argument(
        "--final-holdout-mode",
        choices=("source", "temporal", "random"),
        default="random",
        help="Source is strictest; temporal is useful for chronological drift; random is optimistic.",
    )
    parser.add_argument("--smote", choices=("auto", "on", "off"), default="auto")
    parser.add_argument(
        "--loss",
        choices=("ce", "focal"),
        default="ce",
        help="Cross-entropy is the default for balanced full-dataset training; focal is kept for hard/imbalanced runs.",
    )
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=MODEL_DROPOUT)
    parser.add_argument(
        "--smote-imbalance-ratio",
        type=float,
        default=1.10,
        help="Auto-SMOTE threshold: max train class count / min train class count.",
    )
    parser.add_argument("--no-smote", action="store_true", help="Deprecated alias for --smote off.")
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Enable exact row deduplication before splitting. Disabled by default for Zeek flow data.",
    )
    parser.add_argument("--no-dedupe", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--expected-per-class", type=int, default=60_000, help="Warn when a class count differs from this value. Use 0 to disable.")
    parser.add_argument("--max-categorical-cardinality", type=int, default=1024)
    parser.add_argument("--numeric-valid-ratio", type=float, default=0.98)
    parser.add_argument("--feature-inference-sample-size", type=int, default=120_000)
    parser.add_argument("--save-savedmodel", action="store_true", help="Also export TensorFlow SavedModel directory.")
    return parser


def main() -> None:
    project_root_path = project_root()
    args = build_parser(project_root_path).parse_args()
    artifact_dir = os.path.abspath(args.output_dir)
    os.makedirs(artifact_dir, exist_ok=True)

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    print("=== SE-DWNet Zeek Crossval Training ===")
    print(f"GPUs available: {len(tf.config.list_physical_devices('GPU'))}")
    print(f"Project root: {project_root_path}")
    print(f"CSV:          {args.csv}")
    print(f"Artifacts:    {artifact_dir}")
    print(f"Split mode:   {args.split}")
    print(f"Source mode:  {args.source_group_mode}")
    print(f"SMOTE mode:   {'off' if args.no_smote else args.smote}")
    print(f"Dedupe:       {'on' if args.dedupe and not args.no_dedupe else 'off'}")

    if not os.path.exists(args.csv):
        raise FileNotFoundError(args.csv)

    print("Loading CSV into dataframe...", flush=True)
    df = pd.read_csv(args.csv, low_memory=False, on_bad_lines="skip")
    print(f"Loaded dataframe shape: {df.shape}", flush=True)
    df.columns = df.columns.str.strip()
    target_col = label_column(df, args.label_col)
    if target_col != "type":
        df["type"] = df[target_col]

    labels_norm = df["type"].astype(str).str.strip().str.lower().map(canon_label)
    keep = labels_norm.isin(TARGET_CLASSES)
    dropped = int((~keep).sum())
    if dropped:
        print(f"Dropping {dropped:,} rows with unsupported labels.")
    df = df.loc[keep].copy()
    df["type"] = labels_norm.loc[keep].to_numpy()
    if df.empty:
        raise RuntimeError("No rows left after filtering to the six-class taxonomy.")

    df = df.reset_index(drop=True)
    if args.split == "temporal":
        df["_time_order"] = derive_time_order(df, explicit_col=args.time_col)
    elif "ts" in df.columns:
        df["_time_order"] = pd.to_numeric(df["ts"], errors="coerce")
    else:
        df["_time_order"] = np.arange(len(df), dtype=np.float64)
    df[SOURCE_GROUP_COL] = build_source_groups(df, args.source_group_mode)

    class_counts_after_cleanup = counter_to_int_dict(df["type"])
    print(f"Class counts after cleanup: {class_counts_after_cleanup}")
    if args.expected_per_class > 0:
        expected_mismatch = {cls: count for cls, count in class_counts_after_cleanup.items() if count != args.expected_per_class}
        if expected_mismatch:
            print(f"WARNING: class counts differ from expected {args.expected_per_class:,}: {expected_mismatch}")

    source_counts = counter_to_int_dict(df[SOURCE_GROUP_COL])
    print(f"Source group counts: {source_counts}")

    meta_cols = ["type", "_time_order", SOURCE_GROUP_COL]
    if SOURCE_LABEL_COL in df.columns:
        meta_cols.append(SOURCE_LABEL_COL)
    split_meta_all = df[meta_cols].copy().reset_index(drop=True)

    drop_metadata = sorted((ZEEK_METADATA_COLUMNS | set(LABEL_CANDIDATES) | {"type", "_time_order", SOURCE_GROUP_COL}).intersection(df.columns))
    x_all = df.drop(columns=drop_metadata, errors="ignore").reset_index(drop=True)
    print(f"Dropped metadata columns: {drop_metadata}")
    print(f"Raw feature matrix shape: {x_all.shape}", flush=True)

    y_final_holdout = pd.Series(dtype=str)
    x_final_holdout = pd.DataFrame(columns=x_all.columns)

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
            x_all[col] = pd.to_numeric(x_all[col], errors="coerce").astype("float32")
        if col in x_final_holdout.columns:
            if is_numeric_dtype(x_final_holdout[col]):
                x_final_holdout[col] = x_final_holdout[col].astype("float32", copy=False)
            else:
                x_final_holdout[col] = pd.to_numeric(x_final_holdout[col], errors="coerce").astype("float32")

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

    if not x_all.columns.tolist():
        raise RuntimeError("No usable feature columns remain after preprocessing.")

    x_all = optimize_dtypes(x_all)
    x_final_holdout = optimize_dtypes(x_final_holdout)
    print(f"Cleaned data shape: {x_all.shape}")
    print(f"Numeric columns:     {len(num_cols)}")
    print(f"Categorical columns: {len(valid_cat_cols)}")
    print(f"Dropped features:    {len(dropped_feature_cols) + len(constant_cols)}")

    dedupe_info = {"before": int(len(x_all)), "after": int(len(x_all)), "dropped": 0}
    if args.dedupe and not args.no_dedupe:
        x_all, split_meta_all, dedupe_info = dedupe_rows_with_meta(
            x_all,
            split_meta_all,
        )
    else:
        print("Skipping exact row deduplication; repeated Zeek flows are traffic-volume signal.")
        x_all = x_all.reset_index(drop=True)
        split_meta_all = split_meta_all.reset_index(drop=True)

    split_meta_all["_row_id"] = np.arange(len(split_meta_all), dtype=np.int64)
    print("Reserving final holdout after feature cleanup/dedupe and before SMOTE...")
    if args.final_holdout_mode == "source":
        pool_meta, final_holdout_meta, final_holdout_info = split_final_holdout_by_source_label(
            split_meta_all,
            holdout_size=args.final_holdout_size,
            seed=args.seed,
        )
    elif args.final_holdout_mode == "temporal":
        pool_meta, final_holdout_meta, final_holdout_info = split_final_holdout_temporal(
            split_meta_all,
            holdout_size=args.final_holdout_size,
        )
    else:
        pool_meta, final_holdout_meta, final_holdout_info = split_final_holdout(
            split_meta_all,
            holdout_size=args.final_holdout_size,
            seed=args.seed,
            source_stratified=True,
        )
        final_holdout_info["mode"] = "random"

    pool_ids = pool_meta["_row_id"].astype(int).to_numpy()
    holdout_ids = final_holdout_meta["_row_id"].astype(int).to_numpy()
    x_final_holdout = x_all.iloc[holdout_ids].reset_index(drop=True) if len(holdout_ids) else pd.DataFrame(columns=x_all.columns)
    x_all = x_all.iloc[pool_ids].reset_index(drop=True)
    y_final_holdout = final_holdout_meta["type"].astype(str).reset_index(drop=True) if len(holdout_ids) else pd.Series(dtype=str)
    y_all = pool_meta["type"].astype(str).reset_index(drop=True)
    time_order_all = pd.to_numeric(pool_meta["_time_order"], errors="coerce").reset_index(drop=True)
    split_group_all = pool_meta[SOURCE_GROUP_COL].fillna(pool_meta["type"]).astype(str).reset_index(drop=True)

    print(f"Training pool rows after final holdout reserve: {len(x_all):,}")
    if len(holdout_ids):
        print(f"Final holdout rows: {len(x_final_holdout):,}")
        print(f"Final holdout counts: {counter_to_int_dict(y_final_holdout)}")
    else:
        print("Final holdout disabled or empty.")

    if not (0 < args.val_size < 1 and 0 < args.test_size < 1 and args.val_size + args.test_size < 1):
        raise ValueError("--val-size and --test-size must be positive and sum to less than 1.0")
    train_ratio = 1.0 - args.val_size - args.test_size
    print(f"Splitting data ({args.split} {train_ratio:.2f}/{args.val_size:.2f}/{args.test_size:.2f})...")

    if args.split == "source":
        x_train_df, x_val_df, x_test_df, y_train_str, y_val_str, y_test_str, split_info = split_frames_by_source_group(
            x_all,
            y_all,
            split_group_all,
            train_ratio=train_ratio,
            val_ratio=args.val_size,
            test_ratio=args.test_size,
            seed=args.seed,
        )
    else:
        stratify_group = split_group_all if args.split == "random" else split_group_all
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
            split_group=stratify_group,
        )

    split_group_counts = counter_to_int_dict(split_group_all)
    del df, split_meta_all, pool_meta, final_holdout_meta, x_all, y_all, time_order_all, split_group_all
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
    if num_cols:
        x_train_df[num_cols] = scaler_num.fit_transform(x_train_df[num_cols].values)
        x_val_df[num_cols] = scaler_num.transform(x_val_df[num_cols].values)
        x_test_df[num_cols] = scaler_num.transform(x_test_df[num_cols].values)
        if not x_final_holdout_df.empty:
            x_final_holdout_df[num_cols] = scaler_num.transform(x_final_holdout_df[num_cols].values)

    print(f"Selecting top {args.target_k} features (mutual information)...")
    feature_names = x_train_df.columns.tolist()
    discrete_mask = np.array([col in valid_cat_cols for col in feature_names], dtype=bool)
    mi_scorer = make_mi_scorer(discrete_mask, args.seed)
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

    print("Checking class balance and applying SMOTE on the training split if needed...")
    train_counts = Counter(y_train)
    readable_train_counts = {class_names[int(cls)]: int(count) for cls, count in sorted(train_counts.items())}
    print(f"Pre-SMOTE:  {readable_train_counts}")
    max_class_count = max(train_counts.values())
    min_class_count = min(train_counts.values())
    imbalance_ratio = float(max_class_count) / max(float(min_class_count), 1.0)
    smote_mode = "off" if args.no_smote else args.smote
    use_smote = smote_mode == "on" or (smote_mode == "auto" and imbalance_ratio >= args.smote_imbalance_ratio)
    print(f"SMOTE mode: {smote_mode} (imbalance ratio={imbalance_ratio:.3f}, auto threshold={args.smote_imbalance_ratio:.3f})")

    smote_strategy = {}
    if use_smote:
        for cls, count in train_counts.items():
            if count < 2:
                continue
            target = min(int(count * SMOTE_MAX_MULTIPLIER), max_class_count)
            if target > count:
                smote_strategy[cls] = target

    if smote_strategy:
        if SMOTE is None:
            raise RuntimeError("SMOTE was requested, but imbalanced-learn is not installed. Rerun with --smote off.")
        min_count = min(train_counts[cls] for cls in smote_strategy)
        k_neighbors = max(1, min(5, min_count - 1))
        smote = SMOTE(sampling_strategy=smote_strategy, random_state=args.seed, k_neighbors=k_neighbors)
        x_train_bal, y_train_bal = smote.fit_resample(x_train_sel, y_train)
        readable_bal_counts = {class_names[int(cls)]: int(count) for cls, count in sorted(Counter(y_train_bal).items())}
        print(f"Post-SMOTE: {readable_bal_counts}")
    else:
        x_train_bal, y_train_bal = x_train_sel, y_train
        reason = "disabled" if smote_mode == "off" else "class balance is already close enough" if not use_smote else "no eligible classes"
        print(f"SMOTE skipped ({reason})")

    y_train_onehot = to_categorical(y_train_bal, num_classes=num_classes).astype(np.float32)
    y_val_onehot = to_categorical(y_val, num_classes=num_classes).astype(np.float32)

    model = build_se_dwnet(x_train_bal.shape[1], num_classes, dropout=args.dropout)
    optimizer = Adam(learning_rate=args.learning_rate, clipnorm=1.0)

    counts = np.bincount(y_train_bal, minlength=num_classes).astype(np.float32)
    inv = 1.0 / np.maximum(counts, 1.0)
    alpha_vec = inv / inv.mean()
    alpha_vec = np.clip(alpha_vec, *FOCAL_ALPHA_CLIP).astype(np.float32)
    if args.loss == "focal":
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
        print(f"Using categorical cross-entropy (label_smoothing={args.label_smoothing:.4f})")
        loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing)
        loss_info = {
            "name": "CategoricalCrossentropy",
            "label_smoothing": float(args.label_smoothing),
        }

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
    model_path = os.path.join(artifact_dir, "se_dwnet_zeek_model.keras")
    model.save(model_path)
    compat_model_path = os.path.join(artifact_dir, "resnet_model.keras")
    shutil.copyfile(model_path, compat_model_path)
    se_dwnet_alias_path = os.path.join(artifact_dir, "se_dwnet_model.keras")
    shutil.copyfile(model_path, se_dwnet_alias_path)
    saved_model_dir = None
    if args.save_savedmodel:
        saved_model_dir = os.path.join(artifact_dir, "saved_model")
        if hasattr(model, "export"):
            model.export(saved_model_dir)
        else:
            model.save(saved_model_dir, save_format="tf")

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
        "dataset_name": "zeek_edge_crossval_6class",
        "data_csv": os.path.abspath(args.csv),
        "split_mode": args.split,
        "source_group_mode": args.source_group_mode,
        "metadata_columns_dropped": drop_metadata,
    }
    pipeline_path = os.path.join(artifact_dir, "preprocessing_pipeline.pkl")
    with open(pipeline_path, "wb") as f:
        pickle.dump(pipeline_bundle, f)

    feature_roles_path = os.path.join(artifact_dir, "feature_roles.json")
    with open(feature_roles_path, "w", encoding="utf-8") as f:
        json.dump(feature_role_details, f, indent=2)

    metadata = {
        "script": "resnet_zeek_crossval.py",
        "dataset_name": "zeek_edge_crossval_6class",
        "data_csv": os.path.abspath(args.csv),
        "artifact_dir": artifact_dir,
        "model_path": model_path,
        "compat_model_path": compat_model_path,
        "se_dwnet_alias_path": se_dwnet_alias_path,
        "saved_model_dir": saved_model_dir,
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
        "dropout": args.dropout,
        "split_mode": args.split,
        "source_group_mode": args.source_group_mode,
        "split_info": split_info,
        "split_group_counts": split_group_counts,
        "time_col": args.time_col,
        "temporal_fallback": args.temporal_fallback,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "final_holdout": final_holdout_info,
        "final_holdout_mode": args.final_holdout_mode,
        "final_holdout_stage": "after_feature_cleanup_dedupe_before_smote",
        "dedupe": dedupe_info,
        "dedupe_enabled": bool(args.dedupe and not args.no_dedupe),
        "smote_mode": smote_mode,
        "smote_enabled": bool(use_smote and smote_strategy),
        "smote_imbalance_ratio": imbalance_ratio,
        "smote_auto_threshold": args.smote_imbalance_ratio,
        "loss": loss_info,
        "loss_mode": args.loss,
        "label_smoothing": args.label_smoothing,
        "class_counts_after_cleanup": class_counts_after_cleanup,
        "train_counts": counter_to_int_dict(y_train_str),
        "train_counts_after_smote": {
            class_names[int(cls)]: int(count) for cls, count in sorted(Counter(y_train_bal).items())
        },
        "val_counts": counter_to_int_dict(y_val_str),
        "test_counts": counter_to_int_dict(y_test_str),
        "numeric_columns": num_cols,
        "categorical_columns": valid_cat_cols,
        "dropped_feature_columns": dropped_feature_cols + constant_cols,
        "metadata_columns_dropped": drop_metadata,
        "expected_per_class": args.expected_per_class,
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
        f.write("=== SE-DWNet Zeek Crossval Evaluation ===\n")
        f.write(f"CSV: {os.path.abspath(args.csv)}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Split: {args.split} ({args.source_group_mode})\n")
        f.write(f"Loss: {loss_info}\n\n")
        f.write(report_str)

    cm = confusion_matrix(y_test_readable, y_pred_readable, labels=le_target.classes_)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=le_target.classes_, yticklabels=le_target.classes_, cmap="Blues")
    plt.title("SE-DWNet Zeek Crossval Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(artifact_dir, "se_dwnet_zeek_crossval_confusion_matrix.png"), dpi=200)
    plt.close()

    pred_data = {
        "true_class": y_test_readable,
        "predicted_class": y_pred_readable,
        "confidence": np.max(test_probs, axis=1),
        "correct": y_test_readable == y_pred_readable,
    }
    for index, cls in enumerate(class_names):
        pred_data[f"prob_{cls}"] = test_probs[:, index]
    pd.DataFrame(pred_data).to_csv(os.path.join(artifact_dir, "test_predictions.csv"), index=False)

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
            f.write("=== SE-DWNet Zeek Crossval Final Holdout Evaluation ===\n")
            f.write(f"CSV: {os.path.abspath(args.csv)}\n")
            f.write(f"Model: {model_path}\n")
            f.write(f"Loss: {loss_info}\n")
            f.write(f"Holdout: {final_holdout_info}\n\n")
            f.write(holdout_report)

        holdout_cm = confusion_matrix(holdout_true, holdout_pred_readable, labels=le_target.classes_)
        plt.figure(figsize=(12, 10))
        sns.heatmap(holdout_cm, annot=True, fmt="d", xticklabels=le_target.classes_, yticklabels=le_target.classes_, cmap="Blues")
        plt.title("SE-DWNet Zeek Crossval Final Holdout Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(os.path.join(artifact_dir, "se_dwnet_zeek_crossval_final_holdout_confusion_matrix.png"), dpi=200)
        plt.close()

        holdout_pred_data = {
            "true_class": holdout_true,
            "predicted_class": holdout_pred_readable,
            "confidence": np.max(holdout_probs, axis=1),
            "correct": holdout_true == holdout_pred_readable,
        }
        for index, cls in enumerate(class_names):
            holdout_pred_data[f"prob_{cls}"] = holdout_probs[:, index]
        pd.DataFrame(holdout_pred_data).to_csv(os.path.join(artifact_dir, "final_holdout_predictions.csv"), index=False)

    print(f"DONE. Artifacts saved to: {artifact_dir}")


if __name__ == "__main__":
    main()
