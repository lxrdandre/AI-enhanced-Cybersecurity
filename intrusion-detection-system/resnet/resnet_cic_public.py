"""SE-DWNet training for the public CIC 6-class dataset.

Target taxonomy:
  backdoor, dos_ddos, infiltration, normal, password, scanning

The expected input CSV is the 6-class public CIC build.
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

from app.preprocessing import SafeLabelEncoder  # noqa: E402


TARGET_CLASSES = {"backdoor", "dos_ddos", "infiltration", "normal", "password", "scanning"}
LABEL_CANDIDATES = ("type", "attack", "category", "label", "Label")
TIME_COLS = ("ts", "timestamp", "datetime", "date", "time")
IP_COLS = ("src_ip", "dst_ip", "srcip", "dstip")

TARGET_K = 40
BATCH_SIZE = 1024
MAX_EPOCHS = 100
LEARNING_RATE = 5e-4
SEED = 42
SPLIT_PER_CLASS_CAP = 150_000
SPARE_VALIDATION_PER_CLASS = 1_000
DATASET_NAME = "cic_public_6class"
SPARE_VALIDATION_DATASET_NAME = "cic_public_6class_spare_validation"

USE_FOCAL_LOSS = True
FOCAL_GAMMA = 2.0
FOCAL_ALPHA_CLIP = (0.5, 5.0)
SMOTE_MAX_MULTIPLIER = 2

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

CAT_COLS = [
    "proto",
    "service",
    "conn_state",
    "dns_query",
    "dns_qclass",
    "dns_qtype",
    "dns_rcode",
    "http_user_agent",
    "ssl_version",
    "ssl_cipher",
    "http_method",
    "http_version",
    "src_port",
    "dst_port",
]


def project_root() -> str:
    env_root = os.environ.get("TON_IOT_PROJECT_ROOT")
    if env_root and os.path.isdir(env_root):
        return os.path.abspath(env_root)
    cwd = os.path.abspath(os.getcwd())
    if os.path.isdir(os.path.join(cwd, "data")) and os.path.isdir(os.path.join(cwd, "artifacts")):
        return cwd
    return PROJECT_PARENT


def default_csv(project_root_path: str) -> str:
    return os.path.join(project_root_path, "data", "cic_public_6class.csv")


def default_output_dir(project_root_path: str) -> str:
    return os.path.join(project_root_path, "artifacts", "resnet_cic_public")


def default_spare_validation_csv(project_root_path: str) -> str:
    return os.path.join(project_root_path, "data", f"{SPARE_VALIDATION_DATASET_NAME}.csv")


def canon_label(label: object) -> str:
    value = str(label).strip().lower()
    return "dos_ddos" if value in {"dos", "ddos", "ddos_dos"} else value


def label_column(df: pd.DataFrame, explicit: str | None) -> str:
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
    values = series.dropna().astype(str).str.strip()
    values = values[values != ""].head(n).tolist()
    return values


def numeric_time(series: pd.Series) -> tuple[pd.Series | None, str]:
    cleaned = series.astype(str).str.strip().str.replace(",", "", regex=False)
    values = pd.to_numeric(cleaned, errors="coerce")
    valid = int(values.notna().sum())
    unique = int(values.nunique(dropna=True))
    detail = f"numeric valid={valid:,}/{len(series):,}, unique={unique:,}, sample={sample_values(series)}"
    if valid >= 3 and unique >= 3:
        return values.astype("float64"), detail
    return None, detail


def datetime_time(series: pd.Series) -> tuple[pd.Series | None, str]:
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
    candidates = []
    if explicit_col:
        candidates.append(explicit_col)
    candidates.extend(["ts", "timestamp", "datetime"])

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

    if "date" in df.columns:
        parsed, datetime_detail = datetime_time(df["date"])
        if parsed is not None:
            print(f"Temporal split using datetime column 'date' ({datetime_detail})")
            return parsed
        print(f"Column 'date' is not usable for temporal split ({datetime_detail})")

    raise RuntimeError(
        "Temporal split requires a usable timestamp column. "
        "Expected ts/timestamp/datetime/date+time, or pass --time-col."
    )


def ratio_counts(n: int, train_ratio: float, val_ratio: float) -> tuple[int, int]:
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


def dedupe_with_raw_reference(
    x_raw: pd.DataFrame,
    x_processed: pd.DataFrame,
    y: pd.Series,
    time_order: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict]:
    canonical = x_processed.astype(str).apply(lambda col: col.str.strip())
    feature_hash = pd.util.hash_pandas_object(canonical, index=False).astype("uint64")
    meta = pd.DataFrame(
        {
            "_target": y.to_numpy(),
            "_time_order": pd.to_numeric(time_order, errors="coerce").to_numpy(),
            "_row_order": np.arange(len(x_processed), dtype=np.int64),
            "_feature_hash": feature_hash.to_numpy(),
        }
    )

    label_counts = meta.groupby("_feature_hash")["_target"].nunique()
    conflict_hashes = set(label_counts[label_counts > 1].index.tolist())
    if conflict_hashes:
        conflict_mask = meta["_feature_hash"].isin(conflict_hashes)
        print(f"Dropping {int(conflict_mask.sum()):,} duplicate feature rows with conflicting labels.")
        keep = ~conflict_mask
        x_raw = x_raw.loc[keep].copy()
        x_processed = x_processed.loc[keep].copy()
        y = y.loc[keep].copy()
        time_order = time_order.loc[keep].copy()
        meta = meta.loc[keep].copy()

    before = len(x_processed)
    keep_idx = (
        meta.assign(_sort_time=meta["_time_order"].fillna(np.inf))
        .sort_values(["_sort_time", "_row_order"], kind="mergesort")
        .drop_duplicates("_feature_hash", keep="first")
        .index
    )
    x_raw = x_raw.loc[keep_idx].reset_index(drop=True)
    x_processed = x_processed.loc[keep_idx].reset_index(drop=True)
    y = y.loc[keep_idx].reset_index(drop=True)
    time_order = time_order.loc[keep_idx].reset_index(drop=True)
    info = {"before": int(before), "after": int(len(x_processed)), "dropped": int(before - len(x_processed))}
    print(f"Deduplicated model-visible rows: {before:,} -> {len(x_processed):,}")
    return x_raw, x_processed, y, time_order, info


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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, dict]:
    temp_size = val_ratio + test_ratio
    test_ratio_of_temp = test_ratio / temp_size
    split_info = {"mode": split_mode, "class_splits": {}}

    if split_mode == "random":
        x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=temp_size, stratify=y, random_state=seed)
        x_val, x_test, y_val, y_test = train_test_split(
            x_temp,
            y_temp,
            test_size=test_ratio_of_temp,
            stratify=y_temp,
            random_state=seed,
        )
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
    frame["_row_order"] = np.arange(len(frame), dtype=np.int64)
    train_parts = []
    val_parts = []
    test_parts = []

    for label, group in frame.groupby("_target", sort=True):
        valid_time = group["_time_order"].notna()
        usable_time = group.loc[valid_time, "_time_order"].nunique(dropna=True) >= 3
        if not usable_time:
            if temporal_fallback != "random":
                raise RuntimeError(
                    f"{label}: temporal split requires at least 3 usable unique timestamps. "
                    "Regenerate/fix timestamps or rerun with --temporal-fallback random."
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
        split_info["class_splits"][str(label)] = {
            "kind": split_kind,
            "train": int(len(train)),
            "val": int(len(val)),
            "test": int(len(test)),
        }
        print(f"{split_kind} split {label}: train={len(train):,}, val={len(val):,}, test={len(test):,}")

    train = pd.concat(train_parts, ignore_index=True)
    val = pd.concat(val_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    drop_cols = ["_target", "_time_order", "_row_order"]
    return (
        train.drop(columns=drop_cols).reset_index(drop=True),
        val.drop(columns=drop_cols).reset_index(drop=True),
        test.drop(columns=drop_cols).reset_index(drop=True),
        train["_target"].reset_index(drop=True),
        val["_target"].reset_index(drop=True),
        test["_target"].reset_index(drop=True),
        split_info,
    )


def select_split_pool_and_spare_indices(
    y: pd.Series,
    time_order: pd.Series,
    *,
    split_mode: str,
    split_per_class_cap: int,
    spare_per_class: int,
    seed: int,
) -> tuple[pd.Index, pd.Index, dict]:
    meta = pd.DataFrame(
        {
            "_target": y.astype(str).to_numpy(),
            "_time_order": pd.to_numeric(time_order, errors="coerce").to_numpy(),
            "_row_order": np.arange(len(y), dtype=np.int64),
        }
    )

    split_pool_idx: list[int] = []
    spare_idx: list[int] = []
    selection_info = {"class_selection": {}}

    for offset, label in enumerate(sorted(meta["_target"].astype(str).unique())):
        group = meta.loc[meta["_target"].astype(str) == label].copy()
        valid_time = group["_time_order"].notna()
        usable_time = split_mode == "temporal" and group.loc[valid_time, "_time_order"].nunique(dropna=True) >= 3

        if usable_time:
            ordered = group.sort_values(["_time_order", "_row_order"], kind="mergesort")
            spare = ordered.tail(min(spare_per_class, max(len(ordered) - 3, 0)))
            remaining = ordered.iloc[: len(ordered) - len(spare)]
            if split_per_class_cap > 0 and len(remaining) > split_per_class_cap:
                split_pool = remaining.tail(split_per_class_cap)
                discarded = remaining.iloc[: len(remaining) - len(split_pool)]
            else:
                split_pool = remaining
                discarded = remaining.iloc[0:0]
            selection_mode = "temporal_recent"
        else:
            ordered = group.sample(frac=1.0, random_state=seed + offset)
            spare = ordered.iloc[: min(spare_per_class, max(len(ordered) - 3, 0))]
            remaining = ordered.iloc[len(spare) :]
            if split_per_class_cap > 0 and len(remaining) > split_per_class_cap:
                split_pool = remaining.iloc[:split_per_class_cap]
                discarded = remaining.iloc[split_per_class_cap:]
            else:
                split_pool = remaining
                discarded = remaining.iloc[0:0]
            selection_mode = "random"

        split_pool_idx.extend(split_pool.index.tolist())
        spare_idx.extend(spare.index.tolist())
        selection_info["class_selection"][str(label)] = {
            "mode": selection_mode,
            "available": int(len(group)),
            "split_pool": int(len(split_pool)),
            "spare_validation": int(len(spare)),
            "discarded_after_cap": int(len(discarded)),
        }

    return pd.Index(split_pool_idx), pd.Index(spare_idx), selection_info


def model_row_hashes(x: pd.DataFrame, y: pd.Series) -> pd.Series:
    canonical = x.copy()
    for col in canonical.columns:
        canonical[col] = canonical[col].astype(str).str.strip()
    canonical["_target"] = y.astype(str).str.strip().to_numpy()
    hashes = pd.util.hash_pandas_object(canonical, index=False).astype("uint64")
    return hashes.astype(str)


def build_split_manifest(
    *,
    x_train_fit: pd.DataFrame,
    y_train_fit: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    x_spare: pd.DataFrame,
    y_spare: pd.Series,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    train_fit_manifest = pd.DataFrame(
        {
            "row_hash": model_row_hashes(x_train_fit, y_train_fit).to_numpy(),
            "label": y_train_fit.astype(str).to_numpy(),
            "split": "train_fit",
        }
    )
    rows.append(train_fit_manifest)

    rows.append(
        pd.DataFrame(
            {
                "row_hash": model_row_hashes(x_val, y_val).to_numpy(),
                "label": y_val.astype(str).to_numpy(),
                "split": "val",
            }
        )
    )
    rows.append(
        pd.DataFrame(
            {
                "row_hash": model_row_hashes(x_test, y_test).to_numpy(),
                "label": y_test.astype(str).to_numpy(),
                "split": "test",
            }
        )
    )
    rows.append(
        pd.DataFrame(
            {
                "row_hash": model_row_hashes(x_spare, y_spare).to_numpy(),
                "label": y_spare.astype(str).to_numpy(),
                "split": "holdout_spare",
            }
        )
    )

    manifest = pd.concat(rows, ignore_index=True)
    manifest = manifest.drop_duplicates(subset=["row_hash", "split"], keep="first").reset_index(drop=True)
    return manifest


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == "float64":
            df[col] = df[col].astype("float32")
        if df[col].dtype == "int64":
            df[col] = df[col].astype("int32")
    return df


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
    return Model(inputs, outputs, name="SE_DWNet_CIC_Public")


def build_parser(project_root_path: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train SE-DWNet on the public CIC dataset with the CIC-specific 6-class taxonomy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", default=default_csv(project_root_path), help="Training CSV.")
    parser.add_argument("--label-col", default=None, help="Target label column. Auto-detected when omitted.")
    parser.add_argument("--output-dir", default=default_output_dir(project_root_path), help="Artifact directory.")
    parser.add_argument("--target-k", type=int, default=TARGET_K, help="Number of selected features.")
    parser.add_argument(
        "--split-per-class-cap",
        type=int,
        default=SPLIT_PER_CLASS_CAP,
        help="Cap each class before train/val/test splitting. 0 keeps all rows.",
    )
    parser.add_argument(
        "--spare-validation-per-class",
        type=int,
        default=SPARE_VALIDATION_PER_CLASS,
        help="Reserve this many unseen rows per class for the automatic spare validation dataset.",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--split", choices=("temporal", "random"), default="temporal", help="Validation split strategy.")
    parser.add_argument("--time-col", default="ts", help="Timestamp column for temporal split.")
    parser.add_argument(
        "--temporal-fallback",
        choices=("error", "random"),
        default="random",
        help="What to do when a class cannot be split temporally.",
    )
    parser.add_argument("--test-size", type=float, default=0.20, help="Final held-out test fraction.")
    parser.add_argument("--val-size", type=float, default=0.20, help="Validation fraction.")
    parser.add_argument("--smote", choices=("auto", "on", "off"), default="auto", help="SMOTE balancing mode.")
    parser.add_argument(
        "--smote-imbalance-ratio",
        type=float,
        default=1.25,
        help="Auto-SMOTE threshold: max train class count / min train class count.",
    )
    parser.add_argument("--no-smote", action="store_true", help="Deprecated alias for --smote off.")
    parser.add_argument("--no-dedupe", action="store_true", help="Disable exact row deduplication before splitting.")
    return parser


def main() -> None:
    project_root_path = project_root()
    args = build_parser(project_root_path).parse_args()
    artifact_dir = os.path.abspath(args.output_dir)
    os.makedirs(artifact_dir, exist_ok=True)

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    print("=== SE-DWNet Public CIC Training ===")
    print(f"GPUs available: {len(tf.config.list_physical_devices('GPU'))}")
    print(f"Project root: {project_root_path}")
    print(f"CSV:          {args.csv}")
    print(f"Artifacts:    {artifact_dir}")
    print(f"Split mode:   {args.split}")
    print(f"SMOTE mode:   {'off' if args.no_smote else args.smote}")

    if not os.path.exists(args.csv):
        raise FileNotFoundError(args.csv)

    df = pd.read_csv(args.csv, low_memory=False, dtype=str, on_bad_lines="skip")
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
        raise RuntimeError("No rows left after filtering to the CIC 6-class taxonomy.")
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    class_counts_after_cleanup = dict(Counter(df["type"]))
    print(f"Class counts after cleanup: {class_counts_after_cleanup}")

    ip_cols = [c for c in IP_COLS if c in df.columns]
    df.drop(columns=ip_cols, inplace=True, errors="ignore")
    print(f"Dropped IP columns: {ip_cols}")

    y_all = df["type"].astype(str).reset_index(drop=True)
    time_order_all = pd.to_numeric(df["_time_order"], errors="coerce").reset_index(drop=True)
    x_raw_all = df.drop(columns=["type", "_time_order"]).reset_index(drop=True)
    x_all = x_raw_all.copy()

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
        valid_cat_cols = [c for c in valid_cat_cols if c not in constant_cols]
        num_cols = [c for c in num_cols if c not in constant_cols]
        print(f"Dropped constant columns: {len(constant_cols)}")

    x_all = optimize_dtypes(x_all)
    print(f"Cleaned data shape: {x_all.shape}")

    dedupe_info = {"before": int(len(x_all)), "after": int(len(x_all)), "dropped": 0}
    if not args.no_dedupe:
        x_raw_all, x_all, y_all, time_order_all, dedupe_info = dedupe_with_raw_reference(
            x_raw_all,
            x_all,
            y_all,
            time_order_all,
        )
    else:
        x_raw_all = x_raw_all.reset_index(drop=True)
        x_all = x_all.reset_index(drop=True)
        y_all = y_all.reset_index(drop=True)
        time_order_all = time_order_all.reset_index(drop=True)

    if not (0 < args.val_size < 1 and 0 < args.test_size < 1 and args.val_size + args.test_size < 1):
        raise ValueError("--val-size and --test-size must be positive and sum to less than 1.0")
    train_ratio = 1.0 - args.val_size - args.test_size
    print(f"Splitting data ({args.split} {train_ratio:.2f}/{args.val_size:.2f}/{args.test_size:.2f})...")

    split_pool_idx, spare_idx, selection_info = select_split_pool_and_spare_indices(
        y_all,
        time_order_all,
        split_mode=args.split,
        split_per_class_cap=args.split_per_class_cap,
        spare_per_class=args.spare_validation_per_class,
        seed=args.seed,
    )
    x_split_pool_df = x_all.loc[split_pool_idx].reset_index(drop=True)
    y_split_pool = y_all.loc[split_pool_idx].reset_index(drop=True)
    time_split_pool = time_order_all.loc[split_pool_idx].reset_index(drop=True)
    x_spare_raw_df = x_raw_all.loc[spare_idx].reset_index(drop=True)
    x_spare_df = x_all.loc[spare_idx].reset_index(drop=True)
    y_spare = y_all.loc[spare_idx].reset_index(drop=True)

    split_pool_counts = dict(Counter(y_split_pool))
    spare_counts = dict(Counter(y_spare))
    print(f"Split-pool counts:       {split_pool_counts}")
    print(f"Spare validation counts: {spare_counts}")

    x_train_df, x_val_df, x_test_df, y_train_str, y_val_str, y_test_str, split_info = split_frames(
        x_split_pool_df,
        y_split_pool,
        time_split_pool,
        split_mode=args.split,
        train_ratio=train_ratio,
        val_ratio=args.val_size,
        test_ratio=args.test_size,
        seed=args.seed,
        temporal_fallback=args.temporal_fallback,
    )

    del df, x_raw_all, x_all, y_all, time_order_all, x_split_pool_df, y_split_pool, time_split_pool
    gc.collect()

    split_manifest = build_split_manifest(
        x_train_fit=x_train_df,
        y_train_fit=y_train_str,
        x_val=x_val_df,
        y_val=y_val_str,
        x_test=x_test_df,
        y_test=y_test_str,
        x_spare=x_spare_df,
        y_spare=y_spare,
    )
    split_manifest_path = os.path.join(artifact_dir, "split_membership.csv.gz")
    split_manifest.to_csv(split_manifest_path, index=False, compression="gzip")
    split_manifest_counts = {
        split: int(count)
        for split, count in split_manifest["split"].value_counts().sort_index().items()
    }
    print(f"Split manifest saved: {split_manifest_path}")
    print(f"Split manifest counts: {split_manifest_counts}")

    le_target = LabelEncoder()
    le_target.fit(y_train_str)
    y_train = le_target.transform(y_train_str)
    y_val = le_target.transform(y_val_str)
    y_test = le_target.transform(y_test_str)

    num_classes = len(le_target.classes_)
    class_names = le_target.classes_.tolist()
    print(f"Classes ({num_classes}): {class_names}")

    x_train_df = x_train_df.reset_index(drop=True)
    encoders = {}
    for col in valid_cat_cols:
        encoder = SafeLabelEncoder()
        x_train_df[col] = encoder.fit(x_train_df[col]).transform(x_train_df[col])
        x_val_df[col] = encoder.transform(x_val_df[col])
        x_test_df[col] = encoder.transform(x_test_df[col])
        encoders[col] = encoder

    scaler_num = MinMaxScaler()
    x_train_df[num_cols] = scaler_num.fit_transform(x_train_df[num_cols].values)
    x_val_df[num_cols] = scaler_num.transform(x_val_df[num_cols].values)
    x_test_df[num_cols] = scaler_num.transform(x_test_df[num_cols].values)

    print(f"Selecting top {args.target_k} features (mutual information)...")
    feature_names = x_train_df.columns.tolist()
    discrete_mask = np.array([c in valid_cat_cols for c in feature_names], dtype=bool)
    mi_scorer = partial(mutual_info_classif, discrete_features=discrete_mask, n_neighbors=3, random_state=args.seed, n_jobs=-1)
    selector = SelectKBest(score_func=mi_scorer, k=min(args.target_k, x_train_df.shape[1]))
    selector.fit(x_train_df, y_train)

    x_train_sel = selector.transform(x_train_df).astype(np.float32)
    x_val_sel = selector.transform(x_val_df).astype(np.float32)
    x_test_sel = selector.transform(x_test_df).astype(np.float32)

    selected_mask = selector.get_support()
    final_features = x_train_df.columns[selected_mask].tolist()
    print(f"Selected features ({len(final_features)}): {final_features}")

    with open(os.path.join(artifact_dir, "final_features.txt"), "w") as f:
        f.write("\n".join(final_features) + "\n")

    final_scaler = MinMaxScaler()
    x_train_sel = np.nan_to_num(final_scaler.fit_transform(x_train_sel)).astype(np.float32)
    x_val_sel = np.nan_to_num(final_scaler.transform(x_val_sel)).astype(np.float32)
    x_test_sel = np.nan_to_num(final_scaler.transform(x_test_sel)).astype(np.float32)

    del x_train_df, x_val_df, x_test_df
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
        min_count = min(train_counts[c] for c in smote_strategy)
        k_neighbors = max(1, min(5, min_count - 1))
        smote = SMOTE(sampling_strategy=smote_strategy, random_state=args.seed, k_neighbors=k_neighbors)
        x_train_bal, y_train_bal = smote.fit_resample(x_train_sel, y_train)
        print(f"Post-SMOTE: {dict(Counter(y_train_bal))}")
    else:
        x_train_bal, y_train_bal = x_train_sel, y_train
        reason = "disabled" if smote_mode == "off" else "class balance is already close enough"
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
    model.fit(
        x_train_bal,
        y_train_onehot,
        validation_data=(x_val_sel, y_val_onehot),
        epochs=args.max_epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    print("Saving artifacts...")
    model_path = os.path.join(artifact_dir, "resnet_model.keras")
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
        "dataset_name": DATASET_NAME,
        "data_csv": os.path.abspath(args.csv),
        "split_mode": args.split,
    }

    pipeline_path = os.path.join(artifact_dir, "preprocessing_pipeline.pkl")
    with open(pipeline_path, "wb") as f:
        pickle.dump(pipeline_bundle, f)

    metadata = {
        "script": "resnet_cic_public.py",
        "dataset_name": DATASET_NAME,
        "data_csv": os.path.abspath(args.csv),
        "artifact_dir": artifact_dir,
        "model_path": model_path,
        "pipeline_path": pipeline_path,
        "classes": class_names,
        "selected_features": final_features,
        "target_k": args.target_k,
        "split_per_class_cap": args.split_per_class_cap,
        "spare_validation_per_class": args.spare_validation_per_class,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "split_mode": args.split,
        "split_info": split_info,
        "time_col": args.time_col,
        "temporal_fallback": args.temporal_fallback,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "dedupe": dedupe_info,
        "split_manifest_path": split_manifest_path,
        "split_manifest_counts": split_manifest_counts,
        "selection_info": selection_info,
        "smote_mode": smote_mode,
        "smote_enabled": bool(use_smote and smote_strategy),
        "smote_imbalance_ratio": imbalance_ratio,
        "smote_auto_threshold": args.smote_imbalance_ratio,
        "loss": loss_info,
        "class_counts_after_cleanup": class_counts_after_cleanup,
        "class_counts_after_selection": dict(Counter(y_train_str) + Counter(y_val_str) + Counter(y_test_str) + Counter(y_spare)),
        "split_pool_counts": split_pool_counts,
        "spare_validation_counts": spare_counts,
        "train_counts": dict(Counter(y_train_str)),
        "val_counts": dict(Counter(y_val_str)),
        "test_counts": dict(Counter(y_test_str)),
    }
    with open(os.path.join(artifact_dir, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    with open(os.path.join(artifact_dir, "training_metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    print("Evaluating on test set...")
    test_probs = model.predict(x_test_sel, batch_size=args.batch_size)
    test_pred = np.argmax(test_probs, axis=1)

    y_test_readable = le_target.inverse_transform(y_test)
    y_pred_readable = le_target.inverse_transform(test_pred)

    report_str = classification_report(y_test_readable, y_pred_readable, zero_division=0)
    print("\nClassification Report (TEST):")
    print(report_str)

    with open(os.path.join(artifact_dir, "classification_report.txt"), "w") as f:
        f.write("=== ResNet CIC Public Evaluation ===\n")
        f.write(f"CSV: {os.path.abspath(args.csv)}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Loss: {loss_info}\n\n")
        f.write(report_str)

    cm = confusion_matrix(y_test_readable, y_pred_readable, labels=le_target.classes_)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=le_target.classes_, yticklabels=le_target.classes_, cmap="Blues")
    plt.title("ResNet CIC Public Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(artifact_dir, "resnet_cic_public_confusion_matrix.png"), dpi=200)
    plt.close()

    spare_validation_csv = default_spare_validation_csv(project_root_path)
    spare_validation_df = x_spare_raw_df.copy()
    spare_validation_df["type"] = y_spare.to_numpy()
    os.makedirs(os.path.dirname(spare_validation_csv), exist_ok=True)
    spare_validation_df.to_csv(spare_validation_csv, index=False)

    spare_validation_report = os.path.splitext(spare_validation_csv)[0] + "_report.json"
    with open(spare_validation_report, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_name": SPARE_VALIDATION_DATASET_NAME,
                "output_csv": spare_validation_csv,
                "rows_written": int(len(spare_validation_df)),
                "class_counts": spare_counts,
                "artifact_dir": artifact_dir,
            },
            f,
            indent=2,
        )

    print("\nBuilding spare validation dataset...")
    print(f"Spare validation CSV: {spare_validation_csv}")
    print(f"Spare validation rows: {len(spare_validation_df):,}")

    from validate_cic_holdout import validate as validate_cic_dataset

    secondary_output_dir = os.path.join(artifact_dir, SPARE_VALIDATION_DATASET_NAME)
    validate_cic_dataset(
        csv_path=spare_validation_csv,
        model_dir=artifact_dir,
        output_dir=secondary_output_dir,
        dataset_name=SPARE_VALIDATION_DATASET_NAME,
        label_col="type",
        batch_size=args.batch_size,
        chunk_size=50_000,
        max_samples=None,
    )

    metadata["spare_validation_csv"] = spare_validation_csv
    metadata["spare_validation_report"] = spare_validation_report
    metadata["secondary_validation_output_dir"] = secondary_output_dir
    with open(os.path.join(artifact_dir, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    with open(os.path.join(artifact_dir, "training_metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    print(f"DONE. Artifacts saved to: {artifact_dir}")


if __name__ == "__main__":
    main()
