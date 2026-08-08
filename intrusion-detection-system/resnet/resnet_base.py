"""
SE-DWNet base model training for TON-IoT network intrusion detection.

Architecture: SE-DWNet (Squeeze-Excitation + Depthwise-Separable Conv1D + Residual)
Dataset:      TON-IoT Network dataset (capped, ~5.2M rows)
Classes:      7 (backdoor, dos_ddos, injection, normal, password, scanning, xss)

Dataset distribution (after dropping mitm + ransomware, before merge):
  scanning / xss / dos / ddos / password  ~700 000 each
  normal                                  ~686 500
  backdoor                                ~508 100
  injection                               ~452 700
  → dos + ddos half-sampled and merged into dos_ddos (~700 000)
  → SMOTE brings backdoor + injection up to ~700 000

"""
import os
import gc
import json
import pickle
import argparse
import sys
import warnings
from functools import partial
from collections import Counter

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization, Activation, Add, Multiply,
    Reshape, Conv1D, SeparableConv1D, GlobalAveragePooling1D,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

import seaborn as sns
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_PARENT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_PARENT)

from app.preprocessing import SafeLabelEncoder, transform_with_pipeline


# ── Configuration ─────────────────────────────────────────────────────────────
TARGET_K = 25
BATCH_SIZE = 1024               # H200-friendly large batch
MAX_EPOCHS = 100
SEED = 42

USE_FOCAL_LOSS = True
FOCAL_GAMMA = 2.0
FOCAL_ALPHA_CLIP = (0.5, 5.0)  # clip per-class alpha to keep training stable

SMOTE_MAX_MULTIPLIER = 2       # cap synthetic oversampling (dataset already fairly balanced)

DROP_LABELS = {"mitm", "ransomware"}
LABEL_CANDIDATES = ("type", "attack", "category", "label", "Label")
TIME_COLS = ("ts", "timestamp", "datetime", "date", "time")
IP_COLS = ("src_ip", "dst_ip", "srcip", "dstip")
TARGET_CLASSES_7 = {"backdoor", "dos_ddos", "injection", "normal", "password", "scanning", "xss"}
ROUTER_SAMPLES_PER_DOMAIN = 50_000
ROUTER_THRESHOLD = 0.60
ROUTE_FIELDS = ("domain", "_domain", "source_domain", "_source", "source")

LOG_COLS = [
    "duration", "src_bytes", "dst_bytes", "src_pkts", "dst_pkts",
    "http_request_body_len", "http_response_body_len", "missed_bytes",
]

CAT_COLS = [
    "proto", "service", "conn_state", "dns_query", "dns_qclass",
    "dns_qtype", "dns_rcode", "http_user_agent", "ssl_version",
    "ssl_cipher", "http_method", "http_version", "src_port", "dst_port",
]


# ── Path helpers ──────────────────────────────────────────────────────────────


def _detect_project_root() -> str:
    env_root = os.environ.get("TON_IOT_PROJECT_ROOT")
    if env_root and os.path.isdir(env_root):
        return os.path.abspath(env_root)
    cwd = os.path.abspath(os.getcwd())
    if os.path.isdir(os.path.join(cwd, "data")) and os.path.isdir(os.path.join(cwd, "artifacts")):
        return cwd
    return os.path.normpath(os.path.join(SCRIPT_DIR, ".."))


def _pick_existing_path(candidates: list[str]) -> str | None:
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def _canon_label(label: object) -> str:
    value = str(label).strip().lower()
    return "dos_ddos" if value in {"dos", "ddos", "ddos_dos"} else value


def _find_label_column(df: pd.DataFrame, explicit: str | None) -> str:
    if explicit:
        if explicit not in df.columns:
            raise RuntimeError(f"Label column '{explicit}' not found. Columns: {list(df.columns[:40])}")
        return explicit
    for col in LABEL_CANDIDATES:
        if col not in df.columns:
            continue
        values = set(df[col].dropna().astype(str).head(50_000).map(_canon_label).unique())
        if values.intersection(TARGET_CLASSES_7):
            return col
    raise RuntimeError(
        "Could not identify target label column. "
        f"Tried {LABEL_CANDIDATES}. Columns: {list(df.columns[:40])}"
    )


def _infer_dataset_name(csv_path: str | None) -> str:
    if not csv_path:
        return "base"
    lower = csv_path.lower()
    if "custom" in lower or "tpot" in lower:
        return "custom"
    return "base"


def _default_output_dir(root: str, dataset_name: str, csv_explicit: bool) -> str:
    if dataset_name == "base" and not csv_explicit:
        artifact_name = "resnet_base"
    else:
        artifact_name = f"resnet_{dataset_name}"
    return os.path.normpath(os.path.join(root, "artifacts", artifact_name))


def _base_csv_path(root: str) -> str:
    return os.path.join(root, "data", "Network_dataset_capped.csv")


def _same_path(left: str, right: str) -> bool:
    left_abs = os.path.abspath(left)
    right_abs = os.path.abspath(right)
    try:
        return os.path.samefile(left_abs, right_abs)
    except OSError:
        return left_abs == right_abs


def _sample_values(series: pd.Series, n: int = 5) -> list[str]:
    values = series.dropna().astype(str).str.strip()
    values = values[values != ""].head(n).tolist()
    return values


def _numeric_time(series: pd.Series) -> tuple[pd.Series | None, str]:
    cleaned = series.astype(str).str.strip().str.replace(",", "", regex=False)
    values = pd.to_numeric(cleaned, errors="coerce")
    valid = int(values.notna().sum())
    unique = int(values.nunique(dropna=True))
    detail = f"numeric valid={valid:,}/{len(series):,}, unique={unique:,}, sample={_sample_values(series)}"
    if valid >= 3 and unique >= 3:
        return values.astype("float64"), detail
    return None, detail


def _datetime_time(series: pd.Series) -> tuple[pd.Series | None, str]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        values = pd.to_datetime(series, errors="coerce", utc=True)
    valid = int(values.notna().sum())
    unique = int(values.nunique(dropna=True))
    detail = f"datetime valid={valid:,}/{len(series):,}, unique={unique:,}, sample={_sample_values(series)}"
    if valid < 3 or unique < 3:
        return None, detail
    seconds = pd.Series(np.nan, index=series.index, dtype="float64")
    seconds.loc[values.notna()] = values.loc[values.notna()].astype("int64") / 1_000_000_000
    return seconds, detail


def derive_time_order(df: pd.DataFrame, *, explicit_col: str | None) -> pd.Series:
    candidates = []
    if explicit_col:
        candidates.append(explicit_col)
    candidates.extend(["ts", "timestamp", "datetime"])

    for col in candidates:
        if col not in df.columns:
            continue
        numeric, numeric_detail = _numeric_time(df[col])
        if numeric is not None:
            print(f"Temporal split using numeric column '{col}' ({numeric_detail})")
            return numeric
        parsed, datetime_detail = _datetime_time(df[col])
        if parsed is not None:
            print(f"Temporal split using datetime column '{col}' ({datetime_detail})")
            return parsed
        print(f"Column '{col}' is not usable for temporal split ({numeric_detail}; {datetime_detail})")

    if "date" in df.columns and "time" in df.columns:
        combined = df["date"].astype(str).str.strip() + " " + df["time"].astype(str).str.strip()
        parsed, datetime_detail = _datetime_time(combined)
        if parsed is not None:
            print(f"Temporal split using combined 'date time' ({datetime_detail})")
            return parsed
        print(f"Combined 'date time' is not usable for temporal split ({datetime_detail})")

    if "date" in df.columns:
        parsed, datetime_detail = _datetime_time(df["date"])
        if parsed is not None:
            print(f"Temporal split using datetime column 'date' ({datetime_detail})")
            return parsed
        print(f"Column 'date' is not usable for temporal split ({datetime_detail})")

    raise RuntimeError(
        "Temporal split requires a usable timestamp column. "
        "Expected ts/timestamp/datetime/date+time, or pass --time-col."
    )


def _ratio_counts(n: int, train_ratio: float, val_ratio: float) -> tuple[int, int]:
    if n < 3:
        raise RuntimeError("Need at least 3 rows per class to split train/val/test.")
    train_n = max(1, int(round(n * train_ratio)))
    val_n = max(1, int(round(n * val_ratio)))
    if train_n + val_n >= n:
        val_n = max(1, n - train_n - 1)
    if train_n + val_n >= n:
        train_n = max(1, n - val_n - 1)
    return train_n, val_n


def _boundary_without_time_leak(times: np.ndarray, target_cut: int) -> int:
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
    X: pd.DataFrame,
    y: pd.Series,
    time_order: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict]:
    canonical = X.astype(str).apply(lambda col: col.str.strip())
    feature_hash = pd.util.hash_pandas_object(canonical, index=False).astype("uint64")
    meta = pd.DataFrame({
        "_target": y.to_numpy(),
        "_time_order": pd.to_numeric(time_order, errors="coerce").to_numpy(),
        "_row_order": np.arange(len(X), dtype=np.int64),
        "_feature_hash": feature_hash.to_numpy(),
    })

    label_counts = meta.groupby("_feature_hash")["_target"].nunique()
    conflict_hashes = set(label_counts[label_counts > 1].index.tolist())
    if conflict_hashes:
        conflict_mask = meta["_feature_hash"].isin(conflict_hashes)
        print(f"Dropping {int(conflict_mask.sum()):,} duplicate feature rows with conflicting labels.")
        keep = ~conflict_mask
        X = X.loc[keep].copy()
        y = y.loc[keep].copy()
        time_order = time_order.loc[keep].copy()
        meta = meta.loc[keep].copy()

    before = len(X)
    keep_idx = (
        meta.assign(_sort_time=meta["_time_order"].fillna(np.inf))
        .sort_values(["_sort_time", "_row_order"], kind="mergesort")
        .drop_duplicates("_feature_hash", keep="first")
        .index
    )
    X = X.loc[keep_idx].reset_index(drop=True)
    y = y.loc[keep_idx].reset_index(drop=True)
    time_order = time_order.loc[keep_idx].reset_index(drop=True)
    info = {
        "before": int(before),
        "after": int(len(X)),
        "dropped": int(before - len(X)),
    }
    print(f"Deduplicated model-visible rows: {before:,} -> {len(X):,}")
    return X, y, time_order, info


def split_frames(
    X: pd.DataFrame,
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
        X_train_df, X_temp_df, y_train_str, y_temp_str = train_test_split(
            X, y, test_size=temp_size, stratify=y, random_state=seed,
        )
        X_val_df, X_test_df, y_val_str, y_test_str = train_test_split(
            X_temp_df, y_temp_str, test_size=test_ratio_of_temp, stratify=y_temp_str, random_state=seed,
        )
        return (
            X_train_df.reset_index(drop=True),
            X_val_df.reset_index(drop=True),
            X_test_df.reset_index(drop=True),
            y_train_str.reset_index(drop=True),
            y_val_str.reset_index(drop=True),
            y_test_str.reset_index(drop=True),
            split_info,
        )

    frame = X.copy()
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
            train_n, _ = _ratio_counts(len(group), train_ratio, val_ratio)
            train, temp = train_test_split(group, train_size=train_n, shuffle=True, random_state=seed)
            val, test = train_test_split(temp, train_size=val_ratio / (val_ratio + test_ratio), shuffle=True, random_state=seed)
            split_kind = "random_fallback"
        else:
            invalid_time_rows = group.loc[~valid_time]
            ordered = group.loc[valid_time].sort_values(["_time_order", "_row_order"], kind="mergesort").reset_index(drop=True)
            train_n, val_n = _ratio_counts(len(ordered), train_ratio, val_ratio)
            cut1 = _boundary_without_time_leak(ordered["_time_order"].to_numpy(), train_n)
            cut2 = _boundary_without_time_leak(ordered["_time_order"].to_numpy(), train_n + val_n)
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


def split_final_holdout(
    X: pd.DataFrame,
    y: pd.Series,
    time_order: pd.Series,
    *,
    holdout_size: float,
    holdout_mode: str,
    seed: int,
    temporal_fallback: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series, dict]:
    """Reserve an untouched final holdout before train/validation/test splitting."""
    if holdout_size <= 0:
        empty_x = pd.DataFrame(columns=X.columns)
        empty_y = pd.Series(dtype=str)
        info = {"enabled": False, "holdout_size": float(holdout_size), "mode": "off"}
        return (
            X.reset_index(drop=True),
            y.reset_index(drop=True),
            time_order.reset_index(drop=True),
            empty_x,
            empty_y,
            info,
        )
    if not 0 < holdout_size < 1:
        raise ValueError("--final-holdout-size must be 0 or a fraction between 0 and 1.")

    frame = X.copy()
    frame["_target"] = y.astype(str).to_numpy()
    frame["_time_order"] = pd.to_numeric(time_order, errors="coerce").to_numpy()
    frame["_row_order"] = np.arange(len(frame), dtype=np.int64)

    if holdout_mode == "random":
        pool, holdout = train_test_split(
            frame,
            test_size=holdout_size,
            stratify=frame["_target"],
            random_state=seed,
        )
        info = {
            "enabled": True,
            "holdout_size": float(holdout_size),
            "mode": "random",
            "pool_rows": int(len(pool)),
            "holdout_rows": int(len(holdout)),
            "holdout_counts": dict(Counter(holdout["_target"])),
        }
    else:
        pool_parts = []
        holdout_parts = []
        split_details = {}
        for label, group in frame.groupby("_target", sort=True):
            valid_time = group["_time_order"].notna()
            usable_time = group.loc[valid_time, "_time_order"].nunique(dropna=True) >= 3
            holdout_n = int(round(len(group) * holdout_size))
            holdout_n = max(1, holdout_n) if len(group) >= 10 else holdout_n

            if holdout_n <= 0 or holdout_n >= len(group):
                pool = group
                holdout = group.iloc[0:0]
                split_kind = "empty"
            elif not usable_time:
                if temporal_fallback != "random":
                    raise RuntimeError(
                        f"{label}: temporal final holdout requires at least 3 usable unique timestamps. "
                        "Regenerate/fix timestamps, use --final-holdout-mode random, or rerun with --temporal-fallback random."
                    )
                pool, holdout = train_test_split(
                    group,
                    test_size=holdout_n,
                    stratify=None,
                    random_state=seed,
                )
                split_kind = "random_fallback"
            else:
                invalid_time_rows = group.loc[~valid_time]
                ordered = group.loc[valid_time].sort_values(["_time_order", "_row_order"], kind="mergesort")
                if holdout_n >= len(ordered):
                    raise RuntimeError(f"{label}: temporal final holdout would consume the whole usable class.")
                holdout = ordered.tail(holdout_n)
                pool = pd.concat([invalid_time_rows, ordered.iloc[: len(ordered) - holdout_n]], ignore_index=True)
                split_kind = "temporal"

            pool_parts.append(pool)
            holdout_parts.append(holdout)
            split_details[str(label)] = {
                "kind": split_kind,
                "pool_rows": int(len(pool)),
                "holdout_rows": int(len(holdout)),
            }

        pool = pd.concat(pool_parts, ignore_index=True)
        holdout = pd.concat(holdout_parts, ignore_index=True)
        info = {
            "enabled": True,
            "holdout_size": float(holdout_size),
            "mode": "temporal",
            "pool_rows": int(len(pool)),
            "holdout_rows": int(len(holdout)),
            "holdout_counts": dict(Counter(holdout["_target"])),
            "class_splits": split_details,
        }

    drop_cols = ["_target", "_time_order", "_row_order"]
    pool_y = pool["_target"].astype(str).reset_index(drop=True)
    pool_time = pool["_time_order"].reset_index(drop=True)
    holdout_y = holdout["_target"].astype(str).reset_index(drop=True)
    return (
        pool.drop(columns=drop_cols).reset_index(drop=True),
        pool_y,
        pool_time,
        holdout.drop(columns=drop_cols).reset_index(drop=True),
        holdout_y,
        info,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a SE-DWNet specialist model on TON-IoT or a labelled custom flow CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", default=None, help="Training CSV. Defaults to the TON-IoT capped/base dataset.")
    parser.add_argument("--label-col", default=None, help="Target label column. Auto-detected when omitted.")
    parser.add_argument("--dataset-name", default=None, help="Artifact/report name, e.g. base or custom.")
    parser.add_argument("--output-dir", default=None, help="Artifact directory. Defaults to artifacts/resnet_base or artifacts/resnet_<dataset-name>.")
    parser.add_argument("--target-k", type=int, default=TARGET_K, help="Number of selected features.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--split", choices=("temporal", "random"), default="temporal", help="Validation split strategy.")
    parser.add_argument("--time-col", default=None, help="Timestamp column for temporal split. Auto-detected when omitted.")
    parser.add_argument("--temporal-fallback", choices=("error", "random"), default="error", help="What to do when a class cannot be split temporally.")
    parser.add_argument("--test-size", type=float, default=0.20, help="Final held-out test fraction.")
    parser.add_argument("--val-size", type=float, default=0.20, help="Validation fraction.")
    parser.add_argument(
        "--final-holdout-size",
        type=float,
        default=0.10,
        help="Untouched post-training holdout fraction reserved after cleanup/dedupe and before SMOTE. Use 0 to disable.",
    )
    parser.add_argument(
        "--final-holdout-mode",
        choices=("temporal", "random"),
        default="temporal",
        help="How to reserve the untouched final holdout.",
    )
    parser.add_argument("--smote", choices=("auto", "on", "off"), default="auto", help="SMOTE balancing mode.")
    parser.add_argument("--smote-imbalance-ratio", type=float, default=1.25, help="Auto-SMOTE threshold: max train class count / min train class count.")
    parser.add_argument("--no-smote", action="store_true", help="Deprecated alias for --smote off.")
    parser.add_argument("--no-dedupe", action="store_true", help="Disable exact model-visible row deduplication before splitting.")
    parser.add_argument("--no-merge-dos-ddos", action="store_true", help="Keep dos and ddos as separate labels instead of dos_ddos.")
    parser.add_argument("--no-dos-ddos-half-sample", action="store_true", help="Do not half-sample raw dos/ddos rows before merging.")
    return parser


PROJECT_ROOT = _detect_project_root()
ARGS = build_parser().parse_args()
DATASET_NAME = ARGS.dataset_name or _infer_dataset_name(ARGS.csv)
ARTIFACT_DIR = os.path.abspath(ARGS.output_dir or _default_output_dir(PROJECT_ROOT, DATASET_NAME, bool(ARGS.csv)))
os.makedirs(ARTIFACT_DIR, exist_ok=True)

TARGET_K = ARGS.target_k
BATCH_SIZE = ARGS.batch_size
MAX_EPOCHS = ARGS.max_epochs
SEED = ARGS.seed


# ── Helpers ───────────────────────────────────────────────────────────────────
def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == "float64":
            df[col] = df[col].astype("float32")
        if df[col].dtype == "int64":
            df[col] = df[col].astype("int32")
    return df


def _router_records(path: str, samples: int, seed: int) -> list[dict]:
    df = pd.read_csv(path, low_memory=False, dtype=str, on_bad_lines="skip")
    df.columns = df.columns.str.strip()
    target_col = _find_label_column(df, explicit=None)
    labels = df[target_col].map(_canon_label)
    df = df.loc[labels.isin(TARGET_CLASSES_7)].copy()
    if len(df) > samples:
        df = df.sample(n=samples, random_state=seed)
    drop_cols = set(LABEL_CANDIDATES) | set(TIME_COLS) | set(IP_COLS)
    df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore", inplace=True)
    return df.to_dict("records")


def maybe_train_domain_router(
    *,
    dataset_name: str,
    data_csv_path: str,
    artifact_dir: str,
    pipeline: dict,
    final_features: list[str],
    seed: int,
) -> dict | None:
    if dataset_name != "custom":
        return None

    base_csv = _base_csv_path(PROJECT_ROOT)
    if not os.path.exists(base_csv):
        print(f"Router skipped: base dataset not found at {base_csv}")
        return None
    if _same_path(base_csv, data_csv_path):
        print("Router skipped: custom CSV is the base dataset.")
        return None

    print("Training domain router: original vs custom")
    original_records = _router_records(base_csv, ROUTER_SAMPLES_PER_DOMAIN, seed)
    custom_records = _router_records(data_csv_path, ROUTER_SAMPLES_PER_DOMAIN, seed)
    n = min(len(original_records), len(custom_records))
    if n < 100:
        print(f"Router skipped: not enough domain rows (original={len(original_records)}, custom={len(custom_records)})")
        return None

    original_records = original_records[:n]
    custom_records = custom_records[:n]
    x_original = transform_with_pipeline(original_records, pipeline=pipeline, final_features=final_features)
    x_custom = transform_with_pipeline(custom_records, pipeline=pipeline, final_features=final_features)
    x = np.vstack([x_original, x_custom])
    y = np.concatenate([np.zeros(n, dtype=np.int8), np.ones(n, dtype=np.int8)])

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=seed,
    )
    router = LogisticRegression(max_iter=500, class_weight="balanced", n_jobs=-1)
    router.fit(x_train, y_train)
    accuracy = float(router.score(x_test, y_test))

    router_path = os.path.join(artifact_dir, "domain_router.pkl")
    router_meta = {
        "model": router,
        "threshold": ROUTER_THRESHOLD,
        "classes": {0: "original", 1: "custom"},
        "route_fields": ROUTE_FIELDS,
        "router_accuracy": accuracy,
        "samples_per_domain": int(n),
        "feature_space": "custom_pipeline",
        "base_csv": base_csv,
        "custom_csv": data_csv_path,
    }
    with open(router_path, "wb") as f:
        pickle.dump(router_meta, f)
    print(f"Router saved: {router_path} (accuracy={accuracy:.4f}, samples/domain={n:,})")
    return {k: v for k, v in router_meta.items() if k != "model"}


# ── SE-DWNet Architecture ────────────────────────────────────────────────────
def _se_block_1d(x, reduction: int = 16):
    channels = int(x.shape[-1])
    squeeze = GlobalAveragePooling1D()(x)
    squeeze = Reshape((1, channels))(squeeze)
    hidden = max(channels // reduction, 4)
    excite = Dense(hidden, activation="relu", kernel_initializer="he_normal", use_bias=False)(squeeze)
    excite = Dense(channels, activation="sigmoid", kernel_initializer="he_normal", use_bias=False)(excite)
    return Multiply()([x, excite])


def _sedwnet_block(x, filters: int, stride: int = 1, se_reduction: int = 16, dropout: float = 0.0):
    residual = x
    if stride != 1 or int(x.shape[-1]) != filters:
        residual = Conv1D(filters, 1, strides=stride, padding="same", kernel_initializer="he_normal")(residual)
        residual = BatchNormalization()(residual)

    x = SeparableConv1D(filters, 3, strides=stride, padding="same",
                        depthwise_initializer="he_normal", pointwise_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = SeparableConv1D(filters, 3, strides=1, padding="same",
                        depthwise_initializer="he_normal", pointwise_initializer="he_normal")(x)
    x = BatchNormalization()(x)

    x = _se_block_1d(x, reduction=se_reduction)
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

    x = _sedwnet_block(x, filters=64, stride=1)
    x = _sedwnet_block(x, filters=128, stride=2)
    x = _sedwnet_block(x, filters=256, stride=2)

    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation="softmax", dtype="float32")(x)
    return Model(inputs, outputs, name="SE_DWNet_CyberSec")


# ── Seeding ───────────────────────────────────────────────────────────────────
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("=== SE-DWNet Specialist Training ===")
print(f"GPUs available: {len(tf.config.list_physical_devices('GPU'))}")
print(f"Project root: {PROJECT_ROOT}")
print(f"Dataset:      {DATASET_NAME}")
print(f"Artifacts:    {ARTIFACT_DIR}")
print(f"Split mode:   {ARGS.split}")
print(f"SMOTE mode:   {'off' if ARGS.no_smote else ARGS.smote}")


# ── Step 1: Load Data ────────────────────────────────────────────────────────
csv_candidates = [
    os.path.join(PROJECT_ROOT, "data", "train_test_network.csv"),
    os.path.join(PROJECT_ROOT, "data", "Network_dataset_capped.csv"),
    os.path.join(PROJECT_ROOT, "data", "network_dataset_capped.csv"),
    os.path.join(SCRIPT_DIR, "data", "train_test_network.csv"),
    os.path.join(SCRIPT_DIR, "data", "Network_dataset_capped.csv"),
]
DATA_CSV_PATH = _pick_existing_path(csv_candidates)
if ARGS.csv:
    DATA_CSV_PATH = os.path.abspath(ARGS.csv)

if DATA_CSV_PATH is None:
    raise FileNotFoundError(
        "Could not find training CSV. Tried:\n"
        + "\n".join(f"  - {p}" for p in csv_candidates)
    )
if not os.path.exists(DATA_CSV_PATH):
    raise FileNotFoundError(DATA_CSV_PATH)

print(f"Loading: {DATA_CSV_PATH}")
df = pd.read_csv(DATA_CSV_PATH, low_memory=False, dtype=str, on_bad_lines="skip")
df.columns = df.columns.str.strip()
target_col = _find_label_column(df, ARGS.label_col)
if target_col != "type":
    df["type"] = df[target_col]
if ARGS.split == "temporal" or ARGS.final_holdout_mode == "temporal":
    df["_time_order"] = derive_time_order(df, explicit_col=ARGS.time_col)
else:
    df["_time_order"] = np.arange(len(df), dtype=np.float64)

drop_metadata = [col for col in TIME_COLS if col in df.columns]
drop_metadata += [col for col in LABEL_CANDIDATES if col in df.columns and col != "type"]
df.drop(columns=drop_metadata, errors="ignore", inplace=True)
print(f"Target column: {target_col}")
print(f"Dropped metadata columns: {drop_metadata}")


# ── Step 2: Clean Labels ─────────────────────────────────────────────────────
labels_norm = df["type"].astype(str).str.strip().str.lower()

# Drop rare / problematic classes
dropped = int(labels_norm.isin(DROP_LABELS).sum())
df = df.loc[~labels_norm.isin(DROP_LABELS)].copy()
print(f"Dropped {dropped} rows with type in {sorted(DROP_LABELS)}")

labels_norm = df["type"].astype(str).str.strip().str.lower()
if ARGS.no_merge_dos_ddos:
    df["type"] = labels_norm
    print("Keeping dos and ddos as separate labels.")
else:
    # Merge dos + ddos -> dos_ddos. The original TON-IoT base run half-samples
    # raw dos/ddos because each raw class was capped at ~700k. Custom datasets
    # can opt out with --no-dos-ddos-half-sample.
    dos_df = df.loc[labels_norm == "dos"]
    ddos_df = df.loc[labels_norm == "ddos"]
    other_df = df.loc[~labels_norm.isin({"dos", "ddos"})].copy()
    other_df["type"] = other_df["type"].map(_canon_label)

    if ARGS.no_dos_ddos_half_sample:
        dos_part = dos_df
        ddos_part = ddos_df
    else:
        dos_part = dos_df.sample(n=max(1, len(dos_df) // 2), random_state=SEED) if len(dos_df) > 0 else dos_df
        ddos_part = ddos_df.sample(n=max(1, len(ddos_df) // 2), random_state=SEED) if len(ddos_df) > 0 else ddos_df

    dos_part = dos_part.copy()
    ddos_part = ddos_part.copy()
    dos_part["type"] = "dos_ddos"
    ddos_part["type"] = "dos_ddos"

    df = pd.concat([other_df, dos_part, ddos_part], ignore_index=True)
    print(f"Merged dos({len(dos_part)})+ddos({len(ddos_part)}) -> dos_ddos({len(dos_part)+len(ddos_part)})")

supported_labels = TARGET_CLASSES_7 if not ARGS.no_merge_dos_ddos else (TARGET_CLASSES_7 | {"dos", "ddos"})
unsupported = sorted(set(df["type"].astype(str).str.strip().str.lower()) - supported_labels)
if unsupported:
    count = int(df["type"].astype(str).str.strip().str.lower().isin(unsupported).sum())
    print(f"Dropping {count} rows with unsupported labels: {unsupported}")
    df = df.loc[~df["type"].astype(str).str.strip().str.lower().isin(unsupported)].copy()
if ARGS.no_merge_dos_ddos:
    df["type"] = df["type"].astype(str).str.strip().str.lower()
else:
    df["type"] = df["type"].map(_canon_label)
if df.empty:
    raise RuntimeError("No rows left after label cleanup.")
df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
print(f"Class counts after cleanup: {dict(Counter(df['type']))}")

# Drop IP columns (prevent topology leakage)
ip_cols = [c for c in ["src_ip", "dst_ip", "srcip", "dstip"] if c in df.columns]
df.drop(columns=ip_cols, inplace=True)
print(f"Dropped IP columns: {ip_cols}")


# ── Step 3: Prepare Features ─────────────────────────────────────────────────
y_all = df["type"].astype(str)
time_order_all = pd.to_numeric(df["_time_order"], errors="coerce")
X_all = df.drop(columns=["type", "_time_order"])

valid_cat_cols = [c for c in CAT_COLS if c in X_all.columns]
num_cols = [c for c in X_all.columns if c not in valid_cat_cols]

for col in valid_cat_cols:
    X_all[col] = X_all[col].fillna("missing").replace("-", "missing").astype(str)

for col in num_cols:
    X_all[col] = pd.to_numeric(X_all[col], errors="coerce")

X_all.replace([np.inf, -np.inf], 0, inplace=True)
X_all = X_all.fillna(0)

for col in LOG_COLS:
    if col in X_all.columns:
        X_all[col] = np.log1p(pd.to_numeric(X_all[col], errors="coerce").fillna(0).clip(lower=0))

X_all = optimize_dtypes(X_all)
print(f"Cleaned data shape: {X_all.shape}")

dedupe_info = {"before": int(len(X_all)), "after": int(len(X_all)), "dropped": 0}
if not ARGS.no_dedupe:
    X_all, y_all, time_order_all, dedupe_info = dedupe_model_visible_rows(X_all, y_all.reset_index(drop=True), time_order_all.reset_index(drop=True))
else:
    y_all = y_all.reset_index(drop=True)
    time_order_all = time_order_all.reset_index(drop=True)
    X_all = X_all.reset_index(drop=True)

print("Reserving final untouched holdout...")
X_all, y_all, time_order_all, X_final_holdout_df, y_final_holdout_str, final_holdout_info = split_final_holdout(
    X_all,
    y_all,
    time_order_all,
    holdout_size=ARGS.final_holdout_size,
    holdout_mode=ARGS.final_holdout_mode,
    seed=SEED,
    temporal_fallback=ARGS.temporal_fallback,
)
print(f"Training pool rows after final holdout reserve: {len(X_all):,}")
if not X_final_holdout_df.empty:
    print(f"Final holdout rows: {len(X_final_holdout_df):,}")
    print(f"Final holdout counts: {dict(Counter(y_final_holdout_str))}")
else:
    print("Final holdout disabled or empty.")


# ── Step 4: Stratified Split ─────────────────────────────────────────────────
# FIX: Target encoder is fit on TRAIN only (was previously fit on all data).
if not (0 < ARGS.val_size < 1 and 0 < ARGS.test_size < 1 and ARGS.val_size + ARGS.test_size < 1):
    raise ValueError("--val-size and --test-size must be positive and sum to less than 1.0")
train_ratio = 1.0 - ARGS.val_size - ARGS.test_size
print(f"Splitting data ({ARGS.split} {train_ratio:.2f}/{ARGS.val_size:.2f}/{ARGS.test_size:.2f})...")

X_train_df, X_val_df, X_test_df, y_train_str, y_val_str, y_test_str, split_info = split_frames(
    X_all,
    y_all,
    time_order_all,
    split_mode=ARGS.split,
    train_ratio=train_ratio,
    val_ratio=ARGS.val_size,
    test_ratio=ARGS.test_size,
    seed=SEED,
    temporal_fallback=ARGS.temporal_fallback,
)

del df, X_all, y_all, time_order_all
gc.collect()

# Encode target on train split only
le_target = LabelEncoder()
le_target.fit(y_train_str)
y_train = le_target.transform(y_train_str)
y_val = le_target.transform(y_val_str)
y_test = le_target.transform(y_test_str)

NUM_CLASSES = len(le_target.classes_)
class_names = le_target.classes_.tolist()
print(f"Classes ({NUM_CLASSES}): {class_names}")


# ── Step 5: Fit Preprocessors on Train ────────────────────────────────────────
X_train_df = X_train_df.reset_index(drop=True)
X_final_holdout_df = X_final_holdout_df.reset_index(drop=True)

# Categorical encoding (train-fit only)
encoders = {}
for col in valid_cat_cols:
    le = SafeLabelEncoder()
    X_train_df[col] = le.fit(X_train_df[col]).transform(X_train_df[col])
    X_val_df[col] = le.transform(X_val_df[col])
    X_test_df[col] = le.transform(X_test_df[col])
    if not X_final_holdout_df.empty:
        X_final_holdout_df[col] = le.transform(X_final_holdout_df[col])
    encoders[col] = le

# Scale numerics (needed for mutual-info feature selection to work well)
scaler_num = MinMaxScaler()
X_train_df[num_cols] = scaler_num.fit_transform(X_train_df[num_cols].values)
X_val_df[num_cols] = scaler_num.transform(X_val_df[num_cols].values)
X_test_df[num_cols] = scaler_num.transform(X_test_df[num_cols].values)
if not X_final_holdout_df.empty:
    X_final_holdout_df[num_cols] = scaler_num.transform(X_final_holdout_df[num_cols].values)

# Feature selection (mutual information, top-K)
print(f"Selecting top {TARGET_K} features (mutual information)...")
feature_names = X_train_df.columns.tolist()
discrete_mask = np.array([c in valid_cat_cols for c in feature_names], dtype=bool)

mi_scorer = partial(
    mutual_info_classif, discrete_features=discrete_mask,
    n_neighbors=3, random_state=SEED, n_jobs=-1,
)

selector = SelectKBest(score_func=mi_scorer, k=min(TARGET_K, X_train_df.shape[1]))
selector.fit(X_train_df, y_train)

X_train_sel = selector.transform(X_train_df).astype(np.float32)
X_val_sel = selector.transform(X_val_df).astype(np.float32)
X_test_sel = selector.transform(X_test_df).astype(np.float32)
X_final_holdout_sel = (
    selector.transform(X_final_holdout_df).astype(np.float32)
    if not X_final_holdout_df.empty
    else np.empty((0, min(TARGET_K, X_train_df.shape[1])), dtype=np.float32)
)

selected_mask = selector.get_support()
final_features = X_train_df.columns[selected_mask].tolist()
print(f"Selected features ({len(final_features)}): {final_features}")

with open(os.path.join(ARTIFACT_DIR, "final_features.txt"), "w") as f:
    f.write("\n".join(final_features) + "\n")

# Re-normalize selected features for model input
final_scaler = MinMaxScaler()
X_train_sel = np.nan_to_num(final_scaler.fit_transform(X_train_sel)).astype(np.float32)
X_val_sel = np.nan_to_num(final_scaler.transform(X_val_sel)).astype(np.float32)
X_test_sel = np.nan_to_num(final_scaler.transform(X_test_sel)).astype(np.float32)
if len(X_final_holdout_sel):
    X_final_holdout_sel = np.nan_to_num(final_scaler.transform(X_final_holdout_sel)).astype(np.float32)

del X_train_df, X_val_df, X_test_df, X_final_holdout_df
gc.collect()


# ── Step 6: SMOTE Oversampling ────────────────────────────────────────────────
# FIX: Previous version had zero class balancing — minority classes (backdoor
# ~508k, injection ~453k) lagged behind the 700k-capped majority classes.
# SMOTE brings them closer to the majority count (capped at SMOTE_MAX_MULTIPLIER×).
print("Applying SMOTE for class balancing...")

train_counts = Counter(y_train)
print(f"Pre-SMOTE:  {dict(train_counts)}")

max_class_count = max(train_counts.values())
min_class_count = min(train_counts.values())
imbalance_ratio = float(max_class_count) / max(float(min_class_count), 1.0)
smote_mode = "off" if ARGS.no_smote else ARGS.smote
use_smote = smote_mode == "on" or (smote_mode == "auto" and imbalance_ratio >= ARGS.smote_imbalance_ratio)
print(f"SMOTE mode: {smote_mode} (imbalance ratio={imbalance_ratio:.3f}, auto threshold={ARGS.smote_imbalance_ratio:.3f})")
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
    smote = SMOTE(sampling_strategy=smote_strategy, random_state=SEED, k_neighbors=k_neighbors)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_sel, y_train)
    print(f"Post-SMOTE: {dict(Counter(y_train_bal))}")
else:
    X_train_bal, y_train_bal = X_train_sel, y_train
    if smote_mode == "off":
        reason = "disabled"
    elif not use_smote:
        reason = "class balance is already close enough"
    else:
        reason = "no eligible classes"
    print(f"SMOTE skipped ({reason})")


# ── Step 7: One-Hot Targets ───────────────────────────────────────────────────
y_train_onehot = to_categorical(y_train_bal, num_classes=NUM_CLASSES).astype(np.float32)
y_val_onehot = to_categorical(y_val, num_classes=NUM_CLASSES).astype(np.float32)


# ── Step 8: Build Model & Train ──────────────────────────────────────────────
model = build_se_dwnet(X_train_bal.shape[1], NUM_CLASSES)
optimizer = Adam(learning_rate=ARGS.learning_rate, clipnorm=1.0)

loss_info = {}

if USE_FOCAL_LOSS:
    counts = np.bincount(y_train_bal, minlength=NUM_CLASSES).astype(np.float32)
    inv = 1.0 / np.maximum(counts, 1.0)
    alpha_vec = inv / inv.mean()
    alpha_vec = np.clip(alpha_vec, *FOCAL_ALPHA_CLIP).astype(np.float32)

    print(f"Focal alpha: min={alpha_vec.min():.4f}, mean={alpha_vec.mean():.4f}, max={alpha_vec.max():.4f}")

    loss_fn = tf.keras.losses.CategoricalFocalCrossentropy(
        alpha=alpha_vec.tolist(), gamma=FOCAL_GAMMA, from_logits=False,
    )
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
    X_train_bal, y_train_onehot,
    validation_data=(X_val_sel, y_val_onehot),
    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1,
)


# ── Step 9: Save Artifacts ───────────────────────────────────────────────────
print("Saving artifacts...")
model_path = os.path.join(ARTIFACT_DIR, "resnet_model.keras")
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
    "seed": SEED,
    "loss": loss_info,
    "dataset_name": DATASET_NAME,
    "data_csv": DATA_CSV_PATH,
    "split_mode": ARGS.split,
}

pipeline_path = os.path.join(ARTIFACT_DIR, "preprocessing_pipeline.pkl")
with open(pipeline_path, "wb") as f:
    pickle.dump(pipeline_bundle, f)

router_metadata = maybe_train_domain_router(
    dataset_name=DATASET_NAME,
    data_csv_path=DATA_CSV_PATH,
    artifact_dir=ARTIFACT_DIR,
    pipeline=pipeline_bundle,
    final_features=final_features,
    seed=SEED,
)

metadata = {
    "script": "resnet_base.py",
    "dataset_name": DATASET_NAME,
    "data_csv": DATA_CSV_PATH,
    "artifact_dir": ARTIFACT_DIR,
    "model_path": model_path,
    "pipeline_path": pipeline_path,
    "classes": class_names,
    "selected_features": final_features,
    "target_k": TARGET_K,
    "batch_size": BATCH_SIZE,
    "max_epochs": MAX_EPOCHS,
    "seed": SEED,
    "learning_rate": ARGS.learning_rate,
    "split_mode": ARGS.split,
    "split_info": split_info,
    "temporal_fallback": ARGS.temporal_fallback,
    "val_size": ARGS.val_size,
    "test_size": ARGS.test_size,
    "final_holdout": final_holdout_info,
    "final_holdout_size": ARGS.final_holdout_size,
    "final_holdout_mode": ARGS.final_holdout_mode,
    "final_holdout_stage": "after_feature_cleanup_dedupe_before_smote",
    "dedupe": dedupe_info,
    "smote_mode": smote_mode,
    "smote_enabled": bool(use_smote and smote_strategy),
    "smote_imbalance_ratio": imbalance_ratio,
    "smote_auto_threshold": ARGS.smote_imbalance_ratio,
    "merge_dos_ddos": not ARGS.no_merge_dos_ddos,
    "dos_ddos_half_sample": not ARGS.no_dos_ddos_half_sample,
    "domain_router": router_metadata,
    "loss": loss_info,
    "class_counts_after_cleanup": dict(Counter(y_train_str) + Counter(y_val_str) + Counter(y_test_str) + Counter(y_final_holdout_str)),
    "train_counts": dict(Counter(y_train_str)),
    "val_counts": dict(Counter(y_val_str)),
    "test_counts": dict(Counter(y_test_str)),
    "final_holdout_counts": dict(Counter(y_final_holdout_str)),
}
with open(os.path.join(ARTIFACT_DIR, "training_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)
with open(os.path.join(ARTIFACT_DIR, "training_metadata.pkl"), "wb") as f:
    pickle.dump(metadata, f)


# ── Step 10: Evaluate on Test ────────────────────────────────────────────────
print("Evaluating on test set...")
test_probs = model.predict(X_test_sel, batch_size=BATCH_SIZE)
test_pred = np.argmax(test_probs, axis=1)

y_test_readable = le_target.inverse_transform(y_test)
y_pred_readable = le_target.inverse_transform(test_pred)

report_str = classification_report(y_test_readable, y_pred_readable, zero_division=0)
print("\nClassification Report (TEST):")
print(report_str)

with open(os.path.join(ARTIFACT_DIR, "classification_report.txt"), "w") as f:
    f.write(f"=== ResNet Specialist Evaluation: {DATASET_NAME} ===\n")
    f.write(f"CSV: {DATA_CSV_PATH}\n")
    f.write(f"Model: {model_path}\n")
    f.write(f"Loss: {loss_info}\n\n")
    f.write(report_str)

cm = confusion_matrix(y_test_readable, y_pred_readable, labels=le_target.classes_)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=le_target.classes_,
            yticklabels=le_target.classes_, cmap="Blues")
plt.title(f"ResNet {DATASET_NAME} Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(ARTIFACT_DIR, f"resnet_{DATASET_NAME}_confusion_matrix.png"), dpi=200)
plt.close()

if len(X_final_holdout_sel):
    print("Evaluating on final untouched holdout...")
    holdout_probs = model.predict(X_final_holdout_sel, batch_size=BATCH_SIZE)
    holdout_pred = np.argmax(holdout_probs, axis=1)
    holdout_true_readable = y_final_holdout_str.to_numpy().astype(str)
    holdout_pred_readable = le_target.inverse_transform(holdout_pred)

    holdout_report = classification_report(holdout_true_readable, holdout_pred_readable, zero_division=0)
    print("\nClassification Report (FINAL HOLDOUT):")
    print(holdout_report)

    with open(os.path.join(ARTIFACT_DIR, "final_holdout_classification_report.txt"), "w") as f:
        f.write(f"=== ResNet Specialist Final Holdout Evaluation: {DATASET_NAME} ===\n")
        f.write(f"CSV: {DATA_CSV_PATH}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Loss: {loss_info}\n")
        f.write(f"Holdout: {final_holdout_info}\n\n")
        f.write(holdout_report)

    holdout_cm = confusion_matrix(holdout_true_readable, holdout_pred_readable, labels=le_target.classes_)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        holdout_cm,
        annot=True,
        fmt="d",
        xticklabels=le_target.classes_,
        yticklabels=le_target.classes_,
        cmap="Blues",
    )
    plt.title(f"ResNet {DATASET_NAME} Final Holdout Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, f"resnet_{DATASET_NAME}_final_holdout_confusion_matrix.png"), dpi=200)
    plt.close()

    holdout_pred_data = {
        "true_class": holdout_true_readable,
        "predicted_class": holdout_pred_readable,
        "confidence": np.max(holdout_probs, axis=1),
        "correct": holdout_true_readable == holdout_pred_readable,
    }
    for index, cls in enumerate(le_target.classes_):
        holdout_pred_data[f"prob_{cls}"] = holdout_probs[:, index]
    pd.DataFrame(holdout_pred_data).to_csv(os.path.join(ARTIFACT_DIR, "final_holdout_predictions.csv"), index=False)

print(f"DONE. Artifacts saved to: {ARTIFACT_DIR}")
