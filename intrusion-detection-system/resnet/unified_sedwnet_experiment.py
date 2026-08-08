from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from functools import partial
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import classification_report, confusion_matrix, f1_score
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


SEED = 42
TARGET_CLASSES = ["backdoor", "dos_ddos", "injection", "normal", "password", "scanning", "xss"]
DROP_LABELS = {"mitm", "ransomware"}

TARGET_K = 25
BATCH_SIZE = 1024
MAX_EPOCHS = 100
LEARNING_RATE = 5e-4
EARLY_STOPPING_PATIENCE = 12
LR_PLATEAU_PATIENCE = 4

USE_FOCAL_LOSS = True
FOCAL_GAMMA = 2.0
FOCAL_ALPHA_CLIP = (0.5, 5.0)

BASE_TRAIN_PER_CLASS = 50_000
CUSTOM_TRAIN_PER_CLASS = 0  # 0 means "all custom rows available in train split".
CUSTOM_DOS_DDOS_TRAIN_CAP = 10_000
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

BASE_CSV_REL = os.path.join("data", "Network_dataset_capped.csv")
CUSTOM_CSV_REL = os.path.join("data", "custom", "tpot_finetune.csv")

LABEL_CANDIDATES = ("type", "attack", "category", "label", "Label")
IP_COLS = ("src_ip", "dst_ip", "srcip", "dstip")
TIME_COLS = ("ts", "timestamp", "datetime", "date", "time")
DROP_FEATURE_COLS = set(LABEL_CANDIDATES) | set(IP_COLS) | set(TIME_COLS)
INTERNAL_COLS = {"_target", "_domain", "_time_order", "_row_order", "_feature_hash", "_split_kind"}

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


@dataclass(frozen=True)
class ExperimentConfig:
    split_mode: Literal["random", "temporal"]
    project_root: str
    base_csv: str
    custom_csv: str
    output_dir: str
    base_train_per_class: int
    custom_train_per_class: int
    custom_dos_ddos_train_cap: int
    custom_weight: float | None
    target_k: int
    batch_size: int
    max_epochs: int
    learning_rate: float
    seed: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    base_time_col: str | None
    custom_time_col: str | None


def project_root() -> str:
    env_root = os.environ.get("TON_IOT_PROJECT_ROOT")
    if env_root and os.path.isdir(env_root):
        return os.path.abspath(env_root)
    cwd = os.path.abspath(os.getcwd())
    if os.path.isdir(os.path.join(cwd, "data")) and os.path.isdir(os.path.join(cwd, "artifacts")):
        return cwd
    return PROJECT_PARENT


def canon_label(label: object) -> str:
    value = str(label).strip().lower()
    return "dos_ddos" if value in {"dos", "ddos", "ddos_dos"} else value


def label_column(df: pd.DataFrame) -> str:
    for col in LABEL_CANDIDATES:
        if col not in df.columns:
            continue
        values = set(df[col].dropna().astype(str).head(50_000).map(canon_label).unique())
        if values.intersection(TARGET_CLASSES):
            return col
    raise RuntimeError(f"Could not identify target label column. Columns: {list(df.columns[:40])}")


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


def derive_time_order(df: pd.DataFrame, *, explicit_col: str | None, domain: str) -> pd.Series:
    candidates = []
    if explicit_col:
        candidates.append(explicit_col)
    candidates.extend(["ts", "timestamp", "datetime"])

    for col in candidates:
        if col not in df.columns:
            continue
        numeric, numeric_detail = _numeric_time(df[col])
        if numeric is not None:
            print(f"{domain}: temporal split using numeric column '{col}' ({numeric_detail})")
            return numeric
        parsed, datetime_detail = _datetime_time(df[col])
        if parsed is not None:
            print(f"{domain}: temporal split using datetime column '{col}' ({datetime_detail})")
            return parsed
        print(f"{domain}: column '{col}' is not usable for temporal split ({numeric_detail}; {datetime_detail})")

    if "date" in df.columns and "time" in df.columns:
        combined = df["date"].astype(str).str.strip() + " " + df["time"].astype(str).str.strip()
        parsed, datetime_detail = _datetime_time(combined)
        if parsed is not None:
            print(f"{domain}: temporal split using combined 'date time' ({datetime_detail})")
            return parsed
        print(f"{domain}: combined 'date time' is not usable for temporal split ({datetime_detail})")

    if "date" in df.columns:
        parsed, datetime_detail = _datetime_time(df["date"])
        if parsed is not None:
            print(f"{domain}: temporal split using datetime column 'date' ({datetime_detail})")
            return parsed
        print(f"{domain}: column 'date' is not usable for temporal split ({datetime_detail})")

    raise RuntimeError(
        f"{domain}: temporal split requires a usable timestamp column. "
        "Expected ts/timestamp/datetime/date+time, or pass --base-time-col/--custom-time-col."
    )


def read_domain_csv(
    path: str,
    *,
    domain: Literal["base", "custom"],
    require_time: bool,
    time_col: str | None,
) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    print(f"Loading {domain}: {path}")
    df = pd.read_csv(path, dtype=str, low_memory=False, on_bad_lines="skip")
    df.columns = df.columns.str.strip()

    lab = label_column(df)
    labels = df[lab].map(canon_label)
    keep = labels.isin(TARGET_CLASSES)
    keep &= ~labels.isin(DROP_LABELS)
    df = df.loc[keep].copy()
    labels = labels.loc[keep].reset_index(drop=True)
    df = df.reset_index(drop=True)

    if require_time:
        try:
            time_order = derive_time_order(df, explicit_col=time_col, domain=domain)
            split_kind = "temporal"
        except RuntimeError:
            if time_col:
                raise
            print(f"{domain}: no usable timestamp; using random stratified split for this domain.")
            time_order = pd.Series(np.arange(len(df), dtype=np.float64))
            split_kind = "random"
    else:
        time_order = pd.Series(np.arange(len(df), dtype=np.float64))
        split_kind = "random"

    df["_target"] = labels
    df["_domain"] = domain
    df["_time_order"] = pd.to_numeric(time_order, errors="coerce").to_numpy()
    if split_kind == "random":
        df["_time_order"] = np.arange(len(df), dtype=np.float64)
    df["_row_order"] = np.arange(len(df), dtype=np.int64)
    df["_split_kind"] = split_kind

    df.drop(columns=[col for col in DROP_FEATURE_COLS if col in df.columns], inplace=True, errors="ignore")
    print(f"{domain}: rows={len(df):,}, labels={dict(Counter(df['_target']))}")
    return df


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return sorted(col for col in df.columns if col not in INTERNAL_COLS)


def align_and_deduplicate(frames: list[pd.DataFrame]) -> pd.DataFrame:
    feature_cols = sorted({col for frame in frames for col in _feature_columns(frame)})
    aligned = []
    for frame in frames:
        current = frame.copy()
        for col in feature_cols:
            if col not in current.columns:
                current[col] = ""
        aligned.append(current[feature_cols + ["_target", "_domain", "_time_order", "_row_order", "_split_kind"]])

    df = pd.concat(aligned, ignore_index=True)
    canonical = df[feature_cols].astype(str).apply(lambda col: col.str.strip())
    feature_hash = pd.util.hash_pandas_object(canonical, index=False).astype("uint64")
    df["_feature_hash"] = feature_hash.to_numpy()

    label_counts = df.groupby("_feature_hash")["_target"].nunique()
    conflict_hashes = set(label_counts[label_counts > 1].index.tolist())
    if conflict_hashes:
        conflict_rows = int(df["_feature_hash"].isin(conflict_hashes).sum())
        print(f"Dropping {conflict_rows:,} rows with identical features but conflicting labels.")
        df = df.loc[~df["_feature_hash"].isin(conflict_hashes)].copy()

    before = len(df)
    priority = df["_domain"].map({"custom": 0, "base": 1}).fillna(2)
    df = (
        df.assign(_priority=priority)
        .sort_values(["_priority", "_time_order", "_row_order"], kind="mergesort")
        .drop_duplicates("_feature_hash", keep="first")
        .drop(columns=["_priority"])
        .reset_index(drop=True)
    )
    print(f"Deduplicated model-visible rows: {before:,} -> {len(df):,}")
    print(f"After dedupe by domain/class: {dict(Counter(zip(df['_domain'], df['_target'])))}")
    return df


def _ratio_counts(n: int, train_ratio: float, val_ratio: float) -> tuple[int, int]:
    if n < 3:
        raise RuntimeError("Need at least 3 rows per domain/class group to split train/val/test.")
    train_n = max(1, int(round(n * train_ratio)))
    val_n = max(1, int(round(n * val_ratio)))
    if train_n + val_n >= n:
        val_n = max(1, n - train_n - 1)
    if train_n + val_n >= n:
        train_n = max(1, n - val_n - 1)
    return train_n, val_n


def random_split(df: pd.DataFrame, cfg: ExperimentConfig) -> dict[str, pd.DataFrame]:
    train_parts = []
    val_parts = []
    test_parts = []
    for (domain, label), group in df.groupby(["_domain", "_target"], sort=True):
        train_n, _ = _ratio_counts(len(group), cfg.train_ratio, cfg.val_ratio)
        train, temp = train_test_split(group, train_size=train_n, shuffle=True, random_state=cfg.seed)
        val_ratio_of_temp = cfg.val_ratio / (cfg.val_ratio + cfg.test_ratio)
        val, test = train_test_split(temp, train_size=val_ratio_of_temp, shuffle=True, random_state=cfg.seed)
        train_parts.append(train)
        val_parts.append(val)
        test_parts.append(test)
        print(f"random split {domain}/{label}: train={len(train):,}, val={len(val):,}, test={len(test):,}")
    return {
        "train_full": pd.concat(train_parts, ignore_index=True),
        "val": pd.concat(val_parts, ignore_index=True),
        "test": pd.concat(test_parts, ignore_index=True),
    }


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


def temporal_split(df: pd.DataFrame, cfg: ExperimentConfig) -> dict[str, pd.DataFrame]:
    train_parts = []
    val_parts = []
    test_parts = []
    for (domain, label), group in df.groupby(["_domain", "_target"], sort=True):
        valid_time = group["_time_order"].notna()
        usable_time = group.loc[valid_time, "_time_order"].nunique(dropna=True) >= 3
        if str(group["_split_kind"].iloc[0]) != "temporal" or not usable_time:
            train_n, _ = _ratio_counts(len(group), cfg.train_ratio, cfg.val_ratio)
            train, temp = train_test_split(group, train_size=train_n, shuffle=True, random_state=cfg.seed)
            val_ratio_of_temp = cfg.val_ratio / (cfg.val_ratio + cfg.test_ratio)
            val, test = train_test_split(temp, train_size=val_ratio_of_temp, shuffle=True, random_state=cfg.seed)
            train_parts.append(train)
            val_parts.append(val)
            test_parts.append(test)
            reason = "timestamp unavailable" if str(group["_split_kind"].iloc[0]) != "temporal" else "timestamp missing/constant inside class"
            print(f"random fallback split {domain}/{label} ({reason}): train={len(train):,}, val={len(val):,}, test={len(test):,}")
            continue

        invalid_time_rows = group.loc[~valid_time]
        group = group.loc[valid_time].sort_values(["_time_order", "_row_order"], kind="mergesort").reset_index(drop=True)

        n = len(group)
        train_n, val_n = _ratio_counts(n, cfg.train_ratio, cfg.val_ratio)
        cut1 = _boundary_without_time_leak(group["_time_order"].to_numpy(), train_n)
        cut2 = _boundary_without_time_leak(group["_time_order"].to_numpy(), train_n + val_n)
        if not (0 < cut1 < cut2 < n):
            raise RuntimeError(
                f"{domain}/{label}: temporal boundary collapsed. "
                "Use a finer timestamp column or inspect duplicate timestamps."
            )
        train = group.iloc[:cut1]
        val = group.iloc[cut1:cut2]
        test = group.iloc[cut2:]
        if not invalid_time_rows.empty:
            train = pd.concat([invalid_time_rows, train], ignore_index=True)
        train_parts.append(train)
        val_parts.append(val)
        test_parts.append(test)
        suffix = f", invalid_ts_to_train={len(invalid_time_rows):,}" if not invalid_time_rows.empty else ""
        print(f"temporal split {domain}/{label}: train={len(train):,}, val={len(val):,}, test={len(test):,}{suffix}")
    return {
        "train_full": pd.concat(train_parts, ignore_index=True),
        "val": pd.concat(val_parts, ignore_index=True),
        "test": pd.concat(test_parts, ignore_index=True),
    }


def verify_no_leakage(splits: dict[str, pd.DataFrame]) -> None:
    hash_sets = {name: set(frame["_feature_hash"].astype("uint64").tolist()) for name, frame in splits.items()}
    names = list(hash_sets)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap = hash_sets[left].intersection(hash_sets[right])
            if overlap:
                raise RuntimeError(f"Leakage detected: {len(overlap):,} duplicate feature rows in {left} and {right}.")
    print("Leakage check passed: train/val/test feature hashes are disjoint.")


def sample_training_rows(train_full: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    sampled = []
    for label, group in train_full.loc[train_full["_domain"] == "base"].groupby("_target", sort=True):
        n = min(cfg.base_train_per_class, len(group))
        sampled.append(group.sample(n=n, random_state=cfg.seed))
        if n < cfg.base_train_per_class:
            print(f"base/{label}: only {n:,} train rows available; requested {cfg.base_train_per_class:,}.")

    custom_train = train_full.loc[train_full["_domain"] == "custom"]
    if custom_train.empty:
        raise RuntimeError("Custom train split is empty. Refusing to train because the custom dataset would be ignored.")
    for label, group in custom_train.groupby("_target", sort=True):
        cap = cfg.custom_train_per_class if cfg.custom_train_per_class > 0 else len(group)
        if label == "dos_ddos" and cfg.custom_dos_ddos_train_cap > 0:
            cap = min(cap, cfg.custom_dos_ddos_train_cap)
        n = min(cap, len(group))
        sampled.append(group.sample(n=n, random_state=cfg.seed) if n < len(group) else group)
        if n < len(group):
            print(f"custom/{label}: capped train rows at {n:,} from {len(group):,}.")

    train = pd.concat(sampled, ignore_index=True)
    train = train.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
    print(f"Experiment-B train sample rows: {len(train):,}")
    print(f"Train sample by domain/class: {dict(Counter(zip(train['_domain'], train['_target'])))}")
    return train


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == "float64":
            df[col] = df[col].astype("float32")
        if df[col].dtype == "int64":
            df[col] = df[col].astype("int32")
    return df


def prepare_feature_frame(df: pd.DataFrame, feature_cols: list[str], cat_cols: list[str], num_cols: list[str]) -> pd.DataFrame:
    out = df[feature_cols].copy()
    for col in cat_cols:
        out[col] = out[col].fillna("missing").replace("-", "missing").astype(str)
    for col in num_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out.replace([np.inf, -np.inf], 0, inplace=True)
    out = out.fillna(0)
    for col in LOG_COLS:
        if col in out.columns:
            out[col] = np.log1p(pd.to_numeric(out[col], errors="coerce").fillna(0).clip(lower=0))
    return optimize_dtypes(out)


def fit_preprocessing(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, list[str]]:
    feature_cols = _feature_columns(train_df)
    cat_cols = [col for col in CAT_COLS if col in feature_cols]
    num_cols = [col for col in feature_cols if col not in cat_cols]

    print(f"Feature columns before selection: {len(feature_cols)}")
    print(f"Categorical columns: {cat_cols}")
    print(f"Numeric columns: {len(num_cols)}")

    train_x = prepare_feature_frame(train_df, feature_cols, cat_cols, num_cols)
    val_x = prepare_feature_frame(val_df, feature_cols, cat_cols, num_cols)
    test_x = prepare_feature_frame(test_df, feature_cols, cat_cols, num_cols)

    encoders = {}
    for col in cat_cols:
        encoder = SafeLabelEncoder()
        train_x[col] = encoder.fit(train_x[col]).transform(train_x[col])
        val_x[col] = encoder.transform(val_x[col])
        test_x[col] = encoder.transform(test_x[col])
        encoders[col] = encoder

    scaler_num = MinMaxScaler()
    train_x[num_cols] = scaler_num.fit_transform(train_x[num_cols].values)
    val_x[num_cols] = scaler_num.transform(val_x[num_cols].values)
    test_x[num_cols] = scaler_num.transform(test_x[num_cols].values)

    y_train = LabelEncoder().fit(TARGET_CLASSES).transform(train_df["_target"])
    discrete_mask = np.array([col in cat_cols for col in feature_cols], dtype=bool)
    mi_scorer = partial(mutual_info_classif, discrete_features=discrete_mask, n_neighbors=3, random_state=cfg.seed, n_jobs=-1)
    selector = SelectKBest(score_func=mi_scorer, k=min(cfg.target_k, len(feature_cols)))
    selector.fit(train_x, y_train)

    final_features = train_x.columns[selector.get_support()].tolist()
    print(f"Selected features ({len(final_features)}): {final_features}")

    train_sel = selector.transform(train_x).astype(np.float32)
    val_sel = selector.transform(val_x).astype(np.float32)
    test_sel = selector.transform(test_x).astype(np.float32)

    final_scaler = MinMaxScaler()
    train_sel = np.nan_to_num(final_scaler.fit_transform(train_sel)).astype(np.float32)
    val_sel = np.nan_to_num(final_scaler.transform(val_sel)).astype(np.float32)
    test_sel = np.nan_to_num(final_scaler.transform(test_sel)).astype(np.float32)

    target_encoder = LabelEncoder()
    target_encoder.fit(TARGET_CLASSES)
    pipeline = {
        "scaler_num": scaler_num,
        "selector": selector,
        "final_scaler": final_scaler,
        "encoders": encoders,
        "target_encoder": target_encoder,
        "features": final_features,
        "valid_cat_cols": cat_cols,
        "num_cols": num_cols,
        "seed": cfg.seed,
        "unified_experiment": {
            "name": "experiment_b",
            "split_mode": cfg.split_mode,
            "base_train_per_class": cfg.base_train_per_class,
            "custom_train_per_class": cfg.custom_train_per_class,
            "custom_dos_ddos_train_cap": cfg.custom_dos_ddos_train_cap,
        },
    }
    return train_sel, val_sel, test_sel, pipeline, final_features


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

    x = SeparableConv1D(filters, 3, strides=stride, padding="same", depthwise_initializer="he_normal", pointwise_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = SeparableConv1D(filters, 3, strides=1, padding="same", depthwise_initializer="he_normal", pointwise_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = _se_block_1d(x, reduction=se_reduction)
    if dropout > 0:
        x = Dropout(dropout)(x)
    x = Add()([x, residual])
    return Activation("relu")(x)


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
    return Model(inputs, outputs, name="SE_DWNet_Unified_7Class")


def encoded_labels(frame: pd.DataFrame) -> np.ndarray:
    encoder = LabelEncoder()
    encoder.fit(TARGET_CLASSES)
    return encoder.transform(frame["_target"])


def resolve_custom_weight(train_frame: pd.DataFrame, configured_weight: float | None) -> float:
    if configured_weight is not None:
        return float(configured_weight)
    counts = Counter(train_frame["_domain"])
    custom_count = counts.get("custom", 0)
    if custom_count <= 0:
        raise RuntimeError("Custom train rows are missing; refusing to assign custom_weight.")
    return float(counts.get("base", 1)) / float(custom_count)


def domain_sample_weights(frame: pd.DataFrame, custom_weight: float) -> np.ndarray:
    weights = np.ones(len(frame), dtype=np.float32)
    weights[frame["_domain"].to_numpy() == "custom"] = float(custom_weight)
    return weights


def focal_loss_for(y_train: np.ndarray):
    counts = np.bincount(y_train, minlength=len(TARGET_CLASSES)).astype(np.float32)
    inv = 1.0 / np.maximum(counts, 1.0)
    alpha = np.clip(inv / inv.mean(), *FOCAL_ALPHA_CLIP).astype(np.float32)
    print(f"Focal alpha: min={alpha.min():.4f}, mean={alpha.mean():.4f}, max={alpha.max():.4f}")
    return tf.keras.losses.CategoricalFocalCrossentropy(alpha=alpha.tolist(), gamma=FOCAL_GAMMA, from_logits=False)


def write_report(
    *,
    name: str,
    y_true: np.ndarray,
    probs: np.ndarray,
    output_dir: str,
) -> dict[str, float]:
    y_pred = np.argmax(probs, axis=1)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(TARGET_CLASSES))),
        target_names=TARGET_CLASSES,
        zero_division=0,
        digits=4,
    )
    macro_f1 = float(f1_score(y_true, y_pred, labels=list(range(len(TARGET_CLASSES))), average="macro", zero_division=0))
    accuracy = float(np.mean(y_true == y_pred))
    report_path = os.path.join(output_dir, f"classification_report_{name}.txt")
    with open(report_path, "w") as f:
        f.write(f"=== Unified SEDWNet Experiment B: {name} ===\n")
        f.write(f"accuracy={accuracy:.6f}\n")
        f.write(f"macro_f1={macro_f1:.6f}\n\n")
        f.write(report)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(TARGET_CLASSES))))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=TARGET_CLASSES, yticklabels=TARGET_CLASSES, cmap="Blues")
    plt.title(f"Unified SEDWNet {name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_{name}.png"), dpi=200)
    plt.close()
    print(f"{name}: accuracy={accuracy:.4f}, macro_f1={macro_f1:.4f}")
    return {"accuracy": accuracy, "macro_f1": macro_f1}


def evaluate_all(
    *,
    model: Model,
    x: np.ndarray,
    frame: pd.DataFrame,
    split_name: str,
    output_dir: str,
    batch_size: int,
) -> dict[str, dict[str, float]]:
    y = encoded_labels(frame)
    probs = model.predict(x, batch_size=batch_size, verbose=0)
    metrics = {split_name: write_report(name=split_name, y_true=y, probs=probs, output_dir=output_dir)}
    for domain in ("base", "custom"):
        mask = frame["_domain"].to_numpy() == domain
        if not mask.any():
            continue
        metrics[f"{split_name}_{domain}"] = write_report(
            name=f"{split_name}_{domain}",
            y_true=y[mask],
            probs=probs[mask],
            output_dir=output_dir,
        )
    if "test_base" in metrics and "test_custom" in metrics:
        metrics["test_worst_domain_macro_f1"] = {
            "macro_f1": min(metrics["test_base"]["macro_f1"], metrics["test_custom"]["macro_f1"]),
            "accuracy": min(metrics["test_base"]["accuracy"], metrics["test_custom"]["accuracy"]),
        }
    return metrics


def split_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {f"{domain}/{label}": count for (domain, label), count in Counter(zip(frame["_domain"], frame["_target"])).items()}


def run_experiment(cfg: ExperimentConfig) -> None:
    np.random.seed(cfg.seed)
    tf.keras.utils.set_random_seed(cfg.seed)
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    os.makedirs(cfg.output_dir, exist_ok=True)
    print("=== Unified SEDWNet Experiment B ===")
    print(f"Split mode: {cfg.split_mode}")
    print(f"Output:     {cfg.output_dir}")

    base = read_domain_csv(cfg.base_csv, domain="base", require_time=cfg.split_mode == "temporal", time_col=cfg.base_time_col)
    custom = read_domain_csv(cfg.custom_csv, domain="custom", require_time=cfg.split_mode == "temporal", time_col=cfg.custom_time_col)
    all_rows = align_and_deduplicate([base, custom])
    del base, custom
    gc.collect()

    splits = random_split(all_rows, cfg) if cfg.split_mode == "random" else temporal_split(all_rows, cfg)
    verify_no_leakage(splits)
    train = sample_training_rows(splits["train_full"], cfg)

    x_train, x_val, x_test, pipeline, final_features = fit_preprocessing(train, splits["val"], splits["test"], cfg)
    y_train = encoded_labels(train)
    y_val = encoded_labels(splits["val"])
    y_train_oh = to_categorical(y_train, len(TARGET_CLASSES)).astype(np.float32)
    y_val_oh = to_categorical(y_val, len(TARGET_CLASSES)).astype(np.float32)
    custom_weight = resolve_custom_weight(train, cfg.custom_weight)
    train_weights = domain_sample_weights(train, custom_weight)
    val_weights = domain_sample_weights(splits["val"], custom_weight)
    print(f"Train sample weights by domain: base=1.0, custom={float(train_weights[train['_domain'].to_numpy() == 'custom'][0]):.4f}")
    print(f"Val sample weights by domain: base=1.0, custom={float(val_weights[splits['val']['_domain'].to_numpy() == 'custom'][0]):.4f}")

    loss_fn = focal_loss_for(y_train) if USE_FOCAL_LOSS else tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0.05)
    model = build_se_dwnet(x_train.shape[1], len(TARGET_CLASSES))
    model.compile(optimizer=Adam(learning_rate=cfg.learning_rate, clipnorm=1.0), loss=loss_fn, metrics=["accuracy"])

    history = model.fit(
        x_train,
        y_train_oh,
        sample_weight=train_weights,
        validation_data=(x_val, y_val_oh, val_weights),
        epochs=cfg.max_epochs,
        batch_size=cfg.batch_size,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=LR_PLATEAU_PATIENCE, min_lr=1e-6),
        ],
        verbose=1,
    )

    model_path = os.path.join(cfg.output_dir, "sedwnet_unified_7class.keras")
    pipeline_path = os.path.join(cfg.output_dir, "preprocessing_pipeline.pkl")
    features_path = os.path.join(cfg.output_dir, "final_features.txt")
    model.save(model_path)
    with open(pipeline_path, "wb") as f:
        pickle.dump(pipeline, f)
    with open(features_path, "w") as f:
        f.write("\n".join(final_features) + "\n")
    pd.DataFrame(history.history).to_csv(os.path.join(cfg.output_dir, "training_history.csv"), index=False)

    metrics = {}
    metrics.update(evaluate_all(model=model, x=x_val, frame=splits["val"], split_name="val", output_dir=cfg.output_dir, batch_size=cfg.batch_size))
    metrics.update(evaluate_all(model=model, x=x_test, frame=splits["test"], split_name="test", output_dir=cfg.output_dir, batch_size=cfg.batch_size))

    metadata = {
        "experiment": "unified_sedwnet_experiment_b",
        "split_mode": cfg.split_mode,
        "base_csv": cfg.base_csv,
        "custom_csv": cfg.custom_csv,
        "model_path": model_path,
        "classes": TARGET_CLASSES,
        "selected_features": final_features,
        "train_full_counts": split_counts(splits["train_full"]),
        "train_sample_counts": split_counts(train),
        "val_counts": split_counts(splits["val"]),
        "test_counts": split_counts(splits["test"]),
        "custom_weight_config": cfg.custom_weight if cfg.custom_weight is not None else "auto_domain_balance_from_train_sample",
        "custom_weight_used": custom_weight,
        "metrics": metrics,
        "leakage_controls": [
            "Exact model-visible duplicate rows are deduplicated before splitting.",
            "Rows with identical model-visible features but conflicting labels are dropped.",
            "Train/val/test duplicate feature hashes are checked after splitting.",
            "Encoders, numeric scaler, feature selector, and final scaler are fit on train only.",
            "Timestamp, date, time, IP, and target columns are dropped from model features.",
        ],
    }
    with open(os.path.join(cfg.output_dir, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    with open(os.path.join(cfg.output_dir, "training_metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    print("DONE.")
    print(f"Model:    {model_path}")
    print(f"Pipeline: {pipeline_path}")
    print(f"Reports:  {cfg.output_dir}")


def build_parser(default_output_rel: str) -> argparse.ArgumentParser:
    root = project_root()
    parser = argparse.ArgumentParser(
        description="Train unified SEDWNet Experiment B.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-csv", default=os.path.join(root, BASE_CSV_REL))
    parser.add_argument("--custom-csv", default=os.path.join(root, CUSTOM_CSV_REL))
    parser.add_argument("--output-dir", default=os.path.join(root, default_output_rel))
    parser.add_argument(
        "--base-train-per-class",
        type=int,
        default=BASE_TRAIN_PER_CLASS,
        help="Base/original TON-IoT train rows to sample per class.",
    )
    parser.add_argument("--custom-train-per-class", type=int, default=CUSTOM_TRAIN_PER_CLASS, help="0 means all custom train rows.")
    parser.add_argument("--custom-dos-ddos-train-cap", type=int, default=CUSTOM_DOS_DDOS_TRAIN_CAP)
    parser.add_argument("--custom-weight", type=float, default=None, help="Default auto-balances custom vs base domain loss.")
    parser.add_argument("--target-k", type=int, default=TARGET_K)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--test-ratio", type=float, default=TEST_RATIO)
    parser.add_argument("--base-time-col", default=None)
    parser.add_argument("--custom-time-col", default=None)
    return parser


def config_from_args(
    args: argparse.Namespace,
    *,
    split_mode: Literal["random", "temporal"],
) -> ExperimentConfig:
    ratio_total = args.train_ratio + args.val_ratio + args.test_ratio
    if not np.isclose(ratio_total, 1.0):
        raise ValueError(f"train/val/test ratios must sum to 1.0, got {ratio_total:.4f}")
    return ExperimentConfig(
        split_mode=split_mode,
        project_root=project_root(),
        base_csv=os.path.abspath(args.base_csv),
        custom_csv=os.path.abspath(args.custom_csv),
        output_dir=os.path.abspath(args.output_dir),
        base_train_per_class=args.base_train_per_class,
        custom_train_per_class=args.custom_train_per_class,
        custom_dos_ddos_train_cap=args.custom_dos_ddos_train_cap,
        custom_weight=args.custom_weight,
        target_k=args.target_k,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        base_time_col=args.base_time_col,
        custom_time_col=args.custom_time_col,
    )
