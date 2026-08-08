"""Train/export a CIC 6-class model inside the legacy deployment environment.

Use this script on the AVX-only deployment server, inside the locked venv:

    Python       3.11
    TensorFlow   2.12.1
    NumPy        1.24.3
    scikit-learn 1.3.2
    SciPy        1.11.4

It intentionally avoids artifacts produced by newer NumPy/scikit-learn builds.
The model is exported as TensorFlow SavedModel and the preprocessing pipeline is
exported with standard pickle from the same local runtime.

Expected target taxonomy:

    backdoor, dos_ddos, infiltration, normal, password, scanning

Example:

    python -u resnet/train_cic_public_legacy_tf212.py \
      --csv /data/ton-iot-project/fresh_start/data/cic_public_6class.csv \
      --output-dir /data/ton-iot-project/fresh_start/artifacts/resnet_cic_public_tf212 \
      --target-k 40 \
      --batch-size 512 \
      --max-epochs 50
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import sys
from collections import Counter
from functools import partial
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
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


TARGET_CLASSES = {"backdoor", "dos_ddos", "infiltration", "normal", "password", "scanning"}
LABEL_CANDIDATES = ("type", "attack", "category", "label", "Label")
TIME_COLS = ("ts", "timestamp", "datetime", "date", "time")
IP_COLS = ("src_ip", "dst_ip", "srcip", "dstip")
DROP_ALWAYS = {"flow_id", "raw_label", "source_label", "split_time"}

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

SEED = 42


class SafeLabelEncoder(BaseEstimator, TransformerMixin):
    """Deterministic categorical encoder with unknown values mapped to 0."""

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


def project_root() -> str:
    env_root = os.environ.get("TON_IOT_PROJECT_ROOT")
    if env_root and os.path.isdir(env_root):
        return os.path.abspath(env_root)
    cwd = os.path.abspath(os.getcwd())
    if os.path.isdir(os.path.join(cwd, "data")) and os.path.isdir(os.path.join(cwd, "artifacts")):
        return cwd
    return cwd


def canon_label(label: object) -> str:
    value = str(label).strip().lower()
    return "dos_ddos" if value in {"dos", "ddos", "ddos_dos"} else value


def label_column(df: pd.DataFrame, explicit: str | None) -> str:
    if explicit:
        if explicit not in df.columns:
            raise RuntimeError(f"Label column '{explicit}' not found. First columns: {list(df.columns[:40])}")
        return explicit
    for col in LABEL_CANDIDATES:
        if col not in df.columns:
            continue
        values = set(df[col].dropna().astype(str).head(50_000).map(canon_label).unique())
        if values.intersection(TARGET_CLASSES):
            return col
    raise RuntimeError(f"Could not identify target label column. First columns: {list(df.columns[:40])}")


def counter_to_int_dict(values) -> dict[str, int]:
    return {str(label): int(count) for label, count in Counter(values).items()}


def load_clean_dataset(csv_path: str, label_col: str | None) -> tuple[pd.DataFrame, pd.Series, dict]:
    print("Loading CSV...", flush=True)
    df = pd.read_csv(csv_path, low_memory=False, on_bad_lines="skip")
    df.columns = df.columns.str.strip()
    target_col = label_column(df, label_col)

    labels = df[target_col].map(canon_label)
    keep = labels.isin(TARGET_CLASSES)
    dropped = int((~keep).sum())
    if dropped:
        print(f"Dropping unsupported labels: {dropped:,}")
    df = df.loc[keep].copy()
    labels = labels.loc[keep].astype(str).reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No rows left after filtering to the CIC 6-class taxonomy.")

    drop_cols = set(LABEL_CANDIDATES) | set(TIME_COLS) | set(IP_COLS) | DROP_ALWAYS
    drop_cols.add(target_col)
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True, errors="ignore")
    df = df.reset_index(drop=True)

    info = {
        "target_column": target_col,
        "rows": int(len(df)),
        "columns_after_drop": int(len(df.columns)),
        "class_counts": counter_to_int_dict(labels),
        "dropped_unsupported_labels": dropped,
    }
    print(f"Rows after cleanup: {len(df):,}")
    print(f"Feature columns after metadata drop: {len(df.columns):,}")
    print(f"Class counts: {info['class_counts']}")
    return df, labels, info


def cap_per_class(x: pd.DataFrame, y: pd.Series, cap: int, seed: int) -> tuple[pd.DataFrame, pd.Series, dict]:
    if cap <= 0:
        return x.reset_index(drop=True), y.reset_index(drop=True), {"enabled": False, "cap": int(cap)}

    rng = np.random.default_rng(seed)
    selected_parts = []
    before_counts = counter_to_int_dict(y)
    for label in sorted(y.unique()):
        idx = np.flatnonzero(y.to_numpy() == label)
        if len(idx) > cap:
            idx = rng.choice(idx, size=cap, replace=False)
        selected_parts.append(np.asarray(idx, dtype=np.int64))

    selected = np.concatenate(selected_parts)
    rng.shuffle(selected)
    x_out = x.iloc[selected].reset_index(drop=True)
    y_out = y.iloc[selected].reset_index(drop=True)
    info = {
        "enabled": True,
        "cap": int(cap),
        "before_counts": before_counts,
        "after_counts": counter_to_int_dict(y_out),
    }
    print(f"Applied per-class cap={cap:,}: {info['after_counts']}")
    return x_out, y_out, info


def split_dataset(
    x: pd.DataFrame,
    y: pd.Series,
    *,
    val_size: float,
    test_size: float,
    final_holdout_size: float,
    seed: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series], dict]:
    if not (0 <= final_holdout_size < 1):
        raise ValueError("--final-holdout-size must be >= 0 and < 1")
    if not (0 < val_size < 1 and 0 < test_size < 1 and val_size + test_size < 1):
        raise ValueError("--val-size and --test-size must be positive and sum to less than 1")

    info = {
        "val_size": float(val_size),
        "test_size": float(test_size),
        "final_holdout_size": float(final_holdout_size),
    }

    if final_holdout_size > 0:
        x_pool, x_holdout, y_pool, y_holdout = train_test_split(
            x,
            y,
            test_size=final_holdout_size,
            stratify=y,
            random_state=seed,
        )
    else:
        x_pool, y_pool = x, y
        x_holdout = x.iloc[0:0].copy()
        y_holdout = y.iloc[0:0].copy()

    temp_size = val_size + test_size
    test_ratio_of_temp = test_size / temp_size
    x_train, x_temp, y_train, y_temp = train_test_split(
        x_pool,
        y_pool,
        test_size=temp_size,
        stratify=y_pool,
        random_state=seed,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=test_ratio_of_temp,
        stratify=y_temp,
        random_state=seed,
    )

    frames = {
        "train": x_train.reset_index(drop=True),
        "val": x_val.reset_index(drop=True),
        "test": x_test.reset_index(drop=True),
        "final_holdout": x_holdout.reset_index(drop=True),
    }
    labels = {
        "train": y_train.reset_index(drop=True),
        "val": y_val.reset_index(drop=True),
        "test": y_test.reset_index(drop=True),
        "final_holdout": y_holdout.reset_index(drop=True),
    }
    info["counts"] = {split: counter_to_int_dict(values) for split, values in labels.items()}
    print("Split counts:")
    for split, counts in info["counts"].items():
        print(f"  {split}: {counts}")
    return frames, labels, info


def convert_feature_frames(frames: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], list[str], list[str], list[str]]:
    train = frames["train"].copy()
    cat_cols = [col for col in CAT_COLS if col in train.columns]
    num_cols = [col for col in train.columns if col not in cat_cols]

    converted: dict[str, pd.DataFrame] = {}
    for name, frame in frames.items():
        current = frame.copy()
        for col in cat_cols:
            if col not in current.columns:
                current[col] = "missing"
            current[col] = current[col].fillna("missing").replace("-", "missing").astype(str)
        for col in num_cols:
            if col not in current.columns:
                current[col] = 0
            current[col] = pd.to_numeric(current[col], errors="coerce")
        current.replace([np.inf, -np.inf], 0, inplace=True)
        current = current.fillna(0)
        for col in LOG_COLS:
            if col in current.columns and col in num_cols:
                current[col] = np.log1p(pd.to_numeric(current[col], errors="coerce").fillna(0).clip(lower=0))
        converted[name] = current

    train_converted = converted["train"]
    constant_cols = [col for col in train_converted.columns if train_converted[col].nunique(dropna=False) <= 1]
    if constant_cols:
        print(f"Dropping train-constant columns: {len(constant_cols)}")
        for name in converted:
            converted[name].drop(columns=constant_cols, errors="ignore", inplace=True)
        cat_cols = [col for col in cat_cols if col not in constant_cols]
        num_cols = [col for col in num_cols if col not in constant_cols]

    ordered_cols = converted["train"].columns.tolist()
    for name in converted:
        for col in ordered_cols:
            if col not in converted[name].columns:
                converted[name][col] = "missing" if col in cat_cols else 0
        converted[name] = converted[name][ordered_cols]
        for col in converted[name].columns:
            if converted[name][col].dtype == "float64":
                converted[name][col] = converted[name][col].astype("float32")
            elif converted[name][col].dtype == "int64":
                converted[name][col] = converted[name][col].astype("int32")

    return converted, cat_cols, num_cols, constant_cols


def fit_preprocessing(
    frames: dict[str, pd.DataFrame],
    y_train_encoded: np.ndarray,
    *,
    target_k: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict, list[str]]:
    frames, valid_cat_cols, num_cols, constant_cols = convert_feature_frames(frames)

    encoders = {}
    for col in valid_cat_cols:
        encoder = SafeLabelEncoder()
        frames["train"][col] = encoder.fit(frames["train"][col]).transform(frames["train"][col])
        for split in ("val", "test", "final_holdout"):
            frames[split][col] = encoder.transform(frames[split][col])
        encoders[col] = encoder

    scaler_num = MinMaxScaler()
    if num_cols:
        frames["train"][num_cols] = scaler_num.fit_transform(frames["train"][num_cols].values)
        for split in ("val", "test", "final_holdout"):
            frames[split][num_cols] = scaler_num.transform(frames[split][num_cols].values)
    else:
        scaler_num.fit(np.zeros((1, 1), dtype=np.float32))

    feature_names = frames["train"].columns.tolist()
    if not feature_names:
        raise RuntimeError("No usable feature columns remain after preprocessing.")

    discrete_mask = np.array([col in valid_cat_cols for col in feature_names], dtype=bool)
    k = min(target_k, len(feature_names)) if target_k > 0 else len(feature_names)
    print(f"Selecting top {k} features from {len(feature_names)} columns...")
    mi_scorer = partial(
        mutual_info_classif,
        discrete_features=discrete_mask,
        n_neighbors=3,
        random_state=seed,
    )
    selector = SelectKBest(score_func=mi_scorer, k=k)
    selector.fit(frames["train"], y_train_encoded)

    selected_mask = selector.get_support()
    final_features = [feature for feature, keep in zip(feature_names, selected_mask) if keep]
    print(f"Selected features ({len(final_features)}): {final_features}")

    selected_arrays = {
        split: selector.transform(frame).astype("float32")
        for split, frame in frames.items()
    }

    final_scaler = MinMaxScaler()
    selected_arrays["train"] = np.nan_to_num(final_scaler.fit_transform(selected_arrays["train"])).astype("float32")
    for split in ("val", "test", "final_holdout"):
        selected_arrays[split] = np.nan_to_num(final_scaler.transform(selected_arrays[split])).astype("float32")

    pipeline = {
        "schema_version": "legacy_tf212_sklearn132",
        "scaler_num": scaler_num,
        "selector": selector,
        "final_scaler": final_scaler,
        "encoders": encoders,
        "target_encoder": None,
        "features": final_features,
        "valid_cat_cols": valid_cat_cols,
        "num_cols": num_cols,
        "constant_cols": constant_cols,
        "log_cols": LOG_COLS,
    }
    return selected_arrays, pipeline, final_features


def se_block_1d(x, reduction: int = 16):
    channels = int(x.shape[-1])
    squeeze = GlobalAveragePooling1D()(x)
    squeeze = Reshape((1, channels))(squeeze)
    hidden = max(channels // reduction, 4)
    excite = Dense(hidden, activation="relu", kernel_initializer="he_normal", use_bias=False)(squeeze)
    excite = Dense(channels, activation="sigmoid", kernel_initializer="he_normal", use_bias=False)(excite)
    return Multiply()([x, excite])


def sedwnet_block(x, filters: int, stride: int = 1, dropout: float = 0.0):
    residual = x
    if stride != 1 or int(x.shape[-1]) != filters:
        residual = Conv1D(filters, 1, strides=stride, padding="same", kernel_initializer="he_normal")(residual)
        residual = BatchNormalization()(residual)

    x = SeparableConv1D(filters, 3, strides=stride, padding="same", depthwise_initializer="he_normal", pointwise_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = SeparableConv1D(filters, 3, padding="same", depthwise_initializer="he_normal", pointwise_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = se_block_1d(x)
    if dropout > 0:
        x = Dropout(dropout)(x)
    x = Add()([x, residual])
    return Activation("relu")(x)


def build_model(input_dim: int, num_classes: int) -> Model:
    inputs = Input(shape=(input_dim,), name="tabular_input")
    x = Reshape((input_dim, 1), name="feat_as_1d")(inputs)
    x = Conv1D(48, 3, padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = sedwnet_block(x, filters=48, stride=1)
    x = sedwnet_block(x, filters=96, stride=2)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.35)(x)
    outputs = Dense(num_classes, activation="softmax", dtype="float32")(x)
    return Model(inputs, outputs, name="SE_DWNet_CIC_Public_Legacy")


def class_weights(y_int: np.ndarray) -> dict[int, float]:
    counts = Counter(int(item) for item in y_int)
    total = float(len(y_int))
    num_classes = float(len(counts))
    weights = {}
    for cls, count in counts.items():
        weights[cls] = float(total / (num_classes * max(float(count), 1.0)))
    return weights


def evaluate_split(model, x: np.ndarray, y_int: np.ndarray, encoder: LabelEncoder, class_names: list[str], name: str, output_dir: Path, batch_size: int) -> dict:
    if len(x) == 0:
        return {"enabled": False}
    probs = model.predict(x, batch_size=batch_size, verbose=0)
    pred_int = np.argmax(probs, axis=1)
    y_true = encoder.inverse_transform(y_int)
    y_pred = encoder.inverse_transform(pred_int)
    report = classification_report(
        y_true,
        y_pred,
        labels=class_names,
        target_names=class_names,
        zero_division=0,
        digits=4,
    )
    print(f"\nClassification Report ({name.upper()}):")
    print(report)

    report_path = output_dir / f"{name}_classification_report.txt"
    report_path.write_text(report, encoding="utf-8")
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    cm_path = output_dir / f"{name}_confusion_matrix.json"
    cm_payload = {
        "labels": class_names,
        "matrix": cm.astype(int).tolist(),
    }
    cm_path.write_text(json.dumps(cm_payload, indent=2), encoding="utf-8")

    pred_path = output_dir / f"{name}_predictions.csv"
    pred_data = {
        "true_class": y_true,
        "predicted_class": y_pred,
        "confidence": np.max(probs, axis=1),
        "correct": y_true == y_pred,
    }
    for index, cls in enumerate(class_names):
        pred_data[f"prob_{cls}"] = probs[:, index]
    pd.DataFrame(pred_data).to_csv(pred_path, index=False)

    return {
        "enabled": True,
        "report_path": str(report_path),
        "confusion_matrix_path": str(cm_path),
        "predictions_path": str(pred_path),
    }


def build_parser() -> argparse.ArgumentParser:
    root = project_root()
    parser = argparse.ArgumentParser(
        description="Train/export CIC artifacts inside the TF 2.12 / sklearn 1.3 legacy deployment venv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", default=os.path.join(root, "data", "cic_public_6class.csv"), help="CIC 6-class training CSV.")
    parser.add_argument("--label-col", default=None)
    parser.add_argument("--output-dir", default=os.path.join(root, "artifacts", "resnet_cic_public_tf212"))
    parser.add_argument("--target-k", type=int, default=40)
    parser.add_argument("--per-class-cap", type=int, default=150_000, help="0 keeps all rows.")
    parser.add_argument("--final-holdout-size", type=float, default=0.05)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--no-class-weight", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Legacy CIC Training Export ===")
    print(f"Python:       {sys.version.split()[0]}")
    print(f"TensorFlow:   {tf.__version__}")
    print(f"NumPy:        {np.__version__}")
    print(f"pandas:       {pd.__version__}")
    print(f"scikit-learn: {sklearn.__version__}")
    print(f"CSV:          {args.csv}")
    print(f"Output dir:   {output_dir}")

    x, y, cleanup_info = load_clean_dataset(args.csv, args.label_col)
    x, y, cap_info = cap_per_class(x, y, args.per_class_cap, args.seed)

    frames, labels, split_info = split_dataset(
        x,
        y,
        val_size=args.val_size,
        test_size=args.test_size,
        final_holdout_size=args.final_holdout_size,
        seed=args.seed,
    )
    del x, y
    gc.collect()

    label_encoder = LabelEncoder()
    label_encoder.fit(labels["train"])
    class_names = [str(cls) for cls in label_encoder.classes_.tolist()]
    y_train = label_encoder.transform(labels["train"])
    y_val = label_encoder.transform(labels["val"])
    y_test = label_encoder.transform(labels["test"])
    y_holdout = label_encoder.transform(labels["final_holdout"]) if len(labels["final_holdout"]) else np.asarray([], dtype=np.int64)

    arrays, pipeline, final_features = fit_preprocessing(
        frames,
        y_train,
        target_k=args.target_k,
        seed=args.seed,
    )
    pipeline["target_encoder"] = label_encoder
    pipeline["seed"] = args.seed
    pipeline["dataset_name"] = "cic_public_6class"
    pipeline["data_csv"] = os.path.abspath(args.csv)

    del frames, labels
    gc.collect()

    model = build_model(arrays["train"].shape[1], len(class_names))
    optimizer = Adam(learning_rate=args.learning_rate, clipnorm=1.0)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0.02)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    model.summary()

    train_onehot = to_categorical(y_train, num_classes=len(class_names)).astype("float32")
    val_onehot = to_categorical(y_val, num_classes=len(class_names)).astype("float32")
    weights = None if args.no_class_weight else class_weights(y_train)
    print(f"Class weights: {weights}")

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(2, args.patience // 3), min_lr=1e-6),
    ]
    history = model.fit(
        arrays["train"],
        train_onehot,
        validation_data=(arrays["val"], val_onehot),
        epochs=args.max_epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        class_weight=weights,
        verbose=1,
    )

    saved_model_dir = output_dir / "saved_model"
    pipeline_path = output_dir / "preprocessing_pipeline.pkl"
    features_path = output_dir / "final_features.txt"
    metadata_path = output_dir / "training_metadata.json"

    print("Saving TensorFlow SavedModel...")
    model.save(str(saved_model_dir), save_format="tf", include_optimizer=False)

    print("Saving preprocessing pipeline pickle...")
    with pipeline_path.open("wb") as handle:
        pickle.dump(pipeline, handle, protocol=pickle.HIGHEST_PROTOCOL)
    features_path.write_text("\n".join(final_features) + "\n", encoding="utf-8")

    test_eval = evaluate_split(model, arrays["test"], y_test, label_encoder, class_names, "test", output_dir, args.batch_size)
    holdout_eval = evaluate_split(
        model,
        arrays["final_holdout"],
        y_holdout,
        label_encoder,
        class_names,
        "final_holdout",
        output_dir,
        args.batch_size,
    )

    metadata = {
        "script": "train_cic_public_legacy_tf212.py",
        "schema_version": "legacy_tf212_sklearn132",
        "python": sys.version,
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "data_csv": os.path.abspath(args.csv),
        "output_dir": str(output_dir),
        "saved_model_dir": str(saved_model_dir),
        "pipeline_path": str(pipeline_path),
        "features_path": str(features_path),
        "classes": class_names,
        "selected_features": final_features,
        "cleanup": cleanup_info,
        "cap": cap_info,
        "split": split_info,
        "epochs_ran": int(len(history.history.get("loss", []))),
        "history": {key: [float(item) for item in values] for key, values in history.history.items()},
        "test_eval": test_eval,
        "final_holdout_eval": holdout_eval,
        "class_weight": weights,
        "loss": {"name": "CategoricalCrossentropy", "label_smoothing": 0.02},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nDONE")
    print(f"SavedModel: {saved_model_dir}")
    print(f"Pipeline:   {pipeline_path}")
    print(f"Features:   {features_path}")
    print(f"Metadata:   {metadata_path}")


if __name__ == "__main__":
    main()
