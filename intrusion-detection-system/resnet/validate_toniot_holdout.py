"""
Validate a labelled network-flow CSV with a saved SEDWNet/ResNet artifact.

The default target is the never-seen TON-IoT holdout CSV and the default model
artifact is the temporal unified SEDWNet, when present. The same script also
validates the custom/TPOT CSV because both datasets use the same 7-class label
space after cleanup.

Examples
--------
    # Unseen TON-IoT holdout against the temporal unified model
    python resnet/validate_toniot_holdout.py

    # Custom dataset against a unified model
    python resnet/validate_toniot_holdout.py \
        --csv data/custom/tpot_finetune.csv \
        --dataset-name custom \
        --model-dir artifacts/resnet_unified_7class_temporal
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import re
import sys
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(SCRIPT_DIR, "..")))

from app.preprocessing import transform_with_pipeline  # noqa: E402


class SafeLabelEncoder(BaseEstimator, TransformerMixin):
    """Compatibility shim for older pipeline pickles that reference __main__."""

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
        return (
            pd.Series(y)
            .astype(str)
            .map(self.mapper)
            .fillna(self.unknown_token)
            .astype(np.int32)
            .values
        )


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
LABEL_CANDIDATES = ("type", "attack", "category", "label", "Label")
TIME_COLS = ("ts", "timestamp", "datetime", "date", "time")
IP_COLS = ("src_ip", "dst_ip", "srcip", "dstip")


def _select_probs(raw_output, head: str = "original_head") -> np.ndarray:
    if isinstance(raw_output, dict):
        return np.asarray(raw_output.get(head, next(iter(raw_output.values()))))
    if isinstance(raw_output, (list, tuple)):
        return np.asarray(raw_output[0])
    return np.asarray(raw_output)


def _detect_project_root() -> str:
    env_root = os.environ.get("TON_IOT_PROJECT_ROOT")
    if env_root and os.path.isdir(env_root):
        return os.path.abspath(env_root)
    cwd = os.path.abspath(os.getcwd())
    if os.path.isdir(os.path.join(cwd, "data")) and os.path.isdir(os.path.join(cwd, "artifacts")):
        return cwd
    return os.path.normpath(os.path.join(SCRIPT_DIR, ".."))


def _find_model_dir(project_root: str, explicit: str | None) -> str:
    if explicit and os.path.isdir(explicit):
        return os.path.abspath(explicit)

    candidates = [
        os.path.join(project_root, "artifacts", "resnet_custom"),
        os.path.join(project_root, "artifacts", "resnet_unified_7class_temporal"),
        os.path.join(project_root, "artifacts", "resnet_unified_7class_random"),
        os.path.join(project_root, "artifacts", "resnet_base"),
    ]
    for path in candidates:
        model_files = glob.glob(os.path.join(path, "*.keras"))
        pipeline_files = glob.glob(os.path.join(path, "*pipeline*.pkl"))
        if model_files and pipeline_files:
            return path

    raise FileNotFoundError(
        "Model directory not found. Tried:\n"
        + "\n".join(f"  - {path}" for path in candidates)
        + "\nUse --model-dir to specify explicitly."
    )


def _load_final_features(model_dir: str, pipeline: dict) -> list[str]:
    txt_path = os.path.join(model_dir, "final_features.txt")
    if os.path.exists(txt_path):
        with open(txt_path) as f:
            features = [line.strip() for line in f if line.strip()]
        if features:
            return features

    features = pipeline.get("features")
    if not features:
        raise RuntimeError("Cannot determine final feature list.")
    return [str(feature).strip() for feature in features if str(feature).strip()]


def _pick_model_file(model_dir: str) -> str:
    preferred_names = [
        "sedwnet_unified_7class.keras",
        "resnet_model.keras",
    ]
    for name in preferred_names:
        preferred = os.path.join(model_dir, name)
        if os.path.exists(preferred):
            return preferred

    model_files = sorted(glob.glob(os.path.join(model_dir, "*.keras")))
    if not model_files:
        raise FileNotFoundError(f"No .keras model found in {model_dir}")
    return model_files[-1]


def _pick_pipeline_file(model_dir: str) -> str:
    preferred = os.path.join(model_dir, "preprocessing_pipeline.pkl")
    if os.path.exists(preferred):
        return preferred

    pipeline_files = sorted(glob.glob(os.path.join(model_dir, "*pipeline*.pkl")))
    if not pipeline_files:
        raise FileNotFoundError(f"No pipeline pickle found in {model_dir}")
    return pipeline_files[-1]


def _load_label_encoder(model_dir: str, pipeline: dict) -> tuple[object, list[str]]:
    for meta_name in ("training_metadata.pkl", "transfer_training_metadata.pkl"):
        meta_path = os.path.join(model_dir, meta_name)
        if not os.path.exists(meta_path):
            continue
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        classes = [str(cls) for cls in meta.get("classes", meta.get("all_classes", []))]
        label_encoder = meta.get("label_encoder")
        if label_encoder is not None and classes:
            return label_encoder, classes
        if classes:
            encoder = LabelEncoder()
            encoder.fit(classes)
            return encoder, classes

    pipeline_encoder = pipeline.get("target_encoder")
    if pipeline_encoder is not None and hasattr(pipeline_encoder, "classes_"):
        return pipeline_encoder, [str(cls) for cls in pipeline_encoder.classes_]

    encoder = LabelEncoder()
    encoder.fit(TARGET_CLASSES)
    return encoder, TARGET_CLASSES


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
        if values.intersection(TARGET_CLASSES):
            return col

    raise RuntimeError(
        "Could not identify a label column. "
        f"Tried {LABEL_CANDIDATES}. Columns: {list(df.columns[:40])}"
    )


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip().lower())
    return cleaned.strip("_") or "dataset"


def _infer_dataset_name(csv_path: str, explicit: str | None) -> str:
    if explicit:
        return _safe_name(explicit)
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    if "tpot" in stem.lower() or "custom" in csv_path.lower():
        return "custom"
    if "holdout" in stem.lower() or "toniot" in stem.lower():
        return "toniot_holdout"
    return _safe_name(stem)


def _clean_validation_frame(
    df: pd.DataFrame,
    *,
    class_names: list[str],
    label_col: str | None,
) -> tuple[pd.DataFrame, np.ndarray]:
    df = df.copy()
    df.columns = df.columns.str.strip()

    target_col = _find_label_column(df, label_col)
    labels = df[target_col].map(_canon_label)
    keep = ~labels.isin(DROP_LABELS)
    df = df.loc[keep].copy()
    labels = labels.loc[keep].copy()

    if "ddos_dos" in class_names and "dos_ddos" not in class_names:
        labels = labels.replace({"dos_ddos": "ddos_dos"})

    keep = labels.isin(class_names)
    df = df.loc[keep].copy()
    labels = labels.loc[keep].to_numpy()
    if df.empty:
        raise RuntimeError(f"No validation rows remain after filtering to classes: {class_names}")

    drop_cols = set(LABEL_CANDIDATES) | set(TIME_COLS) | set(IP_COLS)
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True, errors="ignore")
    return df, labels


def validate(
    *,
    csv_path: str,
    model_dir: str,
    output_dir: str | None,
    dataset_name: str,
    label_col: str | None,
    batch_size: int,
    chunk_size: int,
    max_samples: int | None,
) -> None:
    if output_dir is None:
        output_dir = os.path.join(model_dir, f"{dataset_name}_validation")
    os.makedirs(output_dir, exist_ok=True)

    model_path = _pick_model_file(model_dir)
    pipeline_path = _pick_pipeline_file(model_dir)

    print("=== Labelled Dataset Validation ===")
    print(f"Dataset:    {dataset_name}")
    print(f"CSV:        {csv_path}")
    print(f"Model dir:  {model_dir}")
    print(f"Model:      {model_path}")
    print(f"Pipeline:   {pipeline_path}")
    print(f"Output dir: {output_dir}")

    model = tf.keras.models.load_model(model_path, compile=False)

    with open(pipeline_path, "rb") as f:
        pipeline = pickle.load(f)
    final_features = _load_final_features(model_dir, pipeline)
    print(f"Features:   {len(final_features)} -> {final_features}")

    label_encoder, class_names = _load_label_encoder(model_dir, pipeline)
    class_names = [str(cls) for cls in class_names]
    print(f"Classes:    {class_names}")

    print("Loading validation CSV...")
    df = pd.read_csv(csv_path, dtype=str, low_memory=False, on_bad_lines="skip")
    df, y_true = _clean_validation_frame(df, class_names=class_names, label_col=label_col)

    if max_samples and len(df) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(df))[:max_samples]
        df = df.iloc[idx].reset_index(drop=True)
        y_true = y_true[idx]
        print(f"Subsampled to {len(df)} rows")

    print(f"Rows:       {len(df)}")
    print(f"Labels:     {dict(Counter(y_true))}")
    print(f"Columns:    {len(df.columns)}")

    all_probs = []
    records = df.to_dict(orient="records")
    for start in range(0, len(records), chunk_size):
        stop = min(start + chunk_size, len(records))
        x_chunk = transform_with_pipeline(
            records[start:stop],
            pipeline=pipeline,
            final_features=final_features,
        )
        probs = _select_probs(model.predict(x_chunk, batch_size=batch_size, verbose=0))
        all_probs.append(probs)
        print(f"  Processed {stop}/{len(records)}")

    probs = np.vstack(all_probs)
    pred_int = np.argmax(probs, axis=1)
    y_pred = np.asarray(label_encoder.inverse_transform(pred_int)).astype(str)
    confidence = np.max(probs, axis=1)

    report = classification_report(
        y_true,
        y_pred,
        labels=class_names,
        target_names=class_names,
        zero_division=0,
        digits=4,
    )

    print("\n" + "=" * 70)
    print(f"{dataset_name.upper()} VALIDATION")
    print("=" * 70)
    print(report)

    report_path = os.path.join(output_dir, f"{dataset_name}_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"=== {dataset_name} Validation ===\n")
        f.write(f"CSV: {csv_path}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Pipeline: {pipeline_path}\n")
        f.write(f"Rows: {len(y_true)}\n")
        f.write(f"Labels: {dict(Counter(y_true))}\n\n")
        f.write(report)
    print(f"Report saved: {report_path}")

    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="Blues",
    )
    plt.title(f"{dataset_name} Validation")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    cm_path = os.path.join(output_dir, f"{dataset_name}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"Confusion matrix saved: {cm_path}")

    pred_data = {
        "true_class": y_true,
        "predicted_class": y_pred,
        "confidence": confidence,
        "correct": y_true == y_pred,
    }
    for index, cls in enumerate(class_names):
        pred_data[f"prob_{cls}"] = probs[:, index]
    pred_path = os.path.join(output_dir, f"{dataset_name}_predictions.csv")
    pd.DataFrame(pred_data).to_csv(pred_path, index=False)
    print(f"Predictions saved: {pred_path}")

    print(
        "Confidence: "
        f"mean={confidence.mean():.4f}, "
        f"median={np.median(confidence):.4f}, "
        f"min={confidence.min():.4f}"
    )
    for cls in class_names:
        mask = y_true == cls
        if not mask.any():
            continue
        correct = y_pred[mask] == cls
        print(
            f"  {cls:>12s}: n={mask.sum():>7d}  "
            f"conf={confidence[mask].mean():.3f}  "
            f"acc={correct.mean():.3f}"
        )


def main() -> None:
    root = _detect_project_root()
    parser = argparse.ArgumentParser(description="Validate a labelled network-flow CSV with a saved model artifact.")
    parser.add_argument(
        "--csv",
        default=os.path.join(root, "data", "toniot_holdout_validation.csv"),
        help="Labelled validation CSV. Defaults to the unseen TON-IoT holdout CSV.",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Model artifact directory. Defaults to artifacts/resnet_unified_7class_temporal when available.",
    )
    parser.add_argument("--dataset-name", default=None, help="Name used in output files. Auto-derived from CSV if omitted.")
    parser.add_argument("--label-col", default=None, help="Override target label column. Auto-detected if omitted.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    csv_path = os.path.abspath(args.csv)
    model_dir = _find_model_dir(root, args.model_dir)
    dataset_name = _infer_dataset_name(csv_path, args.dataset_name)
    validate(
        csv_path=csv_path,
        model_dir=model_dir,
        output_dir=args.output_dir,
        dataset_name=dataset_name,
        label_col=args.label_col,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
