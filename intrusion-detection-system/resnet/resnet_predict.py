"""
General-purpose prediction script for the SE-DWNet transfer model.

Loads the transfer model + preprocessing pipeline and runs inference
on any Zeek-format CSV.  If labels are present, produces a classification
report; otherwise outputs raw predictions.

Usage:
    # With labels (validation mode)
    python resnet_predict.py --csv /path/to/data.csv --label-col type

    # Without labels (inference mode)
    python resnet_predict.py --csv /path/to/data.csv --output predictions.csv

    # Explicit model directory
    python resnet_predict.py --csv data.csv --model-dir /path/to/artifacts/resnet_transfer_7class
"""
import os
import sys
import argparse
import pickle
import glob
from collections import Counter

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

import seaborn as sns
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow imports from parent directory (app.preprocessing)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(SCRIPT_DIR, "..")))
from app.preprocessing import transform_with_pipeline


TARGET_CLASSES = ["backdoor", "dos_ddos", "injection", "normal",
                  "password", "scanning", "xss"]


# ── Path helpers ──────────────────────────────────────────────────────────────
def _detect_project_root() -> str:
    env_root = os.environ.get("TON_IOT_PROJECT_ROOT")
    if env_root and os.path.isdir(env_root):
        return os.path.abspath(env_root)
    cwd = os.path.abspath(os.getcwd())
    if os.path.isdir(os.path.join(cwd, "data")) and os.path.isdir(
        os.path.join(cwd, "artifacts")
    ):
        return cwd
    return os.path.normpath(os.path.join(SCRIPT_DIR, ".."))


def _find_model_dir(project_root: str, explicit: str | None) -> str:
    """Locate the transfer model artifact directory."""
    if explicit and os.path.isdir(explicit):
        return explicit
    candidates = [
        os.path.join(project_root, "artifacts", "resnet_transfer_7class"),
        os.path.join(project_root, "artifacts", "resnet_transfer"),
        os.path.join(project_root, "artifacts", "resnet_base"),
    ]
    for d in candidates:
        model_files = glob.glob(os.path.join(d, "*.keras"))
        pipeline_files = glob.glob(os.path.join(d, "*pipeline*.pkl"))
        if model_files and pipeline_files:
            return d
    raise FileNotFoundError(
        "Model directory not found. Tried:\n"
        + "\n".join(f"  - {d}" for d in candidates)
        + "\nUse --model-dir to specify explicitly."
    )


def _load_final_features(model_dir: str, pipeline: dict) -> list[str]:
    txt_path = os.path.join(model_dir, "final_features.txt")
    if os.path.exists(txt_path):
        with open(txt_path) as f:
            feats = [line.strip() for line in f if line.strip()]
        if feats:
            return feats
    feats = pipeline.get("features")
    if not feats:
        raise RuntimeError("Cannot determine final features.")
    return [str(feat).strip() for feat in feats if str(feat).strip()]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Run predictions with the SE-DWNet transfer model."
    )
    parser.add_argument(
        "--csv", required=True, help="Input CSV file (Zeek-format features)."
    )
    parser.add_argument(
        "--model-dir", default=None,
        help="Model artifact directory (auto-detected if omitted).",
    )
    parser.add_argument(
        "--label-col", default=None,
        help="Label column name for evaluation mode (e.g. 'type', 'label'). "
             "If omitted, runs inference-only.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output CSV path for predictions (inference mode).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2048,
        help="Prediction batch size.",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Cap total samples (for quick testing).",
    )
    args = parser.parse_args()

    project_root = _detect_project_root()
    model_dir = _find_model_dir(project_root, args.model_dir)

    print("=== SE-DWNet Prediction ===")
    print(f"Input:     {args.csv}")
    print(f"Model dir: {model_dir}")

    # ── Load model + pipeline ─────────────────────────────────────────────
    model_path = sorted(glob.glob(os.path.join(model_dir, "*.keras")))[-1]
    model = tf.keras.models.load_model(model_path, compile=False)
    print(f"Model:     {model_path}")

    pipeline_path = sorted(glob.glob(os.path.join(model_dir, "*pipeline*.pkl")))[-1]
    with open(pipeline_path, "rb") as f:
        pipeline = pickle.load(f)
    print(f"Pipeline:  {pipeline_path}")

    final_features = _load_final_features(model_dir, pipeline)
    print(f"Features:  {len(final_features)}")

    # Load label encoder
    meta_path = os.path.join(model_dir, "transfer_training_metadata.pkl")
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        le_target = meta["label_encoder"]
        class_names = meta["classes"]
    else:
        class_names = TARGET_CLASSES
        from sklearn.preprocessing import LabelEncoder
        le_target = LabelEncoder()
        le_target.fit(class_names)
    print(f"Classes:   {class_names}")

    # ── Load input data ──────────────────────────────────────────────────
    print(f"Loading {args.csv} ...")
    df = pd.read_csv(args.csv, low_memory=False)
    df.columns = df.columns.str.strip()

    # Drop time/metadata columns
    df.drop(columns=["ts", "date", "time", "label"], errors="ignore", inplace=True)

    # Drop IPs
    ip_cols = [c for c in ["src_ip", "dst_ip", "srcip", "dstip"] if c in df.columns]
    if ip_cols:
        df.drop(columns=ip_cols, inplace=True)

    # Extract labels if present
    y_true_str = None
    label_col = args.label_col
    if label_col is None:
        for c in ("type", "label", "Label"):
            if c in df.columns:
                label_col = c
                break

    if label_col and label_col in df.columns:
        y_true_str = df[label_col].astype(str).str.strip().str.lower().values
        df = df.drop(columns=[label_col])
        print(f"Labels found ({label_col}): {dict(Counter(y_true_str))}")
    else:
        print("No label column found — running in inference-only mode.")

    if args.max_samples and len(df) > args.max_samples:
        idx = np.random.RandomState(42).permutation(len(df))[: args.max_samples]
        df = df.iloc[idx].reset_index(drop=True)
        if y_true_str is not None:
            y_true_str = y_true_str[idx]
        print(f"Subsampled to {len(df)} rows")

    print(f"Input shape: {df.shape}")

    # ── Transform + predict ──────────────────────────────────────────────
    print("Transforming features...")
    records = df.to_dict(orient="records")

    CHUNK = 50_000
    all_preds = []
    for i in range(0, len(records), CHUNK):
        chunk_records = records[i : i + CHUNK]
        X_chunk = transform_with_pipeline(
            chunk_records, pipeline=pipeline, final_features=final_features
        )
        preds = model.predict(X_chunk, batch_size=args.batch_size, verbose=0)
        all_preds.append(preds)
        print(f"  Processed {min(i + CHUNK, len(records))}/{len(records)}")

    all_probs = np.vstack(all_preds)
    y_pred_int = np.argmax(all_probs, axis=1)
    y_pred_str = le_target.inverse_transform(y_pred_int)
    confidence = np.max(all_probs, axis=1)

    # ── Output results ───────────────────────────────────────────────────
    if y_true_str is not None:
        # Evaluation mode
        report = classification_report(
            y_true_str, y_pred_str,
            labels=class_names,
            target_names=class_names,
            zero_division=0,
            digits=4,
        )
        print("\n" + "=" * 60)
        print("Classification Report")
        print("=" * 60)
        print(report)

        # Confusion matrix
        cm = confusion_matrix(y_true_str, y_pred_str, labels=class_names)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt="d",
            xticklabels=class_names, yticklabels=class_names,
            cmap="Blues",
        )
        plt.title("Prediction Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        cm_path = os.path.join(model_dir, "prediction_confusion_matrix.png")
        plt.savefig(cm_path, dpi=200)
        plt.close()
        print(f"Confusion matrix saved: {cm_path}")

    # Save predictions
    out_path = args.output
    if out_path is None and y_true_str is None:
        out_path = os.path.splitext(args.csv)[0] + "_predictions.csv"

    if out_path:
        result_df = pd.DataFrame({
            "predicted_class": y_pred_str,
            "confidence": confidence,
        })
        if y_true_str is not None:
            result_df["true_class"] = y_true_str
            result_df["correct"] = y_true_str == y_pred_str

        # Add per-class probabilities
        for i, cls in enumerate(class_names):
            result_df[f"prob_{cls}"] = all_probs[:, i]

        result_df.to_csv(out_path, index=False)
        print(f"Predictions saved: {out_path}")

    # Summary
    print(f"\nTotal predictions: {len(y_pred_str)}")
    print(f"Prediction distribution: {dict(Counter(y_pred_str))}")
    print(f"Mean confidence: {confidence.mean():.4f}")
    print("DONE.")


if __name__ == "__main__":
    main()
