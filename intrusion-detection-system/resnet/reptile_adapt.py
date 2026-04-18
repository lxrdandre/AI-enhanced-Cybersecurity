"""
Reptile few-shot adaptation — extend the model with new attack classes.

Takes the Reptile meta-weights and adapts them to include new class(es) from
a small labelled CSV, producing a deployable expanded model.

Usage
-----
    # Single new class
    python -m resnet.reptile_adapt \\
        --new-class-csv data/custom/botnet_samples.csv \\
        --new-class-name botnet

    # Multiple new classes (CSV has a label column with several new classes)
    python -m resnet.reptile_adapt \\
        --new-class-csv data/custom/new_attacks.csv \\
        --label-column label

    # Custom output / shots
    python -m resnet.reptile_adapt \\
        --new-class-csv data/custom/botnet_samples.csv \\
        --new-class-name botnet \\
        --shots 50 \\
        --output-dir artifacts/resnet_adapted_8class

Input CSV format
----------------
    The CSV must contain the same network flow features used by the base
    pipeline (duration, src_bytes, dst_bytes, proto, etc.).

    If --new-class-name is given, ALL rows in the CSV are treated as that class.
    Otherwise, the CSV must contain a label column (--label-column, default "label")
    with one or more new class names.

Output
------
    artifacts/resnet_adapted_<N>class/
        resnet_adapted_<N>class.keras       – adapted model (expanded head)
        preprocessing_pipeline.pkl          – copied from base
        final_features.txt                  – copied from base
        adaptation_config.json              – class list, metrics, params
        adaptation_report.txt               – classification report
        adaptation_confusion_matrix.png     – confusion matrix plot
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import shutil
import time
from collections import Counter

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from app.preprocessing import SafeLabelEncoder, transform_with_pipeline


# ── Defaults ──────────────────────────────────────────────────────────────────
SEED = 42

# Training hyperparameters
BATCH_SIZE = 128
FREEZE_EPOCHS = 8          # head-only warmup
MAX_EPOCHS = 70            # full fine-tune budget
LR_FROZEN = 1e-3
LR_FINETUNE = 1e-5
PRED_BATCH_SIZE = 2048

# Inner-loop (Reptile-style warmup before standard fine-tuning)
INNER_STEPS_WARMUP = 10
INNER_LR_WARMUP = 3e-4
INNER_CLIPNORM = 1.0

# Fine-tuning strategy
FINE_TUNE_UNFREEZE_LAST_N = 16
FREEZE_BATCH_NORM = False

# Callbacks
EARLY_STOPPING_PATIENCE = 10
LR_PLATEAU_PATIENCE = 3

# Default shots (samples per new class to use as support)
DEFAULT_SHOTS = 20

# Meta-model artifact candidates
META_MODEL_CANDIDATES = [
    os.path.join("artifacts", "resnet_reptile", "resnet_reptile_meta.keras"),
]

META_PIPELINE_CANDIDATES = [
    os.path.join("artifacts", "resnet_reptile", "preprocessing_pipeline.pkl"),
    os.path.join("artifacts", "resnet_base", "preprocessing_pipeline.pkl"),
]

META_CONFIG_CANDIDATES = [
    os.path.join("artifacts", "resnet_reptile", "reptile_meta_config.json"),
]


# ── Path helpers ──────────────────────────────────────────────────────────────
def _detect_project_root() -> str:
    env_root = os.environ.get("TON_IOT_PROJECT_ROOT")
    if env_root and os.path.isdir(env_root):
        return os.path.abspath(env_root)
    cwd = os.path.abspath(os.getcwd())
    if os.path.isdir(os.path.join(cwd, "data")) and os.path.isdir(os.path.join(cwd, "artifacts")):
        return cwd
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(script_dir, ".."))


def _resolve_first(project_root: str, candidates: list[str]) -> str:
    for rel in candidates:
        full = os.path.join(project_root, rel)
        if os.path.exists(full):
            return full
    raise FileNotFoundError(
        "None of the candidate paths exist:\n"
        + "\n".join(f"  - {os.path.join(project_root, c)}" for c in candidates)
    )


def _resolve_optional(project_root: str, candidates: list[str]) -> str | None:
    for rel in candidates:
        full = os.path.join(project_root, rel)
        if os.path.exists(full):
            return full
    return None


def _load_final_features(pipeline_dir: str, pipeline: dict) -> list[str]:
    txt_path = os.path.join(pipeline_dir, "final_features.txt")
    if os.path.exists(txt_path):
        with open(txt_path) as f:
            feats = [line.strip() for line in f if line.strip()]
        if feats:
            return feats
    feats = pipeline.get("features")
    if not feats:
        raise RuntimeError("Cannot determine final features.")
    return [str(feat).strip() for feat in feats if str(feat).strip()]


# ── Model building ───────────────────────────────────────────────────────────
def _build_expanded_model(
    meta_model_path: str,
    old_num_classes: int,
    new_num_classes: int,
) -> Model:
    """Load the meta-model and replace its classification head for more classes.

    The penultimate layer's features are reused; only the final Dense is replaced.
    Weights from the old head are transferred for existing classes.
    """
    base = tf.keras.models.load_model(meta_model_path, compile=False)

    if len(base.layers) < 2:
        raise ValueError("Model has too few layers to replace head.")

    # The last layer is the classification head
    old_head = base.layers[-1]
    old_weights, old_bias = old_head.get_weights()

    # Build feature extractor (everything except the last layer)
    feature_extractor = tf.keras.Model(inputs=base.input, outputs=base.layers[-2].output)
    x = feature_extractor.output

    new_head = Dense(
        new_num_classes, activation="softmax",
        name="adapted_head", dtype="float32",
    )(x)
    model = tf.keras.Model(inputs=feature_extractor.input, outputs=new_head)

    # Transfer old head weights for existing classes
    new_head_layer = model.get_layer("adapted_head")
    new_w, new_b = new_head_layer.get_weights()

    # Copy weights for the first old_num_classes columns
    n_copy = min(old_num_classes, new_num_classes)
    new_w[:, :n_copy] = old_weights[:, :n_copy]
    new_b[:n_copy] = old_bias[:n_copy]

    # Initialize new class weights with small random values
    if new_num_classes > old_num_classes:
        rng = np.random.default_rng(SEED)
        fan_in = new_w.shape[0]
        scale = np.sqrt(2.0 / fan_in)
        new_w[:, old_num_classes:] = rng.normal(0, scale, (fan_in, new_num_classes - old_num_classes)).astype(np.float32)
        new_b[old_num_classes:] = 0.0

    new_head_layer.set_weights([new_w, new_b])
    return model


def _freeze_backbone(model: Model) -> None:
    """Freeze all layers except the classification head."""
    head_name = "adapted_head"
    for layer in model.layers:
        layer.trainable = (layer.name == head_name)


def _unfreeze_tail(model: Model, last_n: int, freeze_bn: bool) -> int:
    """Unfreeze the last N backbone layers. Returns count unfrozen."""
    head_name = "adapted_head"
    backbone = [
        l for l in model.layers
        if l.name != head_name and not isinstance(l, tf.keras.layers.InputLayer)
    ]
    cutoff = max(len(backbone) - last_n, 0)
    unfrozen = 0
    for idx, layer in enumerate(backbone):
        should_train = idx >= cutoff
        if freeze_bn and isinstance(layer, BatchNormalization):
            should_train = False
        layer.trainable = should_train
        if should_train:
            unfrozen += 1
    model.get_layer(head_name).trainable = True
    return unfrozen


# ── Data loading ─────────────────────────────────────────────────────────────
def _load_new_class_data(
    csv_path: str,
    new_class_name: str | None,
    label_column: str,
) -> pd.DataFrame:
    """Load the new-class CSV and ensure it has a 'label' column."""
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip()

    df.drop(columns=["ts", "date", "time"], errors="ignore", inplace=True)

    if new_class_name:
        # All rows are the given class
        df["_adapt_label"] = new_class_name
    else:
        if label_column not in df.columns:
            raise ValueError(
                f"CSV does not contain label column '{label_column}'. "
                f"Available columns: {list(df.columns)}"
            )
        df["_adapt_label"] = df[label_column].astype(str).str.strip()
        df.drop(columns=[label_column], inplace=True, errors="ignore")

    # Drop IP columns
    ip_cols = [c for c in ["src_ip", "dst_ip", "srcip", "dstip"] if c in df.columns]
    if ip_cols:
        df.drop(columns=ip_cols, inplace=True)

    return df


def _load_existing_class_data(
    project_root: str,
    pipeline: dict,
    existing_classes: list[str],
    samples_per_class: int,
) -> pd.DataFrame:
    """Load a balanced subset of existing classes from the base TON-IoT dataset for rehearsal."""
    csv_candidates = [
        os.path.join(project_root, "data", "train_test_network.csv"),
        os.path.join(project_root, "data", "Network_dataset_capped.csv"),
        os.path.join(project_root, "data", "network_dataset_capped.csv"),
        os.path.join(project_root, "data", "custom", "tpot_finetune.csv"),
    ]
    csv_path = None
    for p in csv_candidates:
        if os.path.exists(p):
            csv_path = p
            break

    if csv_path is None:
        print("Warning: No base dataset found for rehearsal — training on new class data only.")
        return pd.DataFrame()

    print(f"Loading rehearsal data from: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False, dtype=str, on_bad_lines="skip")
    df.columns = df.columns.str.strip()
    df.drop(columns=["ts", "date", "time"], errors="ignore", inplace=True)

    # Find label column
    label_col = None
    for c in ("type", "label", "Label"):
        if c in df.columns:
            label_col = c
            break
    if label_col is None:
        return pd.DataFrame()

    # Merge dos+ddos if needed
    labels_norm = df[label_col].astype(str).str.strip().str.lower()
    if "dos_ddos" in existing_classes:
        dos_mask = labels_norm == "dos"
        ddos_mask = labels_norm == "ddos"
        df.loc[dos_mask, label_col] = "dos_ddos"
        df.loc[ddos_mask, label_col] = "dos_ddos"

    # Filter to existing classes only, sample evenly
    df = df[df[label_col].astype(str).isin(existing_classes)].copy()
    parts = []
    for cls in existing_classes:
        cls_df = df[df[label_col].astype(str) == cls]
        if len(cls_df) == 0:
            continue
        n = min(samples_per_class, len(cls_df))
        parts.append(cls_df.sample(n=n, random_state=SEED))
    if not parts:
        return pd.DataFrame()

    result = pd.concat(parts, ignore_index=True)
    result.rename(columns={label_col: "_adapt_label"}, inplace=True)

    # Drop the old label column if it's different
    if label_col != "_adapt_label" and label_col in result.columns:
        result.drop(columns=[label_col], inplace=True, errors="ignore")

    # Drop IPs
    ip_cols = [c for c in ["src_ip", "dst_ip", "srcip", "dstip"] if c in result.columns]
    if ip_cols:
        result.drop(columns=ip_cols, inplace=True)

    result = result.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return result


# ── Main ──────────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Adapt Reptile meta-weights to include new attack classes",
    )
    parser.add_argument(
        "--new-class-csv", required=True,
        help="Path to CSV containing new class samples",
    )
    parser.add_argument(
        "--new-class-name", default=None,
        help="Name for the new class (if all rows in CSV belong to one class). "
             "If omitted, the CSV must have a label column.",
    )
    parser.add_argument(
        "--label-column", default="label",
        help="Label column name in CSV (used when --new-class-name is not set)",
    )
    parser.add_argument(
        "--shots", type=int, default=DEFAULT_SHOTS,
        help=f"Max samples per new class to use (default: {DEFAULT_SHOTS}). "
             "Use 0 for all available samples.",
    )
    parser.add_argument(
        "--rehearsal-samples", type=int, default=200,
        help="Samples per existing class for catastrophic-forgetting rehearsal",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--freeze-epochs", type=int, default=FREEZE_EPOCHS,
    )
    parser.add_argument(
        "--max-epochs", type=int, default=MAX_EPOCHS,
    )
    parser.add_argument(
        "--unfreeze-last-n", type=int, default=FINE_TUNE_UNFREEZE_LAST_N,
    )
    args = parser.parse_args(argv)

    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)

    # GPU setup
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    project_root = _detect_project_root()

    print("=" * 60)
    print("  Reptile Few-Shot Adaptation — Add New Attack Classes")
    print("=" * 60)
    print(f"Project root:    {project_root}")
    print(f"New class CSV:   {args.new_class_csv}")
    print(f"Shots per class: {args.shots if args.shots > 0 else 'all'}")

    # ── Resolve artifacts ─────────────────────────────────────────────────────
    meta_model_path = _resolve_first(project_root, META_MODEL_CANDIDATES)
    meta_pipeline_path = _resolve_first(project_root, META_PIPELINE_CANDIDATES)
    meta_config_path = _resolve_optional(project_root, META_CONFIG_CANDIDATES)

    print(f"Meta model:      {meta_model_path}")
    print(f"Pipeline:        {meta_pipeline_path}")

    with open(meta_pipeline_path, "rb") as f:
        pipeline = pickle.load(f)
    final_features = _load_final_features(os.path.dirname(meta_pipeline_path), pipeline)
    print(f"Features:        {len(final_features)}")

    # Load meta-config to get existing classes
    target_encoder = pipeline.get("target_encoder")
    if target_encoder is None:
        raise RuntimeError("Pipeline missing target_encoder.")
    existing_classes = target_encoder.classes_.tolist()
    existing_num = len(existing_classes)
    print(f"Existing classes ({existing_num}): {existing_classes}")

    if meta_config_path:
        with open(meta_config_path) as f:
            meta_config = json.load(f)
        print(f"Meta-training val_acc: {meta_config.get('best_val_acc', '?')}")
    else:
        meta_config = {}

    # ── Load new class data ───────────────────────────────────────────────────
    csv_path = args.new_class_csv
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(project_root, csv_path)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"New class CSV not found: {csv_path}")

    new_df = _load_new_class_data(csv_path, args.new_class_name, args.label_column)
    new_class_names = sorted(new_df["_adapt_label"].unique().tolist())

    # Check for overlaps with existing classes
    overlaps = [c for c in new_class_names if c in existing_classes]
    if overlaps:
        print(f"Warning: These classes already exist and will be augmented: {overlaps}")

    truly_new = [c for c in new_class_names if c not in existing_classes]
    print(f"New classes: {truly_new}")
    print(f"New class sample counts: {dict(Counter(new_df['_adapt_label']))}")

    # Subsample to --shots per new class
    if args.shots > 0:
        parts = []
        for cls in new_class_names:
            cls_df = new_df[new_df["_adapt_label"] == cls]
            n = min(args.shots, len(cls_df))
            parts.append(cls_df.sample(n=n, random_state=SEED))
        new_df = pd.concat(parts, ignore_index=True)
        print(f"After shot-limit: {dict(Counter(new_df['_adapt_label']))}")

    # ── Load rehearsal data for existing classes ──────────────────────────────
    rehearsal_df = _load_existing_class_data(
        project_root, pipeline, existing_classes, args.rehearsal_samples,
    )
    if not rehearsal_df.empty:
        print(f"Rehearsal samples: {dict(Counter(rehearsal_df['_adapt_label']))}")

    # ── Combine and prepare ───────────────────────────────────────────────────
    if not rehearsal_df.empty:
        combined_df = pd.concat([rehearsal_df, new_df], ignore_index=True)
    else:
        combined_df = new_df.copy()
    if "ts" in combined_df.columns:
        combined_df.drop(columns=["ts"], inplace=True)
    combined_df = combined_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    all_classes = sorted(set(existing_classes + truly_new))
    total_classes = len(all_classes)
    print(f"\nExpanded class set ({total_classes}): {all_classes}")

    # Build label encoder for expanded classes
    le_adapt = LabelEncoder()
    le_adapt.fit(all_classes)

    y_all_str = combined_df["_adapt_label"].astype(str)
    x_all_df = combined_df.drop(columns=["_adapt_label"])

    # Filter out any labels not in the expanded class set
    valid_mask = y_all_str.isin(all_classes)
    if not valid_mask.all():
        dropped = int((~valid_mask).sum())
        print(f"Warning: Dropped {dropped} rows with labels not in expanded class set")
        x_all_df = x_all_df[valid_mask].copy()
        y_all_str = y_all_str[valid_mask].copy()

    # Random stratified split 80/20
    if "ts" in x_all_df.columns:
        x_all_df = x_all_df.drop(columns=["ts"])

    x_train_df, x_test_df, y_train_str, y_test_str = train_test_split(
        x_all_df, y_all_str, test_size=0.2, stratify=y_all_str, random_state=SEED,
    )

    y_train = le_adapt.transform(y_train_str)
    y_test = le_adapt.transform(y_test_str)
    print(f"Random split — Train: {len(x_train_df)}  Test: {len(x_test_df)}")

    del combined_df, x_all_df, y_all_str
    gc.collect()

    # Transform through base pipeline
    print("Transforming features through base pipeline...")
    x_train = transform_with_pipeline(
        x_train_df.to_dict(orient="records"),
        pipeline=pipeline, final_features=final_features,
    )
    x_test = transform_with_pipeline(
        x_test_df.to_dict(orient="records"),
        pipeline=pipeline, final_features=final_features,
    )

    del x_train_df, x_test_df
    gc.collect()

    print(f"Train: {x_train.shape[0]} samples, Test: {x_test.shape[0]} samples")

    # ── Determine output directory ────────────────────────────────────────────
    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.path.join(project_root, "artifacts", f"resnet_adapted_{total_classes}class")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output: {out_dir}")

    # ── Build expanded model ──────────────────────────────────────────────────
    print(f"\nBuilding expanded model: {existing_num} → {total_classes} classes")
    model = _build_expanded_model(meta_model_path, existing_num, total_classes)
    print(f"Model input dim: {model.input_shape[-1]}, output classes: {model.output_shape[-1]}")

    # ── Phase 1: Head-only warmup (frozen backbone) ───────────────────────────
    _freeze_backbone(model)
    model.compile(
        optimizer=Adam(learning_rate=LR_FROZEN),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks_p1 = [
        EarlyStopping(
            monitor="val_loss", mode="min",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", mode="min",
            factor=0.5, patience=LR_PLATEAU_PATIENCE, min_lr=1e-7,
        ),
    ]

    print(f"\nPhase 1: Training head ({args.freeze_epochs} epochs, backbone frozen)...")
    model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=args.freeze_epochs,
        batch_size=BATCH_SIZE,
        callbacks=callbacks_p1,
        verbose=1,
    )

    # ── Phase 2: Fine-tune tail of backbone ───────────────────────────────────
    unfrozen = _unfreeze_tail(model, args.unfreeze_last_n, FREEZE_BATCH_NORM)
    print(f"\nPhase 2: Fine-tuning (unfrozen {unfrozen} backbone layers)...")

    model.compile(
        optimizer=Adam(learning_rate=LR_FINETUNE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks_p2 = [
        EarlyStopping(
            monitor="val_loss", mode="min",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", mode="min",
            factor=0.5, patience=LR_PLATEAU_PATIENCE, min_lr=1e-7,
        ),
    ]

    model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=args.max_epochs,
        batch_size=BATCH_SIZE,
        callbacks=callbacks_p2,
        verbose=1,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\nEvaluating on test set...")
    test_probs = model.predict(x_test, batch_size=PRED_BATCH_SIZE)
    y_pred = np.argmax(test_probs, axis=1)
    class_names = le_adapt.classes_.tolist()

    report = classification_report(
        y_test, y_pred, target_names=class_names, zero_division=0, digits=4,
    )
    report_dict = classification_report(
        y_test, y_pred, target_names=class_names, zero_division=0, output_dict=True,
    )
    print(report)

    # ── Save artifacts ────────────────────────────────────────────────────────
    model_name = f"resnet_adapted_{total_classes}class.keras"
    model_path = os.path.join(out_dir, model_name)
    model.save(model_path)
    print(f"Saved model: {model_path}")

    # Classification report
    report_path = os.path.join(out_dir, "adaptation_report.txt")
    with open(report_path, "w") as f:
        f.write(f"=== Reptile Adaptation Report ===\n")
        f.write(f"Meta model: {meta_model_path}\n")
        f.write(f"New class CSV: {csv_path}\n")
        f.write(f"Existing classes: {existing_classes}\n")
        f.write(f"New classes: {truly_new}\n")
        f.write(f"All classes ({total_classes}): {class_names}\n")
        f.write(f"Shots per class: {args.shots}\n")
        f.write(f"Rehearsal samples: {args.rehearsal_samples}\n\n")
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=list(range(total_classes)))
    plt.figure(figsize=(max(8, total_classes), max(6, total_classes * 0.75)))
    sns.heatmap(
        cm, annot=True, fmt="d",
        xticklabels=class_names, yticklabels=class_names, cmap="Blues",
    )
    plt.title(f"Reptile Adaptation Confusion Matrix ({total_classes} classes)")
    plt.tight_layout()
    cm_path = os.path.join(out_dir, "adaptation_confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()

    # Copy pipeline files
    pipeline_dir = os.path.dirname(meta_pipeline_path)
    for fname in ("preprocessing_pipeline.pkl", "final_features.txt", "calibration.pkl"):
        src = os.path.join(pipeline_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out_dir, fname))
            print(f"Copied {fname} → {out_dir}")

    # Save adaptation config
    config = {
        "algorithm": "reptile_adaptation",
        "seed": SEED,
        "meta_model": meta_model_path,
        "meta_pipeline": meta_pipeline_path,
        "meta_config": meta_config,
        "new_class_csv": csv_path,
        "new_class_name": args.new_class_name,
        "existing_classes": existing_classes,
        "new_classes": truly_new,
        "all_classes": class_names,
        "total_classes": total_classes,
        "shots": args.shots,
        "rehearsal_samples": args.rehearsal_samples,
        "freeze_epochs": args.freeze_epochs,
        "max_epochs": args.max_epochs,
        "unfreeze_last_n": args.unfreeze_last_n,
        "lr_frozen": LR_FROZEN,
        "lr_finetune": LR_FINETUNE,
        "batch_size": BATCH_SIZE,
        "accuracy": round(float(report_dict.get("accuracy", 0.0)), 4),
        "macro_f1": round(float(report_dict.get("macro avg", {}).get("f1-score", 0.0)), 4),
        "weighted_f1": round(float(report_dict.get("weighted avg", {}).get("f1-score", 0.0)), 4),
        "per_class_f1": {
            cls: round(float(report_dict.get(cls, {}).get("f1-score", 0.0)), 4)
            for cls in class_names
        },
    }
    config_path = os.path.join(out_dir, "adaptation_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  DONE — {total_classes}-class adapted model")
    print(f"{'=' * 60}")
    print(f"Accuracy:    {config['accuracy']:.4f}")
    print(f"Macro F1:    {config['macro_f1']:.4f}")
    print(f"Weighted F1: {config['weighted_f1']:.4f}")
    for cls in truly_new:
        f1 = config["per_class_f1"].get(cls, 0.0)
        print(f"  {cls} F1: {f1:.4f}")
    print(f"\nArtifacts: {out_dir}")
    print(f"\nTo deploy this model, update your ids-api.service:")
    print(f'  Environment="TON_IOT_ARTIFACT_DIR={out_dir}"')
    print(f'  Environment="TON_IOT_MODEL_FILENAME={model_name}"')


if __name__ == "__main__":
    main()
