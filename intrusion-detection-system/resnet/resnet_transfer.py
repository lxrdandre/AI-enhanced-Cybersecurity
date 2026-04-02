import os
import gc
import pickle
from collections import Counter

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import mixed_precision

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

import seaborn as sns
import matplotlib.pyplot as plt

from app.preprocessing import SafeLabelEncoder, transform_with_pipeline


# ── Configuration ─────────────────────────────────────────────────────────────
SEED = 42

# 8 raw classes in the custom dataset (1k samples each); dos+ddos merged into dos_ddos
TARGET_CLASSES_RAW = [
    "backdoor", "ddos", "dos", "injection",
    "normal", "password", "scanning", "xss",
]
MERGED_LABEL = "dos_ddos"
# After merge + subsample: 7 classes (dos+ddos → dos_ddos)

# Training hyperparameters (H200-tuned)
BATCH_SIZE = 256                  # smaller batches for fine-tuning stability
FREEZE_EPOCHS = 8                 # head-only warmup
MAX_EPOCHS = 70                   # full fine-tune budget
LR_FROZEN = 1e-3
LR_FINETUNE = 1e-5
PRED_BATCH_SIZE = 2048

# Loss (matching base model)
USE_FOCAL_LOSS = True
FOCAL_GAMMA = 2.0
FOCAL_ALPHA_CLIP = (0.5, 5.0)

# SMOTE
SMOTE_MAX_MULTIPLIER = 2          # cap synthetic oversampling

# Difficult-class weight boosts (applied on top of "balanced" weights)
DIFFICULT_CLASS_BOOSTS: dict[str, float] = {
    "injection": 1.15,
    "xss": 1.10,
    "normal": 1.10,
}

# Fine-tuning strategy
FINE_TUNE_UNFREEZE_LAST_N = 16   # was 8 — more capacity to adapt to new distribution
FREEZE_BATCH_NORM = False         # let BN adapt during phase-2 fine-tuning

# Callbacks
EARLY_STOPPING_PATIENCE = 12
LR_PLATEAU_PATIENCE = 4

# GPU (H200)
USE_MIXED_PRECISION = True
ENABLE_TF32 = True

# Paths (relative to project root)
CUSTOM_CSV_REL = os.path.join("data", "custom", "tpot_finetune.csv")
OUTPUT_DIR_REL = os.path.join("artifacts", "resnet_transfer_7class")

# Model + pipeline are always resolved as a PAIR from the same directory.
# This prevents the silent mismatch bug from the old independent fallback.
ARTIFACT_PAIRS = [
    (
        os.path.join("artifacts", "resnet_base", "resnet_model.keras"),
        os.path.join("artifacts", "resnet_base", "preprocessing_pipeline.pkl"),
    ),
    (
        os.path.join("artifacts", "ton_iot_sedwnet_v13_focalfix_smote_ransomware", "ton_iot_resnet_attn_focalfix_mha.keras"),
        os.path.join("artifacts", "ton_iot_sedwnet_v13_focalfix_smote_ransomware", "pipeline_objects.pkl"),
    ),
    (
        os.path.join("artifacts", "ton_iot_sedwnet_v13_focalfix", "ton_iot_resnet_attn_focalfix_mha.keras"),
        os.path.join("artifacts", "ton_iot_sedwnet_v13_focalfix", "pipeline_objects.pkl"),
    ),
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


def _resolve_artifact_pair(project_root: str) -> tuple[str, str]:
    """Find the first existing model+pipeline pair. Always from the same directory."""
    for model_rel, pipeline_rel in ARTIFACT_PAIRS:
        model_path = os.path.join(project_root, model_rel)
        pipeline_path = os.path.join(project_root, pipeline_rel)
        if os.path.exists(model_path) and os.path.exists(pipeline_path):
            return model_path, pipeline_path
    tried = [f"  - {m} + {p}" for m, p in ARTIFACT_PAIRS]
    raise FileNotFoundError(
        "No matching model+pipeline pair found. Tried:\n" + "\n".join(tried)
    )


def _identify_label_column(df: pd.DataFrame) -> str:
    for c in ("label", "Label", "type"):
        if c in df.columns:
            return c
    raise RuntimeError("Could not find label column (expected 'label', 'Label', or 'type').")


def _load_final_features(pipeline_dir: str, pipeline: dict) -> list[str]:
    """Load feature list from final_features.txt or pipeline dict."""
    txt_path = os.path.join(pipeline_dir, "final_features.txt")
    if os.path.exists(txt_path):
        with open(txt_path) as f:
            feats = [line.strip() for line in f if line.strip()]
        if feats:
            return feats
    feats = pipeline.get("features")
    if not feats:
        raise RuntimeError("Cannot determine final features (no txt file or pipeline['features']).")
    return [str(feat).strip() for feat in feats if str(feat).strip()]


# ── Model builders ────────────────────────────────────────────────────────────
def _build_transfer_model(base_model_path: str, num_classes: int) -> Model:
    """Replace classification head of pretrained model."""
    base = tf.keras.models.load_model(base_model_path, compile=False)
    if len(base.layers) < 2:
        raise ValueError("Base model has too few layers to replace head")
    feature_extractor = tf.keras.Model(inputs=base.input, outputs=base.layers[-2].output)
    x = feature_extractor.output
    out = Dense(num_classes, activation="softmax", name="transfer_head", dtype="float32")(x)
    return tf.keras.Model(inputs=feature_extractor.input, outputs=out)


def _freeze_backbone(model: Model) -> None:
    """Freeze all layers except the transfer head."""
    for layer in model.layers:
        layer.trainable = (layer.name == "transfer_head")


def _unfreeze_tail(model: Model, last_n: int, freeze_bn: bool) -> int:
    """Unfreeze last N backbone layers (excluding head and input). Returns count unfrozen."""
    backbone = [
        layer for layer in model.layers
        if layer.name != "transfer_head" and not isinstance(layer, tf.keras.layers.InputLayer)
    ]
    cutoff = max(len(backbone) - last_n, 0)
    unfrozen = 0
    for idx, layer in enumerate(backbone):
        should_train = idx >= cutoff
        if freeze_bn and isinstance(layer, tf.keras.layers.BatchNormalization):
            should_train = False
        layer.trainable = should_train
        if should_train:
            unfrozen += 1
    model.get_layer("transfer_head").trainable = True
    return unfrozen


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)

    # GPU setup (H200)
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    if ENABLE_TF32:
        tf.config.experimental.enable_tensor_float_32_execution(True)
    if USE_MIXED_PRECISION:
        mixed_precision.set_global_policy("mixed_float16")

    project_root = _detect_project_root()

    # Resolve paths
    custom_csv = os.path.join(project_root, CUSTOM_CSV_REL)
    if not os.path.exists(custom_csv):
        raise FileNotFoundError(f"Custom dataset not found: {custom_csv}")

    base_model_path, base_pipeline_path = _resolve_artifact_pair(project_root)
    out_dir = os.path.join(project_root, OUTPUT_DIR_REL)
    os.makedirs(out_dir, exist_ok=True)

    print("=== Transfer Learning: SE-DWNet/ResNet ===")
    print(f"Base model:   {base_model_path}")
    print(f"Pipeline:     {base_pipeline_path}")
    print(f"Dataset:      {custom_csv}")
    print(f"Output:       {out_dir}")
    print(f"Mixed prec:   {mixed_precision.global_policy().name}")

    # Load pipeline
    with open(base_pipeline_path, "rb") as f:
        pipeline = pickle.load(f)
    final_features = _load_final_features(os.path.dirname(base_pipeline_path), pipeline)
    print(f"Feature set: {len(final_features)} features")

    # ── Load and filter dataset ───────────────────────────────────────────────
    df = pd.read_csv(custom_csv, low_memory=False)
    df.columns = df.columns.str.strip()
    df.drop(columns=["ts", "date", "time"], errors="ignore", inplace=True)

    label_col = _identify_label_column(df)
    df[label_col] = df[label_col].astype(str).str.strip()

    # Keep only target classes
    df = df[df[label_col].isin(TARGET_CLASSES_RAW)].copy()
    if df.empty:
        raise RuntimeError(f"No rows match TARGET_CLASSES_RAW after filtering.")

    # Merge dos+ddos -> dos_ddos, then subsample to keep all classes at equal size
    labels_lower = df[label_col].str.lower()
    dos_df = df.loc[labels_lower == "dos"]
    ddos_df = df.loc[labels_lower == "ddos"]
    other_df = df.loc[~labels_lower.isin({"dos", "ddos"})]

    combined = pd.concat([dos_df, ddos_df], ignore_index=True).copy()
    # Subsample merged class to match per-class count of remaining classes
    per_class_n = int(other_df.groupby(label_col).size().min()) if len(other_df) > 0 else len(combined)
    dos_ddos_df = combined.sample(n=min(per_class_n, len(combined)), random_state=SEED)
    dos_ddos_df = dos_ddos_df.copy()
    dos_ddos_df[label_col] = MERGED_LABEL

    df = pd.concat([other_df, dos_ddos_df], ignore_index=True)
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    print(f"Merged dos({len(dos_df)})+ddos({len(ddos_df)}) -> {MERGED_LABEL}({len(dos_ddos_df)})")

    print(f"Class counts: {dict(Counter(df[label_col]))}")

    # Drop IPs
    ip_cols = [c for c in ["src_ip", "dst_ip", "srcip", "dstip"] if c in df.columns]
    if ip_cols:
        df.drop(columns=ip_cols, inplace=True)

    # ── Split 60/20/20 ───────────────────────────────────────────────────────
    y_all = df[label_col].astype(str)
    X_all = df.drop(columns=[label_col])

    X_train_df, X_temp, y_train_str, y_temp = train_test_split(
        X_all, y_all, test_size=0.4, stratify=y_all, random_state=SEED,
    )
    X_val_df, X_test_df, y_val_str, y_test_str = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED,
    )

    del df, X_all, y_all, X_temp, y_temp
    gc.collect()

    # Encode labels (train-only fit)
    le_target = LabelEncoder()
    le_target.fit(y_train_str)
    y_train = le_target.transform(y_train_str)
    y_val = le_target.transform(y_val_str)
    y_test = le_target.transform(y_test_str)
    class_names = le_target.classes_.tolist()
    print(f"{len(class_names)}-class mapping: {class_names}")

    # Transform features through base pipeline
    X_train = transform_with_pipeline(X_train_df.to_dict(orient="records"),
                                      pipeline=pipeline, final_features=final_features)
    X_val = transform_with_pipeline(X_val_df.to_dict(orient="records"),
                                    pipeline=pipeline, final_features=final_features)
    X_test = transform_with_pipeline(X_test_df.to_dict(orient="records"),
                                     pipeline=pipeline, final_features=final_features)

    del X_train_df, X_val_df, X_test_df
    gc.collect()

    # ── SMOTE oversampling ─────────────────────────────────────────────────
    print("Applying SMOTE for class balancing...")
    train_counts = Counter(y_train)
    print(f"Pre-SMOTE:  {dict(train_counts)}")

    max_class_count = max(train_counts.values())
    smote_strategy = {}
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
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"Post-SMOTE: {dict(Counter(y_train))}")
    else:
        print("SMOTE skipped (no eligible classes)")

    # Compute class weights for balanced training + difficult-class boosts
    unique_classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=unique_classes, y=y_train)
    class_weight_dict = dict(zip(unique_classes.tolist(), weights.tolist()))

    for label, boost in DIFFICULT_CLASS_BOOSTS.items():
        if label in class_names:
            idx = class_names.index(label)
            if idx in class_weight_dict:
                class_weight_dict[idx] *= boost

    print(f"Class weights: { {class_names[k]: f'{v:.2f}' for k, v in class_weight_dict.items()} }")

    # ── Loss function (matching base model) ───────────────────────────────────
    if USE_FOCAL_LOSS:
        counts = np.bincount(y_train, minlength=len(class_names)).astype(np.float32)
        inv = 1.0 / np.maximum(counts, 1.0)
        alpha_vec = inv / inv.mean()
        alpha_vec = np.clip(alpha_vec, *FOCAL_ALPHA_CLIP).astype(np.float32)
        print(f"Focal alpha: min={alpha_vec.min():.4f}, mean={alpha_vec.mean():.4f}, max={alpha_vec.max():.4f}")
        loss_fn = tf.keras.losses.CategoricalFocalCrossentropy(
            alpha=alpha_vec.tolist(), gamma=FOCAL_GAMMA, from_logits=False,
        )
        # Focal loss requires one-hot targets
        from tensorflow.keras.utils import to_categorical
        y_train_cat = to_categorical(y_train, num_classes=len(class_names)).astype(np.float32)
        y_val_cat = to_categorical(y_val, num_classes=len(class_names)).astype(np.float32)
        y_test_cat = to_categorical(y_test, num_classes=len(class_names)).astype(np.float32)
    else:
        loss_fn = "sparse_categorical_crossentropy"
        y_train_cat, y_val_cat, y_test_cat = y_train, y_val, y_test

    # ── Build transfer model ──────────────────────────────────────────────────
    model = _build_transfer_model(base_model_path, num_classes=len(class_names))

    # Phase 1: train head only (frozen backbone)
    _freeze_backbone(model)
    model.compile(
        optimizer=Adam(learning_rate=LR_FROZEN),
        loss=loss_fn,
        metrics=["accuracy"],
    )

    callbacks_p1 = [
        EarlyStopping(
            monitor="val_loss", mode="min",
            patience=EARLY_STOPPING_PATIENCE, min_delta=1e-4,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", mode="min",
            factor=0.5, patience=LR_PLATEAU_PATIENCE, min_lr=1e-7,
        ),
    ]

    print("Phase 1: Training head (frozen backbone)...")
    model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=FREEZE_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=callbacks_p1,
        verbose=1,
    )

    # Phase 2: fine-tune tail of backbone
    unfrozen = _unfreeze_tail(model, FINE_TUNE_UNFREEZE_LAST_N, FREEZE_BATCH_NORM)
    print(f"Phase 2: Fine-tuning (unfrozen {unfrozen} backbone layers, BN frozen={FREEZE_BATCH_NORM})...")

    model.compile(
        optimizer=Adam(learning_rate=LR_FINETUNE),
        loss=loss_fn,
        metrics=["accuracy"],
    )

    callbacks_p2 = [
        EarlyStopping(
            monitor="val_loss", mode="min",
            patience=EARLY_STOPPING_PATIENCE, min_delta=1e-4,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", mode="min",
            factor=0.5, patience=LR_PLATEAU_PATIENCE, min_lr=1e-7,
        ),
    ]

    model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=callbacks_p2,
        verbose=1,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("Evaluating on test set...")
    test_probs = model.predict(X_test, batch_size=PRED_BATCH_SIZE)
    y_pred = np.argmax(test_probs, axis=1)

    # y_test_cat may be one-hot (focal) or integer (sparse) — normalise to int
    y_test_int = np.argmax(y_test_cat, axis=1) if USE_FOCAL_LOSS else y_test

    report = classification_report(y_test_int, y_pred, target_names=class_names, zero_division=0, digits=4)
    print(report)

    suffix = "7class"

    with open(os.path.join(out_dir, f"transfer_classification_report_{suffix}.txt"), "w") as f:
        f.write(f"=== ResNet Transfer Learning ({suffix}) ===\n")
        f.write(f"Base model: {base_model_path}\n")
        f.write(f"Pipeline: {base_pipeline_path}\n")
        f.write(f"Dataset: {custom_csv}\n")
        f.write(f"Classes: {class_names}\n\n")
        f.write(report)

    cm = confusion_matrix(y_test_int, y_pred, labels=list(range(len(class_names))))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names,
                yticklabels=class_names, cmap="Blues")
    plt.title("ResNet Transfer Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"resnet_transfer_cm_{suffix}.png"), dpi=200)
    plt.close()

    # Save model + metadata
    model.save(os.path.join(out_dir, f"resnet_transfer_model_{suffix}.keras"))
    with open(os.path.join(out_dir, "transfer_training_metadata.pkl"), "wb") as f:
        pickle.dump(
            {
                "base_model": base_model_path,
                "base_pipeline": base_pipeline_path,
                "custom_dataset": custom_csv,
                "classes": class_names,
                "merged_label": MERGED_LABEL,
                "label_encoder": le_target,
            },
            f,
        )

    # Copy base pipeline + features into transfer output dir so the API can
    # load everything from a single artifact directory.
    import shutil
    pipeline_dir = os.path.dirname(base_pipeline_path)
    for fname in ("preprocessing_pipeline.pkl", "final_features.txt", "calibration.pkl"):
        src = os.path.join(pipeline_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out_dir, fname))
            print(f"Copied {fname} → {out_dir}")

    print(f"DONE. Artifacts written to: {out_dir}")


if __name__ == "__main__":
    main()
