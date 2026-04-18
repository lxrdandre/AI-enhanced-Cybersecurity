from __future__ import annotations

import os
import pickle
import shutil
import sys
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.normpath(os.path.join(SCRIPT_DIR, "..")))

from app.preprocessing import SafeLabelEncoder, transform_with_pipeline

_PICKLE_COMPAT = SafeLabelEncoder


SEED = 42
CLASSES = ["backdoor", "dos_ddos", "injection", "normal", "password", "scanning", "xss"]

BASE_MODEL_REL = os.path.join("artifacts", "resnet_base", "resnet_model.keras")
BASE_PIPELINE_REL = os.path.join("artifacts", "resnet_base", "preprocessing_pipeline.pkl")
CUSTOM_CSV_REL = os.path.join("data", "custom", "tpot_finetune.csv")
ORIGINAL_CSV_REL = os.path.join("data", "Network_dataset_capped.csv")
OUTPUT_DIR_REL = os.path.join("artifacts", "resnet_transfer_7class")

BATCH_SIZE = 256
FREEZE_EPOCHS = 8
MAX_EPOCHS = 70
LR_FROZEN = 1e-3
LR_FINETUNE = 1e-5
PRED_BATCH_SIZE = 2048

USE_FOCAL_LOSS = True
FOCAL_GAMMA = 2.0
FOCAL_ALPHA_CLIP = (0.5, 5.0)
SMOTE_MAX_MULTIPLIER = 2
DIFFICULT_CLASS_BOOSTS = {"injection": 1.15, "xss": 1.10, "normal": 1.10}

FINE_TUNE_UNFREEZE_LAST_N = 16
FREEZE_BATCH_NORM = False
EARLY_STOPPING_PATIENCE = 12
LR_PLATEAU_PATIENCE = 4

USE_MIXED_PRECISION = True
ENABLE_TF32 = True

ROUTER_SAMPLES_PER_DOMAIN = 50_000
ROUTER_THRESHOLD = 0.60
ROUTE_FIELDS = ("domain", "_domain", "source_domain", "_source", "source")

DROP_COLS = [
    "ts",
    "date",
    "time",
    "type",
    "label",
    "Label",
    "attack",
    "category",
    "src_ip",
    "dst_ip",
    "srcip",
    "dstip",
]


def project_root() -> str:
    env_root = os.environ.get("TON_IOT_PROJECT_ROOT")
    if env_root and os.path.isdir(env_root):
        return os.path.abspath(env_root)
    cwd = os.path.abspath(os.getcwd())
    return cwd if os.path.isdir(os.path.join(cwd, "data")) else os.path.dirname(SCRIPT_DIR)


def paths(root: str) -> tuple[str, str, str, str, str]:
    base_model = os.path.join(root, BASE_MODEL_REL)
    base_pipeline = os.path.join(root, BASE_PIPELINE_REL)
    custom_csv = os.path.join(root, CUSTOM_CSV_REL)
    original_csv = os.path.join(root, ORIGINAL_CSV_REL)
    out_dir = os.path.join(root, OUTPUT_DIR_REL)
    missing = [p for p in (base_model, base_pipeline, custom_csv) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("Missing required file(s):\n" + "\n".join(missing))
    os.makedirs(out_dir, exist_ok=True)
    clean_output_dir(out_dir)
    return base_model, base_pipeline, custom_csv, original_csv, out_dir


def clean_output_dir(out_dir: str) -> None:
    keep = {
        "resnet_transfer_model_7class.keras",
        "preprocessing_pipeline.pkl",
        "final_features.txt",
        "domain_router.pkl",
        "transfer_training_metadata.pkl",
        "transfer_classification_report_7class.txt",
        "resnet_transfer_cm_7class.png",
    }
    for name in os.listdir(out_dir):
        path = os.path.join(out_dir, name)
        if os.path.isfile(path) and name not in keep:
            os.remove(path)


def canon(label: object) -> str:
    label = str(label).strip().lower()
    return "dos_ddos" if label in {"dos", "ddos", "ddos_dos"} else label


def label_column(df: pd.DataFrame) -> str:
    for col in ("type", "attack", "category", "label", "Label"):
        if col not in df.columns:
            continue
        values = set(df[col].dropna().astype(str).head(50_000).map(canon).unique())
        if values.intersection(CLASSES):
            return col
    raise RuntimeError(f"Could not identify label column. Columns: {list(df.columns[:30])}")


def pipeline_classes(pipeline: dict) -> list[str]:
    encoder = pipeline.get("target_encoder")
    if encoder is not None and hasattr(encoder, "classes_"):
        classes = [canon(x) for x in encoder.classes_.tolist()]
        if sorted(classes) == sorted(CLASSES):
            return classes
    return CLASSES


def final_features(pipeline_dir: str, pipeline: dict) -> list[str]:
    txt_path = os.path.join(pipeline_dir, "final_features.txt")
    if os.path.exists(txt_path):
        with open(txt_path) as f:
            features = [line.strip() for line in f if line.strip()]
        if features:
            return features
    features = [str(x).strip() for x in pipeline.get("features", []) if str(x).strip()]
    if not features:
        raise RuntimeError("Cannot determine final features.")
    return features


def read_labeled_csv(path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path, dtype=str, low_memory=False, on_bad_lines="skip")
    df.columns = df.columns.str.strip()
    lab = label_column(df)
    y = df[lab].map(canon)
    df = df[y.isin(CLASSES)].copy()
    y = y[y.isin(CLASSES)].reset_index(drop=True)
    x = df.drop(columns=DROP_COLS, errors="ignore").reset_index(drop=True)
    return x, y


def load_custom(path: str) -> tuple[pd.DataFrame, pd.Series]:
    x, y = read_labeled_csv(path)
    per_class = int(y.value_counts().min())
    idx = pd.concat(
        [group.sample(n=per_class, random_state=SEED) for _, group in y.groupby(y, sort=True)],
    ).index
    x, y = x.loc[idx].reset_index(drop=True), y.loc[idx].reset_index(drop=True)
    order = np.random.default_rng(SEED).permutation(len(y))
    print(f"Custom rows: {dict(Counter(y))}")
    return x.iloc[order].reset_index(drop=True), y.iloc[order].reset_index(drop=True)


def encode(classes: list[str]) -> LabelEncoder:
    encoder = LabelEncoder()
    encoder.classes_ = np.array(classes)
    return encoder


def build_custom_model(base_model_path: str, num_classes: int) -> tf.keras.Model:
    base = tf.keras.models.load_model(base_model_path, compile=False)
    features = tf.keras.Model(base.input, base.layers[-2].output, name="custom_backbone")
    out = Dense(num_classes, activation="softmax", name="custom_head", dtype="float32")(features.output)
    return tf.keras.Model(features.input, out, name="custom_expert")


def freeze_custom_head_only(model: tf.keras.Model) -> None:
    for layer in model.layers:
        layer.trainable = layer.name == "custom_head"


def unfreeze_custom_tail(model: tf.keras.Model) -> int:
    backbone = [
        layer for layer in model.layers
        if layer.name != "custom_head" and not isinstance(layer, tf.keras.layers.InputLayer)
    ]
    cutoff = max(len(backbone) - FINE_TUNE_UNFREEZE_LAST_N, 0)
    unfrozen = 0
    for idx, layer in enumerate(backbone):
        trainable = idx >= cutoff and not (FREEZE_BATCH_NORM and isinstance(layer, tf.keras.layers.BatchNormalization))
        layer.trainable = trainable
        unfrozen += int(trainable)
    model.get_layer("custom_head").trainable = True
    return unfrozen


def callbacks() -> list[tf.keras.callbacks.Callback]:
    return [
        EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, min_delta=1e-4, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=LR_PLATEAU_PATIENCE, min_lr=1e-7),
    ]


def train_router(
    *,
    original_csv: str,
    custom_csv: str,
    pipeline: dict,
    features: list[str],
    out_dir: str,
) -> None:
    if not os.path.exists(original_csv):
        print(f"Router skipped: original dataset not found: {original_csv}")
        return

    original_x, _ = read_labeled_csv(original_csv)
    custom_x, _ = read_labeled_csv(custom_csv)
    n = min(ROUTER_SAMPLES_PER_DOMAIN, len(original_x), len(custom_x))
    original_x = original_x.sample(n=n, random_state=SEED)
    custom_x = custom_x.sample(n=n, random_state=SEED)

    x_original = transform_with_pipeline(original_x.to_dict("records"), pipeline=pipeline, final_features=features)
    x_custom = transform_with_pipeline(custom_x.to_dict("records"), pipeline=pipeline, final_features=features)
    x = np.vstack([x_original, x_custom])
    y = np.concatenate([np.zeros(len(x_original), dtype=np.int8), np.ones(len(x_custom), dtype=np.int8)])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=SEED)
    router = LogisticRegression(max_iter=500, class_weight="balanced", n_jobs=-1)
    router.fit(x_train, y_train)
    acc = float(router.score(x_test, y_test))

    with open(os.path.join(out_dir, "domain_router.pkl"), "wb") as f:
        pickle.dump(
            {
                "model": router,
                "threshold": ROUTER_THRESHOLD,
                "classes": {0: "original", 1: "custom"},
                "route_fields": ROUTE_FIELDS,
                "router_accuracy": acc,
            },
            f,
        )
    print(f"Router accuracy: {acc:.4f} on held-out original-vs-custom split")


def copy_support_files(base_pipeline_path: str, out_dir: str) -> None:
    pipeline_dir = os.path.dirname(base_pipeline_path)
    for fname in ("preprocessing_pipeline.pkl", "final_features.txt"):
        src = os.path.join(pipeline_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out_dir, fname))
            print(f"Copied {fname} -> {out_dir}")


def main() -> None:
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    if ENABLE_TF32:
        tf.config.experimental.enable_tensor_float_32_execution(True)
    if USE_MIXED_PRECISION:
        mixed_precision.set_global_policy("mixed_float16")

    root = project_root()
    base_model_path, base_pipeline_path, custom_csv, original_csv, out_dir = paths(root)
    print("=== Custom Expert Transfer Learning + Domain Router ===")
    print(f"Base model: {base_model_path}")
    print(f"Custom data: {custom_csv}")
    print(f"Output: {out_dir}")

    with open(base_pipeline_path, "rb") as f:
        pipeline = pickle.load(f)
    features = final_features(os.path.dirname(base_pipeline_path), pipeline)
    class_names = pipeline_classes(pipeline)
    encoder = encode(class_names)
    print(f"Classes: {class_names}")
    print(f"Features: {len(features)}")

    x_df, y_str = load_custom(custom_csv)
    x_train_df, x_temp_df, y_train_str, y_temp_str = train_test_split(
        x_df, y_str, test_size=0.4, stratify=y_str, random_state=SEED,
    )
    x_val_df, x_test_df, y_val_str, y_test_str = train_test_split(
        x_temp_df, y_temp_str, test_size=0.5, stratify=y_temp_str, random_state=SEED,
    )

    y_train = encoder.transform(y_train_str)
    y_val = encoder.transform(y_val_str)
    y_test = encoder.transform(y_test_str)
    x_train = transform_with_pipeline(x_train_df.to_dict("records"), pipeline=pipeline, final_features=features)
    x_val = transform_with_pipeline(x_val_df.to_dict("records"), pipeline=pipeline, final_features=features)
    x_test = transform_with_pipeline(x_test_df.to_dict("records"), pipeline=pipeline, final_features=features)

    print("Applying SMOTE for custom class balancing...")
    counts = Counter(y_train)
    max_count = max(counts.values())
    strategy = {
        cls: min(int(count * SMOTE_MAX_MULTIPLIER), max_count)
        for cls, count in counts.items()
        if count > 1 and min(int(count * SMOTE_MAX_MULTIPLIER), max_count) > count
    }
    if strategy:
        k_neighbors = max(1, min(5, min(counts[c] for c in strategy) - 1))
        x_train, y_train = SMOTE(sampling_strategy=strategy, random_state=SEED, k_neighbors=k_neighbors).fit_resample(x_train, y_train)
    print(f"Train rows after SMOTE: {dict(Counter(y_train))}")

    weights = dict(zip(np.unique(y_train), compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)))
    for label, boost in DIFFICULT_CLASS_BOOSTS.items():
        if label in class_names:
            weights[class_names.index(label)] *= boost
    print(f"Class weights: { {class_names[k]: f'{v:.2f}' for k, v in weights.items()} }")

    if USE_FOCAL_LOSS:
        class_counts = np.bincount(y_train, minlength=len(class_names)).astype(np.float32)
        alpha = np.clip((1.0 / np.maximum(class_counts, 1.0)) / (1.0 / np.maximum(class_counts, 1.0)).mean(), *FOCAL_ALPHA_CLIP)
        loss = tf.keras.losses.CategoricalFocalCrossentropy(alpha=alpha.tolist(), gamma=FOCAL_GAMMA)
        y_train_fit = to_categorical(y_train, len(class_names)).astype(np.float32)
        y_val_fit = to_categorical(y_val, len(class_names)).astype(np.float32)
    else:
        loss, y_train_fit, y_val_fit = "sparse_categorical_crossentropy", y_train, y_val

    custom_model = build_custom_model(base_model_path, len(class_names))
    freeze_custom_head_only(custom_model)
    custom_model.compile(optimizer=Adam(LR_FROZEN), loss=loss, metrics=["accuracy"])
    print("Phase 1: custom head only")
    custom_model.fit(
        x_train,
        y_train_fit,
        validation_data=(x_val, y_val_fit),
        epochs=FREEZE_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=weights,
        callbacks=callbacks(),
        verbose=1,
    )

    unfrozen = unfreeze_custom_tail(custom_model)
    print(f"Phase 2: custom expert fine-tune, unfrozen_backbone_layers={unfrozen}, freeze_bn={FREEZE_BATCH_NORM}")
    custom_model.compile(optimizer=Adam(LR_FINETUNE), loss=loss, metrics=["accuracy"])
    custom_model.fit(
        x_train,
        y_train_fit,
        validation_data=(x_val, y_val_fit),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=weights,
        callbacks=callbacks(),
        verbose=1,
    )

    probs = custom_model.predict(x_test, batch_size=PRED_BATCH_SIZE, verbose=0)
    pred = np.argmax(probs, axis=1)
    report = classification_report(y_test, pred, target_names=class_names, zero_division=0, digits=4)
    print(report)

    suffix = "7class"
    with open(os.path.join(out_dir, f"transfer_classification_report_{suffix}.txt"), "w") as f:
        f.write("=== Custom expert test ===\n")
        f.write(f"Base model: {base_model_path}\n")
        f.write(f"Custom data: {custom_csv}\n")
        f.write(f"Classes: {class_names}\n\n")
        f.write(report)

    cm = confusion_matrix(y_test, pred, labels=list(range(len(class_names))))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.title("Custom Expert Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"resnet_transfer_cm_{suffix}.png"), dpi=200)
    plt.close()

    train_router(
        original_csv=original_csv,
        custom_csv=custom_csv,
        pipeline=pipeline,
        features=features,
        out_dir=out_dir,
    )

    custom_model.save(os.path.join(out_dir, f"resnet_transfer_model_{suffix}.keras"))
    with open(os.path.join(out_dir, "transfer_training_metadata.pkl"), "wb") as f:
        pickle.dump(
            {
                "base_model": base_model_path,
                "base_pipeline": base_pipeline_path,
                "custom_dataset": custom_csv,
                "classes": class_names,
                "routing": "API loads artifacts/resnet_base as original expert and this model as custom expert.",
            },
            f,
        )
    copy_support_files(base_pipeline_path, out_dir)
    print(f"DONE. Artifacts written to: {out_dir}")


if __name__ == "__main__":
    main()
