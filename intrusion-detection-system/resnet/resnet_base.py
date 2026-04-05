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
import pickle
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
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

import seaborn as sns
import matplotlib.pyplot as plt

from app.preprocessing import SafeLabelEncoder


# ── Configuration ─────────────────────────────────────────────────────────────
TARGET_K = 25
BATCH_SIZE = 1024               # H200-friendly large batch
MAX_EPOCHS = 100
SEED = 42

USE_FOCAL_LOSS = True
FOCAL_GAMMA = 2.0
FOCAL_ALPHA_CLIP = (0.5, 5.0)  # clip per-class alpha to keep training stable

SMOTE_MAX_MULTIPLIER = 2       # cap synthetic oversampling (dataset already fairly balanced)

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
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


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


PROJECT_ROOT = _detect_project_root()
ARTIFACT_DIR = os.path.normpath(os.path.join(PROJECT_ROOT, "artifacts", "resnet_base"))
os.makedirs(ARTIFACT_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == "float64":
            df[col] = df[col].astype("float32")
        if df[col].dtype == "int64":
            df[col] = df[col].astype("int32")
    return df


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

print("=== TON_IoT SE-DWNet Base Training ===")
print(f"GPUs available: {len(tf.config.list_physical_devices('GPU'))}")
print(f"Project root: {PROJECT_ROOT}")
print(f"Artifacts:    {ARTIFACT_DIR}")


# ── Step 1: Load Data ────────────────────────────────────────────────────────
csv_candidates = [
    os.path.join(PROJECT_ROOT, "data", "train_test_network.csv"),
    os.path.join(PROJECT_ROOT, "data", "Network_dataset_capped.csv"),
    os.path.join(PROJECT_ROOT, "data", "network_dataset_capped.csv"),
    os.path.join(SCRIPT_DIR, "data", "train_test_network.csv"),
    os.path.join(SCRIPT_DIR, "data", "Network_dataset_capped.csv"),
]
DATA_CSV_PATH = _pick_existing_path(csv_candidates)

if DATA_CSV_PATH is None:
    raise FileNotFoundError(
        "Could not find training CSV. Tried:\n"
        + "\n".join(f"  - {p}" for p in csv_candidates)
    )

print(f"Loading: {DATA_CSV_PATH}")
df = pd.read_csv(DATA_CSV_PATH, low_memory=False, dtype=str, on_bad_lines="skip")
df.columns = df.columns.str.strip()
df.drop(columns=["ts", "date", "time", "label"], errors="ignore", inplace=True)

if "type" not in df.columns:
    raise RuntimeError("Missing target column 'type'.")


# ── Step 2: Clean Labels ─────────────────────────────────────────────────────
labels_norm = df["type"].astype(str).str.strip().str.lower()

# Drop rare / problematic classes
drop_labels = {"mitm", "ransomware"}
dropped = int(labels_norm.isin(drop_labels).sum())
df = df.loc[~labels_norm.isin(drop_labels)].copy()
print(f"Dropped {dropped} rows with type in {sorted(drop_labels)}")

# Merge dos + ddos -> dos_ddos (half-sample each to match majority-class size)
labels_norm = df["type"].astype(str).str.strip().str.lower()
dos_df = df.loc[labels_norm == "dos"]
ddos_df = df.loc[labels_norm == "ddos"]
other_df = df.loc[~labels_norm.isin({"dos", "ddos"})]

dos_half = dos_df.sample(n=max(1, len(dos_df) // 2), random_state=SEED) if len(dos_df) > 0 else dos_df
ddos_half = ddos_df.sample(n=max(1, len(ddos_df) // 2), random_state=SEED) if len(ddos_df) > 0 else ddos_df
dos_half = dos_half.copy()
ddos_half = ddos_half.copy()
dos_half["type"] = "dos_ddos"
ddos_half["type"] = "dos_ddos"

df = pd.concat([other_df, dos_half, ddos_half], ignore_index=True)
df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
print(f"Merged dos({len(dos_half)})+ddos({len(ddos_half)}) -> dos_ddos({len(dos_half)+len(ddos_half)})")

# Drop IP columns (prevent topology leakage)
ip_cols = [c for c in ["src_ip", "dst_ip", "srcip", "dstip"] if c in df.columns]
df.drop(columns=ip_cols, inplace=True)
print(f"Dropped IP columns: {ip_cols}")


# ── Step 3: Prepare Features ─────────────────────────────────────────────────
y_all = df["type"].astype(str)
X_all = df.drop(columns=["type"])

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


# ── Step 4: Stratified Split (60/20/20) ──────────────────────────────────────
# FIX: Target encoder is fit on TRAIN only (was previously fit on all data).
print("Splitting data (stratified 60/20/20)...")

X_train_df, X_temp_df, y_train_str, y_temp_str = train_test_split(
    X_all, y_all, test_size=0.4, stratify=y_all, random_state=SEED,
)
X_val_df, X_test_df, y_val_str, y_test_str = train_test_split(
    X_temp_df, y_temp_str, test_size=0.5, stratify=y_temp_str, random_state=SEED,
)

del df, X_all, y_all, X_temp_df, y_temp_str
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

# Categorical encoding (train-fit only)
encoders = {}
for col in valid_cat_cols:
    le = SafeLabelEncoder()
    X_train_df[col] = le.fit(X_train_df[col]).transform(X_train_df[col])
    X_val_df[col] = le.transform(X_val_df[col])
    X_test_df[col] = le.transform(X_test_df[col])
    encoders[col] = le

# Scale numerics (needed for mutual-info feature selection to work well)
scaler_num = MinMaxScaler()
X_train_df[num_cols] = scaler_num.fit_transform(X_train_df[num_cols].values)
X_val_df[num_cols] = scaler_num.transform(X_val_df[num_cols].values)
X_test_df[num_cols] = scaler_num.transform(X_test_df[num_cols].values)

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

selected_mask = selector.get_support()
final_features = X_train_df.columns[selected_mask].tolist()
print(f"Selected features ({len(final_features)}): {final_features}")

with open(os.path.join(ARTIFACT_DIR, "final_features.txt"), "w") as f:
    f.write("\n".join(final_features))

# Re-normalize selected features for model input
final_scaler = MinMaxScaler()
X_train_sel = np.nan_to_num(final_scaler.fit_transform(X_train_sel)).astype(np.float32)
X_val_sel = np.nan_to_num(final_scaler.transform(X_val_sel)).astype(np.float32)
X_test_sel = np.nan_to_num(final_scaler.transform(X_test_sel)).astype(np.float32)

del X_train_df, X_val_df, X_test_df
gc.collect()


# ── Step 6: SMOTE Oversampling ────────────────────────────────────────────────
# FIX: Previous version had zero class balancing — minority classes (backdoor
# ~508k, injection ~453k) lagged behind the 700k-capped majority classes.
# SMOTE brings them closer to the majority count (capped at SMOTE_MAX_MULTIPLIER×).
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
    X_train_bal, y_train_bal = smote.fit_resample(X_train_sel, y_train)
    print(f"Post-SMOTE: {dict(Counter(y_train_bal))}")
else:
    X_train_bal, y_train_bal = X_train_sel, y_train
    print("SMOTE skipped (no eligible classes)")


# ── Step 7: One-Hot Targets ───────────────────────────────────────────────────
y_train_onehot = to_categorical(y_train_bal, num_classes=NUM_CLASSES).astype(np.float32)
y_val_onehot = to_categorical(y_val, num_classes=NUM_CLASSES).astype(np.float32)


# ── Step 8: Build Model & Train ──────────────────────────────────────────────
model = build_se_dwnet(X_train_bal.shape[1], NUM_CLASSES)
optimizer = Adam(learning_rate=5e-4, clipnorm=1.0)

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

pipeline_path = os.path.join(ARTIFACT_DIR, "preprocessing_pipeline.pkl")
with open(pipeline_path, "wb") as f:
    pickle.dump(
        {
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
        },
        f,
    )


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
    f.write("=== ResNet Base Evaluation ===\n")
    f.write(f"Model: {model_path}\n")
    f.write(f"Loss: {loss_info}\n\n")
    f.write(report_str)

cm = confusion_matrix(y_test_readable, y_pred_readable, labels=le_target.classes_)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=le_target.classes_,
            yticklabels=le_target.classes_, cmap="Blues")
plt.title("ResNet Base Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(ARTIFACT_DIR, "resnet_base_confusion_matrix.png"), dpi=200)
plt.close()

print(f"DONE. Artifacts saved to: {ARTIFACT_DIR}")
