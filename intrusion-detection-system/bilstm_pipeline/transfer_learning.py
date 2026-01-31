import gc
import os
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

try:
    from . import config
    from .paths import detect_project_root, resolve_artifacts_dir
    from .models import build_transfer_model
except ImportError:
    import sys

    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import config  # type: ignore
    from paths import detect_project_root, resolve_artifacts_dir  # type: ignore
    from models import build_transfer_model  # type: ignore


def _resolve_first_existing_path(candidates: list[str]) -> str:
    for p in candidates:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError(
        "None of the candidate paths exist:\n" + "\n".join(f"- {p}" for p in candidates)
    )


def _find_model_files(search_root: str) -> list[str]:
    matches: list[str] = []
    for root, _, files in os.walk(search_root):
        for name in files:
            lower = name.lower()
            if lower.endswith(".h5") or lower.endswith(".keras"):
                matches.append(os.path.join(root, name))
    matches.sort(key=lambda p: (os.path.basename(p).lower(), p))
    return matches


def _pick_best_model(model_paths: list[str]) -> str | None:
    if not model_paths:
        return None

    preferred_tokens = ["bilstm", "lstm", "ton", "iot"]
    scored: list[tuple[int, float, str]] = []
    for p in model_paths:
        base = os.path.basename(p).lower()
        score = sum(1 for t in preferred_tokens if t in base)
        try:
            mtime = os.path.getmtime(p)
        except OSError:
            mtime = 0.0
        scored.append((score, mtime, p))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return scored[0][2]


def _load_feature_list(path: str) -> list[str]:
    with open(path, "r") as f:
        feats = [line.strip() for line in f.readlines()]
    feats = [f for f in feats if f]
    if not feats:
        raise RuntimeError(f"Feature list is empty: {path}")
    return feats


def _clean_cat(s: pd.Series) -> pd.Series:
    return s.fillna("missing").replace("-", "missing").astype(str)


def _fit_label_encoder_train_only(train_series: pd.Series) -> LabelEncoder:
    unk = "__UNK__"
    le = LabelEncoder()
    cleaned = _clean_cat(train_series)
    classes = pd.Index(cleaned.unique()).astype(str).tolist()
    if unk not in classes:
        classes.append(unk)
    le.fit(classes)
    return le


def _transform_with_unk(le: LabelEncoder, series: pd.Series) -> np.ndarray:
    unk = "__UNK__"
    cleaned = _clean_cat(series)
    known = set(le.classes_.tolist())
    cleaned = cleaned.where(cleaned.isin(known), other=unk)
    return le.transform(cleaned)


def main() -> None:
    np.random.seed(config.SEED)
    tf.keras.utils.set_random_seed(config.SEED)

    project_root = detect_project_root()

    custom_csv_candidates = [
        os.path.join(project_root, "data", "custom", "tpot_final_ton_iot.csv"),
        os.path.join(project_root, "data", "custom", "tpot_final_ton_iot.csv.gz"),
        os.path.join(project_root, "resnet", "data", "custom", "tpot_final_ton_iot.csv"),
    ]
    custom_csv_path = _resolve_first_existing_path(custom_csv_candidates)

    base_model_candidates = [
        os.path.join(project_root, "artifacts", "bilstm_smote_base.h5"),
        os.path.join(project_root, "artifacts", "ton_iot_h200_production_ports.h5"),
        os.path.join(project_root, "artifacts", "ton_iot_bilstm_final_cw.h5"),
        os.path.join(project_root, "smote", "ton_iot_h200_production_ports.h5"),
    ]

    try:
        base_model_path = _resolve_first_existing_path(base_model_candidates)
    except FileNotFoundError:
        artifacts_dir = os.path.join(project_root, "artifacts")
        found = _find_model_files(artifacts_dir) if os.path.isdir(artifacts_dir) else []
        picked = _pick_best_model(found)
        if picked:
            base_model_path = picked
        else:
            msg = [
                "Could not find a base model (.h5/.keras).",
                "Searched candidates:",
                *(f"- {p}" for p in base_model_candidates),
                "",
                f"Also scanned recursively: {artifacts_dir}",
            ]
            if found:
                msg.append("Found model files:")
                msg.extend(f"- {p}" for p in found)
            msg.append(
                "Fix: copy your trained base model into artifacts/ (any .h5/.keras), "
                "or rename it to one of the expected filenames."
            )
            raise FileNotFoundError("\n".join(msg))

    artifact_dir = resolve_artifacts_dir(project_root, config.TRANSFER_ARTIFACT_SUBDIR)
    os.makedirs(artifact_dir, exist_ok=True)

    print("=== Transfer Learning: BiLSTM -> all classes ===")
    print(f"Base model: {base_model_path}")
    print(f"Custom dataset: {custom_csv_path}")
    print(f"Project root: {project_root}")
    print(f"Artifacts: {artifact_dir}")

    feature_list_candidates = [
        os.path.join(project_root, "artifacts", "final_features_list.txt"),
        os.path.join(project_root, "artifacts", "final_features.txt"),
        os.path.join(os.path.dirname(base_model_path), "final_features_list.txt"),
        os.path.join(os.path.dirname(base_model_path), "final_features.txt"),
    ]

    feature_list_path = _resolve_first_existing_path(feature_list_candidates)
    final_features = _load_feature_list(feature_list_path)
    print(f"Using fixed features from: {feature_list_path}")
    print(f"Fixed feature count: {len(final_features)}")

    df = pd.read_csv(custom_csv_path, low_memory=False)
    df.columns = df.columns.str.strip()

    if "label" in df.columns:
        label_col = "label"
    elif "Label" in df.columns:
        label_col = "Label"
        df.rename(columns={"Label": "label"}, inplace=True)
        label_col = "label"
    elif "type" in df.columns:
        label_col = "type"
    else:
        raise RuntimeError("Could not find label column. Expected one of: 'label', 'Label', 'type'.")

    df.drop(columns=["ts", "date", "time"], errors="ignore", inplace=True)

    df[label_col] = df[label_col].astype(str)

    print("Class counts (raw):", dict(Counter(df[label_col])))

    if config.DROP_IPS:
        ip_cols = ["src_ip", "dst_ip", "srcip", "dstip"]
        cols_to_drop = [c for c in ip_cols if c in df.columns]
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)
            print(f"Dropped IP columns: {cols_to_drop}")

    y_all = df[label_col].astype(str)
    X_all = df.drop(columns=[label_col]).copy()

    X_train_df, X_temp_df, y_train_str, y_temp_str = train_test_split(
        X_all, y_all, test_size=0.4, stratify=y_all, random_state=config.SEED
    )
    X_val_df, X_test_df, y_val_str, y_test_str = train_test_split(
        X_temp_df, y_temp_str, test_size=0.5, stratify=y_temp_str, random_state=config.SEED
    )

    del df, X_all, y_all, X_temp_df, y_temp_str
    gc.collect()

    le_target = LabelEncoder()
    le_target.fit(y_train_str)

    y_train = le_target.transform(y_train_str)
    y_val = le_target.transform(y_val_str)
    y_test = le_target.transform(y_test_str)

    class_names = le_target.classes_.tolist()
    print(f"Label mapping ({len(class_names)} classes): {class_names}")

    missing = [c for c in final_features if c not in X_train_df.columns]
    if missing:
        raise RuntimeError(
            "Custom dataset is missing required columns from final_features_list.txt: "
            + ", ".join(missing[:25])
            + (" ..." if len(missing) > 25 else "")
        )

    X_train_df = X_train_df[final_features].copy()
    X_val_df = X_val_df[final_features].copy()
    X_test_df = X_test_df[final_features].copy()

    cat_cols = [c for c in X_train_df.columns if X_train_df[c].dtype == object]
    num_cols = [c for c in X_train_df.columns if c not in cat_cols]

    encoders: dict[str, LabelEncoder] = {}
    for col in cat_cols:
        le = _fit_label_encoder_train_only(X_train_df[col])
        encoders[col] = le
        X_train_df[col] = _transform_with_unk(le, X_train_df[col]).astype("int32")
        X_val_df[col] = _transform_with_unk(le, X_val_df[col]).astype("int32")
        X_test_df[col] = _transform_with_unk(le, X_test_df[col]).astype("int32")

    for split_df in (X_train_df, X_val_df, X_test_df):
        for col in num_cols:
            split_df[col] = pd.to_numeric(split_df[col], errors="coerce").fillna(0).astype("float32")

    log_cols_present = [c for c in config.LOG_COLS if c in X_train_df.columns]
    for split_df in (X_train_df, X_val_df, X_test_df):
        for col in log_cols_present:
            split_df[col] = np.log1p(np.maximum(split_df[col].astype("float32"), 0.0))

    X_train = X_train_df.values
    X_val = X_val_df.values
    X_test = X_test_df.values

    del X_train_df, X_val_df, X_test_df
    gc.collect()

    scaler = MinMaxScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm = scaler.transform(X_val)
    X_test_norm = scaler.transform(X_test)

    X_train_sel = X_train_norm
    X_val_sel = X_val_norm
    X_test_sel = X_test_norm

    n_feat = X_train_sel.shape[1]
    X_train_3d = X_train_sel.reshape(X_train_sel.shape[0], 1, n_feat)
    X_val_3d = X_val_sel.reshape(X_val_sel.shape[0], 1, n_feat)
    X_test_3d = X_test_sel.reshape(X_test_sel.shape[0], 1, n_feat)

    model = build_transfer_model(
        base_model_path,
        num_classes=len(class_names),
        input_shape=tuple(X_train_3d.shape[1:]),
    )

    for layer in model.layers:
        layer.trainable = layer.name == "transfer_head"

    model.compile(
        optimizer=Adam(learning_rate=config.TRANSFER_LR_FROZEN),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
    ]

    print("Training (frozen base)...")
    model.fit(
        X_train_3d,
        y_train,
        validation_data=(X_val_3d, y_val),
        epochs=config.TRANSFER_FREEZE_EPOCHS,
        batch_size=config.TRANSFER_BATCH_SIZE,
        verbose=1,
        callbacks=callbacks,
    )

    for layer in model.layers:
        layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=config.TRANSFER_LR_FINETUNE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("Training (fine-tune all)...")
    model.fit(
        X_train_3d,
        y_train,
        validation_data=(X_val_3d, y_val),
        epochs=config.TRANSFER_MAX_EPOCHS,
        batch_size=config.TRANSFER_BATCH_SIZE,
        verbose=1,
        callbacks=callbacks,
    )

    print("Evaluating on test...")
    test_probs = model.predict(X_test_3d, batch_size=config.TRANSFER_BATCH_SIZE)
    y_pred = np.argmax(test_probs, axis=1)

    report_txt = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )
    print(report_txt)

    with open(os.path.join(artifact_dir, "classification_report_8class.txt"), "w") as f:
        f.write(report_txt)

    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(class_names))))
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="Blues",
    )
    plt.title(model.name)
    plt.tight_layout()
    plt.savefig(os.path.join(artifact_dir, "confusion_matrix_8class.png"), dpi=200)
    plt.close()

    model.save(os.path.join(artifact_dir, "bilstm_transfer_8class.h5"))

    with open(os.path.join(artifact_dir, "preprocessor_le_target_all.pkl"), "wb") as f:
        pickle.dump(le_target, f)
    with open(os.path.join(artifact_dir, "preprocessor_feature_encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)
    with open(os.path.join(artifact_dir, "preprocessor_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(artifact_dir, "preprocessor_selector.pkl"), "wb") as f:
        pickle.dump({"type": "fixed_features", "features": final_features}, f)

    with open(os.path.join(artifact_dir, "feature_columns_before_selection.txt"), "w") as f:
        f.write("\n".join(final_features))
    with open(os.path.join(artifact_dir, "selected_features.txt"), "w") as f:
        f.write("\n".join(final_features))

    print("DONE. Artifacts written to:", artifact_dir)


if __name__ == "__main__":
    main()
