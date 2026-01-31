import gc
import os
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

try:
    from . import config
    from .paths import detect_project_root, resolve_project_artifacts_dir
    from .preprocessing import SafeLabelEncoder
    from .models import build_transfer_model
except ImportError:
    import sys

    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import config  # type: ignore
    from paths import detect_project_root, resolve_project_artifacts_dir  # type: ignore
    from preprocessing import SafeLabelEncoder  # type: ignore
    from models import build_transfer_model  # type: ignore


class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "SafeLabelEncoder" and module in {
            "__main__",
            "preprocessing",
            "dl_pipeline.preprocessing",
            "resnet.dl_pipeline.preprocessing",
        }:
            return SafeLabelEncoder
        return super().find_class(module, name)


def _load_pipeline(path: str) -> dict:
    with open(path, "rb") as f:
        return _CompatUnpickler(f).load()


def _resolve_first_existing_path(candidates: list[str]) -> str:
    for p in candidates:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError(
        "None of the candidate paths exist:\n" + "\n".join(f"- {p}" for p in candidates)
    )


def _find_files(search_root: str, *, suffixes: tuple[str, ...]) -> list[str]:
    matches: list[str] = []
    for root, _, files in os.walk(search_root):
        for name in files:
            lower = name.lower()
            if any(lower.endswith(suf) for suf in suffixes):
                matches.append(os.path.join(root, name))
    matches.sort(key=lambda p: (os.path.basename(p).lower(), p))
    return matches


def _pick_best_path(paths: list[str], preferred_tokens: list[str]) -> str | None:
    if not paths:
        return None

    scored: list[tuple[int, float, str]] = []
    for p in paths:
        base = os.path.basename(p).lower()
        score = sum(1 for t in preferred_tokens if t in base)
        try:
            mtime = os.path.getmtime(p)
        except OSError:
            mtime = 0.0
        scored.append((score, mtime, p))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return scored[0][2]


def _identify_label_column(df: pd.DataFrame) -> str:
    for c in ("label", "Label", "type"):
        if c in df.columns:
            return c
    raise RuntimeError("Could not find label column. Expected one of: 'label', 'Label', 'type'.")


def _load_final_features(*, features_txt_path: str | None, pipeline: dict) -> list[str]:
    if features_txt_path and os.path.exists(features_txt_path):
        with open(features_txt_path, "r") as f:
            feats = [line.strip() for line in f.readlines()]
        feats = [f for f in feats if f]
        if feats:
            return feats

    feats = pipeline.get("features")
    if feats is None:
        raise RuntimeError(
            "Could not determine final features. Expected final_features.txt next to pipeline_objects.pkl "
            "or pipeline['features'] inside pipeline_objects.pkl."
        )
    feats = [str(f).strip() for f in list(feats)]
    feats = [f for f in feats if f]
    if not feats:
        raise RuntimeError("pipeline['features'] is empty; cannot proceed.")
    return feats


def _transform_with_base_pipeline(
    df: pd.DataFrame,
    *,
    pipeline: dict,
    final_features: list[str],
) -> np.ndarray:
    encoders = pipeline["encoders"]
    scaler_num = pipeline["scaler_num"]
    final_scaler = pipeline["final_scaler"]
    valid_cat_cols = pipeline["valid_cat_cols"]
    num_cols = pipeline["num_cols"]

    required_cols = list(dict.fromkeys(list(valid_cat_cols) + list(num_cols) + list(final_features)))

    X = df.copy()
    for col in required_cols:
        if col in X.columns:
            continue
        if col in valid_cat_cols:
            X[col] = "missing"
        else:
            X[col] = 0

    X = X[required_cols]

    for col in valid_cat_cols:
        if col not in X.columns:
            continue
        X[col] = X[col].fillna("missing").replace("-", "missing").astype(str)
        le = encoders.get(col)
        if le is None:
            X[col] = 0
        else:
            X[col] = le.transform(X[col])

    for col in num_cols:
        if col not in X.columns:
            continue
        X[col] = pd.to_numeric(X[col], errors="coerce")

    for col in final_features:
        if col not in X.columns or col in valid_cat_cols:
            continue
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X.replace([np.inf, -np.inf], 0, inplace=True)
    X = X.fillna(0)

    for col in config.LOG_COLS:
        if col in X.columns:
            X[col] = np.log1p(pd.to_numeric(X[col], errors="coerce").fillna(0).clip(lower=0))

    X_num = scaler_num.transform(X[num_cols].values)
    X.loc[:, num_cols] = X_num

    missing_final = [c for c in final_features if c not in X.columns]
    if missing_final:
        raise RuntimeError(
            "Custom dataset is missing required final features after preprocessing: "
            + ", ".join(missing_final[:20])
            + (" ..." if len(missing_final) > 20 else "")
        )

    X_sel = X[final_features].values

    X_scaled = final_scaler.transform(X_sel)
    X_scaled = np.nan_to_num(X_scaled).astype(np.float32, copy=False)
    return X_scaled


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

    model_candidates = [
        os.path.join(project_root, "artifacts", "ton_iot_resnet_attn_focalfix_mha.keras"),
        os.path.join(project_root, "resnet", "artifacts", "pretrained", "ton_iot_resnet_attn_focalfix_mha.keras"),
        os.path.join(project_root, "resnet", "artifacts", config.BASE_ARTIFACT_SUBDIR, "ton_iot_resnet_attn_focalfix_mha.keras"),
    ]

    pipeline_candidates = [
        os.path.join(project_root, "artifacts", "pipeline_objects.pkl"),
        os.path.join(project_root, "resnet", "artifacts", "pretrained", "pipeline_objects.pkl"),
        os.path.join(project_root, "resnet", "artifacts", config.BASE_ARTIFACT_SUBDIR, "pipeline_objects.pkl"),
    ]

    base_model_path = None
    base_pipeline_path = None

    for p in model_candidates:
        if os.path.exists(p):
            base_model_path = p
            break
    for p in pipeline_candidates:
        if os.path.exists(p):
            base_pipeline_path = p
            break

    if base_model_path is None or base_pipeline_path is None:
        artifacts_root = os.path.join(project_root, "artifacts")
        if os.path.isdir(artifacts_root):
            if base_model_path is None:
                found_models = _find_files(artifacts_root, suffixes=(".keras", ".h5"))
                base_model_path = _pick_best_path(found_models, ["resnet", "sedwnet", "ton_iot", "attn"]) or base_model_path
            if base_pipeline_path is None:
                found_pipes = _find_files(artifacts_root, suffixes=("pipeline_objects.pkl",))
                base_pipeline_path = _pick_best_path(found_pipes, ["pipeline_objects", "sedwnet", "resnet"]) or base_pipeline_path

    if base_model_path is None:
        raise FileNotFoundError(
            "Could not find base ResNet model (.keras/.h5). Expected under artifacts/ or resnet/artifacts/."
        )
    if base_pipeline_path is None:
        raise FileNotFoundError(
            "Could not find pipeline_objects.pkl for the base ResNet model. Expected under artifacts/ or resnet/artifacts/."
        )

    out_dir = resolve_project_artifacts_dir(project_root, "resnet_transfer_8class")
    os.makedirs(out_dir, exist_ok=True)

    print("=== Transfer Learning: SE-DWNet/ResNet -> all classes ===")
    print(f"Project root: {project_root}")
    print(f"Base model: {base_model_path}")
    print(f"Base pipeline: {base_pipeline_path}")
    print(f"Custom dataset: {custom_csv_path}")
    print(f"Artifacts: {out_dir}")

    pipeline = _load_pipeline(base_pipeline_path)

    final_features_path = os.path.join(os.path.dirname(base_pipeline_path), "final_features.txt")
    final_features = _load_final_features(features_txt_path=final_features_path, pipeline=pipeline)
    print(f"Final feature set: {len(final_features)} features")

    df = pd.read_csv(custom_csv_path, low_memory=False)
    df.columns = df.columns.str.strip()

    label_col = _identify_label_column(df)
    df[label_col] = df[label_col].astype(str)

    df.drop(columns=["ts", "date", "time"], errors="ignore", inplace=True)

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

    le_target_4 = LabelEncoder()
    le_target_4.fit(y_train_str)

    y_train = le_target_4.transform(y_train_str)
    y_val = le_target_4.transform(y_val_str)
    y_test = le_target_4.transform(y_test_str)

    class_names = le_target_4.classes_.tolist()
    print(f"Label mapping ({len(class_names)} classes): {class_names}")

    X_train = _transform_with_base_pipeline(X_train_df, pipeline=pipeline, final_features=final_features)
    X_val = _transform_with_base_pipeline(X_val_df, pipeline=pipeline, final_features=final_features)
    X_test = _transform_with_base_pipeline(X_test_df, pipeline=pipeline, final_features=final_features)

    del X_train_df, X_val_df, X_test_df
    gc.collect()

    model = build_transfer_model(base_model_path, num_classes=len(class_names))

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
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=config.TRANSFER_FREEZE_EPOCHS,
        batch_size=config.TRANSFER_BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
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
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=config.TRANSFER_MAX_EPOCHS,
        batch_size=config.TRANSFER_BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    print("Evaluating on test...")
    test_probs = model.predict(X_test, batch_size=config.TRANSFER_BATCH_SIZE)
    y_pred = np.argmax(test_probs, axis=1)

    report_txt = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    print(report_txt)

    with open(os.path.join(out_dir, "classification_report_8class.txt"), "w") as f:
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
    plt.savefig(os.path.join(out_dir, "confusion_matrix_8class.png"), dpi=200)
    plt.close()

    model.save(os.path.join(out_dir, "resnet_transfer_8class.keras"))
    with open(os.path.join(out_dir, "transfer_metadata.pkl"), "wb") as f:
        pickle.dump(
            {
                "base_model": base_model_path,
                "base_pipeline": base_pipeline_path,
                "custom_dataset": custom_csv_path,
                "classes": class_names,
                "label_encoder_4": le_target_4,
            },
            f,
        )

    print("DONE. Artifacts written to:", out_dir)


if __name__ == "__main__":
    main()
