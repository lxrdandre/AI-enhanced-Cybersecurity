import gc
import os
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

try:
    from . import config
    from .data_io import load_csv, drop_label_column
    from .paths import detect_project_root, resolve_data_csv, resolve_resnet_artifacts_dir
    from .preprocessing import (
        build_feature_columns,
        clean_features,
        encode_target,
        split_data,
        fit_preprocessors,
    )
    from .models import build_attention_resnet_mlp, build_se_dwnet
    from .losses import build_focal_loss
    from .training import build_callbacks
    from .evaluation import save_classification_report, save_confusion_matrix
except ImportError:
    import sys

    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import config  # type: ignore
    from data_io import load_csv, drop_label_column  # type: ignore
    from paths import detect_project_root, resolve_data_csv, resolve_resnet_artifacts_dir  # type: ignore
    from preprocessing import (  # type: ignore
        build_feature_columns,
        clean_features,
        encode_target,
        split_data,
        fit_preprocessors,
    )
    from models import build_attention_resnet_mlp, build_se_dwnet  # type: ignore
    from losses import build_focal_loss  # type: ignore
    from training import build_callbacks  # type: ignore
    from evaluation import save_classification_report, save_confusion_matrix  # type: ignore


def _build_model(input_dim: int, num_classes: int) -> tf.keras.Model:
    if config.BASE_MODEL_TYPE == "resnet_mlp":
        return build_attention_resnet_mlp(
            input_dim,
            num_classes,
            use_feature_mha=config.USE_FEATURE_MHA,
            mha_heads=config.MHA_HEADS,
            mha_key_dim=config.MHA_KEY_DIM,
            mha_dropout=config.MHA_DROPOUT,
        )
    return build_se_dwnet(input_dim, num_classes)


def main() -> None:
    np.random.seed(config.SEED)
    tf.random.set_seed(config.SEED)

    project_root = detect_project_root()
    data_csv_path = resolve_data_csv(project_root, config.DATA_CSV_RELATIVE)
    artifact_dir = resolve_resnet_artifacts_dir(config.BASE_ARTIFACT_SUBDIR)
    os.makedirs(artifact_dir, exist_ok=True)

    print("=== TON_IoT Deep Learning Pipeline (Base Training) ===")
    print(f"Project root: {project_root}")
    print(f"Data: {data_csv_path}")
    print(f"Artifacts: {artifact_dir}")

    df = load_csv(data_csv_path)
    df = drop_label_column(df)
    df.columns = df.columns.str.strip()

    if "type" in df.columns and config.DROP_TYPES:
        df = df[~df["type"].isin(config.DROP_TYPES)]

    if config.DROP_IPS:
        topology_cols = ["src_ip", "dst_ip", "srcip", "dstip", "label"]
        cols_to_drop = [c for c in topology_cols if c in df.columns]
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)
            print(f"Dropped IP columns: {cols_to_drop}")

    if "type" not in df.columns:
        raise RuntimeError("Missing target column 'type'.")

    y_raw = df["type"]
    X_raw = df.drop(columns=["type"])

    y_encoded, le_target, num_classes, class_names = encode_target(y_raw)
    print(f"Target encoded. Classes: {num_classes} -> {class_names}")

    valid_cat_cols, num_cols = build_feature_columns(X_raw, config.CAT_COLS)
    X_clean = clean_features(
        X_raw,
        valid_cat_cols=valid_cat_cols,
        num_cols=num_cols,
        log_cols=config.LOG_COLS,
    )
    print(f"Data cleaned. Shape: {X_clean.shape}")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X_clean,
        y_encoded,
        seed=config.SEED,
    )

    del X_clean, X_raw, y_raw
    gc.collect()

    (
        X_train_sel,
        X_val_sel,
        X_test_sel,
        encoders,
        scaler_num,
        selector,
        final_scaler,
        final_features,
    ) = fit_preprocessors(
        X_train,
        X_val,
        X_test,
        y_train,
        valid_cat_cols=valid_cat_cols,
        num_cols=num_cols,
        target_k=config.TARGET_K,
        seed=config.SEED,
    )

    with open(os.path.join(artifact_dir, "final_features.txt"), "w") as f:
        f.write("\n".join(final_features))

    X_train_res, y_train_res = X_train_sel, y_train

    X_train_sel = final_scaler.transform(X_train_sel)
    X_train_res = final_scaler.transform(X_train_res)
    X_val_sel = final_scaler.transform(X_val_sel)
    X_test_sel = final_scaler.transform(X_test_sel)

    X_train_sel = np.nan_to_num(X_train_sel).astype(np.float32, copy=False)
    X_train_res = np.nan_to_num(X_train_res).astype(np.float32, copy=False)
    X_val_sel = np.nan_to_num(X_val_sel).astype(np.float32, copy=False)
    X_test_sel = np.nan_to_num(X_test_sel).astype(np.float32, copy=False)

    y_train_onehot = to_categorical(y_train_res, num_classes=num_classes).astype(np.float32, copy=False)
    y_val_onehot = to_categorical(y_val, num_classes=num_classes).astype(np.float32, copy=False)

    model = _build_model(X_train_res.shape[1], num_classes)
    optimizer = Adam(learning_rate=5e-4, clipnorm=1.0)

    if config.USE_FOCAL_LOSS:
        loss_fn, loss_info = build_focal_loss(
            y_train_res,
            num_classes=num_classes,
            gamma=config.FOCAL_GAMMA,
            alpha_clip=config.FOCAL_ALPHA_CLIP,
        )
    else:
        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0.1)
        loss_info = {"name": "CategoricalCrossentropy", "label_smoothing": 0.1}

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

    callbacks = build_callbacks(patience=12, lr_patience=4)

    print("Starting training...")
    model.fit(
        X_train_res,
        y_train_onehot,
        validation_data=(X_val_sel, y_val_onehot),
        epochs=config.MAX_EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    model_path = os.path.join(artifact_dir, "ton_iot_resnet_attn_focalfix_mha.keras")
    model.save(model_path)

    pipeline_path = os.path.join(artifact_dir, "pipeline_objects.pkl")
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
                "seed": config.SEED,
                "loss": loss_info,
                "use_feature_mha": config.USE_FEATURE_MHA,
                "mha_heads": config.MHA_HEADS,
                "mha_key_dim": config.MHA_KEY_DIM,
            },
            f,
        )

    print("Evaluating on test set...")
    test_probs = model.predict(X_test_sel, batch_size=config.BATCH_SIZE)
    test_pred = np.argmax(test_probs, axis=1)

    y_test_readable = le_target.inverse_transform(y_test)
    y_pred_readable = le_target.inverse_transform(test_pred)

    report_path = os.path.join(artifact_dir, "final_classification_report.txt")
    report_str = save_classification_report(
        y_test_readable,
        y_pred_readable,
        class_names=class_names,
        report_path=report_path,
    )
    print(report_str)

    save_confusion_matrix(
        y_test_readable,
        y_pred_readable,
        class_names=class_names,
        title=model.name,
        out_path=os.path.join(artifact_dir, "confusion_matrix_final.png"),
    )

    print("DONE. Artifacts saved to:", artifact_dir)


if __name__ == "__main__":
    main()
