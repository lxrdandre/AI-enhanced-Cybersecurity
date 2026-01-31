import gc
import os
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

try:
    from . import config
    from .data_io import load_csv, drop_common_time_columns
    from .paths import detect_project_root, resolve_data_csv, resolve_artifacts_dir
    from .preprocessing import (
        optimize_dtypes,
        split_data,
        fit_transform_features,
        select_features_chi2,
    )
    from .models import build_bilstm
    from .training import build_callbacks
    from .evaluation import save_classification_report, save_confusion_matrix
except ImportError:
    import sys

    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import config  # type: ignore
    from data_io import load_csv, drop_common_time_columns  # type: ignore
    from paths import detect_project_root, resolve_data_csv, resolve_artifacts_dir  # type: ignore
    from preprocessing import (  # type: ignore
        optimize_dtypes,
        split_data,
        fit_transform_features,
        select_features_chi2,
    )
    from models import build_bilstm  # type: ignore
    from training import build_callbacks  # type: ignore
    from evaluation import save_classification_report, save_confusion_matrix  # type: ignore


def _apply_smote(X_train: np.ndarray, y_train: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        from imblearn.over_sampling import SMOTE
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency 'imbalanced-learn'. Install with: pip install imbalanced-learn"
        ) from exc

    counts = Counter(y_train)
    if min(counts.values()) < 2:
        return X_train, y_train

    k_neighbors = min(config.SMOTE_K_NEIGHBORS_MAX, max(1, min(counts.values()) - 1))
    smote = SMOTE(random_state=seed, k_neighbors=k_neighbors)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    return X_train_bal, y_train_bal


def main() -> None:
    np.random.seed(config.SEED)
    tf.keras.utils.set_random_seed(config.SEED)

    project_root = detect_project_root()
    data_csv_path = resolve_data_csv(project_root, config.DATA_CSV_RELATIVE)
    artifact_dir = resolve_artifacts_dir(project_root, config.BASE_ARTIFACT_SUBDIR)
    os.makedirs(artifact_dir, exist_ok=True)

    print("=== BiLSTM Base Training (SMOTE + Chi2) ===")
    print(f"Project root: {project_root}")
    print(f"Data: {data_csv_path}")
    print(f"Artifacts: {artifact_dir}")

    df = load_csv(data_csv_path)
    df = drop_common_time_columns(df)
    df.columns = df.columns.str.strip()

    if "type" not in df.columns:
        raise RuntimeError("Missing target column 'type'.")

    if config.DROP_TYPES:
        df = df[~df["type"].isin(config.DROP_TYPES)]

    if config.DROP_IPS:
        ip_cols = ["src_ip", "dst_ip", "srcip", "dstip", "label"]
        cols_to_drop = [c for c in ip_cols if c in df.columns]
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)
            print(f"Dropped IP columns: {cols_to_drop}")

    df = df.sample(frac=1, random_state=config.SEED).reset_index(drop=True)

    min_samples_per_class = 3
    type_counts = df["type"].value_counts()
    rare_types = type_counts[type_counts < min_samples_per_class].index.tolist()
    if rare_types:
        print(
            f"Dropping {len(rare_types)} classes with <{min_samples_per_class} samples: {rare_types}"
        )
        df = df[~df["type"].isin(rare_types)].reset_index(drop=True)

    y_all_str = df["type"].astype(str)
    X_all_df = df.drop(columns=["type"]).copy()

    X_train_df, X_val_df, X_test_df, y_train_str, y_val_str, y_test_str = split_data(
        X_all_df, y_all_str, seed=config.SEED
    )

    del df, X_all_df, y_all_str
    gc.collect()

    le_target = LabelEncoder()
    le_target.fit(y_train_str)

    train_classes = set(le_target.classes_.tolist())
    val_unseen = set(pd.unique(y_val_str)) - train_classes
    test_unseen = set(pd.unique(y_test_str)) - train_classes
    if val_unseen or test_unseen:
        raise ValueError(
            "Unseen target classes present outside training split. "
            f"val_unseen={sorted(val_unseen)}, test_unseen={sorted(test_unseen)}."
        )

    y_train = le_target.transform(y_train_str)
    y_val = le_target.transform(y_val_str)
    y_test = le_target.transform(y_test_str)

    del y_train_str, y_val_str, y_test_str
    gc.collect()

    X_train, X_val, X_test, encoders, scaler, X_cols, _ = fit_transform_features(
        X_train_df, X_val_df, X_test_df, log_cols=config.LOG_COLS
    )

    del X_train_df, X_val_df, X_test_df
    gc.collect()

    X_train_sel, X_val_sel, X_test_sel, scaler, selector, selected_idx = select_features_chi2(
        X_train,
        X_val,
        X_test,
        y_train,
        target_k=config.TARGET_K,
    )

    selected_features = [X_cols[i] for i in selected_idx]

    with open(os.path.join(artifact_dir, "final_features_list.txt"), "w") as f:
        f.write("\n".join(selected_features))

    with open(os.path.join(artifact_dir, "preprocessor_le_target.pkl"), "wb") as f:
        pickle.dump(le_target, f)
    with open(os.path.join(artifact_dir, "preprocessor_feature_encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)
    with open(os.path.join(artifact_dir, "preprocessor_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(artifact_dir, "preprocessor_selector.pkl"), "wb") as f:
        pickle.dump(selector, f)

    X_train_bal, y_train_bal = _apply_smote(X_train_sel, y_train, seed=config.SEED)

    class_weight = compute_class_weight("balanced", classes=np.unique(y_train_bal), y=y_train_bal)
    class_weight_dict = dict(zip(np.unique(y_train_bal), class_weight))

    n_feat = X_train_bal.shape[1]
    X_train_3d = X_train_bal.reshape(X_train_bal.shape[0], 1, n_feat)
    X_val_3d = X_val_sel.reshape(X_val_sel.shape[0], 1, n_feat)
    X_test_3d = X_test_sel.reshape(X_test_sel.shape[0], 1, n_feat)

    model = build_bilstm(input_shape=(1, n_feat), num_classes=len(le_target.classes_))
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = build_callbacks(patience=15, lr_patience=5)

    print("Starting training...")
    model.fit(
        X_train_3d,
        y_train_bal,
        validation_data=(X_val_3d, y_val),
        epochs=config.MAX_EPOCHS,
        batch_size=config.BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1,
    )

    model_path = os.path.join(artifact_dir, "bilstm_smote_base.h5")
    model.save(model_path)

    print("Evaluating...")
    test_probs = model.predict(X_test_3d, batch_size=config.BATCH_SIZE)
    y_pred = np.argmax(test_probs, axis=1)

    report_path = os.path.join(artifact_dir, "classification_report_base.txt")
    report_str = save_classification_report(
        y_test,
        y_pred,
        class_names=le_target.classes_.tolist(),
        report_path=report_path,
    )
    print(report_str)

    save_confusion_matrix(
        y_test,
        y_pred,
        class_names=le_target.classes_.tolist(),
        title=model.name,
        out_path=os.path.join(artifact_dir, "confusion_matrix_base.png"),
    )

    print("DONE. Artifacts saved to:", artifact_dir)


if __name__ == "__main__":
    main()
