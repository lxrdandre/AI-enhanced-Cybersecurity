#!/usr/bin/env python3


from __future__ import annotations

import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif


INPUT_CSV = "/ton-iot-project/fedlearning/data/train_test_network.csv"
OUTPUT_DIR = "/ton-iot-project/fedlearning/data/processed"
LABEL_COL = "label"
TYPE_COL = "type"
NORMAL_LABEL = "0"
NORMAL_TYPE = "normal"
TEST_SIZE = 0.2
RANDOM_STATE = 42
SAVE_SCALER = True
FEATURE_SELECTION = True
TOP_T_FEATURES = 30
STABILITY_RUNS = 3
STABILITY_MIN_COUNT = 2
NUM_CLIENTS = 5


def drop_categorical(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns
    return df.drop(columns=list(cat_cols))


def split_train_test(
    df: pd.DataFrame, label_col: str, test_size: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(
        df,
        test_size=test_size,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=df[label_col].astype(int),
    )


def main() -> None:
    df = pd.read_csv(
        INPUT_CSV,
        engine="python",
        on_bad_lines="skip",
    )
    if LABEL_COL not in df.columns:
        if TYPE_COL in df.columns:
            df[LABEL_COL] = (df[TYPE_COL].astype(str).str.lower() != NORMAL_TYPE).astype(
                int
            )
        else:
            raise ValueError(
                f"Neither '{LABEL_COL}' nor '{TYPE_COL}' found in input."
            )

    # Drop categorical columns (except label)
    label_series = df[LABEL_COL]
    drop_cols = [LABEL_COL]
    if TYPE_COL in df.columns:
        drop_cols.append(TYPE_COL)
    features_df = df.drop(columns=drop_cols)
    features_df = drop_categorical(features_df)

    # Re-attach label for splitting
    df_proc = features_df.copy()
    df_proc[LABEL_COL] = label_series

    # Stable feature selection (mutual information proxy for SHAP stability)
    if FEATURE_SELECTION:
        rng = np.random.RandomState(RANDOM_STATE)
        feature_counts = {c: 0 for c in features_df.columns}
        for _ in range(STABILITY_RUNS):
            idx = rng.choice(len(df_proc), size=min(len(df_proc), 200_000), replace=False)
            x_fs = df_proc.iloc[idx].drop(columns=[LABEL_COL])
            y_fs = df_proc.iloc[idx][LABEL_COL].astype(int)
            mi = mutual_info_classif(x_fs, y_fs, discrete_features=False, random_state=RANDOM_STATE)
            top_idx = np.argsort(mi)[-TOP_T_FEATURES:]
            for col in x_fs.columns[top_idx]:
                feature_counts[col] += 1
        stable_features = [
            f for f, c in feature_counts.items() if c >= STABILITY_MIN_COUNT
        ]
        if len(stable_features) == 0:
            stable_features = list(features_df.columns)
        features_df = features_df[stable_features]
        df_proc = features_df.copy()
        df_proc[LABEL_COL] = label_series

    # Supervised train/test split (balanced input)
    train_df, test_df = split_train_test(df_proc, LABEL_COL, TEST_SIZE)

    # Normalize using train_normal stats
    scaler = StandardScaler()
    feature_cols = [c for c in train_df.columns if c != LABEL_COL]

    train_features = scaler.fit_transform(train_df[feature_cols].values)
    test_features = scaler.transform(test_df[feature_cols].values)

    train_out = pd.DataFrame(train_features, columns=feature_cols)
    train_out[LABEL_COL] = train_df[LABEL_COL].values

    test_out = pd.DataFrame(test_features, columns=feature_cols)
    test_out[LABEL_COL] = test_df[LABEL_COL].values

    # Split training (normal only) into 5 device subsets
    shuffled_train = train_out.sample(
        frac=1.0, random_state=RANDOM_STATE
    ).reset_index(drop=True)
    split_indices = np.array_split(np.arange(len(shuffled_train)), NUM_CLIENTS)
    device_splits = [
        shuffled_train.iloc[idx].reset_index(drop=True) for idx in split_indices
    ]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_path = os.path.join(OUTPUT_DIR, "train_supervised.csv")
    test_path = os.path.join(OUTPUT_DIR, "test_supervised.csv")
    device_paths = [
        os.path.join(OUTPUT_DIR, f"device_{i}.csv")
        for i in range(1, NUM_CLIENTS + 1)
    ]

    train_out.to_csv(train_path, index=False)
    test_out.to_csv(test_path, index=False)
    for split_df, path in zip(device_splits, device_paths):
        split_df.to_csv(path, index=False)

    if SAVE_SCALER:
        scaler_path = os.path.join(OUTPUT_DIR, "scaler.joblib")
        joblib.dump(scaler, scaler_path)

    print(f"Saved: {train_path}")
    print(f"Saved: {test_path}")
    for path in device_paths:
        print(f"Saved: {path}")
    if SAVE_SCALER:
        print(f"Saved: {scaler_path}")


if __name__ == "__main__":
    main()
