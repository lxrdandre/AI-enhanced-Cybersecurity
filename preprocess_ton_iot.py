#!/usr/bin/env python3


from __future__ import annotations

import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


INPUT_CSV = "/ton-iot-project/fedlearning/data/train_test_network.csv"
OUTPUT_DIR = "/ton-iot-project/fedlearning/data/processed"
LABEL_COL = "label"
NORMAL_LABEL = "0"
TEST_SIZE = 0.2
RANDOM_STATE = 42
SAVE_SCALER = True


def drop_categorical(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns
    return df.drop(columns=list(cat_cols))


def split_normal_attack(
    df: pd.DataFrame, label_col: str, normal_label: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    normal_mask = df[label_col].astype(str) == str(normal_label)
    normal_df = df[normal_mask].copy()
    attack_df = df[~normal_mask].copy()
    return normal_df, attack_df


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    if LABEL_COL not in df.columns:
        raise ValueError(f"Label column '{LABEL_COL}' not found in input.")

    # Drop categorical columns (except label)
    label_series = df[LABEL_COL]
    features_df = df.drop(columns=[LABEL_COL])
    features_df = drop_categorical(features_df)

    # Re-attach label for splitting
    df_proc = features_df.copy()
    df_proc[LABEL_COL] = label_series

    normal_df, attack_df = split_normal_attack(df_proc, LABEL_COL, NORMAL_LABEL)

    if len(normal_df) == 0:
        raise ValueError("No normal samples found. Check LABEL_COL/NORMAL_LABEL.")

    # Split normal data into train/test
    train_normal, test_normal = train_test_split(
        normal_df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=None,
    )

    # Combine test normal + all attacks
    test_df = pd.concat([test_normal, attack_df], axis=0).sample(
        frac=1.0, random_state=RANDOM_STATE
    )

    # Normalize using train_normal stats
    scaler = StandardScaler()
    feature_cols = [c for c in train_normal.columns if c != LABEL_COL]

    train_features = scaler.fit_transform(train_normal[feature_cols].values)
    test_features = scaler.transform(test_df[feature_cols].values)

    train_out = pd.DataFrame(train_features, columns=feature_cols)
    train_out[LABEL_COL] = train_normal[LABEL_COL].values

    test_out = pd.DataFrame(test_features, columns=feature_cols)
    test_out[LABEL_COL] = test_df[LABEL_COL].values

    # Split training (normal only) into 3 device subsets
    shuffled_train = train_out.sample(
        frac=1.0, random_state=RANDOM_STATE
    ).reset_index(drop=True)
    split_indices = np.array_split(np.arange(len(shuffled_train)), 3)
    device_splits = [
        shuffled_train.iloc[idx].reset_index(drop=True) for idx in split_indices
    ]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_path = os.path.join(OUTPUT_DIR, "train_normal.csv")
    test_path = os.path.join(OUTPUT_DIR, "test_normal_attacks.csv")
    device_paths = [
        os.path.join(OUTPUT_DIR, "device_1.csv"),
        os.path.join(OUTPUT_DIR, "device_2.csv"),
        os.path.join(OUTPUT_DIR, "device_3.csv"),
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
