import gc
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split


UNK_TOKEN = "__UNK__"


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == "float64":
            df[col] = df[col].astype("float32")
        if df[col].dtype == "int64":
            df[col] = df[col].astype("int32")
    return df


def clean_categorical(series: pd.Series) -> pd.Series:
    return series.fillna("missing").replace("-", "missing").astype(str)


def fit_label_encoder_train_only(train_series: pd.Series) -> LabelEncoder:
    le = LabelEncoder()
    cleaned = clean_categorical(train_series)
    classes = pd.Index(cleaned.unique()).astype(str).tolist()
    if UNK_TOKEN not in classes:
        classes.append(UNK_TOKEN)
    le.fit(classes)
    return le


def transform_with_unk(le: LabelEncoder, series: pd.Series) -> np.ndarray:
    cleaned = clean_categorical(series)
    known = set(le.classes_.tolist())
    cleaned = cleaned.where(cleaned.isin(known), other=UNK_TOKEN)
    return le.transform(cleaned)


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def fit_transform_features(
    X_train_df: pd.DataFrame,
    X_val_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
    *,
    log_cols: List[str],
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Dict[str, LabelEncoder],
    MinMaxScaler,
    List[str],
    List[str],
]:
    valid_cat_cols = [c for c in X_train_df.columns if X_train_df[c].dtype == object]
    num_cols = [c for c in X_train_df.columns if c not in valid_cat_cols]

    encoders: Dict[str, LabelEncoder] = {}
    for col in valid_cat_cols:
        le = fit_label_encoder_train_only(X_train_df[col])
        encoders[col] = le
        X_train_df[col] = transform_with_unk(le, X_train_df[col]).astype("int32")
        X_val_df[col] = transform_with_unk(le, X_val_df[col]).astype("int32")
        X_test_df[col] = transform_with_unk(le, X_test_df[col]).astype("int32")

    for split_df in (X_train_df, X_val_df, X_test_df):
        for col in num_cols:
            split_df[col] = pd.to_numeric(split_df[col], errors="coerce").fillna(0).astype("float32")

    log_cols_present = [c for c in log_cols if c in X_train_df.columns]
    for split_df in (X_train_df, X_val_df, X_test_df):
        for col in log_cols_present:
            split_df[col] = np.log1p(np.maximum(split_df[col].astype("float32"), 0.0))

    X_cols = X_train_df.columns.tolist()

    X_train = X_train_df.values
    X_val = X_val_df.values
    X_test = X_test_df.values

    return X_train, X_val, X_test, encoders, MinMaxScaler(), X_cols, num_cols


def select_features_chi2(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    *,
    target_k: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, SelectKBest, List[int]]:
    scaler = MinMaxScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm = scaler.transform(X_val)
    X_test_norm = scaler.transform(X_test)

    selector = SelectKBest(chi2, k=min(target_k, X_train.shape[1]))
    X_train_sel = selector.fit_transform(X_train_norm, y_train)
    X_val_sel = selector.transform(X_val_norm)
    X_test_sel = selector.transform(X_test_norm)

    del X_train_norm, X_val_norm, X_test_norm
    gc.collect()

    selected_indices = selector.get_support(indices=True)
    return X_train_sel, X_val_sel, X_test_sel, scaler, selector, list(selected_indices)
