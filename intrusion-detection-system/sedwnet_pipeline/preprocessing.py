import gc
from functools import partial
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif


class SafeLabelEncoder(BaseEstimator, TransformerMixin):
    """Deterministic label-to-id mapping with an unknown bucket at 0."""

    def __init__(self):
        self.mapper = {}
        self.unknown_token = 0

    def fit(self, y):
        y_series = pd.Series(y).astype(str)
        unique_labels = np.unique(y_series.values)
        unique_labels = np.sort(unique_labels)
        self.mapper = {label: idx + 1 for idx, label in enumerate(unique_labels)}
        return self

    def transform(self, y):
        return (
            pd.Series(y).astype(str)
            .map(self.mapper)
            .fillna(self.unknown_token)
            .astype(np.int32)
            .values
        )


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == "float64":
            df[col] = df[col].astype("float32")
        if df[col].dtype == "int64":
            df[col] = df[col].astype("int32")
    return df


def split_data(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def encode_target(y_raw: pd.Series) -> Tuple[np.ndarray, LabelEncoder, int, List[str]]:
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y_raw.astype(str))
    num_classes = len(le_target.classes_)
    class_names = le_target.classes_.tolist()
    return y_encoded, le_target, num_classes, class_names


def clean_features(
    X_raw: pd.DataFrame,
    *,
    valid_cat_cols: List[str],
    num_cols: List[str],
    log_cols: List[str],
) -> pd.DataFrame:
    X = X_raw.copy()

    for col in valid_cat_cols:
        X[col] = X[col].fillna("missing").replace("-", "missing").astype(str)

    for col in num_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X.replace([np.inf, -np.inf], 0, inplace=True)
    X = X.fillna(0)

    for col in log_cols:
        if col in X.columns:
            X[col] = np.log1p(pd.to_numeric(X[col], errors="coerce").fillna(0).clip(lower=0))

    X = optimize_dtypes(X)
    return X


def fit_preprocessors(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    *,
    valid_cat_cols: List[str],
    num_cols: List[str],
    target_k: int,
    seed: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Dict[str, SafeLabelEncoder],
    MinMaxScaler,
    SelectKBest,
    MinMaxScaler,
    List[str],
]:
    X_train_trim = X_train.reset_index(drop=True)
    y_train_trim = y_train
    del X_train, y_train
    gc.collect()

    encoders: Dict[str, SafeLabelEncoder] = {}
    for col in valid_cat_cols:
        le = SafeLabelEncoder()
        X_train_trim[col] = le.fit(X_train_trim[col]).transform(X_train_trim[col])
        X_val[col] = le.transform(X_val[col])
        X_test[col] = le.transform(X_test[col])
        encoders[col] = le

    scaler_num = MinMaxScaler()
    X_train_trim[num_cols] = scaler_num.fit_transform(X_train_trim[num_cols].values)
    X_val[num_cols] = scaler_num.transform(X_val[num_cols].values)
    X_test[num_cols] = scaler_num.transform(X_test[num_cols].values)

    feature_names = X_train_trim.columns.tolist()
    discrete_mask = np.array([c in valid_cat_cols for c in feature_names], dtype=bool)

    mi_scorer = partial(
        mutual_info_classif,
        discrete_features=discrete_mask,
        n_neighbors=3,
        random_state=seed,
        n_jobs=-1,
    )

    selector = SelectKBest(score_func=mi_scorer, k=min(target_k, X_train_trim.shape[1]))
    selector.fit(X_train_trim, y_train_trim)

    X_train_sel = selector.transform(X_train_trim)
    X_val_sel = selector.transform(X_val)
    X_test_sel = selector.transform(X_test)

    X_train_sel = np.nan_to_num(X_train_sel).astype(np.float32, copy=False)
    X_val_sel = np.nan_to_num(X_val_sel).astype(np.float32, copy=False)
    X_test_sel = np.nan_to_num(X_test_sel).astype(np.float32, copy=False)

    selected_mask = selector.get_support()
    final_features = X_train_trim.columns[selected_mask].tolist()

    final_scaler = MinMaxScaler()
    final_scaler.fit(X_train_sel)

    return (
        X_train_sel,
        X_val_sel,
        X_test_sel,
        encoders,
        scaler_num,
        selector,
        final_scaler,
        final_features,
    )


def build_feature_columns(X_raw: pd.DataFrame, cat_cols: List[str]) -> Tuple[List[str], List[str]]:
    valid_cat_cols = [c for c in cat_cols if c in X_raw.columns]
    num_cols = [c for c in X_raw.columns if c not in valid_cat_cols]
    return valid_cat_cols, num_cols
