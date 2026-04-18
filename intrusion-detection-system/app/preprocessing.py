from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

LOG_COLS = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "src_pkts",
    "dst_pkts",
    "http_request_body_len",
    "http_response_body_len",
    "missed_bytes",
]


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
            pd.Series(y).astype(str).map(self.mapper).fillna(self.unknown_token).astype(np.int32).values
        )


def transform_with_pipeline(
    raw_records: list[dict],
    *,
    pipeline: dict,
    final_features: list[str],
) -> np.ndarray:
    """Transform raw records into model-ready matrix with saved training pipeline."""
    df = pd.DataFrame(raw_records)

    encoders = pipeline["encoders"]
    scaler_num = pipeline["scaler_num"]
    final_scaler = pipeline["final_scaler"]
    valid_cat_cols = pipeline["valid_cat_cols"]
    num_cols = pipeline["num_cols"]

    required_cols = list(dict.fromkeys(list(valid_cat_cols) + list(num_cols) + list(final_features)))

    for col in required_cols:
        if col in df.columns:
            continue
        if col in valid_cat_cols:
            df[col] = "missing"
        else:
            df[col] = 0

    df = df[required_cols]

    for col in valid_cat_cols:
        if col not in df.columns:
            continue
        df[col] = df[col].fillna("missing").replace("-", "missing").astype(str)
        encoder = encoders.get(col)
        if encoder is None:
            df[col] = 0
        else:
            df[col] = encoder.transform(df[col])

    for col in num_cols:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in final_features:
        if col not in df.columns or col in valid_cat_cols:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.replace([np.inf, -np.inf], 0, inplace=True)
    df = df.fillna(0)

    for col in LOG_COLS:
        if col in df.columns:
            df[col] = np.log1p(pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0))

    df[num_cols] = scaler_num.transform(df[num_cols].values)

    missing_final = [c for c in final_features if c not in df.columns]
    if missing_final:
        raise RuntimeError(
            "Missing required final features after preprocessing: "
            + ", ".join(missing_final[:20])
            + (" ..." if len(missing_final) > 20 else "")
        )

    x_selected = df[final_features].values
    x_scaled = final_scaler.transform(x_selected)
    x_scaled = np.nan_to_num(x_scaled).astype(np.float32, copy=False)
    return x_scaled
