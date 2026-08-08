from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


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
        return pd.Series(y).astype(str).map(self.mapper).fillna(self.unknown_token).astype(np.int32).values
