from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DateFeatures(BaseEstimator, TransformerMixin):
    """Convert a `date` column into year/month/day and drop the original date column."""

    def __init__(self, date_col: str = "date"):
        self.date_col = date_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.date_col in X.columns:
            X[self.date_col] = pd.to_datetime(X[self.date_col], errors="coerce")
            X["year"] = X[self.date_col].dt.year
            X["month"] = X[self.date_col].dt.month
            X["day"] = X[self.date_col].dt.day
            X = X.drop(columns=[self.date_col])
        return X


class HardDropColumns(BaseEstimator, TransformerMixin):
    """Drop known high-cardinality/low-value columns if present."""

    def __init__(self, cols=None):
        self.cols = cols or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        drop_cols = [c for c in self.cols if c in X.columns]
        if drop_cols:
            X = X.drop(columns=drop_cols)
        return X


class DropConstantColumns(BaseEstimator, TransformerMixin):
    """Drop columns with a single unique value (useful for constant columns like country)."""

    def __init__(self, min_unique: int = 2):
        self.min_unique = min_unique

    def fit(self, X, y=None):
        self.columns_to_drop_ = []
        for col in X.columns:
            nunique = X[col].nunique(dropna=False)
            if nunique < self.min_unique:
                self.columns_to_drop_.append(col)
        return self

    def transform(self, X):
        X = X.copy()
        cols = [c for c in self.columns_to_drop_ if c in X.columns]
        if cols:
            X = X.drop(columns=cols)
        return X

