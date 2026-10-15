"""Scikit-learn preprocessing transformer and leakage-safe splitting.

Everything that *learns* from data (imputation values, scaling statistics,
one-hot categories) lives inside the ColumnTransformer, so it is fitted on
training folds only. Splits are grouped by student: OULAD contains students
who enrolled in several presentations, and the same person must never be in
both train and test.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, SEED,
                     TARGET_CLASS)


def make_preprocessor(numeric=None, categorical=None):
    numeric = NUMERIC_FEATURES if numeric is None else numeric
    categorical = CATEGORICAL_FEATURES if categorical is None else categorical
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]), list(numeric)))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False)),
        ]), list(categorical)))
    return ColumnTransformer(transformers, sparse_threshold=0)


def group_split(df, test_frac=0.2, seed=SEED):
    """Stratified (on outcome) hold-out split with students kept in one side only."""
    n_splits = int(round(1 / test_frac))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, test_idx = next(sgkf.split(df, df[TARGET_CLASS], groups=df["id_student"]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def cv_splitter(n_splits=5, seed=SEED):
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def features(df):
    return df[ALL_FEATURES]


def log_counts(df, cols):
    """log1p of heavy-tailed count columns (row-wise, no leakage)."""
    out = df[cols].copy()
    for c in cols:
        out[c] = np.log1p(out[c].clip(lower=0))
    return out
