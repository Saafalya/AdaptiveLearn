"""Cleaning, feature engineering and the scikit-learn preprocessing transformer.

Everything that *learns* from data (imputation values, scaling statistics,
one-hot categories) lives inside the sklearn ColumnTransformer so it is fitted
on training folds only - no leakage into validation/test data.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, SEED,
                     TARGET_LEVEL, TARGET_SCORE, TOPICS, TOPIC_COLS)

MAX_STUDY_HOURS = 60  # more than 60 h/week is treated as a data-entry error


def clean(df):
    """Domain-rule cleaning: clip bounded scores, null impossible values."""
    d = df.copy()
    for col in TOPIC_COLS:
        d[col] = d[col].clip(0, 100)
    d["assignments_completed_pct"] = d["assignments_completed_pct"].clip(0, 100)
    d["prior_cgpa"] = d["prior_cgpa"].clip(0, 10)
    d.loc[d["weekly_study_hours"] > MAX_STUDY_HOURS, "weekly_study_hours"] = np.nan
    return d


def add_features(df):
    """Row-wise engineered features (no dataset-level statistics -> no leakage)."""
    d = df.copy()
    scores = d[TOPIC_COLS].to_numpy(dtype=float)
    observed = ~np.isnan(scores)
    d["n_topics_attempted"] = observed.sum(axis=1)
    d["mean_topic_score"] = d[TOPIC_COLS].mean(axis=1)
    d["min_topic_score"] = d[TOPIC_COLS].min(axis=1)
    d["score_spread"] = d[TOPIC_COLS].max(axis=1) - d["min_topic_score"]

    names = np.array(list(TOPICS), dtype=object)
    weakest = names[np.where(observed, scores, np.inf).argmin(axis=1)]
    weakest = np.where(observed.any(axis=1), weakest, np.nan)
    d["weakest_topic"] = pd.Series(weakest, index=d.index, dtype=object)

    d["engagement_index"] = pd.concat([
        d["logins_per_week"] / 21,
        d["avg_session_min"] / 120,
        d["assignments_completed_pct"] / 100,
    ], axis=1).mean(axis=1)
    return d


def prepare(df):
    """clean -> add_features. Returns the frame used by every model."""
    return add_features(clean(df))


def make_preprocessor(numeric=None, categorical=None):
    numeric = NUMERIC_FEATURES if numeric is None else numeric
    categorical = CATEGORICAL_FEATURES if categorical is None else categorical
    num = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([("num", num, list(numeric)), ("cat", cat, list(categorical))])


def train_test(df, test_size=0.2, seed=SEED):
    """Stratified (on level) hold-out split of the prepared frame."""
    return train_test_split(df, test_size=test_size, random_state=seed,
                            stratify=df[TARGET_LEVEL])


def xy(df):
    return df[ALL_FEATURES], df[TARGET_LEVEL], df[TARGET_SCORE]
