"""Supervised learning on OULAD.

* classification : final_result (Distinction / Pass / Fail / Withdrawn)
* regression     : mean score on assessments due after the cut-off
* analyses       : feature-group ablation, early-prediction sweep, fairness audit
"""
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .config import (ALL_FEATURES, ASSESSMENT_NUM, AT_RISK, CATEGORICAL_FEATURES,
                     DEMOGRAPHIC_CAT, DEMOGRAPHIC_NUM, ENGAGEMENT_NUM, FAIRNESS_ATTRS,
                     RESULT_ORDER, SEED, TARGET_CLASS, TARGET_SCORE)
from .preprocessing import cv_splitter, group_split, make_preprocessor

N_JOBS = -1


def _pipe(model, numeric=None, categorical=None):
    return Pipeline([("prep", make_preprocessor(numeric, categorical)), ("model", model)])


def classifier_zoo():
    return {
        "Baseline (most frequent)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=25, n_jobs=N_JOBS),
        "Decision Tree": DecisionTreeClassifier(max_depth=7, min_samples_leaf=20,
                                                class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=250, min_samples_leaf=3,
                                                class_weight="balanced_subsample",
                                                random_state=SEED, n_jobs=N_JOBS),
        "Hist. Gradient Boosting": HistGradientBoostingClassifier(class_weight="balanced",
                                                                  random_state=SEED),
    }


def regressor_zoo():
    return {
        "Baseline (mean)": DummyRegressor(strategy="mean"),
        "Ridge": Ridge(alpha=10.0),
        "Lasso": Lasso(alpha=0.05, max_iter=10000),
        "K-Nearest Neighbours": KNeighborsRegressor(n_neighbors=25, n_jobs=N_JOBS),
        "Random Forest": RandomForestRegressor(n_estimators=250, min_samples_leaf=3,
                                               random_state=SEED, n_jobs=N_JOBS),
        "Hist. Gradient Boosting": HistGradientBoostingRegressor(random_state=SEED),
    }


# ------------------------------------------------------------------ metrics
def class_metrics(pipe, X, y):
    proba = pipe.predict_proba(X) if hasattr(pipe, "predict_proba") else None
    pred = pipe.predict(X)
    out = {
        "accuracy": accuracy_score(y, pred),
        "precision_macro": precision_score(y, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y, pred, average="macro"),
        "roc_auc_ovr": np.nan, "at_risk_auc": np.nan, "at_risk_ap": np.nan,
    }
    if proba is not None:
        classes = list(pipe.classes_)
        if len(classes) > 2 and len(set(y)) == len(classes):
            out["roc_auc_ovr"] = roc_auc_score(y, proba, multi_class="ovr", average="macro",
                                               labels=classes)
        p_risk = proba[:, [classes.index(c) for c in AT_RISK]].sum(axis=1)
        is_risk = y.isin(AT_RISK).to_numpy()
        out["at_risk_auc"] = roc_auc_score(is_risk, p_risk)
        out["at_risk_ap"] = average_precision_score(is_risk, p_risk)
    return out, pred


def reg_metrics(y, pred):
    return {"rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "mae": float(mean_absolute_error(y, pred)),
            "r2": float(r2_score(y, pred))}


# ----------------------------------------------------------- classification
def run_classification(train, test, quick=False):
    X_tr, y_tr, g_tr = train[ALL_FEATURES], train[TARGET_CLASS], train["id_student"]
    X_te, y_te = test[ALL_FEATURES], test[TARGET_CLASS]
    cv = cv_splitter()
    rows, fitted = [], {}

    def record(name, pipe, cv_mean, cv_std):
        pipe.fit(X_tr, y_tr)
        m, _ = class_metrics(pipe, X_te, y_te)
        rows.append({"model": name, "cv_f1_macro": cv_mean, "cv_f1_std": cv_std,
                     **{f"test_{k}": v for k, v in m.items()}})
        fitted[name] = pipe

    for name, model in classifier_zoo().items():
        pipe = _pipe(model)
        res = cross_validate(pipe, X_tr, y_tr, groups=g_tr, cv=cv, scoring="f1_macro")
        record(name, pipe, res["test_score"].mean(), res["test_score"].std())

    grids = {
        "Random Forest (tuned)": (
            RandomForestClassifier(class_weight="balanced_subsample", random_state=SEED, n_jobs=N_JOBS),
            {"model__n_estimators": [200, 400], "model__max_depth": [10, 16, None],
             "model__min_samples_leaf": [3, 8]}),
        "Hist. Gradient Boosting (tuned)": (
            HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED),
            {"model__learning_rate": [0.03, 0.06, 0.1], "model__max_depth": [3, 6, None],
             "model__l2_regularization": [0.0, 1.0]}),
    }
    best_params = {}
    if quick:
        grids = {k: (m, {kk: vv[:1] for kk, vv in g.items()}) for k, (m, g) in grids.items()}
    for name, (model, grid) in grids.items():
        gs = GridSearchCV(_pipe(model), grid, cv=cv, scoring="f1_macro", refit=True)
        gs.fit(X_tr, y_tr, groups=g_tr)
        best_params[name] = gs.best_params_
        m, _ = class_metrics(gs.best_estimator_, X_te, y_te)
        rows.append({"model": name, "cv_f1_macro": gs.best_score_,
                     "cv_f1_std": gs.cv_results_["std_test_score"][gs.best_index_],
                     **{f"test_{k}": v for k, v in m.items()}})
        fitted[name] = gs.best_estimator_

    table = pd.DataFrame(rows).sort_values("cv_f1_macro", ascending=False).reset_index(drop=True)
    best_name = table.loc[0, "model"]          # chosen on CV, never on the test set
    best = fitted[best_name]
    _, pred = class_metrics(best, X_te, y_te)
    cm = confusion_matrix(y_te, pred, labels=RESULT_ORDER)
    per_class = pd.DataFrame({
        "precision": precision_score(y_te, pred, labels=RESULT_ORDER, average=None, zero_division=0),
        "recall": recall_score(y_te, pred, labels=RESULT_ORDER, average=None, zero_division=0),
        "f1": f1_score(y_te, pred, labels=RESULT_ORDER, average=None),
        "support": pd.Series(y_te).value_counts().reindex(RESULT_ORDER).to_numpy(),
    }, index=RESULT_ORDER)

    sample = test.sample(min(3000, len(test)), random_state=SEED)
    imp = permutation_importance(best, sample[ALL_FEATURES], sample[TARGET_CLASS],
                                 scoring="f1_macro", n_repeats=5, random_state=SEED, n_jobs=1)
    importance = (pd.DataFrame({"feature": ALL_FEATURES, "importance": imp.importances_mean,
                                "std": imp.importances_std})
                  .sort_values("importance", ascending=False).reset_index(drop=True))

    return {"table": table, "best_name": best_name, "best": best, "confusion": cm,
            "per_class": per_class, "importance": importance, "best_params": best_params,
            "fitted": fitted, "test_pred": pred}


def run_ablation(train, test):
    groups = {
        "Demographics only": (DEMOGRAPHIC_NUM, DEMOGRAPHIC_CAT),
        "Engagement (VLE clicks) only": (ENGAGEMENT_NUM, []),
        "Assessments only": (ASSESSMENT_NUM, []),
        "Engagement + assessments": (ENGAGEMENT_NUM + ASSESSMENT_NUM, []),
        "All features": (ENGAGEMENT_NUM + ASSESSMENT_NUM + DEMOGRAPHIC_NUM, CATEGORICAL_FEATURES),
    }
    rows = []
    for label, (num, cat) in groups.items():
        cols = list(num) + list(cat)
        pipe = _pipe(HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED),
                     numeric=num, categorical=cat).fit(train[cols], train[TARGET_CLASS])
        m, _ = class_metrics(pipe, test[cols], test[TARGET_CLASS])
        rows.append({"feature_group": label, "n_features": len(cols), **m})
    return pd.DataFrame(rows)


def run_cutoff_sweep(builder, cutoffs):
    """Re-build features at each cut-off day and measure how early risk is detectable."""
    rows = []
    for c in cutoffs:
        df = builder(c)
        tr, te = group_split(df)
        pipe = _pipe(HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED))
        pipe.fit(tr[ALL_FEATURES], tr[TARGET_CLASS])
        m, _ = class_metrics(pipe, te[ALL_FEATURES], te[TARGET_CLASS])
        rows.append({"cutoff_day": c, "n_students": len(df),
                     "at_risk_share": float(df[TARGET_CLASS].isin(AT_RISK).mean()), **m})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ fairness
def fairness_audit(best, test, attrs=FAIRNESS_ATTRS, min_n=150):
    """Early-warning view: flag a student as at-risk if predicted Fail/Withdrawn."""
    pred = pd.Series(best.predict(test[ALL_FEATURES]), index=test.index)
    flag = pred.isin(AT_RISK)
    actual = test[TARGET_CLASS].isin(AT_RISK)
    rows = []
    for attr in attrs:
        for val, idx in test.groupby(attr).groups.items():
            if len(idx) < min_n:
                continue
            a, f = actual.loc[idx], flag.loc[idx]
            rows.append({
                "attribute": attr, "group": val, "n": len(idx),
                "actual_at_risk_rate": a.mean(),
                "flag_rate": f.mean(),
                "recall_at_risk (TPR)": (f & a).sum() / max(a.sum(), 1),
                "false_alarm_rate (FPR)": (f & ~a).sum() / max((~a).sum(), 1),
                "accuracy_4class": accuracy_score(test.loc[idx, TARGET_CLASS], pred.loc[idx]),
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- regression
def run_regression(train, test):
    tr, te = train[train[TARGET_SCORE].notna()], test[test[TARGET_SCORE].notna()]
    X_tr, y_tr, g_tr = tr[ALL_FEATURES], tr[TARGET_SCORE], tr["id_student"]
    X_te, y_te = te[ALL_FEATURES], te[TARGET_SCORE]
    folds = list(cv_splitter().split(X_tr, tr[TARGET_CLASS], groups=g_tr))

    rows, fitted = [], {}
    for name, model in regressor_zoo().items():
        pipe = _pipe(model)
        res = cross_validate(pipe, X_tr, y_tr, cv=folds,
                             scoring=["neg_root_mean_squared_error", "r2"])
        pipe.fit(X_tr, y_tr)
        m = reg_metrics(y_te, pipe.predict(X_te))
        rows.append({"model": name,
                     "cv_rmse": -res["test_neg_root_mean_squared_error"].mean(),
                     "cv_rmse_std": res["test_neg_root_mean_squared_error"].std(),
                     "cv_r2": res["test_r2"].mean(),
                     **{f"test_{k}": v for k, v in m.items()}})
        fitted[name] = pipe
    table = pd.DataFrame(rows).sort_values("cv_rmse").reset_index(drop=True)
    best_name = table.loc[0, "model"]
    best = fitted[best_name]
    return {"table": table, "best_name": best_name, "best": best,
            "y_true": y_te.to_numpy(), "y_pred": best.predict(X_te), "n_train": len(tr),
            "n_test": len(te)}
