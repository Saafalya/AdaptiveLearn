"""Supervised learning: level classification and final-score regression."""
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier

from .config import (ALL_FEATURES, BEHAVIOUR_COLS, CATEGORICAL_COLS, ENGINEERED_CAT,
                     LEVELS, NUMERIC_FEATURES, SEED, TARGET_LEVEL,
                     TARGET_SCORE, TOPIC_COLS)
from .preprocessing import make_preprocessor

CV_FOLDS = 5


def _pipe(model, **prep_kwargs):
    return Pipeline([("prep", make_preprocessor(**prep_kwargs)), ("model", model)])


def classifier_zoo():
    return {
        "Baseline (most frequent)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=3000),
        "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=15),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
        "SVM (RBF)": SVC(probability=True, random_state=SEED),
    }


def regressor_zoo():
    return {
        "Baseline (mean)": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.05, max_iter=10000),
        "K-Nearest Neighbours": KNeighborsRegressor(n_neighbors=15),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=SEED),
        "SVR (RBF)": SVR(C=50, epsilon=1.0),
    }


def _auc(y_true, proba, classes):
    return roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=list(classes))


def _clf_metrics(pipe, X, y):
    pred = pipe.predict(X)
    out = {
        "accuracy": accuracy_score(y, pred),
        "precision_macro": precision_score(y, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y, pred, average="macro"),
    }
    out["roc_auc_ovr"] = (_auc(y, pipe.predict_proba(X), pipe.classes_)
                          if hasattr(pipe, "predict_proba") else np.nan)
    return out, pred


def run_classification(train, test):
    X_tr, y_tr = train[ALL_FEATURES], train[TARGET_LEVEL]
    X_te, y_te = test[ALL_FEATURES], test[TARGET_LEVEL]
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED)

    rows, fitted = [], {}

    def evaluate(name, pipe, cv_f1=None, cv_f1_std=None):
        if cv_f1 is None:
            res = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring=["accuracy", "f1_macro"], n_jobs=1)
            cv_f1, cv_f1_std = res["test_f1_macro"].mean(), res["test_f1_macro"].std()
            cv_acc = res["test_accuracy"].mean()
        else:
            cv_acc = np.nan
        pipe.fit(X_tr, y_tr)
        m, _ = _clf_metrics(pipe, X_te, y_te)
        rows.append({"model": name, "cv_f1_macro": cv_f1, "cv_f1_std": cv_f1_std,
                     "cv_accuracy": cv_acc, **{f"test_{k}": v for k, v in m.items()}})
        fitted[name] = pipe

    for name, model in classifier_zoo().items():
        evaluate(name, _pipe(model))

    # Small grid search on the two tree ensembles (macro-F1, same CV folds).
    grids = {
        "Random Forest (tuned)": (RandomForestClassifier(random_state=SEED, n_jobs=-1),
                                  {"model__n_estimators": [200, 400],
                                   "model__max_depth": [6, 10, None],
                                   "model__min_samples_leaf": [1, 3]}),
        "Gradient Boosting (tuned)": (GradientBoostingClassifier(random_state=SEED),
                                      {"model__n_estimators": [100, 200],
                                       "model__learning_rate": [0.05, 0.1],
                                       "model__max_depth": [2, 3]}),
    }
    best_params = {}
    for name, (model, grid) in grids.items():
        gs = GridSearchCV(_pipe(model), grid, cv=cv, scoring="f1_macro", n_jobs=1, refit=True)
        gs.fit(X_tr, y_tr)
        std = gs.cv_results_["std_test_score"][gs.best_index_]
        best_params[name] = gs.best_params_
        pipe = gs.best_estimator_
        m, _ = _clf_metrics(pipe, X_te, y_te)
        rows.append({"model": name, "cv_f1_macro": gs.best_score_, "cv_f1_std": std,
                     "cv_accuracy": np.nan, **{f"test_{k}": v for k, v in m.items()}})
        fitted[name] = pipe

    table = pd.DataFrame(rows).sort_values("cv_f1_macro", ascending=False).reset_index(drop=True)
    best_name = table.loc[0, "model"]          # selected on CV, never on the test set
    best = fitted[best_name]
    _, pred = _clf_metrics(best, X_te, y_te)
    cm = confusion_matrix(y_te, pred, labels=LEVELS)

    imp = permutation_importance(best, X_te, y_te, scoring="f1_macro", n_repeats=10,
                                 random_state=SEED, n_jobs=1)
    importance = (pd.DataFrame({"feature": ALL_FEATURES, "importance": imp.importances_mean,
                                "std": imp.importances_std})
                  .sort_values("importance", ascending=False).reset_index(drop=True))

    # Fairness slice (SDG 10): is the model equally good for every access group?
    fair = []
    for grp, part in test.groupby("internet_access"):
        m, _ = _clf_metrics(best, part[ALL_FEATURES], part[TARGET_LEVEL])
        fair.append({"internet_access": grp, "n": len(part),
                     "accuracy": m["accuracy"], "f1_macro": m["f1_macro"]})
    fairness = pd.DataFrame(fair).set_index("internet_access").loc[["limited", "moderate", "good"]].reset_index()

    return {"table": table, "best_name": best_name, "best": best, "confusion": cm,
            "importance": importance, "fairness": fairness, "best_params": best_params}


def _reg_metrics(y, pred):
    return {"rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "mae": float(mean_absolute_error(y, pred)),
            "r2": float(r2_score(y, pred))}


def run_regression(train, test):
    X_tr, y_tr = train[ALL_FEATURES], train[TARGET_SCORE]
    X_te, y_te = test[ALL_FEATURES], test[TARGET_SCORE]
    cv = KFold(CV_FOLDS, shuffle=True, random_state=SEED)

    rows, fitted = [], {}
    for name, model in regressor_zoo().items():
        pipe = _pipe(model)
        res = cross_validate(pipe, X_tr, y_tr, cv=cv,
                             scoring=["neg_root_mean_squared_error", "r2"], n_jobs=1)
        pipe.fit(X_tr, y_tr)
        m = _reg_metrics(y_te, pipe.predict(X_te))
        rows.append({"model": name,
                     "cv_rmse": -res["test_neg_root_mean_squared_error"].mean(),
                     "cv_rmse_std": res["test_neg_root_mean_squared_error"].std(),
                     "cv_r2": res["test_r2"].mean(),
                     **{f"test_{k}": v for k, v in m.items()}})
        fitted[name] = pipe

    table = pd.DataFrame(rows).sort_values("cv_rmse").reset_index(drop=True)
    best_name = table.loc[0, "model"]
    best = fitted[best_name]
    pred = best.predict(X_te)

    # Feature-group ablation with the selected model family.
    topic_num = TOPIC_COLS + ["mean_topic_score", "min_topic_score", "score_spread", "n_topics_attempted"]
    groups = {
        "Topic scores only": (topic_num, ENGINEERED_CAT),
        "Behaviour & context only": (BEHAVIOUR_COLS + ["engagement_index"], CATEGORICAL_COLS),
        "All features": (NUMERIC_FEATURES, CATEGORICAL_COLS + ENGINEERED_CAT),
    }
    abl = []
    for label, (num, cat) in groups.items():
        base_model = regressor_zoo()[best_name]
        pipe = _pipe(base_model, numeric=num, categorical=cat).fit(X_tr, y_tr)
        m = _reg_metrics(y_te, pipe.predict(X_te))
        abl.append({"feature_group": label, **m})
    ablation = pd.DataFrame(abl)

    return {"table": table, "best_name": best_name, "best": best,
            "y_true": y_te.to_numpy(), "y_pred": pred, "ablation": ablation}
