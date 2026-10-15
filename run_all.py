"""Run the full AdaptiveLearn pipeline on the real OULAD data.

    python run_all.py            # full run (several minutes)
    python run_all.py --quick    # smaller hyper-parameter grids, for a smoke test

Needs the 7 OULAD CSV files in data/oulad_raw/ (see README for the download).
Writes tables, figures, metrics.json and models under results/, and the small
derived tables under data/processed/.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from adaptivelearn import viz  # noqa: E402
from adaptivelearn.clustering import run_clustering  # noqa: E402
from adaptivelearn.config import (ALL_FEATURES, CUTOFF_DAY, CUTOFF_SWEEP, MODEL_DIR,  # noqa: E402
                                  PROC_DIR, RAW_DIR, RESULTS_DIR, TABLE_DIR, TARGET_CLASS)
from adaptivelearn.inference import load_artifacts, profile  # noqa: E402
from adaptivelearn.oulad import Oulad  # noqa: E402
from adaptivelearn.preprocessing import group_split  # noqa: E402
from adaptivelearn.recommender import run_recommender  # noqa: E402
from adaptivelearn.supervised import (fairness_audit, run_ablation, run_classification,  # noqa: E402
                                      run_cutoff_sweep, run_regression)


def step(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(quick=False):
    if not (RAW_DIR / "studentVle.csv").exists():
        sys.exit(f"OULAD files not found in {RAW_DIR}. See README.md for the download link.")
    for d in (TABLE_DIR, MODEL_DIR, PROC_DIR):
        d.mkdir(parents=True, exist_ok=True)

    step("1/7 loading OULAD (10.6M click rows) and building day-%d features" % CUTOFF_DAY)
    o = Oulad()
    df = o.build(CUTOFF_DAY)

    step("2/7 exploratory analysis")
    num_cols = ["total_clicks", "active_days", "n_sites", "clicks_last_14d", "days_since_last_active",
                "n_submitted", "submission_rate", "mean_score_early", "num_of_prev_attempts",
                "studied_credits", "future_mean_score"]
    viz.outcomes(df)
    viz.missing(df, [c for c in ALL_FEATURES if c in df.columns] + ["future_mean_score"])
    viz.correlation(df, num_cols)
    df[ALL_FEATURES + ["final_result", "future_mean_score"]].describe(include="all").T.to_csv(
        TABLE_DIR / "data_describe.csv")
    train, test = group_split(df)
    overlap = len(set(train.id_student) & set(test.id_student))
    summary = {
        "n_enrolments_total": int(len(o.enrolments())),
        "n_students_total": int(o.info.id_student.nunique()),
        "n_click_rows_raw": 10655280,
        "cutoff_day": CUTOFF_DAY,
        "n_enrolments_at_cutoff": int(len(df)),
        "n_students_at_cutoff": int(df.id_student.nunique()),
        "n_features": len(ALL_FEATURES),
        "outcome_distribution": df[TARGET_CLASS].value_counts(normalize=True).round(3).to_dict(),
        "missing_pct": (df[ALL_FEATURES + ["future_mean_score"]].isna().mean() * 100).round(2)
                       [lambda s: s > 0].to_dict(),
        "regression_target_available_pct": float(df.future_mean_score.notna().mean() * 100),
        "train": int(len(train)), "test": int(len(test)), "students_in_both_splits": overlap,
    }

    step("3/7 clustering personas")
    cl = run_clustering(df)
    df["persona_id"] = cl["labels"]
    df["persona"] = df["persona_id"].map(cl["personas"])
    for name in ("sweep", "comparison", "profile", "outcome_mix"):
        cl[name].to_csv(TABLE_DIR / f"cluster_{name}.csv", index=name in ("profile", "outcome_mix"))
    viz.cluster_selection(cl["sweep"], cl["best_k"])
    viz.clusters_pca(cl["embedding"], cl["labels"], cl["personas"], cl["explained"])
    viz.persona_outcomes(cl["outcome_mix"], cl["profile"])
    train = train.merge(df[["key", "persona_id", "persona"]], on="key", how="left")
    test = test.merge(df[["key", "persona_id", "persona"]], on="key", how="left")

    step("4/7 classification (grouped CV, model zoo + tuning)")
    clf = run_classification(train, test, quick=quick)
    clf["table"].to_csv(TABLE_DIR / "classification_models.csv", index=False)
    clf["per_class"].to_csv(TABLE_DIR / "classification_per_class.csv")
    clf["importance"].to_csv(TABLE_DIR / "feature_importance.csv", index=False)
    viz.model_bars(clf["table"], "Outcome classification: model comparison",
                   "08_classification_comparison.png", ["cv_f1_macro", "test_f1_macro", "test_accuracy"],
                   ["CV macro-F1", "Test macro-F1", "Test accuracy"])
    viz.confusion(clf["confusion"], f"{clf['best_name']} (test set)")
    viz.importance(clf["importance"])

    step("5/7 ablation, early-prediction sweep, fairness, regression")
    abl = run_ablation(train, test)
    abl.to_csv(TABLE_DIR / "ablation.csv", index=False)
    viz.ablation(abl)
    sweep = run_cutoff_sweep(o.build, CUTOFF_SWEEP if not quick else [30, 90])
    sweep.to_csv(TABLE_DIR / "early_prediction_sweep.csv", index=False)
    viz.cutoff_sweep(sweep)
    fair = fairness_audit(clf["best"], test)
    fair.to_csv(TABLE_DIR / "fairness_audit.csv", index=False)
    viz.fairness(fair)
    reg = run_regression(train, test)
    reg["table"].to_csv(TABLE_DIR / "regression_models.csv", index=False)
    viz.regression_fit(reg["y_true"], reg["y_pred"], f"{reg['best_name']}: future assessment score")

    step("6/7 recommender (22 presentations, temporal hold-out)")
    personas_by_key = dict(zip(df["key"], df["persona_id"]))
    rec = run_recommender(o.visits, df, personas_by_key)
    rec["table"].to_csv(TABLE_DIR / "recommender_models.csv", index=False)
    rec["per_presentation"].to_csv(TABLE_DIR / "recommender_per_presentation.csv", index=False)
    viz.recommender_bars(rec["table"])

    step("7/7 saving models, processed tables and metrics")
    df["split"] = df["key"].isin(set(test["key"])).map({True: "test", False: "train"})
    df.to_csv(PROC_DIR / "enrolments_d90.csv.gz", index=False)
    o.assess.to_csv(PROC_DIR / "assessments.csv", index=False)
    o.site_catalogue().to_csv(PROC_DIR / "sites.csv", index=False)
    joblib.dump(clf["best"], MODEL_DIR / "classifier.joblib")
    joblib.dump(reg["best"], MODEL_DIR / "regressor.joblib")
    joblib.dump({"pipeline": cl["pipeline"], "personas": cl["personas"]}, MODEL_DIR / "clusterer.joblib")
    joblib.dump(rec["artifacts"], MODEL_DIR / "recommender.joblib")

    best_clf, best_reg = clf["table"].iloc[0], reg["table"].iloc[0]
    metrics = {
        "dataset": summary,
        "clustering": {"k": cl["best_k"], "stability_ari": cl["stability_ari"],
                       "personas": {str(k): v for k, v in cl["personas"].items()},
                       "pca_explained_variance": [float(x) for x in cl["explained"]]},
        "classification": {"selected_model": clf["best_name"],
                           "best_params": {k: {p: str(v) for p, v in d.items()}
                                           for k, d in clf["best_params"].items()},
                           **{k: (None if pd.isna(v) else float(v)) for k, v in best_clf.items()
                              if k != "model"}},
        "regression": {"selected_model": reg["best_name"], "n_train": reg["n_train"],
                       "n_test": reg["n_test"],
                       **{k: float(v) for k, v in best_reg.items() if k != "model"}},
        "recommender": {"n_presentations": rec["n_presentations"]},
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    art = load_artifacts()
    row = art["enrolments"][art["enrolments"].split == "test"].iloc[0]
    out = profile(row, art)
    print("\nExample test student -> predicted", out["predicted_result"],
          "| actual", row["final_result"], "| persona", out["persona"],
          f"| P(at-risk)={out['at_risk']:.0%}")
    for s in out["plan"]["steps"][:4]:
        print("  -", s["when"], "|", s["action"])
    step("done - see results/ and docs/")


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
