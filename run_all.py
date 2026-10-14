"""Run the full AdaptiveLearn pipeline end to end.

    python run_all.py

Generates the dataset, trains and evaluates every model, and writes tables,
figures, metrics.json and the saved models under results/.
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
from adaptivelearn.config import (ALL_FEATURES, BEHAVIOUR_COLS, DATA_DIR, MODEL_DIR,  # noqa: E402
                                  RESULTS_DIR, SEED, TABLE_DIR, TOPIC_COLS)
from adaptivelearn.data_generation import generate_dataset, save_dataset  # noqa: E402
from adaptivelearn.inference import load_artifacts, profile_student  # noqa: E402
from adaptivelearn.preprocessing import prepare, train_test  # noqa: E402
from adaptivelearn.recommender import run_recommender  # noqa: E402
from adaptivelearn.supervised import run_classification, run_regression  # noqa: E402


def step(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    for d in (TABLE_DIR, MODEL_DIR, DATA_DIR):
        d.mkdir(parents=True, exist_ok=True)

    step("1/6 generating dataset")
    students_raw, resources, ratings = generate_dataset(seed=SEED)
    save_dataset(students_raw, resources, ratings)
    students = prepare(students_raw)

    step("2/6 exploratory analysis")
    miss_cols = TOPIC_COLS + BEHAVIOUR_COLS
    viz.distributions(students_raw)
    viz.missing(students_raw, miss_cols)
    viz.correlation(students, TOPIC_COLS + BEHAVIOUR_COLS + ["engagement_index", "final_score"])
    summary = {
        "n_students": len(students_raw),
        "n_resources": len(resources),
        "n_ratings": len(ratings),
        "rating_density": len(ratings) / (len(students_raw) * len(resources)),
        "n_raw_columns": students_raw.shape[1],
        "n_model_features": len(ALL_FEATURES),
        "level_distribution": students_raw["level"].value_counts(normalize=True).round(3).to_dict(),
        "missing_pct": (students_raw[miss_cols].isna().mean() * 100).round(2).to_dict(),
        "study_hours_over_60": int((students_raw["weekly_study_hours"] > 60).sum()),
    }
    students_raw.describe().T.round(2).to_csv(TABLE_DIR / "data_describe.csv")

    step("3/6 clustering")
    cl = run_clustering(students)
    cl["sweep"].to_csv(TABLE_DIR / "cluster_k_sweep.csv", index=False)
    cl["comparison"].to_csv(TABLE_DIR / "cluster_algorithm_comparison.csv", index=False)
    cl["profile"].to_csv(TABLE_DIR / "cluster_profiles.csv")
    cl["level_mix"].to_csv(TABLE_DIR / "cluster_level_mix.csv")
    viz.cluster_selection(cl["sweep"], cl["best_k"])
    viz.clusters_pca(cl["embedding"], cl["labels"], cl["personas"], cl["explained"])
    viz.persona_heatmap(cl["profile"])

    step("4/6 classification + regression")
    train, test = train_test(students)
    clf = run_classification(train, test)
    reg = run_regression(train, test)
    clf["table"].to_csv(TABLE_DIR / "classification_models.csv", index=False)
    clf["importance"].to_csv(TABLE_DIR / "feature_importance.csv", index=False)
    clf["fairness"].to_csv(TABLE_DIR / "fairness_by_internet_access.csv", index=False)
    reg["table"].to_csv(TABLE_DIR / "regression_models.csv", index=False)
    reg["ablation"].to_csv(TABLE_DIR / "regression_ablation.csv", index=False)
    viz.model_bars(clf["table"], ["cv_f1_macro", "test_f1_macro", "test_accuracy"],
                   "Level classification: model comparison", "07_classification_comparison.png",
                   ["CV macro-F1", "Test macro-F1", "Test accuracy"])
    viz.confusion(clf["confusion"], f"{clf['best_name']} (test set)")
    viz.importance(clf["importance"])
    viz.pred_vs_actual(reg["y_true"], reg["y_pred"], f"{reg['best_name']}: final score")
    viz.fairness(clf["fairness"])

    step("5/6 recommender")
    rec = run_recommender(students, resources, ratings)
    rec["table"].to_csv(TABLE_DIR / "recommender_models.csv", index=False)
    rec["alpha_sweep"].to_csv(TABLE_DIR / "recommender_alpha_sweep.csv", index=False)
    viz.recommender_bars(rec["table"])

    step("6/6 saving models + metrics")
    joblib.dump(clf["best"], MODEL_DIR / "classifier.joblib")
    joblib.dump(reg["best"], MODEL_DIR / "regressor.joblib")
    joblib.dump({"pipeline": cl["pipeline"], "personas": cl["personas"]}, MODEL_DIR / "clusterer.joblib")
    joblib.dump({"model": rec["content_model"], "resources": resources}, MODEL_DIR / "recommender.joblib")

    best_clf = clf["table"].iloc[0]
    best_reg = reg["table"].iloc[0]
    metrics = {
        "dataset": summary,
        "clustering": {"k": cl["best_k"], "stability_ari": cl["stability_ari"],
                       "personas": {str(k): v for k, v in cl["personas"].items()},
                       "pca_explained_variance": [float(x) for x in cl["explained"]]},
        "classification": {"selected_model": clf["best_name"],
                           "best_params": {k: {p: (v if v is not None else "None") for p, v in d.items()}
                                           for k, d in clf["best_params"].items()},
                           **{k: (None if pd.isna(v) else float(v)) for k, v in best_clf.items() if k != "model"}},
        "regression": {"selected_model": reg["best_name"],
                       **{k: float(v) for k, v in best_reg.items() if k != "model"}},
        "recommender": {"n_train_ratings": rec["n_train"], "n_test_ratings": rec["n_test"],
                        "train_density": rec["density"]},
        "split": {"train": len(train), "test": len(test)},
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # smoke-test the saved artefacts with one example student
    art = load_artifacts()
    example = {"internet_access": "limited", "preferred_format": "video", "weekly_study_hours": 6,
               "logins_per_week": 3, "avg_session_min": 25, "assignments_completed_pct": 55,
               "quiz_attempts": 4, "forum_posts": 0, "days_since_last_active": 9, "prior_cgpa": 6.1,
               "score_python": 88, "score_sql": 42, "score_statistics": 55, "score_linalg": 60,
               "score_dsa": 35, "score_ml": 76}
    out = profile_student(example, art)
    print("\nExample student ->", out["level"], f"(predicted score {out['predicted_score']:.1f}),",
          out["persona"])
    for s in out["learning_path"][:8]:
        print("  ", s["topic"], "|", s["action"], "|", s["resource"])
    step("done - see results/ and docs/")


if __name__ == "__main__":
    main()
