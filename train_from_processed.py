"""Train the app's models from the committed processed table, without the raw OULAD files.

    python train_from_processed.py           # full model search (a few minutes)
    python train_from_processed.py --quick   # smaller grids, for a fast first run

Uses data/processed/enrolments_d90.csv.gz, which already holds the day-90 features and the
same grouped train/test split as run_all.py. It rebuilds the classifier, regressor and persona
model. The next-resource recommender needs the raw click logs (studentVle.csv), so it is only
built by run_all.py; without it the app shows the other outputs and skips recommendations.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from adaptivelearn.clustering import run_clustering  # noqa: E402
from adaptivelearn.config import MODEL_DIR, PROC_DIR  # noqa: E402
from adaptivelearn.inference import load_artifacts, profile  # noqa: E402
from adaptivelearn.supervised import run_classification, run_regression  # noqa: E402


def step(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(quick=False):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROC_DIR / "enrolments_d90.csv.gz")
    step(f"loaded {len(df):,} enrolments from the processed table")

    step("1/3 clustering personas")
    cl = run_clustering(df.drop(columns=["persona_id", "persona"]))
    df["persona_id"], df["persona"] = cl["labels"], pd.Series(cl["labels"]).map(cl["personas"])

    train, test = df[df.split == "train"].copy(), df[df.split == "test"].copy()

    step("2/3 classification")
    clf = run_classification(train, test, quick=quick)
    print(clf["table"].round(3).to_string(index=False))

    step("3/3 regression")
    reg = run_regression(train, test)
    print(reg["table"].round(3).to_string(index=False))

    joblib.dump(clf["best"], MODEL_DIR / "classifier.joblib")
    joblib.dump(reg["best"], MODEL_DIR / "regressor.joblib")
    joblib.dump({"pipeline": cl["pipeline"], "personas": cl["personas"]}, MODEL_DIR / "clusterer.joblib")
    if not (MODEL_DIR / "recommender.joblib").exists():
        joblib.dump({}, MODEL_DIR / "recommender.joblib")   # placeholder until run_all.py is run

    art = load_artifacts()
    row = art["enrolments"][art["enrolments"].split == "test"].iloc[0]
    out = profile(row, art)
    step(f"done - example test student: predicted {out['predicted_result']}, actual "
         f"{row['final_result']}, persona {out['persona']}, P(at-risk)={out['at_risk']:.0%}")


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
