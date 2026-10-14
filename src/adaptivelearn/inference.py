"""Score a single (possibly brand-new) student with the saved models."""
import joblib
import numpy as np
import pandas as pd

from .config import ALL_FEATURES, MODEL_DIR, TOPICS, TOPIC_COLS
from .clustering import CLUSTER_COLS
from .learning_path import build_learning_path
from .preprocessing import prepare
from .recommender import pair_features


def load_artifacts(directory=MODEL_DIR):
    return {name: joblib.load(directory / f"{name}.joblib")
            for name in ("classifier", "regressor", "clusterer", "recommender")}


def profile_student(raw, artifacts, top_k=5):
    """raw: dict with the student columns (topic scores may be None/NaN)."""
    df = pd.DataFrame([raw])
    for col in TOPIC_COLS:  # None -> NaN ("not attempted")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    frame = prepare(df)
    clf, reg = artifacts["classifier"], artifacts["regressor"]
    proba = dict(zip(clf.classes_, clf.predict_proba(frame[ALL_FEATURES])[0]))
    level = clf.predict(frame[ALL_FEATURES])[0]
    predicted_score = float(np.clip(reg.predict(frame[ALL_FEATURES])[0], 0, 100))

    cl = artifacts["clusterer"]
    cluster = int(cl["pipeline"].predict(frame[CLUSTER_COLS])[0])

    rec = artifacts["recommender"]
    resources = rec["resources"]
    n = len(resources)
    feats = pair_features(frame, resources, np.zeros(n, dtype=int), np.arange(n))
    ratings = dict(zip(resources["resource_id"], np.clip(rec["model"].predict(feats), 1, 5)))
    top = (resources.assign(predicted_rating=resources["resource_id"].map(ratings))
           .sort_values("predicted_rating", ascending=False).head(top_k))

    topic_scores = {t: raw.get(f"score_{t}") for t in TOPICS}
    path = build_learning_path(topic_scores, ratings, resources)
    return {"level": level, "level_proba": proba, "predicted_score": predicted_score,
            "persona": cl["personas"][cluster], "cluster": cluster,
            "top_resources": top[["title", "topic", "level", "rtype", "duration_min", "predicted_rating"]],
            "learning_path": path}
