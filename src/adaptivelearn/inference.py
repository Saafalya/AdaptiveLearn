"""Score one real (or what-if modified) enrolment with the saved models."""
import joblib
import numpy as np
import pandas as pd

from .clustering import predict_cluster
from .config import ALL_FEATURES, AT_RISK, MODEL_DIR, PROC_DIR, TARGET_CLASS
from .recommender import recommend_for
from .study_plan import build_study_plan


def load_artifacts(directory=MODEL_DIR, processed=PROC_DIR):
    art = {name: joblib.load(directory / f"{name}.joblib")
           for name in ("classifier", "regressor", "clusterer", "recommender")}
    art["enrolments"] = pd.read_csv(processed / "enrolments_d90.csv.gz")
    art["assessments"] = pd.read_csv(processed / "assessments.csv")
    art["sites"] = pd.read_csv(processed / "sites.csv")
    return art


def peer_benchmark(enrolments, module):
    peers = enrolments[(enrolments.code_module == module)
                       & enrolments[TARGET_CLASS].isin(["Pass", "Distinction"])]
    return float(peers["clicks_last_14d"].median()) if len(peers) else None


def profile(row, art, cutoff=90, top_k=5):
    """row: pandas Series / dict with the model feature columns (+ key, module, presentation)."""
    row = pd.Series(row)
    frame = pd.DataFrame([row])[ALL_FEATURES + ["key"]]
    clf, reg = art["classifier"], art["regressor"]
    proba = dict(zip(clf.classes_, clf.predict_proba(frame[ALL_FEATURES])[0]))
    pred_class = max(proba, key=proba.get)
    pred_score = float(np.clip(reg.predict(frame[ALL_FEATURES])[0], 0, 100))

    persona_id = int(predict_cluster(art["clusterer"], row_for_cluster(row))[0])
    persona = art["clusterer"]["personas"][persona_id]

    pres_key = f"{row['code_module']}_{row['code_presentation']}"
    rec_art = art["recommender"].get(pres_key)
    recs = (recommend_for(rec_art, int(row["key"]), art["sites"], top_k=top_k, persona=persona_id)
            if rec_art is not None else pd.DataFrame())
    up = art["assessments"]
    upcoming = up[(up.code_module == row["code_module"])
                  & (up.code_presentation == row["code_presentation"])
                  & (up.date > cutoff)].sort_values("date")
    plan = build_study_plan(row, proba, pred_score, persona, recs, upcoming,
                            peer_benchmark(art["enrolments"], row["code_module"]))
    return {"proba": proba, "predicted_result": pred_class, "predicted_score": pred_score,
            "persona": persona, "recommendations": recs, "plan": plan,
            "at_risk": sum(proba.get(c, 0) for c in AT_RISK)}


def row_for_cluster(row):
    return pd.DataFrame([row])
