"""Instructor / student dashboard:  streamlit run app/streamlit_app.py
(run `python run_all.py` first - it trains the models this app loads)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from adaptivelearn.config import MODEL_DIR  # noqa: E402
from adaptivelearn.inference import load_artifacts, profile  # noqa: E402

st.set_page_config(page_title="AdaptiveLearn", page_icon="🎓", layout="wide")


@st.cache_resource
def artifacts():
    return load_artifacts()


st.title("🎓 AdaptiveLearn")
st.caption("Early-warning and personalised study plans from the Open University Learning Analytics "
           "Dataset (OULAD, real anonymised data) - predictions use only the first 90 days of a course")

if not (MODEL_DIR / "classifier.joblib").exists():
    st.error("Trained models not found. Run `python run_all.py` first.")
    st.stop()

art = artifacts()
df = art["enrolments"]
test = df[df.split == "test"]          # students the models have never seen

with st.sidebar:
    st.header("Pick a real (held-out) student")
    module = st.selectbox("Module", sorted(test.code_module.unique()))
    pres = st.selectbox("Presentation", sorted(test[test.code_module == module].code_presentation.unique()))
    pool = test[(test.code_module == module) & (test.code_presentation == pres)].reset_index(drop=True)
    idx = st.number_input(f"Student # (0 – {len(pool) - 1})", 0, len(pool) - 1, 0)
    base = pool.iloc[int(idx)].copy()

    st.header("What-if: change their behaviour")
    row = base.copy()
    row["total_clicks"] = st.slider("Total clicks by day 90", 0, 6000, int(min(base.total_clicks, 6000)))
    row["active_days"] = st.slider("Active days", 0, 90, int(min(base.active_days, 90)))
    row["clicks_last_14d"] = st.slider("Clicks in last 14 days", 0, 1500,
                                       int(min(base.clicks_last_14d, 1500)))
    row["days_since_last_active"] = st.slider("Days since last active", 0, 90,
                                              int(min(base.days_since_last_active, 90)))
    row["n_submitted"] = st.slider("Assessments submitted", 0, 6, int(min(base.n_submitted, 6)))
    default_score = 0 if pd.isna(base.mean_score_early) else int(base.mean_score_early)
    row["mean_score_early"] = st.slider("Mean score so far (%)", 0, 100, default_score)
    row["clicks_per_active_day"] = row["total_clicks"] / max(row["active_days"], 1)

out = profile(row, art)
plan = out["plan"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Predicted outcome", out["predicted_result"])
c2.metric("Risk of Fail / Withdrawal", f"{out['at_risk']:.0%}", plan["risk"], delta_color="inverse")
c3.metric("Predicted future score",
          f"{out['predicted_score']:.0f} %", plan["tier"])
c4.metric("Persona", out["persona"])
st.caption(f"Recorded real outcome of this student: **{base.final_result}** "
           f"(shown for transparency; the models never saw it).")

st.bar_chart(pd.Series(out["proba"]).reindex(["Distinction", "Pass", "Fail", "Withdrawn"]))

left, right = st.columns([1, 1.5])
with left:
    st.subheader("Recommended next resources")
    recs = out["recommendations"]
    if len(recs):
        st.dataframe(recs[["activity_type", "week_from", "id_site"]].rename(
            columns={"activity_type": "type", "week_from": "from week", "id_site": "site id"}),
            hide_index=True, width="stretch")
    else:
        st.info("No recommender available for this presentation.")
with right:
    st.subheader("Personalised study plan")
    st.dataframe(pd.DataFrame(plan["steps"])[["when", "action", "detail", "why"]],
                 hide_index=True, width="stretch")
