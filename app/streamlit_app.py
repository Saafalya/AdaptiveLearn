"""Interactive demo:  streamlit run app/streamlit_app.py   (run `python run_all.py` first)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from adaptivelearn.config import MODEL_DIR, TOPICS  # noqa: E402
from adaptivelearn.inference import load_artifacts, profile_student  # noqa: E402

st.set_page_config(page_title="AdaptiveLearn", page_icon="🎓", layout="wide")


@st.cache_resource
def artifacts():
    return load_artifacts()


st.title("🎓 AdaptiveLearn")
st.caption("ML-based personalised learning recommendations from student performance analytics")

if not (MODEL_DIR / "classifier.joblib").exists():
    st.error("Trained models not found. Run `python run_all.py` first.")
    st.stop()

with st.sidebar:
    st.header("Topic scores (%)")
    raw = {}
    for key, name in TOPICS.items():
        attempted = st.checkbox(f"{name} attempted", value=True, key=f"att_{key}")
        raw[f"score_{key}"] = st.slider(name, 0, 100, 60, key=f"s_{key}") if attempted else None
    st.header("Behaviour & context")
    raw.update({
        "weekly_study_hours": st.slider("Study hours / week", 0.0, 40.0, 8.0),
        "logins_per_week": st.slider("Logins / week", 0, 21, 5),
        "avg_session_min": st.slider("Avg session (min)", 5, 120, 35),
        "assignments_completed_pct": st.slider("Assignments completed (%)", 0, 100, 70),
        "quiz_attempts": st.slider("Quiz attempts", 0, 30, 6),
        "forum_posts": st.slider("Forum posts", 0, 20, 2),
        "days_since_last_active": st.slider("Days since last active", 0, 60, 3),
        "prior_cgpa": st.slider("Prior CGPA (0-10)", 3.0, 10.0, 6.8),
        "internet_access": st.selectbox("Internet access", ["limited", "moderate", "good"], index=1),
        "preferred_format": st.selectbox("Preferred format", ["video", "reading", "practice"]),
    })

out = profile_student(raw, artifacts())

c1, c2, c3 = st.columns(3)
c1.metric("Predicted level", out["level"].title())
c2.metric("Predicted final score", f"{out['predicted_score']:.1f} %")
c3.metric("Learner persona", out["persona"])

st.progress(float(out["level_proba"].get(out["level"], 0)),
            text=" · ".join(f"{k}: {v:.0%}" for k, v in out["level_proba"].items()))

left, right = st.columns([1, 1.4])
with left:
    st.subheader("Top recommended resources")
    st.dataframe(out["top_resources"].assign(predicted_rating=lambda d: d.predicted_rating.round(2)),
                 hide_index=True, width="stretch")
with right:
    st.subheader("Personalised learning path")
    if out["learning_path"]:
        st.dataframe(pd.DataFrame(out["learning_path"])[["topic", "action", "resource", "minutes", "why"]],
                     hide_index=True, width="stretch")
    else:
        st.info("No topic is below the remediation threshold (60%) and none is mastered (≥ 75%) - "
                "keep going with the recommended resources on the left.")
    st.caption("Checkpoint rule: score ≥ 75% → advance to the next tier, otherwise repeat practice.")
