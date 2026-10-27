"""Instructor / student dashboard:  streamlit run app/streamlit_app.py
(train the models first with `python run_all.py` or `python train_from_processed.py`)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import ui  # noqa: E402
from adaptivelearn.config import MODEL_DIR, RESULTS_DIR, TABLE_DIR  # noqa: E402
from adaptivelearn.inference import load_artifacts, profile  # noqa: E402

st.set_page_config(page_title="AdaptiveLearn", page_icon="🎓", layout="wide")
ui.inject_css()

if not (MODEL_DIR / "classifier.joblib").exists():
    st.error("Trained models not found. Run `python run_all.py` (needs the raw OULAD files) "
             "or `python train_from_processed.py` (uses the committed processed data).")
    st.stop()


@st.cache_resource
def artifacts():
    return load_artifacts()


@st.cache_data
def reference_tables():
    metrics = json.loads((RESULTS_DIR / "metrics.json").read_text(encoding="utf-8"))
    sweep = pd.read_csv(TABLE_DIR / "early_prediction_sweep.csv")
    return metrics, sweep


art = artifacts()
metrics, sweep = reference_tables()
df = art["enrolments"]
test = df[df.split == "test"]          # students the models have never seen

# ---------- hero ----------
ui.hero(metrics, sweep)
ui.stats(metrics, metrics["recommender"]["n_presentations"], metrics["clustering"]["k"])
ui.features(sweep, ui.persona_shares(df))

# ---------- sidebar controls ----------
with st.sidebar:
    ui.render('<div class="al-side-brand">Adaptive<b>Learn</b></div>'
              '<div class="al-side-note">Pick a held-out student, then change their behaviour '
              'to see how the prediction and plan respond.</div>')
    st.header("Student")
    module = st.selectbox("Module", sorted(test.code_module.unique()))
    pres = st.selectbox("Presentation", sorted(test[test.code_module == module].code_presentation.unique()))
    pool = test[(test.code_module == module) & (test.code_presentation == pres)].reset_index(drop=True)
    idx = st.number_input(f"Student # (0 – {len(pool) - 1})", 0, len(pool) - 1, 0)
    base = pool.iloc[int(idx)].copy()

    st.header("What-if")
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

# ---------- explorer ----------
ui.section_head("student-explorer", "Student explorer", f"{module} · {pres} · student #{int(idx)}",
                "Live predictions from the trained models. Move the sliders in the sidebar to test "
                "what-if scenarios.")
ui.kpis(out, plan, base.final_result)

order = ["Distinction", "Pass", "Fail", "Withdrawn"]
proba = pd.DataFrame({"outcome": order, "probability": [out["proba"].get(c, 0.0) for c in order]})
ui.panel_title("Outcome probabilities")
chart = (
    alt.Chart(proba)
    .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8, size=64)
    .encode(
        x=alt.X("outcome:N", sort=order, title=None,
                axis=alt.Axis(labelAngle=0, labelFontSize=12, labelColor="#0b1220", ticks=False,
                              domain=False, labelPadding=10)),
        y=alt.Y("probability:Q", title=None, scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%", tickCount=5, labelColor="#5b6475", grid=True,
                              gridColor="#e3e6ec", domain=False, ticks=False)),
        color=alt.Color("outcome:N", sort=order, legend=None,
                        scale=alt.Scale(domain=order,
                                        range=["#0f2a6b", "#4f7be8", "#ef6b7f", "#a3112a"])),
        tooltip=["outcome", alt.Tooltip("probability:Q", format=".0%")],
    )
)
labels = chart.mark_text(dy=-10, fontSize=12, fontWeight=600, color="#0b1220").encode(
    text=alt.Text("probability:Q", format=".0%"), color=alt.value("#0b1220"))
st.altair_chart(
    (chart + labels).properties(height=280, background="transparent")
    .configure_view(strokeWidth=0).configure(font="Inter"),
    use_container_width=True,
)

left, right = st.columns([1, 1.5], gap="large")
with left:
    ui.panel_title("Recommended next resources")
    recs = out["recommendations"]
    if len(recs):
        st.dataframe(recs[["activity_type", "week_from", "id_site"]].rename(
            columns={"activity_type": "type", "week_from": "from week", "id_site": "site id"}),
            hide_index=True, use_container_width=True)
    elif not art["recommender"]:
        ui.empty("The resource recommender isn't trained yet. It needs the raw OULAD click logs: "
                 "put the 7 CSVs in <code>data/oulad_raw/</code> and run <code>python run_all.py</code>.")
    else:
        ui.empty("No recommender is available for this presentation.")
with right:
    ui.panel_title("Personalised study plan")
    ui.steps(plan["steps"])

ui.footer()
