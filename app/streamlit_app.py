"""AdaptiveLearn website:  streamlit run app/streamlit_app.py
(train the models first with `python run_all.py` or `python train_from_processed.py`).

Four pages behind one title bar: Overview, Student explorer, Results, About.
"""
import importlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import cohort_map as _cohort  # noqa: E402
import ui  # noqa: E402
from adaptivelearn.config import MODEL_DIR, RESULTS_DIR, TABLE_DIR  # noqa: E402
from adaptivelearn.inference import load_artifacts, profile  # noqa: E402


def _fresh(mod):
    """Streamlit re-runs this file on every interaction but keeps helper modules cached, and on
    some Windows setups it misses edits to them. Reload a helper when its file has changed."""
    if getattr(mod, "LOADED_MTIME", None) != Path(mod.__file__).stat().st_mtime:
        mod = importlib.reload(mod)
    return mod


ui = _fresh(ui)
cohort_map = _fresh(_cohort).cohort_map

st.set_page_config(page_title="AdaptiveLearn", page_icon="🎓", layout="wide",
                   initial_sidebar_state="collapsed")
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
    tables = {"fairness": pd.read_csv(TABLE_DIR / "fairness_audit.csv"),
              "persona_mix": pd.read_csv(TABLE_DIR / "cluster_outcome_mix.csv", index_col=0),
              "recommender": pd.read_csv(TABLE_DIR / "recommender_models.csv")}
    return metrics, sweep, tables


@st.cache_data
def held_out(df):
    """Held-out students plus their position inside their module/presentation pool."""
    test = df[df.split == "test"].reset_index(drop=True)
    test["pool_idx"] = test.groupby(["code_module", "code_presentation"]).cumcount()
    return test


art = artifacts()
metrics, sweep, tables = reference_tables()
df = art["enrolments"]
test = held_out(df)
by_key = test.set_index("key")
modules = sorted(test.code_module.unique())
auc90 = float(sweep.loc[sweep.cutoff_day == 90, "at_risk_auc"].iloc[0])


def presentations(module):
    return sorted(test[test.code_module == module].code_presentation.unique())


# ---------- shared state: the selected student and the appearance ----------
ss = st.session_state
ss.setdefault("mod", modules[0])
ss.setdefault("pres", presentations(ss.mod)[0])
ss.setdefault("num", 1)                     # 1-based student number in the pool

THEME_NAMES = list(ui.THEMES)
if "theme" not in ss:
    q = st.query_params.get("theme")
    ss.theme = q if q in THEME_NAMES else "Dark"


def select_key(key):
    if key is None or key not in by_key.index:
        return False
    r = by_key.loc[key]
    ss.mod, ss.pres, ss.num = r.code_module, r.code_presentation, int(r.pool_idx) + 1
    return True


def on_pick():
    """A dot on the cohort map was clicked: select that student."""
    select_key(ss.cohort.selected)


def on_pick_and_open():
    """A dot on the Overview map was clicked: select the student and open the explorer."""
    if select_key(ss.cohort.selected):
        ss.open_explorer = True


def on_module_change():
    ss.pres = presentations(ss.mod)[0]
    ss.num = 1


def on_theme_change():
    if not ss.theme:                      # clicking the active option deselects it; keep a theme
        ss.theme = st.query_params.get("theme", "Dark")
    st.query_params["theme"] = ss.theme


def current_student():
    pool = test[(test.code_module == ss.mod) & (test.code_presentation == ss.pres)].reset_index(drop=True)
    ss.num = max(1, min(int(ss.num), len(pool)))
    return pool, pool.iloc[int(ss.num) - 1].copy()


# ---------- pages ----------
def page_overview():
    _, base = current_student()
    ui.hero(metrics, auc90)
    cohort_map(test, base, base, None, on_pick_and_open, theme=theme)
    if ss.pop("open_explorer", False):
        st.switch_page(EXPLORER)
    ui.in_short(metrics, tables)
    with st.container(key="cta", horizontal=True, gap="medium"):
        st.page_link(EXPLORER, label="Open the student explorer", query_params={"theme": theme})
        st.page_link(RESULTS, label="Read the results in plain words", query_params={"theme": theme})


def page_explorer():
    pool, base = current_student()
    ui.head("student-explorer", f"Student {ss.num} of {len(pool)} in module {ss.mod}, {ss.pres}",
            "Choose a student on the left or click any dot on the map. Then change how they behaved "
            "and watch the forecast and study plan respond.")
    controls, results = st.columns([1, 2.3], gap="large")

    with controls:
        with st.container(key="controls"):
            st.markdown('<div class="al-panel-title">Student</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2, gap="small")
            c1.selectbox("Module", modules, key="mod", on_change=on_module_change)
            c2.selectbox("Presentation", presentations(ss.mod), key="pres",
                         on_change=lambda: ss.update(num=1))
            n1, n2 = st.columns([1.6, 1], gap="small", vertical_alignment="bottom")
            n1.number_input(f"Student, 1 to {len(pool)}", 1, len(pool), key="num")
            with n2.container(key="random"):
                st.button("Random", on_click=lambda: ss.update(num=int(random.randint(1, len(pool)))),
                          width="stretch", help="Pick another student from this presentation")
            sk = int(base.key)                   # widget keys per student, so sliders reset on a new pick

            real = {
                "total_clicks": int(min(base.total_clicks, 6000)),
                "active_days": int(min(base.active_days, 90)),
                "clicks_last_14d": int(min(base.clicks_last_14d, 1500)),
                "days_since_last_active": int(min(base.days_since_last_active, 90)),
                "n_submitted": int(min(base.n_submitted, 6)),
                "mean_score_early": 0 if pd.isna(base.mean_score_early) else int(base.mean_score_early),
            }
            keys = {f: f"w{sk}_{f}" for f in real}
            changed = [f for f in real if ss.get(keys[f], real[f]) != real[f]]

            def reset():
                for k in keys.values():
                    ss.pop(k, None)

            st.markdown('<hr class="al-rule">', unsafe_allow_html=True)
            h1, h2 = st.columns([1.6, 1], gap="small", vertical_alignment="center")
            h1.markdown('<div class="al-panel-title">What if they had…</div>', unsafe_allow_html=True)
            with h2.container(key="reset"):
                st.button("Reset", on_click=reset, disabled=not changed, width="stretch",
                          help="Put every value back to what this student really did")
            st.markdown(f'<p class="al-hint">{"Values start at what this student really did." if not changed else f"{len(changed)} changed. The ghost dot on the map shows where they would sit now."}</p>',
                        unsafe_allow_html=True)

            def whatif(field, label, hi, unit=""):
                slot = st.empty()
                v = st.slider(label, 0, hi, real[field], key=keys[field], label_visibility="collapsed")
                was = (f'<span class="al-was">was {real[field]:,}{unit}</span>' if v != real[field] else "")
                slot.markdown(f'<div class="al-field{" is-changed" if v != real[field] else ""}">'
                              f'<span class="al-field__label">{label}</span>'
                              f'<span class="al-field__value">{was}<b>{v:,}{unit}</b></span></div>',
                              unsafe_allow_html=True)
                return v

            row = base.copy()
            st.markdown('<div class="al-group">Activity by day 90</div>', unsafe_allow_html=True)
            row["total_clicks"] = whatif("total_clicks", "Page clicks", 6000)
            row["active_days"] = whatif("active_days", "Days active", 90)
            row["clicks_last_14d"] = whatif("clicks_last_14d", "Clicks, last two weeks", 1500)
            row["days_since_last_active"] = whatif("days_since_last_active", "Days since last login", 90)
            st.markdown('<div class="al-group">Assessments</div>', unsafe_allow_html=True)
            row["n_submitted"] = whatif("n_submitted", "Handed in", 6)
            row["mean_score_early"] = whatif("mean_score_early", "Average score", 100, "%")
            row["clicks_per_active_day"] = row["total_clicks"] / max(row["active_days"], 1)

    out = profile(row, art)
    plan = out["plan"]

    with results:
        cohort_map(test, base, row, out["at_risk"], on_pick, theme=theme, height=380)
        ui.readout(out, plan, base.final_result)

        order = ["Distinction", "Pass", "Fail", "Withdrawn"]
        proba = pd.DataFrame({"outcome": order, "probability": [out["proba"].get(c, 0.0) for c in order]})
        ui.sub("How likely each ending is")
        bars = (
            alt.Chart(proba)
            .mark_bar(cornerRadiusTopLeft=7, cornerRadiusTopRight=7, size=58)
            .encode(
                x=alt.X("outcome:N", sort=order, title=None,
                        axis=alt.Axis(labelAngle=0, labelFontSize=12, labelColor=colors["axis"], ticks=False,
                                      domain=False, labelPadding=10)),
                y=alt.Y("probability:Q", title=None, scale=alt.Scale(domain=[0, 1]),
                        axis=alt.Axis(format="%", tickCount=5, labelColor=colors["tick"], grid=True,
                                      gridColor=colors["grid"], domain=False, ticks=False)),
                color=alt.Color("outcome:N", sort=order, legend=None,
                                scale=alt.Scale(domain=order, range=colors["bars"])),
                tooltip=["outcome", alt.Tooltip("probability:Q", format=".0%")],
            )
        )
        labels = bars.mark_text(dy=-10, fontSize=12, fontWeight=600).encode(
            text=alt.Text("probability:Q", format=".0%"), color=alt.value(colors["label"]))
        st.altair_chart(
            (bars + labels).properties(height=230, background="transparent")
            .configure_view(strokeWidth=0).configure(font="Geist, Segoe UI, system-ui, sans-serif"),
            width="stretch",
        )

        plan_col, recs_col = st.columns([1.5, 1], gap="large")
        with plan_col:
            ui.sub("Study plan")
            ui.timeline(plan["steps"])
        with recs_col:
            ui.sub("Resources to open next")
            recs = out["recommendations"]
            if len(recs):
                st.dataframe(recs[["activity_type", "week_from", "id_site"]].rename(
                    columns={"activity_type": "type", "week_from": "from week", "id_site": "site id"}),
                    hide_index=True, width="stretch")
            elif not art["recommender"]:
                ui.note("Resource suggestions need the raw OULAD click logs. Put the 7 CSV files in "
                        "<code>data/oulad_raw/</code> and run <code>python run_all.py</code>.")
            else:
                ui.note("There are no resource suggestions for this presentation.")


def page_results():
    ui.page_title("Results",
                  "How well the system works on students it never saw during training, first in "
                  "plain words and then the two ideas behind it.")
    ui.plain_summary(metrics, tables, heading=False)
    ui.how_it_works(sweep, ui.persona_shares(df))


def page_about():
    ui.about(metrics)


OVERVIEW = st.Page(page_overview, title="Overview", url_path="overview", default=True)
EXPLORER = st.Page(page_explorer, title="Student explorer", url_path="explorer")
RESULTS = st.Page(page_results, title="Results", url_path="results")
ABOUT = st.Page(page_about, title="About", url_path="about")
PAGES = [OVERVIEW, EXPLORER, RESULTS, ABOUT]

current = st.navigation(PAGES, position="hidden")
st.set_page_config(page_title=f"{current.title} | AdaptiveLearn" if current.title != "Overview" else "AdaptiveLearn")

# ---------- title bar ----------
with st.container(key="nav", horizontal=True, vertical_alignment="center", gap="small"):
    with st.container(key="brand", width="content"):
        st.page_link(OVERVIEW, label="AdaptiveLearn", query_params={"theme": ss.theme})
    with st.container(key="links", horizontal=True, width="content", gap="small"):
        for page in PAGES:
            state = "on" if page.url_path == current.url_path else "off"
            with st.container(key=f"link_{page.url_path}_{state}", width="content"):
                st.page_link(page, label=page.title, query_params={"theme": ss.theme})
    st.segmented_control("Appearance", THEME_NAMES, key="theme", label_visibility="collapsed",
                         on_change=on_theme_change, width="content")

theme = ss.theme or "Dark"
ui.theme_css(theme)
colors = ui.chart_colors(theme)

current.run()
ui.footer()
