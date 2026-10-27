"""HTML pieces for the Streamlit front end. Styling lives in style.css."""
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

CSS = Path(__file__).with_name("style.css")
ARROW = ('<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
         'stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
         '<path d="M12 5v14M5 12l7 7 7-7"/></svg>')


def render(html):
    """st.markdown treats indented lines as code and blank lines as paragraph breaks,
    so flatten the markup before handing it over."""
    flat = " ".join(line.strip() for line in html.splitlines() if line.strip())
    st.markdown(flat, unsafe_allow_html=True)


def inject_css():
    st.markdown(f"<style>{CSS.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _fmt_count(n):
    return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n:,}"


def hero(metrics, sweep):
    auc90 = float(sweep.loc[sweep.cutoff_day == 90, "at_risk_auc"].iloc[0]) if len(sweep) else \
        metrics["classification"]["test_at_risk_auc"]
    lo = sweep.at_risk_auc.min() - 0.05 if len(sweep) else 0
    bars = "".join(
        f'<i class="{"is-now" if d == 90 else ""}" '
        f'style="height:{(a - lo) / (sweep.at_risk_auc.max() - lo) * 100:.0f}%"></i>'
        for d, a in zip(sweep.cutoff_day, sweep.at_risk_auc))
    axis = "".join(f"<span>{d}</span>" for d in sweep.cutoff_day)
    render(f"""
    <section class="al-hero">
      <div class="al-hero__top">
        <span class="al-eyebrow">AUCML Lab · SDG 4 &amp; 10</span>
        <span class="al-hero__tag">Open University Learning Analytics Dataset</span>
      </div>
      <div class="al-word" role="heading" aria-level="1"><span>ADAPTIVE</span><span>LEARN</span></div>
      <div class="al-hero__body">
        <p class="al-hero__lede">Spot students who are slipping <strong>within the first 90 days</strong>
        of a course, then hand each one a study plan built around their next real assessment.</p>
        <div class="al-plate" aria-label="At-risk early warning, ROC-AUC {auc90:.2f} at day 90">
          <div class="al-plate__stripe"></div>
          <div class="al-plate__label">At-risk early warning</div>
          <div class="al-plate__num">{auc90:.2f}</div>
          <div class="al-plate__sub">ROC-AUC on held-out students, day 90</div>
          <div class="al-spark">{bars}</div>
          <div class="al-spark-axis">{axis}</div>
        </div>
        <div class="al-hero__cta-wrap">
          <a class="al-cta" href="#student-explorer" target="_self">Explore a student {ARROW}</a>
        </div>
      </div>
    </section>
    """)


def stats(metrics, n_presentations, n_personas):
    d = metrics["dataset"]
    items = [(_fmt_count(d["n_enrolments_total"]), "course enrolments"),
             (_fmt_count(d["n_click_rows_raw"]), "click records analysed"),
             (str(n_presentations), "course presentations"),
             (str(n_personas), "learner personas")]
    cells = "".join(f'<div class="al-stat"><b>{v}</b><span>{escape(k)}</span></div>' for v, k in items)
    render(f'<div class="al-stats">{cells}</div>')


def features(sweep, persona_share):
    cols = "".join(
        f'<div class="{"is-now" if d == 90 else ""}"><em>{a:.2f}</em>'
        f'<i style="height:{(a - 0.6) / (sweep.at_risk_auc.max() - 0.6) * 100:.0f}%"></i>'
        f'<small>day {d}</small></div>'
        for d, a in zip(sweep.cutoff_day, sweep.at_risk_auc))
    rows = "".join(
        f'<div class="al-bar"><span class="al-bar__name">{escape(name)}</span>'
        f'<span class="al-bar__track"><span class="al-bar__fill{" is-red" if "risk" in name.lower() else ""}" '
        f'style="width:{share / persona_share.max() * 100:.0f}%; display:block"></span></span>'
        f'<span class="al-bar__val">{share:.0%}</span></div>'
        for name, share in persona_share.items())
    render(f"""
    <div class="al-features">
      <article class="al-feature">
        <h3>Early answers, before it's too late.</h3>
        <p>Warning quality climbs as the course goes on. Day 90 is the balance point: accurate
        enough to act on, early enough for support to still change the outcome.</p>
        <div class="al-sweep">{cols}</div>
      </article>
      <article class="al-feature al-feature--red">
        <h3>A plan shaped to each learner.</h3>
        <p>Clustering on early behaviour finds five personas. Each one steers the tone and
        content of the study plan a student receives.</p>
        <div class="al-bars">{rows}</div>
      </article>
    </div>
    """)


def section_head(anchor, eyebrow, title, text):
    render(f"""
    <div class="al-section-head" id="{anchor}">
      <span class="al-eyebrow">{escape(eyebrow)}</span>
      <h2>{escape(title)}</h2>
      <p>{escape(text)}</p>
    </div>
    """)


def kpis(out, plan, actual):
    risk = plan["risk"].lower()
    render(f"""
    <div class="al-kpis">
      <div class="al-kpi">
        <div class="al-kpi__label">Predicted outcome</div>
        <div class="al-kpi__value">{escape(out["predicted_result"])}</div>
        <span class="al-pill al-pill--tier">{out["proba"].get(out["predicted_result"], 0):.0%} confidence</span>
      </div>
      <div class="al-kpi al-kpi--risk-{risk}">
        <div class="al-kpi__label">Risk of fail / withdrawal</div>
        <div class="al-kpi__value">{out["at_risk"]:.0%}</div>
        <span class="al-pill al-pill--{risk}">{escape(plan["risk"])} risk</span>
      </div>
      <div class="al-kpi">
        <div class="al-kpi__label">Predicted future score</div>
        <div class="al-kpi__value">{out["predicted_score"]:.0f}%</div>
        <span class="al-pill al-pill--tier">{escape(plan["tier"])} tier</span>
      </div>
      <div class="al-kpi">
        <div class="al-kpi__label">Persona</div>
        <div class="al-kpi__value al-kpi__value--sm">{escape(out["persona"])}</div>
      </div>
    </div>
    <p class="al-truth">Recorded real outcome: <b>{escape(str(actual))}</b>. Shown for transparency;
    the models never saw it.</p>
    """)


def panel_title(text):
    render(f'<div class="al-panel-title">{escape(text)}</div>')


def steps(plan_steps):
    urgent = ("This week", "Today")
    items = "".join(f"""
      <div class="al-step{" al-step--urgent" if s["when"] in urgent else ""}">
        <span class="al-step__when">{escape(s["when"])}</span>
        <div>
          <p class="al-step__action">{escape(s["action"])}</p>
          <p class="al-step__detail">{escape(s["detail"])}</p>
          <p class="al-step__why">{escape(s["why"])}</p>
        </div>
      </div>""" for s in plan_steps)
    render(f'<div class="al-steps">{items}</div>')


def empty(html):
    render(f'<div class="al-empty">{html}</div>')


def footer():
    render("""
    <div class="al-footer">Data: Kuzilek, Hlosta &amp; Zdrahal, Open University Learning Analytics
    dataset, Scientific Data 4:170171 (2017), CC BY 4.0. Predictions use only the first 90 days of
    each course.</div>
    """)


def persona_shares(enrolments):
    return enrolments["persona"].value_counts(normalize=True) if "persona" in enrolments else pd.Series(dtype=float)
