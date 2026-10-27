"""HTML pieces for the Streamlit front end. Styling lives in style.css."""
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

CSS = Path(__file__).with_name("style.css")


def render(html):
    """st.markdown treats indented lines as code and blank lines as paragraph breaks,
    so flatten the markup before handing it over."""
    st.markdown(" ".join(line.strip() for line in html.splitlines() if line.strip()),
                unsafe_allow_html=True)


def inject_css():
    st.markdown(f"<style>{CSS.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def hero(metrics, auc90):
    d = metrics["dataset"]
    render(f"""
    <section class="al-hero">
      <div class="al-word" role="heading" aria-level="1"><span>ADAPTIVE</span><span>LEARN</span></div>
      <div class="al-intro">
        <p class="al-lede">Each dot is a real student the models never saw. Where it sits shows what
        they had done by day 90 of the course; its colour shows how the course actually ended.
        Click one to open their forecast.</p>
        <p class="al-facts">Built from <b>{d["n_enrolments_total"]:,}</b> enrolments and
        <b>{d["n_click_rows_raw"] / 1e6:.1f} million</b> clicks across
        <b>{metrics["recommender"]["n_presentations"]}</b> course presentations of the Open University.
        At day 90 the model flags students who will fail or withdraw with a ROC-AUC of
        <b>{auc90:.2f}</b>.</p>
      </div>
      <div class="al-legend" aria-hidden="true">
        <span><i class="k-blue"></i>Passed, some with distinction</span>
        <span><i class="k-red"></i>Failed or withdrew</span>
        <span><i class="k-sel"></i>Student you're viewing</span>
        <span><i class="k-ghost"></i>Same student after your what-if changes</span>
      </div>
    </section>
    """)


def head(anchor, title, text):
    render(f"""
    <div class="al-head" id="{anchor}">
      <h2>{escape(title)}</h2>
      <p>{escape(text)}</p>
    </div>
    """)


def readout(out, plan, actual):
    risk = plan["risk"].lower()
    conf = out["proba"].get(out["predicted_result"], 0)
    render(f"""
    <div class="al-readout">
      <div class="al-cell">
        <div class="al-cell__label">Predicted outcome</div>
        <div class="al-cell__value">{escape(out["predicted_result"])}</div>
        <div class="al-cell__note">{conf:.0%} likely</div>
      </div>
      <div class="al-cell al-cell--risk is-{risk}">
        <div class="al-cell__label">Chance of failing or withdrawing</div>
        <div class="al-cell__value">{out["at_risk"]:.0%}</div>
        <div class="al-meter" aria-hidden="true"><i style="width:{out["at_risk"] * 100:.0f}%"></i></div>
        <div class="al-cell__note" style="margin-top:6px">{escape(plan["risk"])} risk</div>
      </div>
      <div class="al-cell">
        <div class="al-cell__label">Expected score on what's left</div>
        <div class="al-cell__value">{out["predicted_score"]:.0f}%</div>
        <div class="al-cell__note">{escape(plan["tier"])} study tier</div>
      </div>
      <div class="al-cell">
        <div class="al-cell__label">Learner persona</div>
        <div class="al-cell__value al-cell__value--name">{escape(out["persona"])}</div>
      </div>
    </div>
    <p class="al-truth">How this student's course really ended: <b>{escape(str(actual))}</b>.
    The models never saw this.</p>
    """)


def sub(text):
    render(f'<div class="al-sub">{escape(text)}</div>')


def timeline(steps):
    urgent = ("This week", "Today")
    items = "".join(f"""
      <li class="{"is-urgent" if s["when"] in urgent else ""}">
        <div class="al-when">{escape(s["when"])}</div>
        <p class="al-action">{escape(s["action"])}</p>
        <p class="al-detail">{escape(s["detail"])}</p>
        <p class="al-why">{escape(s["why"])}</p>
      </li>""" for s in steps)
    render(f'<ol class="al-timeline">{items}</ol>')


def note(html):
    render(f'<div class="al-note">{html}</div>')


def how_it_works(sweep, persona_share):
    lo = 0.6
    hi = sweep.at_risk_auc.max()
    cols = "".join(
        f'<div class="{"is-now" if d == 90 else ""}"><em>{a:.2f}</em>'
        f'<i style="height:{(a - lo) / (hi - lo) * 100:.0f}%"></i><small>day {d}</small></div>'
        for d, a in zip(sweep.cutoff_day, sweep.at_risk_auc))
    rows = "".join(
        f'<div class="al-bar"><span class="al-bar__name">{escape(name)}</span>'
        f'<span class="al-bar__track"><span class="al-bar__fill{" is-red" if "risk" in name.lower() else ""}" '
        f'style="width:{share / persona_share.max() * 100:.0f}%"></span></span>'
        f'<span class="al-bar__val">{share:.0%}</span></div>'
        for name, share in persona_share.items())
    render(f"""
    <section class="al-how">
      <div>
        <h3>Why day 90</h3>
        <p>Warnings get sharper the longer a course runs, but help comes too late if you wait.
        Day 90 is accurate enough to act on and early enough to change the outcome. The bars
        show the at-risk ROC-AUC at each cut-off.</p>
        <div class="al-sweep">{cols}</div>
      </div>
      <div>
        <h3>Five kinds of learner</h3>
        <p>Clustering early behaviour finds five personas. A student's persona shapes the tone and
        content of the study plan they get.</p>
        <div class="al-bars">{rows}</div>
      </div>
    </section>
    """)


def footer():
    render("""
    <p class="al-footer">Data: Kuzilek, Hlosta and Zdrahal, Open University Learning Analytics
    dataset, Scientific Data 4:170171 (2017), CC BY 4.0. Predictions use only the first 90 days of
    each course.</p>
    """)


def persona_shares(enrolments):
    return enrolments["persona"].value_counts(normalize=True) if "persona" in enrolments else pd.Series(dtype=float)
