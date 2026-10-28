"""HTML pieces for the Streamlit front end. Styling lives in style.css."""
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

LOADED_MTIME = Path(__file__).stat().st_mtime

CSS = Path(__file__).with_name("style.css")


def render(html):
    """st.markdown treats indented lines as code and blank lines as paragraph breaks,
    so flatten the markup before handing it over."""
    st.markdown(" ".join(line.strip() for line in html.splitlines() if line.strip()),
                unsafe_allow_html=True)


# Three appearances. Blue always means "finished the course", red always means "failed or withdrew".
THEMES = {
    "Dark": {
        "void": "#060a18", "panel": "#0c1328", "panel-2": "#111a35", "field": "#0f1730",
        "line": "rgba(150, 170, 230, 0.12)", "line-strong": "rgba(160, 180, 240, 0.24)",
        "text": "#e7ebf7", "text-2": "#c9d0e4", "mute": "#8e98b6", "tick": "#6f7a99",
        "blue": "#5b86ff", "blue-soft": "#8fb0ff", "red": "#ff4a66", "red-soft": "#ff8a9c", "amber": "#ffb35c",
        "track": "rgba(150, 170, 230, 0.14)", "wash-blue": "rgba(91, 134, 255, 0.16)", "wash-red": "rgba(255, 74, 102, 0.10)",
        "chrome-blue": "linear-gradient(180deg, #dfe7ff 0%, #8aa7ff 30%, #2a4fd6 49%, #0d1a52 51%, #3a62e8 72%, #b9caff 100%)",
        "chrome-red": "linear-gradient(180deg, #ffe1e6 0%, #ff8a9c 30%, #d61f3c 49%, #4c0612 51%, #e8324f 72%, #ffc0ca 100%)",
        "glow-blue": "rgba(42, 79, 214, 0.35)", "glow-red": "rgba(214, 31, 60, 0.35)", "word-edge": "rgba(255, 255, 255, 0.25)",
        "canvas-top": "#0c1328", "canvas-bottom": "#080d1f", "grid": "rgba(150, 170, 230, 0.10)",
        "guide": "rgba(231, 235, 247, 0.18)", "dot-alpha": "0.42", "tip-bg": "rgba(12, 19, 40, 0.94)",
        "inset": "rgba(255, 255, 255, 0.05)", "shadow": "rgba(0, 0, 0, 0.85)",
        "good-bg": "rgba(91, 134, 255, 0.16)", "good-fg": "#a9c0ff", "ok-bg": "rgba(255, 179, 92, 0.16)", "ok-fg": "#ffc98a",
    },
    "Light": {
        "void": "#f2f4f9", "panel": "#ffffff", "panel-2": "#f8f9fc", "field": "#ffffff",
        "line": "rgba(16, 30, 72, 0.12)", "line-strong": "rgba(16, 30, 72, 0.2)",
        "text": "#0e1630", "text-2": "#2c3655", "mute": "#5a6584", "tick": "#7a84a0",
        "blue": "#2b55e0", "blue-soft": "#2449c4", "red": "#d81f3f", "red-soft": "#b8162f", "amber": "#c86a00",
        "track": "rgba(16, 30, 72, 0.08)", "wash-blue": "rgba(43, 85, 224, 0.10)", "wash-red": "rgba(216, 31, 63, 0.07)",
        "chrome-blue": "linear-gradient(180deg, #6f8ff5 0%, #2a4fd6 42%, #0d1a52 50%, #2650c4 70%, #8fa9f5 100%)",
        "chrome-red": "linear-gradient(180deg, #f6788b 0%, #c81c37 42%, #5a0613 50%, #c81c37 70%, #f0909f 100%)",
        "glow-blue": "rgba(43, 85, 224, 0.18)", "glow-red": "rgba(216, 31, 63, 0.16)", "word-edge": "rgba(255, 255, 255, 0.9)",
        "canvas-top": "#ffffff", "canvas-bottom": "#f6f8fc", "grid": "rgba(16, 30, 72, 0.08)",
        "guide": "rgba(14, 22, 48, 0.22)", "dot-alpha": "0.5", "tip-bg": "rgba(255, 255, 255, 0.97)",
        "inset": "rgba(255, 255, 255, 0.9)", "shadow": "rgba(16, 30, 72, 0.25)",
        "good-bg": "rgba(43, 85, 224, 0.10)", "good-fg": "#1f43b8", "ok-bg": "rgba(200, 106, 0, 0.12)", "ok-fg": "#9a5100",
    },
    "Graphite": {
        "void": "#1b1c1f", "panel": "#232428", "panel-2": "#2a2c30", "field": "#2a2c30",
        "line": "rgba(255, 255, 255, 0.08)", "line-strong": "rgba(255, 255, 255, 0.16)",
        "text": "#ececee", "text-2": "#c8c9cd", "mute": "#9a9ca3", "tick": "#7d7f86",
        "blue": "#7d95d9", "blue-soft": "#a3b4e6", "red": "#e06a7e", "red-soft": "#eb9aa8", "amber": "#d9a066",
        "track": "rgba(255, 255, 255, 0.08)", "wash-blue": "rgba(255, 255, 255, 0.04)", "wash-red": "rgba(255, 255, 255, 0.02)",
        "chrome-blue": "linear-gradient(180deg, #f4f5f7 0%, #b9bdc7 30%, #5d6372 49%, #2b2e36 51%, #7c8394 72%, #d9dce3 100%)",
        "chrome-red": "linear-gradient(180deg, #f6eef0 0%, #cbb3b9 30%, #7a5560 49%, #3a2328 51%, #9a6d78 72%, #e6d6da 100%)",
        "glow-blue": "rgba(0, 0, 0, 0.45)", "glow-red": "rgba(0, 0, 0, 0.45)", "word-edge": "rgba(255, 255, 255, 0.2)",
        "canvas-top": "#26272b", "canvas-bottom": "#1f2023", "grid": "rgba(255, 255, 255, 0.07)",
        "guide": "rgba(236, 236, 238, 0.18)", "dot-alpha": "0.45", "tip-bg": "rgba(35, 36, 40, 0.96)",
        "inset": "rgba(255, 255, 255, 0.04)", "shadow": "rgba(0, 0, 0, 0.7)",
        "good-bg": "rgba(125, 149, 217, 0.16)", "good-fg": "#b3c1ea", "ok-bg": "rgba(217, 160, 102, 0.16)", "ok-fg": "#e5bb8f",
    },
}


def theme_css(name):
    t = THEMES.get(name, THEMES["Dark"])
    scheme = "light" if name == "Light" else "dark"
    body = " ".join(f"--{k}: {v};" for k, v in t.items())
    st.markdown(f"<style>:root {{ {body} color-scheme: {scheme}; }}</style>", unsafe_allow_html=True)


def chart_colors(name):
    t = THEMES.get(name, THEMES["Dark"])
    return {"bars": [t["blue-soft"], t["blue"], t["red-soft"], t["red"]],
            "label": t["text"], "axis": t["text-2"], "tick": t["tick"], "grid": t["line"]}


def inject_css():
    st.markdown(f"<style>{CSS.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def hero(metrics, auc90):
    d = metrics["dataset"]
    render(f"""
    <section class="al-hero">
      <div class="al-word" role="heading" aria-level="1"><span>Adaptive</span><span>Learn</span></div>
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


def plain_summary(metrics, tables):
    """The headline results, each with a plain-English reading. Numbers come from results/."""
    c, r = metrics["classification"], metrics["regression"]
    fair, mix, rec = tables["fairness"], tables["persona_mix"], tables["recommender"]
    g = fair[fair.attribute == "gender"]
    w = g.n / g.n.sum()
    caught = float((g["recall_at_risk (TPR)"] * w).sum())
    false_alarm = float((g["false_alarm_rate (FPR)"] * w).sum())
    guess = max(metrics["dataset"]["outcome_distribution"].values())
    personas = {int(k): v for k, v in metrics["clustering"]["personas"].items()}
    risk_of = {personas[i]: float(mix.loc[i, "Fail"] + mix.loc[i, "Withdrawn"]) for i in mix.index}
    worst = max(risk_of, key=risk_of.get)
    best = min(risk_of, key=risk_of.get)
    imd = fair[fair.attribute == "imd_band"].set_index("group")
    poor, rich = imd.loc["0-10%"], imd.loc["90-100%"]
    svd = rec.set_index("model").loc["SVD (k=20)", "hit_rate@10"]
    pop = rec.set_index("model").loc["Popularity", "hit_rate@10"]
    rnd = rec.set_index("model").loc["Random", "hit_rate@10"]
    mae = r["test_mae"]

    rows = [
        ("Can it tell who is likely to fail or drop out, by day 90?",
         f"ROC-AUC {c['test_at_risk_auc']:.2f}",
         f"Show it two students, one who will struggle and one who won't. It picks the right one "
         f"{c['test_at_risk_auc'] * 100:.0f} times out of 100.", "good", "Strong"),
        ("How many struggling students does it actually catch?",
         f"{caught:.0%} caught, {false_alarm:.0%} false alarms",
         f"About {round(caught * 3)} in 3 students who go on to fail or withdraw get flagged. "
         f"Roughly 1 in {round(1 / false_alarm)} students who are fine get flagged by mistake.", "good", "Useful"),
        ("Can it name the exact result: distinction, pass, fail or withdrawn?",
         f"{c['test_accuracy']:.0%} right (guessing gets {guess:.0%})",
         "Right about 6 times in 10. Better than always guessing “pass”, but treat it as a rough "
         "guide, not a verdict.", "ok", "Modest"),
        ("How close are its score predictions?",
         f"Off by {mae:.1f} points on average",
         f"If it predicts 65%, the real score usually lands between about {65 - round(mae)}% and "
         f"{65 + round(mae)}%.", "good", "Good"),
        ("Do the five learner types mean anything?",
         f"{risk_of[worst]:.0%} vs {risk_of[best]:.0%} fail or withdraw",
         f"Students in the “{worst}” group fail or drop out {risk_of[worst]:.0%} of the time. "
         f"For “{best}” it's {risk_of[best]:.0%}. The groups reflect real differences.", "good", "Clear"),
        ("Is it fair to students from poorer areas?",
         f"Catches {poor['recall_at_risk (TPR)']:.0%} vs {rich['recall_at_risk (TPR)']:.0%}",
         f"It flags more students from the poorest areas ({poor.flag_rate:.0%} vs {rich.flag_rate:.0%}), "
         f"but they really do struggle more often ({poor.actual_at_risk_rate:.0%} vs "
         f"{rich.actual_at_risk_rate:.0%}). It catches them at a similar or better rate.", "ok", "Checked"),
        ("Does it suggest useful things to study next?",
         f"Right resource in top 10 for {svd:.0%}",
         f"For {svd:.0%} of students, something they really opened next is in the top 10 suggestions. "
         f"Random picks manage {rnd:.0%} and simply showing the most popular pages gets {pop:.0%}.",
         "ok", "Modest"),
    ]
    body = "".join(
        f"<tr><td>{escape(q)}</td><td>{escape(num)}<br><span class='tag tag-{kind}'>{escape(label)}</span></td>"
        f"<td>{escape(plain_text)}</td></tr>"
        for q, num, plain_text, kind, label in rows)
    render(f"""
    <section class="al-plain" aria-labelledby="plain-title">
      <h2 id="plain-title">What the results mean, in plain words</h2>
      <p>Each line takes one result from testing on students the models never saw and says what it
      means for a teacher or student.</p>
      <div class="al-table-wrap">
        <table class="al-table">
          <thead><tr><th scope="col">The question</th><th scope="col">What we measured</th>
          <th scope="col">What it means</th></tr></thead>
          <tbody>{body}</tbody>
        </table>
      </div>
    </section>
    """)


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
