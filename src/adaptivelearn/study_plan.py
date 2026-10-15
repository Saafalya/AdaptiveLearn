"""Turn model outputs into an explainable, time-ordered study plan.

ML supplies: outcome probabilities, predicted future score, persona, and
recommended sites. A transparent rule layer converts these into concrete steps
tied to the module's *real* upcoming assessments.
"""
import numpy as np
import pandas as pd

from .config import AT_RISK, CHECKPOINT_SCORE, TIER_ADVANCE_MIN, TIER_SUPPORT_MAX

ACTIVITY_HINT = {
    "oucontent": "course content pages", "resource": "downloadable resources",
    "subpage": "course sub-pages", "forumng": "discussion forum", "quiz": "practice quiz",
    "homepage": "module home page", "url": "external links", "page": "course pages",
    "glossary": "glossary", "ouwiki": "wiki", "oucollaborate": "live sessions",
    "questionnaire": "self-check questionnaire", "dataplus": "data tools",
}


def risk_level(p_risk):
    return "High" if p_risk >= 0.6 else "Medium" if p_risk >= 0.35 else "Low"


def score_tier(pred_score):
    if pred_score is None or np.isnan(pred_score):
        return "Foundations"
    return ("Foundations" if pred_score < TIER_SUPPORT_MAX
            else "Core" if pred_score < TIER_ADVANCE_MIN else "Stretch")


def build_study_plan(row, proba, pred_score, persona, recs, upcoming, peer_clicks_14d):
    """
    row              enrolment feature row (Series)
    proba            {result: probability}
    pred_score       predicted mean score of future assessments (float or NaN)
    recs             DataFrame (id_site, activity_type, week_from) from the recommender
    upcoming         DataFrame (assessment_type, date, weight) of assessments after the cut-off
    peer_clicks_14d  median clicks in the last 14 days of Pass/Distinction peers (same module)
    """
    p_risk = sum(proba.get(c, 0.0) for c in AT_RISK)
    tier = score_tier(pred_score)
    plan = {"risk": risk_level(p_risk), "p_at_risk": p_risk, "tier": tier, "steps": []}
    steps = plan["steps"]

    if plan["risk"] == "High":
        steps.append({"when": "This week", "action": "Contact your tutor / student support",
                      "detail": f"Early-warning flag (estimated {p_risk:.0%} chance of Fail/Withdrawal).",
                      "why": "Personal support is the most effective intervention for at-risk students."})

    mine = float(row.get("clicks_last_14d", 0) or 0)
    if peer_clicks_14d and mine < 0.8 * peer_clicks_14d:
        steps.append({"when": "Next 2 weeks", "action": "Raise weekly learning activity",
                      "detail": f"{mine:.0f} clicks in the last 14 days vs about {peer_clicks_14d:.0f} "
                                "for peers who passed.",
                      "why": "Engagement gap versus successful peers in this module."})

    if recs is not None and len(recs):
        kinds = recs["activity_type"].map(lambda a: ACTIVITY_HINT.get(a, a)).tolist()
        steps.append({"when": "This week", "action": "Study these recommended resources",
                      "detail": "; ".join(f"{k} (site {int(s)})" for k, s in
                                          zip(kinds, recs["id_site"])),
                      "why": f"Popular with students in the '{persona}' group that you have not opened yet."})

    for _, a in upcoming.head(3).iterrows():
        kind = {"TMA": "tutor-marked assignment", "CMA": "computer-marked assessment",
                "Exam": "final exam"}.get(a["assessment_type"], a["assessment_type"])
        weight = f", {a['weight']:.0f}% of grade" if pd.notna(a["weight"]) and a["weight"] > 0 else ""
        steps.append({"when": f"Before day {int(a['date'])}" if pd.notna(a["date"]) else "Before the exam",
                      "action": f"Prepare for the {kind}",
                      "detail": f"Tier: {tier}. " + (
                          "Revisit foundations and attempt the practice quiz first." if tier == "Foundations"
                          else "Work through core material, then a practice quiz." if tier == "Core"
                          else "Skim the basics and attempt extension tasks."),
                      "why": f"Checkpoint: self-test score >= {CHECKPOINT_SCORE}% -> move on{weight}; "
                             "otherwise repeat the practice."})
    return plan
