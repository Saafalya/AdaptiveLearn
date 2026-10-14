"""Turn model outputs into an ordered, prerequisite-aware learning path."""
import math

import numpy as np

from .config import ADVANCED_MIN, BEGINNER_MAX, PREREQS, TOPICS, WEAK_TOPIC_MAX

PREREQ_OK = 65  # a prerequisite below this is remediated before its dependent topic


def _target_level(score):
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return 1
    return 1 if score < BEGINNER_MAX else 2 if score < ADVANCED_MIN else 3


def _topological(topics):
    """Order topics so every prerequisite precedes its dependents."""
    ordered, seen = [], set()

    def visit(t):
        if t in seen:
            return
        seen.add(t)
        for p in PREREQS[t]:
            if p in topics:
                visit(p)
        ordered.append(t)

    for t in topics:
        visit(t)
    return ordered


def build_learning_path(topic_scores, predicted_ratings, resources, max_topics=3):
    """
    topic_scores      {topic_key: score or NaN}
    predicted_ratings {resource_id: predicted rating} (higher = better fit)
    resources         resources DataFrame
    """
    def score_of(t):
        v = topic_scores.get(t)
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)

    weak = [t for t in TOPICS if score_of(t) is None or score_of(t) < WEAK_TOPIC_MAX]
    weak.sort(key=lambda t: (score_of(t) is not None, score_of(t) or 0))
    focus = set(weak[:max_topics])
    for t in list(focus):  # pull in shaky prerequisites
        for p in PREREQS[t]:
            if score_of(p) is None or score_of(p) < PREREQ_OK:
                focus.add(p)
    ordered = _topological(focus)

    by_topic = {t: resources[resources.topic == t] for t in TOPICS}

    def best(t, level, rtype=None):
        pool = by_topic[t][by_topic[t].level == level]
        if rtype:
            pool = pool[pool.rtype == rtype]
        if pool.empty:
            return None
        return pool.loc[pool["resource_id"].map(predicted_ratings).idxmax()]

    steps = []
    for t in ordered:
        s, name, lvl = score_of(t), TOPICS[t], _target_level(score_of(t))
        if s is None:
            steps.append({"topic": name, "action": "Diagnostic quiz", "resource": f"{name}: diagnostic",
                          "minutes": 15, "why": "No score yet - establish a baseline first."})
        candidates = [r for r in (best(t, lvl, "video"), best(t, lvl, "reading")) if r is not None]
        learn = max(candidates, key=lambda r: predicted_ratings[r["resource_id"]], default=None)
        practice = best(t, lvl, "practice")
        why = ("below the beginner threshold" if s is not None and s < BEGINNER_MAX
               else "below the remediation threshold" if s is not None else "unassessed topic")
        if t in PREREQS and not focus.isdisjoint({d for d, ps in PREREQS.items() if t in ps}):
            why += "; prerequisite for a topic on this path"
        for res, action in ((learn, "Learn"), (practice, "Practise")):
            if res is not None:
                steps.append({"topic": name, "action": action, "resource": res["title"],
                              "resource_id": res["resource_id"], "minutes": int(res["duration_min"]),
                              "predicted_rating": round(float(predicted_ratings[res["resource_id"]]), 2),
                              "why": why})
        steps.append({"topic": name, "action": "Checkpoint quiz", "resource": f"{name}: checkpoint",
                      "minutes": 15,
                      "why": f"Score >= {ADVANCED_MIN}% -> advance to the next tier; otherwise repeat practice."})

    # Stretch goals for topics already mastered
    for t in TOPICS:
        s = score_of(t)
        if s is not None and s >= ADVANCED_MIN and t not in focus:
            res = best(t, 3)
            if res is not None:
                steps.append({"topic": TOPICS[t], "action": "Stretch", "resource": res["title"],
                              "resource_id": res["resource_id"], "minutes": int(res["duration_min"]),
                              "predicted_rating": round(float(predicted_ratings[res["resource_id"]]), 2),
                              "why": "Already strong - deepen with advanced material."})
    return steps
