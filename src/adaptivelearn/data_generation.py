"""Synthetic learner dataset generator.

No public dataset ships with per-topic diagnostics, behavioural logs *and*
resource-level feedback together, so a documented simulator is used. The
generative story (see docs/report.md, section "Dataset") is:

    aptitude, motivation, internet access  ->  topic mastery (with prerequisite effects)
    mastery + engagement                   ->  hidden final assessment score -> level
    mastery vs. resource difficulty/format ->  resource ratings (1-5)

Missing values, entry-error outliers and a rating-noise floor are injected so
the preprocessing stage has real work to do.
"""
import numpy as np
import pandas as pd

from .config import (ADVANCED_MIN, BEGINNER_MAX, DATA_DIR, LEVELS, SEED,
                     TOPICS)

LEVEL_NAMES = {1: "Fundamentals", 2: "Core", 3: "Advanced"}
RESOURCE_TYPES = [("video", 25), ("reading", 20), ("practice", 35)]


def level_from_score(score):
    """Map a final assessment score (0-100) to beginner/intermediate/advanced."""
    score = np.asarray(score, dtype=float)
    return np.where(score < BEGINNER_MAX, LEVELS[0],
                    np.where(score < ADVANCED_MIN, LEVELS[1], LEVELS[2]))


def appropriate_resource_level(score):
    """Difficulty tier (1-3) that best matches a topic score."""
    score = np.asarray(score, dtype=float)
    return np.where(score < BEGINNER_MAX, 1, np.where(score < ADVANCED_MIN, 2, 3))


def build_resources():
    rows = []
    for key, name in TOPICS.items():
        for lvl, lname in LEVEL_NAMES.items():
            for rtype, minutes in RESOURCE_TYPES:
                rows.append({
                    "resource_id": f"{key[:3].upper()}-L{lvl}-{rtype[:3].upper()}",
                    "topic": key,
                    "level": lvl,
                    "rtype": rtype,
                    "duration_min": minutes + 10 * (lvl - 1),
                    "title": f"{name}: {lname} ({rtype})",
                })
    return pd.DataFrame(rows)


def generate_dataset(n_students=1500, seed=SEED, ratings_per_student=15):
    """Return (students, resources, ratings) DataFrames."""
    rng = np.random.default_rng(seed)
    n = n_students

    # --- latent traits and context -----------------------------------------
    aptitude = rng.normal(0, 1, n)
    motivation = 0.3 * aptitude + rng.normal(0, 0.95, n)
    internet = rng.choice(["limited", "moderate", "good"], n, p=[0.2, 0.4, 0.4])
    net = pd.Series(internet).map({"limited": -1.0, "moderate": 0.0, "good": 0.5}).to_numpy()
    preferred = rng.choice(["video", "reading", "practice"], n, p=[0.4, 0.25, 0.35])

    # --- behavioural logs ---------------------------------------------------
    study_hours = np.clip(9 + 3.5 * motivation + 1.0 * aptitude + rng.normal(0, 2, n), 1, 40)
    logins = np.clip(np.round(5 + 1.8 * motivation + 1.0 * net + rng.normal(0, 1.5, n)), 0, 21)
    session = np.clip(38 + 8 * motivation + 5 * net + rng.normal(0, 9, n), 5, 120)
    assign = np.clip(72 + 11 * motivation + 3 * net + rng.normal(0, 8, n), 5, 100)
    quiz = rng.poisson(np.clip(6 + 2 * motivation, 0.5, None))
    forum = rng.poisson(np.clip(2 + 1.0 * motivation + 0.3 * aptitude, 0.2, None))
    idle = np.clip(np.round(rng.exponential(np.clip(4 - 1.5 * motivation, 0.8, None))), 0, 60)
    cgpa = np.clip(6.8 + 0.9 * aptitude + 0.3 * motivation + rng.normal(0, 0.5, n), 3, 10)

    # --- topic mastery with prerequisite structure --------------------------
    base = 0.85 * aptitude + 0.45 * motivation + 0.25 * net
    m = {t: base + 0.5 * rng.normal(0, 1, n) for t in ("python", "sql", "statistics", "linalg")}
    m["dsa"] = base + 0.5 * rng.normal(0, 1, n) + 0.3 * m["python"]
    m["ml"] = (base + 0.5 * rng.normal(0, 1, n)
               + 0.3 * (m["python"] + m["statistics"] + m["linalg"]))
    true_scores = {t: np.clip(66 + 10 * m[t] + rng.normal(0, 5, n), 0, 100) for t in TOPICS}
    mean_topic = np.mean([true_scores[t] for t in TOPICS], axis=0)

    # --- hidden final assessment (the prediction target) --------------------
    final = (0.75 * mean_topic + 0.17 * assign
             + 0.08 * np.minimum(study_hours, 30) / 30 * 100 + rng.normal(0, 4.5, n))
    final = np.clip(final, 0, 100)

    students = pd.DataFrame({
        "student_id": [f"S{i:04d}" for i in range(1, n + 1)],
        "internet_access": internet,
        "preferred_format": preferred,
        "weekly_study_hours": study_hours.round(1),
        "logins_per_week": logins,
        "avg_session_min": session.round(1),
        "assignments_completed_pct": assign.round(1),
        "quiz_attempts": quiz,
        "forum_posts": forum,
        "days_since_last_active": idle,
        "prior_cgpa": cgpa.round(2),
    })
    for t in TOPICS:
        students[f"score_{t}"] = true_scores[t].round(1)
    students["final_score"] = final.round(1)
    students["level"] = level_from_score(students["final_score"])

    # --- ratings are generated from the TRUE scores, before data corruption --
    resources = build_resources()
    ratings = _generate_ratings(students, true_scores, resources, rng, ratings_per_student)

    # --- data quality problems injected AFTER ratings are generated ---------
    for t in TOPICS:  # "topic not attempted yet"
        students.loc[rng.random(n) < 0.06, f"score_{t}"] = np.nan
    for col, p in [("avg_session_min", 0.03), ("logins_per_week", 0.03),
                   ("prior_cgpa", 0.05), ("quiz_attempts", 0.02)]:
        students.loc[rng.random(n) < p, col] = np.nan
    bad = rng.choice(n, size=max(1, n // 100), replace=False)  # entry errors
    students.loc[bad, "weekly_study_hours"] = rng.uniform(80, 150, len(bad)).round(1)

    return students, resources, ratings


def _generate_ratings(students, true_scores, resources, rng, per_student):
    n, n_res = len(students), len(resources)
    topic_idx = resources["topic"].map({t: i for i, t in enumerate(TOPICS)}).to_numpy()
    score_mat = np.column_stack([true_scores[t] for t in TOPICS])      # n x topics
    res_scores = score_mat[:, topic_idx]                                # n x resources
    appropriate = appropriate_resource_level(res_scores)
    level_gap = np.abs(resources["level"].to_numpy()[None, :] - appropriate)
    fmt_match = (resources["rtype"].to_numpy()[None, :]
                 == students["preferred_format"].to_numpy()[:, None])
    quality = rng.normal(0, 0.3, n_res)                                 # intrinsic resource quality

    fit = (4.3 - 1.2 * level_gap + np.where(fmt_match, 0.5, -0.1)
           + quality[None, :] + rng.normal(0, 0.6, (n, n_res)))
    rating = np.clip(np.round(fit), 1, 5).astype(int)

    # Learners tend to consume resources on topics they are weak at.
    weights = np.exp(-(res_scores - 60) / 30)
    weights /= weights.sum(axis=1, keepdims=True)
    rows = []
    sids = students["student_id"].to_numpy()
    rids = resources["resource_id"].to_numpy()
    for u in range(n):
        picked = rng.choice(n_res, size=per_student, replace=False, p=weights[u])
        rows.extend((sids[u], rids[i], int(rating[u, i])) for i in picked)
    return pd.DataFrame(rows, columns=["student_id", "resource_id", "rating"])


def save_dataset(students, resources, ratings, directory=DATA_DIR):
    directory.mkdir(parents=True, exist_ok=True)
    students.to_csv(directory / "students.csv", index=False)
    resources.to_csv(directory / "resources.csv", index=False)
    ratings.to_csv(directory / "ratings.csv", index=False)


def load_dataset(directory=DATA_DIR):
    return (pd.read_csv(directory / "students.csv"),
            pd.read_csv(directory / "resources.csv"),
            pd.read_csv(directory / "ratings.csv"))
