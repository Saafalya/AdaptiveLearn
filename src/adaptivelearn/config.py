"""Central configuration: paths, topic taxonomy, column groups and thresholds."""
from pathlib import Path

SEED = 42

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"
MODEL_DIR = RESULTS_DIR / "models"

# Topic key -> display name
TOPICS = {
    "python": "Python",
    "sql": "SQL",
    "statistics": "Statistics",
    "linalg": "Linear Algebra",
    "dsa": "Data Structures & Algorithms",
    "ml": "Machine Learning",
}
TOPIC_COLS = [f"score_{t}" for t in TOPICS]

# Topic prerequisite graph used by the learning-path generator
PREREQS = {
    "python": [],
    "sql": [],
    "statistics": [],
    "linalg": [],
    "dsa": ["python"],
    "ml": ["python", "statistics", "linalg"],
}

# Learner level bands on the (hidden) final assessment score, in percent
LEVELS = ["beginner", "intermediate", "advanced"]
BEGINNER_MAX = 55   # score < 55            -> beginner
ADVANCED_MIN = 75   # score >= 75           -> advanced
WEAK_TOPIC_MAX = 60  # topic score below this triggers remediation

BEHAVIOUR_COLS = [
    "weekly_study_hours",
    "logins_per_week",
    "avg_session_min",
    "assignments_completed_pct",
    "quiz_attempts",
    "forum_posts",
    "days_since_last_active",
    "prior_cgpa",
]
CATEGORICAL_COLS = ["internet_access", "preferred_format"]

# Engineered features (see preprocessing.add_features)
ENGINEERED_NUM = [
    "mean_topic_score",
    "min_topic_score",
    "score_spread",
    "n_topics_attempted",
    "engagement_index",
]
ENGINEERED_CAT = ["weakest_topic"]

NUMERIC_FEATURES = TOPIC_COLS + BEHAVIOUR_COLS + ENGINEERED_NUM
CATEGORICAL_FEATURES = CATEGORICAL_COLS + ENGINEERED_CAT
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET_SCORE = "final_score"
TARGET_LEVEL = "level"
