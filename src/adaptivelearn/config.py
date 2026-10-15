"""Central configuration: paths, column groups, thresholds.

Dataset: Open University Learning Analytics Dataset (OULAD), CC BY 4.0.
Kuzilek, Hlosta, Zdrahal, "Open University Learning Analytics dataset",
Scientific Data 4:170171 (2017), doi:10.1038/sdata.2017.171
"""
from pathlib import Path

SEED = 42

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "oulad_raw"          # the 7 official CSV files (not committed, 450 MB)
PROC_DIR = DATA_DIR / "processed"         # derived tables (committed, small)
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"
MODEL_DIR = RESULTS_DIR / "models"

# ---- prediction set-up -------------------------------------------------------
CUTOFF_DAY = 90                       # predict using only data up to day 90 of the course
CUTOFF_SWEEP = [0, 30, 60, 90, 120, 150, 180]   # "how early can we predict?" experiment
RESULT_ORDER = ["Distinction", "Pass", "Fail", "Withdrawn"]
AT_RISK = ("Fail", "Withdrawn")

# Study-plan tiers on the predicted score of future assessments (0-100)
TIER_SUPPORT_MAX = 55    # < 55   -> foundations / extra support
TIER_ADVANCE_MIN = 75    # >= 75  -> advance / stretch
CHECKPOINT_SCORE = 75

# ---- feature groups ----------------------------------------------------------
DEMOGRAPHIC_CAT = ["gender", "region", "highest_education", "imd_band", "age_band",
                   "disability", "code_module", "season"]
DEMOGRAPHIC_NUM = ["num_of_prev_attempts", "studied_credits", "date_registration",
                   "module_presentation_length"]
ENGAGEMENT_NUM = ["total_clicks", "active_days", "n_sites", "clicks_per_active_day",
                  "clicks_before_start", "clicks_last_14d", "clicks_prev_14d", "click_trend",
                  "days_since_last_active", "share_homepage", "share_content", "share_forum",
                  "share_quiz", "share_other"]
ASSESSMENT_NUM = ["n_assessments_due", "n_submitted", "submission_rate", "mean_score_early",
                  "min_score_early", "n_late", "mean_days_early", "n_failed_early"]

NUMERIC_FEATURES = DEMOGRAPHIC_NUM + ENGAGEMENT_NUM + ASSESSMENT_NUM
CATEGORICAL_FEATURES = DEMOGRAPHIC_CAT
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET_CLASS = "final_result"
TARGET_SCORE = "future_mean_score"    # mean score of assessments due AFTER the cutoff

# Fairness audit attributes (SDG 10)
FAIRNESS_ATTRS = ["gender", "disability", "imd_band", "age_band", "highest_education"]
