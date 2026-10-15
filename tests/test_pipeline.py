"""Fast sanity tests (no OULAD download needed):  python -m unittest discover -s tests"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptivelearn.config import (ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES,  # noqa: E402
                                  RESULT_ORDER)
from adaptivelearn.oulad import make_key  # noqa: E402
from adaptivelearn.preprocessing import group_split, make_preprocessor  # noqa: E402
from adaptivelearn.recommender import ranking_metrics  # noqa: E402
from adaptivelearn.study_plan import build_study_plan, risk_level, score_tier  # noqa: E402


def fake_frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(size=n) for c in NUMERIC_FEATURES})
    for c in CATEGORICAL_FEATURES:
        df[c] = rng.choice(["a", "b", "c"], n)
    df["id_student"] = rng.integers(0, 150, n)          # repeated students
    df["final_result"] = rng.choice(RESULT_ORDER, n)
    return df


class PreprocessingTests(unittest.TestCase):
    def test_nan_and_unseen_categories_are_handled(self):
        df = fake_frame()
        df.loc[::7, "mean_score_early"] = np.nan
        df.loc[::11, "gender"] = np.nan
        prep = make_preprocessor().fit(df[ALL_FEATURES])
        new = fake_frame(30, seed=1)
        new["region"] = "never-seen"
        out = prep.transform(new[ALL_FEATURES])
        self.assertFalse(np.isnan(out).any())

    def test_group_split_keeps_students_apart(self):
        tr, te = group_split(fake_frame())
        self.assertFalse(set(tr.id_student) & set(te.id_student))
        self.assertAlmostEqual(len(te) / (len(tr) + len(te)), 0.2, delta=0.08)

    def test_make_key_is_unique_per_enrolment(self):
        mod = pd.Series(["AAA", "AAA", "BBB"])
        pres = pd.Series(["2013J", "2014J", "2013J"])
        stu = pd.Series([5, 5, 5])
        k = make_key(mod, pres, stu, {"AAA": 0, "BBB": 1}, {"2013J": 0, "2014J": 1})
        self.assertEqual(k.nunique(), 3)


class RecommenderTests(unittest.TestCase):
    def test_perfect_ranking_scores_one_and_seen_items_excluded(self):
        X_train = sp.csr_matrix(np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]], dtype=float))
        X_test = sp.csr_matrix(np.array([[0, 0, 1, 0, 0], [0, 0, 0, 1, 0]], dtype=float))
        scores = np.array([[9.0, 0, 5, 0, 0], [0, 9.0, 0, 5, 0]])   # seen item has highest score
        m = ranking_metrics(scores, X_train, X_test, k=1)
        self.assertEqual(m["hit_rate@1"], 1.0)
        self.assertAlmostEqual(m["ndcg@1"], 1.0)


class StudyPlanTests(unittest.TestCase):
    def test_tiers_and_risk(self):
        self.assertEqual(score_tier(40), "Foundations")
        self.assertEqual(score_tier(65), "Core")
        self.assertEqual(score_tier(90), "Stretch")
        self.assertEqual(risk_level(0.7), "High")
        self.assertEqual(risk_level(0.1), "Low")

    def test_high_risk_gets_support_step_first(self):
        up = pd.DataFrame({"assessment_type": ["TMA"], "date": [117], "weight": [20.0]})
        plan = build_study_plan(pd.Series({"clicks_last_14d": 5}),
                                {"Fail": 0.5, "Withdrawn": 0.3, "Pass": 0.2},
                                48.0, "Disengaged / at-risk", pd.DataFrame(), up, 100.0)
        self.assertEqual(plan["risk"], "High")
        self.assertIn("tutor", plan["steps"][0]["action"].lower())
        self.assertTrue(any("Prepare" in s["action"] for s in plan["steps"]))


if __name__ == "__main__":
    unittest.main()
