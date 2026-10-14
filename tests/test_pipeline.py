"""Fast sanity tests: python -m unittest discover -s tests"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptivelearn.config import ALL_FEATURES, PREREQS, TOPICS  # noqa: E402
from adaptivelearn.data_generation import generate_dataset  # noqa: E402
from adaptivelearn.learning_path import build_learning_path  # noqa: E402
from adaptivelearn.preprocessing import make_preprocessor, prepare  # noqa: E402
from adaptivelearn.recommender import ranking_metrics  # noqa: E402


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.students, cls.resources, cls.ratings = generate_dataset(n_students=300, seed=1)

    def test_deterministic(self):
        again, _, _ = generate_dataset(n_students=300, seed=1)
        self.assertTrue(self.students.equals(again))

    def test_shapes_and_ranges(self):
        self.assertEqual(len(self.students), 300)
        self.assertEqual(len(self.ratings), 300 * 15)
        self.assertTrue(self.ratings.rating.between(1, 5).all())
        self.assertTrue(set(self.students.level) <= {"beginner", "intermediate", "advanced"})

    def test_preprocessing_handles_missing_and_outliers(self):
        df = prepare(self.students)
        self.assertLessEqual(df.weekly_study_hours.max(), 60)
        out = make_preprocessor().fit_transform(df[ALL_FEATURES])
        out = out.toarray() if hasattr(out, "toarray") else out
        self.assertFalse(np.isnan(out).any())


class LogicTests(unittest.TestCase):
    def test_prerequisites_come_first(self):
        _, resources, _ = generate_dataset(n_students=50, seed=2)
        scores = {t: 80.0 for t in TOPICS}
        scores.update({"ml": 30.0, "python": 40.0, "statistics": 45.0})
        ratings = {r: 3.0 for r in resources.resource_id}
        steps = build_learning_path(scores, ratings, resources)
        order = []
        for s in steps:
            if s["topic"] not in order and s["action"] != "Stretch":
                order.append(s["topic"])
        names = {t: TOPICS[t] for t in TOPICS}
        for topic, prereqs in PREREQS.items():
            if names[topic] in order:
                for p in prereqs:
                    if names[p] in order:
                        self.assertLess(order.index(names[p]), order.index(names[topic]))

    def test_ranking_metrics_perfect_and_empty(self):
        test = np.full((2, 6), np.nan)
        test[0, 2], test[1, 4] = 5, 4
        train_mask = np.zeros((2, 6), dtype=bool)
        perfect = np.zeros((2, 6))
        perfect[0, 2], perfect[1, 4] = 10, 10
        m = ranking_metrics(perfect, train_mask, test, k=3)
        self.assertEqual(m["hit_rate@3"], 1.0)
        self.assertAlmostEqual(m["ndcg@3"], 1.0)


if __name__ == "__main__":
    unittest.main()
