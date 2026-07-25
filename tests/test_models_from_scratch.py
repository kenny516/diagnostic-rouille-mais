"""Tests des modeles Max-Minority faits maison."""

from __future__ import annotations

import unittest

import numpy as np

from src.models_from_scratch import (
    MaxMinorityDecisionTreeClassifier,
    MaxMinorityRandomForestClassifier,
)


class DecisionTreeTests(unittest.TestCase):
    def test_tree_learns_a_perfect_threshold(self) -> None:
        X = np.array([[0.0], [1.0], [2.0], [8.0], [9.0], [10.0]])
        y = np.array([0, 0, 0, 1, 1, 1])
        model = MaxMinorityDecisionTreeClassifier(max_depth=2, random_state=42)

        model.fit(X, y)

        np.testing.assert_array_equal(model.predict(X), y)
        self.assertAlmostEqual(model.tree_.threshold, 5.0)
        self.assertAlmostEqual(model.feature_importances_.sum(), 1.0)

    def test_max_depth_zero_produces_a_leaf(self) -> None:
        model = MaxMinorityDecisionTreeClassifier(max_depth=0)
        model.fit([[0], [1], [2]], [0, 1, 1])

        self.assertTrue(model.tree_.is_leaf)
        np.testing.assert_array_equal(model.predict([[10], [-5]]), [1, 1])

    def test_predict_before_fit_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            MaxMinorityDecisionTreeClassifier().predict([[1.0]])

    def test_invalid_training_shapes_raise(self) -> None:
        with self.assertRaises(ValueError):
            MaxMinorityDecisionTreeClassifier().fit([[1], [2]], [0])


class RandomForestTests(unittest.TestCase):
    def test_forest_is_reproducible_and_accurate(self) -> None:
        X = np.arange(40, dtype=float).reshape(-1, 1)
        y = (X[:, 0] >= 20).astype(np.int8)
        first = MaxMinorityRandomForestClassifier(
            n_estimators=15, max_depth=3, random_state=7
        ).fit(X, y)
        second = MaxMinorityRandomForestClassifier(
            n_estimators=15, max_depth=3, random_state=7
        ).fit(X, y)

        np.testing.assert_array_equal(first.predict(X), second.predict(X))
        self.assertGreaterEqual(np.mean(first.predict(X) == y), 0.95)
        self.assertEqual(len(first.estimators_), 15)

    def test_probabilities_sum_to_one(self) -> None:
        model = MaxMinorityRandomForestClassifier(
            n_estimators=5, max_depth=2, random_state=1
        ).fit([[0], [1], [8], [9]], [0, 0, 1, 1])

        probabilities = model.predict_proba([[0], [9]])

        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        self.assertEqual(probabilities.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()

