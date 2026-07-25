"""Tests de l'indice Max-Minority et de la recherche de seuil."""

from __future__ import annotations

import unittest

import numpy as np

from src.max_minority import (
    purete_noeud,
    purete_split_ponderee,
    trouver_meilleur_split,
)


def brute_force_reference(
    values: np.ndarray, labels: np.ndarray
) -> tuple[float | None, float]:
    """Version simple utilisee uniquement comme oracle dans les tests."""

    unique_values = np.unique(values)
    if unique_values.size < 2:
        return None, purete_noeud(labels)

    thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0
    candidates = []
    for threshold in thresholds:
        left = labels[values <= threshold]
        right = labels[values > threshold]
        left_counts = np.bincount(left.astype(np.int8), minlength=2)
        right_counts = np.bincount(right.astype(np.int8), minlength=2)
        exact_purity = (np.max(left_counts) + np.max(right_counts)) / labels.size
        candidates.append((float(threshold), float(exact_purity)))
    return max(candidates, key=lambda candidate: (candidate[1], -candidate[0]))


class MaxMinorityTests(unittest.TestCase):
    def test_node_purity_matches_statement_example(self) -> None:
        labels = np.array([0] * 90 + [1] * 10)
        self.assertAlmostEqual(purete_noeud(labels), 0.9)
        self.assertAlmostEqual(purete_noeud([0, 1]), 0.5)

    def test_perfect_split_is_found(self) -> None:
        threshold, purity = trouver_meilleur_split([1, 2, 8, 9], [0, 0, 1, 1])
        self.assertAlmostEqual(threshold, 5.0)
        self.assertAlmostEqual(purity, 1.0)

    def test_duplicate_values_are_not_split(self) -> None:
        threshold, purity = trouver_meilleur_split(
            [1, 1, 1, 3, 3, 3], [0, 0, 1, 1, 1, 0]
        )
        self.assertAlmostEqual(threshold, 2.0)
        self.assertAlmostEqual(purity, 4 / 6)

    def test_constant_feature_returns_parent_purity(self) -> None:
        threshold, purity = trouver_meilleur_split([4, 4, 4], [0, 0, 1])
        self.assertIsNone(threshold)
        self.assertAlmostEqual(purity, 2 / 3)

    def test_optimized_search_matches_brute_force(self) -> None:
        generator = np.random.default_rng(42)
        for _ in range(30):
            values = generator.integers(0, 12, size=80).astype(float)
            labels = generator.integers(0, 2, size=80)
            expected = brute_force_reference(values, labels)
            actual = trouver_meilleur_split(values, labels)
            self.assertAlmostEqual(actual[0], expected[0])
            self.assertAlmostEqual(actual[1], expected[1])

    def test_invalid_inputs_raise_clear_errors(self) -> None:
        with self.assertRaises(ValueError):
            trouver_meilleur_split([1, 2], [0])
        with self.assertRaises(ValueError):
            trouver_meilleur_split([1, np.nan], [0, 1])
        with self.assertRaises(ValueError):
            trouver_meilleur_split([1, 2], [0, 2])


if __name__ == "__main__":
    unittest.main()
