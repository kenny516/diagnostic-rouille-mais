"""Tests des trois caracteristiques visuelles du TP."""

from __future__ import annotations

import unittest

import numpy as np

from src.feature_extraction import create_rust_mask, extract_features


class FeatureExtractionTests(unittest.TestCase):
    def test_rust_percentage_ignores_black_background(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[10:90, 10:90] = (40, 140, 40)
        image[10:50, 10:90] = (20, 100, 180)

        features = extract_features(image)

        self.assertAlmostEqual(features.pct_rouille, 50.0, delta=3.0)

    def test_uniform_leaf_has_low_saturation_heterogeneity(self) -> None:
        image = np.full((80, 80, 3), (40, 150, 40), dtype=np.uint8)

        features = extract_features(image)

        self.assertLess(features.heterogeneite_saturation, 1e-6)

    def test_texture_increases_sobel_roughness(self) -> None:
        flat = np.full((100, 100, 3), (60, 150, 60), dtype=np.uint8)
        textured = flat.copy()
        for column in range(0, 100, 4):
            textured[:, column : column + 2] = (160, 220, 160)

        self.assertGreater(
            extract_features(textured).rugosite,
            extract_features(flat).rugosite + 10.0,
        )

    def test_rust_mask_has_image_dimensions(self) -> None:
        image = np.full((32, 48, 3), (40, 150, 40), dtype=np.uint8)
        leaf_mask = np.full((32, 48), 255, dtype=np.uint8)

        rust_mask = create_rust_mask(image, leaf_mask)

        self.assertEqual(rust_mask.shape, image.shape[:2])
        self.assertEqual(rust_mask.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()

