"""Tests du masque de feuille et de son application."""

from __future__ import annotations

import unittest

import numpy as np

from src.config import DISEASED_DIR, HEALTHY_DIR
from src.image_preprocessing import apply_leaf_mask, create_leaf_mask, load_image


class LeafMaskTests(unittest.TestCase):
    def test_synthetic_leaf_is_separated_from_black_background(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[25:75, 20:80] = (40, 150, 60)

        mask = create_leaf_mask(image)

        self.assertEqual(mask[50, 50], 255)
        self.assertEqual(mask[0, 0], 0)
        self.assertGreater(np.mean(mask > 0), 0.25)
        self.assertLess(np.mean(mask > 0), 0.40)

    def test_mask_removes_background_pixels(self) -> None:
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        image[10:40, 10:40] = (30, 120, 50)
        mask = create_leaf_mask(image)
        result = apply_leaf_mask(image, mask)

        self.assertTrue(np.all(result[0, 0] == 0))
        self.assertTrue(np.any(result[25, 25] > 0))

    def test_real_images_produce_non_empty_masks(self) -> None:
        for directory in (HEALTHY_DIR, DISEASED_DIR):
            image = load_image(next(directory.glob("*.jpg")))
            mask = create_leaf_mask(image)
            self.assertGreater(np.mean(mask > 0), 0.15)


if __name__ == "__main__":
    unittest.main()
