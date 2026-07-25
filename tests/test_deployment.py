"""Tests du decodage, de la prediction et de l'historique Streamlit."""

from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.deployment import (
    PredictionResult,
    decode_uploaded_image,
    load_history,
    load_model_bundle,
    save_detection,
)
from src.feature_extraction import ImageFeatures


class DeploymentTests(unittest.TestCase):
    def setUp(self) -> None:
        image = np.full((32, 48, 3), (40, 150, 40), dtype=np.uint8)
        success, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(success)
        self.image_bytes = encoded.tobytes()

    def test_decode_uploaded_image(self) -> None:
        decoded = decode_uploaded_image(self.image_bytes)
        self.assertEqual(decoded.shape, (32, 48, 3))
        with self.assertRaises(ValueError):
            decode_uploaded_image(b"not-an-image")

    def test_load_model_bundle_validates_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "model.pkl"
            with model_path.open("wb") as model_file:
                pickle.dump({"model": "placeholder"}, model_file)
            with self.assertRaises(ValueError):
                load_model_bundle(model_path)

    def test_history_is_idempotent_for_same_image(self) -> None:
        result = PredictionResult(
            label=0,
            diagnosis="Saine",
            features=ImageFeatures(0.0, 12.0, 0.05),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            uploads_dir = root / "uploads"
            history_path = uploads_dir / "history.json"

            first = save_detection(
                self.image_bytes, "leaf.jpg", result, uploads_dir, history_path
            )
            second = save_detection(
                self.image_bytes, "leaf-copy.jpg", result, uploads_dir, history_path
            )

            self.assertEqual(first["sha256"], second["sha256"])
            self.assertEqual(len(load_history(history_path)), 1)
            self.assertTrue((uploads_dir / first["stored_name"]).exists())


if __name__ == "__main__":
    unittest.main()

