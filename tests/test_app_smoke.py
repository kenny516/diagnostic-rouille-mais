"""Test de demarrage de l'interface Streamlit sans navigateur externe."""

from __future__ import annotations

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class StreamlitAppSmokeTests(unittest.TestCase):
    def test_app_starts_with_model_and_uploader(self) -> None:
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py"))

        app.run(timeout=30)

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.title[0].value, "Diagnostic de la rouille du maïs")
        self.assertEqual(len(app.get("file_uploader")), 1)
        self.assertIn("Arbre maison Max-Minority", app.caption[0].value)


if __name__ == "__main__":
    unittest.main()

