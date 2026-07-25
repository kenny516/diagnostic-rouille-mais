"""Calcule le meilleur split Max-Minority pour chaque caracteristique."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TABLES_DIR, ensure_project_directories  # noqa: E402
from src.max_minority import purete_noeud, trouver_meilleur_split  # noqa: E402


FEATURE_COLUMNS = (
    "pct_rouille",
    "rugosite",
    "heterogeneite_saturation",
)


def main() -> None:
    ensure_project_directories()
    input_path = TABLES_DIR / "features.csv"
    if not input_path.exists():
        raise FileNotFoundError(
            "Le fichier features.csv est absent. Lancez d'abord "
            "`python scripts/extract_features.py`."
        )

    data = pd.read_csv(input_path)
    labels = data["label_malade"].to_numpy()
    parent_purity = purete_noeud(labels)

    rows: list[dict[str, float | str]] = []
    for feature_name in FEATURE_COLUMNS:
        threshold, split_purity = trouver_meilleur_split(
            data[feature_name].to_numpy(), labels
        )
        rows.append(
            {
                "caracteristique": feature_name,
                "seuil_optimal": threshold,
                "purete_parent": parent_purity,
                "purete_split": split_purity,
                "gain_purete": split_purity - parent_purity,
            }
        )

    results = pd.DataFrame(rows).sort_values("purete_split", ascending=False)
    output_path = TABLES_DIR / "max_minority_splits.csv"
    results.to_csv(output_path, index=False)

    print(results.round(6).to_string(index=False))
    print(f"\nResultats enregistres : {output_path}")


if __name__ == "__main__":
    main()

