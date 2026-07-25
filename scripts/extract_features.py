"""Construit le tableau de caracteristiques demande dans la partie 1 du TP."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DISEASED_DIR, HEALTHY_DIR, TABLES_DIR, ensure_project_directories  # noqa: E402
from src.feature_extraction import extract_features  # noqa: E402
from src.image_preprocessing import VALID_IMAGE_EXTENSIONS, load_image  # noqa: E402


CLASS_CONFIG = (
    ("saine", 0, HEALTHY_DIR),
    ("malade", 1, DISEASED_DIR),
)


def build_feature_table() -> pd.DataFrame:
    """Parcourt les deux classes et retourne une ligne par image."""

    rows: list[dict[str, float | int | str]] = []
    for class_name, label, directory in CLASS_CONFIG:
        image_paths = sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
        )
        for image_path in image_paths:
            features = extract_features(load_image(image_path))
            rows.append(
                {
                    "ID_Image": image_path.stem,
                    "pct_rouille": features.pct_rouille,
                    "rugosite": features.rugosite,
                    "heterogeneite_saturation": features.heterogeneite_saturation,
                    "label_malade": label,
                }
            )
        print(f"{class_name.capitalize()} : {len(image_paths)} images traitees")

    return pd.DataFrame(
        rows,
        columns=[
            "ID_Image",
            "pct_rouille",
            "rugosite",
            "heterogeneite_saturation",
            "label_malade",
        ],
    )


def main() -> None:
    ensure_project_directories()
    feature_table = build_feature_table()
    output_path = TABLES_DIR / "features.csv"
    feature_table.to_csv(output_path, index=False)

    summary = feature_table.groupby("label_malade").agg(
        images=("ID_Image", "count"),
        pct_rouille_mediane=("pct_rouille", "median"),
        rugosite_mediane=("rugosite", "median"),
        heterogeneite_saturation_mediane=("heterogeneite_saturation", "median"),
    )
    print("\nResume par classe (0=saine, 1=malade) :")
    print(summary.round(4).to_string())
    print(f"\nTableau cree : {output_path}")


if __name__ == "__main__":
    main()

