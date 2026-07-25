"""Verifie les livrables techniques du TP et produit une preuve JSON.

Ce script ne remplace pas les tests unitaires. Il controle la coherence globale
des donnees, tableaux, modeles, figures et fichiers de deploiement generes par le
pipeline complet.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    DISEASED_DIR,
    FIGURES_DIR,
    HEALTHY_DIR,
    MODELS_DIR,
    TABLES_DIR,
    ensure_project_directories,
)
from src.deployment import load_model_bundle, predict_uploaded_image  # noqa: E402
from src.image_preprocessing import VALID_IMAGE_EXTENSIONS  # noqa: E402


EXPECTED_FEATURE_COLUMNS = [
    "ID_Image",
    "pct_rouille",
    "rugosite",
    "heterogeneite_saturation",
    "label_malade",
]
EXPECTED_MODELS = [
    "arbre_max_minority.joblib",
    "foret_max_minority.joblib",
    "arbre_scikit_gini.joblib",
    "foret_scikit_gini.joblib",
    "modele_deploiement.pkl",
]
EXPECTED_FIGURES = [
    "leaf_mask_examples.png",
    "image_quality_distributions.png",
    "model_performance.png",
    "confusion_matrices.png",
    "feature_importances.png",
]


def require(condition: bool, message: str) -> None:
    """Arrete la verification avec un message lisible si la condition echoue."""

    if not condition:
        raise AssertionError(message)


def image_paths(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
    )


def main() -> None:
    ensure_project_directories()
    checks: dict[str, object] = {}

    healthy_images = image_paths(HEALTHY_DIR)
    diseased_images = image_paths(DISEASED_DIR)
    require(len(healthy_images) == 400, "Le dossier saines doit contenir 400 images.")
    require(len(diseased_images) == 400, "Le dossier malades doit contenir 400 images.")
    checks["dataset"] = {"saines": 400, "malades": 400, "total": 800}

    audit_path = TABLES_DIR / "image_audit.csv"
    require(audit_path.exists(), "image_audit.csv est absent.")
    audit = pd.read_csv(audit_path)
    require(len(audit) == 800, "L'audit doit contenir 800 lignes.")
    require(bool(audit["valide"].all()), "L'audit contient une image invalide.")
    require(int(audit["doublon_sha256"].sum()) == 0, "Un doublon exact a ete detecte.")
    checks["audit"] = {
        "images_valides": int(audit["valide"].sum()),
        "doublons_sha256": int(audit["doublon_sha256"].sum()),
        "images_signalees": int((audit["probleme_qualite"] != "aucun").sum()),
    }

    features_path = TABLES_DIR / "features.csv"
    require(features_path.exists(), "features.csv est absent.")
    features = pd.read_csv(features_path)
    require(
        features.columns.tolist() == EXPECTED_FEATURE_COLUMNS,
        "Les colonnes de features.csv ne correspondent pas a l'enonce.",
    )
    require(features.shape == (800, 5), "features.csv doit avoir la forme (800, 5).")
    require(not bool(features.isna().any().any()), "features.csv contient une valeur manquante.")
    require(features["ID_Image"].nunique() == 800, "Les ID d'images ne sont pas uniques.")
    require(
        features["label_malade"].value_counts().sort_index().to_dict()
        == {0: 400, 1: 400},
        "Les labels de features.csv ne sont pas equilibres 400/400.",
    )
    require(
        bool(features[["pct_rouille", "rugosite", "heterogeneite_saturation"]]
             .apply(np.isfinite)
             .all()
             .all()),
        "Les caracteristiques contiennent NaN ou infini.",
    )
    checks["features"] = {
        "forme": [800, 5],
        "colonnes": EXPECTED_FEATURE_COLUMNS,
        "valeurs_manquantes": 0,
    }

    split_path = TABLES_DIR / "max_minority_splits.csv"
    require(split_path.exists(), "max_minority_splits.csv est absent.")
    splits = pd.read_csv(split_path)
    require(len(splits) == 3, "Trois variables doivent etre evaluees par Max-Minority.")
    require(
        set(splits["caracteristique"])
        == {"pct_rouille", "rugosite", "heterogeneite_saturation"},
        "Les trois variables ne sont pas presentes dans les splits.",
    )
    checks["max_minority"] = {
        "variables_evaluees": 3,
        "meilleure_variable": str(splits.iloc[0]["caracteristique"]),
        "meilleure_purete": float(splits.iloc[0]["purete_split"]),
    }

    comparison_path = TABLES_DIR / "model_comparison.csv"
    require(comparison_path.exists(), "model_comparison.csv est absent.")
    comparison = pd.read_csv(comparison_path)
    require(len(comparison) == 4, "La comparaison doit contenir quatre modeles.")
    for metric in ("accuracy", "precision", "recall"):
        require(metric in comparison, f"La metrique {metric} est absente.")
        require(
            bool(comparison[metric].between(0, 1).all()),
            f"La metrique {metric} doit etre comprise entre 0 et 1.",
        )
    require(
        bool((comparison["vrais_negatifs"] + comparison["faux_positifs"] == 80).all()),
        "Chaque matrice doit contenir 80 feuilles saines de test.",
    )
    require(
        bool((comparison["faux_negatifs"] + comparison["vrais_positifs"] == 80).all()),
        "Chaque matrice doit contenir 80 feuilles malades de test.",
    )
    checks["comparaison_modeles"] = {
        "modeles": 4,
        "images_test": 160,
        "meilleur_recall": float(comparison["recall"].max()),
    }

    missing_models = [name for name in EXPECTED_MODELS if not (MODELS_DIR / name).exists()]
    require(not missing_models, f"Modeles absents : {missing_models}")
    missing_figures = [name for name in EXPECTED_FIGURES if not (FIGURES_DIR / name).exists()]
    require(not missing_figures, f"Figures absentes : {missing_figures}")
    require(
        all((FIGURES_DIR / name).stat().st_size > 10_000 for name in EXPECTED_FIGURES),
        "Une figure est vide ou anormalement petite.",
    )
    checks["artefacts"] = {
        "modeles": EXPECTED_MODELS,
        "figures": EXPECTED_FIGURES,
    }

    bundle = load_model_bundle(MODELS_DIR / "modele_deploiement.pkl")
    healthy_result = predict_uploaded_image(healthy_images[0].read_bytes(), bundle)
    diseased_result = predict_uploaded_image(diseased_images[0].read_bytes(), bundle)
    require(healthy_result.label in (0, 1), "Prediction saine hors domaine binaire.")
    require(diseased_result.label in (0, 1), "Prediction malade hors domaine binaire.")
    checks["deploiement"] = {
        "modele": bundle["model_name"],
        "prediction_exemple_saine": healthy_result.label,
        "prediction_exemple_malade": diseased_result.label,
    }

    require((PROJECT_ROOT / "app.py").exists(), "app.py est absent.")
    require((PROJECT_ROOT / "docs" / "ETAPES_TP.md").exists(), "La documentation est absente.")
    checks["application_et_documentation"] = {
        "app_streamlit": True,
        "documentation": "docs/ETAPES_TP.md",
    }

    output_path = TABLES_DIR / "project_verification.json"
    output_path.write_text(
        json.dumps({"status": "OK", "checks": checks}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({"status": "OK", "checks": checks}, indent=2, ensure_ascii=False))
    print(f"\nPreuve enregistree : {output_path}")


if __name__ == "__main__":
    main()

