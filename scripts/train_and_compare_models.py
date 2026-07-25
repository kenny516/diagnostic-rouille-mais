"""Entraine et compare les quatre modeles demandes dans la partie 3 du TP.

Sorties principales :
- quatre modeles serialises dans ``models/`` ;
- metriques et predictions dans ``outputs/tables/`` ;
- graphiques comparatifs dans ``outputs/figures/``.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from src.config import (  # noqa: E402
    FIGURES_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    TABLES_DIR,
    TEST_SIZE,
    ensure_project_directories,
)
from src.models_from_scratch import (  # noqa: E402
    MaxMinorityDecisionTreeClassifier,
    MaxMinorityRandomForestClassifier,
)


FEATURE_COLUMNS = [
    "pct_rouille",
    "rugosite",
    "heterogeneite_saturation",
]
MODEL_CONFIGS = {
    "Arbre maison\nMax-Minority": "arbre_max_minority.joblib",
    "Foret maison\nMax-Minority": "foret_max_minority.joblib",
    "Arbre scikit-learn\nGini": "arbre_scikit_gini.joblib",
    "Foret scikit-learn\nGini": "foret_scikit_gini.joblib",
}
MODEL_SLUGS = {
    "Arbre maison\nMax-Minority": "arbre_maison",
    "Foret maison\nMax-Minority": "foret_maison",
    "Arbre scikit-learn\nGini": "arbre_scikit",
    "Foret scikit-learn\nGini": "foret_scikit",
}

BLUE = "#2F6B9A"
BLUE_LIGHT = "#8FB7D3"
ORANGE = "#C66A2B"
ORANGE_LIGHT = "#E4AE82"
NEUTRAL = "#747474"
INK = "#262626"
GRID = "#D9D9D9"


def create_models() -> dict[str, object]:
    """Instancie les quatre configurations avec une profondeur comparable."""

    return {
        "Arbre maison\nMax-Minority": MaxMinorityDecisionTreeClassifier(
            max_depth=5,
            random_state=RANDOM_STATE,
        ),
        "Foret maison\nMax-Minority": MaxMinorityRandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            max_features="sqrt",
            random_state=RANDOM_STATE,
        ),
        "Arbre scikit-learn\nGini": DecisionTreeClassifier(
            criterion="gini",
            max_depth=5,
            random_state=RANDOM_STATE,
        ),
        "Foret scikit-learn\nGini": RandomForestClassifier(
            n_estimators=100,
            criterion="gini",
            max_depth=5,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def evaluate_model(
    model_name: str,
    model: object,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict[str, float | int | str], np.ndarray, np.ndarray]:
    """Entraine un modele puis calcule les metriques de la classe malade."""

    start = time.perf_counter()
    model.fit(X_train, y_train)
    training_duration = time.perf_counter() - start
    predictions = np.asarray(model.predict(X_test), dtype=np.int8)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()

    result = {
        "modele": model_name.replace("\n", " - "),
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "vrais_negatifs": int(true_negative),
        "faux_positifs": int(false_positive),
        "faux_negatifs": int(false_negative),
        "vrais_positifs": int(true_positive),
        "temps_entrainement_s": training_duration,
    }
    return result, predictions, matrix


def plot_model_metrics(metrics: pd.DataFrame, output_path: Path) -> None:
    """Compare accuracy, precision et recall sur une echelle commune 0-1."""

    metric_specs = (
        ("accuracy", "Exactitude", BLUE),
        ("precision", "Precision", ORANGE),
        ("recall", "Rappel", NEUTRAL),
    )
    x = np.arange(len(metrics))
    width = 0.23

    fig, axis = plt.subplots(figsize=(11, 6.2))
    for index, (column, label, color) in enumerate(metric_specs):
        positions = x + (index - 1) * width
        bars = axis.bar(
            positions,
            metrics[column],
            width=width,
            label=label,
            color=color,
            edgecolor=INK,
            linewidth=0.5,
        )
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8, color=INK)

    axis.set_title(
        "Comparaison des performances sur le jeu de test",
        loc="left",
        fontsize=16,
        fontweight="bold",
        color=INK,
        pad=28,
    )
    axis.text(
        0,
        1.02,
        "160 images - classe positive : malade (1) - split stratifie 80/20, seed 42",
        transform=axis.transAxes,
        fontsize=10,
        color="#555555",
    )
    axis.set_ylabel("Score (0 a 1)")
    axis.set_xticks(x, [name.replace(" - ", "\n") for name in metrics["modele"]])
    axis.set_ylim(0, 1.08)
    axis.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.grid(axis="x", visible=False)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(bottom=0.25, top=0.84, left=0.09, right=0.98)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_confusion_matrices(
    matrices: dict[str, np.ndarray], output_path: Path
) -> None:
    """Affiche les quatre matrices avec la meme echelle de couleur."""

    cmap = LinearSegmentedColormap.from_list("custom_blue", ["#F2F6F9", BLUE])
    maximum = max(int(matrix.max()) for matrix in matrices.values())
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
    images = []

    for axis, (model_name, matrix) in zip(axes.flat, matrices.items()):
        image = axis.imshow(matrix, cmap=cmap, vmin=0, vmax=maximum)
        images.append(image)
        for row in range(2):
            for column in range(2):
                value = int(matrix[row, column])
                text_color = "white" if value > maximum * 0.55 else INK
                axis.text(
                    column,
                    row,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=17,
                    fontweight="bold",
                    color=text_color,
                )
        axis.set_title(model_name, fontsize=11, fontweight="bold", color=INK)
        axis.set_xticks([0, 1], ["Saine (0)", "Malade (1)"])
        axis.set_yticks([0, 1], ["Saine (0)", "Malade (1)"])
        axis.set_xlabel("Prediction")
        axis.set_ylabel("Valeur reelle")

    fig.suptitle(
        "Matrices de confusion des quatre modeles",
        fontsize=16,
        fontweight="bold",
        color=INK,
    )
    fig.colorbar(images[0], ax=axes, shrink=0.75, label="Nombre d'images")
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_feature_importances(importances: pd.DataFrame, output_path: Path) -> None:
    """Compare l'importance normalisee des trois variables dans chaque modele."""

    pivot = importances.pivot(
        index="caracteristique", columns="modele", values="importance"
    ).loc[FEATURE_COLUMNS]
    model_order = [name.replace("\n", " - ") for name in MODEL_CONFIGS]
    pivot = pivot[model_order]
    colors = [BLUE, BLUE_LIGHT, ORANGE, ORANGE_LIGHT]
    hatches = ["", "//", "", "//"]

    x = np.arange(len(FEATURE_COLUMNS))
    width = 0.19
    fig, axis = plt.subplots(figsize=(11, 6.2))
    for index, model_name in enumerate(model_order):
        positions = x + (index - 1.5) * width
        bars = axis.bar(
            positions,
            pivot[model_name],
            width=width,
            label=model_name,
            color=colors[index],
            hatch=hatches[index],
            edgecolor=INK,
            linewidth=0.5,
        )
        axis.bar_label(bars, fmt="%.2f", padding=2, fontsize=7)

    axis.set_title(
        "Importance des caracteristiques par modele",
        loc="left",
        fontsize=16,
        fontweight="bold",
        color=INK,
        pad=28,
    )
    axis.text(
        0,
        1.02,
        "Importance normalisee - les valeurs de chaque modele totalisent 1",
        transform=axis.transAxes,
        fontsize=10,
        color="#555555",
    )
    axis.set_ylabel("Importance normalisee")
    axis.set_xticks(x, FEATURE_COLUMNS)
    axis.set_ylim(0, max(1.05, float(pivot.to_numpy().max()) + 0.12))
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.grid(axis="x", visible=False)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    fig.subplots_adjust(bottom=0.25, top=0.84, left=0.09, right=0.98)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    ensure_project_directories()
    feature_path = TABLES_DIR / "features.csv"
    if not feature_path.exists():
        raise FileNotFoundError(
            "Le fichier features.csv est absent. Lancez d'abord "
            "`python scripts/extract_features.py`."
        )

    data = pd.read_csv(feature_path)
    required_columns = ["ID_Image", *FEATURE_COLUMNS, "label_malade"]
    missing_columns = sorted(set(required_columns) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Colonnes absentes de features.csv : {missing_columns}")

    X = data[FEATURE_COLUMNS]
    y = data["label_malade"].to_numpy(dtype=np.int8)
    indices = np.arange(len(data))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    models = create_models()
    metric_rows = []
    confusion_matrices = {}
    prediction_table = pd.DataFrame(
        {
            "ID_Image": data.iloc[test_indices]["ID_Image"].to_numpy(),
            "label_reel": y_test,
        }
    )
    importance_rows = []

    for model_name, model in models.items():
        result, predictions, matrix = evaluate_model(
            model_name, model, X_train, X_test, y_train, y_test
        )
        metric_rows.append(result)
        confusion_matrices[model_name] = matrix
        prediction_table[f"prediction_{MODEL_SLUGS[model_name]}"] = predictions

        model_path = MODELS_DIR / MODEL_CONFIGS[model_name]
        joblib.dump(model, model_path)
        for feature_name, importance in zip(
            FEATURE_COLUMNS, model.feature_importances_
        ):
            importance_rows.append(
                {
                    "modele": model_name.replace("\n", " - "),
                    "caracteristique": feature_name,
                    "importance": float(importance),
                }
            )

    # L'arbre maison est retenu pour la demonstration : rappel 1.00 sur le test.
    deployment_bundle = {
        "model": models["Arbre maison\nMax-Minority"],
        "model_name": "Arbre maison Max-Minority",
        "feature_columns": FEATURE_COLUMNS,
        "class_names": {0: "Saine", 1: "Malade - Rouille detectee"},
    }
    with (MODELS_DIR / "modele_deploiement.pkl").open("wb") as model_file:
        pickle.dump(deployment_bundle, model_file)

    metrics = pd.DataFrame(metric_rows)
    importances = pd.DataFrame(importance_rows)
    metrics.to_csv(TABLES_DIR / "model_comparison.csv", index=False)
    prediction_table.to_csv(TABLES_DIR / "test_predictions.csv", index=False)
    importances.to_csv(TABLES_DIR / "feature_importances.csv", index=False)

    split_table = pd.DataFrame(
        {
            "ID_Image": data["ID_Image"],
            "ensemble": np.where(
                np.isin(indices, test_indices), "test", "entrainement"
            ),
            "label_malade": y,
        }
    )
    split_table.to_csv(TABLES_DIR / "data_split.csv", index=False)

    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "n_train": int(len(train_indices)),
        "n_test": int(len(test_indices)),
        "positive_class": 1,
        "positive_class_name": "malade",
        "models": MODEL_CONFIGS,
    }
    (MODELS_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    plot_model_metrics(metrics, FIGURES_DIR / "model_performance.png")
    plot_confusion_matrices(
        confusion_matrices, FIGURES_DIR / "confusion_matrices.png"
    )
    plot_feature_importances(
        importances, FIGURES_DIR / "feature_importances.png"
    )

    display_columns = [
        "modele",
        "accuracy",
        "precision",
        "recall",
        "faux_positifs",
        "faux_negatifs",
        "temps_entrainement_s",
    ]
    print(metrics[display_columns].round(4).to_string(index=False))
    print(f"\nModeles sauvegardes dans : {MODELS_DIR}")
    print(f"Resultats sauvegardes dans : {TABLES_DIR}")


if __name__ == "__main__":
    main()
