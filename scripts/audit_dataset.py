"""Audite les images, mesure le biais de fond et genere des figures de controle."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src.config import (  # noqa: E402
    DISEASED_DIR,
    FIGURES_DIR,
    HEALTHY_DIR,
    TABLES_DIR,
    ensure_project_directories,
)
from src.image_preprocessing import (  # noqa: E402
    VALID_IMAGE_EXTENSIONS,
    apply_leaf_mask,
    create_leaf_mask,
    load_image,
    masked_values,
)


CLASS_CONFIG = (
    ("saine", 0, HEALTHY_DIR),
    ("malade", 1, DISEASED_DIR),
)
PALETTE = {"saine": "#2F6B9A", "malade": "#C66A2B"}


def sha256_file(path: Path) -> str:
    """Calcule l'empreinte d'un fichier image."""

    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for block in iter(lambda: image_file.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_image(path: Path, class_name: str, label: int) -> dict[str, object]:
    """Calcule les indicateurs de qualite d'une image."""

    record: dict[str, object] = {
        "id_image": path.stem,
        "fichier": str(path.relative_to(PROJECT_ROOT)),
        "classe": class_name,
        "label_malade": label,
        "taille_ko": round(path.stat().st_size / 1024, 3),
        "sha256": sha256_file(path),
        "valide": False,
        "erreur": "",
    }

    try:
        image = load_image(path)
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        leaf_mask = create_leaf_mask(image)
        leaf_pixels = leaf_mask > 0
        dark_pixels = np.max(image, axis=2) <= 20
        laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_64F))

        record.update(
            {
                "largeur": width,
                "hauteur": height,
                "ratio_fond_sombre": float(np.mean(dark_pixels)),
                "ratio_feuille": float(np.mean(leaf_pixels)),
                "luminosite_feuille": float(np.mean(masked_values(gray, leaf_mask))),
                "nettete_feuille": float(np.var(masked_values(laplacian, leaf_mask))),
                "valide": True,
            }
        )
    except Exception as exc:  # L'audit doit continuer et consigner l'image fautive.
        record["erreur"] = str(exc)

    return record


def build_audit_dataframe() -> pd.DataFrame:
    """Parcourt les deux classes et retourne une ligne d'audit par image."""

    records: list[dict[str, object]] = []
    for class_name, label, directory in CLASS_CONFIG:
        paths = sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
        )
        records.extend(audit_image(path, class_name, label) for path in paths)

    audit = pd.DataFrame(records)
    if audit.empty:
        raise RuntimeError("Aucune image trouvee dans le dataset.")

    audit["doublon_sha256"] = audit.duplicated("sha256", keep=False)
    audit["probleme_qualite"] = "aucun"
    audit.loc[~audit["valide"], "probleme_qualite"] = "image_invalide"
    audit.loc[audit["doublon_sha256"], "probleme_qualite"] = "doublon"
    valid = audit["valide"]
    audit.loc[valid & (audit["ratio_feuille"] < 0.20), "probleme_qualite"] = (
        "feuille_trop_petite"
    )
    audit.loc[valid & (audit["nettete_feuille"] < 8.0), "probleme_qualite"] = (
        "image_tres_floue"
    )
    return audit


def build_summary(audit: pd.DataFrame) -> pd.DataFrame:
    """Produit un profil compact par classe et pour l'ensemble."""

    valid = audit[audit["valide"]].copy()
    summary = (
        valid.groupby("classe", sort=False)
        .agg(
            images=("id_image", "count"),
            doublons=("doublon_sha256", "sum"),
            ratio_fond_sombre_moyen=("ratio_fond_sombre", "mean"),
            ratio_feuille_moyen=("ratio_feuille", "mean"),
            luminosite_mediane=("luminosite_feuille", "median"),
            nettete_mediane=("nettete_feuille", "median"),
        )
        .reset_index()
    )
    summary["images_invalides"] = audit.groupby("classe")["valide"].apply(
        lambda values: int((~values).sum())
    ).values

    numeric_columns = summary.select_dtypes(include="number").columns
    summary[numeric_columns] = summary[numeric_columns].round(4)
    return summary


def plot_audit_distributions(audit: pd.DataFrame, output_path: Path) -> None:
    """Compare les distributions des indicateurs entre les deux classes."""

    valid = audit[audit["valide"]].copy()
    metrics = (
        ("ratio_fond_sombre", "Pixels sombres dans l'image", "Proportion"),
        ("ratio_feuille", "Surface retenue par le masque", "Proportion"),
        ("luminosite_feuille", "Luminosite de la feuille", "Intensite 0-255"),
        ("nettete_feuille", "Nettete de la feuille", "Variance du Laplacien"),
    )

    sns.set_theme(style="whitegrid", font_scale=0.95)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Distribution des indicateurs de qualite par classe", fontsize=16, y=0.98)
    fig.text(
        0.5,
        0.94,
        f"{len(valid)} images valides - mesures brutes et mesures limitees a la feuille",
        ha="center",
        color="#555555",
        fontsize=10,
    )

    for axis, (column, title, ylabel) in zip(axes.flat, metrics, strict=True):
        sns.boxplot(
            data=valid,
            x="classe",
            y=column,
            hue="classe",
            order=["saine", "malade"],
            hue_order=["saine", "malade"],
            palette=PALETTE,
            legend=False,
            width=0.55,
            linewidth=1.1,
            fliersize=2.5,
            ax=axis,
        )
        axis.set_title(title, loc="left", fontsize=11, fontweight="bold")
        axis.set_xlabel("")
        axis.set_ylabel(ylabel)
        axis.grid(axis="x", visible=False)
        axis.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(output_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_mask_examples(audit: pd.DataFrame, output_path: Path) -> None:
    """Affiche deux exemples par classe : original, masque, resultat nettoye."""

    selected: list[tuple[str, Path]] = []
    for class_name in ("saine", "malade"):
        class_rows = audit[(audit["classe"] == class_name) & audit["valide"]].sort_values(
            "id_image"
        )
        indices = [0, len(class_rows) // 2]
        selected.extend(
            (class_name, PROJECT_ROOT / class_rows.iloc[index]["fichier"])
            for index in indices
        )

    fig, axes = plt.subplots(len(selected), 3, figsize=(10, 11))
    fig.suptitle("Controle visuel du masque de feuille", fontsize=16, y=0.99)
    fig.text(
        0.5,
        0.96,
        "Le fond est exclu sans modifier les images originales",
        ha="center",
        color="#555555",
        fontsize=10,
    )

    column_titles = ("Image originale", "Masque binaire", "Feuille isolee")
    for column, title in enumerate(column_titles):
        axes[0, column].set_title(title, fontsize=11, fontweight="bold")

    for row, (class_name, image_path) in enumerate(selected):
        image_bgr = load_image(image_path)
        leaf_mask = create_leaf_mask(image_bgr)
        masked = apply_leaf_mask(image_bgr, leaf_mask)
        axes[row, 0].imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        axes[row, 1].imshow(leaf_mask, cmap="gray", vmin=0, vmax=255)
        axes[row, 2].imshow(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB))
        axes[row, 0].set_ylabel(class_name.capitalize(), fontweight="bold")
        for column, axis in enumerate(axes[row]):
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(column == 1)
                spine.set_color("#555555")
                spine.set_linewidth(0.8)
        retained_ratio = 100 * np.count_nonzero(leaf_mask) / leaf_mask.size
        axes[row, 1].set_xlabel(
            f"{retained_ratio:.1f} % retenu", fontsize=8, color="#444444"
        )

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Genere uniquement les tableaux CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_project_directories()

    audit = build_audit_dataframe()
    summary = build_summary(audit)

    audit_path = TABLES_DIR / "image_audit.csv"
    summary_path = TABLES_DIR / "image_quality_summary.csv"
    audit.to_csv(audit_path, index=False, encoding="utf-8")
    summary.to_csv(summary_path, index=False, encoding="utf-8")

    if not args.skip_figures:
        plot_audit_distributions(audit, FIGURES_DIR / "image_quality_distributions.png")
        plot_mask_examples(audit, FIGURES_DIR / "leaf_mask_examples.png")

    print(summary.to_string(index=False))
    print(f"\nAudit detaille : {audit_path}")
    print(f"Resume : {summary_path}")


if __name__ == "__main__":
    main()
