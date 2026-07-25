"""Application Streamlit de diagnostic de la rouille sur une feuille de mais."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.config import MODELS_DIR, UPLOADS_DIR, ensure_project_directories
from src.deployment import (
    load_history,
    load_model_bundle,
    predict_uploaded_image,
    save_detection,
)


MODEL_PATH = MODELS_DIR / "modele_deploiement.pkl"
HISTORY_PATH = UPLOADS_DIR / "history.json"


@st.cache_resource
def get_model_bundle() -> dict:
    """Charge le modele une seule fois pendant la session Streamlit."""

    return load_model_bundle(MODEL_PATH)


def display_gallery() -> None:
    """Affiche les detections les plus recentes dans une grille de trois colonnes."""

    st.subheader("Historique des détections")
    try:
        history = load_history(HISTORY_PATH)
    except ValueError as error:
        st.warning(str(error))
        return

    if not history:
        st.caption("Aucune image analysée pour le moment.")
        return

    columns = st.columns(3)
    for index, record in enumerate(history):
        image_path = UPLOADS_DIR / record["stored_name"]
        with columns[index % 3]:
            if image_path.exists():
                st.image(str(image_path), width="stretch")
            if int(record["label"]) == 1:
                st.error("Feuille malade")
            else:
                st.success("Feuille saine")
            st.caption(
                f"{record['original_name']} · rouille : "
                f"{record['features']['pct_rouille']:.2f} %"
            )


def main() -> None:
    st.set_page_config(
        page_title="Diagnostic de la rouille du maïs",
        page_icon="🌽",
        layout="wide",
    )
    ensure_project_directories()

    st.title("Diagnostic de la rouille du maïs")
    st.write(
        "Téléversez une photo de feuille. L'application isole la feuille, calcule "
        "les caractéristiques HSV et Sobel, puis applique le modèle Max-Minority."
    )
    st.info(
        "Prototype pédagogique : ce résultat est une aide au diagnostic et ne "
        "remplace pas l'avis d'un technicien agricole."
    )

    try:
        bundle = get_model_bundle()
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
        st.code("python scripts/train_and_compare_models.py")
        display_gallery()
        return

    st.caption(f"Modèle chargé : {bundle['model_name']}")
    uploaded_file = st.file_uploader(
        "Photo de la feuille",
        type=["png", "jpg", "jpeg"],
        help="Formats acceptés : PNG, JPG et JPEG.",
    )

    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        image_column, result_column = st.columns([1.15, 1])
        with image_column:
            st.image(
                image_bytes,
                caption=uploaded_file.name,
                width="stretch",
            )

        try:
            result = predict_uploaded_image(image_bytes, bundle)
            save_detection(
                image_bytes=image_bytes,
                original_name=uploaded_file.name,
                result=result,
                uploads_dir=UPLOADS_DIR,
                history_path=HISTORY_PATH,
            )
        except (ValueError, OSError) as error:
            with result_column:
                st.error(f"Analyse impossible : {error}")
        else:
            with result_column:
                if result.label == 1:
                    st.error("ATTENTION : feuille malade - rouille détectée")
                else:
                    st.success("Feuille saine")

                st.subheader("Caractéristiques extraites")
                metric_columns = st.columns(3)
                metric_columns[0].metric(
                    "Pixels de rouille", f"{result.features.pct_rouille:.2f} %"
                )
                metric_columns[1].metric(
                    "Rugosité Sobel", f"{result.features.rugosite:.2f}"
                )
                metric_columns[2].metric(
                    "Hétérogénéité", f"{result.features.heterogeneite_saturation:.3f}"
                )

    st.divider()
    display_gallery()


if __name__ == "__main__":
    main()
