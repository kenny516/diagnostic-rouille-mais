"""Fonctions partagees par l'application Streamlit de diagnostic.

La logique de decodage, prediction et persistance est separee de l'interface afin
de pouvoir etre testee sans demarrer un navigateur.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from src.feature_extraction import ImageFeatures, extract_features


@dataclass(frozen=True)
class PredictionResult:
    """Resultat complet affiche et sauvegarde par l'application."""

    label: int
    diagnosis: str
    features: ImageFeatures


def load_model_bundle(model_path: Path) -> dict[str, Any]:
    """Charge le modele et ses metadonnees avec pickle.load."""

    if not model_path.exists():
        raise FileNotFoundError(
            f"Modele introuvable : {model_path}. Lancez le script d'entrainement."
        )
    with model_path.open("rb") as model_file:
        bundle = pickle.load(model_file)

    required_keys = {"model", "model_name", "feature_columns", "class_names"}
    missing_keys = required_keys - set(bundle)
    if missing_keys:
        raise ValueError(f"Le fichier modele est incomplet : {sorted(missing_keys)}")
    return bundle


def decode_uploaded_image(image_bytes: bytes) -> np.ndarray:
    """Decode un fichier PNG/JPEG en image BGR OpenCV."""

    if not image_bytes:
        raise ValueError("Le fichier televerse est vide.")
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Le fichier n'est pas une image PNG ou JPEG valide.")
    return image_bgr


def predict_uploaded_image(
    image_bytes: bytes, bundle: dict[str, Any]
) -> PredictionResult:
    """Extrait les trois variables puis applique le modele serialise."""

    image_bgr = decode_uploaded_image(image_bytes)
    features = extract_features(image_bgr)
    feature_values = {
        "pct_rouille": features.pct_rouille,
        "rugosite": features.rugosite,
        "heterogeneite_saturation": features.heterogeneite_saturation,
    }
    feature_columns = list(bundle["feature_columns"])
    missing_features = sorted(set(feature_columns) - set(feature_values))
    if missing_features:
        raise ValueError(f"Variables non calculees : {missing_features}")

    feature_frame = pd.DataFrame(
        [[feature_values[column] for column in feature_columns]],
        columns=feature_columns,
    )
    label = int(bundle["model"].predict(feature_frame)[0])
    class_names = bundle["class_names"]
    diagnosis = class_names.get(label, class_names.get(str(label), str(label)))
    return PredictionResult(label=label, diagnosis=diagnosis, features=features)


def load_history(history_path: Path) -> list[dict[str, Any]]:
    """Lit l'historique JSON ; un fichier absent correspond a une galerie vide."""

    if not history_path.exists():
        return []
    try:
        content = json.loads(history_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"Historique illisible : {history_path}") from error
    if not isinstance(content, list):
        raise ValueError("Le contenu de l'historique doit etre une liste JSON.")
    return content


def save_detection(
    image_bytes: bytes,
    original_name: str,
    result: PredictionResult,
    uploads_dir: Path,
    history_path: Path,
) -> dict[str, Any]:
    """Sauvegarde une detection une seule fois, identifiee par SHA-256."""

    uploads_dir.mkdir(parents=True, exist_ok=True)
    history = load_history(history_path)
    digest = hashlib.sha256(image_bytes).hexdigest()

    for record in history:
        if record.get("sha256") == digest:
            return record

    suffix = Path(original_name).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png"}:
        suffix = ".jpg"
    stored_name = f"{digest[:16]}{suffix}"
    image_path = uploads_dir / stored_name
    image_path.write_bytes(image_bytes)

    record = {
        "sha256": digest,
        "original_name": Path(original_name).name,
        "stored_name": stored_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "label": result.label,
        "diagnosis": result.diagnosis,
        "features": asdict(result.features),
    }
    history.insert(0, record)
    history_path.write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return record

