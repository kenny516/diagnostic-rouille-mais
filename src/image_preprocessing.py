"""Chargement des images et isolation non destructive de la feuille."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load_image(image_path: str | Path) -> np.ndarray:
    """Charge une image en BGR et signale clairement les fichiers illisibles."""

    path = Path(image_path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Image illisible ou format non pris en charge : {path}")
    return image


def create_leaf_mask(image_bgr: np.ndarray, dark_threshold: int = 20) -> np.ndarray:
    """Retourne un masque binaire 0/255 couvrant la feuille.

    PlantVillage contient souvent un fond noir autour des feuilles malades. Le masque
    commence par retirer les pixels presque noirs, consolide la zone utile par morphologie,
    puis conserve et remplit le plus grand contour externe. Cela inclut les pustules et les
    zones necrosees internes sans reintroduire le fond.
    """

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("L'image doit avoir trois canaux BGR.")

    max_channel = np.max(image_bgr, axis=2)
    initial_mask = (max_channel > dark_threshold).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(initial_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.full(image_bgr.shape[:2], 255, dtype=np.uint8)

    largest_contour = max(contours, key=cv2.contourArea)
    leaf_mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(leaf_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # Une image presque entierement non noire correspond a une feuille cadrant tout le plan.
    if np.mean(initial_mask > 0) > 0.98:
        leaf_mask[:] = 255

    return leaf_mask


def apply_leaf_mask(image_bgr: np.ndarray, leaf_mask: np.ndarray) -> np.ndarray:
    """Met a zero le fond tout en conservant les pixels de la feuille."""

    if image_bgr.shape[:2] != leaf_mask.shape:
        raise ValueError("Le masque et l'image doivent avoir les memes dimensions.")
    return cv2.bitwise_and(image_bgr, image_bgr, mask=leaf_mask)


def masked_values(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Extrait les valeurs situees dans le masque, avec controle du masque vide."""

    values = array[mask > 0]
    if values.size == 0:
        raise ValueError("Le masque de feuille ne contient aucun pixel.")
    return values
