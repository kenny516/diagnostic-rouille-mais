"""Extraction des caracteristiques visuelles demandees dans la partie 1 du TP."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.image_preprocessing import create_leaf_mask, masked_values


# OpenCV code la teinte HSV entre 0 et 179.
# Cet intervalle vise les tons brun-orange typiques des pustules de rouille.
RUST_HSV_LOWER = np.array([3, 55, 25], dtype=np.uint8)
RUST_HSV_UPPER = np.array([30, 255, 255], dtype=np.uint8)


@dataclass(frozen=True)
class ImageFeatures:
    """Vecteur de caracteristiques d'une image de feuille."""

    pct_rouille: float
    rugosite: float
    heterogeneite_saturation: float


def create_rust_mask(image_bgr: np.ndarray, leaf_mask: np.ndarray) -> np.ndarray:
    """Isole les pixels brun-orange, uniquement a l'interieur de la feuille."""

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv, RUST_HSV_LOWER, RUST_HSV_UPPER)
    return cv2.bitwise_and(color_mask, leaf_mask)


def _interior_leaf_mask(leaf_mask: np.ndarray) -> np.ndarray:
    """Retire le contour feuille/fond qui fausserait le gradient de Sobel."""

    kernel = np.ones((5, 5), dtype=np.uint8)
    interior = cv2.erode(leaf_mask, kernel, iterations=1)
    if np.count_nonzero(interior) < 0.5 * np.count_nonzero(leaf_mask):
        return leaf_mask
    return interior


def extract_features(image_bgr: np.ndarray) -> ImageFeatures:
    """Calcule pct_rouille, rugosite Sobel et heterogeneite de saturation."""

    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("L'image est vide ou illisible.")

    leaf_mask = create_leaf_mask(image_bgr)
    leaf_pixel_count = int(np.count_nonzero(leaf_mask))
    if leaf_pixel_count == 0:
        raise ValueError("Aucun pixel de feuille n'a ete detecte.")

    rust_mask = create_rust_mask(image_bgr, leaf_mask)
    pct_rouille = 100.0 * np.count_nonzero(rust_mask) / leaf_pixel_count

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(sobel_x, sobel_y)
    rugosite = float(np.mean(masked_values(gradient, _interior_leaf_mask(leaf_mask))))

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    saturation = masked_values(hsv[:, :, 1], leaf_mask).astype(np.float32) / 255.0
    heterogeneite_saturation = float(np.std(saturation))

    return ImageFeatures(
        pct_rouille=float(pct_rouille),
        rugosite=rugosite,
        heterogeneite_saturation=heterogeneite_saturation,
    )

