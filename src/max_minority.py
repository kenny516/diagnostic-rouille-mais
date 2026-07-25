"""Indice Max-Minority et recherche du meilleur seuil, sans scikit-learn."""

from __future__ import annotations

import numpy as np


def _validate_binary_labels(y: np.ndarray) -> None:
    """Verifie que les labels sont non vides et appartiennent a {0, 1}."""

    if y.size == 0:
        raise ValueError("Le tableau de labels ne peut pas etre vide.")
    if not np.all(np.isin(y, (0, 1))):
        raise ValueError("Les labels doivent uniquement contenir 0 et 1.")


def purete_noeud(y: np.ndarray | list[int]) -> float:
    """Retourne P(t), la proportion de la classe majoritaire du noeud."""

    labels = np.asarray(y).reshape(-1)
    _validate_binary_labels(labels)
    counts = np.bincount(labels.astype(np.int8), minlength=2)
    return float(np.max(counts) / labels.size)


def purete_split_ponderee(
    y_gauche: np.ndarray | list[int],
    y_droite: np.ndarray | list[int],
) -> float:
    """Calcule |G|/N P(G) + |D|/N P(D)."""

    gauche = np.asarray(y_gauche).reshape(-1)
    droite = np.asarray(y_droite).reshape(-1)
    if gauche.size == 0 or droite.size == 0:
        raise ValueError("Les deux sous-groupes doivent etre non vides.")
    _validate_binary_labels(gauche)
    _validate_binary_labels(droite)

    total = gauche.size + droite.size
    return float(
        (gauche.size / total) * purete_noeud(gauche)
        + (droite.size / total) * purete_noeud(droite)
    )


def trouver_meilleur_split(
    X_column: np.ndarray | list[float],
    y: np.ndarray | list[int],
) -> tuple[float | None, float]:
    """Trouve le seuil qui maximise la purete Max-Minority ponderee.

    Les donnees sont triees une seule fois. Les effectifs des classes a gauche
    et a droite sont ensuite obtenus par sommes cumulees. En cas d'egalite,
    le plus petit seuil est choisi pour garantir un resultat reproductible.

    Si la variable ne contient qu'une valeur unique, aucun split n'est
    possible : la fonction retourne ``(None, purete_du_noeud_parent)``.
    """

    values = np.asarray(X_column, dtype=np.float64).reshape(-1)
    labels = np.asarray(y).reshape(-1)

    if values.size != labels.size:
        raise ValueError("X_column et y doivent avoir la meme longueur.")
    if values.size == 0:
        raise ValueError("X_column et y ne peuvent pas etre vides.")
    if not np.all(np.isfinite(values)):
        raise ValueError("X_column contient une valeur NaN ou infinie.")
    _validate_binary_labels(labels)
    labels = labels.astype(np.int8)

    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_labels = labels[order]

    # Un seuil n'est valide qu'entre deux valeurs consecutives distinctes.
    candidate_positions = np.flatnonzero(sorted_values[:-1] < sorted_values[1:])
    if candidate_positions.size == 0:
        return None, purete_noeud(sorted_labels)

    cumulative_ones = np.cumsum(sorted_labels, dtype=np.int64)
    total_ones = int(cumulative_ones[-1])
    total_zeros = int(labels.size - total_ones)

    left_sizes = candidate_positions + 1
    left_ones = cumulative_ones[candidate_positions]
    left_zeros = left_sizes - left_ones
    right_ones = total_ones - left_ones
    right_zeros = total_zeros - left_zeros

    # Les ponderations se simplifient : |G|P(G) est l'effectif majoritaire de G.
    weighted_purities = (
        np.maximum(left_zeros, left_ones) + np.maximum(right_zeros, right_ones)
    ) / labels.size

    best_candidate_index = int(np.argmax(weighted_purities))
    split_position = int(candidate_positions[best_candidate_index])
    lower = sorted_values[split_position]
    upper = sorted_values[split_position + 1]
    threshold = lower + (upper - lower) / 2.0

    return float(threshold), float(weighted_purities[best_candidate_index])

