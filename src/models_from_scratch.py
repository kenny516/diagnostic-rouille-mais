"""Arbre et foret Max-Minority implementes uniquement avec NumPy.

Ces classes reprennent volontairement une petite partie de l'interface de
scikit-learn (``fit``, ``predict`` et ``predict_proba``) afin de faciliter la
comparaison, mais leur apprentissage est entierement code dans ce projet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.max_minority import purete_noeud, trouver_meilleur_split


MaxFeatures = int | float | Literal["sqrt", "log2"] | None


@dataclass
class TreeNode:
    """Un noeud de l'arbre de decision.

    Un noeud feuille ne possede ni ``feature_index`` ni enfants. Tous les
    noeuds gardent leur prediction majoritaire, ce qui permet un repli propre.
    """

    prediction: int
    n_samples: int
    purity: float
    feature_index: int | None = None
    threshold: float | None = None
    split_purity: float | None = None
    left: TreeNode | None = None
    right: TreeNode | None = None

    @property
    def is_leaf(self) -> bool:
        """Indique si le noeud est terminal."""

        return self.feature_index is None


def _as_feature_matrix(X: object) -> tuple[np.ndarray, np.ndarray | None]:
    """Convertit les entrees en matrice 2D et conserve les noms de colonnes."""

    feature_names = None
    if hasattr(X, "columns"):
        feature_names = np.asarray(getattr(X, "columns"), dtype=object)
    matrix = np.asarray(X, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("X doit etre une matrice 2D non vide.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("X contient une valeur NaN ou infinie.")
    return matrix, feature_names


def _as_binary_labels(y: object, expected_size: int) -> np.ndarray:
    """Valide et convertit les labels binaires."""

    labels = np.asarray(y).reshape(-1)
    if labels.size != expected_size:
        raise ValueError("X et y doivent contenir le meme nombre de lignes.")
    if not np.all(np.isin(labels, (0, 1))):
        raise ValueError("y doit uniquement contenir les labels 0 et 1.")
    return labels.astype(np.int8)


def _majority_class(y: np.ndarray) -> int:
    """Retourne la classe majoritaire (0 en cas d'egalite)."""

    counts = np.bincount(y, minlength=2)
    return int(np.argmax(counts))


def _resolve_max_features(max_features: MaxFeatures, n_features: int) -> int:
    """Transforme le parametre max_features en nombre de variables."""

    if max_features is None:
        return n_features
    if max_features == "sqrt":
        return max(1, int(np.sqrt(n_features)))
    if max_features == "log2":
        return max(1, int(np.log2(n_features)))
    if isinstance(max_features, int) and not isinstance(max_features, bool):
        if not 1 <= max_features <= n_features:
            raise ValueError("max_features entier doit etre compris entre 1 et n_features.")
        return max_features
    if isinstance(max_features, float):
        if not 0.0 < max_features <= 1.0:
            raise ValueError("max_features flottant doit appartenir a ]0, 1].")
        return max(1, int(np.ceil(max_features * n_features)))
    raise ValueError("max_features doit etre None, un nombre, 'sqrt' ou 'log2'.")


class MaxMinorityDecisionTreeClassifier:
    """Arbre binaire dont chaque split maximise la purete Max-Minority."""

    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 2,
        max_features: MaxFeatures = None,
        min_purity_gain: float = 0.0,
        random_state: int | None = None,
    ) -> None:
        if max_depth < 0:
            raise ValueError("max_depth doit etre positif ou nul.")
        if min_samples_split < 2:
            raise ValueError("min_samples_split doit etre superieur ou egal a 2.")
        if min_purity_gain < 0:
            raise ValueError("min_purity_gain doit etre positif ou nul.")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.min_purity_gain = min_purity_gain
        self.random_state = random_state

    def fit(self, X: object, y: object) -> MaxMinorityDecisionTreeClassifier:
        """Construit recursivement l'arbre a partir des donnees d'apprentissage."""

        matrix, feature_names = _as_feature_matrix(X)
        labels = _as_binary_labels(y, matrix.shape[0])

        self.n_features_in_ = matrix.shape[1]
        self.feature_names_in_ = (
            feature_names
            if feature_names is not None
            else np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)
        )
        self.classes_ = np.array([0, 1], dtype=np.int8)
        self._max_features_count = _resolve_max_features(
            self.max_features, self.n_features_in_
        )
        self._rng = np.random.default_rng(self.random_state)
        self._raw_feature_importances = np.zeros(self.n_features_in_, dtype=np.float64)
        self.tree_ = self._build_tree(matrix, labels, depth=0)

        total_importance = float(np.sum(self._raw_feature_importances))
        if total_importance > 0:
            self.feature_importances_ = self._raw_feature_importances / total_importance
        else:
            self.feature_importances_ = self._raw_feature_importances.copy()
        return self

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> TreeNode:
        """Cree un noeud, cherche son meilleur split puis construit ses enfants."""

        node_purity = purete_noeud(y)
        node = TreeNode(
            prediction=_majority_class(y),
            n_samples=y.size,
            purity=node_purity,
        )

        # Conditions d'arret demandees par l'enonce, plus la taille minimale.
        if (
            node_purity >= 1.0
            or depth >= self.max_depth
            or y.size < self.min_samples_split
        ):
            return node

        if self._max_features_count == self.n_features_in_:
            candidate_features = np.arange(self.n_features_in_)
        else:
            candidate_features = np.sort(
                self._rng.choice(
                    self.n_features_in_,
                    size=self._max_features_count,
                    replace=False,
                )
            )

        best_feature = None
        best_threshold = None
        best_split_purity = -np.inf

        for feature_index in candidate_features:
            threshold, split_purity = trouver_meilleur_split(X[:, feature_index], y)
            if threshold is None:
                continue
            if split_purity > best_split_purity + 1e-12:
                best_feature = int(feature_index)
                best_threshold = threshold
                best_split_purity = split_purity

        purity_gain = best_split_purity - node_purity
        if (
            best_feature is None
            or best_threshold is None
            or purity_gain <= self.min_purity_gain + 1e-12
        ):
            return node

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        if not np.any(left_mask) or not np.any(right_mask):
            return node

        node.feature_index = best_feature
        node.threshold = float(best_threshold)
        node.split_purity = float(best_split_purity)
        self._raw_feature_importances[best_feature] += y.size * purity_gain
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return node

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "tree_"):
            raise RuntimeError("Le modele doit etre entraine avec fit avant la prediction.")

    def _validate_prediction_matrix(self, X: object) -> np.ndarray:
        self._check_is_fitted()
        matrix, _ = _as_feature_matrix(X)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError("X ne contient pas le meme nombre de variables qu'a l'entrainement.")
        return matrix

    def _predict_row(self, row: np.ndarray) -> int:
        node = self.tree_
        while not node.is_leaf:
            if row[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.prediction

    def predict(self, X: object) -> np.ndarray:
        """Predit 0 (saine) ou 1 (malade) pour chaque ligne."""

        matrix = self._validate_prediction_matrix(X)
        return np.asarray([self._predict_row(row) for row in matrix], dtype=np.int8)

    def predict_proba(self, X: object) -> np.ndarray:
        """Retourne des probabilites binaires basees sur la classe des feuilles."""

        predictions = self.predict(X).astype(np.float64)
        return np.column_stack((1.0 - predictions, predictions))


class MaxMinorityRandomForestClassifier:
    """Foret simplifiee : bootstrap, sous-ensemble de variables et vote majoritaire."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        min_samples_split: int = 2,
        max_features: MaxFeatures = "sqrt",
        min_purity_gain: float = 0.0,
        random_state: int | None = None,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators doit etre superieur ou egal a 1.")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.min_purity_gain = min_purity_gain
        self.random_state = random_state

    def fit(self, X: object, y: object) -> MaxMinorityRandomForestClassifier:
        """Entraine chaque arbre sur un bootstrap tire avec remplacement."""

        matrix, feature_names = _as_feature_matrix(X)
        labels = _as_binary_labels(y, matrix.shape[0])
        self.n_features_in_ = matrix.shape[1]
        self.feature_names_in_ = (
            feature_names
            if feature_names is not None
            else np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)
        )
        self.classes_ = np.array([0, 1], dtype=np.int8)

        generator = np.random.default_rng(self.random_state)
        self.estimators_: list[MaxMinorityDecisionTreeClassifier] = []
        n_samples = matrix.shape[0]

        for _ in range(self.n_estimators):
            # np.random.choice est la methode de bagging explicitement demandee.
            bootstrap_indices = generator.choice(
                n_samples, size=n_samples, replace=True
            )
            tree_seed = int(generator.integers(0, np.iinfo(np.int32).max))
            tree = MaxMinorityDecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                min_purity_gain=self.min_purity_gain,
                random_state=tree_seed,
            )
            tree.fit(matrix[bootstrap_indices], labels[bootstrap_indices])
            self.estimators_.append(tree)

        mean_importance = np.mean(
            [tree.feature_importances_ for tree in self.estimators_], axis=0
        )
        importance_total = float(np.sum(mean_importance))
        self.feature_importances_ = (
            mean_importance / importance_total
            if importance_total > 0
            else mean_importance
        )
        return self

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "estimators_"):
            raise RuntimeError("Le modele doit etre entraine avec fit avant la prediction.")

    def predict_proba(self, X: object) -> np.ndarray:
        """Moyenne les votes des arbres pour produire une probabilite."""

        self._check_is_fitted()
        matrix, _ = _as_feature_matrix(X)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError("X ne contient pas le meme nombre de variables qu'a l'entrainement.")
        votes_malades = np.mean(
            [tree.predict(matrix) for tree in self.estimators_], axis=0
        )
        return np.column_stack((1.0 - votes_malades, votes_malades))

    def predict(self, X: object) -> np.ndarray:
        """Applique le vote majoritaire (egalite classee malade par prudence)."""

        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int8)

