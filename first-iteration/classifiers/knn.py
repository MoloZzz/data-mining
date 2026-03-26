"""
classifiers/knn.py
------------------
Реалізація k-Nearest Neighbours (kNN) з нуля.

Відстані:
  euclidean  — √Σ(xᵢ - yᵢ)²
  manhattan  — Σ|xᵢ - yᵢ|
  cosine     — 1 - (x·y)/(‖x‖·‖y‖)

Зважування:
  uniform  — кожен сусід має однаковий голос
  distance — голос зважується як 1 / d  (ближчі важливіші)

Примітка: перед використанням рекомендується нормалізація (min-max або z-score).
"""

import numpy as np
from collections import Counter


class KNNClassifier:
    """
    Параметри
    ---------
    k          : int    — кількість сусідів (непарне значення зменшує нічиї)
    metric     : str    — 'euclidean' | 'manhattan' | 'cosine'
    weights    : str    — 'uniform' | 'distance'
    """

    METRICS = ("euclidean", "manhattan", "cosine")
    WEIGHTS = ("uniform", "distance")

    def __init__(self,
                 k: int = 5,
                 metric: str = "euclidean",
                 weights: str = "uniform"):
        if metric not in self.METRICS:
            raise ValueError(f"metric має бути одним із {self.METRICS}")
        if weights not in self.WEIGHTS:
            raise ValueError(f"weights має бути одним із {self.WEIGHTS}")
        if k < 1:
            raise ValueError("k >= 1")

        self.k       = k
        self.metric  = metric
        self.weights = weights

        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None

    # ──────────────────────────────────────────
    # fit  — просто запам'ятовуємо дані
    # ──────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        self._X_train = np.array(X, dtype=float)
        self._y_train = np.array(y)
        print(f"[kNN] k={self.k}  metric={self.metric}  "
              f"weights={self.weights}  |  "
              f"збережено {len(y)} прикладів")
        return self

    # ──────────────────────────────────────────
    # Відстані
    # ──────────────────────────────────────────

    def _distances(self, x: np.ndarray) -> np.ndarray:
        """Відстані від одного вектора x до всіх навчальних точок."""
        if self.metric == "euclidean":
            diff = self._X_train - x
            return np.sqrt((diff ** 2).sum(axis=1))

        elif self.metric == "manhattan":
            return np.abs(self._X_train - x).sum(axis=1)

        elif self.metric == "cosine":
            dot    = self._X_train @ x
            norm_t = np.linalg.norm(self._X_train, axis=1)
            norm_x = np.linalg.norm(x)
            denom  = norm_t * norm_x
            denom[denom == 0] = 1e-10
            return 1.0 - dot / denom

    # ──────────────────────────────────────────
    # Голосування по k сусідах
    # ──────────────────────────────────────────

    def _vote(self, distances: np.ndarray,
              neighbor_labels: np.ndarray) -> int:
        if self.weights == "uniform":
            return int(Counter(neighbor_labels).most_common(1)[0][0])

        # weights == 'distance'
        inv_dist = np.where(distances == 0, 1e10, 1.0 / distances)
        votes: dict = {}
        for label, w in zip(neighbor_labels, inv_dist):
            votes[label] = votes.get(label, 0.0) + w
        return int(max(votes, key=votes.get))

    # ──────────────────────────────────────────
    # predict (з батч-векторизацією)
    # ──────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Для кожного тестового прикладу знаходимо k найближчих сусідів
        і проводимо голосування.

        Векторизована версія для euclidean дозволяє уникнути Python-циклу
        по навчальним прикладам: використовуємо (A-B)² = A²-2AB+B².
        Для manhattan/cosine — стандартний рядковий цикл.
        """
        X = np.array(X, dtype=float)

        if self.metric == "euclidean" and self.weights == "uniform":
            return self._predict_euclidean_fast(X)

        return np.array([self._predict_one(x) for x in X])

    def _predict_one(self, x: np.ndarray) -> int:
        dists   = self._distances(x)
        k_idx   = np.argpartition(dists, self.k)[:self.k]
        k_dists = dists[k_idx]
        k_label = self._y_train[k_idx]
        return self._vote(k_dists, k_label)

    def _predict_euclidean_fast(self, X: np.ndarray) -> np.ndarray:
        """
        Векторизований euclidean + uniform: обчислюємо матрицю відстаней
        одним матричним множенням.
        shape X_train: (n, d),  X_test: (m, d)
        dist²[i,j] = ||Xtr[j]||² - 2·Xtr[j]·Xte[i]ᵀ + ||Xte[i]||²
        """
        sq_train = (self._X_train ** 2).sum(axis=1)   # (n,)
        sq_test  = (X ** 2).sum(axis=1)               # (m,)
        cross    = X @ self._X_train.T                # (m, n)

        dist2 = sq_train[None, :] - 2 * cross + sq_test[:, None]
        dist2 = np.maximum(dist2, 0)                  # числова стабільність

        # Топ-k індексів по кожному тестовому прикладу
        preds = []
        for i in range(len(X)):
            k_idx  = np.argpartition(dist2[i], self.k)[:self.k]
            labels = self._y_train[k_idx]
            preds.append(Counter(labels).most_common(1)[0][0])
        return np.array(preds)

    # ──────────────────────────────────────────
    # predict_proba  (м'яке голосування)
    # ──────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Повертає матрицю (n_samples, n_classes) з частками голосів.
        """
        X = np.array(X, dtype=float)
        classes = np.unique(self._y_train)
        class_idx = {c: i for i, c in enumerate(classes)}
        proba = np.zeros((len(X), len(classes)))

        for i, x in enumerate(X):
            dists  = self._distances(x)
            k_idx  = np.argpartition(dists, self.k)[:self.k]
            labels = self._y_train[k_idx]
            for lbl in labels:
                proba[i, class_idx[lbl]] += 1.0
        proba /= self.k
        return proba
