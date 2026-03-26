"""
classifiers/naive_bayes.py
--------------------------
Реалізація Gaussian Naive Bayes з нуля.

Формула:
  P(y | x) ∝ P(y) · ∏ P(x_j | y)

  P(x_j | y) — Gaussian PDF:
      f(x) = 1 / sqrt(2π·σ²) · exp( -(x - μ)² / (2σ²) )

  Класифікація:
      ŷ = argmax_c [ log P(c) + Σ_j log P(x_j | c) ]
      (логарифми для чисельної стабільності)
"""

import numpy as np
from collections import Counter


class NaiveBayesClassifier:
    """
    Параметри
    ---------
    var_smoothing : float
        Додається до дисперсії кожної ознаки, щоб уникнути ділення на 0
        (аналог epsilon у sklearn GaussianNB).
    """

    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing

        # Заповнюються під час fit()
        self.classes_: np.ndarray = np.array([])
        self.log_priors_: dict    = {}   # log P(c)
        self.means_: dict         = {}   # μ[c] — shape (n_features,)
        self.vars_: dict          = {}   # σ²[c] — shape (n_features,)

    # ──────────────────────────────────────────
    # fit
    # ──────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NaiveBayesClassifier":
        """
        Обчислює апріорні ймовірності класів та параметри Гауссіани
        (μ, σ²) для кожної ознаки в кожному класі.
        """
        n_samples = X.shape[0]
        self.classes_ = np.unique(y)
        counts = Counter(y)

        # Глобальна дисперсія — для var_smoothing (як у sklearn)
        global_var = X.var(axis=0)

        for c in self.classes_:
            mask = y == c
            X_c  = X[mask]

            self.log_priors_[c] = np.log(counts[c] / n_samples)
            self.means_[c]      = X_c.mean(axis=0)
            self.vars_[c]       = (X_c.var(axis=0)
                                   + self.var_smoothing * global_var)

        print(f"[NaiveBayes] Навчено на {n_samples} прикладах  |  "
              f"класи: {self.classes_.tolist()}  |  "
              f"var_smoothing={self.var_smoothing}")
        return self

    # ──────────────────────────────────────────
    # Логарифм правдоподібності для одного класу
    # ──────────────────────────────────────────

    def _log_likelihood(self, X: np.ndarray, c) -> np.ndarray:
        """
        Повертає вектор (n_samples,) — сума log P(x_j | c) по всіх j.
        """
        mu  = self.means_[c]
        var = self.vars_[c]

        # log Gaussian PDF: -0.5·log(2π·σ²) - (x-μ)²/(2σ²)
        log_pdf = (
            -0.5 * np.log(2 * np.pi * var)
            - 0.5 * ((X - mu) ** 2) / var
        )
        return log_pdf.sum(axis=1)

    # ──────────────────────────────────────────
    # predict_log_proba
    # ──────────────────────────────────────────

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Повертає матрицю (n_samples, n_classes) з
        ненормованими логарифмічними апостеріорними ймовірностями.
        """
        log_posteriors = np.column_stack([
            self.log_priors_[c] + self._log_likelihood(X, c)
            for c in self.classes_
        ])
        return log_posteriors

    # ──────────────────────────────────────────
    # predict_proba
    # ──────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Повертає нормалізовані ймовірності через softmax у лог-просторі
        (для чисельної стабільності).
        """
        log_p = self.predict_log_proba(X)
        # log-sum-exp trick
        log_p -= log_p.max(axis=1, keepdims=True)
        p = np.exp(log_p)
        return p / p.sum(axis=1, keepdims=True)

    # ──────────────────────────────────────────
    # predict
    # ──────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        log_posteriors = self.predict_log_proba(X)
        best_class_idx = np.argmax(log_posteriors, axis=1)
        return self.classes_[best_class_idx]
