"""
classifiers/one_rule.py
-----------------------
Реалізація алгоритму 1-Rule (One Rule / 1R).

Алгоритм:
  Для кожної ознаки будуємо одне правило: розбиваємо значення на бакети
  (bins), для кожного бакету визначаємо клас більшості, рахуємо кількість
  помилок. Обираємо ознаку з найменшою кількістю помилок.

  Для неперервних ознак — дискретизація на рівні інтервали (n_bins).
"""

import numpy as np
from collections import Counter


class OneRuleClassifier:
    """
    Параметри
    ---------
    n_bins : int
        Кількість інтервалів для дискретизації неперервних ознак.
    """

    def __init__(self, n_bins: int = 5):
        self.n_bins = n_bins
        self.best_feature_idx: int = -1
        self.rules: dict = {}          # bin_id → predicted_class
        self.bin_edges: np.ndarray | None = None
        self.default_class: int = 0    # клас у разі невідомого бакету

    # ──────────────────────────────────────────
    # Дискретизація однієї ознаки
    # ──────────────────────────────────────────

    def _discretize(self, values: np.ndarray,
                    edges: np.ndarray) -> np.ndarray:
        """Перетворює неперервні значення на індекси бакетів."""
        return np.digitize(values, edges[1:-1])  # індекси 0..n_bins-1

    def _compute_edges(self, col: np.ndarray) -> np.ndarray:
        """Рівномірні межі бакетів для одного стовпця."""
        return np.linspace(col.min(), col.max(), self.n_bins + 1)

    # ──────────────────────────────────────────
    # Побудова правила для однієї ознаки
    # ──────────────────────────────────────────

    def _build_rule(self, bins: np.ndarray,
                    y: np.ndarray) -> tuple[dict, int]:
        """
        Для кожного унікального значення бакету → клас більшості.
        Повертає (rules_dict, n_errors).
        """
        rules = {}
        errors = 0
        for bin_id in np.unique(bins):
            mask = bins == bin_id
            majority = Counter(y[mask]).most_common(1)[0][0]
            rules[bin_id] = majority
            errors += int(np.sum(y[mask] != majority))
        return rules, errors

    # ──────────────────────────────────────────
    # fit
    # ──────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OneRuleClassifier":
        """
        Навчання: перебираємо всі ознаки, обираємо ту,
        що дає найменшу кількість помилок.
        """
        n_features = X.shape[1]
        best_errors = np.inf
        best_idx    = 0
        best_rules  = {}
        best_edges  = None

        for j in range(n_features):
            col   = X[:, j]
            edges = self._compute_edges(col)
            bins  = self._discretize(col, edges)
            rules, errors = self._build_rule(bins, y)

            if errors < best_errors:
                best_errors = errors
                best_idx    = j
                best_rules  = rules
                best_edges  = edges

        self.best_feature_idx = best_idx
        self.rules            = best_rules
        self.bin_edges        = best_edges
        self.default_class    = int(Counter(y).most_common(1)[0][0])

        err_rate = best_errors / len(y)
        print(f"[1-Rule] Обрана ознака #{best_idx}  |  "
              f"train error={err_rate:.4f}  |  "
              f"правил: {len(best_rules)}")
        return self

    # ──────────────────────────────────────────
    # predict
    # ──────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        col  = X[:, self.best_feature_idx]
        bins = self._discretize(col, self.bin_edges)
        return np.array([
            self.rules.get(b, self.default_class) for b in bins
        ])
