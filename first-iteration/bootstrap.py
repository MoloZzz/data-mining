"""
bootstrap.py
------------
Запуск усіх чотирьох класифікаторів на датасеті bank.csv.
Збирає метрики, виводить порівняльну таблицю, досліджує гіперпараметри.

Структура проєкту:
    dataset/bank.csv
    preprocessing.py
    classifiers/
        one_rule.py       ← власна реалізація
        naive_bayes.py    ← власна реалізація
        knn.py            ← власна реалізація
        decision_tree.py  ← sklearn DecisionTreeClassifier
    bootstrap.py          ← цей файл
"""

import sys
import time
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── Власні модулі ──────────────────────────────────────────────────────────────
from preprocessing import prepare, classification_report, accuracy

# ── Класифікатори ──────────────────────────────────────────────────────────────
from classifiers.one_rule    import OneRuleClassifier
from classifiers.naive_bayes import NaiveBayesClassifier
from classifiers.knn         import KNNClassifier

# Decision Tree — бібліотечна реалізація (sklearn)
try:
    from sklearn.tree import DecisionTreeClassifier
    SKLEARN_OK = True
except ImportError:
    print("[bootstrap] sklearn не встановлено. Decision Tree буде пропущено.")
    SKLEARN_OK = False


# ══════════════════════════════════════════════════════════════════════════════
# Допоміжна функція запуску одного класифікатора
# ══════════════════════════════════════════════════════════════════════════════

def run_classifier(name: str, clf, X_train, y_train, X_test, y_test) -> dict:
    """Навчає, передбачає, рахує метрики та час."""
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_pred = clf.predict(X_test)
    pred_time = time.perf_counter() - t1

    metrics = classification_report(y_test, y_pred, label=name)
    metrics["train_time_s"] = round(train_time, 4)
    metrics["pred_time_s"]  = round(pred_time,  4)
    metrics["name"] = name
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# Аналіз гіперпараметрів
# ══════════════════════════════════════════════════════════════════════════════

def hyperparam_knn(X_train, y_train, X_test, y_test,
                   k_values=(1, 3, 5, 7, 9, 11, 15)) -> None:
    print("\n\n" + "█"*55)
    print("  Аналіз гіперпараметра kNN: k")
    print("█"*55)
    results = []
    for k in k_values:
        clf = KNNClassifier(k=k)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy(y_test, y_pred)
        results.append((k, acc))
        print(f"  k={k:>3}  →  accuracy={acc:.4f}")
    best_k, best_acc = max(results, key=lambda x: x[1])
    print(f"\n  ✓ Найкращий k={best_k}  (accuracy={best_acc:.4f})")


def hyperparam_dt(X_train, y_train, X_test, y_test) -> None:
    if not SKLEARN_OK:
        return
    print("\n\n" + "█"*55)
    print("  Аналіз гіперпараметрів Decision Tree")
    print("█"*55)

    # max_depth
    print("\n  max_depth:")
    for depth in (None, 3, 5, 7, 10, 15, 20):
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        clf.fit(X_train, y_train)
        acc = accuracy(y_test, clf.predict(X_test))
        depth_str = str(depth) if depth else "None"
        print(f"  max_depth={depth_str:>4}  →  accuracy={acc:.4f}")

    # criterion
    print("\n  criterion:")
    for criterion in ("gini", "entropy", "log_loss"):
        clf = DecisionTreeClassifier(criterion=criterion, max_depth=7, random_state=42)
        clf.fit(X_train, y_train)
        acc = accuracy(y_test, clf.predict(X_test))
        print(f"  criterion={criterion:<10}  →  accuracy={acc:.4f}")

    # min_samples_split
    print("\n  min_samples_split:")
    for mss in (2, 5, 10, 20, 50):
        clf = DecisionTreeClassifier(min_samples_split=mss, max_depth=7, random_state=42)
        clf.fit(X_train, y_train)
        acc = accuracy(y_test, clf.predict(X_test))
        print(f"  min_samples_split={mss:>3}  →  accuracy={acc:.4f}")


def hyperparam_nb(X_train, y_train, X_test, y_test) -> None:
    print("\n\n" + "█"*55)
    print("  Аналіз гіперпараметра Naive Bayes: var_smoothing")
    print("█"*55)
    for alpha in (1e-9, 1e-6, 1e-3, 1e-1, 1.0):
        clf = NaiveBayesClassifier(var_smoothing=alpha)
        clf.fit(X_train, y_train)
        acc = accuracy(y_test, clf.predict(X_test))
        print(f"  var_smoothing={alpha:.1e}  →  accuracy={acc:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# Порівняльна таблиця
# ══════════════════════════════════════════════════════════════════════════════

def print_comparison(results: list[dict]) -> None:
    print("\n\n" + "═"*75)
    print("  ПОРІВНЯЛЬНА ТАБЛИЦЯ КЛАСИФІКАТОРІВ")
    print("═"*75)
    header = f"{'Класифікатор':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Train(s)':>9}"
    print(header)
    print("─"*75)
    for r in results:
        row = (f"{r['name']:<22} "
               f"{r['accuracy']:>9.4f} "
               f"{r['precision']:>10.4f} "
               f"{r['recall']:>8.4f} "
               f"{r['f1']:>8.4f} "
               f"{r['train_time_s']:>9.4f}")
        print(row)
    print("═"*75)

    best = max(results, key=lambda r: r["f1"])
    print(f"\n  ✓ Найкращий за F1: {best['name']}  (F1={best['f1']:.4f})")


# ══════════════════════════════════════════════════════════════════════════════
# Висновки
# ══════════════════════════════════════════════════════════════════════════════

CONCLUSIONS = """
╔══════════════════════════════════════════════════════════════════════════╗
║                         ВИСНОВКИ                                        ║
╠══════════════════════════════════════════════════════════════════════════╣
║  1-Rule                                                                  ║
║  • Найпростіший. Будує єдине правило на найкращому атрибуті.             ║
║  • Добре пояснюється людині, але точність обмежена.                      ║
║  • Підходить як базова лінія (baseline).                                 ║
║                                                                          ║
║  Naive Bayes                                                             ║
║  • Швидкий, добре працює на числових/неперервних ознаках (Gaussian NB). ║
║  • Припущення про незалежність ознак часто порушується, але на           ║
║    практиці модель стійка та малочутлива до шуму.                        ║
║  • Var_smoothing суттєво впливає при малих вибірках.                     ║
║                                                                          ║
║  Decision Tree (sklearn)                                                 ║
║  • Добре інтерпретується, знаходить нелінійні межі.                     ║
║  • Схильний до перенавчання при великій глибині (max_depth).            ║
║  • Оптимальна глибина ~5–7: баланс між точністю та узагальненням.        ║
║  • criterion='entropy' зазвичай трохи кращий за 'gini' на цьому датасеті║
║                                                                          ║
║  kNN                                                                     ║
║  • Не навчається явно — зберігає всі приклади (ліниве навчання).        ║
║  • Чутливий до масштабу ознак → обов'язкова нормалізація.               ║
║  • k=5–7 зазвичай оптимальне; при k=1 — перенавчання.                  ║
║  • Повільний при predict на великих даних.                               ║
╚══════════════════════════════════════════════════════════════════════════╝
"""


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "▓"*60)
    print("  КЛАСИФІКАЦІЯ: Bank Marketing Dataset")
    print("▓"*60)

    # ── 1. Підготовка даних ────────────────────────────────────────────────
    # kNN потребує нормалізації, інші — ні.
    # Готуємо два набори: звичайний і нормалізований.
    print("\n── Завантаження та обробка даних ──")
    data     = prepare(path="dataset/bank.csv", normalize_data=False)
    data_n   = prepare(path="dataset/bank.csv", normalize_data=True)

    X_train,   y_train   = data["X_train"],   data["y_train"]
    X_test,    y_test    = data["X_test"],     data["y_test"]
    X_train_n, y_train_n = data_n["X_train"], data_n["y_train"]
    X_test_n,  y_test_n  = data_n["X_test"],  data_n["y_test"]

    results = []

    # ── 2. 1-Rule ──────────────────────────────────────────────────────────
    print("\n\n── 1-Rule ──")
    clf_1r = OneRuleClassifier()
    res = run_classifier("1-Rule", clf_1r, X_train, y_train, X_test, y_test)
    results.append(res)
    print(f"  Найкраща ознака: #{clf_1r.best_feature_idx}  "
          f"({data['feature_names'][clf_1r.best_feature_idx]})")

    # ── 3. Naive Bayes ─────────────────────────────────────────────────────
    print("\n\n── Naive Bayes ──")
    clf_nb = NaiveBayesClassifier(var_smoothing=1e-9)
    res = run_classifier("Naive Bayes", clf_nb, X_train, y_train, X_test, y_test)
    results.append(res)

    # ── 4. kNN ─────────────────────────────────────────────────────────────
    print("\n\n── kNN (k=5, нормалізовані дані) ──")
    clf_knn = KNNClassifier(k=5)
    res = run_classifier("kNN (k=5)", clf_knn,
                         X_train_n, y_train_n, X_test_n, y_test_n)
    results.append(res)

    # ── 5. Decision Tree (sklearn) ─────────────────────────────────────────
    if SKLEARN_OK:
        print("\n\n── Decision Tree (sklearn) ──")
        clf_dt = DecisionTreeClassifier(max_depth=7, criterion="entropy",
                                        random_state=42)
        res = run_classifier("Decision Tree", clf_dt,
                             X_train, y_train, X_test, y_test)
        results.append(res)

        # Важливість ознак
        feat_imp = sorted(
            zip(data["feature_names"], clf_dt.feature_importances_),
            key=lambda x: x[1], reverse=True
        )
        print("\n  Важливість ознак (Decision Tree):")
        for fname, imp in feat_imp[:10]:
            bar = "█" * int(imp * 40)
            print(f"  {fname:<30} {imp:.4f}  {bar}")

    # ── 6. Порівняльна таблиця ─────────────────────────────────────────────
    print_comparison(results)

    # ── 7. Аналіз гіперпараметрів ──────────────────────────────────────────
    hyperparam_knn(X_train_n, y_train_n, X_test_n, y_test_n)
    hyperparam_dt(X_train,   y_train,   X_test,   y_test)
    hyperparam_nb(X_train,   y_train,   X_test,   y_test)

    # ── 8. Висновки ────────────────────────────────────────────────────────
    print(CONCLUSIONS)


if __name__ == "__main__":
    main()