"""
preprocessing.py
----------------
Завантаження та попередня обробка датасету bank.csv.
Датасет: Bank Marketing (UCI / Kaggle bankbalanced)
Цільова змінна: 'y' (yes/no → 1/0) — чи оформив клієнт строковий депозит.
"""

import pandas as pd
import numpy as np
from collections import Counter


# ──────────────────────────────────────────────
# 1. Завантаження
# ──────────────────────────────────────────────

def load_data(path: str = "dataset/bank.csv") -> pd.DataFrame:
    """Зчитує CSV, автоматично визначає роздільник (';' або ',')."""
    # Спробуємо крапку з комою (стандарт UCI), потім кому
    for sep in (";", ","):
        try:
            df = pd.read_csv(path, sep=sep)
            if df.shape[1] > 2:          # валідний датафрейм
                print(f"[load] Завантажено {df.shape[0]} рядків, {df.shape[1]} стовпців (sep='{sep}')")
                return df
        except Exception:
            pass
    raise ValueError(f"Не вдалося зчитати файл: {path}")


# ──────────────────────────────────────────────
# 2. Базове дослідження
# ──────────────────────────────────────────────

def explore(df: pd.DataFrame) -> None:
    """Виводить зведену інформацію про датасет."""
    print("\n=== ДОСЛІДЖЕННЯ ДАТАСЕТУ ===")
    print(f"Розмір: {df.shape}")
    print(f"\nТипи стовпців:\n{df.dtypes}")
    print(f"\nПропущені значення:\n{df.isnull().sum()}")
    print(f"\nЦільова змінна 'deposit':\n{df['deposit'].value_counts()}")
    print(f"\nБаланс класів: {dict(Counter(df['deposit']))}")

    print("\nЧислові ознаки — статистика:")
    print(df.select_dtypes(include=np.number).describe().round(2))

    print("\nКатегоріальні ознаки — унікальні значення:")
    for col in df.select_dtypes(include="object").columns:
        if col != "deposit":
            print(f"  {col}: {sorted(df[col].unique())}")


# ──────────────────────────────────────────────
# 3. Очищення
# ──────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Замінює 'unknown' на NaN, потім заповнює модою для категоріальних
      і медіаною для числових.
    - Видаляє дублікати.
    """
    df = df.copy()

    # 'unknown' → NaN
    df.replace("unknown", np.nan, inplace=True)

    # Заповнення пропусків
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype == object:
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
                print(f"[clean] '{col}': NaN → мода '{mode_val}'")
            else:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"[clean] '{col}': NaN → медіана {median_val}")

    # Дублікати
    before = len(df)
    df.drop_duplicates(inplace=True)
    removed = before - len(df)
    if removed:
        print(f"[clean] Видалено {removed} дублікатів")

    return df


# ──────────────────────────────────────────────
# 4. Кодування категоріальних ознак
# ──────────────────────────────────────────────

BINARY_COLS = ["default", "housing", "loan"]   # yes/no → 1/0
ORDINAL_EDUCATION = {
    "primary": 0, "secondary": 1, "tertiary": 2
}
ORDINAL_MONTH = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,  "may": 5,  "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """
    Повертає DataFrame з лише числовими стовпцями.
    Цільова змінна 'deposit': yes→1, no→0.
    """
    df = df.copy()

    # Цільова змінна
    df["deposit"] = df["deposit"].map({"yes": 1, "no": 0}).astype(int)

    # Бінарні yes/no → 1/0
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].map({"yes": 1, "no": 0})

    # Освіта — порядкова шкала
    if "education" in df.columns:
        df["education"] = df["education"].map(ORDINAL_EDUCATION)

    # Місяць → число
    if "month" in df.columns:
        df["month"] = df["month"].map(ORDINAL_MONTH)

    # Решта категоріальних → one-hot (contact, job, marital, poutcome)
    cat_cols = [c for c in df.select_dtypes(include="object").columns if c != "y"]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        print(f"[encode] One-hot: {cat_cols}")

    return df


# ──────────────────────────────────────────────
# 5. Відбір ознак (кореляція + дисперсія)
# ──────────────────────────────────────────────

def select_features(df: pd.DataFrame,
                    corr_threshold: float = 0.03,
                    top_n: int = 15) -> tuple[pd.DataFrame, list[str]]:
    """
    Повертає df із відібраними ознаками та список їхніх назв.
    Критерій: |кореляція з 'deposit'| >= corr_threshold, беремо top_n.
    """
    corr = df.drop(columns=["deposit"]).corrwith(df["deposit"]).abs().sort_values(ascending=False)

    print(f"\n=== КОРЕЛЯЦІЯ ОЗНАК З ЦІЛЬОВОЮ ЗМІННОЮ ===")
    print(corr.round(4).to_string())

    selected = corr[corr >= corr_threshold].head(top_n).index.tolist()
    print(f"\n[features] Обрано {len(selected)} ознак: {selected}")

    return df[selected + ["deposit"]], selected


# ──────────────────────────────────────────────
# 6. Нормалізація (min-max) — для kNN
# ──────────────────────────────────────────────

def normalize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Min-max нормалізація. Повертає (X_norm, min_vals, max_vals)."""
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)
    denom = max_vals - min_vals
    denom[denom == 0] = 1          # уникаємо ділення на 0
    X_norm = (X - min_vals) / denom
    return X_norm, min_vals, max_vals


# ──────────────────────────────────────────────
# 7. Train/test split (без sklearn)
# ──────────────────────────────────────────────

def train_test_split(X: np.ndarray, y: np.ndarray,
                     test_size: float = 0.2,
                     random_state: int = 42
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Перемішує та ділить дані на train/test."""
    rng = np.random.default_rng(random_state)
    idx = rng.permutation(len(y))
    split = int(len(y) * (1 - test_size))
    train_idx, test_idx = idx[:split], idx[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# ──────────────────────────────────────────────
# 8. Метрики (без sklearn)
# ──────────────────────────────────────────────

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray
                     ) -> tuple[int, int, int, int]:
    """Повертає (TP, FP, FN, TN)."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    return tp, fp, fn, tn


def classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                           label: str = "") -> dict:
    """Виводить та повертає словник метрик."""
    tp, fp, fn, tn = confusion_matrix(y_true, y_pred)
    acc  = (tp + tn) / (tp + fp + fn + tn + 1e-9)
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)

    header = f"  [{label}]" if label else ""
    print(f"\n{'='*45}")
    print(f"  Класифікатор:{header}")
    print(f"{'='*45}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-score : {f1:.4f}")
    print(f"  Матриця помилок:")
    print(f"           Pred 0   Pred 1")
    print(f"  True 0:  {tn:>6}   {fp:>6}")
    print(f"  True 1:  {fn:>6}   {tp:>6}")
    print(f"{'='*45}")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# ──────────────────────────────────────────────
# 9. Головна функція — повний пайплайн
# ──────────────────────────────────────────────

def prepare(path: str = "dataset/bank.csv",
            test_size: float = 0.2,
            normalize_data: bool = False,
            top_n_features: int = 15,
            random_state: int = 42
            ) -> dict:
    """
    Повний пайплайн: load → explore → clean → encode → select → split → (normalize).

    Повертає словник:
        X_train, X_test, y_train, y_test  — numpy arrays
        feature_names                      — list[str]
        df_encoded                         — pd.DataFrame після кодування
    """
    df = load_data(path)
    explore(df)
    df = clean(df)
    df = encode(df)
    df, feature_names = select_features(df, top_n=top_n_features)

    X = df.drop(columns=["deposit"]).values.astype(float)
    y = df["deposit"].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if normalize_data:
        X_train, min_v, max_v = normalize(X_train)
        denom = max_v - min_v
        denom[denom == 0] = 1
        X_test = (X_test - min_v) / denom

    print(f"\n[prepare] Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"[prepare] Баланс train: {Counter(y_train)}")
    print(f"[prepare] Баланс test : {Counter(y_test)}")

    return {
        "X_train": X_train,
        "X_test":  X_test,
        "y_train": y_train,
        "y_test":  y_test,
        "feature_names": feature_names,
        "df_encoded": df,
    }


if __name__ == "__main__":
    data = prepare()