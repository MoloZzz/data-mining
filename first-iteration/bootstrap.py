from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

from one_rule import OneRule
from kNN import KNN
from naive_bayes import NaiveBayes

def load_data(path):
    df = pd.read_csv(path, sep=";")
    return df

def preprocess(df):
    y = df["y"]
    X = df.drop("y", axis=1)

    X = X.replace("unknown", np.nan)

    for col in X.columns:
        if X[col].dtype == "object":
            X[col].fillna(X[col].mode()[0], inplace=True)
        else:
            X[col].fillna(X[col].mean(), inplace=True)

    X_encoded = pd.get_dummies(X)

    return X, X_encoded, y


def main():
    df = load_data("dataset/bank.csv")

    X_raw, X_encoded, y = preprocess(df)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42
    )

    X_train_enc, X_test_enc, _, _ = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_enc)
    X_test_scaled = scaler.transform(X_test_enc)

    print("=== RESULTS ===")

    one_rule = OneRule()
    one_rule.fit(X_train_raw, y_train)
    preds = one_rule.predict(X_test_raw)

    print("1-Rule Accuracy:", accuracy_score(y_test, preds))

    knn = KNN(k=5)
    knn.fit(pd.DataFrame(X_train_scaled), y_train.reset_index(drop=True))
    preds = knn.predict(pd.DataFrame(X_test_scaled))

    print("kNN Accuracy:", accuracy_score(y_test, preds))

    nb = NaiveBayes()
    nb.fit(X_train_scaled, y_train.values)
    preds = nb.predict(X_test_scaled)

    print("Naive Bayes Accuracy:", accuracy_score(y_test, preds))


if __name__ == "__main__":
    main()