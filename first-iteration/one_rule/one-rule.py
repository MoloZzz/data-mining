import pandas as pd
from collections import Counter

class OneRule:
    def __init__(self):
        self.best_feature = None
        self.rules = {}

    def fit(self, X, y):
        min_error = float('inf')

        for feature in X.columns:
            feature_rules = {}
            error = 0

            for value in X[feature].unique():
                subset_y = y[X[feature] == value]
                most_common = Counter(subset_y).most_common(1)[0][0]
                feature_rules[value] = most_common

                # рахуємо помилки
                error += sum(subset_y != most_common)

            if error < min_error:
                min_error = error
                self.best_feature = feature
                self.rules = feature_rules

    def predict(self, X):
        predictions = []
        for _, row in X.iterrows():
            value = row[self.best_feature]
            prediction = self.rules.get(value, None)
            predictions.append(prediction)
        return predictions