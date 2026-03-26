from collections import Counter

import numpy as np

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X.values
        self.y_train = y.values

    def euclidean_distance(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    def predict(self, X):
        predictions = []

        for x in X.values:
            distances = []

            for i, x_train in enumerate(self.X_train):
                dist = self.euclidean_distance(x, x_train)
                distances.append((dist, self.y_train[i]))

            distances.sort(key=lambda x: x[0])

            neighbors = [label for _, label in distances[:self.k]]

            most_common = Counter(neighbors).most_common(1)[0][0]
            predictions.append(most_common)

        return predictions