# In The Name of God
# =======================================
# [] File Name : knn.py
#
# [] Creation Date : 23-01-2020
#
# [] Created By : Parham Alvani <parham.alvani@gmail.com>
# =======================================
import numpy as np

class KNN:
    def __init__(self, k, distance='euclidean'):
        self.k = k
        self.distance = distance
        self.X = []
        self.y = []

    def fit(self, X, y):
        '''
        X: np.ndarray, shape = [n_examples, n_features]
        y: np.ndarray, shape = [n_examples]
        '''
        self.X = X
        self.y = y
        return self

    def predict(self, X):
        r = []
        for x in X:
            distances = []
            if self.distance == 'euclidean':
                distances = np.linalg.norm(self.X - x, axis=1)
            elif self.distance == 'manhattan':
                distances = np.sum(np.abs(self.X - x), axis=1)
            elif self.distance == 'chebyshev':
                distances = np.max(np.abs(self.X - x), axis=1)
            idx = np.argpartition(distances, self.k - 1) # creates a partition that has k-smallest elements' indeces.
            r.append(np.argmax(np.bincount(self.y[idx][:self.k])))
        return np.array(r)
