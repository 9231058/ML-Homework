import numpy as np


class KMeans:
    def __init__(self, k: int, n_iter: int = 50, random_state: int = 1):
        self.k = k
        self.random_state = random_state
        self.n_iter = n_iter
        self._centroids = []
        self._labels = []

    def centroid(self, X):
        '''
        X: np.ndarray, shape = [n_examples, n_features]
        '''
        return np.mean(X, axis=0)

    def distance(self, x):
        '''
        Return nearest centroid based on the Euclidean distance
        x: np.ndarray, shape = [n_features]
        '''
        return np.linalg.norm(self._centroids - x, axis=1).argmin()

    def fit(self, X):
        '''
        X: np.ndarray, shape = [n_examples, n_features]
        '''
        rgen = np.random.RandomState(self.random_state)
        self._centroids = X[rgen.choice(X.shape[0], self.k, replace=False)]
        for _ in range(self.n_iter):
            self._labels = np.empty(X.shape[0])
            for idx, x in enumerate(X):
                self._labels[idx] = self.distance(x)
            for cl in range(self.k):
                self._centroids[cl] = self.centroid(X[self._labels == cl])
