import numpy as np


class KMeans:
    def __init__(self, k: int, n_iter: int = 50, random_state: int = 1):
        self.k = k
        self.random_state = random_state
        self.n_iter = n_iter
        self._centroids = []
        self._labels = []
        self._Si = []
        self.dbi = []

    def Si(self, X, centroid):
        '''
        Calculate the Davies–Bouldin index (S_i) for given cluster
        X: np.ndarray, shape = [n_examples, n_features]
        '''
        return np.mean(np.linalg.norm(X - centroid, axis=1))

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
            self._Si = np.empty(self.k)
            self._Rij = np.empty([self.k, self.k])

            for idx, x in enumerate(X):
                self._labels[idx] = self.distance(x)

            for cl in range(self.k):
                self._centroids[cl] = self.centroid(X[self._labels == cl])
                self._Si[cl] = self.Si(X[self._labels == cl], self._centroids[cl])

            for cl1 in range(self.k):
                for cl2 in range(self.k):
                    if cl1 != cl2:
                        m = np.linalg.norm(self._centroids[cl1] - self._centroids[cl2])
                        self._Rij[cl1][cl2] = (self._Si[cl1] + self._Si[cl2]) / m

            self.dbi.append(np.mean(np.max(self._Rij, axis = 0)))
