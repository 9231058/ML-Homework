# In The Name of God
# =======================================
# [] File Name : dbscan.py
#
# [] Creation Date : 31-01-2020
#
# [] Created By : Parham Alvani <parham.alvani@gmail.com>
# =======================================
import numpy as np

class DBSCAN:
    def __init__(self, eps: float = 0.5, min_samples: int = 5, metric = 'euclidean'):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self._labels = []
        self.label = -1


    def dense_region(self, x, X):
        '''
        x: np.ndarray, shape = [n_features]
        X: np.ndarray, shape = [n_examples, n_features]
        Find a dense region for given point. if there isn't any dense region returns None.
        '''
        dinstances = []
        if self.metric == 'euclidean':
            distances = np.linalg.norm(X - x, axis=1)
        retval = distances <= self.eps
        if np.sum(retval) >= self.min_samples:
            return retval
        return None

    def fit(self, X):
        '''
        X: np.ndarray, shape = [n_examples, n_features]
        '''
        self._labels = np.full(X.shape[0], -1)
        for idx, x in enumerate(X):
            if self._labels[idx] > -1:
                continue

            region = self.dense_region(x, X)
            # we have a dense region so let check its neighbors
            if region is not None:
                self.label += 1
                regions = [region]
                # BFS-like loop to check neighbors
                while len(regions) > 0:
                    region = regions[0]
                    regions = regions[1:]
                    for idx, in_region in enumerate(region):
                        # check label to make sure its new
                        if in_region == True and self._labels[idx] == -1:
                            self._labels[idx] = self.label
                            r = self.dense_region(X[idx], X)
                            if r is not None:
                                regions.append(r)


if __name__ == '__main__':
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
        [10, 10]
    ])

    cl = DBSCAN(eps=1, min_samples=2)
    cl.fit(X)
    print(cl._labels)
