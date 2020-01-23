# In The Name of God
# =======================================
# [] File Name : gnb.py
#
# [] Creation Date : 24-01-2020
#
# [] Created By : Parham Alvani <parham.alvani@gmail.com>
# =======================================
import numpy as np
import scipy.stats

class GaussianNB:
    def __init__(self):
        self._dists = []
        self._classes = []

    def fit(self, X, y):
        self._dists = []
        self._classes = np.unique(y)

        for cl in self._classes:
            loc = np.mean(X[y == cl], axis=0)
            scale = np.std(X[y == cl], axis=0)
            vnorm = np.vectorize(scipy.stats.norm)
            self._dists.append(vnorm(loc, scale))
        return self

    def predict(self, X):
        r = []
        for x in X:
            p = []
            for dist in self._dists:
                p.append(np.prod([disti.pdf(xi) for xi, disti in zip(x, dist)]))
            r.append(self._classes[np.argmax(p)])
        return np.array(r)
