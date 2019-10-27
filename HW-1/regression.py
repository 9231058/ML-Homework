# In The Name of God
# =======================================
# [] File Name : regression.py
#
# [] Creation Date : 27-10-2019
#
# [] Created By : Parham Alvani <parham.alvani@gmail.com>
# =======================================
import numpy as np
from typing import List, Dict

def regression(data: Dict[float, float], n: int, alpha: float, steps: int, lmb: float) -> List[int]:
    '''
    single variable regression based on gradient descent
    x, y are arrays of input data
    alpha is a learning rate
    n is a degree of regression
    lmb is a regularization parameter
    steps is a number of steps in the algorithm
    '''
    teta = [1 for _ in range(n)]
    for _ in range(steps):
        # calculate MSE
        mse = 0
        for x in data:
            h = np.polynomial.polynomial.polyval(x, teta)
            mse += (1/ (2 * len(data))) * ((h - data[x]) ** 2)
        print(mse)

        # gradient descent
        new_teta = [0 for _ in range(n)]
        for j in range(n):
            new_teta[j] = teta[j]
            new_teta[j] -= 2 * alpha * lmb * teta[j]
            for x in data:
                h = np.polynomial.polynomial.polyval(x, teta)
                new_teta[j] -= (1/len(data)) * (alpha * (h - data[x]) * (x ** j))
        teta = new_teta
    return teta


if __name__ == '__main__':
    data = {
        1: 1,
        2: 4,
        3: 9,
        4: 16,
        5: 25,
    }

    h = regression(data=data, n=3, alpha=0.001, steps=100, lmb=2)
    print(h)
    print(np.polynomial.polynomial.polyval(6, h))
