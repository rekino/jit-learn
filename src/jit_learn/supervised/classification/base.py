import numpy as np
import cvxpy as cp
from numpy.typing import NDArray
from typing import Sequence


class BaseClassifier(object):
    def __init__(self, xdim: int, memory: int) -> None:
        self.xdim = xdim
        self.memory = memory
        w = self.weights = cp.Variable(xdim)
        e = self.errors = cp.Variable(memory)
        p = self.preds = cp.Variable(memory)
        b = self.bias = cp.Variable()

        X = self.features = cp.Parameter((memory, xdim))
        y = self.labels = cp.Parameter(memory)
        C = self.param = cp.Parameter()

        constraints = [
            cp.multiply(y, p) >= 1 - e,
            p == X @ w + b,
            e >= 0
            ]

        objective = cp.Minimize(cp.sum_squares(w) / 2 + C * cp.sum(e))

        self.problem = cp.Problem(objective, constraints)

    def _sample(self, x: NDArray | None = None) -> NDArray:
        if x is None:
            idx = np.arange(self.X.shape[0])
            np.random.shuffle(idx)
            idx = idx[:self.memory]
            return self.X[idx], self.y[idx]

        raise NotImplementedError("conditional sampling is not implemented.")

    def _transform(self, x: NDArray, X: NDArray) -> NDArray:
        return X - x

    def _embed(self, X: NDArray) -> NDArray:
        return X

    def _train(self, X: NDArray, y: NDArray) -> Sequence:
        self.features.value = X
        self.labels.value = y
        self.param.value = 1

        try:
            self.problem.solve()
        except Exception as e:
            print(X)
            print(y)
            raise e

        w = self.weights.value
        e = self.errors.value

        is_positive = y > 0
        is_negative = y < 0

        Xp_size = np.sum(is_positive)
        Xn_size = np.sum(is_negative)

        Xp, Xn = X[y > 0], X[y < 0]
        ep, en = e[y > 0], e[y < 0]

        lower_bound = (1 - ep - Xp @ w).max() if Xp_size > 0 else -1
        upper_bound = (en - 1 - Xn @ w).min() if Xn_size > 0 else 1

        return np.asarray([lower_bound, upper_bound])

    def fit(self, X: NDArray, y: NDArray) -> None:
        self.X = X
        self.y = y

    def predict(self, x: NDArray) -> Sequence:
        samples, labels = self._sample()
        transformed = self._transform(x, samples)
        features = self._embed(transformed)

        bound = self._train(features, labels)

        return bound
