import numpy as np
import cvxpy as cp
from numpy.typing import NDArray

from .base import Model, Distribution, Automorphism, Embeding


class LinearSVC(Model):
    def __call__(self, x: NDArray) -> NDArray:
        if self.problem is None:
            raise Exception("the model has been called before it is trained")

        samples, labels = super(Distribution, self).sample(self.sample_size)
        transformed = super(Automorphism, self).transform(x, samples)
        features = super(Embeding, self).embed(transformed)

        self.features.value = features
        self.labels.value = labels
        self.param.value = 1

        self.problem.solve()

        w = self.weights.value
        e = self.errors.value

        is_positive = labels > 0
        is_negative = labels < 0

        Xp_size = np.sum(is_positive)
        Xn_size = np.sum(is_negative)

        Xp, Xn = features[is_positive], features[is_negative]
        ep, en = e[is_positive], e[is_negative]

        lower_bound = (1 - ep - Xp @ w).max() if Xp_size > 0 else 0
        upper_bound = (en - 1 - Xn @ w).min() if Xn_size > 0 else 0

        return np.asarray([lower_bound, upper_bound])

    def train(self, X: NDArray, y: NDArray, sample_size: int) -> None:
        super(Distribution, self).fit(data=(X, y))
        features = super(Embeding, self).embed(X[0])

        if len(features) > 2:
            raise Exception("Only flat embedings are supported")

        embeding_dim = features[-1]
        self.sample_size = sample_size

        w = self.weights = cp.Variable(embeding_dim)
        e = self.errors = cp.Variable(sample_size)
        p = self.preds = cp.Variable(sample_size)
        b = self.bias = cp.Variable()

        X = self.features = cp.Parameter((sample_size, embeding_dim))
        y = self.labels = cp.Parameter(sample_size)
        C = self.param = cp.Parameter()

        constraints = [cp.multiply(y, p) >= 1 - e, p == X @ w + b, e >= 0]

        objective = cp.Minimize(cp.sum_squares(w) + C * cp.sum(e))

        self.problem = cp.Problem(objective, constraints)

    @property
    def is_ready(self):
        return self.problem is not None
