import numpy as np
import cvxpy as cp
from numpy.typing import NDArray
from typing import Sequence

from .base import Model


class LinearSVC(Model):
    def __init__(self, sample_size: int) -> None:
        super(Model, self).__init__()

        self.sample_size = sample_size
        embeding_dim = self.embeding_out_shape

        if len(embeding_dim) > 1:
            raise Exception("Only flat embedings are supported")

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

    def predict(self, x: NDArray) -> Sequence:
        samples, labels = super().sample(self.sample_size)
        transformed = super().transform(x, samples)
        features = super().embed(transformed)

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
