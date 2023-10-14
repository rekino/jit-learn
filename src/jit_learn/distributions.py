import numpy as np
from numpy.typing import NDArray
from typing import Sequence

from .base import Distribution


class EmpiricalDistribution(Distribution):
    def __init__(self, X: NDArray, y: NDArray | None = None) -> None:
        self.X = X
        self.y = y

    def sample(self, count: int, **kwargs) -> NDArray:
        size = self.X.shape[0]
        idx = np.arange(size)
        np.random.shuffle(idx)

        if self.y is None:
            return self.X[idx]

        return self.X[idx], self.y[idx]

    @property
    def x_shape(self) -> Sequence[int]:
        return self.X.shape[1:]

    @property
    def y_shape(self) -> Sequence[int]:
        if self.y is None:
            return None

        return self.y.shape[1:]
