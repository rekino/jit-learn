import numpy as np
from numpy.typing import NDArray

from .base import Distribution


class EmpiricalDistribution(Distribution):
    def fit(self, data: NDArray) -> None:
        self.data = data

    def sample(self, size: int, **kwargs) -> NDArray:
        count = self.data.shape[0]
        idx = np.arange(count)
        np.random.shuffle(idx)
        idx = idx[:size]

        return self.data[idx]
