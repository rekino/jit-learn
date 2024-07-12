from typing import Any
from numpy.typing import NDArray


class Distribution:
    def fit(self, data: NDArray, **kwargs: Any) -> None:
        raise NotImplementedError()

    def sample(self, size: int, **kwargs: Any) -> NDArray:
        raise NotImplementedError()


class Automorphism:
    def transform(self, x: NDArray, X: NDArray, **kwargs: Any) -> NDArray:
        raise NotImplementedError()


class Embeding:
    def embed(self, X: NDArray, **kwargs: Any) -> NDArray:
        raise NotImplementedError()


class Model:
    def train(self, X: NDArray, y: NDArray, **kwargs: Any) -> None:
        raise NotImplementedError()

    def __call__(self, x: NDArray, **kwargs: Any) -> NDArray:
        raise NotImplementedError()

    @property
    def is_ready(self):
        return False
