from numpy.typing import NDArray

from .base import Embeding


class FlatEmbeding(Embeding):
    def embed(self, X: NDArray) -> NDArray:
        return X.reshape(X.shape[0], -1)
