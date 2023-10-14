from numpy.typing import NDArray


class Distribution:
    def fit(self, data: NDArray, **kwargs) -> None:
        raise NotImplementedError()

    def sample(self, size: int, **kwargs) -> NDArray:
        raise NotImplementedError()


class Automorphism:
    def transform(self, x: NDArray, X: NDArray, **kwargs) -> NDArray:
        raise NotImplementedError()


class Embeding:
    def embed(self, X: NDArray, **kwargs) -> NDArray:
        raise NotImplementedError()


class Model:
    def train(self, X: NDArray, y: NDArray, **kwargs) -> None:
        raise NotImplementedError()

    def predict(self, x: NDArray, **kwargs) -> NDArray:
        raise NotImplementedError()
