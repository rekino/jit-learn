from numpy.typing import NDArray
from typing import Sequence


class Distribution:
    def sample(self, count: int, **kwargs) -> (NDArray, NDArray):
        raise NotImplementedError()

    @property
    def x_shape(self) -> Sequence[int]:
        raise NotImplementedError()

    @property
    def y_shape(self) -> Sequence[int]:
        raise NotImplementedError()


class Automorphism:
    def transform(self, x: NDArray, X: NDArray) -> NDArray:
        raise NotImplementedError()

    @property
    def shape(self) -> Sequence[int]:
        raise NotImplementedError()


class Embeding:
    def embed(self, X: NDArray) -> NDArray:
        raise NotImplementedError()

    @property
    def in_shape(self) -> Sequence[int]:
        raise NotImplementedError()

    @property
    def out_shape(self) -> Sequence[int]:
        raise NotImplementedError()


class LearningRule:
    dist_x_shape: Sequence[int]
    dist_y_shape: Sequence[int]
    auto_shape: Sequence[int]
    embeding_in_shape: Sequence[int]
    embeding_out_shape: Sequence[int]

    def __init__(self) -> None:
        self.dist_x_shape = super().x_shape()
        self.dist_y_shape = super().y_shape()
        self.auto_shape = super().shape()
        self.embeding_in_shape = super().in_shape()
        self.embeding_out_shape = super().out_shape()

        if not self.dist_x_shape == self.auto_shape:
            raise Exception("distribution and automorphism are not compatible")

        if not self.embeding_in_shape == self.auto_shape:
            raise Exception("embedinng and automorphism are not compatible")
