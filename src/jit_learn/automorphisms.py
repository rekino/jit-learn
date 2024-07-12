from numpy.typing import NDArray

from .base import Automorphism


class ShiftAutomorphism(Automorphism):
    def transform(self, x: NDArray, X: NDArray) -> NDArray:
        return X - x
