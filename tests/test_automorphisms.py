import unittest
import numpy as np

from src.jit_learn.automorphisms import ShiftAutomorphism


class Testautomorphisms(unittest.TestCase):
    X = np.random.randn(2, 3)
    x = np.zeros((1, 3))

    def test_shift(self):
        auto = ShiftAutomorphism()

        transformed = auto.transform(self.x, self.X)

        assert np.allclose(transformed, self.X)
