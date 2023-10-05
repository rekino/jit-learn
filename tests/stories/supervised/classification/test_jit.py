import unittest
import numpy.typing as ntp

import numpy as np

import src.jit_learn.supervised.classification.base as jit


class TestInterface(unittest.TestCase):
    X_train: ntp.NDArray
    y_train: ntp.NDArray

    def setUp(self) -> None:
        self.X_train = np.linspace(-1, 1)
        self.y_train = np.sign(self.X_train)

    def test_base(self) -> None:
        learner = jit.BaseClassifier()

        learner.fit(self.X_train, self.y_train)

        y_pred = learner.predict(self.X_train)

        self.assertAlmostEquals(y_pred, self.y_train)
