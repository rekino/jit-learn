import unittest
import numpy.typing as ntp

import numpy as np

import src.jit_learn.supervised.classification.base as jit


class TestInterface(unittest.TestCase):
    X_train: ntp.NDArray
    y_train: ntp.NDArray

    x_test: float

    def setUp(self) -> None:
        self.X_train = np.linspace(-1, 1, 4)[:, None]
        self.y_train = np.sign(self.X_train).squeeze()

        self.x_test = 0

    def test_base(self) -> None:
        learner = jit.BaseClassifier(1, 4)

        learner.fit(self.X_train, self.y_train)

        y_interval = learner.predict(self.x_test)

        self.assertEqual(len(y_interval), 2)
        self.assertAlmostEquals(y_interval, (0, 0))
