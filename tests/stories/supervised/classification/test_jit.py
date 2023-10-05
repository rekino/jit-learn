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

    def test_supervised_base_classifier(self):
        learner = jit.BaseClassifier()

        learner.fit(self.X_train, self.y_train)

        y_pred = learner.predict(self.X_train)

        self.assertEquals(y_pred.shape, self.y_train.shape)
