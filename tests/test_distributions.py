import unittest
import numpy as np

from src.jit_learn.distributions import EmpiricalDistribution


class TestDistributions(unittest.TestCase):
    X = np.random.randn(2, 3)

    def test_empirical(self):
        dist = EmpiricalDistribution()
        dist.fit(self.X)
        self.assertIsNotNone(dist.data)

        sample = dist.sample(1)
        self.assertEquals(sample.shape, (1, 3))

        self.assertRaises(Exception, dist.sample, 3)
