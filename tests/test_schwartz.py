import unittest

import torch

from src.jit_learn.test_function import SchwartzNeuron


class TestSchwartzNeuron(unittest.TestCase):
    def test_init(self) -> None:
        rho = SchwartzNeuron(2)

        self.assertIsNotNone(rho)

    def test_forward(self) -> None:
        rho = SchwartzNeuron(2)

        x = torch.randn(3, 2)
        out = rho(x)

        self.assertEquals(out.shape, (3, 1))

    def test_laplacian(self) -> None:
        rho = SchwartzNeuron(2)

        x = torch.randn(3, 2)
        out = rho.laplacian(x)

        self.assertEquals(out.shape, (3, 1))
