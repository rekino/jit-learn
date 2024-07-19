import unittest
from unittest.mock import MagicMock

import torch

from src.jit_learn.distribution import GeneralizaedSLP


class TestGeneralizaedSLP(unittest.TestCase):
    def test_init(self) -> None:
        gslp = GeneralizaedSLP(2, 5, 3)

        self.assertIsNotNone(gslp)

    def test_val(self) -> None:
        gslp = GeneralizaedSLP(2, 5, 3)

        x = torch.randn(4, 2)
        out = gslp.val(x)

        self.assertEquals(out.shape, (4, 3))

    def test_qr(self) -> None:
        gslp = GeneralizaedSLP(2, 5, 3)

        w = torch.randn(2, 1)
        r, R = gslp._qr(w)

        self.assertAlmostEqual(r, torch.linalg.norm(w))
        self.assertEquals(R.shape, (2, 5))

    def test_forward(self) -> None:
        gslp = GeneralizaedSLP(2, 5, 3)

        rho = MagicMock()
        rho.w = torch.randn(2, 1)
        rho.b = torch.randn(1)
        rho.activation = torch.relu

        out = gslp(rho)
        self.assertEquals(out.shape, (3,))

    def test_laplacian(self) -> None:
        gslp = GeneralizaedSLP(2, 5, 3)

        rho = MagicMock()
        rho.w = torch.randn(2, 1)
        rho.b = torch.randn(1)
        rho.activation = torch.relu

        out = gslp.laplacian(rho)
        self.assertEquals(out.shape, (3,))
