import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from numpy.polynomial.hermite import hermgauss

from .test_function import SchwartzNeuron


class GeneralizaedSLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, activation=F.tanh, deg=10) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.deg = deg

        self.v = nn.Parameter(torch.randn(hidden_dim, out_dim))
        self.v0 = nn.Parameter(torch.randn(out_dim))
        self.W = nn.Parameter(torch.randn(in_dim, hidden_dim))
        self.b = nn.Parameter(torch.randn(hidden_dim))
        self.activation = activation

        u, c = hermgauss(deg)

        self.u = torch.Tensor(u) * np.sqrt(2)
        self.c = torch.Tensor(c)

        Ui, Uj = torch.meshgrid(self.u, self.u, indexing='xy')
        self.U = torch.stack([Ui, Uj], dim=-1)

        Ci, Cj = torch.meshgrid(self.c, self.c, indexing='xy')
        self.C = Ci * Cj
    
    def val(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(x @ self.W + self.b)

        return out @ self.v + self.v0

    def _qr(self, w: torch.Tensor) -> torch.Tensor:
        w_extended = w * torch.ones_like(self.W)
        A = torch.stack([w_extended, self.W], dim=0)
        A = A.permute(*torch.arange(A.ndim - 1, -1, -1))
        _, R = torch.linalg.qr(A)

        return R[0, 0, 0], R[:, :, 1].T

    def forward(self, phi: SchwartzNeuron) -> torch.Tensor:
        r, R = self._qr(phi.w)

        s = phi.activation(self.u * r + phi.b)
        phi_mean = torch.sum(self.c * s) / np.sqrt(np.pi)

        S = self.activation(self.U @ R + self.b)
        s = torch.ones(self.deg, self.deg) * s

        out = (self.C * s)[:, :, None]
        out = torch.sum(out * S, dim=(0, 1)) / np.pi

        return out @ self.v + self.v0 * phi_mean
    
    def laplacian(self, phi: SchwartzNeuron) -> torch.Tensor:
        r, R = self._qr(phi.w)

        U1, U2 = self.U[:, :, 0], self.U[:, :, 1]

        s = phi.activation(self.u * r + phi.b)
        s_x = 1 - s**2
        s_xx = 2 * (s**3 - s)
        lap_phi_mean = self.c * (s_xx * r**2 - 2 * s_x * r * self.u + s * (self.u**2 - 1))
        lap_phi_mean = torch.sum(lap_phi_mean) / np.sqrt(np.pi)

        S = self.activation(self.U @ R + self.b)
        s = torch.ones(self.deg, self.deg) * s

        out = self.C * (s_xx * r**2 - 2 * s_x * r * U1 + s * (U1**2 + U2**2 - 2))
        out = out[:, :, None]
        out = torch.sum(out * S, dim=(0, 1)) / np.pi

        return out @ self.v + self.v0 * lap_phi_mean


# if __name__ == '__main__':
#     from scipy.integrate import dblquad

#     ann = GeneralizaedSLP(2, 5, 1, deg=50, activation=F.tanh)
#     rho = SchwartzNeuron(2)

#     def integrand(x, y):
#         X = np.array([x, y])[None, :]
#         X = torch.Tensor(X)

#         out = ann.val(X) * rho(X)
#         return np.squeeze(out.detach().numpy())
    
#     def integrand2(x, y):
#         X = np.array([x, y])[None, :]
#         X = torch.Tensor(X)

#         out = ann.val(X) * rho.laplacian(X)
#         return np.squeeze(out.detach().numpy())

#     out_hat = ann(rho)
#     out_lap_hat = ann.laplacian(rho)

#     out, _ = dblquad(integrand, -np.inf, np.inf, -np.inf, np.inf)
#     print(out - out_hat)

#     out_lap, _ = dblquad(integrand2, -np.inf, np.inf, -np.inf, np.inf)
#     print(out_lap - out_lap_hat)