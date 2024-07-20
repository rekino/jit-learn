import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SchwartzNeuron(nn.Module):
    def __init__(self, dim: int, activation=F.tanh) -> None:
        super().__init__()

        self.dim = dim
        self.w = nn.Parameter(torch.randn(dim, 1))
        self.b = nn.Parameter(torch.randn(1))
        self.activation = activation

        self.norm = np.exp(-dim * np.log(2 * np.pi) / 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = (x * x).sum(dim=-1, keepdim=True)
        out = torch.exp(-out / 2) * self.norm
        out *= self.activation(x @ self.w + self.b)
        return out

    def laplacian(self, x: torch.Tensor) -> torch.Tensor:
        w_w = self.w.T @ self.w
        w_x = x @ self.w
        x_x = (x * x).sum(dim=-1, keepdim=True)

        s = self.activation(x @ self.w + self.b)
        s_x = 1 - s**2
        s_xx = 2 * (s**3 - s)

        out = s_xx * w_w - 2 * s_x * w_x + s * (x_x - self.dim)
        out *= torch.exp(-x_x / 2) * self.norm

        return out
    
    def norm(self) -> torch.Tensor:
        raise NotImplementedError()
