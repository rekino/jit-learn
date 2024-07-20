from typing import Callable

import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.autograd import grad

from .distribution import GeneralizaedSLP
from .test_function import SchwartzNeuron


def jit_harmonic_loss(
        gslp: GeneralizaedSLP,
        rho: SchwartzNeuron,
        loss: Callable,
        train_samples: torch.Tensor,
        train_targets: torch.Tensor | None = None
        ) -> None:
    train_pred = gslp.val(train_samples)

    if train_targets is None:
        l = loss(train_pred)
    else:
        l = loss(train_pred, train_targets)
    
    l_h = grad(l.sum(), train_pred, retain_graph=True)[0]
