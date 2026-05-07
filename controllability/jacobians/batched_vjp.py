from __future__ import annotations

import torch
from torch import Tensor


def batched_vjp(outputs: Tensor, inputs: Tensor, vectors: Tensor) -> Tensor:
    grads = []
    for vector in vectors:
        grads.append(torch.autograd.grad((outputs * vector).sum(), inputs, retain_graph=True)[0])
    return torch.stack(grads)
