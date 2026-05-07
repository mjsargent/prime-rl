from __future__ import annotations

import torch
from torch import Tensor


def controllability_tensor(jacobian: Tensor, residual_covariance: Tensor, regularization: float) -> tuple[Tensor, Tensor]:
    metric = jacobian @ residual_covariance @ jacobian.transpose(-1, -2)
    inverse = torch.linalg.inv(metric + regularization * torch.eye(metric.shape[-1], device=metric.device))
    return metric, inverse
