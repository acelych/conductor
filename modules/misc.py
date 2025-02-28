from typing import Tuple

import torch
from torch import nn, Tensor

class CrossHadaNorm(nn.Module):
    def __init__(self, cs_expand):
        super().__init__()
        self.cs_expamd = cs_expand
        self.cov = nn.Parameter(torch.randn(cs_expand) * 0.05)
        
    def forward(self, x: Tensor, batch_norm: nn.BatchNorm2d, i_idx: Tensor, j_idx: Tensor) -> Tensor:
        mean: Tuple[Tensor] = (batch_norm.running_mean[i_idx], batch_norm.running_mean[j_idx])
        var: Tuple[Tensor] = (batch_norm.running_var[i_idx], batch_norm.running_var[j_idx])

        # calculate cross mean var
        mean_cross = (self.cov + mean[0] * mean[1]).view(1, -1, 1, 1)
        var_cross = (var[0] * var[1] + var[0] * mean[1]**2 + mean[0]**2 * var[1] - 2 * self.cov * mean[0] * mean[1] + self.cov**2).view(1, -1, 1, 1)

        # calculate normalize
        x = (x - mean_cross) / torch.sqrt(var_cross + 1e-5)
        return x

        