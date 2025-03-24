from typing import Tuple

import torch
from torch import nn, Tensor

class CrossHadaNorm(nn.Module):
    def __init__(self, cs_expand):
        super().__init__()
        self.cs_expamd = cs_expand
        self.cov = nn.Parameter(torch.zeros(cs_expand))
        
    def forward(self, x: Tensor, batch_norm: nn.BatchNorm2d, topk_idx: Tensor, i_idx: Tensor, j_idx: Tensor) -> Tensor:
        running_mean = batch_norm.running_mean[topk_idx]
        running_var = batch_norm.running_var[topk_idx]
        mean: Tuple[Tensor] = (running_mean[:,i_idx], running_mean[:,j_idx])
        var: Tuple[Tensor] = (running_var[:,i_idx], running_var[:,j_idx])

        # calculate cross mean var
        mean_cross = (self.cov + mean[0] * mean[1]).view(x.size(0), -1, 1, 1)
        var_cross = (var[0] * var[1] + var[0] * mean[1]**2 + mean[0]**2 * var[1] - 2 * self.cov * mean[0] * mean[1] + self.cov**2).view(x.size(0), -1, 1, 1)

        # calculate normalize
        x = (x - mean_cross) / torch.sqrt(var_cross + 1e-5)
        return x
    
    
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.sqrt(torch.mean(x.pow(2), dim=(1, 2, 3), keepdim=True) + self.eps)
        x_normed = x / rms
        return self.gamma * x_normed + self.beta
    
    
class DyT(nn.Module):
    # Dynamic Tanh, cite: https://arxiv.org/abs/2503.10622
    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        
    def forward(self, x: Tensor) -> Tensor:
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias

        