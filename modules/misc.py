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
    
    
class DySig(nn.Module):
    def __init__(self, dim, alpha_init_value=0.35):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1) * 2)
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        x = torch.sigmoid(self.alpha * x) - 0.5
        return self.weight * x + self.beta
    
    
class DySoft(nn.Module):
    def __init__(self, dim, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, dim, 1, 1))
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.alpha * x
        x = x / (1 + torch.abs(x))
        return x * self.weight + self.bias
    
    
class DyAlge(nn.Module):
    def __init__(self, dim, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, dim, 1, 1))
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.alpha * x
        x = x / torch.sqrt(1 + x * x)
        return x * self.weight + self.bias
    
    
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


class ECA(nn.Module):
    """Constructs a ECA module.
    Args:
        k_size: Adaptive selection of kernel size
        weights_only: If True, returns the weights only without applying sigmoid activation.
    """
    def __init__(self, k_size=3, weights_only=False):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        if weights_only:
            self.forward = self.wo_forw
        else:
            self.sigmoid = nn.Sigmoid()
            self.forward = self.reg_forw

    def reg_forw(self, x: Tensor) -> Tensor:
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Apply sigmoid activation
        y = self.sigmoid(y)

        return y
    
    def wo_forw(self, x: Tensor) -> Tensor:
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Directly return weights
        return y
        