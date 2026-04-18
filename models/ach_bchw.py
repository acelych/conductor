from typing import List, Sequence, Tuple, Union
import math

import torch
from torch import nn, Tensor

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
    
class DySoft(nn.Module):
    def __init__(self, dim, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, dim, 1, 1))
        
    def forward(self, x: Tensor) -> Tensor:
        # if self.training:
        x = self.alpha * x
        x = x / (1 + torch.abs(x))
        return x * self.weight + self.bias
        # else:
        #     dysoft(x, self.alpha, self.weight, self.bias)
        #     return x

class AdaptiveCrossHadamard(nn.Module):
    def __init__(self, c1, cs, norm: nn.Module):
        super().__init__()

        self.c1 = c1
        self.cs = cs
        self.cs_expand = cs * (cs - 1) // 2
        self.ce = c1 + self.cs_expand
        
        # fc-expand
        self.fc = nn.Conv2d(c1, c1, 1)
        self.norm_x = nn.BatchNorm2d(c1)
        
        # eva-net
        self.eva_net = ECA(5, True)

        # gumbel-softmax
        self.tau = nn.Parameter(torch.tensor(4.0), requires_grad=False)
        self.tau_adj = nn.Parameter(torch.tensor(0), requires_grad=False)
        
        # cross-hadamard
        self.hadamard_i: Tensor
        self.hadamard_j: Tensor
        self.register_buffer("hadamard_i", torch.zeros(self.cs_expand, dtype=torch.int64, requires_grad=False))
        self.register_buffer("hadamard_j", torch.zeros(self.cs_expand, dtype=torch.int64, requires_grad=False))
        h_idx = 0
        for i in range(self.cs):
            for j in range(i + 1, self.cs):
                self.hadamard_i[h_idx] = i
                self.hadamard_j[h_idx] = j
                h_idx += 1

        # expand normalize
        self.norm = norm(self.cs_expand)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x)
        x = self.norm_x(x)
        
        x_sel_ex = self._get_selected(x)
        x_sel_ex = self.norm(x_sel_ex)

        return torch.cat([x, x_sel_ex], dim=1)
        
    def _get_selected(self, x: Tensor) -> Tensor:
        logits = self.eva_net(x).flatten(1)
        if self.training:
            _shape = list(x.shape)
            _shape[1] = -1
            
            mask = nn.functional.gumbel_softmax(logits, tau=self.tau)
            self._adjust_tau_with_grad(self.eva_net.conv.weight.grad) # 提取单层网络梯度
            _, topk_idx = torch.topk(mask, self.cs)
            _batchs = torch.arange(_shape[0]).unsqueeze(-1).expand_as(topk_idx).flatten(0)
            _rows = torch.arange(self.cs).unsqueeze(0).expand_as(topk_idx).flatten(0)
            
            hard_mask_ = torch.zeros_like(logits)
            hard_mask_[_batchs, topk_idx.flatten(0)] = 1.0
            hard_mask = hard_mask_ + mask.detach() - mask
            
            # to one-hot matrix
            mask_mat_ = torch.zeros(_shape[0], self.cs, self.c1, device=hard_mask.device, dtype=torch.float32)
            mask_mat_[_batchs, _rows, topk_idx.flatten(0)] = 1.0
            mask_mat = mask_mat_ * hard_mask.unsqueeze(1)
            
            x_sel = (mask_mat @ x.flatten(2)).view(_shape)
            x_sel_ex = x_sel[:, self.hadamard_i, ...] * x_sel[:, self.hadamard_j, ...]
        else:
            _, topk_idx = torch.topk(logits, self.cs)
            batch_idx = torch.arange(x.size(0)).unsqueeze(-1).expand_as(topk_idx)
            x_sel = x[batch_idx, topk_idx]
            x_sel_ex = x_sel[:, self.hadamard_i, ...] * x_sel[:, self.hadamard_j, ...]
        return x_sel_ex

    def _adjust_tau_with_grad(self, grad: Tensor, alpha: float = 0.005):
        if self.tau_adj.data != 0 and grad is not None:
            alpha *= 1.0 if self.tau_adj <= grad.norm() else -1.0
            self.tau.data = torch.clamp(self.tau * (1.0 + alpha), max=4.0, min=0.01)
        if grad is not None:
            self.tau_adj.data = grad.norm()
            
def _autopad(k, p=None, d=1):
    if p is None:
        if isinstance(k, int) and isinstance(d, int):
            p = (k - 1) * d // 2
        elif isinstance(k, (int, Sequence)) and isinstance(d, (int, Sequence)):
            _dim = len(k) if isinstance(k, Sequence) else len(d)
            k = k if isinstance(k, Sequence) else tuple(k for _ in range(_dim))
            d = d if isinstance(d, Sequence) else tuple(d for _ in range(_dim))
            p = tuple((k[i] - 1) * d[i] // 2 for i in range(_dim))
        else:
            raise TypeError(f"expect kernel-size & dilation to be int or Sequence, got k:'{k.__class__.__name__}'; d:'{d.__class__.__name__}' instead.")
    return p
            
class ConvNormAct(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, norm=nn.BatchNorm2d, act=nn.SiLU, **kwargs):
        super().__init__(**kwargs)
        self.conv = nn.Conv2d(c1, c2, k, s, _autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = norm(c2) if norm else nn.Identity()
        self.act = act() if act else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))
                     
class GhostModule(nn.Module):
    def __init__(self, c1, ce, ratio=2, k_p=1, k_c=3):
        super().__init__()

        c_init = math.ceil(ce / ratio)
        c_new = c_init * (ratio - 1)
        self.primary_conv = ConvNormAct(c1, c_init, k_p, 1, norm=nn.BatchNorm2d, act=nn.Hardswish)
        self.cheap_operation = ConvNormAct(c_init, c_new, k_c, 1, g=c_init, norm=nn.BatchNorm2d, act=None)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)