from typing import List, Tuple

import torch
from torch import nn, Tensor
from torchvision.ops import SqueezeExcitation as SElayer

from .conv import ConvNormAct, MeanFilter
from ._utils import _convert_str2class, BaseModule

class InvertedResidual(BaseModule):
    def __init__(self, c1, c2, ce, k, s, d, se, act):
        super().__init__()
        
        assert 1 <= s <= 2, f"illegal stride value '{s}'"
        self.use_res_connect = s == 1 and c1 == c2
        layers: List[nn.Module] = []
        
        # expand
        if c1 != ce:
            layers.append(
                ConvNormAct(
                    c1, ce, k=1, norm=nn.BatchNorm2d, act=act
                )
            )
        
        # depthwise
        s = 1 if d > 1 else s
        layers.append(
            ConvNormAct(
                ce, ce, k=k, s=s, g=ce, d=d, norm=nn.BatchNorm2d, act=act
            )
        )
        if se:
            layers.append(SElayer(ce, ce // 4, scale_activation=nn.Hardsigmoid))
            
        # project
        layers.append(
            ConvNormAct(
                ce, c2, k=1, norm=nn.BatchNorm2d, act=None
            )
        )
        
        self.block = nn.Sequential(*layers)
        
    def forward(self, x: Tensor) -> Tensor:
        res = self.block(x)
        if self.use_res_connect:
            res = x + res
        return res
    
    @staticmethod
    def yaml_args_parser(channels, former, modules, args) -> Tuple[int, int, list, dict]:
        '''
        yaml format:
        [former, repeats, InvertedResidual, [c2, ce, k, s, d, false, ReLU]]
        '''
        c1 = channels[former]
        c2 = args[0]
        act_type = _convert_str2class(args[-1], modules)  # get act
        return c1, c2, [c1] + args[:-1] + [act_type], dict()
    

class HadamardExpansion(nn.Module):
    def __init__(self, c1, ce):
        super().__init__()

        self.c1 = c1
        self.ce = ce
        self.candis_num = c1 * (c1 - 1) // 2
        assert self.ce <= self.candis_num, f"too much expansion channels required"
        
        # full-connect
        # self.fc = nn.Conv2d(c1, c1, 1)
        
        # hadamard-pairs selected
        self.selected_met: Tensor
        self.selected_seq: Tensor
        self.register_buffer("selected_seq", torch.zeros((2, self.ce), dtype=torch.int64))
        
        # hadamard-pairs candidates
        can_idx_loc = torch.zeros((3, self.candis_num * 2), dtype=torch.int64)
        can_idx_val = torch.ones(self.candis_num * 2)
        can_idx = 0
        for i in range(c1):
            for j in range(i + 1, c1):
                can_idx_loc[:, can_idx * 2] = torch.tensor([0, can_idx, i])
                can_idx_loc[:, can_idx * 2 + 1] = torch.tensor([1, can_idx, j])
                can_idx += 1
        self.candis_met: Tensor
        self.register_buffer('candis_met', torch.sparse_coo_tensor(indices=can_idx_loc, values=can_idx_val).to_dense())

        # gumbel-softmax
        self.logits = nn.Parameter(torch.randn(self.candis_num))
        self.tau = nn.Parameter(torch.tensor(2.0))
        self.tau_adj = nn.Parameter(torch.tensor(0), requires_grad=False)

        # normalize
        self.mean = MeanFilter(ce)
        self.norm = nn.BatchNorm2d(ce)
        
        # initialize
        torch.nn.init.uniform_(self.logits, a=-0.1, b=0.1)

    def forward(self, x: Tensor) -> Tensor:
        # x = self.fc(x)
        if self.training:
            self._update_mask()
            _shape = list(x.shape)
            _shape[1] = -1
            x = x.flatten(2)
            x_i = (self.selected_met[0, ...] @ x).view(_shape)
            x_j = (self.selected_met[1, ...] @ x).view(_shape)
            x = x.view(_shape)
        else:
            x_i = x[:, self.selected_seq[0], ...]
            x_j = x[:, self.selected_seq[1], ...]
        x_expand = x_i * x_j
        # if x_expand.isnan().any():
        #     ma, mi = x_expand.max(), x_expand.min()
        x_expand = self.norm(x_expand)
        # x_expand = self.mean(x_expand)
        # return x_expand
        return torch.cat([x, x_expand], dim=1)
        
    def _update_mask(self):
        mask = nn.functional.gumbel_softmax(self.logits, tau=self.tau)
        self._adjust_tau_with_grad(self.logits.grad)
        
        # topk & straight-through estimator
        _, topk_idx = torch.topk(mask, self.ce)
        _hard_mask = torch.zeros_like(self.logits).scatter_(0, topk_idx, 1.0)
        hard_mask = _hard_mask + mask.detach() - mask
        
        # candidates to selected
        _candis_to_select = torch.zeros(self.ce, self.candis_num, device=self.logits.device)
        _candis_to_select[torch.arange(self.ce), topk_idx] = 1.0
        candis_to_select = _candis_to_select * hard_mask.unsqueeze(0)

        # selected channel
        self.selected_met = candis_to_select @ self.candis_met
        self.selected_seq = torch.argmax(self.selected_met, dim=2)

    def _adjust_tau_with_grad(self, grad: Tensor, alpha: float = 0.01):
        if self.tau_adj.data != 0 and grad is not None:
            alpha *= 1 if self.tau_adj <= grad.norm() else -1
            self.tau.data = torch.clamp(self.tau * (1 + alpha), max=4.0, min=0.1)
        if grad is not None:
            self.tau_adj.data = grad.norm()
            
            
class HadamardExpansionV2(nn.Module):
    def __init__(self, c1, ce):
        super().__init__()

        self.c1 = c1
        self.ce = ce

        # gumbel-softmax
        self.logits = nn.Parameter(torch.randn(2, ce, c1))
        self.tau = nn.Parameter(torch.tensor(2.0))
        self.tau_adj = nn.Parameter(torch.tensor(0), requires_grad=False)
        self.mask_saved: Tensor
        self.register_buffer("mask_saved", torch.zeros((2, ce), dtype=torch.int64))

        # normalize
        self.mean = MeanFilter(ce)
        self.norm = nn.BatchNorm2d(ce)
        
        # initialize
        torch.nn.init.uniform_(self.logits, a=-0.1, b=0.1)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mask = self._update_mask()
            _shape = list(x.shape)
            _shape[1] = -1
            x = x.flatten(2)
            x_i = (mask[0, ...] @ x).view(_shape)
            x_j = (mask[1, ...] @ x).view(_shape)
            x = x.view(_shape)
        else:
            x_i = x[:, self.mask_saved[0], ...]
            x_j = x[:, self.mask_saved[1], ...]
        x_expand = x_i * x_j
        x_expand = self.norm(x_expand)
        return x_expand
        
    def _update_mask(self):
        mask = nn.functional.gumbel_softmax(self.logits, tau=self.tau, hard=True)
        self._adjust_tau_with_grad(self.logits.grad)
        self.mask_saved = torch.argmax(mask, dim=2)
        return mask

    def _adjust_tau_with_grad(self, grad: Tensor, alpha: float = 0.01):
        if self.tau_adj.data != 0 and grad is not None:
            alpha *= 1 if self.tau_adj <= grad.norm() else -1
            self.tau.data = torch.clamp(self.tau * (1 + alpha), max=4.0, min=0.1)
        if grad is not None:
            self.tau_adj.data = grad.norm()


class HadamardCompression(nn.Module):
    def __init__(self, c1, norm = nn.BatchNorm2d):
        super().__init__()



class HadamardResidual(BaseModule):
    def __init__(self, c1, c2, ce, k, s, d, se, act):
        super().__init__()
        
        assert 1 <= s <= 2, f"illegal stride value '{s}'"
        self.use_res_connect = s == 1 and c1 == c2
        layers: List[nn.Module] = []

        # hadamard-expansion
        if c1 != ce:
            layers.append(
                HadamardExpansion(c1, ce - c1)
            )
        
        # # depthwise-convolution
        # s = 1 if d > 1 else s
        # layers.append(
        #     ConvNormAct(ce, ce, k=k, s=s, g=ce, d=d, norm=nn.BatchNorm2d, act=act)
        # )
        # if se:
        #     layers.append(SElayer(ce, ce // 4, scale_activation=nn.Hardsigmoid))
            
        # # project
        # layers.append(
        #     ConvNormAct(ce, c2, k=1, norm=nn.BatchNorm2d, act=None)
        # )
            
        # project
        layers.append(
            ConvNormAct(ce, c2, k=1, norm=nn.BatchNorm2d, act=None)
        )
        if se:
            layers.append(SElayer(c2, c2 // 4, scale_activation=nn.Hardsigmoid))
        
        # depthwise-convolution
        s = 1 if d > 1 else s
        layers.append(
            ConvNormAct(c2, c2, k=k, s=s, g=c2, d=d, norm=nn.BatchNorm2d, act=act)
        )
        
        self.block = nn.Sequential(*layers)
        
    def forward(self, x: Tensor) -> Tensor:
        res = self.block(x)
        if self.use_res_connect:
            res = x + res
        return res
    
    @staticmethod
    def yaml_args_parser(channels, former, modules, args) -> Tuple[int, int, list, dict]:
        '''
        yaml format:
        [former, repeats, InvertedResidual, [c2, ce, k, s, d, se, act]]
        '''
        c1 = channels[former]
        c2 = args[0]
        act_type = _convert_str2class(args[-1], modules)  # get act
        return c1, c2, [c1] + args[:-1] + [act_type], dict()
