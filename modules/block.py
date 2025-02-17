import math

from typing import List, Tuple

import torch
from torch import nn, Tensor
from torchvision.ops import SqueezeExcitation as SElayer

from .conv import ConvNormAct
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
        args[-1] = _convert_str2class(args[-1], modules)  # get act
        return c1, c2, [c1] + args, dict()
    

class HadamardExpansion(nn.Module):
    def __init__(self, c1, ce):
        super().__init__()

        self.c1 = c1
        self.ce = ce
        self.candidates_num = c1 * (c1 - 1) // 2
        assert self.ce <= self.candidates_num, f"too much expansion channels required"
        
        self.selected_met: Tensor
        self.selected_seq: Tensor
        
        can_idx_loc = torch.zeros((3, self.candidates_num * 2), dtype=torch.int64)
        can_idx_val = torch.ones(self.candidates_num * 2)
        can_idx = 0
        for i in range(c1):
            for j in range(i + 1, c1):
                can_idx_loc[can_idx * 2] = (0, can_idx, i)
                can_idx_loc[can_idx * 2 + 1] = (1, can_idx, j)
                can_idx += 1
        self.candidates_met = torch.sparse_coo_tensor(indices=can_idx_loc, values=can_idx_val)

        # gumbel-softmax
        self.logits = nn.Parameter(torch.randn(self.candidates_num))
        self.hard_mask = nn.Parameter(torch.zeros(self.candidates_num))
        self.tau = nn.Parameter(torch.tensor(2.0))

        # instance normalize
        self.norm = nn.InstanceNorm2d(c1 + ce, affine=True)
        
        # initialize
        torch.nn.init.uniform_(self.logits, a=-0.1, b=0.1)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            self._update_mask()
            _shape = x.shape
            x = x.flatten(1)
            x_i = (self.selected_met[0, ...] @ x).view(_shape)
            x_j = (self.selected_met[1, ...] @ x).view(_shape)
        else:
            x_i = x[:, self.selected_seq[0], ...]
            x_j = x[:, self.selected_seq[1], ...]
        x_expand = x_i * x_j
        return self.norm(torch.cat([x, x_expand], dim=1))
        
    def _update_mask(self):
        torch.clamp(self.tau, max=4.0, min=0.1, out=self.tau)
        mask = nn.functional.gumbel_softmax(self.logits, tau=self.tau)
        _, topk_idx = torch.topk(mask, self.ce)

        self.hard_mask.zero_()
        self.hard_mask[topk_idx] = 1.0

        # straight-through estimator
        self.hard_mask = self.hard_mask - mask.detach() + mask

        # slice selected channel
        self.selected_met = self.hard_mask.unsqueeze(1) * self.candidates_met
        non_zero = torch.sum(self.selected_met, dim=1) != 0
        self.selected_met = self.selected_met[non_zero]
        self.selected_seq = torch.argmax(self.selected_met, dim=1)
    

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
        layers.append(
            HadamardExpansion(c1, ce)
        )
        ce += c1
        
        # depthwise-convolution
        s = 1 if d > 1 else s
        layers.append(
            ConvNormAct(ce, ce, k=k, s=s, g=ce, d=d, norm=nn.BatchNorm2d, act=act)
        )
        if se:
            layers.append(SElayer(ce, ce // 4, scale_activation=nn.Hardsigmoid))
            
        # project
        layers.append(
            ConvNormAct(ce, c2, k=1, norm=nn.BatchNorm2d, act=None)
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
        args[-1] = _convert_str2class(args[-1], modules)  # get act
        return c1, c2, [c1] + args, dict()
