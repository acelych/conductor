from typing import List, Tuple

import torch
from torch import nn, Tensor
from torchvision.ops import SqueezeExcitation as SElayer

from .conv import ConvNormAct, MeanFilter
from .misc import CrossHadaNorm
from ._utils import _convert_str2class, BaseModule, TensorCollector

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
        self.norm = nn.BatchNorm2d(ce)
        # self.norm = nn.LayerNorm(ce)
        # self.norm = nn.InstanceNorm2d(ce, affine=True)
        
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
        # TensorCollector.collect(x_expand, "expand")
        # TensorCollector.collect(x, "input")
        x_expand = self.norm(x_expand)
        # x = self.norm(torch.cat([x, x_expand], dim=1))
        
        # TensorCollector.collect(x_expand, "norm")
        # if (x_expand > 1e5).any() or (x > 1e5).any():
        #     TensorCollector.collect(x_expand, "expand_nan")
        #     TensorCollector.collect(x, "input_nan")
        #     TensorCollector.collect(x_expand, "norm_nan")
            
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
        if res.isnan().any():
            TensorCollector.collect(x, "HR_nan_in")
            TensorCollector.collect(res, "HR_nan_out")
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
            
            
class HadamardExpansionV2(nn.Module):
    def __init__(self, c1, cs):
        super().__init__()

        self.c1 = c1
        self.cs = cs
        self.cs_expand = cs * (cs - 1) // 2
        self.ce = c1 + self.cs_expand
        
        # fc-expand
        self.fc = nn.Conv2d(c1, c1, 1)
        self.norm_x = nn.BatchNorm2d(c1)
        
        # eva-net
        self.eva_avg = nn.AdaptiveAvgPool2d(1)
        self.eva_net = nn.Linear(c1, c1)

        # gumbel-softmax
        self.tau = nn.Parameter(torch.tensor(1.8))
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
        self.norm = CrossHadaNorm(self.cs_expand)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x)
        x = self.norm_x(x)
        
        x_sel = self._get_selected(x)
        x_sel_ex = x_sel[:, self.hadamard_i, ...] * x_sel[:, self.hadamard_j, ...]
        x_sel_ex = self.norm(x_sel_ex, self.norm_x, self.hadamard_i, self.hadamard_j)

        return torch.cat([x, x_sel_ex], dim=1)
        
    def _get_selected(self, x: Tensor) -> Tensor:
        logits = self.eva_net(self.eva_avg(x).flatten(1))
        if self.training:
            _shape = list(x.shape)
            _shape[1] = -1
            
            mask = nn.functional.gumbel_softmax(logits, tau=self.tau)
            self._adjust_tau_with_grad(self.eva_net.weight.grad)
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
            
            x_selected = (mask_mat @ x.flatten(2)).view(_shape)
        else:
            _, topk_idx = torch.topk(logits, self.cs)
            batch_idx = torch.arange(x.size(0)).unsqueeze(-1).expand_as(topk_idx)
            x_selected = x[batch_idx, topk_idx]
        return x_selected        

    def _adjust_tau_with_grad(self, grad: Tensor, alpha: float = 0.01):
        if self.tau_adj.data != 0 and grad is not None:
            alpha *= 1 if self.tau_adj <= grad.norm() else -1
            self.tau.data = torch.clamp(self.tau * (1 + alpha), max=4.0, min=0.1)
        if grad is not None:
            self.tau_adj.data = grad.norm()


class HadamardResidualV2(BaseModule):
    def __init__(self, c1, c2, cs, k, s, d, se, act):
        super().__init__()
        
        assert 1 <= s <= 2, f"illegal stride value '{s}'"
        self.use_res_connect = s == 1 and c1 == c2
        layers: List[nn.Module] = []

        # hadamard-expansion
        he_layer = HadamardExpansionV2(c1, cs)
        ce = he_layer.ce
        layers.append(he_layer)
        
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
        [former, repeats, HadamardResidualV2, [c2, cs, k, s, d, se, act]]
        '''
        c1 = channels[former]
        c2 = args[0]
        act_type = _convert_str2class(args[-1], modules)  # get act
        return c1, c2, [c1] + args[:-1] + [act_type], dict()


class StarBlock(BaseModule):
    def __init__(self, dim, mlp_ratio=3):
        super().__init__()
        self.dwconv = ConvNormAct(dim, dim, 7, 1, (7 - 1) // 2, g=dim, norm=nn.BatchNorm2d, act=None)
        self.f1 = ConvNormAct(dim, mlp_ratio * dim, 1, norm=None, act=None)
        self.f2 = ConvNormAct(dim, mlp_ratio * dim, 1, norm=None, act=None)
        self.g = ConvNormAct(mlp_ratio * dim, dim, 1, norm=nn.BatchNorm2d, act=None)
        self.dwconv2 = ConvNormAct(dim, dim, 7, 1, (7 - 1) // 2, g=dim, norm=nn.BatchNorm2d, act=None)
        self.act = nn.ReLU6()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + x
        return x
    
    @staticmethod
    def yaml_args_parser(channels, former, modules, args) -> Tuple[int, int, list, dict]:
        '''
        yaml format:
        [former, repeats, BaseModule, [dim, ratio]]
        '''
        c1 = channels[former]
        c2 = args[0]
        return c1, c2, args, dict()