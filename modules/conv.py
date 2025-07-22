from typing import Sequence, Tuple, List

import yaml
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from ._utils import _convert_str2class, _autopad, BaseModule
from .nas import SearchableModule


class ConvNormAct(BaseModule):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, norm=nn.BatchNorm2d, act=nn.SiLU, **kwargs):
        super().__init__(**kwargs)
        self.conv = nn.Conv2d(c1, c2, k, s, _autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = norm(c2) if norm else nn.Identity()
        self.act = act() if act else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))
    
    @staticmethod
    def yaml_args_parser(channels, former, modules, args) -> Tuple[int, int, list, dict]:
        '''
        yaml format:
        [former, repeats, ConvNormAct, [[c2, k, ...], BatchNorm2d, ReLU]]
        '''
        c1 = channels[former]
        c2 = args[0][0]
        norm = _convert_str2class(args[1], modules)
        act = _convert_str2class(args[2], modules)
        return c1, c2, [c1] + args[0], {'norm': norm, 'act': act}
    

class SearchableConvNormAct(SearchableModule):
    def __init__(self, c1, c2, k: List[int], s=1, p=None, g=1, d=1, norm=nn.BatchNorm2d, act=nn.SiLU, **kwargs):
        convs = [nn.Conv2d(c1, c2, k_it, s, _autopad(k_it, p, d), groups=g, dilation=d, bias=False) for k_it in k]                
        super().__init__(*convs, **kwargs)
        self.bn = norm(c2) if norm else nn.Identity()
        self.act = act() if act else nn.Identity()
        
    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(super().forward(x)))
    
    def get_yaml_obj(self) -> str:
        """
        Returns the YAML object of the optimal module. (Unparseable)
        """
        old = super(SearchableModule, self).get_yaml_obj()
        new = old.copy()
        confs = F.softmax(self.nas_alpha, dim=-1).tolist()
        new[3][1] = [[op.kernel_size, conf] for op, conf in zip(self.ops, confs)]
        return new
    
    @staticmethod
    def yaml_args_parser(channels, former, modules, args) -> Tuple[int, int, list, dict]:
        """
        yaml format:
        [former, repeats, SearchableConvNormAct, [[c2, [k...], ...], BatchNorm2d, ReLU]]
        """
        c1 = channels[former]
        c2 = args[0][0]
        norm = _convert_str2class(args[1], modules)
        act = _convert_str2class(args[2], modules)
        return c1, c2, [c1] + args[0], {'norm': norm, 'act': act}
    

class RepConv(nn.Module):
    def __init__(self, ):
        super().__init__()

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1,1,1,1])

    @staticmethod
    def _fuse_bn_tensor(conv: nn.Conv2d, bn: nn.BatchNorm2d):
        kernel = conv.weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std
    
    
class MeanFilter(nn.Module):
    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        self.channels = channels
        self.k_size: tuple = k_size if isinstance(k_size, tuple) else (k_size, k_size)
        self.padding = k_size // 2
        
        self.kernel: Tensor
        self.register_buffer(
            'kernel', 
            torch.ones(channels, 1, *self.k_size) / (k_size * k_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.conv2d(x, self.kernel, padding=self.padding, groups=self.channels)
    
    
__all__ = [
    "ConvNormAct",
    "SearchableConvNormAct",
    "MeanFilter"
]