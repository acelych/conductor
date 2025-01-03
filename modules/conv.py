from typing import Sequence, Tuple, List

from torch import nn, Tensor

from ._utils import _convert_str2class


def autopad(k, p=None, d=1):
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
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, norm=nn.BatchNorm2d, act=nn.SiLU):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
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