from typing import List, Tuple

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
        if ce != c2:
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
        [former, repeats, InvertedResidual, [c2, ce, k, s, d, false, ReLU]
        '''
        c1 = channels[former]
        c2 = args[0]
        args[-1] = _convert_str2class(args[-1], modules)  # get act
        return c1, c2, [c1] + args, None