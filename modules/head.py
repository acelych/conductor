from typing import List, Tuple

import torch
from torch import nn, Tensor

class Classifier(nn.Module):
    def __init__(self, channel_in, num_classes, channel_expand, dropout: float = 0.2):
        super().__init__()
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(channel_in, channel_expand),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(channel_expand, num_classes)
        )
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
    
    @staticmethod
    def yaml_args_parser(channels, former, modules, args) -> Tuple[int, int, list, dict]:
        '''
        yaml format:
        [former, repeats, Classifier, [nc, ce, ...]
        '''
        c1 = channels[former]
        c2 = args[0]
        return c1, c2, [c1] + args, None