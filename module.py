import torch
from torch import nn, optim

class ModuleProvider():
    def __init__(self):
        pass
    
    @classmethod
    def __call__(cls, *args, **kwds) -> nn.Module:
        pass