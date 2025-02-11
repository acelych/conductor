import torch
from torch import nn, optim

from .block import *
from .conv import *
from .head import *
from ._utils import BaseModule

class ModuleProvider():
    _modules = [
        ConvNormAct,
        InvertedResidual,
        Classifier
    ]
    _modules: dict = {m.__name__: m for m in _modules}
    
    @classmethod
    def get_module(cls, module_name: str) -> BaseModule:
        assert module_name in cls._modules, f"unexpected module '{module_name}'"
        return cls._modules.get(module_name)
    
    @classmethod
    def get_modules(cls) -> dict:
        return cls._modules