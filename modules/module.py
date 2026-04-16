import torch
from torch import nn, optim
from typing import Optional, Tuple, Dict, Sequence, List, Union, Any

def _convert_str2class(m_str: str, modules: dict) -> Optional[Union[type, nn.Module]]:
    if m_str is None:
        return None
    
    if hasattr(nn, m_str):
        m = getattr(nn, m_str)  # from torch modules
    elif m_str in modules:
        m = modules[m_str]  # from local modules
    else:
        raise AssertionError(f"could not find any matching module for '{m_str}'")
    
    return m

class BaseModule(nn.Module):
    def __init__(self, yaml_obj: object = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.yaml_obj = yaml_obj
        
    def get_yaml_obj(self) -> object:
        return self.yaml_obj

    @staticmethod
    def yaml_args_parser(channels: List[int], former: Union[List[int], int], modules: dict, args: List[Any]) -> Tuple[int, int, list, dict]:
        raise NotImplementedError

class ModuleProvider():
    _modules_dict = None
    
    @classmethod
    def get_module(cls, module_name: str) -> BaseModule:
        if cls._modules_dict is None:
            cls._init_modules()
        assert module_name in cls._modules_dict, f"unexpected module '{module_name}'"
        return cls._modules_dict.get(module_name)
    
    @classmethod
    def get_modules(cls) -> dict:
        if cls._modules_dict is None:
            cls._init_modules()
        return cls._modules_dict

    @classmethod
    def _init_modules(cls):
        from .block import InvertedResidual, UniversalInvertedBottleneck, HadamardResidual, HadamardResidualV2, AdaptiveBottleneck, StarBlock
        from .conv import ConvNormAct, SearchableConvNormAct
        from .head import Classifier, ClassifierSimple
        from .nas import SearchableBlank, SearchableModule, SearchableBaseModule

        _modules_list = [
            ConvNormAct, InvertedResidual, UniversalInvertedBottleneck,
            HadamardResidual, HadamardResidualV2, AdaptiveBottleneck,
            StarBlock, Classifier, ClassifierSimple, SearchableBlank,
            SearchableModule, SearchableBaseModule, SearchableConvNormAct,
        ]
        cls._modules_dict = {m.__name__: m for m in _modules_list}
