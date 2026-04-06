from typing import Optional, Tuple, Dict, Sequence, List, Union, Any

import torch
from torch import nn, Tensor


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

def _autopad(k, p=None, d=1):
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

class BaseModule(nn.Module):
    def __init__(self, yaml_obj: object = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.yaml_obj = yaml_obj
        
    def get_yaml_obj(self) -> object:
        return self.yaml_obj

    @staticmethod
    def yaml_args_parser(channels: List[int], former: Union[List[int], int], modules: dict, args: List[Any]) -> Tuple[int, int, list, dict]:
        raise NotImplementedError
    

class TensorCollector:
    _enable: bool = False
    _collector: Dict[str, Tensor] = dict()
    
    @classmethod
    def enable(cls):
        cls._enable = True
        
    @classmethod
    def disable(cls):
        cls._enable = False
    
    @classmethod
    def collect(cls, t: Tensor, key: str):
        if not cls._enable:
            return
        if cls._collector.get(key) is not None:
            return
        cls._collector[key] = t
        
    @classmethod
    def get(cls, key) -> Tensor:
        return cls._collector.get(key)
        
    @classmethod
    def clear(cls):
        cls._collector.clear()
    
    
class Functional:
    
    @staticmethod
    def gumbel_topk(logits: Tensor, k: int, tau: float = 1, hard: bool = False, dim: int = -1) -> Tensor:
        gumbels = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
        )  # ~Gumbel(0,1)
        gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
        y_soft = gumbels.softmax(dim)
        _, topk_idx = torch.topk(y_soft, k)
        
        if hard:
            with torch.no_grad():
                y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, topk_idx, 1.0)
            # ret = y_hard - y_soft.detach() + y_soft
            ret = torch.where(y_hard == 1, y_hard, y_soft)
        else:
            ret = y_soft
        return ret, topk_idx
    
    @staticmethod
    def gumbel_softmax(logits: Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1) -> Tensor:
        gumbels = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
        )  # ~Gumbel(0,1)
        gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
        y_soft = gumbels.softmax(dim)

        if hard:
            # Straight through.
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Reparametrization trick.
            ret = y_soft
        return ret