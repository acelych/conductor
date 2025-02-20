from typing import Tuple, Callable

import torch
from torch import nn, Tensor


def _convert_str2class(m_str: str, modules: dict):
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

    @staticmethod
    def yaml_args_parser(channels, former, modules, args) -> Tuple[int, int, list, dict]:
        raise NotImplementedError
    
    
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