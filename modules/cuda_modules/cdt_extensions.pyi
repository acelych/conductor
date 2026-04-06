from __future__ import annotations
import torch
__all__: list[str] = ['cross_hada', 'cross_hada_balanced', 'cross_hada_mixed', 'dysoft']
def cross_hada(arg0: torch.Tensor) -> torch.Tensor:
    """
    Cross Hadamard Product
    """
def cross_hada_balanced(arg0: torch.Tensor) -> torch.Tensor:
    """
    Cross Hadamard Product (Balanced)
    """
def cross_hada_mixed(arg0: torch.Tensor, arg1: torch.Tensor, arg2: int) -> torch.Tensor:
    """
    Cross Hadamard Product (Mixed TopK Ops)
    """
def dysoft(arg0: torch.Tensor, arg1: torch.Tensor, arg2: torch.Tensor, arg3: torch.Tensor) -> None:
    """
    DySoft Normalization
    """
