import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from typing import List, Tuple, Dict, Union, Optional, Callable, Literal

from ._utils import BaseModule, _convert_str2class


class TauScheduler:
    def __init__(
        self,
        params: List[nn.Parameter],
        total_epochs: int,
        max_tau: float = 4.0,
        min_tau: float = 0.1,
        annealing: Literal["linear", "exp", "cos"] = "cos",
    ):
        self.params = params
        self.max_tau = max_tau
        self.min_tau = min_tau
        self.total_epochs = total_epochs
        self.annealing_fn = self._get_annealing_fn(annealing)

    def _get_annealing_fn(self, method: str) -> Callable[[int], float]:
        strategies = {
            "linear": lambda e: self.max_tau - (self.max_tau - self.min_tau) * (e / self.total_epochs),
            "exp": lambda e: self.max_tau * ((self.min_tau / self.max_tau) ** (e / self.total_epochs)),
            "cos": lambda e: self.min_tau + 0.5 * (self.max_tau - self.min_tau) * (1 + math.cos(math.pi * e / self.total_epochs)),
        }
        return strategies[method]

    def step(self, epoch: int) -> float:
        tau = self.annealing_fn(epoch)
        for param in self.params:
            param.data.fill_(tau)
        return tau


class SearchableBlank(BaseModule):
    """
    A blank module that can be used as a placeholder in the architecture search.
    It does not perform any operation and is used to maintain the structure of the model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return x

    @staticmethod
    def yaml_args_parser(channels, former, modules, args) -> Tuple[int, int, list, dict]:
        """
        yaml format:
        [former, repeats, SearchableModule, [c2, [SearchableBlank, []], ...]]
        """
        c1 = channels[former]
        c2 = args[0]
        assert c1 == c2, "SearchableBlank should not change the channel size."
        return c1, c2, [], {}
    
    
class SearchableModule(BaseModule):
    """
    A searchable module that can contain multiple candidate operations 
    which is an implement of GDAS (Gradient-based DARTS) style architecture search.
    
    Since this module accept all forms of nn.Module, 
    person who use this module should ensure that all candidates are 
    compatible with the input and output shapes.
    """
    
    def __init__(self, *candidates: nn.Module, **kwargs):
        super().__init__(**kwargs)
        self.ops = nn.ModuleList(list(candidates))
        self.nas_alpha = Parameter(torch.randn(len(self.ops)) * 1e-1)  # Architecture parameters
        self.nas_tau = nn.Parameter(torch.tensor(4), requires_grad=False)  # Temperature param for Gumbel softmax
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = F.gumbel_softmax(self.nas_alpha, tau=self.nas_tau.data, hard=True, dim=-1)
        out = sum(w * op(x) for w, op in zip(weights, self.ops))
        return out
            
    def get_optimal(self) -> nn.Module:
        """
        Returns the module with the highest weight.
        """
        selected_idx = int(self.nas_alpha.argmax().item())
        optimal = self.ops[selected_idx]
        if isinstance(optimal, SearchableModule):
            return optimal.get_optimal()
        return optimal
    
    def get_yaml_obj(self) -> dict:
        """
        Returns the YAML object of the optimal module. (Unparseable)
        """
        old = super().get_yaml_obj()
        new = old.copy()
        confs = F.softmax(self.nas_alpha, dim=-1).tolist()
        new[3] = [new[3][0]]  # keep the first element (c2)
        new[3] += [[m.__class__.__module__ + '.' + m.__class__.__name__, conf] for m, conf in zip(self.ops, confs)]
        return new
    
    @staticmethod
    def yaml_args_parser(channels, former, modules, args) -> Tuple[int, int, list, dict]:
        """
        yaml format:
        [former, repeats, SearchableModule, [c2, [Module1, [...]], [Module2, [...]]]]
        
        note: 
        this parser is not responsible for parsing the sub-modules.
        please provide all params at the phase of building yaml file.
        """
        c1 = channels[former]
        c2 = args[0]
        modules_list = []
        
        for module_args in args[1:]:
            m: nn.Module = _convert_str2class(module_args[0], modules)  # get module class
            modules_list.append(m(*module_args[1]))
            
        return c1, c2, modules_list, dict()


class SearchableBaseModule(SearchableModule):
    """
    A searchable module that inherits from `SearchableModule`.
    
    This module is designed to be used with `BaseModule` for more accurate YAML representation.
    It'll call the parser of the secondary module when parsing.
    """

    def __init__(self, *candidates: BaseModule, **kwargs):
        super().__init__(*candidates, **kwargs)
        
    def get_yaml_obj(self) -> dict:
        """
        Returns the YAML object of the optimal module. (Unparseable)
        """
        old = super(SearchableModule, self).get_yaml_obj()
        new = old.copy()
        confs = F.softmax(self.nas_alpha, dim=-1).tolist()
        new[3] = [new[3][0]]  # keep the first element (c2)
        new[3] += [[m.get_yaml_obj(), conf] for m, conf in zip(self.ops, confs)]
        return new
    
    @staticmethod
    def yaml_args_parser(channels, former, modules, args) -> Tuple[int, int, list, dict]:
        """
        yaml format:
        [former, repeats, SearchableModule, [c2, [Module1, [...]], [Module2, [...]]]]
        """
        c1 = channels[former]
        c2 = args[0]
        modules_list = []
        
        for module_args in args[1:]:
            m: BaseModule = _convert_str2class(module_args[0], modules)  # get module class
            assert c2 == module_args[1][0], f"expect c2 of {m.__name__} to be {c2}, got {module_args[1][0]}"
            _, _, args_, kwargs_ = m.yaml_args_parser(channels, former, modules, module_args[1])
            kwargs_['yaml_obj'] = [former, 1, module_args[0], module_args[1]]  # make sub-module yaml obj
            modules_list.append(m(*args_, **kwargs_))
            
        return c1, c2, modules_list, dict()
    