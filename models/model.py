import os
import yaml
from typing import List, Set, Iterable

import torch
from torch import nn, optim, Tensor

from ..modules._utils import BaseModule
from ..modules.module import ModuleProvider
from ..utils.res import ResourceManager


class Model(nn.Module):
    def __init__(self, layers: Iterable, save: Iterable):
        super().__init__()
        self.layers = layers
        self.save = save
                
    def forward(self, x):
        saved = dict()
        for layer in self.layers:
            # forward
            if x.f != -1:
                x = layer(*(saved.get(i) for i in ([x.f] if isinstance(x.f, int) else x.f)))
            else:
                x = layer(x)
                
            # save layer's output
            if x.i in self.save:
                saved[x.i] = x
        return x
        

class ModelManager():
    def __init__(self, yaml_path: str, task: str):
        self.task = task
        self.parse_yaml(yaml_path)
    
    def parse_yaml(self, yaml_path: str):
        with open(yaml_path, 'r') as f:
            model_desc = yaml.safe_load(f)
            
        with open(ResourceManager.get_task_pattern(self.task), 'r') as f:
            task_pat = yaml.safe_load(f)
        
        for key in task_pat:
            assert key in model_desc, f"Can not found argument '{key}' for {self.task} in {yaml_path}"
            
            if task_pat[key] == 'int':                   # pat require integer
                assert isinstance(model_desc[key], int)
            elif task_pat[key] == 'str':                 # pat require string
                assert isinstance(model_desc[key], str)
            elif isinstance(task_pat[key], list):        # pat require list
                assert isinstance(model_desc[key], list)
                                
        self.model_desc: dict = model_desc
        
    def build_model(self, input_dim=3):
        
        channels = [input_dim]
        layers: List[nn.Module] = []
        save: Set[int] = set()
        layers_desc = self.model_desc.get("backbone") + self.model_desc.get("head")
        
        for i, (f, n, m, args) in enumerate(layers_desc):
            assert all(i >= x for x in ([f] if isinstance(f, int) else f)), f"expect former layers of layer {i}, got {f}"
            assert n >= 1, f"expect n of layer {i} >= 1, got {n}"
            
            m: BaseModule = ModuleProvider(m)
            c1, c2, args, kwargs = m.yaml_args_parser(channels, f, ModuleProvider.get_modules(), args)
            
            if i == 0:
                channels = [c2]
            else:
                channels.append(c2)
                
            save.union(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
            _m = nn.Sequential(*(m(*args, **kwargs) for _ in range(n))) if n > 1 else m(*args, **kwargs)
            _m.p, _m.i, _m.f = sum(x.numel() for x in _m.parameters()), i, f
            layers.append(_m)
            
        return Model(nn.Sequential(*layers), sorted(list(save)))
        
                    
                
            
                