import os
import yaml
from typing import List, Set

import torch
from torch import nn, optim, Tensor

from .modules._utils import BaseModule
from .modules.module import ModuleProvider
from .utils import ResourceManager, ConfigManager


class Model(nn.Module):
    def __init__(self, layers: nn.Sequential, save: list):
        super().__init__()
        self.layers: nn.Sequential = layers
        self.save: list = save
        self.apply(self._init_weights)
                
    def forward(self, x):
        saved = dict()
        for layer in self.layers:
            # forward
            if layer.f != -1:
                x = layer(*(saved.get(i) for i in ([layer.f] if isinstance(layer.f, int) else layer.f)))
            else:
                x = layer(x)
                
            # save layer's output
            if layer.i in self.save:
                saved[layer.i] = x
        return x
    
    def info(self) -> List[str]:
        info_list = []
        info_list.append(f"{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
        for layer in self.layers:
            info_list.append(f"{layer.i:>3}{str(layer.f):>20}{layer.n:>3}{layer.p:10.0f}  {layer.t:<45}{str(layer.args):<30}")
        return info_list
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        

class ModelManager():
    def __init__(self, cm: ConfigManager):
        self.task = cm.task
        self.parse_yaml(cm.model_yaml_path)
    
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
            
            m: BaseModule = ModuleProvider.get_module(m)
            c1, c2, args, kwargs = m.yaml_args_parser(channels, f, ModuleProvider.get_modules(), args)
            
            if i == 0:
                channels = [c2]
            else:
                channels.append(c2)
                
            save.union(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
            _m = nn.Sequential(*(m(*args, **kwargs) for _ in range(n))) if n > 1 else m(*args, **kwargs)
            _m.i, _m.f, _m.n, _m.p, _m.t, _m.args = i, f, n, sum(x.numel() for x in _m.parameters()), m.__module__ + '.' + m.__name__, args
            layers.append(_m)
            
        return Model(nn.Sequential(*layers), sorted(list(save)))
        
                    
                
            
                