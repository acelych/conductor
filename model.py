import os
import yaml
from pathlib import Path
from typing import List, Set, Union

from torch import nn, Tensor
from torchvision import models

from .modules._utils import BaseModule
from .modules.module import ModuleProvider
from .utils import ResourceManager, ConfigManager, get_module_class_str, isbuiltin


class Model(nn.Module):
    def __init__(self, layers: nn.ModuleList, save: list):
        super().__init__()
        self.layers: nn.ModuleList = layers
        self.save: Tensor = Tensor(save)
        self.apply(self._init_weights)
                
    def _forward_impl(self, x: Tensor) -> Tensor:
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
    
    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
    
    def info(self) -> List[str]:
        head = [f"{'':>3}{'from':>8}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}"]
        layerinfo = [f"{item.i:>3}{str(item.f):>8}{item.n:>3}{item.p:10.0f}  {item.t:<45}{item.args:<30}" for item in self.layers]
        return head + layerinfo
    
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
        self.model_desc = dict()
        
        if cm.model_yaml_path.exists():
            # yaml model
            self.parse_yaml(cm.model_yaml_path)
            self.model_type = 'yaml'
        else:
            # torchvision model
            self.setup_tv_model(cm.model_yaml_path.__str__())
            self.model_type = 'tv'
    
    def parse_yaml(self, yaml_path: Path):
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
                                
        self.model_desc.update(model_desc)

    def setup_tv_model(self, instruct: str):
        import ast

        tv_model_cfg = [sub.strip() for sub in instruct.split(',')]
        assert len(tv_model_cfg) == 2, f"expect tv model cfg as: 'builder,num_classes', got '{instruct}'"
        assert hasattr(models, tv_model_cfg[0]), f"unexpect torchvision.models attribute '{tv_model_cfg[0]}'"
        try:
            kwargs = dict(ast.literal_eval(tv_model_cfg[1]))
        except Exception as e:
            raise AssertionError(f"expect tv model cfg second args to be a dict") from e
        assert 'num_classes' in kwargs, f"expect tv model cfg has 'num_classes' key"

        self.tv_builder = getattr(models, tv_model_cfg[0])
        self.tv_kwargs = kwargs
        self.model_desc['nc'] = kwargs.get('num_classes')

    @staticmethod
    def tv_info(obj: object):
        return f"(torchvision model) {get_module_class_str(obj)}"
        
    def build_model(self, input_dim=3) -> Union[Model, nn.Module]:
        
        if self.model_type == 'yaml':
            return self.build_desc_model(self.model_desc, input_dim)
        
        elif self.model_type == 'tv':
            tv_model = self.tv_builder(**self.tv_kwargs)
            setattr(tv_model, 'info', lambda :self.tv_info(tv_model))
            return tv_model
        
    @staticmethod
    def build_desc_model(model_desc: dict, input_dim=3) -> Model:
        channels = [input_dim]
        layers = nn.ModuleList()
        save: Set[int] = set()
        layers_desc = model_desc.get("backbone") + model_desc.get("head")
        
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
            _m.i, _m.f, _m.n = i, f, n
            _m.p = sum(x.numel() for x in _m.parameters())
            _m.t = m.__module__ + '.' + m.__name__

            args_desc = [str([(item if isbuiltin(item) else get_module_class_str(item)) for item in args])]
            if len(kwargs) > 0:
                args_desc.append(str({k: (v if isbuiltin(v) else get_module_class_str(v)) for k, v in kwargs.items()}))
            _m.args = ' '.join(args_desc)

            layers.append(_m)
            
        return Model(layers, sorted(list(save)))
            
                