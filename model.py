import os
import yaml

import torch
from torch import nn, optim

from .utils.res import ResourceManager


class Model():
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
                
                if isinstance(task_pat[key][0], list):   # pat require list of list
                    list_pat = task_pat[key][0]
                    for row in model_desc[key]:
                        assert isinstance(row, list) and len(row) == len(list_pat)
                        for term, tar in zip(row, list_pat):
                            if tar == 'int':
                                assert isinstance(term, int)
                            if tar == 'str':
                                assert isinstance(term, str)
                            if tar == 'list':
                                assert isinstance(term, list)
                                
        self.model_desc = model_desc
        
    def build_model(self):
        model = nn.Sequential()
        
                    
                
            
                