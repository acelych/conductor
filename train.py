import torch
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP

from .models import ModelManager

class Trainer():
    def __init__(self):
        self.dataloader = None
        self.net: nn.Module = None
        self.criti: nn.Module = None
        self.optim: optim.Optimizer = None

    def train_epoch(self):
        self.net.train()
        for x, label in self.dataloader:
            output = self.net(x)
            self.optim.zero_grad()
            loss = self.criti(output, label)
            loss.backward()
            self.optim.step()
            
class DistributedTrainer(Trainer):
    def __init__(self):
        pass
    
class TrainerManager():
    def __init__(self, model_desc: str, ):
        pass