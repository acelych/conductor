import torch
from torch import nn, optim

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