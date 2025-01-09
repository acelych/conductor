import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from .models import ModelManager
from .data import DataLoaderManager
from .utils import CommandDetails, IndexLogger

class Trainer():
    def __init__(self, cd: CommandDetails, model_mng: ModelManager, data_mng: DataLoaderManager, rank: int):
        self.cd = cd
        self.model_mng = model_mng
        self.data_mng = data_mng
        self.rank = rank
        
    def main_work(self):
        train_dataloader = self.data_mng.get_dataloader("train", self.rank)
        val_dataloader = self.data_mng.get_dataloader("val", self.rank)
        
        model = self.model_mng.build_model()
        ddp_model = DDP(model, device_ids=[self.rank])
        self.criterion = self.cd.criterion()
        self.optimizer = self.cd.optimizer(model.parameters(), self.cd.learn_rate)
        
        for epoch in range(self.cd.epochs):
            self.train_epoch(ddp_model, train_dataloader)
            dist.barrier()
            self.val_epoch(ddp_model, val_dataloader)
    
    def train_epoch(self, model: DDP, dataloader: DataLoader):
        model.train()
        for x, label in dataloader:
            output = model(x)
            self.optimizer.zero_grad()
            loss = self.criterion(output, label)
            loss.backward()
            self.optimizer.step()

    def val_epoch(self, model: nn.Module, dataloader: DataLoader):
        model.eval()
        for x, label in dataloader:
            output = model(x)
            loss = self.criterion(output, label)
            
    @staticmethod
    def init_trainer(rank, cd: CommandDetails, model_mng: ModelManager, data_mng: DataLoaderManager):
        dist.init_process_group("nccl", rank=rank, world_size=len(cd.world))
        trainer = Trainer(cd, model_mng, data_mng, rank)
        trainer.main_work()

    
class TrainerManager():
    def __init__(self, cd: CommandDetails):
        self.model_mng = ModelManager(cd.model_yaml_path, cd.task)
        self.data_mng = DataLoaderManager(cd.data_yaml_path, cd)
        self.logger = IndexLogger(cd.task)
        self.cd = cd
        
    def train(self):
        mp.spawn(Trainer.init_trainer, args=(self.cd, self.model_mng, self.data_mng), nprocs=len(self.cd.world), join=True)