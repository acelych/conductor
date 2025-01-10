import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from .models import ModelManager
from .data import DataLoaderManager
from .utils import CommandDetails, IndexLogger, Recorder

class Trainer():
    def __init__(self, rank: int, cd: CommandDetails, model_mng: ModelManager, data_mng: DataLoaderManager, logger: IndexLogger):
        torch.cuda.set_device(cd.world[rank])
        dist.init_process_group("nccl", rank=rank, world_size=len(cd.world))
        self.rank = rank
        self.cd = cd
        self.model_mng = model_mng
        self.data_mng = data_mng
        self.logger = logger
        
    def main_work(self):
        train_dataloader = self.data_mng.get_dataloader("train", rank=self.rank)
        val_dataloader = self.data_mng.get_dataloader("val", rank=self.rank)
        
        model = self.model_mng.build_model().to(torch.cuda.current_device())
        ddp_model = DDP(model, device_ids=[torch.cuda.current_device()])
        self.criterion: nn.Module = self.cd.criterion()
        self.optimizer: optim.Optimizer = self.cd.optimizer(model.parameters(), self.cd.learn_rate)
        
        for epoch in range(self.cd.epochs):
            self.logger.start()
            
            if self.cd.task == "classify":
                index = IndexLogger.ClassifyIndex(epoch=epoch, time=self.logger.get_time())
            elif self.cd.task == "detect":
                index = IndexLogger.DetectIndex(epoch=epoch, time=self.logger.get_time())
            
            # learn & val
            self.train_epoch(ddp_model, train_dataloader, index)
            dist.barrier()
            self.val_epoch(ddp_model, val_dataloader, index)
            
            self.logger(index)
    
    def train_epoch(self, model: DDP, dataloader: DataLoader, index: IndexLogger.Index):
        model.train()
        losses = []
        for x, label in dataloader:
            output = model(x)
            self.optimizer.zero_grad()
            loss: Tensor = self.criterion(output, label)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        mean_loss = Recorder.converge_loss(losses)
        
        # let one of the gpu fill the blanks
        if dist.get_rank() == 0:
            index.record_train(mean_loss)
        dist.barrier()  # wait till it's done

    def val_epoch(self, model: nn.Module, dataloader: DataLoader, index: IndexLogger.Index):
        model.eval()
        recorder = Recorder(self.model_mng.model_desc.get("nc"))
        for x, label in dataloader:
            output = model(x)
            loss = self.criterion(output, label)
            recorder(output, label, loss)
        recorder.converge()
        
        # let one of the gpu fill the blanks
        if dist.get_rank() == 0:
            index.record_val(recorder)
        dist.barrier()  # wait till it's done
            
    @staticmethod
    def init_trainer(rank, cd: CommandDetails, model_mng: ModelManager, data_mng: DataLoaderManager, logger: IndexLogger):
        trainer = Trainer(rank, cd, model_mng, data_mng, logger)
        trainer.main_work()

    
class TrainerManager():
    def __init__(self, cd: CommandDetails):
        self.model_mng = ModelManager(cd.model_yaml_path, cd.task)
        self.data_mng = DataLoaderManager(cd.data_yaml_path, cd)
        self.logger = IndexLogger(cd, self.model_mng.model_desc)
        self.cd = cd
        
    def train(self):
        mp.spawn(Trainer.init_trainer, args=(self.cd, self.model_mng, self.data_mng, self.logger), nprocs=len(self.cd.world), join=True)