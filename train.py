import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from .utils.cli import CommandDetails

from .models import ModelManager
from .data import DataLoaderManager
from .utils import IndexManager, Recorder

class Trainer:
    def __init__(self, rank: int, cd: CommandDetails, model_mng: ModelManager, data_mng: DataLoaderManager, idx_mng: IndexManager):
        torch.cuda.set_device(cd.world[rank])
        dist.init_process_group(
            "nccl" if dist.is_nccl_available() else "gloo", 
            rank=rank, 
            world_size=len(cd.world)
        )
        self.rank = rank
        self.cd = cd
        self.model_mng = model_mng
        self.data_mng = data_mng
        self.idx_mng = idx_mng
        
    def train(self):
        train_dataloader = self.data_mng.get_dataloader("train", rank=self.rank)
        val_dataloader = self.data_mng.get_dataloader("val", rank=self.rank)
        
        model = self.model_mng.build_model()
        ddp_model = DDP(model, device_ids=[torch.cuda.current_device()])
        self.criterion: nn.Module = self.cd.criterion()
        self.optimizer: optim.Optimizer = self.cd.build_optimizer(ddp_model.parameters())
        
        for epoch in range(self.cd.epochs):
            self.idx_mng.start()
            
            if self.cd.task == "classify":
                index = IndexManager.ClassifyIndex(epoch=epoch)
            elif self.cd.task == "detect":
                index = IndexManager.DetectIndex(epoch=epoch)
            
            # learn & val
            self.train_epoch(ddp_model, train_dataloader, index)
            dist.barrier()
            self.val_epoch(ddp_model, val_dataloader, index)
            
            # done, get time & commit
            index.time = self.idx_mng.get_time()
            self.idx_mng(index)
    
    def train_epoch(self, model: DDP, dataloader: DataLoader, index: IndexManager.Index):
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

    def val_epoch(self, model: nn.Module, dataloader: DataLoader, index: IndexManager.Index):
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
    def init_trainer(rank, cd: CommandDetails, model_mng: ModelManager, data_mng: DataLoaderManager, idx_mng: IndexManager):
        trainer = Trainer(rank, cd, model_mng, data_mng, idx_mng)
        trainer.train()

    
class TrainerManager:
    def __init__(self, cd: CommandDetails):
        self.model_mng = ModelManager(cd.model_yaml_path, cd.task)
        self.data_mng = DataLoaderManager(cd.data_yaml_path, cd)
        self.idx_mng = IndexManager(cd, self.model_mng.model_desc)
        self.cd = cd
        
    def train(self):
        mp.spawn(Trainer.init_trainer, args=(self.cd, self.model_mng, self.data_mng, self.idx_mng), nprocs=len(self.cd.world), join=True)