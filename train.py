import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from .utils.cli import CommandDetails, LR_Scheduler

from .models import ModelManager
from .data import DataLoaderManager
from .utils import MetricsManager, Recorder

class Trainer:
    def __init__(self, rank: int, cd: CommandDetails, model_mng: ModelManager, data_mng: DataLoaderManager, met_mng: MetricsManager):
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
        self.met_mng = met_mng
        
    def train(self):
        train_dataloader = self.data_mng.get_dataloader("train", rank=self.rank)
        val_dataloader = self.data_mng.get_dataloader("val", rank=self.rank)
        
        model = self.model_mng.build_model()
        ddp_model = DDP(model, device_ids=[torch.cuda.current_device()])
        self.criterion: nn.Module = self.cd.criterion()
        self.optimizer: optim.Optimizer = self.cd.build_optimizer(ddp_model.parameters())
        self.scheduler: LR_Scheduler = self.cd.build_scheduler(self.optimizer, len(train_dataloader))
        
        self.recorder: Recorder = Recorder(self.model_mng.model_desc.get("nc"))
        self.met_mng.start()
        
        for epoch in range(self.cd.epochs):
            metrics = self.met_mng.get_index(epoch)
            
            # learn & val
            self.train_epoch(ddp_model, train_dataloader, metrics)
            val_loss = self.val_epoch(ddp_model, val_dataloader, metrics)
            
            # scheduler
            self.step_scheduler(val_loss)
            
            # done
            self.met_mng(metrics)
    
    def train_epoch(self, model: DDP, dataloader: DataLoader, metrics: MetricsManager.Metrics):
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
            metrics.record_train(mean_loss)

    def val_epoch(self, model: nn.Module, dataloader: DataLoader, metrics: MetricsManager.Metrics):
        model.eval()
        self.recorder.clear()
        for x, label in dataloader:
            output = model(x)
            loss = self.criterion(output, label)
            self.recorder(output, label, loss)
        self.recorder.converge()
        
        # let one of the gpu fill the blanks
        if dist.get_rank() == 0:
            metrics.record_val(self.recorder)
            
        return self.recorder.get_mean_loss()
    
    def step_scheduler(self, val_loss: Tensor):
        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(metrics=val_loss)
        else:
            self.scheduler.step()
            
    @staticmethod
    def init_trainer(rank, cd: CommandDetails, model_mng: ModelManager, data_mng: DataLoaderManager, idx_mng: MetricsManager):
        trainer = Trainer(rank, cd, model_mng, data_mng, idx_mng)
        trainer.train()

    
class TrainerManager:
    def __init__(self, cd: CommandDetails):
        self.model_mng = ModelManager(cd.model_yaml_path, cd.task)
        self.data_mng = DataLoaderManager(cd.data_yaml_path, cd)
        self.met_mng = MetricsManager(cd, self.model_mng.model_desc)
        self.cd = cd
        
    def train(self):
        mp.spawn(Trainer.init_trainer, args=(self.cd, self.model_mng, self.data_mng, self.met_mng), nprocs=len(self.cd.world), join=True)