import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from .utils.misc import LR_Scheduler
from .utils.cli import CommandDetails

from .models import ModelManager
from .data import DataLoaderManager
from .utils import MetricsManager, Recorder, Logger

class Trainer:
    def __init__(self, 
                 rank: int, 
                 cd: CommandDetails, 
                 model_mng: ModelManager, 
                 data_mng: DataLoaderManager, 
                 met_mng: MetricsManager, 
                 logger: Logger,
                 ckpt: dict = None):
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
        self.logger = logger
        self.ckpt = ckpt
        
    def train(self):
        train_dataloader = self.data_mng.get_dataloader("train", rank=self.rank)
        val_dataloader = self.data_mng.get_dataloader("val", rank=self.rank)
        
        model = self.model_mng.build_model()
        self.model = DDP(model, device_ids=[torch.cuda.current_device()])
        self.criterion: nn.Module = self.cd.criterion()
        self.optimizer: optim.Optimizer = self.cd.build_optimizer(self.model.parameters())
        self.scheduler: LR_Scheduler = self.cd.build_scheduler(self.optimizer, len(train_dataloader))

        self.recorder: Recorder = Recorder(self.model_mng.model_desc.get("nc"))
        self.best_fitness = None
        self.met_mng.start()
        
        for self.epoch in range(self.load_state(), self.cd.epochs):
            metrics = self.met_mng.get_metrics_holder(self.epoch)
            
            # learn & val
            self.train_epoch(self.model, train_dataloader, metrics)
            val_loss: Tensor = self.val_epoch(self.model, val_dataloader, metrics)
            
            # scheduler & fitness
            self.step_scheduler(val_loss)
            self.best_fitness(val_loss)
            
            # metrics & save
            self.met_mng(metrics)
            self.save_state(val_loss.item())
    
    def train_epoch(self, dataloader: DataLoader, metrics: MetricsManager.Metrics):
        self.model.train()
        losses = []
        for x, label in dataloader:
            output = self.model(x)
            self.optimizer.zero_grad()
            loss: Tensor = self.criterion(output, label)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        mean_loss = Recorder.converge_loss(losses)
        
        # let one of the gpu fill the blanks
        if dist.get_rank() == 0:
            metrics.record_train(mean_loss)

    def val_epoch(self, dataloader: DataLoader, metrics: MetricsManager.Metrics):
        self.model.eval()
        self.recorder.clear()
        for x, label in dataloader:
            output = self.model(x)
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

    def step_fitness(self, val_loss: Tensor):
        if self.best_fitness is None:
            self.best_fitness = val_loss.item()
        self.best_fitness = min(self.best_fitness, val_loss.item())

    def save_state(self, curr_fitness):
        import io

        buffer = io.BytesIO()
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'opt_state_dict': self.optimizer.state_dict(),
                'lrs_state_dict': self.scheduler.state_dict(),
                'epoch': self.epoch,
            },
            buffer
        )
        serialized_ckpt = buffer.getvalue()

        self.logger.last.write_bytes(serialized_ckpt)
        if curr_fitness == self.best_fitness:
            self.logger.best.write_bytes(serialized_ckpt)

    def load_state(self) -> int:
        if self.ckpt:
            self.model.load_state_dict(self.ckpt.get('model_state_dict'))
            self.optimizer.load_state_dict(self.ckpt.get('opt_state_dict'))
            self.scheduler.load_state_dict(self.ckpt.get('lrs_state_dict'))
            return self.ckpt.get('epoch') + 1
        else:
            return 0
            
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