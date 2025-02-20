import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn, optim, Tensor
from torch.types import Number
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from .model import ModelManager
from .data import DataLoaderManager
from .utils import ConfigManager, ArtifactManager, MetricsManager, Calculate, Recorder, LogInterface, LR_Scheduler, get_model_assessment
from .test import Tester

class Trainer(Tester):
    def __init__(self, 
                 rank: int, 
                 cm: ConfigManager, 
                 am: ArtifactManager, 
                 log: LogInterface, 
                 model_mng: ModelManager, 
                 data_mng: DataLoaderManager, 
                 met_mng: MetricsManager, ):
        
        self.active_rank = {-1, }
        self.device = torch.device(cm.device)

        if cm.isddp():
            torch.cuda.set_device(cm.world[rank])
            dist.init_process_group(
                "nccl" if dist.is_nccl_available() else "gloo", 
                rank=rank, 
                world_size=len(cm.world)
            )
            self.active_rank.add(cm.world[0])
            self.device = torch.cuda.device(rank)

        self.rank = rank
        self.cm = cm
        self.am = am
        self.log = log
        self.model_mng = model_mng
        self.data_mng = data_mng
        self.met_mng = met_mng

        
    def train(self):
        if self.isactive():
            self.log.info("initializing training")

        train_dataloader = self.data_mng.get_dataloader("train", rank=self.rank)
        val_dataloader = self.data_mng.get_dataloader("val", rank=self.rank)
        test_dataleader = self.data_mng.get_dataloader("test", rank=self.rank)
        
        model = self.model_mng.build_model()
        if self.cm.isddp():
            self.model = DDP(model, device_ids=[torch.cuda.current_device()])
        else:
            self.model = model.to(self.cm.device)

        self.criterion: nn.Module = self.cm.criterion()
        self.optimizer: optim.Optimizer = self.cm.build_optimizer(self.model)
        self.scheduler: LR_Scheduler = self.cm.build_scheduler(self.optimizer, len(train_dataloader))

        self.recorder: Recorder = Recorder(self.model_mng.model_desc.get("nc"))
        self.best_fitness = None
        
        if self.isactive():
            self.log.info(self.model.info(), fn=True)
            model_assessment, _ = get_model_assessment(self.model, self.cm.imgsz, lambda:self.model_mng.build_model().to(self.cm.device))
            self.log.info(model_assessment, fn=True)

        self.met_mng.start()

        for self.epoch in range(self.load_state(self.am.ckpt), self.cm.epochs):
            metrics = self.met_mng.get_metrics_holder(self.cm.task, self.epoch)
            
            # learn & val
            self.train_epoch(train_dataloader, metrics)
            val_loss: Number = self.val_epoch(val_dataloader, metrics)
            
            if not self.isactive():  # following codes should be ran only once a cycle
                continue

            # scheduler & fitness
            self.step_scheduler(val_loss, metrics)
            self.step_fitness(val_loss)
            
            # metrics & save
            metrics = self.met_mng(metrics)  # add time stick & save to met_mng
            self.log.metrics(vars(metrics))  # write to metrics.csv
            self.save_state(val_loss)  # save last & probably best

        if self.isactive():
            # test
            metrics = self.met_mng.get_metrics_holder(self.cm.task, self.epoch)
            best_epoch = self.load_state(torch.load(self.am.best)) - 1
            test_report, precision, _ = self.test_epoch(test_dataleader, metrics, best_epoch)
            metrics.dummy_fill()
            self.log.metrics(vars(metrics), save=False)
            self.log.info(test_report)
            # latency
            self.latency()
            # sampling
            self.sampling("train")
            self.sampling("test")
            # focusing
            worst_category = precision.argmin()
            self.focusing("train", worst_category.item())
            self.focusing("test", worst_category.item())
            self.am.plot_metrics(self.met_mng.metrics_collect)
            # save caches
            self.data_mng.save_caches()
    
    def train_epoch(self, dataloader: DataLoader, metrics: MetricsManager.Metrics):
        self.model.train()
        losses = []

        # let one of the devices handle process bar
        if self.isactive():
            self.log.bar_init(len(dataloader), self.get_bar_desc('train'), fn=True)

        for x, label in dataloader:
            x, label = self.move_batch(x, label)
            output = self.model(x)
            self.optimizer.zero_grad()
            loss: Tensor = self.criterion(output, label)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            if self.isactive():
                self.log.bar_update()
        if self.isactive():
            self.log.bar_close()
        mean_loss = Recorder.converge_loss(losses)
        
        # let one of the devices fill the blanks
        if self.isactive():
            metrics.record_train(mean_loss)

    def val_epoch(self, dataloader: DataLoader, metrics: MetricsManager.Metrics) -> Number:
        self.model.eval()
        self.recorder.clear()

        # let one of the devices handle process bar
        if self.isactive():
            self.log.bar_init(len(dataloader), self.get_bar_desc('val'))

        with torch.no_grad():
            for x, label in dataloader:
                x, label = self.move_batch(x, label)
                output = self.model(x)
                loss: Tensor = self.criterion(output, label)
                self.recorder(output, label, loss)
                if self.isactive():
                    self.log.bar_update()
            if self.isactive():
                self.log.bar_close()
            self.recorder.converge()
        
        # let one of the devices fill the blanks
        if self.isactive():
            metrics.record_val(self.recorder)
            
        return self.recorder.get_mean_loss()
    
    def step_scheduler(self, val_loss: Number, met: MetricsManager.Metrics):
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning)

        if self.cm.scheduler is optim.lr_scheduler.ReduceLROnPlateau and not (self.epoch < self.cm.get_warmup_epochs()):
            self.scheduler.step(metrics=val_loss)
        else:
            self.scheduler.step()
        met.learn_rate = self.scheduler.get_last_lr()[0]

    def step_fitness(self, val_loss: Number):
        if self.best_fitness is None:
            self.best_fitness = val_loss
        self.best_fitness = min(self.best_fitness, val_loss)

    def save_state(self, curr_fitness):
        import io

        buffer = io.BytesIO()
        torch.save(
            {
                'model_desc': self.model_mng.model_desc,
                'model_state_dict': self.model.state_dict(),
                'opt_state_dict': self.optimizer.state_dict(),
                'lrs_state_dict': self.scheduler.state_dict(),
                'epoch': self.epoch,
                'best_fitness': self.best_fitness,
            },
            buffer
        )
        serialized_ckpt = buffer.getvalue()

        self.am.last.write_bytes(serialized_ckpt)
        if curr_fitness == self.best_fitness:
            self.am.best.write_bytes(serialized_ckpt)

    def load_state(self, ckpt: dict) -> int:
        if ckpt:
            self.model.load_state_dict(ckpt.get('model_state_dict'))
            self.optimizer.load_state_dict(ckpt.get('opt_state_dict'))
            self.scheduler.load_state_dict(ckpt.get('lrs_state_dict'))
            self.best_fitness = ckpt.get('best_fitness')
            return ckpt.get('epoch') + 1
        else:
            return 0
    
    def get_bar_desc(self, stage: str, show_epoch = True, etc = None) -> str:
        result = f"({stage})"
        if show_epoch:
            result += f" {self.epoch}/{self.cm.epochs}"
        if etc is not None:
            result += f" {etc}"
        return result
    
    def isactive(self) -> bool:
        return self.rank in self.active_rank
            
    @staticmethod
    def init_trainer(rank, cm: ConfigManager, am: ArtifactManager, log: LogInterface, model_mng: ModelManager, data_mng: DataLoaderManager, met_mng: MetricsManager):
        trainer = Trainer(rank, cm, am, log, model_mng, data_mng, met_mng)
        trainer.train()

    
class TrainerManager:
    def __init__(self, cm: ConfigManager, am: ArtifactManager, log: LogInterface):
        self.cm = cm
        self.am = am
        self.log = log
        self.model_mng = ModelManager(self.cm)
        self.data_mng = DataLoaderManager(self.cm, self.log)
        self.met_mng = MetricsManager()
        
    def train(self):
        if self.cm.isddp():
            self.log.info(f"initializing distributed data parallel training, using gpu {self.cm.world}")
            mp.spawn(Trainer.init_trainer, args=(self.cm, self.am, self.log, self.model_mng, self.data_mng, self.met_mng), nprocs=len(self.cm.world), join=True)
        else:
            Trainer.init_trainer(-1, self.cm, self.am, self.log, self.model_mng, self.data_mng, self.met_mng)