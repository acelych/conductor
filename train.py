import random
from typing import Tuple, Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn, optim, Tensor
from torch.types import Number
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from .model import ModelManager, Model
from .data import DataLoaderManager
from .utils import ConfigManager, ArtifactManager, MetricsManager, Calculate, Recorder, LogInterface, LR_Scheduler, get_model_assessment
from .test import Tester
from .modules._utils import BaseModule, TensorCollector
from .modules.nas import TauScheduler

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

        if cm.isddp():
            torch.cuda.set_device(cm.world[rank])
            dist.init_process_group(
                "nccl" if dist.is_nccl_available() else "gloo", 
                rank=rank, 
                world_size=len(cm.world)
            )
            self.active_rank.add(cm.world[0])
            self.device = torch.cuda.device(rank)
        else:
            torch.cuda.set_device(cm.world[0])
            self.device = torch.cuda.current_device()

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

        train_dataloader = self.data_mng.get_dataloader("train", rank=self.rank, shuffle=True)
        val_dataloader = self.data_mng.get_dataloader("val", rank=self.rank, shuffle=False)
        test_dataleader = self.data_mng.get_dataloader("test", rank=self.rank, shuffle=False)
        
        model = self.model_mng.build_model()
        if self.cm.isddp():
            self.model = DDP(model, device_ids=[torch.cuda.current_device()])
        else:
            self.model = model.to(self.cm.device)
            
        self.model_params = []
        self.arch_params = []
        self.tau_params = []
        for name, param in self.model.named_parameters():
            if 'nas_alpha' in name:
                self.arch_params.append(param)
            elif 'nas_tau' in name:
                self.tau_params.append(param)
            else:
                self.model_params.append(param)

        self.criterion: nn.Module = self.cm.criterion()
        self.optimizer: optim.Optimizer = self.cm.build_optimizer(self.model_params, self.cm.optimizer, self.cm.learn_rate, self.cm.decay)
        self.scheduler: LR_Scheduler = self.cm.build_scheduler(self.optimizer, len(train_dataloader))

        self.recorder: Recorder = Recorder(self.model_mng.model_desc.get("nc"))
        self.best_fitness = None
        
        if self.cm.nas:
            self.nas_optimizer: optim.Optimizer = self.cm.build_optimizer(
                self.arch_params,
                self.cm.nas.get('optimizer'), 
                self.cm.nas.get('learn_rate'),
                self.cm.nas.get('decay'))
            self.nas_tau_scheduler: TauScheduler = TauScheduler(
                self.tau_params, 
                self.cm.epochs, 
                self.cm.nas.get("max_tau"),
                self.cm.nas.get("min_tau"),
                self.cm.nas.get("annealing"))
            nas_dataloader: DataLoader = self.data_mng.get_dataloader("val", rank=self.rank, shuffle=True)
        
        if self.isactive():
            self.log.info(self.model.info(), fn=True)
            model_assessment, _ = get_model_assessment(self.model, self.cm.imgsz, lambda:self.model_mng.build_model().to(self.cm.device))
            self.log.info(model_assessment, fn=True)

        self.met_mng.start()

        for self.epoch in range(self.load_state(self.am.ckpt), self.cm.epochs):
            metrics = self.met_mng.get_metrics_holder(self.cm.task, self.epoch)
            
            # learn & val
            if self.cm.nas:
                tau = self.nas_tau_scheduler.step(self.epoch)
                self.nas_epoch(train_dataloader, nas_dataloader, metrics, tau)
            else:
                self.train_epoch(train_dataloader, metrics)
            val_loss: Number = self.val_epoch(val_dataloader, metrics)
            
            if not self.isactive():  # following codes should be ran only once a cycle
                continue

            # scheduler & fitness
            best_fitness, rule = self.get_best_fitness(metrics)
            self.step_scheduler(val_loss, metrics)
            self.step_fitness(best_fitness, rule)
            
            # metrics & save
            metrics = self.met_mng(metrics)  # add time stick & save to met_mng
            self.log.metrics(vars(metrics))  # write to metrics.csv
            self.save_state(best_fitness)  # save last & probably best

        if self.isactive():
            # test
            best_epoch = self.load_state(torch.load(self.am.best)) - 1
            metrics = self.met_mng.get_metrics_holder(self.cm.task, best_epoch)
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
            # nas state
            if self.cm.nas:
                self.nas_state()
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
            # TensorCollector.enable()
            for x, label in dataloader:
                x, label = self.move_batch(x, label)
                output = self.model(x)
                loss: Tensor = self.criterion(output, label)
                # if loss.isnan().any() or loss.item() > 10:
                #     batch = 0
                #     for idx, row in enumerate(output):
                #         if row.isnan().any():
                #             batch = idx
                #             break
                #     self.am.vis_matrices(TensorCollector.get("expand"), "expand", batch)
                #     self.am.vis_matrices(TensorCollector.get("input"), "input", batch)
                #     self.am.vis_matrices(TensorCollector.get("norm"), "norm", batch)
                #     self.am.vis_matrices(TensorCollector.get("expand_nan"), "expand_nan", batch)
                #     self.am.vis_matrices(TensorCollector.get("input_nan"), "input_nan", batch)
                #     self.am.vis_matrices(TensorCollector.get("norm_nan"), "norm_nan", batch)
                #     self.am.vis_matrices(TensorCollector.get("HR_nan_in"), "HR_nan_in", batch)
                #     self.am.vis_matrices(TensorCollector.get("HR_nan_out"), "HR_nan_out", batch)
                # TensorCollector.clear()
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
    
    def nas_epoch(self, dataloader: DataLoader, nas_dataloader: DataLoader, metrics: MetricsManager.Metrics, tau: float):
        losses = []

        # let one of the devices handle process bar
        if self.isactive():
            self.log.info(f"[nas training] tau: {tau}", fn=True)
            self.log.bar_init(len(dataloader), self.get_bar_desc('train'))

        for x, label in dataloader:
            # model params
            self.model.train()
            x, label = self.move_batch(x, label)
            output = self.model(x)
            self.optimizer.zero_grad()
            loss: Tensor = self.criterion(output, label)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            
            # arch params
            self.model.eval()
            x, label = next(iter(nas_dataloader))
            x, label = self.move_batch(x, label)
            output = self.model(x)
            self.nas_optimizer.zero_grad()
            loss: Tensor = self.criterion(output, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.arch_params, max_norm=1.0)  # gradient clipping
            self.nas_optimizer.step()
            
            if self.isactive():
                self.log.bar_update()
                
        if self.isactive():
            self.log.bar_close()
        mean_loss = Recorder.converge_loss(losses)
        
        # let one of the devices fill the blanks
        if self.isactive():
            metrics.record_train(mean_loss)
    
    def step_scheduler(self, val_loss: Number, met: MetricsManager.Metrics):
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning)

        if self.cm.scheduler is optim.lr_scheduler.ReduceLROnPlateau and not (self.epoch < self.cm.get_warmup_epochs()):
            self.scheduler.step(metrics=val_loss)
        else:
            self.scheduler.step()
        met.learn_rate = self.scheduler.get_last_lr()[0]

    def step_fitness(self, best_fitness: Number, rule: Callable):
        if self.best_fitness is None:
            self.best_fitness = best_fitness
        self.best_fitness = rule(self.best_fitness, best_fitness)

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
        
    def get_best_fitness(self, metrics: MetricsManager.Metrics) -> Tuple[Number, Callable]:
        assert self.cm.best_metrics in metrics.__dict__, f"unexpect metrics option '{self.cm.best_metrics}'"
        rule = min if self.cm.best_metrics.endswith('loss') else max
        return metrics.__dict__.get(self.cm.best_metrics), rule
    
    def get_bar_desc(self, stage: str, show_epoch = True, etc = None) -> str:
        result = f"({stage})"
        if show_epoch:
            result += f" {self.epoch}/{self.cm.epochs}"
        if etc is not None:
            result += f" {etc}"
        return result
    
    def isactive(self) -> bool:
        return self.rank in self.active_rank
    
    def nas_state(self):
        import yaml
        assert isinstance(self.model, Model)
        yaml_ls = ['- ' + str(layer.get_yaml_obj()) for layer in self.model.layers if isinstance(layer, BaseModule)]
        self.log.info("...nas result...", fn=True)
        self.log.info(yaml_ls)
            
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