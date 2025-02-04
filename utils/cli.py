import yaml
import shutil
import datetime
from pathlib import Path
from typing import Union

import pandas as pd
from torch import cuda, nn, optim
from tqdm import tqdm

from .resources import ResourceManager
from .misc import LR_Scheduler, get_module_class_str

## ========== CONSOLE OUTPUTS FORMAT ========== ##

class Logger:
    def __init__(self, output_dir: Union[str, Path], taskdir_name: str = None):
        output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir
        assert output_dir.exists(), f"output directory '{output_dir.__str__}' is not exist"

        if taskdir_name is None:
            taskdir_name = get_default_taskdir_name(output_dir)

        self.taskdir = output_dir / taskdir_name
        if self.taskdir.exists():
            shutil.rmtree(self.taskdir)
        self.taskdir.mkdir()

        self.console_logger_path = self.taskdir / 'console.log'
        self.metrics_logger_path = self.taskdir / 'metrics.csv'
        self.weights_dir = self.taskdir / 'weights'
        self.weights_dir.mkdir()
        self.best = self.weights_dir / 'best.pt'
        self.last = self.weights_dir / 'last.pt'
        self.save_fitness = None

        self.info(f"Conductor --- {datetime.datetime.now()}\n")  # init console logger

    def init_metrics(self, metrics_heads: tuple):
        pd.DataFrame({k: [] for k in metrics_heads}).to_csv(self.metrics_logger_path, index=False)  # init indexes logger

    def info(self, content: str):
        if isinstance(content, str):
            print(content)
            with open(self.console_logger_path, 'a') as f:
                f.write(content + '\n')
                
        elif isinstance(content, list):                
            (self.info(row) for row in content)

    def metrics(self, content: dict):
        pd.DataFrame(content).to_csv(self.metrics_logger_path, mode='a', header=False, index=False)


def get_default_taskdir_name(output_dir: Path):
    default_name = "task"
    default_idx = 0
    dirs = [f.name for f in output_dir.iterdir() if f.is_dir()]
    while f'{default_name}_{default_idx}' in dirs:
        default_idx += 1
    return f'{default_name}_{default_idx}'

## ========== CONSOLE INPUTS FORMAT ========== ##

class InstructDetails:
    def __init__(
        self,
        model_yaml_path: str,
        data_yaml_path: str,
        command: str,  # train, predict
        task: str,
        device: str,  # cpu, cuda
        world: list = None,  # giving [...] of devices of using cuda
        criterion: nn.Module = nn.CrossEntropyLoss,
        optimizer: optim.Optimizer = optim.AdamW,
        scheduler: LR_Scheduler = optim.lr_scheduler.ReduceLROnPlateau,
        learn_rate: float = 0.001,
        momentum: float = 0.9,
        decay: float = 1e-5,
        batch_size: int = None,
        epochs: int = 300,
        **kwargs,  # accept unexpected args
        ):
        self.model_yaml_path: str = model_yaml_path
        self.data_yaml_path: str = data_yaml_path
        self.command: str = command
        self.task: str = task
        self.device: str = device
        self.world: list = world
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.learn_rate = float(learn_rate)
        self.momentum = float(momentum)
        self.decay = float(decay)
        self.batch_size = batch_size if batch_size else (len(world) * 16 if world else 16)
        self.epochs = int(epochs)

    def logging(self, logger: Logger):
        info = "### Task Arguments ###\n"
        for k, v in vars(self).items():
            if v.__class__ not in [str, list, float, int]:
                v = get_module_class_str(v)
            info += f"{k}: {v}\n"
        logger.info(info)
        
    def build_optimizer(self, model: nn.Module):
        if self.optimizer in [optim.Adam, optim.AdamW, optim.Adamax, optim.NAdam, optim.RAdam]:
            opt_instance = self.optimizer(model.parameters(), lr=self.learn_rate, betas=(self.momentum, 0.999), weight_decay=self.decay)
        elif self.optimizer == optim.RMSprop:
            opt_instance = optim.RMSprop(model.parameters(), lr=self.learn_rate, momentum=self.momentum, weight_decay=self.decay)
        elif self.optimizer == optim.SGD:
            opt_instance = optim.SGD(model.parameters(), lr=self.learn_rate, momentum=self.momentum, weight_decay=self.decay, nesterov=True)
        else:
            raise NotImplementedError(f"unexpected optimizer '{self.optimizer.__name__}'")
        return opt_instance
    
    def build_scheduler(self, opt: optim.Optimizer, steps_per_epoch: int, **kwargs):
        '''
        conductor supports lr schedulers as follows:
        1. Cosine Annealing (cosine)
            gradually reduce lr rate by following a cosine curve.
            universal, smooth, especially for long term training.
        2. Step Decay (step)
            reduce lr rate by a fixed percentage every fixed training steps
            easy to use, for smoothly converging & simple task
        3. One-Cycle (cycle)
            start from a minor lr, guadually increase to a major lr, then swiftly decrease back in cycle
            faster converging for giant model-dataset, avoiding local optimal solution
        4. ReduceLROnPlateau (reduce)
            adjust lr according to val loss, reduce lr when metrics remains stagnant
            universal, adaptive, avoiding premature lr decline
        '''
        if self.scheduler in [optim.lr_scheduler.ReduceLROnPlateau]:
            sch_instance = self.scheduler(optimizer=opt, **kwargs)
        elif self.scheduler is optim.lr_scheduler.StepLR:
            sch_instance = self.scheduler(optimizer=opt, step_size=int(self.epochs * 0.1), **kwargs)
        elif self.scheduler is optim.lr_scheduler.CosineAnnealingLR:
            sch_instance = self.scheduler(optimizer=opt, T_max=self.epochs, eta_min=1e-6, **kwargs)
        elif self.scheduler is optim.lr_scheduler.OneCycleLR:
            sch_instance = self.scheduler(optimizer=opt, max_lr=0.01, steps_per_epoch=steps_per_epoch, total_steps=self.epochs, **kwargs)
        else:
            raise AssertionError(f"unexpected scheduler '{self.scheduler.__name__}'")

        return sch_instance
    
    @classmethod
    def get_instance(cls, logger: Logger, model_yaml_path: str, data_yaml_path: str, command: str, **kwargs):
        import ast
        
        # task
        task = kwargs.get('task')
        if not task:
            with open(model_yaml_path, 'r') as f:
                model_desc: dict = yaml.safe_load(f)
                assert 'task' in model_desc, f"expect 'task' key in model yaml or given kwargs"
                task = model_desc.get('task')
        assert task in ResourceManager.get_legal_tasks(), f"unexpect task '{task}'"
        kwargs['task'] = task
                
        # device
        device = kwargs.get('device')
        device_count = cuda.device_count()
        if device:  # defined by user
            assert device in ['cuda', 'cpu'], f"expect 'device' to be 'cuda' or 'cpu', got {device}"
            if device == 'cuda' and device_count == 0:
                logger.info("changing device to 'cpu' since cuda is not available")
                device = 'cpu'
        else:
            device = 'cuda' if device_count > 0 else 'cpu'
        kwargs['device'] = device
        
        # world
        world: str = kwargs.get('world')
        if world:
            if device != 'cuda':
                logger.info("changing world to None since cuda is not available")
                world = None
            else:
                try:
                    world = list(ast.literal_eval(world))
                except Exception as e:
                    raise AssertionError(f"expect world to be a parseable string, got '{world}'") from e
                assert len(world) <= device_count, f"expect using {device_count} gpu device(s) at most, got {world}"
        else:
            world = [0] if device == 'cuda' else None
        kwargs['world'] = world
                
        # criterion
        criterion: str = kwargs.get('criterion')
        if criterion:
            criterion_cls = getattr(nn, criterion.lstrip('nn.'), None)
            assert criterion_cls, f"unexpected criterion '{criterion}'"
        kwargs['criterion'] = criterion_cls
        
        # optimizer
        optimizer: str = kwargs.get('optimizer')
        if optimizer:
            optimizer_cls = getattr(optim, optimizer.lstrip('optim.'), None)
            assert optimizer_cls, f"unexpected optimizer '{optimizer}'"
        kwargs['optimizer'] = optimizer_cls
        
        # scheduler
        scheduler: str = kwargs.get('scheduler')
        if scheduler:
            scheduler_cls = getattr(optim.lr_scheduler, scheduler.lstrip('optim.').lstrip('lr_scheduler.'), None)
            assert scheduler_cls, f"unexpected scheduler '{scheduler}'"
        kwargs['scheduler'] = scheduler_cls

        # batch_size
        batch_size = kwargs.get('batch_size')
        if batch_size:
            assert kwargs.get('batch_size') % len(world) == 0, f"expect batch size a multiple of amount of devices."
        
        return cls(model_yaml_path, data_yaml_path, command, **kwargs)
            

def is_using_ddp(id: InstructDetails):
    return id.world is not None and len(id.world) > 1
