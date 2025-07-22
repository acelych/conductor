import yaml
import shutil
from pathlib import Path
from typing import Any, Dict, Tuple, List, Union, Sequence
from PIL import Image

import pandas as pd

import torch
from torch import nn, optim, cuda, Tensor

from .resources import ResourceManager
from .stat import MetricsManager
from .misc import LR_Scheduler, get_module_class_str, isbuiltin
from .plot import Plot

"""
ConfigManager is a utility class to manage configurations for training, testing, and profiling tasks.
"""
class ConfigManager:
    def __init__(
        self,
        model_yaml_path: str,
        data_yaml_path: str,
        command: str,  # train, test, profile
        task: str,
        device: str,  # cpu, cuda
        world: list = None,  # giving [...] of devices of using cuda
        criterion: nn.Module = nn.CrossEntropyLoss,
        optimizer: optim.Optimizer = optim.AdamW,
        scheduler: LR_Scheduler = None,
        learn_rate: float = 0.001,
        warmup_rate = 0.05,
        momentum: float = 0.9,
        decay: float = 1e-5,
        batch_size: int = None,
        epochs: int = 300,
        imgsz: Union[int, tuple] = (224, 224),
        best_metrics: str = "val_loss",
        nas: Dict[str, Any] = None,  # whether to use nas module
        **kwargs: Dict[str, Any],  # accept unexpected args
        ):
        self.model_yaml_path: Path = Path(model_yaml_path)
        self.data_yaml_path: Path = Path(data_yaml_path)
        self.command: str = command
        self.task: str = task
        self.device: str = device
        self.world: list = world
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.learn_rate = float(learn_rate)
        self.warmup_rate = float(warmup_rate)
        self.momentum = float(momentum)
        self.decay = float(decay)
        self.batch_size = batch_size if batch_size else (len(world) * 16 if world else 16)
        self.epochs = int(epochs)
        self.imgsz = imgsz
        self.best_metrics = best_metrics
        self.nas = nas  # whether to use nas module
        self.__dict__.update(kwargs)

    def info(self) -> list:
        info_list = ["###  CONFIG  ###"]
        for k, v in vars(self).items():
            if not isbuiltin(v) and not isinstance(v, Path):
                v = get_module_class_str(v)
            info_list.append(f"{k}: {v}")
        return info_list
        
    def build_optimizer(self, params, opt, lr) -> optim.Optimizer:
        if opt in [optim.Adam, optim.AdamW, optim.Adamax, optim.NAdam, optim.RAdam]:
            opt_instance = opt(params, lr=lr, betas=(self.momentum, 0.999), weight_decay=self.decay)
        elif opt == optim.RMSprop:
            opt_instance = optim.RMSprop(params, lr=lr, momentum=self.momentum, weight_decay=self.decay)
        elif opt == optim.SGD:
            opt_instance = optim.SGD(params, lr=lr, momentum=self.momentum, weight_decay=self.decay, nesterov=True)
        else:
            raise NotImplementedError(f"unexpected optimizer '{opt.__name__}'")
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
        elif self.scheduler is None:
            sch_instance = optim.lr_scheduler.LambdaLR(optimizer=opt, lr_lambda=lambda epoch:1)  # empty scheduler
        else:
            raise AssertionError(f"unexpected scheduler '{self.scheduler.__name__}'")
        
        warmup_epo = self.get_warmup_epochs()
        warmup_sch = optim.lr_scheduler.LambdaLR(optimizer=opt, lr_lambda=lambda epoch: (epoch + 1) / warmup_epo if epoch < warmup_epo else 1)
        combined_sch = optim.lr_scheduler.SequentialLR(
            optimizer=opt,
            schedulers=[warmup_sch, sch_instance],
            milestones=[warmup_epo]
        )

        return combined_sch
    
    @classmethod
    def get_instance(cls, **kwargs) -> Tuple[Any, list]:
        '''
        Get an instance of ConfigManager with validated and processed configurations.
        `get_instance` will validate the provided configurations from YAML, set default values where necessary,
        and return an instance of ConfigManager along with a list of info messages.

        Args:
            **kwargs: Keyword arguments containing configuration parameters. Expected keys include:
                - model_yaml_path (str): Path to the model YAML file.
                - data_yaml_path (str): Path to the data YAML file.
                - command (str): Command to execute ('train', 'test', 'profiler').
                - task (str): Task type (e.g., 'classification', 'detection').
                - device (str): Device to use ('cpu' or 'cuda').
                - world (list, optional): List of device indices for distributed training.
                - criterion (str, optional): Loss function class name.
                - optimizer (str, optional): Optimizer class name.
                - scheduler (str, optional): Learning rate scheduler class name.
                - learn_rate (float, optional): Learning rate for the optimizer.
                - warmup_rate (float, optional): Warmup rate for learning rate scheduling.
                - momentum (float, optional): Momentum for the optimizer.
                - decay (float, optional): Weight decay for the optimizer.
                - batch_size (int, optional): Batch size for training.
                - epochs (int, optional): Number of epochs for training.
                - imgsz (Union[int, tuple], optional): Input image size for the model.
                - best_metrics (str, optional): Metric to determine the best model.

        Returns:
            - ConfigManager, list: An instance of ConfigManager and a list of info messages. (since logging is not implemented yet)
        '''
        import ast

        model_yaml_path = kwargs.get('model_yaml_path')
        info = []

        # data_yaml_path
        data_yaml_path = kwargs.get('data_yaml_path')
        if not data_yaml_path:
            if kwargs.get('command') != 'profile':
                raise AssertionError("expect 'data_yaml_path' for training or testing")
            else:
                kwargs['data_yaml_path'] = ''
        
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
                info.append("changing device to 'cpu' since cuda is not available")
                device = 'cpu'
        else:
            device = 'cuda' if device_count > 0 else 'cpu'
        kwargs['device'] = device
        
        # world
        world: str = kwargs.get('world')
        if world:
            if device != 'cuda':
                info.append("changing world to None since cuda is not available")
                world = None
            else:
                try:
                    world = list(ast.literal_eval(world))
                except Exception as e:
                    raise AssertionError(f"expect world to be a parseable string, got '{world}'") from e
                assert len(world) <= device_count, f"expect using {device_count} gpu device(s) at most, got {world}"
                assert max(world) < device_count, f"expect device sequence less than {device_count}, got {max(world)}"
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
            assert batch_size % len(world) == 0, f"expect batch size a multiple of amount of devices."

        # imgsz
        imgsz = kwargs.get('imgsz')
        if imgsz:
            if isinstance(imgsz, Sequence):
                assert len(imgsz) == 2 and all((isinstance(it, int) for it in imgsz)), f"expect imgsz to be a 2 integers' sequence"
            kwargs['imgsz'] = tuple(imgsz)
            
        # nas
        nas = kwargs.get('nas')
        if nas:
            assert isinstance(nas, dict), f"expect nas to be a dict, got {type(nas)}"
            assert nas.get('optimizer') is not None, f"expect nas optimizer to be defined"
            assert nas.get('learn_rate') is not None, f"expect nas learn_rate to be defined"
            assert nas.get('min_tau') is not None, f"expect nas minimum of tau to be defined"
            assert nas.get('max_tau') is not None, f"expect nas maximum of tau to be defined"
            
            nas_optimizer_cls = getattr(optim, nas.get('optimizer').lstrip('optim.'), None)
            assert nas_optimizer_cls, f"unexpected optimizer '{nas.get('optimizer')}'"
            
            nas['optimizer'] = nas_optimizer_cls
            nas['learn_rate'] = float(nas.get('learn_rate'))
            nas['min_tau'] = float(nas.get('min_tau'))
            nas['max_tau'] = float(nas.get('max_tau'))
            
            assert nas['min_tau'] > 0, f"expect nas minimum of tau greater than zero"
            assert nas['max_tau'] > 0, f"expect nas maximum of tau greater than zero"
            assert nas['min_tau'] < nas['max_tau'], f"expect nas minimum of tau less than maximum"
            kwargs['nas'] = nas
        
        return cls(**kwargs), info
    
    def isddp(self) -> bool:
        return self.world is not None and len(self.world) > 1
    
    def get_warmup_epochs(self) -> int:
        return int(self.epochs * self.warmup_rate)
    

"""
ArtifactManager is used to manage the artifacts generated during the training or testing process.
"""
class ArtifactManager:
    def __init__(self, cm: ConfigManager):
        self.cm = cm
        output_dir = getattr(cm, 'output_dir')
        output_dir: Path = Path(output_dir) if output_dir else Path('.')
        assert output_dir.exists(), f"output directory '{output_dir.__str__}' is not exist"

        self.taskdir = output_dir / get_default_taskdir_name(output_dir)
        if self.taskdir.exists():
            shutil.rmtree(self.taskdir)
        self.taskdir.mkdir()

        # Path
        self.console_logger_path = self.taskdir / 'console.log'
        self.metrics_logger_path = self.taskdir / 'metrics.csv'
        self.onnx_model_path = self.taskdir / 'model.onnx'
        self.weights_dir = self.taskdir / 'weights'
        self.best = self.weights_dir / 'best.pt'
        self.last = self.weights_dir / 'last.pt'
        self.result = self.taskdir / 'result.png'
        self.profiler_dir = self.taskdir / 'profiler'

        # File
        ckpt = getattr(cm, 'ckpt')
        self.ckpt: dict = torch.load(ckpt) if ckpt else None  # load ckpt if available
        if cm.command == 'test' and self.ckpt is None:
            raise AssertionError(f"expect ckpt for testing")
        
        if cm.command == 'train':
            self.weights_dir.mkdir()  # prepare dir for trained weights
            metrics_heads = MetricsManager.get_metrics_holder(cm.task).get_heads()
            pd.DataFrame({k: [] for k in metrics_heads}).to_csv(self.metrics_logger_path, index=False)  # init indexes logger
        elif cm.command == 'profile':
            self.profiler_dir.mkdir()  # prepare dir for profiler results

    def info(self, content: str):
        with open(self.console_logger_path, 'a') as f:
            f.write(content + '\n')
            
    def metrics(self, content: dict):
        content = {k: [v] for k, v in content.items()}
        pd.DataFrame(content).to_csv(self.metrics_logger_path, mode='a', header=False, index=False)
        
    def plot_metrics(self, metrics_collcet: List[MetricsManager.Metrics]):
        Plot.plot_line_chart(metrics_collcet, self.result)
        
    def plot_samples(self, stage: str, samples: List[Image.Image], label: list, pred: list, category_name: str = None):
        filename = f"sample{'_' + category_name if isinstance(category_name, str) else ''}_{stage}.png"
        sample_path = self.taskdir / filename
        Plot.plot_classify_sampling(samples, label, pred, sample_path)
        
    def vis_matrices(self, mat: Tensor, info: str, batch: int = 0):
        if mat is None:
            return
        assert len(mat.shape) == 4, f"expect BCHW Tensor, got dimension {len(mat.shape)}"
        filename = f"visual_{info}_{batch}.png"
        vis_path = self.taskdir / filename
        mat_chw = mat[batch, ...]
        imgs = [Plot.vis_matrix(mat, with_anno=True) for mat in mat_chw]
        Plot.plot_imgs(vis_path, imgs)


def get_default_taskdir_name(output_dir: Path):
    default_name = "task"
    default_idx = 0
    dirs = [f.name for f in output_dir.iterdir() if f.is_dir()]
    while f'{default_name}_{default_idx}' in dirs:
        default_idx += 1
    return f'{default_name}_{default_idx}'