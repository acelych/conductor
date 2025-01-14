import datetime
from pathlib import Path
from typing import Union

import pandas as pd
from torch import cuda, nn, optim
from tqdm import tqdm

## ========== CONSOLE OUTPUTS FORMAT ========== ##

class Logger:
    def __init__(self, output_dir: Union[str, Path], indexes_heads: dict, task_name: str = None):
        output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir
        assert output_dir.exists(), f"output directory '{output_dir.__str__}' is not exist"

        if task_name is None:
            task_name = get_default_task_name(output_dir)

        self.runs_dir = output_dir / task_name
        self.runs_dir.mkdir()
        self.console_logger_path = self.runs_dir / 'console.log'
        self.indexes_logger_path = self.runs_dir / 'indexes.csv'

        self.info(f"Conductor --- {datetime.datetime.now()}\n")  # init console logger
        pd.DataFrame({k: [] for k in indexes_heads.keys()}).to_csv(self.indexes_logger_path, index=False)  # init indexes logger

    def info(self, content: str):
        if isinstance(content, str):
            print(content)
            with open(self.console_logger_path, 'a') as f:
                f.write(content + '\n')
                
        elif isinstance(content, list):                
            (self.info(row) for row in content)
                

    def index(self, content: dict):
        pd.DataFrame(content).to_csv(self.indexes_logger_path, mode='a', header=False, index=False)
        
    def info_nn_struct(self, content: pd.DataFrame):
        content.to_dict


def get_default_task_name(output_dir: Path):
    default_name = "task"
    default_idx = 0
    dirs = [f.name for f in output_dir.iterdir() if f.is_dir()]
    while f'{default_name}_{default_idx}' in dirs:
        default_idx += 1
    return f'{default_name}_{default_idx}'

## ========== CONSOLE INPUTS FORMAT ========== ##

class CommandDetails:
    def __init__(
        self,
        model_yaml_path: str,
        data_yaml_path: str,
        command: str,  # train, predict
        task: str,
        device = "cuda",  # cpu, cuda, [...]
        world = None,
        criterion: nn.Module = nn.CrossEntropyLoss,
        optimizer: optim.Optimizer = optim.AdamW,
        learn_rate: int = 0.001,
        momentum: float = 0.9,
        decay: float = 1e-5,
        batch_size: int = 16,
        epochs: int = 300,
        ):
        self.model_yaml_path: str = model_yaml_path
        self.data_yaml_path: str = data_yaml_path
        self.command = command
        self.task = task

        device_count = cuda.device_count()
        assert device == "cuda" or device == "cpu", f"expect 'device' to be 'cuda' or 'cpu', got {device}"
        if device == "cuda" and device_count == 0:
            # TODO: warning, could not find any cuda device
            self.device = "cpu"
        else:
            self.device = device

        if isinstance(world, int):
            assert world <= device_count, f"expect using {device_count} gpu device(s) at most, got {world}"
            self.world = list(range(0, world))
        elif isinstance(world, list):
            assert all(n < device_count for n in world), f"expect device index less than {device_count}, got {world}"
        elif device == "cuda" and world is None:
            world = [0]
        self.world: list = None if device == "cpu" else world

        self.criterion = criterion
        self.optimizer: optim.Optimizer = optimizer
        self.learn_rate = learn_rate
        self.batch_size = batch_size
        self.momentum = momentum
        self.decay = decay
        self.epochs = epochs
        
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


def is_using_ddp(cd: CommandDetails):
    return cd.world is not None and len(cd.world) > 1
