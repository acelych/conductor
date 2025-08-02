import yaml
from typing import Union
from pathlib import Path

import torch

from .model import ModelManager
from .data import DataLoaderManager
from .train import TrainerManager
from .test import Tester
from .profiler import Profiler
from .utils import ConfigManager, ArtifactManager, LogInterface, MetricsManager


class Conductor:
    def __init__(self, cfg_path: Union[str, Path]):
        if isinstance(cfg_path, str):
            cfg_path = Path(cfg_path)
        with open(cfg_path, 'r') as f:
            cfg: dict = yaml.safe_load(f)

        self.cm, cm_log = ConfigManager.get_instance(**cfg)
        self.cm: ConfigManager
        self.am = ArtifactManager(self.cm)

        self.log = LogInterface(self.am)
        self.log.info(cm_log, fn=True)
        self.log.info(self.cm.info(), fn=True, bn=True)

    def run(self):

        if self.cm.command == 'train':
            orch = TrainerManager(self.cm, self.am, self.log)
            orch.train()
        elif self.cm.command == 'test':
            orch = Tester(self.cm, self.am, self.log)
            orch.test()
        elif self.cm.command == 'profile':
            orch = Profiler(self.cm, self.am, self.log)
            orch.profile()
        else:
            self.log.info(f"Unknown command: {self.cm.command}, exiting...")
            pass
        

        