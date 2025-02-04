import shutil
from pathlib import Path

import torch

class FileManager:
    def __init__(self, output_dir: str = None, ckpt: str = None):
        output_dir: Path = Path(output_dir) if output_dir else Path('.')
        assert output_dir.exists(), f"output directory '{output_dir.__str__}' is not exist"

        self.taskdir = output_dir / get_default_taskdir_name(output_dir)
        if self.taskdir.exists():
            shutil.rmtree(self.taskdir)
        self.taskdir.mkdir()

        self.console_logger_path = self.taskdir / 'console.log'
        self.metrics_logger_path = self.taskdir / 'metrics.csv'
        self.weights_dir = self.taskdir / 'weights'
        self.weights_dir.mkdir()
        self.best = self.weights_dir / 'best.pt'
        self.last = self.weights_dir / 'last.pt'

        self.ckpt: dict = torch.load(ckpt)


def get_default_taskdir_name(output_dir: Path):
    default_name = "task"
    default_idx = 0
    dirs = [f.name for f in output_dir.iterdir() if f.is_dir()]
    while f'{default_name}_{default_idx}' in dirs:
        default_idx += 1
    return f'{default_name}_{default_idx}'