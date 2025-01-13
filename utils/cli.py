import datetime
from pathlib import Path
from typing import Union, List
from collections import namedtuple

import pandas as pd
from tqdm import tqdm

LayerInfo = namedtuple('LayerInfo', ['idx', 'former', 'n', 'params', 'module', 'arguments'])

def get_default_task_name(output_dir: Path):
    default_name = "task"
    default_idx = 0
    dirs = [f.name for f in output_dir.iterdir() if f.is_dir()]
    while f'{default_name}_{default_idx}' in dirs:
        default_idx += 1
    return f'{default_name}_{default_idx}'

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
                
        elif isinstance(content, List):
            self.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
            for item in content:
                assert isinstance(item, LayerInfo)
                self.info(
                    f"{item.idx:>3}{str(item.former):>20}"
                    f"{item.n:>3}{item.params:10.0f}  "
                    f"{item.module:<45}{str(item.arguments):<30}"
                )
                

    def index(self, content: dict):
        pd.DataFrame(content).to_csv(self.indexes_logger_path, mode='a', header=False, index=False)
        
    def info_nn_struct(self, content: pd.DataFrame):
        content.to_dict
