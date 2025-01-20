import yaml
from typing import Union
from pathlib import Path

from .model import ModelManager
from .data import DataLoaderManager
from .utils import InstructDetails, Logger, MetricsManager

class Conductor:
    def __init__(self, instruct: Union[str, Path]):
        if isinstance(instruct, str):
            instruct = Path(instruct)
        with open(instruct.__str__(), 'r') as f:
            instruct_dict: dict = yaml.safe_load(f)

        self.logger = Logger(
            output_dir=instruct_dict.setdefault('output_dir', '.'), 
            taskdir_name=instruct_dict.get('taskdir_name')
        )
        self.id = InstructDetails.get_instance(
            logger=self.logger, 
            **instruct_dict
        )

    def run(self):
        self.model_mng = ModelManager(self.id.model_yaml_path, self.id.task)
        self.data_mng = DataLoaderManager(self.id.data_yaml_path, self.id)
        self.met_mng = MetricsManager(self.id, self.model_mng.model_desc)
        
        self.logger.start(metrics_heads=self.met_mng.get_metrics_holder().get_heads())

        