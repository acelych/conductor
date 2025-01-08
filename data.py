import yaml
from pathlib import Path

import cv2
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader


class ClassifyDataset(Dataset):

    def __init__(self, data_path: str, mode: str):
        super().__init__()
        self.data_path = data_path

        if mode == 'parquet':
            self.data = pd.read_parquet(self.data_path)
            self.__getitem__ = self._getitem_parquet
            self.__len__ = self._len_of_all
    
    def _len_of_all(self):
        return len(self.data)

    def _getitem_parquet(self, idx):
        tar = self.data.loc[idx]
        img, label = np.frombuffer(tar['img']['bytes'], dtype=np.uint8), tar['fine_label']
        return cv2.imdecode(img, cv2.IMREAD_COLOR), label


class UnifiedDataLoader:
    def __init__(self, yaml_path: str, task: str):
        pass
            
    def parse_yaml(self, yaml_path: str, task: str):
        
        with open(yaml_path, 'r') as f:
            data_desc: dict = yaml.safe_load(f)
        assert data_desc.get('task') == task, f"task of dataset '{yaml_path}' ({data_desc.get('task')}) does not match required task ({task})"
        assert data_desc.get("test") or data_desc.get("val"), f"expected to have at least one of test and val in dataset, got none"
            
        dataset_path = Path(data_desc.get('path'))
        if not dataset_path.is_absolute():
            dataset_path = yaml_path / dataset_path

        if not data_desc.get("test"):
            data_desc["test"] = data_desc.get("val")
        elif not data_desc.get("val"):
            data_desc["val"] = data_desc.get("test")

        train_path: Path = dataset_path / data_desc.get("train")
        test_path: Path = dataset_path / data_desc.get("test")
        val_path: Path = dataset_path / data_desc.get("val")
            
        if task == "classify":
            if train_path.is_file() and train_path.suffix == ".parquet":
                mode = "parquet"

            self.train_data = ClassifyDataset(train_path, mode)
            self.test_data = ClassifyDataset(test_path, mode)
            self.val_data = ClassifyDataset(val_path, mode) if test_path != val_path else self.test_data
        else:
            pass  #TODO