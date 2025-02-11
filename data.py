import yaml
from typing import Callable
from pathlib import Path

import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from .utils.back import ConfigManager


class ClassifyDataset(Dataset):
    def __init__(self, data_path: str, mode: str):
        super().__init__()
        self.data_path = data_path
        self.mode = mode

        if mode == 'parquet':
            self.data = pd.read_parquet(self.data_path)
        else:
            pass  #TODO
    
    def __len__(self):
        if self.mode == 'parquet':
            return self._len_of_all()
        
    def __getitem__(self, index):
        if self.mode == 'parquet':
            return self._getitem_parquet(index)

    def _len_of_all(self):
        return len(self.data)

    def _getitem_parquet(self, index):
        tar = self.data.loc[index]
        img, label = tar['img']['bytes'], tar['fine_label']
        img = np.frombuffer(img, dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        img = torch.Tensor(img).permute(2, 0, 1).float()
        return img, label


class DataLoaderManager:
    def __init__(self, cm: ConfigManager):
        self.parse_yaml(cm.data_yaml_path, cm.task)
        self.cm = cm
            
    def parse_yaml(self, yaml_path: str, task: str):
        
        with open(yaml_path, 'r') as f:
            data_desc: dict = yaml.safe_load(f)
        assert data_desc.get('task') == task, f"task of dataset '{yaml_path}' ({data_desc.get('task')}) does not match required task ({task})"
        assert data_desc.get("test") or data_desc.get("val"), f"expected to have at least one of test and val in dataset, got none"
            
        dataset_path = Path(data_desc.get('path'))
        if not dataset_path.is_absolute():
            dataset_path = Path(yaml_path).parent / dataset_path

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
            
        self.names = data_desc.get("names")
        
    def get_dataloader(self, stage: str, rank = None):
        using_ddp = self.cm.is_using_ddp()
        if using_ddp:
            assert isinstance(rank, int), f"expect exact rank number for ddp training."
        
        if stage == "train":
            dataset = self.train_data
            shuffle = True
        elif stage == "test":
            dataset = self.test_data
            shuffle = False
        elif stage == "val":
            dataset = self.val_data
            shuffle = False
        else:
            raise AssertionError(f"expect legal stage (train, test, val), got {stage}")
        
        sampler = DistributedSampler(dataset, num_replicas=len(self.cm.world), rank=rank) if using_ddp else None
        return DataLoader(dataset, batch_size=self.cm.batch_size, sampler=sampler, shuffle=shuffle)
            