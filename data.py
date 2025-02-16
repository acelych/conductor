import yaml
import pickle
from io import BytesIO
from typing import Any
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms

from .utils.back import ConfigManager
from .utils.front import LogInterface


class ClassifyDataset(Dataset):
    def __init__(
            self, 
            data_path: Path, 
            format: str, 
            log: LogInterface, 
            imgsz: tuple,
            mean: list = None, 
            std: list = None, 
            enhance: bool = False,
            **kwargs
        ):
        super().__init__()
        self.data_path = data_path
        self.data: Any = None
        self.cache = get_cache(data_path)
        self.format = format
        self.trm = self.get_transforms(imgsz, enhance, mean, std)

        if format == 'parquet':
            self.data = pd.read_parquet(self.data_path)
        else:
            pass  #TODO
    
    def __len__(self):
        if self.format == 'parquet':
            return self._len_all()
        
    def __getitem__(self, index):
        if index in self.cache:
            img, label = self.cache[index]
        else:
            if self.format == 'parquet':
                img, label = self._getitem_parquet(index)
            elif True:
                pass # TODO
            self.cache[index] = (img, label)
            
        return self.trm(img), label

    def _len_all(self):
        return len(self.data)

    def _getitem_parquet(self, index):
        tar = self.data.loc[index]
        img, label = tar['img']['bytes'], tar['fine_label']
        img = Image.open(BytesIO(img))
        return img, label
    
    @staticmethod
    def get_transforms(imgsz: tuple, enhance: bool = False, mean: list = None, std: list = None):
        trm = [transforms.Resize(imgsz)]
        if enhance:
            trm.extend([
                transforms.RandomCrop(224, padding=4),
                transforms.RandomHorizontalFlip(),
            ])
        trm.append(transforms.ToTensor())
        if mean and std:
            trm.append(transforms.Normalize(mean=mean, std=std))
        return transforms.Compose(trm)


class DataLoaderManager:
    def __init__(self, cm: ConfigManager, log: LogInterface):
        self.cm = cm
        self.log = log
        self.parse_yaml(cm.data_yaml_path)
            
    def parse_yaml(self, yaml_path: str):
        
        with open(yaml_path, 'r') as f:
            data_desc: dict = yaml.safe_load(f)
        assert data_desc.get('task') == self.cm.task, f"task of dataset '{yaml_path}' ({data_desc.get('task')}) does not match required task ({self.cm.task})"
        assert data_desc.get('train'), f"expected to have train in dataset"
        assert data_desc.get("test") or data_desc.get("val"), f"expected to have at least one of test and val in dataset, got none"
            
        dataset_path = Path(data_desc.get('path'))
        if not dataset_path.is_absolute():
            dataset_path = Path(yaml_path).parent / dataset_path

        if not data_desc.get("test"):
            data_desc["test"] = data_desc.get("val")
        elif not data_desc.get("val"):
            data_desc["val"] = data_desc.get("test")

        train_path: Path = dataset_path / data_desc.get("train").get('path')
        test_path: Path = dataset_path / data_desc.get("test").get('path')
        val_path: Path = dataset_path / data_desc.get("val").get('path')
            
        if self.cm.task == "classify":
            if train_path.is_file() and train_path.suffix == ".parquet":
                mode = "parquet"

            self.train_data = ClassifyDataset(train_path, mode, self.log, self.cm.imgsz, enhance=True, **data_desc.get("train"))
            self.test_data = ClassifyDataset(test_path, mode, self.log, self.cm.imgsz, **data_desc.get("test"))
            self.val_data = ClassifyDataset(val_path, mode, self.log, self.cm.imgsz, **data_desc.get("val")) if test_path != val_path else self.test_data
        else:
            pass  #TODO
            
        self.names = data_desc.get("names")

    def save_caches(self):
        all_data = [self.train_data, self.test_data]
        if self.val_data is not self.test_data:
            all_data.append(self.val_data)
        for data in all_data:
            cache_path = save_cache(data.data_path, data.cache)
            if cache_path:
                self.log.info(f"dataset cache saved to {cache_path}")
        
    def get_dataloader(self, stage: str, rank = None):
        using_ddp = self.cm.isddp()
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

def get_cache_path(path: Path) -> Path:
    if path.is_file():
        return path.parent / (path.stem + '.cache')
    if path.is_dir():
        return path / '.cache'
    
def get_cache(path: Path) -> dict:
    cache_path = get_cache_path(path)
    if cache_path.is_file():
        with open(cache_path, 'rb') as file:
            cache = pickle.load(file)
        if isinstance(cache, dict):
            return cache
    return dict()

def save_cache(data_path: Path, cache: Any) -> Path:
    if cache:
        cache_path = get_cache_path(data_path)
        with open(cache_path, 'wb') as file:
            pickle.dump(cache, file)
        return cache_path
    return None
            