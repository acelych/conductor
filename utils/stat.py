import time
from typing import List, Dict, Any

class IndexLogger():
    
    classify_indexes_init = {
        'train_loss': [],
        'train_top1': [],
        'train_top5': [],
        'val_loss': [],
        'val_top1': [],
        'val_top5': [],
        'precision': [],
        'recall': [],
    }
    
    detect_indexes_init = {
        'train_box_loss': [],
        'train_cls_loss': [],
        'train_dfl_loss': [],
        'val_dfl_loss': [],
        'val_dfl_loss': [],
        'val_dfl_loss': [],
        'precision': [],
        'recall': [],
        'mAP50': [],
        'mAP50-95': [],
    }

    def __init__(self, task: str):
        self.timer = None
        self.task = task
        self.indexes: Dict[str: List] = {
            'epoch': [],
            'time': [],
        }
        if task == 'classify':
            self.indexes.update(IndexLogger.classify_indexes_init)
        elif task == 'detect':
            self.indexes.update(IndexLogger.detect_indexes_init)

    def start(self):
        self.timer = time.time()

    def update(self, *args, **kwargs):
        for k, v in kwargs.items():
            assert isinstance(self.indexes[k], list)
            self.indexes[k].append(v)
