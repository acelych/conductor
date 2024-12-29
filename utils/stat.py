import time
from typing import List, Dict, Any

class IndexLogger():
    
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

    def __init__(self, mission: str):
        self.timer = None
        self.mission = mission
        self.indexes: Dict[str: List] = {
            'epoch': [],
            'time': [],
        }
        if mission == 'classify':
            pass
        elif mission == 'detect':
            self.indexes.update(IndexLogger.detect_indexes_init)

    def start(self):
        self.timer = time.time()

    def update(self, *args, **kwargs):
        for k, v in kwargs.items():
            assert isinstance(self.indexes[k], list)
            self.indexes[k].append(v)
