import time
from typing import List, Dict, Any

import numpy as np
from torch import Tensor

from .cmd import CommandDetails

class Calculate:
    @staticmethod
    def update_conf_mat(conf_mat: np.ndarray, output: Tensor, label: Tensor):
        _, pred = output.max(1)  # Get Pred Tag
        for index in range(conf_mat.shape[0]):
            conf_mat[label[index], pred[index]] += 1

    @staticmethod
    def topk_accuracy(output: Tensor, label: Tensor, k: int = 5):
        _, topk_indices = output.topk(k)  # Get top-k [[top1, top2, ..., topk],...]
        correct = topk_indices.eq(label.unsqueeze(-1))  # Get top-k correct [[False, True, ..., False], ...]
        return correct.sum().item() / label.size(0)
    
    @staticmethod
    def top1_accuracy(conf_mat: np.ndarray):
        '''
        This method is used by confusion matrix,
        for calculate top1-acc in dataloader, using topk_accuracy(..., k=1)
        '''
        TP_all = sum(conf_mat[i, i] for i in range(conf_mat.shape[0]))
        return TP_all / np.sum(conf_mat)

    @staticmethod
    def precision(conf_mat: np.ndarray):
        res = np.zeros(conf_mat.shape[0], dtype=np.float32)
        for i in range(conf_mat.shape[0]):
            TP = conf_mat[i, i]
            TP_FP = np.sum(conf_mat[:, i])
            res[i] = TP / TP_FP
        return res

    @staticmethod
    def recall(conf_mat: np.ndarray):
        res = np.zeros(conf_mat.shape[0], dtype=np.float32)
        for i in range(conf_mat.shape[0]):
            TP = conf_mat[i, i]
            TP_FN = np.sum(conf_mat[i, :])
            res[i] = TP / TP_FN
        return res

class IndexLogger:

    class Recorder:
        def __init__(self, num_classes: int, k: int = 5):
            self.loss = []
            self.conf_mat = np.zeros((num_classes, num_classes), dtype=np.uint32)
            self.topk = []
            self.k = k

        def __call__(self, output: Tensor, label: Tensor, loss: Tensor):
            self.loss.append(loss.item())
            Calculate.update_conf_mat(self.conf_mat, output, label)
            Calculate.topk_accuracy(output, label, self.k)

        def get_mean_loss(self):
            return sum(self.loss) / len(self.loss)
        
        def get_mean_topk_acc(self):
            return sum(self.topk) / len(self.topk)
    
    classify_indexes_init = {
        'train_loss': [],
        'val_loss': [],
        'top1': [],
        'top5': [],
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

    def __init__(self, cd: CommandDetails, model_desc: dict):
        self.timer = None
        self.cd = cd
        self.model_desc = model_desc
        self.indexes: Dict[str: List] = {
            'epoch': [],
            'time': [],
        }
        if cd.task == 'classify':
            self.indexes.update(IndexLogger.classify_indexes_init)
        elif cd.task == 'detect':
            self.indexes.update(IndexLogger.detect_indexes_init)

    def start(self):
        self.timer = time.time()

    def update(self, *args, **kwargs):
        for k, v in kwargs.items():
            assert isinstance(self.indexes[k], list)
            self.indexes[k].append(v)

    def record_classify_epoch_result(self, recoder: Recorder):
        num_classes = self.model_desc.get("nc")  # Get Number of Classes

        self.indexes['val_loss'].append(recoder.get_mean_loss())
        self.indexes['top5'].append(recoder.get_mean_topk_acc())
        self.indexes['top1'].append(Calculate.top1_accuracy(recoder.conf_mat))
        self.indexes['precision'].append(Calculate.precision(recoder.conf_mat))
        self.indexes['recall'].append(Calculate.recall(recoder.conf_mat))
        
