import time
from typing import List

import torch
import torch.distributed as dist
from torch import Tensor


class Calculate:
    @staticmethod
    def update_conf_mat(conf_mat: Tensor, output: Tensor, label: Tensor):
        _, pred = output.max(1)  # Get Pred Tag
        for index in range(label.shape[0]):
            conf_mat[label[index], pred[index]] += 1

    @staticmethod
    def topk_accuracy(output: Tensor, label: Tensor, k: int = 5):
        _, topk_indices = output.topk(k)  # Get top-k [[top1, top2, ..., topk],...]
        correct: Tensor = topk_indices.eq(label.unsqueeze(-1))  # Get top-k correct [[False, True, ..., False], ...]
        return correct.sum().item() / label.size(0)
    
    @staticmethod
    def top1_accuracy(conf_mat: Tensor) -> Tensor:
        '''
        This method should be used with confusion matrix,
        for calculate top1-acc in dataloader, using topk_accuracy(..., k=1)
        '''
        TP_all = sum(conf_mat[i, i] for i in range(conf_mat.shape[0]))
        TP_all = conf_mat.diagonal().sum()
        return TP_all / conf_mat.sum()

    @staticmethod
    def precision(conf_mat: Tensor) -> Tensor:
        res = torch.zeros(conf_mat.shape[0], dtype=torch.float32, device=torch.cuda.current_device())
        for i in range(conf_mat.shape[0]):
            TP = conf_mat[i, i]
            TP_FP = conf_mat[:, i].sum()
            res[i] = TP / (TP_FP + 1e-8)
        return res

    @staticmethod
    def recall(conf_mat: Tensor) -> Tensor:
        res = torch.zeros(conf_mat.shape[0], dtype=torch.float32, device=torch.cuda.current_device())
        for i in range(conf_mat.shape[0]):
            TP = conf_mat[i, i]
            TP_FN = conf_mat[i, :].sum()
            res[i] = TP / (TP_FN + 1e-8)
        return res
    

class Recorder:
    def __init__(self, num_classes: int, k: int = 5):
        self.device = torch.cuda.current_device()
        self.loss: list = []
        self.topk: list = []
        self.conf_mat: Tensor = torch.zeros((num_classes, num_classes), dtype=torch.int32, device=self.device)
        self.k: int = k
        
        self.mean_loss = None
        self.mean_topk = None
        self.is_converged = False

    def __call__(self, output: Tensor, label: Tensor, loss: Tensor):
        self.loss.append(loss.item())
        self.topk.append(Calculate.topk_accuracy(output, label, self.k))
        Calculate.update_conf_mat(self.conf_mat, output, label)
        
    def clear(self):
        self.loss.clear()
        self.topk.clear()
        self.conf_mat.zero_()
        self.mean_loss = None
        self.mean_topk = None
        self.is_converged = False
        
    def converge(self):
        '''
        This method should be only called ONCE for converging data from all devices
        '''
        assert not self.is_converged, "recorder is already converged"
        self.loss = [loss for loss in self.loss if loss != float('nan')]
        self.mean_loss = sum(self.loss) / len(self.loss)
        self.mean_topk = sum(self.topk) / len(self.topk)
        if not dist.is_initialized():
            self.is_converged = True
            return
        mean_loss = Tensor([self.mean_loss], device=self.device)
        mean_topk = Tensor([self.mean_topk], device=self.device)
        dist.barrier()
        dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(mean_topk, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.conf_mat, op=dist.ReduceOp.SUM)
        self.mean_loss = (mean_loss / dist.get_world_size()).item()
        self.mean_topk = (mean_topk / dist.get_world_size()).item()
        self.is_converged = True

    def get_mean_loss(self):
        assert self.is_converged, "converge the recorder before get any vals"
        return self.mean_loss
    
    def get_mean_topk_acc(self):
        assert self.is_converged, "converge the recorder before get any vals"
        return self.mean_topk
        
    def get_conf_mat(self):
        assert self.is_converged, "converge the recorder before get any vals"
        return self.conf_mat
    
    @staticmethod
    def converge_loss(losses: List) -> float:
        mean_loss = sum(losses) / len(losses)
        if not dist.is_initialized():
            return mean_loss
        mean_loss = Tensor([mean_loss], device=torch.cuda.current_device())
        dist.barrier()
        dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
        return (mean_loss / dist.get_world_size()).item()

class MetricsManager:

    class Metrics:
        def __init__(self, *args, **kwargs):
            self.epoch = kwargs.get('epoch')
            self.time = kwargs.get('time')
            self.learn_rate = kwargs.get('learn_rate')
            
        def filled(self) -> bool:
            return all(v is not None for v in vars(self).values())
        
        def dummy_fill(self):
            for key in self.__dict__:
                if self.__dict__[key] is None:
                    self.__dict__[key] = float('nan')
        
        def get_heads(self) -> tuple:
            return tuple(vars(self).keys())
        
        def get_plot_form(self) -> dict:
            return {'x_axis': 'epoch'}
        
        def record_train(self, loss: Tensor):
            raise NotImplementedError
        
        def record_val(self, recorder: Recorder):
            raise NotImplementedError
        
    class ClassifyMetrics(Metrics):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.train_loss = kwargs.get('train_loss')
            self.val_loss = kwargs.get('val_loss')
            self.top1_acc = kwargs.get('top1_acc')
            self.top5_acc = kwargs.get('top5_acc')
            self.precision = kwargs.get('precision')
            self.recall = kwargs.get('recall')
            
        def get_plot_form(self):
            form = super().get_plot_form()
            form.update({
                'y_axis': [
                    ('train_loss', 'val_loss'),
                    'top1_acc',
                    'top5_acc',
                    'precision',
                    'recall',
                    'learn_rate',
                ]
            })
            return form
            
        def record_train(self, loss):
            self.train_loss = loss
            
        def record_val(self, recorder: Recorder):
            self.val_loss = recorder.get_mean_loss()
            self.top5_acc = recorder.get_mean_topk_acc()
            self.top1_acc = Calculate.top1_accuracy(recorder.get_conf_mat()).mean().item()
            self.precision = Calculate.precision(recorder.get_conf_mat()).mean().item()
            self.recall = Calculate.recall(recorder.get_conf_mat()).mean().item()
            
            
    class DetectMetrics(Metrics):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.train_box_loss = kwargs.get("train_box_loss")
            self.train_cls_loss = kwargs.get("train_cls_loss")
            self.train_dfl_loss = kwargs.get("train_dfl_loss")
            self.val_dfl_loss = kwargs.get("val_dfl_loss")
            self.val_dfl_loss = kwargs.get("val_dfl_loss")
            self.val_dfl_loss = kwargs.get("val_dfl_loss")
            self.precision = kwargs.get("precision")
            self.recall = kwargs.get("recall")
            self.mAP50 = kwargs.get("mAP50")
            self.mAP50_95 = kwargs.get("mAP50_95")

    def __init__(self):
        self.start_time = None
        self.metrics_collect: List[MetricsManager.Metrics] = []
        
    def __call__(self, metrics: Metrics) -> Metrics:
        metrics.time = self.get_time()  # record submission time (comparing with start)
        assert metrics.filled(), f"logger received an unfilled index"
        self.metrics_collect.append(metrics)
        return metrics

    def start(self):
        self.start_time = time.time()
        
    def get_time(self) -> float:
        assert self.start_time is not None, f"expect a start time for comparison"
        return time.time() - self.start_time
    
    @classmethod
    def get_metrics_holder(cls, task: str, epoch: int = -1) -> Metrics:
        if task == 'classify':
            return MetricsManager.ClassifyMetrics(epoch=epoch)
        if task == 'detect':
            return MetricsManager.DetectMetrics(epoch=epoch)
        
