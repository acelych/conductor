from typing import Tuple, Union, Sequence
from copy import deepcopy

import thop
import torch
from torch import nn, optim

LR_Scheduler = Union[optim.lr_scheduler.LRScheduler, optim.lr_scheduler.ReduceLROnPlateau]

def get_module_class_str(obj: object) -> str:
    if not isinstance(obj, type):
        obj = obj.__class__
    module_str = obj.__module__
    class_str = obj.__name__
    if module_str != '__main__':
        return '.'.join((module_str, class_str))
    else:
        return class_str

def get_model_assessment(model: nn.Module, imgsz=244) -> Tuple[str, Tuple]:
    if not isinstance(imgsz, Sequence):
        imgsz = (imgsz, imgsz)  # expand if int/float

    n_l = len(list(model.modules()))  # number of layers
    n_p = sum(x.numel() for x in model.parameters())  # number of parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number of gradients

    p = next(model.parameters())
    im = torch.empty((1, p.shape[1], *imgsz), device=p.device)  # input image in BCHW format
    model_ = deepcopy(model.module if isinstance(model, (nn.parallel.DistributedDataParallel)) else model)
    gflops = thop.profile(model_, inputs=[im], verbose=False)[0] / 1e9 * 2  # imgsz GFLOPs

    info = f"model summary: {n_l:,} layers; {n_p:,} parameters; {n_g:,} gradients; {gflops:,} GFLOPs (within {imgsz})"
    return info, (n_l, n_p, n_g, gflops)
