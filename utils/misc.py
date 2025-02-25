from typing import Tuple, Union, Sequence, Callable
from copy import deepcopy

import thop
import torch
from torch import nn, optim

try:
    _lr_sch = getattr(optim.lr_scheduler, 'LRScheduler')
except AttributeError:
    _lr_sch = getattr(optim.lr_scheduler, '_LRScheduler')
LR_Scheduler = Union[_lr_sch, optim.lr_scheduler.ReduceLROnPlateau]

BUILTIN_TYPE = [int, float, str, list, dict, tuple, set, frozenset, bytes, bytearray]

def get_module_class_str(obj: object) -> str:
    if not isinstance(obj, type):
        obj = obj.__class__
    module_str = obj.__module__
    class_str = obj.__name__
    if module_str != '__main__':
        return '.'.join((module_str, class_str))
    else:
        return class_str

def get_model_assessment(model: nn.Module, imgsz: Union[int, Sequence] = 244, model_mirror: Callable = None) -> Tuple[str, Tuple]:
    if not isinstance(imgsz, Sequence):
        imgsz = (imgsz, imgsz)  # expand if int/float

    n_l = len(list(model.modules()))  # number of layers
    n_p = sum(x.numel() for x in model.parameters())  # number of parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number of gradients

    p = next(model.parameters())
    im = torch.empty((1, p.shape[1], *imgsz), device=p.device)  # input image in BCHW format
    try:
        model_ = deepcopy(model.module if isinstance(model, (nn.parallel.DistributedDataParallel)) else model)
    except Exception as e:
        if model_mirror:
            model_ = model_mirror()
        else:
            raise(e)
    gflops = thop.profile(model_, inputs=[im], verbose=False)[0] / 1e9 * 2  # imgsz GFLOPs

    info = f"model summary: {n_l:,} layers; {n_p:,} parameters; {n_g:,} gradients; {gflops:,} GFLOPs (within {imgsz})"
    return info, (n_l, n_p, n_g, gflops)

def isbuiltin(obj: object):
    return obj.__class__ in BUILTIN_TYPE or obj is None
