from typing import Tuple, Union
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

def get_model_info(model: nn.Module, imgsz=640) -> Tuple[str, Tuple]:
    if not isinstance(imgsz, list):
        imgsz = [imgsz, imgsz]  # expand if int/float

    n_l = len(list(model.modules()))  # number of layers
    n_p = sum(x.numel() for x in model.parameters())  # number of parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number of gradients

    p = next(model.parameters())
    im = torch.empty((1, p.shape[1], *imgsz), device=p.device)  # input image in BCHW format
    model_ = deepcopy(model.module if isinstance(model, (nn.parallel.DistributedDataParallel)) else model)
    gflops = thop.profile(model_, inputs=[im], verbose=False)[0] / 1e9 * 2  # imgsz GFLOPs

    info = f"model summary: {n_l:,} layers; {n_p:,} parameters; {n_g:,} gradients; {gflops:,} GFLOPs"
    return info, (n_l, n_p, n_g, gflops)

def get_layers_info(model: nn.Module) -> Tuple[str, list]:
    head = f"{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}"
    layerinfo = [f"{item.i:>3}{str(item.f):>20}{item.n:>3}{item.p:10.0f}  {item.t:<45}{str(item.args):<30}" for item in model.layers]
    return head, layerinfo
