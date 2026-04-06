import torch
import time
import tqdm
import numpy as np
from tqdm import tqdm
from functools import wraps

from cdt_extensions import *

# a = torch.Tensor([[[[1., 1.], [1., 1.]], 
#                    [[2., 2.], [2., 2.]], 
#                    [[3., 3.], [3., 3.]],
#                    [[4., 4.], [4., 4.]],
#                    [[5., 5.], [5., 5.]]]]).to(device='cuda')
# print(a.shape)
# print(cross_hada_balanced(a))
# exit(0)

def benchmark(repeat=10, warmup=3):
    def decorator(func):
        @wraps(func)
        def wrapper(ls: list = None, *args, **kwargs):
            # warm up
            for _ in range(warmup):
                func(*args, **kwargs)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # run
            start = time.time()
            for _ in range(repeat):
                result = func(*args, **kwargs)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            elapsed = (time.time() - start) / repeat
            if ls is not None:
                ls.append(elapsed)
            else:
                print(f"{func.__name__} - avg time: {elapsed:.6f}s (repeats: {repeat})")
                
            return result
        return wrapper
    return decorator

def ground_truth_cross_hada(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    p = c * (c - 1) // 2
    y = torch.empty((b, p, h, w), dtype=x.dtype, device=x.device)

    pid = 0
    for i in range(c):
        for j in range(i + 1, c):
            y[:, pid, :, :] = x[:, i, :, :] * x[:, j, :, :]
            pid += 1

    return y

def ground_truth_cross_hada_mixed(x: torch.Tensor, logits: torch.Tensor, k: int) -> torch.Tensor:
    b, c, h, w = x.shape
    p = k * (k - 1) // 2
    y = torch.empty((b, p, h, w), dtype=x.dtype, device=x.device)
    
    _, indices = logits.topk(k)

    pid = 0
    for i in range(k):
        for j in range(i + 1, k):
            for b in range(x.shape[0]):
                y[b, pid, :, :] = x[b, indices[b, i], :, :] * x[b, indices[b, j], :, :]
            pid += 1

    return y

b = 64
c = 17
k = 4
input_tensor = torch.randn(b, c, 123, 123, device='cuda')
logits = torch.randn(b, c, device='cuda')

print("Starting cross_hada computation...")
result = cross_hada_balanced(input_tensor) == ground_truth_cross_hada(input_tensor)
print(f"Cross Hada result matches ground truth: {result.all().item()}")

print("Starting cross_hada_mixed computation...")
# tkt = torch.Tensor([
#     [[[1, 1], [1, 1]], 
#     [[2, 2], [2, 2]], 
#     [[3, 3], [3, 3]], 
#     [[4, 4], [4, 4]], 
#     [[5, 5], [5, 5]]],
#     [[[6, 6], [6, 6]], 
#     [[7, 7], [7, 7]], 
#     [[8, 8], [8, 8]], 
#     [[9, 9], [9, 9]], 
#     [[0, 0], [0, 0]]]
#     ]).to(device='cuda')
# logits = torch.Tensor([[9.8, 0.2, 0.99, 2.7, 0.1],[0.1, 6.8, 7.7, 2.4, 0.7]]).to(device='cuda')
# print(ground_truth_cross_hada_mixed(tkt, logits, 3))
# print(cross_hada_mixed(input_tensor, logits, 3))
result = ground_truth_cross_hada_mixed(input_tensor, logits, k) == cross_hada_mixed(input_tensor, logits, k)
print(f"Cross Hada Mixed result matches ground truth: {result.all().item()}")

# input_tensor = torch.randn(64, 12, 2, 2, device='cuda')

# ========== EXPERIMENT ========== #

timed_func1 = benchmark(repeat=500, warmup=10)(cross_hada)
timed_func2 = benchmark(repeat=500, warmup=10)(cross_hada_balanced)

input_tensor = torch.randn(1, 64, 1, 1, device='cuda')
timed_func1(None, input_tensor)
timed_func2(None, input_tensor)
exit()

# Exp
batch_ls = range(1, 65)    # [1, 5, 9, ..., 61]
channel_ls = range(3, 67)  # [3, 7, 11, ..., 63]
edge_ls = range(1, 65)     # [1, 5, 9, ..., 61]

# Batch-Channel
f1_score = []
f2_score = []
for b in tqdm(batch_ls):
    for c in channel_ls:
        input_tensor = torch.randn(b, c, 64, 64, device='cuda')
        timed_func1(f1_score, input_tensor)
        timed_func2(f2_score, input_tensor)
        del input_tensor
batch_channel = np.stack((
    np.array(f1_score).reshape(len(batch_ls), len(channel_ls)),
    np.array(f2_score).reshape(len(batch_ls), len(channel_ls))
    ))

# Batch-Edge
f1_score = []
f2_score = []
for b in tqdm(batch_ls):
    for e in edge_ls:
        input_tensor = torch.randn(b, 64, e, e, device='cuda')
        timed_func1(f1_score, input_tensor)
        timed_func2(f2_score, input_tensor)
        del input_tensor
batch_edge = np.stack((
    np.array(f1_score).reshape(len(batch_ls), len(edge_ls)),
    np.array(f2_score).reshape(len(batch_ls), len(edge_ls))
    ))

# Channel-Edge
f1_score = []
f2_score = []
for c in tqdm(channel_ls):
    for e in edge_ls:
        input_tensor = torch.randn(64, c, e, e, device='cuda')
        timed_func1(f1_score, input_tensor)
        timed_func2(f2_score, input_tensor)
        del input_tensor
channel_edge = np.stack((
    np.array(f1_score).reshape(len(channel_ls), len(edge_ls)),
    np.array(f2_score).reshape(len(channel_ls), len(edge_ls))
    ))

assert batch_channel.shape == batch_edge.shape and batch_edge.shape == channel_edge.shape
np.savetxt('bc.csv', np.hstack((batch_channel[0], batch_channel[1])), delimiter=',')
np.savetxt('be.csv', np.hstack((batch_edge[0], batch_edge[1])), delimiter=',')
np.savetxt('ce.csv', np.hstack((channel_edge[0], channel_edge[1])), delimiter=',')
np.savez("exp_res.npz", bc=batch_channel, be=batch_edge, ce=channel_edge)
print("Done.")