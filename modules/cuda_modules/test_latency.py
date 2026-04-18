import torch
import time
import tqdm
import numpy as np
from tqdm import tqdm
from functools import wraps

try:
    from cdt_extensions import dysoft, cross_hada, cross_hada_balanced
except ImportError:
    print("✗ Failed to import cdt_extensions. Ensure 'python setup.py build_ext --inplace' was run.")
    exit(1)

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

def dysoft_pytorch_native(matrices, alpha, weight, bias):
    """Native PyTorch implementation of DySoft for latency baseline"""
    result = matrices.clone()
    result = alpha * result
    result = result / (1 + torch.abs(result))
    weight_4d = weight.view(1, -1, 1, 1)
    bias_4d = bias.view(1, -1, 1, 1)
    result = result * weight_4d + bias_4d
    return result

def benchmark_dysoft():
    """Benchmark DySoft CUDA implementation vs PyTorch native"""
    print("=== DySoft CUDA vs Native PyTorch Latency Benchmark ===\n")
    
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping.")
        return
    
    test_cases = [
        (1, 64, 224, 224),     # Small resolution feature map
        (4, 128, 112, 112),    # Medium
        (8, 256, 56, 56),      # Common backbone size
        (16, 512, 28, 28),     # Deep features
        (32, 1024, 14, 14),    # Large batch, small feature map
        (1, 2048, 7, 7),       # High channel count
    ]
    
    print(f"{'Shape':<20} {'PyTorch(ms)':<12} {'CUDA(ms)':<12} {'Speedup':<10} {'Memory Saved'}")
    print("-" * 75)
    
    for b, c, h, w in test_cases:
        x = torch.randn(b, c, h, w, device='cuda')
        alpha = torch.tensor([0.7], device='cuda')
        weight = torch.ones(c, device='cuda')
        bias = torch.zeros(c, device='cuda')
        
        # Warmup
        for _ in range(5):
            _ = dysoft_pytorch_native(x, alpha, weight, bias)
            x_copy = x.clone()
            dysoft(x_copy, alpha, weight, bias)
        
        torch.cuda.synchronize()
        
        start_time = time.time()
        for _ in range(100):
            result_pytorch = dysoft_pytorch_native(x, alpha, weight, bias)
        torch.cuda.synchronize()
        pytorch_time = (time.time() - start_time) / 100 * 1000
        
        start_time = time.time()
        for _ in range(100):
            x_copy = x.clone()
            dysoft(x_copy, alpha, weight, bias)
        torch.cuda.synchronize()
        cuda_time = (time.time() - start_time) / 100 * 1000
        
        tensor_size_mb = x.element_size() * x.nelement() / 1024**2
        memory_saved = tensor_size_mb  # In-place operation saves an entire tensor allocation
        speedup = pytorch_time / cuda_time
        
        print(f"({b},{c},{h},{w}): {pytorch_time:>10.3f} {cuda_time:>10.3f} {speedup:>9.1f}x {memory_saved:>8.1f}MB")
    print()

def generate_cross_hada_benchmark_npz():
    """Generates the heavy heatmap .npz benchmark arrays for cross_hada plotting."""
    print("=== Cross Hadamard Heavy Heatmap Generation ===")
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping.")
        return

    print("This will take several minutes. Results will be saved to 'exp_res.npz'...")
    timed_func1 = benchmark(repeat=500, warmup=10)(cross_hada)
    timed_func2 = benchmark(repeat=500, warmup=10)(cross_hada_balanced)

    batch_ls = range(1, 65)    # [1, 2, ..., 64]
    channel_ls = range(3, 67)  # [3, 4, ..., 66]
    edge_ls = range(1, 65)     # [1, 2, ..., 64]

    # Batch-Channel
    print("Generating Batch-Channel benchmarks...")
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
    print("Generating Batch-Edge benchmarks...")
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
    print("Generating Channel-Edge benchmarks...")
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
    print("Done. Saved to 'exp_res.npz' and CSV files.")

if __name__ == "__main__":
    print("\nStarting CUDA Latency Benchmarks...\n" + "="*40)
    benchmark_dysoft()
    generate_cross_hada_benchmark_npz()
    print("="*40 + "\nAll benchmarks finished.")
