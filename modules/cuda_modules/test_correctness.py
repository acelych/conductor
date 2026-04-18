import torch
import torch.nn as nn
import numpy as np

try:
    from cdt_extensions import dysoft, cross_hada, cross_hada_balanced, cross_hada_mixed
except ImportError:
    print("✗ Failed to import cdt_extensions. Ensure 'python setup.py build_ext --inplace' was run.")
    exit(1)

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
            for batch_idx in range(b):
                y[batch_idx, pid, :, :] = x[batch_idx, indices[batch_idx, i], :, :] * x[batch_idx, indices[batch_idx, j], :, :]
            pid += 1
    return y

def dysoft_cpu(matrices, alpha, weight, bias):
    result = matrices.clone()
    result = alpha * result
    result = result / (1 + torch.abs(result))
    channels = matrices.shape[1]
    weight_4d = weight.view(1, channels, 1, 1)
    bias_4d = bias.view(1, channels, 1, 1)
    result = result * weight_4d + bias_4d
    return result

def test_cross_hada_correctness():
    print("=== Cross Hadamard CUDA Correctness Test ===")
    
    if not torch.cuda.is_available():
        print("✗ CUDA not available. Skipping.")
        return

    b, c, k = 64, 17, 4
    input_tensor = torch.randn(b, c, 12, 12, device='cuda')
    logits = torch.randn(b, c, device='cuda')

    # Test standard and balanced cross_hada
    gt_result = ground_truth_cross_hada(input_tensor)
    res_normal = cross_hada(input_tensor)
    res_balanced = cross_hada_balanced(input_tensor)
    
    # We use allclose with a small tolerance to account for floating point differences
    if torch.allclose(gt_result, res_normal, atol=1e-5):
        print("✓ cross_hada matches ground truth.")
    else:
        print("✗ cross_hada FAILED.")
        
    if torch.allclose(gt_result, res_balanced, atol=1e-5):
        print("✓ cross_hada_balanced matches ground truth.")
    else:
        print("✗ cross_hada_balanced FAILED.")

    # Test cross_hada_mixed
    gt_mixed = ground_truth_cross_hada_mixed(input_tensor, logits, k)
    res_mixed = cross_hada_mixed(input_tensor, logits, k)
    
    if torch.allclose(gt_mixed, res_mixed, atol=1e-5):
        print("✓ cross_hada_mixed matches ground truth.")
    else:
        print("✗ cross_hada_mixed FAILED.")
    print()

def test_dysoft_correctness():
    print("=== DySoft CUDA Correctness Test ===")
    
    if not torch.cuda.is_available():
        print("✗ CUDA not available. Skipping.")
        return

    torch.manual_seed(42)
    batch_size, channels, height, width = 2, 3, 4, 4
    alpha = torch.tensor([0.7])
    weight = torch.ones(channels)
    bias = torch.zeros(channels)
    
    x_cpu = torch.randn(batch_size, channels, height, width)
    y_cpu = dysoft_cpu(x_cpu, alpha, weight, bias)
    
    x_gpu = x_cpu.cuda()
    alpha_gpu = alpha.cuda()
    weight_gpu = weight.cuda()
    bias_gpu = bias.cuda()
    
    # In-place operation
    x_gpu_copy = x_gpu.clone()
    dysoft(x_gpu_copy, alpha_gpu, weight_gpu, bias_gpu)
    y_gpu_cpu = x_gpu_copy.cpu()
    
    diff = torch.abs(y_gpu_cpu - y_cpu)
    max_diff = diff.max().item()
    
    if max_diff < 1e-4:
        print("✓ Basic DySoft implementation is correct.")
    else:
        print(f"✗ DySoft FAILED. Max difference: {max_diff:.6e}")
        
    # Edge case: zero input
    x_zero = torch.zeros(1, channels, 1, 1)
    y_zero_cpu = dysoft_cpu(x_zero, alpha, weight, bias)
    x_zero_gpu = x_zero.cuda()
    dysoft(x_zero_gpu, alpha_gpu, weight_gpu, bias_gpu)
    if torch.allclose(y_zero_cpu, x_zero_gpu.cpu(), atol=1e-4):
        print("✓ Edge Case: Zero input handled correctly.")
    else:
        print("✗ Edge Case: Zero input FAILED.")
        
    # Weight Broadcasting test
    weight2 = torch.tensor([0.5, 1.0, 2.0])
    bias2 = torch.tensor([-0.1, 0.0, 0.1])
    y_param_cpu = dysoft_cpu(x_cpu, alpha, weight2, bias2)
    
    x_gpu2 = x_cpu.cuda()
    dysoft(x_gpu2, alpha_gpu, weight2.cuda(), bias2.cuda())
    
    if torch.allclose(y_param_cpu, x_gpu2.cpu(), atol=1e-4):
        print("✓ Channel Broadcasting (Weight/Bias) is correct.")
    else:
        print("✗ Channel Broadcasting FAILED.")
    print()

if __name__ == "__main__":
    print("\nStarting CUDA Correctness Tests...\n" + "="*40)
    test_cross_hada_correctness()
    test_dysoft_correctness()
    print("="*40 + "\nAll correctness tests finished.")
