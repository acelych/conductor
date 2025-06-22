#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCAtomics.cuh>

using namespace at;

__global__ void topk_gather_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ x,
    float* __restrict__ out,
    int* __restrict__ indices,
    int batch_size,
    int channels,
    int spatial_dim,
    int k
) {
    int b = blockIdx.x;
    if (b >= batch_size) return;

    // 每个线程块处理一个 batch sample
    auto logits_b = logits + b * channels;
    auto x_b = x + b * channels * spatial_dim;
    auto out_b = out + b * k * spatial_dim;

    // 使用堆排序找出 TopK
    struct Pair {
        float val;
        int idx;
    };
    Pair topk_vals[64];  // 支持最多 k=64

    for (int i = 0; i < k; ++i) {
        topk_vals[i].val = -1e9;
        topk_vals[i].idx = -1;
    }

    for (int c = 0; c < channels; ++c) {
        float val = logits_b[c];
        int insert_pos = -1;
        for (int i = 0; i < k; ++i) {
            if (val > topk_vals[i].val) {
                insert_pos = i;
                break;
            }
        }
        if (insert_pos != -1) {
            for (int i = k - 1; i > insert_pos; --i) {
                topk_vals[i] = topk_vals[i - 1];
            }
            topk_vals[insert_pos].val = val;
            topk_vals[insert_pos].idx = c;
        }
    }

    // Save indices
    for (int i = 0; i < k; ++i) {
        indices[b * k + i] = topk_vals[i].idx;
        for (int s = 0; s < spatial_dim; ++s) {
            out_b[i * spatial_dim + s] = x_b[topk_vals[i].idx * spatial_dim + s];
        }
    }
}

std::tuple<Tensor, Tensor> topk_gather_cuda(const Tensor& x, const Tensor& logits, int k) {
    int batch_size = logits.size(0);
    int channels = logits.size(1);
    int spatial_dim = x.numel() / (batch_size * channels);

    Tensor out = torch::empty({batch_size, k, spatial_dim}, x.options());
    Tensor indices = torch::empty({batch_size, k}, logits.options().dtype(torch::kInt32));

    dim3 grid(batch_size);
    topk_gather_kernel<<<grid, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
        logits.data_ptr<float>(),
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        indices.data_ptr<int>(),
        batch_size,
        channels,
        spatial_dim,
        k
    );

    // Reshape output to match input shape
    int H = x.size(2), W = x.size(3);
    out = out.view({batch_size, k, H, W});
    return std::make_tuple(out, indices);
}