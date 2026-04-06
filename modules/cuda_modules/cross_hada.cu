#include "cdt_extensions.h"

namespace cdt
{

template <typename scalar_t>
__global__ void
crossHadamardKernel(const scalar_t *matrices, scalar_t *result, int p, int c, int h, int w)
{
    size_t bid = blockIdx.x / p;
    size_t pid = blockIdx.x % p;
    size_t hid = blockIdx.y * blockDim.y + threadIdx.y;
    size_t wid = blockIdx.z * blockDim.z + threadIdx.z;

    if (hid >= h || wid >= w)
        return;

    size_t tile_id = threadIdx.y * blockDim.z + threadIdx.z;
    size_t matg_id = hid * w + wid;

    size_t cc = 2 * c - 1;
    size_t i  = (cc - sqrtf(cc * cc - 8 * pid)) / 2;
    size_t j  = i + 1 + pid - (i * (cc - i) / 2);

    result[(bid * p + pid) * h * w + matg_id] =
        matrices[(bid * c + i) * h * w + matg_id] * matrices[(bid * c + j) * h * w + matg_id];
}

template <typename scalar_t>
__global__ void crossHadamardMixedKernel(const int64_t  *indices,
                                         const scalar_t *matrices,
                                         scalar_t       *result,
                                         int             p,
                                         int             c,
                                         int             h,
                                         int             w,
                                         int             k)
{
    size_t bid = blockIdx.x / p;
    size_t pid = blockIdx.x % p;
    size_t hid = blockIdx.y * blockDim.y + threadIdx.y;
    size_t wid = blockIdx.z * blockDim.z + threadIdx.z;

    if (hid >= h || wid >= w)
        return;

    size_t tile_id = threadIdx.y * blockDim.z + threadIdx.z;
    size_t matg_id = hid * w + wid;

    size_t kk = 2 * k - 1;
    size_t i  = (kk - sqrtf(kk * kk - 8 * pid)) / 2;
    size_t j  = i + 1 + pid - (i * (kk - i) / 2);

    result[(bid * p + pid) * h * w + matg_id] =
        matrices[(bid * c + indices[bid * k + i]) * h * w + matg_id] *
        matrices[(bid * c + indices[bid * k + j]) * h * w + matg_id];
}

torch::Tensor crossHadamard(torch::Tensor matrices)
{
    // 检查输入设备是否为CUDA
    AT_ASSERTM(matrices.is_cuda(), "Input must be a CUDA tensor");
    AT_ASSERTM(matrices.dim() == 4, "Input must be 4D [b, c, h, w]");
    AT_ASSERTM(matrices.is_contiguous(), "Input must be contiguous");

    cudaSetDevice(matrices.get_device());

    auto b = matrices.size(0);
    auto c = matrices.size(1);
    auto h = matrices.size(2);
    auto w = matrices.size(3);

    AT_ASSERTM(c > 1, "Input matrices must have at least 2 channels.");

    // 准备输出张量
    auto p       = c * (c - 1) / 2; // 哈达玛积对的数量
    auto options = torch::TensorOptions().dtype(matrices.dtype()).device(matrices.device());
    auto output  = torch::empty({b, p, h, w}, options);

    // 调用CUDA Kernel
    const int thd_edge = 32;
    dim3      block(b * p, (h + thd_edge - 1) / thd_edge, (w + thd_edge - 1) / thd_edge);
    dim3      thread(1, thd_edge, thd_edge);

    AT_DISPATCH_FLOATING_TYPES(matrices.scalar_type(), "cross_hadamard",
                               (
                                   [&]
                                   {
                                       crossHadamardKernel<scalar_t><<<block, thread>>>(
                                           matrices.data_ptr<scalar_t>(),
                                           output.data_ptr<scalar_t>(), p, c, h, w);
                                   }));

    return output;
}

torch::Tensor crossHadamardMixed(torch::Tensor matrices, torch::Tensor logits, const int k)
{
    AT_ASSERTM(matrices.is_cuda(), "Input matrices must be a CUDA tensor");
    AT_ASSERTM(matrices.dim() == 4, "Input matrices must be 4D [b, c, h, w]");
    AT_ASSERTM(matrices.is_contiguous(), "Input matrices must be contiguous");

    AT_ASSERTM(logits.is_cuda(), "Input logits must be a CUDA tensor");
    AT_ASSERTM(logits.dim() == 2, "Input logits must be 2D [b, c]");
    AT_ASSERTM(logits.is_contiguous(), "Input logits must be contiguous");

    AT_ASSERTM(matrices.get_device() == logits.get_device(), "Inputs must be on same device.");
    AT_ASSERTM(matrices.size(0) == logits.size(0), "Inputs must have same batch size.");
    AT_ASSERTM(matrices.size(1) == logits.size(1), "Inputs must have same channel size.");

    cudaSetDevice(matrices.get_device());

    auto b = matrices.size(0);
    auto c = matrices.size(1);
    auto h = matrices.size(2);
    auto w = matrices.size(3);

    AT_ASSERTM(c > 1, "Input matrices must have at least 2 channels.");

    // 准备输出张量
    auto p       = k * (k - 1) / 2; // 哈达玛积对的数量
    auto options = torch::TensorOptions().dtype(matrices.dtype()).device(matrices.device());
    auto output  = torch::empty({b, p, h, w}, options);

    // 调用CUDA Kernel
    const int thd_edge = 32;
    dim3      block(b * p, (h + thd_edge - 1) / thd_edge, (w + thd_edge - 1) / thd_edge);
    dim3      thread(1, thd_edge, thd_edge);

    auto [val, indices] = logits.topk(k);
    auto indices_ptr    = indices.data_ptr<int64_t>();

    AT_DISPATCH_FLOATING_TYPES(matrices.scalar_type(), "cross_hadamard_mixed",
                               (
                                   [&]
                                   {
                                       crossHadamardMixedKernel<scalar_t><<<block, thread>>>(
                                           indices_ptr, matrices.data_ptr<scalar_t>(),
                                           output.data_ptr<scalar_t>(), p, c, h, w, k);
                                   }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}

template <typename scalar_t>
__global__ void
crossHadamardBalancedKernel(const scalar_t *matrices, scalar_t *result, int p, int c, int h, int w)
{
    int64_t bid = blockIdx.x / c;
    int64_t cid = blockIdx.x % c;
    int64_t hid = blockIdx.y * blockDim.y + threadIdx.y;
    int64_t wid = blockIdx.z * blockDim.z + threadIdx.z;

    if (hid >= h || wid >= w)
        return;

    int64_t tile_id = threadIdx.y * blockDim.z + threadIdx.z;
    int64_t matg_id = hid * w + wid;

    int64_t pid, i, j;

    for (int64_t cit = 0, i; cit < c; cit++)
    {
        if (cit < cid && !((cid - cit) & 1))
        {
            i = cit;
            j = cid;
        }
        else if (cit > cid && (cid - cit) & 1)
        {
            i = cid;
            j = cit;
        }
        else
            continue;

        pid = i * (2 * c - i - 1) / 2 + (j - i - 1);

        result[(bid * p + pid) * h * w + matg_id] =
            matrices[(bid * c + i) * h * w + matg_id] * matrices[(bid * c + j) * h * w + matg_id];
    }
}

torch::Tensor crossHadamardBalanced(torch::Tensor matrices)
{
    // 检查输入设备是否为CUDA
    AT_ASSERTM(matrices.is_cuda(), "Input must be a CUDA tensor");
    AT_ASSERTM(matrices.dim() == 4, "Input must be 4D [b, c, h, w]");
    AT_ASSERTM(matrices.is_contiguous(), "Input must be contiguous");

    cudaSetDevice(matrices.get_device());

    auto b = matrices.size(0);
    auto c = matrices.size(1);
    auto h = matrices.size(2);
    auto w = matrices.size(3);

    AT_ASSERTM(c > 1, "Input matrices must have at least 2 channels.");

    // 准备输出张量
    auto p       = c * (c - 1) / 2; // 哈达玛积对的数量
    auto options = torch::TensorOptions().dtype(matrices.dtype()).device(matrices.device());
    auto output  = torch::empty({b, p, h, w}, options);

    // 调用CUDA Kernel
    const int thd_edge = 32;
    dim3      block(b * c, (h + thd_edge - 1) / thd_edge, (w + thd_edge - 1) / thd_edge);
    dim3      thread(1, thd_edge, thd_edge);

    AT_DISPATCH_FLOATING_TYPES(matrices.scalar_type(), "cross_hadamard_balanced",
                               (
                                   [&]
                                   {
                                       crossHadamardBalancedKernel<scalar_t><<<block, thread>>>(
                                           matrices.data_ptr<scalar_t>(),
                                           output.data_ptr<scalar_t>(), p, c, h, w);
                                   }));

    return output;
}

}; // namespace cdt