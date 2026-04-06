#include "cdt_extensions.h"

namespace cdt
{

template <typename scalar_t>
__global__ void dySoftKernel(scalar_t *__restrict__ matrices,
                             const scalar_t *__restrict__ alpha,
                             const scalar_t *__restrict__ weight,
                             const scalar_t *__restrict__ bias,
                             int b,
                             int c,
                             int h,
                             int w)
{
    int64_t bid = blockIdx.x / c;
    int64_t cid = blockIdx.x % c;
    int64_t hid = blockIdx.y * blockDim.y + threadIdx.y;
    int64_t wid = blockIdx.z * blockDim.z + threadIdx.z;

    if (hid >= h || wid >= w)
        return;

    int64_t idx = ((bid * c + cid) * h + hid) * w + wid;

    scalar_t num        = matrices[idx];
    scalar_t weight_val = weight[cid];
    scalar_t bias_val   = bias[cid];
    scalar_t alpha_val  = alpha[0];

    num *= alpha_val;
    num /= (scalar_t(1) + abs(num));
    num = num * weight_val + bias_val;

    matrices[idx] = num;
}

void dySoft(torch::Tensor matrices, torch::Tensor alpha, torch::Tensor weight, torch::Tensor bias)
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

    AT_ASSERTM(weight.size(0) == c, "Weight size must match channels");
    AT_ASSERTM(bias.size(0) == c, "Bias size must match channels");

    // 调用CUDA Kernel
    const int thd_edge = 32;
    dim3      block(b * c, (h + thd_edge - 1) / thd_edge, (w + thd_edge - 1) / thd_edge);
    dim3      thread(1, thd_edge, thd_edge);

    AT_DISPATCH_FLOATING_TYPES(matrices.scalar_type(), "dysoft",
                               (
                                   [&]
                                   {
                                       dySoftKernel<scalar_t><<<block, thread>>>(
                                           matrices.data_ptr<scalar_t>(),
                                           alpha.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(),
                                           bias.data_ptr<scalar_t>(), b, c, h, w);
                                   }));
}

} // namespace cdt