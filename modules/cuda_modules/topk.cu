#include "cdt_extensions.h"
#include <ATen/ATen.h>

namespace cdt
{

torch::Tensor sortAPI(torch::Tensor &matrices, const int k)
{
    // 检查输入设备是否为CUDA
    AT_ASSERTM(matrices.is_cuda(), "Input must be a CUDA tensor");
    AT_ASSERTM(matrices.dim() == 2, "Input must be 2D [b, c]");
    AT_ASSERTM(matrices.is_contiguous(), "Input must be contiguous");

    cudaSetDevice(matrices.get_device());

    auto [val, indices] = matrices.topk(k);

    return indices;
}

}; // namespace cdt