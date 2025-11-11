#ifndef __CDT_EXTENSIONS_H__
#define __CDT_EXTENSIONS_H__

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace cdt
{

torch::Tensor crossHadamard(torch::Tensor matrices);
torch::Tensor crossHadamardMixed(torch::Tensor matrices, torch::Tensor logits, const int k);
torch::Tensor crossHadamardBalanced(torch::Tensor matrices);
void dySoft(torch::Tensor matrices, torch::Tensor alpha, torch::Tensor weight, torch::Tensor bias);

}; // namespace cdt

#endif //__CDT_EXTENSIONS_H__
