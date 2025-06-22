#include <torch/extension.h>

std::tuple<at::Tensor, at::Tensor> topk_gather_cuda(const at::Tensor &x, const at::Tensor &logits, int k);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("topk_gather", &topk_gather_cuda, "TopK + Gather in CUDA");
}