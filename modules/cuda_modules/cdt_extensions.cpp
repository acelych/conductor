#include "cdt_extensions.h"

namespace cdt
{

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("cross_hada", &crossHadamard, "Cross Hadamard Product");
    m.def("cross_hada_mixed", &crossHadamardMixed, "Cross Hadamard Product (Mixed TopK Ops)");
    m.def("top_k", &sortAPI, "Top K");
}

}; // namespace cdt