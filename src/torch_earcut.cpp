#include <stdexcept>
#include <vector>
#include <array>
#include <utility>
#include <span>
#include <cstring>
#include <omp.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include "earcut.hpp"

template <typename F, size_t D, typename N>
std::vector<torch::Tensor> triangulate(torch::Tensor contours)
{
    using Point = std::array<F, D>;

    if (contours.dim() != 3 || contours.size(2) != D)
        throw std::invalid_argument("Contours should have shape (B, L, D)");

    if (contours.dtype() != torch::CppTypeToScalarType<F>())
        throw std::invalid_argument("Contours should have type F");

    if (!contours.is_contiguous())
        throw std::invalid_argument("Contours should be contiguous in memory");

    if (contours.device() != torch::kCPU)
        throw std::invalid_argument("Contours should be in the CPU memory");

    auto options = torch::TensorOptions().dtype(torch::CppTypeToScalarType<N>());
    auto indices = std::vector<torch::Tensor>(contours.size(0));

#pragma omp parallel shared(indices) num_threads(int(omp_get_num_procs() * 0.75))
    {
        mapbox::detail::Earcut<N> earcut;
#pragma omp for schedule(dynamic, 1)
        for (size_t i = 0; i < contours.size(0); i++)
        {
            std::array<std::span<Point>, 1> contourView({std::span<Point>(reinterpret_cast<Point *>(contours[i].data_ptr()), contours.size(1))});
            earcut(contourView);
            indices[i] = torch::from_blob(earcut.indices.data(), {earcut.indices.size()}, options).clone();
        }
    }
    return indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("triangulate_float_2_int32", &triangulate<float, 2, int32_t>);
}