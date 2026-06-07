#pragma once

#include "gptcu/layers_cpu.hpp"

#include <vector>

namespace gptcu {

Tensor2D layer_norm_cuda(const Tensor2D& input,
                         const std::vector<Tensor2D::Value>& gamma,
                         const std::vector<Tensor2D::Value>& beta,
                         Tensor2D::Value epsilon = 1.0e-5f);
Tensor2D gelu_cuda(const Tensor2D& input);
Tensor2D linear_cuda(const CpuLinear& linear, const Tensor2D& input);

} // namespace gptcu
