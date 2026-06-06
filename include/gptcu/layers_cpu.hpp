#pragma once

#include <cstddef>
#include <vector>

namespace gptcu {

struct Tensor2D {
    using Value = float;

    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<Value> values;

    Tensor2D() = default;
    Tensor2D(std::size_t row_count, std::size_t col_count, std::vector<Value> data);

    Value& at(std::size_t row, std::size_t col);
    Value at(std::size_t row, std::size_t col) const;
};

class CpuLinear {
public:
    using Value = Tensor2D::Value;

    CpuLinear() = default;
    CpuLinear(std::size_t input_dim,
              std::size_t output_dim,
              std::vector<Value> weights,
              std::vector<Value> bias = {});

    std::size_t input_dim() const;
    std::size_t output_dim() const;
    Tensor2D forward(const Tensor2D& input) const;

private:
    std::size_t input_dim_ = 0;
    std::size_t output_dim_ = 0;
    std::vector<Value> weights_;
    std::vector<Value> bias_;
};

class CpuFeedForward {
public:
    CpuFeedForward(CpuLinear up_projection, CpuLinear down_projection);

    Tensor2D forward(const Tensor2D& input) const;

private:
    CpuLinear up_projection_;
    CpuLinear down_projection_;
};

Tensor2D residual_add(const Tensor2D& lhs, const Tensor2D& rhs);
Tensor2D layer_norm(const Tensor2D& input,
                    const std::vector<Tensor2D::Value>& gamma,
                    const std::vector<Tensor2D::Value>& beta,
                    Tensor2D::Value epsilon = 1.0e-5f);
Tensor2D gelu(const Tensor2D& input);
Tensor2D causal_self_attention(const Tensor2D& query, const Tensor2D& key, const Tensor2D& value);

} // namespace gptcu