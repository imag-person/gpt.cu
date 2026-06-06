#include "gptcu/layers_cpu.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace gptcu {
namespace {

constexpr Tensor2D::Value kGeluScale = 0.7978845608028654f;
constexpr Tensor2D::Value kGeluCubic = 0.044715f;

void require_same_shape(const Tensor2D& lhs, const Tensor2D& rhs, const char* operation) {
    if (lhs.rows != rhs.rows || lhs.cols != rhs.cols) {
        throw std::invalid_argument(std::string(operation) + " requires tensors with the same shape");
    }
}

} // namespace

Tensor2D::Tensor2D(std::size_t row_count, std::size_t col_count, std::vector<Value> data)
    : rows(row_count), cols(col_count), values(std::move(data)) {
    if (rows * cols != values.size()) {
        throw std::invalid_argument("Tensor2D data size does not match rows * cols");
    }
}

Tensor2D::Value& Tensor2D::at(std::size_t row, std::size_t col) {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Tensor2D index is out of range");
    }
    return values[row * cols + col];
}

Tensor2D::Value Tensor2D::at(std::size_t row, std::size_t col) const {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Tensor2D index is out of range");
    }
    return values[row * cols + col];
}

CpuLinear::CpuLinear(std::size_t input_dim,
                     std::size_t output_dim,
                     std::vector<Value> weights,
                     std::vector<Value> bias)
    : input_dim_(input_dim), output_dim_(output_dim), weights_(std::move(weights)), bias_(std::move(bias)) {
    if (input_dim_ == 0 || output_dim_ == 0) {
        throw std::invalid_argument("linear dimensions must be greater than zero");
    }
    if (weights_.size() != input_dim_ * output_dim_) {
        throw std::invalid_argument("linear weights must have output_dim * input_dim values");
    }
    if (!bias_.empty() && bias_.size() != output_dim_) {
        throw std::invalid_argument("linear bias must be empty or have output_dim values");
    }
}

std::size_t CpuLinear::input_dim() const {
    return input_dim_;
}

std::size_t CpuLinear::output_dim() const {
    return output_dim_;
}

Tensor2D CpuLinear::forward(const Tensor2D& input) const {
    if (input.cols != input_dim_) {
        throw std::invalid_argument("linear input width does not match input_dim");
    }

    Tensor2D output(input.rows, output_dim_, std::vector<Value>(input.rows * output_dim_, 0.0f));
    for (std::size_t row = 0; row < input.rows; ++row) {
        for (std::size_t out_col = 0; out_col < output_dim_; ++out_col) {
            Value sum = bias_.empty() ? 0.0f : bias_[out_col];
            for (std::size_t in_col = 0; in_col < input_dim_; ++in_col) {
                sum += input.at(row, in_col) * weights_[out_col * input_dim_ + in_col];
            }
            output.at(row, out_col) = sum;
        }
    }
    return output;
}

CpuFeedForward::CpuFeedForward(CpuLinear up_projection, CpuLinear down_projection)
    : up_projection_(std::move(up_projection)), down_projection_(std::move(down_projection)) {
    if (up_projection_.output_dim() != down_projection_.input_dim()) {
        throw std::invalid_argument("feed-forward projections have incompatible dimensions");
    }
}

Tensor2D CpuFeedForward::forward(const Tensor2D& input) const {
    return down_projection_.forward(gelu(up_projection_.forward(input)));
}

Tensor2D residual_add(const Tensor2D& lhs, const Tensor2D& rhs) {
    require_same_shape(lhs, rhs, "residual_add");
    Tensor2D result(lhs.rows, lhs.cols, lhs.values);
    for (std::size_t i = 0; i < result.values.size(); ++i) {
        result.values[i] += rhs.values[i];
    }
    return result;
}

Tensor2D layer_norm(const Tensor2D& input,
                    const std::vector<Tensor2D::Value>& gamma,
                    const std::vector<Tensor2D::Value>& beta,
                    Tensor2D::Value epsilon) {
    if (input.cols == 0) {
        throw std::invalid_argument("layer_norm requires at least one column");
    }
    if (gamma.size() != input.cols || beta.size() != input.cols) {
        throw std::invalid_argument("layer_norm gamma and beta must match input width");
    }
    if (epsilon < 0.0f) {
        throw std::invalid_argument("layer_norm epsilon must be non-negative");
    }

    Tensor2D output(input.rows, input.cols, std::vector<Tensor2D::Value>(input.values.size(), 0.0f));
    for (std::size_t row = 0; row < input.rows; ++row) {
        Tensor2D::Value mean = 0.0f;
        for (std::size_t col = 0; col < input.cols; ++col) {
            mean += input.at(row, col);
        }
        mean /= static_cast<Tensor2D::Value>(input.cols);

        Tensor2D::Value variance = 0.0f;
        for (std::size_t col = 0; col < input.cols; ++col) {
            const auto centered = input.at(row, col) - mean;
            variance += centered * centered;
        }
        variance /= static_cast<Tensor2D::Value>(input.cols);
        const auto inverse_stddev = 1.0f / std::sqrt(variance + epsilon);

        for (std::size_t col = 0; col < input.cols; ++col) {
            output.at(row, col) = (input.at(row, col) - mean) * inverse_stddev * gamma[col] + beta[col];
        }
    }
    return output;
}

Tensor2D gelu(const Tensor2D& input) {
    Tensor2D output(input.rows, input.cols, std::vector<Tensor2D::Value>(input.values.size(), 0.0f));
    for (std::size_t i = 0; i < input.values.size(); ++i) {
        const auto x = input.values[i];
        output.values[i] = 0.5f * x * (1.0f + std::tanh(kGeluScale * (x + kGeluCubic * x * x * x)));
    }
    return output;
}

Tensor2D causal_self_attention(const Tensor2D& query, const Tensor2D& key, const Tensor2D& value) {
    if (query.rows != key.rows || key.rows != value.rows) {
        throw std::invalid_argument("causal_self_attention requires equal sequence lengths");
    }
    if (query.cols == 0 || query.cols != key.cols) {
        throw std::invalid_argument("causal_self_attention query/key widths must match and be non-zero");
    }

    Tensor2D output(query.rows, value.cols, std::vector<Tensor2D::Value>(query.rows * value.cols, 0.0f));
    const auto scale = 1.0f / std::sqrt(static_cast<Tensor2D::Value>(query.cols));
    for (std::size_t row = 0; row < query.rows; ++row) {
        std::vector<Tensor2D::Value> scores(row + 1, 0.0f);
        for (std::size_t key_row = 0; key_row <= row; ++key_row) {
            Tensor2D::Value dot = 0.0f;
            for (std::size_t col = 0; col < query.cols; ++col) {
                dot += query.at(row, col) * key.at(key_row, col);
            }
            scores[key_row] = dot * scale;
        }

        const auto max_score = *std::max_element(scores.begin(), scores.end());
        Tensor2D::Value denominator = 0.0f;
        for (auto& score : scores) {
            score = std::exp(score - max_score);
            denominator += score;
        }
        if (denominator <= std::numeric_limits<Tensor2D::Value>::min()) {
            throw std::runtime_error("causal_self_attention softmax denominator underflowed");
        }

        for (std::size_t key_row = 0; key_row <= row; ++key_row) {
            const auto probability = scores[key_row] / denominator;
            for (std::size_t col = 0; col < value.cols; ++col) {
                output.at(row, col) += probability * value.at(key_row, col);
            }
        }
    }
    return output;
}

} // namespace gptcu