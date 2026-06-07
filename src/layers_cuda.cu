#include "gptcu/layers_cuda.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace gptcu {
namespace {

using Value = Tensor2D::Value;

__global__ void layer_norm_kernel(const Value* input,
                                  std::size_t rows,
                                  std::size_t cols,
                                  const Value* gamma,
                                  const Value* beta,
                                  Value epsilon,
                                  Value* output) {
    const auto row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    const Value* in = input + row * cols;
    Value mean = 0.0f;
    for (std::size_t col = 0; col < cols; ++col) {
        mean += in[col];
    }
    mean /= static_cast<Value>(cols);

    Value variance = 0.0f;
    for (std::size_t col = 0; col < cols; ++col) {
        const auto centered = in[col] - mean;
        variance += centered * centered;
    }
    variance /= static_cast<Value>(cols);
    const auto inverse_stddev = 1.0f / sqrtf(variance + epsilon);

    for (std::size_t col = 0; col < cols; ++col) {
        output[row * cols + col] = (in[col] - mean) * inverse_stddev * gamma[col] + beta[col];
    }
}

__global__ void gelu_kernel(const Value* input, std::size_t size, Value* output) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        constexpr Value kGeluScale = 0.7978845608028654f;
        constexpr Value kGeluCubic = 0.044715f;
        const auto x = input[index];
        output[index] = 0.5f * x * (1.0f + tanhf(kGeluScale * (x + kGeluCubic * x * x * x)));
    }
}

__global__ void linear_kernel(const Value* input,
                              std::size_t rows,
                              std::size_t input_dim,
                              const Value* weights,
                              const Value* bias,
                              bool has_bias,
                              std::size_t output_dim,
                              Value* output) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto total = rows * output_dim;
    if (index >= total) {
        return;
    }
    const auto row = index / output_dim;
    const auto out_col = index % output_dim;
    Value sum = has_bias ? bias[out_col] : 0.0f;
    for (std::size_t in_col = 0; in_col < input_dim; ++in_col) {
        sum += input[row * input_dim + in_col] * weights[out_col * input_dim + in_col];
    }
    output[index] = sum;
}

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

Value* device_copy(const std::vector<Value>& host, const char* label) {
    Value* device = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&device), host.size() * sizeof(Value)),
               label);
    try {
        check_cuda(cudaMemcpy(device, host.data(), host.size() * sizeof(Value), cudaMemcpyHostToDevice),
                   label);
    } catch (...) {
        cudaFree(device);
        throw;
    }
    return device;
}

} // namespace

Tensor2D layer_norm_cuda(const Tensor2D& input,
                         const std::vector<Value>& gamma,
                         const std::vector<Value>& beta,
                         Value epsilon) {
    if (input.cols == 0) {
        throw std::invalid_argument("layer_norm requires at least one column");
    }
    if (gamma.size() != input.cols || beta.size() != input.cols) {
        throw std::invalid_argument("layer_norm gamma and beta must match input width");
    }
    if (epsilon < 0.0f) {
        throw std::invalid_argument("layer_norm epsilon must be non-negative");
    }

    Tensor2D output(input.rows, input.cols, std::vector<Value>(input.values.size(), 0.0f));
    if (input.rows == 0) {
        return output;
    }

    Value* device_input = nullptr;
    Value* device_gamma = nullptr;
    Value* device_beta = nullptr;
    Value* device_output = nullptr;
    try {
        device_input = device_copy(input.values, "cudaMalloc layer_norm input");
        device_gamma = device_copy(gamma, "cudaMalloc layer_norm gamma");
        device_beta = device_copy(beta, "cudaMalloc layer_norm beta");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_output), output.values.size() * sizeof(Value)),
                   "cudaMalloc layer_norm output");

        constexpr int kBlockSize = 256;
        const auto blocks = static_cast<int>((input.rows + kBlockSize - 1) / kBlockSize);
        layer_norm_kernel<<<blocks, kBlockSize>>>(device_input, input.rows, input.cols, device_gamma,
                                                  device_beta, epsilon, device_output);
        check_cuda(cudaGetLastError(), "layer_norm_kernel launch");
        check_cuda(cudaDeviceSynchronize(), "layer_norm_kernel sync");
        check_cuda(cudaMemcpy(output.values.data(), device_output, output.values.size() * sizeof(Value),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy layer_norm output");
    } catch (...) {
        cudaFree(device_input);
        cudaFree(device_gamma);
        cudaFree(device_beta);
        cudaFree(device_output);
        throw;
    }
    cudaFree(device_input);
    cudaFree(device_gamma);
    cudaFree(device_beta);
    cudaFree(device_output);
    return output;
}

Tensor2D gelu_cuda(const Tensor2D& input) {
    Tensor2D output(input.rows, input.cols, std::vector<Value>(input.values.size(), 0.0f));
    if (input.values.empty()) {
        return output;
    }

    Value* device_input = nullptr;
    Value* device_output = nullptr;
    try {
        device_input = device_copy(input.values, "cudaMalloc gelu input");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_output), output.values.size() * sizeof(Value)),
                   "cudaMalloc gelu output");

        constexpr int kBlockSize = 256;
        const auto blocks = static_cast<int>((input.values.size() + kBlockSize - 1) / kBlockSize);
        gelu_kernel<<<blocks, kBlockSize>>>(device_input, input.values.size(), device_output);
        check_cuda(cudaGetLastError(), "gelu_kernel launch");
        check_cuda(cudaDeviceSynchronize(), "gelu_kernel sync");
        check_cuda(cudaMemcpy(output.values.data(), device_output, output.values.size() * sizeof(Value),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy gelu output");
    } catch (...) {
        cudaFree(device_input);
        cudaFree(device_output);
        throw;
    }
    cudaFree(device_input);
    cudaFree(device_output);
    return output;
}

Tensor2D linear_cuda(const CpuLinear& linear, const Tensor2D& input) {
    if (input.cols != linear.input_dim()) {
        throw std::invalid_argument("linear input width does not match input_dim");
    }

    const auto output_dim = linear.output_dim();
    const bool has_bias = !linear.bias().empty();
    Tensor2D output(input.rows, output_dim, std::vector<Value>(input.rows * output_dim, 0.0f));
    if (input.rows == 0) {
        return output;
    }

    Value* device_input = nullptr;
    Value* device_weights = nullptr;
    Value* device_bias = nullptr;
    Value* device_output = nullptr;
    try {
        device_input = device_copy(input.values, "cudaMalloc linear input");
        device_weights = device_copy(linear.weights(), "cudaMalloc linear weights");
        if (has_bias) {
            device_bias = device_copy(linear.bias(), "cudaMalloc linear bias");
        }
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_output), output.values.size() * sizeof(Value)),
                   "cudaMalloc linear output");

        constexpr int kBlockSize = 256;
        const auto blocks = static_cast<int>((output.values.size() + kBlockSize - 1) / kBlockSize);
        linear_kernel<<<blocks, kBlockSize>>>(device_input, input.rows, linear.input_dim(),
                                              device_weights, device_bias, has_bias, output_dim,
                                              device_output);
        check_cuda(cudaGetLastError(), "linear_kernel launch");
        check_cuda(cudaDeviceSynchronize(), "linear_kernel sync");
        check_cuda(cudaMemcpy(output.values.data(), device_output, output.values.size() * sizeof(Value),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy linear output");
    } catch (...) {
        cudaFree(device_input);
        cudaFree(device_weights);
        cudaFree(device_bias);
        cudaFree(device_output);
        throw;
    }
    cudaFree(device_input);
    cudaFree(device_weights);
    cudaFree(device_bias);
    cudaFree(device_output);
    return output;
}

} // namespace gptcu
