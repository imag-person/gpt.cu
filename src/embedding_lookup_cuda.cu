#include "gptcu/embedding_lookup_cuda.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace gptcu {
namespace {

using TokenId = CpuEmbeddingTable::TokenId;
using Value = CpuEmbeddingTable::Value;

__global__ void gather_kernel(const Value* weights,
                              const TokenId* token_ids,
                              std::size_t n_tokens,
                              std::size_t dim,
                              Value* output) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto total = n_tokens * dim;
    if (index < total) {
        const auto token = index / dim;
        const auto col = index % dim;
        output[index] = weights[static_cast<std::size_t>(token_ids[token]) * dim + col];
    }
}

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

} // namespace

std::vector<Value> embedding_lookup_cuda(const CpuEmbeddingTable& table,
                                         const std::vector<TokenId>& token_ids) {
    if (token_ids.empty()) {
        return {};
    }

    const auto dim = table.embedding_dim();
    const auto vocab_size = table.vocab_size();
    for (const TokenId token_id : token_ids) {
        if (token_id < 0 || static_cast<std::size_t>(token_id) >= vocab_size) {
            throw std::out_of_range("token id is out of range for embedding lookup");
        }
    }

    const auto& weights = table.weights();
    std::vector<Value> output(token_ids.size() * dim);

    Value* device_weights = nullptr;
    TokenId* device_ids = nullptr;
    Value* device_output = nullptr;

    try {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_weights), weights.size() * sizeof(Value)),
                   "cudaMalloc weights");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_ids), token_ids.size() * sizeof(TokenId)),
                   "cudaMalloc ids");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_output), output.size() * sizeof(Value)),
                   "cudaMalloc output");

        check_cuda(cudaMemcpy(device_weights, weights.data(), weights.size() * sizeof(Value),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy weights");
        check_cuda(cudaMemcpy(device_ids, token_ids.data(), token_ids.size() * sizeof(TokenId),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy ids");

        constexpr int kBlockSize = 256;
        const auto blocks = static_cast<int>((output.size() + kBlockSize - 1) / kBlockSize);
        gather_kernel<<<blocks, kBlockSize>>>(device_weights, device_ids, token_ids.size(), dim,
                                              device_output);
        check_cuda(cudaGetLastError(), "gather_kernel launch");
        check_cuda(cudaDeviceSynchronize(), "gather_kernel sync");

        check_cuda(cudaMemcpy(output.data(), device_output, output.size() * sizeof(Value),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy output");
    } catch (...) {
        cudaFree(device_weights);
        cudaFree(device_ids);
        cudaFree(device_output);
        throw;
    }

    cudaFree(device_weights);
    cudaFree(device_ids);
    cudaFree(device_output);
    return output;
}

} // namespace gptcu
