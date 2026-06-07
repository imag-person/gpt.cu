#include "gptcu/tokenizer_bpe_cuda.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace gptcu {
namespace {

using TokenId = CpuBpeTokenizer::TokenId;

__global__ void encode_bytes_kernel(const unsigned char* input,
                                    std::size_t size,
                                    const TokenId* byte_tokens,
                                    TokenId* output) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        output[index] = byte_tokens[input[index]];
    }
}

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

} // namespace

ByteTokenTable make_byte_token_table(const CpuBpeTokenizer& tokenizer) {
    ByteTokenTable table{};
    for (std::size_t byte = 0; byte < table.size(); ++byte) {
        const std::string token(1, static_cast<char>(byte));
        table[byte] = tokenizer.token_id_for(token);
    }
    return table;
}

std::vector<TokenId> encode_bytes_cuda(const std::string& text, const ByteTokenTable& byte_tokens) {
    if (text.empty()) {
        return {};
    }

    unsigned char* device_input = nullptr;
    TokenId* device_table = nullptr;
    TokenId* device_output = nullptr;
    std::vector<TokenId> output(text.size());

    try {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_input), text.size()), "cudaMalloc input");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_table), byte_tokens.size() * sizeof(TokenId)),
                   "cudaMalloc table");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_output), output.size() * sizeof(TokenId)),
                   "cudaMalloc output");

        check_cuda(cudaMemcpy(device_input, text.data(), text.size(), cudaMemcpyHostToDevice),
                   "cudaMemcpy input");
        check_cuda(cudaMemcpy(device_table, byte_tokens.data(), byte_tokens.size() * sizeof(TokenId),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy table");

        constexpr int kBlockSize = 256;
        const auto blocks = static_cast<int>((text.size() + kBlockSize - 1) / kBlockSize);
        encode_bytes_kernel<<<blocks, kBlockSize>>>(device_input, text.size(), device_table, device_output);
        check_cuda(cudaGetLastError(), "encode_bytes_kernel launch");
        check_cuda(cudaDeviceSynchronize(), "encode_bytes_kernel sync");

        check_cuda(cudaMemcpy(output.data(), device_output, output.size() * sizeof(TokenId),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy output");
    } catch (...) {
        cudaFree(device_input);
        cudaFree(device_table);
        cudaFree(device_output);
        throw;
    }

    cudaFree(device_input);
    cudaFree(device_table);
    cudaFree(device_output);
    return output;
}

} // namespace gptcu