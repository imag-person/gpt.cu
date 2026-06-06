#include "gptcu/tokenizer_bpe_cuda.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

template <typename T>
void expect_eq(const T& actual, const T& expected, const std::string& label) {
    if (actual != expected) {
        throw std::runtime_error("failed assertion: " + label);
    }
}

void test_cuda_byte_encoding() {
    const gptcu::CpuBpeTokenizer tokenizer({{"a", 7}, {"b", 8}, {"<unk>", 99}}, {}, 99);
    const auto table = gptcu::make_byte_token_table(tokenizer);
    expect_eq(gptcu::encode_bytes_cuda("aba?", table),
              std::vector<gptcu::CpuBpeTokenizer::TokenId>({7, 8, 7, 99}),
              "CUDA byte token encoding");
}

} // namespace

int main() {
    try {
        test_cuda_byte_encoding();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}