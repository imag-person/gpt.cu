#include "gptcu/embedding_lookup_cuda.hpp"

#include "gptcu/embedding_lookup_cpu.hpp"

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

template <typename Fn>
void expect_throws(Fn&& fn, const std::string& label) {
    try {
        fn();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error("failed assertion: " + label);
}

void test_cuda_matches_cpu_gather() {
    const gptcu::CpuEmbeddingTable table(3, {0.1f, 0.2f, 0.3f, 1.0f, 1.1f, 1.2f, -2.0f, -2.1f, -2.2f});
    const std::vector<gptcu::CpuEmbeddingTable::TokenId> ids{2, 0, 1, 2};

    expect_eq(gptcu::embedding_lookup_cuda(table, ids), table.lookup_many(ids),
              "CUDA gather matches CPU lookup_many bit-for-bit");
    expect_eq(gptcu::embedding_lookup_cuda(table, {}),
              std::vector<gptcu::CpuEmbeddingTable::Value>{}, "CUDA gather of empty ids is empty");
}

void test_invalid_token_id() {
    const gptcu::CpuEmbeddingTable table(2, {0.0f, 0.1f, 1.0f, 1.1f});
    expect_throws([&]() { static_cast<void>(gptcu::embedding_lookup_cuda(table, {-1})); },
                  "negative token id rejected");
    expect_throws([&]() { static_cast<void>(gptcu::embedding_lookup_cuda(table, {2})); },
                  "token id past vocab rejected");
}

} // namespace

int main() {
    try {
        test_cuda_matches_cpu_gather();
        test_invalid_token_id();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
