#include "gptcu/layers_cuda.hpp"

#include "gptcu/layers_cpu.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void expect_close(const std::vector<float>& actual,
                  const std::vector<float>& expected,
                  const std::string& label,
                  float tolerance = 1.0e-5f) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error("failed assertion: " + label + " size mismatch");
    }
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (std::fabs(actual[i] - expected[i]) > tolerance) {
            throw std::runtime_error("failed assertion: " + label);
        }
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

void test_layer_norm_matches_cpu() {
    const gptcu::Tensor2D input(2, 3, {1.0f, 3.0f, -2.0f, 0.5f, 0.0f, 4.0f});
    const std::vector<float> gamma{1.0f, 0.5f, 2.0f};
    const std::vector<float> beta{0.0f, 1.0f, -1.0f};
    const auto cpu = gptcu::layer_norm(input, gamma, beta);
    const auto cuda = gptcu::layer_norm_cuda(input, gamma, beta);
    expect_close(cuda.values, cpu.values, "CUDA layer_norm matches CPU");
}

void test_gelu_matches_cpu() {
    const gptcu::Tensor2D input(2, 2, {-1.0f, 0.0f, 1.0f, 2.5f});
    expect_close(gptcu::gelu_cuda(input).values, gptcu::gelu(input).values, "CUDA gelu matches CPU");
}

void test_linear_matches_cpu() {
    const gptcu::Tensor2D input(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
    const gptcu::CpuLinear linear(2, 3, {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f}, {0.5f, -0.5f, 1.0f});
    expect_close(gptcu::linear_cuda(linear, input).values, linear.forward(input).values,
                 "CUDA linear matches CPU");

    const gptcu::CpuLinear no_bias(2, 1, {2.0f, -1.0f});
    expect_close(gptcu::linear_cuda(no_bias, input).values, no_bias.forward(input).values,
                 "CUDA linear without bias matches CPU");
}

void test_invalid_shapes() {
    const gptcu::Tensor2D input(1, 2, {1.0f, 2.0f});
    expect_throws([&]() { static_cast<void>(gptcu::layer_norm_cuda(input, {1.0f}, {0.0f})); },
                  "layer_norm gamma width validation");
    expect_throws([&]() {
        static_cast<void>(gptcu::linear_cuda(gptcu::CpuLinear(3, 1, {1.0f, 2.0f, 3.0f}), input));
    }, "linear input width validation");
}

} // namespace

int main() {
    try {
        test_layer_norm_matches_cpu();
        test_gelu_matches_cpu();
        test_linear_matches_cpu();
        test_invalid_shapes();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
