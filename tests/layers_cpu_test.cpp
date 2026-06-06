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

void test_linear_projection() {
    const gptcu::Tensor2D input(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
    const gptcu::CpuLinear linear(2, 3, {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f},
                                  {0.5f, -0.5f, 1.0f});
    const auto output = linear.forward(input);
    expect_close(output.values, {1.5f, 1.5f, 4.0f, 3.5f, 3.5f, 8.0f}, "linear projection");
}

void test_layer_norm_and_residual_add() {
    const gptcu::Tensor2D input(1, 2, {1.0f, 3.0f});
    const auto normalized = gptcu::layer_norm(input, {1.0f, 1.0f}, {0.0f, 0.0f}, 0.0f);
    expect_close(normalized.values, {-1.0f, 1.0f}, "layer norm without epsilon");

    const auto residual = gptcu::residual_add(input, gptcu::Tensor2D(1, 2, {0.5f, -0.5f}));
    expect_close(residual.values, {1.5f, 2.5f}, "residual add");
}

void test_gelu_and_feed_forward() {
    const gptcu::Tensor2D input(1, 2, {0.0f, 1.0f});
    const auto activated = gptcu::gelu(input);
    expect_close(activated.values, {0.0f, 0.841192f}, "gelu activation", 1.0e-4f);

    const gptcu::CpuFeedForward feed_forward(
        gptcu::CpuLinear(2, 2, {1.0f, 0.0f, 0.0f, 1.0f}),
        gptcu::CpuLinear(2, 1, {1.0f, 1.0f}));
    expect_close(feed_forward.forward(input).values, {0.841192f}, "feed-forward block", 1.0e-4f);
}

void test_causal_self_attention() {
    const gptcu::Tensor2D query(2, 1, {0.0f, 0.0f});
    const gptcu::Tensor2D key(2, 1, {0.0f, 0.0f});
    const gptcu::Tensor2D value(2, 1, {2.0f, 4.0f});
    const auto output = gptcu::causal_self_attention(query, key, value);
    expect_close(output.values, {2.0f, 3.0f}, "causal attention uses only current and previous rows");
}

void test_invalid_shapes() {
    expect_throws([]() { gptcu::Tensor2D(2, 2, {1.0f, 2.0f, 3.0f}); }, "tensor shape validation");
    expect_throws([]() {
        gptcu::residual_add(gptcu::Tensor2D(1, 2, {1.0f, 2.0f}), gptcu::Tensor2D(2, 1, {1.0f, 2.0f}));
    }, "residual shape validation");
    expect_throws([]() {
        gptcu::CpuLinear(3, 1, {1.0f, 2.0f}).forward(gptcu::Tensor2D(1, 2, {1.0f, 2.0f}));
    }, "linear input width validation");
}

} // namespace

int main() {
    try {
        test_linear_projection();
        test_layer_norm_and_residual_add();
        test_gelu_and_feed_forward();
        test_causal_self_attention();
        test_invalid_shapes();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}