#include "simple_transformer_reference.h"

#include <cmath>
#include <iostream>
#include <vector>

namespace {

std::vector<float> expectedLogits() {
  return {
      -0.01645244f, 0.01707258f,  -0.01713129f, 0.01662662f,  -0.01557519f,
      0.01401156f,  -0.01198715f, 0.00956854f,  -0.00683527f, 0.00387721f,
      -0.00079165f, -0.00231994f, 0.01102682f,  -0.01341424f, 0.01536053f,
      -0.01680169f, 0.01769031f,  -0.01799718f, 0.01771221f,  -0.01684476f,
      0.01542337f,  -0.01349477f, 0.01112240f,  -0.00838426f, 0.03494637f,
      -0.03635882f, 0.03657559f,  -0.03558956f, 0.03343316f,  -0.03017729f,
      0.02592904f,  -0.02082809f, 0.01504221f,  -0.00876166f, 0.00219298f,
      0.00444782f,  -0.01580957f, 0.02060716f,  -0.02472708f, 0.02803384f,
      -0.03041869f, 0.03180322f,  -0.03214188f, 0.03142355f,  -0.02967184f,
      0.02694437f,  -0.02333081f, 0.01895002f,
  };
}

}  // namespace

int main() {
  const auto demo = simple_transformer_reference::makeDemoInputs();
  const auto actual = simple_transformer_reference::computeReferenceLogits(demo);
  const auto expected = expectedLogits();

  if (actual.size() != expected.size()) {
    std::cerr << "Unexpected logits size: " << actual.size() << " vs "
              << expected.size() << '\n';
    return 1;
  }

  const float max_diff = simple_transformer_reference::maxAbsDiff(actual, expected);
  if (!std::isfinite(max_diff) || max_diff > 1.0e-6f) {
    std::cerr << "Reference regression failed, max diff = " << max_diff << '\n';
    return 1;
  }

  std::cout << "simple_transformer_reference_test passed, max diff = "
            << max_diff << '\n';
  return 0;
}