#include "simple_transformer_reference.h"

#include <cmath>
#include <iostream>
#include <vector>

namespace {

std::vector<float> expectedProbabilities() {
  return {
      0.08275954f, 0.08402319f, 0.08261360f, 0.08412480f, 0.08256303f,
      0.08412448f, 0.08261423f, 0.08402226f, 0.08276071f, 0.08383184f,
      0.08298379f, 0.08357853f, 0.08514541f, 0.08111084f, 0.08574300f,
      0.08069278f, 0.08601534f, 0.08060645f, 0.08592376f, 0.08086205f,
      0.08548132f, 0.08142921f, 0.08475046f, 0.08223947f, 0.08691728f,
      0.07932428f, 0.08703401f, 0.07945991f, 0.08662571f, 0.08006144f,
      0.08575235f, 0.08105969f, 0.08453970f, 0.08233615f, 0.08315682f,
      0.08373269f, 0.08273365f, 0.08413337f, 0.08222622f, 0.08458877f,
      0.08185975f, 0.08487587f, 0.08167998f, 0.08495538f, 0.08170912f,
      0.08481634f, 0.08194359f, 0.08447788f,
  };
}

}  // namespace

int main() {
  const auto demo = simple_transformer_reference::makeDemoInputs();
  const auto actual =
      simple_transformer_reference::computeReferenceProbabilities(demo);
  const auto expected = expectedProbabilities();

  if (actual.size() != expected.size()) {
    std::cerr << "Unexpected probability size: " << actual.size() << " vs "
              << expected.size() << '\n';
    return 1;
  }

  const float max_diff =
      simple_transformer_reference::maxAbsDiff(actual, expected);
  if (!std::isfinite(max_diff) || max_diff > 1.0e-6f) {
    std::cerr << "Reference probability regression failed, max diff = "
              << max_diff << '\n';
    return 1;
  }

  std::cout << "simple_transformer_reference_test passed, max diff = "
            << max_diff << '\n';
  return 0;
}
