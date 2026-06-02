#include <cuda_runtime.h>

#ifdef GELU_TEST_MAIN
#include <cmath>
#include <cstdio>
#include <vector>
#endif

namespace {

constexpr int kGeluThreadsPerBlock = 256;
constexpr float kInvSqrt2 = 0.70710678118654752440f;

__device__ __forceinline__ float gelu(float value) {
  return 0.5f * value * (1.0f + erff(value * kInvSqrt2));
}

__global__ void gelu_kernel(const float* input, float* output, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  output[index] = gelu(input[index]);
}

}  // namespace

cudaError_t gelu_forward(const float* input, float* output, int count,
                         cudaStream_t stream) {
  if (input == nullptr || output == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (count <= 0) {
    return cudaSuccess;
  }

  const int blocks = (count + kGeluThreadsPerBlock - 1) / kGeluThreadsPerBlock;
  gelu_kernel<<<blocks, kGeluThreadsPerBlock, 0, stream>>>(input, output, count);
  return cudaGetLastError();
}

#ifdef GELU_TEST_MAIN
namespace {

bool nearly_equal(float lhs, float rhs) {
  return std::fabs(lhs - rhs) <= 1e-6f;
}

bool check_cuda(cudaError_t status, const char* operation) {
  if (status == cudaSuccess) {
    return true;
  }
  std::fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(status));
  return false;
}

std::vector<float> reference_gelu(const std::vector<float>& input) {
  std::vector<float> output(input.size());
  for (size_t index = 0; index < input.size(); ++index) {
    const float value = input[index];
    output[index] = 0.5f * value * (1.0f + std::erf(value * kInvSqrt2));
  }
  return output;
}

}  // namespace

int main() {
  const std::vector<float> input = {
      -5.0f, -3.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 3.0f, 5.0f,
  };
  std::vector<float> output(input.size(), 0.0f);

  float* device_input = nullptr;
  float* device_output = nullptr;
  if (!check_cuda(cudaMalloc(&device_input, input.size() * sizeof(float)),
                  "cudaMalloc(input)")) {
    return 1;
  }
  if (!check_cuda(cudaMalloc(&device_output, output.size() * sizeof(float)),
                  "cudaMalloc(output)")) {
    cudaFree(device_input);
    return 1;
  }
  if (!check_cuda(cudaMemcpy(device_input, input.data(),
                             input.size() * sizeof(float),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy(input)")) {
    cudaFree(device_input);
    cudaFree(device_output);
    return 1;
  }

  cudaError_t status = gelu_forward(device_input, device_output,
                                    static_cast<int>(input.size()), nullptr);
  if (status == cudaSuccess) {
    status = cudaDeviceSynchronize();
  }
  if (status == cudaSuccess) {
    status = cudaMemcpy(output.data(), device_output,
                        output.size() * sizeof(float), cudaMemcpyDeviceToHost);
  }

  cudaFree(device_input);
  cudaFree(device_output);

  if (status != cudaSuccess) {
    std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
    return 1;
  }

  const std::vector<float> expected = reference_gelu(input);
  for (size_t index = 0; index < output.size(); ++index) {
    if (!nearly_equal(output[index], expected[index])) {
      std::fprintf(stderr, "Mismatch at %zu: got %.8f expected %.8f\n", index,
                   output[index], expected[index]);
      return 1;
    }
  }

  std::puts("gelu validation passed");
  return 0;
}
#endif