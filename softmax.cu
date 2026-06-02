#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifdef SOFTMAX_TEST_MAIN
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>
#endif

namespace {

constexpr int kMaxThreadsPerBlock = 1024;

int next_power_of_two(int value) {
  int result = 1;
  while (result < value && result < kMaxThreadsPerBlock) {
    result <<= 1;
  }
  return result;
}

__global__ void softmax_kernel(const float* input, float* output, int rows,
                               int cols) {
  extern __shared__ float scratch[];

  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  const float* row_input = input + row * cols;
  float* row_output = output + row * cols;

  float thread_max = -CUDART_INF_F;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    thread_max = fmaxf(thread_max, row_input[col]);
  }

  scratch[threadIdx.x] = thread_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x],
                                  scratch[threadIdx.x + stride]);
    }
    __syncthreads();
  }

  const float row_max = scratch[0];
  float thread_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    thread_sum += expf(row_input[col] - row_max);
  }

  scratch[threadIdx.x] = thread_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float inv_sum = 1.0f / scratch[0];
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    row_output[col] = expf(row_input[col] - row_max) * inv_sum;
  }
}

__global__ void softmax_kernel_fp16(const __half* input, __half* output,
                                    int rows, int cols) {
  extern __shared__ float scratch[];

  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  const __half* row_input = input + row * cols;
  __half* row_output = output + row * cols;

  float thread_max = -CUDART_INF_F;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    thread_max = fmaxf(thread_max, __half2float(row_input[col]));
  }

  scratch[threadIdx.x] = thread_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x],
                                  scratch[threadIdx.x + stride]);
    }
    __syncthreads();
  }

  const float row_max = scratch[0];
  float thread_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    thread_sum += expf(__half2float(row_input[col]) - row_max);
  }

  scratch[threadIdx.x] = thread_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float inv_sum = 1.0f / scratch[0];
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    row_output[col] =
        __float2half(expf(__half2float(row_input[col]) - row_max) * inv_sum);
  }
}

}  // namespace

cudaError_t softmax_forward(const float* input, float* output, int rows,
                            int cols, cudaStream_t stream) {
  if (input == nullptr || output == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (rows <= 0 || cols <= 0) {
    return cudaSuccess;
  }

  const int threads = next_power_of_two(cols);
  const size_t shared_bytes = static_cast<size_t>(threads) * sizeof(float);
  softmax_kernel<<<rows, threads, shared_bytes, stream>>>(input, output, rows,
                                                          cols);
  return cudaGetLastError();
}

cudaError_t softmax_forward_fp16(const __half* input, __half* output, int rows,
                                 int cols, cudaStream_t stream) {
  if (input == nullptr || output == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (rows <= 0 || cols <= 0) {
    return cudaSuccess;
  }

  const int threads = next_power_of_two(cols);
  const size_t shared_bytes = static_cast<size_t>(threads) * sizeof(float);
  softmax_kernel_fp16<<<rows, threads, shared_bytes, stream>>>(input, output,
                                                               rows, cols);
  return cudaGetLastError();
}

#ifdef SOFTMAX_TEST_MAIN
namespace {

bool nearly_equal(float lhs, float rhs, float tolerance = 1e-5f) {
  return std::fabs(lhs - rhs) <= tolerance;
}

bool check_cuda(cudaError_t status, const char* operation) {
  if (status == cudaSuccess) {
    return true;
  }
  std::fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(status));
  return false;
}

std::vector<float> reference_softmax(const std::vector<float>& input, int rows,
                                     int cols) {
  std::vector<float> output(input.size());
  for (int row = 0; row < rows; ++row) {
    const int offset = row * cols;
    const auto begin = input.begin() + offset;
    const auto end = begin + cols;
    const float row_max = *std::max_element(begin, end);
    float sum = 0.0f;
    for (int col = 0; col < cols; ++col) {
      output[offset + col] = std::exp(input[offset + col] - row_max);
      sum += output[offset + col];
    }
    for (int col = 0; col < cols; ++col) {
      output[offset + col] /= sum;
    }
  }
  return output;
}

}  // namespace

int main() {
  constexpr int rows = 3;
  constexpr int cols = 5;
  const std::vector<float> input = {
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
      -1.0f, -2.0f, -3.0f, -4.0f, -5.0f,
      1000.0f, 1001.0f, 999.0f, 1002.0f, 998.0f,
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

  cudaError_t status = softmax_forward(device_input, device_output, rows, cols,
                                       nullptr);
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

  const std::vector<float> expected = reference_softmax(input, rows, cols);
  for (size_t index = 0; index < output.size(); ++index) {
    if (!nearly_equal(output[index], expected[index])) {
      std::fprintf(stderr, "Mismatch at %zu: got %.8f expected %.8f\n", index,
                   output[index], expected[index]);
      return 1;
    }
  }

  std::vector<__half> half_input(input.size());
  std::vector<__half> half_output(input.size());
  for (size_t index = 0; index < input.size(); ++index) {
    half_input[index] = __float2half(input[index]);
  }

  __half* device_half_input = nullptr;
  __half* device_half_output = nullptr;
  if (!check_cuda(cudaMalloc(&device_half_input,
                             half_input.size() * sizeof(__half)),
                  "cudaMalloc(half input)")) {
    return 1;
  }
  if (!check_cuda(cudaMalloc(&device_half_output,
                             half_output.size() * sizeof(__half)),
                  "cudaMalloc(half output)")) {
    cudaFree(device_half_input);
    return 1;
  }
  if (!check_cuda(cudaMemcpy(device_half_input, half_input.data(),
                             half_input.size() * sizeof(__half),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy(half input)")) {
    cudaFree(device_half_input);
    cudaFree(device_half_output);
    return 1;
  }

  status = softmax_forward_fp16(device_half_input, device_half_output, rows, cols,
                                nullptr);
  if (status == cudaSuccess) {
    status = cudaDeviceSynchronize();
  }
  if (status == cudaSuccess) {
    status = cudaMemcpy(half_output.data(), device_half_output,
                        half_output.size() * sizeof(__half),
                        cudaMemcpyDeviceToHost);
  }

  cudaFree(device_half_input);
  cudaFree(device_half_output);

  if (status != cudaSuccess) {
    std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
    return 1;
  }

  for (size_t index = 0; index < half_output.size(); ++index) {
    const float got = __half2float(half_output[index]);
    if (!nearly_equal(got, expected[index], 1e-3f)) {
      std::fprintf(stderr, "FP16 mismatch at %zu: got %.8f expected %.8f\n",
                   index, got, expected[index]);
      return 1;
    }
  }

  std::puts("softmax validation passed");
  return 0;
}
#endif