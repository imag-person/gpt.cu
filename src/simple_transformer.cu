#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t status = (call);                                               \
    if (status != cudaSuccess) {                                               \
      std::cerr << "CUDA error at " << __FILE__ << ':' << __LINE__ << ": "    \
                << cudaGetErrorString(status) << std::endl;                   \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (false)

namespace {

constexpr int kSeqLen = 4;
constexpr int kModelDim = 8;
constexpr int kHiddenDim = 16;
constexpr int kVocabSize = 12;
constexpr int kMaxSeqLen = 16;

__global__ void embedKernel(const int* tokens, const float* token_embedding,
                            const float* position_embedding, float* x,
                            int seq_len, int dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = seq_len * dim;
  if (idx >= total) return;

  int token_pos = idx / dim;
  int channel = idx % dim;
  int token_id = tokens[token_pos];
  x[idx] = token_embedding[token_id * dim + channel] +
           position_embedding[token_pos * dim + channel];
}

__global__ void linearKernel(const float* input, const float* weight,
                             const float* bias, float* output, int rows,
                             int in_dim, int out_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * out_dim;
  if (idx >= total) return;

  int row = idx / out_dim;
  int out_col = idx % out_dim;
  float sum = bias[out_col];
  for (int in_col = 0; in_col < in_dim; ++in_col) {
    sum += input[row * in_dim + in_col] * weight[out_col * in_dim + in_col];
  }
  output[idx] = sum;
}

__global__ void attentionKernel(const float* q, const float* k, const float* v,
                                float* out, int seq_len, int dim) {
  int query_pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (query_pos >= seq_len) return;

  float scores[kMaxSeqLen];
  float max_score = -1.0e30f;
  float scale = rsqrtf(static_cast<float>(dim));

  for (int key_pos = 0; key_pos < seq_len; ++key_pos) {
    float dot = 0.0f;
    for (int channel = 0; channel < dim; ++channel) {
      dot += q[query_pos * dim + channel] * k[key_pos * dim + channel];
    }
    scores[key_pos] = dot * scale;
    max_score = fmaxf(max_score, scores[key_pos]);
  }

  float normalizer = 0.0f;
  for (int key_pos = 0; key_pos < seq_len; ++key_pos) {
    scores[key_pos] = expf(scores[key_pos] - max_score);
    normalizer += scores[key_pos];
  }

  for (int channel = 0; channel < dim; ++channel) {
    float value = 0.0f;
    for (int key_pos = 0; key_pos < seq_len; ++key_pos) {
      float probability = scores[key_pos] / normalizer;
      value += probability * v[key_pos * dim + channel];
    }
    out[query_pos * dim + channel] = value;
  }
}

__global__ void reluKernel(float* values, int total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total) values[idx] = fmaxf(values[idx], 0.0f);
}

__global__ void addKernel(const float* lhs, const float* rhs, float* out,
                          int total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total) out[idx] = lhs[idx] + rhs[idx];
}

std::vector<float> makeValues(std::size_t count, float scale) {
  std::vector<float> values(count);
  for (std::size_t i = 0; i < count; ++i) {
    values[i] = scale * std::sin(static_cast<float>(i + 1) * 0.37f);
  }
  return values;
}

template <typename T>
T* copyToDevice(const std::vector<T>& host) {
  T* device = nullptr;
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&device),
                        host.size() * sizeof(T)));
  CHECK_CUDA(cudaMemcpy(device, host.data(), host.size() * sizeof(T),
                        cudaMemcpyHostToDevice));
  return device;
}

float* allocateDevice(std::size_t count) {
  float* device = nullptr;
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&device),
                        count * sizeof(float)));
  return device;
}

std::vector<float> cpuLinear(const std::vector<float>& input,
                             const std::vector<float>& weight,
                             const std::vector<float>& bias, int rows,
                             int in_dim, int out_dim) {
  std::vector<float> output(rows * out_dim, 0.0f);
  for (int row = 0; row < rows; ++row) {
    for (int out_col = 0; out_col < out_dim; ++out_col) {
      float sum = bias[out_col];
      for (int in_col = 0; in_col < in_dim; ++in_col) {
        sum += input[row * in_dim + in_col] * weight[out_col * in_dim + in_col];
      }
      output[row * out_dim + out_col] = sum;
    }
  }
  return output;
}

std::vector<float> cpuAttention(const std::vector<float>& q,
                                const std::vector<float>& k,
                                const std::vector<float>& v) {
  std::vector<float> output(kSeqLen * kModelDim, 0.0f);
  float scale = 1.0f / std::sqrt(static_cast<float>(kModelDim));

  for (int query_pos = 0; query_pos < kSeqLen; ++query_pos) {
    std::array<float, kSeqLen> scores{};
    float max_score = -1.0e30f;
    for (int key_pos = 0; key_pos < kSeqLen; ++key_pos) {
      float dot = 0.0f;
      for (int channel = 0; channel < kModelDim; ++channel) {
        dot += q[query_pos * kModelDim + channel] *
               k[key_pos * kModelDim + channel];
      }
      scores[key_pos] = dot * scale;
      max_score = std::max(max_score, scores[key_pos]);
    }

    float normalizer = 0.0f;
    for (float& score : scores) {
      score = std::exp(score - max_score);
      normalizer += score;
    }

    for (int channel = 0; channel < kModelDim; ++channel) {
      for (int key_pos = 0; key_pos < kSeqLen; ++key_pos) {
        output[query_pos * kModelDim + channel] +=
            (scores[key_pos] / normalizer) * v[key_pos * kModelDim + channel];
      }
    }
  }
  return output;
}

void addInPlace(std::vector<float>& lhs, const std::vector<float>& rhs) {
  for (std::size_t i = 0; i < lhs.size(); ++i) lhs[i] += rhs[i];
}

float maxAbsDiff(const std::vector<float>& lhs, const std::vector<float>& rhs) {
  float max_diff = 0.0f;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    max_diff = std::max(max_diff, std::fabs(lhs[i] - rhs[i]));
  }
  return max_diff;
}

}  // namespace

int main() {
  static_assert(kSeqLen <= kMaxSeqLen, "attentionKernel stack buffer too small");

  std::vector<int> tokens = {1, 4, 2, 7};
  auto token_embedding = makeValues(kVocabSize * kModelDim, 0.08f);
  auto position_embedding = makeValues(kSeqLen * kModelDim, 0.03f);
  auto wq = makeValues(kModelDim * kModelDim, 0.06f);
  auto wk = makeValues(kModelDim * kModelDim, 0.05f);
  auto wv = makeValues(kModelDim * kModelDim, 0.07f);
  auto wo = makeValues(kModelDim * kModelDim, 0.04f);
  auto w1 = makeValues(kHiddenDim * kModelDim, 0.09f);
  auto w2 = makeValues(kModelDim * kHiddenDim, 0.05f);
  auto lm_head = makeValues(kVocabSize * kModelDim, 0.08f);
  std::vector<float> zero_model_bias(kModelDim, 0.0f);
  std::vector<float> zero_hidden_bias(kHiddenDim, 0.0f);
  std::vector<float> zero_vocab_bias(kVocabSize, 0.0f);

  std::vector<float> cpu_x(kSeqLen * kModelDim);
  for (int pos = 0; pos < kSeqLen; ++pos) {
    for (int channel = 0; channel < kModelDim; ++channel) {
      cpu_x[pos * kModelDim + channel] =
          token_embedding[tokens[pos] * kModelDim + channel] +
          position_embedding[pos * kModelDim + channel];
    }
  }
  auto cpu_q = cpuLinear(cpu_x, wq, zero_model_bias, kSeqLen, kModelDim, kModelDim);
  auto cpu_k = cpuLinear(cpu_x, wk, zero_model_bias, kSeqLen, kModelDim, kModelDim);
  auto cpu_v = cpuLinear(cpu_x, wv, zero_model_bias, kSeqLen, kModelDim, kModelDim);
  auto cpu_attn = cpuAttention(cpu_q, cpu_k, cpu_v);
  auto cpu_projected = cpuLinear(cpu_attn, wo, zero_model_bias, kSeqLen, kModelDim, kModelDim);
  addInPlace(cpu_x, cpu_projected);
  auto cpu_hidden = cpuLinear(cpu_x, w1, zero_hidden_bias, kSeqLen, kModelDim, kHiddenDim);
  for (float& value : cpu_hidden) value = std::max(value, 0.0f);
  auto cpu_ffn = cpuLinear(cpu_hidden, w2, zero_model_bias, kSeqLen, kHiddenDim, kModelDim);
  addInPlace(cpu_x, cpu_ffn);
  auto cpu_logits = cpuLinear(cpu_x, lm_head, zero_vocab_bias, kSeqLen, kModelDim, kVocabSize);

  int* d_tokens = copyToDevice(tokens);
  float* d_token_embedding = copyToDevice(token_embedding);
  float* d_position_embedding = copyToDevice(position_embedding);
  float* d_wq = copyToDevice(wq);
  float* d_wk = copyToDevice(wk);
  float* d_wv = copyToDevice(wv);
  float* d_wo = copyToDevice(wo);
  float* d_w1 = copyToDevice(w1);
  float* d_w2 = copyToDevice(w2);
  float* d_lm_head = copyToDevice(lm_head);
  float* d_zero_model_bias = copyToDevice(zero_model_bias);
  float* d_zero_hidden_bias = copyToDevice(zero_hidden_bias);
  float* d_zero_vocab_bias = copyToDevice(zero_vocab_bias);
  float* d_x = allocateDevice(kSeqLen * kModelDim);
  float* d_q = allocateDevice(kSeqLen * kModelDim);
  float* d_k = allocateDevice(kSeqLen * kModelDim);
  float* d_v = allocateDevice(kSeqLen * kModelDim);
  float* d_attn = allocateDevice(kSeqLen * kModelDim);
  float* d_projected = allocateDevice(kSeqLen * kModelDim);
  float* d_hidden = allocateDevice(kSeqLen * kHiddenDim);
  float* d_ffn = allocateDevice(kSeqLen * kModelDim);
  float* d_logits = allocateDevice(kSeqLen * kVocabSize);

  int model_values = kSeqLen * kModelDim;
  int hidden_values = kSeqLen * kHiddenDim;
  int vocab_values = kSeqLen * kVocabSize;
  int threads = 128;
  embedKernel<<<(model_values + threads - 1) / threads, threads>>>(
      d_tokens, d_token_embedding, d_position_embedding, d_x, kSeqLen, kModelDim);
  linearKernel<<<(model_values + threads - 1) / threads, threads>>>(
      d_x, d_wq, d_zero_model_bias, d_q, kSeqLen, kModelDim, kModelDim);
  linearKernel<<<(model_values + threads - 1) / threads, threads>>>(
      d_x, d_wk, d_zero_model_bias, d_k, kSeqLen, kModelDim, kModelDim);
  linearKernel<<<(model_values + threads - 1) / threads, threads>>>(
      d_x, d_wv, d_zero_model_bias, d_v, kSeqLen, kModelDim, kModelDim);
  attentionKernel<<<1, kSeqLen>>>(d_q, d_k, d_v, d_attn, kSeqLen, kModelDim);
  linearKernel<<<(model_values + threads - 1) / threads, threads>>>(
      d_attn, d_wo, d_zero_model_bias, d_projected, kSeqLen, kModelDim, kModelDim);
  addKernel<<<(model_values + threads - 1) / threads, threads>>>(
      d_x, d_projected, d_x, model_values);
  linearKernel<<<(hidden_values + threads - 1) / threads, threads>>>(
      d_x, d_w1, d_zero_hidden_bias, d_hidden, kSeqLen, kModelDim, kHiddenDim);
  reluKernel<<<(hidden_values + threads - 1) / threads, threads>>>(d_hidden,
                                                                  hidden_values);
  linearKernel<<<(model_values + threads - 1) / threads, threads>>>(
      d_hidden, d_w2, d_zero_model_bias, d_ffn, kSeqLen, kHiddenDim, kModelDim);
  addKernel<<<(model_values + threads - 1) / threads, threads>>>(d_x, d_ffn, d_x,
                                                                 model_values);
  linearKernel<<<(vocab_values + threads - 1) / threads, threads>>>(
      d_x, d_lm_head, d_zero_vocab_bias, d_logits, kSeqLen, kModelDim,
      kVocabSize);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<float> gpu_logits(kSeqLen * kVocabSize);
  CHECK_CUDA(cudaMemcpy(gpu_logits.data(), d_logits,
                        gpu_logits.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  float max_diff = maxAbsDiff(cpu_logits, gpu_logits);
  std::cout << "Simple CUDA transformer block demo\n";
  std::cout << "Max |CPU - GPU| logits diff: " << max_diff << "\n";
  std::cout << "Last-token logits:";
  for (int vocab = 0; vocab < kVocabSize; ++vocab) {
    std::cout << ' ' << std::fixed << std::setprecision(5)
              << gpu_logits[(kSeqLen - 1) * kVocabSize + vocab];
  }
  std::cout << '\n';

  CHECK_CUDA(cudaFree(d_tokens));
  CHECK_CUDA(cudaFree(d_token_embedding));
  CHECK_CUDA(cudaFree(d_position_embedding));
  CHECK_CUDA(cudaFree(d_wq));
  CHECK_CUDA(cudaFree(d_wk));
  CHECK_CUDA(cudaFree(d_wv));
  CHECK_CUDA(cudaFree(d_wo));
  CHECK_CUDA(cudaFree(d_w1));
  CHECK_CUDA(cudaFree(d_w2));
  CHECK_CUDA(cudaFree(d_lm_head));
  CHECK_CUDA(cudaFree(d_zero_model_bias));
  CHECK_CUDA(cudaFree(d_zero_hidden_bias));
  CHECK_CUDA(cudaFree(d_zero_vocab_bias));
  CHECK_CUDA(cudaFree(d_x));
  CHECK_CUDA(cudaFree(d_q));
  CHECK_CUDA(cudaFree(d_k));
  CHECK_CUDA(cudaFree(d_v));
  CHECK_CUDA(cudaFree(d_attn));
  CHECK_CUDA(cudaFree(d_projected));
  CHECK_CUDA(cudaFree(d_hidden));
  CHECK_CUDA(cudaFree(d_ffn));
  CHECK_CUDA(cudaFree(d_logits));

  if (!std::isfinite(max_diff) || max_diff > 1.0e-4f) {
    std::cerr << "Validation failed: GPU result diverged from CPU reference\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
