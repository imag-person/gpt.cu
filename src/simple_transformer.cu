#include <cuda_runtime.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "simple_transformer_reference.h"

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

  float scores[simple_transformer_reference::kMaxSeqLen];
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

}  // namespace

int main() {
  static_assert(simple_transformer_reference::kSeqLen <=
                    simple_transformer_reference::kMaxSeqLen,
                "attentionKernel stack buffer too small");

  const auto demo = simple_transformer_reference::makeDemoInputs();
  const auto& tokens = demo.tokens;
  const auto& token_embedding = demo.token_embedding;
  const auto& position_embedding = demo.position_embedding;
  const auto& wq = demo.wq;
  const auto& wk = demo.wk;
  const auto& wv = demo.wv;
  const auto& wo = demo.wo;
  const auto& w1 = demo.w1;
  const auto& w2 = demo.w2;
  const auto& lm_head = demo.lm_head;
  const auto& zero_model_bias = demo.zero_model_bias;
  const auto& zero_hidden_bias = demo.zero_hidden_bias;
  const auto& zero_vocab_bias = demo.zero_vocab_bias;
  const auto cpu_logits = simple_transformer_reference::computeReferenceLogits(demo);

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
  float* d_x = allocateDevice(simple_transformer_reference::kSeqLen *
                              simple_transformer_reference::kModelDim);
  float* d_q = allocateDevice(simple_transformer_reference::kSeqLen *
                              simple_transformer_reference::kModelDim);
  float* d_k = allocateDevice(simple_transformer_reference::kSeqLen *
                              simple_transformer_reference::kModelDim);
  float* d_v = allocateDevice(simple_transformer_reference::kSeqLen *
                              simple_transformer_reference::kModelDim);
  float* d_attn = allocateDevice(simple_transformer_reference::kSeqLen *
                                 simple_transformer_reference::kModelDim);
  float* d_projected = allocateDevice(simple_transformer_reference::kSeqLen *
                                      simple_transformer_reference::kModelDim);
  float* d_hidden = allocateDevice(simple_transformer_reference::kSeqLen *
                                   simple_transformer_reference::kHiddenDim);
  float* d_ffn = allocateDevice(simple_transformer_reference::kSeqLen *
                                simple_transformer_reference::kModelDim);
  float* d_logits = allocateDevice(simple_transformer_reference::kSeqLen *
                                   simple_transformer_reference::kVocabSize);

  int model_values = simple_transformer_reference::kSeqLen *
                     simple_transformer_reference::kModelDim;
  int hidden_values = simple_transformer_reference::kSeqLen *
                      simple_transformer_reference::kHiddenDim;
  int vocab_values = simple_transformer_reference::kSeqLen *
                     simple_transformer_reference::kVocabSize;
  int threads = 128;
  embedKernel<<<(model_values + threads - 1) / threads, threads>>>(
      d_tokens, d_token_embedding, d_position_embedding, d_x,
      simple_transformer_reference::kSeqLen,
      simple_transformer_reference::kModelDim);
  linearKernel<<<(model_values + threads - 1) / threads, threads>>>(
      d_x, d_wq, d_zero_model_bias, d_q, simple_transformer_reference::kSeqLen,
      simple_transformer_reference::kModelDim,
      simple_transformer_reference::kModelDim);
  linearKernel<<<(model_values + threads - 1) / threads, threads>>>(
      d_x, d_wk, d_zero_model_bias, d_k, simple_transformer_reference::kSeqLen,
      simple_transformer_reference::kModelDim,
      simple_transformer_reference::kModelDim);
  linearKernel<<<(model_values + threads - 1) / threads, threads>>>(
      d_x, d_wv, d_zero_model_bias, d_v, simple_transformer_reference::kSeqLen,
      simple_transformer_reference::kModelDim,
      simple_transformer_reference::kModelDim);
  attentionKernel<<<1, simple_transformer_reference::kSeqLen>>>(
      d_q, d_k, d_v, d_attn, simple_transformer_reference::kSeqLen,
      simple_transformer_reference::kModelDim);
  linearKernel<<<(model_values + threads - 1) / threads, threads>>>(
      d_attn, d_wo, d_zero_model_bias, d_projected,
      simple_transformer_reference::kSeqLen,
      simple_transformer_reference::kModelDim,
      simple_transformer_reference::kModelDim);
  addKernel<<<(model_values + threads - 1) / threads, threads>>>(
      d_x, d_projected, d_x, model_values);
  linearKernel<<<(hidden_values + threads - 1) / threads, threads>>>(
      d_x, d_w1, d_zero_hidden_bias, d_hidden,
      simple_transformer_reference::kSeqLen,
      simple_transformer_reference::kModelDim,
      simple_transformer_reference::kHiddenDim);
  reluKernel<<<(hidden_values + threads - 1) / threads, threads>>>(d_hidden,
                                                                  hidden_values);
  linearKernel<<<(model_values + threads - 1) / threads, threads>>>(
      d_hidden, d_w2, d_zero_model_bias, d_ffn,
      simple_transformer_reference::kSeqLen,
      simple_transformer_reference::kHiddenDim,
      simple_transformer_reference::kModelDim);
  addKernel<<<(model_values + threads - 1) / threads, threads>>>(d_x, d_ffn, d_x,
                                                                 model_values);
  linearKernel<<<(vocab_values + threads - 1) / threads, threads>>>(
      d_x, d_lm_head, d_zero_vocab_bias, d_logits,
      simple_transformer_reference::kSeqLen,
      simple_transformer_reference::kModelDim,
      simple_transformer_reference::kVocabSize);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<float> gpu_logits(simple_transformer_reference::kSeqLen *
                                simple_transformer_reference::kVocabSize);
  CHECK_CUDA(cudaMemcpy(gpu_logits.data(), d_logits,
                        gpu_logits.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  float max_diff = simple_transformer_reference::maxAbsDiff(cpu_logits, gpu_logits);
  std::cout << "Simple CUDA transformer block demo\n";
  std::cout << "Max |CPU - GPU| logits diff: " << max_diff << "\n";
  std::cout << "Last-token logits:";
  for (int vocab = 0; vocab < simple_transformer_reference::kVocabSize; ++vocab) {
    std::cout << ' ' << std::fixed << std::setprecision(5)
              << gpu_logits[(simple_transformer_reference::kSeqLen - 1) *
                                simple_transformer_reference::kVocabSize +
                            vocab];
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
