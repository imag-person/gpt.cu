#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
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

__device__ float bfloat16BitsToFloatDevice(std::uint16_t bits) {
  return __uint_as_float(static_cast<unsigned int>(bits) << 16u);
}

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

__global__ void embedBfloat16Kernel(const int* tokens,
                                    const std::uint16_t* token_embedding,
                                    const std::uint16_t* position_embedding,
                                    float* x, int seq_len, int dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = seq_len * dim;
  if (idx >= total) return;

  int token_pos = idx / dim;
  int channel = idx % dim;
  int token_id = tokens[token_pos];
  x[idx] = bfloat16BitsToFloatDevice(token_embedding[token_id * dim + channel]) +
           bfloat16BitsToFloatDevice(position_embedding[token_pos * dim + channel]);
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

__global__ void linearBfloat16Kernel(const float* input,
                                     const std::uint16_t* weight,
                                     const std::uint16_t* bias, float* output,
                                     int rows, int in_dim, int out_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * out_dim;
  if (idx >= total) return;

  int row = idx / out_dim;
  int out_col = idx % out_dim;
  float sum = bfloat16BitsToFloatDevice(bias[out_col]);
  for (int in_col = 0; in_col < in_dim; ++in_col) {
    sum += input[row * in_dim + in_col] *
           bfloat16BitsToFloatDevice(weight[out_col * in_dim + in_col]);
  }
  output[idx] = sum;
}

__global__ void layerNormKernel(const float* input, const float* gamma,
                                const float* beta, float* output, int rows,
                                int dim) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;

  float mean = 0.0f;
  for (int channel = 0; channel < dim; ++channel) {
    mean += input[row * dim + channel];
  }
  mean /= static_cast<float>(dim);

  float variance = 0.0f;
  for (int channel = 0; channel < dim; ++channel) {
    float centered = input[row * dim + channel] - mean;
    variance += centered * centered;
  }
  variance /= static_cast<float>(dim);

  float inv_std = rsqrtf(variance + 1.0e-5f);
  for (int channel = 0; channel < dim; ++channel) {
    float normalized = (input[row * dim + channel] - mean) * inv_std;
    output[row * dim + channel] = normalized * gamma[channel] + beta[channel];
  }
}

__global__ void layerNormBfloat16Kernel(const float* input,
                                        const std::uint16_t* gamma,
                                        const std::uint16_t* beta, float* output,
                                        int rows, int dim) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;

  float mean = 0.0f;
  for (int channel = 0; channel < dim; ++channel) {
    mean += input[row * dim + channel];
  }
  mean /= static_cast<float>(dim);

  float variance = 0.0f;
  for (int channel = 0; channel < dim; ++channel) {
    float centered = input[row * dim + channel] - mean;
    variance += centered * centered;
  }
  variance /= static_cast<float>(dim);

  float inv_std = rsqrtf(variance + 1.0e-5f);
  for (int channel = 0; channel < dim; ++channel) {
    float normalized = (input[row * dim + channel] - mean) * inv_std;
    output[row * dim + channel] =
        normalized * bfloat16BitsToFloatDevice(gamma[channel]) +
        bfloat16BitsToFloatDevice(beta[channel]);
  }
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

__global__ void softmaxRowsKernel(const float* input, float* output, int rows,
                                  int cols) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;

  float max_value = -1.0e30f;
  for (int col = 0; col < cols; ++col) {
    max_value = fmaxf(max_value, input[row * cols + col]);
  }

  float normalizer = 0.0f;
  for (int col = 0; col < cols; ++col) {
    float probability = expf(input[row * cols + col] - max_value);
    output[row * cols + col] = probability;
    normalizer += probability;
  }

  for (int col = 0; col < cols; ++col) {
    output[row * cols + col] /= normalizer;
  }
}

__global__ void meanPoolSequenceKernel(const float* states, float* pooled,
                                       int seq_len, int dim) {
  int channel = blockIdx.x * blockDim.x + threadIdx.x;
  if (channel >= dim) return;

  float sum = 0.0f;
  for (int pos = 0; pos < seq_len; ++pos) {
    sum += states[pos * dim + channel];
  }
  pooled[channel] = sum / static_cast<float>(seq_len);
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
  const auto& norm_gamma = demo.norm_gamma;
  const auto& norm_beta = demo.norm_beta;
  const auto& zero_model_bias = demo.zero_model_bias;
  const auto& zero_hidden_bias = demo.zero_hidden_bias;
  const auto& zero_vocab_bias = demo.zero_vocab_bias;
  const auto cpu_probabilities =
      simple_transformer_reference::computeReferenceProbabilities(demo);
  const auto cpu_encoder_pooled =
      simple_transformer_reference::computeEncoderPooled(demo);
  const auto cpu_bfloat16_probabilities =
      simple_transformer_reference::computeBfloat16ReferenceProbabilities(demo);
  const auto cpu_bfloat16_encoder_pooled =
      simple_transformer_reference::computeBfloat16EncoderPooled(demo);
  const auto token_embedding_bf16 =
      simple_transformer_reference::quantizeToBfloat16(token_embedding);
  const auto position_embedding_bf16 =
      simple_transformer_reference::quantizeToBfloat16(position_embedding);
  const auto wq_bf16 = simple_transformer_reference::quantizeToBfloat16(wq);
  const auto wk_bf16 = simple_transformer_reference::quantizeToBfloat16(wk);
  const auto wv_bf16 = simple_transformer_reference::quantizeToBfloat16(wv);
  const auto wo_bf16 = simple_transformer_reference::quantizeToBfloat16(wo);
  const auto w1_bf16 = simple_transformer_reference::quantizeToBfloat16(w1);
  const auto w2_bf16 = simple_transformer_reference::quantizeToBfloat16(w2);
  const auto lm_head_bf16 =
      simple_transformer_reference::quantizeToBfloat16(lm_head);
  const auto norm_gamma_bf16 =
      simple_transformer_reference::quantizeToBfloat16(norm_gamma);
  const auto norm_beta_bf16 =
      simple_transformer_reference::quantizeToBfloat16(norm_beta);
  const auto zero_model_bias_bf16 =
      simple_transformer_reference::quantizeToBfloat16(zero_model_bias);
  const auto zero_hidden_bias_bf16 =
      simple_transformer_reference::quantizeToBfloat16(zero_hidden_bias);
  const auto zero_vocab_bias_bf16 =
      simple_transformer_reference::quantizeToBfloat16(zero_vocab_bias);

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
  float* d_norm_gamma = copyToDevice(norm_gamma);
  float* d_norm_beta = copyToDevice(norm_beta);
  float* d_zero_model_bias = copyToDevice(zero_model_bias);
  float* d_zero_hidden_bias = copyToDevice(zero_hidden_bias);
  float* d_zero_vocab_bias = copyToDevice(zero_vocab_bias);
  std::uint16_t* d_token_embedding_bf16 = copyToDevice(token_embedding_bf16);
  std::uint16_t* d_position_embedding_bf16 = copyToDevice(position_embedding_bf16);
  std::uint16_t* d_wq_bf16 = copyToDevice(wq_bf16);
  std::uint16_t* d_wk_bf16 = copyToDevice(wk_bf16);
  std::uint16_t* d_wv_bf16 = copyToDevice(wv_bf16);
  std::uint16_t* d_wo_bf16 = copyToDevice(wo_bf16);
  std::uint16_t* d_w1_bf16 = copyToDevice(w1_bf16);
  std::uint16_t* d_w2_bf16 = copyToDevice(w2_bf16);
  std::uint16_t* d_lm_head_bf16 = copyToDevice(lm_head_bf16);
  std::uint16_t* d_norm_gamma_bf16 = copyToDevice(norm_gamma_bf16);
  std::uint16_t* d_norm_beta_bf16 = copyToDevice(norm_beta_bf16);
  std::uint16_t* d_zero_model_bias_bf16 = copyToDevice(zero_model_bias_bf16);
  std::uint16_t* d_zero_hidden_bias_bf16 = copyToDevice(zero_hidden_bias_bf16);
  std::uint16_t* d_zero_vocab_bias_bf16 = copyToDevice(zero_vocab_bias_bf16);
  float* d_x = allocateDevice(simple_transformer_reference::kSeqLen *
                              simple_transformer_reference::kModelDim);
  float* d_norm1 = allocateDevice(simple_transformer_reference::kSeqLen *
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
  float* d_norm2 = allocateDevice(simple_transformer_reference::kSeqLen *
                                  simple_transformer_reference::kModelDim);
  float* d_hidden = allocateDevice(simple_transformer_reference::kSeqLen *
                                   simple_transformer_reference::kHiddenDim);
  float* d_ffn = allocateDevice(simple_transformer_reference::kSeqLen *
                                simple_transformer_reference::kModelDim);
  float* d_logits = allocateDevice(simple_transformer_reference::kSeqLen *
                                   simple_transformer_reference::kVocabSize);
  float* d_probabilities = allocateDevice(simple_transformer_reference::kSeqLen *
                                          simple_transformer_reference::kVocabSize);
  float* d_encoder_pooled = allocateDevice(simple_transformer_reference::kModelDim);

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
  layerNormKernel<<<1, simple_transformer_reference::kSeqLen>>>(
      d_x, d_norm_gamma, d_norm_beta, d_norm1,
      simple_transformer_reference::kSeqLen,
      simple_transformer_reference::kModelDim);
  linearKernel<<<(model_values + threads - 1) / threads, threads>>>(
      d_norm1, d_wq, d_zero_model_bias, d_q,
      simple_transformer_reference::kSeqLen,
      simple_transformer_reference::kModelDim,
      simple_transformer_reference::kModelDim);
  linearKernel<<<(model_values + threads - 1) / threads, threads>>>(
      d_norm1, d_wk, d_zero_model_bias, d_k,
      simple_transformer_reference::kSeqLen,
      simple_transformer_reference::kModelDim,
      simple_transformer_reference::kModelDim);
  linearKernel<<<(model_values + threads - 1) / threads, threads>>>(
      d_norm1, d_wv, d_zero_model_bias, d_v,
      simple_transformer_reference::kSeqLen,
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
  layerNormKernel<<<1, simple_transformer_reference::kSeqLen>>>(
      d_x, d_norm_gamma, d_norm_beta, d_norm2,
      simple_transformer_reference::kSeqLen,
      simple_transformer_reference::kModelDim);
  linearKernel<<<(hidden_values + threads - 1) / threads, threads>>>(
      d_norm2, d_w1, d_zero_hidden_bias, d_hidden,
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
  softmaxRowsKernel<<<1, simple_transformer_reference::kSeqLen>>>(
      d_logits, d_probabilities, simple_transformer_reference::kSeqLen,
      simple_transformer_reference::kVocabSize);
  meanPoolSequenceKernel<<<1, simple_transformer_reference::kModelDim>>>(
      d_x, d_encoder_pooled, simple_transformer_reference::kSeqLen,
      simple_transformer_reference::kModelDim);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<float> gpu_probabilities(simple_transformer_reference::kSeqLen *
                                       simple_transformer_reference::kVocabSize);
  CHECK_CUDA(cudaMemcpy(gpu_probabilities.data(), d_probabilities,
                        gpu_probabilities.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  std::vector<float> gpu_encoder_pooled(simple_transformer_reference::kModelDim);
  CHECK_CUDA(cudaMemcpy(gpu_encoder_pooled.data(), d_encoder_pooled,
                        gpu_encoder_pooled.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  embedBfloat16Kernel<<<(model_values + threads - 1) / threads, threads>>>(
      d_tokens, d_token_embedding_bf16, d_position_embedding_bf16, d_x,
      simple_transformer_reference::kSeqLen,
      simple_transformer_reference::kModelDim);
  layerNormBfloat16Kernel<<<1, simple_transformer_reference::kSeqLen>>>(
      d_x, d_norm_gamma_bf16, d_norm_beta_bf16, d_norm1,
      simple_transformer_reference::kSeqLen,
      simple_transformer_reference::kModelDim);
  linearBfloat16Kernel<<<(model_values + threads - 1) / threads, threads>>>(
      d_norm1, d_wq_bf16, d_zero_model_bias_bf16, d_q,
      simple_transformer_reference::kSeqLen,
      simple_transformer_reference::kModelDim,
      simple_transformer_reference::kModelDim);
  linearBfloat16Kernel<<<(model_values + threads - 1) / threads, threads>>>(
      d_norm1, d_wk_bf16, d_zero_model_bias_bf16, d_k,
      simple_transformer_reference::kSeqLen,
      simple_transformer_reference::kModelDim,
      simple_transformer_reference::kModelDim);
  linearBfloat16Kernel<<<(model_values + threads - 1) / threads, threads>>>(
      d_norm1, d_wv_bf16, d_zero_model_bias_bf16, d_v,
      simple_transformer_reference::kSeqLen,
      simple_transformer_reference::kModelDim,
      simple_transformer_reference::kModelDim);
  attentionKernel<<<1, simple_transformer_reference::kSeqLen>>>(
      d_q, d_k, d_v, d_attn, simple_transformer_reference::kSeqLen,
      simple_transformer_reference::kModelDim);
  linearBfloat16Kernel<<<(model_values + threads - 1) / threads, threads>>>(
      d_attn, d_wo_bf16, d_zero_model_bias_bf16, d_projected,
      simple_transformer_reference::kSeqLen,
      simple_transformer_reference::kModelDim,
      simple_transformer_reference::kModelDim);
  addKernel<<<(model_values + threads - 1) / threads, threads>>>(
      d_x, d_projected, d_x, model_values);
  layerNormBfloat16Kernel<<<1, simple_transformer_reference::kSeqLen>>>(
      d_x, d_norm_gamma_bf16, d_norm_beta_bf16, d_norm2,
      simple_transformer_reference::kSeqLen,
      simple_transformer_reference::kModelDim);
  linearBfloat16Kernel<<<(hidden_values + threads - 1) / threads, threads>>>(
      d_norm2, d_w1_bf16, d_zero_hidden_bias_bf16, d_hidden,
      simple_transformer_reference::kSeqLen,
      simple_transformer_reference::kModelDim,
      simple_transformer_reference::kHiddenDim);
  reluKernel<<<(hidden_values + threads - 1) / threads, threads>>>(d_hidden,
                                                                  hidden_values);
  linearBfloat16Kernel<<<(model_values + threads - 1) / threads, threads>>>(
      d_hidden, d_w2_bf16, d_zero_model_bias_bf16, d_ffn,
      simple_transformer_reference::kSeqLen,
      simple_transformer_reference::kHiddenDim,
      simple_transformer_reference::kModelDim);
  addKernel<<<(model_values + threads - 1) / threads, threads>>>(d_x, d_ffn, d_x,
                                                                 model_values);
  linearBfloat16Kernel<<<(vocab_values + threads - 1) / threads, threads>>>(
      d_x, d_lm_head_bf16, d_zero_vocab_bias_bf16, d_logits,
      simple_transformer_reference::kSeqLen,
      simple_transformer_reference::kModelDim,
      simple_transformer_reference::kVocabSize);
  softmaxRowsKernel<<<1, simple_transformer_reference::kSeqLen>>>(
      d_logits, d_probabilities, simple_transformer_reference::kSeqLen,
      simple_transformer_reference::kVocabSize);
  meanPoolSequenceKernel<<<1, simple_transformer_reference::kModelDim>>>(
      d_x, d_encoder_pooled, simple_transformer_reference::kSeqLen,
      simple_transformer_reference::kModelDim);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<float> gpu_bfloat16_probabilities(
      simple_transformer_reference::kSeqLen *
      simple_transformer_reference::kVocabSize);
  CHECK_CUDA(cudaMemcpy(gpu_bfloat16_probabilities.data(), d_probabilities,
                        gpu_bfloat16_probabilities.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  std::vector<float> gpu_bfloat16_encoder_pooled(
      simple_transformer_reference::kModelDim);
  CHECK_CUDA(cudaMemcpy(gpu_bfloat16_encoder_pooled.data(), d_encoder_pooled,
                        gpu_bfloat16_encoder_pooled.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  float max_diff = simple_transformer_reference::maxAbsDiff(cpu_probabilities,
                                                            gpu_probabilities);
  float encoder_max_diff = simple_transformer_reference::maxAbsDiff(
      cpu_encoder_pooled, gpu_encoder_pooled);
  float bfloat16_max_diff = simple_transformer_reference::maxAbsDiff(
      cpu_bfloat16_probabilities, gpu_bfloat16_probabilities);
  float bfloat16_encoder_max_diff = simple_transformer_reference::maxAbsDiff(
      cpu_bfloat16_encoder_pooled, gpu_bfloat16_encoder_pooled);
  std::cout << "Simple CUDA transformer block demo\n";
  std::cout << "Max |CPU - GPU| probability diff: " << max_diff << "\n";
  std::cout << "Max |CPU - GPU| encoder pooled diff: " << encoder_max_diff
            << "\n";
  std::cout << "Max |CPU - GPU| BF16 probability diff: " << bfloat16_max_diff
            << "\n";
  std::cout << "Max |CPU - GPU| BF16 encoder pooled diff: "
            << bfloat16_encoder_max_diff << "\n";
  std::cout << "Last-token probabilities:";
  for (int vocab = 0; vocab < simple_transformer_reference::kVocabSize; ++vocab) {
    std::cout << ' ' << std::fixed << std::setprecision(5)
              << gpu_probabilities[(simple_transformer_reference::kSeqLen - 1) *
                                       simple_transformer_reference::kVocabSize +
                                   vocab];
  }
  std::cout << "\nPooled encoder embedding:";
  for (float value : gpu_encoder_pooled) {
    std::cout << ' ' << std::fixed << std::setprecision(5) << value;
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
  CHECK_CUDA(cudaFree(d_norm_gamma));
  CHECK_CUDA(cudaFree(d_norm_beta));
  CHECK_CUDA(cudaFree(d_zero_model_bias));
  CHECK_CUDA(cudaFree(d_zero_hidden_bias));
  CHECK_CUDA(cudaFree(d_zero_vocab_bias));
  CHECK_CUDA(cudaFree(d_token_embedding_bf16));
  CHECK_CUDA(cudaFree(d_position_embedding_bf16));
  CHECK_CUDA(cudaFree(d_wq_bf16));
  CHECK_CUDA(cudaFree(d_wk_bf16));
  CHECK_CUDA(cudaFree(d_wv_bf16));
  CHECK_CUDA(cudaFree(d_wo_bf16));
  CHECK_CUDA(cudaFree(d_w1_bf16));
  CHECK_CUDA(cudaFree(d_w2_bf16));
  CHECK_CUDA(cudaFree(d_lm_head_bf16));
  CHECK_CUDA(cudaFree(d_norm_gamma_bf16));
  CHECK_CUDA(cudaFree(d_norm_beta_bf16));
  CHECK_CUDA(cudaFree(d_zero_model_bias_bf16));
  CHECK_CUDA(cudaFree(d_zero_hidden_bias_bf16));
  CHECK_CUDA(cudaFree(d_zero_vocab_bias_bf16));
  CHECK_CUDA(cudaFree(d_x));
  CHECK_CUDA(cudaFree(d_norm1));
  CHECK_CUDA(cudaFree(d_q));
  CHECK_CUDA(cudaFree(d_k));
  CHECK_CUDA(cudaFree(d_v));
  CHECK_CUDA(cudaFree(d_attn));
  CHECK_CUDA(cudaFree(d_projected));
  CHECK_CUDA(cudaFree(d_norm2));
  CHECK_CUDA(cudaFree(d_hidden));
  CHECK_CUDA(cudaFree(d_ffn));
  CHECK_CUDA(cudaFree(d_logits));
  CHECK_CUDA(cudaFree(d_probabilities));
  CHECK_CUDA(cudaFree(d_encoder_pooled));

  if (!std::isfinite(max_diff) || max_diff > 1.0e-4f ||
      !std::isfinite(encoder_max_diff) || encoder_max_diff > 1.0e-4f ||
      !std::isfinite(bfloat16_max_diff) || bfloat16_max_diff > 1.0e-4f ||
      !std::isfinite(bfloat16_encoder_max_diff) ||
      bfloat16_encoder_max_diff > 1.0e-4f) {
    std::cerr << "Validation failed: GPU outputs diverged from CPU reference\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
