#include "simple_transformer_reference.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

namespace simple_transformer_reference {

std::vector<float> makeValues(std::size_t count, float scale) {
  std::vector<float> values(count);
  for (std::size_t i = 0; i < count; ++i) {
    values[i] = scale * std::sin(static_cast<float>(i + 1) * 0.37f);
  }
  return values;
}

std::uint16_t floatToBfloat16Bits(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x00007fffu + ((bits >> 16u) & 1u);
  return static_cast<std::uint16_t>(bits >> 16u);
}

float bfloat16BitsToFloat(std::uint16_t bits) {
  std::uint32_t widened = static_cast<std::uint32_t>(bits) << 16u;
  float value = 0.0f;
  std::memcpy(&value, &widened, sizeof(value));
  return value;
}

std::vector<std::uint16_t> quantizeToBfloat16(
    const std::vector<float>& values) {
  std::vector<std::uint16_t> quantized(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    quantized[i] = floatToBfloat16Bits(values[i]);
  }
  return quantized;
}

std::vector<float> dequantizeFromBfloat16(
    const std::vector<std::uint16_t>& values) {
  std::vector<float> dequantized(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    dequantized[i] = bfloat16BitsToFloat(values[i]);
  }
  return dequantized;
}

DemoInputs makeDemoInputs() {
  DemoInputs demo;
  demo.tokens = {1, 4, 2, 7};
  demo.token_embedding = makeValues(kVocabSize * kModelDim, 0.08f);
  demo.position_embedding = makeValues(kSeqLen * kModelDim, 0.03f);
  demo.wq = makeValues(kModelDim * kModelDim, 0.06f);
  demo.wk = makeValues(kModelDim * kModelDim, 0.05f);
  demo.wv = makeValues(kModelDim * kModelDim, 0.07f);
  demo.wo = makeValues(kModelDim * kModelDim, 0.04f);
  demo.w1 = makeValues(kHiddenDim * kModelDim, 0.09f);
  demo.w2 = makeValues(kModelDim * kHiddenDim, 0.05f);
  demo.lm_head = makeValues(kVocabSize * kModelDim, 0.08f);
  demo.norm_gamma.assign(kModelDim, 1.0f);
  demo.norm_beta.assign(kModelDim, 0.0f);
  demo.zero_model_bias.assign(kModelDim, 0.0f);
  demo.zero_hidden_bias.assign(kHiddenDim, 0.0f);
  demo.zero_vocab_bias.assign(kVocabSize, 0.0f);
  return demo;
}

DemoInputs quantizeDemoInputsToBfloat16(const DemoInputs& demo) {
  DemoInputs quantized = demo;
  auto quantize = [](const std::vector<float>& values) {
    return dequantizeFromBfloat16(quantizeToBfloat16(values));
  };
  quantized.token_embedding = quantize(demo.token_embedding);
  quantized.position_embedding = quantize(demo.position_embedding);
  quantized.wq = quantize(demo.wq);
  quantized.wk = quantize(demo.wk);
  quantized.wv = quantize(demo.wv);
  quantized.wo = quantize(demo.wo);
  quantized.w1 = quantize(demo.w1);
  quantized.w2 = quantize(demo.w2);
  quantized.lm_head = quantize(demo.lm_head);
  quantized.norm_gamma = quantize(demo.norm_gamma);
  quantized.norm_beta = quantize(demo.norm_beta);
  quantized.zero_model_bias = quantize(demo.zero_model_bias);
  quantized.zero_hidden_bias = quantize(demo.zero_hidden_bias);
  quantized.zero_vocab_bias = quantize(demo.zero_vocab_bias);
  return quantized;
}

std::vector<float> embedTokens(const DemoInputs& demo) {
  std::vector<float> x(kSeqLen * kModelDim);
  for (int pos = 0; pos < kSeqLen; ++pos) {
    for (int channel = 0; channel < kModelDim; ++channel) {
      x[pos * kModelDim + channel] =
          demo.token_embedding[demo.tokens[pos] * kModelDim + channel] +
          demo.position_embedding[pos * kModelDim + channel];
    }
  }
  return x;
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

std::vector<float> cpuLayerNormRows(const std::vector<float>& input,
                                    const std::vector<float>& gamma,
                                    const std::vector<float>& beta, int rows,
                                    int dim) {
  std::vector<float> output(rows * dim, 0.0f);
  constexpr float epsilon = 1.0e-5f;

  for (int row = 0; row < rows; ++row) {
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

    float inv_std = 1.0f / std::sqrt(variance + epsilon);
    for (int channel = 0; channel < dim; ++channel) {
      float normalized = (input[row * dim + channel] - mean) * inv_std;
      output[row * dim + channel] = normalized * gamma[channel] + beta[channel];
    }
  }
  return output;
}

void cpuReluInPlace(std::vector<float>& values) {
  for (float& value : values) value = std::max(value, 0.0f);
}

std::vector<float> cpuSoftmaxRows(const std::vector<float>& input, int rows,
                                  int cols) {
  std::vector<float> output(rows * cols, 0.0f);
  for (int row = 0; row < rows; ++row) {
    float max_value = -1.0e30f;
    for (int col = 0; col < cols; ++col) {
      max_value = std::max(max_value, input[row * cols + col]);
    }

    float normalizer = 0.0f;
    for (int col = 0; col < cols; ++col) {
      float probability = std::exp(input[row * cols + col] - max_value);
      output[row * cols + col] = probability;
      normalizer += probability;
    }

    for (int col = 0; col < cols; ++col) {
      output[row * cols + col] /= normalizer;
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

std::vector<float> meanPoolSequence(const std::vector<float>& states, int seq_len,
                                    int dim) {
  std::vector<float> pooled(dim, 0.0f);
  for (int channel = 0; channel < dim; ++channel) {
    for (int pos = 0; pos < seq_len; ++pos) {
      pooled[channel] += states[pos * dim + channel];
    }
    pooled[channel] /= static_cast<float>(seq_len);
  }
  return pooled;
}

std::vector<float> computeEncoderStates(const DemoInputs& demo) {
  auto x = embedTokens(demo);
  auto norm1 = cpuLayerNormRows(x, demo.norm_gamma, demo.norm_beta, kSeqLen,
                                kModelDim);
  auto q = cpuLinear(norm1, demo.wq, demo.zero_model_bias, kSeqLen, kModelDim,
                     kModelDim);
  auto k = cpuLinear(norm1, demo.wk, demo.zero_model_bias, kSeqLen, kModelDim,
                     kModelDim);
  auto v = cpuLinear(norm1, demo.wv, demo.zero_model_bias, kSeqLen, kModelDim,
                     kModelDim);
  auto attn = cpuAttention(q, k, v);
  auto projected = cpuLinear(attn, demo.wo, demo.zero_model_bias, kSeqLen,
                             kModelDim, kModelDim);
  addInPlace(x, projected);
  auto norm2 = cpuLayerNormRows(x, demo.norm_gamma, demo.norm_beta, kSeqLen,
                                kModelDim);
  auto hidden = cpuLinear(norm2, demo.w1, demo.zero_hidden_bias, kSeqLen,
                          kModelDim, kHiddenDim);
  cpuReluInPlace(hidden);
  auto ffn = cpuLinear(hidden, demo.w2, demo.zero_model_bias, kSeqLen,
                       kHiddenDim, kModelDim);
  addInPlace(x, ffn);
  return x;
}

std::vector<float> computeEncoderPooled(const DemoInputs& demo) {
  return meanPoolSequence(computeEncoderStates(demo), kSeqLen, kModelDim);
}

std::vector<float> computeReferenceLogits(const DemoInputs& demo) {
  auto x = computeEncoderStates(demo);
  return cpuLinear(x, demo.lm_head, demo.zero_vocab_bias, kSeqLen, kModelDim,
                   kVocabSize);
}

std::vector<float> computeReferenceProbabilities(const DemoInputs& demo) {
  return cpuSoftmaxRows(computeReferenceLogits(demo), kSeqLen, kVocabSize);
}

std::vector<float> computeBfloat16ReferenceProbabilities(const DemoInputs& demo) {
  return computeReferenceProbabilities(quantizeDemoInputsToBfloat16(demo));
}

std::vector<float> computeBfloat16EncoderPooled(const DemoInputs& demo) {
  return computeEncoderPooled(quantizeDemoInputsToBfloat16(demo));
}

}  // namespace simple_transformer_reference
