#include "simple_transformer_reference.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace simple_transformer_reference {

std::vector<float> makeValues(std::size_t count, float scale) {
  std::vector<float> values(count);
  for (std::size_t i = 0; i < count; ++i) {
    values[i] = scale * std::sin(static_cast<float>(i + 1) * 0.37f);
  }
  return values;
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
  demo.zero_model_bias.assign(kModelDim, 0.0f);
  demo.zero_hidden_bias.assign(kHiddenDim, 0.0f);
  demo.zero_vocab_bias.assign(kVocabSize, 0.0f);
  return demo;
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

std::vector<float> computeReferenceLogits(const DemoInputs& demo) {
  auto x = embedTokens(demo);
  auto q = cpuLinear(x, demo.wq, demo.zero_model_bias, kSeqLen, kModelDim,
                     kModelDim);
  auto k = cpuLinear(x, demo.wk, demo.zero_model_bias, kSeqLen, kModelDim,
                     kModelDim);
  auto v = cpuLinear(x, demo.wv, demo.zero_model_bias, kSeqLen, kModelDim,
                     kModelDim);
  auto attn = cpuAttention(q, k, v);
  auto projected = cpuLinear(attn, demo.wo, demo.zero_model_bias, kSeqLen,
                             kModelDim, kModelDim);
  addInPlace(x, projected);
  auto hidden = cpuLinear(x, demo.w1, demo.zero_hidden_bias, kSeqLen,
                          kModelDim, kHiddenDim);
  for (float& value : hidden) value = std::max(value, 0.0f);
  auto ffn = cpuLinear(hidden, demo.w2, demo.zero_model_bias, kSeqLen,
                       kHiddenDim, kModelDim);
  addInPlace(x, ffn);
  return cpuLinear(x, demo.lm_head, demo.zero_vocab_bias, kSeqLen, kModelDim,
                   kVocabSize);
}

}  // namespace simple_transformer_reference