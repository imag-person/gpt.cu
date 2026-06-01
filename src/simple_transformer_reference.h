#pragma once

#include <cstddef>
#include <vector>

namespace simple_transformer_reference {

constexpr int kSeqLen = 4;
constexpr int kModelDim = 8;
constexpr int kHiddenDim = 16;
constexpr int kVocabSize = 12;
constexpr int kMaxSeqLen = 16;

struct DemoInputs {
  std::vector<int> tokens;
  std::vector<float> token_embedding;
  std::vector<float> position_embedding;
  std::vector<float> wq;
  std::vector<float> wk;
  std::vector<float> wv;
  std::vector<float> wo;
  std::vector<float> w1;
  std::vector<float> w2;
  std::vector<float> lm_head;
  std::vector<float> zero_model_bias;
  std::vector<float> zero_hidden_bias;
  std::vector<float> zero_vocab_bias;
};

std::vector<float> makeValues(std::size_t count, float scale);
DemoInputs makeDemoInputs();
std::vector<float> embedTokens(const DemoInputs& demo);
std::vector<float> cpuLinear(const std::vector<float>& input,
                             const std::vector<float>& weight,
                             const std::vector<float>& bias, int rows,
                             int in_dim, int out_dim);
std::vector<float> cpuAttention(const std::vector<float>& q,
                                const std::vector<float>& k,
                                const std::vector<float>& v);
void addInPlace(std::vector<float>& lhs, const std::vector<float>& rhs);
float maxAbsDiff(const std::vector<float>& lhs, const std::vector<float>& rhs);
std::vector<float> computeReferenceLogits(const DemoInputs& demo);

}  // namespace simple_transformer_reference