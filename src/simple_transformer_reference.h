#pragma once

#include <cstddef>
#include <cstdint>
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
  std::vector<float> norm_gamma;
  std::vector<float> norm_beta;
  std::vector<float> zero_model_bias;
  std::vector<float> zero_hidden_bias;
  std::vector<float> zero_vocab_bias;
};

std::vector<float> makeValues(std::size_t count, float scale);
std::uint16_t floatToBfloat16Bits(float value);
float bfloat16BitsToFloat(std::uint16_t bits);
std::vector<std::uint16_t> quantizeToBfloat16(
    const std::vector<float>& values);
std::vector<float> dequantizeFromBfloat16(
    const std::vector<std::uint16_t>& values);
std::uint16_t floatToFloat16Bits(float value);
float float16BitsToFloat(std::uint16_t bits);
std::vector<std::uint16_t> quantizeToFloat16(
    const std::vector<float>& values);
std::vector<float> dequantizeFromFloat16(
    const std::vector<std::uint16_t>& values);
DemoInputs makeDemoInputs();
DemoInputs quantizeDemoInputsToBfloat16(const DemoInputs& demo);
DemoInputs quantizeDemoInputsToFloat16(const DemoInputs& demo);
std::vector<float> embedTokens(const DemoInputs& demo);
std::vector<float> cpuLinear(const std::vector<float>& input,
                             const std::vector<float>& weight,
                             const std::vector<float>& bias, int rows,
                             int in_dim, int out_dim);
std::vector<float> cpuAttention(const std::vector<float>& q,
                                const std::vector<float>& k,
                                const std::vector<float>& v);
std::vector<float> cpuLayerNormRows(const std::vector<float>& input,
                                    const std::vector<float>& gamma,
                                    const std::vector<float>& beta, int rows,
                                    int dim);
void cpuReluInPlace(std::vector<float>& values);
std::vector<float> cpuSoftmaxRows(const std::vector<float>& input, int rows,
                                  int cols);
void addInPlace(std::vector<float>& lhs, const std::vector<float>& rhs);
float maxAbsDiff(const std::vector<float>& lhs, const std::vector<float>& rhs);
std::vector<float> meanPoolSequence(const std::vector<float>& states, int seq_len,
                                    int dim);
std::vector<float> computeEncoderStates(const DemoInputs& demo);
std::vector<float> computeEncoderPooled(const DemoInputs& demo);
std::vector<float> computeReferenceLogits(const DemoInputs& demo);
std::vector<float> computeReferenceProbabilities(const DemoInputs& demo);
std::vector<float> computeBfloat16ReferenceProbabilities(const DemoInputs& demo);
std::vector<float> computeBfloat16EncoderPooled(const DemoInputs& demo);
std::vector<float> computeFloat16ReferenceProbabilities(const DemoInputs& demo);
std::vector<float> computeFloat16EncoderPooled(const DemoInputs& demo);

}  // namespace simple_transformer_reference
