#pragma once

#include "gptcu/tokenizer_bpe_cpu.hpp"

#include <array>
#include <string>
#include <vector>

namespace gptcu {

using ByteTokenTable = std::array<CpuBpeTokenizer::TokenId, 256>;

ByteTokenTable make_byte_token_table(const CpuBpeTokenizer& tokenizer);

std::vector<CpuBpeTokenizer::TokenId> encode_bytes_cuda(const std::string& text,
                                                        const ByteTokenTable& byte_tokens);

} // namespace gptcu