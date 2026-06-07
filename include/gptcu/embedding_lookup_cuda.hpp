#pragma once

#include "gptcu/embedding_lookup_cpu.hpp"

#include <vector>

namespace gptcu {

std::vector<CpuEmbeddingTable::Value> embedding_lookup_cuda(
    const CpuEmbeddingTable& table,
    const std::vector<CpuEmbeddingTable::TokenId>& token_ids);

} // namespace gptcu
