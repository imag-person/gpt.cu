#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace gptcu {

class CpuEmbeddingTable {
public:
    using TokenId = std::int32_t;
    using Value = float;

    CpuEmbeddingTable() = default;
    CpuEmbeddingTable(std::size_t embedding_dim, std::vector<Value> weights);

    static CpuEmbeddingTable FromTextFile(const std::filesystem::path& path);

    std::size_t vocab_size() const;
    std::size_t embedding_dim() const;

    std::vector<Value> lookup(TokenId token_id) const;
    std::vector<Value> lookup_many(const std::vector<TokenId>& token_ids) const;

    const std::vector<Value>& weights() const;

private:
    std::size_t offset_for(TokenId token_id) const;

    std::size_t embedding_dim_ = 0;
    std::size_t vocab_size_ = 0;
    std::vector<Value> weights_;
};

} // namespace gptcu