#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace gptcu {

class CpuBpeTokenizer {
public:
    using TokenId = std::int32_t;
    using Merge = std::pair<std::string, std::string>;

    static constexpr TokenId kUnknownTokenId = -1;

    CpuBpeTokenizer() = default;
    CpuBpeTokenizer(std::unordered_map<std::string, TokenId> vocab,
                    const std::vector<Merge>& merges,
                    TokenId unknown_token_id = kUnknownTokenId);

    static CpuBpeTokenizer FromFiles(const std::filesystem::path& vocab_path,
                                     const std::filesystem::path& merges_path,
                                     TokenId unknown_token_id = kUnknownTokenId);

    std::vector<std::string> tokenize(const std::string& text) const;
    std::vector<TokenId> encode(const std::string& text) const;
    std::string decode(const std::vector<TokenId>& token_ids) const;

    bool has_token(const std::string& token) const;
    TokenId token_id_for(const std::string& token) const;
    bool has_merges() const;
    std::size_t merge_count() const;

private:
    struct PairKey {
        std::string left;
        std::string right;

        bool operator==(const PairKey& other) const {
            return left == other.left && right == other.right;
        }
    };

    struct PairKeyHash {
        std::size_t operator()(const PairKey& key) const;
    };

    std::unordered_map<std::string, TokenId> vocab_;
    std::unordered_map<TokenId, std::string> id_to_token_;
    std::unordered_map<PairKey, std::int32_t, PairKeyHash> merge_ranks_;
    TokenId unknown_token_id_ = kUnknownTokenId;
};

} // namespace gptcu