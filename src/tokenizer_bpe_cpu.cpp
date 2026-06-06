#include "gptcu/tokenizer_bpe_cpu.hpp"

#include <algorithm>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace gptcu {
namespace {

std::string trim(const std::string& value) {
    const auto begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return "";
    }
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

std::string unescape_token(const std::string& token) {
    std::string out;
    out.reserve(token.size());
    for (std::size_t i = 0; i < token.size(); ++i) {
        if (token[i] != '\\' || i + 1 == token.size()) {
            out.push_back(token[i]);
            continue;
        }
        const char escaped = token[++i];
        if (escaped == 's') {
            out.push_back(' ');
        } else if (escaped == 't') {
            out.push_back('\t');
        } else if (escaped == 'n') {
            out.push_back('\n');
        } else {
            out.push_back(escaped);
        }
    }
    return out;
}

std::vector<std::string> bytes_as_symbols(const std::string& text) {
    std::vector<std::string> symbols;
    symbols.reserve(text.size());
    for (const unsigned char byte : text) {
        symbols.emplace_back(1, static_cast<char>(byte));
    }
    return symbols;
}

} // namespace

std::size_t CpuBpeTokenizer::PairKeyHash::operator()(const PairKey& key) const {
    const auto left_hash = std::hash<std::string>{}(key.left);
    const auto right_hash = std::hash<std::string>{}(key.right);
    return left_hash ^ (right_hash + 0x9e3779b97f4a7c15ULL + (left_hash << 6U) + (left_hash >> 2U));
}

CpuBpeTokenizer::CpuBpeTokenizer(std::unordered_map<std::string, TokenId> vocab,
                                 const std::vector<Merge>& merges,
                                 TokenId unknown_token_id)
    : vocab_(std::move(vocab)), unknown_token_id_(unknown_token_id) {
    for (const auto& entry : vocab_) {
        const auto [_, inserted] = id_to_token_.emplace(entry.second, entry.first);
        if (!inserted) {
            throw std::invalid_argument("duplicate token id in BPE vocab");
        }
    }

    for (std::size_t rank = 0; rank < merges.size(); ++rank) {
        PairKey key{merges[rank].first, merges[rank].second};
        merge_ranks_.emplace(std::move(key), static_cast<std::int32_t>(rank));
    }
}

CpuBpeTokenizer CpuBpeTokenizer::FromFiles(const std::filesystem::path& vocab_path,
                                          const std::filesystem::path& merges_path,
                                          TokenId unknown_token_id) {
    std::ifstream vocab_file(vocab_path);
    if (!vocab_file) {
        throw std::runtime_error("failed to open vocab file: " + vocab_path.string());
    }

    std::unordered_map<std::string, TokenId> vocab;
    std::string line;
    while (std::getline(vocab_file, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream stream(line);
        std::string token;
        TokenId id = kUnknownTokenId;
        if (!(stream >> token >> id)) {
            throw std::runtime_error("invalid vocab line: " + line);
        }
        vocab.emplace(unescape_token(token), id);
    }

    std::ifstream merges_file(merges_path);
    if (!merges_file) {
        throw std::runtime_error("failed to open merges file: " + merges_path.string());
    }

    std::vector<Merge> merges;
    while (std::getline(merges_file, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream stream(line);
        std::string left;
        std::string right;
        if (!(stream >> left >> right)) {
            throw std::runtime_error("invalid merge line: " + line);
        }
        merges.emplace_back(unescape_token(left), unescape_token(right));
    }

    return CpuBpeTokenizer(std::move(vocab), merges, unknown_token_id);
}

std::vector<std::string> CpuBpeTokenizer::tokenize(const std::string& text) const {
    auto symbols = bytes_as_symbols(text);
    while (symbols.size() > 1) {
        std::size_t best_index = symbols.size();
        std::int32_t best_rank = std::numeric_limits<std::int32_t>::max();

        for (std::size_t i = 0; i + 1 < symbols.size(); ++i) {
            const auto found = merge_ranks_.find(PairKey{symbols[i], symbols[i + 1]});
            if (found != merge_ranks_.end() && found->second < best_rank) {
                best_rank = found->second;
                best_index = i;
            }
        }

        if (best_index == symbols.size()) {
            break;
        }

        symbols[best_index] += symbols[best_index + 1];
        symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(best_index + 1));
    }
    return symbols;
}

std::vector<CpuBpeTokenizer::TokenId> CpuBpeTokenizer::encode(const std::string& text) const {
    const auto tokens = tokenize(text);
    std::vector<TokenId> ids;
    ids.reserve(tokens.size());
    for (const auto& token : tokens) {
        const auto found = vocab_.find(token);
        ids.push_back(found == vocab_.end() ? unknown_token_id_ : found->second);
    }
    return ids;
}

std::string CpuBpeTokenizer::decode(const std::vector<TokenId>& token_ids) const {
    std::string text;
    for (const TokenId id : token_ids) {
        const auto found = id_to_token_.find(id);
        if (found == id_to_token_.end()) {
            if (id == unknown_token_id_) {
                text += "<unk>";
                continue;
            }
            throw std::runtime_error("token id is not present in BPE vocab");
        }
        text += found->second;
    }
    return text;
}

bool CpuBpeTokenizer::has_token(const std::string& token) const {
    return vocab_.find(token) != vocab_.end();
}

} // namespace gptcu