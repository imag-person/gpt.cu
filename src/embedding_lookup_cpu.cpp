#include "gptcu/embedding_lookup_cpu.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

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

} // namespace

CpuEmbeddingTable::CpuEmbeddingTable(std::size_t embedding_dim, std::vector<Value> weights)
    : embedding_dim_(embedding_dim), weights_(std::move(weights)) {
    if (embedding_dim_ == 0) {
        throw std::invalid_argument("embedding dimension must be greater than zero");
    }
    if (weights_.size() % embedding_dim_ != 0) {
        throw std::invalid_argument("embedding weights must be divisible by the embedding dimension");
    }
    vocab_size_ = weights_.size() / embedding_dim_;
}

CpuEmbeddingTable CpuEmbeddingTable::FromTextFile(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("failed to open embedding file: " + path.string());
    }

    std::vector<Value> weights;
    std::size_t embedding_dim = 0;
    std::string line;
    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::istringstream stream(line);
        std::vector<Value> row;
        Value value = 0.0f;
        while (stream >> value) {
            row.push_back(value);
        }
        if (stream.fail() && !stream.eof()) {
            throw std::runtime_error("invalid embedding value in row: " + line);
        }
        if (row.empty()) {
            throw std::runtime_error("embedding row is empty: " + line);
        }
        if (embedding_dim == 0) {
            embedding_dim = row.size();
        } else if (row.size() != embedding_dim) {
            throw std::runtime_error("embedding rows must all have the same width");
        }
        weights.insert(weights.end(), row.begin(), row.end());
    }

    if (embedding_dim == 0) {
        throw std::runtime_error("embedding file did not contain any rows");
    }

    return CpuEmbeddingTable(embedding_dim, std::move(weights));
}

std::size_t CpuEmbeddingTable::vocab_size() const {
    return vocab_size_;
}

std::size_t CpuEmbeddingTable::embedding_dim() const {
    return embedding_dim_;
}

std::vector<CpuEmbeddingTable::Value> CpuEmbeddingTable::lookup(TokenId token_id) const {
    const auto offset = offset_for(token_id);
    return std::vector<Value>(weights_.begin() + static_cast<std::ptrdiff_t>(offset),
                              weights_.begin() + static_cast<std::ptrdiff_t>(offset + embedding_dim_));
}

std::vector<CpuEmbeddingTable::Value> CpuEmbeddingTable::lookup_many(const std::vector<TokenId>& token_ids) const {
    std::vector<Value> result;
    result.reserve(token_ids.size() * embedding_dim_);
    for (const TokenId token_id : token_ids) {
        const auto offset = offset_for(token_id);
        result.insert(result.end(),
                      weights_.begin() + static_cast<std::ptrdiff_t>(offset),
                      weights_.begin() + static_cast<std::ptrdiff_t>(offset + embedding_dim_));
    }
    return result;
}

std::size_t CpuEmbeddingTable::offset_for(TokenId token_id) const {
    if (token_id < 0 || static_cast<std::size_t>(token_id) >= vocab_size_) {
        throw std::out_of_range("token id is out of range for embedding lookup");
    }
    return static_cast<std::size_t>(token_id) * embedding_dim_;
}

} // namespace gptcu