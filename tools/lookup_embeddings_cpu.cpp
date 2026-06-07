#include "gptcu/embedding_lookup_cpu.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: lookup_embeddings_cpu <embeddings.txt> <token_id> [token_id...]\n";
        return EXIT_FAILURE;
    }

    try {
        const auto table = gptcu::CpuEmbeddingTable::FromTextFile(argv[1]);
        std::vector<gptcu::CpuEmbeddingTable::TokenId> token_ids;
        token_ids.reserve(static_cast<std::size_t>(argc - 2));
        for (int i = 2; i < argc; ++i) {
            token_ids.push_back(static_cast<gptcu::CpuEmbeddingTable::TokenId>(std::stoi(argv[i])));
        }

        const auto embeddings = table.lookup_many(token_ids);
        const auto dim = table.embedding_dim();
        for (std::size_t row = 0; row < token_ids.size(); ++row) {
            for (std::size_t col = 0; col < dim; ++col) {
                if (col != 0) {
                    std::cout << ' ';
                }
                std::cout << embeddings[row * dim + col];
            }
            std::cout << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "embedding lookup failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}