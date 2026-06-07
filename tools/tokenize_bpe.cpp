#include "gptcu/tokenizer_bpe_cpu.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: tokenize_bpe_cpu <vocab.txt> <merges.txt> <text>\n";
        return EXIT_FAILURE;
    }

    try {
        const auto tokenizer = gptcu::CpuBpeTokenizer::FromFiles(argv[1], argv[2]);
        const auto ids = tokenizer.encode(argv[3]);
        for (std::size_t i = 0; i < ids.size(); ++i) {
            if (i != 0) {
                std::cout << ' ';
            }
            std::cout << ids[i];
        }
        std::cout << '\n';
    } catch (const std::exception& error) {
        std::cerr << "tokenization failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}