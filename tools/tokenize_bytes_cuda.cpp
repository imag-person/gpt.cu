#include "gptcu/tokenizer_bpe_cuda.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: tokenize_bytes_cuda <vocab.txt> <merges.txt> <text>\n";
        return EXIT_FAILURE;
    }

    try {
        const auto tokenizer = gptcu::CpuBpeTokenizer::FromFiles(argv[1], argv[2]);
        if (tokenizer.has_merges()) {
            std::cerr << "warning: CUDA byte tokenizer ignores " << tokenizer.merge_count()
                      << " BPE merges; use tokenize_bpe_cpu for full BPE tokenization\n";
        }

        const auto table = gptcu::make_byte_token_table(tokenizer);
        const auto ids = gptcu::encode_bytes_cuda(argv[3], table);
        for (std::size_t i = 0; i < ids.size(); ++i) {
            if (i != 0) {
                std::cout << ' ';
            }
            std::cout << ids[i];
        }
        std::cout << '\n';
    } catch (const std::exception& error) {
        std::cerr << "CUDA tokenization failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}