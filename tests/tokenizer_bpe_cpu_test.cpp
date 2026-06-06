#include "gptcu/tokenizer_bpe_cpu.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

template <typename T>
void expect_eq(const T& actual, const T& expected, const std::string& label) {
    if (actual != expected) {
        throw std::runtime_error("failed assertion: " + label);
    }
}

gptcu::CpuBpeTokenizer make_toy_tokenizer() {
    return gptcu::CpuBpeTokenizer(
        {{"l", 0}, {"o", 1}, {"w", 2}, {"e", 3}, {"r", 4}, {"lo", 5},
         {"low", 6}, {"er", 7}, {"lower", 8}, {" ", 9}, {"lower ", 10},
         {"n", 11}, {"ne", 12}, {"new", 13}, {"<unk>", 14}},
        {{"l", "o"}, {"lo", "w"}, {"e", "r"}, {"low", "er"}, {"lower", " "},
         {"n", "e"}, {"ne", "w"}},
        14);
}

void test_greedy_bpe_merges() {
    const auto tokenizer = make_toy_tokenizer();
    expect_eq(tokenizer.tokenize("lower new"), std::vector<std::string>({"lower ", "new"}),
              "tokenize lower new");
    expect_eq(tokenizer.encode("lower new"), std::vector<gptcu::CpuBpeTokenizer::TokenId>({10, 13}),
              "encode lower new");
    expect_eq(tokenizer.decode({10, 13}), std::string("lower new"), "decode lower new");
}

void test_unknown_token() {
    const auto tokenizer = make_toy_tokenizer();
    expect_eq(tokenizer.encode("z"), std::vector<gptcu::CpuBpeTokenizer::TokenId>({14}),
              "unknown byte uses configured token");
    expect_eq(tokenizer.token_id_for("z"), gptcu::CpuBpeTokenizer::TokenId{14},
              "token_id_for uses configured unknown token");
}

void test_file_loading_with_escaped_space() {
    const auto dir = std::filesystem::temp_directory_path() / "gptcu_bpe_test";
    std::filesystem::create_directories(dir);
    const auto vocab_path = dir / "vocab.txt";
    const auto merges_path = dir / "merges.txt";

    {
        std::ofstream vocab(vocab_path);
        vocab << "a 0\n";
        vocab << "b 1\n";
        vocab << "ab 2\n";
        vocab << "\\s 3\n";
        vocab << "ab\\s 4\n";
    }
    {
        std::ofstream merges(merges_path);
        merges << "a b\n";
        merges << "ab \\s\n";
    }

    const auto tokenizer = gptcu::CpuBpeTokenizer::FromFiles(vocab_path, merges_path);
    expect_eq(tokenizer.encode("ab "), std::vector<gptcu::CpuBpeTokenizer::TokenId>({4}),
              "load escaped-space merge from files");
}

void test_merge_introspection() {
    const auto tokenizer = make_toy_tokenizer();
    expect_eq(tokenizer.has_merges(), true, "tokenizer reports merges");
    expect_eq(tokenizer.merge_count(), std::size_t{7}, "tokenizer merge count");
}

} // namespace

int main() {
    try {
        test_greedy_bpe_merges();
        test_unknown_token();
        test_file_loading_with_escaped_space();
        test_merge_introspection();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}