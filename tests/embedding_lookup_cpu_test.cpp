#include "gptcu/embedding_lookup_cpu.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

template <typename T>
void expect_eq(const T& actual, const T& expected, const std::string& label) {
    if (actual != expected) {
        throw std::runtime_error("failed assertion: " + label);
    }
}

template <typename Fn>
void expect_throws(Fn&& fn, const std::string& label) {
    try {
        fn();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error("failed assertion: " + label);
}

void test_single_and_batch_lookup() {
    const gptcu::CpuEmbeddingTable table(3, {0.1f, 0.2f, 0.3f, 1.0f, 1.1f, 1.2f});
    expect_eq(table.vocab_size(), std::size_t{2}, "vocab size inferred from weights");
    expect_eq(table.embedding_dim(), std::size_t{3}, "embedding dim stored");
    expect_eq(table.lookup(1), std::vector<float>({1.0f, 1.1f, 1.2f}), "single lookup returns row");
    expect_eq(table.lookup_many({1, 0}), std::vector<float>({1.0f, 1.1f, 1.2f, 0.1f, 0.2f, 0.3f}),
              "batch lookup is flattened row-major");
}

void test_file_loading() {
    const auto dir = std::filesystem::temp_directory_path() / "gptcu_embedding_test";
    std::filesystem::create_directories(dir);
    const auto path = dir / "embeddings.txt";

    std::ofstream file(path);
    file << "# token 0\n";
    file << "0.5 0.0 -0.5\n";
    file << "1.5 1.0 0.5\n";
    file.close();

    const auto table = gptcu::CpuEmbeddingTable::FromTextFile(path);
    expect_eq(table.lookup(0), std::vector<float>({0.5f, 0.0f, -0.5f}), "load first row from file");
    expect_eq(table.lookup(1), std::vector<float>({1.5f, 1.0f, 0.5f}), "load second row from file");
}

void test_invalid_token_id() {
    const gptcu::CpuEmbeddingTable table(2, {0.0f, 0.1f, 1.0f, 1.1f});
    expect_throws([&]() { static_cast<void>(table.lookup(-1)); }, "negative token id rejected");
    expect_throws([&]() { static_cast<void>(table.lookup(2)); }, "token id past vocab rejected");
}

void test_invalid_embedding_value_rejected() {
    const auto dir = std::filesystem::temp_directory_path() / "gptcu_embedding_invalid_value_test";
    std::filesystem::create_directories(dir);
    const auto path = dir / "embeddings.txt";

    std::ofstream file(path);
    file << "0.5 0.0 -0.5\n";
    file << "1.5 bad 0.5\n";
    file.close();

    expect_throws([&]() { static_cast<void>(gptcu::CpuEmbeddingTable::FromTextFile(path)); },
                  "invalid embedding token rejected");
}

} // namespace

int main() {
    try {
        test_single_and_batch_lookup();
        test_file_loading();
        test_invalid_token_id();
        test_invalid_embedding_value_rejected();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}