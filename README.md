# gpt.cu

Small CUDA/C++ experiments for GPT-style inference components.

## CPU BPE tokenization

This repository includes a dependency-free C++17 CPU Byte Pair Encoding tokenizer in
`include/gptcu/tokenizer_bpe_cpu.hpp` and `src/tokenizer_bpe_cpu.cpp`.

The tokenizer starts from input bytes, repeatedly applies the lowest-rank adjacent
merge from a BPE merges list, then maps resulting token strings through a vocab.
This keeps tokenization usable on CPU without requiring CUDA or third-party
libraries.

### Build and test

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

### CLI usage

`tokenize_bpe_cpu` expects a vocab file, a merges file, and text to tokenize:

```sh
./build/tokenize_bpe_cpu vocab.txt merges.txt "hello world"
```

Vocab lines use `TOKEN ID`; merge lines use `LEFT RIGHT`. Use `\s` for a literal
space token, `\t` for tab, `\n` for newline, and `\\` for a literal backslash.