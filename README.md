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

## Optional CUDA byte tokenization

When a CUDA compiler is available, CMake also builds a CUDA byte-token encoder in
`include/gptcu/tokenizer_bpe_cuda.hpp` and `src/tokenizer_bpe_cuda.cu`.

```sh
cmake -S . -B build -DGPTCU_ENABLE_CUDA=ON
cmake --build build --target tokenize_bytes_cuda
./build/tokenize_bytes_cuda vocab.txt merges.txt "hello world"
```

The CUDA path maps each input byte to a vocab ID in parallel, which is useful for
byte-level tokenizers and GPU-side input staging. Full greedy BPE merges remain on
the CPU path because they are sequential and produce variable-length symbols.

## CPU embedding lookup

This repository also includes a dependency-free CPU embedding table in
`include/gptcu/embedding_lookup_cpu.hpp` and `src/embedding_lookup_cpu.cpp`.

The embedding file format is one whitespace-separated float row per token ID; the
row index is the token ID used for lookup.

```sh
./build/lookup_embeddings_cpu embeddings.txt 0 3 7
```

Each requested token ID prints one embedding row, which makes it easy to inspect
or stage token embeddings before adding more model layers.

## Optional CUDA embedding lookup

When a CUDA compiler is available, CMake also builds a GPU embedding gather in
`include/gptcu/embedding_lookup_cuda.hpp` and `src/embedding_lookup_cuda.cu`.

`embedding_lookup_cuda` gathers the requested token rows on the GPU and returns
the same flattened, row-major result as `CpuEmbeddingTable::lookup_many`, so the
CUDA path matches the CPU reference.

## CPU next-layer primitives

The `gptcu_layers_cpu` library adds small CPU building blocks for the layers that
normally follow token and position embeddings:

- `Tensor2D` for row-major sequence batches
- `CpuLinear` for dense projections
- `layer_norm`, `gelu`, and `residual_add`
- `causal_self_attention` for a single pre-projected causal attention head
- `CpuFeedForward` for a GELU MLP block

These routines are intentionally dependency-free and tested as reference CPU
implementations.

## Optional CUDA next-layer primitives

When a CUDA compiler is available, CMake also builds matching CUDA kernels for
the core transformer-block layers in `include/gptcu/layers_cuda.hpp` and
`src/layers_cuda.cu`:

- `layer_norm_cuda` mirrors `layer_norm`
- `gelu_cuda` mirrors `gelu` (GPT-2 tanh approximation)
- `linear_cuda` mirrors `CpuLinear::forward`

Each kernel reproduces the CPU math operation-for-operation (one thread per
independent output, same accumulation order), and the CUDA targets are compiled
with `--fmad=false` so results match the CPU reference. The
`layers_cuda_test`/`embedding_lookup_cuda_test` suites compare the CUDA output
against the CPU path and only build when `nvcc` is present.