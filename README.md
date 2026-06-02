# gpt.cu

Small CUDA C++ experiments for transformer-style model building blocks.

## Simple transformer CUDA demo

This repository includes a self-contained CUDA C++ example that runs one tiny
GPT-style transformer block over four token IDs:

- token and positional embeddings
- Q/K/V projections
- scaled dot-product self-attention
- layer normalization before attention and the feed-forward network
- ReLU activation in the feed-forward network
- output projection with residual connection
- two-layer feed-forward network
- language-model head producing vocabulary logits and softmax probabilities

The executable also computes a CPU reference path and fails if the CUDA
probabilities diverge from the CPU result.

## Build and run

Requirements: CMake 3.18+, a CUDA toolkit with `nvcc`, and a CUDA-capable GPU.

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

If you only want to verify the deterministic CPU reference path without a CUDA
toolkit, compile and run the regression test directly with `g++`:

```sh
g++ -std=c++17 tests/simple_transformer_reference_test.cpp \
  src/simple_transformer_reference.cpp -Isrc -o /tmp/simple_transformer_reference_test
/tmp/simple_transformer_reference_test
```

Run the demo directly:

```sh
./build/simple_transformer_cuda
```

The repository also includes a golden-output regression test for the CPU
reference implementation, including layer normalization, ReLU, and softmax.
