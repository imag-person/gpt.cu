# gpt.cu

Single-file CUDA implementation of a GPT-style decoder-only transformer
(forward pass / greedy inference). Written for readability — naive fp32
kernels, no cuBLAS, no tensor cores, no kv-cache. Useful as a reference for
what a minimal end-to-end transformer forward pass looks like on the GPU.

## Build

```sh
nvcc -O3 -std=c++17 gpt.cu -o gpt
```

## Run

```sh
./gpt              # generate 16 tokens with the default config
./gpt 64           # generate 64 tokens
```

The model is initialised with a fixed RNG seed, so the output is
deterministic across runs (but meaningless — the weights are random).

## What's in the file

- Token + positional embedding lookup
- Pre-LayerNorm transformer blocks: causal multi-head self-attention + MLP
  (GELU, 4× expansion)
- Final LayerNorm and a tied `lm_head` (shares weights with the token
  embedding)
- Greedy decoding loop with timing

Default config is `n_layer=4`, `n_head=4`, `n_embd=128`, `block_size=64`,
`vocab_size=1024`. Edit `main()` in `gpt.cu` to scale up.