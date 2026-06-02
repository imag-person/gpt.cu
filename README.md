# gpt.cu

Single-file CUDA reference implementations of transformer forward passes.
Two flavours, same style — naive fp32 kernels, no cuBLAS, no tensor cores,
no kv-cache. Useful as a reference for what a minimal end-to-end transformer
forward pass looks like on the GPU.

- **`gpt.cu`** — GPT-style decoder-only transformer with causal
  self-attention, tied `lm_head`, and a greedy decoding loop.
- **`encoder.cu`** — BERT-style encoder transformer with bidirectional
  self-attention, mean-pooled classifier head, and a single forward pass.

## Build

```sh
nvcc -O3 -std=c++17 gpt.cu -o gpt
nvcc -O3 -std=c++17 encoder.cu -o encoder
```

## Run

```sh
./gpt              # generate 16 tokens with the default config
./gpt 64           # generate 64 tokens

./encoder          # one forward pass over a 32-token random input
./encoder 16       # one forward pass over a 16-token random input
```

Both models are initialised with a fixed RNG seed, so the output is
deterministic across runs (but meaningless — the weights are random).

## What's in `gpt.cu`

- Token + positional embedding lookup
- Pre-LayerNorm transformer blocks: causal multi-head self-attention + MLP
  (GELU, 4× expansion)
- Final LayerNorm and a tied `lm_head` (shares weights with the token
  embedding)
- Greedy decoding loop with timing

## What's in `encoder.cu`

- Token + positional embedding lookup
- Pre-LayerNorm transformer blocks: **bidirectional** multi-head
  self-attention + MLP (GELU, 4× expansion)
- Final LayerNorm, mean-pool over positions, linear classifier head
- Single forward pass with timing; prints class logits and argmax

The encoder shares most of the decoder's kernels — the only substantive
difference is that the attention scores kernel doesn't apply a causal mask.

Default config (both): `n_layer=4`, `n_head=4`, `n_embd=128`,
`block_size=64`, `vocab_size=1024` (and `n_classes=4` for the encoder).
Edit `main()` in either file to scale up.