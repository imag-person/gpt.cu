# gpt.cu

## BPE Tokenizer (CPU)

A minimal, self-contained GPT-2 style byte-level Byte Pair Encoding (BPE)
tokenizer that runs entirely on the CPU. It lives in [`tokenizer/`](tokenizer/)
and has no external dependencies.

Features:

- Train merges from raw text up to a target vocabulary size
- Encode text into token ids and decode ids back into bytes (lossless)
- Save and load the learned model

### Build & test

```sh
cd tokenizer
make test        # builds bpe_demo and runs the round-trip / save-load self-test
```

The demo can also train on your own corpus:

```sh
./bpe_demo path/to/corpus.txt 512   # 512 = target vocab size
```

### API

Include `tokenizer/bpe.h` and link `tokenizer/bpe.c`:

```c
BpeTokenizer tok;
bpe_init(&tok);
bpe_train(&tok, text, text_len, /*vocab_size=*/512, /*verbose=*/0);

size_t n_ids;
int *ids = bpe_encode(&tok, text, text_len, &n_ids);

size_t out_len;
unsigned char *bytes = bpe_decode(&tok, ids, n_ids, &out_len);

bpe_save(&tok, "bpe.model");
free(ids);
free(bytes);
bpe_free(&tok);
```

## CUDA batch encoding (optional)

Encoding is independent per sequence, so tokenizing a corpus is embarrassingly
parallel. [`tokenizer/bpe_cuda.cu`](tokenizer/bpe_cuda.cu) uploads a trained
tokenizer's merges to the GPU and encodes a whole batch of sequences at once,
one thread per sequence. Output is bit-for-bit identical to the CPU `bpe_encode`.

Requires `nvcc` and a CUDA-capable GPU at runtime:

```sh
cd tokenizer
make cuda-test   # builds bpe_cuda_demo and verifies GPU output matches the CPU encoder
```

```c
#include "bpe_cuda.h"

/* Sequences packed CSR-style: seq i is data[offsets[i] .. offsets[i+1]). */
int *ids;            /* concatenated token ids   */
size_t *out_offsets; /* length n_seqs + 1        */
bpe_encode_batch_cuda(&tok, data, offsets, n_seqs, &ids, &out_offsets);
free(ids);
free(out_offsets);
```

## Embedding lookup

The first layer of a GPT model maps token ids to dense vectors.
[`tokenizer/embedding.h`](tokenizer/embedding.c) provides a `[vocab_size x dim]`
embedding table and a gather that turns a token-id sequence into an
`[n_tokens x dim]` matrix. A CUDA gather
([`tokenizer/embedding_cuda.cu`](tokenizer/embedding_cuda.cu)) does the same on
the GPU, one thread per output element, bit-for-bit identical to the CPU path.

```sh
cd tokenizer
make embed-test         # text -> BPE ids -> embeddings (CPU), with a gather check
make embed-cuda-test    # verifies the GPU gather matches the CPU (needs nvcc + a GPU)
```

```c
#include "embedding.h"

Embedding emb;
embedding_init(&emb, tok.vocab_size, /*dim=*/16);
embedding_randomize(&emb, /*seed=*/1234, /*scale=*/0.1f);

float *out = malloc(n_ids * 16 * sizeof(float)); /* [n_ids x dim], row-major */
embedding_lookup(&emb, ids, n_ids, out);

free(out);
embedding_free(&emb);
```

## Transformer-block layers

The core forward primitives that, together with attention, make up a GPT block
live in [`tokenizer/nn.h`](tokenizer/nn.c): **LayerNorm**, **Linear**, and the
**GELU** activation (GPT-2 tanh approximation). All operate on row-major float
matrices where each row is one token. CUDA counterparts
([`tokenizer/nn_cuda.cu`](tokenizer/nn_cuda.cu)) match the CPU layers up to
floating-point rounding (CPU accumulates in double, the GPU in float).

```sh
cd tokenizer
make nn-test         # layernorm row stats, an MLP block, and GELU spot checks
make nn-cuda-test    # verifies the GPU layers match the CPU (needs nvcc + a GPU)
```

```c
#include "nn.h"

/* MLP block: layernorm -> linear(dim->4*dim) -> gelu -> linear(4*dim->dim) */
layernorm_forward(x, gamma, beta, n_tokens, dim, 1e-5f, ln);
linear_forward(ln, w1, b1, n_tokens, dim, 4 * dim, h);
gelu_forward(h, (size_t)n_tokens * 4 * dim, h);
linear_forward(h, w2, b2, n_tokens, 4 * dim, dim, out);
```