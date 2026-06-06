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