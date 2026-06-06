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