/*
 * bpe.h - CPU byte-level Byte Pair Encoding (BPE) tokenizer for gpt.cu.
 *
 * A minimal, self-contained GPT-2 style byte-level BPE implementation that
 * runs entirely on the CPU. It supports training merges from raw text,
 * encoding text into token ids, decoding ids back into bytes, and
 * saving/loading the learned model.
 */
#ifndef BPE_H
#define BPE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* A merge rule: token `first` followed by token `second` becomes a new id. */
typedef struct {
    int first;
    int second;
} BpePair;

typedef struct {
    int vocab_size;            /* total number of tokens (>= 256)            */
    int num_merges;            /* vocab_size - 256                           */
    BpePair *merges;           /* length num_merges; merges[i] -> id 256 + i */
    unsigned char **vocab;     /* vocab[id] is the byte string for that id   */
    int *vocab_lens;           /* vocab_lens[id] is the length of vocab[id]  */
} BpeTokenizer;

/* Initialize an empty tokenizer (no merges, base 256 byte vocabulary). */
void bpe_init(BpeTokenizer *t);

/* Release all memory owned by the tokenizer. */
void bpe_free(BpeTokenizer *t);

/*
 * Train BPE merges on `text` (length `text_len`) until the vocabulary reaches
 * `vocab_size` tokens. `vocab_size` must be >= 256. When `verbose` is non-zero
 * each merge is printed to stdout. Replaces any existing merges in `t`.
 */
void bpe_train(BpeTokenizer *t, const unsigned char *text, size_t text_len,
               int vocab_size, int verbose);

/*
 * Encode `text` into token ids. Returns a malloc'd array of length `*out_len`
 * that the caller must free. Returns NULL on allocation failure.
 */
int *bpe_encode(const BpeTokenizer *t, const unsigned char *text,
                size_t text_len, size_t *out_len);

/*
 * Decode `n_ids` token ids into raw bytes. Returns a malloc'd buffer of length
 * `*out_len` that the caller must free. Returns NULL on allocation failure.
 */
unsigned char *bpe_decode(const BpeTokenizer *t, const int *ids, size_t n_ids,
                          size_t *out_len);

/* Save the learned merges to `path`. Returns 0 on success, non-zero on error. */
int bpe_save(const BpeTokenizer *t, const char *path);

/* Load merges from `path`, rebuilding the vocabulary. Returns 0 on success. */
int bpe_load(BpeTokenizer *t, const char *path);

#ifdef __cplusplus
}
#endif

#endif /* BPE_H */
