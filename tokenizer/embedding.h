/*
 * embedding.h - Token embedding table and lookup for gpt.cu.
 *
 * Maps token ids (e.g. the output of the BPE tokenizer) to dense embedding
 * vectors by gathering rows from a [vocab_size x dim] weight matrix. This is
 * the first layer of a GPT model: ids -> embeddings.
 */
#ifndef EMBEDDING_H
#define EMBEDDING_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* A [vocab_size x dim] row-major table of float weights. Row `id` is the
 * embedding for token `id`, stored at weight[id * dim .. id * dim + dim). */
typedef struct {
    int vocab_size;
    int dim;
    float *weight;
} Embedding;

/* Allocate a [vocab_size x dim] table with all weights zeroed. */
void embedding_init(Embedding *e, int vocab_size, int dim);

/* Fill the table with deterministic pseudo-random values in [-scale, scale]
 * (splitmix64 seeded by `seed`). Useful for demos and tests. */
void embedding_randomize(Embedding *e, unsigned long long seed, float scale);

/* Release memory owned by the table. */
void embedding_free(Embedding *e);

/*
 * Gather embeddings for `n_ids` token ids into `out`, which the caller must
 * preallocate with length n_ids * dim (row-major):
 *   out[i * dim + j] = weight[ids[i] * dim + j]
 * Out-of-range ids produce a zero row.
 */
void embedding_lookup(const Embedding *e, const int *ids, size_t n_ids,
                      float *out);

#ifdef __cplusplus
}
#endif

#endif /* EMBEDDING_H */
