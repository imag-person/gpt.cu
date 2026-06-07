/*
 * embedding_cuda.h - CUDA gather for the gpt.cu embedding table.
 *
 * Embedding lookup is a pure gather, so it parallelizes perfectly on the GPU:
 * one thread per output element. Results are bit-for-bit identical to the CPU
 * embedding_lookup (no arithmetic, just row copies).
 */
#ifndef EMBEDDING_CUDA_H
#define EMBEDDING_CUDA_H

#include <stddef.h>

#include "embedding.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Gather embeddings on the GPU:
 *   out[i * dim + j] = weight[ids[i] * dim + j]   (zero row for out-of-range ids)
 * `out` is a host buffer of length n_ids * dim that the caller preallocates.
 * Returns 0 on success, non-zero on error (no CUDA device / runtime error).
 */
int embedding_lookup_cuda(const Embedding *e, const int *ids, size_t n_ids,
                          float *out);

#ifdef __cplusplus
}
#endif

#endif /* EMBEDDING_CUDA_H */
