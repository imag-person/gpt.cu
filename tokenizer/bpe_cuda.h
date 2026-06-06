/*
 * bpe_cuda.h - CUDA-accelerated batch encoding for the gpt.cu BPE tokenizer.
 *
 * The CPU implementation in bpe.h encodes one sequence at a time. When tokenizing
 * a large corpus for training, the encode step is "embarrassingly parallel" across
 * sequences: each sequence is independent. This API uploads a trained tokenizer's
 * merges to the GPU and encodes a whole batch of byte sequences at once, running
 * one thread per sequence. Results are identical (bit-for-bit) to bpe_encode.
 */
#ifndef BPE_CUDA_H
#define BPE_CUDA_H

#include <stddef.h>

#include "bpe.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Batch-encode `n_seqs` byte sequences on the GPU using a trained tokenizer.
 *
 * Input layout (compressed sparse row style):
 *   data    - concatenation of every sequence's bytes
 *   offsets - length n_seqs + 1; sequence i is data[offsets[i] .. offsets[i+1])
 *
 * Output (both malloc'd; the caller must free them):
 *   *out_ids     - concatenation of every sequence's encoded token ids
 *   *out_offsets - length n_seqs + 1; ids for sequence i live in
 *                  (*out_ids)[(*out_offsets)[i] .. (*out_offsets)[i+1])
 *
 * Returns 0 on success, non-zero on error (no CUDA device, allocation failure,
 * or a CUDA runtime error). On error, *out_ids and *out_offsets are left NULL.
 */
int bpe_encode_batch_cuda(const BpeTokenizer *t,
                          const unsigned char *data, const size_t *offsets,
                          size_t n_seqs,
                          int **out_ids, size_t **out_offsets);

#ifdef __cplusplus
}
#endif

#endif /* BPE_CUDA_H */
