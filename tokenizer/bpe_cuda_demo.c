/*
 * bpe_cuda_demo.c - Demonstrates and self-tests the CUDA batch encoder.
 *
 * Trains a BPE model on a sample corpus (CPU), then encodes a batch of byte
 * sequences on the GPU with bpe_encode_batch_cuda() and checks that every
 * sequence matches the reference CPU encoder bpe_encode() exactly.
 *
 * Compiled with nvcc (see the Makefile `cuda` target). Despite the .c name it
 * is fed to nvcc; the GPU code lives in bpe_cuda.cu.
 *
 * Usage:  ./bpe_cuda_demo [vocab_size]
 */
#include "bpe.h"
#include "bpe_cuda.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *SAMPLE =
    "the quick brown fox jumps over the lazy dog. "
    "the dog was not amused, and the fox ran away quickly. "
    "tokenization with byte pair encoding compresses repeated patterns "
    "like the, ing, and tion into single tokens for the gpt model.";

/* A handful of sequences to encode in parallel, including out-of-corpus text. */
static const char *BATCH[] = {
    "the quick brown fox",
    "the lazy dog ran away quickly",
    "byte pair encoding",
    "tokenization for the gpt model",
    "the the the the the",
    "completely different unseen string!!!",
};
#define N_SEQS (sizeof(BATCH) / sizeof(BATCH[0]))

int main(int argc, char **argv) {
    int vocab_size = 276;
    if (argc >= 2) {
        vocab_size = atoi(argv[1]);
        if (vocab_size < 256) vocab_size = 256;
    }

    BpeTokenizer tok;
    bpe_init(&tok);
    bpe_train(&tok, (const unsigned char *)SAMPLE, strlen(SAMPLE), vocab_size, 0);

    printf("=== gpt.cu BPE tokenizer (CUDA batch encode) ===\n");
    printf("vocab size : %d (%d merges)\n", tok.vocab_size, tok.num_merges);
    printf("sequences  : %zu\n\n", (size_t)N_SEQS);

    /* Pack the batch into the (data, offsets) layout the GPU API expects. */
    size_t offsets[N_SEQS + 1];
    offsets[0] = 0;
    for (size_t s = 0; s < N_SEQS; s++)
        offsets[s + 1] = offsets[s] + strlen(BATCH[s]);
    size_t total = offsets[N_SEQS];
    unsigned char *data = (unsigned char *)malloc(total ? total : 1);
    for (size_t s = 0; s < N_SEQS; s++)
        memcpy(data + offsets[s], BATCH[s], strlen(BATCH[s]));

    int *gpu_ids = NULL;
    size_t *gpu_off = NULL;
    int rc = bpe_encode_batch_cuda(&tok, data, offsets, N_SEQS, &gpu_ids, &gpu_off);
    if (rc != 0) {
        fprintf(stderr, "bpe_encode_batch_cuda failed (rc=%d). "
                        "A CUDA-capable GPU is required.\n", rc);
        free(data);
        bpe_free(&tok);
        return 1;
    }

    /* Compare each GPU sequence against the reference CPU encoder. */
    int all_ok = 1;
    for (size_t s = 0; s < N_SEQS; s++) {
        size_t cpu_n = 0;
        int *cpu_ids = bpe_encode(&tok, (const unsigned char *)BATCH[s],
                                  strlen(BATCH[s]), &cpu_n);
        size_t gpu_n = gpu_off[s + 1] - gpu_off[s];
        int ok = (gpu_n == cpu_n) &&
                 (memcmp(cpu_ids, gpu_ids + gpu_off[s], cpu_n * sizeof(int)) == 0);
        printf("  seq %zu: %2zu bytes -> %2zu tokens  %s  \"%s\"\n",
               s, strlen(BATCH[s]), gpu_n, ok ? "OK" : "FAIL", BATCH[s]);
        if (!ok) all_ok = 0;
        free(cpu_ids);
    }

    printf("\nGPU vs CPU match: %s\n", all_ok ? "ALL OK" : "MISMATCH");

    free(gpu_ids);
    free(gpu_off);
    free(data);
    bpe_free(&tok);
    return all_ok ? 0 : 1;
}
