/*
 * embedding_cuda_demo.c - Self-test for the CUDA embedding gather.
 *
 * Builds a random embedding table, encodes a string with BPE, then gathers the
 * embeddings on both the CPU and the GPU and checks they are identical. Gather
 * does no arithmetic, so the comparison is exact (bit-for-bit).
 *
 * Compiled with nvcc (see the Makefile `embed-cuda` target).
 */
#include "bpe.h"
#include "embedding.h"
#include "embedding_cuda.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *SAMPLE =
    "the quick brown fox jumps over the lazy dog. "
    "the dog was not amused, and the fox ran away quickly. "
    "tokenization with byte pair encoding compresses repeated patterns "
    "like the, ing, and tion into single tokens for the gpt model.";

int main(void) {
    const char *text = "tokenization for the gpt model";
    const int dim = 32;

    BpeTokenizer tok;
    bpe_init(&tok);
    bpe_train(&tok, (const unsigned char *)SAMPLE, strlen(SAMPLE), 276, 0);

    size_t n_ids = 0;
    int *ids = bpe_encode(&tok, (const unsigned char *)text, strlen(text), &n_ids);

    Embedding emb;
    embedding_init(&emb, tok.vocab_size, dim);
    embedding_randomize(&emb, 1234ULL, 0.1f);

    size_t n = (n_ids ? n_ids : 1) * (size_t)dim;
    float *cpu = (float *)malloc(n * sizeof(float));
    float *gpu = (float *)malloc(n * sizeof(float));

    embedding_lookup(&emb, ids, n_ids, cpu);
    int rc = embedding_lookup_cuda(&emb, ids, n_ids, gpu);
    if (rc != 0) {
        fprintf(stderr, "embedding_lookup_cuda failed (rc=%d). "
                        "A CUDA-capable GPU is required.\n", rc);
        free(cpu); free(gpu); free(ids);
        embedding_free(&emb); bpe_free(&tok);
        return 1;
    }

    int ok = (memcmp(cpu, gpu, (size_t)n_ids * dim * sizeof(float)) == 0);

    printf("=== gpt.cu embedding lookup (CUDA) ===\n");
    printf("text       : \"%s\"\n", text);
    printf("tokens     : %zu\n", n_ids);
    printf("embed dim  : %d\n", dim);
    printf("output     : [%zu x %d]\n\n", n_ids, dim);
    printf("GPU vs CPU gather: %s\n", ok ? "ALL OK" : "MISMATCH");

    free(cpu);
    free(gpu);
    free(ids);
    embedding_free(&emb);
    bpe_free(&tok);
    return ok ? 0 : 1;
}
