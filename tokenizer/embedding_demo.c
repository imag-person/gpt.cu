/*
 * embedding_demo.c - End-to-end CPU demo: text -> BPE ids -> embeddings.
 *
 * Trains a small BPE model, encodes a string into token ids, builds a random
 * embedding table, looks up the ids, prints the resulting [n_tokens x dim]
 * matrix shape, and verifies the gather matches the table exactly.
 */
#include "bpe.h"
#include "embedding.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *SAMPLE =
    "the quick brown fox jumps over the lazy dog. "
    "the dog was not amused, and the fox ran away quickly. "
    "tokenization with byte pair encoding compresses repeated patterns "
    "like the, ing, and tion into single tokens for the gpt model.";

int main(void) {
    const char *text = "the quick brown fox";
    const int dim = 16;

    BpeTokenizer tok;
    bpe_init(&tok);
    bpe_train(&tok, (const unsigned char *)SAMPLE, strlen(SAMPLE), 276, 0);

    size_t n_ids = 0;
    int *ids = bpe_encode(&tok, (const unsigned char *)text, strlen(text), &n_ids);

    Embedding emb;
    embedding_init(&emb, tok.vocab_size, dim);
    embedding_randomize(&emb, 1234ULL, 0.1f);

    float *out = (float *)malloc((n_ids ? n_ids : 1) * (size_t)dim * sizeof(float));
    embedding_lookup(&emb, ids, n_ids, out);

    printf("=== gpt.cu embedding lookup (CPU) ===\n");
    printf("text       : \"%s\"\n", text);
    printf("vocab size : %d\n", emb.vocab_size);
    printf("tokens     : %zu\n", n_ids);
    printf("embed dim  : %d\n", dim);
    printf("output     : [%zu x %d]\n\n", n_ids, dim);

    if (n_ids > 0) {
        printf("emb[token 0 = id %d]:", ids[0]);
        for (int j = 0; j < dim && j < 6; j++) printf(" % .4f", out[j]);
        printf(" ...\n");
    }

    /* Verify each gathered row matches the table row for that id. */
    int ok = 1;
    for (size_t i = 0; i < n_ids && ok; i++) {
        int id = ids[i];
        const float *row =
            (id >= 0 && id < emb.vocab_size) ? emb.weight + (size_t)id * dim : NULL;
        for (int j = 0; j < dim; j++) {
            float expected = row ? row[j] : 0.0f;
            if (out[i * (size_t)dim + j] != expected) { ok = 0; break; }
        }
    }
    printf("\nlookup correct : %s\n", ok ? "OK" : "FAIL");

    free(out);
    free(ids);
    embedding_free(&emb);
    bpe_free(&tok);
    return ok ? 0 : 1;
}
