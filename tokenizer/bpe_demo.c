/*
 * bpe_demo.c - Demonstrates and self-tests the CPU BPE tokenizer.
 *
 * Trains a small BPE model on a sample corpus, reports the compression ratio,
 * verifies that encode -> decode round-trips exactly, and exercises model
 * save/load. Optionally trains on a text file passed as the first argument.
 *
 * Usage:  ./bpe_demo [corpus.txt] [vocab_size]
 */
#include "bpe.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *SAMPLE =
    "the quick brown fox jumps over the lazy dog. "
    "the dog was not amused, and the fox ran away quickly. "
    "tokenization with byte pair encoding compresses repeated patterns "
    "like the, ing, and tion into single tokens for the gpt model.";

/* Read an entire file into a malloc'd buffer; sets *len. Returns NULL on error. */
static unsigned char *read_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 0) { fclose(f); return NULL; }
    unsigned char *buf = (unsigned char *)malloc((size_t)sz + 1);
    *len = fread(buf, 1, (size_t)sz, f);
    buf[*len] = 0;
    fclose(f);
    return buf;
}

static int roundtrip_ok(const BpeTokenizer *t, const unsigned char *text,
                        size_t text_len) {
    size_t n_ids = 0, dec_len = 0;
    int *ids = bpe_encode(t, text, text_len, &n_ids);
    unsigned char *dec = bpe_decode(t, ids, n_ids, &dec_len);
    int ok = (dec_len == text_len) && (memcmp(dec, text, text_len) == 0);
    free(ids);
    free(dec);
    return ok;
}

int main(int argc, char **argv) {
    size_t text_len;
    unsigned char *text = NULL;
    int vocab_size = 276; /* 256 base bytes + 20 merges */

    if (argc >= 2) {
        text = read_file(argv[1], &text_len);
        if (!text) {
            fprintf(stderr, "could not read '%s', using built-in sample\n", argv[1]);
        }
    }
    if (argc >= 3) {
        vocab_size = atoi(argv[2]);
        if (vocab_size < 256) vocab_size = 256;
    }
    if (!text) {
        text_len = strlen(SAMPLE);
        text = (unsigned char *)malloc(text_len + 1);
        memcpy(text, SAMPLE, text_len + 1);
    }

    printf("=== gpt.cu BPE tokenizer (CPU) ===\n");
    printf("corpus bytes : %zu\n", text_len);
    printf("vocab size   : %d (%d merges)\n", vocab_size, vocab_size - 256);

    BpeTokenizer tok;
    bpe_init(&tok);
    bpe_train(&tok, text, text_len, vocab_size, /*verbose=*/1);

    size_t n_ids = 0;
    int *ids = bpe_encode(&tok, text, text_len, &n_ids);
    printf("\nencoded tokens : %zu\n", n_ids);
    if (n_ids > 0)
        printf("compression    : %.3fx\n", (double)text_len / (double)n_ids);

    printf("roundtrip      : %s\n", roundtrip_ok(&tok, text, text_len) ? "OK" : "FAIL");

    /* Round-trip a short out-of-corpus string too. */
    const char *probe = "the lazy fox!";
    int probe_ok = roundtrip_ok(&tok, (const unsigned char *)probe, strlen(probe));
    printf("probe '%s' : %s\n", probe, probe_ok ? "OK" : "FAIL");

    /* Save, reload, and confirm the reloaded model still round-trips. */
    const char *model_path = "bpe.model";
    int saved = bpe_save(&tok, model_path);
    BpeTokenizer loaded;
    bpe_init(&loaded);
    int loaded_ok = (saved == 0) && (bpe_load(&loaded, model_path) == 0) &&
                    roundtrip_ok(&loaded, text, text_len);
    printf("save/load      : %s (%s)\n", loaded_ok ? "OK" : "FAIL", model_path);

    int all_ok = roundtrip_ok(&tok, text, text_len) && probe_ok && loaded_ok;

    free(ids);
    free(text);
    bpe_free(&tok);
    bpe_free(&loaded);
    return all_ok ? 0 : 1;
}
