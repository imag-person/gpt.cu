/* embedding.c - CPU token embedding table and lookup. See embedding.h. */
#include "embedding.h"

#include <stdlib.h>
#include <string.h>

void embedding_init(Embedding *e, int vocab_size, int dim) {
    e->vocab_size = vocab_size;
    e->dim = dim;
    size_t n = (size_t)vocab_size * (size_t)dim;
    e->weight = (float *)calloc(n ? n : 1, sizeof(float));
}

/* splitmix64: a small, fast, deterministic PRNG. */
static unsigned long long sm64(unsigned long long *s) {
    unsigned long long z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

void embedding_randomize(Embedding *e, unsigned long long seed, float scale) {
    unsigned long long s = seed;
    size_t n = (size_t)e->vocab_size * (size_t)e->dim;
    for (size_t i = 0; i < n; i++) {
        /* 53-bit mantissa -> uniform double in [0, 1), then map to [-scale, scale). */
        double r = (double)(sm64(&s) >> 11) * (1.0 / 9007199254740992.0);
        e->weight[i] = (float)((2.0 * r - 1.0) * (double)scale);
    }
}

void embedding_free(Embedding *e) {
    free(e->weight);
    e->weight = NULL;
    e->vocab_size = 0;
    e->dim = 0;
}

void embedding_lookup(const Embedding *e, const int *ids, size_t n_ids,
                      float *out) {
    int dim = e->dim;
    for (size_t i = 0; i < n_ids; i++) {
        int id = ids[i];
        float *dst = out + i * (size_t)dim;
        if (id >= 0 && id < e->vocab_size)
            memcpy(dst, e->weight + (size_t)id * (size_t)dim,
                   (size_t)dim * sizeof(float));
        else
            memset(dst, 0, (size_t)dim * sizeof(float));
    }
}
