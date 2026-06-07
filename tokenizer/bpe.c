/* bpe.c - CPU byte-level BPE tokenizer implementation. See bpe.h. */
#include "bpe.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ----------------------------------------------------------------------------
 * Internal open-addressing hash map: packed (a, b) pair -> int value.
 * Used both for counting pair frequencies during training and for storing
 * merge ranks during encoding.
 * ------------------------------------------------------------------------- */
typedef struct {
    long long *keys; /* -1 marks an empty slot */
    int *vals;
    int cap;
} PairMap;

static long long pack_pair(int a, int b) {
    return ((long long)a << 32) | (unsigned int)b;
}

static unsigned long long mix64(unsigned long long x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

static void pm_init(PairMap *m, int min_cap) {
    int cap = 16;
    while (cap < min_cap * 2) cap <<= 1;
    m->cap = cap;
    m->keys = (long long *)malloc((size_t)cap * sizeof(long long));
    m->vals = (int *)malloc((size_t)cap * sizeof(int));
    for (int i = 0; i < cap; i++) m->keys[i] = -1;
}

static void pm_free(PairMap *m) {
    free(m->keys);
    free(m->vals);
    m->keys = NULL;
    m->vals = NULL;
    m->cap = 0;
}

/* Add `delta` to the value stored for (a, b), inserting if absent. */
static void pm_add(PairMap *m, int a, int b, int delta) {
    long long key = pack_pair(a, b);
    unsigned int mask = (unsigned int)m->cap - 1;
    unsigned int i = (unsigned int)mix64((unsigned long long)key) & mask;
    while (m->keys[i] != -1 && m->keys[i] != key) i = (i + 1) & mask;
    if (m->keys[i] == -1) {
        m->keys[i] = key;
        m->vals[i] = delta;
    } else {
        m->vals[i] += delta;
    }
}

/* Look up the value for (a, b); returns -1 if absent. */
static int pm_get(const PairMap *m, int a, int b) {
    long long key = pack_pair(a, b);
    unsigned int mask = (unsigned int)m->cap - 1;
    unsigned int i = (unsigned int)mix64((unsigned long long)key) & mask;
    while (m->keys[i] != -1) {
        if (m->keys[i] == key) return m->vals[i];
        i = (i + 1) & mask;
    }
    return -1;
}

/* ----------------------------------------------------------------------------
 * Tokenizer lifecycle and vocabulary construction.
 * ------------------------------------------------------------------------- */
void bpe_init(BpeTokenizer *t) {
    t->vocab_size = 0;
    t->num_merges = 0;
    t->merges = NULL;
    t->vocab = NULL;
    t->vocab_lens = NULL;
}

static void free_vocab(BpeTokenizer *t) {
    if (t->vocab) {
        for (int i = 0; i < t->vocab_size; i++) free(t->vocab[i]);
        free(t->vocab);
        t->vocab = NULL;
    }
    free(t->vocab_lens);
    t->vocab_lens = NULL;
}

void bpe_free(BpeTokenizer *t) {
    free(t->merges);
    t->merges = NULL;
    free_vocab(t);
    t->vocab_size = 0;
    t->num_merges = 0;
}

/* Build vocab[] / vocab_lens[] from the base bytes plus the merge list. */
static void build_vocab(BpeTokenizer *t) {
    free_vocab(t);
    int V = t->vocab_size;
    t->vocab = (unsigned char **)malloc((size_t)V * sizeof(unsigned char *));
    t->vocab_lens = (int *)malloc((size_t)V * sizeof(int));
    for (int i = 0; i < 256; i++) {
        t->vocab[i] = (unsigned char *)malloc(1);
        t->vocab[i][0] = (unsigned char)i;
        t->vocab_lens[i] = 1;
    }
    for (int i = 0; i < t->num_merges; i++) {
        int a = t->merges[i].first, b = t->merges[i].second;
        int id = 256 + i;
        int la = t->vocab_lens[a], lb = t->vocab_lens[b];
        t->vocab_lens[id] = la + lb;
        t->vocab[id] = (unsigned char *)malloc((size_t)(la + lb));
        memcpy(t->vocab[id], t->vocab[a], (size_t)la);
        memcpy(t->vocab[id] + la, t->vocab[b], (size_t)lb);
    }
}

/* Replace every adjacent (a, b) in ids[0..n) with new_id; returns new length. */
static size_t merge_inplace(int *ids, size_t n, int a, int b, int new_id) {
    size_t w = 0;
    for (size_t i = 0; i < n;) {
        if (i + 1 < n && ids[i] == a && ids[i + 1] == b) {
            ids[w++] = new_id;
            i += 2;
        } else {
            ids[w++] = ids[i++];
        }
    }
    return w;
}

/* ----------------------------------------------------------------------------
 * Training.
 * ------------------------------------------------------------------------- */
void bpe_train(BpeTokenizer *t, const unsigned char *text, size_t text_len,
               int vocab_size, int verbose) {
    if (vocab_size < 256) vocab_size = 256;
    int num_merges = vocab_size - 256;

    int *ids = (int *)malloc((text_len ? text_len : 1) * sizeof(int));
    size_t n = text_len;
    for (size_t i = 0; i < text_len; i++) ids[i] = text[i];

    free(t->merges);
    t->merges = (BpePair *)malloc((size_t)(num_merges ? num_merges : 1) * sizeof(BpePair));
    int done = 0;

    for (int step = 0; step < num_merges; step++) {
        if (n < 2) break;
        PairMap counts;
        pm_init(&counts, (int)n);
        for (size_t i = 0; i + 1 < n; i++)
            pm_add(&counts, ids[i], ids[i + 1], 1);

        /* Find the most frequent pair. */
        long long best_key = -1;
        int best_count = 0;
        for (int i = 0; i < counts.cap; i++) {
            if (counts.keys[i] != -1 && counts.vals[i] > best_count) {
                best_count = counts.vals[i];
                best_key = counts.keys[i];
            }
        }
        pm_free(&counts);
        if (best_key < 0 || best_count < 1) break;

        int a = (int)(best_key >> 32);
        int b = (int)(best_key & 0xffffffff);
        int new_id = 256 + step;
        n = merge_inplace(ids, n, a, b, new_id);
        t->merges[step].first = a;
        t->merges[step].second = b;
        done++;
        if (verbose)
            printf("merge %d: (%d, %d) -> %d  (count %d)\n",
                   step + 1, a, b, new_id, best_count);
    }

    free(ids);
    t->num_merges = done;
    t->vocab_size = 256 + done;
    build_vocab(t);
}

/* ----------------------------------------------------------------------------
 * Encoding / decoding.
 * ------------------------------------------------------------------------- */
int *bpe_encode(const BpeTokenizer *t, const unsigned char *text,
                size_t text_len, size_t *out_len) {
    int *ids = (int *)malloc((text_len ? text_len : 1) * sizeof(int));
    if (!ids) return NULL;
    size_t n = text_len;
    for (size_t i = 0; i < text_len; i++) ids[i] = text[i];

    /* Map each merge pair to its rank (lower rank = applied earlier). */
    PairMap ranks;
    pm_init(&ranks, t->num_merges ? t->num_merges : 1);
    for (int i = 0; i < t->num_merges; i++)
        pm_add(&ranks, t->merges[i].first, t->merges[i].second, i + 1);

    while (n >= 2) {
        int best_rank = -1, best_a = 0, best_b = 0;
        for (size_t i = 0; i + 1 < n; i++) {
            int r = pm_get(&ranks, ids[i], ids[i + 1]);
            if (r > 0 && (best_rank == -1 || r < best_rank)) {
                best_rank = r;
                best_a = ids[i];
                best_b = ids[i + 1];
            }
        }
        if (best_rank == -1) break;
        n = merge_inplace(ids, n, best_a, best_b, 256 + (best_rank - 1));
    }
    pm_free(&ranks);
    *out_len = n;
    return ids;
}

unsigned char *bpe_decode(const BpeTokenizer *t, const int *ids, size_t n_ids,
                          size_t *out_len) {
    size_t total = 0;
    for (size_t i = 0; i < n_ids; i++) {
        int id = ids[i];
        if (id >= 0 && id < t->vocab_size) total += (size_t)t->vocab_lens[id];
    }
    unsigned char *out = (unsigned char *)malloc(total ? total : 1);
    if (!out) return NULL;
    size_t w = 0;
    for (size_t i = 0; i < n_ids; i++) {
        int id = ids[i];
        if (id >= 0 && id < t->vocab_size) {
            memcpy(out + w, t->vocab[id], (size_t)t->vocab_lens[id]);
            w += (size_t)t->vocab_lens[id];
        }
    }
    *out_len = w;
    return out;
}

/* ----------------------------------------------------------------------------
 * Persistence. Format: "bpe v1\n", vocab_size, then one "a b" line per merge.
 * ------------------------------------------------------------------------- */
int bpe_save(const BpeTokenizer *t, const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) return 1;
    fprintf(f, "bpe v1\n%d\n", t->vocab_size);
    for (int i = 0; i < t->num_merges; i++)
        fprintf(f, "%d %d\n", t->merges[i].first, t->merges[i].second);
    fclose(f);
    return 0;
}

int bpe_load(BpeTokenizer *t, const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return 1;
    char header[16] = {0};
    int vocab_size = 0;
    if (fscanf(f, "%15s v1", header) != 1 || fscanf(f, "%d", &vocab_size) != 1) {
        fclose(f);
        return 2;
    }
    if (vocab_size < 256) {   /* must cover the 256 base byte tokens */
        fclose(f);
        return 2;
    }
    bpe_free(t);
    bpe_init(t);
    t->vocab_size = vocab_size;
    t->num_merges = vocab_size - 256;
    if (t->num_merges < 0) t->num_merges = 0;
    t->merges = (BpePair *)malloc((size_t)(t->num_merges ? t->num_merges : 1) * sizeof(BpePair));
    for (int i = 0; i < t->num_merges; i++) {
        if (fscanf(f, "%d %d", &t->merges[i].first, &t->merges[i].second) != 2) {
            fclose(f);
            return 3;
        }
    }
    fclose(f);
    build_vocab(t);
    return 0;
}
