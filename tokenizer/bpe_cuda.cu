/* bpe_cuda.cu - CUDA batch encoder for the BPE tokenizer. See bpe_cuda.h. */
#include "bpe_cuda.h"

#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>

#define CUDA_TRY(call)                                                      \
    do {                                                                   \
        cudaError_t err__ = (call);                                        \
        if (err__ != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                    \
                    cudaGetErrorString(err__), __FILE__, __LINE__);        \
            goto fail;                                                     \
        }                                                                  \
    } while (0)

/* Device-side mirror of the host open-addressing pair map (lookup only). */
__device__ static long long d_pack(int a, int b) {
    return ((long long)a << 32) | (unsigned int)b;
}

__device__ static unsigned long long d_mix64(unsigned long long x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

__device__ static int d_rank(const long long *keys, const int *vals, int cap,
                             int a, int b) {
    long long key = d_pack(a, b);
    unsigned int mask = (unsigned int)cap - 1;
    unsigned int i = (unsigned int)d_mix64((unsigned long long)key) & mask;
    while (keys[i] != -1) {
        if (keys[i] == key) return vals[i];
        i = (i + 1) & mask;
    }
    return -1;
}

/* One thread per sequence: greedy lowest-rank merge, encoded in place. */
__global__ static void encode_kernel(int *work, const size_t *off, size_t n_seqs,
                                     const long long *keys, const int *vals,
                                     int cap, size_t *out_len) {
    size_t s = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_seqs) return;
    int *ids = work + off[s];
    size_t n = off[s + 1] - off[s];
    while (n >= 2) {
        int best_rank = -1, best_a = 0, best_b = 0;
        for (size_t i = 0; i + 1 < n; i++) {
            int r = d_rank(keys, vals, cap, ids[i], ids[i + 1]);
            if (r > 0 && (best_rank == -1 || r < best_rank)) {
                best_rank = r; best_a = ids[i]; best_b = ids[i + 1];
            }
        }
        if (best_rank == -1) break;
        size_t w = 0;
        for (size_t i = 0; i < n;) {
            if (i + 1 < n && ids[i] == best_a && ids[i + 1] == best_b) {
                ids[w++] = 256 + (best_rank - 1); i += 2;
            } else {
                ids[w++] = ids[i++];
            }
        }
        n = w;
    }
    out_len[s] = n;
}

extern "C" int bpe_encode_batch_cuda(const BpeTokenizer *t,
                                     const unsigned char *data,
                                     const size_t *offsets, size_t n_seqs,
                                     int **out_ids, size_t **out_offsets) {
    *out_ids = NULL;
    *out_offsets = NULL;
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count < 1) {
        fprintf(stderr, "no CUDA device available\n");
        return 1;
    }

    size_t total = offsets[n_seqs];

    /* Build the merge-rank hash table on the host (same layout as bpe.c). */
    int cap = 16;
    while (cap < (t->num_merges + 1) * 2) cap <<= 1;
    long long *h_keys = (long long *)malloc((size_t)cap * sizeof(long long));
    int *h_vals = (int *)malloc((size_t)cap * sizeof(int));
    int *h_work = (int *)malloc((total ? total : 1) * sizeof(int));
    size_t *h_outlen = (size_t *)malloc(n_seqs * sizeof(size_t));
    if (!h_keys || !h_vals || !h_work || !h_outlen) goto host_fail;
    for (int i = 0; i < cap; i++) h_keys[i] = -1;
    for (int i = 0; i < t->num_merges; i++) {
        long long key = ((long long)t->merges[i].first << 32) |
                        (unsigned int)t->merges[i].second;
        unsigned int mask = (unsigned int)cap - 1;
        unsigned long long h = key;
        h ^= h >> 33; h *= 0xff51afd7ed558ccdULL;
        h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53ULL; h ^= h >> 33;
        unsigned int j = (unsigned int)h & mask;
        while (h_keys[j] != -1) j = (j + 1) & mask;
        h_keys[j] = key; h_vals[j] = i + 1;
    }
    for (size_t i = 0; i < total; i++) h_work[i] = data[i];

    {
        int *d_work = NULL, *d_vals = NULL;
        long long *d_keys = NULL;
        size_t *d_off = NULL, *d_outlen = NULL;
        CUDA_TRY(cudaMalloc(&d_work, (total ? total : 1) * sizeof(int)));
        CUDA_TRY(cudaMalloc(&d_keys, (size_t)cap * sizeof(long long)));
        CUDA_TRY(cudaMalloc(&d_vals, (size_t)cap * sizeof(int)));
        CUDA_TRY(cudaMalloc(&d_off, (n_seqs + 1) * sizeof(size_t)));
        CUDA_TRY(cudaMalloc(&d_outlen, n_seqs * sizeof(size_t)));
        CUDA_TRY(cudaMemcpy(d_work, h_work, total * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_keys, h_keys, (size_t)cap * sizeof(long long), cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_vals, h_vals, (size_t)cap * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_off, offsets, (n_seqs + 1) * sizeof(size_t), cudaMemcpyHostToDevice));

        int threads = 128;
        int blocks = (int)((n_seqs + threads - 1) / threads);
        encode_kernel<<<blocks, threads>>>(d_work, d_off, n_seqs, d_keys, d_vals, cap, d_outlen);
        CUDA_TRY(cudaGetLastError());
        CUDA_TRY(cudaMemcpy(h_work, d_work, total * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_TRY(cudaMemcpy(h_outlen, d_outlen, n_seqs * sizeof(size_t), cudaMemcpyDeviceToHost));
        cudaFree(d_work); cudaFree(d_keys); cudaFree(d_vals);
        cudaFree(d_off); cudaFree(d_outlen);
    }

    /* Compact each sequence's prefix into contiguous output. */
    {
        size_t *o_off = (size_t *)malloc((n_seqs + 1) * sizeof(size_t));
        size_t acc = 0;
        for (size_t s = 0; s < n_seqs; s++) { o_off[s] = acc; acc += h_outlen[s]; }
        o_off[n_seqs] = acc;
        int *o_ids = (int *)malloc((acc ? acc : 1) * sizeof(int));
        for (size_t s = 0; s < n_seqs; s++)
            for (size_t i = 0; i < h_outlen[s]; i++)
                o_ids[o_off[s] + i] = h_work[offsets[s] + i];
        *out_ids = o_ids;
        *out_offsets = o_off;
    }

    free(h_keys); free(h_vals); free(h_work); free(h_outlen);
    return 0;

fail:
    free(h_keys); free(h_vals); free(h_work); free(h_outlen);
    return 2;
host_fail:
    free(h_keys); free(h_vals); free(h_work); free(h_outlen);
    return 3;
}
