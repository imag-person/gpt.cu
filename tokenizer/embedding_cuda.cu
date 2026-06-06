/* embedding_cuda.cu - CUDA gather for the embedding table. See embedding_cuda.h. */
#include "embedding_cuda.h"

#include <stdio.h>

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

/* One thread per output element: out[idx] picks weight row ids[idx/dim]. */
__global__ static void gather_kernel(const float *weight, int vocab_size,
                                     int dim, const int *ids, size_t n_ids,
                                     float *out) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = n_ids * (size_t)dim;
    if (idx >= total) return;
    size_t row = idx / (size_t)dim;
    int col = (int)(idx % (size_t)dim);
    int id = ids[row];
    out[idx] = (id >= 0 && id < vocab_size)
                   ? weight[(size_t)id * (size_t)dim + col]
                   : 0.0f;
}

extern "C" int embedding_lookup_cuda(const Embedding *e, const int *ids,
                                     size_t n_ids, float *out) {
    int dev = 0;
    if (cudaGetDeviceCount(&dev) != cudaSuccess || dev < 1) {
        fprintf(stderr, "no CUDA device available\n");
        return 1;
    }

    float *d_w = NULL, *d_out = NULL;
    int *d_ids = NULL;
    size_t wn = (size_t)e->vocab_size * (size_t)e->dim;
    size_t on = n_ids * (size_t)e->dim;

    CUDA_TRY(cudaMalloc(&d_w, (wn ? wn : 1) * sizeof(float)));
    CUDA_TRY(cudaMalloc(&d_ids, (n_ids ? n_ids : 1) * sizeof(int)));
    CUDA_TRY(cudaMalloc(&d_out, (on ? on : 1) * sizeof(float)));
    CUDA_TRY(cudaMemcpy(d_w, e->weight, wn * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_TRY(cudaMemcpy(d_ids, ids, n_ids * sizeof(int), cudaMemcpyHostToDevice));

    {
        int threads = 256;
        int blocks = (int)((on + threads - 1) / threads);
        if (blocks < 1) blocks = 1;
        gather_kernel<<<blocks, threads>>>(d_w, e->vocab_size, e->dim, d_ids,
                                           n_ids, d_out);
        CUDA_TRY(cudaGetLastError());
        CUDA_TRY(cudaMemcpy(out, d_out, on * sizeof(float), cudaMemcpyDeviceToHost));
    }

    cudaFree(d_w);
    cudaFree(d_ids);
    cudaFree(d_out);
    return 0;

fail:
    if (d_w) cudaFree(d_w);
    if (d_ids) cudaFree(d_ids);
    if (d_out) cudaFree(d_out);
    return 2;
}
