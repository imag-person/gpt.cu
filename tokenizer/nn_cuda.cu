/* nn_cuda.cu - CUDA layernorm, linear, and gelu forward passes. See nn_cuda.h. */
#include "nn_cuda.h"

#include <cuda_runtime.h>
#include <math.h>

#define CUDA_OK(call)                                                       \
    do {                                                                    \
        cudaError_t err__ = (call);                                         \
        if (err__ != cudaSuccess) return (int)err__;                       \
    } while (0)

/* One thread per row: compute mean/variance, then normalize and scale. */
__global__ void layernorm_kernel(const float *x, const float *gamma,
                                 const float *beta, int n_rows, int dim,
                                 float eps, float *out) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rows) return;
    const float *xr = x + (size_t)r * dim;
    float *yr = out + (size_t)r * dim;
    float mean = 0.0f;
    for (int i = 0; i < dim; i++) mean += xr[i];
    mean /= dim;
    float var = 0.0f;
    for (int i = 0; i < dim; i++) {
        float d = xr[i] - mean;
        var += d * d;
    }
    var /= dim;
    float inv = rsqrtf(var + eps);
    for (int i = 0; i < dim; i++)
        yr[i] = (xr[i] - mean) * inv * gamma[i] + beta[i];
}

/* One thread per output element out[t, o]. */
__global__ void linear_kernel(const float *x, const float *weight,
                              const float *bias, int n_rows, int in_f,
                              int out_f, float *out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_rows * out_f) return;
    int r = idx / out_f;
    int o = idx % out_f;
    const float *xr = x + (size_t)r * in_f;
    const float *wr = weight + (size_t)o * in_f;
    float acc = bias ? bias[o] : 0.0f;
    for (int i = 0; i < in_f; i++) acc += xr[i] * wr[i];
    out[idx] = acc;
}

#define GELU_C 0.7978845608028654f /* sqrt(2 / pi) */

__global__ void gelu_kernel(const float *x, size_t n, float *out) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    float inner = GELU_C * (v + 0.044715f * v * v * v);
    out[i] = 0.5f * v * (1.0f + tanhf(inner));
}

int layernorm_forward_cuda(const float *x, const float *gamma, const float *beta,
                           int n_rows, int dim, float eps, float *out) {
    size_t n = (size_t)n_rows * dim;
    float *dx = NULL, *dg = NULL, *db = NULL, *dy = NULL;
    CUDA_OK(cudaMalloc(&dx, n * sizeof(float)));
    CUDA_OK(cudaMalloc(&dg, (size_t)dim * sizeof(float)));
    CUDA_OK(cudaMalloc(&db, (size_t)dim * sizeof(float)));
    CUDA_OK(cudaMalloc(&dy, n * sizeof(float)));
    CUDA_OK(cudaMemcpy(dx, x, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dg, gamma, (size_t)dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(db, beta, (size_t)dim * sizeof(float), cudaMemcpyHostToDevice));
    int threads = 256, blocks = (n_rows + threads - 1) / threads;
    layernorm_kernel<<<blocks, threads>>>(dx, dg, db, n_rows, dim, eps, dy);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaMemcpy(out, dy, n * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(dx); cudaFree(dg); cudaFree(db); cudaFree(dy);
    return 0;
}

int linear_forward_cuda(const float *x, const float *weight, const float *bias,
                        int n_rows, int in_f, int out_f, float *out) {
    size_t nx = (size_t)n_rows * in_f, ny = (size_t)n_rows * out_f;
    size_t nw = (size_t)out_f * in_f;
    float *dx = NULL, *dw = NULL, *dbias = NULL, *dy = NULL;
    CUDA_OK(cudaMalloc(&dx, nx * sizeof(float)));
    CUDA_OK(cudaMalloc(&dw, nw * sizeof(float)));
    CUDA_OK(cudaMalloc(&dy, ny * sizeof(float)));
    CUDA_OK(cudaMemcpy(dx, x, nx * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dw, weight, nw * sizeof(float), cudaMemcpyHostToDevice));
    if (bias) {
        CUDA_OK(cudaMalloc(&dbias, (size_t)out_f * sizeof(float)));
        CUDA_OK(cudaMemcpy(dbias, bias, (size_t)out_f * sizeof(float), cudaMemcpyHostToDevice));
    }
    int threads = 256, blocks = (int)((ny + threads - 1) / threads);
    linear_kernel<<<blocks, threads>>>(dx, dw, dbias, n_rows, in_f, out_f, dy);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaMemcpy(out, dy, ny * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(dx); cudaFree(dw); cudaFree(dbias); cudaFree(dy);
    return 0;
}

int gelu_forward_cuda(const float *x, size_t n, float *out) {
    float *dx = NULL, *dy = NULL;
    CUDA_OK(cudaMalloc(&dx, n * sizeof(float)));
    CUDA_OK(cudaMalloc(&dy, n * sizeof(float)));
    CUDA_OK(cudaMemcpy(dx, x, n * sizeof(float), cudaMemcpyHostToDevice));
    int threads = 256;
    size_t blocks = (n + threads - 1) / threads;
    gelu_kernel<<<blocks, threads>>>(dx, n, dy);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaMemcpy(out, dy, n * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(dx); cudaFree(dy);
    return 0;
}
