/*
 * nn_cuda.h - CUDA forward passes for the gpt.cu transformer layers.
 *
 * GPU counterparts of layernorm_forward / linear_forward / gelu_forward in
 * nn.h. Each takes host buffers, manages device memory internally, and returns
 * 0 on success or non-zero on error (no CUDA device / runtime error). Results
 * match the CPU layers up to floating-point rounding.
 */
#ifndef NN_CUDA_H
#define NN_CUDA_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int layernorm_forward_cuda(const float *x, const float *gamma, const float *beta,
                           int n_rows, int dim, float eps, float *out);

int linear_forward_cuda(const float *x, const float *weight, const float *bias,
                        int n_rows, int in_f, int out_f, float *out);

int gelu_forward_cuda(const float *x, size_t n, float *out);

#ifdef __cplusplus
}
#endif

#endif /* NN_CUDA_H */
