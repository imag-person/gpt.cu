/* nn.c - CPU layernorm, linear, and gelu forward passes. See nn.h. */
#include "nn.h"

#include <math.h>

void layernorm_forward(const float *x, const float *gamma, const float *beta,
                       int n_rows, int dim, float eps, float *out) {
    for (int r = 0; r < n_rows; r++) {
        const float *xr = x + (size_t)r * dim;
        float *yr = out + (size_t)r * dim;
        double mean = 0.0;
        for (int i = 0; i < dim; i++) mean += xr[i];
        mean /= dim;
        double var = 0.0;
        for (int i = 0; i < dim; i++) {
            double d = xr[i] - mean;
            var += d * d;
        }
        var /= dim;
        float inv = (float)(1.0 / sqrt(var + eps));
        for (int i = 0; i < dim; i++)
            yr[i] = (float)(xr[i] - mean) * inv * gamma[i] + beta[i];
    }
}

void linear_forward(const float *x, const float *weight, const float *bias,
                    int n_rows, int in_f, int out_f, float *out) {
    for (int r = 0; r < n_rows; r++) {
        const float *xr = x + (size_t)r * in_f;
        float *yr = out + (size_t)r * out_f;
        for (int o = 0; o < out_f; o++) {
            const float *wr = weight + (size_t)o * in_f;
            double acc = bias ? (double)bias[o] : 0.0;
            for (int i = 0; i < in_f; i++) acc += (double)xr[i] * wr[i];
            yr[o] = (float)acc;
        }
    }
}

#define GELU_C 0.7978845608028654f /* sqrt(2 / pi) */

void gelu_forward(const float *x, size_t n, float *out) {
    for (size_t i = 0; i < n; i++) {
        float v = x[i];
        float inner = GELU_C * (v + 0.044715f * v * v * v);
        out[i] = 0.5f * v * (1.0f + tanhf(inner));
    }
}
