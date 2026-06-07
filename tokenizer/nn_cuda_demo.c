/*
 * nn_cuda_demo.c - verify the CUDA nn layers match the CPU reference.
 *
 * Runs layernorm, linear, and gelu on the same deterministic random input
 * through both the CPU path (nn.h) and the GPU path (nn_cuda.h) and reports the
 * max absolute difference. CPU accumulates linear/layernorm in double while the
 * GPU uses float, so an exact match is not expected - we check a tolerance.
 */
#include "nn.h"
#include "nn_cuda.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static unsigned int g_state = 1234u;
static float frand(void) {
    g_state = g_state * 1664525u + 1013904557u;
    return ((g_state >> 8) / (float)(1u << 24)) * 2.0f - 1.0f;
}
static void fill(float *p, size_t n) {
    for (size_t i = 0; i < n; i++) p[i] = frand();
}
static float maxdiff(const float *a, const float *b, size_t n) {
    float m = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

int main(void) {
    const int T = 16, dim = 32, out_f = 64;
    const float eps = 1e-5f;
    int fails = 0;

    printf("=== gpt.cu nn layers: CUDA vs CPU ===\n");
    printf("shape : T=%d dim=%d out_f=%d\n\n", T, dim, out_f);

    /* LayerNorm */
    {
        size_t n = (size_t)T * dim;
        float *x = malloc(n * sizeof(float));
        float *g = malloc((size_t)dim * sizeof(float));
        float *b = malloc((size_t)dim * sizeof(float));
        float *c = malloc(n * sizeof(float)), *d = malloc(n * sizeof(float));
        fill(x, n); fill(g, dim); fill(b, dim);
        layernorm_forward(x, g, b, T, dim, eps, c);
        int rc = layernorm_forward_cuda(x, g, b, T, dim, eps, d);
        if (rc != 0) { printf("layernorm : CUDA error %d\n", rc); fails++; }
        else {
            float m = maxdiff(c, d, n);
            int ok = m < 1e-4f;
            printf("layernorm : max|cpu-gpu| = %.2e  %s\n", m, ok ? "OK" : "FAIL");
            if (!ok) fails++;
        }
        free(x); free(g); free(b); free(c); free(d);
    }

    /* Linear */
    {
        size_t nx = (size_t)T * dim, ny = (size_t)T * out_f;
        size_t nw = (size_t)out_f * dim;
        float *x = malloc(nx * sizeof(float));
        float *w = malloc(nw * sizeof(float));
        float *b = malloc((size_t)out_f * sizeof(float));
        float *c = malloc(ny * sizeof(float)), *d = malloc(ny * sizeof(float));
        fill(x, nx); fill(w, nw); fill(b, out_f);
        linear_forward(x, w, b, T, dim, out_f, c);
        int rc = linear_forward_cuda(x, w, b, T, dim, out_f, d);
        if (rc != 0) { printf("linear    : CUDA error %d\n", rc); fails++; }
        else {
            float m = maxdiff(c, d, ny);
            int ok = m < 1e-3f;
            printf("linear    : max|cpu-gpu| = %.2e  %s\n", m, ok ? "OK" : "FAIL");
            if (!ok) fails++;
        }
        free(x); free(w); free(b); free(c); free(d);
    }

    /* GELU */
    {
        size_t n = (size_t)T * out_f;
        float *x = malloc(n * sizeof(float));
        float *c = malloc(n * sizeof(float)), *d = malloc(n * sizeof(float));
        fill(x, n);
        gelu_forward(x, n, c);
        int rc = gelu_forward_cuda(x, n, d);
        if (rc != 0) { printf("gelu      : CUDA error %d\n", rc); fails++; }
        else {
            float m = maxdiff(c, d, n);
            int ok = m < 1e-5f;
            printf("gelu      : max|cpu-gpu| = %.2e  %s\n", m, ok ? "OK" : "FAIL");
            if (!ok) fails++;
        }
        free(x); free(c); free(d);
    }

    printf("\nresult : %s\n", fails == 0 ? "OK" : "FAIL");
    return fails == 0 ? 0 : 1;
}
