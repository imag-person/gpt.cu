/*
 * nn_demo.c - CPU self-test for the transformer-block primitives in nn.h.
 *
 * Builds a small MLP block (layernorm -> linear up -> gelu -> linear down) on
 * deterministic random input, then checks two invariants:
 *   1. after layernorm each row has ~zero mean and ~unit variance
 *   2. gelu(0) == 0 and gelu is monotonic on a small probe
 */
#include "nn.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* Tiny LCG so the demo is dependency-free and reproducible. */
static unsigned int g_state = 1234u;
static float frand(void) {
    g_state = g_state * 1664525u + 1013904557u;
    return ((g_state >> 8) / (float)(1u << 24)) * 2.0f - 1.0f; /* [-1, 1) */
}

static void fill(float *p, size_t n) {
    for (size_t i = 0; i < n; i++) p[i] = frand();
}

int main(void) {
    const int T = 4, dim = 8, hidden = 4 * 8;
    const float eps = 1e-5f;

    float *x = malloc((size_t)T * dim * sizeof(float));
    float *ln = malloc((size_t)T * dim * sizeof(float));
    float *gamma = malloc((size_t)dim * sizeof(float));
    float *beta = malloc((size_t)dim * sizeof(float));
    float *w1 = malloc((size_t)hidden * dim * sizeof(float));
    float *b1 = malloc((size_t)hidden * sizeof(float));
    float *h = malloc((size_t)T * hidden * sizeof(float));
    float *w2 = malloc((size_t)dim * hidden * sizeof(float));
    float *b2 = malloc((size_t)dim * sizeof(float));
    float *out = malloc((size_t)T * dim * sizeof(float));

    fill(x, (size_t)T * dim);
    for (int i = 0; i < dim; i++) { gamma[i] = 1.0f; beta[i] = 0.0f; }
    fill(w1, (size_t)hidden * dim);
    fill(b1, (size_t)hidden);
    fill(w2, (size_t)dim * hidden);
    fill(b2, (size_t)dim);

    printf("=== gpt.cu nn layers (CPU) ===\n");
    printf("block      : layernorm -> linear(%d->%d) -> gelu -> linear(%d->%d)\n",
           dim, hidden, hidden, dim);
    printf("tokens     : %d   dim : %d\n\n", T, dim);

    /* 1. LayerNorm sanity: each row should be ~zero mean, ~unit variance. */
    layernorm_forward(x, gamma, beta, T, dim, eps, ln);
    int ln_ok = 1;
    for (int r = 0; r < T; r++) {
        const float *row = ln + r * dim;
        double mean = 0.0;
        for (int i = 0; i < dim; i++) mean += row[i];
        mean /= dim;
        double var = 0.0;
        for (int i = 0; i < dim; i++) { double d = row[i] - mean; var += d * d; }
        var /= dim;
        if (fabs(mean) > 1e-4 || fabs(var - 1.0) > 1e-3) ln_ok = 0;
    }
    printf("layernorm row stats (mean~0, var~1) : %s\n", ln_ok ? "OK" : "FAIL");

    /* 2. MLP block forward. */
    linear_forward(ln, w1, b1, T, dim, hidden, h);
    gelu_forward(h, (size_t)T * hidden, h);
    linear_forward(h, w2, b2, T, hidden, dim, out);
    printf("mlp out[0,0..3]  : %.4f %.4f %.4f %.4f\n",
           out[0], out[1], out[2], out[3]);

    /* 3. GELU spot checks. GELU is not globally monotonic (it dips below zero
     * near x=-0.75), so we check gelu(0)=0 and monotonicity on x >= 0. */
    float probe[5] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f}, g[5];
    gelu_forward(probe, 5, g);
    int gelu_ok = (fabs(g[0]) < 1e-6f);
    for (int i = 1; i < 5; i++) if (g[i] <= g[i - 1]) gelu_ok = 0;
    printf("gelu(0)=0 & monotonic on x>=0    : %s\n", gelu_ok ? "OK" : "FAIL");

    int rc = (ln_ok && gelu_ok) ? 0 : 1;
    printf("\nresult : %s\n", rc == 0 ? "OK" : "FAIL");

    free(x); free(ln); free(gamma); free(beta); free(w1); free(b1);
    free(h); free(w2); free(b2); free(out);
    return rc;
}
