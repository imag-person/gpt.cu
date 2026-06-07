/*
 * nn.h - Core transformer-block layers for gpt.cu (CPU).
 *
 * The forward primitives that, together with attention, make up a GPT block:
 * LayerNorm, Linear (fully connected), and the GELU activation. All operate on
 * row-major float matrices where each row is one token.
 */
#ifndef NN_H
#define NN_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * LayerNorm over the last dimension. For each of `n_rows` rows of length `dim`:
 *   out = (x - mean) / sqrt(var + eps) * gamma + beta
 * x, out: [n_rows x dim] row-major. gamma, beta: [dim].
 */
void layernorm_forward(const float *x, const float *gamma, const float *beta,
                       int n_rows, int dim, float eps, float *out);

/*
 * Linear layer (PyTorch nn.Linear convention):
 *   out[t, o] = sum_i x[t, i] * weight[o, i] + bias[o]
 * x: [n_rows x in_f], weight: [out_f x in_f], bias: [out_f] or NULL,
 * out: [n_rows x out_f].
 */
void linear_forward(const float *x, const float *weight, const float *bias,
                    int n_rows, int in_f, int out_f, float *out);

/*
 * GELU activation (GPT-2 tanh approximation), elementwise over `n` values:
 *   out = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 */
void gelu_forward(const float *x, size_t n, float *out);

#ifdef __cplusplus
}
#endif

#endif /* NN_H */
