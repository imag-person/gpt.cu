# gpt.cu

CUDA kernels and utilities for GPT-style models.

## Softmax

`softmax.cu` contains numerically stable row-wise softmax implementations for
contiguous `float` and FP16 tensors. Each CUDA block processes one row and
reduces over the row to compute the maximum and normalization sum before writing
normalized probabilities. The FP16 path computes reductions in FP32 and writes
FP16 outputs.

Build the standalone validation binary with:

```sh
nvcc -DSOFTMAX_TEST_MAIN softmax.cu -o softmax_test
```

Run it with:

```sh
./softmax_test
```

## GELU

`gelu.cu` contains elementwise Gaussian Error Linear Unit (GELU)
implementations for contiguous `float` and FP16 tensors. It applies the exact
GELU formula, `0.5 * x * (1 + erf(x / sqrt(2)))`, in a simple 1D CUDA kernel.
The FP16 path computes the activation in FP32 and writes FP16 outputs.

Build the standalone validation binary with:

```sh
nvcc -DGELU_TEST_MAIN gelu.cu -o gelu_test
```

Run it with:

```sh
./gelu_test
```
