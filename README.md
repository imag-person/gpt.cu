# gpt.cu

CUDA kernels and utilities for GPT-style models.

## Softmax

`softmax.cu` contains a numerically stable row-wise softmax implementation for
contiguous `float` tensors. Each CUDA block processes one row and reduces over
the row to compute the maximum and normalization sum before writing normalized
probabilities.

Build the standalone validation binary with:

```sh
nvcc -DSOFTMAX_TEST_MAIN softmax.cu -o softmax_test
```

Run it with:

```sh
./softmax_test
```
