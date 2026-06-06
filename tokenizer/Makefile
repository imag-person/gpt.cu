# Makefile for the gpt.cu BPE tokenizer (CPU + optional CUDA).
CC      ?= cc
CFLAGS  ?= -O2 -std=c11 -Wall -Wextra
NVCC    ?= nvcc
NVFLAGS ?= -O2

.PHONY: all test embed embed-test cuda cuda-test embed-cuda embed-cuda-test clean

all: bpe_demo embedding_demo

bpe_demo: bpe_demo.c bpe.c bpe.h
	$(CC) $(CFLAGS) -o $@ bpe_demo.c bpe.c

# Build and run the CPU self-test demo (round-trip + save/load checks).
test: bpe_demo
	./bpe_demo

# CPU embedding lookup demo (text -> BPE ids -> embeddings).
embed: embedding_demo

embedding_demo: embedding_demo.c embedding.c bpe.c embedding.h bpe.h
	$(CC) $(CFLAGS) -o $@ embedding_demo.c embedding.c bpe.c

embed-test: embedding_demo
	./embedding_demo

# Optional CUDA batch encoder. Requires nvcc and a CUDA-capable GPU at runtime.
cuda: bpe_cuda_demo

bpe_cuda_demo: bpe_cuda_demo.c bpe_cuda.cu bpe_cuda.h bpe.c bpe.h
	$(NVCC) $(NVFLAGS) -o $@ bpe_cuda_demo.c bpe_cuda.cu bpe.c

# Build and run the CUDA demo (verifies GPU output matches the CPU encoder).
cuda-test: bpe_cuda_demo
	./bpe_cuda_demo

# Optional CUDA embedding gather. Requires nvcc and a CUDA-capable GPU.
embed-cuda: embedding_cuda_demo

embedding_cuda_demo: embedding_cuda_demo.c embedding_cuda.cu embedding.c bpe.c embedding_cuda.h embedding.h bpe.h
	$(NVCC) $(NVFLAGS) -o $@ embedding_cuda_demo.c embedding_cuda.cu embedding.c bpe.c

# Build and run the CUDA embedding demo (verifies GPU gather matches the CPU).
embed-cuda-test: embedding_cuda_demo
	./embedding_cuda_demo

clean:
	rm -f bpe_demo bpe_cuda_demo embedding_demo embedding_cuda_demo bpe.model
