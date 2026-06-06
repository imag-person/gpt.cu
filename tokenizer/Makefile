# Makefile for the gpt.cu CPU BPE tokenizer.
CC      ?= cc
CFLAGS  ?= -O2 -std=c11 -Wall -Wextra

.PHONY: all test clean

all: bpe_demo

bpe_demo: bpe_demo.c bpe.c bpe.h
	$(CC) $(CFLAGS) -o $@ bpe_demo.c bpe.c

# Build and run the self-test demo (round-trip + save/load checks).
test: bpe_demo
	./bpe_demo

clean:
	rm -f bpe_demo bpe.model
