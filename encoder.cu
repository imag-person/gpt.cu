// encoder.cu — single-file CUDA implementation of a BERT-style encoder
// transformer (bidirectional self-attention, mean-pooled classifier head).
//
// Companion to gpt.cu. Same conventions: forward pass only, fp32, naive
// kernels, no cuBLAS / cuDNN / tensor cores / kv-cache. The goal is a
// readable reference, not a fast one.
//
// Build:  nvcc -O3 -std=c++17 encoder.cu -o encoder
// Run:    ./encoder
//
// Differences vs gpt.cu:
//   - Attention is non-causal: every position attends to every position.
//   - There is no autoregressive loop and no tied lm_head. The model emits
//     contextualised hidden states; a mean-pool + linear head turns them
//     into class logits for a single forward-pass demo.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _err = (expr);                                             \
        if (_err != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",               \
                         #expr, __FILE__, __LINE__, cudaGetErrorString(_err)); \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

struct Config {
    int vocab_size;
    int n_layer;
    int n_head;
    int n_embd;
    int block_size; // max sequence length
    int n_classes;  // classifier head output size
};

static inline int cdiv(int a, int b) { return (a + b - 1) / b; }

// ----------------------------------------------------------------------------
// Kernels
// ----------------------------------------------------------------------------

// Token + positional embedding lookup. Same as gpt.cu.
__global__ void embed_kernel(const int *tokens, const float *wte,
                             const float *wpe, float *out,
                             int T, int C, int V) {
    int t = blockIdx.x;
    int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (t >= T || c >= C) return;
    int tok = tokens[t];
    if (tok < 0 || tok >= V) tok = 0;
    out[t * C + c] = wte[tok * C + c] + wpe[t * C + c];
}

// LayerNorm over the last dim. One block per row.
__global__ void layernorm_kernel(const float *x, const float *g,
                                 const float *b, float *out,
                                 int N, int C, float eps) {
    int row = blockIdx.x;
    if (row >= N) return;
    extern __shared__ float smem[];

    const float *xr = x + row * C;
    float *or_ = out + row * C;

    // Two-pass: mean first, then variance over (x - mean)^2. Avoids the
    // E[x^2] - E[x]^2 cancellation when |x| is large relative to var.
    float local_sum = 0.f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) local_sum += xr[i];
    smem[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }
    float mean = smem[0] / C;
    __syncthreads();

    float local_sq = 0.f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float d = xr[i] - mean;
        local_sq += d * d;
    }
    smem[threadIdx.x] = local_sq;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }
    float rstd = rsqrtf(smem[0] / C + eps);

    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float n = (xr[i] - mean) * rstd;
        or_[i] = n * g[i] + b[i];
    }
}

// Naive matmul: out = x @ W + (b ? bias : 0).
__global__ void matmul_kernel(const float *x, const float *W, const float *bias,
                              float *out, int M, int N, int K) {
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || n >= N) return;
    float acc = 0.f;
    const float *xr = x + m * K;
    for (int k = 0; k < K; ++k) acc += xr[k] * W[k * N + n];
    if (bias) acc += bias[n];
    out[m * N + n] = acc;
}

__global__ void residual_kernel(const float *a, const float *b, float *out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = a[i] + b[i];
}

// GPT-2 / GELU (tanh approximation). Used in the MLP just like in gpt.cu.
__global__ void gelu_kernel(float *x, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float v = x[i];
    float c = 0.7978845608028654f; // sqrt(2/pi)
    float t = c * (v + 0.044715f * v * v * v);
    x[i] = 0.5f * v * (1.f + tanhf(t));
}

// Bidirectional scaled dot-product attention + row-wise softmax. The only
// substantive change vs the decoder's kernel is that there is no causal
// mask: every query position attends to every key position.
//   q,k: [T, H, D]
//   att: [H, T, T]   att[h, i, j] = softmax_j(q_i . k_j / sqrt(D))
__global__ void attn_scores_softmax_kernel(const float *q, const float *k,
                                           float *att, int T, int H, int D) {
    int h = blockIdx.x;
    int i = blockIdx.y;
    if (h >= H || i >= T) return;
    extern __shared__ float smem[];
    float scale = rsqrtf((float)D);

    // 1. raw scores, all positions (no mask).
    const float *qv = q + (i * H + h) * D;
    for (int j = threadIdx.x; j < T; j += blockDim.x) {
        const float *kv = k + (j * H + h) * D;
        float s = 0.f;
        for (int d = 0; d < D; ++d) s += qv[d] * kv[d];
        att[(h * T + i) * T + j] = s * scale;
    }
    __syncthreads();

    // 2. row max.
    float local_max = -INFINITY;
    for (int j = threadIdx.x; j < T; j += blockDim.x) {
        float v = att[(h * T + i) * T + j];
        if (v > local_max) local_max = v;
    }
    smem[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            float a = smem[threadIdx.x];
            float b = smem[threadIdx.x + stride];
            smem[threadIdx.x] = a > b ? a : b;
        }
        __syncthreads();
    }
    float row_max = smem[0];

    // 3. exp + sum.
    float local_sum = 0.f;
    for (int j = threadIdx.x; j < T; j += blockDim.x) {
        float e = expf(att[(h * T + i) * T + j] - row_max);
        att[(h * T + i) * T + j] = e;
        local_sum += e;
    }
    smem[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }
    float row_sum = smem[0];

    // 4. normalise.
    float inv = 1.f / row_sum;
    for (int j = threadIdx.x; j < T; j += blockDim.x) {
        att[(h * T + i) * T + j] *= inv;
    }
}

// out = att @ V, per head. Sums over all key positions (no mask).
//   att: [H, T, T]
//   v:   [T, H, D]
//   out: [T, H, D]
__global__ void attn_value_kernel(const float *att, const float *v, float *out,
                                  int T, int H, int D) {
    int t = blockIdx.x;
    int h = blockIdx.y;
    int d = threadIdx.x;
    if (t >= T || h >= H || d >= D) return;
    float acc = 0.f;
    for (int j = 0; j < T; ++j) {
        float a = att[(h * T + t) * T + j];
        acc += a * v[(j * H + h) * D + d];
    }
    out[(t * H + h) * D + d] = acc;
}

// Split a [T, 3*C] qkv tensor into three [T, H, D] tensors (C = H * D).
__global__ void split_qkv_kernel(const float *qkv, float *q, float *k, float *v,
                                 int T, int C) {
    int t = blockIdx.x;
    int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (t >= T || c >= C) return;
    const float *row = qkv + t * 3 * C;
    q[t * C + c] = row[0 * C + c];
    k[t * C + c] = row[1 * C + c];
    v[t * C + c] = row[2 * C + c];
}

// Mean-pool the [T, C] hidden state into a single [C] vector. One thread per
// channel; reads T rows. T is small (<= block_size).
__global__ void mean_pool_kernel(const float *x, float *out, int T, int C) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float acc = 0.f;
    for (int t = 0; t < T; ++t) acc += x[t * C + c];
    out[c] = acc / (float)T;
}


// ----------------------------------------------------------------------------
// Model
// ----------------------------------------------------------------------------

// Per-layer weights, same layout as the decoder's Block. The kernels above
// don't depend on causality, so the structure matches.
struct Block {
    float *ln1_g, *ln1_b;       // [C]
    float *qkv_w, *qkv_b;       // [C, 3C], [3C]
    float *proj_w, *proj_b;     // [C, C],  [C]
    float *ln2_g, *ln2_b;       // [C]
    float *fc_w,  *fc_b;        // [C, 4C], [4C]
    float *fcp_w, *fcp_b;       // [4C, C], [C]
};

struct Encoder {
    Config cfg;
    float *wte;                 // [V, C]
    float *wpe;                 // [block_size, C]
    std::vector<Block> blocks;
    float *lnf_g, *lnf_b;       // [C]
    // Classifier head: pooled [C] -> [n_classes].
    float *cls_w, *cls_b;       // [C, n_classes], [n_classes]
};

static float *alloc_init(std::mt19937 &rng, int n, float std) {
    std::normal_distribution<float> dist(0.f, std);
    std::vector<float> host(n);
    for (int i = 0; i < n; ++i) host[i] = dist(rng);
    float *d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d, host.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    return d;
}

static float *alloc_zeros(int n) {
    float *d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(d, 0, n * sizeof(float)));
    return d;
}

static Encoder build_model(const Config &cfg, uint32_t seed) {
    std::mt19937 rng(seed);
    Encoder m;
    m.cfg = cfg;
    int C = cfg.n_embd;
    auto ones = [&](int n) {
        std::vector<float> host(n, 1.f);
        float *d = nullptr;
        CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d, host.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        return d;
    };

    m.wte = alloc_init(rng, cfg.vocab_size * C, 0.02f);
    m.wpe = alloc_init(rng, cfg.block_size * C, 0.02f);
    m.blocks.resize(cfg.n_layer);
    for (int l = 0; l < cfg.n_layer; ++l) {
        Block &b = m.blocks[l];
        b.ln1_g  = ones(C);
        b.ln1_b  = alloc_zeros(C);
        b.qkv_w  = alloc_init(rng, C * 3 * C, 0.02f);
        b.qkv_b  = alloc_zeros(3 * C);
        b.proj_w = alloc_init(rng, C * C, 0.02f);
        b.proj_b = alloc_zeros(C);
        b.ln2_g  = ones(C);
        b.ln2_b  = alloc_zeros(C);
        b.fc_w   = alloc_init(rng, C * 4 * C, 0.02f);
        b.fc_b   = alloc_zeros(4 * C);
        b.fcp_w  = alloc_init(rng, 4 * C * C, 0.02f);
        b.fcp_b  = alloc_zeros(C);
    }
    m.lnf_g = ones(C);
    m.lnf_b = alloc_zeros(C);
    m.cls_w = alloc_init(rng, C * cfg.n_classes, 0.02f);
    m.cls_b = alloc_zeros(cfg.n_classes);
    return m;
}

static void free_model(Encoder &m) {
    cudaFree(m.wte);
    cudaFree(m.wpe);
    for (auto &b : m.blocks) {
        cudaFree(b.ln1_g);  cudaFree(b.ln1_b);
        cudaFree(b.qkv_w);  cudaFree(b.qkv_b);
        cudaFree(b.proj_w); cudaFree(b.proj_b);
        cudaFree(b.ln2_g);  cudaFree(b.ln2_b);
        cudaFree(b.fc_w);   cudaFree(b.fc_b);
        cudaFree(b.fcp_w);  cudaFree(b.fcp_b);
    }
    cudaFree(m.lnf_g); cudaFree(m.lnf_b);
    cudaFree(m.cls_w); cudaFree(m.cls_b);
}


// ----------------------------------------------------------------------------
// Forward pass
// ----------------------------------------------------------------------------

struct Workspace {
    float *x;        // [T, C]    residual stream
    float *h;        // [T, C]    block input copy
    float *t1;       // [T, C]    LN output / proj output
    float *qkv;      // [T, 3C]
    float *q, *k, *v;// [T, C]
    float *att;      // [H, T, T]
    float *attn_out; // [T, C]
    float *ff1;      // [T, 4C]
    float *pooled;   // [C]
    float *logits;   // [n_classes]
};

static Workspace make_workspace(const Config &cfg) {
    int T = cfg.block_size, C = cfg.n_embd, H = cfg.n_head;
    Workspace w{};
    w.x        = alloc_zeros(T * C);
    w.h        = alloc_zeros(T * C);
    w.t1       = alloc_zeros(T * C);
    w.qkv      = alloc_zeros(T * 3 * C);
    w.q        = alloc_zeros(T * C);
    w.k        = alloc_zeros(T * C);
    w.v        = alloc_zeros(T * C);
    w.att      = alloc_zeros(H * T * T);
    w.attn_out = alloc_zeros(T * C);
    w.ff1      = alloc_zeros(T * 4 * C);
    w.pooled   = alloc_zeros(C);
    w.logits   = alloc_zeros(cfg.n_classes);
    return w;
}

static void free_workspace(Workspace &w) {
    cudaFree(w.x); cudaFree(w.h); cudaFree(w.t1);
    cudaFree(w.qkv); cudaFree(w.q); cudaFree(w.k); cudaFree(w.v);
    cudaFree(w.att); cudaFree(w.attn_out); cudaFree(w.ff1);
    cudaFree(w.pooled); cudaFree(w.logits);
}

static void launch_matmul(const float *X, const float *W, const float *b,
                          float *out, int M, int N, int K) {
    dim3 bs(16, 16);
    dim3 gs(cdiv(N, bs.x), cdiv(M, bs.y));
    matmul_kernel<<<gs, bs>>>(X, W, b, out, M, N, K);
}

static void launch_layernorm(const float *x, const float *g, const float *bi,
                             float *out, int N, int C) {
    int threads = 256;
    size_t shm  = threads * sizeof(float);
    layernorm_kernel<<<N, threads, shm>>>(x, g, bi, out, N, C, 1e-5f);
}

static void launch_residual(const float *a, const float *b, float *out, int N) {
    int threads = 256;
    residual_kernel<<<cdiv(N, threads), threads>>>(a, b, out, N);
}

// One encoder block: x <- x + attn(ln1(x)); x <- x + mlp(ln2(x)). Identical
// in structure to the decoder block; the only behavioural change is in the
// non-causal attention kernel above.
static void block_forward(const Block &bl, const Config &cfg, Workspace &w,
                          int T) {
    int C = cfg.n_embd, H = cfg.n_head, D = C / H;
    int NC = T * C;

    // --- attention ---
    CUDA_CHECK(cudaMemcpy(w.h, w.x, NC * sizeof(float), cudaMemcpyDeviceToDevice));
    launch_layernorm(w.x, bl.ln1_g, bl.ln1_b, w.t1, T, C);
    launch_matmul(w.t1, bl.qkv_w, bl.qkv_b, w.qkv, T, 3 * C, C);

    {
        dim3 bs(256, 1);
        dim3 gs(T, cdiv(C, 256));
        split_qkv_kernel<<<gs, bs>>>(w.qkv, w.q, w.k, w.v, T, C);
    }
    {
        int threads = 128;
        size_t shm  = threads * sizeof(float);
        dim3 gs(H, T);
        attn_scores_softmax_kernel<<<gs, threads, shm>>>(w.q, w.k, w.att, T, H, D);
    }
    {
        // D=32 (one warp/block) gives low occupancy, but keeps the per-thread
        // indexing trivial; fine for a naive reference kernel.
        dim3 gs(T, H);
        attn_value_kernel<<<gs, D>>>(w.att, w.v, w.attn_out, T, H, D);
    }
    launch_matmul(w.attn_out, bl.proj_w, bl.proj_b, w.t1, T, C, C);
    launch_residual(w.h, w.t1, w.x, NC);

    // --- mlp ---
    CUDA_CHECK(cudaMemcpy(w.h, w.x, NC * sizeof(float), cudaMemcpyDeviceToDevice));
    launch_layernorm(w.x, bl.ln2_g, bl.ln2_b, w.t1, T, C);
    launch_matmul(w.t1, bl.fc_w, bl.fc_b, w.ff1, T, 4 * C, C);
    {
        int n = T * 4 * C, threads = 256;
        gelu_kernel<<<cdiv(n, threads), threads>>>(w.ff1, n);
    }
    launch_matmul(w.ff1, bl.fcp_w, bl.fcp_b, w.t1, T, C, 4 * C);
    launch_residual(w.h, w.t1, w.x, NC);
}

// Forward through the whole encoder. `tokens_d` is a device buffer of T int32
// token ids. Writes pooled class logits into w.logits (size n_classes).
static void forward(const Encoder &m, Workspace &w, const int *tokens_d, int T) {
    const Config &cfg = m.cfg;
    int C = cfg.n_embd, V = cfg.vocab_size;

    {
        dim3 bs(256, 1);
        dim3 gs(T, cdiv(C, 256));
        embed_kernel<<<gs, bs>>>(tokens_d, m.wte, m.wpe, w.x, T, C, V);
    }
    for (const auto &bl : m.blocks) block_forward(bl, cfg, w, T);
    launch_layernorm(w.x, m.lnf_g, m.lnf_b, w.t1, T, C);

    // Mean pool over the T positions, then a single linear -> n_classes.
    int threads = 256;
    mean_pool_kernel<<<cdiv(C, threads), threads>>>(w.t1, w.pooled, T, C);
    launch_matmul(w.pooled, m.cls_w, m.cls_b, w.logits, 1, cfg.n_classes, C);
}

// ----------------------------------------------------------------------------
// Driver
// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
    Config cfg{};
    cfg.vocab_size = 1024;
    cfg.n_layer    = 4;
    cfg.n_head     = 4;
    cfg.n_embd     = 128;   // head dim = 32
    cfg.block_size = 64;
    cfg.n_classes  = 4;

    int T_in = 32;
    uint32_t seed = 1337;
    if (argc > 1) T_in = std::atoi(argv[1]);
    if (T_in > cfg.block_size) T_in = cfg.block_size;
    if (T_in < 1) T_in = 1;

    int device = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::printf("encoder.cu — device: %s (sm_%d%d)\n", prop.name,
                prop.major, prop.minor);
    std::printf("config: V=%d L=%d H=%d C=%d block=%d  T=%d  classes=%d\n",
                cfg.vocab_size, cfg.n_layer, cfg.n_head, cfg.n_embd,
                cfg.block_size, T_in, cfg.n_classes);

    Encoder model = build_model(cfg, seed);
    Workspace w = make_workspace(cfg);

    std::mt19937 rng(seed ^ 0xA5A5A5A5u);
    std::uniform_int_distribution<int> tok_dist(0, cfg.vocab_size - 1);
    std::vector<int> tokens(T_in);
    for (int i = 0; i < T_in; ++i) tokens[i] = tok_dist(rng);

    int *tokens_d = nullptr;
    CUDA_CHECK(cudaMalloc(&tokens_d, cfg.block_size * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(tokens_d, tokens.data(), T_in * sizeof(int),
                          cudaMemcpyHostToDevice));

    std::vector<float> logits_h(cfg.n_classes);
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    std::printf("input: ");
    for (int t : tokens) std::printf("%d ", t);
    std::printf("\n");

    cudaEventRecord(ev_start);
    forward(model, w, tokens_d, T_in);
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);
    CUDA_CHECK(cudaMemcpy(logits_h.data(), w.logits,
                          cfg.n_classes * sizeof(float),
                          cudaMemcpyDeviceToHost));

    int argmax = 0;
    float best = logits_h[0];
    for (int c = 1; c < cfg.n_classes; ++c) {
        if (logits_h[c] > best) { best = logits_h[c]; argmax = c; }
    }
    std::printf("logits:");
    for (float v : logits_h) std::printf(" %.4f", v);
    std::printf("\nargmax: %d\n", argmax);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, ev_start, ev_stop);
    std::printf("forward: %.2f ms (T=%d)\n", ms, T_in);

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cudaFree(tokens_d);
    free_workspace(w);
    free_model(model);
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}
