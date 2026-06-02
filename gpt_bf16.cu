// gpt_bf16.cu — bf16-storage variant of gpt.cu.
//
// Same model and same naive kernels as gpt.cu, but weights and activations
// are stored in __nv_bfloat16 (half the memory bandwidth of fp32). Kernels
// load bf16 -> fp32, accumulate in fp32, and store fp32 -> bf16 on the way
// out. The attention score matrix and the final lm_head logits stay in fp32
// (softmax / argmax precision matters more than storage for these).
//
// Requires sm_80+ (Ampere) for __nv_bfloat16.
// Build:  nvcc -O3 -std=c++17 -arch=sm_80 gpt_bf16.cu -o gpt_bf16
// Run:    ./gpt_bf16

#include <cuda_runtime.h>
#include <cuda_bf16.h>
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

using bf16 = __nv_bfloat16;

// Shorthand for the load/store conversions used everywhere below.
__device__ __forceinline__ float    b2f(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16     f2b(float x) { return __float2bfloat16(x); }

struct Config {
    int vocab_size;
    int n_layer;
    int n_head;
    int n_embd;
    int block_size;
};

static inline int cdiv(int a, int b) { return (a + b - 1) / b; }

// ----------------------------------------------------------------------------
// Kernels
// ----------------------------------------------------------------------------

// fp32 -> bf16 conversion. Used during weight upload (host fp32 -> device bf16).
__global__ void f32_to_bf16_kernel(const float *src, bf16 *dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = f2b(src[i]);
}

// Token + positional embedding lookup.
//   tokens: [T] int32
//   wte:    [V, C]    bf16
//   wpe:    [block_size, C]  bf16
//   out:    [T, C]    bf16
__global__ void embed_kernel(const int *tokens, const bf16 *wte,
                             const bf16 *wpe, bf16 *out,
                             int T, int C, int V) {
    int t = blockIdx.x;
    int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (t >= T || c >= C) return;
    int tok = tokens[t];
    if (tok < 0 || tok >= V) tok = 0;
    out[t * C + c] = f2b(b2f(wte[tok * C + c]) + b2f(wpe[t * C + c]));
}

// LayerNorm over the last dim. One block per row.
//   x:   [N, C]  bf16
//   g,b: [C]     bf16
//   out: [N, C]  bf16
__global__ void layernorm_kernel(const bf16 *x, const bf16 *g,
                                 const bf16 *b, bf16 *out,
                                 int N, int C, float eps) {
    int row = blockIdx.x;
    if (row >= N) return;
    extern __shared__ float smem[];

    const bf16 *xr = x + row * C;
    bf16 *or_ = out + row * C;

    // Two-pass: mean first, then variance over (x - mean)^2. Avoids the
    // E[x^2] - E[x]^2 cancellation when |x| is large relative to var.
    float local_sum = 0.f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) local_sum += b2f(xr[i]);
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
        float d = b2f(xr[i]) - mean;
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
        float n = (b2f(xr[i]) - mean) * rstd;
        or_[i] = f2b(n * b2f(g[i]) + b2f(b[i]));
    }
}

// Naive matmul: out = x @ W + (bias ? bias : 0).
//   x:    [M, K]  bf16
//   W:    [K, N]  bf16  (row-major: W[k * N + n])
//   bias: [N]     bf16 or nullptr
//   out:  [M, N]  bf16
// Accumulate in fp32 then narrow.
__global__ void matmul_kernel(const bf16 *x, const bf16 *W, const bf16 *bias,
                              bf16 *out, int M, int N, int K) {
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || n >= N) return;
    float acc = 0.f;
    const bf16 *xr = x + m * K;
    for (int k = 0; k < K; ++k) acc += b2f(xr[k]) * b2f(W[k * N + n]);
    if (bias) acc += b2f(bias[n]);
    out[m * N + n] = f2b(acc);
}

// out[i] = a[i] + b[i]
__global__ void residual_kernel(const bf16 *a, const bf16 *b, bf16 *out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = f2b(b2f(a[i]) + b2f(b[i]));
}

// GPT-2 / GELU (tanh approximation). In-place on bf16, computed in fp32.
__global__ void gelu_kernel(bf16 *x, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float v = b2f(x[i]);
    float c = 0.7978845608028654f; // sqrt(2/pi)
    float t = c * (v + 0.044715f * v * v * v);
    x[i] = f2b(0.5f * v * (1.f + tanhf(t)));
}



// Scaled dot-product attention scores with a causal mask, followed by a
// row-wise softmax. q,k stay in bf16; att stays in fp32 (softmax precision).
//   q,k: [T, H, D]  bf16
//   att: [H, T, T]  fp32
__global__ void attn_scores_softmax_kernel(const bf16 *q, const bf16 *k,
                                           float *att, int T, int H, int D) {
    int h = blockIdx.x;
    int i = blockIdx.y;
    if (h >= H || i >= T) return;
    extern __shared__ float smem[];
    float scale = rsqrtf((float)D);

    // 1. raw scores (causal). For j > i write -inf so it drops out of softmax.
    const bf16 *qv = q + (i * H + h) * D;
    for (int j = threadIdx.x; j < T; j += blockDim.x) {
        if (j > i) {
            att[(h * T + i) * T + j] = -INFINITY;
        } else {
            const bf16 *kv = k + (j * H + h) * D;
            float s = 0.f;
            for (int d = 0; d < D; ++d) s += b2f(qv[d]) * b2f(kv[d]);
            att[(h * T + i) * T + j] = s * scale;
        }
    }
    __syncthreads();

    // 2. row max.
    float local_max = -INFINITY;
    for (int j = threadIdx.x; j <= i; j += blockDim.x) {
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
    for (int j = threadIdx.x; j <= i; j += blockDim.x) {
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
        if (j <= i) att[(h * T + i) * T + j] *= inv;
        else        att[(h * T + i) * T + j]  = 0.f;
    }
}

// out = att @ V, per head. att is fp32, v and out are bf16.
//   att: [H, T, T]  fp32
//   v:   [T, H, D]  bf16
//   out: [T, H, D]  bf16
__global__ void attn_value_kernel(const float *att, const bf16 *v, bf16 *out,
                                  int T, int H, int D) {
    int t = blockIdx.x;
    int h = blockIdx.y;
    int d = threadIdx.x;
    if (t >= T || h >= H || d >= D) return;
    // No causal guard: attn_scores_softmax_kernel already zeros att[h,t,j]
    // for j > t, so those terms contribute 0.
    float acc = 0.f;
    for (int j = 0; j < T; ++j) {
        float a = att[(h * T + t) * T + j];
        acc += a * b2f(v[(j * H + h) * D + d]);
    }
    out[(t * H + h) * D + d] = f2b(acc);
}

// Split a [T, 3*C] qkv tensor into three [T, H, D] tensors.
__global__ void split_qkv_kernel(const bf16 *qkv, bf16 *q, bf16 *k, bf16 *v,
                                 int T, int C) {
    int t = blockIdx.x;
    int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (t >= T || c >= C) return;
    const bf16 *row = qkv + t * 3 * C;
    q[t * C + c] = row[0 * C + c];
    k[t * C + c] = row[1 * C + c];
    v[t * C + c] = row[2 * C + c];
}

// logits[v] = dot(x_last[:], wte[v, :]). x_last and wte are bf16; logits fp32.
__global__ void lm_head_kernel(const bf16 *x_last, const bf16 *wte,
                               float *logits, int V, int C) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= V) return;
    const bf16 *row = wte + v * C;
    float acc = 0.f;
    for (int c = 0; c < C; ++c) acc += b2f(x_last[c]) * b2f(row[c]);
    logits[v] = acc;
}


// ----------------------------------------------------------------------------
// Model
// ----------------------------------------------------------------------------

// All per-layer weights, on the device, in bf16.
struct Block {
    bf16 *ln1_g, *ln1_b;        // [C]
    bf16 *qkv_w, *qkv_b;        // [C, 3C], [3C]
    bf16 *proj_w, *proj_b;      // [C, C],  [C]
    bf16 *ln2_g, *ln2_b;        // [C]
    bf16 *fc_w,  *fc_b;         // [C, 4C], [4C]
    bf16 *fcp_w, *fcp_b;        // [4C, C], [C]
};

struct GPT {
    Config cfg;
    bf16 *wte;                  // [V, C]
    bf16 *wpe;                  // [block_size, C]
    std::vector<Block> blocks;
    bf16 *lnf_g, *lnf_b;        // [C]
    // lm_head is tied to wte.
};

// Allocate a device bf16 buffer of n elements, fill it by uploading fp32
// values from host via a temporary fp32 device buffer + conversion kernel.
// (host -> device conversions for bf16 aren't trivially available, so the
// staging round-trip keeps things portable.)
static bf16 *upload_bf16(const std::vector<float> &host) {
    int n = (int)host.size();
    bf16 *d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, n * sizeof(bf16)));
    float *staging = nullptr;
    CUDA_CHECK(cudaMalloc(&staging, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(staging, host.data(), n * sizeof(float),
                          cudaMemcpyHostToDevice));
    int threads = 256;
    f32_to_bf16_kernel<<<cdiv(n, threads), threads>>>(staging, d, n);
    CUDA_CHECK(cudaFree(staging));
    return d;
}

static bf16 *alloc_init(std::mt19937 &rng, int n, float std) {
    std::normal_distribution<float> dist(0.f, std);
    std::vector<float> host(n);
    for (int i = 0; i < n; ++i) host[i] = dist(rng);
    return upload_bf16(host);
}

static bf16 *alloc_zeros_bf16(int n) {
    bf16 *d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, n * sizeof(bf16)));
    // bf16 zero is bit-pattern 0, so memset(0) is valid.
    CUDA_CHECK(cudaMemset(d, 0, n * sizeof(bf16)));
    return d;
}

static bf16 *alloc_ones(int n) {
    return upload_bf16(std::vector<float>(n, 1.f));
}

static float *alloc_zeros_f32(int n) {
    float *d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(d, 0, n * sizeof(float)));
    return d;
}

static GPT build_model(const Config &cfg, uint32_t seed) {
    std::mt19937 rng(seed);
    GPT m;
    m.cfg = cfg;
    int C = cfg.n_embd;
    // GPT-2-style init: linear weights ~ N(0, 0.02), biases zero, LN affine
    // params at (g=1, b=0).
    m.wte = alloc_init(rng, cfg.vocab_size * C, 0.02f);
    m.wpe = alloc_init(rng, cfg.block_size * C, 0.02f);
    m.blocks.resize(cfg.n_layer);
    for (int l = 0; l < cfg.n_layer; ++l) {
        Block &b = m.blocks[l];
        b.ln1_g  = alloc_ones(C);
        b.ln1_b  = alloc_zeros_bf16(C);
        b.qkv_w  = alloc_init(rng, C * 3 * C, 0.02f);
        b.qkv_b  = alloc_zeros_bf16(3 * C);
        b.proj_w = alloc_init(rng, C * C, 0.02f);
        b.proj_b = alloc_zeros_bf16(C);
        b.ln2_g  = alloc_ones(C);
        b.ln2_b  = alloc_zeros_bf16(C);
        b.fc_w   = alloc_init(rng, C * 4 * C, 0.02f);
        b.fc_b   = alloc_zeros_bf16(4 * C);
        b.fcp_w  = alloc_init(rng, 4 * C * C, 0.02f);
        b.fcp_b  = alloc_zeros_bf16(C);
    }
    m.lnf_g = alloc_ones(C);
    m.lnf_b = alloc_zeros_bf16(C);
    return m;
}

static void free_model(GPT &m) {
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
}


// ----------------------------------------------------------------------------
// Forward pass
// ----------------------------------------------------------------------------

// Scratch buffers reused across blocks. Activations are bf16, except `att`
// (kept in fp32 for softmax precision) and `logits` (kept in fp32 for argmax).
struct Workspace {
    bf16  *x;        // [T, C]
    bf16  *h;        // [T, C]
    bf16  *t1;       // [T, C]
    bf16  *qkv;      // [T, 3C]
    bf16  *q, *k, *v;// [T, C]
    float *att;      // [H, T, T]   fp32
    bf16  *attn_out; // [T, C]
    bf16  *ff1;      // [T, 4C]
    float *logits;   // [V]         fp32
};

static Workspace make_workspace(const Config &cfg) {
    int T = cfg.block_size, C = cfg.n_embd, H = cfg.n_head, V = cfg.vocab_size;
    Workspace w{};
    w.x        = alloc_zeros_bf16(T * C);
    w.h        = alloc_zeros_bf16(T * C);
    w.t1       = alloc_zeros_bf16(T * C);
    w.qkv      = alloc_zeros_bf16(T * 3 * C);
    w.q        = alloc_zeros_bf16(T * C);
    w.k        = alloc_zeros_bf16(T * C);
    w.v        = alloc_zeros_bf16(T * C);
    w.att      = alloc_zeros_f32(H * T * T);
    w.attn_out = alloc_zeros_bf16(T * C);
    w.ff1      = alloc_zeros_bf16(T * 4 * C);
    w.logits   = alloc_zeros_f32(V);
    return w;
}

static void free_workspace(Workspace &w) {
    cudaFree(w.x); cudaFree(w.h); cudaFree(w.t1);
    cudaFree(w.qkv); cudaFree(w.q); cudaFree(w.k); cudaFree(w.v);
    cudaFree(w.att); cudaFree(w.attn_out); cudaFree(w.ff1); cudaFree(w.logits);
}

// Launch helpers.
static void launch_matmul(const bf16 *X, const bf16 *W, const bf16 *b,
                          bf16 *out, int M, int N, int K) {
    dim3 bs(16, 16);
    dim3 gs(cdiv(N, bs.x), cdiv(M, bs.y));
    matmul_kernel<<<gs, bs>>>(X, W, b, out, M, N, K);
}

static void launch_layernorm(const bf16 *x, const bf16 *g, const bf16 *bi,
                             bf16 *out, int N, int C) {
    int threads = 256;
    size_t shm  = threads * sizeof(float);
    layernorm_kernel<<<N, threads, shm>>>(x, g, bi, out, N, C, 1e-5f);
}

static void launch_residual(const bf16 *a, const bf16 *b, bf16 *out, int N) {
    int threads = 256;
    residual_kernel<<<cdiv(N, threads), threads>>>(a, b, out, N);
}

// Run a single transformer block: x <- x + attn(ln1(x)); x <- x + mlp(ln2(x)).
static void block_forward(const Block &bl, const Config &cfg, Workspace &w,
                          int T) {
    int C = cfg.n_embd, H = cfg.n_head, D = C / H;
    int NC = T * C;

    // --- attention ---
    CUDA_CHECK(cudaMemcpy(w.h, w.x, NC * sizeof(bf16), cudaMemcpyDeviceToDevice));
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
    CUDA_CHECK(cudaMemcpy(w.h, w.x, NC * sizeof(bf16), cudaMemcpyDeviceToDevice));
    launch_layernorm(w.x, bl.ln2_g, bl.ln2_b, w.t1, T, C);
    launch_matmul(w.t1, bl.fc_w, bl.fc_b, w.ff1, T, 4 * C, C);
    {
        int n = T * 4 * C, threads = 256;
        gelu_kernel<<<cdiv(n, threads), threads>>>(w.ff1, n);
    }
    launch_matmul(w.ff1, bl.fcp_w, bl.fcp_b, w.t1, T, C, 4 * C);
    launch_residual(w.h, w.t1, w.x, NC);
}

// Forward through the whole network. `tokens_d` is a device buffer of T int32
// token ids. Writes logits for the LAST position into w.logits (size V).
static void forward(const GPT &m, Workspace &w, const int *tokens_d, int T) {
    const Config &cfg = m.cfg;
    int C = cfg.n_embd, V = cfg.vocab_size;

    {
        dim3 bs(256, 1);
        dim3 gs(T, cdiv(C, 256));
        embed_kernel<<<gs, bs>>>(tokens_d, m.wte, m.wpe, w.x, T, C, V);
    }
    for (const auto &bl : m.blocks) block_forward(bl, cfg, w, T);
    launch_layernorm(w.x, m.lnf_g, m.lnf_b, w.t1, T, C);

    int threads = 256;
    lm_head_kernel<<<cdiv(V, threads), threads>>>(w.t1 + (T - 1) * C, m.wte,
                                                  w.logits, V, C);
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

    int prompt_len = 8;
    int n_generate = 16;
    uint32_t seed  = 1337;

    if (argc > 1) n_generate = std::atoi(argv[1]);

    int device = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::printf("gpt_bf16.cu — device: %s (sm_%d%d)\n", prop.name,
                prop.major, prop.minor);
    if (prop.major < 8) {
        std::fprintf(stderr,
            "warning: bf16 arithmetic requires sm_80+; this device is sm_%d%d\n",
            prop.major, prop.minor);
    }
    std::printf("config: V=%d L=%d H=%d C=%d block=%d  prompt=%d  generate=%d\n",
                cfg.vocab_size, cfg.n_layer, cfg.n_head, cfg.n_embd,
                cfg.block_size, prompt_len, n_generate);

    GPT model = build_model(cfg, seed);
    Workspace w = make_workspace(cfg);

    std::mt19937 rng(seed ^ 0xA5A5A5A5u);
    std::uniform_int_distribution<int> tok_dist(0, cfg.vocab_size - 1);
    std::vector<int> tokens;
    tokens.reserve(prompt_len + n_generate);
    for (int i = 0; i < prompt_len; ++i) tokens.push_back(tok_dist(rng));

    int *tokens_d = nullptr;
    CUDA_CHECK(cudaMalloc(&tokens_d, cfg.block_size * sizeof(int)));

    std::vector<float> logits_h(cfg.vocab_size);
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    std::printf("prompt: ");
    for (int t : tokens) std::printf("%d ", t);
    std::printf("\ngenerated:");
    std::fflush(stdout);

    cudaEventRecord(ev_start);
    for (int step = 0; step < n_generate; ++step) {
        int T = (int)tokens.size();
        if (T > cfg.block_size) {
            tokens.erase(tokens.begin(), tokens.begin() + (T - cfg.block_size));
            T = cfg.block_size;
        }
        CUDA_CHECK(cudaMemcpy(tokens_d, tokens.data(), T * sizeof(int),
                              cudaMemcpyHostToDevice));
        forward(model, w, tokens_d, T);
        CUDA_CHECK(cudaMemcpy(logits_h.data(), w.logits,
                              cfg.vocab_size * sizeof(float),
                              cudaMemcpyDeviceToHost));
        int next = 0;
        float best = logits_h[0];
        for (int v = 1; v < cfg.vocab_size; ++v) {
            if (logits_h[v] > best) { best = logits_h[v]; next = v; }
        }
        tokens.push_back(next);
        std::printf(" %d", next);
        std::fflush(stdout);
    }
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, ev_start, ev_stop);
    std::printf("\n%d tokens in %.2f ms (%.2f tok/s)\n", n_generate, ms,
                n_generate * 1000.f / ms);

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cudaFree(tokens_d);
    free_workspace(w);
    free_model(model);
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}
