// RMSNorm and activation-quantize kernels, the dequant GEMVs and the PDL / CUDA-graph launch helpers.
// Part of the forward_qwen.cu unity build: included in order, not a standalone header.

// __launch_bounds__(1024, 1) on the whole stripe-reduce rmsnorm/argmax family below:
// every kernel is always launched at exactly 1024 threads/block, and on nvcc 12.9
// ptxas allocates up to 80 regs/thread for these -- 80 x 1024 exceeds the 64K-register
// SM budget, so each launch failed at runtime (spec decode died after the first
// token). The bound forces ptxas into the 64-reg budget on any toolchain; excess
// pressure becomes spills instead of dead launches.
__global__ void __launch_bounds__(1024, 1)
k_tq_qwen_rmsnorm(float *out, const float *x, const uint16_t *weight, int N, float eps) {
    __shared__ float smem[32];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;

    float sum_sq = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float v = x[i];
        sum_sq += v * v;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }
    if (lane == 0) smem[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        sum_sq = (lane < (blockDim.x + 31) / 32) ? smem[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        }
        if (lane == 0) smem[0] = rsqrtf(sum_sq / (float)N + eps);
    }
    __syncthreads();

    float rms_inv = smem[0];
    // Each block computes the full (identical) sum above, then scales only its
    // own output stripe. This spreads the normalize pass across gridDim.x SMs
    // instead of one, while keeping the result bit-identical to the grid=1 path.
    int stripe = (N + gridDim.x - 1) / gridDim.x;
    int begin = blockIdx.x * stripe;
    int end = begin + stripe;
    if (end > N) end = N;
    for (int i = begin + tid; i < end; i += blockDim.x) {
        float w = 1.0f + tq_bf16_to_float(weight[i]);
        out[i] = x[i] * rms_inv * w;
    }
}

// Programmatic dependent launch (paged decode step, TQ_PDL): the kernels of the step call trigger + wait first
// (or, the EXL3 GEMM, wait before its first dependent access); both are no-ops for a launch without the attribute.
// wait = the predecessor grid has completed and its writes are visible; trigger = dependents may launch now.
__device__ __forceinline__ void tq_pdl_wait(void) { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void tq_pdl_trigger(void) { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ void tq_pdl_enter(void) { tq_pdl_trigger(); tq_pdl_wait(); }

// Host side: the paged decode step sets the flags. TQ_PDL=1 launches bfrag and the EXL3 GEMM with the attribute
// (g_pdl_on); TQ_PDL=2 also the step's other kernels (g_pdl_small), which measured slower at 32K: their CTAs,
// resident early and waiting, cost more than the launch latency they hide.
static int g_pdl_on = 0, g_pdl_small = 0;
template <typename... KArgs, typename... Args>
static void tq_launch(bool pdl, void (*kern)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t st,
                      Args... args) {
    if (!pdl) { kern<<<grid, block, smem, st>>>(args...); return; }
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid; cfg.blockDim = block; cfg.dynamicSmemBytes = smem; cfg.stream = st;
    cfg.attrs = attr; cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, kern, static_cast<KArgs>(args)...);
}

// CUDA graphs for the paged decode step (TQ_GRAPH=1, session 10 "decode-graph"): the step's kernel sequence is
// captured once per shape class (clients N, GQA key-split count S, MTP columns, mode flags) and replayed; the
// per-step inputs (tokens, slot ids, positions, block tables, encoded lengths, MTP control rows) reach the kernels
// through device buffers uploaded before the replay, so the replayed kernels run with exactly the eager step's
// parameters (bitwise identical). Every device buffer a captured step may reference is owned by the engine and
// (re)allocated only in a handful of places; each of them bumps g_graph_gen, which drops every cached graph.
// TQ_GRAPH=1: the paged decode step as a graph; TQ_GRAPH=2: also the verify wave and the MTP draft (measured: no
// gain over the eager spec round, and the verify key varies with the wave's column count -> more captures).
// DEFAULT 1 since 2026-10-17 (owner's decision (g), with TQ_GEMM_FORK: -0.35 ms per step at every N / context,
// bitwise); TQ_GRAPH=0 = the eager launch sequence.
static unsigned g_graph_gen = 1;
static void graph_cache_clear(void);
static void graph_invalidate(void) { g_graph_gen++; graph_cache_clear(); }

// Grid of the batched norms below (and k_tq_qwen_rmsnorm_b_out): every block of a row computes the row's FULL
// reduction and the gridDim.x stripes only split the WRITES, so the outputs do not depend on the stripe count. 8 stripes
// spread a few rows over more SMs (the decode step); for a wave of many rows (prefill, verify) one block per row does
// 1/8 of the reads -- session 17: the prefill wave's 128 launches at 11.8 us were 1.5 ms of a 48.6 ms wave.
#ifndef TQ_NORM_ONE_STRIPE_ROWS
#define TQ_NORM_ONE_STRIPE_ROWS 16
#endif
static inline dim3 tq_norm_grid(int rows) { return dim3(rows >= TQ_NORM_ONE_STRIPE_ROWS ? 1 : 8, rows); }

// Batched rmsnorm over `nodes` contiguous vectors (vector v at v*N). Grid = (stripes,
// nodes), node = blockIdx.y. Bit-identical to k_tq_qwen_rmsnorm per node (same .x stripe
// count); one launch replaces N for the Step-5 batched verify (cuts launch overhead).
__global__ void __launch_bounds__(1024, 1)
k_tq_qwen_rmsnorm_b(float *out, const float *x, const uint16_t *weight, int N, float eps) {
    tq_pdl_enter();
    __shared__ float smem[32];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    const float *xn = x + (size_t)blockIdx.y * N;
    float *outn = out + (size_t)blockIdx.y * N;
    float sum_sq = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) { float v = xn[i]; sum_sq += v * v; }
    for (int offset = 16; offset > 0; offset >>= 1) sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    if (lane == 0) smem[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        sum_sq = (lane < (blockDim.x + 31) / 32) ? smem[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        if (lane == 0) smem[0] = rsqrtf(sum_sq / (float)N + eps);
    }
    __syncthreads();
    float rms_inv = smem[0];
    int stripe = (N + gridDim.x - 1) / gridDim.x;
    int begin = blockIdx.x * stripe;
    int end = begin + stripe;
    if (end > N) end = N;
    for (int i = begin + tid; i < end; i += blockDim.x) {
        float w = 1.0f + tq_bf16_to_float(weight[i]);
        outn[i] = xn[i] * rms_inv * w;
    }
}

// Fused add-residual + RMSNorm: computes resid[i] = a[i] + b[i], then
// out[i] = rmsnorm(resid, weight). Replaces separate add_vec_b + rmsnorm_b.
__global__ void __launch_bounds__(1024, 1)
k_tq_add_rmsnorm_b(float *out, float *resid, const float *a, const float *b,
                                   const uint16_t *weight, int N, float eps) {
    tq_pdl_enter();
    __shared__ float smem[32];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    const float *an = a + (size_t)blockIdx.y * N;
    const float *bn = b + (size_t)blockIdx.y * N;
    float *rn = resid + (size_t)blockIdx.y * N;
    float *outn = out + (size_t)blockIdx.y * N;
    float sum_sq = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float v = an[i] + bn[i];
        rn[i] = v;
        sum_sq += v * v;
    }
    for (int offset = 16; offset > 0; offset >>= 1) sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    if (lane == 0) smem[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        sum_sq = (lane < (blockDim.x + 31) / 32) ? smem[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        if (lane == 0) smem[0] = rsqrtf(sum_sq / (float)N + eps);
    }
    __syncthreads();
    float rms_inv = smem[0];
    int stripe = (N + gridDim.x - 1) / gridDim.x;
    int begin = blockIdx.x * stripe;
    int end = begin + stripe;
    if (end > N) end = N;
    for (int i = begin + tid; i < end; i += blockDim.x) {
        float w = 1.0f + tq_bf16_to_float(weight[i]);
        outn[i] = rn[i] * rms_inv * w;
    }
}

__global__ void k_tq_debug_qmma_gemv(float *out, const uint8_t *a, int M, int K, int Kt,
                                     float inv_scale, const float *x) {
    __shared__ float smem[256];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= M) return;

    float sum = 0.0f;
    for (int col = tid; col < K; col += blockDim.x) {
        uint8_t q = tq_qmma_a_byte(a, Kt, row, col);
        sum += tq_e4m3_to_float(q) * inv_scale * x[col];
    }
    smem[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out[row] = smem[0];
}

__global__ void k_tq_debug_qmma_gemv_block_scaled(float *out, const uint8_t *a, int M, int K, int Kt,
                                                  const float *scale_inv, int scale_cols,
                                                  const float *x) {
    __shared__ float smem[256];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= M) return;

    int scale_row = row >> 7;
    float sum = 0.0f;
    for (int col = tid; col < K; col += blockDim.x) {
        uint8_t q = tq_qmma_a_byte(a, Kt, row, col);
        float s = scale_inv[scale_row * scale_cols + (col >> 7)];
        sum += tq_e4m3_to_float(q) * s * x[col];
    }
    smem[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out[row] = smem[0];
}

__global__ void k_tq_debug_qmma_gemv_block_scaled_rowmajor(float *out, const uint8_t *a, int M, int K,
                                                           const float *scale_inv, int scale_cols,
                                                           const float *x) {
    __shared__ float smem[256];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= M) return;

    int scale_row = row >> 7;
    const uint8_t *row_ptr = a + (size_t)row * K;
    float sum = 0.0f;
    for (int col = tid; col < K; col += blockDim.x) {
        float s = scale_inv[scale_row * scale_cols + (col >> 7)];
        sum += tq_e4m3_to_float(row_ptr[col]) * s * x[col];
    }
    smem[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out[row] = smem[0];
}

__device__ __forceinline__ uint8_t tq_float_to_e4m3(float val) {
    // E4M3FN activation encoder: full normal range up to 448 (biased exponent up to
    // 15), round-to-nearest-even (rintf), subnormals in exponent field 0, and
    // saturation to 448 (never emits the NaN code S.1111.111). Mirrors the fixed
    // weight encoder in tools/convert.py:float_to_e4m3. (Previously the biased
    // exponent was clipped to 14 -> max ~240, clipping large QMMA.SF activations.)
    float ax = fabsf(val);
    uint8_t sign = val < 0.0f ? 0x80 : 0;
    if (!isfinite(ax)) ax = 448.0f;
    if (ax < 0x1p-6f) {                          // subnormal: step 2^-9, mantissa 0..8
        int m = (int)rintf(ax * 512.0f);         // ax / 2^-9, round-half-to-even
        if (m < 0) m = 0;
        if (m > 8) m = 8;                        // m==8 carries to smallest normal (0x08)
        return sign | (uint8_t)m;
    }
    int e = (int)floorf(log2f(ax));              // unbiased exponent
    int biased = e + 7;
    int m = (int)rintf(ax / ldexpf(1.0f, e) * 8.0f) - 8;   // RNE mantissa, 0..8
    if (m >= 8) { m = 0; biased += 1; }          // mantissa overflow -> bump exponent
    if (biased > 15 || (biased == 15 && m > 6)) { biased = 15; m = 6; }   // saturate to 448
    if (biased < 1) biased = 1;
    if (m < 0) m = 0;
    else if (m > 7) m = 7;
    return sign | ((uint8_t)biased << 3) | (uint8_t)m;
}

__global__ void k_tq_quantize_bfrag_pow2_fused(uint8_t *b, float *scale_inv, const float *x, int K) {
    __shared__ float smem[128];
    __shared__ float block_scale_inv;
    int block = blockIdx.x;
    int tid = threadIdx.x;
    int base = block * 128;
    int base_tile = base >> 5;
    int remaining = K - base;
    int live_tiles = remaining >= 128 ? 4 : (remaining + 31) / 32;

    for (int i = tid; i < live_tiles * 256; i += blockDim.x) {
        b[base_tile * 256 + i] = 0;
    }

    float local = 0.0f;
    for (int i = tid; i < 128 && base + i < K; i += blockDim.x) {
        float v = fabsf(x[base + i]);
        if (v > local) local = v;
    }
    smem[tid] = local;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride && smem[tid + stride] > smem[tid]) smem[tid] = smem[tid + stride];
        __syncthreads();
    }
    if (tid == 0) {
        float raw = smem[0] > 0.0f ? smem[0] / 448.0f : 1.0f;
        block_scale_inv = exp2f(ceilf(log2f(raw)));
        scale_inv[block] = block_scale_inv;
    }
    __syncthreads();

    for (int i = tid; i < 128 && base + i < K; i += blockDim.x) {
        int idx = base + i;
        int tile = idx >> 5;
        int k_in_tile = idx & 31;
        b[tile * 256 + k_in_tile] = tq_float_to_e4m3(x[idx] / block_scale_inv);
    }
}

// E4M3 quantization with shared pow2 scale (max across columns), matching the
// per-token path. After RMSNorm the per-column absmax values are similar (~3-5),
// so the shared scale loses at most 1 pow2 step; with E4M3's 3 mantissa bits
// this still provides better precision than E5M2's 2 mantissa bits.
__global__ void k_tq_quantize_bfrag_pow2_fused_nk(uint8_t *b, float *scale_inv,
                                                  const float *x, int K, int nvar) {
    __shared__ float smem[128];
    __shared__ float block_scale_inv;
    int block = blockIdx.x;
    int tid = threadIdx.x;
    int base = block * 128;
    int base_tile = base >> 5;
    int remaining = K - base;
    int live_tiles = remaining >= 128 ? 4 : (remaining + 31) / 32;

    size_t gstride = (size_t)((K + 31) / 32) * 256;
    for (int i = tid; i < live_tiles * 256; i += blockDim.x) {
        b[base_tile * 256 + i] = 0;
        if (nvar > 8) b[gstride + base_tile * 256 + i] = 0;
    }

    float local = 0.0f;
    for (int v = 0; v < nvar; v++) {
        const float *xv = x + (size_t)v * K;
        for (int i = tid; i < 128 && base + i < K; i += blockDim.x) {
            float a = fabsf(xv[base + i]);
            if (a > local) local = a;
        }
    }
    smem[tid] = local;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride && smem[tid + stride] > smem[tid]) smem[tid] = smem[tid + stride];
        __syncthreads();
    }
    if (tid == 0) {
        float raw = smem[0] > 0.0f ? smem[0] / 448.0f : 1.0f;
        block_scale_inv = exp2f(ceilf(log2f(raw)));
        scale_inv[block] = block_scale_inv;
    }
    __syncthreads();

    for (int v = 0; v < nvar; v++) {
        const float *xv = x + (size_t)v * K;
        uint8_t *bg = b + (v >> 3) * gstride;
        int vc = v & 7;
        for (int i = tid; i < 128 && base + i < K; i += blockDim.x) {
            int idx = base + i;
            int tile = idx >> 5;
            int k_in_tile = idx & 31;
            bg[tile * 256 + 32 * vc + k_in_tile] = tq_float_to_e4m3(xv[idx] / block_scale_inv);
        }
    }
}
