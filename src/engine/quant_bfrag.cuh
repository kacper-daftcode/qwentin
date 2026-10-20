// activation quantizers (bfrag) and the small fused kernels around the projections.
// Part of the forward_qwen.cu unity build: included in order, not a standalone header.

// ===========================================================================
// FAST drop-in replacement for k_tq_quantize_bfrag_pow2_fused_nk (ADDITIVE).
//
// Byte-identical output, but reads x ONCE instead of twice and runs at high
// occupancy.  The baseline reads x twice (absmax pass + quant pass) and launches
// only 128 threads / 128-K block (~5% occupancy on a 170-SM RTX 5090 -> ~180
// GB/s).  This version, per 128-K block:
//   * caches the block's [nvar x 128] activations in shared memory while
//     computing the absmax (single global read of x),
//   * uses 128-bit (float4) coalesced loads and 32-bit (uchar4) packed stores,
//   * runs many warps/block (dynamic blockDim) for memory-latency hiding.
// The math is identical to the baseline: per-128-K-block absmax over all nvar
// columns, pow2 block_scale_inv = exp2f(ceilf(log2f(absmax/448))) (1.0 if 0),
// byte = e4m3(x / block_scale_inv), same packing + same zeroing of group 0 (and
// group 1 when nvar>8) -> bit-exact b and scale_inv.
//
// NOTE: each thread reads back exactly the shared slots it wrote, so the only
// cross-thread shared dependency is the scalar block_scale_inv (one syncthreads).
// Launch via tq_quant_bfrag_fast_launch (opts into >48KB dynamic shared).
// ===========================================================================
__global__ void k_tq_quantize_bfrag_pow2_fused_nk_fast(uint8_t *__restrict__ b,
                                                       float *__restrict__ scale_inv,
                                                       const float *__restrict__ x,
                                                       int K, int nvar) {
    extern __shared__ float sx[];           // [nvar][live] activation cache (<=128*nvar)
    __shared__ float red[32];               // per-warp partial maxima (<=32 warps)
    __shared__ float block_scale_inv;
    const int BDIM = blockDim.x;
    const int block = blockIdx.x;
    const int tid = threadIdx.x;
    const int base = block * 128;
    const int base_tile = base >> 5;
    const int remaining = K - base;
    const int live = remaining >= 128 ? 128 : remaining;
    const int live_tiles = remaining >= 128 ? 4 : (remaining + 31) / 32;
    const size_t gstride = (size_t)((K + 31) / 32) * 256;
    const bool vec = (live == 128) && ((K & 3) == 0);   // 16B-aligned float4 fast path

    // (1) zero group 0 and (nvar>8) group 1 -- identical to the baseline.
    for (int i = tid; i < live_tiles * 256; i += BDIM) {
        b[(size_t)base_tile * 256 + i] = 0;
        if (nvar > 8) b[gstride + (size_t)base_tile * 256 + i] = 0;
    }

    // (2) load activations into shared, accumulate this thread's running absmax.
    float local = 0.0f;
    if (vec) {
        const float4 *x4 = reinterpret_cast<const float4 *>(x);
        float4 *s4 = reinterpret_cast<float4 *>(sx);
        const int total4 = 32 * nvar;                  // 32 float4 per column-block
        const int kq = K >> 2, bq = base >> 2;
        for (int e = tid; e < total4; e += BDIM) {
            int v = e >> 5, j = e & 31;
            float4 f = __ldg(&x4[(size_t)v * kq + bq + j]);
            s4[v * 32 + j] = f;
            float m = fmaxf(fmaxf(fabsf(f.x), fabsf(f.y)), fmaxf(fabsf(f.z), fabsf(f.w)));
            local = fmaxf(local, m);
        }
    } else {
        const int total = live * nvar;
        for (int e = tid; e < total; e += BDIM) {
            int v = e / live, kk = e - v * live;
            float val = __ldg(&x[(size_t)v * K + base + kk]);
            sx[v * live + kk] = val;
            local = fmaxf(local, fabsf(val));
        }
    }

    // (3) block-wide max -> identical block_scale_inv (max is order-independent,
    //     so the reduced value is bit-identical to the baseline's serial max).
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) local = fmaxf(local, __shfl_down_sync(0xffffffffu, local, o));
    int lane = tid & 31, warp = tid >> 5;
    if (lane == 0) red[warp] = local;
    __syncthreads();
    if (warp == 0) {
        int nwarps = BDIM >> 5;
        float v = (lane < nwarps) ? red[lane] : 0.0f;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, o));
        if (lane == 0) {
            float raw = v > 0.0f ? v / 448.0f : 1.0f;
            block_scale_inv = exp2f(ceilf(log2f(raw)));
            scale_inv[block] = block_scale_inv;
        }
    }
    __syncthreads();
    const float s = block_scale_inv;

    // (4) quantize from the shared cache (each thread re-reads its own slots).
    if (vec) {
        const float4 *s4 = reinterpret_cast<const float4 *>(sx);
        const int total4 = 32 * nvar;
        for (int e = tid; e < total4; e += BDIM) {
            int v = e >> 5, j = e & 31;
            float4 f = s4[v * 32 + j];
            int kk = 4 * j;
            int tile = (base + kk) >> 5;
            int k_in_tile = (base + kk) & 31;          // 4-aligned -> uchar4 store
            int vc = v & 7;
            uint8_t *bg = b + (size_t)(v >> 3) * gstride;
            uchar4 out;
            out.x = tq_float_to_e4m3(f.x / s);
            out.y = tq_float_to_e4m3(f.y / s);
            out.z = tq_float_to_e4m3(f.z / s);
            out.w = tq_float_to_e4m3(f.w / s);
            *reinterpret_cast<uchar4 *>(&bg[(size_t)tile * 256 + 32 * vc + k_in_tile]) = out;
        }
    } else {
        const int total = live * nvar;
        for (int e = tid; e < total; e += BDIM) {
            int v = e / live, kk = e - v * live;
            int idx = base + kk;
            int tile = idx >> 5, k_in_tile = idx & 31, vc = v & 7;
            uint8_t *bg = b + (size_t)(v >> 3) * gstride;
            bg[(size_t)tile * 256 + 32 * vc + k_in_tile] = tq_float_to_e4m3(sx[v * live + kk] / s);
        }
    }
}

// blockDim for the fast quantizer. With 64KB shared (nvar=128) only one block
// fits per SM, so more warps = better latency hiding; but for small nvar a huge
// block wastes warps and inflates the reduction, so scale threads with the work
// (32*nvar float4 loads per 128-K block).
static int tq_bfrag_bdim(int nvar) {
    int work = 32 * nvar;                          // float4s per 128-K block
    return work <= 256 ? 256 : (work <= 512 ? 512 : 1024);
}

// Host launcher for the fast quantizer. Opts into >48KB dynamic shared memory
// (needed for nvar up to 128 -> 64KB) once, then launches with the cache sized
// to 128*nvar floats. Drop-in for the k_tq_quantize_bfrag_pow2_fused_nk launch.
static inline void tq_quant_bfrag_fast_launch(uint8_t *b, float *scale_inv, const float *x,
                                              int K, int nvar, cudaStream_t stream) {
    int nblocks = (K + 127) / 128;
    int bdim = tq_bfrag_bdim(nvar);
    size_t smem = (size_t)128 * (size_t)nvar * sizeof(float);
    static int s_attr = 0;
    if (!s_attr) {
        cudaFuncSetAttribute(k_tq_quantize_bfrag_pow2_fused_nk_fast,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             128 * 128 * (int)sizeof(float));   // 64KB upper bound (nvar<=128)
        s_attr = 1;
    }
    k_tq_quantize_bfrag_pow2_fused_nk_fast<<<nblocks, bdim, smem, stream>>>(b, scale_inv, x, K, nvar);
}

// Deterministic activation fill for the bit-exact gate / bench. Wide magnitude
// spread (2^-11 .. 2^8) + sign + occasional exact zeros and saturating spikes,
// distinct per element so any packing/indexing/absmax-scope bug surfaces.
__global__ void k_tq_bfrag_fill_det(float *x, long total, unsigned salt) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    unsigned h = (unsigned)i * 2654435761u + salt * 40503u + 1013904223u;
    h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;
    float mant = 1.0f + (float)(h & 0x7fffffu) / (float)0x800000u;     // [1,2)
    int e = (int)((h >> 23) % 19u) - 11;                              // [-11, 7]
    float val = ldexpf(mant, e);
    if (h & 0x80000000u) val = -val;
    unsigned r = h % 101u;
    if (r == 0u) val = 0.0f;                                          // exact zeros
    else if (r == 1u) val = (h & 0x40000000u) ? 1024.0f : -1024.0f;   // saturate to 448
    x[i] = val;
}

// MANDATORY correctness gate. Fills a deterministic x[N x K], runs the CURRENT
// kernel into (b_ref, scale_ref) and the FAST kernel into (b_new, scale_new),
// and reports the max byte diff of b (*max_abs_diff_bytes) and the max diff of
// scale_inv (*max_abs_diff_scale). Both pointers are optional. Returns 0 on
// success (caller MUST check that both diffs are exactly 0 for bit-exactness);
// negative on CUDA/alloc error.
extern "C" int qwn_quant_bfrag_check(int K, int N, float *max_abs_diff_bytes,
                                     float *max_abs_diff_scale) {
    if (K <= 0 || N <= 0) return -1;
    int nblocks = (K + 127) / 128;
    int G = (N + 7) / 8;
    size_t gstride = (size_t)((K + 31) / 32) * 256;
    size_t bbytes = (size_t)G * gstride;
    size_t xcount = (size_t)N * (size_t)K;

    float *d_x = NULL, *d_s_ref = NULL, *d_s_new = NULL;
    uint8_t *d_b_ref = NULL, *d_b_new = NULL;
    if (cudaMalloc(&d_x, xcount * sizeof(float)) != cudaSuccess) return -2;
    if (cudaMalloc(&d_b_ref, bbytes) != cudaSuccess ||
        cudaMalloc(&d_b_new, bbytes) != cudaSuccess ||
        cudaMalloc(&d_s_ref, (size_t)nblocks * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_s_new, (size_t)nblocks * sizeof(float)) != cudaSuccess) {
        cudaFree(d_x); cudaFree(d_b_ref); cudaFree(d_b_new); cudaFree(d_s_ref); cudaFree(d_s_new);
        return -3;
    }

    long tot = (long)xcount;
    k_tq_bfrag_fill_det<<<(unsigned)((tot + 255) / 256), 256>>>(d_x, tot, 0x1234u);

    // Pre-zero both b buffers exactly as the production caller does.
    cudaMemset(d_b_ref, 0, bbytes);
    cudaMemset(d_b_new, 0, bbytes);
    cudaMemset(d_s_ref, 0, (size_t)nblocks * sizeof(float));
    cudaMemset(d_s_new, 0, (size_t)nblocks * sizeof(float));

    k_tq_quantize_bfrag_pow2_fused_nk<<<nblocks, 128>>>(d_b_ref, d_s_ref, d_x, K, N);
    tq_quant_bfrag_fast_launch(d_b_new, d_s_new, d_x, K, N, 0);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "[bfrag_check K=%d N=%d] CUDA error: %s\n", K, N, cudaGetErrorString(err));
        cudaFree(d_x); cudaFree(d_b_ref); cudaFree(d_b_new); cudaFree(d_s_ref); cudaFree(d_s_new);
        return -5;
    }

    uint8_t *hb_ref = (uint8_t *)malloc(bbytes), *hb_new = (uint8_t *)malloc(bbytes);
    float *hs_ref = (float *)malloc((size_t)nblocks * sizeof(float));
    float *hs_new = (float *)malloc((size_t)nblocks * sizeof(float));
    if (!hb_ref || !hb_new || !hs_ref || !hs_new) {
        free(hb_ref); free(hb_new); free(hs_ref); free(hs_new);
        cudaFree(d_x); cudaFree(d_b_ref); cudaFree(d_b_new); cudaFree(d_s_ref); cudaFree(d_s_new);
        return -6;
    }
    cudaMemcpy(hb_ref, d_b_ref, bbytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(hb_new, d_b_new, bbytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(hs_ref, d_s_ref, (size_t)nblocks * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hs_new, d_s_new, (size_t)nblocks * sizeof(float), cudaMemcpyDeviceToHost);

    int maxb = 0; size_t first_bad = (size_t)-1;
    for (size_t i = 0; i < bbytes; i++) {
        int d = (int)hb_ref[i] - (int)hb_new[i]; if (d < 0) d = -d;
        if (d > maxb) { maxb = d; }
        if (d != 0 && first_bad == (size_t)-1) first_bad = i;
    }
    float maxs = 0.0f; int s_bitdiff = 0;
    for (int i = 0; i < nblocks; i++) {
        float d = fabsf(hs_ref[i] - hs_new[i]); if (d > maxs) maxs = d;
        unsigned a, c; memcpy(&a, &hs_ref[i], 4); memcpy(&c, &hs_new[i], 4);
        if (a != c) s_bitdiff++;
    }
    if (max_abs_diff_bytes) *max_abs_diff_bytes = (float)maxb;
    if (max_abs_diff_scale) *max_abs_diff_scale = maxs;
    fprintf(stderr,
            "[bfrag_check K=%d N=%d] bbytes=%zu nblocks=%d  max|db|=%d  max|dscale|=%g  scale_bitdiff=%d%s\n",
            K, N, bbytes, nblocks, maxb, (double)maxs, s_bitdiff,
            (maxb == 0 && s_bitdiff == 0) ? "  [BIT-EXACT]" : "  [MISMATCH]");
    if (maxb != 0)
        fprintf(stderr, "         first mismatching byte at offset %zu (ref=%u new=%u)\n",
                first_bad, (unsigned)hb_ref[first_bad], (unsigned)hb_new[first_bad]);

    free(hb_ref); free(hb_new); free(hs_ref); free(hs_new);
    cudaFree(d_x); cudaFree(d_b_ref); cudaFree(d_b_new); cudaFree(d_s_ref); cudaFree(d_s_new);
    return 0;
}

// Timing harness: CURRENT vs FAST quantizer. Reports per-launch ms and achieved
// bandwidth using the LOGICAL byte budget (x read once + b written), the same
// convention as the nsys 180 GB/s figure, so gbps_fast is the true DRAM rate and
// gbps_ref is comparable to the reported baseline. Returns 0 on success.
extern "C" int qwn_quant_bfrag_bench(int K, int N, int iters,
                                     double *t_ref_ms, double *t_fast_ms,
                                     double *gbps_ref, double *gbps_fast) {
    if (K <= 0 || N <= 0) return -1;
    if (iters < 1) iters = 1;
    int nblocks = (K + 127) / 128;
    int G = (N + 7) / 8;
    size_t gstride = (size_t)((K + 31) / 32) * 256;
    size_t bbytes = (size_t)G * gstride;
    size_t xcount = (size_t)N * (size_t)K;

    float *d_x = NULL, *d_s = NULL; uint8_t *d_b = NULL;
    if (cudaMalloc(&d_x, xcount * sizeof(float)) != cudaSuccess) return -2;
    if (cudaMalloc(&d_b, bbytes) != cudaSuccess ||
        cudaMalloc(&d_s, (size_t)nblocks * sizeof(float)) != cudaSuccess) {
        cudaFree(d_x); cudaFree(d_b); cudaFree(d_s); return -3;
    }
    k_tq_bfrag_fill_det<<<(unsigned)((xcount + 255) / 256), 256>>>(d_x, (long)xcount, 7u);
    cudaMemset(d_b, 0, bbytes);

    for (int i = 0; i < 5; i++) {                       // warmup both
        k_tq_quantize_bfrag_pow2_fused_nk<<<nblocks, 128>>>(d_b, d_s, d_x, K, N);
        tq_quant_bfrag_fast_launch(d_b, d_s, d_x, K, N, 0);
    }
    if (cudaDeviceSynchronize() != cudaSuccess) { cudaFree(d_x); cudaFree(d_b); cudaFree(d_s); return -4; }

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int i = 0; i < iters; i++)
        k_tq_quantize_bfrag_pow2_fused_nk<<<nblocks, 128>>>(d_b, d_s, d_x, K, N);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms_ref = 0.0f; cudaEventElapsedTime(&ms_ref, e0, e1); ms_ref /= iters;

    cudaEventRecord(e0);
    for (int i = 0; i < iters; i++)
        tq_quant_bfrag_fast_launch(d_b, d_s, d_x, K, N, 0);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms_fast = 0.0f; cudaEventElapsedTime(&ms_fast, e0, e1); ms_fast /= iters;
    cudaEventDestroy(e0); cudaEventDestroy(e1);

    double logical = (double)xcount * 4.0 + (double)bbytes;       // x once + b
    double gr = logical / (ms_ref * 1e-3) / 1e9;
    double gf = logical / (ms_fast * 1e-3) / 1e9;
    if (t_ref_ms) *t_ref_ms = ms_ref;
    if (t_fast_ms) *t_fast_ms = ms_fast;
    if (gbps_ref) *gbps_ref = gr;
    if (gbps_fast) *gbps_fast = gf;
    fprintf(stderr,
            "[bfrag_bench K=%d N=%d] ref=%.4f ms (%.0f GB/s)  fast=%.4f ms (%.0f GB/s)  speedup=%.2fx  bdim=%d\n",
            K, N, ms_ref, gr, ms_fast, gf, ms_ref / ms_fast, tq_bfrag_bdim(N));

    cudaFree(d_x); cudaFree(d_b); cudaFree(d_s);
    return 0;
}

static __device__ __forceinline__ uint32_t tq_ue8m0_pack4(float scale) {
    uint32_t e = (__float_as_uint(scale) >> 23) & 0xffu;
    return e * 0x01010101u;
}

static __device__ __forceinline__ void tq_prefetch_qmma_a_l2(
    const uint32_t *a_base, int Kt, int tk, int lane, int word_major, int distance) {
    int ptk = tk + distance;
    if (distance <= 0 || ptk >= Kt) return;
    int a_idx = word_major ? (ptk * 128 + lane) : (ptk * 128 + lane * 4);
    const uint32_t *ptr = &a_base[a_idx];
    asm volatile("prefetch.global.L2 [%0];" :: "l"(ptr));
}

static __device__ __forceinline__ uint32_t tq_rowmajor_a_reg(const uint8_t *a, int K, int tile_m,
                                                             int tk, int lane, int word) {
    int g = lane >> 2;
    int sub = lane & 3;
    uint32_t out = 0;
    #pragma unroll
    for (int b = 0; b < 4; b++) {
        int byte_idx = word * 4 + b;
        int half = byte_idx >> 3;
        int rem = byte_idx & 7;
        int pair = rem >> 2;
        int elem = rem & 3;
        int r = g * 2 + pair;
        int c = sub * 8 + half * 4 + elem;
        uint8_t v = a[(size_t)(tile_m * 16 + r) * K + tk * 32 + c];
        out |= ((uint32_t)v) << (8 * b);
    }
    return out;
}

__global__ __launch_bounds__(32, 1)
void k_tq_qmma_sf_gemv_rowmajor(float *out, const uint8_t *a, const uint8_t *b,
                                int K, int Kt, const float *weight_scale_inv,
                                int weight_scale_cols, const float *act_scale_inv) {
    int tile_m = blockIdx.x;
    int lane = threadIdx.x & 31;
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
    const uint32_t *b_base = (const uint32_t *)b;
    int scale_row = tile_m >> 3;

    for (int tk = 0; tk < Kt; tk++) {
        uint32_t a0 = tq_rowmajor_a_reg(a, K, tile_m, tk, lane, 0);
        uint32_t a1 = tq_rowmajor_a_reg(a, K, tile_m, tk, lane, 1);
        uint32_t a2 = tq_rowmajor_a_reg(a, K, tile_m, tk, lane, 2);
        uint32_t a3 = tq_rowmajor_a_reg(a, K, tile_m, tk, lane, 3);
        int b_idx = tk * 64 + lane * 2;
        uint32_t b0 = __ldg(&b_base[b_idx + 0]);
        uint32_t b1 = __ldg(&b_base[b_idx + 1]);
        uint32_t sfa = tq_ue8m0_pack4(weight_scale_inv[scale_row * weight_scale_cols + (weight_scale_cols == Kt ? tk : (tk >> 2))]);
        uint32_t sfb = tq_ue8m0_pack4(act_scale_inv[tk >> 2]);
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col"
            ".kind::mxf8f6f4.block_scale.scale_vec::1X"
            ".f32.e4m3.e4m3.f32.ue8m0 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%10, %11, %12, %13}, "
            "%14, {0, 1}, "
            "%15, {0, 1};"
            : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
              "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(sfa), "r"(sfb));
    }

    if ((lane & 3) == 0) {
        int g = lane >> 2;
        out[tile_m * 16 + g * 2] = c0;
        out[tile_m * 16 + g * 2 + 1] = c2;
    }
}

// Fast 16x 6-bit E2M3 -> byte-lane unpack: split the 96-bit stream into four
// 24-bit groups (funnel shifts), expand each to 4 zero-extended bytes with
// masked shifts (LOP3-fused). Bit-identical output to the per-code loop at
// ~2.5x fewer instructions -- the GEMV inner loop is issue-rate-bound, not
// bandwidth-bound, so unpack ALU directly gates DRAM throughput.
static __device__ __forceinline__ void tq_unpack_e2m3_16(uint32_t p0, uint32_t p1, uint32_t p2,
                                                         uint32_t &a0, uint32_t &a1,
                                                         uint32_t &a2, uint32_t &a3) {
    uint32_t g0 = p0;
    uint32_t g1 = __funnelshift_r(p0, p1, 24);
    uint32_t g2 = __funnelshift_r(p1, p2, 16);
    uint32_t g3 = p2 >> 8;
    a0 = (g0 & 0x3Fu) | ((g0 << 2) & 0x3F00u) | ((g0 << 4) & 0x3F0000u) | ((g0 << 6) & 0x3F000000u);
    a1 = (g1 & 0x3Fu) | ((g1 << 2) & 0x3F00u) | ((g1 << 4) & 0x3F0000u) | ((g1 << 6) & 0x3F000000u);
    a2 = (g2 & 0x3Fu) | ((g2 << 2) & 0x3F00u) | ((g2 << 4) & 0x3F0000u) | ((g2 << 6) & 0x3F000000u);
    a3 = (g3 & 0x3Fu) | ((g3 << 2) & 0x3F00u) | ((g3 << 4) & 0x3F0000u) | ((g3 << 6) & 0x3F000000u);
}

// 16x 4-bit E2M1 -> byte-lane unpack (8 nibbles per word, no straddles).
// SM120's "narrow" MMA width class decodes ONE 6-bit layout (E2M1 and E2M3 have
// identical SASS; see blackwell-isa QMMA_SF_ENCODING) -- so present each E2M1
// code as its EXACT E2M3 equivalent, which is just nibble << 2 (sign b3->b5,
// exp b2:1->b4:3, mantissa b0->b2: (1+m/2)*2^(e-1) == (1+(m<<2)/8)*2^(e-1)).
static __device__ __forceinline__ void tq_unpack_e2m1_16(uint32_t p0, uint32_t p1,
                                                         uint32_t &a0, uint32_t &a1,
                                                         uint32_t &a2, uint32_t &a3) {
    uint32_t h0 = p0, h1 = p0 >> 16, h2 = p1, h3 = p1 >> 16;
    a0 = ((h0 & 0xFu) | ((h0 << 4) & 0xF00u) | ((h0 << 8) & 0xF0000u) | ((h0 << 12) & 0xF000000u)) << 2;
    a1 = ((h1 & 0xFu) | ((h1 << 4) & 0xF00u) | ((h1 << 8) & 0xF0000u) | ((h1 << 12) & 0xF000000u)) << 2;
    a2 = ((h2 & 0xFu) | ((h2 << 4) & 0xF00u) | ((h2 << 8) & 0xF0000u) | ((h2 << 12) & 0xF000000u)) << 2;
    a3 = ((h3 & 0xFu) | ((h3 << 4) & 0xF00u) | ((h3 << 8) & 0xF0000u) | ((h3 << 12) & 0xF000000u)) << 2;
}

static __device__ __forceinline__ void tq_qmma_sf_gemv_tiled_tile_range(
    float *out, const uint32_t *a, const uint32_t *b, int Kt,
    const float *weight_scale_inv, int weight_scale_cols,
    const float *act_scale_inv, int tile_m, int tk_begin, int tk_end,
    int word_major, int prefetch_distance, int a_fmt) {
    int lane = threadIdx.x & 31;
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
    const uint32_t *b_base = b;
    int scale_row = tile_m >> 3;

    if (a_fmt == 2) {
        // HW-direct E2M3 byte containers (128 words/tile, code in low 6 bits of
        // each byte): loads feed the e2m3 QMMA with zero decode ALU. Same codes
        // and scales as the dense 6-bit payload -> bit-exact, ~2.5x fewer issue
        // slots, +33% weight bytes.
        const uint32_t *a_base = a + (size_t)tile_m * Kt * 128;
        for (int tk = tk_begin; tk < tk_end; tk++) {
            int a_idx = tk * 128 + lane * 4;
            int b_idx = tk * 64 + lane * 2;
            uint32_t a0 = __ldg(&a_base[a_idx + 0]);
            uint32_t a1 = __ldg(&a_base[a_idx + 1]);
            uint32_t a2 = __ldg(&a_base[a_idx + 2]);
            uint32_t a3 = __ldg(&a_base[a_idx + 3]);
            uint32_t b0 = __ldg(&b_base[b_idx + 0]);
            uint32_t b1 = __ldg(&b_base[b_idx + 1]);
            uint32_t sfa = tq_ue8m0_pack4(weight_scale_inv[scale_row * weight_scale_cols + (weight_scale_cols == Kt ? tk : (tk >> 2))]);
            uint32_t sfb = tq_ue8m0_pack4(act_scale_inv[tk >> 2]);
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X"
                ".f32.e2m3.e4m3.f32.ue8m0 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, {0,1}, %15, {0,1};"
                : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                  "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(sfa), "r"(sfb));
        }
    } else if (a_fmt == 1) {
        // 4-bit E2M1 (64 words/tile): unpack nibble<<2 to 6-bit twins, same MMA
        const uint32_t *a_base = a + (size_t)tile_m * Kt * 64;
        for (int tk = tk_begin; tk < tk_end; tk++) {
            int a_idx = tk * 64 + lane * 2;
            int b_idx = tk * 64 + lane * 2;
            uint32_t Q0 = __ldg(&a_base[a_idx + 0]);
            uint32_t Q1 = __ldg(&a_base[a_idx + 1]);
            uint32_t a0, a1, a2, a3;
            tq_unpack_e2m1_16(Q0, Q1, a0, a1, a2, a3);
            uint32_t b0 = __ldg(&b_base[b_idx + 0]);
            uint32_t b1 = __ldg(&b_base[b_idx + 1]);
            uint32_t sfa = tq_ue8m0_pack4(weight_scale_inv[scale_row * weight_scale_cols + (weight_scale_cols == Kt ? tk : (tk >> 2))]);
            uint32_t sfb = tq_ue8m0_pack4(act_scale_inv[tk >> 2]);
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X"
                ".f32.e2m3.e4m3.f32.ue8m0 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, {0,1}, %15, {0,1};"
                : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                  "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(sfa), "r"(sfb));
        }
    } else if (gc_weight_e2m3) {
        // Dense 6-bit E2M3 weights: 12 bytes/lane (3 uint32) per K-tile,
        // expanded to the native padded 16-byte A-fragment before the MMA.
        // B (activations) stays E4M3. Stride is Kt*96 uint32 per m16 tile.
        // Software-pipelined pair: both tiles' A/B words load before the first
        // unpack (pure load reordering; per-tile math order unchanged).
        const uint32_t *a_base = a + (size_t)tile_m * Kt * 96;
        int tk = tk_begin;
        for (; tk + 3 < tk_end; tk += 4) {
            int a_idx = tk * 96 + lane * 3;
            int b_idx = tk * 64 + lane * 2;
            uint32_t P0[3] = {
                __ldg(&a_base[a_idx + 0]), __ldg(&a_base[a_idx + 1]), __ldg(&a_base[a_idx + 2]),
            };
            uint32_t P1[3] = {
                __ldg(&a_base[a_idx + 96]), __ldg(&a_base[a_idx + 97]), __ldg(&a_base[a_idx + 98]),
            };
            uint32_t P2[3] = {
                __ldg(&a_base[a_idx + 192]), __ldg(&a_base[a_idx + 193]), __ldg(&a_base[a_idx + 194]),
            };
            uint32_t P3[3] = {
                __ldg(&a_base[a_idx + 288]), __ldg(&a_base[a_idx + 289]), __ldg(&a_base[a_idx + 290]),
            };
            uint32_t b00 = __ldg(&b_base[b_idx + 0]);
            uint32_t b01 = __ldg(&b_base[b_idx + 1]);
            uint32_t b10 = __ldg(&b_base[b_idx + 64]);
            uint32_t b11 = __ldg(&b_base[b_idx + 65]);
            uint32_t b20 = __ldg(&b_base[b_idx + 128]);
            uint32_t b21 = __ldg(&b_base[b_idx + 129]);
            uint32_t b30 = __ldg(&b_base[b_idx + 192]);
            uint32_t b31 = __ldg(&b_base[b_idx + 193]);
            #pragma unroll
            for (int half = 0; half < 4; half++) {
                const uint32_t *P = (half == 0) ? P0 : (half == 1) ? P1 : (half == 2) ? P2 : P3;
                uint32_t bb0 = (half == 0) ? b00 : (half == 1) ? b10 : (half == 2) ? b20 : b30;
                uint32_t bb1 = (half == 0) ? b01 : (half == 1) ? b11 : (half == 2) ? b21 : b31;
                int tkh = tk + half;
                uint32_t a0, a1, a2, a3;
            tq_unpack_e2m3_16(P[0], P[1], P[2], a0, a1, a2, a3);
                uint32_t sfa = tq_ue8m0_pack4(weight_scale_inv[scale_row * weight_scale_cols + (weight_scale_cols == Kt ? tkh : (tkh >> 2))]);
                uint32_t sfb = tq_ue8m0_pack4(act_scale_inv[tkh >> 2]);
                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X"
                    ".f32.e2m3.e4m3.f32.ue8m0 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, {0,1}, %15, {0,1};"
                    : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(bb0), "r"(bb1),
                      "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(sfa), "r"(sfb));
            }
        }
        for (; tk < tk_end; tk++) {
            int a_idx = tk * 96 + lane * 3;
            uint32_t P[3] = {
                __ldg(&a_base[a_idx + 0]),
                __ldg(&a_base[a_idx + 1]),
                __ldg(&a_base[a_idx + 2]),
            };
            uint32_t a0, a1, a2, a3;
            tq_unpack_e2m3_16(P[0], P[1], P[2], a0, a1, a2, a3);
            int b_idx = tk * 64 + lane * 2;
            uint32_t b0 = __ldg(&b_base[b_idx + 0]);
            uint32_t b1 = __ldg(&b_base[b_idx + 1]);
            uint32_t sfa = tq_ue8m0_pack4(weight_scale_inv[scale_row * weight_scale_cols + (weight_scale_cols == Kt ? tk : (tk >> 2))]);
            uint32_t sfb = tq_ue8m0_pack4(act_scale_inv[tk >> 2]);
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X"
                ".f32.e2m3.e4m3.f32.ue8m0 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, {0,1}, %15, {0,1};"
                : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                  "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(sfa), "r"(sfb));
        }
    } else {
        const uint32_t *a_base = a + (size_t)tile_m * Kt * 128;
        for (int tk = tk_begin; tk < tk_end; tk++) {
            tq_prefetch_qmma_a_l2(a_base, tk_end, tk, lane, word_major, prefetch_distance);
            int a_idx = word_major ? (tk * 128 + lane) : (tk * 128 + lane * 4);
            int b_idx = tk * 64 + lane * 2;
            uint32_t a0 = __ldg(&a_base[a_idx + 0]);
            uint32_t a1 = __ldg(&a_base[a_idx + (word_major ? 32 : 1)]);
            uint32_t a2 = __ldg(&a_base[a_idx + (word_major ? 64 : 2)]);
            uint32_t a3 = __ldg(&a_base[a_idx + (word_major ? 96 : 3)]);
            uint32_t b0 = __ldg(&b_base[b_idx + 0]);
            uint32_t b1 = __ldg(&b_base[b_idx + 1]);
            uint32_t sfa = tq_ue8m0_pack4(weight_scale_inv[scale_row * weight_scale_cols + (weight_scale_cols == Kt ? tk : (tk >> 2))]);
            uint32_t sfb = tq_ue8m0_pack4(act_scale_inv[tk >> 2]);
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col"
                ".kind::mxf8f6f4.block_scale.scale_vec::1X"
                ".f32.e4m3.e4m3.f32.ue8m0 "
                "{%0, %1, %2, %3}, "
                "{%4, %5, %6, %7}, "
                "{%8, %9}, "
                "{%10, %11, %12, %13}, "
                "%14, {0, 1}, "
                "%15, {0, 1};"
                : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                  "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(sfa), "r"(sfb));
        }
    }

    if ((lane & 3) == 0) {
        int g = lane >> 2;
        out[tile_m * 16 + g * 2] = c0;
        out[tile_m * 16 + g * 2 + 1] = c2;
    }
}


