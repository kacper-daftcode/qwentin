// wide FP6 projection GEMMs and their launchers.
// Part of the forward_qwen.cu unity build: included in order, not a standalone header.

// ===================== D.1: wide FP6 projection GEMM =====================
// NG = ceil(N/8) column groups: the e2m3 weight A-fragment is loaded ONCE per k-tile and
// reused across all NG m16n8 MMAs, so a weight tile is read once and amortized
// over NG*8 token-columns (the compute-bound prefill lever). e2m3 weight x e4m3
// activation, QMMA.SF block-scaled (shared pow2 activation scale per 128-K block,
// validated for prefill via needle). The batched path's FP6 projections (wide_proj) run it;
// qwn_wide_gemm_check / _bench test it. Accumulators c[NG][4] live in registers (NG=32 -> 128 regs/thread).
template<int NG>
__global__ void k_tq_fp6_wide_gemm(float *out, const uint32_t *a, const uint32_t *b,
                                   int M, int Kt, const float *wscale_inv,
                                   int wscale_cols, const float *ascale_inv, int nvar) {
    int tile_m = blockIdx.x;
    int lane = threadIdx.x & 31;
    float c[NG][4];
    #pragma unroll
    for (int g = 0; g < NG; g++) { c[g][0] = c[g][1] = c[g][2] = c[g][3] = 0.0f; }
    int scale_row = tile_m >> 3;
    const uint32_t *a_base = a + (size_t)tile_m * Kt * 96;   // e2m3: 96 words / k-tile
    const size_t bgstride = (size_t)Kt * 64;                 // uint32 per column-group
    for (int tk = 0; tk < Kt; tk++) {
        int a_idx = tk * 96 + lane * 3;
        uint32_t P0 = __ldg(&a_base[a_idx + 0]);
        uint32_t P1 = __ldg(&a_base[a_idx + 1]);
        uint32_t P2 = __ldg(&a_base[a_idx + 2]);
        uint32_t a0, a1, a2, a3;
        tq_unpack_e2m3_16(P0, P1, P2, a0, a1, a2, a3);
        uint32_t sfa = tq_ue8m0_pack4(wscale_inv[scale_row * wscale_cols + (wscale_cols == Kt ? tk : (tk >> 2))]);
        uint32_t sfb = tq_ue8m0_pack4(ascale_inv[tk >> 2]);
        int b_idx = tk * 64 + lane * 2;
        #pragma unroll
        for (int g = 0; g < NG; g++) {
            uint32_t b0 = __ldg(&b[(size_t)g * bgstride + b_idx + 0]);
            uint32_t b1 = __ldg(&b[(size_t)g * bgstride + b_idx + 1]);
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X"
                ".f32.e2m3.e4m3.f32.ue8m0 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, {0,1}, %15, {0,1};"
                : "+f"(c[g][0]), "+f"(c[g][1]), "+f"(c[g][2]), "+f"(c[g][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                  "f"(c[g][0]), "f"(c[g][1]), "f"(c[g][2]), "f"(c[g][3]), "r"(sfa), "r"(sfb));
        }
    }
    int cc = lane & 3;
    int g_row = lane >> 2;
    #pragma unroll
    for (int g = 0; g < NG; g++) {
        int cola = g * 8 + 2 * cc;
        int colb = g * 8 + 2 * cc + 1;
        if (cola < nvar) {
            out[(size_t)cola * M + tile_m * 16 + 2 * g_row + 0] = c[g][0];
            out[(size_t)cola * M + tile_m * 16 + 2 * g_row + 1] = c[g][2];
        }
        if (colb < nvar) {
            out[(size_t)colb * M + tile_m * 16 + 2 * g_row + 0] = c[g][1];
            out[(size_t)colb * M + tile_m * 16 + 2 * g_row + 1] = c[g][3];
        }
    }
}

// K-split wide GEMM: blockIdx.y = split processes a Kt sub-range -> partials
// [split][nvar*M], summed by k_tq_reduce_qmma_sf_splits. Fills the GPU for
// large-K / small-Mt weights (mlp_down K=17408 -> few tiles, long serial K).
__global__ void k_tq_reduce_qmma_sf_splits(float *out, const float *partials, int M, int k_splits);

template<int NG>
__global__ void k_tq_fp6_wide_gemm_split(float *partials, const uint32_t *a, const uint32_t *b,
                                         int M, int Kt, const float *wscale_inv, int wscale_cols,
                                         const float *ascale_inv, int nvar, int k_splits) {
    int tile_m = blockIdx.x;
    int split = blockIdx.y;
    int lane = threadIdx.x & 31;
    float c[NG][4];
    #pragma unroll
    for (int g = 0; g < NG; g++) { c[g][0] = c[g][1] = c[g][2] = c[g][3] = 0.0f; }
    int scale_row = tile_m >> 3;
    int kt_per = Kt / k_splits;
    int tk_begin = split * kt_per, tk_end = tk_begin + kt_per;
    const uint32_t *a_base = a + (size_t)tile_m * Kt * 96;
    const size_t bgstride = (size_t)Kt * 64;
    for (int tk = tk_begin; tk < tk_end; tk++) {
        int a_idx = tk * 96 + lane * 3;
        uint32_t P0 = __ldg(&a_base[a_idx + 0]);
        uint32_t P1 = __ldg(&a_base[a_idx + 1]);
        uint32_t P2 = __ldg(&a_base[a_idx + 2]);
        uint32_t a0, a1, a2, a3;
        tq_unpack_e2m3_16(P0, P1, P2, a0, a1, a2, a3);
        uint32_t sfa = tq_ue8m0_pack4(wscale_inv[scale_row * wscale_cols + (wscale_cols == Kt ? tk : (tk >> 2))]);
        uint32_t sfb = tq_ue8m0_pack4(ascale_inv[tk >> 2]);
        int b_idx = tk * 64 + lane * 2;
        #pragma unroll
        for (int g = 0; g < NG; g++) {
            uint32_t b0 = __ldg(&b[(size_t)g * bgstride + b_idx + 0]);
            uint32_t b1 = __ldg(&b[(size_t)g * bgstride + b_idx + 1]);
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X"
                ".f32.e2m3.e4m3.f32.ue8m0 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, {0,1}, %15, {0,1};"
                : "+f"(c[g][0]), "+f"(c[g][1]), "+f"(c[g][2]), "+f"(c[g][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                  "f"(c[g][0]), "f"(c[g][1]), "f"(c[g][2]), "f"(c[g][3]), "r"(sfa), "r"(sfb));
        }
    }
    int cc = lane & 3, g_row = lane >> 2;
    float *pout = partials + (size_t)split * nvar * M;
    #pragma unroll
    for (int g = 0; g < NG; g++) {
        int cola = g * 8 + 2 * cc, colb = g * 8 + 2 * cc + 1;
        if (cola < nvar) {
            pout[(size_t)cola * M + tile_m * 16 + 2 * g_row + 0] = c[g][0];
            pout[(size_t)cola * M + tile_m * 16 + 2 * g_row + 1] = c[g][2];
        }
        if (colb < nvar) {
            pout[(size_t)colb * M + tile_m * 16 + 2 * g_row + 0] = c[g][1];
            pout[(size_t)colb * M + tile_m * 16 + 2 * g_row + 1] = c[g][3];
        }
    }
}

// instantiated NG = ceil(N/8) rounded UP to a power of 2 (1,2,4,8,16,32); the kernel
// masks columns >= nvar, so a chunk of e.g. 80 tokens (10 groups) runs as NG=16.
static inline int wide_ng(int N) { int g = (N + 7) / 8, ng = 1; while (ng < g) ng <<= 1; return ng; }

#define TQ_WIDE_SINGLE(NG) k_tq_fp6_wide_gemm<NG><<<Mt, 32, 0, st>>>(out, a, bb, M, Kt, ws, sc, act_scale, N)
#define TQ_WIDE_SPLIT(NG)  k_tq_fp6_wide_gemm_split<NG><<<grid, 32, 0, st>>>(partials, a, bb, M, Kt, ws, sc, act_scale, N, k_splits)
#define TQ_WIDE_SWITCH(LAUNCH) \
    switch (G) { \
        case 1:  LAUNCH(1);  break; case 2:  LAUNCH(2);  break; \
        case 4:  LAUNCH(4);  break; case 8:  LAUNCH(8);  break; \
        case 16: LAUNCH(16); break; case 32: LAUNCH(32); break; \
        default: return -1; }   /* bench N in {8,16,32,64,128,256} -> G in {1,2,4,8,16,32} */

static int launch_fp6_wide_gemm_tile(float *out, const tq_qmma_weight_t *w, const uint8_t *b,
                                     const float *act_scale, int N, float *partials, int k_splits,
                                     cudaStream_t st) {
    const uint32_t *a = (const uint32_t *)w->d_A;
    const uint32_t *bb = (const uint32_t *)b;
    int M = w->M, Kt = w->Kt, Mt = w->Mt, sc = w->scale_cols;
    const float *ws = w->d_block_scale_inv;
    int G = wide_ng(N);                          // pow2 group count; kernel masks col>=N
    if (k_splits > 1 && (Kt % k_splits) == 0 && partials) {
        dim3 grid(Mt, k_splits);
        TQ_WIDE_SWITCH(TQ_WIDE_SPLIT);
        k_tq_reduce_qmma_sf_splits<<<((size_t)N * M + 255) / 256, 256, 0, st>>>(out, partials, N * M, k_splits);
        return 0;
    }
    TQ_WIDE_SWITCH(TQ_WIDE_SINGLE);
    return 0;
}
#undef TQ_WIDE_SINGLE
#undef TQ_WIDE_SPLIT
#undef TQ_WIDE_SWITCH

// N-tiling wrapper: cap columns per launch at 128 (NG<=16 -> <=64 accumulator
// registers/thread). N=256 with NG=32 (128 acc regs) spilled and REGRESSED
// (gate 285->392us); two 128-col tiles scale linearly instead. Each tile reads
// the weight once; the shared per-128K-block activation scale is global so a
// sub-tile reuses it unchanged. b is grouped 8 cols/group (Kt*256 bytes/group).
static int launch_fp6_wide_gemm(float *out, const tq_qmma_weight_t *w, const uint8_t *b,
                                const float *act_scale, int N, float *partials, int k_splits,
                                cudaStream_t st) {
    const int TILE = 128;
    int M = w->M, Kt = w->Kt;
    for (int off = 0; off < N; off += TILE) {
        int nt = (N - off < TILE) ? (N - off) : TILE;
        int rc = launch_fp6_wide_gemm_tile(out + (size_t)off * M, w,
                                           b + (size_t)(off / 8) * Kt * 256,
                                           act_scale, nt, partials, k_splits, st);
        if (rc) return rc;
    }
    return 0;
}

static __device__ __forceinline__ void tq_qmma_sf_gemv_tiled_tile(
    float *out, const uint32_t *a, const uint32_t *b, int Kt,
    const float *weight_scale_inv, int weight_scale_cols,
    const float *act_scale_inv, int tile_m, int word_major, int prefetch_distance,
    int a_byte) {
    tq_qmma_sf_gemv_tiled_tile_range(out, a, b, Kt, weight_scale_inv, weight_scale_cols,
                                     act_scale_inv, tile_m, 0, Kt, word_major, prefetch_distance,
                                     a_byte);
}

__global__ __launch_bounds__(32, 1)
void k_tq_qmma_sf_gemv_tiled(float *out, const uint32_t *a, const uint32_t *b,
                             int Kt, const float *weight_scale_inv, int weight_scale_cols,
                             const float *act_scale_inv, int word_major, int prefetch_distance,
                             int a_byte) {
    tq_qmma_sf_gemv_tiled_tile(out, a, b, Kt, weight_scale_inv, weight_scale_cols,
                               act_scale_inv, blockIdx.x, word_major, prefetch_distance, a_byte);
}

__global__ __launch_bounds__(32, 1)
void k_tq_qmma_sf_gemv_tiled_split(float *partials, const uint32_t *a, const uint32_t *b,
                                   int M, int Kt, const float *weight_scale_inv,
                                   int weight_scale_cols, const float *act_scale_inv,
                                   int k_splits, int word_major, int a_byte) {
    int tile_m = blockIdx.x;
    int split = blockIdx.y;
    int kt_per_split = Kt / k_splits;
    int tk_begin = split * kt_per_split;
    int tk_end = tk_begin + kt_per_split;
    tq_qmma_sf_gemv_tiled_tile_range(partials + (size_t)split * M, a, b, Kt,
                                     weight_scale_inv, weight_scale_cols,
                                     act_scale_inv, tile_m, tk_begin, tk_end, word_major, 0,
                                     a_byte);
}

__global__ void k_tq_reduce_qmma_sf_splits(float *out, const float *partials, int M, int k_splits) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    float sum = 0.0f;
    for (int s = 0; s < k_splits; s++) {
        sum += partials[(size_t)s * M + row];
    }
    out[row] = sum;
}

__global__ __launch_bounds__(128, 1)
void k_tq_qmma_sf_gemv_tiled_multi(float *out0, const uint32_t *a0, int mt0,
                                   const float *scale0, int scale_cols0,
                                   float *out1, const uint32_t *a1, int mt1,
                                   const float *scale1, int scale_cols1,
                                   float *out2, const uint32_t *a2, int mt2,
                                   const float *scale2, int scale_cols2,
                                   float *out3, const uint32_t *a3, int mt3,
                                   const float *scale3, int scale_cols3,
                                   const uint32_t *b, int Kt,
                                   const float *act_scale_inv,
                                   int warps_per_block, int word_major, int prefetch_distance,
                                   int a_byte) {
    int warp = threadIdx.x >> 5;
    if (warp >= warps_per_block) return;
    int tile = blockIdx.x * warps_per_block + warp;
    if (tile < mt0) {
        tq_qmma_sf_gemv_tiled_tile(out0, a0, b, Kt, scale0, scale_cols0, act_scale_inv, tile, word_major, prefetch_distance, a_byte);
        return;
    }
    tile -= mt0;
    if (tile < mt1) {
        tq_qmma_sf_gemv_tiled_tile(out1, a1, b, Kt, scale1, scale_cols1, act_scale_inv, tile, word_major, prefetch_distance, a_byte);
        return;
    }
    tile -= mt1;
    if (tile < mt2) {
        tq_qmma_sf_gemv_tiled_tile(out2, a2, b, Kt, scale2, scale_cols2, act_scale_inv, tile, word_major, prefetch_distance, a_byte);
        return;
    }
    tile -= mt2;
    if (tile < mt3) {
        tq_qmma_sf_gemv_tiled_tile(out3, a3, b, Kt, scale3, scale_cols3, act_scale_inv, tile, word_major, prefetch_distance, a_byte);
    }
}

__global__ __launch_bounds__(128, 1)
void k_tq_qmma_sf_gemv_tiled_multi_split(float *partials,
                                         const uint32_t *a0, int mt0, const float *scale0, int scale_cols0,
                                         const uint32_t *a1, int mt1, const float *scale1, int scale_cols1,
                                         const uint32_t *a2, int mt2, const float *scale2, int scale_cols2,
                                         const uint32_t *a3, int mt3, const float *scale3, int scale_cols3,
                                         const uint32_t *b, int Kt, const float *act_scale_inv,
                                         int warps_per_block, int k_splits, int total_m,
                                         int word_major, int a_byte) {
    int warp = threadIdx.x >> 5;
    if (warp >= warps_per_block) return;
    int tile = blockIdx.x * warps_per_block + warp;
    int split = blockIdx.y;
    int kt_per_split = Kt / k_splits;
    int tk_begin = split * kt_per_split;
    int tk_end = tk_begin + kt_per_split;
    float *split_partials = partials + (size_t)split * total_m;

    if (tile < mt0) {
        tq_qmma_sf_gemv_tiled_tile_range(split_partials, a0, b, Kt, scale0, scale_cols0,
                                         act_scale_inv, tile, tk_begin, tk_end, word_major, 0, a_byte);
        return;
    }
    tile -= mt0;
    int row_offset = mt0 * 16;
    if (tile < mt1) {
        tq_qmma_sf_gemv_tiled_tile_range(split_partials + row_offset, a1, b, Kt, scale1, scale_cols1,
                                         act_scale_inv, tile, tk_begin, tk_end, word_major, 0, a_byte);
        return;
    }
    tile -= mt1;
    row_offset += mt1 * 16;
    if (tile < mt2) {
        tq_qmma_sf_gemv_tiled_tile_range(split_partials + row_offset, a2, b, Kt, scale2, scale_cols2,
                                         act_scale_inv, tile, tk_begin, tk_end, word_major, 0, a_byte);
        return;
    }
    tile -= mt2;
    row_offset += mt2 * 16;
    if (tile < mt3) {
        tq_qmma_sf_gemv_tiled_tile_range(split_partials + row_offset, a3, b, Kt, scale3, scale_cols3,
                                         act_scale_inv, tile, tk_begin, tk_end, word_major, 0, a_byte);
    }
}

__global__ void k_tq_reduce_qmma_sf_multi_splits(float *out0, int m0, float *out1, int m1,
                                                float *out2, int m2, float *out3, int m3,
                                                const float *partials, int total_m, int k_splits) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= total_m) return;
    float sum = 0.0f;
    for (int s = 0; s < k_splits; s++) {
        sum += partials[(size_t)s * total_m + row];
    }
    int r = row;
    if (r < m0) {
        out0[r] = sum;
        return;
    }
    r -= m0;
    if (r < m1) {
        out1[r] = sum;
        return;
    }
    r -= m1;
    if (r < m2) {
        out2[r] = sum;
        return;
    }
    r -= m2;
    if (r < m3) out3[r] = sum;
}

__global__ void k_tq_linear_conv_update(float *out, float *state, const float *x,
                                        const uint16_t *conv_weight,
                                        int channels, int kernel_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= channels) return;
    float sum = 0.0f;
    float *s = state + i * kernel_size;
    for (int j = 0; j < kernel_size - 1; j++) {
        s[j] = s[j + 1];
        sum += s[j] * tq_bf16_to_float(conv_weight[i * kernel_size + j]);
    }
    s[kernel_size - 1] = x[i];
    sum += x[i] * tq_bf16_to_float(conv_weight[i * kernel_size + (kernel_size - 1)]);
    out[i] = sum / (1.0f + expf(-sum));
}

__global__ void k_tq_linear_decode_core_gated(float *out, float *recurrent_state,
                                              const float *qkv_conv, const float *z,
                                              const float *b_proj, const float *a_proj,
                                              const float *A_log, const uint16_t *dt_bias,
                                              const float *norm_weight,
                                              int value_heads, int key_heads, int dim, float eps) {
    __shared__ float partial[4];
    __shared__ float q_sum_shared;
    __shared__ float k_sum_shared;
    __shared__ float core_sum_shared;
    __shared__ float qn[128];
    __shared__ float kn[128];
    int head = blockIdx.x;
    int tid = threadIdx.x;
    if (head >= value_heads || tid >= dim) return;
    int group = value_heads / key_heads;
    int key_head = head / group;
    int key_dim = key_heads * dim;
    const float *q = qkv_conv + key_head * dim;
    const float *k = qkv_conv + key_dim + key_head * dim;
    const float *v = qkv_conv + 2 * key_dim + head * dim;
    const float *z_head = z + head * dim;
    float qv = q[tid];
    float kv = k[tid];
    float qsum = qv * qv;
    float ksum = kv * kv;
    for (int offset = 16; offset > 0; offset >>= 1) {
        qsum += __shfl_down_sync(0xffffffff, qsum, offset);
        ksum += __shfl_down_sync(0xffffffff, ksum, offset);
    }
    if ((tid & 31) == 0) partial[tid >> 5] = qsum;
    __syncthreads();
    if (tid < 4) {
        float vtmp = partial[tid];
        for (int offset = 2; offset > 0; offset >>= 1) vtmp += __shfl_down_sync(0x0f, vtmp, offset);
        if (tid == 0) q_sum_shared = vtmp;
    }
    __syncthreads();
    if ((tid & 31) == 0) partial[tid >> 5] = ksum;
    __syncthreads();
    if (tid < 4) {
        float vtmp = partial[tid];
        for (int offset = 2; offset > 0; offset >>= 1) vtmp += __shfl_down_sync(0x0f, vtmp, offset);
        if (tid == 0) k_sum_shared = vtmp;
    }
    __syncthreads();
    qn[tid] = qv * rsqrtf(q_sum_shared + 1.0e-6f) * rsqrtf((float)dim);
    kn[tid] = kv * rsqrtf(k_sum_shared + 1.0e-6f);
    __syncthreads();

    float beta = 1.0f / (1.0f + expf(-b_proj[head]));
    float softplus_arg = a_proj[head] + tq_bf16_to_float(dt_bias[head]);
    float softplus = log1pf(expf(-fabsf(softplus_arg))) + fmaxf(softplus_arg, 0.0f);
    float g = -expf(A_log[head]) * softplus;
    float decay = expf(g);
    float *state = recurrent_state + (size_t)head * dim * dim;

    float kv_mem = 0.0f;
    for (int kk = 0; kk < dim; kk++) {
        kv_mem += state[kk * dim + tid] * decay * kn[kk];
    }
    float delta = (v[tid] - kv_mem) * beta;
    float core = 0.0f;
    for (int kk = 0; kk < dim; kk++) {
        float newv = state[kk * dim + tid] * decay + kn[kk] * delta;
        state[kk * dim + tid] = newv;
        core += newv * qn[kk];
    }

    float core_sq = core * core;
    for (int offset = 16; offset > 0; offset >>= 1) {
        core_sq += __shfl_down_sync(0xffffffff, core_sq, offset);
    }
    if ((tid & 31) == 0) partial[tid >> 5] = core_sq;
    __syncthreads();
    if (tid < 4) {
        float vtmp = partial[tid];
        for (int offset = 2; offset > 0; offset >>= 1) vtmp += __shfl_down_sync(0x0f, vtmp, offset);
        if (tid == 0) core_sum_shared = vtmp;
    }
    __syncthreads();
    float normed = core * rsqrtf(core_sum_shared / (float)dim + eps) * norm_weight[tid];
    float gate = z_head[tid];
    out[head * dim + tid] = normed * (gate / (1.0f + expf(-gate)));
}

// Shared-memory tree reduction over the first n (power-of-two) entries of sh.
// All threads of the block must call this uniformly. Result broadcast via sh[0].
static __device__ __forceinline__ float tq_block_reduce_sum_sh(float *sh, int n, int t) {
    for (int stride = n >> 1; stride > 0; stride >>= 1) {
        if (t < stride) sh[t] += sh[t + stride];
        __syncthreads();
    }
    return sh[0];
}

// Occupancy-filling variant of the linear-attention decode core.
//
// The original kernel launches one block (dim threads) per value head, i.e. only
// `value_heads` blocks total (48 for Qwen3.6-27B). With one warp's worth of work
// per head and ~6k threads total, the GPU is starved (<=1 warp/SM, ~2% DRAM),
// so the kernel runs ~12x slower than its tiny memory footprint warrants.
//
// Here each output element (head, tid) is handled by G threads that split the
// contraction (kk) loop, raising the thread count G-fold (G=8 => 1024 threads
// per block => full occupancy on the active SMs) while caching the per-thread
// state slice in registers so the recurrent state is read once instead of twice.
// The kk reduction is a warp shuffle; the cross-tid reductions (q/k L2 norm and
// the gated RMSNorm) stay in shared memory. Math is equivalent to the original
// up to floating-point summation order.
template<int G>
__global__ void k_tq_linear_decode_core_gated_par(float *out, float *recurrent_state,
                                                  const float *qkv_conv, const float *z,
                                                  const float *b_proj, const float *a_proj,
                                                  const float *A_log, const uint16_t *dt_bias,
                                                  const float *norm_weight,
                                                  int value_heads, int key_heads, int dim, float eps) {
    extern __shared__ float smem[];
    float *qn = smem;            // [dim]
    float *kn = smem + dim;      // [dim]
    float *red = smem + 2 * dim; // [dim] scratch for cross-tid reductions
    int head = blockIdx.x;
    if (head >= value_heads) return;
    int t = threadIdx.x;
    int tid = t / G;
    int klane = t % G;

    int group = value_heads / key_heads;
    int key_head = head / group;
    int key_dim = key_heads * dim;
    const float *q = qkv_conv + key_head * dim;
    const float *k = qkv_conv + key_dim + key_head * dim;
    const float *v = qkv_conv + 2 * key_dim + head * dim;
    const float *z_head = z + head * dim;

    // q/k L2 normalisation (reduction over the head's dim, one value per tid).
    float qv = 0.0f, kv = 0.0f;
    if (klane == 0) { qv = q[tid]; kv = k[tid]; }
    if (klane == 0) red[tid] = qv * qv;
    __syncthreads();
    float qsum = tq_block_reduce_sum_sh(red, dim, t);
    __syncthreads();
    if (klane == 0) red[tid] = kv * kv;
    __syncthreads();
    float ksum = tq_block_reduce_sum_sh(red, dim, t);
    __syncthreads();
    if (klane == 0) {
        qn[tid] = qv * rsqrtf(qsum + 1.0e-6f) * rsqrtf((float)dim);
        kn[tid] = kv * rsqrtf(ksum + 1.0e-6f);
    }
    __syncthreads();

    float beta = 1.0f / (1.0f + expf(-b_proj[head]));
    float softplus_arg = a_proj[head] + tq_bf16_to_float(dt_bias[head]);
    float softplus = log1pf(expf(-fabsf(softplus_arg))) + fmaxf(softplus_arg, 0.0f);
    float gg = -expf(A_log[head]) * softplus;
    float decay = expf(gg);
    float *state = recurrent_state + (size_t)head * dim * dim;

    // Phase 1: kv_mem[tid] = decay * sum_kk state[kk, tid] * kn[kk].
    // Cache this thread's state slice so phase 2 does not re-read it.
    float st[(128 + G - 1) / G];
    int nkk = 0;
    float psum = 0.0f;
    for (int kk = klane; kk < dim; kk += G) {
        float s = state[(size_t)kk * dim + tid];
        st[nkk++] = s;
        psum += s * kn[kk];
    }
    #pragma unroll
    for (int off = G >> 1; off > 0; off >>= 1)
        psum += __shfl_down_sync(0xffffffffu, psum, off, G);
    float kv_mem = __shfl_sync(0xffffffffu, psum, 0, G) * decay;
    float delta = (v[tid] - kv_mem) * beta;

    // Phase 2: update recurrent state and accumulate core[tid] = sum_kk newv*qn[kk].
    float pcore = 0.0f;
    nkk = 0;
    for (int kk = klane; kk < dim; kk += G) {
        float newv = st[nkk] * decay + kn[kk] * delta;
        state[(size_t)kk * dim + tid] = newv;
        pcore += newv * qn[kk];
        nkk++;
    }
    #pragma unroll
    for (int off = G >> 1; off > 0; off >>= 1)
        pcore += __shfl_down_sync(0xffffffffu, pcore, off, G);
    float core = __shfl_sync(0xffffffffu, pcore, 0, G);

    // Gated RMSNorm over the head's dim.
    if (klane == 0) red[tid] = core * core;
    __syncthreads();
    float core_sum = tq_block_reduce_sum_sh(red, dim, t);
    __syncthreads();
    if (klane == 0) {
        float normed = core * rsqrtf(core_sum / (float)dim + eps) * norm_weight[tid];
        float gate = z_head[tid];
        out[head * dim + tid] = normed * (gate / (1.0f + expf(-gate)));
    }
}

// Head-split variant. core_par above still launches only `value_heads` blocks,
// so at most that many SMs (48 on Qwen3.6-27B) issue memory traffic for the
// recurrent state, capping effective DRAM bandwidth at ~value_heads/total_SMs.
// Here each head is additionally split into HS blocks over disjoint tid stripes
// (the per-element work is independent), so value_heads*HS blocks engage all SMs.
// The cross-tid gated RMSNorm cannot be done per-stripe, so this kernel writes
// the raw core values and k_tq_linear_decode_gated_norm finalises them.
template<int G, int HS>
__global__ void k_tq_linear_decode_core_hs(float *core_out, float *recurrent_state,
                                           const float *qkv_conv,
                                           const float *b_proj, const float *a_proj,
                                           const float *A_log, const uint16_t *dt_bias,
                                           int value_heads, int key_heads, int dim) {
    extern __shared__ float smem[];
    float *qn = smem;            // [dim]
    float *kn = smem + dim;      // [dim]
    float *red = smem + 2 * dim; // [dim]
    int blk = blockIdx.x;
    int head = blk / HS;
    int part = blk % HS;
    if (head >= value_heads) return;
    int t = threadIdx.x;
    int bs = blockDim.x;
    int tids_per = dim / HS;
    int tid_local = t / G;
    int klane = t % G;
    int tid = part * tids_per + tid_local;

    int group = value_heads / key_heads;
    int key_head = head / group;
    int key_dim = key_heads * dim;
    const float *q = qkv_conv + key_head * dim;
    const float *k = qkv_conv + key_dim + key_head * dim;
    const float *v = qkv_conv + 2 * key_dim + head * dim;

    // Full (redundant) q/k L2 normalisation: every block needs all kn/qn[kk].
    for (int i = t; i < dim; i += bs) red[i] = q[i] * q[i];
    __syncthreads();
    float qsum = tq_block_reduce_sum_sh(red, dim, t);
    __syncthreads();
    for (int i = t; i < dim; i += bs) red[i] = k[i] * k[i];
    __syncthreads();
    float ksum = tq_block_reduce_sum_sh(red, dim, t);
    __syncthreads();
    for (int i = t; i < dim; i += bs) {
        qn[i] = q[i] * rsqrtf(qsum + 1.0e-6f) * rsqrtf((float)dim);
        kn[i] = k[i] * rsqrtf(ksum + 1.0e-6f);
    }
    __syncthreads();

    float beta = 1.0f / (1.0f + expf(-b_proj[head]));
    float softplus_arg = a_proj[head] + tq_bf16_to_float(dt_bias[head]);
    float softplus = log1pf(expf(-fabsf(softplus_arg))) + fmaxf(softplus_arg, 0.0f);
    float gg = -expf(A_log[head]) * softplus;
    float decay = expf(gg);
    float *state = recurrent_state + (size_t)head * dim * dim;

    float st[(128 + G - 1) / G];
    int nkk = 0;
    float psum = 0.0f;
    for (int kk = klane; kk < dim; kk += G) {
        float s = state[(size_t)kk * dim + tid];
        st[nkk++] = s;
        psum += s * kn[kk];
    }
    #pragma unroll
    for (int off = G >> 1; off > 0; off >>= 1)
        psum += __shfl_down_sync(0xffffffffu, psum, off, G);
    float kv_mem = __shfl_sync(0xffffffffu, psum, 0, G) * decay;
    float delta = (v[tid] - kv_mem) * beta;

    float pcore = 0.0f;
    nkk = 0;
    for (int kk = klane; kk < dim; kk += G) {
        float newv = st[nkk] * decay + kn[kk] * delta;
        state[(size_t)kk * dim + tid] = newv;
        pcore += newv * qn[kk];
        nkk++;
    }
    #pragma unroll
    for (int off = G >> 1; off > 0; off >>= 1)
        pcore += __shfl_down_sync(0xffffffffu, pcore, off, G);
    float core = __shfl_sync(0xffffffffu, pcore, 0, G);

    if (klane == 0) core_out[head * dim + tid] = core;
}

