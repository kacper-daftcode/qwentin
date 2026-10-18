// wide FP6 / E2M1 projection GEMMs and their launchers.
// Part of the forward_qwen.cu unity build: included in order, not a standalone header.

// ===================== D.1: wide FP6 projection GEMM =====================
// Generalizes k_tq_qmma_sf_gemv_tiled_nk from 2 column-groups (N<=16) to
// NG = ceil(N/8): the e2m3 weight A-fragment is loaded ONCE per k-tile and
// reused across all NG m16n8 MMAs, so a weight tile is read once and amortized
// over NG*8 token-columns (the compute-bound prefill lever). e2m3 weight x e4m3
// activation, QMMA.SF block-scaled (shared pow2 activation scale per 128-K block,
// same approximation as the verify nk path -- validated for prefill via needle,
// not here). Standalone: only qwn_wide_gemm_bench/check use it; no shipping path
// touches it. Accumulators c[NG][4] live in registers (NG=32 -> 128 regs/thread).
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

// ===== milestone 3.4: WIDE 4-bit (E2M1) GEMM (throughput tier) =====
// Identical to k_tq_fp6_wide_gemm except the WEIGHT a-fragment load: E2M1 packs 64 words/tile
// (vs E2M3's 96) and unpacks 2 words -> 4 a-fragments via tq_unpack_e2m1_16 (which embeds each
// 4-bit code as its exact 6-bit E2M3 twin, so the SAME mxf8f6f4 MMA + scales apply). Weights are
// repacked E2M3->E2M1 on load (TQ_W_E2M1) freeing 1/3 of weight VRAM; B-frag/activations/scales
// and the output layout are unchanged. Opt-in throughput tier -- FP6 single-stream untouched.
template<int NG>
__global__ void k_tq_fp4_wide_gemm(float *out, const uint32_t *a, const uint32_t *b,
                                   int M, int Kt, const float *wscale_inv,
                                   int wscale_cols, const float *ascale_inv, int nvar) {
    int tile_m = blockIdx.x;
    int lane = threadIdx.x & 31;
    float c[NG][4];
    #pragma unroll
    for (int g = 0; g < NG; g++) { c[g][0] = c[g][1] = c[g][2] = c[g][3] = 0.0f; }
    int scale_row = tile_m >> 3;
    const uint32_t *a_base = a + (size_t)tile_m * Kt * 64;   // e2m1: 64 words / k-tile
    const size_t bgstride = (size_t)Kt * 64;
    for (int tk = 0; tk < Kt; tk++) {
        int a_idx = tk * 64 + lane * 2;
        uint32_t Q0 = __ldg(&a_base[a_idx + 0]);
        uint32_t Q1 = __ldg(&a_base[a_idx + 1]);
        uint32_t a0, a1, a2, a3;
        tq_unpack_e2m1_16(Q0, Q1, a0, a1, a2, a3);
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

template<int NG>
__global__ void k_tq_fp4_wide_gemm_split(float *partials, const uint32_t *a, const uint32_t *b,
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
    const uint32_t *a_base = a + (size_t)tile_m * Kt * 64;
    const size_t bgstride = (size_t)Kt * 64;
    for (int tk = tk_begin; tk < tk_end; tk++) {
        int a_idx = tk * 64 + lane * 2;
        uint32_t Q0 = __ldg(&a_base[a_idx + 0]);
        uint32_t Q1 = __ldg(&a_base[a_idx + 1]);
        uint32_t a0, a1, a2, a3;
        tq_unpack_e2m1_16(Q0, Q1, a0, a1, a2, a3);
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

#define TQ_FP4_SINGLE(NG) k_tq_fp4_wide_gemm<NG><<<Mt, 32, 0, st>>>(out, a, bb, M, Kt, ws, sc, act_scale, N)
#define TQ_FP4_SPLIT(NG)  k_tq_fp4_wide_gemm_split<NG><<<grid, 32, 0, st>>>(partials, a, bb, M, Kt, ws, sc, act_scale, N, k_splits)
#define TQ_FP4_SWITCH(LAUNCH) \
    switch (G) { \
        case 1:  LAUNCH(1);  break; case 2:  LAUNCH(2);  break; \
        case 4:  LAUNCH(4);  break; case 8:  LAUNCH(8);  break; \
        case 16: LAUNCH(16); break; case 32: LAUNCH(32); break; \
        default: return -1; }

static int launch_fp4_wide_gemm_tile(float *out, const tq_qmma_weight_t *w, const uint8_t *b,
                                     const float *act_scale, int N, float *partials, int k_splits,
                                     cudaStream_t st) {
    const uint32_t *a = (const uint32_t *)w->d_A;
    const uint32_t *bb = (const uint32_t *)b;
    int M = w->M, Kt = w->Kt, Mt = w->Mt, sc = w->scale_cols;
    const float *ws = w->d_block_scale_inv;
    int G = wide_ng(N);
    if (k_splits > 1 && (Kt % k_splits) == 0 && partials) {
        dim3 grid(Mt, k_splits);
        TQ_FP4_SWITCH(TQ_FP4_SPLIT);
        k_tq_reduce_qmma_sf_splits<<<((size_t)N * M + 255) / 256, 256, 0, st>>>(out, partials, N * M, k_splits);
        return 0;
    }
    TQ_FP4_SWITCH(TQ_FP4_SINGLE);
    return 0;
}
#undef TQ_FP4_SINGLE
#undef TQ_FP4_SPLIT
#undef TQ_FP4_SWITCH

static int launch_fp4_wide_gemm(float *out, const tq_qmma_weight_t *w, const uint8_t *b,
                                const float *act_scale, int N, float *partials, int k_splits,
                                cudaStream_t st) {
    const int TILE = 128;
    int M = w->M, Kt = w->Kt;
    for (int off = 0; off < N; off += TILE) {
        int nt = (N - off < TILE) ? (N - off) : TILE;
        int rc = launch_fp4_wide_gemm_tile(out + (size_t)off * M, w,
                                           b + (size_t)(off / 8) * Kt * 256,
                                           act_scale, nt, partials, k_splits, st);
        if (rc) return rc;
    }
    return 0;
}

// K-split variant of the N=k batched GEMV (Step 5): blockIdx.y = split processes a
// K-tile sub-range so large-K weights (mlp_down/o_proj: small Mt, long Kt) fill the
// GPU instead of one warp serially streaming all of K. Writes per-split partials
// laid out [split][nvar*M]; k_tq_reduce_qmma_sf_splits(out, partials, nvar*M, splits)
// sums them. Mirrors the per-token k_tq_qmma_sf_gemv_tiled_split path.
__global__ void k_tq_qmma_sf_gemv_tiled_nk_split(
    float *partials, const uint32_t *a, const uint32_t *b, int M, int Kt,
    const float *weight_scale_inv, int weight_scale_cols,
    const float *act_scale_inv, int nvar, int k_splits, int a_fmt) {
    int tile_m = blockIdx.x;
    int split = blockIdx.y;
    int lane = threadIdx.x & 31;
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
    float c4 = 0.0f, c5 = 0.0f, c6 = 0.0f, c7 = 0.0f;
    const uint32_t *b_base = b;
    int g2 = nvar > 8;
    const uint32_t *b_base2 = b_base + (size_t)Kt * 64;
    int scale_row = tile_m >> 3;
    int kt_per = Kt / k_splits;
    int tk_begin = split * kt_per;
    int tk_end = tk_begin + kt_per;

    if (a_fmt == 2) {
        // HW-direct E2M3 byte containers (128 words/tile): zero-unpack loop.
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
            if (g2) {
                uint32_t bb0 = __ldg(&b_base2[b_idx + 0]);
                uint32_t bb1 = __ldg(&b_base2[b_idx + 1]);
                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X"
                    ".f32.e2m3.e4m3.f32.ue8m0 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, {0,1}, %15, {0,1};"
                    : "+f"(c4), "+f"(c5), "+f"(c6), "+f"(c7)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(bb0), "r"(bb1),
                      "f"(c4), "f"(c5), "f"(c6), "f"(c7), "r"(sfa), "r"(sfb));
            }
        }
    } else if (a_fmt == 1) {
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
            if (g2) {
                uint32_t bb0 = __ldg(&b_base2[b_idx + 0]);
                uint32_t bb1 = __ldg(&b_base2[b_idx + 1]);
                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X"
                    ".f32.e2m3.e4m3.f32.ue8m0 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, {0,1}, %15, {0,1};"
                    : "+f"(c4), "+f"(c5), "+f"(c6), "+f"(c7)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(bb0), "r"(bb1),
                      "f"(c4), "f"(c5), "f"(c6), "f"(c7), "r"(sfa), "r"(sfb));
            }
        }
    } else if (gc_weight_e2m3) {
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
            uint32_t B2[4][2];
            if (g2) {
                #pragma unroll
                for (int u = 0; u < 4; u++) {
                    B2[u][0] = __ldg(&b_base2[b_idx + 64 * u + 0]);
                    B2[u][1] = __ldg(&b_base2[b_idx + 64 * u + 1]);
                }
            }
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
                if (g2) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X"
                        ".f32.e2m3.e4m3.f32.ue8m0 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, {0,1}, %15, {0,1};"
                        : "+f"(c4), "+f"(c5), "+f"(c6), "+f"(c7)
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(B2[half][0]), "r"(B2[half][1]),
                          "f"(c4), "f"(c5), "f"(c6), "f"(c7), "r"(sfa), "r"(sfb));
                }
            }
        }
        for (; tk < tk_end; tk++) {
            int a_idx = tk * 96 + lane * 3;
            uint32_t P[3] = {
                __ldg(&a_base[a_idx + 0]), __ldg(&a_base[a_idx + 1]), __ldg(&a_base[a_idx + 2]),
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
            if (g2) {
                uint32_t bb0 = __ldg(&b_base2[b_idx + 0]);
                uint32_t bb1 = __ldg(&b_base2[b_idx + 1]);
                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X"
                    ".f32.e2m3.e4m3.f32.ue8m0 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, {0,1}, %15, {0,1};"
                    : "+f"(c4), "+f"(c5), "+f"(c6), "+f"(c7)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(bb0), "r"(bb1),
                      "f"(c4), "f"(c5), "f"(c6), "f"(c7), "r"(sfa), "r"(sfb));
            }
        }
    } else {
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
                ".f32.e4m3.e4m3.f32.ue8m0 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, {0,1}, %15, {0,1};"
                : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                  "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(sfa), "r"(sfb));
        }
    }

    int cc = lane & 3;
    int g = lane >> 2;
    int cola = 2 * cc;
    int colb = 2 * cc + 1;
    size_t pbase = (size_t)split * nvar * M;
    if (cola < nvar) {
        partials[pbase + (size_t)cola * M + tile_m * 16 + 2 * g] = c0;
        partials[pbase + (size_t)cola * M + tile_m * 16 + 2 * g + 1] = c2;
    }
    if (colb < nvar) {
        partials[pbase + (size_t)colb * M + tile_m * 16 + 2 * g] = c1;
        partials[pbase + (size_t)colb * M + tile_m * 16 + 2 * g + 1] = c3;
    }
    if (g2) {
        int cola2 = 8 + 2 * cc;
        int colb2 = 9 + 2 * cc;
        if (cola2 < nvar) {
            partials[pbase + (size_t)cola2 * M + tile_m * 16 + 2 * g] = c4;
            partials[pbase + (size_t)cola2 * M + tile_m * 16 + 2 * g + 1] = c6;
        }
        if (colb2 < nvar) {
            partials[pbase + (size_t)colb2 * M + tile_m * 16 + 2 * g] = c5;
            partials[pbase + (size_t)colb2 * M + tile_m * 16 + 2 * g + 1] = c7;
        }
    }
}

// Pack an activation vector x[K] into the sparse-GEMV B operand (dense n8xk64,
// n=0 column = contiguous bytes 0..63 per 512-byte k64-tile). The caller must
// cudaMemset b to 0 first (the non-n0 bytes stay zero).
__global__ void k_tq_pack_b_k64(uint8_t *b, const float *x, int K) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < K) {
        int tile = k / 64, kk = k % 64;
        b[tile * 512 + kk] = tq_float_to_e4m3(x[k]);
    }
}

// Residual e4m3 split for full-precision activations: x ~= hi + lo where
// hi=e4m3(x), lo=e4m3(x - decode(hi)). Two e4m3 passes ~ BF16 (~7 mantissa bits).
// Caller memsets both buffers first.
__global__ void k_tq_pack_b_k64_residual(uint8_t *b_hi, uint8_t *b_lo,
                                         const float *x, int K) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < K) {
        int tile = k / 64, kk = k % 64;
        uint8_t hi = tq_float_to_e4m3(x[k]);
        float hi_v = tq_e4m3_to_float(hi);
        uint8_t lo = tq_float_to_e4m3(x[k] - hi_v);
        b_hi[tile * 512 + kk] = hi;
        b_lo[tile * 512 + kk] = lo;
    }
}

// Residual k64 activation packer WITH per-128-K-block pow2 scale (matches the
// dense activation packer's scale), for the forward sparse draft. hi=e4m3(x/s),
// lo=e4m3(x/s - decode(hi)); both passes share the block scale s (ue8m0 sfb).
// Caller memsets b_hi/b_lo first. Grid = (K+127)/128 blocks, 128 threads.
__global__ void k_tq_pack_b_k64_residual_scaled(uint8_t *b_hi, uint8_t *b_lo,
                                                float *scale_inv, const float *x, int K) {
    __shared__ float smem[128];
    __shared__ float block_scale_inv;
    int block = blockIdx.x;
    int tid = threadIdx.x;
    int base = block * 128;
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
        int tile = idx >> 6;
        int kk = idx & 63;
        float xs = x[idx] / block_scale_inv;
        uint8_t hi = tq_float_to_e4m3(xs);
        b_hi[tile * 512 + kk] = hi;
        b_lo[tile * 512 + kk] = tq_float_to_e4m3(xs - tq_e4m3_to_float(hi));
    }
}

// Optimistic sparse-FP4 draft GEMV (engine copy of tl_probe_qmma_sp_gemv_e2m3):
// tiled sparse 2:4, 6-bit E2M3 weights x E4M3 acts, ue8m0 block scales. The k64
// tile tk lives in the 128-K block (tk>>1); weight scale per (8-row group, 128-K).
__global__ void k_tq_sparse_gemv_e2m3(
    float *out, const uint32_t *a, const uint32_t *b, const uint32_t *meta,
    const float *wscale_inv, const float *ascale_inv, int M, int Kt64, int wscale_cols) {
    int tile_m = blockIdx.x;
    int lane = threadIdx.x & 31;
    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    int scale_row = tile_m >> 3;
    const uint32_t *a_base = a + (size_t)tile_m * Kt64 * 96;
    for (int tk = 0; tk < Kt64; tk++) {
        int a_idx = tk * 96 + lane * 3;
        uint32_t P[3] = {__ldg(&a_base[a_idx + 0]), __ldg(&a_base[a_idx + 1]),
                         __ldg(&a_base[a_idx + 2])};
        uint32_t a0, a1, a2, a3;
            tq_unpack_e2m3_16(P[0], P[1], P[2], a0, a1, a2, a3);
        int b_idx = tk * 128 + lane * 4;
        uint32_t b0 = __ldg(&b[b_idx + 0]), b1 = __ldg(&b[b_idx + 1]);
        uint32_t b2 = __ldg(&b[b_idx + 2]), b3 = __ldg(&b[b_idx + 3]);
        uint32_t re = __ldg(&meta[((size_t)tile_m * Kt64 + tk) * 32 + lane]);
        uint32_t sfa = tq_ue8m0_pack4(wscale_inv[scale_row * wscale_cols + (tk >> 1)]);
        uint32_t sfb = tq_ue8m0_pack4(ascale_inv[tk >> 1]);
        asm volatile(
            "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col"
            ".kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e2m3.e4m3.f32.ue8m0 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, "
            "%16, 0x0, %17, {0,1}, %18, {0,1};"
            : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(b2), "r"(b3),
              "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(re), "r"(sfa), "r"(sfb));
    }
    if ((lane & 3) == 0) {
        int g = lane >> 2;
        out[tile_m * 16 + g * 2] = c0;
        out[tile_m * 16 + g * 2 + 1] = c2;
    }
}

static __device__ __forceinline__ void tq_qmma_sf_gemv_tiled_tile_range_atomic(
    float *out, const uint32_t *a, const uint32_t *b, int Kt,
    const float *weight_scale_inv, int weight_scale_cols,
    const float *act_scale_inv, int tile_m, int tk_begin, int tk_end,
    int word_major, int prefetch_distance) {
    int lane = threadIdx.x & 31;
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
    const uint32_t *a_base = a + (size_t)tile_m * Kt * 128;
    const uint32_t *b_base = b;
    int scale_row = tile_m >> 3;

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

    if ((lane & 3) == 0) {
        int g = lane >> 2;
        atomicAdd(&out[tile_m * 16 + g * 2], c0);
        atomicAdd(&out[tile_m * 16 + g * 2 + 1], c2);
    }
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

static __device__ __forceinline__ void tq_qmma_sf_gemv_tiled_tile_pair(
    float *out, const uint32_t *a, const uint32_t *b, int Kt,
    const float *weight_scale_inv, int weight_scale_cols,
    const float *act_scale_inv, int tile_m0, int word_major, int prefetch_distance) {
    int lane = threadIdx.x & 31;
    int tile_m1 = tile_m0 + 1;
    float c00 = 0.0f, c01 = 0.0f, c02 = 0.0f, c03 = 0.0f;
    float c10 = 0.0f, c11 = 0.0f, c12 = 0.0f, c13 = 0.0f;
    const uint32_t *a_base0 = a + (size_t)tile_m0 * Kt * 128;
    const uint32_t *a_base1 = a + (size_t)tile_m1 * Kt * 128;
    const uint32_t *b_base = b;
    int scale_row0 = tile_m0 >> 3;
    int scale_row1 = tile_m1 >> 3;

    for (int tk = 0; tk < Kt; tk++) {
        tq_prefetch_qmma_a_l2(a_base0, Kt, tk, lane, word_major, prefetch_distance);
        tq_prefetch_qmma_a_l2(a_base1, Kt, tk, lane, word_major, prefetch_distance);
        int a_idx = word_major ? (tk * 128 + lane) : (tk * 128 + lane * 4);
        int a_stride1 = word_major ? 32 : 1;
        int a_stride2 = word_major ? 64 : 2;
        int a_stride3 = word_major ? 96 : 3;
        uint32_t a00 = __ldg(&a_base0[a_idx + 0]);
        uint32_t a01 = __ldg(&a_base0[a_idx + a_stride1]);
        uint32_t a02 = __ldg(&a_base0[a_idx + a_stride2]);
        uint32_t a03 = __ldg(&a_base0[a_idx + a_stride3]);
        uint32_t a10 = __ldg(&a_base1[a_idx + 0]);
        uint32_t a11 = __ldg(&a_base1[a_idx + a_stride1]);
        uint32_t a12 = __ldg(&a_base1[a_idx + a_stride2]);
        uint32_t a13 = __ldg(&a_base1[a_idx + a_stride3]);
        int b_idx = tk * 64 + lane * 2;
        uint32_t b0 = __ldg(&b_base[b_idx + 0]);
        uint32_t b1 = __ldg(&b_base[b_idx + 1]);
        uint32_t sfa0 = tq_ue8m0_pack4(weight_scale_inv[scale_row0 * weight_scale_cols + (weight_scale_cols == Kt ? tk : (tk >> 2))]);
        uint32_t sfa1 = tq_ue8m0_pack4(weight_scale_inv[scale_row1 * weight_scale_cols + (weight_scale_cols == Kt ? tk : (tk >> 2))]);
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
            : "+f"(c00), "+f"(c01), "+f"(c02), "+f"(c03)
            : "r"(a00), "r"(a01), "r"(a02), "r"(a03), "r"(b0), "r"(b1),
              "f"(c00), "f"(c01), "f"(c02), "f"(c03), "r"(sfa0), "r"(sfb));
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
            : "+f"(c10), "+f"(c11), "+f"(c12), "+f"(c13)
            : "r"(a10), "r"(a11), "r"(a12), "r"(a13), "r"(b0), "r"(b1),
              "f"(c10), "f"(c11), "f"(c12), "f"(c13), "r"(sfa1), "r"(sfb));
    }

    if ((lane & 3) == 0) {
        int g = lane >> 2;
        out[tile_m0 * 16 + g * 2] = c00;
        out[tile_m0 * 16 + g * 2 + 1] = c02;
        out[tile_m1 * 16 + g * 2] = c10;
        out[tile_m1 * 16 + g * 2 + 1] = c12;
    }
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

__global__ void k_tq_reduce_qmma_sf_splits_add(float *out, const float *partials,
                                               const float *residual, int M, int k_splits) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    float sum = residual[row];
    for (int s = 0; s < k_splits; s++) {
        sum += partials[(size_t)s * M + row];
    }
    out[row] = sum;
}

// K-split reduce with the nk post-correction folded in: out[col*M+row] =
// (sum of split partials) / factors[col]. Bit-identical to reduce-then-postcorrect.
__global__ void k_tq_reduce_qmma_sf_splits_pc(float *out, const float *partials,
                                              const float *factors, int M, int nvar,
                                              int k_splits) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nvar * M;
    if (idx >= total) return;
    float sum = 0.0f;
    for (int s = 0; s < k_splits; s++) {
        sum += partials[(size_t)s * total + idx];
    }
    out[idx] = sum / factors[idx / M];
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
void k_tq_qmma_sf_gemv_tiled_multi_persistent(float *out0, const uint32_t *a0, int mt0,
                                              const float *scale0, int scale_cols0,
                                              float *out1, const uint32_t *a1, int mt1,
                                              const float *scale1, int scale_cols1,
                                              float *out2, const uint32_t *a2, int mt2,
                                              const float *scale2, int scale_cols2,
                                              float *out3, const uint32_t *a3, int mt3,
                                              const float *scale3, int scale_cols3,
                                              const uint32_t *b, int Kt,
                                              const float *act_scale_inv,
                                              int warps_per_block, int total_mt,
                                              int word_major, int a_byte,
                                              int *work_counter) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= warps_per_block) return;
    while (1) {
        int tile = 0;
        if (lane == 0) tile = atomicAdd(work_counter, 1);
        tile = __shfl_sync(0xffffffff, tile, 0);
        if (tile >= total_mt) return;
        if (tile < mt0) {
            tq_qmma_sf_gemv_tiled_tile(out0, a0, b, Kt, scale0, scale_cols0, act_scale_inv, tile, word_major, 0, a_byte);
            continue;
        }
        tile -= mt0;
        if (tile < mt1) {
            tq_qmma_sf_gemv_tiled_tile(out1, a1, b, Kt, scale1, scale_cols1, act_scale_inv, tile, word_major, 0, a_byte);
            continue;
        }
        tile -= mt1;
        if (tile < mt2) {
            tq_qmma_sf_gemv_tiled_tile(out2, a2, b, Kt, scale2, scale_cols2, act_scale_inv, tile, word_major, 0, a_byte);
            continue;
        }
        tile -= mt2;
        if (tile < mt3) {
            tq_qmma_sf_gemv_tiled_tile(out3, a3, b, Kt, scale3, scale_cols3, act_scale_inv, tile, word_major, 0, a_byte);
        }
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

__global__ __launch_bounds__(128, 1)
void k_tq_qmma_sf_gemv_tiled_multi_split_atomic(float *out0, const uint32_t *a0, int mt0,
                                                const float *scale0, int scale_cols0,
                                                float *out1, const uint32_t *a1, int mt1,
                                                const float *scale1, int scale_cols1,
                                                float *out2, const uint32_t *a2, int mt2,
                                                const float *scale2, int scale_cols2,
                                                float *out3, const uint32_t *a3, int mt3,
                                                const float *scale3, int scale_cols3,
                                                const uint32_t *b, int Kt,
                                                const float *act_scale_inv,
                                                int warps_per_block, int k_splits,
                                                int word_major) {
    int warp = threadIdx.x >> 5;
    if (warp >= warps_per_block) return;
    int tile = blockIdx.x * warps_per_block + warp;
    int split = blockIdx.y;
    int kt_per_split = Kt / k_splits;
    int tk_begin = split * kt_per_split;
    int tk_end = tk_begin + kt_per_split;

    if (tile < mt0) {
        tq_qmma_sf_gemv_tiled_tile_range_atomic(out0, a0, b, Kt, scale0, scale_cols0,
                                                act_scale_inv, tile, tk_begin, tk_end, word_major, 0);
        return;
    }
    tile -= mt0;
    if (tile < mt1) {
        tq_qmma_sf_gemv_tiled_tile_range_atomic(out1, a1, b, Kt, scale1, scale_cols1,
                                                act_scale_inv, tile, tk_begin, tk_end, word_major, 0);
        return;
    }
    tile -= mt1;
    if (tile < mt2) {
        tq_qmma_sf_gemv_tiled_tile_range_atomic(out2, a2, b, Kt, scale2, scale_cols2,
                                                act_scale_inv, tile, tk_begin, tk_end, word_major, 0);
        return;
    }
    tile -= mt2;
    if (tile < mt3) {
        tq_qmma_sf_gemv_tiled_tile_range_atomic(out3, a3, b, Kt, scale3, scale_cols3,
                                                act_scale_inv, tile, tk_begin, tk_end, word_major, 0);
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

__global__ __launch_bounds__(32, 1)
void k_tq_qmma_gemv_fp8(float *raw, const uint32_t *a, const uint32_t *b, int Kt) {
    int tile_m = blockIdx.x;
    int lane = threadIdx.x & 31;
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
    const uint32_t *a_base = a + (size_t)tile_m * Kt * 128;
    const uint32_t *b_base = b;
    for (int tk = 0; tk < Kt; tk++) {
        int a_idx = tk * 128 + lane * 4;
        int b_idx = tk * 64 + lane * 2;
        uint32_t a0 = __ldg(&a_base[a_idx + 0]);
        uint32_t a1 = __ldg(&a_base[a_idx + 1]);
        uint32_t a2 = __ldg(&a_base[a_idx + 2]);
        uint32_t a3 = __ldg(&a_base[a_idx + 3]);
        uint32_t b0 = __ldg(&b_base[b_idx + 0]);
        uint32_t b1 = __ldg(&b_base[b_idx + 1]);
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
            : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
              "f"(c0), "f"(c1), "f"(c2), "f"(c3));
    }
    raw[tile_m * 128 + lane * 4 + 0] = c0;
    raw[tile_m * 128 + lane * 4 + 1] = c1;
    raw[tile_m * 128 + lane * 4 + 2] = c2;
    raw[tile_m * 128 + lane * 4 + 3] = c3;
}

__global__ __launch_bounds__(32, 1)
void k_tq_qmma_gemv_k1024(float *raw, const uint32_t *a, const uint32_t *b) {
    int tile_m = blockIdx.x;
    int lane = threadIdx.x & 31;
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
    const uint32_t *a_base = a + (size_t)tile_m * 32 * 128;
    const uint32_t *b_base = b;
    #pragma unroll
    for (int tk = 0; tk < 32; tk++) {
        int a_idx = tk * 128 + lane * 4;
        int b_idx = tk * 64 + lane * 2;
        uint32_t a0 = __ldg(&a_base[a_idx + 0]);
        uint32_t a1 = __ldg(&a_base[a_idx + 1]);
        uint32_t a2 = __ldg(&a_base[a_idx + 2]);
        uint32_t a3 = __ldg(&a_base[a_idx + 3]);
        uint32_t b0 = __ldg(&b_base[b_idx + 0]);
        uint32_t b1 = __ldg(&b_base[b_idx + 1]);
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
            : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
              "f"(c0), "f"(c1), "f"(c2), "f"(c3));
    }
    raw[tile_m * 128 + lane * 4 + 0] = c0;
    raw[tile_m * 128 + lane * 4 + 1] = c1;
    raw[tile_m * 128 + lane * 4 + 2] = c2;
    raw[tile_m * 128 + lane * 4 + 3] = c3;
}

__global__ void k_tq_extract_descale(float *out, const float *raw, int M, float inv_w_scale,
                                     const float *d_scale) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    int tile = row >> 4;
    int r = row & 15;
    int g = r >> 1;
    int pair = r & 1;
    float v = raw[tile * 128 + (g * 4) * 4 + pair * 2];
    out[row] = v * inv_w_scale / d_scale[0];
}

__global__ void k_tq_linear_conv_first_token(float *out, const float *x,
                                             const uint16_t *conv_weight,
                                             int channels, int kernel_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= channels) return;
    float w = tq_bf16_to_float(conv_weight[i * kernel_size + (kernel_size - 1)]);
    float v = x[i] * w;
    out[i] = v / (1.0f + expf(-v));
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

// Batched conv update over a group of sibling nodes (Step-5 verify). grid=(blocks, gc);
// node = group_ids[blockIdx.y]. Each node has its own conv state slab (stride
// state_stride) and its own qkv vector (stride channels). Bit-identical per node to
// k_tq_linear_conv_update. conv_weight is shared (layer weight).
__global__ void k_tq_linear_conv_update_b(float *out, float *state, const float *x,
                                          const uint16_t *conv_weight, int channels, int kernel_size,
                                          const int *group_ids, int state_stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= channels) return;
    int node = group_ids[blockIdx.y];
    float *s = state + (size_t)node * state_stride + (size_t)i * kernel_size;
    const float *xn = x + (size_t)node * channels;
    float *outn = out + (size_t)node * channels;
    float sum = 0.0f;
    for (int j = 0; j < kernel_size - 1; j++) {
        s[j] = s[j + 1];
        sum += s[j] * tq_bf16_to_float(conv_weight[i * kernel_size + j]);
    }
    s[kernel_size - 1] = xn[i];
    sum += xn[i] * tq_bf16_to_float(conv_weight[i * kernel_size + (kernel_size - 1)]);
    outn[i] = sum / (1.0f + expf(-sum));
}

// Fused fork + conv update: reads from PARENT's conv state, writes to CHILD's.
// Eliminates the per-sibling cudaMemcpyAsync for conv state. For root nodes
// (parent < 0), reads from the committed live state. Bit-identical to
// k_tq_linear_conv_update_b preceded by a state fork copy.
__global__ void k_tq_linear_conv_update_b_fork(float *out, float *state, const float *x,
                                               const uint16_t *conv_weight, int channels, int kernel_size,
                                               const int *group_ids, const int *parent_ids,
                                               int state_stride, const float *committed_state) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= channels) return;
    int node = group_ids[blockIdx.y];
    int par = parent_ids[node];
    const float *src = (par >= 0) ? state + (size_t)par * state_stride + (size_t)i * kernel_size
                                  : committed_state + (size_t)i * kernel_size;
    float *dst = state + (size_t)node * state_stride + (size_t)i * kernel_size;
    const float *xn = x + (size_t)node * channels;
    float *outn = out + (size_t)node * channels;
    float sum = 0.0f;
    for (int j = 0; j < kernel_size - 1; j++) {
        dst[j] = src[j + 1];
        sum += dst[j] * tq_bf16_to_float(conv_weight[i * kernel_size + j]);
    }
    dst[kernel_size - 1] = xn[i];
    sum += xn[i] * tq_bf16_to_float(conv_weight[i * kernel_size + (kernel_size - 1)]);
    outn[i] = sum / (1.0f + expf(-sum));
}

// Chunk-parallel conv update for the WIDE prefill: position p of the chunk
// reads x[p-3..p] straight from the chunk buffer (positions before the chunk
// come from the incoming conv state), so all n positions run in ONE launch
// (grid (channels/256, n)) instead of n serial per-token launches (measured
// 9.5% of a 64k prefill = 3.1M launches of 1.3 us). blockIdx.y == n writes the
// NEW state (the last kernel_size-1 x columns + padding from the old state for
// short chunks); it touches only `state`, no out column, and no other block
// reads state, so the in-launch order is free. Per-position math is identical
// to k_tq_linear_conv_update -> bit-exact. out must NOT alias x (the serial
// kernel overwrote x[p] after use; parallel positions still need it).
__global__ void k_tq_linear_conv_update_wide(float *out, float *state, const float *x,
                                             const uint16_t *conv_weight,
                                             int channels, int kernel_size, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= channels) return;
    int pos = blockIdx.y;
    if (pos == n) {                                // new state = the last K inputs
        float tail[8];                             // kernel_size <= 8 (host-gated)
        for (int j = 0; j < kernel_size; j++) {
            int k = n - kernel_size + j;           // x index feeding new s[j]
            tail[j] = (k >= 0) ? x[(size_t)k * channels + i]
                               : state[(size_t)i * kernel_size + (kernel_size + k)];
        }
        for (int j = 0; j < kernel_size; j++) state[(size_t)i * kernel_size + j] = tail[j];
        return;
    }
    float sum = 0.0f;
    #pragma unroll 4
    for (int j = 0; j < kernel_size; j++) {
        int k = pos - (kernel_size - 1) + j;       // absolute chunk index of tap j
        float v = (k >= 0) ? x[(size_t)k * channels + i]
                           : state[(size_t)i * kernel_size + (kernel_size + k)];
        sum += v * tq_bf16_to_float(conv_weight[i * kernel_size + j]);
    }
    out[(size_t)pos * channels + i] = sum / (1.0f + expf(-sum));
}

// Fused-tree conv update: processes ALL depth groups in a single launch.
// Blocks self-synchronise via a global atomic counter so that depth d waits
// for every block at depths 0..d-1 before reading parent conv state.
// Bit-identical per node to k_tq_linear_conv_update_b_fork.
__global__ void k_tq_linear_conv_update_fused_tree(float *out, float *state, const float *x,
                                                   const uint16_t *conv_weight, int channels, int kernel_size,
                                                   const int *group_node_ids, const int *parent_ids,
                                                   int state_stride, const float *committed_state,
                                                   const int *depth_offsets, int n_depths,
                                                   int *depth_done) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int slot = blockIdx.y;
    int my_depth = n_depths - 1;
    for (int d = 0; d < n_depths; d++) {
        if (slot < depth_offsets[d + 1]) { my_depth = d; break; }
    }
    int blocks_before = depth_offsets[my_depth] * (int)gridDim.x;
    if (blocks_before > 0) {
        if (threadIdx.x == 0)
            while (atomicAdd(depth_done, 0) < blocks_before) {}
        __syncthreads();
        __threadfence();
    }
    if (i < channels) {
        int node = group_node_ids[slot];
        int par = parent_ids[node];
        const float *src = (par >= 0) ? state + (size_t)par * state_stride + (size_t)i * kernel_size
                                      : committed_state + (size_t)i * kernel_size;
        float *dst = state + (size_t)node * state_stride + (size_t)i * kernel_size;
        const float *xn = x + (size_t)node * channels;
        float *outn = out + (size_t)node * channels;
        float sum = 0.0f;
        for (int j = 0; j < kernel_size - 1; j++) {
            dst[j] = src[j + 1];
            sum += dst[j] * tq_bf16_to_float(conv_weight[i * kernel_size + j]);
        }
        dst[kernel_size - 1] = xn[i];
        sum += xn[i] * tq_bf16_to_float(conv_weight[i * kernel_size + (kernel_size - 1)]);
        outn[i] = sum / (1.0f + expf(-sum));
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(depth_done, 1);
}

__global__ void k_tq_linear_first_core_gated(float *out, const float *qkv_conv, const float *z,
                                             const float *b_proj, const float *norm_weight,
                                             int value_heads, int key_heads, int dim, float eps) {
    __shared__ float partial[4];
    __shared__ float q_sum_shared;
    __shared__ float k_sum_shared;
    __shared__ float dot_shared;
    __shared__ float core_sum_shared;
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
        float v = partial[tid];
        for (int offset = 2; offset > 0; offset >>= 1) v += __shfl_down_sync(0x0f, v, offset);
        if (tid == 0) q_sum_shared = v;
    }
    __syncthreads();
    if ((tid & 31) == 0) partial[tid >> 5] = ksum;
    __syncthreads();
    if (tid < 4) {
        float v = partial[tid];
        for (int offset = 2; offset > 0; offset >>= 1) v += __shfl_down_sync(0x0f, v, offset);
        if (tid == 0) k_sum_shared = v;
    }
    __syncthreads();

    float q_norm = qv * rsqrtf(q_sum_shared + 1.0e-6f);
    float k_norm = kv * rsqrtf(k_sum_shared + 1.0e-6f);
    float dot = q_norm * k_norm;
    for (int offset = 16; offset > 0; offset >>= 1) {
        dot += __shfl_down_sync(0xffffffff, dot, offset);
    }
    if ((tid & 31) == 0) partial[tid >> 5] = dot;
    __syncthreads();
    if (tid < 4) {
        float v = partial[tid];
        for (int offset = 2; offset > 0; offset >>= 1) v += __shfl_down_sync(0x0f, v, offset);
        if (tid == 0) dot_shared = v * rsqrtf((float)dim);
    }
    __syncthreads();

    float beta = 1.0f / (1.0f + expf(-b_proj[head]));
    float core = dot_shared * beta * v[tid];
    float core_sq = core * core;
    for (int offset = 16; offset > 0; offset >>= 1) {
        core_sq += __shfl_down_sync(0xffffffff, core_sq, offset);
    }
    if ((tid & 31) == 0) partial[tid >> 5] = core_sq;
    __syncthreads();
    if (tid < 4) {
        float v = partial[tid];
        for (int offset = 2; offset > 0; offset >>= 1) v += __shfl_down_sync(0x0f, v, offset);
        if (tid == 0) core_sum_shared = v;
    }
    __syncthreads();

    float normed = core * rsqrtf(core_sum_shared / (float)dim + eps) * norm_weight[tid];
    float gate = z_head[tid];
    float silu_gate = gate / (1.0f + expf(-gate));
    out[head * dim + tid] = normed * silu_gate;
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

// Batched head-split DeltaNet core over a group of sibling nodes (Step-5 verify).
// grid=(value_heads*HS, gc); node = group_ids[blockIdx.y]. Each node has its own
// recurrent state (stride recur_stride), qkv (conv_dim), b/a proj (value_heads) and
// core out (core_stride). Bit-identical per node to k_tq_linear_decode_core_hs.
template <int G, int HS>
__global__ void k_tq_linear_decode_core_hs_b(float *core_out, float *recurrent_state,
                                             const float *qkv_conv,
                                             const float *b_proj, const float *a_proj,
                                             const float *A_log, const uint16_t *dt_bias,
                                             int value_heads, int key_heads, int dim,
                                             const int *group_ids, int recur_stride,
                                             int conv_dim, int core_stride) {
    extern __shared__ float smem[];
    float *qn = smem;            // [dim]
    float *kn = smem + dim;      // [dim]
    float *red = smem + 2 * dim; // [dim]
    int node = group_ids[blockIdx.y];
    float *core_o = core_out + (size_t)node * core_stride;
    float *rstate = recurrent_state + (size_t)node * recur_stride;
    qkv_conv += (size_t)node * conv_dim;
    b_proj += (size_t)node * value_heads;
    a_proj += (size_t)node * value_heads;
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
    float *state = rstate + (size_t)head * dim * dim;

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

    if (klane == 0) core_o[head * dim + tid] = core;
}

// Fused-tree head-split DeltaNet core: processes ALL depth groups in a single
// launch. Blocks at depth d spin-wait on a global atomic counter until every
// block at depths 0..d-1 has finished, then compute their node's head slice.
// Bit-identical per node to k_tq_linear_decode_core_hs_b preceded by a copy of the parent's state.
template <int G, int HS>
__global__ void k_tq_linear_decode_core_hs_fused_tree(float *core_out, float *recurrent_state,
                                             const float *qkv_conv,
                                             const float *b_proj, const float *a_proj,
                                             const float *A_log, const uint16_t *dt_bias,
                                             int value_heads, int key_heads, int dim,
                                             const int *group_node_ids, const int *parent_ids,
                                             int recur_stride, int conv_dim, int core_stride,
                                             const float *committed_recur,
                                             const int *depth_offsets, int n_depths,
                                             int *depth_done) {
    extern __shared__ float smem[];
    float *qn = smem;
    float *kn = smem + dim;
    float *red = smem + 2 * dim;
    int slot = blockIdx.y;

    int my_depth = n_depths - 1;
    for (int d = 0; d < n_depths; d++) {
        if (slot < depth_offsets[d + 1]) { my_depth = d; break; }
    }
    int blocks_before = depth_offsets[my_depth] * (int)gridDim.x;
    if (blocks_before > 0) {
        if (threadIdx.x == 0)
            while (atomicAdd(depth_done, 0) < blocks_before) {}
        __syncthreads();
        __threadfence();
    }

    int node = group_node_ids[slot];
    int par = parent_ids[node];
    float *core_o = core_out + (size_t)node * core_stride;
    const float *src_rstate = (par >= 0) ? recurrent_state + (size_t)par * recur_stride
                                         : committed_recur;
    float *dst_rstate = recurrent_state + (size_t)node * recur_stride;
    qkv_conv += (size_t)node * conv_dim;
    b_proj += (size_t)node * value_heads;
    a_proj += (size_t)node * value_heads;
    int blk = blockIdx.x;
    int head = blk / HS;
    int part = blk % HS;
    if (head >= value_heads) {
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) atomicAdd(depth_done, 1);
        return;
    }
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
    const float *src_state = src_rstate + (size_t)head * dim * dim;
    float *dst_state = dst_rstate + (size_t)head * dim * dim;

    float st[(128 + G - 1) / G];
    int nkk = 0;
    float psum = 0.0f;
    for (int kk = klane; kk < dim; kk += G) {
        float s = src_state[(size_t)kk * dim + tid];
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
        dst_state[(size_t)kk * dim + tid] = newv;
        pcore += newv * qn[kk];
        nkk++;
    }
    #pragma unroll
    for (int off = G >> 1; off > 0; off >>= 1)
        pcore += __shfl_down_sync(0xffffffffu, pcore, off, G);
    float core = __shfl_sync(0xffffffffu, pcore, 0, G);

    if (klane == 0) core_o[head * dim + tid] = core;

    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(depth_done, 1);
}

// Fully-fused DeltaNet kernel: conv update + core recurrence for ALL depth groups
// in a SINGLE kernel launch. Replaces the separate conv-fused-tree + core-fused-tree
// pair, saving one kernel launch per linear layer (~48 launches total).
// Each block computes the conv output for its head's q/k/v channels in shared memory
// (avoiding the global-memory round-trip), writes its partition of the conv state for
// children, then proceeds with the core recurrence. Depth synchronisation via a single
// global atomic counter (same scheme as the separate fused-tree kernels).
// Bit-identical per node to the separate conv + core fused-tree path.
template <int G, int HS>
__global__ void k_tq_deltanet_fused_depths(float *core_out, float *recurrent_state,
                                           float *conv_state, const float *qkv_in,
                                           const float *b_proj, const float *a_proj,
                                           const float *A_log, const uint16_t *dt_bias,
                                           const uint16_t *conv_weight,
                                           int value_heads, int key_heads, int dim,
                                           int kernel_size,
                                           const int *group_node_ids, const int *parent_ids,
                                           int recur_stride, int conv_dim, int core_stride,
                                           int conv_state_stride,
                                           const float *committed_recur,
                                           const float *committed_conv,
                                           const int *depth_offsets, int n_depths,
                                           int *depth_done) {
    extern __shared__ float smem[];
    float *qn = smem;
    float *kn = smem + dim;
    float *red = smem + 2 * dim;
    int slot = blockIdx.y;

    int my_depth = n_depths - 1;
    for (int d = 0; d < n_depths; d++) {
        if (slot < depth_offsets[d + 1]) { my_depth = d; break; }
    }
    int blocks_before = depth_offsets[my_depth] * (int)gridDim.x;
    if (blocks_before > 0) {
        if (threadIdx.x == 0)
            while (atomicAdd(depth_done, 0) < blocks_before) { __nanosleep(100); }
        __syncthreads();
        __threadfence();
    }

    int node = group_node_ids[slot];
    int par = parent_ids[node];
    const float *src_conv = (par >= 0) ? conv_state + (size_t)par * conv_state_stride
                                       : committed_conv;
    float *dst_conv = conv_state + (size_t)node * conv_state_stride;
    const float *src_rstate = (par >= 0) ? recurrent_state + (size_t)par * recur_stride
                                         : committed_recur;
    float *dst_rstate = recurrent_state + (size_t)node * recur_stride;
    const float *xn = qkv_in + (size_t)node * conv_dim;
    float *core_o = core_out + (size_t)node * core_stride;

    int blk = blockIdx.x;
    int head = blk / HS;
    int part = blk % HS;
    int t = threadIdx.x;
    int bs = blockDim.x;

    // --- Conv state write: each block handles its partition of channels ---
    int total_blocks_x = value_heads * HS;
    int ch_per_block = (conv_dim + total_blocks_x - 1) / total_blocks_x;
    int ch_start = blk * ch_per_block;
    int ch_end = ch_start + ch_per_block;
    if (ch_end > conv_dim) ch_end = conv_dim;
    for (int ch = ch_start + t; ch < ch_end; ch += bs) {
        const float *s_src = src_conv + (size_t)ch * kernel_size;
        float *s_dst = dst_conv + (size_t)ch * kernel_size;
        for (int j = 0; j < kernel_size - 1; j++)
            s_dst[j] = s_src[j + 1];
        s_dst[kernel_size - 1] = xn[ch];
    }

    if (head >= value_heads) {
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) atomicAdd(depth_done, 1);
        return;
    }

    // --- Conv output for q/k channels → shared memory ---
    int group = value_heads / key_heads;
    int key_head = head / group;
    int key_dim_total = key_heads * dim;
    int q_off = key_head * dim;
    int k_off = key_dim_total + key_head * dim;
    for (int i = t; i < dim; i += bs) {
        int qch = q_off + i;
        const float *qs = src_conv + (size_t)qch * kernel_size;
        float qval = 0.0f;
        for (int j = 0; j < kernel_size - 1; j++)
            qval += qs[j + 1] * tq_bf16_to_float(conv_weight[qch * kernel_size + j]);
        qval += xn[qch] * tq_bf16_to_float(conv_weight[qch * kernel_size + (kernel_size - 1)]);
        qn[i] = qval / (1.0f + expf(-qval));

        int kch = k_off + i;
        const float *ks = src_conv + (size_t)kch * kernel_size;
        float kval = 0.0f;
        for (int j = 0; j < kernel_size - 1; j++)
            kval += ks[j + 1] * tq_bf16_to_float(conv_weight[kch * kernel_size + j]);
        kval += xn[kch] * tq_bf16_to_float(conv_weight[kch * kernel_size + (kernel_size - 1)]);
        kn[i] = kval / (1.0f + expf(-kval));
    }
    __syncthreads();

    // --- Conv output for v channel → register ---
    int tids_per = dim / HS;
    int tid_local = t / G;
    int klane = t % G;
    int tid = part * tids_per + tid_local;
    int v_off = 2 * key_dim_total + head * dim;
    int vch = v_off + tid;
    const float *vs = src_conv + (size_t)vch * kernel_size;
    float vsum = 0.0f;
    for (int j = 0; j < kernel_size - 1; j++)
        vsum += vs[j + 1] * tq_bf16_to_float(conv_weight[vch * kernel_size + j]);
    vsum += xn[vch] * tq_bf16_to_float(conv_weight[vch * kernel_size + (kernel_size - 1)]);
    float v_val = vsum / (1.0f + expf(-vsum));

    // --- q/k normalization ---
    for (int i = t; i < dim; i += bs) red[i] = qn[i] * qn[i];
    __syncthreads();
    float qsum = tq_block_reduce_sum_sh(red, dim, t);
    __syncthreads();
    for (int i = t; i < dim; i += bs) red[i] = kn[i] * kn[i];
    __syncthreads();
    float ksum = tq_block_reduce_sum_sh(red, dim, t);
    __syncthreads();
    for (int i = t; i < dim; i += bs) {
        qn[i] = qn[i] * rsqrtf(qsum + 1.0e-6f) * rsqrtf((float)dim);
        kn[i] = kn[i] * rsqrtf(ksum + 1.0e-6f);
    }
    __syncthreads();

    // --- DeltaNet core recurrence ---
    const float *bp = b_proj + (size_t)node * value_heads;
    const float *ap = a_proj + (size_t)node * value_heads;
    float beta = 1.0f / (1.0f + expf(-bp[head]));
    float softplus_arg = ap[head] + tq_bf16_to_float(dt_bias[head]);
    float softplus = log1pf(expf(-fabsf(softplus_arg))) + fmaxf(softplus_arg, 0.0f);
    float gg = -expf(A_log[head]) * softplus;
    float decay = expf(gg);
    const float *src_state = src_rstate + (size_t)head * dim * dim;
    float *dst_state = dst_rstate + (size_t)head * dim * dim;

    float st[(128 + G - 1) / G];
    int nkk = 0;
    float psum = 0.0f;
    for (int kk = klane; kk < dim; kk += G) {
        float s = src_state[(size_t)kk * dim + tid];
        st[nkk++] = s;
        psum += s * kn[kk];
    }
    #pragma unroll
    for (int off = G >> 1; off > 0; off >>= 1)
        psum += __shfl_down_sync(0xffffffffu, psum, off, G);
    float kv_mem = __shfl_sync(0xffffffffu, psum, 0, G) * decay;
    float delta = (v_val - kv_mem) * beta;

    float pcore = 0.0f;
    nkk = 0;
    for (int kk = klane; kk < dim; kk += G) {
        float newv = st[nkk] * decay + kn[kk] * delta;
        dst_state[(size_t)kk * dim + tid] = newv;
        pcore += newv * qn[kk];
        nkk++;
    }
    #pragma unroll
    for (int off = G >> 1; off > 0; off >>= 1)
        pcore += __shfl_down_sync(0xffffffffu, pcore, off, G);
    float core = __shfl_sync(0xffffffffu, pcore, 0, G);

    if (klane == 0) core_o[head * dim + tid] = core;

    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(depth_done, 1);
}
