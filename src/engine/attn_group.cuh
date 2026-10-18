// the single-stream verify path's GROUP / GROUP2 attention kernels.
// Part of the forward_qwen.cu unity build: included in order, not a standalone header.

// ===== GROUP attention (TQ_ATTN_MMA_GROUP, standalone verify path only) =====
// One item = a WHOLE kv group (6 q-heads = 3 pairs) x one position chunk, so the
// group's K codes/scales are loaded once for 3 pairs' score MMAs and the V codes
// once for 3 pairs' P.V MMAs (the pair path re-reads them per pair-item; the
// phase is L1TEX-latency-bound, DRAM ~6%). To hold 3 pairs' O accumulators under
// the register budget the P.V phase runs in TWO column halves (outer half loop
// re-runs the score pass; K re-read once more, still 2/3 of the pair path's K
// traffic and 1/3 of its V traffic). Per-row math and the chunk-partial layout
// are IDENTICAL to the pair path (same tile order, same merge) -> bit-exact.
// Q A-fragments come pre-staged from k_tq_attn_qa_stage (one small launch per
// layer) -- the in-item Q-prep would run 3x more often at the 3x finer chunking
// this mode wants. MODE 3 (prod Q4 KV) only; requires N <= 8, group == 6,
// nchunks > 1 (partial+merge path; at the >=96k gate this always holds).
//
// dynamic smem layout (words):
#define TQ_AMG_QA(p)   ((p) * 2048)         // [3 pairs][16 chunks][32 lanes][4] staged A-frags
#define TQ_AMG_PST(p)  (6144 + (p) * 1024)  // [3 pairs][8 tiles][32][4] P A-frag staging
#define TQ_AMG_WMAX    9216                 // [8 warps][16 rows] tile row-max (per-pair, reused)
#define TQ_AMG_WSUM    9344                 // [8 warps][16 rows]
#define TQ_AMG_MST(p)  (9472 + (p) * 16)    // [3][16] running row max
#define TQ_AMG_LST(p)  (9520 + (p) * 16)    // [3][16] running row denom
#define TQ_AMG_ALPH    9568                 // [16] per-row rescale (transient per pair)
#define TQ_AMG_AMASK   9584                 // [16] ancestor slot bitmask per row
#define TQ_AMG_WORDS   9600
#define TQ_ATTN_MMA_GROUP_SMEM ((size_t)TQ_AMG_WORDS * 4)

// Stage the Q A-fragments for all (pair, node) rows of one layer: grid = nh/2
// pair-blocks x 256 threads; block p writes the exact QA words the pair kernel's
// in-item Q-prep would build (same math incl. FWHT) to stage[p * 2048 ..].
__global__ void k_tq_attn_qa_stage(
    uint32_t *stage, const float *q_proj_base, const uint16_t *q_norm_w,
    const int *pos_arr, int N, int nh, int hd,
    int q_stride, float eps, float rope_theta) {
    __shared__ float qtmp[256];
    __shared__ float red[9];
    __shared__ uint32_t qa[2048];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int prg = blockIdx.x;                      // global pair id = head/2
    const int head = 2 * prg;
    for (int i = tid; i < 2048; i += 256) qa[i] = 0;
    __syncthreads();
    for (int rq = 0; rq < 16; rq++) {
        const int n = rq & 7;
        if (n >= N) continue;
        const int hh = head + (rq >> 3);
        const float *q_proj = q_proj_base + (size_t)n * q_stride;
        int pos = pos_arr[n];
        float qv = q_proj[hh * (2 * hd) + tid];
        float qsum = qv * qv;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) qsum += __shfl_down_sync(0xffffffffu, qsum, o);
        if (lane == 0) red[warp] = qsum;
        __syncthreads();
        if (tid < 8) {
            float v = red[tid];
            for (int o = 4; o > 0; o >>= 1) v += __shfl_down_sync(0xffu, v, o);
            if (tid == 0) red[8] = v;
        }
        __syncthreads();
        float q_sum_shared = red[8];
        qv = qv * rsqrtf(q_sum_shared / (float)hd + eps) * (1.0f + tq_bf16_to_float(q_norm_w[tid]));
        if (tid < 64) {
            int idx = tid & 31;
            float freq = powf(rope_theta, -((float)(2 * idx) / 64.0f));
            float angle = (float)pos * freq;
            float c = cosf(angle), s = sinf(angle);
            float q_pair = q_proj[hh * (2 * hd) + ((tid < 32) ? tid + 32 : tid - 32)];
            q_pair = q_pair * rsqrtf(q_sum_shared / (float)hd + eps) *
                     (1.0f + tq_bf16_to_float(q_norm_w[(tid < 32) ? tid + 32 : tid - 32]));
            float q_rot = (tid < 32) ? -q_pair : q_pair;
            qv = qv * c + q_rot * s;
        }
        __syncthreads();
        qtmp[tid] = qv;
        __syncthreads();
        tq_fwht256(qtmp, tid);                       // MODE 3 rotated-space q
        if (tid < 128) {
            int c16 = tid >> 3, tt = (tid >> 1) & 3, rsel = tid & 1;
            int d0 = 16 * c16 + 2 * tt + 8 * rsel;
            int l = 4 * (rq & 7) + tt, r = ((rq >> 3) & 1) + 2 * rsel;
            qa[(c16 * 32 + l) * 4 + r] = tq_pack_bf16(qtmp[d0], qtmp[d0 + 1]);
        }
        __syncthreads();
    }
    for (int i = tid; i < 2048; i += 256) stage[(size_t)prg * 2048 + i] = qa[i];
}

// The group attention kernel: grid (nkv, nchunks); MODE 3, partial-writing only.
// __launch_bounds__(256, 2): unconstrained the kernel takes 136 registers ->
// 1 CTA/SM and loses to the pair path below ~180k (measured); capping at 128
// (a few local spills) keeps 2 CTAs/SM like the pair kernel.
__global__ void __launch_bounds__(256, 2) k_tq_spec_attn_group_mma(
    const uint32_t *qa_stage,
    const uint8_t *kc4, const float *kq4s, const uint8_t *vc8, const float *vscale,
    const float *ka, const float *va,
    const int *pos_arr, const int *anc, int N,
    int nh, int nkv, int hd, int kv_m, float *part) {
    extern __shared__ uint32_t sm[];
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int gr = lane >> 2, t4 = lane & 3;
    const int base = pos_arr[0] - 1;                 // root at depth 0 (graph-safe)
    const int kv_head = blockIdx.x, chunk = blockIdx.y, nchunks = gridDim.y;
    const int pair0 = kv_head * 3;                   // global pair ids pair0..pair0+2
    const size_t cstride4 = (size_t)(nkv * hd) >> 1, szstride = (size_t)nkv * 16;
    const size_t astride = (size_t)kv_m;
    const uint8_t *kcb4 = kc4 + (size_t)kv_head * (hd >> 1);
    const float *ksz4 = q4s_adv(kq4s, (size_t)kv_head * 16);
    const uint8_t *vcb8 = vc8 + (size_t)kv_head * hd;
    const float *vscb = vscale + kv_head;
    const float *kab = ka + (size_t)kv_head * hd;
    const float *vab = va + (size_t)kv_head * hd;
    float *wmax = (float *)(sm + TQ_AMG_WMAX), *wsum = (float *)(sm + TQ_AMG_WSUM);
    float *alph = (float *)(sm + TQ_AMG_ALPH);

    // staged A-frags for the group's 3 pairs + ancestor masks
    for (int p = 0; p < 3; p++)
        for (int i = tid; i < 2048; i += 256)
            sm[TQ_AMG_QA(p) + i] = qa_stage[(size_t)(pair0 + p) * 2048 + i];
    if (tid < 16) {
        uint32_t m = 0;
        int node = tid & 7;
        if (node < N) {
            int len = pos_arr[node] - base;          // chain incl. the node itself
            const int *an = anc + node * TQ_SPEC_MAX_N;
            for (int d = 0; d < len; d++) m |= 1u << an[d];
        }
        sm[TQ_AMG_AMASK + tid] = m;
    }
    __syncthreads();
    const uint32_t amask_lo = sm[TQ_AMG_AMASK + gr], amask_hi = sm[TQ_AMG_AMASK + 8 + gr];

    const float rsc = rsqrtf((float)hd);
    const int nst_all = base >= 0 ? (base + 128) >> 7 : 0;
    const int st_per = (nst_all + nchunks - 1) / nchunks;
    const int st_lo = chunk * st_per;
    int st_hi_t = (chunk + 1) * st_per;
    const int st_hi = st_hi_t < nst_all ? st_hi_t : nst_all;
    const int do_arch = (chunk == 0);
    const int nst = st_hi > st_lo ? st_hi - st_lo : 0;

    for (int half = 0; half < 2; half++) {
        // per-half softmax state reset (both halves recompute identical m/l)
        if (tid < 16) {
            for (int p = 0; p < 3; p++) {
                ((float *)(sm + TQ_AMG_MST(p)))[tid] = TQ_AMM_NINF;
                ((float *)(sm + TQ_AMG_LST(p)))[tid] = 0.0f;
            }
        }
        __syncthreads();
        float oacc[3][2][4];                          // [pair][col-frag of this half][j]
        #pragma unroll
        for (int p = 0; p < 3; p++)
            #pragma unroll
            for (int f = 0; f < 2; f++)
                #pragma unroll
                for (int j = 0; j < 4; j++) oacc[p][f][j] = 0.0f;

        for (int it = 0; it < nst + do_arch; it++) {
            const int is_arch = (it == nst);
            const int nwt = is_arch ? 1 : 8;
            const int t0 = (st_lo + it) << 7;
            float s[3][8];                            // 3 pairs' score tiles (s0|s1)
            if (warp < nwt) {
                #pragma unroll
                for (int p = 0; p < 3; p++)
                    #pragma unroll
                    for (int j = 0; j < 8; j++) s[p][j] = 0.0f;
                if (is_arch) {
                    const float *kr0 = kab + (size_t)gr * astride;
                    const float *kr1 = kab + (size_t)(8 + gr) * astride;
                    #pragma unroll
                    for (int c = 0; c < 16; c++) {
                        int d0 = 16 * c + 2 * t4;
                        float2 x0 = *(const float2 *)(kr0 + d0);
                        float2 x1 = *(const float2 *)(kr0 + d0 + 8);
                        float2 y0 = *(const float2 *)(kr1 + d0);
                        float2 y1 = *(const float2 *)(kr1 + d0 + 8);
                        uint32_t b00 = tq_pack_bf16(x0.x, x0.y), b01 = tq_pack_bf16(x1.x, x1.y);
                        uint32_t b10 = tq_pack_bf16(y0.x, y0.y), b11 = tq_pack_bf16(y1.x, y1.y);
                        #pragma unroll
                        for (int p = 0; p < 3; p++) {
                            const uint32_t *a = &sm[TQ_AMG_QA(p) + (c * 32 + lane) * 4];
                            tq_mma_bf16_m16n8k16(&s[p][0], a, b00, b01);
                            tq_mma_bf16_m16n8k16(&s[p][4], a, b10, b11);
                        }
                    }
                    #pragma unroll
                    for (int p = 0; p < 3; p++)
                        #pragma unroll
                        for (int j = 0; j < 4; j++) {
                            int col = 2 * t4 + (j & 1);
                            uint32_t am = (j < 2) ? amask_lo : amask_hi;
                            s[p][j]     = ((am >> col) & 1) ? s[p][j] * rsc : TQ_AMM_NINF;
                            s[p][4 + j] = ((am >> (col + 8)) & 1) ? s[p][4 + j] * rsc : TQ_AMM_NINF;
                        }
                } else {
                    const int tp = t0 + warp * 16;
                    int p0 = tp + gr, p1 = tp + 8 + gr;
                    const uint8_t *kr0 = kcb4 + (size_t)min(p0, base) * cstride4;
                    const uint8_t *kr1 = kcb4 + (size_t)min(p1, base) * cstride4;
                    const float *sz0 = q4s_adv(ksz4, (size_t)min(p0, base) * szstride);
                    const float *sz1 = q4s_adv(ksz4, (size_t)min(p1, base) * szstride);
                    #pragma unroll
                    for (int g = 0; g < 8; g++) {
                        float2 p0s = q4s_pair(sz0, 2 * g);
                        float2 p1s = q4s_pair(sz1, 2 * g);
                        #pragma unroll
                        for (int cc = 0; cc < 2; cc++) {
                            int c = 2 * g + cc;
                            uint64_t w0 = __ldg((const unsigned long long *)(kr0 + 8 * c));
                            uint64_t w1 = __ldg((const unsigned long long *)(kr1 + 8 * c));
                            uint8_t a0 = (uint8_t)(w0 >> (8 * t4));
                            uint8_t a8 = (uint8_t)(w0 >> (8 * t4 + 32));
                            uint8_t b0v = (uint8_t)(w1 >> (8 * t4));
                            uint8_t b8 = (uint8_t)(w1 >> (8 * t4 + 32));
                            uint32_t kb00 = tq_pack_bf16(((float)(a0 & 15) - p0s.y) * p0s.x,
                                                         ((float)(a0 >> 4) - p0s.y) * p0s.x);
                            uint32_t kb01 = tq_pack_bf16(((float)(a8 & 15) - p0s.y) * p0s.x,
                                                         ((float)(a8 >> 4) - p0s.y) * p0s.x);
                            uint32_t kb10 = tq_pack_bf16(((float)(b0v & 15) - p1s.y) * p1s.x,
                                                         ((float)(b0v >> 4) - p1s.y) * p1s.x);
                            uint32_t kb11 = tq_pack_bf16(((float)(b8 & 15) - p1s.y) * p1s.x,
                                                         ((float)(b8 >> 4) - p1s.y) * p1s.x);
                            #pragma unroll
                            for (int p = 0; p < 3; p++) {
                                const uint32_t *a = &sm[TQ_AMG_QA(p) + (c * 32 + lane) * 4];
                                tq_mma_bf16_m16n8k16(&s[p][0], a, kb00, kb01);
                                tq_mma_bf16_m16n8k16(&s[p][4], a, kb10, kb11);
                            }
                        }
                    }
                    #pragma unroll
                    for (int p = 0; p < 3; p++)
                        #pragma unroll
                        for (int j = 0; j < 8; j++) s[p][j] *= rsc;
                    // causal cut at base (uniform across rows)
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        int c0 = tp + 2 * t4 + (j & 1);
                        #pragma unroll
                        for (int p = 0; p < 3; p++) {
                            if (c0 > base) s[p][j] = TQ_AMM_NINF;
                            if (c0 + 8 > base) s[p][4 + j] = TQ_AMM_NINF;
                        }
                    }
                }
            }
            // per pair: online-softmax update + P staging (V scale folded)
            for (int p = 0; p < 3; p++) {
                float *mst = (float *)(sm + TQ_AMG_MST(p));
                float *lst = (float *)(sm + TQ_AMG_LST(p));
                if (warp < nwt) {
                    float lm_lo = fmaxf(fmaxf(s[p][0], s[p][1]), fmaxf(s[p][4], s[p][5]));
                    float lm_hi = fmaxf(fmaxf(s[p][2], s[p][3]), fmaxf(s[p][6], s[p][7]));
                    #pragma unroll
                    for (int o = 1; o <= 2; o <<= 1) {
                        lm_lo = fmaxf(lm_lo, __shfl_xor_sync(0xffffffffu, lm_lo, o));
                        lm_hi = fmaxf(lm_hi, __shfl_xor_sync(0xffffffffu, lm_hi, o));
                    }
                    if (t4 == 0) { wmax[warp * 16 + gr] = lm_lo; wmax[warp * 16 + 8 + gr] = lm_hi; }
                }
                __syncthreads();
                if (warp == 0 && lane < 16) {
                    float m_old = mst[lane], m_new = m_old;
                    for (int w = 0; w < nwt; w++) m_new = fmaxf(m_new, wmax[w * 16 + lane]);
                    alph[lane] = expf(m_old - m_new);
                    mst[lane] = m_new;
                }
                __syncthreads();
                if (warp < nwt) {
                    float mn_lo = mst[gr], mn_hi = mst[8 + gr];
                    float p00 = expf(s[p][0] - mn_lo), p01 = expf(s[p][1] - mn_lo);
                    float p02 = expf(s[p][2] - mn_hi), p03 = expf(s[p][3] - mn_hi);
                    float p10 = expf(s[p][4] - mn_lo), p11 = expf(s[p][5] - mn_lo);
                    float p12 = expf(s[p][6] - mn_hi), p13 = expf(s[p][7] - mn_hi);
                    float ls_lo = p00 + p01 + p10 + p11;
                    float ls_hi = p02 + p03 + p12 + p13;
                    #pragma unroll
                    for (int o = 1; o <= 2; o <<= 1) {
                        ls_lo += __shfl_xor_sync(0xffffffffu, ls_lo, o);
                        ls_hi += __shfl_xor_sync(0xffffffffu, ls_hi, o);
                    }
                    if (t4 == 0) { wsum[warp * 16 + gr] = ls_lo; wsum[warp * 16 + 8 + gr] = ls_hi; }
                    if (!is_arch) {
                        int p0c = t0 + warp * 16 + 2 * t4, p1c = p0c + 8;
                        float v00 = vscb[(size_t)min(p0c, base) * nkv];
                        float v01 = vscb[(size_t)min(p0c + 1, base) * nkv];
                        float v10 = vscb[(size_t)min(p1c, base) * nkv];
                        float v11 = vscb[(size_t)min(p1c + 1, base) * nkv];
                        p00 *= v00; p01 *= v01; p02 *= v00; p03 *= v01;
                        p10 *= v10; p11 *= v11; p12 *= v10; p13 *= v11;
                    }
                    uint32_t *pst = &sm[TQ_AMG_PST(p) + (warp * 32 + lane) * 4];
                    pst[0] = tq_pack_bf16(p00, p01);
                    pst[1] = tq_pack_bf16(p02, p03);
                    pst[2] = tq_pack_bf16(p10, p11);
                    pst[3] = tq_pack_bf16(p12, p13);
                }
                __syncthreads();
                if (warp == 0 && lane < 16) {
                    float l = lst[lane] * alph[lane];
                    for (int w = 0; w < nwt; w++) l += wsum[w * 16 + lane];
                    lst[lane] = l;
                }
                // rescale this pair's O slice while alph is fresh
                {
                    float a_lo = alph[gr], a_hi = alph[8 + gr];
                    #pragma unroll
                    for (int f = 0; f < 2; f++) {
                        oacc[p][f][0] *= a_lo; oacc[p][f][1] *= a_lo;
                        oacc[p][f][2] *= a_hi; oacc[p][f][3] *= a_hi;
                    }
                }
                __syncthreads();                      // alph reused by the next pair
            }
            // P.V for this half's columns: V fragments built once, 3 pairs' MMAs
            for (int tt = 0; tt < nwt; tt++) {
                #pragma unroll
                for (int f = 0; f < 2; f++) {
                    int fg = 2 * half + f;
                    int col = warp * 32 + fg * 8 + gr;
                    uint32_t b0, b1;
                    if (is_arch) {
                        int r0 = 2 * t4;
                        b0 = tq_pack_bf16(vab[(size_t)r0 * astride + col],
                                          vab[(size_t)(r0 + 1) * astride + col]);
                        b1 = tq_pack_bf16(vab[(size_t)(r0 + 8) * astride + col],
                                          vab[(size_t)(r0 + 9) * astride + col]);
                    } else {
                        int r0 = t0 + tt * 16 + 2 * t4;
                        b0 = tq_pack_bf16(
                            tq_e4m3_dec_fast(vcb8[(size_t)min(r0, base) * (size_t)(nkv * hd) + col]),
                            tq_e4m3_dec_fast(vcb8[(size_t)min(r0 + 1, base) * (size_t)(nkv * hd) + col]));
                        b1 = tq_pack_bf16(
                            tq_e4m3_dec_fast(vcb8[(size_t)min(r0 + 8, base) * (size_t)(nkv * hd) + col]),
                            tq_e4m3_dec_fast(vcb8[(size_t)min(r0 + 9, base) * (size_t)(nkv * hd) + col]));
                    }
                    #pragma unroll
                    for (int p = 0; p < 3; p++) {
                        const uint32_t *pa = &sm[TQ_AMG_PST(p) + (tt * 32 + lane) * 4];
                        uint32_t a[4] = {pa[0], pa[1], pa[2], pa[3]};
                        tq_mma_bf16_m16n8k16(oacc[p][f], a, b0, b1);
                    }
                }
            }
            __syncthreads();                          // PST/wmax reuse next iter
        }
        // chunk partials for this half's columns; m/l written by half 0 only
        // (half 1 recomputes identical values). Pair p's halves land in the
        // per-head slabs of heads 6*kv_head + 2p and +2p+1 like the pair path.
        for (int p = 0; p < 3; p++) {
            float *mst = (float *)(sm + TQ_AMG_MST(p));
            float *lst = (float *)(sm + TQ_AMG_LST(p));
            int head_lo = 6 * kv_head + 2 * p, head_hi = head_lo + 1;
            float *ps = part + (size_t)(head_lo * nchunks + chunk) * TQ_AMM_PART_F;
            float *ph = part + (size_t)(head_hi * nchunks + chunk) * TQ_AMM_PART_F;
            if (half == 0) {
                if (tid < 8)       { ps[tid] = mst[tid]; ps[16 + tid] = lst[tid]; }
                else if (tid < 16) { ph[tid - 8] = mst[tid]; ph[16 + tid - 8] = lst[tid]; }
            }
            float *po = ps + 32, *qo = ph + 32;
            #pragma unroll
            for (int f = 0; f < 2; f++) {
                int col = warp * 32 + (2 * half + f) * 8 + 2 * t4;
                #pragma unroll
                for (int j = 0; j < 2; j++) {
                    po[(size_t)gr * 256 + col + j] = oacc[p][f][j];
                    qo[(size_t)gr * 256 + col + j] = oacc[p][f][2 + j];
                }
            }
        }
        __syncthreads();
    }
}

// ===== GROUP2: producer/consumer group attention (TQ_ATTN_MMA_GROUP2) =====
// Same item = whole kv group x position chunk, but ONE 512-thread CTA at
// __launch_bounds__(512, 1): warps 0-7 (producers) run the score pass -- K
// codes/scales read ONCE per super-tile (the 2-half kernel above re-runs the
// whole score pass per column half = 2x K traffic and 2x score MMAs) -- and
// stage softmax'd P into a double-buffered smem slab; warps 8-15 (consumers)
// run P.V for the FULL 256 columns of the PREVIOUS tile in parallel (the
// 4-frag oacc fits because consumers hold no score state). Producer-internal
// softmax reductions use named barrier 1 (256 threads); one full barrier per
// tile hands the slot over. Per-row math and accumulation order are IDENTICAL
// to the 2-half kernel -> bit-exact streams; occupancy is the same 16 warps/SM
// but the two phases overlap instead of serializing.
#define TQ_AMG2_QA(p)      ((p) * 2048)          // [3 pairs][16 chunks][32][4] staged Q A-frags
#define TQ_AMG2_PST(s, p)  (6144 + (s) * 3072 + (p) * 1024)  // [2 slots][3][8 tiles][32][4] P staging
#define TQ_AMG2_WMAX       12288                 // [8 warps][16 rows] (producer scratch)
#define TQ_AMG2_WSUM       12416                 // [8 warps][16 rows]
#define TQ_AMG2_MST(p)     (12544 + (p) * 16)    // [3][16] running row max (producer)
#define TQ_AMG2_LST(p)     (12592 + (p) * 16)    // [3][16] running row denom (producer)
#define TQ_AMG2_ALPH(s, p) (12640 + (s) * 48 + (p) * 16)     // [2 slots][3][16] per-tile rescale
#define TQ_AMG2_AMASK      12736                 // [16] ancestor slot bitmask per row
#define TQ_AMG2_WORDS      12752
#define TQ_ATTN_MMA_GROUP2_SMEM ((size_t)TQ_AMG2_WORDS * 4)
// STAGE=1 extension: cp.async double-buffered staging of the K codes (128 pos x
// 128 B) and fp16 (scale,zp) rows (128 pos x 32 B) of the NEXT super-tile while
// the producers score the current one -- the in-flight bytes live in smem, not
// registers, so the L1TEX load-use chains disappear without touching the reg
// budget. fp32-scale escape hatch (TQ_KV_Q4_FP32S) falls back to STAGE=0.
#define TQ_AMG2_KST(s)     (12752 + (s) * 4096)  // [2 slots][128 pos][32 words] K codes
#define TQ_AMG2_SST(s)     (20944 + (s) * 1024)  // [2 slots][128 pos][8 words] fp16 scales
#define TQ_AMG2_WORDS_STG  22992
#define TQ_ATTN_MMA_GROUP2_SMEM_STG ((size_t)TQ_AMG2_WORDS_STG * 4)

static __device__ __forceinline__ void tq_bar_sync(int bar, int nthreads) {
    asm volatile("bar.sync %0, %1;" :: "r"(bar), "r"(nthreads));
}

static __device__ __forceinline__ void tq_cp_async16(uint32_t dst_smem, const void *src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst_smem), "l"(src));
}
static __device__ __forceinline__ void tq_cp_async8(uint32_t dst_smem, const void *src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" :: "r"(dst_smem), "l"(src));
}
static __device__ __forceinline__ void tq_cp_async4(uint32_t dst_smem, const void *src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(dst_smem), "l"(src));
}

template <int STAGE>
__global__ void __launch_bounds__(512, 1) k_tq_spec_attn_group_mma2(
    const uint32_t *qa_stage,
    const uint8_t *kc4, const float *kq4s, const uint8_t *vc8, const float *vscale,
    const float *ka, const float *va,
    const int *pos_arr, const int *anc, int N,
    int nh, int nkv, int hd, int kv_m, float *part) {
    extern __shared__ uint32_t sm[];
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int gr = lane >> 2, t4 = lane & 3;
    const int prod = warp < 8;                   // producers score, consumers P.V
    const int cw = warp & 7;                     // role-local warp id
    const int base = pos_arr[0] - 1;
    const int kv_head = blockIdx.x, chunk = blockIdx.y, nchunks = gridDim.y;
    const int pair0 = kv_head * 3;
    const size_t cstride4 = (size_t)(nkv * hd) >> 1, szstride = (size_t)nkv * 16;
    const size_t astride = (size_t)kv_m;
    const uint8_t *kcb4 = kc4 + (size_t)kv_head * (hd >> 1);
    const float *ksz4 = q4s_adv(kq4s, (size_t)kv_head * 16);
    const uint8_t *vcb8 = vc8 + (size_t)kv_head * hd;
    const float *vscb = vscale + kv_head;
    const float *kab = ka + (size_t)kv_head * hd;
    const float *vab = va + (size_t)kv_head * hd;
    float *wmax = (float *)(sm + TQ_AMG2_WMAX), *wsum = (float *)(sm + TQ_AMG2_WSUM);

    for (int p = 0; p < 3; p++)
        for (int i = tid; i < 2048; i += 512)
            sm[TQ_AMG2_QA(p) + i] = qa_stage[(size_t)(pair0 + p) * 2048 + i];
    if (tid < 16) {
        uint32_t m = 0;
        int node = tid & 7;
        if (node < N) {
            int len = pos_arr[node] - base;
            const int *an = anc + node * TQ_SPEC_MAX_N;
            for (int d = 0; d < len; d++) m |= 1u << an[d];
        }
        sm[TQ_AMG2_AMASK + tid] = m;
        for (int p = 0; p < 3; p++) {
            ((float *)(sm + TQ_AMG2_MST(p)))[tid] = TQ_AMM_NINF;
            ((float *)(sm + TQ_AMG2_LST(p)))[tid] = 0.0f;
        }
    }
    tq_bar_sync(0, 512);
    const uint32_t amask_lo = sm[TQ_AMG2_AMASK + gr], amask_hi = sm[TQ_AMG2_AMASK + 8 + gr];

    const float rsc = rsqrtf((float)hd);
    const int nst_all = base >= 0 ? (base + 128) >> 7 : 0;
    const int st_per = (nst_all + nchunks - 1) / nchunks;
    const int st_lo = chunk * st_per;
    int st_hi_t = (chunk + 1) * st_per;
    const int st_hi = st_hi_t < nst_all ? st_hi_t : nst_all;
    const int do_arch = (chunk == 0);
    const int nst = st_hi > st_lo ? st_hi - st_lo : 0;
    const int T = nst + do_arch;

    // phase ph: producers build tile ph into slot ph&1; consumers drain tile
    // ph-1 from slot (ph-1)&1. One full barrier per phase is the handoff. The
    // roles run DISJOINT loops (same bar-0 count) so the consumer-only oacc
    // does not inflate the producers' register allocation.
    if (prod) {
    const int ptid = tid;                         // producer-local tid (warps 0-7)
    // STAGE: cp.async one K super-tile (codes 128 x 128 B + fp16 scales
    // 128 x 32 B) into stage slot `sl`, clamped rows like the direct path.
    // 1280 16-B chunks / 256 threads = 5 issues per thread + one commit.
    auto stage_issue = [&](int it_s, int sl) {
        if (!STAGE || it_s >= nst) return;
        const int t0s = (st_lo + it_s) << 7;
        for (int id = ptid; id < 1024; id += 256) {
            int pos = id >> 3, sub = id & 7;
            const uint8_t *src = kcb4 + (size_t)min(t0s + pos, base) * cstride4 + 16 * sub;
            tq_cp_async16((uint32_t)__cvta_generic_to_shared(
                              sm + TQ_AMG2_KST(sl) + pos * 32 + sub * 4), src);
        }
        for (int id = ptid; id < 256; id += 256) {
            int pos = id >> 1, sub = id & 1;
            const __half *src = (const __half *)kq4s + (size_t)kv_head * 16 +
                                (size_t)min(t0s + pos, base) * szstride + 8 * sub;
            tq_cp_async16((uint32_t)__cvta_generic_to_shared(
                              sm + TQ_AMG2_SST(sl) + pos * 8 + sub * 4), src);
        }
        asm volatile("cp.async.commit_group;\n");
    };
    stage_issue(0, 0);
    for (int ph = 0; ph <= T; ph++) {
        if (ph < T) {
            const int it = ph, slot = ph & 1;
            const int is_arch = (it == nst);
            const int nwt = is_arch ? 1 : 8;
            const int t0 = (st_lo + it) << 7;
            if (STAGE && it < nst) {
                stage_issue(it + 1, (it + 1) & 1);   // overlap next tile's copy
                if (it + 1 < nst) asm volatile("cp.async.wait_group 1;\n");
                else              asm volatile("cp.async.wait_group 0;\n");
                tq_bar_sync(1, 256);                 // staged bytes visible to all producers
            }
            float s[3][8];
            if (cw < nwt) {
                #pragma unroll
                for (int p = 0; p < 3; p++)
                    #pragma unroll
                    for (int j = 0; j < 8; j++) s[p][j] = 0.0f;
                if (is_arch) {
                    const float *kr0 = kab + (size_t)gr * astride;
                    const float *kr1 = kab + (size_t)(8 + gr) * astride;
                    #pragma unroll
                    for (int c = 0; c < 16; c++) {
                        int d0 = 16 * c + 2 * t4;
                        float2 x0 = *(const float2 *)(kr0 + d0);
                        float2 x1 = *(const float2 *)(kr0 + d0 + 8);
                        float2 y0 = *(const float2 *)(kr1 + d0);
                        float2 y1 = *(const float2 *)(kr1 + d0 + 8);
                        uint32_t b00 = tq_pack_bf16(x0.x, x0.y), b01 = tq_pack_bf16(x1.x, x1.y);
                        uint32_t b10 = tq_pack_bf16(y0.x, y0.y), b11 = tq_pack_bf16(y1.x, y1.y);
                        #pragma unroll
                        for (int p = 0; p < 3; p++) {
                            const uint32_t *a = &sm[TQ_AMG2_QA(p) + (c * 32 + lane) * 4];
                            tq_mma_bf16_m16n8k16(&s[p][0], a, b00, b01);
                            tq_mma_bf16_m16n8k16(&s[p][4], a, b10, b11);
                        }
                    }
                    #pragma unroll
                    for (int p = 0; p < 3; p++)
                        #pragma unroll
                        for (int j = 0; j < 4; j++) {
                            int col = 2 * t4 + (j & 1);
                            uint32_t am = (j < 2) ? amask_lo : amask_hi;
                            s[p][j]     = ((am >> col) & 1) ? s[p][j] * rsc : TQ_AMM_NINF;
                            s[p][4 + j] = ((am >> (col + 8)) & 1) ? s[p][4 + j] * rsc : TQ_AMM_NINF;
                        }
                } else {
                    const int tp = t0 + cw * 16;
                    int p0 = tp + gr, p1 = tp + 8 + gr;
                    // STAGE: rows come from the smem stage slot (clamping was
                    // applied at copy time); direct path reads global.
                    const int r0i = cw * 16 + gr, r1i = r0i + 8;
                    const uint8_t *kr0 = STAGE
                        ? (const uint8_t *)(sm + TQ_AMG2_KST(slot) + r0i * 32)
                        : kcb4 + (size_t)min(p0, base) * cstride4;
                    const uint8_t *kr1 = STAGE
                        ? (const uint8_t *)(sm + TQ_AMG2_KST(slot) + r1i * 32)
                        : kcb4 + (size_t)min(p1, base) * cstride4;
                    const float *sz0 = STAGE
                        ? (const float *)(sm + TQ_AMG2_SST(slot) + r0i * 8)
                        : q4s_adv(ksz4, (size_t)min(p0, base) * szstride);
                    const float *sz1 = STAGE
                        ? (const float *)(sm + TQ_AMG2_SST(slot) + r1i * 8)
                        : q4s_adv(ksz4, (size_t)min(p1, base) * szstride);
                    #pragma unroll
                    for (int g = 0; g < 8; g++) {
                        float2 p0s = q4s_pair(sz0, 2 * g);
                        float2 p1s = q4s_pair(sz1, 2 * g);
                        #pragma unroll
                        for (int cc = 0; cc < 2; cc++) {
                            int c = 2 * g + cc;
                            uint64_t w0 = STAGE
                                ? *(const unsigned long long *)(kr0 + 8 * c)
                                : __ldg((const unsigned long long *)(kr0 + 8 * c));
                            uint64_t w1 = STAGE
                                ? *(const unsigned long long *)(kr1 + 8 * c)
                                : __ldg((const unsigned long long *)(kr1 + 8 * c));
                            uint8_t a0 = (uint8_t)(w0 >> (8 * t4));
                            uint8_t a8 = (uint8_t)(w0 >> (8 * t4 + 32));
                            uint8_t b0v = (uint8_t)(w1 >> (8 * t4));
                            uint8_t b8 = (uint8_t)(w1 >> (8 * t4 + 32));
                            uint32_t kb00 = tq_pack_bf16(((float)(a0 & 15) - p0s.y) * p0s.x,
                                                         ((float)(a0 >> 4) - p0s.y) * p0s.x);
                            uint32_t kb01 = tq_pack_bf16(((float)(a8 & 15) - p0s.y) * p0s.x,
                                                         ((float)(a8 >> 4) - p0s.y) * p0s.x);
                            uint32_t kb10 = tq_pack_bf16(((float)(b0v & 15) - p1s.y) * p1s.x,
                                                         ((float)(b0v >> 4) - p1s.y) * p1s.x);
                            uint32_t kb11 = tq_pack_bf16(((float)(b8 & 15) - p1s.y) * p1s.x,
                                                         ((float)(b8 >> 4) - p1s.y) * p1s.x);
                            #pragma unroll
                            for (int p = 0; p < 3; p++) {
                                const uint32_t *a = &sm[TQ_AMG2_QA(p) + (c * 32 + lane) * 4];
                                tq_mma_bf16_m16n8k16(&s[p][0], a, kb00, kb01);
                                tq_mma_bf16_m16n8k16(&s[p][4], a, kb10, kb11);
                            }
                        }
                    }
                    #pragma unroll
                    for (int p = 0; p < 3; p++)
                        #pragma unroll
                        for (int j = 0; j < 8; j++) s[p][j] *= rsc;
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        int c0 = tp + 2 * t4 + (j & 1);
                        #pragma unroll
                        for (int p = 0; p < 3; p++) {
                            if (c0 > base) s[p][j] = TQ_AMM_NINF;
                            if (c0 + 8 > base) s[p][4 + j] = TQ_AMM_NINF;
                        }
                    }
                }
            }
            for (int p = 0; p < 3; p++) {
                float *mst = (float *)(sm + TQ_AMG2_MST(p));
                float *lst = (float *)(sm + TQ_AMG2_LST(p));
                float *alph = (float *)(sm + TQ_AMG2_ALPH(slot, p));
                if (cw < nwt) {
                    float lm_lo = fmaxf(fmaxf(s[p][0], s[p][1]), fmaxf(s[p][4], s[p][5]));
                    float lm_hi = fmaxf(fmaxf(s[p][2], s[p][3]), fmaxf(s[p][6], s[p][7]));
                    #pragma unroll
                    for (int o = 1; o <= 2; o <<= 1) {
                        lm_lo = fmaxf(lm_lo, __shfl_xor_sync(0xffffffffu, lm_lo, o));
                        lm_hi = fmaxf(lm_hi, __shfl_xor_sync(0xffffffffu, lm_hi, o));
                    }
                    if (t4 == 0) { wmax[cw * 16 + gr] = lm_lo; wmax[cw * 16 + 8 + gr] = lm_hi; }
                }
                tq_bar_sync(1, 256);
                if (cw == 0 && lane < 16) {
                    float m_old = mst[lane], m_new = m_old;
                    for (int w = 0; w < nwt; w++) m_new = fmaxf(m_new, wmax[w * 16 + lane]);
                    alph[lane] = expf(m_old - m_new);
                    mst[lane] = m_new;
                }
                tq_bar_sync(1, 256);
                if (cw < nwt) {
                    float mn_lo = mst[gr], mn_hi = mst[8 + gr];
                    float p00 = expf(s[p][0] - mn_lo), p01 = expf(s[p][1] - mn_lo);
                    float p02 = expf(s[p][2] - mn_hi), p03 = expf(s[p][3] - mn_hi);
                    float p10 = expf(s[p][4] - mn_lo), p11 = expf(s[p][5] - mn_lo);
                    float p12 = expf(s[p][6] - mn_hi), p13 = expf(s[p][7] - mn_hi);
                    float ls_lo = p00 + p01 + p10 + p11;
                    float ls_hi = p02 + p03 + p12 + p13;
                    #pragma unroll
                    for (int o = 1; o <= 2; o <<= 1) {
                        ls_lo += __shfl_xor_sync(0xffffffffu, ls_lo, o);
                        ls_hi += __shfl_xor_sync(0xffffffffu, ls_hi, o);
                    }
                    if (t4 == 0) { wsum[cw * 16 + gr] = ls_lo; wsum[cw * 16 + 8 + gr] = ls_hi; }
                    if (!is_arch) {
                        int p0c = t0 + cw * 16 + 2 * t4, p1c = p0c + 8;
                        float v00 = vscb[(size_t)min(p0c, base) * nkv];
                        float v01 = vscb[(size_t)min(p0c + 1, base) * nkv];
                        float v10 = vscb[(size_t)min(p1c, base) * nkv];
                        float v11 = vscb[(size_t)min(p1c + 1, base) * nkv];
                        p00 *= v00; p01 *= v01; p02 *= v00; p03 *= v01;
                        p10 *= v10; p11 *= v11; p12 *= v10; p13 *= v11;
                    }
                    uint32_t *pst = &sm[TQ_AMG2_PST(slot, p) + (cw * 32 + lane) * 4];
                    pst[0] = tq_pack_bf16(p00, p01);
                    pst[1] = tq_pack_bf16(p02, p03);
                    pst[2] = tq_pack_bf16(p10, p11);
                    pst[3] = tq_pack_bf16(p12, p13);
                }
                tq_bar_sync(1, 256);
                if (cw == 0 && lane < 16) {
                    float l = lst[lane] * alph[lane];
                    for (int w = 0; w < nwt; w++) l += wsum[w * 16 + lane];
                    lst[lane] = l;
                }
                // wmax/wsum reused by the next pair: producers-only barrier
                tq_bar_sync(1, 256);
            }
        }
        tq_bar_sync(0, 512);                      // slot handoff (P + alpha visible)
    }
    // m/l partials from the producers' running state (threads 0..15; same
    // threads that wrote mst/lst, so no extra sync is needed).
    for (int p = 0; p < 3; p++) {
        float *mst = (float *)(sm + TQ_AMG2_MST(p));
        float *lst = (float *)(sm + TQ_AMG2_LST(p));
        int head_lo = 6 * kv_head + 2 * p, head_hi = head_lo + 1;
        float *ps = part + (size_t)(head_lo * nchunks + chunk) * TQ_AMM_PART_F;
        float *ph = part + (size_t)(head_hi * nchunks + chunk) * TQ_AMM_PART_F;
        if (tid < 8)       { ps[tid] = mst[tid]; ps[16 + tid] = lst[tid]; }
        else if (tid < 16) { ph[tid - 8] = mst[tid]; ph[16 + tid - 8] = lst[tid]; }
    }
    } else {
    float oacc[3][4][4];
    #pragma unroll
    for (int p = 0; p < 3; p++)
        #pragma unroll
        for (int f = 0; f < 4; f++)
            #pragma unroll
            for (int j = 0; j < 4; j++) oacc[p][f][j] = 0.0f;
    for (int ph = 0; ph <= T; ph++) {
        if (ph > 0) {
            const int it = ph - 1, slot = (ph - 1) & 1;
            const int is_arch = (it == nst);
            const int nwt = is_arch ? 1 : 8;
            const int t0 = (st_lo + it) << 7;
            #pragma unroll
            for (int p = 0; p < 3; p++) {
                const float *alph = (const float *)(sm + TQ_AMG2_ALPH(slot, p));
                float a_lo = alph[gr], a_hi = alph[8 + gr];
                #pragma unroll
                for (int f = 0; f < 4; f++) {
                    oacc[p][f][0] *= a_lo; oacc[p][f][1] *= a_lo;
                    oacc[p][f][2] *= a_hi; oacc[p][f][3] *= a_hi;
                }
            }
            for (int tt = 0; tt < nwt; tt++) {
                #pragma unroll
                for (int f = 0; f < 4; f++) {
                    int col = cw * 32 + f * 8 + gr;
                    uint32_t b0, b1;
                    if (is_arch) {
                        int r0 = 2 * t4;
                        b0 = tq_pack_bf16(vab[(size_t)r0 * astride + col],
                                          vab[(size_t)(r0 + 1) * astride + col]);
                        b1 = tq_pack_bf16(vab[(size_t)(r0 + 8) * astride + col],
                                          vab[(size_t)(r0 + 9) * astride + col]);
                    } else {
                        int r0 = t0 + tt * 16 + 2 * t4;
                        b0 = tq_pack_bf16(
                            tq_e4m3_dec_fast(vcb8[(size_t)min(r0, base) * (size_t)(nkv * hd) + col]),
                            tq_e4m3_dec_fast(vcb8[(size_t)min(r0 + 1, base) * (size_t)(nkv * hd) + col]));
                        b1 = tq_pack_bf16(
                            tq_e4m3_dec_fast(vcb8[(size_t)min(r0 + 8, base) * (size_t)(nkv * hd) + col]),
                            tq_e4m3_dec_fast(vcb8[(size_t)min(r0 + 9, base) * (size_t)(nkv * hd) + col]));
                    }
                    #pragma unroll
                    for (int p = 0; p < 3; p++) {
                        const uint32_t *pa = &sm[TQ_AMG2_PST(slot, p) + (tt * 32 + lane) * 4];
                        uint32_t a[4] = {pa[0], pa[1], pa[2], pa[3]};
                        tq_mma_bf16_m16n8k16(oacc[p][f], a, b0, b1);
                    }
                }
            }
        }
        tq_bar_sync(0, 512);                      // slot handoff (P + alpha visible)
    }
    // O-column partials (full 256-col window per pair); same slab layout and
    // values as the 2-half kernel.
    for (int p = 0; p < 3; p++) {
        int head_lo = 6 * kv_head + 2 * p, head_hi = head_lo + 1;
        float *ps = part + (size_t)(head_lo * nchunks + chunk) * TQ_AMM_PART_F;
        float *ph = part + (size_t)(head_hi * nchunks + chunk) * TQ_AMM_PART_F;
        float *po = ps + 32, *qo = ph + 32;
        #pragma unroll
        for (int f = 0; f < 4; f++) {
            int col = cw * 32 + f * 8 + 2 * t4;
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                po[(size_t)gr * 256 + col + j] = oacc[p][f][j];
                qo[(size_t)gr * 256 + col + j] = oacc[p][f][2 + j];
            }
        }
    }
    }
}

// Merge phase: flash-decoding combine of one head's chunk partials (m* = max m_c;
// l*/O* = sum exp(m_c - m*) {l_c, O_c}) + the same epilogue as the single-chunk
// path. Block = 256 threads (one output column each), rows looped; sm9 = 9-float
// scratch for the per-row absmax fold (persist only).
static __device__ void tq_attn_mma_merge_block(
    const float *q_proj_base, size_t q_stride, int head, int r_lo, int r_hi, int hd,
    int nchunks, const float *scratch,
    float *out, size_t out_stride, float *absmax, float *sm9) {
    int tid = threadIdx.x;
    const float *ps0 = scratch + (size_t)head * nchunks * TQ_AMM_PART_F;
    for (int r = r_lo; r < r_hi; r++) {
        float m_star = TQ_AMM_NINF;
        for (int c = 0; c < nchunks; c++)
            m_star = fmaxf(m_star, ps0[(size_t)c * TQ_AMM_PART_F + r]);
        float l = 0.0f, o = 0.0f;
        for (int c = 0; c < nchunks; c++) {
            const float *ps = ps0 + (size_t)c * TQ_AMM_PART_F;
            float w = expf(ps[r] - m_star);
            l += w * ps[16 + r];
            o += w * ps[32 + (size_t)r * 256 + tid];
        }
        float gate = q_proj_base[(size_t)r * q_stride + head * (2 * hd) + hd + tid];
        float v = (o / l) * (1.0f / (1.0f + expf(-gate)));
        out[(size_t)r * out_stride + head * hd + tid] = v;
        if (absmax) {
            float am = fabsf(v);
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                am = fmaxf(am, __shfl_down_sync(0xffffffffu, am, off));
            if ((tid & 31) == 0) sm9[tid >> 5] = am;
            __syncthreads();
            if (tid == 0) {
                float m = sm9[0];
                for (int w8 = 1; w8 < 8; w8++) m = fmaxf(m, sm9[w8]);
                atomicMax((unsigned int *)&absmax[r], __float_as_uint(m));
            }
            __syncthreads();
        }
    }
}

// Standalone replacement for k_tq_spec_attn_tree under TQ_ATTN_MMA: grid =
// (nh or nh/2 with `pair`, nchunks), all N nodes handled per block (launch with
// TQ_ATTN_MMA_SMEM dynamic smem). gridDim.y == 1 writes the output directly;
// > 1 writes chunk partials into `scratch` for the merge kernel below.
template <int MODE>
__global__ void k_tq_spec_attn_tree_mma(float *out, const float *q_proj_base,
                                        const uint16_t *q_norm_w,
                                        const float *k_cache, const float *v_cache,
                                        const uint8_t *k_cache8, const float *k_scale,
                                        const uint8_t *v_cache8, const float *v_scale,
                                        const uint8_t *k_cache4, const float *k_q4s,
                                        const float *k_arch, const float *v_arch,
                                        const int *pos_arr, const int *anc, int N,
                                        int pair,
                                        int nh, int nkv, int hd, int kv_m,
                                        int q_stride, int out_stride,
                                        float eps, float rope_theta, int base_in,
                                        float *scratch) {
    extern __shared__ uint32_t tq_dyn_sm[];
    int nchunks = gridDim.y;
    tq_attn_mma_block<MODE>(q_proj_base, q_stride, q_norm_w,
                            k_cache, v_cache, k_cache8, k_scale, v_cache8, v_scale,
                            k_cache4, k_q4s,
                            k_arch, v_arch, pos_arr, anc, base_in, N,
                            pair ? 2 * blockIdx.x : blockIdx.x, pair,
                            nh, nkv, hd, kv_m, eps, rope_theta,
                            blockIdx.y, nchunks,
                            out, out_stride, NULL, nchunks > 1 ? scratch : NULL,
                            tq_dyn_sm);
}

// grid (nh, N): one (head, row) per block -- at the >=96k gate the group path
// runs ~96 chunks and a 24-block/8-serial-row merge was 2.6 ms/round (7.6% of
// the round, 13% GPU occupancy); per-row blocks make it ~8x wider.
__global__ void k_tq_spec_attn_tree_mma_merge(float *out, const float *q_proj_base,
                                              int N, int nh, int hd,
                                              int q_stride, int out_stride,
                                              int nchunks, const float *scratch) {
    int r = blockIdx.y;
    if (r >= N) return;
    tq_attn_mma_merge_block(q_proj_base, (size_t)q_stride, blockIdx.x, r, r + 1, hd,
                            nchunks, scratch, out, (size_t)out_stride, NULL, NULL);
}

static int ensure_attn_scores(void);   // defined with the other host helpers below
static int paged_sm_count(void);       // cached multiProcessorCount (defined below)

static int attn_mma_chunk_tokens(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("TQ_ATTN_MMA_CHUNK");
        // 1024 swept best @10k (219us/layer vs 385 @2048, 277 @512): ~240 items
        // fill the 340-CTA grid without per-item Q-prep overhead taking over
        cached = e ? atoi(e) : 1024;
        if (cached < 256) cached = 256;
    }
    return cached;
}

// Chunk-count cap for the flash-decoding split below (TQ_ATTN_MMA_NCHUNKS to
// A/B). MEASURED (RTX PRO 6000, prod Q4): unpaired items want the old
// cap 16 (384 items ~= one wave of the ~376-CTA grid; 32 chunks = 28.5 vs 26.7
// ms/round @32k -- extra per-item Q-prep, no balance win). PAIRED items halve
// the head count, so cap 16 leaves half the grid idle: with cap 32 the wave is
// 12 heads x 32 = 384 items again -- @64k 32.8 -> 27.3 ms/round (+26% tok/s).
// 64 loses to 32 (2 items/CTA + doubled Q-prep): target ~1 item per CTA.
static int attn_mma_nchunks_cap(int pair) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("TQ_ATTN_MMA_NCHUNKS");
        cached = e ? atoi(e) : 0;                    // 0 = auto (16 / 32 with pair)
        if (cached < 0) cached = 0;
        if (cached > 256) cached = 256;
    }
    if (cached > 0) return cached;
    return pair ? 32 : 16;
}

// number of position chunks for a verify/wave round at `base`: ~1 chunk per
// TQ_ATTN_MMA_CHUNK positions, capped by attn_mma_nchunks_cap() and the
// partial-slab budget (nh x nchunks x TQ_AMM_PART_F floats inside the scores
// scratch). The kernel splits the prefix super-tiles evenly across nchunks.
static int attn_mma_nchunks(int base, int pair) {
    int ch = attn_mma_chunk_tokens();
    int n = base >= 0 ? (base + ch) / ch : 1;
    if (n < 1) n = 1;
    int cap = attn_mma_nchunks_cap(pair);
    int max_seq = g_qwen.max_seq > 0 ? g_qwen.max_seq : 8192;
    int slab_cap = (int)(((size_t)TQ_ATTN_SCORE_SLOTS * max_seq) /
                         ((size_t)g_qwen.nh * TQ_AMM_PART_F));
    if (slab_cap < 1) slab_cap = 1;
    if (cap > slab_cap) cap = slab_cap;
    if (n > cap) n = cap;
    return n;
}

// Two q-heads of one kv group per item (same K/V bytes read once, both m16 tile
// halves live) -- see tq_attn_mma_block PAIR mode. Requires N <= 8 (production
// tree budget) and an even GQA group so (2b, 2b+1) share a kv head. Per-row
// numerics are bit-identical to unpaired items; TQ_ATTN_MMA_PAIR=0 disables.
static int attn_mma_pair_ok(int N) {
    static int en = -1;
    if (en < 0) {
        const char *e = getenv("TQ_ATTN_MMA_PAIR");
        en = e ? (atoi(e) != 0) : 1;
    }
    return en && N <= 8 && (g_qwen.nh & 1) == 0 && ((g_qwen.nh / g_qwen.nkv) & 1) == 0;
}

// GROUP mode (whole kv group per item, staged Q) for the standalone launcher.
// MODE 3 only, needs the exact production GQA shape. HISTORY: the 2-half group
// kernel beat the pair path only past ~196k on the 188-SM PRO 6000 and from
// ~32k on the 170-SM 5090 (sweeps; SM-aware thresholds). With the
// GROUP2 producer/consumer kernel (default) the group mode wins
// from 32k up on BOTH dies -- 5090: 32k 23.8 vs 24.2, 128k 26.5 vs 33.9,
// 250k 30.8 vs 40.2; PRO 6000: 32k 22.7 vs 23.8, 245k 28.0 vs 34.7 ms/round
// -- so the auto threshold is a flat 32768. TQ_ATTN_MMA_GROUP=1/0 forces
// always/never; TQ_ATTN_MMA_GROUP_MIN moves it.
static int attn_mma_group_ok(int N, int mode, int host_base) {
    static int en = -2, gmin = -1;
    if (en == -2) {
        const char *e = getenv("TQ_ATTN_MMA_GROUP");
        en = (e && *e) ? (atoi(e) != 0 ? 1 : 0) : -1;           // -1 = auto
        const char *m = getenv("TQ_ATTN_MMA_GROUP_MIN");
        // 8192 with GROUP2 (5090 sweep: 8k 22.3 -> 21.4, 16k 22.5
        // -> 21.7 ms/round free-run; teacher-forced verify -0.8 ms/round @8k)
        gmin = (m && *m) ? atoi(m) : 8192;
    }
    if (en == 0) return 0;
    if (en == -1 && host_base < gmin) return 0;
    return mode == 3 && N <= 8 && g_qwen.hd == 256 &&
           g_qwen.nh / g_qwen.nkv == 6 && g_qwen.nh == 6 * g_qwen.nkv;
}

static uint32_t *g_attn_qa_stage = NULL;   // [nh/2 pairs x 2048] staged Q A-frags
static int ensure_attn_qa_stage(void) {
    if (g_attn_qa_stage) return 0;
    size_t words = (size_t)(g_qwen.nh / 2) * 2048;
    return cudaMalloc(&g_attn_qa_stage, words * 4) == cudaSuccess ? 0 : -1;
}

// GROUP2 (producer/consumer single-pass kernel above): DEFAULT ON (A/B, 200
// rounds each, ms/round vs the 2-half group kernel):
//   5090 / 170 SM: 32k 23.8->22.8, 64k 26.4->23.9, 128k 31.4->26.5,
//                  200k 36.7->29.1 (->27.2 with the one-wave nc3), 250k 40.2->30.8
//   PRO 6000 / 188 SM: 128k 28.8->25.5, 200k 33.3->27.9, 245k 34.7->28.0
// Streams bit-exact vs the group kernel at matched nc3 (@128k stream_dump
// diff = 0). TQ_ATTN_MMA_GROUP2=0 reverts to the 2-half kernel; the 51 KB
// dynamic smem needs a one-time opt-in.
static int attn_mma_group2_enabled(void) {
    static int en = -1;
    if (en < 0) {
        const char *e = getenv("TQ_ATTN_MMA_GROUP2");
        en = (e && *e) ? (atoi(e) != 0 ? 1 : 0) : 1;
    }
    if (!en) return 0;
    static int smem_ok = -1;
    if (smem_ok < 0)
        smem_ok = cudaFuncSetAttribute(k_tq_spec_attn_group_mma2<0>,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       TQ_ATTN_MMA_GROUP2_SMEM) == cudaSuccess;
    return smem_ok;
}

// STAGE extension gate (TQ_ATTN_MMA_GROUP2_STAGE, default OFF -- measured
// NEUTRAL on the 5090: 64k 23.15 vs 23.13, 128k 25.17 vs 25.11,
// 200k 27.27 vs 27.27 ms/round; streams bit-exact). After the producer/consumer
// split the producers' K-load stalls already hide behind consumer P.V work; the
// remaining latency sits in the consumers' V byte loads (8 useful bytes per
// 32 B sector, inherent to the position-major V layout), which the K staging
// cannot touch. Kept as an opt-in in case a V-layout change revisits this.
// cp.async K/scale staging; fp16 scales only (the fp32 escape hatch falls back).
static int attn_mma_group2_stage_enabled(void) {
    static int en = -1;
    if (en < 0) {
        const char *e = getenv("TQ_ATTN_MMA_GROUP2_STAGE");
        en = (e && *e) ? (atoi(e) != 0 ? 1 : 0) : 0;
    }
    if (!en || g_qwen.kv_q4_fp32s) return 0;
    static int smem_ok = -1;
    if (smem_ok < 0)
        smem_ok = cudaFuncSetAttribute(k_tq_spec_attn_group_mma2<1>,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       TQ_ATTN_MMA_GROUP2_SMEM_STG) == cudaSuccess;
    return smem_ok;
}

static int launch_attn_tree_mma(float *out, const float *q_proj_base,
                                const uint16_t *q_norm_w,
                                const float *k_cache, const float *v_cache,
                                const uint8_t *k_cache8, const float *k_scale,
                                const uint8_t *v_cache8, const float *v_scale,
                                const uint8_t *k_cache4, const float *k_q4s,
                                const float *k_arch, const float *v_arch,
                                const int *pos_arr, const int *anc, int N,
                                int q_stride, int out_stride, int base_in,
                                int host_base, int mode) {
    if (ensure_attn_scores() != 0) return -97;       // chunk-partial scratch
    int pair = attn_mma_pair_ok(N);
    int nchunks = attn_mma_nchunks(host_base, pair);
    // GROUP path: one item per (kv head, chunk) with staged Q; 3x the chunk
    // count keeps the grid at the pair path's ~384 blocks (4 kv heads vs 12
    // pair-heads). Only meaningful with the partial+merge path (nchunks > 1).
    if (attn_mma_group_ok(N, mode, host_base) && nchunks * 3 > 1 && ensure_attn_qa_stage() == 0) {
        int nc3 = nchunks * 3;
        if (nc3 > 128) nc3 = 128;
        k_tq_attn_qa_stage<<<g_qwen.nh / 2, 256, 0, g_qwen.stream>>>(
            g_attn_qa_stage, q_proj_base, q_norm_w, pos_arr, N, g_qwen.nh, g_qwen.hd,
            q_stride, g_qwen.eps, g_qwen.rope_theta);
        if (attn_mma_group2_enabled()) {
            // 512-thread CTAs run at 1 CTA/SM, so the sweet spot is ONE full
            // wave: nkv * nc3 == SM count (200k sweep on 170 SM: nc3 42 -> 27.2
            // ms/round vs 48 -> 30.9 (1.13-wave tail) vs 24 -> 30.6 (half the
            // GPU idle)), clamped to the super-tile count so short contexts do
            // not launch empty chunks. TQ_ATTN_MMA_NCHUNKS overrides via
            // nchunks*3.
            if (!getenv("TQ_ATTN_MMA_NCHUNKS")) {
                nc3 = paged_sm_count() / g_qwen.nkv;
                int nst_all = host_base >= 0 ? (host_base + 128) >> 7 : 1;
                if (nc3 > nst_all) nc3 = nst_all;
                if (nc3 < 2) nc3 = 2;
                if (nc3 > 128) nc3 = 128;
                int slab_cap3 = (int)(((size_t)TQ_ATTN_SCORE_SLOTS * g_qwen.max_seq) /
                                      ((size_t)g_qwen.nh * TQ_AMM_PART_F));
                if (nc3 > slab_cap3) nc3 = slab_cap3 > 0 ? slab_cap3 : 1;
            }
            if (attn_mma_group2_stage_enabled())
                k_tq_spec_attn_group_mma2<1><<<dim3(g_qwen.nkv, nc3), 512,
                                               TQ_ATTN_MMA_GROUP2_SMEM_STG, g_qwen.stream>>>(
                    g_attn_qa_stage, k_cache4, k_q4s, v_cache8, v_scale, k_arch, v_arch,
                    pos_arr, anc, N, g_qwen.nh, g_qwen.nkv, g_qwen.hd, g_qwen.nkv * g_qwen.hd,
                    g_qwen.d_attn_scores);
            else
                k_tq_spec_attn_group_mma2<0><<<dim3(g_qwen.nkv, nc3), 512,
                                               TQ_ATTN_MMA_GROUP2_SMEM, g_qwen.stream>>>(
                    g_attn_qa_stage, k_cache4, k_q4s, v_cache8, v_scale, k_arch, v_arch,
                    pos_arr, anc, N, g_qwen.nh, g_qwen.nkv, g_qwen.hd, g_qwen.nkv * g_qwen.hd,
                    g_qwen.d_attn_scores);
        } else
            k_tq_spec_attn_group_mma<<<dim3(g_qwen.nkv, nc3), 256, TQ_ATTN_MMA_GROUP_SMEM,
                                       g_qwen.stream>>>(
                g_attn_qa_stage, k_cache4, k_q4s, v_cache8, v_scale, k_arch, v_arch,
                pos_arr, anc, N, g_qwen.nh, g_qwen.nkv, g_qwen.hd, g_qwen.nkv * g_qwen.hd,
                g_qwen.d_attn_scores);
        k_tq_spec_attn_tree_mma_merge<<<dim3(g_qwen.nh, N), 256, 0, g_qwen.stream>>>(
            out, q_proj_base, N, g_qwen.nh, g_qwen.hd, q_stride, out_stride,
            nc3, g_qwen.d_attn_scores);
        return 0;
    }
    dim3 grid(pair ? g_qwen.nh / 2 : g_qwen.nh, nchunks);
    if (mode == 3)
        k_tq_spec_attn_tree_mma<3><<<grid, 256, TQ_ATTN_MMA_SMEM, g_qwen.stream>>>(
            out, q_proj_base, q_norm_w, k_cache, v_cache, k_cache8, k_scale, v_cache8,
            v_scale, k_cache4, k_q4s, k_arch, v_arch, pos_arr, anc, N, pair, g_qwen.nh,
            g_qwen.nkv, g_qwen.hd,
            g_qwen.nkv * g_qwen.hd, q_stride, out_stride, g_qwen.eps, g_qwen.rope_theta,
            base_in, g_qwen.d_attn_scores);
    else if (mode == 2)
        k_tq_spec_attn_tree_mma<2><<<grid, 256, TQ_ATTN_MMA_SMEM, g_qwen.stream>>>(
            out, q_proj_base, q_norm_w, k_cache, v_cache, k_cache8, k_scale, v_cache8,
            v_scale, k_cache4, k_q4s, k_arch, v_arch, pos_arr, anc, N, pair, g_qwen.nh,
            g_qwen.nkv, g_qwen.hd,
            g_qwen.nkv * g_qwen.hd, q_stride, out_stride, g_qwen.eps, g_qwen.rope_theta,
            base_in, g_qwen.d_attn_scores);
    else
        k_tq_spec_attn_tree_mma<1><<<grid, 256, TQ_ATTN_MMA_SMEM, g_qwen.stream>>>(
            out, q_proj_base, q_norm_w, k_cache, v_cache, k_cache8, k_scale, v_cache8,
            v_scale, k_cache4, k_q4s, k_arch, v_arch, pos_arr, anc, N, pair, g_qwen.nh,
            g_qwen.nkv, g_qwen.hd,
            g_qwen.nkv * g_qwen.hd, q_stride, out_stride, g_qwen.eps, g_qwen.rope_theta,
            base_in, g_qwen.d_attn_scores);
    if (nchunks > 1)
        k_tq_spec_attn_tree_mma_merge<<<dim3(g_qwen.nh, N), 256, 0, g_qwen.stream>>>(
            out, q_proj_base, N, g_qwen.nh, g_qwen.hd, q_stride, out_stride,
            nchunks, g_qwen.d_attn_scores);
    return 0;
}

__global__ void k_tq_full_attn_decode_arr(float *out, const float *q_proj, const float *k_proj,
                                           const float *v_proj, const uint16_t *q_norm_w,
                                           const uint16_t *k_norm_w, float *k_cache,
                                           float *v_cache,
                                           uint8_t *k_cache8, float *k_scale,
                                           uint8_t *v_cache8, float *v_scale,
                                           uint8_t *k_cache4, float *k_q4s,
                                           const int *pos_arr, int node_idx,
                                           int nh, int nkv, int hd, float eps, float rope_theta,
                                           float *scores_g, int sstride) {
    int pos = pos_arr[node_idx];
    __shared__ float partial[8];
    __shared__ float q_sum_shared;
    __shared__ float k_sum_shared;
    __shared__ float gsz[16];
    float *scores = scores_g + (size_t)(blockIdx.x) * sstride;
    __shared__ float max_score_shared;
    __shared__ float denom_shared;
    __shared__ float qn[256];
    int head = blockIdx.x;
    int tid = threadIdx.x;
    if (head >= nh || tid >= hd) return;
    int group = nh / nkv;
    int kv_head = head / group;
    int rotary_dim = 64;
    float qv = q_proj[head * (2 * hd) + tid];
    float kv = k_proj[kv_head * hd + tid];
    float qsum = qv * qv;
    float ksum = kv * kv;
    for (int offset = 16; offset > 0; offset >>= 1) {
        qsum += __shfl_down_sync(0xffffffff, qsum, offset);
        ksum += __shfl_down_sync(0xffffffff, ksum, offset);
    }
    if ((tid & 31) == 0) partial[tid >> 5] = qsum;
    __syncthreads();
    if (tid < 8) {
        float vtmp = partial[tid];
        for (int offset = 4; offset > 0; offset >>= 1) vtmp += __shfl_down_sync(0xff, vtmp, offset);
        if (tid == 0) q_sum_shared = vtmp;
    }
    __syncthreads();
    if ((tid & 31) == 0) partial[tid >> 5] = ksum;
    __syncthreads();
    if (tid < 8) {
        float vtmp = partial[tid];
        for (int offset = 4; offset > 0; offset >>= 1) vtmp += __shfl_down_sync(0xff, vtmp, offset);
        if (tid == 0) k_sum_shared = vtmp;
    }
    __syncthreads();
    qv = qv * rsqrtf(q_sum_shared / (float)hd + eps) * (1.0f + tq_bf16_to_float(q_norm_w[tid]));
    kv = kv * rsqrtf(k_sum_shared / (float)hd + eps) * (1.0f + tq_bf16_to_float(k_norm_w[tid]));
    if (tid < rotary_dim) {
        int idx = tid & 31;
        float freq = powf(rope_theta, -((float)(2 * idx) / (float)rotary_dim));
        float angle = (float)pos * freq;
        float c = cosf(angle);
        float s = sinf(angle);
        float q_pair = q_proj[head * (2 * hd) + ((tid < 32) ? tid + 32 : tid - 32)];
        float k_pair = k_proj[kv_head * hd + ((tid < 32) ? tid + 32 : tid - 32)];
        q_pair = q_pair * rsqrtf(q_sum_shared / (float)hd + eps) * (1.0f + tq_bf16_to_float(q_norm_w[(tid < 32) ? tid + 32 : tid - 32]));
        k_pair = k_pair * rsqrtf(k_sum_shared / (float)hd + eps) * (1.0f + tq_bf16_to_float(k_norm_w[(tid < 32) ? tid + 32 : tid - 32]));
        float q_rot = (tid < 32) ? -q_pair : q_pair;
        float k_rot = (tid < 32) ? -k_pair : k_pair;
        qv = qv * c + q_rot * s;
        kv = kv * c + k_rot * s;
    }
    qn[tid] = qv;
    if (k_cache4) {
        // int4-K cache write (TQ_KV_Q4): V row keeps the E4M3 path; the K row is
        // FWHT-rotated (borrowing qn as the transform buffer) then group-quantized.
        float vv = v_proj[kv_head * hd + tid];
        __syncthreads();
        float vscl = tq_kv_block_scale(vv, partial, &k_sum_shared);
        v_cache8[((size_t)pos * nkv + kv_head) * hd + tid] = tq_float_to_e4m3(vv / vscl);
        if (tid == 0) v_scale[(size_t)pos * nkv + kv_head] = vscl;
        qn[tid] = kv;
        __syncthreads();
        tq_fwht256(qn, tid);
        float krot = qn[tid];
        __syncthreads();
        uint8_t code = tq_k_q4_quant(krot, tid, gsz);
        unsigned hi4 = __shfl_down_sync(0xffffffffu, (unsigned)code, 1);
        if ((tid & 1) == 0)
            k_cache4[((size_t)pos * nkv + kv_head) * (hd >> 1) + (tid >> 1)] =
                (uint8_t)(code | ((hi4 & 15u) << 4));
        if (tid < 16)
            q4s_put(k_q4s, ((size_t)pos * nkv + kv_head) * 16 + tid, gsz[tid]);
        __syncthreads();
        qn[tid] = qv;                  // tail q lives in the rotated space
        __syncthreads();
        tq_fwht256(qn, tid);
    } else if (k_cache8) {
        // FP8 cache write: block absmax -> pow2 scale -> E4M3 codes. All nh blocks
        // of one kv_head compute identical scales/codes (redundant, benign -- same
        // as the redundant fp32 row writes on the else path).
        float vv = v_proj[kv_head * hd + tid];
        __syncthreads();
        float kscl = tq_kv_block_scale(kv, partial, &k_sum_shared);
        k_cache8[((size_t)pos * nkv + kv_head) * hd + tid] = tq_float_to_e4m3(kv / kscl);
        __syncthreads();
        float vscl = tq_kv_block_scale(vv, partial, &k_sum_shared);
        v_cache8[((size_t)pos * nkv + kv_head) * hd + tid] = tq_float_to_e4m3(vv / vscl);
        if (tid == 0) {
            k_scale[(size_t)pos * nkv + kv_head] = kscl;
            v_scale[(size_t)pos * nkv + kv_head] = vscl;
        }
    } else {
        k_cache[((size_t)pos * nkv + kv_head) * hd + tid] = kv;
        v_cache[((size_t)pos * nkv + kv_head) * hd + tid] = v_proj[kv_head * hd + tid];
    }
    __syncthreads();
    float ratio;
    if (hd == 256) {
        ratio = tq_attn_tail_pp(qn, scores, qn,
                                k_cache + (size_t)kv_head * hd, v_cache + (size_t)kv_head * hd,
                                (size_t)nkv * hd,
                                k_cache8 ? k_cache8 + (size_t)kv_head * hd : NULL,
                                k_scale + kv_head,
                                v_cache8 ? v_cache8 + (size_t)kv_head * hd : NULL,
                                v_scale + kv_head, nkv,
                                k_cache4 ? k_cache4 + (size_t)kv_head * (hd >> 1) : NULL,
                                k_cache4 ? q4s_adv(k_q4s, (size_t)kv_head * 16) : NULL,
                                NULL, NULL, 0, NULL, pos, pos, hd, tid);
    } else {
        float max_score = -3.402823466e38f;
        for (int t = 0; t <= pos; t++) {
            float dot = qn[tid] * k_cache[((size_t)t * nkv + kv_head) * hd + tid];
            for (int offset = 16; offset > 0; offset >>= 1) dot += __shfl_down_sync(0xffffffff, dot, offset);
            if ((tid & 31) == 0) partial[tid >> 5] = dot;
            __syncthreads();
            if (tid < 8) {
                float vtmp = partial[tid];
                for (int offset = 4; offset > 0; offset >>= 1) vtmp += __shfl_down_sync(0xff, vtmp, offset);
                if (tid == 0) {
                    float score = vtmp * rsqrtf((float)hd);
                    scores[t] = score;
                    if (t == 0 || score > max_score_shared) max_score_shared = score;
                }
            }
            __syncthreads();
        }
        max_score = max_score_shared;
        if (tid == 0) {
            float denom = 0.0f;
            for (int t = 0; t <= pos; t++) denom += expf(scores[t] - max_score);
            denom_shared = denom;
        }
        __syncthreads();
        float acc = 0.0f;
        for (int t = 0; t <= pos; t++) {
            float p = expf(scores[t] - max_score) / denom_shared;
            acc += p * v_cache[((size_t)t * nkv + kv_head) * hd + tid];
        }
        ratio = acc;
    }
    float gate = q_proj[head * (2 * hd) + hd + tid];
    out[head * hd + tid] = ratio * (1.0f / (1.0f + expf(-gate)));
}
