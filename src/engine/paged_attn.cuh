// paged decode / verify attention kernels (GQA MMA, staged V, segment-aware, fp16, trellis twins).
// Part of the forward_qwen.cu unity build: included in order, not a standalone header.

// ===== GQA-aware tensor-core paged DECODE attention (Phase 0.5) =====
// The scalar decode kernels above spend ~160 warp-instructions per (q-head, key) and read every
// K/V row once per q-head (6x per kv group); measured 0.165 us per context token per client per
// step = ~10x the HBM bound from 32K to 128K. Here one block = (kv group, client column, key
// split): the group's q-heads (nh/nkv = 6) are the rows of ONE m16 tile (rows >= group inert),
// so each K/V row is loaded and dequantized once for all heads, and Q.K^T / P.V run on HMMA --
// the k_tq_wide_attn_mma<4,1> floor plan with rows = heads instead of positions. Grid (nkv, N,
// S); S > 1 writes the same (m, l, acc) partials as k_tq_paged_attn_q4_split, so the existing
// k_tq_paged_attn_q4_merge combines them. page must be a multiple of 128 (a 128-key super-tile
// lies in one physical block). Enabled by TQ_PAGED_ATTN_MMA (see paged_attn_gqa_mma_enabled).
// USE_TOPK=0 is the dense walk. USE_TOPK=1 consumes a per-(client, kv head) page list
// (Quest top-k; count 0 means this client stays dense). Separate instantiations so the dense
// kernel's register count does not include the list path.
// VQ4 = 1: int4 V (TQ_KV_V4); O accumulates in the rotated space, rotated back once per q-head
// in the epilogue (S == 1) or in k_tq_paged_attn_q4_merge<1> (S > 1).
// T4 = 1 (TQ_KV_T4, requires VQ4 = 1): the pool blocks hold trellis codes (k4_pool / v8_pool = the code bytes,
// t4p.kside / vside the side words); rows < the column's position are trellis, the column's own key (row pos) is
// read exact from the fp16 staging; the fp16 dequant table is staged in smem once per block.
template <int USE_TOPK, int VQ4, int T4 = 0>
__global__ void __launch_bounds__(256) k_tq_paged_attn_q4_gqa_mma(
    float *out, const float *q_proj_base, const uint16_t *q_norm_w,
    const uint8_t *k4_pool, const float *kq4s_pool, const uint8_t *v8_pool, const float *vscale_pool,
    const int *positions, const int *slot_ids,
    const int *block_table, int max_blocks, int page, int page_log,
    int nh, int nkv, int hd, int q_m, int attn_m, int S,
    float eps, float rope_theta, float *part_acc, float *part_ml,
    const int *topk_list, const int *topk_count, int topk_cap, const int *col_S,
    const tq_t4_pool t4p = tq_t4_pool()) {
    tq_pdl_enter();
    __shared__ uint32_t sm[TQ_AMM_WORDS];
    __shared__ __align__(16) uint16_t t4tab[T4 ? 4096 : 8];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int gr = lane >> 2, t4 = lane & 3;
    const int kv_head = blockIdx.x, col = blockIdx.y, split = blockIdx.z;
    // col_S (verify waves): this column's own key-split count Sc <= S (the grid's); the partial layout keeps the
    // grid stride S. NULL = every column uses S (the decode step; unchanged arithmetic).
    const int Sc = col_S ? col_S[col] : S;
    if (split >= Sc) return;
    const int group = nh / nkv;                                 // q-heads per kv head = tile rows
    const int pos = positions[col], slot = slot_ids[col];
    const size_t cstride = (size_t)nkv * hd, cstride4 = cstride >> 1, szstride = (size_t)nkv * 16;
    const uint8_t *kcb4 = k4_pool + (size_t)kv_head * (hd >> 1);
    const float *ksz4 = q4s_adv(kq4s_pool, (size_t)kv_head * 16);
    const uint8_t *vcb8 = v8_pool + (size_t)kv_head * (VQ4 ? (hd >> 1) : hd);
    const float *vscb = VQ4 ? q4s_adv(vscale_pool, (size_t)kv_head * 16) : vscale_pool + kv_head;
    float *qtmp = (float *)(sm + TQ_AMM_QTMP);
    float *red = (float *)(sm + TQ_AMM_RED);
    float *wmax = (float *)(sm + TQ_AMM_WMAX), *wsum = (float *)(sm + TQ_AMM_WSUM);
    float *mst = (float *)(sm + TQ_AMM_MST), *lst = (float *)(sm + TQ_AMM_LST);
    float *alph = (float *)(sm + TQ_AMM_ALPH);
    for (int i = tid; i < TQ_AMM_PST + 1024; i += 256) sm[i] = 0;
    if (tid < 16) { mst[tid] = TQ_AMM_NINF; lst[tid] = 0.0f; }
    if constexpr (T4) tq_t4_load_table(t4p.table, t4tab, tid);
    __syncthreads();
    // ---- Q prep: row r = q-head kv_head*group + r at this client's position (norm + rope +
    // FWHT into the rotated int4-K space), packed into m16k16 A-fragments; rows >= group stay 0.
    const float *q_proj = q_proj_base + (size_t)col * q_m;
    for (int r = 0; r < group; r++) {
        const int hh = kv_head * group + r;
        float qv = q_proj[hh * (2 * hd) + tid];
        float qsum = qv * qv;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) qsum += __shfl_down_sync(0xffffffffu, qsum, o);
        if (lane == 0) red[warp] = qsum;
        __syncthreads();
        if (tid < 8) { float v = red[tid]; for (int o = 4; o > 0; o >>= 1) v += __shfl_down_sync(0xffu, v, o); if (tid == 0) red[8] = v; }
        __syncthreads();
        float qss = red[8];
        qv = qv * rsqrtf(qss / (float)hd + eps) * (1.0f + tq_bf16_to_float(q_norm_w[tid]));
        if (tid < 64) {
            int idx = tid & 31;
            float freq = powf(rope_theta, -((float)(2 * idx) / 64.0f));
            float angle = (float)pos * freq, c = cosf(angle), s = sinf(angle);
            float q_pair = q_proj[hh * (2 * hd) + ((tid < 32) ? tid + 32 : tid - 32)];
            q_pair = q_pair * rsqrtf(qss / (float)hd + eps) * (1.0f + tq_bf16_to_float(q_norm_w[(tid < 32) ? tid + 32 : tid - 32]));
            float q_rot = (tid < 32) ? -q_pair : q_pair;
            qv = qv * c + q_rot * s;
        }
        __syncthreads();
        qtmp[tid] = qv;
        __syncthreads();
        tq_fwht256(qtmp, tid);
        if (tid < 128) {
            int c16 = tid >> 3, tt = (tid >> 1) & 3, rsel = tid & 1;
            int d0 = 16 * c16 + 2 * tt + 8 * rsel;
            int l = 4 * (r & 7) + tt, rr = ((r >> 3) & 1) + 2 * rsel;
            sm[TQ_AMM_QA + (c16 * 32 + l) * 4 + rr] = tq_pack_bf16(qtmp[d0], qtmp[d0 + 1]);
        }
        __syncthreads();
    }
    // ---- main loop: this split's slice of the 128-key super-tiles over [0..pos] ----
    // USE_TOPK=0 keeps only the dense walk (compiled out of the list branch). USE_TOPK=1
    // walks the ranked page list when this client has one, else the same dense walk.
    float oacc[4][4];
    #pragma unroll
    for (int f = 0; f < 4; f++) for (int j = 0; j < 4; j++) oacc[f][j] = 0.0f;
    const float rsc = rsqrtf((float)hd);
    const int nst_all = (pos + 128) >> 7;                        // ceil((pos+1)/128)
    int it_lo = 0, nst = nst_all;
    if (Sc > 1) {
        int per = (nst_all + Sc - 1) / Sc;
        it_lo = split * per;
        int hi = it_lo + per;
        nst = hi < nst_all ? hi : nst_all;
    }
    const size_t bt_base = (size_t)slot * max_blocks;
    // T4: every key before the column's own position is a trellis row (prefill and verify waves encode their
    // columns per layer before the attention, decode steps right after the step); the column's own key (row
    // pos) is read exact from the fp16 staging. A super-tile is all-trellis or the mixed one holding row pos.
#define TQ_GQA_SUPERTILE(t0d, physd) \
    do { \
        if constexpr (T4) { \
            const int nenc_ = pos - (t0d); \
            const int stg_ = slot; \
            if (nenc_ >= 128) \
                tq_gqa_mma_supertile_t4(oacc, sm, wmax, wsum, mst, lst, alph, tid, warp, lane, gr, t4, \
                    (t0d), (physd), pos, page, group, nkv, rsc, kcb4, t4p.kside, vcb8, t4p.vside, kv_head, t4tab); \
            else \
                tq_gqa_mma_supertile_mix(oacc, sm, wmax, wsum, mst, lst, alph, tid, warp, lane, gr, t4, \
                    (t0d), (physd), stg_, nenc_ < 0 ? 0 : nenc_, pos, page, group, nkv, rsc, kcb4, t4p.kside, \
                    vcb8, t4p.vside, kv_head, t4tab, t4p.f16s); \
        } else { \
            tq_gqa_mma_supertile<VQ4>(oacc, sm, wmax, wsum, mst, lst, alph, tid, warp, lane, gr, t4, \
                (t0d), (physd), pos, page, group, nkv, rsc, kcb4, cstride4, ksz4, szstride, vcb8, cstride, vscb); \
        } \
    } while (0)
#define TQ_GQA_DENSE_WALK \
    for (int it = it_lo; it < nst; it++) { \
        int t0d = it << 7; \
        int physd = block_table[bt_base + (t0d >> page_log)]; \
        TQ_GQA_SUPERTILE(t0d, physd); \
    }
    if constexpr (USE_TOPK) {
        int list_cnt = 0;
        const int *list_base = NULL;
        if (topk_count) {
            list_cnt = topk_count[(size_t)col * nkv + kv_head];
            if (list_cnt > 0 && topk_list)
                list_base = topk_list + ((size_t)(col * nkv + kv_head) * (size_t)topk_cap) * 2;
            else
                list_cnt = 0;
        }
        if (list_cnt <= 0) {
            TQ_GQA_DENSE_WALK
        } else {
            int per = (list_cnt + Sc - 1) / Sc;
            int lo = split * per;
            int hi = lo + per;
            if (hi > list_cnt) hi = list_cnt;
            for (int li = lo; li < hi; li++) {
                int phys = list_base[li * 2];
                int lb = list_base[li * 2 + 1];
                int page_base = lb << page_log;
                for (int sub = 0; sub < page; sub += 128) {
                    int t0 = page_base + sub;
                    if (t0 > pos) break;
                    TQ_GQA_SUPERTILE(t0, phys);
                }
            }
        }
    } else {
        (void)topk_list; (void)topk_count; (void)topk_cap;
        TQ_GQA_DENSE_WALK
    }
#undef TQ_GQA_DENSE_WALK
#undef TQ_GQA_SUPERTILE
    // ---- output: Sc == 1 -> normalized + gated rows; Sc > 1 -> split partials for the merge ----
    if (Sc > 1) {
        // partial layout of k_tq_paged_attn_q4_split: [(head * N + col) * S + split][hd], (m, l)
        #pragma unroll
        for (int f = 0; f < 4; f++) {
            int cidx = warp * 32 + f * 8 + 2 * t4;
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                const int ch = VQ4 ? warp * 32 + 4 * (2 * t4 + j) + f : cidx + j;
                if (gr < group) {
                    size_t pidx = ((size_t)((kv_head * group + gr) * gridDim.y + col) * S + split);
                    part_acc[pidx * hd + ch] = oacc[f][j];
                }
                if (8 + gr < group) {
                    size_t pidx = ((size_t)((kv_head * group + 8 + gr) * gridDim.y + col) * S + split);
                    part_acc[pidx * hd + ch] = oacc[f][2 + j];
                }
            }
        }
        if (tid < group) {
            size_t pidx = ((size_t)((kv_head * group + tid) * gridDim.y + col) * S + split);
            part_ml[pidx * 2 + 0] = mst[tid]; part_ml[pidx * 2 + 1] = lst[tid];
        }
        return;
    }
    float inv_lo = (gr < group) ? 1.0f / lst[gr] : 0.0f;
    float inv_hi = (8 + gr < group) ? 1.0f / lst[8 + gr] : 0.0f;
    if constexpr (VQ4) {
        // rotated rows -> smem ([group][256] lies below lst), one FWHT-256 per q-head by a warp
        float *ob = (float *)sm;
        __syncthreads();
        #pragma unroll
        for (int f = 0; f < 4; f++) {
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                const int ch = warp * 32 + 4 * (2 * t4 + j) + f;
                if (gr < group) ob[gr * 256 + ch] = oacc[f][j] * inv_lo;
                if (8 + gr < group) ob[(8 + gr) * 256 + ch] = oacc[f][2 + j] * inv_hi;
            }
        }
        __syncthreads();
        for (int r = warp; r < group; r += 8) {
            float x[8];
            #pragma unroll
            for (int e = 0; e < 8; e++) x[e] = ob[r * 256 + lane + 32 * e];
            tq_fwht256_warp(x, lane);
            const int hh = kv_head * group + r;
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                const int d = lane + 32 * e;
                float gate = q_proj[hh * (2 * hd) + hd + d];
                out[(size_t)col * attn_m + hh * hd + d] = x[e] * (1.0f / (1.0f + expf(-gate)));
            }
        }
        return;
    }
    #pragma unroll
    for (int f = 0; f < 4; f++) {
        int cidx = warp * 32 + f * 8 + 2 * t4;
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            if (gr < group) {
                int hh = kv_head * group + gr;
                float gate = q_proj[hh * (2 * hd) + hd + cidx + j];
                out[(size_t)col * attn_m + hh * hd + cidx + j] = oacc[f][j] * inv_lo * (1.0f / (1.0f + expf(-gate)));
            }
            if (8 + gr < group) {
                int hh = kv_head * group + 8 + gr;
                float gate = q_proj[hh * (2 * hd) + hd + cidx + j];
                out[(size_t)col * attn_m + hh * hd + cidx + j] = oacc[f][2 + j] * inv_hi * (1.0f / (1.0f + expf(-gate)));
            }
        }
    }
}

// ===== GQA decode attention, staged-V variant (task 10 / PP4 memo: the attention kernel's second pass) =====
// Same math and MMA order as k_tq_paged_attn_q4_gqa_mma (bitwise identical outputs), two memory-pipeline changes:
//  (1) the int4 V codes (128 rows x 128 B) and their (scale, zp) rows (128 x 32 B) of the super-tile are copied
//      into shared memory with cp.async at the START of the tile, so their DRAM latency overlaps the K loads, the
//      Q.K^T MMAs and the softmax instead of being paid as 8 dependent load rounds in the P.V loop;
//  (2) the K code bytes are extracted with shifts instead of a byte-indexed register array (which lived in local
//      memory: 64 LDL + 16 STL per tile per thread).
// Dense walk only (the Quest page-list path stays on the base kernel); int4 V (VQ4 = 1) only, fp16 scales only.
// 20 KB more static smem (38.6 KB): still 3 blocks/SM by registers.
#define TQ_AST_VST   TQ_AMM_WORDS                 // [128 rows][32 words] V codes
#define TQ_AST_VSS   (TQ_AMM_WORDS + 4096)        // [128 rows][8 words] V (scale, zp) fp16 pairs
#define TQ_AST_WORDS (TQ_AMM_WORDS + 5120)

static __device__ __forceinline__ void tq_gqa_mma_supertile_st(
    float oacc[4][4], uint32_t *sm, float *wmax, float *wsum, float *mst, float *lst, float *alph,
    int tid, int warp, int lane, int gr, int t4,
    int t0, int phys, int pos, int page, int group, int nkv, float rsc,
    const uint8_t *kcb4, size_t cstride4, const float *ksz4, size_t szstride,
    const uint8_t *vcb8, const float *vscb) {
    const size_t pr0 = (size_t)phys * page + (size_t)(t0 & (page - 1));
#define TQ_GROW(lp) (pr0 + (size_t)(min((lp), pos) - t0))
    // (1) stage this tile's V codes + scales (rows clamped to pos like the direct reads): 5 x 16 B per thread
    {
        uint32_t *vst = sm + TQ_AST_VST, *vss = sm + TQ_AST_VSS;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            int id = tid + 256 * k, row = id >> 3, sub = id & 7;
            const uint8_t *src = vcb8 + TQ_GROW(t0 + row) * cstride4 + 16 * sub;
            tq_cp_async16((uint32_t)__cvta_generic_to_shared(vst + row * 32 + sub * 4), src);
        }
        {
            int row = tid >> 1, sub = tid & 1;
            const __half *src = (const __half *)q4s_adv(vscb, TQ_GROW(t0 + row) * szstride) + 8 * sub;
            tq_cp_async16((uint32_t)__cvta_generic_to_shared(vss + row * 8 + sub * 4), src);
        }
        asm volatile("cp.async.commit_group;\n");
    }
    float s0[4], s1[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) { s0[j] = 0.0f; s1[j] = 0.0f; }
    const int tp = t0 + warp * 16;
    const int p0 = tp + gr, p1 = tp + 8 + gr;
    {
        const uint8_t *kr0 = kcb4 + TQ_GROW(p0) * cstride4;
        const uint8_t *kr1 = kcb4 + TQ_GROW(p1) * cstride4;
        const float *sz0 = q4s_adv(ksz4, TQ_GROW(p0) * szstride);
        const float *sz1 = q4s_adv(ksz4, TQ_GROW(p1) * szstride);
        const int sh = 8 * t4;
        #pragma unroll
        for (int g = 0; g < 8; g++) {
            float2 p0s = q4s_pair(sz0, 2 * g), p1s = q4s_pair(sz1, 2 * g);
            float sc0 = p0s.x, zp0 = p0s.y, sc1 = p1s.x, zp1 = p1s.y;
            uint4 kp0 = *(const uint4 *)(kr0 + 16 * g);
            uint4 kp1 = *(const uint4 *)(kr1 + 16 * g);
            #pragma unroll
            for (int cc = 0; cc < 2; cc++) {
                int c = 2 * g + cc;
                const uint32_t *a = &sm[TQ_AMM_QA + (c * 32 + lane) * 4];
                // (2) byte bi = (8c + t4) & 15 of the 16-byte chunk: word (cc ? 2 : 0) (+1 for bi + 4), byte t4
                const uint32_t wa0 = cc ? kp0.z : kp0.x, wa8 = cc ? kp0.w : kp0.y;
                const uint32_t we0 = cc ? kp1.z : kp1.x, we8 = cc ? kp1.w : kp1.y;
                uint8_t a0 = (uint8_t)(wa0 >> sh), a8 = (uint8_t)(wa8 >> sh);
                uint8_t e0 = (uint8_t)(we0 >> sh), e8 = (uint8_t)(we8 >> sh);
                uint32_t bk00 = tq_pack_bf16(((float)(a0 & 15) - zp0) * sc0, ((float)(a0 >> 4) - zp0) * sc0);
                uint32_t bk08 = tq_pack_bf16(((float)(a8 & 15) - zp0) * sc0, ((float)(a8 >> 4) - zp0) * sc0);
                uint32_t bk10 = tq_pack_bf16(((float)(e0 & 15) - zp1) * sc1, ((float)(e0 >> 4) - zp1) * sc1);
                uint32_t bk18 = tq_pack_bf16(((float)(e8 & 15) - zp1) * sc1, ((float)(e8 >> 4) - zp1) * sc1);
                tq_mma_bf16_m16n8k16(s0, a, bk00, bk08);
                tq_mma_bf16_m16n8k16(s1, a, bk10, bk18);
            }
        }
    }
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        int qrow = (j < 2) ? gr : (gr + 8);
        int k0 = tp + 2 * t4 + (j & 1), k1 = tp + 8 + 2 * t4 + (j & 1);
        bool rv = qrow < group;
        s0[j] = (rv && k0 <= pos) ? s0[j] * rsc : TQ_AMM_NINF;
        s1[j] = (rv && k1 <= pos) ? s1[j] * rsc : TQ_AMM_NINF;
    }
    float lm_lo = fmaxf(fmaxf(s0[0], s0[1]), fmaxf(s1[0], s1[1]));
    float lm_hi = fmaxf(fmaxf(s0[2], s0[3]), fmaxf(s1[2], s1[3]));
    #pragma unroll
    for (int o = 1; o <= 2; o <<= 1) {
        lm_lo = fmaxf(lm_lo, __shfl_xor_sync(0xffffffffu, lm_lo, o));
        lm_hi = fmaxf(lm_hi, __shfl_xor_sync(0xffffffffu, lm_hi, o));
    }
    if (t4 == 0) { wmax[warp * 16 + gr] = lm_lo; wmax[warp * 16 + 8 + gr] = lm_hi; }
    __syncthreads();
    if (warp == 0 && lane < 16) {
        float m_old = mst[lane], m_new = m_old;
        for (int w = 0; w < 8; w++) m_new = fmaxf(m_new, wmax[w * 16 + lane]);
        alph[lane] = expf(m_old - m_new); mst[lane] = m_new;
    }
    __syncthreads();
    float mn_lo = mst[gr], mn_hi = mst[8 + gr];
    float p00 = expf(s0[0] - mn_lo), p01 = expf(s0[1] - mn_lo), p02 = expf(s0[2] - mn_hi), p03 = expf(s0[3] - mn_hi);
    float p10 = expf(s1[0] - mn_lo), p11 = expf(s1[1] - mn_lo), p12 = expf(s1[2] - mn_hi), p13 = expf(s1[3] - mn_hi);
    float ls_lo = p00 + p01 + p10 + p11, ls_hi = p02 + p03 + p12 + p13;
    #pragma unroll
    for (int o = 1; o <= 2; o <<= 1) {
        ls_lo += __shfl_xor_sync(0xffffffffu, ls_lo, o);
        ls_hi += __shfl_xor_sync(0xffffffffu, ls_hi, o);
    }
    if (t4 == 0) { wsum[warp * 16 + gr] = ls_lo; wsum[warp * 16 + 8 + gr] = ls_hi; }
    uint32_t *pst = &sm[TQ_AMM_PST + (warp * 32 + lane) * 4];
    pst[0] = tq_pack_bf16(p00, p01); pst[1] = tq_pack_bf16(p02, p03);
    pst[2] = tq_pack_bf16(p10, p11); pst[3] = tq_pack_bf16(p12, p13);
    asm volatile("cp.async.wait_group 0;\n");          // this tile's V has landed (per thread) ...
    __syncthreads();                                   // ... and is visible to every warp
    if (warp == 0 && lane < 16) {
        float l = lst[lane] * alph[lane];
        for (int w = 0; w < 8; w++) l += wsum[w * 16 + lane];
        lst[lane] = l;
    }
    float a_lo = alph[gr], a_hi = alph[8 + gr];
    #pragma unroll
    for (int f = 0; f < 4; f++) { oacc[f][0] *= a_lo; oacc[f][1] *= a_lo; oacc[f][2] *= a_hi; oacc[f][3] *= a_hi; }
    const uint8_t *vst8 = (const uint8_t *)(sm + TQ_AST_VST);
    const float *vssf = (const float *)(sm + TQ_AST_VSS);
    #pragma unroll 2
    for (int tt = 0; tt < 8; tt++) {
        const uint32_t *pa = &sm[TQ_AMM_PST + (tt * 32 + lane) * 4];
        uint32_t a[4] = {pa[0], pa[1], pa[2], pa[3]};
        const int r0 = tt * 16 + 2 * t4;                   // row within the staged tile
        const size_t cb = (size_t)(16 * warp + 2 * gr);
        const uint32_t c0 = *(const uint16_t *)(vst8 + (size_t)r0 * 128 + cb);
        const uint32_t c1 = *(const uint16_t *)(vst8 + (size_t)(r0 + 1) * 128 + cb);
        const uint32_t c8 = *(const uint16_t *)(vst8 + (size_t)(r0 + 8) * 128 + cb);
        const uint32_t c9 = *(const uint16_t *)(vst8 + (size_t)(r0 + 9) * 128 + cb);
        const float2 z0 = q4s_pair(vssf + (size_t)r0 * 8, 2 * warp);
        const float2 z1 = q4s_pair(vssf + (size_t)(r0 + 1) * 8, 2 * warp);
        const float2 z8 = q4s_pair(vssf + (size_t)(r0 + 8) * 8, 2 * warp);
        const float2 z9 = q4s_pair(vssf + (size_t)(r0 + 9) * 8, 2 * warp);
        #pragma unroll
        for (int f = 0; f < 4; f++) {
            const int sh = 4 * f;
            uint32_t b0 = tq_pack_bf16(((float)((c0 >> sh) & 15u) - z0.y) * z0.x,
                                       ((float)((c1 >> sh) & 15u) - z1.y) * z1.x);
            uint32_t b1 = tq_pack_bf16(((float)((c8 >> sh) & 15u) - z8.y) * z8.x,
                                       ((float)((c9 >> sh) & 15u) - z9.y) * z9.x);
            tq_mma_bf16_m16n8k16(oacc[f], a, b0, b1);
        }
    }
#undef TQ_GROW
    __syncthreads();
}

#ifndef TQ_AST_MINB
#define TQ_AST_MINB 2                              // -DTQ_AST_MINB=3 caps the registers for 3 blocks/SM (A/B build)
#endif
__global__ void __launch_bounds__(256, TQ_AST_MINB) k_tq_paged_attn_q4_gqa_mma_st(
    float *out, const float *q_proj_base, const uint16_t *q_norm_w,
    const uint8_t *k4_pool, const float *kq4s_pool, const uint8_t *v8_pool, const float *vscale_pool,
    const int *positions, const int *slot_ids,
    const int *block_table, int max_blocks, int page, int page_log,
    int nh, int nkv, int hd, int q_m, int attn_m, int S,
    float eps, float rope_theta, float *part_acc, float *part_ml, const int *col_S) {
    tq_pdl_enter();
    __shared__ __align__(16) uint32_t sm[TQ_AST_WORDS];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int gr = lane >> 2, t4 = lane & 3;
    const int kv_head = blockIdx.x, col = blockIdx.y, split = blockIdx.z;
    const int Sc = col_S ? col_S[col] : S;
    if (split >= Sc) return;
    const int group = nh / nkv;
    const int pos = positions[col], slot = slot_ids[col];
    const size_t cstride = (size_t)nkv * hd, cstride4 = cstride >> 1, szstride = (size_t)nkv * 16;
    const uint8_t *kcb4 = k4_pool + (size_t)kv_head * (hd >> 1);
    const float *ksz4 = q4s_adv(kq4s_pool, (size_t)kv_head * 16);
    const uint8_t *vcb8 = v8_pool + (size_t)kv_head * (hd >> 1);
    const float *vscb = q4s_adv(vscale_pool, (size_t)kv_head * 16);
    float *qtmp = (float *)(sm + TQ_AMM_QTMP);
    float *red = (float *)(sm + TQ_AMM_RED);
    float *wmax = (float *)(sm + TQ_AMM_WMAX), *wsum = (float *)(sm + TQ_AMM_WSUM);
    float *mst = (float *)(sm + TQ_AMM_MST), *lst = (float *)(sm + TQ_AMM_LST);
    float *alph = (float *)(sm + TQ_AMM_ALPH);
    for (int i = tid; i < TQ_AMM_PST + 1024; i += 256) sm[i] = 0;
    if (tid < 16) { mst[tid] = TQ_AMM_NINF; lst[tid] = 0.0f; }
    __syncthreads();
    const float *q_proj = q_proj_base + (size_t)col * q_m;
    for (int r = 0; r < group; r++) {
        const int hh = kv_head * group + r;
        float qv = q_proj[hh * (2 * hd) + tid];
        float qsum = qv * qv;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) qsum += __shfl_down_sync(0xffffffffu, qsum, o);
        if (lane == 0) red[warp] = qsum;
        __syncthreads();
        if (tid < 8) { float v = red[tid]; for (int o = 4; o > 0; o >>= 1) v += __shfl_down_sync(0xffu, v, o); if (tid == 0) red[8] = v; }
        __syncthreads();
        float qss = red[8];
        qv = qv * rsqrtf(qss / (float)hd + eps) * (1.0f + tq_bf16_to_float(q_norm_w[tid]));
        if (tid < 64) {
            int idx = tid & 31;
            float freq = powf(rope_theta, -((float)(2 * idx) / 64.0f));
            float angle = (float)pos * freq, c = cosf(angle), s = sinf(angle);
            float q_pair = q_proj[hh * (2 * hd) + ((tid < 32) ? tid + 32 : tid - 32)];
            q_pair = q_pair * rsqrtf(qss / (float)hd + eps) * (1.0f + tq_bf16_to_float(q_norm_w[(tid < 32) ? tid + 32 : tid - 32]));
            float q_rot = (tid < 32) ? -q_pair : q_pair;
            qv = qv * c + q_rot * s;
        }
        __syncthreads();
        qtmp[tid] = qv;
        __syncthreads();
        tq_fwht256(qtmp, tid);
        if (tid < 128) {
            int c16 = tid >> 3, tt = (tid >> 1) & 3, rsel = tid & 1;
            int d0 = 16 * c16 + 2 * tt + 8 * rsel;
            int l = 4 * (r & 7) + tt, rr = ((r >> 3) & 1) + 2 * rsel;
            sm[TQ_AMM_QA + (c16 * 32 + l) * 4 + rr] = tq_pack_bf16(qtmp[d0], qtmp[d0 + 1]);
        }
        __syncthreads();
    }
    float oacc[4][4];
    #pragma unroll
    for (int f = 0; f < 4; f++) for (int j = 0; j < 4; j++) oacc[f][j] = 0.0f;
    const float rsc = rsqrtf((float)hd);
    const int nst_all = (pos + 128) >> 7;
    int it_lo = 0, nst = nst_all;
    if (Sc > 1) {
        int per = (nst_all + Sc - 1) / Sc;
        it_lo = split * per;
        int hi = it_lo + per;
        nst = hi < nst_all ? hi : nst_all;
    }
    const size_t bt_base = (size_t)slot * max_blocks;
    for (int it = it_lo; it < nst; it++) {
        int t0d = it << 7;
        int physd = block_table[bt_base + (t0d >> page_log)];
        tq_gqa_mma_supertile_st(oacc, sm, wmax, wsum, mst, lst, alph,
            tid, warp, lane, gr, t4, t0d, physd, pos, page, group, nkv, rsc,
            kcb4, cstride4, ksz4, szstride, vcb8, vscb);
    }
    if (Sc > 1) {
        #pragma unroll
        for (int f = 0; f < 4; f++) {
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                const int ch = warp * 32 + 4 * (2 * t4 + j) + f;
                if (gr < group) {
                    size_t pidx = ((size_t)((kv_head * group + gr) * gridDim.y + col) * S + split);
                    part_acc[pidx * hd + ch] = oacc[f][j];
                }
                if (8 + gr < group) {
                    size_t pidx = ((size_t)((kv_head * group + 8 + gr) * gridDim.y + col) * S + split);
                    part_acc[pidx * hd + ch] = oacc[f][2 + j];
                }
            }
        }
        if (tid < group) {
            size_t pidx = ((size_t)((kv_head * group + tid) * gridDim.y + col) * S + split);
            part_ml[pidx * 2 + 0] = mst[tid]; part_ml[pidx * 2 + 1] = lst[tid];
        }
        return;
    }
    float inv_lo = (gr < group) ? 1.0f / lst[gr] : 0.0f;
    float inv_hi = (8 + gr < group) ? 1.0f / lst[8 + gr] : 0.0f;
    float *ob = (float *)sm;
    __syncthreads();
    #pragma unroll
    for (int f = 0; f < 4; f++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            const int ch = warp * 32 + 4 * (2 * t4 + j) + f;
            if (gr < group) ob[gr * 256 + ch] = oacc[f][j] * inv_lo;
            if (8 + gr < group) ob[(8 + gr) * 256 + ch] = oacc[f][2 + j] * inv_hi;
        }
    }
    __syncthreads();
    for (int r = warp; r < group; r += 8) {
        float x[8];
        #pragma unroll
        for (int e = 0; e < 8; e++) x[e] = ob[r * 256 + lane + 32 * e];
        tq_fwht256_warp(x, lane);
        const int hh = kv_head * group + r;
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            const int d = lane + 32 * e;
            float gate = q_proj[hh * (2 * hd) + hd + d];
            out[(size_t)col * attn_m + hh * hd + d] = x[e] * (1.0f / (1.0f + expf(-gate)));
        }
    }
}

// TQ_PAGED_ATTN_ST: 0 = base GQA decode kernel, 1 = staged-V kernel (bitwise = base; the default until 10-16),
// 2 = fp16 int4 path for P.V (k_tq_paged_attn_q4_gqa_f16<QT,0>: P in fp16 with the V scale folded in, raw nibbles as
// operands, exact V), 3 = also the fp16 K dequant and fp16 Q (k_tq_paged_attn_q4_gqa_f16<QT,1>). 2/3 serve the verify
// waves with the same kernel. **Default 3 since 10-16 (owner, HANDOFF Decisions 16e)**: e2e-neutral within +-0.0001 KL,
// verify == decode, -1.5 ms per step at 32K N = 8; TQ_PAGED_ATTN_ST=1 reverts to the kernel that is bitwise e1ecb3a's.
// The page-list (Quest) path and E4M3 V always use the base kernel; with TQ_KV_T4 the trellis pool runs its own GQA
// kernels and this knob is a no-op (see launch_paged_attn_q4).
// The effective TQ_PAGED_ATTN_ST (for the server's startup line).
extern "C" int qwn_paged_attn_st(void) { return g_cfg.paged_attn_st; }

// ===== Segment-aware GQA attention for the verify wave (task 10, stage 5 performance item (a)) =====
// One block = (kv head, segment of up to QT*16/group consecutive columns of ONE slot, key split): the slot's
// K/V super-tiles are streamed and dequantized once for all the segment's columns instead of once per column.
// Row (t, r) of the QT m16 tiles is column t's q-head r with ITS OWN causal limit pos_t, split count S_t and
// split tile range [lo_t, hi_t) = exactly what k_tq_paged_attn_q4_gqa_mma computes for that column, so every
// row's arithmetic (tile order, masks, online-softmax updates, partial layout) is the per-column kernel's bit
// for bit: a tile outside a row's range leaves the row's state untouched (alpha = 1, p = 0, oacc + 0 = oacc).
// The per-column merge kernel combines the partials unchanged (layout stride S = the grid's, ncols = the wave's).
#define TQ_SEG_QA     0                         // [QT*16 chunks][32 lanes][4] bf16 Q A-frags (QT <= 2)
#define TQ_SEG_PST    4096                      // [QT*8 tiles][32][4] P A-frag staging
#define TQ_SEG_WMAX   6144                      // [8 warps][32 rows]
#define TQ_SEG_WSUM   6400
#define TQ_SEG_MST    6656                      // [32] running row max
#define TQ_SEG_LST    6688                      // [32] running row denominator
#define TQ_SEG_ALPH   6720                      // [32] per-row rescale
#define TQ_SEG_QTMP   6752                      // [256]
#define TQ_SEG_RED    7008                      // [16]
#define TQ_SEG_RPOS   7024                      // [32] per row: position
#define TQ_SEG_RLO    7056                      // [32] first tile of the row's split range
#define TQ_SEG_RHI    7088                      // [32] one past the last
#define TQ_SEG_RSC    7120                      // [32] the row's split count S_t
#define TQ_SEG_RCOL   7152                      // [32] the row's wave column
#define TQ_SEG_WORDS  7184

template <int VQ4, int QT>
static __device__ __forceinline__ void tq_gqa_seg_supertile(
    float oacc[QT][4][4], uint32_t *sm, int tid, int warp, int lane, int gr, int t4,
    int it, int t0, int phys, int pos_max, int page, int nrows, int nkv, float rsc,
    const uint8_t *kcb4, size_t cstride4, const float *ksz4, size_t szstride,
    const uint8_t *vcb8, size_t cstride, const float *vscb) {
    float *wmax = (float *)(sm + TQ_SEG_WMAX), *wsum = (float *)(sm + TQ_SEG_WSUM);
    float *mst = (float *)(sm + TQ_SEG_MST), *lst = (float *)(sm + TQ_SEG_LST), *alph = (float *)(sm + TQ_SEG_ALPH);
    const int *rpos = (const int *)(sm + TQ_SEG_RPOS), *rlo = (const int *)(sm + TQ_SEG_RLO), *rhi = (const int *)(sm + TQ_SEG_RHI);
    const size_t pr0 = (size_t)phys * page + (size_t)(t0 & (page - 1));
#define TQ_GROW(lp) (pr0 + (size_t)(min((lp), pos_max) - t0))
    float s0[QT][4], s1[QT][4];
    #pragma unroll
    for (int q = 0; q < QT; q++)
        #pragma unroll
        for (int j = 0; j < 4; j++) { s0[q][j] = 0.0f; s1[q][j] = 0.0f; }
    const int tp = t0 + warp * 16;
    const int p0 = tp + gr, p1 = tp + 8 + gr;
    {
        const uint8_t *kr0 = kcb4 + TQ_GROW(p0) * cstride4;
        const uint8_t *kr1 = kcb4 + TQ_GROW(p1) * cstride4;
        const float *sz0 = q4s_adv(ksz4, TQ_GROW(p0) * szstride);
        const float *sz1 = q4s_adv(ksz4, TQ_GROW(p1) * szstride);
        #pragma unroll
        for (int g = 0; g < 8; g++) {
            float2 p0s = q4s_pair(sz0, 2 * g), p1s = q4s_pair(sz1, 2 * g);
            float sc0 = p0s.x, zp0 = p0s.y, sc1 = p1s.x, zp1 = p1s.y;
            uint4 kp0 = *(const uint4 *)(kr0 + 16 * g);
            uint4 kp1 = *(const uint4 *)(kr1 + 16 * g);
            uint32_t w0[4] = {kp0.x, kp0.y, kp0.z, kp0.w};
            uint32_t w1[4] = {kp1.x, kp1.y, kp1.z, kp1.w};
            const uint8_t *b0 = (const uint8_t *)w0, *b1 = (const uint8_t *)w1;
            #pragma unroll
            for (int cc = 0; cc < 2; cc++) {
                int c = 2 * g + cc;
                int bi = (8 * c + t4) & 15;
                uint8_t a0 = b0[bi], a8 = b0[bi + 4], e0 = b1[bi], e8 = b1[bi + 4];
                uint32_t bk00 = tq_pack_bf16(((float)(a0 & 15) - zp0) * sc0, ((float)(a0 >> 4) - zp0) * sc0);
                uint32_t bk08 = tq_pack_bf16(((float)(a8 & 15) - zp0) * sc0, ((float)(a8 >> 4) - zp0) * sc0);
                uint32_t bk10 = tq_pack_bf16(((float)(e0 & 15) - zp1) * sc1, ((float)(e0 >> 4) - zp1) * sc1);
                uint32_t bk18 = tq_pack_bf16(((float)(e8 & 15) - zp1) * sc1, ((float)(e8 >> 4) - zp1) * sc1);
                #pragma unroll
                for (int q = 0; q < QT; q++) {
                    const uint32_t *a = &sm[TQ_SEG_QA + ((q * 16 + c) * 32 + lane) * 4];
                    tq_mma_bf16_m16n8k16(s0[q], a, bk00, bk08);
                    tq_mma_bf16_m16n8k16(s1[q], a, bk10, bk18);
                }
            }
        }
    }
    // scale + mask per ROW: keys > pos_t, tiles outside the row's split range, and rows >= nrows go to -inf
    bool act_lo[QT], act_hi[QT];
    #pragma unroll
    for (int q = 0; q < QT; q++) {
        const int rl = q * 16 + gr, rh = rl + 8;
        act_lo[q] = rl < nrows && it >= rlo[rl] && it < rhi[rl];
        act_hi[q] = rh < nrows && it >= rlo[rh] && it < rhi[rh];
        const int pl = act_lo[q] ? rpos[rl] : -1, ph = act_hi[q] ? rpos[rh] : -1;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int k0 = tp + 2 * t4 + (j & 1), k1 = tp + 8 + 2 * t4 + (j & 1);
            bool rv = (j < 2) ? act_lo[q] : act_hi[q];
            int prow = (j < 2) ? pl : ph;
            s0[q][j] = (rv && k0 <= prow) ? s0[q][j] * rsc : TQ_AMM_NINF;
            s1[q][j] = (rv && k1 <= prow) ? s1[q][j] * rsc : TQ_AMM_NINF;
        }
        float lm_lo = fmaxf(fmaxf(s0[q][0], s0[q][1]), fmaxf(s1[q][0], s1[q][1]));
        float lm_hi = fmaxf(fmaxf(s0[q][2], s0[q][3]), fmaxf(s1[q][2], s1[q][3]));
        #pragma unroll
        for (int o = 1; o <= 2; o <<= 1) {
            lm_lo = fmaxf(lm_lo, __shfl_xor_sync(0xffffffffu, lm_lo, o));
            lm_hi = fmaxf(lm_hi, __shfl_xor_sync(0xffffffffu, lm_hi, o));
        }
        if (t4 == 0) { wmax[warp * 32 + rl] = lm_lo; wmax[warp * 32 + rh] = lm_hi; }
    }
    __syncthreads();
    if (tid < QT * 16) {
        const int row = tid;
        const bool act = row < nrows && it >= rlo[row] && it < rhi[row];
        float m_old = mst[row], m_new = m_old;
        for (int w = 0; w < 8; w++) m_new = fmaxf(m_new, wmax[w * 32 + row]);
        alph[row] = act ? expf(m_old - m_new) : 1.0f;
        mst[row] = act ? m_new : m_old;
    }
    __syncthreads();
    #pragma unroll
    for (int q = 0; q < QT; q++) {
        const int rl = q * 16 + gr, rh = rl + 8;
        float mn_lo = mst[rl], mn_hi = mst[rh];
        float p00 = act_lo[q] ? expf(s0[q][0] - mn_lo) : 0.0f, p01 = act_lo[q] ? expf(s0[q][1] - mn_lo) : 0.0f;
        float p02 = act_hi[q] ? expf(s0[q][2] - mn_hi) : 0.0f, p03 = act_hi[q] ? expf(s0[q][3] - mn_hi) : 0.0f;
        float p10 = act_lo[q] ? expf(s1[q][0] - mn_lo) : 0.0f, p11 = act_lo[q] ? expf(s1[q][1] - mn_lo) : 0.0f;
        float p12 = act_hi[q] ? expf(s1[q][2] - mn_hi) : 0.0f, p13 = act_hi[q] ? expf(s1[q][3] - mn_hi) : 0.0f;
        float ls_lo = p00 + p01 + p10 + p11, ls_hi = p02 + p03 + p12 + p13;
        #pragma unroll
        for (int o = 1; o <= 2; o <<= 1) {
            ls_lo += __shfl_xor_sync(0xffffffffu, ls_lo, o);
            ls_hi += __shfl_xor_sync(0xffffffffu, ls_hi, o);
        }
        if (t4 == 0) { wsum[warp * 32 + rl] = ls_lo; wsum[warp * 32 + rh] = ls_hi; }
        if constexpr (!VQ4) {   // fold the per-key V scale into p (the denominator keeps the unscaled p)
            int p0c = t0 + warp * 16 + 2 * t4, p1c = p0c + 8;
            float v00 = vscb[TQ_GROW(p0c) * nkv], v01 = vscb[TQ_GROW(p0c + 1) * nkv];
            float v10 = vscb[TQ_GROW(p1c) * nkv], v11 = vscb[TQ_GROW(p1c + 1) * nkv];
            p00 *= v00; p01 *= v01; p02 *= v00; p03 *= v01;
            p10 *= v10; p11 *= v11; p12 *= v10; p13 *= v11;
        }
        uint32_t *pst = &sm[TQ_SEG_PST + ((q * 8 + warp) * 32 + lane) * 4];
        pst[0] = tq_pack_bf16(p00, p01); pst[1] = tq_pack_bf16(p02, p03);
        pst[2] = tq_pack_bf16(p10, p11); pst[3] = tq_pack_bf16(p12, p13);
    }
    __syncthreads();
    if (tid < QT * 16) {
        float l = lst[tid] * alph[tid];
        for (int w = 0; w < 8; w++) l += wsum[w * 32 + tid];
        lst[tid] = l;
    }
    #pragma unroll
    for (int q = 0; q < QT; q++) {
        float a_lo = alph[q * 16 + gr], a_hi = alph[q * 16 + 8 + gr];
        #pragma unroll
        for (int f = 0; f < 4; f++) { oacc[q][f][0] *= a_lo; oacc[q][f][1] *= a_lo; oacc[q][f][2] *= a_hi; oacc[q][f][3] *= a_hi; }
    }
    for (int tt = 0; tt < 8; tt++) {
        if constexpr (VQ4) {
            const int r0 = t0 + tt * 16 + 2 * t4;
            const size_t q0 = TQ_GROW(r0), q1 = TQ_GROW(r0 + 1), q8 = TQ_GROW(r0 + 8), q9 = TQ_GROW(r0 + 9);
            const size_t cb = (size_t)(16 * warp + 2 * gr);
            const uint32_t c0 = *(const uint16_t *)(vcb8 + q0 * cstride4 + cb);
            const uint32_t c1 = *(const uint16_t *)(vcb8 + q1 * cstride4 + cb);
            const uint32_t c8 = *(const uint16_t *)(vcb8 + q8 * cstride4 + cb);
            const uint32_t c9 = *(const uint16_t *)(vcb8 + q9 * cstride4 + cb);
            const float2 z0 = q4s_pair(q4s_adv(vscb, q0 * szstride), 2 * warp);
            const float2 z1 = q4s_pair(q4s_adv(vscb, q1 * szstride), 2 * warp);
            const float2 z8 = q4s_pair(q4s_adv(vscb, q8 * szstride), 2 * warp);
            const float2 z9 = q4s_pair(q4s_adv(vscb, q9 * szstride), 2 * warp);
            #pragma unroll
            for (int f = 0; f < 4; f++) {
                const int sh = 4 * f;
                uint32_t b0 = tq_pack_bf16(((float)((c0 >> sh) & 15u) - z0.y) * z0.x,
                                           ((float)((c1 >> sh) & 15u) - z1.y) * z1.x);
                uint32_t b1 = tq_pack_bf16(((float)((c8 >> sh) & 15u) - z8.y) * z8.x,
                                           ((float)((c9 >> sh) & 15u) - z9.y) * z9.x);
                #pragma unroll
                for (int q = 0; q < QT; q++) {
                    const uint32_t *pa = &sm[TQ_SEG_PST + ((q * 8 + tt) * 32 + lane) * 4];
                    uint32_t a[4] = {pa[0], pa[1], pa[2], pa[3]};
                    tq_mma_bf16_m16n8k16(oacc[q][f], a, b0, b1);
                }
            }
            continue;
        }
        #pragma unroll
        for (int f = 0; f < 4; f++) {
            int cidx = warp * 32 + f * 8 + gr;
            int r0 = t0 + tt * 16 + 2 * t4;
            uint32_t b0 = tq_pack_bf16(tq_e4m3_dec_fast(vcb8[TQ_GROW(r0) * cstride + cidx]),
                                       tq_e4m3_dec_fast(vcb8[TQ_GROW(r0 + 1) * cstride + cidx]));
            uint32_t b1 = tq_pack_bf16(tq_e4m3_dec_fast(vcb8[TQ_GROW(r0 + 8) * cstride + cidx]),
                                       tq_e4m3_dec_fast(vcb8[TQ_GROW(r0 + 9) * cstride + cidx]));
            #pragma unroll
            for (int q = 0; q < QT; q++) {
                const uint32_t *pa = &sm[TQ_SEG_PST + ((q * 8 + tt) * 32 + lane) * 4];
                uint32_t a[4] = {pa[0], pa[1], pa[2], pa[3]};
                tq_mma_bf16_m16n8k16(oacc[q][f], a, b0, b1);
            }
        }
    }
#undef TQ_GROW
    __syncthreads();
}

// seg_col0[seg], seg_n[seg]: sub-segments of <= QT*16/group consecutive columns of one slot (consecutive
// positions); col_S: per-column split count (NULL = S); ncols = the wave's column count (partial layout).
template <int VQ4, int QT>
__global__ void __launch_bounds__(256) k_tq_paged_attn_q4_gqa_seg(
    float *out, const float *q_proj_base, const uint16_t *q_norm_w,
    const uint8_t *k4_pool, const float *kq4s_pool, const uint8_t *v8_pool, const float *vscale_pool,
    const int *positions, const int *slot_ids, const int *col_S, const int *seg_col0, const int *seg_n,
    const int *block_table, int max_blocks, int page, int page_log,
    int nh, int nkv, int hd, int q_m, int attn_m, int S, int ncols,
    float eps, float rope_theta, float *part_acc, float *part_ml) {
    tq_pdl_enter();
    __shared__ uint32_t sm[TQ_SEG_WORDS];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int gr = lane >> 2, t4 = lane & 3;
    const int kv_head = blockIdx.x, seg = blockIdx.y, split = blockIdx.z;
    const int group = nh / nkv;
    const int col0 = seg_col0[seg], n = seg_n[seg];
    const int nrows = n * group;
    const int slot = slot_ids[col0];
    const size_t cstride = (size_t)nkv * hd, cstride4 = cstride >> 1, szstride = (size_t)nkv * 16;
    const uint8_t *kcb4 = k4_pool + (size_t)kv_head * (hd >> 1);
    const float *ksz4 = q4s_adv(kq4s_pool, (size_t)kv_head * 16);
    const uint8_t *vcb8 = v8_pool + (size_t)kv_head * (VQ4 ? (hd >> 1) : hd);
    const float *vscb = VQ4 ? q4s_adv(vscale_pool, (size_t)kv_head * 16) : vscale_pool + kv_head;
    float *qtmp = (float *)(sm + TQ_SEG_QTMP), *red = (float *)(sm + TQ_SEG_RED);
    float *mst = (float *)(sm + TQ_SEG_MST), *lst = (float *)(sm + TQ_SEG_LST);
    int *rpos = (int *)(sm + TQ_SEG_RPOS), *rlo = (int *)(sm + TQ_SEG_RLO), *rhi = (int *)(sm + TQ_SEG_RHI);
    int *rSc = (int *)(sm + TQ_SEG_RSC), *rcol = (int *)(sm + TQ_SEG_RCOL);
    for (int i = tid; i < TQ_SEG_PST + QT * 1024; i += 256) sm[i] = 0;
    if (tid < 32) {
        mst[tid] = TQ_AMM_NINF; lst[tid] = 0.0f;
        int row = tid, lo = 0, hi = 0, pos = -1, Sc = 0, col = -1;
        if (row < nrows) {
            col = col0 + row / group; pos = positions[col]; Sc = col_S ? col_S[col] : S;
            int nst_all = (pos + 128) >> 7;
            hi = nst_all;
            if (Sc > 1) { int per = (nst_all + Sc - 1) / Sc; lo = split * per; hi = lo + per < nst_all ? lo + per : nst_all; }
            if (split >= Sc) { lo = 0; hi = 0; }                  // this split does not exist for the row
        }
        rpos[row] = pos; rlo[row] = lo; rhi[row] = hi; rSc[row] = Sc; rcol[row] = col;
    }
    __syncthreads();
    int it_lo = 1 << 30, it_hi = 0, pos_max = 0;
    for (int row = 0; row < nrows; row++) {
        if (rhi[row] > rlo[row]) { it_lo = min(it_lo, rlo[row]); it_hi = max(it_hi, rhi[row]); }
        pos_max = max(pos_max, rpos[row]);
    }
    // (no early exit when no row is active in this split: the tile loop is empty and the rows' NINF / 0
    //  partials still get written, as the per-column kernel does for an empty split)
    // ---- Q prep per row (column t, q-head r): norm + rope at the column's position + FWHT -> A fragments
    for (int row = 0; row < nrows; row++) {
        const int t = row / group, r = row - t * group, col = col0 + t, pos = rpos[row];
        const float *q_proj = q_proj_base + (size_t)col * q_m;
        const int hh = kv_head * group + r;
        float qv = q_proj[hh * (2 * hd) + tid];
        float qsum = qv * qv;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) qsum += __shfl_down_sync(0xffffffffu, qsum, o);
        if (lane == 0) red[warp] = qsum;
        __syncthreads();
        if (tid < 8) { float v = red[tid]; for (int o = 4; o > 0; o >>= 1) v += __shfl_down_sync(0xffu, v, o); if (tid == 0) red[8] = v; }
        __syncthreads();
        float qss = red[8];
        qv = qv * rsqrtf(qss / (float)hd + eps) * (1.0f + tq_bf16_to_float(q_norm_w[tid]));
        if (tid < 64) {
            int idx = tid & 31;
            float freq = powf(rope_theta, -((float)(2 * idx) / 64.0f));
            float angle = (float)pos * freq, c = cosf(angle), s = sinf(angle);
            float q_pair = q_proj[hh * (2 * hd) + ((tid < 32) ? tid + 32 : tid - 32)];
            q_pair = q_pair * rsqrtf(qss / (float)hd + eps) * (1.0f + tq_bf16_to_float(q_norm_w[(tid < 32) ? tid + 32 : tid - 32]));
            float q_rot = (tid < 32) ? -q_pair : q_pair;
            qv = qv * c + q_rot * s;
        }
        __syncthreads();
        qtmp[tid] = qv;
        __syncthreads();
        tq_fwht256(qtmp, tid);
        if (tid < 128) {
            int c16 = tid >> 3, tt = (tid >> 1) & 3, rsel = tid & 1;
            int d0 = 16 * c16 + 2 * tt + 8 * rsel;
            int l = 4 * (row & 7) + tt, rr = ((row >> 3) & 1) + 2 * rsel;
            sm[TQ_SEG_QA + (((row >> 4) * 16 + c16) * 32 + l) * 4 + rr] = tq_pack_bf16(qtmp[d0], qtmp[d0 + 1]);
        }
        __syncthreads();
    }
    float oacc[QT][4][4];
    #pragma unroll
    for (int q = 0; q < QT; q++)
        #pragma unroll
        for (int f = 0; f < 4; f++)
            #pragma unroll
            for (int j = 0; j < 4; j++) oacc[q][f][j] = 0.0f;
    const float rsc = rsqrtf((float)hd);
    const size_t bt_base = (size_t)slot * max_blocks;
    for (int it = it_lo; it < it_hi; it++) {
        int t0d = it << 7;
        int physd = block_table[bt_base + (t0d >> page_log)];
        tq_gqa_seg_supertile<VQ4, QT>(oacc, sm, tid, warp, lane, gr, t4, it, t0d, physd, pos_max, page, nrows, nkv, rsc,
                                      kcb4, cstride4, ksz4, szstride, vcb8, cstride, vscb);
    }
    // ---- output per row: S_t > 1 -> split partials (the per-column merge's layout); S_t == 1 -> final rows ----
    #pragma unroll
    for (int q = 0; q < QT; q++) {
        #pragma unroll
        for (int f = 0; f < 4; f++) {
            int cidx = warp * 32 + f * 8 + 2 * t4;
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                const int ch = VQ4 ? warp * 32 + 4 * (2 * t4 + j) + f : cidx + j;
                const int rl = q * 16 + gr, rh = rl + 8;
                if (rl < nrows && rSc[rl] > 1 && split < rSc[rl]) {
                    const int head = kv_head * group + (rl - (rl / group) * group);
                    size_t pidx = ((size_t)(head * ncols + rcol[rl]) * S + split);
                    part_acc[pidx * hd + ch] = oacc[q][f][j];
                }
                if (rh < nrows && rSc[rh] > 1 && split < rSc[rh]) {
                    const int head = kv_head * group + (rh - (rh / group) * group);
                    size_t pidx = ((size_t)(head * ncols + rcol[rh]) * S + split);
                    part_acc[pidx * hd + ch] = oacc[q][f][2 + j];
                }
            }
        }
    }
    if (tid < nrows && rSc[tid] > 1 && split < rSc[tid]) {
        const int head = kv_head * group + (tid - (tid / group) * group);
        size_t pidx = ((size_t)(head * ncols + rcol[tid]) * S + split);
        part_ml[pidx * 2 + 0] = mst[tid]; part_ml[pidx * 2 + 1] = lst[tid];
    }
    // rows with S_t == 1 (only split 0 reaches here for them): normalize (+ rotate back for int4 V) + gate
    __syncthreads();
    float *ob = (float *)sm;                                      // 8 rows x 256 (aliases the A fragments)
    for (int rb = 0; rb < nrows; rb += 8) {
        #pragma unroll
        for (int q = 0; q < QT; q++) {
            #pragma unroll
            for (int f = 0; f < 4; f++) {
                int cidx = warp * 32 + f * 8 + 2 * t4;
                #pragma unroll
                for (int j = 0; j < 2; j++) {
                    const int ch = VQ4 ? warp * 32 + 4 * (2 * t4 + j) + f : cidx + j;
                    const int rl = q * 16 + gr, rh = rl + 8;
                    if (rl >= rb && rl < rb + 8 && rl < nrows && rSc[rl] == 1) ob[(rl - rb) * 256 + ch] = oacc[q][f][j] * (1.0f / lst[rl]);
                    if (rh >= rb && rh < rb + 8 && rh < nrows && rSc[rh] == 1) ob[(rh - rb) * 256 + ch] = oacc[q][f][2 + j] * (1.0f / lst[rh]);
                }
            }
        }
        __syncthreads();
        {
            const int row = rb + warp;
            if (row < nrows && rSc[row] == 1) {
                const int t = row / group, r = row - t * group, col = rcol[row];
                const int hh = kv_head * group + r;
                const float *q_proj = q_proj_base + (size_t)col * q_m;
                float x[8];
                #pragma unroll
                for (int e = 0; e < 8; e++) x[e] = ob[warp * 256 + lane + 32 * e];
                if constexpr (VQ4) tq_fwht256_warp(x, lane);
                #pragma unroll
                for (int e = 0; e < 8; e++) {
                    const int d = lane + 32 * e;
                    float gate = q_proj[hh * (2 * hd) + hd + d];
                    out[(size_t)col * attn_m + hh * hd + d] = x[e] * (1.0f / (1.0f + expf(-gate)));
                }
            }
        }
        __syncthreads();
    }
}

// ===== fp16 int4 GQA attention (decode + verify; task 10, attention third pass) =====
// One kernel for the decode step (segments of one column) and the verify wave (sub-segments of <= QT*16/group
// columns of one slot): rows = (column t, q-head r) with their own causal limit / split range as in
// k_tq_paged_attn_q4_gqa_seg. Per 128-key super-tile and lane the base kernels spend ~1300 of ~2100 instructions
// dequantizing int4 to bf16 (extract, I2FP, -zp, *sc, pack per value); here:
//   V (always): the codes + (scale, zp) rows are cp.async-staged in smem; P is packed to fp16 and the per-key V scale
//      is folded into the P A-fragment (4 HMUL2 per 16-key tile: the warp's 32 channels are one scale group); the raw
//      nibbles are the B operand -- ((w & 0x000F000F) | 0x64006400) - 1024 = (n_a, n_b) exactly, odd f carry 16 n --
//      and the zero-point term C2 = sum_k p'_k zp_k comes from one extra HMMA per tile whose B column 0 holds zp_k;
//      out = (acc_f / 16^(f&1) - C2) / l. V enters exactly (no per-value rounding; p' = fp16(fp16(p) * sc) instead of
//      bf16(p) * bf16((n - zp) sc)): ~20 instead of ~90 instructions per 16-key tile per lane.
//   K (KF16): fp16 pairs via SHF + LOP3 (1024 + n), HSUB2 1024, HSUB2 zp, HMUL2 sc = 5 ops per pair instead of 10,
//      two roundings at 11 bits instead of one at 8; lane t4 takes channels (t4, t4 + 4) of each 8-channel word
//      (odd lanes carry 16 n and 16 zp; Q's odd channels are pre-divided by 16), Q in fp16 with the same permutation.
//      KF16 = 0 keeps the bf16 K path of the staged kernel (the P.V change alone).
// The decode step and the verify wave share this code, so verify == decode bitwise at the same client count holds
// for TQ_PAGED_ATTN_ST = 2 / 3 as it does for 0 / 1. int4 V, fp16 scales, dense walk only.
// smem (words): the Q-prep scratch aliases the P staging (dead by then), so the V stage fits under the 48 KB static cap.
#define TQ_SEG2_QA    0                     // [QT*16 chunks][32 lanes][4] Q A-frags (QT <= 2)
#define TQ_SEG2_PST   4096                  // [QT*8 tiles][32][4] P A-frag staging (fp16)
#define TQ_SEG2_QTMP  4096                  // [256] one row's normed + roped q (Q-prep only; aliases PST)
#define TQ_SEG2_RED   4352                  // [16] block-reduce scratch (Q-prep only; aliases PST)
#define TQ_SEG2_WMAX  6144                  // [8 warps][32 rows]
#define TQ_SEG2_WSUM  6400
#define TQ_SEG2_MST   6656                  // [32] running row max
#define TQ_SEG2_LST   6688                  // [32] running row denominator
#define TQ_SEG2_ALPH  6720                  // [32] per-row rescale
#define TQ_SEG2_RPOS  6752                  // [32] per row: position
#define TQ_SEG2_RLO   6784                  // [32] first tile of the row's split range
#define TQ_SEG2_RHI   6816                  // [32] one past the last
#define TQ_SEG2_RSC   6848                  // [32] the row's split count S_t
#define TQ_SEG2_RCOL  6880                  // [32] the row's wave column
#define TQ_SEG2_VST   6912                  // [128 rows][32 words] staged V codes (16-B aligned)
#define TQ_SEG2_VSS   11008                 // [128 rows][8 words] staged V (scale, zp) fp16 pairs
#define TQ_SEG2_WORDS 12032                 // 47 KB
#define TQ_F16_1024x2 0x64006400u

static __device__ __forceinline__ uint32_t tq_pack_f16(float lo, float hi) {
    uint32_t d;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
    return d;
}
static __device__ __forceinline__ uint32_t tq_hsub2(uint32_t a, uint32_t b) {
    uint32_t d;
    asm("sub.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
    return d;
}
static __device__ __forceinline__ uint32_t tq_hmul2(uint32_t a, uint32_t b) {
    uint32_t d;
    asm("mul.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
    return d;
}
static __device__ __forceinline__ void tq_mma_f16_m16n8k16(float *c, const uint32_t *a, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}
// int4 K pair of lane t4 from the 8-channel code word w: channels (t4, t4+4) (sh8 = 8*(t4>>1), lmask = odd t4 ?
// 0x00F000F0 : 0x000F000F); odd lanes get 16 (n - zp) (zp2 pre-scaled by 16). fp16x2 (n - zp) * sc.
static __device__ __forceinline__ uint32_t tq_q4_kpair_f16(uint32_t w, int sh8, uint32_t lmask, uint32_t zp2, uint32_t sc2) {
    uint32_t h = ((w >> sh8) & lmask) | TQ_F16_1024x2;
    h = tq_hsub2(h, TQ_F16_1024x2);
    h = tq_hsub2(h, zp2);
    return tq_hmul2(h, sc2);
}
// int4 V: the four B words (f = 0..3; odd f = 16 n) of a key pair from the two 16-bit code words of channels 4gr..4gr+3.
static __device__ __forceinline__ void tq_q4_vwords_f16(uint32_t ca, uint32_t cb, uint32_t b[4]) {
    uint32_t w01 = ca | (cb << 16), w23 = w01 >> 8;
    b[0] = tq_hsub2((w01 & 0x000F000Fu) | TQ_F16_1024x2, TQ_F16_1024x2);
    b[1] = tq_hsub2((w01 & 0x00F000F0u) | TQ_F16_1024x2, TQ_F16_1024x2);
    b[2] = tq_hsub2((w23 & 0x000F000Fu) | TQ_F16_1024x2, TQ_F16_1024x2);
    b[3] = tq_hsub2((w23 & 0x00F000F0u) | TQ_F16_1024x2, TQ_F16_1024x2);
}

template <int QT, int KF16>
static __device__ __forceinline__ void tq_gqa_tile_f16(
    float oacc[QT][4][4], float cacc[QT][4], uint32_t *sm, int tid, int warp, int lane, int gr, int t4,
    int it, int t0, int phys, int pos_max, int page, int nrows, int nkv, float rsc,
    const uint8_t *kcb4, size_t cstride4, const float *ksz4, size_t szstride,
    const uint8_t *vcb8, const float *vscb) {
    float *wmax = (float *)(sm + TQ_SEG2_WMAX), *wsum = (float *)(sm + TQ_SEG2_WSUM);
    float *mst = (float *)(sm + TQ_SEG2_MST), *lst = (float *)(sm + TQ_SEG2_LST), *alph = (float *)(sm + TQ_SEG2_ALPH);
    const int *rpos = (const int *)(sm + TQ_SEG2_RPOS), *rlo = (const int *)(sm + TQ_SEG2_RLO), *rhi = (const int *)(sm + TQ_SEG2_RHI);
    const size_t pr0 = (size_t)phys * page + (size_t)(t0 & (page - 1));
#define TQ_GROW(lp) (pr0 + (size_t)(min((lp), pos_max) - t0))
    {   // stage this tile's V codes + scales (rows clamped to pos_max like the direct reads)
        uint32_t *vst = sm + TQ_SEG2_VST, *vss = sm + TQ_SEG2_VSS;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            int id = tid + 256 * k, row = id >> 3, sub = id & 7;
            tq_cp_async16((uint32_t)__cvta_generic_to_shared(vst + row * 32 + sub * 4),
                          vcb8 + TQ_GROW(t0 + row) * cstride4 + 16 * sub);
        }
        {
            int row = tid >> 1, sub = tid & 1;
            tq_cp_async16((uint32_t)__cvta_generic_to_shared(vss + row * 8 + sub * 4),
                          (const __half *)q4s_adv(vscb, TQ_GROW(t0 + row) * szstride) + 8 * sub);
        }
        asm volatile("cp.async.commit_group;\n");
    }
    float s0[QT][4], s1[QT][4];
    #pragma unroll
    for (int q = 0; q < QT; q++)
        #pragma unroll
        for (int j = 0; j < 4; j++) { s0[q][j] = 0.0f; s1[q][j] = 0.0f; }
    const int tp = t0 + warp * 16;
    const int p0 = tp + gr, p1 = tp + 8 + gr;
    {
        const uint8_t *kr0 = kcb4 + TQ_GROW(p0) * cstride4;
        const uint8_t *kr1 = kcb4 + TQ_GROW(p1) * cstride4;
        const float *sz0 = q4s_adv(ksz4, TQ_GROW(p0) * szstride);
        const float *sz1 = q4s_adv(ksz4, TQ_GROW(p1) * szstride);
        const int sh = 8 * t4, sh8 = 8 * (t4 >> 1);
        const uint32_t lmask = (t4 & 1) ? 0x00F000F0u : 0x000F000Fu;
        const uint32_t k16 = (t4 & 1) ? 0x4C004C00u : 0x3C003C00u;      // fp16x2 (16, 16) / (1, 1)
        #pragma unroll
        for (int g = 0; g < 8; g++) {
            uint4 kp0 = *(const uint4 *)(kr0 + 16 * g);
            uint4 kp1 = *(const uint4 *)(kr1 + 16 * g);
            float sc0 = 0.0f, zp0 = 0.0f, sc1 = 0.0f, zp1 = 0.0f;
            uint32_t sc02 = 0, zp02 = 0, sc12 = 0, zp12 = 0;
            if constexpr (KF16) {
                uint32_t z0w = ((const uint32_t *)sz0)[g], z1w = ((const uint32_t *)sz1)[g];
                sc02 = __byte_perm(z0w, z0w, 0x1010); zp02 = tq_hmul2(__byte_perm(z0w, z0w, 0x3232), k16);
                sc12 = __byte_perm(z1w, z1w, 0x1010); zp12 = tq_hmul2(__byte_perm(z1w, z1w, 0x3232), k16);
            } else {
                float2 p0s = q4s_pair(sz0, 2 * g), p1s = q4s_pair(sz1, 2 * g);
                sc0 = p0s.x; zp0 = p0s.y; sc1 = p1s.x; zp1 = p1s.y;
            }
            #pragma unroll
            for (int cc = 0; cc < 2; cc++) {
                const int c = 2 * g + cc;
                const uint32_t wa0 = cc ? kp0.z : kp0.x, wa8 = cc ? kp0.w : kp0.y;
                const uint32_t we0 = cc ? kp1.z : kp1.x, we8 = cc ? kp1.w : kp1.y;
                uint32_t bk00, bk08, bk10, bk18;
                if constexpr (KF16) {
                    bk00 = tq_q4_kpair_f16(wa0, sh8, lmask, zp02, sc02); bk08 = tq_q4_kpair_f16(wa8, sh8, lmask, zp02, sc02);
                    bk10 = tq_q4_kpair_f16(we0, sh8, lmask, zp12, sc12); bk18 = tq_q4_kpair_f16(we8, sh8, lmask, zp12, sc12);
                } else {
                    uint8_t a0 = (uint8_t)(wa0 >> sh), a8 = (uint8_t)(wa8 >> sh);
                    uint8_t e0 = (uint8_t)(we0 >> sh), e8 = (uint8_t)(we8 >> sh);
                    bk00 = tq_pack_bf16(((float)(a0 & 15) - zp0) * sc0, ((float)(a0 >> 4) - zp0) * sc0);
                    bk08 = tq_pack_bf16(((float)(a8 & 15) - zp0) * sc0, ((float)(a8 >> 4) - zp0) * sc0);
                    bk10 = tq_pack_bf16(((float)(e0 & 15) - zp1) * sc1, ((float)(e0 >> 4) - zp1) * sc1);
                    bk18 = tq_pack_bf16(((float)(e8 & 15) - zp1) * sc1, ((float)(e8 >> 4) - zp1) * sc1);
                }
                #pragma unroll
                for (int q = 0; q < QT; q++) {
                    const uint32_t *a = &sm[TQ_SEG2_QA + ((q * 16 + c) * 32 + lane) * 4];
                    if constexpr (KF16) { tq_mma_f16_m16n8k16(s0[q], a, bk00, bk08); tq_mma_f16_m16n8k16(s1[q], a, bk10, bk18); }
                    else { tq_mma_bf16_m16n8k16(s0[q], a, bk00, bk08); tq_mma_bf16_m16n8k16(s1[q], a, bk10, bk18); }
                }
            }
        }
    }
    // scale + mask per ROW: keys > pos_t, tiles outside the row's split range, rows >= nrows -> -inf
    bool act_lo[QT], act_hi[QT];
    #pragma unroll
    for (int q = 0; q < QT; q++) {
        const int rl = q * 16 + gr, rh = rl + 8;
        act_lo[q] = rl < nrows && it >= rlo[rl] && it < rhi[rl];
        act_hi[q] = rh < nrows && it >= rlo[rh] && it < rhi[rh];
        const int pl = act_lo[q] ? rpos[rl] : -1, ph = act_hi[q] ? rpos[rh] : -1;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int k0 = tp + 2 * t4 + (j & 1), k1 = tp + 8 + 2 * t4 + (j & 1);
            bool rv = (j < 2) ? act_lo[q] : act_hi[q];
            int prow = (j < 2) ? pl : ph;
            s0[q][j] = (rv && k0 <= prow) ? s0[q][j] * rsc : TQ_AMM_NINF;
            s1[q][j] = (rv && k1 <= prow) ? s1[q][j] * rsc : TQ_AMM_NINF;
        }
        float lm_lo = fmaxf(fmaxf(s0[q][0], s0[q][1]), fmaxf(s1[q][0], s1[q][1]));
        float lm_hi = fmaxf(fmaxf(s0[q][2], s0[q][3]), fmaxf(s1[q][2], s1[q][3]));
        #pragma unroll
        for (int o = 1; o <= 2; o <<= 1) {
            lm_lo = fmaxf(lm_lo, __shfl_xor_sync(0xffffffffu, lm_lo, o));
            lm_hi = fmaxf(lm_hi, __shfl_xor_sync(0xffffffffu, lm_hi, o));
        }
        if (t4 == 0) { wmax[warp * 32 + rl] = lm_lo; wmax[warp * 32 + rh] = lm_hi; }
    }
    __syncthreads();
    if (tid < QT * 16) {
        const int row = tid;
        const bool act = row < nrows && it >= rlo[row] && it < rhi[row];
        float m_old = mst[row], m_new = m_old;
        for (int w = 0; w < 8; w++) m_new = fmaxf(m_new, wmax[w * 32 + row]);
        alph[row] = act ? expf(m_old - m_new) : 1.0f;
        mst[row] = act ? m_new : m_old;
    }
    __syncthreads();
    #pragma unroll
    for (int q = 0; q < QT; q++) {
        const int rl = q * 16 + gr, rh = rl + 8;
        float mn_lo = mst[rl], mn_hi = mst[rh];
        float p00 = act_lo[q] ? expf(s0[q][0] - mn_lo) : 0.0f, p01 = act_lo[q] ? expf(s0[q][1] - mn_lo) : 0.0f;
        float p02 = act_hi[q] ? expf(s0[q][2] - mn_hi) : 0.0f, p03 = act_hi[q] ? expf(s0[q][3] - mn_hi) : 0.0f;
        float p10 = act_lo[q] ? expf(s1[q][0] - mn_lo) : 0.0f, p11 = act_lo[q] ? expf(s1[q][1] - mn_lo) : 0.0f;
        float p12 = act_hi[q] ? expf(s1[q][2] - mn_hi) : 0.0f, p13 = act_hi[q] ? expf(s1[q][3] - mn_hi) : 0.0f;
        float ls_lo = p00 + p01 + p10 + p11, ls_hi = p02 + p03 + p12 + p13;
        #pragma unroll
        for (int o = 1; o <= 2; o <<= 1) {
            ls_lo += __shfl_xor_sync(0xffffffffu, ls_lo, o);
            ls_hi += __shfl_xor_sync(0xffffffffu, ls_hi, o);
        }
        if (t4 == 0) { wsum[warp * 32 + rl] = ls_lo; wsum[warp * 32 + rh] = ls_hi; }
        uint32_t *pst = &sm[TQ_SEG2_PST + ((q * 8 + warp) * 32 + lane) * 4];
        pst[0] = tq_pack_f16(p00, p01); pst[1] = tq_pack_f16(p02, p03);
        pst[2] = tq_pack_f16(p10, p11); pst[3] = tq_pack_f16(p12, p13);
    }
    asm volatile("cp.async.wait_group 0;\n");          // this tile's V has landed (per thread) ...
    __syncthreads();                                   // ... and is visible to every warp, as is P
    if (tid < QT * 16) {
        float l = lst[tid] * alph[tid];
        for (int w = 0; w < 8; w++) l += wsum[w * 32 + tid];
        lst[tid] = l;
    }
    #pragma unroll
    for (int q = 0; q < QT; q++) {
        float a_lo = alph[q * 16 + gr], a_hi = alph[q * 16 + 8 + gr];
        #pragma unroll
        for (int f = 0; f < 4; f++) { oacc[q][f][0] *= a_lo; oacc[q][f][1] *= a_lo; oacc[q][f][2] *= a_hi; oacc[q][f][3] *= a_hi; }
        cacc[q][0] *= a_lo; cacc[q][1] *= a_lo; cacc[q][2] *= a_hi; cacc[q][3] *= a_hi;
    }
    const uint16_t *vst16 = (const uint16_t *)(sm + TQ_SEG2_VST);
    const uint32_t *vssw = sm + TQ_SEG2_VSS;
    const int cbh = 8 * warp + gr;                     // 16-bit index of the lane's 4 channels in a 128-B row
    const uint32_t zsel = (gr == 0) ? 0xffffffffu : 0u; // the zp column of the C2 MMA lives in B column 0
    #pragma unroll 2
    for (int tt = 0; tt < 8; tt++) {
        const int r0 = tt * 16 + 2 * t4;
        const uint32_t c0 = vst16[r0 * 64 + cbh], c1 = vst16[(r0 + 1) * 64 + cbh];
        const uint32_t c8 = vst16[(r0 + 8) * 64 + cbh], c9 = vst16[(r0 + 9) * 64 + cbh];
        const uint32_t z0 = vssw[r0 * 8 + warp], z1 = vssw[(r0 + 1) * 8 + warp];
        const uint32_t z8 = vssw[(r0 + 8) * 8 + warp], z9 = vssw[(r0 + 9) * 8 + warp];
        const uint32_t sc_lo = __byte_perm(z0, z1, 0x5410), sc_hi = __byte_perm(z8, z9, 0x5410);
        const uint32_t zp_lo = __byte_perm(z0, z1, 0x7632) & zsel, zp_hi = __byte_perm(z8, z9, 0x7632) & zsel;
        uint32_t b0[4], b1[4];
        tq_q4_vwords_f16(c0, c1, b0);
        tq_q4_vwords_f16(c8, c9, b1);
        #pragma unroll
        for (int q = 0; q < QT; q++) {
            const uint32_t *pa = &sm[TQ_SEG2_PST + ((q * 8 + tt) * 32 + lane) * 4];
            uint32_t a[4] = {tq_hmul2(pa[0], sc_lo), tq_hmul2(pa[1], sc_lo), tq_hmul2(pa[2], sc_hi), tq_hmul2(pa[3], sc_hi)};
            tq_mma_f16_m16n8k16(cacc[q], a, zp_lo, zp_hi);
            #pragma unroll
            for (int f = 0; f < 4; f++) tq_mma_f16_m16n8k16(oacc[q][f], a, b0[f], b1[f]);
        }
    }
#undef TQ_GROW
    __syncthreads();
}

// grid (nkv, segments, S). seg_col0[seg], seg_n[seg]: sub-segments of <= QT*16/group consecutive columns of one slot
// (the decode step passes iota / ones: one column per segment); col_S: per-column split count (NULL = S);
// ncols = the wave's column count (partial layout). 2 blocks/SM (48 KB smem, <= 128 registers).
template <int QT, int KF16>
__global__ void __launch_bounds__(256, 2) k_tq_paged_attn_q4_gqa_f16(
    float *out, const float *q_proj_base, const uint16_t *q_norm_w,
    const uint8_t *k4_pool, const float *kq4s_pool, const uint8_t *v8_pool, const float *vscale_pool,
    const int *positions, const int *slot_ids, const int *col_S, const int *seg_col0, const int *seg_n,
    const int *block_table, int max_blocks, int page, int page_log,
    int nh, int nkv, int hd, int q_m, int attn_m, int S, int ncols,
    float eps, float rope_theta, float *part_acc, float *part_ml) {
    tq_pdl_enter();
    __shared__ __align__(16) uint32_t sm[TQ_SEG2_WORDS];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int gr = lane >> 2, t4 = lane & 3;
    const int kv_head = blockIdx.x, seg = blockIdx.y, split = blockIdx.z;
    const int group = nh / nkv;
    const int col0 = seg_col0[seg], n = seg_n[seg];
    const int nrows = n * group;
    const int slot = slot_ids[col0];
    const size_t cstride = (size_t)nkv * hd, cstride4 = cstride >> 1, szstride = (size_t)nkv * 16;
    const uint8_t *kcb4 = k4_pool + (size_t)kv_head * (hd >> 1);
    const float *ksz4 = q4s_adv(kq4s_pool, (size_t)kv_head * 16);
    const uint8_t *vcb8 = v8_pool + (size_t)kv_head * (hd >> 1);
    const float *vscb = q4s_adv(vscale_pool, (size_t)kv_head * 16);
    float *qtmp = (float *)(sm + TQ_SEG2_QTMP), *red = (float *)(sm + TQ_SEG2_RED);
    float *mst = (float *)(sm + TQ_SEG2_MST), *lst = (float *)(sm + TQ_SEG2_LST);
    int *rpos = (int *)(sm + TQ_SEG2_RPOS), *rlo = (int *)(sm + TQ_SEG2_RLO), *rhi = (int *)(sm + TQ_SEG2_RHI);
    int *rSc = (int *)(sm + TQ_SEG2_RSC), *rcol = (int *)(sm + TQ_SEG2_RCOL);
    for (int i = tid; i < TQ_SEG2_PST + QT * 1024; i += 256) sm[i] = 0;
    if (tid < 32) {
        mst[tid] = TQ_AMM_NINF; lst[tid] = 0.0f;
        int row = tid, lo = 0, hi = 0, pos = -1, Sc = 0, col = -1;
        if (row < nrows) {
            col = col0 + row / group; pos = positions[col]; Sc = col_S ? col_S[col] : S;
            int nst_all = (pos + 128) >> 7;
            hi = nst_all;
            if (Sc > 1) { int per = (nst_all + Sc - 1) / Sc; lo = split * per; hi = lo + per < nst_all ? lo + per : nst_all; }
            if (split >= Sc) { lo = 0; hi = 0; }
        }
        rpos[row] = pos; rlo[row] = lo; rhi[row] = hi; rSc[row] = Sc; rcol[row] = col;
    }
    __syncthreads();
    int it_lo = 1 << 30, it_hi = 0, pos_max = 0;
    for (int row = 0; row < nrows; row++) {
        if (rhi[row] > rlo[row]) { it_lo = min(it_lo, rlo[row]); it_hi = max(it_hi, rhi[row]); }
        pos_max = max(pos_max, rpos[row]);
    }
    // ---- Q prep per row: norm + rope at the column's position + FWHT -> A fragments (fp16 permuted for KF16)
    for (int row = 0; row < nrows; row++) {
        const int t = row / group, r = row - t * group, col = col0 + t, pos = rpos[row];
        const float *q_proj = q_proj_base + (size_t)col * q_m;
        const int hh = kv_head * group + r;
        float qv = q_proj[hh * (2 * hd) + tid];
        float qsum = qv * qv;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) qsum += __shfl_down_sync(0xffffffffu, qsum, o);
        if (lane == 0) red[warp] = qsum;
        __syncthreads();
        if (tid < 8) { float v = red[tid]; for (int o = 4; o > 0; o >>= 1) v += __shfl_down_sync(0xffu, v, o); if (tid == 0) red[8] = v; }
        __syncthreads();
        float qss = red[8];
        qv = qv * rsqrtf(qss / (float)hd + eps) * (1.0f + tq_bf16_to_float(q_norm_w[tid]));
        if (tid < 64) {
            int idx = tid & 31;
            float freq = powf(rope_theta, -((float)(2 * idx) / 64.0f));
            float angle = (float)pos * freq, c = cosf(angle), s = sinf(angle);
            float q_pair = q_proj[hh * (2 * hd) + ((tid < 32) ? tid + 32 : tid - 32)];
            q_pair = q_pair * rsqrtf(qss / (float)hd + eps) * (1.0f + tq_bf16_to_float(q_norm_w[(tid < 32) ? tid + 32 : tid - 32]));
            float q_rot = (tid < 32) ? -q_pair : q_pair;
            qv = qv * c + q_rot * s;
        }
        __syncthreads();
        qtmp[tid] = qv;
        __syncthreads();
        tq_fwht256(qtmp, tid);
        if (tid < 128) {
            int c16 = tid >> 3, tt = (tid >> 1) & 3, rsel = tid & 1;
            int l = 4 * (row & 7) + tt, rr = ((row >> 3) & 1) + 2 * rsel;
            uint32_t *dst = &sm[TQ_SEG2_QA + (((row >> 4) * 16 + c16) * 32 + l) * 4 + rr];
            if constexpr (KF16) {
                int d0 = 16 * c16 + tt + 8 * rsel;            // k pair (2tt, 2tt+1) <-> channels (d0, d0 + 4)
                float f = (tt & 1) ? 0.0625f : 1.0f;          // the K side carries 16 n on odd lanes
                *dst = tq_pack_f16(qtmp[d0] * f, qtmp[d0 + 4] * f);
            } else {
                int d0 = 16 * c16 + 2 * tt + 8 * rsel;
                *dst = tq_pack_bf16(qtmp[d0], qtmp[d0 + 1]);
            }
        }
        __syncthreads();
    }
    float oacc[QT][4][4], cacc[QT][4];
    #pragma unroll
    for (int q = 0; q < QT; q++) {
        #pragma unroll
        for (int f = 0; f < 4; f++)
            #pragma unroll
            for (int j = 0; j < 4; j++) oacc[q][f][j] = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; j++) cacc[q][j] = 0.0f;
    }
    const float rsc = rsqrtf((float)hd);
    const size_t bt_base = (size_t)slot * max_blocks;
    for (int it = it_lo; it < it_hi; it++) {
        int t0d = it << 7;
        int physd = block_table[bt_base + (t0d >> page_log)];
        tq_gqa_tile_f16<QT, KF16>(oacc, cacc, sm, tid, warp, lane, gr, t4, it, t0d, physd, pos_max, page, nrows, nkv, rsc,
                                  kcb4, cstride4, ksz4, szstride, vcb8, vscb);
    }
    // zero-point corrections per row (C2 of rows gr / gr+8 sits in lane t4 = 0 of the quad) and the 1/16 of odd f
    float c2lo[QT], c2hi[QT];
    #pragma unroll
    for (int q = 0; q < QT; q++) {
        c2lo[q] = __shfl_sync(0xffffffffu, cacc[q][0], lane & ~3);
        c2hi[q] = __shfl_sync(0xffffffffu, cacc[q][2], lane & ~3);
    }
#define TQ_F16_OUT(q, f, j) ((oacc[q][f][j] * (((f) & 1) ? 0.0625f : 1.0f)) - (((j) < 2) ? c2lo[q] : c2hi[q]))
    // ---- output per row: S_t > 1 -> split partials (the per-column merge's layout); S_t == 1 -> final rows ----
    #pragma unroll
    for (int q = 0; q < QT; q++) {
        #pragma unroll
        for (int f = 0; f < 4; f++) {
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                const int ch = warp * 32 + 4 * (2 * t4 + j) + f;
                const int rl = q * 16 + gr, rh = rl + 8;
                if (rl < nrows && rSc[rl] > 1 && split < rSc[rl]) {
                    const int head = kv_head * group + (rl - (rl / group) * group);
                    size_t pidx = ((size_t)(head * ncols + rcol[rl]) * S + split);
                    part_acc[pidx * hd + ch] = TQ_F16_OUT(q, f, j);
                }
                if (rh < nrows && rSc[rh] > 1 && split < rSc[rh]) {
                    const int head = kv_head * group + (rh - (rh / group) * group);
                    size_t pidx = ((size_t)(head * ncols + rcol[rh]) * S + split);
                    part_acc[pidx * hd + ch] = TQ_F16_OUT(q, f, 2 + j);
                }
            }
        }
    }
    if (tid < nrows && rSc[tid] > 1 && split < rSc[tid]) {
        const int head = kv_head * group + (tid - (tid / group) * group);
        size_t pidx = ((size_t)(head * ncols + rcol[tid]) * S + split);
        part_ml[pidx * 2 + 0] = mst[tid]; part_ml[pidx * 2 + 1] = lst[tid];
    }
    __syncthreads();
    float *ob = (float *)sm;                                      // 8 rows x 256 (aliases the A fragments)
    for (int rb = 0; rb < nrows; rb += 8) {
        #pragma unroll
        for (int q = 0; q < QT; q++) {
            #pragma unroll
            for (int f = 0; f < 4; f++) {
                #pragma unroll
                for (int j = 0; j < 2; j++) {
                    const int ch = warp * 32 + 4 * (2 * t4 + j) + f;
                    const int rl = q * 16 + gr, rh = rl + 8;
                    if (rl >= rb && rl < rb + 8 && rl < nrows && rSc[rl] == 1) ob[(rl - rb) * 256 + ch] = TQ_F16_OUT(q, f, j) * (1.0f / lst[rl]);
                    if (rh >= rb && rh < rb + 8 && rh < nrows && rSc[rh] == 1) ob[(rh - rb) * 256 + ch] = TQ_F16_OUT(q, f, 2 + j) * (1.0f / lst[rh]);
                }
            }
        }
        __syncthreads();
        {
            const int row = rb + warp;
            if (row < nrows && rSc[row] == 1) {
                const int t = row / group, r = row - t * group, col = rcol[row];
                const int hh = kv_head * group + r;
                const float *q_proj = q_proj_base + (size_t)col * q_m;
                float x[8];
                #pragma unroll
                for (int e = 0; e < 8; e++) x[e] = ob[warp * 256 + lane + 32 * e];
                tq_fwht256_warp(x, lane);
                #pragma unroll
                for (int e = 0; e < 8; e++) {
                    const int d = lane + 32 * e;
                    float gate = q_proj[hh * (2 * hd) + hd + d];
                    out[(size_t)col * attn_m + hh * hd + d] = x[e] * (1.0f / (1.0f + expf(-gate)));
                }
            }
        }
        __syncthreads();
    }
#undef TQ_F16_OUT
}

// ===== TQ_KV_T4 twin of the segment-aware verify attention (B2-eng, session 10) =====
// Same block geometry and per-row bookkeeping as k_tq_paged_attn_q4_gqa_seg; the slot's keys are trellis rows
// streamed once for all the sub-segment's columns (table dequant), and each row's OWN key (the column's token at
// pos_t) is exact fp16 from the staging -- the per-column T4 kernel's semantics (rows < pos trellis, row pos exact).
// Diagonal patch for the super-tile(s) holding the segment's positions [seg_pos0, seg_pos1]:
//  (1) Q.K^T runs two streams there: the shared trellis stream and an "exact" stream whose B columns at the
//      segment's positions carry the exact fp16 keys; a score element takes the exact value iff the key is the row's
//      own position. An MMA output element depends only on its A row and B column, so the selected element is bitwise
//      the per-column kernel's (which has the exact key in that B column).
//  (2) P.V for a 16-key chunk holding own keys runs one MMA variant per own key kk (B = exact V at kk, trellis
//      elsewhere) and each row takes the variant of its own key -- the per-column kernel's B for that row -- while rows
//      whose own key lies elsewhere take the plain trellis MMA. Every other chunk / super-tile is the shared stream.
// F16 = 1 (TQ_T4_ATTN_F16, the attention third pass on the trellis pool): Q, K, P and V as fp16 MMA operands. The K
// and V operands are the table's fp16 values themselves (no per-element convert / multiply / bf16 pack); a key's
// scale multiplies its score column after Q.K^T (one FMUL per score, folded into the rsc step) and a value row's
// scale is folded into P (4 HMUL2 per 16 keys, as the int4 fp16 kernel does); exact own keys (fp16 staging) are the
// operands as stored with scale 1. F16 = 0 is the bf16 path above (bitwise the per-column kernel's).
// Shared-memory layout of the T4 segment kernel, sized by QT (the bf16 segment kernel's layout above is QT = 2's);
// the dequant table follows it (tq_t4_tab, TR copies). MEASURED 2026-10-16 (notes 12): TR = 4 (32 KB) cut the
// lookups' LDS wavefronts by only 5 % (3.39 -> 3.22 per instruction) and cost +32 % instructions for the index math
// -> +8 % kernel time; TR = 1 is the default. V staging (VST/VSS, QT = 1 only): the super-tile's 128 V code rows at a
// 152-byte stride (38 words: rows r0, r0+1, r0+8, r0+9 of a lane quad land on disjoint banks) + the 128 side words,
// cp.async'd during Q.K^T (one 16 KB copy per super-tile instead of 96 per-key 1/2/4-byte loads per thread).
// QT = 1: 14.9 + 19.5 + 8 KB = 42.4 KB; QT = 2: 28.7 + 8 KB -- both static, both at 2 blocks/SM.
#ifndef TQ_T4_TABREP
#define TQ_T4_TABREP 1
#endif
#ifndef TQ_T4_VSTAGE
#define TQ_T4_VSTAGE 1
#endif
template <int QT> struct tq_t4_seg_sm {
    static constexpr int ROWS = QT * 16;
    static constexpr int TR   = (QT == 1) ? TQ_T4_TABREP : ((TQ_T4_TABREP >= 2) ? 2 : 1);
    static constexpr bool VSTAGE = (QT == 1) && (TQ_T4_VSTAGE != 0);
    static constexpr int VROW = 38;                        // words per staged V code row (128 B + 24 B pad)
    static constexpr int QA   = 0;                        // [QT*16 chunks][32 lanes][4] Q A-frags
    static constexpr int PST  = QA + QT * 2048;           // [QT*8 tiles][32][4] P A-frag staging
    static constexpr int WMAX = PST + QT * 1024;          // [8 warps][ROWS]
    static constexpr int WSUM = WMAX + 8 * ROWS;
    static constexpr int MST  = WSUM + 8 * ROWS;          // [ROWS] running row max
    static constexpr int LST  = MST + ROWS;               // [ROWS] running row denominator
    static constexpr int ALPH = LST + ROWS;               // [ROWS] per-row rescale
    static constexpr int RPOS = ALPH + ROWS;              // [ROWS] per row: position
    static constexpr int RLO  = RPOS + ROWS;              // first tile of the row's split range
    static constexpr int RHI  = RLO + ROWS;               // one past the last
    static constexpr int RSC  = RHI + ROWS;               // the row's split count S_t
    static constexpr int RCOL = RSC + ROWS;               // the row's wave column
    static constexpr int QTMP = RCOL + ROWS;              // [256]
    static constexpr int RED  = QTMP + 256;               // [16]
    static constexpr int VST  = RED + 16;                 // [128 rows][VROW] staged V codes (QT = 1)
    static constexpr int VSS  = VST + (VSTAGE ? 128 * VROW : 0);   // [128] staged V side words
    static constexpr int WORDS = VSS + (VSTAGE ? 128 : 0);
};
template <int QT, int F16>
static __device__ __forceinline__ void tq_gqa_seg_supertile_t4(
    float oacc[QT][4][4], uint32_t *sm, int tid, int warp, int lane, int gr, int t4,
    int it, int t0, int phys, int pos_max, int page, int nrows, int nkv, float rsc,
    const uint8_t *kcb, const uint8_t *kside, const uint8_t *vcb, const uint8_t *vside, int kv_head,
    const uint16_t *tab, const uint16_t *f16s, int slot, int seg_pos0, int seg_pos1) {
    typedef tq_t4_seg_sm<QT> L;
    constexpr int TR = L::TR, ROWS = L::ROWS;
    const uint32_t cr = tq_t4_tab<TR>::cr(lane);
    float *wmax = (float *)(sm + L::WMAX), *wsum = (float *)(sm + L::WSUM);
    float *mst = (float *)(sm + L::MST), *lst = (float *)(sm + L::LST), *alph = (float *)(sm + L::ALPH);
    const int *rpos = (const int *)(sm + L::RPOS), *rlo = (const int *)(sm + L::RLO), *rhi = (const int *)(sm + L::RHI);
    const size_t pr0 = (size_t)phys * page + (size_t)(t0 & (page - 1));
    const size_t cstride4 = (size_t)nkv * 128;
    const size_t stg0 = (size_t)slot * page;                      // staging row of position p: stg0 + (p & (page-1))
#define TQ_GROW(lp) (pr0 + (size_t)(min((lp), pos_max) - t0))
#define TQ_STGK(p) (f16s + ((stg0 + (size_t)((p) & (page - 1))) * nkv + kv_head) * 512)
    const bool diag = seg_pos1 >= t0 && seg_pos0 < t0 + 128;      // block-uniform
    if constexpr (L::VSTAGE) {
        // stage this super-tile's V codes (128 rows x 128 B, 8-byte copies into the 152-byte-stride rows) and side
        // words (rows clamped to pos_max like the direct reads); lands during Q.K^T, waited for before P.V
        uint32_t *vst = sm + L::VST, *vss = sm + L::VSS;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            const int id = tid + 256 * k, row = id >> 4, sub = id & 15;
            tq_cp_async8((uint32_t)__cvta_generic_to_shared(vst + row * L::VROW + sub * 2),
                         vcb + TQ_GROW(t0 + row) * cstride4 + 8 * sub);
        }
        if (tid < 128) tq_cp_async4((uint32_t)__cvta_generic_to_shared(vss + tid), vside + (TQ_GROW(t0 + tid) * nkv + kv_head) * 4);
        asm volatile("cp.async.commit_group;\n");
    }
    float s0[QT][4], s1[QT][4];
    #pragma unroll
    for (int q = 0; q < QT; q++)
        #pragma unroll
        for (int j = 0; j < 4; j++) { s0[q][j] = 0.0f; s1[q][j] = 0.0f; }
    const int tp = t0 + warp * 16;
    const int p0 = tp + gr, p1 = tp + 8 + gr;
    {
        const size_t r0 = TQ_GROW(p0), r1 = TQ_GROW(p1);
        const uint8_t *kr0 = kcb + r0 * cstride4, *kr1 = kcb + r1 * cstride4;
        const uint32_t sw0 = tq_t4_side(kside, r0, nkv, kv_head), sw1 = tq_t4_side(kside, r1, nkv, kv_head);
        const float sc0 = tq_t4_scale(sw0), sc1 = tq_t4_scale(sw1);
        uint32_t prev0 = sw0 & 0xFFu, prev1 = sw1 & 0xFFu;
        // F16: the scales of the C fragment's keys (tp + 2 t4 + j and tp + 8 + 2 t4 + j) live in the quad of lane 4 (2 t4 + j)
        float ksc0[2] = {1.0f, 1.0f}, ksc1[2] = {1.0f, 1.0f};
        if constexpr (F16) {
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                ksc0[j] = __shfl_sync(0xffffffffu, sc0, (2 * t4 + j) * 4);
                ksc1[j] = __shfl_sync(0xffffffffu, sc1, (2 * t4 + j) * 4);
            }
        }
        if (!diag) {
            #pragma unroll
            for (int g = 0; g < 8; g++) {
                const uint4 kp0 = *(const uint4 *)(kr0 + 16 * g);
                const uint4 kp1 = *(const uint4 *)(kr1 + 16 * g);
                uint32_t bk0[2], bk8[2], ek0[2], ek8[2];
                if constexpr (F16) { tq_t4_kfrags_raw<TR>(kp0, prev0, t4, tab, bk0, bk8, cr); tq_t4_kfrags_raw<TR>(kp1, prev1, t4, tab, ek0, ek8, cr); }
                else { tq_t4_kfrags<TR>(kp0, prev0, t4, sc0, tab, bk0, bk8, cr); tq_t4_kfrags<TR>(kp1, prev1, t4, sc1, tab, ek0, ek8, cr); }
                prev0 = kp0.w >> 24; prev1 = kp1.w >> 24;
                #pragma unroll
                for (int cc = 0; cc < 2; cc++) {
                    const int c = 2 * g + cc;
                    #pragma unroll
                    for (int q = 0; q < QT; q++) {
                        const uint32_t *a = &sm[L::QA + ((q * 16 + c) * 32 + lane) * 4];
                        if constexpr (F16) { tq_mma_f16_m16n8k16(s0[q], a, bk0[cc], bk8[cc]); tq_mma_f16_m16n8k16(s1[q], a, ek0[cc], ek8[cc]); }
                        else { tq_mma_bf16_m16n8k16(s0[q], a, bk0[cc], bk8[cc]); tq_mma_bf16_m16n8k16(s1[q], a, ek0[cc], ek8[cc]); }
                    }
                }
            }
            if constexpr (F16) {
                #pragma unroll
                for (int q = 0; q < QT; q++)
                    #pragma unroll
                    for (int j = 0; j < 4; j++) { s0[q][j] *= ksc0[j & 1]; s1[q][j] *= ksc1[j & 1]; }
            }
        } else {
            // the diagonal super-tile: trellis stream (s) + exact stream (x) whose B columns at the segment's
            // positions are the exact fp16 keys; select per element afterwards
            float x0[QT][4], x1[QT][4];
            #pragma unroll
            for (int q = 0; q < QT; q++)
                #pragma unroll
                for (int j = 0; j < 4; j++) { x0[q][j] = 0.0f; x1[q][j] = 0.0f; }
            const bool own0 = p0 >= seg_pos0 && p0 <= seg_pos1, own1 = p1 >= seg_pos0 && p1 <= seg_pos1;
            const uint16_t *kf0 = TQ_STGK(p0), *kf1 = TQ_STGK(p1);
            #pragma unroll
            for (int g = 0; g < 8; g++) {
                const uint4 kp0 = *(const uint4 *)(kr0 + 16 * g);
                const uint4 kp1 = *(const uint4 *)(kr1 + 16 * g);
                uint32_t bk0[2], bk8[2], ek0[2], ek8[2];
                if constexpr (F16) { tq_t4_kfrags_raw<TR>(kp0, prev0, t4, tab, bk0, bk8, cr); tq_t4_kfrags_raw<TR>(kp1, prev1, t4, tab, ek0, ek8, cr); }
                else { tq_t4_kfrags<TR>(kp0, prev0, t4, sc0, tab, bk0, bk8, cr); tq_t4_kfrags<TR>(kp1, prev1, t4, sc1, tab, ek0, ek8, cr); }
                prev0 = kp0.w >> 24; prev1 = kp1.w >> 24;
                #pragma unroll
                for (int cc = 0; cc < 2; cc++) {
                    const int c = 2 * g + cc, ch = 16 * c + 2 * t4;
                    uint32_t xb0, xb8, xe0, xe8;
                    if constexpr (F16) {                              // exact keys: the fp16 pairs as stored
                        xb0 = own0 ? *(const uint32_t *)(kf0 + ch) : bk0[cc];
                        xb8 = own0 ? *(const uint32_t *)(kf0 + ch + 8) : bk8[cc];
                        xe0 = own1 ? *(const uint32_t *)(kf1 + ch) : ek0[cc];
                        xe8 = own1 ? *(const uint32_t *)(kf1 + ch + 8) : ek8[cc];
                    } else {
                        xb0 = own0 ? tq_t4_f16x2_bf16(*(const uint32_t *)(kf0 + ch)) : bk0[cc];
                        xb8 = own0 ? tq_t4_f16x2_bf16(*(const uint32_t *)(kf0 + ch + 8)) : bk8[cc];
                        xe0 = own1 ? tq_t4_f16x2_bf16(*(const uint32_t *)(kf1 + ch)) : ek0[cc];
                        xe8 = own1 ? tq_t4_f16x2_bf16(*(const uint32_t *)(kf1 + ch + 8)) : ek8[cc];
                    }
                    #pragma unroll
                    for (int q = 0; q < QT; q++) {
                        const uint32_t *a = &sm[L::QA + ((q * 16 + c) * 32 + lane) * 4];
                        if constexpr (F16) {
                            tq_mma_f16_m16n8k16(s0[q], a, bk0[cc], bk8[cc]);
                            tq_mma_f16_m16n8k16(s1[q], a, ek0[cc], ek8[cc]);
                            tq_mma_f16_m16n8k16(x0[q], a, xb0, xb8);
                            tq_mma_f16_m16n8k16(x1[q], a, xe0, xe8);
                        } else {
                            tq_mma_bf16_m16n8k16(s0[q], a, bk0[cc], bk8[cc]);
                            tq_mma_bf16_m16n8k16(s1[q], a, ek0[cc], ek8[cc]);
                            tq_mma_bf16_m16n8k16(x0[q], a, xb0, xb8);
                            tq_mma_bf16_m16n8k16(x1[q], a, xe0, xe8);
                        }
                    }
                }
            }
            // a row's own key takes the exact score (C-fragment layout: j < 2 -> row gr, keys 2t4 + (j&1); j >= 2 -> row gr+8);
            // F16: every other score takes its key's scale here (the exact key's scale is 1)
            #pragma unroll
            for (int q = 0; q < QT; q++) {
                const int rl = q * 16 + gr, rh = rl + 8;
                const int pl = rl < nrows ? rpos[rl] : -1, ph = rh < nrows ? rpos[rh] : -1;
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    const int k0 = tp + 2 * t4 + (j & 1), k1 = tp + 8 + 2 * t4 + (j & 1);
                    const int prow = (j < 2) ? pl : ph;
                    if constexpr (F16) {
                        s0[q][j] = (k0 == prow) ? x0[q][j] : s0[q][j] * ksc0[j & 1];
                        s1[q][j] = (k1 == prow) ? x1[q][j] : s1[q][j] * ksc1[j & 1];
                    } else {
                        if (k0 == prow) s0[q][j] = x0[q][j];
                        if (k1 == prow) s1[q][j] = x1[q][j];
                    }
                }
            }
        }
    }
    // scale + mask per ROW: keys > pos_t, tiles outside the row's split range, and rows >= nrows go to -inf
    bool act_lo[QT], act_hi[QT];
    #pragma unroll
    for (int q = 0; q < QT; q++) {
        const int rl = q * 16 + gr, rh = rl + 8;
        act_lo[q] = rl < nrows && it >= rlo[rl] && it < rhi[rl];
        act_hi[q] = rh < nrows && it >= rlo[rh] && it < rhi[rh];
        const int pl = act_lo[q] ? rpos[rl] : -1, ph = act_hi[q] ? rpos[rh] : -1;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int k0 = tp + 2 * t4 + (j & 1), k1 = tp + 8 + 2 * t4 + (j & 1);
            bool rv = (j < 2) ? act_lo[q] : act_hi[q];
            int prow = (j < 2) ? pl : ph;
            s0[q][j] = (rv && k0 <= prow) ? s0[q][j] * rsc : TQ_AMM_NINF;
            s1[q][j] = (rv && k1 <= prow) ? s1[q][j] * rsc : TQ_AMM_NINF;
        }
        float lm_lo = fmaxf(fmaxf(s0[q][0], s0[q][1]), fmaxf(s1[q][0], s1[q][1]));
        float lm_hi = fmaxf(fmaxf(s0[q][2], s0[q][3]), fmaxf(s1[q][2], s1[q][3]));
        #pragma unroll
        for (int o = 1; o <= 2; o <<= 1) {
            lm_lo = fmaxf(lm_lo, __shfl_xor_sync(0xffffffffu, lm_lo, o));
            lm_hi = fmaxf(lm_hi, __shfl_xor_sync(0xffffffffu, lm_hi, o));
        }
        if (t4 == 0) { wmax[warp * ROWS + rl] = lm_lo; wmax[warp * ROWS + rh] = lm_hi; }
    }
    __syncthreads();
    if (tid < QT * 16) {
        const int row = tid;
        const bool act = row < nrows && it >= rlo[row] && it < rhi[row];
        float m_old = mst[row], m_new = m_old;
        for (int w = 0; w < 8; w++) m_new = fmaxf(m_new, wmax[w * ROWS + row]);
        alph[row] = act ? expf(m_old - m_new) : 1.0f;
        mst[row] = act ? m_new : m_old;
    }
    __syncthreads();
    #pragma unroll
    for (int q = 0; q < QT; q++) {
        const int rl = q * 16 + gr, rh = rl + 8;
        float mn_lo = mst[rl], mn_hi = mst[rh];
        float p00 = act_lo[q] ? expf(s0[q][0] - mn_lo) : 0.0f, p01 = act_lo[q] ? expf(s0[q][1] - mn_lo) : 0.0f;
        float p02 = act_hi[q] ? expf(s0[q][2] - mn_hi) : 0.0f, p03 = act_hi[q] ? expf(s0[q][3] - mn_hi) : 0.0f;
        float p10 = act_lo[q] ? expf(s1[q][0] - mn_lo) : 0.0f, p11 = act_lo[q] ? expf(s1[q][1] - mn_lo) : 0.0f;
        float p12 = act_hi[q] ? expf(s1[q][2] - mn_hi) : 0.0f, p13 = act_hi[q] ? expf(s1[q][3] - mn_hi) : 0.0f;
        float ls_lo = p00 + p01 + p10 + p11, ls_hi = p02 + p03 + p12 + p13;
        #pragma unroll
        for (int o = 1; o <= 2; o <<= 1) {
            ls_lo += __shfl_xor_sync(0xffffffffu, ls_lo, o);
            ls_hi += __shfl_xor_sync(0xffffffffu, ls_hi, o);
        }
        if (t4 == 0) { wsum[warp * ROWS + rl] = ls_lo; wsum[warp * ROWS + rh] = ls_hi; }
        uint32_t *pst = &sm[L::PST + ((q * 8 + warp) * 32 + lane) * 4];
        if constexpr (F16) {
            pst[0] = tq_pack_f16(p00, p01); pst[1] = tq_pack_f16(p02, p03);
            pst[2] = tq_pack_f16(p10, p11); pst[3] = tq_pack_f16(p12, p13);
        } else {
            pst[0] = tq_pack_bf16(p00, p01); pst[1] = tq_pack_bf16(p02, p03);
            pst[2] = tq_pack_bf16(p10, p11); pst[3] = tq_pack_bf16(p12, p13);
        }
    }
    __syncthreads();
    if (tid < QT * 16) {
        float l = lst[tid] * alph[tid];
        for (int w = 0; w < 8; w++) l += wsum[w * ROWS + tid];
        lst[tid] = l;
    }
    #pragma unroll
    for (int q = 0; q < QT; q++) {
        float a_lo = alph[q * 16 + gr], a_hi = alph[q * 16 + 8 + gr];
        #pragma unroll
        for (int f = 0; f < 4; f++) { oacc[q][f][0] *= a_lo; oacc[q][f][1] *= a_lo; oacc[q][f][2] *= a_hi; oacc[q][f][3] *= a_hi; }
    }
    const int cb = 16 * warp + 2 * gr;                            // stream byte of the thread's first channel
    if constexpr (L::VSTAGE) {
        asm volatile("cp.async.wait_group 0;\n");                // this super-tile's V has landed (per thread) ...
        __syncthreads();                                          // ... and is visible to every warp
    }
    const uint32_t *vstw = sm + L::VST, *vssw = sm + L::VSS;
    for (int tt = 0; tt < 8; tt++) {
        const int r0 = t0 + tt * 16 + 2 * t4;
        float v[4][4];                                            // [key q0,q1,q8,q9][f]: trellis V (bf16 path, dequantized)
        uint32_t vr[4][4];                                        // F16: the table's fp16 bits
        uint32_t zs[4];
        #pragma unroll
        for (int rr = 0; rr < 4; rr++) {
            uint32_t z, c;
            if constexpr (L::VSTAGE) {
                // staged: row-local index (the clamp to pos_max happened at the copy), 3 code bytes cb-1 .. cb+1 from the
                // two words around them (cb = 0: the prefix byte of the side word)
                const int lr = tt * 16 + 2 * t4 + (rr >> 1) * 8 + (rr & 1);
                z = vssw[lr];
                const uint32_t *rw = vstw + lr * L::VROW;
                if (cb == 0) {
                    c = (z & 0xFFu) | ((rw[0] & 0xFFFFu) << 8);
                } else {
                    const int o = cb - 1;
                    const uint32_t w0 = rw[o >> 2], w1 = rw[(o >> 2) + 1];
                    c = __funnelshift_r(w0, w1, 8 * (o & 3)) & 0xFFFFFFu;
                }
            } else {
                const size_t qq = TQ_GROW(r0 + (rr >> 1) * 8 + (rr & 1));
                z = tq_t4_side(vside, qq, nkv, kv_head);
                c = tq_t4_vbits(vcb + qq * cstride4, z, cb);
            }
            zs[rr] = z;
            if constexpr (F16) {
                #pragma unroll
                for (int f = 0; f < 4; f++) vr[rr][f] = tq_t4_val_raw<TR>(c, f, tab, cr);
            } else {
                const float vs = tq_t4_scale(z);
                #pragma unroll
                for (int f = 0; f < 4; f++) v[rr][f] = tq_t4_val<TR>(c, f, tab, vs, cr);
            }
        }
        // F16: the V scales of the thread's key pairs (r0, r0+1) / (r0+8, r0+9) as fp16x2, folded into the P pairs
        const uint32_t scv_lo = F16 ? tq_t4_scale2(zs[0], zs[1]) : 0u, scv_hi = F16 ? tq_t4_scale2(zs[2], zs[3]) : 0u;
        const int ca = max(seg_pos0, t0 + 16 * tt), ckb = min(seg_pos1, t0 + 16 * tt + 15);
        if (!(diag && ca <= ckb)) {                               // no own key in this chunk: the shared stream
            #pragma unroll
            for (int f = 0; f < 4; f++) {
                const uint32_t b0 = F16 ? (vr[0][f] | (vr[1][f] << 16)) : tq_pack_bf16(v[0][f], v[1][f]);
                const uint32_t b1 = F16 ? (vr[2][f] | (vr[3][f] << 16)) : tq_pack_bf16(v[2][f], v[3][f]);
                #pragma unroll
                for (int q = 0; q < QT; q++) {
                    const uint32_t *pa = &sm[L::PST + ((q * 8 + tt) * 32 + lane) * 4];
                    if constexpr (F16) {
                        uint32_t a[4] = {tq_hmul2(pa[0], scv_lo), tq_hmul2(pa[1], scv_lo), tq_hmul2(pa[2], scv_hi), tq_hmul2(pa[3], scv_hi)};
                        tq_mma_f16_m16n8k16(oacc[q][f], a, b0, b1);
                    } else {
                        uint32_t a[4] = {pa[0], pa[1], pa[2], pa[3]};
                        tq_mma_bf16_m16n8k16(oacc[q][f], a, b0, b1);
                    }
                }
            }
            continue;
        }
        // own keys ca..ckb in this chunk: per (q, f) the plain MMA for the rows whose own key lies elsewhere, then one
        // variant MMA per own key kk (exact V at kk, P scale 1 there) for the rows owning kk
        #pragma unroll
        for (int q = 0; q < QT; q++) {
            const uint32_t *pa = &sm[L::PST + ((q * 8 + tt) * 32 + lane) * 4];
            uint32_t a[4];
            if constexpr (F16) { a[0] = tq_hmul2(pa[0], scv_lo); a[1] = tq_hmul2(pa[1], scv_lo); a[2] = tq_hmul2(pa[2], scv_hi); a[3] = tq_hmul2(pa[3], scv_hi); }
            else { a[0] = pa[0]; a[1] = pa[1]; a[2] = pa[2]; a[3] = pa[3]; }
            const int rl = q * 16 + gr, rh = rl + 8;
            const int pl = rl < nrows ? rpos[rl] : -1, ph = rh < nrows ? rpos[rh] : -1;
            #pragma unroll
            for (int f = 0; f < 4; f++) {
                float res[4] = {oacc[q][f][0], oacc[q][f][1], oacc[q][f][2], oacc[q][f][3]};
                const uint32_t b0 = F16 ? (vr[0][f] | (vr[1][f] << 16)) : tq_pack_bf16(v[0][f], v[1][f]);
                const uint32_t b1 = F16 ? (vr[2][f] | (vr[3][f] << 16)) : tq_pack_bf16(v[2][f], v[3][f]);
                if constexpr (F16) tq_mma_f16_m16n8k16(res, a, b0, b1); else tq_mma_bf16_m16n8k16(res, a, b0, b1);
                for (int kk = ca; kk <= ckb; kk++) {
                    const int rel = kk - r0;                      // the thread's keys: r0, r0+1, r0+8, r0+9
                    float t[4] = {oacc[q][f][0], oacc[q][f][1], oacc[q][f][2], oacc[q][f][3]};
                    if constexpr (F16) {
                        uint32_t ve = 0u;
                        if (rel == 0 || rel == 1 || rel == 8 || rel == 9) ve = (uint32_t)TQ_STGK(kk)[256 + 2 * cb + f];
                        const uint32_t vb0 = (rel == 0) ? (ve | (vr[1][f] << 16)) : (rel == 1) ? (vr[0][f] | (ve << 16)) : b0;
                        const uint32_t vb1 = (rel == 8) ? (ve | (vr[3][f] << 16)) : (rel == 9) ? (vr[2][f] | (ve << 16)) : b1;
                        const uint32_t sl = (rel == 0) ? ((scv_lo & 0xFFFF0000u) | 0x3C00u) : (rel == 1) ? ((scv_lo & 0xFFFFu) | 0x3C000000u) : scv_lo;
                        const uint32_t sh = (rel == 8) ? ((scv_hi & 0xFFFF0000u) | 0x3C00u) : (rel == 9) ? ((scv_hi & 0xFFFFu) | 0x3C000000u) : scv_hi;
                        uint32_t av[4] = {tq_hmul2(pa[0], sl), tq_hmul2(pa[1], sl), tq_hmul2(pa[2], sh), tq_hmul2(pa[3], sh)};
                        tq_mma_f16_m16n8k16(t, av, vb0, vb1);
                    } else {
                        float ve = 0.0f;
                        if (rel == 0 || rel == 1 || rel == 8 || rel == 9)
                            ve = __half2float(__ushort_as_half(TQ_STGK(kk)[256 + 2 * cb + f]));
                        const uint32_t vb0 = (rel == 0) ? tq_pack_bf16(ve, v[1][f]) : (rel == 1) ? tq_pack_bf16(v[0][f], ve) : b0;
                        const uint32_t vb1 = (rel == 8) ? tq_pack_bf16(ve, v[3][f]) : (rel == 9) ? tq_pack_bf16(v[2][f], ve) : b1;
                        tq_mma_bf16_m16n8k16(t, a, vb0, vb1);
                    }
                    if (pl == kk) { res[0] = t[0]; res[1] = t[1]; }
                    if (ph == kk) { res[2] = t[2]; res[3] = t[3]; }
                }
                oacc[q][f][0] = res[0]; oacc[q][f][1] = res[1]; oacc[q][f][2] = res[2]; oacc[q][f][3] = res[3];
            }
        }
    }
#undef TQ_STGK
#undef TQ_GROW
    __syncthreads();
}

// TQ_KV_T4 segment-aware verify attention: k_tq_paged_attn_q4_gqa_seg's geometry, bookkeeping, partial layout and
// epilogue with tq_gqa_seg_supertile_t4 (trellis stream + exact own keys); the fp16 table in smem.
template <int QT, int F16>
__global__ void __launch_bounds__(256) k_tq_paged_attn_q4_gqa_seg_t4(
    float *out, const float *q_proj_base, const uint16_t *q_norm_w,
    const uint8_t *k4_pool, const uint8_t *v8_pool,
    const int *positions, const int *slot_ids, const int *col_S, const int *seg_col0, const int *seg_n,
    const int *block_table, int max_blocks, int page, int page_log,
    int nh, int nkv, int hd, int q_m, int attn_m, int S, int ncols,
    float eps, float rope_theta, float *part_acc, float *part_ml, const tq_t4_pool t4p) {
    tq_pdl_enter();
    typedef tq_t4_seg_sm<QT> L;
    constexpr int ROWS = L::ROWS, TR = L::TR;
    __shared__ __align__(16) uint32_t sm[L::WORDS];
    __shared__ __align__(16) uint16_t t4tab[tq_t4_tab<TR>::ENTRIES];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int gr = lane >> 2, t4 = lane & 3;
    const int kv_head = blockIdx.x, seg = blockIdx.y, split = blockIdx.z;
    const int group = nh / nkv;
    const int col0 = seg_col0[seg], n = seg_n[seg];
    const int nrows = n * group;
    const int slot = slot_ids[col0];
    const uint8_t *kcb4 = k4_pool + (size_t)kv_head * (hd >> 1);
    const uint8_t *vcb8 = v8_pool + (size_t)kv_head * (hd >> 1);
    float *qtmp = (float *)(sm + L::QTMP), *red = (float *)(sm + L::RED);
    float *mst = (float *)(sm + L::MST), *lst = (float *)(sm + L::LST);
    int *rpos = (int *)(sm + L::RPOS), *rlo = (int *)(sm + L::RLO), *rhi = (int *)(sm + L::RHI);
    int *rSc = (int *)(sm + L::RSC), *rcol = (int *)(sm + L::RCOL);
    for (int i = tid; i < L::WMAX; i += 256) sm[i] = 0;
    tq_t4_tab<TR>::load(t4p.table, t4tab, tid);
    if (tid < ROWS) {
        mst[tid] = TQ_AMM_NINF; lst[tid] = 0.0f;
        int row = tid, lo = 0, hi = 0, pos = -1, Sc = 0, col = -1;
        if (row < nrows) {
            col = col0 + row / group; pos = positions[col]; Sc = col_S ? col_S[col] : S;
            int nst_all = (pos + 128) >> 7;
            hi = nst_all;
            if (Sc > 1) { int per = (nst_all + Sc - 1) / Sc; lo = split * per; hi = lo + per < nst_all ? lo + per : nst_all; }
            if (split >= Sc) { lo = 0; hi = 0; }                  // this split does not exist for the row
        }
        rpos[row] = pos; rlo[row] = lo; rhi[row] = hi; rSc[row] = Sc; rcol[row] = col;
    }
    __syncthreads();
    int it_lo = 1 << 30, it_hi = 0, pos_max = 0;
    for (int row = 0; row < nrows; row++) {
        if (rhi[row] > rlo[row]) { it_lo = min(it_lo, rlo[row]); it_hi = max(it_hi, rhi[row]); }
        pos_max = max(pos_max, rpos[row]);
    }
    const int seg_pos0 = rpos[0], seg_pos1 = rpos[nrows - 1];     // the segment's (consecutive) positions
    // ---- Q prep per row (column t, q-head r): norm + rope at the column's position + FWHT -> A fragments
    for (int row = 0; row < nrows; row++) {
        const int t = row / group, r = row - t * group, col = col0 + t, pos = rpos[row];
        const float *q_proj = q_proj_base + (size_t)col * q_m;
        const int hh = kv_head * group + r;
        float qv = q_proj[hh * (2 * hd) + tid];
        float qsum = qv * qv;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) qsum += __shfl_down_sync(0xffffffffu, qsum, o);
        if (lane == 0) red[warp] = qsum;
        __syncthreads();
        if (tid < 8) { float v = red[tid]; for (int o = 4; o > 0; o >>= 1) v += __shfl_down_sync(0xffu, v, o); if (tid == 0) red[8] = v; }
        __syncthreads();
        float qss = red[8];
        qv = qv * rsqrtf(qss / (float)hd + eps) * (1.0f + tq_bf16_to_float(q_norm_w[tid]));
        if (tid < 64) {
            int idx = tid & 31;
            float freq = powf(rope_theta, -((float)(2 * idx) / 64.0f));
            float angle = (float)pos * freq, c = cosf(angle), s = sinf(angle);
            float q_pair = q_proj[hh * (2 * hd) + ((tid < 32) ? tid + 32 : tid - 32)];
            q_pair = q_pair * rsqrtf(qss / (float)hd + eps) * (1.0f + tq_bf16_to_float(q_norm_w[(tid < 32) ? tid + 32 : tid - 32]));
            float q_rot = (tid < 32) ? -q_pair : q_pair;
            qv = qv * c + q_rot * s;
        }
        __syncthreads();
        qtmp[tid] = qv;
        __syncthreads();
        tq_fwht256(qtmp, tid);
        if (tid < 128) {
            int c16 = tid >> 3, tt = (tid >> 1) & 3, rsel = tid & 1;
            int d0 = 16 * c16 + 2 * tt + 8 * rsel;
            int l = 4 * (row & 7) + tt, rr = ((row >> 3) & 1) + 2 * rsel;
            sm[L::QA + (((row >> 4) * 16 + c16) * 32 + l) * 4 + rr] =
                F16 ? tq_pack_f16(qtmp[d0], qtmp[d0 + 1]) : tq_pack_bf16(qtmp[d0], qtmp[d0 + 1]);
        }
        __syncthreads();
    }
    float oacc[QT][4][4];
    #pragma unroll
    for (int q = 0; q < QT; q++)
        #pragma unroll
        for (int f = 0; f < 4; f++)
            #pragma unroll
            for (int j = 0; j < 4; j++) oacc[q][f][j] = 0.0f;
    const float rsc = rsqrtf((float)hd);
    const size_t bt_base = (size_t)slot * max_blocks;
    for (int it = it_lo; it < it_hi; it++) {
        int t0d = it << 7;
        int physd = block_table[bt_base + (t0d >> page_log)];
        tq_gqa_seg_supertile_t4<QT, F16>(oacc, sm, tid, warp, lane, gr, t4, it, t0d, physd, pos_max, page, nrows, nkv, rsc,
                                    kcb4, t4p.kside, vcb8, t4p.vside, kv_head, t4tab, t4p.f16s, slot, seg_pos0, seg_pos1);
    }
    // ---- output per row: S_t > 1 -> split partials (the per-column merge's layout); S_t == 1 -> final rows ----
    #pragma unroll
    for (int q = 0; q < QT; q++) {
        #pragma unroll
        for (int f = 0; f < 4; f++) {
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                const int ch = warp * 32 + 4 * (2 * t4 + j) + f;
                const int rl = q * 16 + gr, rh = rl + 8;
                if (rl < nrows && rSc[rl] > 1 && split < rSc[rl]) {
                    const int head = kv_head * group + (rl - (rl / group) * group);
                    size_t pidx = ((size_t)(head * ncols + rcol[rl]) * S + split);
                    part_acc[pidx * hd + ch] = oacc[q][f][j];
                }
                if (rh < nrows && rSc[rh] > 1 && split < rSc[rh]) {
                    const int head = kv_head * group + (rh - (rh / group) * group);
                    size_t pidx = ((size_t)(head * ncols + rcol[rh]) * S + split);
                    part_acc[pidx * hd + ch] = oacc[q][f][2 + j];
                }
            }
        }
    }
    if (tid < nrows && rSc[tid] > 1 && split < rSc[tid]) {
        const int head = kv_head * group + (tid - (tid / group) * group);
        size_t pidx = ((size_t)(head * ncols + rcol[tid]) * S + split);
        part_ml[pidx * 2 + 0] = mst[tid]; part_ml[pidx * 2 + 1] = lst[tid];
    }
    // rows with S_t == 1 (only split 0 reaches here for them): normalize + rotate back (rotated V space) + gate
    __syncthreads();
    float *ob = (float *)sm;                                      // 8 rows x 256 (aliases the A fragments)
    for (int rb = 0; rb < nrows; rb += 8) {
        #pragma unroll
        for (int q = 0; q < QT; q++) {
            #pragma unroll
            for (int f = 0; f < 4; f++) {
                #pragma unroll
                for (int j = 0; j < 2; j++) {
                    const int ch = warp * 32 + 4 * (2 * t4 + j) + f;
                    const int rl = q * 16 + gr, rh = rl + 8;
                    if (rl >= rb && rl < rb + 8 && rl < nrows && rSc[rl] == 1) ob[(rl - rb) * 256 + ch] = oacc[q][f][j] * (1.0f / lst[rl]);
                    if (rh >= rb && rh < rb + 8 && rh < nrows && rSc[rh] == 1) ob[(rh - rb) * 256 + ch] = oacc[q][f][2 + j] * (1.0f / lst[rh]);
                }
            }
        }
        __syncthreads();
        {
            const int row = rb + warp;
            if (row < nrows && rSc[row] == 1) {
                const int t = row / group, r = row - t * group, col = rcol[row];
                const int hh = kv_head * group + r;
                const float *q_proj = q_proj_base + (size_t)col * q_m;
                float x[8];
                #pragma unroll
                for (int e = 0; e < 8; e++) x[e] = ob[warp * 256 + lane + 32 * e];
                tq_fwht256_warp(x, lane);
                #pragma unroll
                for (int e = 0; e < 8; e++) {
                    const int d = lane + 32 * e;
                    float gate = q_proj[hh * (2 * hd) + hd + d];
                    out[(size_t)col * attn_m + hh * hd + d] = x[e] * (1.0f / (1.0f + expf(-gate)));
                }
            }
        }
        __syncthreads();
    }
}

// TQ_PAGED_ATTN_MMA: 1 (default) = GQA tensor-core decode attention, 0 = the scalar/split-K
// kernels above. Requires page % 128 == 0, hd == 256 and group <= 16 (else the launcher falls
// back). MEASURED 2026-09-22 (Qwen3.8-27B FP6+Q4, RTX 5090, tokens identical to the scalar path
// over 32 greedy steps at every N, needle 9/9 at 32K/64K/128K, compute-sanitizer clean):
//   32K  N=1 21.3 -> 17.0 ms/step, N=8 57.3 -> 24.4 (agg 140 -> 329 tok/s); +1.05 ms per extra
//        32K client instead of +5.2 (HBM bound ~0.5 ms)
//   128K N=1 37.1 -> 19.6, N=2 58.8 -> 22.6; +3.0 ms per 128K client instead of +21.7 (bound ~1.9)
//   -> ~0.023 us per context token per client per step (was 0.165).

// key splits for the GQA MMA decode kernel: enough (nkv x N x S) blocks to cover the SMs ~2x,
// each split >= 2 super-tiles, at most 64 (the merge reads S partials per head).
// TQ_PAGED_GQA_SPLIT=s pins the split (clamped by the context only, so a column's partition is the
// same at any N: the bitwise verify-vs-decode check of analysis/spec_paged_check.py). Default auto.
static int paged_gqa_split_S(int N, int pos_max) {
    const int forced = g_cfg.paged_gqa_split;
    if (g_split_n_override > 0) N = g_split_n_override;
    int nst = (pos_max + 128) >> 7;
    int sm = paged_sm_count();
    int blocks0 = g_qwen.nkv * N;
    int S = forced >= 1 ? forced : (2 * sm + blocks0 - 1) / blocks0;
    if (S > nst / 2) S = nst / 2;
    if (S > 64) S = 64;
    return S < 1 ? 1 : S;
}

// The fp16 kernels (TQ_PAGED_ATTN_ST >= 2): resident-block capacity of the instantiation in use (occupancy API, cached
// per QT) and the largest key-split count whose grid nkv x N x S fits ONE wave of it -- the old picker's
// ceil(2 SMs / blocks0) put 352 blocks on 340 slots (2 blocks/SM) and the 12 leftovers ran a second full block time
// (32K N=8 @550 W: 22.7 -> 21.7 ms per step with S = 10 instead of 11). Each split keeps >= 2 tiles, S <= 64 (merge).
static int paged_attn_f16_cap(int qt, int kf16) {
    static int cap[2][2] = {{0, 0}, {0, 0}};
    int &c = cap[qt == 2][kf16 != 0];
    if (c == 0) {
        int nb = 0;
        cudaError_t e = (qt == 2)
            ? (kf16 ? cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k_tq_paged_attn_q4_gqa_f16<2, 1>, 256, 0)
                    : cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k_tq_paged_attn_q4_gqa_f16<2, 0>, 256, 0))
            : (kf16 ? cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k_tq_paged_attn_q4_gqa_f16<1, 1>, 256, 0)
                    : cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k_tq_paged_attn_q4_gqa_f16<1, 0>, 256, 0));
        if (e != cudaSuccess || nb < 1) nb = 2;
        c = nb * paged_sm_count();
    }
    return c;
}
static int paged_gqa_split_S_f16(int N, int pos_max, int qt, int kf16) {
    const int forced = g_cfg.paged_gqa_split;
    if (g_split_n_override > 0) N = g_split_n_override;
    int nst = (pos_max + 128) >> 7;
    int blocks0 = g_qwen.nkv * N;
    int S = forced >= 1 ? forced : paged_attn_f16_cap(qt, kf16) / blocks0;
    if (S > nst / 2) S = nst / 2;
    if (S > 64) S = 64;
    return S < 1 ? 1 : S;
}
// The key-split count of the kernel the launcher will pick for this mode (decode: qt = 1; verify: its QT).
static int g_pg_v4 = 0;                         // TQ_KV_V4, read by qwn_paged_init (the pool's V is int4 like K)
static int g_pg_t4 = 0;                         // TQ_KV_T4 (trellis pool, below)
// TQ_T4_ATTN_F16 (DEFAULT 1 since 2026-10-17, owner's decision (b); it was 0 on main from 10-16 23:37 to 10-17 08:56): 1 = the
// trellis pool's decode step and verify wave run k_tq_paged_attn_q4_gqa_seg_t4<QT, 1> (fp16 operands, one kernel for
// both, cp.async-staged V codes for the decode step); 0 = the bf16 per-column decode kernel + the bf16 segment kernel
// (the 10-15 path, bit for bit). MEASURED (notes/session10-b2.md
// 11.1, 12): gates pass (spec_paged_check verify == decode, 4K dec8 e2e dKL -0.00001 +-0.00004 vs the bf16 kernels,
// needle 6/6 at 32K / 128K); the fp16 operands alone were speed-neutral (the kernel is LSU-bound: table-lookup bank
// conflicts 3.4 wavefronts/LDS, 3x the FP4 kernel's global load requests), the V staging (TQ_T4_VSTAGE) made it
// 335 -> 277 us per layer at 32K N=8 (-1.0 ms per step vs the bf16 kernels; FP4 + ST3 = 249 us).
// resident-block capacity of the T4 fp16 kernel (QT = 1's: decode and verify share one picker, so a column's split
// count -- and bits -- is the same in both)
static int paged_attn_t4_cap(void) {
    static int c = 0;
    if (c == 0) {
        int nb = 0;
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k_tq_paged_attn_q4_gqa_seg_t4<1, 1>, 256, 0) != cudaSuccess || nb < 1) nb = 2;
        c = nb * paged_sm_count();
    }
    return c;
}
static int paged_gqa_split_S_cap(int N, int pos_max, int cap) {
    const int forced = g_cfg.paged_gqa_split;
    if (g_split_n_override > 0) N = g_split_n_override;
    int nst = (pos_max + 128) >> 7;
    int blocks0 = g_qwen.nkv * N;
    int S = forced >= 1 ? forced : cap / blocks0;
    if (S > nst / 2) S = nst / 2;
    if (S > 64) S = 64;
    return S < 1 ? 1 : S;
}
static int paged_attn_split_S(int N, int pos_max, int qt) {
    int st = g_cfg.paged_attn_st;
    if (g_pg_t4) return g_cfg.t4_attn_f16 ? paged_gqa_split_S_cap(N, pos_max, paged_attn_t4_cap()) : paged_gqa_split_S(N, pos_max);
    return (st >= 2 && g_pg_v4) ? paged_gqa_split_S_f16(N, pos_max, qt, st == 3) : paged_gqa_split_S(N, pos_max);
}

// pick S so nh*N*S ~ 4x SM count, but only when few-client + long-ctx (else S=1 -> single kernel,
// so short-ctx / high-concurrency is byte-for-byte unchanged).
static int paged_split_S(int N, int max_pos) {
    int total = max_pos + 1, sm = paged_sm_count(), blocks0 = g_qwen.nh * N;
    if (blocks0 >= 2 * sm || total < 2048) return 1;     // already enough blocks / short ctx
    int s_occ = (6 * sm + blocks0 - 1) / blocks0;        // aim ~6x SMs of blocks
    int s_work = total / 512;                             // >=512 rows/split (keep partials cheap)
    int S = s_occ < s_work ? s_occ : s_work;
    if (S < 1) S = 1;
    if (S > 32) S = 32;
    return S;
}

static int ensure_attn_partials(size_t units) {        // units = nh*N*S
    if (units > g_attn_pacc_n || units > g_attn_pml_n) graph_invalidate();
    if (units > g_attn_pacc_n) {
        if (g_attn_pacc) cudaFree(g_attn_pacc);
        if (cudaMalloc(&g_attn_pacc, units * g_qwen.hd * sizeof(float)) != cudaSuccess) { g_attn_pacc = NULL; g_attn_pacc_n = 0; return -1; }
        g_attn_pacc_n = units;
    }
    if (units > g_attn_pml_n) {
        if (g_attn_pml) cudaFree(g_attn_pml);
        if (cudaMalloc(&g_attn_pml, units * 2 * sizeof(float)) != cudaSuccess) { g_attn_pml = NULL; g_attn_pml_n = 0; return -2; }
        g_attn_pml_n = units;
    }
    return 0;
}

// Page top-k (Quest). Pool + list buffers are defined here so the launcher can see them;
// alloc/free happens with the rest of the paged pool. do_topk=0 keeps dense attention
// (prefill). Summaries refresh whenever TQ_PAGED_TOPK_PAGES>0 so a later decode has them.
static float *g_pool_kmm[TQ_MAX_LAYERS] = {0};
static int *g_kmm_slot_d = NULL, *g_kmm_pos_d = NULL;
static int g_kmm_ready = 0;
static int *g_topk_list = NULL, *g_topk_count = NULL;
static int g_topk_cap = 0, g_topk_k_alloc = 0;
static int paged_kmm_refresh(int layer, const int *pos_h, const int *slot_h, int n, cudaStream_t st);
static int paged_topk_rank(int layer, const float *q_proj, const uint16_t *q_norm_w,
                           const int *positions, const int *slot_ids, int N, int max_pos,
                           int kpages, cudaStream_t st);

// --- paged pool host state (shared block pool + per-slot block table + DeltaNet) ---
static uint8_t *g_pool_k4[TQ_MAX_LAYERS] = {0};
static float   *g_pool_kq4s[TQ_MAX_LAYERS] = {0};
static uint8_t *g_pool_v8[TQ_MAX_LAYERS] = {0};
static float   *g_pool_vscale[TQ_MAX_LAYERS] = {0};
static void    *g_pg_recur[TQ_MAX_LAYERS] = {0};   // [max_slots][recur_f] in the g_pg_dstate format (fp32 / bf16 / fp16)
static float   *g_pg_conv[TQ_MAX_LAYERS] = {0};    // [max_slots][conv_f] fp32
// A slot's recurrent state (device pointer) and its byte size in the current storage format.
static inline void *pg_recur_slot(int L, int slot) {
    return (char *)g_pg_recur[L] + (size_t)slot * (size_t)batched_recur_floats() * g_pg_ds_es;
}
static inline size_t pg_recur_slot_bytes(void) { return (size_t)batched_recur_floats() * g_pg_ds_es; }
static int     *g_block_table = NULL, *h_block_table = NULL;
static int     *g_slot_ids = NULL, *g_iota = NULL, *g_ones = NULL;   // g_iota/g_ones: pool-wide 0..max_slots-1 / all-ones
static int     *g_tok_d = NULL;                                      // device token ids of the step's columns (TQ_GRAPH)
static int     *h_pin_ctl = NULL;                                    // pinned [3][g_pin_cols]: tokens, slot ids, positions
static int      g_pin_cols = 0;
static int     *g_pf_colslot = NULL; static int g_pf_colslot_cap = 0;  // prefill wave column->slot
static float   *g_pg_logits = NULL;
static int     *g_pg_argmax = NULL;
static float   *g_pg_sink = NULL;                                    // qwn_paged_logits_sink (caller-owned)
static int      g_pg_sink_cap = 0, g_pg_sink_head = 0;
static int     *h_free = NULL, *h_slot_nb = NULL;
static int      g_pg_nfree = 0;
static int      g_pg_page = 0, g_pg_plog = 0, g_pg_nblocks = 0, g_pg_maxslots = 0, g_pg_maxblk = 0;
static int      g_pg_ready = 0;

// TQ_KV_V4, read by qwn_paged_init: the pool's V is int4 like K (k_tq_paged_attn_kv_write_q4<1>);
// every paged attention launch below takes its VQ4 instantiation from it.
// g_pg_v4: defined above (split picker)

// TQ_KV_T4 (B2, session 10): the main pool holds every COMMITTED (token, kv head) K and V vector as a trellis
// code (see tq_t4_pool above); the columns of the wave / step in flight sit in a per-slot staging area (2 pages
// per slot, ring by logical-block parity) as FP4 K+V (what the attention of the step reads for them) together
// with the fp16 rotated K / V the KV-write kernel has after its FWHT (the encoder's input). Per full-attention layer:
//   g_pool_k4 / g_pool_v8 [rows][nkv*128]      trellis code bytes (the FP4 code arrays, same footprint)
//   g_t4_kside / g_t4_vside [rows][nkv*4]       prefix byte | fp16 scale (replace the 32 B of (scale, zp))
//   g_t4_stg_* [2*max_slots*page rows]          FP4 K / V codes + (scale, zp) of the in-flight columns
//   g_t4_stg_f16 [2*max_slots*page][nkv][512]   fp16 rotated K | V of the in-flight columns
// h_slot_encl[slot] = the slot's encoded length (= committed length after every entry point); the device copy
// g_t4_encl_d tells the attention kernels which rows are trellis (pool block) and which FP4 (staging page).
// Prefill waves encode their columns per layer before that layer's attention (every key trellis = the
// exllamav3 simulation); decode / verify steps attend their own new columns as FP4 and encode them afterwards
// (decode step) / at qwn_paged_commit (verify). The MTP drafter's KV rows (g_mtp_*) stay FP4.
// g_pg_t4: defined above (split picker)
static uint8_t *g_t4_kside[TQ_MAX_LAYERS] = {0}, *g_t4_vside[TQ_MAX_LAYERS] = {0};
static uint16_t *g_t4_stg_f16[TQ_MAX_LAYERS] = {0};
static float   *g_t4_cb = NULL;                       // [4096] fp32 codebook (analysis order)
static uint16_t *g_t4_tab = NULL;                     // [4096] fp16 table (engine nibble order)
static int     *h_slot_encl = NULL, *h_t4_encl_tmp = NULL;   // per slot: encoded length (host), wave view
static int     *g_t4_encl_d = NULL;                          // device copy read by the T4 attention kernels
static int g_t4_stg_rows = 0;
// TQ_KV_T4_ASYNC (default 1): a prefill wave attends its own columns exact from the fp16 staging (like a decode
// step) and their trellis encode runs per layer on a side stream behind the layer loop; g_t4_done is recorded after
// the wave's last encode and every later use of the pool waits on it. 0 = encode synchronously before each layer's
// attention (every key trellis, the exllamav3 simulation's semantics).
static int g_t4_async = 1;
static cudaStream_t g_t4_stream = NULL;
static cudaEvent_t g_t4_ev = NULL, g_t4_done = NULL;
static void paged_t4_sync(cudaStream_t st) { if (g_pg_t4 && g_t4_async && g_t4_done) cudaStreamWaitEvent(st, g_t4_done, 0); }
static tq_t4_pool t4_pool_of(int L) {
    tq_t4_pool p;
    p.kside = g_t4_kside[L]; p.vside = g_t4_vside[L];
    p.table = g_t4_tab; p.f16s = g_t4_stg_f16[L]; p.encl = g_t4_encl_d; p.nblk = g_pg_nblocks;
    return p;
}
static int paged_t4_encode_wave_layer(const int *seg_slot, const int *seg_off, const int *seg_len, const int *col_pos,
                                      int K, int L, cudaStream_t st);

// t4 != NULL: trellis pool (TQ_KV_T4) -- the write goes to the slot's FP4 staging page + fp16 staging
static void paged_kv_write(int ncol, uint8_t *k4_pool, float *kq4s_pool, uint8_t *v8_pool, float *vscale_pool,
                           const float *k_proj, const float *v_proj, const uint16_t *k_norm_w,
                           const int *positions, const int *slot_ids, const int *block_table,
                           int max_blocks, int page, int page_log, cudaStream_t st, int t4_layer = -1) {
    int nkv = g_qwen.nkv, hd = g_qwen.hd, kv_m = nkv * hd;
    if (t4_layer >= 0) {
        int L = t4_layer;
        tq_launch(g_pdl_small, k_tq_paged_attn_kv_write_q4<1, 1>, dim3(nkv, ncol), dim3(hd), 0, st,
                  (uint8_t *)NULL, (float *)NULL, (uint8_t *)NULL, (float *)NULL, k_proj, v_proj, k_norm_w,
                  positions, slot_ids, block_table, max_blocks, page, page_log, nkv, hd, kv_m, g_qwen.eps,
                  g_qwen.rope_theta, g_t4_stg_f16[L], g_pg_nblocks);
        return;
    }
    tq_launch(g_pdl_small, g_pg_v4 ? k_tq_paged_attn_kv_write_q4<1> : k_tq_paged_attn_kv_write_q4<0>,
              dim3(nkv, ncol), dim3(hd), 0, st,
              k4_pool, kq4s_pool, v8_pool, vscale_pool, k_proj, v_proj, k_norm_w, positions, slot_ids,
              block_table, max_blocks, page, page_log, nkv, hd, kv_m, g_qwen.eps, g_qwen.rope_theta,
              (uint16_t *)NULL, 0);
}

static void paged_attn_merge(float *out, const float *q_proj, const int *positions, int N, int S,
                             cudaStream_t st, const int *col_S = NULL) {
    int nh = g_qwen.nh, hd = g_qwen.hd, q_m = nh * hd * 2, attn_m = nh * hd;
    tq_launch(g_pdl_small, g_pg_v4 ? k_tq_paged_attn_q4_merge<1> : k_tq_paged_attn_q4_merge<0>,
              dim3(nh, N), dim3(hd), 0, st,
              out, q_proj, positions, nh, hd, q_m, attn_m, S, g_attn_pacc, g_attn_pml, col_S);
}
// Verify waves: per-column key-split counts (device array, NULL for the decode step) and their max, and the
// sub-segment list (col0, n) for the segment-aware kernel (count 0 = per-column kernel).
// g_iota, g_ones: declared with the pool state above
static const int *g_attn_col_S = NULL;
static int g_attn_col_S_max = 0;
static const int *g_attn_seg_c0 = NULL, *g_attn_seg_n = NULL;
static int g_attn_seg_cnt = 0, g_attn_seg_qt = 0;

// t4_layer >= 0: the main pool in trellis mode (TQ_KV_T4) for full-attention layer t4_layer (the MTP
// pool and the FP4 / E4M3 pools pass -1).
static int launch_paged_attn_q4(float *out, const float *q_proj, const uint16_t *q_norm_w,
                                const float *k_proj, const float *v_proj, const uint16_t *k_norm_w,
                                uint8_t *k4_pool, float *kq4s_pool, uint8_t *v8_pool, float *vscale_pool,
                                const int *positions, const int *slot_ids, const int *block_table,
                                int max_blocks, int page, int page_log, int N, int max_pos,
                                int layer, const int *pos_host, const int *slot_host, int do_topk,
                                cudaStream_t st, int t4_layer = -1,
                                const int *wave_seg_slot = NULL, const int *wave_seg_off = NULL,
                                const int *wave_seg_len = NULL, int wave_K = 0) {
    int nh = g_qwen.nh, nkv = g_qwen.nkv, hd = g_qwen.hd;
    int q_m = nh * hd * 2, attn_m = nh * hd;
    const int t4 = t4_layer >= 0;
    const tq_t4_pool t4p = t4 ? t4_pool_of(t4_layer) : tq_t4_pool();
    paged_kv_write(N, k4_pool, kq4s_pool, v8_pool, vscale_pool, k_proj, v_proj, k_norm_w, positions,
                   slot_ids, block_table, max_blocks, page, page_log, st, t4_layer);
    if (t4 && wave_seg_slot &&          // prefill (scalar path) / verify wave: the wave's columns -> trellis for this layer
        paged_t4_encode_wave_layer(wave_seg_slot, wave_seg_off, wave_seg_len, pos_host, wave_K, t4_layer, st) != 0) return -5;
    int kpages = g_cfg.paged_topk_pages;
    if (kpages > 0 && pos_host && slot_host) {
        if (paged_kmm_refresh(layer, pos_host, slot_host, N, st) != 0) return -3;
    }
    int nblk = (max_pos >> page_log) + 1;
    // Full coverage and short contexts take the dense walk. Ranking them only adds a launch.
    int use_topk = do_topk && kpages > 0 && nblk > kpages && max_pos + 1 >= g_cfg.paged_topk_min_ctx
                   && g_cfg.paged_attn_mma
                   && (page & 127) == 0 && hd == 256 && nh % nkv == 0 && nh / nkv <= 16;
    if (g_cfg.paged_attn_mma && (page & 127) == 0 && hd == 256 && nh % nkv == 0 && nh / nkv <= 16) {
        int S;
        if (use_topk) {
            int budget = kpages + 3;
            int fake = budget * 128 - 1;
            if (fake > max_pos) fake = max_pos;
            if (fake < 0) fake = 0;
            S = paged_gqa_split_S(N, fake);
            if (paged_topk_rank(layer, q_proj, q_norm_w, positions, slot_ids, N, max_pos, kpages, st) != 0)
                return -4;
        } else {
            S = paged_attn_split_S(N, max_pos, 1);
        }
        const int *col_S = NULL;
        if (g_attn_col_S && !use_topk) { col_S = g_attn_col_S; S = g_attn_col_S_max; }   // per-column S (verify)
        if (S > 1 && ensure_attn_partials((size_t)nh * N * S) != 0) { S = 1; col_S = NULL; }
        const int st_mode = g_cfg.paged_attn_st;
        if (st_mode >= 2 && !use_topk && g_pg_v4 && !t4 && batched_q4s_elem_bytes() == 2) {
            // fp16 int4 path (TQ_PAGED_ATTN_ST=2: P.V; 3: + K/Q): the decode step as N one-column segments, the
            // verify wave as its slot sub-segments -- one kernel, so verify == decode bitwise.
            const bool seg = col_S && g_attn_seg_cnt > 0;
            const int *c0 = seg ? g_attn_seg_c0 : g_iota, *sn = seg ? g_attn_seg_n : g_ones;
            const int nseg = seg ? g_attn_seg_cnt : N, qt = seg ? g_attn_seg_qt : 1;
#define TQ_F16_LAUNCH(QTV, KF) \
            tq_launch(g_pdl_small, k_tq_paged_attn_q4_gqa_f16<QTV, KF>, dim3(nkv, nseg, S), dim3(256), 0, st, \
                out, q_proj, q_norm_w, k4_pool, kq4s_pool, v8_pool, vscale_pool, positions, slot_ids, col_S, c0, sn, \
                block_table, max_blocks, page, page_log, nh, nkv, hd, q_m, attn_m, S, N, \
                g_qwen.eps, g_qwen.rope_theta, g_attn_pacc, g_attn_pml)
            if (st_mode == 3) { if (qt == 2) TQ_F16_LAUNCH(2, 1); else TQ_F16_LAUNCH(1, 1); }
            else              { if (qt == 2) TQ_F16_LAUNCH(2, 0); else TQ_F16_LAUNCH(1, 0); }
#undef TQ_F16_LAUNCH
            if (S > 1) paged_attn_merge(out, q_proj, positions, N, S, st, col_S);
            return cudaGetLastError() == cudaSuccess ? 0 : -1;
        }
        if (t4 && !use_topk && g_cfg.t4_attn_f16) {
            // TQ_KV_T4 fp16 path (attention third pass on the trellis pool): the decode step as N one-column segments,
            // the verify wave as its slot sub-segments -- one kernel, so verify == decode bitwise
            const bool seg = col_S && g_attn_seg_cnt > 0;
            const int *c0 = seg ? g_attn_seg_c0 : g_iota, *sn = seg ? g_attn_seg_n : g_ones;
            const int nseg = seg ? g_attn_seg_cnt : N, qt = seg ? g_attn_seg_qt : 1;
#define TQ_T4F16_LAUNCH(QTV) \
            tq_launch(g_pdl_small, k_tq_paged_attn_q4_gqa_seg_t4<QTV, 1>, dim3(nkv, nseg, S), dim3(256), 0, st, \
                out, q_proj, q_norm_w, k4_pool, v8_pool, positions, slot_ids, col_S, c0, sn, \
                block_table, max_blocks, page, page_log, nh, nkv, hd, q_m, attn_m, S, N, \
                g_qwen.eps, g_qwen.rope_theta, g_attn_pacc, g_attn_pml, t4p)
            if (qt == 2) TQ_T4F16_LAUNCH(2); else TQ_T4F16_LAUNCH(1);
#undef TQ_T4F16_LAUNCH
            if (S > 1) paged_attn_merge(out, q_proj, positions, N, S, st, col_S);
            return cudaGetLastError() == cudaSuccess ? 0 : -1;
        }
        if (col_S && g_attn_seg_cnt > 0) {
            // verify wave: one K/V stream per slot sub-segment (k_tq_paged_attn_q4_gqa_seg), rows = (column, q-head);
            // TQ_KV_T4: its twin with the exact-own-key diagonal patch (k_tq_paged_attn_q4_gqa_seg_t4)
#define TQ_SEG_LAUNCH(QTV, VQ) \
            tq_launch(g_pdl_small, k_tq_paged_attn_q4_gqa_seg<VQ, QTV>, dim3(nkv, g_attn_seg_cnt, S), dim3(256), 0, st, \
                out, q_proj, q_norm_w, k4_pool, kq4s_pool, v8_pool, vscale_pool, positions, slot_ids, col_S, \
                g_attn_seg_c0, g_attn_seg_n, block_table, max_blocks, page, page_log, nh, nkv, hd, q_m, attn_m, S, N, \
                g_qwen.eps, g_qwen.rope_theta, g_attn_pacc, g_attn_pml)
#define TQ_SEG_LAUNCH_T4(QTV) \
            tq_launch(g_pdl_small, k_tq_paged_attn_q4_gqa_seg_t4<QTV, 0>, dim3(nkv, g_attn_seg_cnt, S), dim3(256), 0, st, \
                out, q_proj, q_norm_w, k4_pool, v8_pool, positions, slot_ids, col_S, \
                g_attn_seg_c0, g_attn_seg_n, block_table, max_blocks, page, page_log, nh, nkv, hd, q_m, attn_m, S, N, \
                g_qwen.eps, g_qwen.rope_theta, g_attn_pacc, g_attn_pml, t4p)
            if (t4)                      { if (g_attn_seg_qt == 2) TQ_SEG_LAUNCH_T4(2); else TQ_SEG_LAUNCH_T4(1); }
            else if (g_attn_seg_qt == 2) { if (g_pg_v4) TQ_SEG_LAUNCH(2, 1); else TQ_SEG_LAUNCH(2, 0); }
            else                         { if (g_pg_v4) TQ_SEG_LAUNCH(1, 1); else TQ_SEG_LAUNCH(1, 0); }
#undef TQ_SEG_LAUNCH
#undef TQ_SEG_LAUNCH_T4
            if (S > 1) paged_attn_merge(out, q_proj, positions, N, S, st, col_S);
            return cudaGetLastError() == cudaSuccess ? 0 : -1;
        }
        if (!use_topk && g_pg_v4 && !t4 && st_mode == 1 && batched_q4s_elem_bytes() == 2) {
            // dense int4-V decode attention: the staged-V kernel (same bits, V via cp.async into smem); the trellis
            // pool (TQ_KV_T4) keeps its own kernel below
            tq_launch(g_pdl_small, k_tq_paged_attn_q4_gqa_mma_st, dim3(nkv, N, S), dim3(256), 0, st,
                out, q_proj, q_norm_w, k4_pool, kq4s_pool, v8_pool, vscale_pool, positions, slot_ids,
                block_table, max_blocks, page, page_log, nh, nkv, hd, q_m, attn_m, S,
                g_qwen.eps, g_qwen.rope_theta, g_attn_pacc, g_attn_pml, col_S);
            if (S > 1) paged_attn_merge(out, q_proj, positions, N, S, st, col_S);
            return cudaGetLastError() == cudaSuccess ? 0 : -1;
        }
#define TQ_GQA_LAUNCH(TK, VQ, TT, LIST, CNT, CAP) \
            tq_launch(g_pdl_small, k_tq_paged_attn_q4_gqa_mma<TK, VQ, TT>, dim3(nkv, N, S), dim3(256), 0, st, \
                out, q_proj, q_norm_w, k4_pool, kq4s_pool, v8_pool, vscale_pool, positions, slot_ids, \
                block_table, max_blocks, page, page_log, nh, nkv, hd, q_m, attn_m, S, \
                g_qwen.eps, g_qwen.rope_theta, g_attn_pacc, g_attn_pml, LIST, CNT, CAP, col_S, t4p)
        if (use_topk) {
            if (t4)          TQ_GQA_LAUNCH(1, 1, 1, g_topk_list, g_topk_count, g_topk_cap);
            else if (g_pg_v4) TQ_GQA_LAUNCH(1, 1, 0, g_topk_list, g_topk_count, g_topk_cap);
            else             TQ_GQA_LAUNCH(1, 0, 0, g_topk_list, g_topk_count, g_topk_cap);
        } else {
            if (t4)          TQ_GQA_LAUNCH(0, 1, 1, (const int *)NULL, (const int *)NULL, 0);
            else if (g_pg_v4) TQ_GQA_LAUNCH(0, 1, 0, (const int *)NULL, (const int *)NULL, 0);
            else             TQ_GQA_LAUNCH(0, 0, 0, (const int *)NULL, (const int *)NULL, 0);
        }
#undef TQ_GQA_LAUNCH
        if (S > 1) paged_attn_merge(out, q_proj, positions, N, S, st, col_S);
        return cudaGetLastError() == cudaSuccess ? 0 : -1;
    }
    int S = paged_split_S(N, max_pos);
    if (S <= 1) {
        if (t4)
            k_tq_paged_attn_q4<1, 1><<<dim3(nh, N), hd, 0, st>>>(
                out, q_proj, q_norm_w, k4_pool, kq4s_pool, v8_pool, vscale_pool, positions, slot_ids,
                block_table, max_blocks, page, page_log, nh, nkv, hd, q_m, attn_m, g_qwen.eps, g_qwen.rope_theta, t4p);
        else if (g_pg_v4)
            k_tq_paged_attn_q4<1><<<dim3(nh, N), hd, 0, st>>>(
                out, q_proj, q_norm_w, k4_pool, kq4s_pool, v8_pool, vscale_pool, positions, slot_ids,
                block_table, max_blocks, page, page_log, nh, nkv, hd, q_m, attn_m, g_qwen.eps, g_qwen.rope_theta);
        else
            k_tq_paged_attn_q4<0><<<dim3(nh, N), hd, 0, st>>>(
                out, q_proj, q_norm_w, k4_pool, kq4s_pool, v8_pool, vscale_pool, positions, slot_ids,
                block_table, max_blocks, page, page_log, nh, nkv, hd, q_m, attn_m, g_qwen.eps, g_qwen.rope_theta);
        return cudaGetLastError() == cudaSuccess ? 0 : -1;
    }
    if (ensure_attn_partials((size_t)nh * N * S) != 0) return -2;
    if (t4)
        k_tq_paged_attn_q4_split<1, 1><<<dim3(nh, N, S), hd, 0, st>>>(
            q_proj, q_norm_w, k4_pool, kq4s_pool, v8_pool, vscale_pool, positions, slot_ids,
            block_table, max_blocks, page, page_log, nh, nkv, hd, q_m, S, g_qwen.eps, g_qwen.rope_theta,
            g_attn_pacc, g_attn_pml, t4p);
    else if (g_pg_v4)
        k_tq_paged_attn_q4_split<1><<<dim3(nh, N, S), hd, 0, st>>>(
            q_proj, q_norm_w, k4_pool, kq4s_pool, v8_pool, vscale_pool, positions, slot_ids,
            block_table, max_blocks, page, page_log, nh, nkv, hd, q_m, S, g_qwen.eps, g_qwen.rope_theta,
            g_attn_pacc, g_attn_pml);
    else
        k_tq_paged_attn_q4_split<0><<<dim3(nh, N, S), hd, 0, st>>>(
            q_proj, q_norm_w, k4_pool, kq4s_pool, v8_pool, vscale_pool, positions, slot_ids,
            block_table, max_blocks, page, page_log, nh, nkv, hd, q_m, S, g_qwen.eps, g_qwen.rope_theta,
            g_attn_pacc, g_attn_pml);
    paged_attn_merge(out, q_proj, positions, N, S, st);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

// Ragged-prefill MMA attention (step 2b): write all T columns' Q4 KV, then run the
// tensor-core wide attention per CLIENT SEGMENT (ascending positions, one slot -> one
// block table; a 128-aligned super-tile lies in one page>=128 block). Compute-bound
// prefill regime -> tensor cores win; the memory-bound paged DECODE keeps scalar/split-K.
// seg_slot/seg_off/seg_len are HOST arrays (K segments over the T concatenated columns).
static int launch_paged_attn_q4_mma_prefill(
    float *out, const float *q_proj, const uint16_t *q_norm_w,
    const float *k_proj, const float *v_proj, const uint16_t *k_norm_w,
    uint8_t *k4_pool, float *kq4s_pool, uint8_t *v8_pool, float *vscale_pool,
    const int *positions, const int *slot_ids, const int *block_table,
    int max_blocks, int page, int page_log,
    const int *seg_slot, const int *seg_off, const int *seg_len,
    const int *col_pos_host, const int *col_slot_host, int layer, int K, int T,
    cudaStream_t st, int t4_layer = -1) {
    int nh = g_qwen.nh, nkv = g_qwen.nkv, hd = g_qwen.hd;
    int q_m = nh * hd * 2, attn_m = nh * hd;
    const int t4 = t4_layer >= 0;
    const tq_t4_pool t4p = t4 ? t4_pool_of(t4_layer) : tq_t4_pool();
    paged_kv_write(T, k4_pool, kq4s_pool, v8_pool, vscale_pool, k_proj, v_proj, k_norm_w, positions,
                   slot_ids, block_table, max_blocks, page, page_log, st, t4_layer);
    if (t4) {
        if (g_t4_async) {                      // the attention reads the wave's columns from the fp16 staging; encode behind it
            cudaEventRecord(g_t4_ev, st);
            cudaStreamWaitEvent(g_t4_stream, g_t4_ev, 0);
            if (paged_t4_encode_wave_layer(seg_slot, seg_off, seg_len, col_pos_host, K, t4_layer, g_t4_stream) != 0) return -5;
        } else if (paged_t4_encode_wave_layer(seg_slot, seg_off, seg_len, col_pos_host, K, t4_layer, st) != 0) return -5;
    }
    // Prefill attention stays dense. Summaries of the pages this wave wrote are refreshed
    // so a later pure decode step can rank them. Decode columns inside a mixed wave are
    // covered here too (they share this KV write).
    if (g_cfg.paged_topk_pages > 0 && col_pos_host && col_slot_host) {
        if (paged_kmm_refresh(layer, col_pos_host, col_slot_host, T, st) != 0) return -3;
    }
    for (int k = 0; k < K; k++) {
        int off = seg_off[k], n = seg_len[k], slot = seg_slot[k];
        // 32-row query tiles (QT=2) like the single-stream wide path: the kernel is L2-bound on
        // the K/V super-tile stream, so two tiles per block halve that traffic per query token
        // (the ragged path ran QT=1 until 2026-09-22: 128K prefill 115 s vs 85 s single-stream).
        // Segments shorter than 32 rows (decode columns in mixed waves, short follow-ups) keep QT=1, and so does every
        // segment with TQ_WIDE_ATTN_QROWS=16.
        int qr = (g_cfg.wide_attn_qrows == 32 && n > 16) ? 32 : 16;
        int nqt = (n + qr - 1) / qr;
        int part_f = 2 * qr + qr * 256;
        // key-split by this segment's own prefix depth (its last column's slot pos);
        // segments run sequentially on one stream, so the partial slab is reused
        int max_pos = col_pos_host ? col_pos_host[off + n - 1] : -1;
        // session 11: the GQA-grouped fp16 kernel for trellis segments of >= 32 columns (TQ_PATTN_GQA=1)
        if (t4 && g_cfg.pattn_gqa > 0 && n >= 32 && max_pos >= 0 &&
            launch_pattn_t4(out + (size_t)off * attn_m, q_proj + (size_t)off * q_m, q_norm_w, k4_pool, v8_pool,
                            positions + off, slot, block_table, max_blocks, page, page_log, n, max_pos, t4p, st) == 0)
            continue;
        int S = wide_attn_split_S(max_pos);
        if (S > 1 && ensure_wide_attn_part((size_t)nh * nqt * S * part_f) != 0)
            S = 1;
#define TQ_WIDE_PAGED(QTV, VQ) \
            k_tq_wide_attn_mma<4, QTV, VQ><<<dim3(nh, nqt, S), 256, 0, st>>>( \
                out + (size_t)off * attn_m, q_proj + (size_t)off * q_m, q_norm_w, \
                NULL, NULL, k4_pool, kq4s_pool, v8_pool, vscale_pool, \
                positions + off, slot, block_table, max_blocks, page, page_log, \
                n, nh, nkv, hd, q_m, attn_m, g_qwen.eps, g_qwen.rope_theta, \
                g_wide_attn_part)
#define TQ_WIDE_PAGED_T4(QTV) \
            k_tq_wide_attn_mma<4, QTV, 1, 1><<<dim3(nh, nqt, S), 256, 0, st>>>( \
                out + (size_t)off * attn_m, q_proj + (size_t)off * q_m, q_norm_w, \
                NULL, NULL, k4_pool, kq4s_pool, v8_pool, vscale_pool, \
                positions + off, slot, block_table, max_blocks, page, page_log, \
                n, nh, nkv, hd, q_m, attn_m, g_qwen.eps, g_qwen.rope_theta, \
                g_wide_attn_part, t4p)
#define TQ_WIDE_MERGE(QTV, VQ) \
            k_tq_wide_attn_mma_merge<QTV, VQ><<<dim3(nh, nqt), 256, 0, st>>>( \
                out + (size_t)off * attn_m, q_proj + (size_t)off * q_m, \
                n, nh, hd, q_m, attn_m, S, g_wide_attn_part)
        if (t4) { if (qr == 32) TQ_WIDE_PAGED_T4(2); else TQ_WIDE_PAGED_T4(1); }
        else if (qr == 32) { if (g_pg_v4) TQ_WIDE_PAGED(2, 1); else TQ_WIDE_PAGED(2, 0); }
        else          { if (g_pg_v4) TQ_WIDE_PAGED(1, 1); else TQ_WIDE_PAGED(1, 0); }
        if (S > 1) {
            if (qr == 32) { if (g_pg_v4) TQ_WIDE_MERGE(2, 1); else TQ_WIDE_MERGE(2, 0); }
            else          { if (g_pg_v4) TQ_WIDE_MERGE(1, 1); else TQ_WIDE_MERGE(1, 0); }
        }
#undef TQ_WIDE_PAGED
#undef TQ_WIDE_PAGED_T4
#undef TQ_WIDE_MERGE
    }
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

// paged MTP drafter (qwn_paged_mtp_init): the MTP layer's KV pool rows + per-slot last hidden
static int      g_mtp_on = 0;
static uint8_t *g_mtp_k4 = NULL;  static float *g_mtp_kq4s = NULL;
static uint8_t *g_mtp_v8 = NULL;  static float *g_mtp_vscale = NULL;
static float   *g_mtp_last_h = NULL;                 // [max_slots][H]
static int mtp_hook_decode(const int *tokens, const int *slot_ids_h, const int *positions, int N);
static int mtp_hook_decode_prep(const int *tokens, const int *slot_ids_h, const int *positions, int N);
static int mtp_hook_decode_kernels(int m, int N);
static int mtp_draft_body(int N, int K, int max_pos);
static int mtp_hook_prefill(const int *tokens, const int *col_pos, const int *seg_slot, const int *seg_off,
                            const int *seg_len, int K, int T);
static int mtp_hook_commit(int m);
static void vf_free_all(void);                                       // spec verify/commit buffers (below)
static void mtp_paged_free(void);                                    // paged MTP drafter pool (below)

static void park_ring_free(void);
static void paged_t4_free(void);
static void paged_free_all(void) {
    graph_invalidate();
    vf_free_all();
    mtp_paged_free();
    park_ring_free();
    paged_t4_free();
    for (int L = 0; L < TQ_MAX_LAYERS; L++) {
        if (g_pool_k4[L]) { cudaFree(g_pool_k4[L]); g_pool_k4[L] = NULL; }
        if (g_pool_kq4s[L]) { cudaFree(g_pool_kq4s[L]); g_pool_kq4s[L] = NULL; }
        if (g_pool_v8[L]) { cudaFree(g_pool_v8[L]); g_pool_v8[L] = NULL; }
        if (g_pool_vscale[L]) { cudaFree(g_pool_vscale[L]); g_pool_vscale[L] = NULL; }
        if (g_pool_kmm[L]) { cudaFree(g_pool_kmm[L]); g_pool_kmm[L] = NULL; }
        if (g_pg_recur[L]) { cudaFree(g_pg_recur[L]); g_pg_recur[L] = NULL; }
        if (g_pg_conv[L]) { cudaFree(g_pg_conv[L]); g_pg_conv[L] = NULL; }
    }
    if (g_block_table) { cudaFree(g_block_table); g_block_table = NULL; }
    if (g_slot_ids) { cudaFree(g_slot_ids); g_slot_ids = NULL; }
    if (g_tok_d) { cudaFree(g_tok_d); g_tok_d = NULL; }
    if (h_pin_ctl) { cudaFreeHost(h_pin_ctl); h_pin_ctl = NULL; }
    if (g_iota) { cudaFree(g_iota); g_iota = NULL; }
    if (g_ones) { cudaFree(g_ones); g_ones = NULL; }
    if (g_pf_colslot) { cudaFree(g_pf_colslot); g_pf_colslot = NULL; } g_pf_colslot_cap = 0;
    if (g_attn_pacc) { cudaFree(g_attn_pacc); g_attn_pacc = NULL; } g_attn_pacc_n = 0;
    if (g_attn_pml) { cudaFree(g_attn_pml); g_attn_pml = NULL; } g_attn_pml_n = 0;
    if (g_pg_logits) { cudaFree(g_pg_logits); g_pg_logits = NULL; }
    if (g_pg_argmax) { cudaFree(g_pg_argmax); g_pg_argmax = NULL; }
    if (g_kmm_slot_d) { cudaFree(g_kmm_slot_d); g_kmm_slot_d = NULL; }
    if (g_kmm_pos_d) { cudaFree(g_kmm_pos_d); g_kmm_pos_d = NULL; }
    if (g_topk_list) { cudaFree(g_topk_list); g_topk_list = NULL; }
    if (g_topk_count) { cudaFree(g_topk_count); g_topk_count = NULL; }
    g_kmm_ready = 0; g_topk_k_alloc = 0; g_topk_cap = 0;
    if (h_block_table) { free(h_block_table); h_block_table = NULL; }
    if (h_free) { free(h_free); h_free = NULL; }
    if (h_slot_nb) { free(h_slot_nb); h_slot_nb = NULL; }
    g_pg_nfree = 0; g_pg_page = 0; g_pg_plog = 0; g_pg_nblocks = 0;
    g_pg_maxslots = 0; g_pg_maxblk = 0; g_pg_ready = 0;
    g_pg_sink = NULL; g_pg_sink_cap = 0; g_pg_sink_head = 0;
    g_pg_dstate = 0; g_pg_ds_es = sizeof(float);
}

static int paged_alloc_block(void) { return g_pg_nfree > 0 ? h_free[--g_pg_nfree] : -1; }

static int paged_ensure_blocks(int slot, int pos) {
    int need = (pos >> g_pg_plog) + 1;
    if (need > g_pg_maxblk) return -1;
    int *row = h_block_table + (size_t)slot * g_pg_maxblk;
    for (int lb = h_slot_nb[slot]; lb < need; lb++) {
        int b = paged_alloc_block();
        if (b < 0) return -2;                 // pool exhausted -> admission failure
        row[lb] = b;
    }
    if (need > h_slot_nb[slot]) h_slot_nb[slot] = need;
    return 0;
}
