// trellis-coded KV pages (TQ_KV_T4): device-side code and decode tables.
// Part of the forward_qwen.cu unity build: included in order, not a standalone header.

// ===== B2 — trellis-coded KV pages (TQ_KV_T4, session 10) =====
// A COMPLETE 128-token page of the paged pool holds every (token, kv head) K and V vector as a
// bitshift-trellis code: 12-bit states, 4 bits per value, value_t = table[s_t] * scale, where the
// state s_t is the 12-bit window at nibble offset t of the 258-nibble stream
//     [prefix nibbles n0 n1 = s_0 >> 8, (s_0 >> 4) & 15][b_0 .. b_255]
// stored little-endian (nibble k = byte k/2, low nibble first), i.e. s_t = (stream >> 4t) & 0xFFF.
// Layout: the 128 symbol bytes live in the pool's code arrays exactly where the FP4 codes would
// (k4 / v8 rows, 128 B per vector), the prefix byte (n0 | n1 << 4) and the fp16 RMS scale in a
// 4-byte side word per vector ([rows][nkv] uint32: byte 0 prefix, bytes 1-2 scale, byte 3 unused)
// that replaces the 32 B of FP4 (scale, zp) pairs -> 132 B per vector, 16.5 KiB per token.
// The dequant table (4096 fp16) is the trained codebook in the engine's nibble order; the
// analysis-side encoder (analysis/kv_trellis_cuda) numbers states big-endian, so
// table[e] = cb[swap nibbles 0 and 2 of e]. The newest (incomplete) page of every slot stays FP4
// in a per-slot staging area (block ids >= nblk address staging page id - nblk) and is encoded as a
// whole when the slot's committed length passes its end (k_t4_forward / k_t4_walk below).
struct tq_t4_pool {
    const uint8_t *kside, *vside;      // pool side words [rows][nkv] (4 B each) for the K / V codes
    const uint16_t *table;             // [4096] fp16 dequant table (engine nibble order)
    const uint16_t *f16s;              // staging fp16 rotated K | V [max_slots * page rows][nkv][512]: the in-flight
                                       // columns of a wave / step, row = slot * page + (pos & (page - 1))
    const int *encl;                   // [max_slots] encoded length (host bookkeeping mirror; the kernels decide by
                                       // the column's own position: rows < pos trellis, row pos exact)
    int nblk;                          // pool blocks
};

// 8 KB table -> smem (256 threads, 2 x 16 B each)
static __device__ __forceinline__ void tq_t4_load_table(const uint16_t *g, uint16_t *s, int tid) {
    const uint4 *src = (const uint4 *)g;
    uint4 *dst = (uint4 *)s;
    dst[tid] = src[tid];
    dst[tid + 256] = src[tid + 256];
}
// Table replication (B2-eng lever 1, 2026-10-16): the lookups' 12-bit indices are random, so 32 lanes hit random banks
// (3.4 wavefronts per LDS measured). TR interleaved copies, copy r's word j (entries 2j, 2j+1) at word TR*j + r, put
// copy r on the banks == r (mod TR) only; lane l reads copy (l & (TR-1)), so conflicts stay within the 32/TR lanes of a
// copy over 32/TR banks (expected wavefronts 3.53 / 3.07 / 2.61 for TR = 1 / 2 / 4). Entry e of copy r is at uint16
// index ((e << LOG) & ~(2 TR - 1)) | 2 r | (e & 1); the per-lane 2 r is precomputed (cr).
template <int TR> struct tq_t4_tab {
    static constexpr int LOG = (TR == 4) ? 2 : (TR == 2) ? 1 : 0;
    static constexpr int ENTRIES = TR * 4096;
    static __device__ __forceinline__ uint32_t cr(int lane) { return (uint32_t)((lane & (TR - 1)) << 1); }
    static __device__ __forceinline__ uint32_t idx(uint32_t e, uint32_t cr) {
        return TR == 1 ? e : ((((e << LOG) & ~(uint32_t)(2 * TR - 1)) | cr) | (e & 1u));
    }
    static __device__ __forceinline__ void load(const uint16_t *g, uint16_t *s, int tid) {
        const uint32_t *src = (const uint32_t *)g;             // word w = entries (2w, 2w+1)
        uint32_t *dst = (uint32_t *)s;
        for (int w = tid; w < 2048; w += 256) {
            const uint32_t v = src[w];
            #pragma unroll
            for (int r = 0; r < TR; r++) dst[w * TR + r] = v;
        }
    }
};
// one vector's side word: prefix byte | fp16 scale bits << 8
static __device__ __forceinline__ uint32_t tq_t4_side(const uint8_t *side, size_t row, int nkv, int kv_head) {
    return *(const uint32_t *)(side + (row * nkv + kv_head) * 4);
}
static __device__ __forceinline__ float tq_t4_scale(uint32_t sw) {
    return __half2float(__ushort_as_half((unsigned short)((sw >> 8) & 0xFFFFu)));
}
// two adjacent states (ch, ch+1), ch even, from the 16-bit window that starts at stream byte ch/2:
// x = stream bytes (b-1 .. b+2) as a little-endian word, y = stream byte b+3, where b = 4k is the
// word's byte offset inside the 16-byte block and the window starts at byte b - 1 + t4.
static __device__ __forceinline__ uint32_t tq_t4_pair(uint32_t x, uint32_t y, int t4) {
    return __funnelshift_r(x, y, 8 * t4) & 0xFFFFu;
}
template <int TR = 1>
static __device__ __forceinline__ uint32_t tq_t4_bk(uint32_t pair, const uint16_t *tab, float sc, uint32_t cr = 0) {
    const float lo = __half2float(__ushort_as_half(tab[tq_t4_tab<TR>::idx(pair & 0xFFFu, cr)])) * sc;
    const float hi = __half2float(__ushort_as_half(tab[tq_t4_tab<TR>::idx((pair >> 4) & 0xFFFu, cr)])) * sc;
    return tq_pack_bf16(lo, hi);
}
// Q.K^T B-fragments of one key row for 32-channel group g (two k16 chunks): the thread (gr, t4)
// supplies channels {16c + 2t4, +1} (bk0[cc]) and {16c + 2t4 + 8, +9} (bk8[cc]) of chunk c = 2g + cc.
// w = the row's 16 code bytes of group g, prevb = the code byte before them (the prefix byte at g = 0).
template <int TR = 1>
static __device__ __forceinline__ void tq_t4_kfrags(const uint4 kp, uint32_t prevb, int t4, float sc,
                                                    const uint16_t *tab, uint32_t bk0[2], uint32_t bk8[2], uint32_t cr = 0) {
    const uint32_t w[4] = {kp.x, kp.y, kp.z, kp.w};
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        const uint32_t lowb = (k == 0) ? prevb : (w[k - 1] >> 24);
        const uint32_t x = (w[k] << 8) | lowb, y = w[k] >> 24;
        const uint32_t frag = tq_t4_bk<TR>(tq_t4_pair(x, y, t4), tab, sc, cr);
        if (k & 1) bk8[k >> 1] = frag; else bk0[k >> 1] = frag;
    }
}
// fp16 path (attention third pass, B2-eng): the table values themselves as the fp16 MMA operands -- no scale, no
// conversion; the vector's scale is applied to the Q.K^T score column (K) or folded into the P row (V) afterwards.
template <int TR = 1>
static __device__ __forceinline__ uint32_t tq_t4_bk_raw(uint32_t pair, const uint16_t *tab, uint32_t cr = 0) {
    return (uint32_t)tab[tq_t4_tab<TR>::idx(pair & 0xFFFu, cr)] | ((uint32_t)tab[tq_t4_tab<TR>::idx((pair >> 4) & 0xFFFu, cr)] << 16);
}
template <int TR = 1>
static __device__ __forceinline__ void tq_t4_kfrags_raw(const uint4 kp, uint32_t prevb, int t4,
                                                        const uint16_t *tab, uint32_t bk0[2], uint32_t bk8[2], uint32_t cr = 0) {
    const uint32_t w[4] = {kp.x, kp.y, kp.z, kp.w};
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        const uint32_t lowb = (k == 0) ? prevb : (w[k - 1] >> 24);
        const uint32_t x = (w[k] << 8) | lowb, y = w[k] >> 24;
        const uint32_t frag = tq_t4_bk_raw<TR>(tq_t4_pair(x, y, t4), tab, cr);
        if (k & 1) bk8[k >> 1] = frag; else bk0[k >> 1] = frag;
    }
}
// the fp16 scales of two side words as an fp16x2 (lo = sw0's, hi = sw1's)
static __device__ __forceinline__ uint32_t tq_t4_scale2(uint32_t sw0, uint32_t sw1) { return __byte_perm(sw0, sw1, 0x6521); }
// P.V B-fragment values of one key row for the thread's four adjacent channels ch..ch+3
// (ch = 32 warp + 4 gr): states (bits >> 4f) & 0xFFF of the 3 stream bytes at ch/2 .. ch/2 + 2.
static __device__ __forceinline__ uint32_t tq_t4_vbits(const uint8_t *codes_row, uint32_t sw, int cb /* = ch/2 */) {
    const uint32_t b0 = (cb == 0) ? (sw & 0xFFu) : (uint32_t)codes_row[cb - 1];
    const uint32_t b12 = *(const uint16_t *)(codes_row + cb);
    return b0 | (b12 << 8);
}
static __device__ __forceinline__ uint32_t tq_t4_f16x2_bf16(uint32_t h2) {
    return tq_pack_bf16(__half2float(__ushort_as_half((unsigned short)(h2 & 0xFFFFu))),
                        __half2float(__ushort_as_half((unsigned short)(h2 >> 16))));
}
template <int TR = 1>
static __device__ __forceinline__ float tq_t4_val(uint32_t bits, int f, const uint16_t *tab, float sc, uint32_t cr = 0) {
    return __half2float(__ushort_as_half(tab[tq_t4_tab<TR>::idx((bits >> (4 * f)) & 0xFFFu, cr)])) * sc;
}
template <int TR = 1>
static __device__ __forceinline__ uint32_t tq_t4_val_raw(uint32_t bits, int f, const uint16_t *tab, uint32_t cr = 0) {
    return (uint32_t)tab[tq_t4_tab<TR>::idx((bits >> (4 * f)) & 0xFFFu, cr)];
}
// scalar paths: value of channel d of a row (codes_row = the 128 code bytes, sw = its side word)
static __device__ __forceinline__ float tq_t4_dq(const uint8_t *codes_row, uint32_t sw, int d, const uint16_t *tab) {
    const int b = d >> 1;                                 // window starts at stream byte b (= code byte b - 1)
    const uint32_t lo = (b == 0) ? (sw & 0xFFu) : (uint32_t)codes_row[b - 1];
    const uint32_t hi = (uint32_t)codes_row[b];
    const uint32_t w = lo | (hi << 8);
    return __half2float(__ushort_as_half(tab[(w >> (4 * (d & 1))) & 0xFFFu])) * tq_t4_scale(sw);
}

// Flash-decoding split: positions are partitioned into `nchunks` ranges (one
// (head, chunk) item each, 128-aligned even split) so the persistent grid stays
// fed at long context; each chunk runs its own online softmax and either writes
// the final output directly (nchunks == 1) or a (m[16], l[16], O[16x256]) partial
// into the scratch slab, combined by a cheap merge phase per head.
#define TQ_AMM_PART_F 4128            // floats per (head, chunk) partial slab
static __device__ __forceinline__ float *tq_attn_mma_part_slab(float *scratch, int head,
                                                               int nchunks, int chunk) {
    return scratch + (size_t)(head * nchunks + chunk) * TQ_AMM_PART_F;
}

// Whole-block MMA tree attention for one (q-head, position-chunk) item (block =
// 256 threads, hd = 256). chunk/nchunks partition the prefix [0..base]; chunk 0
// also owns the 16-slot archive tile. part_out == NULL (single chunk): writes
// out[node * out_stride + head*hd + col] = attn * sigmoid(gate) and, when absmax
// != NULL, folds the per-node |out| max via atomicMax (persist pack2 contract).
// part_out != NULL: writes the unnormalized (m, l, O) partial instead.
// MODE 3 (TQ_KV_Q4): prefix K rows are rotated int4 codes, dequantized to bf16
// at B-fragment build and run through the bf16 HMMA exactly like mode 1; Q is
// FWHT-rotated at prep (so the fp32 tree-archive seam, whose K rows were
// rotated at kv_prep, shares the same A-fragments); V stays E4M3 like mode 2.
// PAIR mode (pair != 0, requires N <= 8, even GQA group): one item covers TWO
// q-heads of the same kv group -- `head` and `head + 1` -- with head/lo nodes in
// tile rows 0..N-1 and head+1 nodes in rows 8..8+N-1. The m16 tile is otherwise
// half-masked at the production N=8, so this reads the SAME K/V bytes once for
// two heads (the long-ctx phase is L1/L2-traffic/latency-bound, DRAM ~6%) at
// identical per-row MMA/softmax math: each row's scores, partials and epilogue
// are bit-identical to the unpaired item of its head.
template <int MODE>
static __device__ void tq_attn_mma_block(
    const float *q_proj_base, size_t q_stride, const uint16_t *q_norm_w,
    const float *kc, const float *vc,                       // fp32 prefix (mode 1)
    const uint8_t *kc8, const float *kscale,                // E4M3 prefix (mode 2)
    const uint8_t *vc8, const float *vscale,
    const uint8_t *kc4, const float *kq4s,                  // int4 prefix K (mode 3)
    const float *ka, const float *va,                       // fp32 tree archive
    const int *pos_arr, const int *anc, int base_in, int N,
    int head, int pair, int nh, int nkv, int hd, int kv_m,
    float eps, float rope_theta, int chunk, int nchunks,
    float *out, size_t out_stride, float *absmax, float *part_out,
    uint32_t *sm) {
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int gr = lane >> 2, t4 = lane & 3;                // m16n8 fragment coords
    const int group = nh / nkv, kv_head = head / group;
    const int base = base_in >= 0 ? base_in : pos_arr[0] - 1;
    const size_t cstride = (size_t)nkv * hd, astride = (size_t)kv_m;
    const float *kcb = kc ? kc + (size_t)kv_head * hd : NULL;
    const float *vcb = vc ? vc + (size_t)kv_head * hd : NULL;
    const uint8_t *kcb8 = kc8 ? kc8 + (size_t)kv_head * hd : NULL;
    const uint8_t *vcb8 = vc8 ? vc8 + (size_t)kv_head * hd : NULL;
    const uint8_t *kcb4 = kc4 ? kc4 + (size_t)kv_head * (hd >> 1) : NULL;
    const float *ksz4 = kq4s ? q4s_adv(kq4s, (size_t)kv_head * 16) : NULL;
    const size_t cstride4 = cstride >> 1, szstride = (size_t)nkv * 16;
    const float *kscb = kscale ? kscale + kv_head : NULL;
    const float *vscb = vscale ? vscale + kv_head : NULL;
    const float *kab = ka + (size_t)kv_head * hd;
    const float *vab = va + (size_t)kv_head * hd;
    float *qtmp = (float *)(sm + TQ_AMM_QTMP);
    float *red = (float *)(sm + TQ_AMM_RED);
    float *wmax = (float *)(sm + TQ_AMM_WMAX), *wsum = (float *)(sm + TQ_AMM_WSUM);
    float *mst = (float *)(sm + TQ_AMM_MST), *lst = (float *)(sm + TQ_AMM_LST);
    float *alph = (float *)(sm + TQ_AMM_ALPH);
    float *qsc = (float *)(sm + TQ_AMM_QSC);

    // ---- init: zero the fragment staging (rows >= N stay inert), softmax state,
    // ancestor bitmasks (bit j = archive slot j is on node i's chain incl. self).
    // PAIR: tile row r maps to node (r & 7) of head + (r >> 3); rows 8-15 carry
    // head+1's copies of the same nodes, so they take the same node mask.
    for (int i = tid; i < TQ_AMM_PST + 1024; i += 256) sm[i] = 0;
    if (tid < 16) {
        mst[tid] = TQ_AMM_NINF;
        lst[tid] = 0.0f;
        uint32_t m = 0;
        int node = pair ? (tid & 7) : tid;
        if (node < N) {
            int len = pos_arr[node] - base - 1 + 1;         // chain incl. the node itself
            const int *an = anc + node * TQ_SPEC_MAX_N;
            for (int d = 0; d < len; d++) m |= 1u << an[d];
        }
        sm[TQ_AMM_AMASK + tid] = m;
    }
    __syncthreads();

    // ---- Q prep: per tile row, the same norm+rope math as the scalar kernels,
    // then packed straight into MMA A-fragments (bf16 always; e4m3 with a
    // per-row pow2 scale for mode 2)
    const int nqr = pair ? 16 : N;                          // packed tile rows
    for (int rq = 0; rq < nqr; rq++) {
        const int n = pair ? (rq & 7) : rq;                 // node of this row
        if (pair && n >= N) continue;                       // uniform whole-block skip
        const int hh = head + (pair ? (rq >> 3) : 0);       // this row's q-head
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
        __syncthreads();                                    // red[] free for reuse
        float qscl = 1.0f;
        if (MODE == 2) {
            qscl = tq_kv_block_scale(qv, red, red + 8);     // pow2 absmax scale
            if (tid == 0) qsc[rq] = qscl;
        }
        qtmp[tid] = qv;
        __syncthreads();
        if (MODE == 3)
            tq_fwht256(qtmp, tid);   // rotated-space q (matches int4 K + rotated arch)
        // bf16 A-frags: 128 words per row (16 k16-chunks x 4 quad-lanes x 2 regs)
        if (tid < 128) {
            int c16 = tid >> 3, tt = (tid >> 1) & 3, rsel = tid & 1;
            int d0 = 16 * c16 + 2 * tt + 8 * rsel;
            int l = 4 * (rq & 7) + tt, r = ((rq >> 3) & 1) + 2 * rsel;
            sm[TQ_AMM_QA + (c16 * 32 + l) * 4 + r] = tq_pack_bf16(qtmp[d0], qtmp[d0 + 1]);
        }
        // e4m3 A-frags (mode 2): 64 words per row (8 k32-chunks x 4 x 2)
        if (MODE == 2 && tid < 64) {
            int c32 = tid >> 3, tt = (tid >> 1) & 3, rsel = tid & 1;
            int d0 = 32 * c32 + 4 * tt + 16 * rsel;
            int l = 4 * (rq & 7) + tt, r = ((rq >> 3) & 1) + 2 * rsel;
            float inv = 1.0f / qscl;
            sm[TQ_AMM_QA8 + (c32 * 32 + l) * 4 + r] =
                tq_pack_e4m3x4(qtmp[d0] * inv, qtmp[d0 + 1] * inv,
                               qtmp[d0 + 2] * inv, qtmp[d0 + 3] * inv);
        }
        __syncthreads();
    }

    // ---- main loop: prefix super-tiles (8 warps x 16 positions) + 1 archive tile
    float oacc[4][4];                                       // O slice: 16 rows x 32 cols
    #pragma unroll
    for (int f = 0; f < 4; f++)
        #pragma unroll
        for (int j = 0; j < 4; j++) oacc[f][j] = 0.0f;
    const float rsc = rsqrtf((float)hd);
    // 128-aligned even split of the prefix super-tiles across the chunks; the
    // archive tile rides with chunk 0 (always present, also when base < 0).
    const int nst_all = base >= 0 ? (base + 128) >> 7 : 0;  // ceil((base+1)/128)
    const int st_per = (nst_all + nchunks - 1) / nchunks;
    const int st_lo = chunk * st_per;
    int st_hi_t = (chunk + 1) * st_per;
    const int st_hi = st_hi_t < nst_all ? st_hi_t : nst_all;
    const int do_arch = (chunk == 0);
    const int nst = st_hi > st_lo ? st_hi - st_lo : 0;      // this chunk's super-tiles
    const uint32_t amask_lo = sm[TQ_AMM_AMASK + gr], amask_hi = sm[TQ_AMM_AMASK + 8 + gr];

    for (int it = 0; it < nst + do_arch; it++) {
        const int is_arch = (it == nst);
        const int nwt = is_arch ? 1 : 8;                    // active score tiles
        const int t0 = (st_lo + it) << 7;
        float s0[4], s1[4];                                 // the warp's 16x16 score tile
        // phase 1: scores via MMA + mask + per-row tile max
        if (warp < nwt) {
            #pragma unroll
            for (int j = 0; j < 4; j++) { s0[j] = 0.0f; s1[j] = 0.0f; }
            if (is_arch) {
                // tree tile: 16 archive slots, Q bf16 x arch K bf16
                const float *kr0 = kab + (size_t)gr * astride;        // slot gr
                const float *kr1 = kab + (size_t)(8 + gr) * astride;  // slot 8+gr
                #pragma unroll
                for (int c = 0; c < 16; c++) {
                    const uint32_t *a = &sm[TQ_AMM_QA + (c * 32 + lane) * 4];
                    int d0 = 16 * c + 2 * t4;
                    float2 x0 = *(const float2 *)(kr0 + d0);
                    float2 x1 = *(const float2 *)(kr0 + d0 + 8);
                    tq_mma_bf16_m16n8k16(s0, a, tq_pack_bf16(x0.x, x0.y),
                                         tq_pack_bf16(x1.x, x1.y));
                    float2 y0 = *(const float2 *)(kr1 + d0);
                    float2 y1 = *(const float2 *)(kr1 + d0 + 8);
                    tq_mma_bf16_m16n8k16(s1, a, tq_pack_bf16(y0.x, y0.y),
                                         tq_pack_bf16(y1.x, y1.y));
                }
                // scale + per-(row, slot) ancestor mask
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    int col = 2 * t4 + (j & 1);             // slot for s0; +8 for s1
                    uint32_t am = (j < 2) ? amask_lo : amask_hi;
                    s0[j] = ((am >> col) & 1) ? s0[j] * rsc : TQ_AMM_NINF;
                    s1[j] = ((am >> (col + 8)) & 1) ? s1[j] * rsc : TQ_AMM_NINF;
                }
            } else {
                const int tp = t0 + warp * 16;              // this warp's 16 positions
                int p0 = tp + gr, p1 = tp + 8 + gr;
                if (MODE == 2) {
                    const uint8_t *kr0 = kcb8 + (size_t)min(p0, base) * cstride;
                    const uint8_t *kr1 = kcb8 + (size_t)min(p1, base) * cstride;
                    #pragma unroll
                    for (int c = 0; c < 8; c++) {
                        const uint32_t *a = &sm[TQ_AMM_QA8 + (c * 32 + lane) * 4];
                        int d0 = 32 * c + 4 * t4;
                        tq_mma_e4m3_m16n8k32(s0, a, *(const uint32_t *)(kr0 + d0),
                                             *(const uint32_t *)(kr0 + d0 + 16));
                        tq_mma_e4m3_m16n8k32(s1, a, *(const uint32_t *)(kr1 + d0),
                                             *(const uint32_t *)(kr1 + d0 + 16));
                    }
                    // pull the pow2 scales back in: per-row Q scale x per-pos K scale
                    float qs_lo = qsc[gr] * rsc, qs_hi = qsc[8 + gr] * rsc;
                    #pragma unroll
                    for (int h = 0; h < 2; h++) {
                        float *s = h ? s1 : s0;
                        int p = (h ? p1 : p0) - gr + 2 * t4;        // col 2t position
                        float k0 = kscb[(size_t)min(p, base) * nkv];
                        float k1 = kscb[(size_t)min(p + 1, base) * nkv];
                        s[0] *= qs_lo * k0; s[1] *= qs_lo * k1;
                        s[2] *= qs_hi * k0; s[3] *= qs_hi * k1;
                    }
                } else if (MODE == 3) {
                    // int4 K: dequant (code - zp) * scale -> bf16 -> HMMA. The
                    // (d0, d0+1) channel pair shares one byte and one group
                    // (g = c>>1: even d0, groups of 32); k16-chunk c covers
                    // channels 16c..16c+15, all in group c>>1. The (scale, zp)
                    // pair is one 32-bit load hoisted over its two c's (same
                    // values/order as per-half loads -- bit-exact, fewer
                    // scattered L1TEX transactions in the hottest loop).
                    const uint8_t *kr0 = kcb4 + (size_t)min(p0, base) * cstride4;
                    const uint8_t *kr1 = kcb4 + (size_t)min(p1, base) * cstride4;
                    const float *sz0 = q4s_adv(ksz4, (size_t)min(p0, base) * szstride);
                    const float *sz1 = q4s_adv(ksz4, (size_t)min(p1, base) * szstride);
                    #pragma unroll
                    for (int g = 0; g < 8; g++) {
                        float2 p0s = q4s_pair(sz0, 2 * g);
                        float2 p1s = q4s_pair(sz1, 2 * g);
                        float sc0 = p0s.x, zp0 = p0s.y;
                        float sc1 = p1s.x, zp1 = p1s.y;
                        #pragma unroll
                        for (int cc = 0; cc < 2; cc++) {
                            int c = 2 * g + cc;
                            const uint32_t *a = &sm[TQ_AMM_QA + (c * 32 + lane) * 4];
                            // the quad's 8 int4-code bytes [8c .. 8c+8) in ONE
                            // 8-byte read-only load (lane t4 uses bytes t4 and
                            // t4+4); same bytes as the scattered loads -- bit-exact
                            uint64_t w0 = __ldg((const unsigned long long *)(kr0 + 8 * c));
                            uint64_t w1 = __ldg((const unsigned long long *)(kr1 + 8 * c));
                            uint8_t a0 = (uint8_t)(w0 >> (8 * t4));
                            uint8_t a8 = (uint8_t)(w0 >> (8 * t4 + 32));
                            uint8_t b0v = (uint8_t)(w1 >> (8 * t4));
                            uint8_t b8 = (uint8_t)(w1 >> (8 * t4 + 32));
                            tq_mma_bf16_m16n8k16(
                                s0, a,
                                tq_pack_bf16(((float)(a0 & 15) - zp0) * sc0,
                                             ((float)(a0 >> 4) - zp0) * sc0),
                                tq_pack_bf16(((float)(a8 & 15) - zp0) * sc0,
                                             ((float)(a8 >> 4) - zp0) * sc0));
                            tq_mma_bf16_m16n8k16(
                                s1, a,
                                tq_pack_bf16(((float)(b0v & 15) - zp1) * sc1,
                                             ((float)(b0v >> 4) - zp1) * sc1),
                                tq_pack_bf16(((float)(b8 & 15) - zp1) * sc1,
                                             ((float)(b8 >> 4) - zp1) * sc1));
                        }
                    }
                    #pragma unroll
                    for (int j = 0; j < 4; j++) { s0[j] *= rsc; s1[j] *= rsc; }
                } else {
                    const float *kr0 = kcb + (size_t)min(p0, base) * cstride;
                    const float *kr1 = kcb + (size_t)min(p1, base) * cstride;
                    #pragma unroll
                    for (int c = 0; c < 16; c++) {
                        const uint32_t *a = &sm[TQ_AMM_QA + (c * 32 + lane) * 4];
                        int d0 = 16 * c + 2 * t4;
                        float2 x0 = *(const float2 *)(kr0 + d0);
                        float2 x1 = *(const float2 *)(kr0 + d0 + 8);
                        tq_mma_bf16_m16n8k16(s0, a, tq_pack_bf16(x0.x, x0.y),
                                             tq_pack_bf16(x1.x, x1.y));
                        float2 y0 = *(const float2 *)(kr1 + d0);
                        float2 y1 = *(const float2 *)(kr1 + d0 + 8);
                        tq_mma_bf16_m16n8k16(s1, a, tq_pack_bf16(y0.x, y0.y),
                                             tq_pack_bf16(y1.x, y1.y));
                    }
                    #pragma unroll
                    for (int j = 0; j < 4; j++) { s0[j] *= rsc; s1[j] *= rsc; }
                }
                // causal cut at base (uniform across rows)
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    int c0 = tp + 2 * t4 + (j & 1);
                    if (c0 > base) s0[j] = TQ_AMM_NINF;
                    if (c0 + 8 > base) s1[j] = TQ_AMM_NINF;
                }
            }
            float lm_lo = fmaxf(fmaxf(s0[0], s0[1]), fmaxf(s1[0], s1[1]));
            float lm_hi = fmaxf(fmaxf(s0[2], s0[3]), fmaxf(s1[2], s1[3]));
            #pragma unroll
            for (int o = 1; o <= 2; o <<= 1) {
                lm_lo = fmaxf(lm_lo, __shfl_xor_sync(0xffffffffu, lm_lo, o));
                lm_hi = fmaxf(lm_hi, __shfl_xor_sync(0xffffffffu, lm_hi, o));
            }
            if (t4 == 0) { wmax[warp * 16 + gr] = lm_lo; wmax[warp * 16 + 8 + gr] = lm_hi; }
        }
        __syncthreads();
        // online-softmax state step (one warp owns the 16 rows)
        if (warp == 0 && lane < 16) {
            float m_old = mst[lane], m_new = m_old;
            for (int w = 0; w < nwt; w++) m_new = fmaxf(m_new, wmax[w * 16 + lane]);
            alph[lane] = expf(m_old - m_new);
            mst[lane] = m_new;
        }
        __syncthreads();
        // phase 2: p = exp(s - m_new), per-row tile sums, pack P into A-frag staging
        if (warp < nwt) {
            float mn_lo = mst[gr], mn_hi = mst[8 + gr];
            float p00 = expf(s0[0] - mn_lo), p01 = expf(s0[1] - mn_lo);
            float p02 = expf(s0[2] - mn_hi), p03 = expf(s0[3] - mn_hi);
            float p10 = expf(s1[0] - mn_lo), p11 = expf(s1[1] - mn_lo);
            float p12 = expf(s1[2] - mn_hi), p13 = expf(s1[3] - mn_hi);
            float ls_lo = p00 + p01 + p10 + p11;
            float ls_hi = p02 + p03 + p12 + p13;
            #pragma unroll
            for (int o = 1; o <= 2; o <<= 1) {
                ls_lo += __shfl_xor_sync(0xffffffffu, ls_lo, o);
                ls_hi += __shfl_xor_sync(0xffffffffu, ls_hi, o);
            }
            if (t4 == 0) { wsum[warp * 16 + gr] = ls_lo; wsum[warp * 16 + 8 + gr] = ls_hi; }
            if (MODE >= 2 && !is_arch) {
                // fold the V row scale into p (denominator keeps unscaled p)
                int p0c = t0 + warp * 16 + 2 * t4, p1c = p0c + 8;
                float v00 = vscb[(size_t)min(p0c, base) * nkv];
                float v01 = vscb[(size_t)min(p0c + 1, base) * nkv];
                float v10 = vscb[(size_t)min(p1c, base) * nkv];
                float v11 = vscb[(size_t)min(p1c + 1, base) * nkv];
                p00 *= v00; p01 *= v01; p02 *= v00; p03 *= v01;
                p10 *= v10; p11 *= v11; p12 *= v10; p13 *= v11;
            }
            uint32_t *pst = &sm[TQ_AMM_PST + (warp * 32 + lane) * 4];
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
        // phase 3: O slice rescale + P*V MMAs over this iteration's tiles
        {
            float a_lo = alph[gr], a_hi = alph[8 + gr];
            #pragma unroll
            for (int f = 0; f < 4; f++) {
                oacc[f][0] *= a_lo; oacc[f][1] *= a_lo;
                oacc[f][2] *= a_hi; oacc[f][3] *= a_hi;
            }
            for (int tt = 0; tt < nwt; tt++) {
                const uint32_t *pa = &sm[TQ_AMM_PST + (tt * 32 + lane) * 4];
                uint32_t a[4] = {pa[0], pa[1], pa[2], pa[3]};
                #pragma unroll
                for (int f = 0; f < 4; f++) {
                    int col = warp * 32 + f * 8 + gr;
                    uint32_t b0, b1;
                    if (is_arch) {
                        int r0 = 2 * t4;
                        b0 = tq_pack_bf16(vab[(size_t)r0 * astride + col],
                                          vab[(size_t)(r0 + 1) * astride + col]);
                        b1 = tq_pack_bf16(vab[(size_t)(r0 + 8) * astride + col],
                                          vab[(size_t)(r0 + 9) * astride + col]);
                    } else if (MODE >= 2) {
                        int r0 = t0 + tt * 16 + 2 * t4;
                        b0 = tq_pack_bf16(
                            tq_e4m3_dec_fast(vcb8[(size_t)min(r0, base) * cstride + col]),
                            tq_e4m3_dec_fast(vcb8[(size_t)min(r0 + 1, base) * cstride + col]));
                        b1 = tq_pack_bf16(
                            tq_e4m3_dec_fast(vcb8[(size_t)min(r0 + 8, base) * cstride + col]),
                            tq_e4m3_dec_fast(vcb8[(size_t)min(r0 + 9, base) * cstride + col]));
                    } else {
                        int r0 = t0 + tt * 16 + 2 * t4;
                        b0 = tq_pack_bf16(vcb[(size_t)min(r0, base) * cstride + col],
                                          vcb[(size_t)min(r0 + 1, base) * cstride + col]);
                        b1 = tq_pack_bf16(vcb[(size_t)min(r0 + 8, base) * cstride + col],
                                          vcb[(size_t)min(r0 + 9, base) * cstride + col]);
                    }
                    tq_mma_bf16_m16n8k16(oacc[f], a, b0, b1);
                }
            }
        }
        __syncthreads();                                    // pst/wmax reuse next iter
    }

    // ---- chunk partial: unnormalized (m, l, O) for the merge phase. PAIR:
    // tile rows 8-15 are head+1's copies of nodes 0-7 -> they land in head+1's
    // per-head slab rows 0-7 (the merge phase stays head-indexed and unchanged).
    if (part_out) {
        float *ps = tq_attn_mma_part_slab(part_out, head, nchunks, chunk);
        float *ph = pair ? tq_attn_mma_part_slab(part_out, head + 1, nchunks, chunk)
                         : ps + 8;                     // unpaired: rows 8-15 in-slab
        if (tid < 8)       { ps[tid] = mst[tid]; ps[16 + tid] = lst[tid]; }
        else if (tid < 16) { ph[tid - 8] = mst[tid]; ph[16 + tid - 8] = lst[tid]; }
        float *po = ps + 32, *qo = pair ? ph + 32 : po + 8 * 256;
        #pragma unroll
        for (int f = 0; f < 4; f++) {
            int col = warp * 32 + f * 8 + 2 * t4;
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                po[(size_t)gr * 256 + col + j] = oacc[f][j];
                qo[(size_t)gr * 256 + col + j] = oacc[f][2 + j];
            }
        }
        __syncthreads();
        return;
    }

    // ---- epilogue: out = (O / l) * sigmoid(gate); optional per-node absmax fold
    const int n_hi = pair ? gr : 8 + gr;                // hi half: node / head of row 8+gr
    const int h_hi = pair ? head + 1 : head;
    float inv_lo = 1.0f / lst[gr], inv_hi = 1.0f / lst[8 + gr];
    float am_lo = 0.0f, am_hi = 0.0f;
    #pragma unroll
    for (int f = 0; f < 4; f++) {
        int col = warp * 32 + f * 8 + 2 * t4;
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            if (gr < N) {
                const float *q_proj = q_proj_base + (size_t)gr * q_stride;
                float gate = q_proj[head * (2 * hd) + hd + col + j];
                float v = oacc[f][j] * inv_lo * (1.0f / (1.0f + expf(-gate)));
                out[(size_t)gr * out_stride + head * hd + col + j] = v;
                am_lo = fmaxf(am_lo, fabsf(v));
            }
            if (n_hi < N) {
                const float *q_proj = q_proj_base + (size_t)n_hi * q_stride;
                float gate = q_proj[h_hi * (2 * hd) + hd + col + j];
                float v = oacc[f][2 + j] * inv_hi * (1.0f / (1.0f + expf(-gate)));
                out[(size_t)n_hi * out_stride + h_hi * hd + col + j] = v;
                am_hi = fmaxf(am_hi, fabsf(v));
            }
        }
    }
    if (absmax) {
        #pragma unroll
        for (int o = 1; o <= 2; o <<= 1) {
            am_lo = fmaxf(am_lo, __shfl_xor_sync(0xffffffffu, am_lo, o));
            am_hi = fmaxf(am_hi, __shfl_xor_sync(0xffffffffu, am_hi, o));
        }
        if (t4 == 0) { wmax[warp * 16 + gr] = am_lo; wmax[warp * 16 + 8 + gr] = am_hi; }
        __syncthreads();
        if (warp == 0 && lane < 16) {
            int node = pair ? (lane & 7) : lane;        // rows 8-15 fold into node 0-7
            if (node < N) {
                float m = wmax[lane];
                for (int w = 1; w < 8; w++) m = fmaxf(m, wmax[w * 16 + lane]);
                atomicMax((unsigned int *)&absmax[node], __float_as_uint(m));
            }
        }
    }
    __syncthreads();
}
