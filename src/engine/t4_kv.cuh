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
