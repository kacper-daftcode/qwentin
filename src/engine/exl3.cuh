// EXL3 trellis weights (dequant GEMM / GEMV, launchers).
// Part of the forward_qwen.cu unity build: included in order, not a standalone header.

// ===== EXL3 trellis weights on the wide path (TQ_FLAG_EXL3_WEIGHTS) =====
// A Linear y = x W^T with W [M][K] is stored as W = diag(svh) (H128 W_hat H128)^T diag(suh):
// W_hat is [K][M] in 16x16 trellis tiles (int16 [K/16][M/16][16*KB]), H128 the 128-point
// Sylvester Hadamard scaled 1/sqrt(128), applied per 128-block on each side. Forward:
//   xh = H128(x * suh)  ->  y_hat = xh W_hat  ->  y = H128(y_hat) * svh.
// Tile stream: 256*KB bits, circular, MSB-first inside uint32 words; weight t (tensor-core
// fragment order, t = lane*8 + j) has the 16-bit codeword ending at stream bit (t+1)*KB.
// mul1 codebook. Same arithmetic as exllamav3 (exl3_dq.cuh / codebook.cuh, cb == 2), so the
// dequantized fp16 weights are bit-identical to that engine's.
static __device__ __forceinline__ __half tq_exl3_mul1(uint32_t x) {
    x *= 0x83DCD12Du;
    uint32_t sum = __dp4a(x, 0x01010101u, 0x6400u);           // byte sum + 1024.0 in fp16 bits
    return __hfma(__ushort_as_half((uint16_t)sum), __ushort_as_half(0x1eee), __ushort_as_half(0xc931));
}

static __device__ __forceinline__ uint32_t tq_exl3_pack(__half lo, __half hi) {
    return (uint32_t)__half_as_ushort(lo) | ((uint32_t)__half_as_ushort(hi) << 16);
}
// Two codewords -> one packed fp16 pair (an A-fragment word): the per-element arithmetic of tq_exl3_mul1 (same
// IMAD / DP4A, one HFMA2 = two fp16 fmas with the same rounding), paired in fragment order -- ptxas pairs the scalar
// version's hfmas arbitrarily and spends 8 PRMTs per tile re-pairing the results (session 10, prefill thread).
static __device__ __forceinline__ uint32_t tq_exl3_mul1x2(uint32_t x0, uint32_t x1) {
    x0 *= 0x83DCD12Du; x1 *= 0x83DCD12Du;
    uint32_t s0 = __dp4a(x0, 0x01010101u, 0x6400u), s1 = __dp4a(x1, 0x01010101u, 0x6400u);
    uint32_t p = __byte_perm(s0, s1, 0x5410);                 // lo16(s0) | lo16(s1) << 16
    __half2 h = __hfma2(*reinterpret_cast<const __half2 *>(&p),
                        __halves2half2(__ushort_as_half(0x1eee), __ushort_as_half(0x1eee)),
                        __halves2half2(__ushort_as_half(0xc931), __ushort_as_half(0xc931)));
    return *reinterpret_cast<uint32_t *>(&h);
}

// Codeword extraction (the exllamav3 dq4 / dq2x2 / dq8 scheme). The lane's 8 weights are the
// A operand of mma.m16n8k16 (rows = output features n, cols = k): a0 (n=l/4, k=2(l%4)+{0,1}),
// a1 (n+8), a2 (k+8,+9), a3 (n+8, k+8,+9) = codewords t = 8*lane + {0,1}, {4,5}, {2,3}, {6,7}.
// They are taken in groups of NCW consecutive codewords, each group from ONE pair of stream
// words (a = word of the group's first bit, b = word of its last bit) via a 64-bit shift. The
// group fits two words because of the fragment order: 8 codewords start at 8*lane*KB, so the
// offset of a group's first bit inside its word plus (NCW-1)*KB+16 never exceeds 64 for the
// NCW chosen per KB (1..3 bits: 8; 4,5,6,8 bits: 4; 7 bits: 2). Word indices and shifts depend
// on the lane only, so they are computed once per warp; per tile each lane loads 2 words per
// group (L1 absorbs the overlap between lanes; the tile leaves DRAM once).
template <int KB> struct tq_exl3_ncw { static constexpr int v = (KB <= 3) ? 8 : (KB == 7 ? 2 : 4); };

template <int KB>
struct tq_exl3_win {
    static constexpr int NCW = tq_exl3_ncw<KB>::v;
    static constexpr int NGRP = 8 / NCW;
    static constexpr int NW = 8 * KB;
    int ia[NGRP], ib[NGRP], s2[NGRP];
    __device__ __forceinline__ void init(int lane) {
        #pragma unroll
        for (int g = 0; g < NGRP; g++) {
            int t = lane * 8 + g * NCW;
            int b0 = (t + 257) * KB - 16;               // start of the group's first codeword
            int b2 = b0 + (NCW - 1) * KB + 16;          // end of its last codeword
            int i0 = b0 >> 5, i2 = (b2 - 1) >> 5;
            s2[g] = ((i2 + 1) << 5) - b2;
            ia[g] = i0 % NW;
            ib[g] = i2 % NW;
        }
    }
    // Per-lane direct loads of the group words (2 x 32-bit per group per tile per lane). STREAM: evict-first
    // (decode-size N reads a tile once); wide N re-reads every tile from its token-group units, so it keeps the
    // normal L2 policy (ld.global.nc) -- ncu had the 128-column kernel at 72-77 % L2 hits, long-scoreboard bound.
    template <bool STREAM = true>
    __device__ __forceinline__ void load(const uint32_t *tile, uint32_t *r) const {
        #pragma unroll
        for (int g = 0; g < NGRP; g++) {
            if (STREAM) { r[2 * g] = __ldcs(tile + ia[g]); r[2 * g + 1] = __ldcs(tile + ib[g]); }
            else        { r[2 * g] = __ldg(tile + ia[g]);  r[2 * g + 1] = __ldg(tile + ib[g]); }
        }
    }
    // L2 prefetch of a tile some tiles ahead of the register ring (one line per lane group; no registers)
    __device__ __forceinline__ void prefetch(const uint32_t *tile) const {
        asm volatile("prefetch.global.L2 [%0];" :: "l"(tile + ia[0]));
    }
    // PAIR: decode the codewords two at a time straight into the fragment words (tq_exl3_mul1x2; the wide units).
    template <bool PAIR = false>
    __device__ __forceinline__ void frag(const uint32_t *r, uint32_t &a0, uint32_t &a1, uint32_t &a2, uint32_t &a3) const {
        if (PAIR) {
            uint32_t cw[8];
            #pragma unroll
            for (int g = 0; g < NGRP; g++) {
                uint64_t m = ((uint64_t)r[2 * g] << 32) | (uint64_t)r[2 * g + 1];
                #pragma unroll
                for (int j = 0; j < NCW; j++)
                    cw[g * NCW + j] = (uint32_t)(m >> (s2[g] + (NCW - 1 - j) * KB)) & 0xffffu;
            }
            a0 = tq_exl3_mul1x2(cw[0], cw[1]);
            a2 = tq_exl3_mul1x2(cw[2], cw[3]);
            a1 = tq_exl3_mul1x2(cw[4], cw[5]);
            a3 = tq_exl3_mul1x2(cw[6], cw[7]);
            return;
        }
        __half w[8];
        #pragma unroll
        for (int g = 0; g < NGRP; g++) {
            uint64_t m = ((uint64_t)r[2 * g] << 32) | (uint64_t)r[2 * g + 1];
            #pragma unroll
            for (int j = 0; j < NCW; j++)
                w[g * NCW + j] = tq_exl3_mul1((uint32_t)(m >> (s2[g] + (NCW - 1 - j) * KB)) & 0xffffu);
        }
        a0 = tq_exl3_pack(w[0], w[1]);
        a2 = tq_exl3_pack(w[2], w[3]);
        a1 = tq_exl3_pack(w[4], w[5]);
        a3 = tq_exl3_pack(w[6], w[7]);
    }
};

static __device__ __forceinline__ float tq_fp16_to_float(uint16_t bits) {
    return __half2float(__ushort_as_half(bits));
}

// 4-then-32 butterfly over a warp: 128-point Hadamard of (lane*4 + i) values (natural order),
// same sign convention as exllamav3 shuffle_had_f4x32.
static __device__ __forceinline__ void tq_had128_warp(float &v0, float &v1, float &v2, float &v3, int lane) {
    float s0 = v0 + v1, d0 = v0 - v1, s1 = v2 + v3, d1 = v2 - v3;
    v0 = s0 + s1; v1 = d0 + d1; v2 = s0 - s1; v3 = d0 - d1;
    #pragma unroll
    for (int i = 1; i < 32; i <<= 1) {
        float p0 = __shfl_xor_sync(0xffffffffu, v0, i);
        float p1 = __shfl_xor_sync(0xffffffffu, v1, i);
        float p2 = __shfl_xor_sync(0xffffffffu, v2, i);
        float p3 = __shfl_xor_sync(0xffffffffu, v3, i);
        float sg = (lane & i) ? -1.0f : 1.0f;
        v0 = sg * v0 + p0; v1 = sg * v1 + p1; v2 = sg * v2 + p2; v3 = sg * v3 + p3;
    }
}

#define TQ_EXL3_RSCALE 0.08838834764831845f   // 1/sqrt(128)

// x fp32 [N][K] token-major -> fp16 B fragments of H128(x*suh) for mma.m16n8k16:
// layout [k16 tile][group pair gp = n/16][lane][2 groups][2 words] (GP = ceil(Gt/2) pairs);
// lane = (n%8)*4 + tig holds k = 2*tig+{0,1} (word 0) and k+8,+9 (word 1). A warp's 128-bit
// load of a group pair is one contiguous 512-byte run. One warp per (token, 128-block of K);
// tokens N..Npad-1 (the pad of the last group) are written as zeros, so no per-call memset.
// Up to three Linears that share x (q/k/v, gate/up, qkv/z) in one launch: blockIdx.y picks the (buffer, suh) pair.
struct tq_bfrag_set { uint32_t *b0, *b1, *b2; const uint16_t *s0, *s1, *s2; };

__global__ void __launch_bounds__(128) k_tq_exl3_bfrag(tq_bfrag_set set, const float *x, const float *x2,
                                                       int K, int N, int Npad, int Kt16, int GP) {
    // the next GEMM may start fetching its weights; x comes from the predecessor and the previous GEMM
    // may still read bfrag, so wait before touching either
    tq_pdl_enter();
    uint32_t *bfrag = blockIdx.y == 0 ? set.b0 : blockIdx.y == 1 ? set.b1 : set.b2;
    const uint16_t *suh = blockIdx.y == 0 ? set.s0 : blockIdx.y == 1 ? set.s1 : set.s2;
    int wg = blockIdx.x * 4 + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    int nblk = K >> 7;
    int n = wg / nblk, bk = wg - n * nblk;
    if (n >= Npad) return;
    float4 v = n < N ? *(const float4 *)(x + (size_t)n * K + bk * 128 + lane * 4) : make_float4(0.f, 0.f, 0.f, 0.f);
    if (x2 && n < N) {           // fused SiLU gate: x = silu(gate) * up, the expression of k_tq_silu_mul (same bits)
        float4 u = *(const float4 *)(x2 + (size_t)n * K + bk * 128 + lane * 4);
        v.x = (v.x / (1.0f + expf(-v.x))) * u.x; v.y = (v.y / (1.0f + expf(-v.y))) * u.y;
        v.z = (v.z / (1.0f + expf(-v.z))) * u.z; v.w = (v.w / (1.0f + expf(-v.w))) * u.w;
    }
    const uint16_t *sp = suh + bk * 128 + lane * 4;
    float v0 = v.x * tq_fp16_to_float(sp[0]), v1 = v.y * tq_fp16_to_float(sp[1]);
    float v2 = v.z * tq_fp16_to_float(sp[2]), v3 = v.w * tq_fp16_to_float(sp[3]);
    tq_had128_warp(v0, v1, v2, v3, lane);
    __half h0 = __float2half_rn(v0 * TQ_EXL3_RSCALE), h1 = __float2half_rn(v1 * TQ_EXL3_RSCALE);
    __half h2 = __float2half_rn(v2 * TQ_EXL3_RSCALE), h3 = __float2half_rn(v3 * TQ_EXL3_RSCALE);
    int kt = bk * 8 + (lane >> 2);
    int w = (lane & 2) ? 1 : 0;
    int tig = (lane & 1) * 2;
    int g = n >> 3, r = n & 7;
    uint32_t *dst = bfrag + (((size_t)kt * GP + (g >> 1)) * 32 + r * 4 + tig) * 4 + (g & 1) * 2 + w;
    dst[0] = tq_exl3_pack(h0, h1);
    dst[4] = tq_exl3_pack(h2, h3);                // next lane (tig+1), same group
}

// Streaming GEMM with the output Hadamard fused. Work unit = (128 output features, one
// k-split); a block of 8 warps (one 16-feature trellis tile each) takes units from a counter
// until none are left (no tail from a static grid; the grid is sized to what is resident).
// For k_splits == 1 the block's smem epilogue applies H128 and svh and writes out directly.
// For k_splits > 1 every split stores its y_hat partial; the LAST split to arrive at a
// feature block (per-block arrival counter) sums the partials in split order -- so the
// result does not depend on arrival order -- and runs the same epilogue. The counters
// (ctr[0] units, ctr[1] blocks done, ctr[2..] per-feature-block arrivals) are reset by the
// block that finishes last; launches rotate through counter slots, so a launch that starts
// early under PDL never takes units from its predecessor's counters.
// Unit = (128 output features) x (8*NG tokens = one token group) x (one k-split); ntg token
// groups. Wide N runs as several token groups over the same weights (L2 serves the re-reads)
// instead of huge accumulator sets or deep k-splits.
// KB=7 keeps 8 words per lane per tile in the prefetch ring (NCW=2): at 4 blocks/SM (64 registers)
// it spills, and the local-memory traffic halves its decode bandwidth (750 vs ~1400 GB/s for KB=6);
// 2 blocks/SM (128 registers) run it spill-free at ~1200 GB/s (3 blocks at 80 registers still spill).
// Wide units (NG = 8, the 128-column prefill waves). ncu (session 10, B2-eng): tensor pipe 41-47 % active, 3-4 warps per
// scheduler with 0.5-0.7 eligible, a warp issues every ~10 cycles of which 4 are long-scoreboard (the per-lane weight
// loads): latency-bound. prefetch.global.L2 of the tile TQ_EXL3_L2PF tiles ahead of the register ring turns the ring's
// loads into L2 hits (no registers): <5,8> 105 -> 100 us, <6,8> 78 -> 75 (-3.6 % of the wave's GEMM time, bitwise);
// KB=7 (4 groups, 8 ring words per lane) got slower with it and keeps the plain ring. The normal-L2-policy loads
// (TQ_EXL3_WIDE_LDG=1, ld.global.nc instead of evict-first) measured +5 % and stay off. Both measured 2026-10-16.
#ifndef TQ_EXL3_WIDE_LDG
#define TQ_EXL3_WIDE_LDG 0
#endif
#ifndef TQ_EXL3_L2PF
#define TQ_EXL3_L2PF 6
#endif
#ifndef TQ_EXL3_L2PF_KB7
#define TQ_EXL3_L2PF_KB7 0
#endif
// 128-token units (NG = 16; session 10, prefill-GEMM thread): ncu of the 64-token units has the tensor pipe 36-39 %
// active -- one m16n8k16 f16 HMMA takes 16 pipe cycles per sub-partition, and every warp spends ~100 issue slots of
// codeword extraction + loads per 8 HMMAs, with only 4 warps per scheduler (128 registers, 2 blocks/SM) to interleave
// them (8.5 cycles per issued instruction: long-scoreboard 2.1, math-pipe throttle 2.1, fixed-latency wait 1.8). A
// 128-token unit feeds 16 HMMAs (256 pipe cycles) per dequantized tile; it runs at ONE block per SM (255 registers:
// 96 accumulators, a double-buffered B-fragment set, a 4-tile weight ring, no spills), so the two warps of a scheduler
// alternate HMMA bursts and dequant phases. Same k-walk, fold schedule and -- for even feature-block counts -- the same
// k-split count as the 64-token units at 128 columns: bitwise identical outputs.
#ifndef TQ_EXL3_NG16_BLOCKS
#define TQ_EXL3_NG16_BLOCKS 1
#endif
#ifndef TQ_EXL3_NG16_PF
#define TQ_EXL3_NG16_PF 4
#endif
#ifndef TQ_EXL3_PAIR
#define TQ_EXL3_PAIR 1
#endif
// Dropped (session 10, prefill thread, measured 10-16): a cp.async shared-memory stage of 8 tiles of B fragments for the
// 64-token units (one copy per block, double-buffered, one barrier per stage, LDS.128 reads) -- bitwise, but +9 % per
// prefill wave (76.1 vs 69.7 ms at 32K): the two 38 KB blocks per SM shrink L1 under the weight ring and the LDS
// wavefronts equal the L1-hit LDG.128s they replace. The register double buffer of the 128-token units is the keeper.
// 64-token units with fp32 accumulators (NG = 8, F32; session 17): the verify wave's units for 33..64 columns. The verify
// used to cap its units at 32 tokens (fp32 accumulators = the decode step's numerics), so a wave of 33..64 columns ran
// as a 32-column pass plus a tail pass over the SAME weights with 8..32-token units -- a second full read + decode of
// every Linear per wave (N = 8 slots at K = 4: 40 columns, the tail pass cost ~22 ms of kernel time per round; N = 12 at
// K = 2: 36 columns, the served rule's regime at full occupancy). The F32 instantiation keeps the decode-size path per
// column (fp32 MMA, the unpaired codeword decode, the same k-walk and k-splits), so every column is bit-identical to the
// 32-token units' and to the decode step's; token groups past nvar_u skip their MMAs (the B fragments there are the
// pad) -- and skip their B-fragment loads, which is what makes the unit fast. Measured 2026-10-22 on the 32K verify
// wave, 12 slots prefilled (spec_paged_check.py --bench), T = 40 / 36 / 60 / 64 columns: the served two-pass wave
// 41.0 / 42.6 / 48.8 / 47.6 ms; this unit loading every group's B fragments 41.1 / 42.4 / 54.3 / 58.4 (no gain: the
// dead groups' loads sit on each tile's critical path); loading the live groups only 32.75 / 34.4 / 43.5 / 47.4 --
// T = 40 costs 4.9 ms over T = 32 (27.85) instead of 13.2. At 8 live groups (T = 64) the fp32 MMAs bound it like the
// two passes did. The 128-token unit's B double buffer (BPF) measured the same (32.3 at PF 2, 33.0 at PF 4) but its
// PF = 2 build computed wrong columns (PF 4 + BPF was exact) -- not root-caused, so the F32 unit keeps the NG <= 4
// schedule: a 4-tile weight ring, B fragments loaded right before their MMAs.
template <int KB, int NG, bool F32 = false>
__global__ void __launch_bounds__(256, (NG <= 2) ? ((KB == 7) ? 2 : 4) : (NG >= 16 ? TQ_EXL3_NG16_BLOCKS : 2))
k_tq_exl3_wide_gemm(float *out, float *partials, size_t split_stride,
                    const uint16_t *trellis, const uint32_t *bfrag,
                    const uint16_t *svh, int M, int Kt16, int nvar,
                    int GP, int g_base, int ntg, int k_splits, int per, int *ctr) {
    static_assert(!F32 || NG == 8, "the fp32-accumulate wide unit is the 64-token one");
    constexpr int NW = 8 * KB;
    constexpr int PF = (NG <= 4 || F32) ? 4 : (NG >= 16 ? TQ_EXL3_NG16_PF : 2);   // tiles in flight per warp (register ring)
    constexpr bool BPF = (NG >= 16);                // B fragments double-buffered one tile ahead
    constexpr bool PAIR = (TQ_EXL3_PAIR != 0) && (NG >= 8) && !F32;   // paired codeword decode (tq_exl3_mul1x2); decode N unchanged
    constexpr int RW = 2 * tq_exl3_win<KB>::NGRP;   // words per lane per tile
    __shared__ int s_unit;
    __shared__ __align__(16) float s_red[8][128];
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int tig = lane & 3, gid = lane >> 2;
    int MB = M >> 7;
    int total = MB * ntg * k_splits;
    const size_t kstride = NW;                    // tiles stored [nt][kt]: k-walk is contiguous
    tq_exl3_win<KB> win;
    win.init(lane);
    // PDL: before the first wait this block only takes units from its launch's own counters and loads
    // weights; bfrag, the partials, the arrival counters and out belong to the stream order
    bool waited = false;
    for (;;) {
        if (threadIdx.x == 0) s_unit = atomicAdd(&ctr[0], 1);
        __syncthreads();
        int u = s_unit;
        __syncthreads();
        if (u >= total) break;
        if (u + (int)gridDim.x >= total) tq_pdl_trigger();   // final round: let the next kernel fill the tail
        int split = u % k_splits;
        int bt = u / k_splits;                    // (feature block, token group), token group fastest
        int tg = bt % ntg, bm = bt / ntg;
        int nt = bm * 8 + warp;
        int t0 = split * per, t1 = t0 + per;
        if (t1 > Kt16) t1 = Kt16;                 // ragged last split
        const uint32_t *tr = (const uint32_t *)trellis + (size_t)nt * Kt16 * NW;
        // this unit's tokens: tg*8*NG .. ; bfrag groups and out rows offset accordingly
        const int tok_base = tg * 8 * NG;
        const int nvar_u = (nvar - tok_base < 8 * NG) ? (nvar - tok_base) : 8 * NG;
        const int gp0 = (g_base + tg * NG) >> 1;  // first group pair of this unit (g_base + tg*NG even when NG >= 2)
        const int half0 = (g_base + tg * NG) & 1; // NG == 1: which half of the pair
        float *out_u = out + (size_t)tok_base * M;
        float *partials_u = partials + (size_t)tok_base * M;
        float c[NG][4];
        #pragma unroll
        for (int g = 0; g < NG; g++) { c[g][0] = c[g][1] = c[g][2] = c[g][3] = 0.0f; }
        // Wide N (prefill) is tensor-pipe bound (ncu: 79 % tensor active, math-pipe throttle),
        // and GeForce runs f16-accumulate MMAs at twice the f32-accumulate rate: accumulate
        // TQ_EXL3_FOLD tiles (16*FOLD products) in fp16 and fold into the fp32 accumulators.
        // Decode-size N keeps the f32 MMA (tensor work is negligible there); so does the verify wave's F32 unit.
        constexpr bool ACC16 = (NG >= 8) && !F32;
        constexpr int FOLD = 8;
        uint32_t ch[ACC16 ? NG : 1][2];
        #pragma unroll
        for (int g = 0; g < (ACC16 ? NG : 1); g++) { ch[g][0] = 0u; ch[g][1] = 0u; }
        auto fold16 = [&]() {
            if (ACC16) {
                #pragma unroll
                for (int g = 0; g < NG; g++) {
                    __half2 h0 = *reinterpret_cast<__half2 *>(&ch[g][0]);
                    __half2 h1 = *reinterpret_cast<__half2 *>(&ch[g][1]);
                    c[g][0] += __low2float(h0); c[g][1] += __high2float(h0);
                    c[g][2] += __low2float(h1); c[g][3] += __high2float(h1);
                    ch[g][0] = 0u; ch[g][1] = 0u;
                }
            }
        };
        // One tile's NG MMAs against the tokens (A fragment already dequantized). Group pairs
        // are contiguous per lane: two groups per 16-byte load, coalesced across the warp.
        auto load_b = [&](int kt, uint32_t (&bw)[NG][2]) {
            const uint32_t *bp = bfrag + (((size_t)kt * GP + gp0) * 32 + lane) * 4;
            if (NG >= 2) {
                #pragma unroll
                for (int g = 0; g < NG; g += 2) {
                    if (F32 && g * 8 >= nvar_u) {   // dead token groups of a ragged verify wave: no B load (their MMAs are skipped)
                        bw[g][0] = bw[g][1] = bw[g + 1][0] = bw[g + 1][1] = 0u;
                        continue;
                    }
                    uint4 q = __ldg((const uint4 *)(bp + (size_t)(g >> 1) * 128));
                    bw[g][0] = q.x; bw[g][1] = q.y; bw[g + 1][0] = q.z; bw[g + 1][1] = q.w;
                }
            } else {
                uint2 q = __ldg((const uint2 *)(bp + half0 * 2));
                bw[0][0] = q.x; bw[0][1] = q.y;
            }
        };
        auto mma_b = [&](uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, const uint32_t (&bw)[NG][2]) {
            #pragma unroll
            for (int g = 0; g < NG; g++) {
                if (F32 && g * 8 >= nvar_u) break;   // ragged verify wave: token groups past the live columns (uniform per unit)
                if (ACC16) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                        : "+r"(ch[g][0]), "+r"(ch[g][1])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(bw[g][0]), "r"(bw[g][1]));
                } else {
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+f"(c[g][0]), "+f"(c[g][1]), "+f"(c[g][2]), "+f"(c[g][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(bw[g][0]), "r"(bw[g][1]));
                }
            }
        };
        auto tile_mma = [&](uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, int kt) {
            uint32_t bw[NG][2];
            load_b(kt, bw);
            mma_b(a0, a1, a2, a3, bw);
        };
        // Register ring: pf[d] holds the lane's words of tile kt+d (direct per-lane loads of
        // the two words of each codeword group; L1 absorbs the overlap between lanes).
        constexpr bool STREAM = !(TQ_EXL3_WIDE_LDG && NG >= 8);
        constexpr int L2PF = (NG >= 8 && (KB != 7 || TQ_EXL3_L2PF_KB7)) ? TQ_EXL3_L2PF : 0;
        uint32_t pf[PF][RW];
        #pragma unroll
        for (int d = 0; d < PF; d++)
            if (t0 + d < t1) win.template load<STREAM>(tr + (size_t)(t0 + d) * kstride, pf[d]);
        if (L2PF > 0) {
            #pragma unroll
            for (int d = PF; d < PF + L2PF; d++)
                if (t0 + d < t1) win.prefetch(tr + (size_t)(t0 + d) * kstride);
        }
        if (!waited) { tq_pdl_wait(); waited = true; }
        int kt = t0;
        if (BPF) {
            // 128-token units: B fragments double-buffered one tile ahead (bwb[(kt - t0 + d) & 1]; PF is even, so the
            // parity is d & 1 in every iteration), the ring and the L2 prefetch as below. Same MMA order per tile.
            static_assert(!BPF || (PF % 2) == 0, "B double buffer needs an even ring");
            static_assert(!BPF || FOLD == 8, "the fold rule below encodes FOLD = 8 with the 2-tile ring's tail");
            // fp16 fold points of the 2-tile-ring schedule below (bitwise identity with the 64-token units): after
            // every 8th tile while at least 4 tiles remain, then after the tail's 2-tile chunks -- i.e. after tile
            // c (1-based) when c % 8 == 0 && c <= nE - 2, or c == nE, or c == n, with nE = n rounded down to even.
            const int nE = (t1 - t0) & ~1;
            uint32_t bwb[2][NG][2];
            if (t0 < t1) load_b(t0, bwb[0]);
            for (; kt + 2 * PF <= t1; kt += PF) {     // c <= n - PF here: only the 8-tile rule applies
                #pragma unroll
                for (int d = 0; d < PF; d++) {
                    uint32_t r[RW];
                    #pragma unroll
                    for (int q = 0; q < RW; q++) r[q] = pf[d][q];
                    win.template load<STREAM>(tr + (size_t)(kt + d + PF) * kstride, pf[d]);
                    if (L2PF > 0 && kt + d + PF + L2PF < t1) win.prefetch(tr + (size_t)(kt + d + PF + L2PF) * kstride);
                    load_b(kt + d + 1, bwb[(d + 1) & 1]);          // kt + d + 1 < t1: 2*PF tiles are left
                    uint32_t a0, a1, a2, a3;
                    win.template frag<PAIR>(r, a0, a1, a2, a3);
                    mma_b(a0, a1, a2, a3, bwb[d & 1]);
                    if (ACC16 && (((kt - t0 + d + 1) & 7) == 0)) fold16();
                }
            }
            for (; kt < t1; kt += PF) {
                #pragma unroll
                for (int d = 0; d < PF; d++) {
                    if (kt + d >= t1) break;
                    uint32_t r[RW];
                    #pragma unroll
                    for (int q = 0; q < RW; q++) r[q] = pf[d][q];
                    if (kt + d + PF < t1) win.template load<STREAM>(tr + (size_t)(kt + d + PF) * kstride, pf[d]);
                    if (kt + d + 1 < t1) load_b(kt + d + 1, bwb[(d + 1) & 1]);
                    uint32_t a0, a1, a2, a3;
                    win.template frag<PAIR>(r, a0, a1, a2, a3);
                    mma_b(a0, a1, a2, a3, bwb[d & 1]);
                    const int c = kt - t0 + d + 1;
                    if (ACC16 && ((((c & 7) == 0) && c <= nE - 2) || c == nE)) fold16();
                }
            }
        } else {
        // main loop: PF tiles in hand and PF more guaranteed ahead -> no per-tile branches
        for (; kt + 2 * PF <= t1; kt += PF) {
            #pragma unroll
            for (int d = 0; d < PF; d++) {
                uint32_t r[RW];
                #pragma unroll
                for (int q = 0; q < RW; q++) r[q] = pf[d][q];
                win.template load<STREAM>(tr + (size_t)(kt + d + PF) * kstride, pf[d]);
                if (L2PF > 0 && kt + d + PF + L2PF < t1) win.prefetch(tr + (size_t)(kt + d + PF + L2PF) * kstride);
                uint32_t a0, a1, a2, a3;
                win.template frag<PAIR>(r, a0, a1, a2, a3);
                tile_mma(a0, a1, a2, a3, kt + d);
            }
            if (ACC16 && (((kt - t0) / PF + 1) % (FOLD / PF)) == 0) fold16();
        }
        // tail: fewer than 2*PF tiles left
        for (; kt < t1; kt += PF) {
            #pragma unroll
            for (int d = 0; d < PF; d++) {
                if (kt + d >= t1) break;
                uint32_t r[RW];
                #pragma unroll
                for (int q = 0; q < RW; q++) r[q] = pf[d][q];
                if (kt + d + PF < t1) win.template load<STREAM>(tr + (size_t)(kt + d + PF) * kstride, pf[d]);
                uint32_t a0, a1, a2, a3;
                win.template frag<PAIR>(r, a0, a1, a2, a3);
                tile_mma(a0, a1, a2, a3, kt + d);
            }
            if (ACC16) fold16();
        }
        }
        fold16();
        int row0 = nt * 16 + gid, row1 = row0 + 8;
        bool finish = true;
        if (k_splits > 1) {
            // publish this split's y_hat partial, then see whether we are the last split of (bm, tg)
            float *pout = partials_u + (size_t)split * split_stride;
            #pragma unroll
            for (int g = 0; g < NG; g++) {
                int tok0 = g * 8 + 2 * tig, tok1 = tok0 + 1;
                if (tok0 < nvar_u) { pout[(size_t)tok0 * M + row0] = c[g][0]; pout[(size_t)tok0 * M + row1] = c[g][2]; }
                if (tok1 < nvar_u) { pout[(size_t)tok1 * M + row0] = c[g][1]; pout[(size_t)tok1 * M + row1] = c[g][3]; }
            }
            __threadfence();
            __syncthreads();
            if (threadIdx.x == 0) {
                int arrived = atomicAdd(&ctr[2 + bt], 1);
                s_unit = (arrived == k_splits - 1) ? 1 : 0;
                if (arrived == k_splits - 1) ctr[2 + bt] = 0;
            }
            __syncthreads();
            finish = s_unit != 0;
            __syncthreads();
            if (finish) __threadfence();
        }
        if (finish) {
            #pragma unroll
            for (int g = 0; g < NG; g++) {
                if (g * 8 >= nvar_u) break;
                // this warp's 16 features x 8 tokens of group g -> s_red[token][feature]
                int f0 = warp * 16 + gid, f1 = f0 + 8;
                if (k_splits > 1) {
                    // deterministic: partials summed in split order 0..k_splits-1
                    int tok0 = g * 8 + 2 * tig, tok1 = tok0 + 1;
                    float v00 = 0.f, v01 = 0.f, v10 = 0.f, v11 = 0.f;
                    for (int s = 0; s < k_splits; s++) {
                        const float *ps = partials_u + (size_t)s * split_stride;   // other blocks' data: skip L1
                        if (tok0 < nvar_u) { v00 += __ldcg(ps + (size_t)tok0 * M + row0); v10 += __ldcg(ps + (size_t)tok0 * M + row1); }
                        if (tok1 < nvar_u) { v01 += __ldcg(ps + (size_t)tok1 * M + row0); v11 += __ldcg(ps + (size_t)tok1 * M + row1); }
                    }
                    s_red[2 * tig][f0] = v00; s_red[2 * tig + 1][f0] = v01;
                    s_red[2 * tig][f1] = v10; s_red[2 * tig + 1][f1] = v11;
                } else {
                    s_red[2 * tig][f0] = c[g][0]; s_red[2 * tig + 1][f0] = c[g][1];
                    s_red[2 * tig][f1] = c[g][2]; s_red[2 * tig + 1][f1] = c[g][3];
                }
                __syncthreads();
                // warp w: token g*8+w, 128 features -> H128 * svh -> out
                int tok = g * 8 + warp;
                if (tok < nvar_u) {
                    float4 v = *(const float4 *)&s_red[warp][lane * 4];
                    tq_had128_warp(v.x, v.y, v.z, v.w, lane);
                    const uint16_t *sp = svh + bm * 128 + lane * 4;
                    float4 o;
                    o.x = v.x * TQ_EXL3_RSCALE * tq_fp16_to_float(sp[0]);
                    o.y = v.y * TQ_EXL3_RSCALE * tq_fp16_to_float(sp[1]);
                    o.z = v.z * TQ_EXL3_RSCALE * tq_fp16_to_float(sp[2]);
                    o.w = v.w * TQ_EXL3_RSCALE * tq_fp16_to_float(sp[3]);
                    *(float4 *)(out_u + (size_t)tok * M + bm * 128 + lane * 4) = o;
                }
                __syncthreads();
            }
        }
    }
    // the last block to finish resets the unit counter for the next launch
    if (threadIdx.x == 0) {
        __threadfence();
        int done = atomicAdd(&ctr[1], 1);
        if (done == (int)gridDim.x - 1) { ctr[0] = 0; ctr[1] = 0; __threadfence(); }
    }
}

// Linears the checkpoint left in bf16 (tiny: in_proj_a/b): out[n][m] = sum_k W[m][k] x[n][k].
// grid (M, ceil(N/8)), 1024 threads (few rows -> the K loop must be short); one weight row
// read per 8 tokens.
__global__ void __launch_bounds__(1024) k_tq_exl3_bf16_wide(float *out, const uint16_t *w, const float *x,
                                                            int M, int K, int N) {
    __shared__ float red[8][1024];
    int m = blockIdx.x, n0 = blockIdx.y * 8, tid = threadIdx.x;
    const uint16_t *wr = w + (size_t)m * K;
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int k = tid; k < K; k += 1024) {
        float wv = tq_bf16_to_float(__ldg(wr + k));
        #pragma unroll
        for (int c = 0; c < 8; c++)
            if (n0 + c < N) acc[c] += wv * __ldg(x + (size_t)(n0 + c) * K + k);
    }
    #pragma unroll
    for (int c = 0; c < 8; c++) red[c][tid] = acc[c];
    __syncthreads();
    for (int stride = 512; stride > 0; stride >>= 1) {
        if (tid < stride) {
            #pragma unroll
            for (int c = 0; c < 8; c++) red[c][tid] += red[c][tid + stride];
        }
        __syncthreads();
    }
    if (tid < 8 && n0 + tid < N) out[(size_t)(n0 + tid) * M + m] = red[tid][0];
}

// The same Linears at prefill N, 4 rows per block (session 10, prefill thread): the block-per-row kernel above cost
// 30 µs per call at N = 128 (512 blocks each re-reading 8 token rows = ~80 MB of L2 traffic per 320 KB weight, and a
// 10-step 1024-thread tree per block: 2.9 ms of a 69 ms prefill wave for in_proj_b + in_proj_a). Same per-thread partial
// sums (k = tid + 1024 i, i ascending) and the same tree (stride 512 .. 1, red[tid] += red[tid + stride]) -> bitwise;
// the five in-warp levels (stride 16 .. 1) run as shfl_down on warp 0 (the same operand pairs), the rows one after the
// other through one 32 KB smem buffer. grid (M / 4, ceil(N / 8)). K % 1024 == 0, M % 4 == 0.
__global__ void __launch_bounds__(1024) k_tq_exl3_bf16_wide_r4(float *out, const uint16_t *w, const float *x,
                                                               int M, int K, int N) {
    __shared__ float red[8][1024];
    int m0 = blockIdx.x * 4, n0 = blockIdx.y * 8, tid = threadIdx.x;
    int nn = N - n0 < 8 ? N - n0 : 8;
    float acc[4][8];
    #pragma unroll
    for (int r = 0; r < 4; r++)
        #pragma unroll
        for (int c = 0; c < 8; c++) acc[r][c] = 0.0f;
    for (int k = tid; k < K; k += 1024) {
        float xv[8];
        #pragma unroll
        for (int c = 0; c < 8; c++) xv[c] = (c < nn) ? __ldg(x + (size_t)(n0 + c) * K + k) : 0.0f;
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            float wv = tq_bf16_to_float(__ldg(w + (size_t)(m0 + r) * K + k));
            #pragma unroll
            for (int c = 0; c < 8; c++) acc[r][c] += wv * xv[c];
        }
    }
    int lane = tid & 31;
    for (int r = 0; r < 4; r++) {
        if (r > 0) __syncthreads();                   // red is free again (warp 0 finished the previous row)
        #pragma unroll
        for (int c = 0; c < 8; c++) red[c][tid] = acc[r][c];
        __syncthreads();
        for (int stride = 512; stride >= 32; stride >>= 1) {
            if (tid < stride) {
                #pragma unroll
                for (int c = 0; c < 8; c++) red[c][tid] += red[c][tid + stride];
            }
            __syncthreads();
        }
        if (tid < 32) {
            float v[8];
            #pragma unroll
            for (int c = 0; c < 8; c++) v[c] = red[c][lane];
            #pragma unroll
            for (int stride = 16; stride > 0; stride >>= 1) {
                #pragma unroll
                for (int c = 0; c < 8; c++) {
                    float o = __shfl_down_sync(0xffffffffu, v[c], stride);
                    if (lane < stride) v[c] += o;
                }
            }
            if (lane == 0) {
                for (int c = 0; c < nn; c++) out[(size_t)(n0 + c) * M + m0 + r] = v[c];
            }
        }
    }
}

// Decode-size N: the two bf16 Linears that share an input (GDN in_proj_b / in_proj_a, M rows each)
// in one launch. Block = (row of either weight, group of 8 tokens), 256 threads x 4-wide K chunks,
// warp shuffles + one smem pass to reduce (the block-per-row kernel above spends ~4 us per call in
// its 1024-thread tree reduction). K % 1024 == 0.
__global__ void __launch_bounds__(256) k_tq_exl3_bf16_pair(float *out0, const uint16_t *w0, float *out1,
                                                           const uint16_t *w1, const float *x, int M, int K, int N) {
    tq_pdl_enter();
    __shared__ float red[8][8];
    int r = blockIdx.x, n0 = blockIdx.y * 8, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int m = r < M ? r : r - M;
    const uint16_t *wr = (r < M ? w0 : w1) + (size_t)m * K;
    float *o = r < M ? out0 : out1;
    int nn = N - n0 < 8 ? N - n0 : 8;
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int k = tid * 4; k < K; k += 1024) {
        uint2 wv = __ldg((const uint2 *)(wr + k));
        float w4[4] = {tq_bf16_to_float((uint16_t)(wv.x & 0xffffu)), tq_bf16_to_float((uint16_t)(wv.x >> 16)),
                       tq_bf16_to_float((uint16_t)(wv.y & 0xffffu)), tq_bf16_to_float((uint16_t)(wv.y >> 16))};
        #pragma unroll
        for (int c = 0; c < 8; c++) {
            if (c < nn) {
                float4 xv = __ldg((const float4 *)(x + (size_t)(n0 + c) * K + k));
                acc[c] += w4[0] * xv.x + w4[1] * xv.y + w4[2] * xv.z + w4[3] * xv.w;
            }
        }
    }
    #pragma unroll
    for (int c = 0; c < 8; c++) {
        float v = acc[c];
        #pragma unroll
        for (int s = 16; s > 0; s >>= 1) v += __shfl_xor_sync(0xffffffffu, v, s);
        if (lane == 0) red[warp][c] = v;
    }
    __syncthreads();
    if (tid < nn) {
        float v = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; w++) v += red[w][tid];
        o[(size_t)(n0 + tid) * M + m] = v;
    }
}

// ---- D.1 integration: wide projection via k_tq_fp6_wide_gemm (shared per-128K-block
// activation scale: NOT bit-exact vs per-token for differing tokens -- validated by
// needle, not bit-exact). Scratch grows lazily. ---------------------------------------
static uint8_t *g_wproj_b = NULL; static float *g_wproj_bscale = NULL, *g_wproj_part = NULL;
static size_t g_wproj_b_bytes = 0, g_wproj_bscale_n = 0, g_wproj_part_bytes = 0;
// TQ_GEMM_FORK=1 (session 10, decode-graph; DEFAULT ON since 2026-10-17, owner's decision (g)): the 2nd / 3rd Linear of a group that shares one input
// (k/v after q, z after qkv, up after gate) run on side streams concurrently with the first -- they are independent
// (the fragment buffer is read-only, the outputs distinct), so the small k/v_proj GEMMs (80 of 680 blocks) and every
// member's tail fill the GPU instead of serializing. Same kernels, same grids, same k-splits -> bitwise; each side
// member gets its own k-split partial buffer (the shared one would race) and, as every launch, its own counter slot.
// TQ_GEMM_FORK_MAXN (session 11, wave-sched thread; default 64): the fork only up to this N. At decode / verify
// sizes the members underfill the GPU and the fork pays (-0.35 ms per step with the graph); at the 128-column
// prefill wave every member is a full one-block-per-SM pass (gate + up: 2 x 136 blocks on 170 SMs), the side
// member only runs when the main one retires (175 us concurrent vs 2 x 90 serial = no overlap), and the join
// breaks the PDL chain -- measured 0.0 ms either way (notes/session11-sched.md), so the wave runs its groups
// serially: one stream, no join, the idle SMs of each pass free for the trellis encoder's side stream.
static float *g_wproj_part_side[2] = {NULL, NULL}; static size_t g_wproj_part_side_bytes[2] = {0, 0};
static int g_exl3_part_sel = 0;                     // 0 = the main partial buffer, 1..2 = side member i's
static cudaStream_t g_fork_st[2] = {NULL, NULL};
static cudaEvent_t g_fork_ev0 = NULL, g_fork_ev[2] = {NULL, NULL};
static int fork_streams_ok(void) {
    static int state = 0;                           // 0 = not tried, 1 = ok, -1 = failed
    if (state == 0) {
        state = 1;
        int prio = 0;                               // the members are the main stream's work: its priority
        if (!g_qwen.stream || cudaStreamGetPriority(g_qwen.stream, &prio) != cudaSuccess) prio = 0;
        for (int i = 0; i < 2 && state == 1; i++)
            if (cudaStreamCreateWithPriority(&g_fork_st[i], cudaStreamNonBlocking, prio) != cudaSuccess ||
                cudaEventCreateWithFlags(&g_fork_ev[i], cudaEventDisableTiming) != cudaSuccess) state = -1;
        if (state == 1 && cudaEventCreateWithFlags(&g_fork_ev0, cudaEventDisableTiming) != cudaSuccess) state = -1;
        if (state < 0) fprintf(stderr, "[gemm-fork] side streams unavailable -- serial GEMM groups\n");
    }
    return state == 1;
}
// EXL3: wide_quant_input only records the fp32 input (the transform depends on each Linear's suh);
// exl3_wide_proj builds the fp16 fragments per projection.
static const float *g_exl3_x = NULL; static int g_exl3_K = 0, g_exl3_N = 0;   // (g_exl3_x2 declared above)
static uint32_t *g_exl3_b = NULL; static size_t g_exl3_b_words = 0;
#define TQ_EXL3_CTR_STRIDE (2 + 8192)          // per slot: work/done/arrival counters, self-resetting
#define TQ_EXL3_CTR_SLOTS 16
static int *g_exl3_ctr = NULL;                 // [TQ_EXL3_CTR_SLOTS][TQ_EXL3_CTR_STRIDE]
static unsigned g_exl3_ctr_next = 0;
static int g_exl3_grid[9][6];                  // resident blocks per (KB, log2 NG; 5 = the fp32 64-token unit), 0 = not queried
static int g_exl3_nsm = 0;

// Resident block capacity of one instantiation (occupancy x SMs), cached.
template <int KB, int NG, bool F32 = false>
static int exl3_gemm_cap(void) {
    int lg = F32 ? 5 : (NG == 1) ? 0 : (NG == 2) ? 1 : (NG == 4) ? 2 : (NG == 8) ? 3 : 4;
    int &cap = g_exl3_grid[KB][lg];
    if (cap == 0) {
        int nb = 0;
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k_tq_exl3_wide_gemm<KB, NG, F32>, 256, 0) != cudaSuccess || nb < 1) nb = 1;
        if (g_exl3_nsm == 0) {
            int dev = 0; cudaGetDevice(&dev);
            cudaDeviceGetAttribute(&g_exl3_nsm, cudaDevAttrMultiProcessorCount, dev);
            if (g_exl3_nsm <= 0) g_exl3_nsm = 1;
        }
        cap = nb * g_exl3_nsm;
    }
    return cap;
}

// Tokens per unit (8*NG). Decode-size N keeps small accumulator sets (occupancy). For wide N
// one dequantized tile should feed as many MMAs as possible (the dequant is the cost there),
// but a unit set must still fill the resident blocks: take the largest NG whose
// (feature blocks x token groups) covers >= 3/4 of the capacity, else the smallest (NG=4).
// g_exl3_ng_max caps the unit width; g_exl3_vf = the verify wave's regime (fp32 accumulators at every unit width: it
// sets ng_max 8, so 33..64 columns run as ONE pass of 64-token F32 units -- until session 17 it set 4 and such a wave
// paid a second pass over the weights for its tail).
static int g_exl3_ng_max = 16;
static int g_exl3_vf = 0;
static int exl3_unit_ng(int N, int MB) {
    int g = wide_ng(N);
    if (g <= 4) return g;
    if (g_exl3_ng_max <= 8) return g_exl3_ng_max;
    // measured (5090, 128 tokens, fp16-accumulate MMAs): 64-token units are the best trade
    // between dequant amortization and k-split partial traffic for every projection except the
    // lm_head, whose 1940 feature blocks fill the GPU with 128-token units and no split
    // (1.47 vs 1.63 ms); the k/v projections (8 feature blocks) need 32-token units.
    int units16 = MB * ((N + 127) / 128);
    // TQ_EXL3_WIDE_NG16 (default 1; session 10, prefill thread): 128-token units at one block per SM for every Linear
    // with >= 32 of them (the k/v projections keep their 32-token units) -- see k_tq_exl3_wide_gemm's NG = 16 notes.
    // Measured 10-16 (GPU 2, 550 W, trellis stack): KB=5 GEMMs 99 -> 87 us, KB=6 74.5 -> 67.6, KB=7 unchanged;
    // 128-column wave at 32K 69.7 -> 66.1 ms (-5.1 %), at 128K 120.0 -> 116.9 (-2.6 %); bitwise (per-Linear vs the
    // 64-token units at N = 128 / 130 / 256, prefill-mode e2e). =0 restores the 64-token units.
    if (g_cfg.exl3_wide_ng16 && N >= 128 && units16 >= 32) return 16;
    if (units16 >= 1024) return 16;
    int units8 = MB * ((N + 63) / 64);
    if (units8 >= 64) return 8;
    return 4;
}

// k-splits so that (feature blocks x token groups x splits) fills the resident blocks in ONE
// round (a GEMM is only 1-2 rounds of the whole GPU, so any unit tail costs a large fraction),
// with >= 32 tiles per split; ragged last split allowed. The y_hat partial traffic is capped
// at half the weight bytes. Decode-size N rounds down (no second round for a few blocks:
// in_proj_qkv had 720 units on 680 blocks, KB=7 MLPs 408 on 340) and allows 16 tiles per split
// where 32 would leave most of the GPU idle (k/v_proj: 80 of 680 blocks); at 40-48 feature
// blocks (in_proj_z, o_proj) shorter splits cost more in partial reduction than the fill gains.
// g_split_n_override > 0: the k-split / key-split pickers see this client count instead of the column count
// (the verify wave sets its slot count, so a column's splits -- and bits -- equal the decode step's at that
// many clients; the attention launches T x S blocks instead of the fewer a T-column step would pick).
static int g_split_n_override = 0;
static int exl3_pick_splits(int cap, int units, int Kt16, int N, int M, int KB, int *per_out) {
    int fixed = g_cfg.qmma_sf_k_split;
    int ks;
    if (g_split_n_override > 0) N = g_split_n_override;
    if (fixed >= 1) ks = fixed;
    else {
        bool decode = N <= 16;
        ks = decode ? cap / units : (cap + units / 2) / units;
        if (ks < 1) ks = 1;
        int min_per = (decode && 2 * units * (Kt16 / 32) < cap) ? 16 : 32;
        int ks_max = Kt16 / min_per; if (ks_max < 1) ks_max = 1;
        if (ks > ks_max) ks = ks_max;
        double wbytes = (double)M * (double)Kt16 * 16.0 * KB / 8.0;
        int ks_bw = (int)(wbytes / (2.0 * N * (double)M * 4.0));
        if (ks_bw < 1) ks_bw = 1;
        if (ks > ks_bw) ks = ks_bw;
    }
    int per = (Kt16 + ks - 1) / ks;
    ks = (Kt16 + per - 1) / per;
    *per_out = per;
    return ks;
}

// Resident-block capacity of the instantiation a decode step at N clients would run (NG = wide_ng(N)).
template <int KB>
static int exl3_gemm_cap_decode(int N) {
    switch (wide_ng(N)) {
        case 1: return exl3_gemm_cap<KB, 1>();
        case 2: return exl3_gemm_cap<KB, 2>();
        case 4: return exl3_gemm_cap<KB, 4>();
        case 8: return exl3_gemm_cap<KB, 8>();
        default: return exl3_gemm_cap<KB, 16>();
    }
}

template <int KB, int NG, bool F32 = false>
static int exl3_launch_gemm(float *out, float *partials, size_t split_stride, const tq_qmma_weight_t *w,
                            const uint32_t *bp, int M, int Kt16, int N, int Gt, int g_base, cudaStream_t st) {
    int cap = exl3_gemm_cap<KB, NG, F32>();
    int ntg = (N + 8 * NG - 1) / (8 * NG);
    int units = (M / 128) * ntg;
    if (units > 8192) return -117;                // arrival counters
    int per = 1;
    // g_split_n_override: pick the k-splits as the decode step at that many clients does (its instantiation's
    // capacity, one token group), so the per-column numerics equal that step's; the launch keeps this grid.
    int ks = g_split_n_override > 0
        ? exl3_pick_splits(exl3_gemm_cap_decode<KB>(g_split_n_override), M / 128, Kt16, N, M, KB, &per)
        : exl3_pick_splits(cap, units, Kt16, N, M, KB, &per);
    float *pout = NULL;
    if (ks > 1) {
        size_t need = (size_t)ks * split_stride * sizeof(float);
        float **pp = g_exl3_part_sel ? &g_wproj_part_side[g_exl3_part_sel - 1] : &g_wproj_part;
        size_t *pb = g_exl3_part_sel ? &g_wproj_part_side_bytes[g_exl3_part_sel - 1] : &g_wproj_part_bytes;
        if (need > *pb) {
            graph_invalidate();
            if (*pp) cudaFree(*pp);
            if (cudaMalloc(pp, need) != cudaSuccess) { *pp = NULL; *pb = 0; return -91; }
            *pb = need;
        }
        pout = *pp;
    }
    int total = units * ks;
    int grid = total < cap ? total : cap;
    int *ctr = g_exl3_ctr + (size_t)(g_exl3_ctr_next++ % TQ_EXL3_CTR_SLOTS) * TQ_EXL3_CTR_STRIDE;
    tq_launch(g_pdl_on, k_tq_exl3_wide_gemm<KB, NG, F32>, dim3(grid), dim3(256), 0, st, out, pout, split_stride,
              w->d_exl3_trellis, bp, w->d_exl3_svh, M, Kt16, N, (Gt + 1) / 2, g_base, ntg, ks, per, ctr);
    if (g_cfg.exl3_debug) {   // counters must be back at zero after every launch
        int h[2 + 64];
        cudaStreamSynchronize(st);
        cudaMemcpy(h, ctr, sizeof(h), cudaMemcpyDeviceToHost);
        int bad = 0;
        for (int i = 0; i < 2 + 64; i++) if (h[i]) bad++;
        fprintf(stderr, "[exl3-gemm] KB=%d NG=%d%s M=%d K=%d N=%d ntg=%d ks=%d per=%d grid=%d units=%d ctr0=%d ctr1=%d nonzero=%d err=%s\n",
                KB, NG, F32 ? " f32" : "", M, Kt16 * 16, N, ntg, ks, per, grid, units, h[0], h[1], bad, cudaGetErrorString(cudaGetLastError()));
    }
    return 0;
}

#define TQ_EXL3_LAUNCH(KB_, NG_, F32_) \
    return exl3_launch_gemm<KB_, NG_, F32_>(out + (size_t)n0 * M, NULL, split_stride, w, bp, M, Kt16, nn, Gt, n0 / 8, st)
// G = 8 is the fp16-accumulate 64-token unit of the prefill waves, or -- in the verify wave's regime -- its fp32 twin.
#define TQ_EXL3_SWITCH_NG(KB_) \
    switch (G) { \
        case 1: TQ_EXL3_LAUNCH(KB_, 1, false); case 2: TQ_EXL3_LAUNCH(KB_, 2, false); \
        case 4: TQ_EXL3_LAUNCH(KB_, 4, false); \
        case 8: if (g_exl3_vf) TQ_EXL3_LAUNCH(KB_, 8, true); else TQ_EXL3_LAUNCH(KB_, 8, false); \
        case 16: TQ_EXL3_LAUNCH(KB_, 16, false); default: return -113; }

// Tokens [n0, n0+nn) of the prepared fragments with NG-wide units.
static int exl3_launch_part(const tq_qmma_weight_t *w, const uint32_t *bp, float *out, int n0, int nn, int G, int Gt,
                            int M, int Kt16, size_t split_stride, cudaStream_t st) {
    switch (w->exl3_kbits) {
        case 2: TQ_EXL3_SWITCH_NG(2);
        case 3: TQ_EXL3_SWITCH_NG(3);
        case 4: TQ_EXL3_SWITCH_NG(4);
        case 5: TQ_EXL3_SWITCH_NG(5);
        case 6: TQ_EXL3_SWITCH_NG(6);
        case 7: TQ_EXL3_SWITCH_NG(7);
        case 8: TQ_EXL3_SWITCH_NG(8);
        default: return -114;
    }
}

// fp16 B fragments of g_exl3_x for n (<= 3) quantized Linears of the same K in one launch; Linear i's fragments
// start at g_exl3_b + i * *need_w.
static int exl3_build_frags(const tq_qmma_weight_t *const *ws, int n, int N, size_t *need_w_out) {
    int K = ws[0]->K;
    if (n < 1 || n > 3 || !g_exl3_x || g_exl3_K != K || g_exl3_N < N) return -110;
    if (!g_exl3_ctr) {
        size_t nb = (size_t)TQ_EXL3_CTR_SLOTS * TQ_EXL3_CTR_STRIDE * sizeof(int);
        graph_invalidate();
        if (cudaMalloc(&g_exl3_ctr, nb) != cudaSuccess) { g_exl3_ctr = NULL; return -116; }
        cudaMemset(g_exl3_ctr, 0, nb);
    }
    int Kt16 = K / 16;
    int Gt = wide_ng(N), GP = (Gt + 1) / 2;
    size_t need_w = (size_t)GP * Kt16 * 128;      // [kt][pair][32 lanes][4 words]
    if (3 * need_w > g_exl3_b_words) {
        graph_invalidate();
        if (g_exl3_b) cudaFree(g_exl3_b);
        if (cudaMalloc(&g_exl3_b, 3 * need_w * sizeof(uint32_t)) != cudaSuccess) { g_exl3_b = NULL; g_exl3_b_words = 0; return -112; }
        g_exl3_b_words = 3 * need_w;
    }
    tq_bfrag_set set;
    set.b0 = g_exl3_b; set.b1 = g_exl3_b + need_w; set.b2 = g_exl3_b + 2 * need_w;
    set.s0 = ws[0]->d_exl3_suh;
    set.s1 = n > 1 ? ws[1]->d_exl3_suh : ws[0]->d_exl3_suh;
    set.s2 = n > 2 ? ws[2]->d_exl3_suh : ws[0]->d_exl3_suh;
    int nwarps = 8 * Gt * (K / 128);
    tq_launch(g_pdl_on, k_tq_exl3_bfrag, dim3((nwarps + 3) / 4, n), dim3(128), 0, g_qwen.stream, set, g_exl3_x, g_exl3_x2,
              K, N, 8 * Gt, Kt16, GP);
    *need_w_out = need_w;
    return 0;
}

static int exl3_gemm_frags(const tq_qmma_weight_t *w, const uint32_t *bp, float *out, int N, cudaStream_t st = NULL) {
    if (!st) st = g_qwen.stream;
    int M = w->M, K = w->K;
    int Kt16 = K / 16;
    int Gt = wide_ng(N);
    size_t split_stride = (size_t)N * M;
    // Ragged N (a mixed prefill wave = 128 prompt columns + a few decoders): full 8*G-token
    // units for the head, then the remainder with the narrowest units that cover it -- one
    // more streaming pass of the weights instead of a full wide unit pass per feature block.
    int G = exl3_unit_ng(N, M / 128);
    int unit_tokens = 8 * G;
    int n_main = (N >= unit_tokens) ? (N / unit_tokens) * unit_tokens : N;
    int r = 0;
    if (n_main > 0) r = exl3_launch_part(w, bp, out, 0, n_main, G, Gt, M, Kt16, split_stride, st);
    if (r == 0 && n_main < N)
        r = exl3_launch_part(w, bp, out, n_main, N - n_main, wide_ng(N - n_main), Gt, M, Kt16, split_stride, st);
    return r;
}

static int exl3_wide_proj(const tq_qmma_weight_t *w, float *out, int N) {
    int M = w->M, K = w->K;
    if (!g_exl3_x || g_exl3_K != K || g_exl3_N < N) return -110;
    if (w->exl3_kbits == 0) {
        if (!w->d_exl3_bf16) return -111;
        if (g_cfg.exl3_bf16_r4 && N > 16 && (M & 3) == 0 && (K & 1023) == 0)
            k_tq_exl3_bf16_wide_r4<<<dim3(M / 4, (N + 7) / 8), 1024, 0, g_qwen.stream>>>(out, w->d_exl3_bf16, g_exl3_x, M, K, N);
        else
            k_tq_exl3_bf16_wide<<<dim3(M, (N + 7) / 8), 1024, 0, g_qwen.stream>>>(out, w->d_exl3_bf16, g_exl3_x, M, K, N);
        return 0;
    }
    size_t need_w = 0;
    int r = exl3_build_frags(&w, 1, N, &need_w);
    return r != 0 ? r : exl3_gemm_frags(w, g_exl3_b, out, N);
}

// Linears sharing the recorded input: one fragment launch, then the GEMMs back to back (under PDL each GEMM
// streams its first weights during its predecessor's tail). Other formats fall back to wide_proj one by one.
static int wide_proj_group(const tq_qmma_weight_t *const *ws, float *const *outs, int n, int N) {
    bool grouped = n >= 2 && n <= 3;
    for (int i = 0; i < n && grouped; i++)
        grouped = ws[i]->exl3 && ws[i]->exl3_kbits > 0 && ws[i]->K == ws[0]->K;
    if (!grouped) {
        for (int i = 0; i < n; i++) { int r = wide_proj(ws[i], outs[i], N); if (r != 0) return r; }
        return 0;
    }
    size_t need_w = 0;
    int r = exl3_build_frags(ws, n, N, &need_w);
    if (r == 0 && g_cfg.gemm_fork && N <= g_cfg.gemm_fork_maxn && fork_streams_ok()) {
        // members 1.. on the side streams, concurrently with member 0 (fork after the fragments, join before the
        // next op of the main stream: the consumer and the next PDL launch wait for every member)
        cudaStream_t ms = g_qwen.stream;
        cudaEventRecord(g_fork_ev0, ms);
        for (int i = 1; i < n; i++) cudaStreamWaitEvent(g_fork_st[i - 1], g_fork_ev0, 0);
        r = exl3_gemm_frags(ws[0], g_exl3_b, outs[0], N, ms);
        for (int i = 1; i < n; i++) {
            if (r == 0) {
                g_exl3_part_sel = i;
                r = exl3_gemm_frags(ws[i], g_exl3_b + i * need_w, outs[i], N, g_fork_st[i - 1]);
                g_exl3_part_sel = 0;
            }
            cudaEventRecord(g_fork_ev[i - 1], g_fork_st[i - 1]);
            cudaStreamWaitEvent(ms, g_fork_ev[i - 1], 0);
        }
        return r;
    }
    for (int i = 0; i < n && r == 0; i++) r = exl3_gemm_frags(ws[i], g_exl3_b + i * need_w, outs[i], N);
    return r;
}
#undef TQ_EXL3_LAUNCH
#undef TQ_EXL3_SWITCH_NG

// quantize activations [N x K] (token-major) -> B-frag groups in g_wproj_b (e4m3, shared
// pow2 scale per 128-K block in g_wproj_bscale). Mirrors the bench path exactly.
static int wide_quant_input(const float *d_x, int K, int N) {
    if (g_qwen.flags & TQ_FLAG_EXL3_WEIGHTS) {      // fp16 fragments are built per Linear (suh)
        g_exl3_x = d_x; g_exl3_x2 = NULL; g_exl3_K = K; g_exl3_N = N;
        return 0;
    }
    int Kt = (K + 31) / 32, G = wide_ng(N), nblocks = (K + 127) / 128;  // NG groups (pow2); extras zeroed
    size_t need_b = (size_t)G * Kt * 256;
    if (need_b > g_wproj_b_bytes) {
        graph_invalidate();
        if (g_wproj_b) cudaFree(g_wproj_b);
        if (cudaMalloc(&g_wproj_b, need_b) != cudaSuccess) return -1;
        g_wproj_b_bytes = need_b;
    }
    if ((size_t)nblocks > g_wproj_bscale_n) {
        graph_invalidate();
        if (g_wproj_bscale) cudaFree(g_wproj_bscale);
        if (cudaMalloc(&g_wproj_bscale, (size_t)nblocks * sizeof(float)) != cudaSuccess) return -2;
        g_wproj_bscale_n = nblocks;
    }
    cudaMemsetAsync(g_wproj_b, 0, need_b, g_qwen.stream);
    tq_quant_bfrag_fast_launch(g_wproj_b, g_wproj_bscale, d_x, K, N, g_qwen.stream);  // 7.6x, bit-exact
    (void)nblocks;
    return 0;
}

// run one wide projection from the prepared B-frag (call wide_quant_input first).
// the wide path takes E2M3 (6-bit) block-scaled weights (or EXL3).
static inline int tq_wide_fmt_ok(const tq_qmma_weight_t *w) {
    if (w->exl3) return 1;
    return can_use_qmma_sf_weight(w) && !w->row_major && w->e2m3;
}

static int wide_proj(const tq_qmma_weight_t *w, float *out, int N) {
    if (w->exl3) return exl3_wide_proj(w, out, N);
    if (!can_use_qmma_sf_weight(w) || w->row_major || !w->e2m3) return -90;
    int k_splits = qmma_sf_k_split_auto(w->Mt, w->Kt);
    if (k_splits > 1) {
        size_t need = (size_t)k_splits * N * w->M * sizeof(float);
        if (need > g_wproj_part_bytes) {
            graph_invalidate();
            if (g_wproj_part) cudaFree(g_wproj_part);
            if (cudaMalloc(&g_wproj_part, need) != cudaSuccess) return -91;
            g_wproj_part_bytes = need;
        }
    }
    return launch_fp6_wide_gemm(out, w, g_wproj_b, g_wproj_bscale, N, g_wproj_part, k_splits, g_qwen.stream);
}

// Two projections of the same input (call wide_quant_input first). The GDN in_proj_b / in_proj_a
// pair that EXL3 checkpoints keep in bf16 runs as one kernel at decode-size N.
static int exl3_bf16_pair_ok(const tq_qmma_weight_t *w0, const tq_qmma_weight_t *w1, int N) {
    return w0->exl3 && w1->exl3 && w0->exl3_kbits == 0 && w1->exl3_kbits == 0 && w0->d_exl3_bf16 && w1->d_exl3_bf16 &&
           w0->M == w1->M && w0->K == w1->K && (w0->K & 1023) == 0 &&
           g_exl3_x && g_exl3_K == w0->K && g_exl3_N >= N;
}
static void exl3_bf16_pair_launch(const tq_qmma_weight_t *w0, float *out0, const tq_qmma_weight_t *w1, float *out1, int N) {
    tq_launch(g_pdl_small, k_tq_exl3_bf16_pair, dim3(2 * w0->M, (N + 7) / 8), dim3(256), 0, g_qwen.stream,
              out0, w0->d_exl3_bf16, out1, w1->d_exl3_bf16, g_exl3_x, w0->M, w0->K, N);
}
static int wide_proj_pair(const tq_qmma_weight_t *w0, float *out0, const tq_qmma_weight_t *w1, float *out1, int N) {
    if (N <= 16 && exl3_bf16_pair_ok(w0, w1, N)) { exl3_bf16_pair_launch(w0, out0, w1, out1, N); return 0; }
    int ret = wide_proj(w0, out0, N);
    return ret != 0 ? ret : wide_proj(w1, out1, N);
}
// The verify wave: the pair kernel at any N (per-column math identical to the decode step's; the
// block-per-row wide kernel reduces in another order).
static int wide_proj_pair_seq(const tq_qmma_weight_t *w0, float *out0, const tq_qmma_weight_t *w1, float *out1, int N) {
    if (exl3_bf16_pair_ok(w0, w1, N)) { exl3_bf16_pair_launch(w0, out0, w1, out1, N); return 0; }
    return wide_proj_pair(w0, out0, w1, out1, N);
}

// Validation ABI: run one Linear of the loaded model through the wide path on host data.
// which: 0 q_proj 1 k_proj 2 v_proj 3 o_proj 4 mlp_gate 5 mlp_up 6 mlp_down 7 linear_in_qkv
// 8 linear_in_z 9 linear_in_b 10 linear_in_a 11 linear_out 12 lm_head (layer ignored).
// h_x [N][K] fp32 token-major -> h_out [N][M]. With h_x == NULL only *out_M/*out_K are filled.
extern "C" int qwn_wide_linear_test(int layer, int which, const float *h_x, int N, float *h_out,
                                    int *out_M, int *out_K, int *out_kbits) {
    if (!g_qwen.initialized) return -1;
    const tq_qmma_weight_t *w = NULL;
    if (which == 12) w = &g_qwen.lm_head;
    else {
        if (layer < 0 || layer >= g_qwen.L) return -2;
        tq_layer_t *l = &g_qwen.layers[layer];
        switch (which) {
            case 0: w = &l->q_proj; break;      case 1: w = &l->k_proj; break;
            case 2: w = &l->v_proj; break;      case 3: w = &l->o_proj; break;
            case 4: w = &l->mlp_gate; break;    case 5: w = &l->mlp_up; break;
            case 6: w = &l->mlp_down; break;    case 7: w = &l->linear_in_qkv; break;
            case 8: w = &l->linear_in_z; break; case 9: w = &l->linear_in_b; break;
            case 10: w = &l->linear_in_a; break; case 11: w = &l->linear_out; break;
            default: return -3;
        }
    }
    if (w->M <= 0) return -4;
    if (out_M) *out_M = w->M;
    if (out_K) *out_K = w->K;
    if (out_kbits) *out_kbits = w->exl3 ? w->exl3_kbits : -1;
    if (!h_x) return 0;
    if (N < 1) return -5;
    float *d_x = NULL, *d_out = NULL;
    size_t xb = (size_t)N * w->K * sizeof(float), ob = (size_t)N * w->M * sizeof(float);
    if (cudaMalloc(&d_x, xb) != cudaSuccess) return -6;
    if (cudaMalloc(&d_out, ob) != cudaSuccess) { cudaFree(d_x); return -6; }
    int ret = 0;
    // g_qwen.stream is non-blocking: order the upload on it (a synchronous cudaMemcpy from
    // pageable memory may return before the DMA lands, and the kernels would read a partial x).
    if (cudaMemcpyAsync(d_x, h_x, xb, cudaMemcpyHostToDevice, g_qwen.stream) != cudaSuccess) ret = -7;
    if (ret == 0 && cudaStreamSynchronize(g_qwen.stream) != cudaSuccess) ret = -7;
    if (ret == 0 && (ret = wide_quant_input(d_x, w->K, N)) != 0) ret = -100 + ret;
    if (ret == 0 && (ret = wide_proj(w, d_out, N)) != 0) ret = -200 + ret;
    if (ret == 0 && cudaStreamSynchronize(g_qwen.stream) != cudaSuccess) ret = -8;
    if (ret == 0 && cudaMemcpy(h_out, d_out, ob, cudaMemcpyDeviceToHost) != cudaSuccess) ret = -9;
    cudaFree(d_x); cudaFree(d_out);
    return ret;
}

// Timing ABI: ms per wide_quant_input + wide_proj of one Linear at N tokens (random-ish input).
extern "C" int qwn_wide_linear_bench(int layer, int which, int N, int iters, float *out_ms, long long *out_wbytes) {
    if (!g_qwen.initialized) return -1;
    int M = 0, K = 0, kb = 0;
    int r = qwn_wide_linear_test(layer, which, NULL, 0, NULL, &M, &K, &kb);
    if (r != 0) return r;
    const tq_qmma_weight_t *w = NULL;
    if (which == 12) w = &g_qwen.lm_head;
    else {
        tq_layer_t *l = &g_qwen.layers[layer];
        const tq_qmma_weight_t *tab[12] = {&l->q_proj, &l->k_proj, &l->v_proj, &l->o_proj, &l->mlp_gate, &l->mlp_up,
                                           &l->mlp_down, &l->linear_in_qkv, &l->linear_in_z, &l->linear_in_b,
                                           &l->linear_in_a, &l->linear_out};
        w = tab[which];
    }
    float *d_x = NULL, *d_out = NULL;
    if (cudaMalloc(&d_x, (size_t)N * K * sizeof(float)) != cudaSuccess) return -6;
    if (cudaMalloc(&d_out, (size_t)N * M * sizeof(float)) != cudaSuccess) { cudaFree(d_x); return -6; }
    cudaMemset(d_x, 0x3c, (size_t)N * K * sizeof(float));   // ~0.0107 everywhere
    int ret = 0;
    for (int i = 0; i < 3 && ret == 0; i++) {                // warm-up
        if ((ret = wide_quant_input(d_x, K, N)) != 0) break;
        if ((ret = wide_proj(w, d_out, N)) != 0) break;
    }
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0, g_qwen.stream);
    for (int i = 0; i < iters && ret == 0; i++) {
        if ((ret = wide_quant_input(d_x, K, N)) != 0) break;
        if ((ret = wide_proj(w, d_out, N)) != 0) break;
    }
    cudaEventRecord(e1, g_qwen.stream);
    cudaEventSynchronize(e1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, e0, e1);
    if (out_ms) *out_ms = ms / (iters > 0 ? iters : 1);
    if (out_wbytes) {
        long long wb = (w->exl3 && w->exl3_kbits > 0) ? (long long)M * K * w->exl3_kbits / 8
                     : (w->exl3 ? (long long)M * K * 2 : (long long)w->Mt * w->Kt * (w->e2m3 ? 384 : 512));
        *out_wbytes = wb;
    }
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaFree(d_x); cudaFree(d_out);
    return ret;
}

