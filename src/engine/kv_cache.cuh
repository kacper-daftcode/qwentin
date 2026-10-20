// the int4-K / E4M3-V cache layout helpers ((scale, zp) groups, Hadamard rotation) of the single-stream decoder and the paged pool.
// Part of the forward_qwen.cu unity build: included in order, not a standalone header.

// ===== E4M3 KV rows (the V rows of TQ_KV_Q4) =====
// Codes are stored as u8 E4M3 with one pow2 scale per (pos, kv_head) row section
// (absmax over the hd values): scale = exp2(ceil(log2(absmax/448))), so val/scale
// is an exact exponent shift and only the E4M3 mantissa rounding is lossy.
__device__ __forceinline__ uint8_t tq_float_to_e4m3(float val);   // defined below

static __device__ __forceinline__ float tq_kv_fp8_scale(float absmax) {
    if (!(absmax > 0.0f)) return 1.0f;
    return exp2f(ceilf(log2f(absmax / 448.0f)));
}

// Fast E4M3 decoder for the attention hot loop: shift the sign to bit 31 and the
// 7 magnitude bits into the f32 exponent/mantissa fields, then rescale by 2^120
// (handles normals AND subnormals exactly; same magnitudes as tq_e4m3_to_float
// for every encoder output). Branchless: 2 SHL + LOP3 + FMUL.
static __device__ __forceinline__ float tq_e4m3_dec_fast(uint8_t v) {
    uint32_t u = (uint32_t)v;
    uint32_t bits = ((u & 0x80u) << 24) | ((u & 0x7fu) << 20);
    return __uint_as_float(bits) * 0x1p+120f;
}

// Block-wide pow2 KV scale from one value per thread (blockDim multiple of 32,
// <= 256). sm8 = 8-float scratch, out = 1-float broadcast slot; the caller must
// __syncthreads() before re-using those slots. Identical value in all threads.
static __device__ __forceinline__ float tq_kv_block_scale(float v, float *sm8, float *out) {
    float m = fabsf(v);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float t = __shfl_down_sync(0xffffffffu, m, off);
        if (t > m) m = t;
    }
    if ((threadIdx.x & 31) == 0) sm8[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x == 0) {
        float mm = sm8[0];
        for (int w = 1; w < (int)(blockDim.x >> 5); w++)
            if (sm8[w] > mm) mm = sm8[w];
        *out = tq_kv_fp8_scale(mm);
    }
    __syncthreads();
    return *out;
}

// ===== 4-bit K prefix cache helpers (TQ_KV_Q4=1) =====
// K rows are stored Hadamard-rotated: k' = H k with H = H_2^x8 / 16 (orthogonal,
// symmetric -> self-inverse), so q'.k' == q.k exactly in R and the score path
// just rotates q once at prep. The rotation smears the per-channel outliers
// that break per-(pos,head) int4 grouping (proto: no-rot min cos 0.9936 vs
// 0.99918 rotated). Codes: asymmetric int4 per 32-channel group,
// x ~ (code - zp) * scale, (scale, zp) stored as floats.

// In-place FWHT over a 256-float smem buffer; every thread of a 256-thread
// block participates (tid 0..255 owns element tid). The 1/16 normalization
// makes H orthogonal (H self-inverse). Caller must __syncthreads() after any
// write to buf before calling.
static __device__ __forceinline__ void tq_fwht256(float *buf, int tid) {
    #pragma unroll
    for (int h = 1; h < 256; h <<= 1) {
        float a = buf[tid], b = buf[tid ^ h];
        __syncthreads();
        buf[tid] = (tid & h) ? (b - a) : (a + b);
        __syncthreads();
    }
    buf[tid] *= 0.0625f;
    __syncthreads();
}

// tq_fwht256 of one 256-vector held by a whole warp, no smem and no block sync: lane owns
// elements lane + 32*e (e = 0..7). Same butterflies in the same order with the same 1/16, so
// the result is bit-identical to tq_fwht256. Every lane of the warp must call it.
static __device__ __forceinline__ void tq_fwht256_warp(float x[8], int lane) {
    #pragma unroll
    for (int h = 1; h < 32; h <<= 1) {
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            float b = __shfl_xor_sync(0xffffffffu, x[e], h);
            x[e] = (lane & h) ? (b - x[e]) : (x[e] + b);
        }
    }
    #pragma unroll
    for (int h = 1; h < 8; h <<= 1) {
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            if (e & h) continue;
            float a = x[e], b = x[e | h];
            x[e] = a + b;
            x[e | h] = a - b;
        }
    }
    #pragma unroll
    for (int e = 0; e < 8; e++) x[e] *= 0.0625f;
}

// ===== TQ_KV_Q4 K (scale, zp) group storage helpers =====
// The d_k_q4s buffer holds (scale, zp) pairs as fp16 (default) or fp32 (escape
// hatch). To keep every signature `float*`, these reinterpret the base by the
// runtime type. q4s_adv advances a (tagged) base by `n` scale-elements; q4s_at
// loads element i; q4s_put stores. Only compose q4s_adv outputs with q4s_at/_put
// (never raw float* arithmetic on the result).
static __device__ __forceinline__ const float *q4s_adv(const float *p, size_t n) {
    return gc_kv_q4_fp32s ? (p + n) : (const float *)((const __half *)p + n);
}
static __device__ __forceinline__ float q4s_at(const float *p, size_t i) {
    return gc_kv_q4_fp32s ? p[i] : __half2float(((const __half *)p)[i]);
}
// One group's (scale, zp) pair in a single load (the pair is adjacent in both
// storage types). Bit-identical values to two q4s_at calls; halves the number
// of scattered scale loads in the attention hot loop.
static __device__ __forceinline__ float2 q4s_pair(const float *p, size_t g2) {
    if (gc_kv_q4_fp32s) {
        return *(const float2 *)(p + g2);
    }
    __half2 h = *(const __half2 *)((const __half *)p + g2);
    return __half22float2(h);
}
static __device__ __forceinline__ void q4s_put(float *p, size_t i, float v) {
    if (gc_kv_q4_fp32s) p[i] = v;
    else ((__half *)p)[i] = __float2half(v);
}

// Group-asymmetric int4 quantizer: one value per thread (256-thread block,
// 8 groups of 32 lanes = one warp each), returns this thread's code and leaves
// (scale, zp) of group g in gsz[2g], gsz[2g+1] (16-float smem scratch).
static __device__ __forceinline__ uint8_t tq_k_q4_quant(float v, int tid, float *gsz) {
    int lane = tid & 31;
    float lo = v, hi = v;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        lo = fminf(lo, __shfl_down_sync(0xffffffffu, lo, off));
        hi = fmaxf(hi, __shfl_down_sync(0xffffffffu, hi, off));
    }
    if (lane == 0) {
        int g = tid >> 5;
        float sc = (hi - lo) * (1.0f / 15.0f);
        if (!(sc > 1.175494e-38f)) sc = 1.175494e-38f;
        float zp = rintf(-lo / sc);
        zp = fminf(fmaxf(zp, 0.0f), 15.0f);
        gsz[2 * g] = sc;
        gsz[2 * g + 1] = zp;
    }
    __syncthreads();
    float sc = gsz[2 * (tid >> 5)], zp = gsz[2 * (tid >> 5) + 1];
    float q = rintf(v / sc + zp);
    return (uint8_t)fminf(fmaxf(q, 0.0f), 15.0f);
}

// int4-K variant of the attention tail (TQ_KV_Q4): qn arrives ROTATED (the
// caller runs tq_fwht256 on it; archive K rows are written rotated too, so
// both loops use the same q). kc4/ks4 pre-offset by kv_head like the fp8 tail;
// V stays E4M3 (identical fold as tq_attn_tail_pp_fp8).
static __device__ __noinline__ float tq_attn_tail_pp_q4(
    const float *qn, float *scores, float *red,
    const uint8_t *kc4, const float *ks4, const uint8_t *vc8, const float *vsc,
    size_t cstride, int sstride,
    const float *ka, const float *va, size_t astride, const int *anc_n,
    int base, int pos, int hd, int tid) {
    int warp = tid >> 5, lane = tid & 31;
    const size_t cstride4 = cstride >> 1;          // int4 row stride in bytes
    for (int t = warp; t <= base; t += 8) {
        const uint8_t *krow = kc4 + (size_t)t * cstride4;
        const float *ksz = q4s_adv(ks4, (size_t)t * (sstride * 16));
        float dot = 0.0f;
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            int d = lane + 32 * e;                 // channel; group g == e
            uint8_t byte = krow[d >> 1];
            float code = (float)((d & 1) ? (byte >> 4) : (byte & 15));
            dot += qn[d] * ((code - q4s_at(ksz, 2 * e + 1)) * q4s_at(ksz, 2 * e));
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            dot += __shfl_down_sync(0xffffffffu, dot, off);
        if (lane == 0) scores[t] = dot * rsqrtf((float)hd);
    }
    for (int t = base + 1 + warp; t <= pos; t += 8) {
        const float *krow = ka + (size_t)anc_n[t - base - 1] * astride;
        float dot = 0.0f;
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            int d = lane + 32 * e;
            dot += qn[d] * krow[d];
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            dot += __shfl_down_sync(0xffffffffu, dot, off);
        if (lane == 0) scores[t] = dot * rsqrtf((float)hd);
    }
    __syncthreads();
    float m = -3.402823466e38f;
    for (int t = tid; t <= pos; t += 256)
        if (scores[t] > m) m = scores[t];
    red[tid] = m;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s && red[tid + s] > red[tid]) red[tid] = red[tid + s];
        __syncthreads();
    }
    float max_score = red[0];
    __syncthreads();
    float ssum = 0.0f;
    for (int t = tid; t <= pos; t += 256) {
        float p = expf(scores[t] - max_score);
        ssum += p;
        scores[t] = (t <= base) ? p * vsc[(size_t)t * sstride] : p;
    }
    red[tid] = ssum;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    float denom = red[0];
    __syncthreads();
    float acc = 0.0f;
    const uint8_t *vrow8 = vc8 + tid;
    #pragma unroll 4
    for (int t = 0; t <= base; t++)
        acc += scores[t] * tq_e4m3_dec_fast(vrow8[(size_t)t * cstride]);
    for (int t = base + 1; t <= pos; t++)
        acc += scores[t] * va[(size_t)anc_n[t - base - 1] * astride + tid];
    return acc / denom;
}

// FP8 variant of the attention tail: rows <= base decode E4M3 codes
// from the prefix cache; the pow2 row scale is pulled out of the dot product
// (score = dot * ksc[t] * rsqrt(hd)) and the V scale is folded into p[t] before
// the hd-wide FMA stream. Scale pointers arrive pre-offset by kv_head and stride
// by sstride (= nkv); arch rows (> base) stay fp32. Loops split at base so the
// hot prefix loops stay branchless (a per-row t<=base branch defeats unrolling).
// __noinline__ keeps this path's register pressure from perturbing the fp32
// codegen of the (inlined) fp32 tail in every caller, incl. the persist kernel.
static __device__ __noinline__ float tq_attn_tail_pp_fp8(
    const float *qn, float *scores, float *red,
    const uint8_t *kc8, const float *ksc, const uint8_t *vc8, const float *vsc,
    size_t cstride, int sstride,
    const float *ka, const float *va, size_t astride, const int *anc_n,
    int base, int pos, int hd, int tid) {
    int warp = tid >> 5, lane = tid & 31;
    for (int t = warp; t <= base; t += 8) {
        // same lane->element mapping and summation order as the fp32 tail (only
        // the K element is decoded from E4M3): keeps the fp8 scores as close as
        // possible to fp32 on near-ties. A pair-reordered HW-cvt variant
        // (cvt.rn.f16x2.e4m3x2) measured ~6% faster on the attn phase but cost
        // 0.03-0.05 accept/round on prose via reassociation -- rejected.
        const uint8_t *krow = kc8 + (size_t)t * cstride;
        float dot = 0.0f;
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            int d = lane + 32 * e;
            dot += qn[d] * tq_e4m3_dec_fast(krow[d]);
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            dot += __shfl_down_sync(0xffffffffu, dot, off);
        if (lane == 0) scores[t] = dot * ksc[(size_t)t * sstride] * rsqrtf((float)hd);
    }
    for (int t = base + 1 + warp; t <= pos; t += 8) {
        const float *krow = ka + (size_t)anc_n[t - base - 1] * astride;
        float dot = 0.0f;
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            int d = lane + 32 * e;
            dot += qn[d] * krow[d];
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            dot += __shfl_down_sync(0xffffffffu, dot, off);
        if (lane == 0) scores[t] = dot * rsqrtf((float)hd);
    }
    __syncthreads();
    float m = -3.402823466e38f;
    for (int t = tid; t <= pos; t += 256)
        if (scores[t] > m) m = scores[t];
    red[tid] = m;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s && red[tid + s] > red[tid]) red[tid] = red[tid + s];
        __syncthreads();
    }
    float max_score = red[0];
    __syncthreads();
    // fold the V row scale into p[t] here (parallel strided pass) so the serial
    // hd-wide accumulation below stays at one decode + one FMA per row; the
    // denominator keeps the unscaled probabilities.
    float ssum = 0.0f;
    for (int t = tid; t <= pos; t += 256) {
        float p = expf(scores[t] - max_score);
        ssum += p;
        scores[t] = (t <= base) ? p * vsc[(size_t)t * sstride] : p;
    }
    red[tid] = ssum;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    float denom = red[0];
    __syncthreads();
    float acc = 0.0f;
    const uint8_t *vrow8 = vc8 + tid;
    #pragma unroll 4
    for (int t = 0; t <= base; t++)
        acc += scores[t] * tq_e4m3_dec_fast(vrow8[(size_t)t * cstride]);
    for (int t = base + 1; t <= pos; t++)
        acc += scores[t] * va[(size_t)anc_n[t - base - 1] * astride + tid];
    return acc / denom;
}

// Position-parallel attention tail (hd == 256, block == 256). The old tail ran
// an O(pos) loop with two block barriers per position (465 us/layer at ctx~1000,
// 34% of total GPU time at long context). Here independent warps compute score
// slices with no block syncs, max/denom are parallel tree reductions, the
// probabilities are materialized into scores[], and the V accumulation is a pure
// FMA stream. New summation order (max is exact; denom/acc tree-ordered) -- a
// deliberate numerics-baseline change applied to BOTH the dense and spec paths,
// gated by accept/divergence.
// Rows <= base come from the (cache, cstride) pair; rows > base resolve through
// anc_n into the (arch, astride) pair. Dense callers pass base == pos.
// `red` may alias qn (qn is dead once the score slices are done).
// kc8 != NULL routes to the FP8 tail above; kc4 != NULL routes to
// the int4-K tail (TQ_KV_Q4, qn must arrive rotated); the fp32 body below is
// untouched.
static __device__ __forceinline__ float tq_attn_tail_pp(
    const float *qn, float *scores, float *red,
    const float *kc, const float *vc, size_t cstride,
    const uint8_t *kc8, const float *ksc, const uint8_t *vc8, const float *vsc, int sstride,
    const uint8_t *kc4, const float *ks4,
    const float *ka, const float *va, size_t astride, const int *anc_n,
    int base, int pos, int hd, int tid) {
    if (kc4)
        return tq_attn_tail_pp_q4(qn, scores, red, kc4, ks4, vc8, vsc, cstride, sstride,
                                  ka, va, astride, anc_n, base, pos, hd, tid);
    if (kc8)
        return tq_attn_tail_pp_fp8(qn, scores, red, kc8, ksc, vc8, vsc, cstride, sstride,
                                   ka, va, astride, anc_n, base, pos, hd, tid);
    int warp = tid >> 5, lane = tid & 31;
    for (int t = warp; t <= pos; t += 8) {
        const float *krow = (t <= base) ? kc + (size_t)t * cstride
                                        : ka + (size_t)anc_n[t - base - 1] * astride;
        float dot = 0.0f;
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            int d = lane + 32 * e;
            dot += qn[d] * krow[d];
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            dot += __shfl_down_sync(0xffffffffu, dot, off);
        if (lane == 0) scores[t] = dot * rsqrtf((float)hd);
    }
    __syncthreads();
    float m = -3.402823466e38f;
    for (int t = tid; t <= pos; t += 256)
        if (scores[t] > m) m = scores[t];
    red[tid] = m;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s && red[tid + s] > red[tid]) red[tid] = red[tid + s];
        __syncthreads();
    }
    float max_score = red[0];
    __syncthreads();
    float ssum = 0.0f;
    for (int t = tid; t <= pos; t += 256) {
        float p = expf(scores[t] - max_score);
        scores[t] = p;
        ssum += p;
    }
    red[tid] = ssum;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    float denom = red[0];
    __syncthreads();
    float acc = 0.0f;
    for (int t = 0; t <= pos; t++) {
        const float *vrow = (t <= base) ? vc + (size_t)t * cstride
                                        : va + (size_t)anc_n[t - base - 1] * astride;
        acc += scores[t] * vrow[tid];
    }
    return acc / denom;
}

// ===== Shared-memory layout of the MMA attention kernels (wide prefill, paged decode / verify) =====
// Positions are tiled 16 wide; online softmax (running m / l, fp32 rescale) across tiles.
// dynamic smem layout (32-bit words)
#define TQ_AMM_QA      0                    // [16 chunks][32 lanes][4] bf16 Q A-frags
#define TQ_AMM_PST     3072                 // [8 tiles][32][4] P A-frag staging
#define TQ_AMM_WMAX    4096                 // [8 warps][16 rows] tile row-max
#define TQ_AMM_WSUM    4224                 // [8 warps][16 rows] tile row-sum
#define TQ_AMM_MST     4352                 // [16] running row max
#define TQ_AMM_LST     4368                 // [16] running row denom
#define TQ_AMM_ALPH    4384                 // [16] per-row rescale exp(m_old-m_new)
#define TQ_AMM_QTMP    4400                 // [256] one node's normed+roped q
#define TQ_AMM_QSC     4656                 // [16] per-node Q pow2 scale (mode 2)
#define TQ_AMM_RED     4688                 // [9] block-reduce scratch
#define TQ_AMM_WORDS   4704
#define TQ_ATTN_SCORE_SLOTS 512             // [512 x max_seq] gmem score/partial slab
#define TQ_AMM_NINF    (-3.402823466e38f)   // finite -inf stand-in (exp underflows to 0)

static __device__ __forceinline__ uint32_t tq_pack_bf16(float lo, float hi) {
    uint32_t d;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
    return d;
}

static __device__ __forceinline__ void tq_mma_bf16_m16n8k16(float *c, const uint32_t *a,
                                                            uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

