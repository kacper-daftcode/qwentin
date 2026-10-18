// prefill attention: wide flash-causal, MMA wide prefill, the GQA-grouped trellis-pool kernel (TQ_PATTN_GQA).
// Part of the forward_qwen.cu unity build: included in order, not a standalone header.

// ===================== D.1 (2): wide flash-causal prefill attention =====================
// A dedicated N-wide full attention for a prefill chunk, with NO 16-slot archive (the
// k_tq_spec_attn_tree_mma archive overflows for chunks > ~16 and corrupts long context).
// It mirrors k_tq_full_attn_decode_arr EXACTLY -- q/k RMSNorm over hd, partial RoPE
// (rotary_dim 64, rotate-half +-32 pairing), GQA head->kv_head map, the
// tq_attn_tail_pp online softmax, and the sigmoid(gate) fold -- but processes N query
// tokens in one launch by reading straight from the fp32 KV cache rows [0..pos]
// (pos = positions[token]) instead of a per-chunk archive tile.
//
// Two phases, launched in order on the same stream (so phase 1 finishes first):
//   (1) k_tq_wide_attn_kv_write : write every chunk token's normed+roped K and raw V
//       into the live cache at its absolute slot, so token i can attend causally to
//       the earlier chunk tokens 0..i-1 (and itself) -- identical bytes to the K/V
//       store inside k_tq_full_attn_decode_arr.
//   (2) k_tq_wide_attn          : each (head, token) block runs the dense online
//       softmax over cache rows [0..pos]. Scores live in DYNAMIC SHARED (pos+1 floats),
//       so there is no nh*N*max_seq gmem scores slab; tq_attn_tail_pp does not care
//       whether `scores` is gmem or smem (same write-then-read, same reduction order).
// Running this pair is bit-identical to invoking k_tq_full_attn_decode_arr once per
// chunk token in ascending order against the same prefix cache.
// SCOPE: fp32 KV cache only. TQ_KV_FP8 / TQ_KV_Q4 are a deliberate TODO (the callers
// reject those modes); the fp8/q4 K/V store + tail variants are not wired here yet.

__global__ void k_tq_wide_attn_kv_write(
    float *k_cache, float *v_cache,
    const float *k_proj_base, const float *v_proj_base,
    const uint16_t *k_norm_w, const int *positions,
    int nkv, int hd, int kv_m, float eps, float rope_theta) {
    __shared__ float partial[8];
    __shared__ float k_sum_shared;
    int kv_head = blockIdx.x;
    int i = blockIdx.y;          // chunk token index
    int tid = threadIdx.x;
    if (kv_head >= nkv || tid >= hd) return;
    int pos = positions[i];
    const float *k_proj = k_proj_base + (size_t)i * kv_m;
    const float *v_proj = v_proj_base + (size_t)i * kv_m;
    int rotary_dim = 64;
    float kv = k_proj[kv_head * hd + tid];
    float ksum = kv * kv;
    for (int offset = 16; offset > 0; offset >>= 1)
        ksum += __shfl_down_sync(0xffffffff, ksum, offset);
    if ((tid & 31) == 0) partial[tid >> 5] = ksum;
    __syncthreads();
    if (tid < 8) {
        float vtmp = partial[tid];
        for (int offset = 4; offset > 0; offset >>= 1) vtmp += __shfl_down_sync(0xff, vtmp, offset);
        if (tid == 0) k_sum_shared = vtmp;
    }
    __syncthreads();
    kv = kv * rsqrtf(k_sum_shared / (float)hd + eps) * (1.0f + tq_bf16_to_float(k_norm_w[tid]));
    if (tid < rotary_dim) {
        int idx = tid & 31;
        float freq = powf(rope_theta, -((float)(2 * idx) / (float)rotary_dim));
        float angle = (float)pos * freq;
        float c = cosf(angle);
        float s = sinf(angle);
        float k_pair = k_proj[kv_head * hd + ((tid < 32) ? tid + 32 : tid - 32)];
        k_pair = k_pair * rsqrtf(k_sum_shared / (float)hd + eps) * (1.0f + tq_bf16_to_float(k_norm_w[(tid < 32) ? tid + 32 : tid - 32]));
        float k_rot = (tid < 32) ? -k_pair : k_pair;
        kv = kv * c + k_rot * s;
    }
    k_cache[((size_t)pos * nkv + kv_head) * hd + tid] = kv;
    v_cache[((size_t)pos * nkv + kv_head) * hd + tid] = v_proj[kv_head * hd + tid];
}

// Flash key-chunk size: the cache rows [0..pos] are processed in fixed-width chunks
// with a running online softmax, so the per-block shared-memory footprint is O(chunk)
// (a few KB), NOT O(pos). This is what lets the wide path scale past ctx ~32k -- the
// old version stored the whole score row (pos+1 floats) in dynamic shared memory and
// launch_wide_attn died (-11) once that exceeded the SM120 per-block smem cap (~100KB)
// near 25-32k. 2048 keeps s_chunk at 8 KB (total static smem ~10 KB, no opt-in needed).
#ifndef TQ_WIDE_ATTN_CHUNK
#define TQ_WIDE_ATTN_CHUNK 2048
#endif

// Flash-style key-chunked rewrite of the wide-prefill full attention (D.1(2) ctx fix).
// Per (head, query-token) block it walks the fp32 KV cache rows [0..pos] in
// TQ_WIDE_ATTN_CHUNK-sized chunks, keeping the softmax running max/denom in shared
// memory and the per-thread output column (dim = tid) in a register, rescaling on each
// chunk exactly like tq_attn_mma_block's flash-decoding split. The q/k RMSNorm + partial
// RoPE prologue and the dot*rsqrt(hd) score scaling are byte-for-byte the same as the
// per-token reference k_tq_full_attn_decode_arr / tq_attn_tail_pp; only the softmax
// reduction is tiled (chunked m/l rescale) instead of one flat pass over a pos-sized
// score buffer, so the result is mathematically identical and matches the reference to
// fp reassociation (~1e-6), far inside the 1e-3 gate. fp32 KV only (fp8/q4 TODO).
__global__ void k_tq_wide_attn(
    float *out, const float *q_proj_base, const uint16_t *q_norm_w,
    const float *k_cache, const float *v_cache,
    const int *positions, int nh, int nkv, int hd, int q_m, int attn_m,
    float eps, float rope_theta) {
    __shared__ float partial[8];
    __shared__ float q_sum_shared;
    __shared__ float qn[256];                       // normed+roped q, persists across chunks
    __shared__ float s_chunk[TQ_WIDE_ATTN_CHUNK];   // one chunk of scores/probs (O(chunk))
    __shared__ float red[256];                      // block-reduce scratch (chunk max/sum)
    __shared__ float m_run, l_run;                  // running online-softmax max / denom
    int head = blockIdx.x;
    int i = blockIdx.y;          // chunk token index
    int tid = threadIdx.x;
    if (head >= nh || tid >= hd) return;
    int pos = positions[i];
    int group = nh / nkv;
    int kv_head = head / group;
    int rotary_dim = 64;
    const float *q_proj = q_proj_base + (size_t)i * q_m;
    float qv = q_proj[head * (2 * hd) + tid];
    float qsum = qv * qv;
    for (int offset = 16; offset > 0; offset >>= 1)
        qsum += __shfl_down_sync(0xffffffff, qsum, offset);
    if ((tid & 31) == 0) partial[tid >> 5] = qsum;
    __syncthreads();
    if (tid < 8) {
        float vtmp = partial[tid];
        for (int offset = 4; offset > 0; offset >>= 1) vtmp += __shfl_down_sync(0xff, vtmp, offset);
        if (tid == 0) q_sum_shared = vtmp;
    }
    __syncthreads();
    qv = qv * rsqrtf(q_sum_shared / (float)hd + eps) * (1.0f + tq_bf16_to_float(q_norm_w[tid]));
    if (tid < rotary_dim) {
        int idx = tid & 31;
        float freq = powf(rope_theta, -((float)(2 * idx) / (float)rotary_dim));
        float angle = (float)pos * freq;
        float c = cosf(angle);
        float s = sinf(angle);
        float q_pair = q_proj[head * (2 * hd) + ((tid < 32) ? tid + 32 : tid - 32)];
        q_pair = q_pair * rsqrtf(q_sum_shared / (float)hd + eps) * (1.0f + tq_bf16_to_float(q_norm_w[(tid < 32) ? tid + 32 : tid - 32]));
        float q_rot = (tid < 32) ? -q_pair : q_pair;
        qv = qv * c + q_rot * s;
    }
    qn[tid] = qv;
    if (tid == 0) { m_run = -3.402823466e38f; l_run = 0.0f; }
    __syncthreads();

    // Flash online softmax over the causal prefix rows [0..pos] of the fp32 cache. Each
    // cache row is [t][kv_head][hd]; stride between rows is nkv*hd. Thread `tid` owns
    // output dim `tid` and keeps its running (rescaled) numerator in the register `acc`.
    const float *kc = k_cache + (size_t)kv_head * hd;
    const float *vc = v_cache + (size_t)kv_head * hd;
    size_t cstride = (size_t)nkv * hd;
    int warp = tid >> 5, lane = tid & 31;
    float scale = rsqrtf((float)hd);
    float acc = 0.0f;
    int total = pos + 1;
    for (int c0 = 0; c0 < total; c0 += TQ_WIDE_ATTN_CHUNK) {
        int clen = total - c0;
        if (clen > TQ_WIDE_ATTN_CHUNK) clen = TQ_WIDE_ATTN_CHUNK;
        // (1) scores for this chunk: one warp per row, 8 dims/lane, warp-reduced dot.
        for (int j = warp; j < clen; j += 8) {
            const float *krow = kc + (size_t)(c0 + j) * cstride;
            float dot = 0.0f;
            #pragma unroll
            for (int e = 0; e < 8; e++) { int d = lane + 32 * e; dot += qn[d] * krow[d]; }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) dot += __shfl_down_sync(0xffffffffu, dot, off);
            if (lane == 0) s_chunk[j] = dot * scale;
        }
        __syncthreads();
        // (2) chunk max (block reduce over s_chunk[0..clen)).
        float m = -3.402823466e38f;
        for (int j = tid; j < clen; j += 256) if (s_chunk[j] > m) m = s_chunk[j];
        red[tid] = m;
        __syncthreads();
        for (int s = 128; s > 0; s >>= 1) {
            if (tid < s && red[tid + s] > red[tid]) red[tid] = red[tid + s];
            __syncthreads();
        }
        float m_chunk = red[0];
        __syncthreads();
        // (3) running-max update + rescale factor (m_run = -inf on chunk 0 -> alpha = 0).
        float m_old = m_run;
        float m_new = fmaxf(m_old, m_chunk);
        float alpha = expf(m_old - m_new);
        // (4) probabilities exp(score - m_new) (overwrite s_chunk) + chunk denom.
        float lsum = 0.0f;
        for (int j = tid; j < clen; j += 256) {
            float p = expf(s_chunk[j] - m_new);
            s_chunk[j] = p;
            lsum += p;
        }
        red[tid] = lsum;
        __syncthreads();
        for (int s = 128; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            __syncthreads();
        }
        float l_chunk = red[0];
        // (5) rescale the running numerator and fold in this chunk's weighted V column.
        acc *= alpha;
        for (int j = 0; j < clen; j++)
            acc += s_chunk[j] * vc[(size_t)(c0 + j) * cstride + tid];
        if (tid == 0) { l_run = l_run * alpha + l_chunk; m_run = m_new; }
        __syncthreads();
    }
    float ratio = acc / l_run;
    float gate = q_proj[head * (2 * hd) + hd + tid];
    out[(size_t)i * attn_m + head * hd + tid] = ratio * (1.0f / (1.0f + expf(-gate)));
}

// ===== milestone 3.6: MMA (tensor-core) wide-prefill attention (fp32 KV) =====
// PREFILL attention is COMPUTE-bound (~170 FLOP/byte: 16 query rows reuse the same K), so
// tensor cores win here (unlike the memory-bound decode, kept scalar). Clean causal flash
// attention -- 16-query m16 tiles (grid.y = ceil(N/16)) attend the fp32 KV cache causally
// (per-row STAIRCASE mask: token attends keys [0..its pos]), MMA Q.K^T -> online softmax ->
// MMA P.V. Reuses tq_attn_mma_block's mode-1 fragment layout / helpers; NO spec-tree /
// ancestor / archive machinery (that path stays intact). Replaces the scalar k_tq_wide_attn
// where it beats the baseline tensor-core path (>~16k), lifting the wide-prefill ctx cap.
// Key-split (gridDim.z = S > 1): the causal key range [0..maxpos] is cut into S
// even super-tile chunks -- one (head, qtile, kchunk) block each -- so a long
// prefix no longer serializes in nh x ceil(N/16) blocks (192 blocks left half
// the GPU idle and each walked ~1000 super-tiles at 128k). Chunks write
// unnormalized (m, l, O) partials to `part`; k_tq_wide_attn_mma_merge combines
// them (same flash-decoding math as the verify path). S == 1 keeps the exact
// single-pass epilogue (bit-identical to the pre-split kernel).
// VQ4 = 1 (MODE 4 only): the paged V is int4 like K (TQ_KV_V4, k_tq_paged_attn_kv_write_q4), so O
// accumulates in the rotated space; the epilogue / merge run one FWHT-256 per row before the gate.
// T4 = 1 (TQ_KV_T4, MODE 4 + VQ4 only): block ids < t4p.nblk are complete trellis pages (k4 / v8 = code
// bytes, side words in t4p); every key of a prefill wave is a trellis row (the wave's columns are encoded per
// layer before the attention); fp16 table in smem.
template <int MODE, int QT = 1, int VQ4 = 0, int T4 = 0>   // 1 = fp32 KV, 3 = contiguous Q4, 4 = paged Q4; QT = 16-row query tiles per block
__global__ void k_tq_wide_attn_mma(
    float *out, const float *q_proj_base, const uint16_t *q_norm_w,
    const float *k_cache, const float *v_cache,
    const uint8_t *k4, const float *kq4s, const uint8_t *v8, const float *vscale,
    const int *positions,
    int slot, const int *block_table, int max_blocks, int page, int page_log,
    int N, int nh, int nkv, int hd, int q_m, int attn_m, float eps, float rope_theta,
    float *part, const tq_t4_pool t4p = tq_t4_pool()) {
    // QT=2 packs TWO 16-row query tiles per block so the 128-key super-tile
    // stream is read from L2 once per 32 queries (the prefill kernel is
    // L2-bandwidth-bound: 72% L2 vs 23% SM at MODE 3). Layout is a relocation
    // of the QT=1 floor plan; numerics are bit-identical per query row.
    constexpr int QROWS   = 16 * QT;
    constexpr int OFF_QA  = 0;
    constexpr int OFF_QA2 = 2048;                    // QT=2 only
    constexpr int OFF_PST  = (QT == 2) ? 4096 : TQ_AMM_PST;
    constexpr int OFF_PST2 = 5120;                   // QT=2 only
    constexpr int OFF_WMAX = (QT == 2) ? 6144 : TQ_AMM_WMAX;
    constexpr int OFF_WSUM = (QT == 2) ? 6400 : TQ_AMM_WSUM;
    constexpr int OFF_MST  = (QT == 2) ? 6656 : TQ_AMM_MST;
    constexpr int OFF_LST  = (QT == 2) ? 6688 : TQ_AMM_LST;
    constexpr int OFF_ALPH = (QT == 2) ? 6720 : TQ_AMM_ALPH;
    constexpr int OFF_QTMP = (QT == 2) ? 6752 : TQ_AMM_QTMP;
    constexpr int OFF_QSC  = (QT == 2) ? 7008 : TQ_AMM_QSC;
    constexpr int OFF_REDC = (QT == 2) ? 7072 : TQ_AMM_RED;
    constexpr int ZERCAP   = (QT == 2) ? 6144 : TQ_AMM_PST + 1024;
    constexpr int SMEM_W   = (QT == 2) ? 7104 : TQ_AMM_WORDS;
    static_assert(!VQ4 || MODE == 4, "int4 V exists only in the paged pool");
    static_assert(!T4 || (MODE == 4 && VQ4 == 1), "trellis pages exist only in the paged pool with int4 V staging");
    __shared__ uint32_t sm[SMEM_W];
    __shared__ __align__(16) uint16_t t4tab[T4 ? 4096 : 8];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int gr = lane >> 2, t4 = lane & 3;
    int head = blockIdx.x, qtile = blockIdx.y, qtile_base = qtile * QROWS;
    if (head >= nh) return;
    const int group = nh / nkv, kv_head = head / group;
    const size_t cstride = (size_t)nkv * hd;
    const float *kcb = k_cache ? k_cache + (size_t)kv_head * hd : NULL;
    const float *vcb = v_cache ? v_cache + (size_t)kv_head * hd : NULL;
    const uint8_t *kcb4 = k4 ? k4 + (size_t)kv_head * (hd >> 1) : NULL;
    const float *ksz4 = kq4s ? q4s_adv(kq4s, (size_t)kv_head * 16) : NULL;
    // VQ4: v8 / vscale hold int4 V codes / (scale, zp) pairs in K's layout
    const uint8_t *vcb8 = v8 ? v8 + (size_t)kv_head * (VQ4 ? (hd >> 1) : hd) : NULL;
    const float *vscb = vscale ? (VQ4 ? q4s_adv(vscale, (size_t)kv_head * 16) : vscale + kv_head) : NULL;
    const size_t cstride4 = cstride >> 1, szstride = (size_t)nkv * 16;
    if constexpr (T4) tq_t4_load_table(t4p.table, t4tab, tid);
    float *qtmp = (float *)(sm + OFF_QTMP);
    float *red = (float *)(sm + OFF_REDC);
    float *wmax = (float *)(sm + OFF_WMAX), *wsum = (float *)(sm + OFF_WSUM);
    float *mst = (float *)(sm + OFF_MST), *lst = (float *)(sm + OFF_LST);
    float *alph = (float *)(sm + OFF_ALPH);
    int nvalid = N - qtile_base; if (nvalid > QROWS) nvalid = QROWS;
    int maxpos = positions[qtile_base + nvalid - 1];           // ascending -> tile's max key pos
    for (int i = tid; i < ZERCAP; i += 256) sm[i] = 0;
    if (tid < QROWS) { mst[tid] = TQ_AMM_NINF; lst[tid] = 0.0f; }
    __syncthreads();
    // ---- Q prep: norm + partial rope (no fwht; fp32), pack into m16k16 A-fragments ----
    for (int n = 0; n < nvalid; n++) {
        const float *q_proj = q_proj_base + (size_t)(qtile_base + n) * q_m;
        int pos = positions[qtile_base + n];
        float qv = q_proj[head * (2 * hd) + tid];
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
            float q_pair = q_proj[head * (2 * hd) + ((tid < 32) ? tid + 32 : tid - 32)];
            q_pair = q_pair * rsqrtf(qss / (float)hd + eps) * (1.0f + tq_bf16_to_float(q_norm_w[(tid < 32) ? tid + 32 : tid - 32]));
            float q_rot = (tid < 32) ? -q_pair : q_pair;
            qv = qv * c + q_rot * s;
        }
        __syncthreads();
        qtmp[tid] = qv;
        __syncthreads();
        if (MODE != 1) tq_fwht256(qtmp, tid);          // rotated-space q (matches rotated int4 K)
        if (tid < 128) {
            int c16 = tid >> 3, tt = (tid >> 1) & 3, rsel = tid & 1;
            int d0 = 16 * c16 + 2 * tt + 8 * rsel;
            int nn = n & 15;
            int l = 4 * (nn & 7) + tt, r = ((nn >> 3) & 1) + 2 * rsel;
            int qoff = (QT == 2 && n >= 16) ? OFF_QA2 : OFF_QA;
            sm[qoff + (c16 * 32 + l) * 4 + r] = tq_pack_bf16(qtmp[d0], qtmp[d0 + 1]);
        }
        __syncthreads();
    }
    // ---- main loop: 128-key super-tiles over [0..maxpos], online softmax.
    // Key-split: this block only walks its gridDim.z-slice of the super-tiles.
    float oacc[4][4];
    float oacc2[QT == 2 ? 4 : 1][4];
    #pragma unroll
    for (int f = 0; f < 4; f++) for (int j = 0; j < 4; j++) oacc[f][j] = 0.0f;
    if constexpr (QT == 2)
        for (int f = 0; f < 4; f++) for (int j = 0; j < 4; j++) oacc2[f][j] = 0.0f;
    const float rsc = rsqrtf((float)hd);
    int nst_all = (maxpos + 128) >> 7;                          // ceil((maxpos+1)/128)
    int it_lo = 0, nst = nst_all;
    if (gridDim.z > 1) {
        int per = (nst_all + gridDim.z - 1) / gridDim.z;
        it_lo = blockIdx.z * per;
        int hi = it_lo + per;
        nst = hi < nst_all ? hi : nst_all;                      // exclusive upper tile
    }
    for (int it = it_lo; it < nst; it++) {
        const int t0 = it << 7;
        // paged (MODE 4): a 128-aligned super-tile lies in ONE physical block (page>=128),
        // so one block-table lookup gives a contiguous physical-row base for the 128 keys.
        size_t pr0 = 0;
        constexpr bool t4blk = T4 != 0;
        // T4: page-local rows < t4nenc are trellis rows of the pool block, rows >= t4nenc the wave's own in-flight
        // columns, read exact from the slot's fp16 staging (TQ_KV_T4_ASYNC: their trellis encode runs on a side
        // stream behind the layer loop; in the synchronous mode encl was raised to the wave's end before the launch)
        int t4nenc = 256;
        if (MODE == 4) {
            int phys = block_table[(size_t)slot * max_blocks + (t0 >> page_log)];
            pr0 = (size_t)phys * page + (size_t)(t0 & (page - 1));
            if constexpr (T4) t4nenc = t4p.encl[slot] - t0;
        }
        const uint16_t *t4f16 = T4 ? t4p.f16s + ((size_t)slot * page * nkv + kv_head) * 512 : NULL;   // + (pos & (page-1)) * nkv * 512
        const uint8_t *kcb4_it = kcb4;
        const float *ksz4_it = ksz4;
        const uint8_t *vcb8_it = vcb8;
        const float *vscb_it = vscb;
#define TQ_PROW(lp) (MODE == 4 ? (pr0 + (size_t)(min((lp), maxpos) - t0)) : (size_t)min((lp), maxpos))
        float s0[4], s1[4], s2[4], s3[4];
        #pragma unroll
        for (int j = 0; j < 4; j++) { s0[j] = 0.0f; s1[j] = 0.0f; s2[j] = 0.0f; s3[j] = 0.0f; }
        const int tp = t0 + warp * 16;
        int p0 = tp + gr, p1 = tp + 8 + gr;
        if (T4 && t4blk && t4nenc >= 128) {                    // all 128 keys trellis: the straight-line table path
            const size_t r0 = TQ_PROW(p0), r1 = TQ_PROW(p1);
            const uint8_t *kr0 = kcb4 + r0 * cstride4;
            const uint8_t *kr1 = kcb4 + r1 * cstride4;
            const uint32_t sw0 = tq_t4_side(t4p.kside, r0, nkv, kv_head), sw1 = tq_t4_side(t4p.kside, r1, nkv, kv_head);
            const float sc0 = tq_t4_scale(sw0), sc1 = tq_t4_scale(sw1);
            uint32_t prev0 = sw0 & 0xFFu, prev1 = sw1 & 0xFFu;
            #pragma unroll
            for (int g = 0; g < 8; g++) {
                const uint4 kp0 = *(const uint4 *)(kr0 + 16 * g);
                const uint4 kp1 = *(const uint4 *)(kr1 + 16 * g);
                uint32_t bk0[2], bk8[2], ek0[2], ek8[2];
                tq_t4_kfrags(kp0, prev0, t4, sc0, t4tab, bk0, bk8);
                tq_t4_kfrags(kp1, prev1, t4, sc1, t4tab, ek0, ek8);
                prev0 = kp0.w >> 24; prev1 = kp1.w >> 24;
                #pragma unroll
                for (int cc = 0; cc < 2; cc++) {
                    const int c = 2 * g + cc;
                    const uint32_t *a = &sm[OFF_QA + (c * 32 + lane) * 4];
                    tq_mma_bf16_m16n8k16(s0, a, bk0[cc], bk8[cc]);
                    tq_mma_bf16_m16n8k16(s1, a, ek0[cc], ek8[cc]);
                    if constexpr (QT == 2) {
                        const uint32_t *a2 = &sm[OFF_QA2 + (c * 32 + lane) * 4];
                        tq_mma_bf16_m16n8k16(s2, a2, bk0[cc], bk8[cc]);
                        tq_mma_bf16_m16n8k16(s3, a2, ek0[cc], ek8[cc]);
                    }
                }
            }
        } else if (T4 && t4blk) {                              // the wave's own super-tile: trellis / exact fp16 per row
            const int o0 = min(p0, maxpos) - t0, o1 = min(p1, maxpos) - t0;
            const bool tr0 = o0 < t4nenc, tr1 = o1 < t4nenc;
            const size_t r0 = TQ_PROW(p0), r1 = TQ_PROW(p1);
            const uint8_t *kr0 = tr0 ? kcb4 + r0 * cstride4 : NULL;
            const uint8_t *kr1 = tr1 ? kcb4 + r1 * cstride4 : NULL;
            const uint16_t *kf0 = tr0 ? NULL : t4f16 + (size_t)((min(p0, maxpos)) & (page - 1)) * nkv * 512;
            const uint16_t *kf1 = tr1 ? NULL : t4f16 + (size_t)((min(p1, maxpos)) & (page - 1)) * nkv * 512;
            const uint32_t sw0 = tr0 ? tq_t4_side(t4p.kside, r0, nkv, kv_head) : 0u;
            const uint32_t sw1 = tr1 ? tq_t4_side(t4p.kside, r1, nkv, kv_head) : 0u;
            const float sc0 = tq_t4_scale(sw0), sc1 = tq_t4_scale(sw1);
            uint32_t prev0 = sw0 & 0xFFu, prev1 = sw1 & 0xFFu;
            #pragma unroll
            for (int g = 0; g < 8; g++) {
                uint32_t bk0[2], bk8[2], ek0[2], ek8[2];
                if (tr0) {
                    const uint4 kp0 = *(const uint4 *)(kr0 + 16 * g);
                    tq_t4_kfrags(kp0, prev0, t4, sc0, t4tab, bk0, bk8);
                    prev0 = kp0.w >> 24;
                } else {
                    #pragma unroll
                    for (int cc = 0; cc < 2; cc++) {
                        const int ch = 32 * g + 16 * cc + 2 * t4;
                        bk0[cc] = tq_t4_f16x2_bf16(*(const uint32_t *)(kf0 + ch));
                        bk8[cc] = tq_t4_f16x2_bf16(*(const uint32_t *)(kf0 + ch + 8));
                    }
                }
                if (tr1) {
                    const uint4 kp1 = *(const uint4 *)(kr1 + 16 * g);
                    tq_t4_kfrags(kp1, prev1, t4, sc1, t4tab, ek0, ek8);
                    prev1 = kp1.w >> 24;
                } else {
                    #pragma unroll
                    for (int cc = 0; cc < 2; cc++) {
                        const int ch = 32 * g + 16 * cc + 2 * t4;
                        ek0[cc] = tq_t4_f16x2_bf16(*(const uint32_t *)(kf1 + ch));
                        ek8[cc] = tq_t4_f16x2_bf16(*(const uint32_t *)(kf1 + ch + 8));
                    }
                }
                #pragma unroll
                for (int cc = 0; cc < 2; cc++) {
                    const int c = 2 * g + cc;
                    const uint32_t *a = &sm[OFF_QA + (c * 32 + lane) * 4];
                    tq_mma_bf16_m16n8k16(s0, a, bk0[cc], bk8[cc]);
                    tq_mma_bf16_m16n8k16(s1, a, ek0[cc], ek8[cc]);
                    if constexpr (QT == 2) {
                        const uint32_t *a2 = &sm[OFF_QA2 + (c * 32 + lane) * 4];
                        tq_mma_bf16_m16n8k16(s2, a2, bk0[cc], bk8[cc]);
                        tq_mma_bf16_m16n8k16(s3, a2, ek0[cc], ek8[cc]);
                    }
                }
            }
        } else if (MODE != 1) {                                // int4-K dequant (rotated) into the Q·Kᵀ tile
            const uint8_t *kr0 = kcb4_it + TQ_PROW(p0) * cstride4;
            const uint8_t *kr1 = kcb4_it + TQ_PROW(p1) * cstride4;
            const float *sz0 = q4s_adv(ksz4_it, TQ_PROW(p0) * szstride);
            const float *sz1 = q4s_adv(ksz4_it, TQ_PROW(p1) * szstride);
            #pragma unroll
            for (int g = 0; g < 8; g++) {                      // 32-channel group == c>>1
                float2 p0s = q4s_pair(sz0, 2 * g);             // (scale, zp) one load
                float2 p1s = q4s_pair(sz1, 2 * g);
                float sc0 = p0s.x, zp0 = p0s.y;
                float sc1 = p1s.x, zp1 = p1s.y;
                // Wide-load the whole 16-byte window this (key,g) touches: the four
                // byte gathers per (row, cc) below sit at byte offsets
                // {16g+t4, 16g+8+t4} and {t4+4}, so one 128-bit fetch per key row
                // replaces 8 scalar LDG.U8 (4x fewer L1/L2 wavefronts; values are
                // the identical nibbles, so results stay bit-identical).
                uint4 kp0 = *(const uint4 *)(kr0 + 16 * g);
                uint4 kp1 = *(const uint4 *)(kr1 + 16 * g);
                uint32_t w0[4] = {kp0.x, kp0.y, kp0.z, kp0.w};
                uint32_t w1[4] = {kp1.x, kp1.y, kp1.z, kp1.w};
                const uint8_t *b0 = (const uint8_t *)w0;       // b0[i] = kr0[16g+i]
                const uint8_t *b1 = (const uint8_t *)w1;
                #pragma unroll
                for (int cc = 0; cc < 2; cc++) {
                    int c = 2 * g + cc;
                    const uint32_t *a = &sm[OFF_QA + (c * 32 + lane) * 4];
                    int bi = (8 * c + t4) & 15;                 // window-relative row-byte index
                    uint8_t a0 = b0[bi], a8 = b0[bi + 4];
                    uint8_t e0 = b1[bi], e8 = b1[bi + 4];
                    uint32_t bk00 = tq_pack_bf16(((float)(a0 & 15) - zp0) * sc0, ((float)(a0 >> 4) - zp0) * sc0);
                    uint32_t bk08 = tq_pack_bf16(((float)(a8 & 15) - zp0) * sc0, ((float)(a8 >> 4) - zp0) * sc0);
                    uint32_t bk10 = tq_pack_bf16(((float)(e0 & 15) - zp1) * sc1, ((float)(e0 >> 4) - zp1) * sc1);
                    uint32_t bk18 = tq_pack_bf16(((float)(e8 & 15) - zp1) * sc1, ((float)(e8 >> 4) - zp1) * sc1);
                    tq_mma_bf16_m16n8k16(s0, a, bk00, bk08);
                    tq_mma_bf16_m16n8k16(s1, a, bk10, bk18);
                    if constexpr (QT == 2) {
                        const uint32_t *a2 = &sm[OFF_QA2 + (c * 32 + lane) * 4];
                        tq_mma_bf16_m16n8k16(s2, a2, bk00, bk08);
                        tq_mma_bf16_m16n8k16(s3, a2, bk10, bk18);
                    }
                }
            }
        } else {
            const float *kr0 = kcb + (size_t)min(p0, maxpos) * cstride;
            const float *kr1 = kcb + (size_t)min(p1, maxpos) * cstride;
            #pragma unroll
            for (int c = 0; c < 16; c++) {
                const uint32_t *a = &sm[OFF_QA + (c * 32 + lane) * 4];
                int d0 = 16 * c + 2 * t4;
                float2 x0 = *(const float2 *)(kr0 + d0), x1 = *(const float2 *)(kr0 + d0 + 8);
                tq_mma_bf16_m16n8k16(s0, a, tq_pack_bf16(x0.x, x0.y), tq_pack_bf16(x1.x, x1.y));
                float2 y0 = *(const float2 *)(kr1 + d0), y1 = *(const float2 *)(kr1 + d0 + 8);
                tq_mma_bf16_m16n8k16(s1, a, tq_pack_bf16(y0.x, y0.y), tq_pack_bf16(y1.x, y1.y));
            }
        }
        // scale + per-row causal staircase mask (token attends keys [0..its pos])
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int qrow = (j < 2) ? gr : (gr + 8);
            int qpos = (qrow < nvalid) ? positions[qtile_base + qrow] : -1;
            int k0 = tp + 2 * t4 + (j & 1), k1 = tp + 8 + 2 * t4 + (j & 1);
            s0[j] = (k0 <= qpos) ? s0[j] * rsc : TQ_AMM_NINF;
            s1[j] = (k1 <= qpos) ? s1[j] * rsc : TQ_AMM_NINF;
        }
        if constexpr (QT == 2) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                int qrow = ((j < 2) ? gr : (gr + 8)) + 16;
                int qpos = (qrow < nvalid) ? positions[qtile_base + qrow] : -1;
                int k0 = tp + 2 * t4 + (j & 1), k1 = tp + 8 + 2 * t4 + (j & 1);
                s2[j] = (k0 <= qpos) ? s2[j] * rsc : TQ_AMM_NINF;
                s3[j] = (k1 <= qpos) ? s3[j] * rsc : TQ_AMM_NINF;
            }
        }
        float lm_lo = fmaxf(fmaxf(s0[0], s0[1]), fmaxf(s1[0], s1[1]));
        float lm_hi = fmaxf(fmaxf(s0[2], s0[3]), fmaxf(s1[2], s1[3]));
        float lm2_lo, lm2_hi;
        if constexpr (QT == 2) {
            lm2_lo = fmaxf(fmaxf(s2[0], s2[1]), fmaxf(s3[0], s3[1]));
            lm2_hi = fmaxf(fmaxf(s2[2], s2[3]), fmaxf(s3[2], s3[3]));
        }
        #pragma unroll
        for (int o = 1; o <= 2; o <<= 1) {
            lm_lo = fmaxf(lm_lo, __shfl_xor_sync(0xffffffffu, lm_lo, o));
            lm_hi = fmaxf(lm_hi, __shfl_xor_sync(0xffffffffu, lm_hi, o));
            if constexpr (QT == 2) {
                lm2_lo = fmaxf(lm2_lo, __shfl_xor_sync(0xffffffffu, lm2_lo, o));
                lm2_hi = fmaxf(lm2_hi, __shfl_xor_sync(0xffffffffu, lm2_hi, o));
            }
        }
        if (t4 == 0) {
            wmax[warp * QROWS + gr] = lm_lo; wmax[warp * QROWS + 8 + gr] = lm_hi;
            if constexpr (QT == 2) {
                wmax[warp * QROWS + 16 + gr] = lm2_lo; wmax[warp * QROWS + 24 + gr] = lm2_hi;
            }
        }
        __syncthreads();
        if (warp == 0 && lane < QROWS) {
            float m_old = mst[lane], m_new = m_old;
            for (int w = 0; w < 8; w++) m_new = fmaxf(m_new, wmax[w * QROWS + lane]);
            alph[lane] = expf(m_old - m_new); mst[lane] = m_new;
        }
        __syncthreads();
        float mn_lo = mst[gr], mn_hi = mst[8 + gr];
        float p00 = expf(s0[0] - mn_lo), p01 = expf(s0[1] - mn_lo), p02 = expf(s0[2] - mn_hi), p03 = expf(s0[3] - mn_hi);
        float p10 = expf(s1[0] - mn_lo), p11 = expf(s1[1] - mn_lo), p12 = expf(s1[2] - mn_hi), p13 = expf(s1[3] - mn_hi);
        float ls_lo = p00 + p01 + p10 + p11, ls_hi = p02 + p03 + p12 + p13;
        float p20 = 0.f, p21 = 0.f, p22 = 0.f, p23 = 0.f, p30 = 0.f, p31 = 0.f, p32 = 0.f, p33 = 0.f;
        float ls2_lo = 0.f, ls2_hi = 0.f;
        if constexpr (QT == 2) {
            float mn2_lo = mst[16 + gr], mn2_hi = mst[24 + gr];
            p20 = expf(s2[0] - mn2_lo); p21 = expf(s2[1] - mn2_lo);
            p22 = expf(s2[2] - mn2_hi); p23 = expf(s2[3] - mn2_hi);
            p30 = expf(s3[0] - mn2_lo); p31 = expf(s3[1] - mn2_lo);
            p32 = expf(s3[2] - mn2_hi); p33 = expf(s3[3] - mn2_hi);
            ls2_lo = p20 + p21 + p30 + p31; ls2_hi = p22 + p23 + p32 + p33;
        }
        #pragma unroll
        for (int o = 1; o <= 2; o <<= 1) {
            ls_lo += __shfl_xor_sync(0xffffffffu, ls_lo, o);
            ls_hi += __shfl_xor_sync(0xffffffffu, ls_hi, o);
            if constexpr (QT == 2) {
                ls2_lo += __shfl_xor_sync(0xffffffffu, ls2_lo, o);
                ls2_hi += __shfl_xor_sync(0xffffffffu, ls2_hi, o);
            }
        }
        if (t4 == 0) {
            wsum[warp * QROWS + gr] = ls_lo; wsum[warp * QROWS + 8 + gr] = ls_hi;
            if constexpr (QT == 2) {
                wsum[warp * QROWS + 16 + gr] = ls2_lo; wsum[warp * QROWS + 24 + gr] = ls2_hi;
            }
        }
        if (MODE != 1 && !VQ4) {                          // fold per-key V scale into p (denom keeps unscaled)
            int p0c = t0 + warp * 16 + 2 * t4, p1c = p0c + 8;
            float v00 = vscb_it[TQ_PROW(p0c) * nkv];
            float v01 = vscb_it[TQ_PROW(p0c + 1) * nkv];
            float v10 = vscb_it[TQ_PROW(p1c) * nkv];
            float v11 = vscb_it[TQ_PROW(p1c + 1) * nkv];
            p00 *= v00; p01 *= v01; p02 *= v00; p03 *= v01;
            p10 *= v10; p11 *= v11; p12 *= v10; p13 *= v11;
            if constexpr (QT == 2) {
                p20 *= v00; p21 *= v01; p22 *= v00; p23 *= v01;
                p30 *= v10; p31 *= v11; p32 *= v10; p33 *= v11;
            }
        }
        uint32_t *pst = &sm[OFF_PST + (warp * 32 + lane) * 4];
        pst[0] = tq_pack_bf16(p00, p01); pst[1] = tq_pack_bf16(p02, p03);
        pst[2] = tq_pack_bf16(p10, p11); pst[3] = tq_pack_bf16(p12, p13);
        if constexpr (QT == 2) {
            uint32_t *pst2 = &sm[OFF_PST2 + (warp * 32 + lane) * 4];
            pst2[0] = tq_pack_bf16(p20, p21); pst2[1] = tq_pack_bf16(p22, p23);
            pst2[2] = tq_pack_bf16(p30, p31); pst2[3] = tq_pack_bf16(p32, p33);
        }
        __syncthreads();
        if (warp == 0 && lane < QROWS) {
            float l = lst[lane] * alph[lane];
            for (int w = 0; w < 8; w++) l += wsum[w * QROWS + lane];
            lst[lane] = l;
        }
        float a_lo = alph[gr], a_hi = alph[8 + gr];
        #pragma unroll
        for (int f = 0; f < 4; f++) { oacc[f][0] *= a_lo; oacc[f][1] *= a_lo; oacc[f][2] *= a_hi; oacc[f][3] *= a_hi; }
        if constexpr (QT == 2) {
            float a2_lo = alph[16 + gr], a2_hi = alph[24 + gr];
            #pragma unroll
            for (int f = 0; f < 4; f++) { oacc2[f][0] *= a2_lo; oacc2[f][1] *= a2_lo; oacc2[f][2] *= a2_hi; oacc2[f][3] *= a2_hi; }
        }
        for (int tt = 0; tt < 8; tt++) {
            const uint32_t *pa = &sm[OFF_PST + (tt * 32 + lane) * 4];
            uint32_t a[4] = {pa[0], pa[1], pa[2], pa[3]};
            uint32_t a2[4];
            if constexpr (QT == 2) {
                const uint32_t *pa2 = &sm[OFF_PST2 + (tt * 32 + lane) * 4];
                a2[0] = pa2[0]; a2[1] = pa2[1]; a2[2] = pa2[2]; a2[3] = pa2[3];
            }
            if (T4 && t4blk && t4nenc >= 128) {
                // trellis V: states (bits >> 4f) & 0xFFF of the 3 stream bytes at ch/2 of a key row
                const int r0 = t0 + tt * 16 + 2 * t4;
                const size_t q0 = TQ_PROW(r0), q1 = TQ_PROW(r0 + 1), q8 = TQ_PROW(r0 + 8), q9 = TQ_PROW(r0 + 9);
                const int cb = 16 * warp + 2 * gr;
                const uint32_t z0 = tq_t4_side(t4p.vside, q0, nkv, kv_head), z1 = tq_t4_side(t4p.vside, q1, nkv, kv_head);
                const uint32_t z8 = tq_t4_side(t4p.vside, q8, nkv, kv_head), z9 = tq_t4_side(t4p.vside, q9, nkv, kv_head);
                const uint32_t c0 = tq_t4_vbits(vcb8 + q0 * cstride4, z0, cb), c1 = tq_t4_vbits(vcb8 + q1 * cstride4, z1, cb);
                const uint32_t c8 = tq_t4_vbits(vcb8 + q8 * cstride4, z8, cb), c9 = tq_t4_vbits(vcb8 + q9 * cstride4, z9, cb);
                const float v0 = tq_t4_scale(z0), v1 = tq_t4_scale(z1), v8 = tq_t4_scale(z8), v9 = tq_t4_scale(z9);
                #pragma unroll
                for (int f = 0; f < 4; f++) {
                    uint32_t b0 = tq_pack_bf16(tq_t4_val(c0, f, t4tab, v0), tq_t4_val(c1, f, t4tab, v1));
                    uint32_t b1 = tq_pack_bf16(tq_t4_val(c8, f, t4tab, v8), tq_t4_val(c9, f, t4tab, v9));
                    tq_mma_bf16_m16n8k16(oacc[f], a, b0, b1);
                    if constexpr (QT == 2) tq_mma_bf16_m16n8k16(oacc2[f], a2, b0, b1);
                }
                continue;
            }
            if (T4 && t4blk) {
                // the wave's own super-tile: trellis rows below t4nenc, the in-flight rows exact fp16 from the staging
                const int r0 = t0 + tt * 16 + 2 * t4;
                const int cb = 16 * warp + 2 * gr;
                float v[4][4];
                #pragma unroll
                for (int rr = 0; rr < 4; rr++) {
                    const int rp = min(r0 + (rr >> 1) * 8 + (rr & 1), maxpos);
                    if (rp - t0 < t4nenc) {
                        const size_t q = TQ_PROW(rp);
                        const uint32_t z = tq_t4_side(t4p.vside, q, nkv, kv_head);
                        const uint32_t c = tq_t4_vbits(vcb8 + q * cstride4, z, cb);
                        const float vs = tq_t4_scale(z);
                        #pragma unroll
                        for (int f = 0; f < 4; f++) v[rr][f] = tq_t4_val(c, f, t4tab, vs);
                    } else {
                        const uint2 w = *(const uint2 *)(t4f16 + (size_t)(rp & (page - 1)) * nkv * 512 + 256 + 2 * cb);
                        v[rr][0] = __half2float(__ushort_as_half((unsigned short)(w.x & 0xFFFFu)));
                        v[rr][1] = __half2float(__ushort_as_half((unsigned short)(w.x >> 16)));
                        v[rr][2] = __half2float(__ushort_as_half((unsigned short)(w.y & 0xFFFFu)));
                        v[rr][3] = __half2float(__ushort_as_half((unsigned short)(w.y >> 16)));
                    }
                }
                #pragma unroll
                for (int f = 0; f < 4; f++) {
                    uint32_t b0 = tq_pack_bf16(v[0][f], v[1][f]);
                    uint32_t b1 = tq_pack_bf16(v[2][f], v[3][f]);
                    tq_mma_bf16_m16n8k16(oacc[f], a, b0, b1);
                    if constexpr (QT == 2) tq_mma_bf16_m16n8k16(oacc2[f], a2, b0, b1);
                }
                continue;
            }
            if constexpr (VQ4) {
                // int4 V: the warp's 32 output channels are one (scale, zp) group (g = warp). MMA f's
                // column gr carries channel warp*32 + 4*gr + f, so a thread's four columns are two
                // adjacent code bytes of a key row: one 16-bit load per row.
                const int r0 = t0 + tt * 16 + 2 * t4;
                const size_t q0 = TQ_PROW(r0), q1 = TQ_PROW(r0 + 1), q8 = TQ_PROW(r0 + 8), q9 = TQ_PROW(r0 + 9);
                const size_t cb = (size_t)(16 * warp + 2 * gr);
                const uint32_t c0 = *(const uint16_t *)(vcb8_it + q0 * cstride4 + cb);
                const uint32_t c1 = *(const uint16_t *)(vcb8_it + q1 * cstride4 + cb);
                const uint32_t c8 = *(const uint16_t *)(vcb8_it + q8 * cstride4 + cb);
                const uint32_t c9 = *(const uint16_t *)(vcb8_it + q9 * cstride4 + cb);
                const float2 z0 = q4s_pair(q4s_adv(vscb_it, q0 * szstride), 2 * warp);
                const float2 z1 = q4s_pair(q4s_adv(vscb_it, q1 * szstride), 2 * warp);
                const float2 z8 = q4s_pair(q4s_adv(vscb_it, q8 * szstride), 2 * warp);
                const float2 z9 = q4s_pair(q4s_adv(vscb_it, q9 * szstride), 2 * warp);
                #pragma unroll
                for (int f = 0; f < 4; f++) {
                    const int sh = 4 * f;
                    uint32_t b0 = tq_pack_bf16(((float)((c0 >> sh) & 15u) - z0.y) * z0.x,
                                               ((float)((c1 >> sh) & 15u) - z1.y) * z1.x);
                    uint32_t b1 = tq_pack_bf16(((float)((c8 >> sh) & 15u) - z8.y) * z8.x,
                                               ((float)((c9 >> sh) & 15u) - z9.y) * z9.x);
                    tq_mma_bf16_m16n8k16(oacc[f], a, b0, b1);
                    if constexpr (QT == 2) tq_mma_bf16_m16n8k16(oacc2[f], a2, b0, b1);
                }
                continue;
            }
            #pragma unroll
            for (int f = 0; f < 4; f++) {
                int col = warp * 32 + f * 8 + gr;
                int r0 = t0 + tt * 16 + 2 * t4;
                uint32_t b0, b1;
                if (MODE != 1) {                               // E4M3-V dequant (scale already folded into p)
                    b0 = tq_pack_bf16(tq_e4m3_dec_fast(vcb8[TQ_PROW(r0) * cstride + col]),
                                      tq_e4m3_dec_fast(vcb8[TQ_PROW(r0 + 1) * cstride + col]));
                    b1 = tq_pack_bf16(tq_e4m3_dec_fast(vcb8[TQ_PROW(r0 + 8) * cstride + col]),
                                      tq_e4m3_dec_fast(vcb8[TQ_PROW(r0 + 9) * cstride + col]));
                } else {
                    b0 = tq_pack_bf16(vcb[(size_t)min(r0, maxpos) * cstride + col],
                                      vcb[(size_t)min(r0 + 1, maxpos) * cstride + col]);
                    b1 = tq_pack_bf16(vcb[(size_t)min(r0 + 8, maxpos) * cstride + col],
                                      vcb[(size_t)min(r0 + 9, maxpos) * cstride + col]);
                }
                tq_mma_bf16_m16n8k16(oacc[f], a, b0, b1);
                if constexpr (QT == 2) tq_mma_bf16_m16n8k16(oacc2[f], a2, b0, b1);
            }
        }
#undef TQ_PROW
        __syncthreads();
    }
    // ---- key-split partial: unnormalized (m, l, O) for the merge kernel ----
    if (gridDim.z > 1) {
        constexpr int PART_F = 2 * QROWS + QROWS * 256;
        float *ps = part + (size_t)((head * gridDim.y + qtile) * gridDim.z + blockIdx.z)
                               * PART_F;
        if (tid < QROWS) { ps[tid] = mst[tid]; ps[QROWS + tid] = lst[tid]; }
        float *po = ps + 2 * QROWS;
        #pragma unroll
        for (int f = 0; f < 4; f++) {
            int col = warp * 32 + f * 8 + 2 * t4;
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                const int ch = VQ4 ? warp * 32 + 4 * (2 * t4 + j) + f : col + j;
                po[(size_t)gr * 256 + ch] = oacc[f][j];
                po[(size_t)(8 + gr) * 256 + ch] = oacc[f][2 + j];
                if constexpr (QT == 2) {
                    po[(size_t)(16 + gr) * 256 + ch] = oacc2[f][j];
                    po[(size_t)(24 + gr) * 256 + ch] = oacc2[f][2 + j];
                }
            }
        }
        return;
    }
    // ---- epilogue: out = (O / l) * sigmoid(gate) ----
    float inv_lo = 1.0f / lst[gr], inv_hi = 1.0f / lst[8 + gr];
    float inv2_lo, inv2_hi;
    if constexpr (QT == 2) { inv2_lo = 1.0f / lst[16 + gr]; inv2_hi = 1.0f / lst[24 + gr]; }
    if constexpr (VQ4) {
        // rotated rows of one 16-row tile -> smem (the main loop is done with it; lst lies above
        // the 4096 words), one FWHT-256 per row by a warp, then the gate
        float *ob = (float *)sm;
        #pragma unroll
        for (int tile = 0; tile < QT; tile++) {
            __syncthreads();
            #pragma unroll
            for (int f = 0; f < 4; f++) {
                #pragma unroll
                for (int j = 0; j < 2; j++) {
                    const int ch = warp * 32 + 4 * (2 * t4 + j) + f;
                    if (tile == 0) {
                        ob[gr * 256 + ch] = oacc[f][j] * inv_lo;
                        ob[(8 + gr) * 256 + ch] = oacc[f][2 + j] * inv_hi;
                    } else if constexpr (QT == 2) {
                        ob[gr * 256 + ch] = oacc2[f][j] * inv2_lo;
                        ob[(8 + gr) * 256 + ch] = oacc2[f][2 + j] * inv2_hi;
                    }
                }
            }
            __syncthreads();
            for (int rr = warp; rr < 16; rr += 8) {
                const int row = 16 * tile + rr;
                if (row >= nvalid) break;
                float x[8];
                #pragma unroll
                for (int e = 0; e < 8; e++) x[e] = ob[rr * 256 + lane + 32 * e];
                tq_fwht256_warp(x, lane);
                const float *q_proj = q_proj_base + (size_t)(qtile_base + row) * q_m;
                #pragma unroll
                for (int e = 0; e < 8; e++) {
                    const int d = lane + 32 * e;
                    float gate = q_proj[head * (2 * hd) + hd + d];
                    out[(size_t)(qtile_base + row) * attn_m + head * hd + d] = x[e] * (1.0f / (1.0f + expf(-gate)));
                }
            }
        }
        return;
    }
    #pragma unroll
    for (int f = 0; f < 4; f++) {
        int col = warp * 32 + f * 8 + 2 * t4;
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            if (gr < nvalid) {
                const float *q_proj = q_proj_base + (size_t)(qtile_base + gr) * q_m;
                float gate = q_proj[head * (2 * hd) + hd + col + j];
                out[(size_t)(qtile_base + gr) * attn_m + head * hd + col + j] =
                    oacc[f][j] * inv_lo * (1.0f / (1.0f + expf(-gate)));
            }
            if (8 + gr < nvalid) {
                const float *q_proj = q_proj_base + (size_t)(qtile_base + 8 + gr) * q_m;
                float gate = q_proj[head * (2 * hd) + hd + col + j];
                out[(size_t)(qtile_base + 8 + gr) * attn_m + head * hd + col + j] =
                    oacc[f][2 + j] * inv_hi * (1.0f / (1.0f + expf(-gate)));
            }
            if constexpr (QT == 2) {
                if (16 + gr < nvalid) {
                    const float *q_proj = q_proj_base + (size_t)(qtile_base + 16 + gr) * q_m;
                    float gate = q_proj[head * (2 * hd) + hd + col + j];
                    out[(size_t)(qtile_base + 16 + gr) * attn_m + head * hd + col + j] =
                        oacc2[f][j] * inv2_lo * (1.0f / (1.0f + expf(-gate)));
                }
                if (24 + gr < nvalid) {
                    const float *q_proj = q_proj_base + (size_t)(qtile_base + 24 + gr) * q_m;
                    float gate = q_proj[head * (2 * hd) + hd + col + j];
                    out[(size_t)(qtile_base + 24 + gr) * attn_m + head * hd + col + j] =
                        oacc2[f][2 + j] * inv2_hi * (1.0f / (1.0f + expf(-gate)));
                }
            }
        }
    }
}

// Merge phase for the key-split wide-prefill attention: grid (nh, nqt), 256
// threads; per query row r combine the S chunk partials (m* = max, l/O rescaled
// by exp(m_c - m*)) and run the same gate epilogue the single-pass kernel uses.
// Fully-masked chunks wrote m = TQ_AMM_NINF, l = 0, O = 0 and vanish under the
// exp underflow (row 0's chunk 0 always has at least key 0, so l* > 0).
// VQ4: the partials are in the rotated V space; rows are staged 16 at a time and each gets one
// FWHT-256 (a warp per row) before the gate.
template <int QT = 1, int VQ4 = 0>
__global__ void k_tq_wide_attn_mma_merge(
    float *out, const float *q_proj_base, int N, int nh, int hd,
    int q_m, int attn_m, int S, const float *part) {
    constexpr int QROWS = 16 * QT;
    constexpr int PART_F = 2 * QROWS + QROWS * 256;
    __shared__ float ob[VQ4 ? 16 * 256 : 1];
    int head = blockIdx.x, qtile = blockIdx.y, qtile_base = qtile * QROWS;
    int tid = threadIdx.x;
    int nvalid = N - qtile_base; if (nvalid > QROWS) nvalid = QROWS;
    const float *ps0 = part + (size_t)((head * gridDim.y + qtile) * S) * PART_F;
    if constexpr (VQ4) {
        const int warp = tid >> 5, lane = tid & 31;
        for (int r0 = 0; r0 < nvalid; r0 += 16) {
            const int nr = (nvalid - r0 < 16) ? nvalid - r0 : 16;
            for (int rr = 0; rr < nr; rr++) {
                const int r = r0 + rr;
                float m_star = TQ_AMM_NINF;
                for (int c = 0; c < S; c++)
                    m_star = fmaxf(m_star, ps0[(size_t)c * PART_F + r]);
                float l = 0.0f, o = 0.0f;
                for (int c = 0; c < S; c++) {
                    const float *ps = ps0 + (size_t)c * PART_F;
                    float w = expf(ps[r] - m_star);
                    l += w * ps[QROWS + r];
                    o += w * ps[2 * QROWS + (size_t)r * 256 + tid];
                }
                ob[rr * 256 + tid] = o / l;
            }
            __syncthreads();
            for (int rr = warp; rr < nr; rr += 8) {
                float x[8];
                #pragma unroll
                for (int e = 0; e < 8; e++) x[e] = ob[rr * 256 + lane + 32 * e];
                tq_fwht256_warp(x, lane);
                const float *q_proj = q_proj_base + (size_t)(qtile_base + r0 + rr) * q_m;
                #pragma unroll
                for (int e = 0; e < 8; e++) {
                    const int d = lane + 32 * e;
                    float gate = q_proj[head * (2 * hd) + hd + d];
                    out[(size_t)(qtile_base + r0 + rr) * attn_m + head * hd + d] = x[e] * (1.0f / (1.0f + expf(-gate)));
                }
            }
            __syncthreads();
        }
        return;
    }
    for (int r = 0; r < nvalid; r++) {
        float m_star = TQ_AMM_NINF;
        for (int c = 0; c < S; c++)
            m_star = fmaxf(m_star, ps0[(size_t)c * PART_F + r]);
        float l = 0.0f, o = 0.0f;
        for (int c = 0; c < S; c++) {
            const float *ps = ps0 + (size_t)c * PART_F;
            float w = expf(ps[r] - m_star);
            l += w * ps[QROWS + r];
            o += w * ps[2 * QROWS + (size_t)r * 256 + tid];
        }
        const float *q_proj = q_proj_base + (size_t)(qtile_base + r) * q_m;
        float gate = q_proj[head * (2 * hd) + hd + tid];
        out[(size_t)(qtile_base + r) * attn_m + head * hd + tid] =
            (o / l) * (1.0f / (1.0f + expf(-gate)));
    }
}

// ===== Session 11: GQA-grouped tensor-core prefill attention on the trellis pool (TQ_PATTN_GQA) =====
// k_tq_wide_attn_mma<4, 2, 1, 1> decodes every trellis K / V element in the warp that consumes it: a block holds ONE
// q head x 32 query rows, so each 12-bit table lookup (LDS.U16 at ~3.4 wavefronts from random bank conflicts) feeds
// 32 query rows, and with nh / nkv = 6 heads per kv head every key is decoded 6 x 4 = 24 times per wave and layer.
// ncu at 128K (notes/session11-pattn.md): tensor pipe 56 % active, L1 79 %, 865 M LDS wavefronts per launch.
// This kernel decodes each K / V tile ONCE per block into fp16 shared-memory tiles that serve all six q heads of the
// kv head: a block is 128 (head, position) rows = 8 m16 tiles (warp w = m-tile rb * 8 + w; m-tile mt = position tile
// mt / G, head kv_head * G + mt % G), every warp keeps its 16 rows' Q A-fragments (16 k16 chunks) and the fp32 O
// accumulator (16 x 256) in registers (FA-2 structure, online softmax per 32-key tile in registers, no block-wide
// reductions), the K and V tiles (32 keys x 256 dims fp16, 528-byte rows: ldmatrix conflict-free) are double-buffered
// and dequantized by all 256 threads from cp.async-staged codes (each thread stages the 16 code bytes + side word it
// decodes itself, so no barrier sits between the copy and the decode; one __syncthreads per tile). Numerics follow
// the fp16 third pass of the decode kernel (TQ_T4_ATTN_F16, gated in session 10): Q, K, P, V are fp16 MMA operands,
// K and V the table's fp16 values themselves, a key's scale multiplies its score column, a value row's scale is
// folded into P; fp32 accumulation and fp32 online softmax. NOT bitwise vs the bf16 kernel (operand precision,
// 32-key softmax steps, 7-way key split): gated by the e2e yardstick + needle. Grid (nkv, row blocks, S); every block
// writes an (m, l, O) partial, k_tq_pattn_merge normalizes, un-rotates (FWHT) and gates. Default off (TQ_PATTN_GQA=1).
#define TQ_PA_TK     32                                 // keys per tile
#define TQ_PA_LD     264                                // fp16 tile row stride in halfs (512 B + 16 B pad)
#define TQ_PA_LDB    (TQ_PA_LD * 2)                     // ... in bytes (528)
#define TQ_PA_TILE_B (TQ_PA_TK * TQ_PA_LDB)             // one K or V tile: 16 896 B
#define TQ_PA_CROW   144                                // staged code row: 128 bytes + 16 pad (8 keys of a quarter-warp on 8 bank groups)
#define TQ_PA_CODE_B (TQ_PA_TK * TQ_PA_CROW)            // staged codes of one tile: 4 608 B
// dynamic smem layout (bytes): KT | VT | CK | CV | SK[32] | SV[32] | table | ksc[2][32] | vsc[2][32]
// Single fp16 K and V tiles: phase A of tile i runs Q.K^T on KT = K_i while decoding V_i into VT, phase C runs the
// softmax + P.V on VT while decoding K_(i+1) into KT (one __syncthreads per phase). The code buffers are single too:
// K_(i+1)'s codes are copied during phase A(i) (consumed in C(i)), V_(i+1)'s during phase C(i) (consumed in A(i+1)).
// 49.5 KB per block, so the trellis encoder of the wave's columns (37 KB, side stream) co-resides on the SM as it did
// with the old kernel (at 97 KB it ran only after the attention and pushed the merge + GEMMs by ~7 ms per wave).
#define TQ_PA_OFF_KT   0
#define TQ_PA_OFF_VT   (TQ_PA_OFF_KT + TQ_PA_TILE_B)
#define TQ_PA_OFF_CK   (TQ_PA_OFF_VT + TQ_PA_TILE_B)
#define TQ_PA_OFF_CV   (TQ_PA_OFF_CK + TQ_PA_CODE_B)
#define TQ_PA_OFF_SK   (TQ_PA_OFF_CV + TQ_PA_CODE_B)
#define TQ_PA_OFF_SV   (TQ_PA_OFF_SK + TQ_PA_TK * 4)
#define TQ_PA_OFF_TAB  (TQ_PA_OFF_SV + TQ_PA_TK * 4)
#define TQ_PA_OFF_KSC  (TQ_PA_OFF_TAB + 8192)
#define TQ_PA_OFF_VSC  (TQ_PA_OFF_KSC + 2 * TQ_PA_TK * 4)
#define TQ_PA_OFF_QH   (TQ_PA_OFF_VSC + 2 * TQ_PA_TK * 4)     // QH: Q A-fragments of chunks 8..15, [8 warps][8][32][4]
#define TQ_PA_SMEM0    TQ_PA_OFF_QH                            // 50 688 B (QH = 0)
#define TQ_PA_SMEM1    (TQ_PA_OFF_QH + 8 * 8 * 32 * 16)        // 83 456 B (QH = 1)
#define TQ_PA_ROWS     128
#define TQ_PA_PART_F   (2 * TQ_PA_ROWS + TQ_PA_ROWS * 256)    // floats per (kv head, row block, split) partial

static __device__ __forceinline__ void tq_pa_ldsm_x4(uint32_t r[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
static __device__ __forceinline__ void tq_pa_ldsm_x4_trans(uint32_t r[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
static __device__ __forceinline__ void tq_pa_mma_f16(float *c, const uint32_t *a, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}
// fp16-accumulate variant (AQ / AP): C and D are two f16x2 registers, (row gr, cols 2 t4, 2 t4 + 1) and (row gr + 8, ...)
static __device__ __forceinline__ void tq_pa_mma_f16acc(uint32_t *c, const uint32_t *a, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1},{%2,%3,%4,%5},{%6,%7},{%8,%9};"
        : "+r"(c[0]), "+r"(c[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(c[0]), "r"(c[1]));
}
// fold an f16x2 accumulator pair into the fp32 C fragment (c0 += lo(d0), c1 += hi(d0), c2 += lo(d1), c3 += hi(d1))
// (asm volatile: nvcc otherwise sinks every fold of a phase below the phase's last HMMA, ~100 dependent ALU
//  instructions during which the tensor pipe idles; the callers issue a group's fold one group late, behind the next
//  group's HMMAs, so the fold never waits for an HMMA result)
static __device__ __forceinline__ void tq_pa_fold16(float *c, const uint32_t *d) {
    asm volatile("{\n .reg .b16 h0, h1, h2, h3;\n .reg .f32 f0, f1, f2, f3;\n"
                 " mov.b32 {h0, h1}, %4;\n mov.b32 {h2, h3}, %5;\n"
                 " cvt.f32.f16 f0, h0;\n cvt.f32.f16 f1, h1;\n cvt.f32.f16 f2, h2;\n cvt.f32.f16 f3, h3;\n"
                 " add.f32 %0, %0, f0;\n add.f32 %1, %1, f1;\n add.f32 %2, %2, f2;\n add.f32 %3, %3, f3;\n}"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(d[0]), "r"(d[1]));
}
static __device__ __forceinline__ float tq_pa_ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
static __device__ __forceinline__ uint32_t tq_pa_pack_f16(float lo, float hi) {
    uint32_t d;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
    return d;
}
// Decode 32 trellis values (dims 32 g .. 32 g + 31 of one row) from the row's 16 code bytes at byte 16 g (cw) and the
// code byte before them (prevb; the prefix byte at g = 0) into raw fp16 table values, 8 per 16-byte store.
static __device__ __forceinline__ void tq_pa_dequant32(const uint4 cw, uint32_t prevb, const uint16_t *tab, uint4 *dst) {
    const uint32_t w[4] = {cw.x, cw.y, cw.z, cw.w};
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        const uint32_t lowb = (k == 0) ? prevb : (w[k - 1] >> 24);
        const uint32_t x = (w[k] << 8) | lowb, y = w[k] >> 24;
        uint32_t v[4];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const uint32_t pr = __funnelshift_r(x, y, 8 * j);
            v[j] = (uint32_t)tab[pr & 0xFFFu] | ((uint32_t)tab[(pr >> 4) & 0xFFFu] << 16);
        }
        dst[k] = make_uint4(v[0], v[1], v[2], v[3]);
    }
}

// G = q heads per kv head; QH = 1: the Q A-fragments of chunks 8..15 live in smem (32 KB, one LDS.128 per chunk and
// tile) instead of registers (frees 32 registers); AQ = 1: Q.K^T accumulates in fp16 over TQ_PA_FOLD chunks and folds
// into the fp32 scores; AP = 1: P.V accumulates the tile's 32 keys in fp16 per 8-n-tile group and folds into the
// fp32 O. The fp16 accumulators run the tensor pipe at twice the fp32 rate (515 vs 258 TFLOPS on GB202).
#ifndef TQ_PA_FOLD
#define TQ_PA_FOLD 4
#endif
template <int G, int QH, int AQ, int AP>
__global__ void __launch_bounds__(256, 1) k_tq_pattn_t4(
    const float *q_proj_base, const uint16_t *q_norm_w,
    const uint8_t *k4, const uint8_t *v8,
    const int *positions, int slot, const int *block_table, int max_blocks, int page, int page_log,
    int N, int nkv, int q_m, float eps, float rope_theta,
    float *part, int S, const tq_t4_pool t4p) {
    extern __shared__ __align__(16) uint8_t pa_sm[];
    uint16_t *tab = (uint16_t *)(pa_sm + TQ_PA_OFF_TAB);
    float *ksc = (float *)(pa_sm + TQ_PA_OFF_KSC), *vsc = (float *)(pa_sm + TQ_PA_OFF_VSC);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gr = lane >> 2, t4 = lane & 3;
    const int kv_head = blockIdx.x, rb = blockIdx.y, z = blockIdx.z;
    const int nptile = (N + 15) >> 4;
    const int mt = rb * 8 + warp, ptile = mt / G, hl = mt - ptile * G;
    const int head = kv_head * G + hl;
    const bool wvalid = ptile < nptile;
    const int pidx_lo = ptile * 16 + gr, pidx_hi = pidx_lo + 8;
    const int qpos_lo = (wvalid && pidx_lo < N) ? positions[pidx_lo] : -1;
    const int qpos_hi = (wvalid && pidx_hi < N) ? positions[pidx_hi] : -1;
    // block key range: [0 .. max position of the block's rows], 32-key tiles, this split's slice
    int ptile_hi = (rb * 8 + 7) / G; if (ptile_hi > nptile - 1) ptile_hi = nptile - 1;
    int last = (ptile_hi + 1) * 16; if (last > N) last = N;
    const int maxpos = positions[last - 1];
    const int minpos = positions[(rb * 8 / G) * 16];
    const int nst = (maxpos + TQ_PA_TK) / TQ_PA_TK;
    const int per = (nst + S - 1) / S;
    const int lo = z * per, hi = (lo + per < nst) ? lo + per : nst;
    const int encl = t4p.encl[slot];
    const size_t cstride4 = (size_t)nkv * 128;
    const uint8_t *kcb = k4 + (size_t)kv_head * 128, *vcb = v8 + (size_t)kv_head * 128;
    const uint16_t *stg = t4p.f16s + ((size_t)slot * page * nkv + kv_head) * 512;   // + (pos & (page-1)) * nkv * 512
    const float rsc = rsqrtf(256.0f);

    // ---- Q prep (one warp per m-tile, warps 0-3 then 4-7 in the 33 KB tile area): norm, partial rope, FWHT,
    //      fp16 [16 rows][256] -> ldmatrix A-fragments (16 chunks x 4 registers)
    constexpr int NQR = QH ? 8 : 16;                    // resident Q chunks (the rest in smem, QH)
    uint32_t qa[NQR][4];
    uint32_t *qh = (uint32_t *)(pa_sm + TQ_PA_OFF_QH) + warp * (8 * 32 * 4);
    #pragma unroll 1
    for (int half = 0; half < 2; half++) {
        if ((warp >> 2) == half) {
            uint8_t *qs = pa_sm + TQ_PA_OFF_KT + (warp & 3) * (16 * TQ_PA_LDB);
            for (int r = 0; r < 16; r++) {
                const int pidx = ptile * 16 + r;
                float x[8];
                if (wvalid && pidx < N) {
                    const float *q = q_proj_base + (size_t)pidx * q_m + head * 512;
                    const int pos = positions[pidx];
                    float ss = 0.0f;
                    #pragma unroll
                    for (int e = 0; e < 8; e++) { x[e] = q[lane + 32 * e]; ss += x[e] * x[e]; }
                    #pragma unroll
                    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
                    const float nrm = rsqrtf(ss / 256.0f + eps);
                    #pragma unroll
                    for (int e = 0; e < 8; e++) x[e] = x[e] * nrm * (1.0f + tq_bf16_to_float(q_norm_w[lane + 32 * e]));
                    // partial rope over dims 0..63: pairs (lane, lane + 32) = (x[0], x[1])
                    const float freq = powf(rope_theta, -((float)(2 * lane) / 64.0f));
                    const float angle = (float)pos * freq, c = cosf(angle), s = sinf(angle);
                    const float x0 = x[0], x1 = x[1];
                    x[0] = x0 * c - x1 * s;
                    x[1] = x1 * c + x0 * s;
                    tq_fwht256_warp(x, lane);
                } else {
                    #pragma unroll
                    for (int e = 0; e < 8; e++) x[e] = 0.0f;
                }
                uint16_t *qrow = (uint16_t *)(qs + r * TQ_PA_LDB);
                #pragma unroll
                for (int e = 0; e < 8; e++) qrow[lane + 32 * e] = __half_as_ushort(__float2half_rn(x[e]));
            }
            __syncwarp();
            const uint32_t qbase = (uint32_t)__cvta_generic_to_shared(qs) + ((lane & 7) + 8 * ((lane >> 3) & 1)) * TQ_PA_LDB + (lane >> 4) * 16;
            #pragma unroll
            for (int c = 0; c < 16; c++) {
                if (c < NQR) tq_pa_ldsm_x4(qa[c], qbase + c * 32);
                else {
                    uint32_t t[4];
                    tq_pa_ldsm_x4(t, qbase + c * 32);
                    *(uint4 *)(qh + ((c - 8) * 32 + lane) * 4) = make_uint4(t[0], t[1], t[2], t[3]);
                }
            }
        }
        __syncthreads();
    }
    tq_t4_load_table(t4p.table, tab, tid);              // after the Q prep (the table shares no bytes with it, but
                                                        // the loop's first use is after the next barrier anyway)
    // ---- per-thread staging / dequant assignment: key kk = tid >> 3 of the tile, 32-dim chunk g = tid & 7
    // (warp g decodes the 32-dim chunk g of all 32 keys: a quarter-warp's 8 keys then store 16 B each to 8 distinct
    //  bank groups -- 528-byte rows -- and read their codes from 8 distinct groups -- 144-byte rows; the earlier
    //  (key = tid >> 3, chunk = tid & 7) mapping put a quarter-warp's 8 stores on 2 bank groups: 16 wavefronts per STS.128)
    const int kk = tid & 31, g = warp;
    const uint32_t sm_base = (uint32_t)__cvta_generic_to_shared(pa_sm);
    // copy the codes (this thread's 16 bytes) + side word (one per key) of tile `it`'s K or V into CK / CV
    auto stage = [&](int it, bool kpart) {
        const int t0 = it * TQ_PA_TK;
        const int phys = block_table[(size_t)slot * max_blocks + (t0 >> page_log)];
        const size_t row = (size_t)phys * page + (size_t)(t0 & (page - 1)) + kk;
        tq_cp_async16(sm_base + (kpart ? TQ_PA_OFF_CK : TQ_PA_OFF_CV) + kk * TQ_PA_CROW + g * 16, (kpart ? kcb : vcb) + row * cstride4 + g * 16);
        if (g == 0) tq_cp_async4(sm_base + (kpart ? TQ_PA_OFF_SK : TQ_PA_OFF_SV) + kk * 4, (kpart ? t4p.kside : t4p.vside) + (row * nkv + kv_head) * 4);
        asm volatile("cp.async.commit_group;\n");
    };
    // decode this thread's values 8 k .. 8 k + 7 of its 32-dim chunk (one code word) of tile `it`'s K or V from CK / CV
    // into KT / VT (k = 0 also writes the row's scale to ksc / vsc [it & 1]). Inline-PTX shared-memory accesses keep
    // the four pieces of a tile in program order relative to the HMMA stream (asm volatile), i.e. interleaved with
    // the MMAs instead of hoisted into one burst at the phase start (nvcc did that with plain loads: the LSU then
    // served ~100 lookups per warp before the first ldmatrix and the tensor pipe idled for the first ~1000 cycles).
    const uint32_t tab_s = sm_base + TQ_PA_OFF_TAB;
    auto piece = [&](int it, bool kpart, int k) {
        const int pos = it * TQ_PA_TK + kk;
        const uint32_t crow_s = sm_base + (kpart ? TQ_PA_OFF_CK : TQ_PA_OFF_CV) + kk * TQ_PA_CROW + g * 16;
        const uint32_t dst_s = sm_base + (kpart ? TQ_PA_OFF_KT : TQ_PA_OFF_VT) + kk * TQ_PA_LDB + g * 64 + k * 16;
        const uint32_t sc_s = sm_base + (kpart ? TQ_PA_OFF_KSC : TQ_PA_OFF_VSC) + ((it & 1) * TQ_PA_TK + kk) * 4;
        uint32_t v[4];
        if (pos < encl) {
            uint32_t w, lowb;
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(w) : "r"(crow_s + 4 * k));
            if (k == 0) {
                uint32_t sw;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(sw) : "r"(sm_base + (kpart ? TQ_PA_OFF_SK : TQ_PA_OFF_SV) + kk * 4));
                if (g == 0) {
                    lowb = sw & 0xFFu;
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(sc_s), "f"(tq_t4_scale(sw)));
                } else {
                    asm volatile("ld.shared.u8 %0, [%1];" : "=r"(lowb) : "r"(crow_s - 1));
                }
            } else {
                asm volatile("ld.shared.u8 %0, [%1];" : "=r"(lowb) : "r"(crow_s + 4 * k - 1));
            }
            const uint32_t x = (w << 8) | lowb, y = w >> 24;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                const uint32_t pr = __funnelshift_r(x, y, 8 * j);
                uint32_t lo16, hi16;
                asm volatile("ld.shared.u16 %0, [%1];" : "=r"(lo16) : "r"(tab_s + 2 * (pr & 0xFFFu)));
                asm volatile("ld.shared.u16 %0, [%1];" : "=r"(hi16) : "r"(tab_s + 2 * ((pr >> 4) & 0xFFFu)));
                v[j] = lo16 | (hi16 << 16);
            }
        } else {                                        // the wave's own column: exact fp16 from the slot's staging
            const uint4 t = *(const uint4 *)(stg + (size_t)(pos & (page - 1)) * nkv * 512 + (kpart ? 0 : 256) + g * 32 + k * 8);
            v[0] = t.x; v[1] = t.y; v[2] = t.z; v[3] = t.w;
            if (k == 0 && g == 0) asm volatile("st.shared.f32 [%0], %1;" :: "r"(sc_s), "f"(1.0f));
        }
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(dst_s), "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3]));
    };

    float o[32][4];
    #pragma unroll
    for (int j = 0; j < 32; j++) { o[j][0] = 0.0f; o[j][1] = 0.0f; o[j][2] = 0.0f; o[j][3] = 0.0f; }
    // the online softmax runs in log2 units: scores x (key scale x rsc x log2 e), ex2.approx for p and alpha; the
    // partial's m is in log2 units too (k_tq_pattn_merge uses exp2f)
    const float rsl = rsqrtf(256.0f) * 1.4426950408889634f;
    float m_lo = TQ_AMM_NINF, m_hi = TQ_AMM_NINF, l_lo = 0.0f, l_hi = 0.0f;
    // ldmatrix lane bases: K (non-trans) rows = keys 8 j + (lane & 7) + 8 (lane >> 4), dims + 8 ((lane >> 3) & 1);
    // V (trans) rows = keys (lane & 7) + 8 ((lane >> 3) & 1), dims + 8 (lane >> 4)
    const uint32_t kt = sm_base + TQ_PA_OFF_KT + ((lane & 7) + 8 * (lane >> 4)) * TQ_PA_LDB + ((lane >> 3) & 1) * 16;
    const uint32_t vt = sm_base + TQ_PA_OFF_VT + ((lane & 7) + 8 * ((lane >> 3) & 1)) * TQ_PA_LDB + (lane >> 4) * 16;

    if (lo < hi) {                                      // prologue: K_lo decoded into KT, V_lo's codes staged
        stage(lo, true);
        asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < 4; k++) piece(lo, true, k);
        stage(lo, false);
        asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();
    }
    for (int it = lo; it < hi; it++) {
        const int t0 = it * TQ_PA_TK;
        const bool have_next = it + 1 < hi;
        // ---- phase A: Q.K^T on KT = K_it; V_it decoded into VT (4 pieces at chunks 1, 5, 9, 13); K_(it+1)'s codes copied
        if (have_next) stage(it + 1, true);
        float s[4][4];
        #pragma unroll
        for (int j = 0; j < 4; j++) { s[j][0] = 0.0f; s[j][1] = 0.0f; s[j][2] = 0.0f; s[j][3] = 0.0f; }
        uint32_t s16[2][4][2];                          // AQ: ping-pong fp16 accumulators of consecutive fold groups
        #pragma unroll
        for (int c = 0; c < 16; c++) {
            uint32_t kb0[4], kb1[4], aq[4];
            tq_pa_ldsm_x4(kb0, kt + c * 32);
            tq_pa_ldsm_x4(kb1, kt + c * 32 + 16 * TQ_PA_LDB);
            if (c < NQR) { aq[0] = qa[c][0]; aq[1] = qa[c][1]; aq[2] = qa[c][2]; aq[3] = qa[c][3]; }
            else { const uint4 t = *(const uint4 *)(qh + ((c - 8) * 32 + lane) * 4); aq[0] = t.x; aq[1] = t.y; aq[2] = t.z; aq[3] = t.w; }
            if constexpr (AQ) {
                constexpr int F = TQ_PA_FOLD;
                uint32_t (*acc)[2] = s16[(c / F) & 1];
                if ((c % F) == 0) {
                    #pragma unroll
                    for (int j = 0; j < 4; j++) { acc[j][0] = 0u; acc[j][1] = 0u; }
                }
                tq_pa_mma_f16acc(acc[0], aq, kb0[0], kb0[1]);
                tq_pa_mma_f16acc(acc[1], aq, kb0[2], kb0[3]);
                tq_pa_mma_f16acc(acc[2], aq, kb1[0], kb1[1]);
                tq_pa_mma_f16acc(acc[3], aq, kb1[2], kb1[3]);
                if ((c % F) == F - 1 && c >= F) {       // fold the PREVIOUS group (its HMMAs are long done)
                    #pragma unroll
                    for (int j = 0; j < 4; j++) tq_pa_fold16(s[j], s16[((c / F) & 1) ^ 1][j]);
                }
                if (c == 15) {
                    #pragma unroll
                    for (int j = 0; j < 4; j++) tq_pa_fold16(s[j], acc[j]);
                }
            } else {
                tq_pa_mma_f16(s[0], aq, kb0[0], kb0[1]);
                tq_pa_mma_f16(s[1], aq, kb0[2], kb0[3]);
                tq_pa_mma_f16(s[2], aq, kb1[0], kb1[1]);
                tq_pa_mma_f16(s[3], aq, kb1[2], kb1[3]);
            }
            if ((c & 3) == 1) piece(it, false, c >> 2);
        }
        asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();
        // ---- phase C: online softmax, P.V on VT = V_it; K_(it+1) decoded into KT (4 pieces); V_(it+1)'s codes copied
        if (have_next) stage(it + 1, false);
        const float *ks = ksc + (it & 1) * TQ_PA_TK, *vs = vsc + (it & 1) * TQ_PA_TK;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const float2 sc = *(const float2 *)(ks + 8 * j + 2 * t4);
            const float f0 = sc.x * rsl, f1 = sc.y * rsl;
            s[j][0] *= f0; s[j][1] *= f1; s[j][2] *= f0; s[j][3] *= f1;
        }
        if (t0 + TQ_PA_TK - 1 > minpos) {               // causal staircase (block-uniform test)
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                const int k0 = t0 + 8 * j + 2 * t4;
                if (k0 > qpos_lo) s[j][0] = TQ_AMM_NINF;
                if (k0 + 1 > qpos_lo) s[j][1] = TQ_AMM_NINF;
                if (k0 > qpos_hi) s[j][2] = TQ_AMM_NINF;
                if (k0 + 1 > qpos_hi) s[j][3] = TQ_AMM_NINF;
            }
        }
        float tm_lo = fmaxf(fmaxf(s[0][0], s[0][1]), fmaxf(s[1][0], s[1][1]));
        float tm_hi = fmaxf(fmaxf(s[0][2], s[0][3]), fmaxf(s[1][2], s[1][3]));
        tm_lo = fmaxf(tm_lo, fmaxf(fmaxf(s[2][0], s[2][1]), fmaxf(s[3][0], s[3][1])));
        tm_hi = fmaxf(tm_hi, fmaxf(fmaxf(s[2][2], s[2][3]), fmaxf(s[3][2], s[3][3])));
        tm_lo = fmaxf(tm_lo, __shfl_xor_sync(0xffffffffu, tm_lo, 1));
        tm_hi = fmaxf(tm_hi, __shfl_xor_sync(0xffffffffu, tm_hi, 1));
        tm_lo = fmaxf(tm_lo, __shfl_xor_sync(0xffffffffu, tm_lo, 2));
        tm_hi = fmaxf(tm_hi, __shfl_xor_sync(0xffffffffu, tm_hi, 2));
        const float mn_lo = fmaxf(m_lo, tm_lo), mn_hi = fmaxf(m_hi, tm_hi);
        const float a_lo = tq_pa_ex2(m_lo - mn_lo), a_hi = tq_pa_ex2(m_hi - mn_hi);
        m_lo = mn_lo; m_hi = mn_hi;
        float ps_lo = 0.0f, ps_hi = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            s[j][0] = tq_pa_ex2(s[j][0] - mn_lo); s[j][1] = tq_pa_ex2(s[j][1] - mn_lo);
            s[j][2] = tq_pa_ex2(s[j][2] - mn_hi); s[j][3] = tq_pa_ex2(s[j][3] - mn_hi);
            ps_lo += s[j][0] + s[j][1]; ps_hi += s[j][2] + s[j][3];
        }
        l_lo = l_lo * a_lo + ps_lo; l_hi = l_hi * a_hi + ps_hi;
        if (__any_sync(0xffffffffu, (a_lo != 1.0f) | (a_hi != 1.0f))) {
            #pragma unroll
            for (int j = 0; j < 32; j++) { o[j][0] *= a_lo; o[j][1] *= a_lo; o[j][2] *= a_hi; o[j][3] *= a_hi; }
        }
        // P (x the value rows' scales) as fp16 A-fragments of the two 16-key chunks
        uint32_t pa[2][4];
        #pragma unroll
        for (int kc = 0; kc < 2; kc++) {
            const float2 v0 = *(const float2 *)(vs + 16 * kc + 2 * t4);
            const float2 v1 = *(const float2 *)(vs + 16 * kc + 8 + 2 * t4);
            pa[kc][0] = tq_pa_pack_f16(s[2 * kc][0] * v0.x, s[2 * kc][1] * v0.y);
            pa[kc][1] = tq_pa_pack_f16(s[2 * kc][2] * v0.x, s[2 * kc][3] * v0.y);
            pa[kc][2] = tq_pa_pack_f16(s[2 * kc + 1][0] * v1.x, s[2 * kc + 1][1] * v1.y);
            pa[kc][3] = tq_pa_pack_f16(s[2 * kc + 1][2] * v1.x, s[2 * kc + 1][3] * v1.y);
        }
        if constexpr (AP) {
            uint32_t o16[2][4][2];                      // ping-pong fp16 accumulators of consecutive 4-n-tile groups
            #pragma unroll
            for (int grp = 0; grp < 8; grp++) {         // n-tiles 4 grp .. 4 grp + 3: fp16 over the tile's 32 keys, then fold
                uint32_t (*acc)[2] = o16[grp & 1];
                #pragma unroll
                for (int jj = 0; jj < 4; jj++) { acc[jj][0] = 0u; acc[jj][1] = 0u; }
                #pragma unroll
                for (int kc = 0; kc < 2; kc++) {
                    #pragma unroll
                    for (int jp = 0; jp < 2; jp++) {
                        uint32_t vb[4];
                        tq_pa_ldsm_x4_trans(vb, vt + kc * (16 * TQ_PA_LDB) + (2 * grp + jp) * 32);
                        tq_pa_mma_f16acc(acc[2 * jp], pa[kc], vb[0], vb[1]);
                        tq_pa_mma_f16acc(acc[2 * jp + 1], pa[kc], vb[2], vb[3]);
                    }
                }
                if (grp >= 1) {                         // fold the previous group behind this group's HMMAs
                    #pragma unroll
                    for (int jj = 0; jj < 4; jj++) tq_pa_fold16(o[4 * (grp - 1) + jj], o16[(grp & 1) ^ 1][jj]);
                }
                if (grp == 7) {
                    #pragma unroll
                    for (int jj = 0; jj < 4; jj++) tq_pa_fold16(o[4 * grp + jj], acc[jj]);
                }
                if ((grp & 1) == 0 && have_next) piece(it + 1, true, grp >> 1);
            }
        } else {
            #pragma unroll
            for (int kc = 0; kc < 2; kc++) {
                #pragma unroll
                for (int jp = 0; jp < 16; jp++) {
                    uint32_t vb[4];
                    tq_pa_ldsm_x4_trans(vb, vt + kc * (16 * TQ_PA_LDB) + jp * 32);
                    tq_pa_mma_f16(o[2 * jp], pa[kc], vb[0], vb[1]);
                    tq_pa_mma_f16(o[2 * jp + 1], pa[kc], vb[2], vb[3]);
                    if (kc == 0 && (jp & 3) == 3 && have_next) piece(it + 1, true, jp >> 2);
                }
            }
        }
        asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();
    }
    // ---- partial (m, l, O) of this split for the merge
    l_lo += __shfl_xor_sync(0xffffffffu, l_lo, 1); l_hi += __shfl_xor_sync(0xffffffffu, l_hi, 1);
    l_lo += __shfl_xor_sync(0xffffffffu, l_lo, 2); l_hi += __shfl_xor_sync(0xffffffffu, l_hi, 2);
    float *ps = part + (size_t)((kv_head * gridDim.y + rb) * S + z) * TQ_PA_PART_F;
    const int r_lo = warp * 16 + gr, r_hi = r_lo + 8;
    if (t4 == 0) { ps[r_lo] = m_lo; ps[r_hi] = m_hi; ps[TQ_PA_ROWS + r_lo] = l_lo; ps[TQ_PA_ROWS + r_hi] = l_hi; }
    float *po = ps + 2 * TQ_PA_ROWS;
    #pragma unroll
    for (int j = 0; j < 32; j++) {
        *(float2 *)(po + (size_t)r_lo * 256 + 8 * j + 2 * t4) = make_float2(o[j][0], o[j][1]);
        *(float2 *)(po + (size_t)r_hi * 256 + 8 * j + 2 * t4) = make_float2(o[j][2], o[j][3]);
    }
}

// Merge of k_tq_pattn_t4's partials: grid (nkv * row blocks * 8 m-tiles), 256 threads; per row m* = max over the S
// splits, l / O rescaled by exp(m_s - m*), O / l, one FWHT-256 per row (the pool's V is in the rotated space), the gate.
template <int G>
__global__ void __launch_bounds__(256) k_tq_pattn_merge(
    float *out, const float *q_proj_base, int N, int nblk, int S, int q_m, int attn_m, const float *part) {
    __shared__ float ob[16 * 256];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int w = blockIdx.x & 7, blk = blockIdx.x >> 3;
    const int kv_head = blk / nblk, rb = blk - kv_head * nblk;
    const int nptile = (N + 15) >> 4;
    const int mt = rb * 8 + w, ptile = mt / G, hl = mt - ptile * G, head = kv_head * G + hl;
    if (ptile >= nptile) return;
    const float *ps0 = part + (size_t)((kv_head * nblk + rb) * S) * TQ_PA_PART_F;
    for (int rr = 0; rr < 16; rr++) {
        if (ptile * 16 + rr >= N) break;
        const int r = w * 16 + rr;
        float m_star = TQ_AMM_NINF;
        for (int c = 0; c < S; c++) m_star = fmaxf(m_star, ps0[(size_t)c * TQ_PA_PART_F + r]);
        float l = 0.0f, o = 0.0f;
        for (int c = 0; c < S; c++) {
            const float *ps = ps0 + (size_t)c * TQ_PA_PART_F;
            const float wgt = exp2f(ps[r] - m_star);          // m is in log2 units (ex2 softmax)
            l += wgt * ps[TQ_PA_ROWS + r];
            o += wgt * ps[2 * TQ_PA_ROWS + (size_t)r * 256 + tid];
        }
        ob[rr * 256 + tid] = o / l;
    }
    __syncthreads();
    for (int rr = warp; rr < 16; rr += 8) {
        const int pidx = ptile * 16 + rr;
        if (pidx >= N) break;
        float x[8];
        #pragma unroll
        for (int e = 0; e < 8; e++) x[e] = ob[rr * 256 + lane + 32 * e];
        tq_fwht256_warp(x, lane);
        const float *q_proj = q_proj_base + (size_t)pidx * q_m;
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            const int d = lane + 32 * e;
            const float gate = q_proj[head * 512 + 256 + d];
            out[(size_t)pidx * attn_m + head * 256 + d] = x[e] * (1.0f / (1.0f + expf(-gate)));
        }
    }
}

// TQ_PATTN_GQA: 0 = the per-head bf16 prefill attention kernel (the 10-16 path, bit for bit); 1 = the GQA-grouped tensor-core
// kernel on the trellis pool with fp32 accumulation -- DEFAULT since 2026-10-17 (session 11: attention 17.0 -> 9.3 ms per
// 128-column wave at 32K, 67.6 -> 34.2 at 128K; e2e dKL within the CI at 4K / 32K on two seeds, spec exactness ALL PASS,
// needle 6/6; notes/session11-pattn.md); 3 / 4 / 5 = fp16-accumulate variants (opt-in: a position-dependent +0.00005 dKL
// at >= 24K); 2 = instrumentation.
static int pattn_gqa_mode(void) {
    static int c = -1;
    if (c < 0) { const char *e = getenv("TQ_PATTN_GQA"); c = (e && *e) ? atoi(e) : 1; }
    return c;
}
static int pattn_split_override(void) {                 // TQ_PATTN_S: force the key-split count (experiments)
    static int c = -1;
    if (c < 0) { const char *e = getenv("TQ_PATTN_S"); c = e ? atoi(e) : 0; }
    return c;
}
static float *g_pattn_part = NULL;
static size_t g_pattn_part_f = 0;
static int g_pattn_sms = 0;
// Key-split count: about one block per SM (nkv x nblk x S ~ SMs), at least 16 tiles (512 keys) per split, <= 16.
static int pattn_split_S(int nkv, int nblk, int max_pos) {
    if (pattn_split_override() > 0) return pattn_split_override();
    if (g_pattn_sms <= 0) { cudaDeviceGetAttribute(&g_pattn_sms, cudaDevAttrMultiProcessorCount, 0); if (g_pattn_sms <= 0) g_pattn_sms = 170; }
    int S = g_pattn_sms / (nkv * nblk);
    const int nst = (max_pos + TQ_PA_TK) / TQ_PA_TK;
    if (S > nst / 16) S = nst / 16;
    if (S > 16) S = 16;
    return S < 1 ? 1 : S;
}
// Launch the GQA-grouped kernel + merge for one segment of n columns. Returns 0, or < 0 (caller falls back).
static int launch_pattn_t4(float *out, const float *q_proj, const uint16_t *q_norm_w, const uint8_t *k4_pool,
                           const uint8_t *v8_pool, const int *positions, int slot, const int *block_table,
                           int max_blocks, int page, int page_log, int n, int max_pos, const tq_t4_pool &t4p,
                           cudaStream_t st) {
    const int nh = g_qwen.nh, nkv = g_qwen.nkv, hd = g_qwen.hd, G = nh / nkv;
    if (hd != 256 || G * nkv != nh || G != 6 || page < 128) return -1;
    const int q_m = nh * hd * 2, attn_m = nh * hd;
    const int nptile = (n + 15) >> 4, nblk = (nptile * G + 7) >> 3;
    const int S = pattn_split_S(nkv, nblk, max_pos);
    const size_t need = (size_t)nkv * nblk * S * TQ_PA_PART_F;
    if (g_pattn_part_f < need) {
        if (g_pattn_part) { cudaFree(g_pattn_part); g_pattn_part = NULL; g_pattn_part_f = 0; }
        if (cudaMalloc(&g_pattn_part, need * sizeof(float)) != cudaSuccess) return -2;
        g_pattn_part_f = need;
    }
    // TQ_PATTN_GQA: 1 = fp32 accumulate, Q in registers; 2 = Q half in smem (QH); 3 = QH + fp16-accumulate Q.K^T (AQ);
    // 4 = QH + AQ + fp16-accumulate P.V (AP); 5 = QH + AP only
    const int mode = pattn_gqa_mode();
    static int attr_done = 0;
    if (!attr_done) {
        if (cudaFuncSetAttribute(k_tq_pattn_t4<6, 0, 0, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, TQ_PA_SMEM0) != cudaSuccess ||
            cudaFuncSetAttribute(k_tq_pattn_t4<6, 1, 0, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, TQ_PA_SMEM1) != cudaSuccess ||
            cudaFuncSetAttribute(k_tq_pattn_t4<6, 1, 1, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, TQ_PA_SMEM1) != cudaSuccess ||
            cudaFuncSetAttribute(k_tq_pattn_t4<6, 1, 1, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, TQ_PA_SMEM1) != cudaSuccess ||
            cudaFuncSetAttribute(k_tq_pattn_t4<6, 1, 0, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, TQ_PA_SMEM1) != cudaSuccess) return -3;
        attr_done = 1;
    }
#define TQ_PATTN_LAUNCH(QH, AQ, AP) \
    k_tq_pattn_t4<6, QH, AQ, AP><<<dim3(nkv, nblk, S), 256, QH ? TQ_PA_SMEM1 : TQ_PA_SMEM0, st>>>( \
        q_proj, q_norm_w, k4_pool, v8_pool, positions, slot, block_table, max_blocks, page, page_log, \
        n, nkv, q_m, g_qwen.eps, g_qwen.rope_theta, g_pattn_part, S, t4p)
    switch (mode) {
        case 2:  TQ_PATTN_LAUNCH(1, 0, 0); break;
        case 3:  TQ_PATTN_LAUNCH(1, 1, 0); break;
        case 4:  TQ_PATTN_LAUNCH(1, 1, 1); break;
        case 5:  TQ_PATTN_LAUNCH(1, 0, 1); break;
        default: TQ_PATTN_LAUNCH(0, 0, 0); break;
    }
#undef TQ_PATTN_LAUNCH
    k_tq_pattn_merge<6><<<dim3(nkv * nblk * 8), 256, 0, st>>>(out, q_proj, n, nblk, S, q_m, attn_m, g_pattn_part);
    if (getenv("TQ_PATTN_DEBUG")) {                     // scan the partials of this launch for non-finite values
        static int shown = 0;
        if (shown < 4) {
            shown++;
            cudaStreamSynchronize(st);
            float *h = (float *)malloc(need * sizeof(float));
            cudaMemcpy(h, g_pattn_part, need * sizeof(float), cudaMemcpyDeviceToHost);
            for (int kvh = 0; kvh < nkv; kvh++) for (int rbb = 0; rbb < nblk; rbb++) for (int zz = 0; zz < S; zz++) {
                const float *ps = h + ((size_t)(kvh * nblk + rbb) * S + zz) * TQ_PA_PART_F;
                int bad_m = 0, bad_l = 0, bad_o = 0, ninf_m = 0;
                for (int r = 0; r < TQ_PA_ROWS; r++) {
                    if (!isfinite(ps[r])) bad_m++; else if (ps[r] < -1e37f) ninf_m++;
                    if (!isfinite(ps[TQ_PA_ROWS + r])) bad_l++;
                }
                for (int i = 0; i < TQ_PA_ROWS * 256; i++) if (!isfinite(ps[2 * TQ_PA_ROWS + i])) bad_o++;
                if (bad_m || bad_l || bad_o || (kvh == 0 && rbb == 0))
                    fprintf(stderr, "[pattn-debug] n=%d max_pos=%d S=%d kv %d rb %d z %d: nonfinite m %d l %d O %d; ninf m %d; row0 m %g l %g O[0] %g O[1] %g; row 17 m %g l %g\n",
                            n, max_pos, S, kvh, rbb, zz, bad_m, bad_l, bad_o, ninf_m, ps[0], ps[TQ_PA_ROWS], ps[2 * TQ_PA_ROWS], ps[2 * TQ_PA_ROWS + 1],
                            ps[17], ps[TQ_PA_ROWS + 17]);
            }
            free(h);
        }
    }
    return cudaGetLastError() == cudaSuccess ? 0 : -4;
}

// S key-chunks for a wide-prefill attention launch over prefix [0..max_pos]:
// 1 (bit-identical single pass) until the prefix outgrows what nh x nqt blocks
// keep parallel, then ~1 chunk per TQ_WIDE_ATTN_SPLIT positions, capped at 8.
static int wide_attn_split_S(int max_pos) {
    static int chunk = -1;
    if (chunk < 0) {
        const char *e = getenv("TQ_WIDE_ATTN_SPLIT");
        chunk = e ? atoi(e) : 16384;
        if (chunk == 0) chunk = 1 << 30;               // 0 = never split
        if (chunk < 2048) chunk = 2048;
    }
    int S = max_pos >= 0 ? (max_pos + chunk) / chunk : 1;
    if (S > 8) S = 8;
    return S < 1 ? 1 : S;
}

// Dedicated partial slab for the split (nh x nqt x S x TQ_AMM_PART_F floats,
// ~25 MB at the production 24 x 8 x 8 -- NOT the verify scores scratch, so the
// batched/paged tier does not pay the 0.5 GiB d_attn_scores allocation here).
static float *g_wide_attn_part = NULL;
static size_t g_wide_attn_part_f = 0;
static int ensure_wide_attn_part(size_t floats) {
    if (g_wide_attn_part_f >= floats) return 0;
    if (g_wide_attn_part) { cudaFree(g_wide_attn_part); g_wide_attn_part = NULL; g_wide_attn_part_f = 0; }
    if (cudaMalloc(&g_wide_attn_part, floats * sizeof(float)) != cudaSuccess) return -1;
    g_wide_attn_part_f = floats;
    return 0;
}

// Query rows per attention block on the Q4 wide-prefill path: 32 (QT=2, default)
// halves the L2 K/V stream per query token on the L2-bound kernel; 16 = legacy
// (escape: TQ_WIDE_ATTN_QROWS=16).
static int wide_attn_qrows(void) {
    static int c = -1;
    if (c < 0) { const char *e = getenv("TQ_WIDE_ATTN_QROWS"); c = (e && atoi(e) == 16) ? 16 : 32; }
    return c;
}

static int wide_attn_mma_enabled(void) {
    static int c = -1;
    if (c < 0) { const char *e = getenv("TQ_WIDE_ATTN_MMA"); c = (e && atoi(e)) ? 1 : 0; }
    return c;
}

// Launch the wide-attn pair for a chunk of N tokens (fp32 KV only). `positions` is a
// device array of the N absolute slots; max_pos is their maximum (the dynamic scores
// smem size). k_proj_base/v_proj_base/q_proj_base hold the RAW (un-normed) projections
// laid out [token][proj]. Returns 0 on success, negative on launch error.
static int launch_wide_attn(float *out, const float *q_proj, const uint16_t *q_norm_w,
                            const float *k_proj, const float *v_proj, const uint16_t *k_norm_w,
                            float *k_cache, float *v_cache, const int *positions,
                            int N, int max_pos, cudaStream_t st) {
    int nh = g_qwen.nh, nkv = g_qwen.nkv, hd = g_qwen.hd;
    int q_m = nh * hd * 2, kv_m = nkv * hd, attn_m = nh * hd;
    k_tq_wide_attn_kv_write<<<dim3(nkv, N), hd, 0, st>>>(
        k_cache, v_cache, k_proj, v_proj, k_norm_w, positions, nkv, hd, kv_m,
        g_qwen.eps, g_qwen.rope_theta);
    // Flash key-chunking: k_tq_wide_attn now uses a fixed O(TQ_WIDE_ATTN_CHUNK) static
    // shared-memory footprint (a few KB), so there is NO dynamic smem and no opt-in cap
    // to raise as context grows -- this is exactly what removes the old O(pos)-smem launch
    // failure (-11) at ctx ~32k. `max_pos` is retained only for ABI compatibility.
    (void)max_pos;
    if (wide_attn_mma_enabled()) {       // tensor-core prefill attention (compute-bound regime)
        int nqt = (N + 15) / 16;
        int S = wide_attn_split_S(max_pos);
        if (S > 1 && ensure_wide_attn_part((size_t)nh * nqt * S * TQ_AMM_PART_F) != 0)
            S = 1;
        k_tq_wide_attn_mma<1, 1><<<dim3(nh, nqt, S), 256, 0, st>>>(
            out, q_proj, q_norm_w, k_cache, v_cache, NULL, NULL, NULL, NULL, positions,
            0, NULL, 0, 0, 0, N, nh, nkv, hd, q_m, attn_m,
            g_qwen.eps, g_qwen.rope_theta, g_wide_attn_part);
        if (S > 1)
            k_tq_wide_attn_mma_merge<1><<<dim3(nh, nqt), 256, 0, st>>>(
                out, q_proj, N, nh, hd, q_m, attn_m, S, g_wide_attn_part);
    } else {
        k_tq_wide_attn<<<dim3(nh, N), hd, 0, st>>>(
            out, q_proj, q_norm_w, k_cache, v_cache, positions, nh, nkv, hd, q_m, attn_m,
            g_qwen.eps, g_qwen.rope_theta);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

__global__ void k_tq_spec_archive_kv(float *dst, const float *cache, const int *pos_arr,
                                      int node_idx, int kv_m,
                                      const uint8_t *cache8, const float *cscale,
                                      int nkv, int hd,
                                      const uint8_t *cache4, const float *cq4s) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= kv_m) return;
    int pos = pos_arr[node_idx];
    if (cache4) {   // int4 K cache -> fp32 archive (stays in the ROTATED space)
        uint8_t byte = cache4[((size_t)pos * kv_m + tid) >> 1];
        float code = (float)((tid & 1) ? (byte >> 4) : (byte & 15));
        int kv_head = tid / hd, g = (tid % hd) >> 5;
        const float *sz = q4s_adv(cq4s, ((size_t)pos * nkv + kv_head) * 16 + 2 * g);
        dst[tid] = (code - q4s_at(sz, 1)) * q4s_at(sz, 0);
        return;
    }
    if (cache8) {   // FP8 cache -> fp32 archive (dequant; arch stays fp32)
        dst[tid] = tq_e4m3_dec_fast(cache8[(size_t)pos * kv_m + tid]) *
                   cscale[(size_t)pos * nkv + tid / hd];
        return;
    }
    dst[tid] = cache[(size_t)pos * kv_m + tid];
}
