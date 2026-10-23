// the paged KV pool: per-slot argmax and state sizes, KV write kernels and the trellis page encoder.
// Part of the forward_qwen.cu unity build: included in order, not a standalone header.

// Per-client greedy argmax over [N x V] logits (one block/client, lowest-index tie-break).
__global__ void k_tq_batched_argmax(int *out_ids, const float *logits, int V, int N) {
    tq_pdl_enter();
    int c = blockIdx.x;
    if (c >= N) return;
    const float *row = logits + (size_t)c * V;
    int tid = threadIdx.x, bs = blockDim.x;
    float best = -3.402823466e38f; int besti = 0;
    for (int i = tid; i < V; i += bs) {
        float v = row[i];
        if (v > best) { best = v; besti = i; }
    }
    __shared__ float sv[256];
    __shared__ int si[256];
    sv[tid] = best; si[tid] = besti;
    __syncthreads();
    for (int s = bs >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            if (sv[tid + s] > sv[tid] || (sv[tid + s] == sv[tid] && si[tid + s] < si[tid])) {
                sv[tid] = sv[tid + s]; si[tid] = si[tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) out_ids[c] = si[0];
}

// The drafter's argmax over the DRAFT lm_head's V' rows (TQ_FLAG_DRAFT_HEAD): the winning row is mapped to its
// token id through d2t (stored as float: exact below 2^24). Same reduction as k_tq_batched_argmax.
__global__ void k_tq_batched_argmax_map(int *out_ids, const float *logits, int V, int N, const float *d2t) {
    tq_pdl_enter();
    int c = blockIdx.x;
    if (c >= N) return;
    const float *row = logits + (size_t)c * V;
    int tid = threadIdx.x, bs = blockDim.x;
    float best = -3.402823466e38f; int besti = 0;
    for (int i = tid; i < V; i += bs) {
        float v = row[i];
        if (v > best) { best = v; besti = i; }
    }
    __shared__ float sv[256];
    __shared__ int si[256];
    sv[tid] = best; si[tid] = besti;
    __syncthreads();
    for (int s = bs >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            if (sv[tid + s] > sv[tid] || (sv[tid + s] == sv[tid] && si[tid + s] < si[tid])) {
                sv[tid] = sv[tid + s]; si[tid] = si[tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) out_ids[c] = (int)d2t[si[0]];
}

// (scale,zp) element size for the Q4 K scale store (fp16).
static size_t batched_q4s_elem_bytes(void) { return sizeof(__half); }

static int batched_recur_floats(void) {
    return g_qwen.linear_num_value_heads * g_qwen.linear_key_head_dim * g_qwen.linear_value_head_dim;
}
static int batched_conv_floats(void) {
    int key_dim = g_qwen.linear_num_key_heads * g_qwen.linear_key_head_dim;
    int value_dim = g_qwen.linear_num_value_heads * g_qwen.linear_value_head_dim;
    int conv_dim = 2 * key_dim + value_dim;
    return conv_dim * g_qwen.linear_conv_kernel_dim;
}

// ============== Roadmap E milestone 3: PAGED KV + continuous batching ==============
// Replaces the contiguous bat_ctx-per-client KV slab with a shared POOL of fixed-size
// blocks (g_page positions each). A block table maps each slot's logical block -> a
// physical pool block, so short clients use few blocks and long ones grow on demand out
// of the SAME pool. Two decouplings vs the milestone-2 contiguous path:
//   - attention reads/writes the Q4 KV at a PHYSICAL row = block_table[slot][t/page]*page
//     + t%page (per-row indirection; physical rows precomputed once per chunk in smem).
//   - the active batch is N columns; column j -> persistent slot_ids[j]. The DeltaNet
//     IO is column-major (the wide projection output) but the recurrent/conv STATE is
//     slot-major (persists across steps while clients join/leave) -> paged kernels index
//     IO by blockIdx.y and state by slot_ids[blockIdx.y].
// Q4(K)+E4M3(V) only (the @128k tier). Default-off; merged contiguous batched + wide
// prefill paths untouched.

// --- paged Q4 attention (block-table addressing; column j -> slot slot_ids[j]) ---
// VQ4 = 1 (TQ_KV_V4): V is stored like K -- FWHT-256 rotated, asymmetric int4 per 32 channels --
// with v8_pool / vscale_pool holding the codes / (scale, zp) pairs in K's layout (20.0 instead of
// 26.2 KiB per token). Attention then accumulates P.V in the rotated space and undoes the
// rotation once per (query, head): P.(V H) = (P.V) H, H self-inverse.
// T4 = 1 (TQ_KV_T4): the pool pointers are the per-slot FP4 STAGING arrays; block ids >= t4_nblk
// address staging page (id - t4_nblk) and the rotated fp32 K / V are also kept as fp16 in
// f16_stg ([stg row][nkv][K 256 | V 256]) for the page's trellis encode when it completes.
template <int VQ4, int T4 = 0>
__global__ void k_tq_paged_attn_kv_write_q4(
    uint8_t *k4_pool, float *kq4s_pool, uint8_t *v8_pool, float *vscale_pool,
    const float *k_proj_base, const float *v_proj_base,
    const uint16_t *k_norm_w, const int *positions, const int *slot_ids,
    const int *block_table, int max_blocks, int page, int page_log,
    int nkv, int hd, int kv_m, float eps, float rope_theta,
    uint16_t *f16_stg = NULL, int t4_nblk = 0) {
    tq_pdl_enter();
    __shared__ float partial[8];
    __shared__ float k_sum_shared;
    __shared__ float gsz[16];
    __shared__ float rot[256];
    int kv_head = blockIdx.x;
    int col = blockIdx.y;
    int tid = threadIdx.x;
    if (kv_head >= nkv || tid >= hd) return;
    int pos = positions[col];
    int slot = slot_ids[col];
    const float *k_proj = k_proj_base + (size_t)col * kv_m;
    const float *v_proj = v_proj_base + (size_t)col * kv_m;
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
        float c = cosf(angle), s = sinf(angle);
        float k_pair = k_proj[kv_head * hd + ((tid < 32) ? tid + 32 : tid - 32)];
        k_pair = k_pair * rsqrtf(k_sum_shared / (float)hd + eps) * (1.0f + tq_bf16_to_float(k_norm_w[(tid < 32) ? tid + 32 : tid - 32]));
        float k_rot = (tid < 32) ? -k_pair : k_pair;
        kv = kv * c + k_rot * s;
    }
    int phys = block_table[(size_t)slot * max_blocks + (pos >> page_log)];
    float vv = v_proj[kv_head * hd + tid];
    __syncthreads();
    if constexpr (T4) {
        // the column's rotated K | V go as fp16 to the slot's staging row (pos & (page-1)); the trellis encode
        // moves them into the pool block before the layer's attention (prefill / verify) or after the step
        // (decode). The in-flight row is attended exact, so no FP4 codes are produced for it.
        (void)t4_nblk; (void)phys; (void)gsz;
        const size_t pr = (size_t)slot * page + (pos & (page - 1));
        uint16_t *f16 = f16_stg + ((pr * nkv + kv_head) << 9);
        rot[tid] = vv;
        __syncthreads();
        tq_fwht256(rot, tid);
        f16[256 + tid] = __half_as_ushort(__float2half(rot[tid]));
        __syncthreads();
        rot[tid] = kv;
        __syncthreads();
        tq_fwht256(rot, tid);
        f16[tid] = __half_as_ushort(__float2half(rot[tid]));
        return;
    }
    size_t pr = (size_t)phys * page + (pos & (page - 1));   // physical KV row
    if constexpr (VQ4) {
        rot[tid] = vv;
        __syncthreads();
        tq_fwht256(rot, tid);
        float vrot = rot[tid];
        __syncthreads();
        uint8_t vcode = tq_k_q4_quant(vrot, tid, gsz);
        unsigned vhi4 = __shfl_down_sync(0xffffffffu, (unsigned)vcode, 1);
        if ((tid & 1) == 0)
            v8_pool[pr * (kv_m >> 1) + (size_t)kv_head * (hd >> 1) + (tid >> 1)] =
                (uint8_t)(vcode | ((vhi4 & 15u) << 4));
        if (tid < 16)
            q4s_put(vscale_pool, pr * (nkv * 16) + (size_t)kv_head * 16 + tid, gsz[tid]);
        __syncthreads();                                   // rot / gsz are K's next
    } else {
        float vscl = tq_kv_block_scale(vv, partial, &k_sum_shared);
        v8_pool[pr * kv_m + (size_t)kv_head * hd + tid] = tq_float_to_e4m3(vv / vscl);
        if (tid == 0) vscale_pool[pr * nkv + kv_head] = vscl;
    }
    rot[tid] = kv;
    __syncthreads();
    tq_fwht256(rot, tid);
    float krot = rot[tid];
    __syncthreads();
    uint8_t code = tq_k_q4_quant(krot, tid, gsz);
    unsigned hi4 = __shfl_down_sync(0xffffffffu, (unsigned)code, 1);
    if ((tid & 1) == 0)
        k4_pool[pr * (kv_m >> 1) + (size_t)kv_head * (hd >> 1) + (tid >> 1)] =
            (uint8_t)(code | ((hi4 & 15u) << 4));
    if (tid < 16)
        q4s_put(kq4s_pool, pr * (nkv * 16) + (size_t)kv_head * 16 + tid, gsz[tid]);
}

// ===== B2 trellis page encoder (TQ_KV_T4): the Viterbi of analysis/kv_trellis_cuda/trellis.cu =====
// One complete page = 128 tokens x nkv heads x {K, V} vectors per full-attention layer, read as fp16
// rotated values from the slot's staging rows, written as trellis codes + side words into the page's
// pool block. Tiles of a launch: tile -> (layer index fl, token t, head h, kv); <= TQ_T4_CHUNK per launch
// (the traceback scratch is 32 KB per tile). Numerics = the analysis kernel (fp32 costs, sub/mul/add
// without FMA contraction, first-occurrence ties): same input rows -> bitwise the same codes.
#define TQ_T4_L       12
#define TQ_T4_NT      256                     // threads = predecessor classes = 2^(L-4)
#define TQ_T4_MROW    20
#define TQ_T4_CHUNK   65536                   // tiles per launch (grid.x; no scratch since the fused kernel)
#define TQ_T4_MAXL    16                      // full-attention layers per job
#define TQ_T4_MAXR    64                      // position ranges per job (slots x page straddles)
struct tq_t4_job {
    const uint16_t *src[TQ_T4_MAXL];          // fp16 staging [stg row][nkv][512]
    uint8_t *dk[TQ_T4_MAXL], *dv[TQ_T4_MAXL]; // pool code arrays [row][nkv * 128]
    uint8_t *sk[TQ_T4_MAXL], *sv[TQ_T4_MAXL]; // pool side arrays [row][nkv * 4]
    unsigned src_row0[TQ_T4_MAXR], dst_row0[TQ_T4_MAXR];   // per range: first staging row / first pool row
    int pre[TQ_T4_MAXR + 1];                  // prefix sum of the ranges' position counts
    int nr, npos, nkv, nfull, tile0;          // ranges, total positions, heads, layers, first tile of this launch
};
static __device__ __forceinline__ void tq_t4_min2(float &v, int &a, float v2, int a2) {
    const bool lt = v2 < v;
    v = lt ? v2 : v;
    a = lt ? a2 : a;
}
static __device__ __forceinline__ void tq_t4_min16(const float (&c)[16], float &v, int &a) {
    float w[16];
    int ix[16];
    #pragma unroll
    for (int j = 0; j < 16; ++j) { w[j] = c[j]; ix[j] = j; }
    #pragma unroll
    for (int k = 0; k < 8; ++k) tq_t4_min2(w[2 * k], ix[2 * k], w[2 * k + 1], ix[2 * k + 1]);
    #pragma unroll
    for (int k = 0; k < 4; ++k) tq_t4_min2(w[4 * k], ix[4 * k], w[4 * k + 2], ix[4 * k + 2]);
    #pragma unroll
    for (int k = 0; k < 2; ++k) tq_t4_min2(w[8 * k], ix[8 * k], w[8 * k + 4], ix[8 * k + 4]);
    tq_t4_min2(w[0], ix[0], w[8], ix[8]);
    v = w[0];
    a = ix[0];
}
// tile -> (layer, source staging row, destination pool row, head, K/V)
static __device__ __forceinline__ void tq_t4_tile_coords(const tq_t4_job &jb, int tile, int &fl, size_t &srow, size_t &drow,
                                                         int &h, int &kv) {
    const int per_layer = jb.npos * jb.nkv * 2;
    fl = tile / per_layer;
    int rem = tile - fl * per_layer;
    const int p = rem / (jb.nkv * 2);
    rem -= p * (jb.nkv * 2);
    h = rem >> 1;
    kv = rem & 1;
    int r = 0;
    while (r + 1 < jb.nr && p >= jb.pre[r + 1]) r++;
    const int t = p - jb.pre[r];
    srow = (size_t)jb.src_row0[r] + t;
    drow = (size_t)jb.dst_row0[r] + t;
}
// Fused encode, one CTA per tile: the Viterbi forward pass with the traceback in SHARED memory (256 steps x 256
// classes x 4 bits = 32 KB) and the walk by one thread right after the block argmin (255 dependent smem reads,
// ~3 us) -- the two-kernel version with the traceback in global memory was latency-bound at ~100 us per launch for
// the small batches of a verify wave / decode step (255 dependent L2 loads per tile). x from the fp16 staging row,
// unit RMS with the fp16-rounded scale (kv_fakequant), codes + side word straight into the pool rows.
// Session 17: each thread packs its own argmin nibbles of 8 consecutive steps into one word (a funnel shift per step)
// and stores it once per 8 steps -- layout [step / 8][class], nibble step & 7 -- where the previous kernel packed 8
// classes' nibbles per step through a 3-shuffle chain that sat on the barrier's critical path; the step loop is
// unrolled by two (the m buffers' parity is then static). The arithmetic per state is the previous kernel's
// instruction for instruction (sub / mul / add without contraction, the same min tree, the same ties): the codes are
// bit for bit the same (tools: analysis/t4_encode_ab.py), the 1024-tile launch 235 -> 178 us.
// The body is shared by the two kernels below: the job by value (kernel parameter; the eager decode / verify /
// prefill paths) or read from device memory (k_t4_encode_dev: the graph-captured verify wave, whose jobs the host
// writes before every replay). Same code, same arithmetic -> the same codes.
static __device__ __forceinline__ void t4_encode_body(const tq_t4_job &jb, const float *__restrict__ cb) {
    constexpr int NT = TQ_T4_NT, NG = NT / 16, NW = NT / 32, SH = TQ_T4_L - 4;
    __shared__ float xs[256];
    __shared__ __align__(16) float ms[2][NG * TQ_T4_MROW];
    __shared__ float rv[NW];
    __shared__ int rs[NW];
    __shared__ uint32_t tbs[32 * NT];                  // traceback nibbles: [step / 8][class], nibble step & 7
    const int tile = blockIdx.x, i = threadIdx.x, lane = i & 31, warp = i >> 5, g = i >> 4;
    int fl, h, kv;
    size_t srow, drow;
    tq_t4_tile_coords(jb, jb.tile0 + tile, fl, srow, drow, h, kv);
    const uint16_t *src = jb.src[fl] + ((srow * jb.nkv + h) << 9) + kv * 256;
    const float xi = __half2float(__ushort_as_half(src[i]));
    float ss = __fmul_rn(xi, xi);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, o));
    if (lane == 0) rv[warp] = ss;
    __syncthreads();
    float tot = 0.0f;
    #pragma unroll
    for (int w = 0; w < NW; w++) tot = __fadd_rn(tot, rv[w]);
    float rms = __half2float(__float2half(sqrtf(tot * (1.0f / 256.0f))));
    if (!(rms > 1e-12f)) rms = 1e-12f;
    xs[i] = xi / rms;
    float cbv[16], cost[16];
    #pragma unroll
    for (int j = 0; j < 16; ++j) cbv[j] = cb[(j << SH) | i];
    __syncthreads();
    {
        const float x0 = xs[0];
        #pragma unroll
        for (int j = 0; j < 16; ++j) { const float d = __fsub_rn(x0, cbv[j]); cost[j] = __fmul_rn(d, d); }
    }
    const int mslot = (i & (NG - 1)) * TQ_T4_MROW + (i >> 4);
    uint32_t tbw = 0u;
    #pragma unroll 2
    for (int st = 1; st < 256; ++st) {
        float m;
        int a;
        tq_t4_min16(cost, m, a);
        const int par = st & 1;
        ms[par][mslot] = m;
        tbw = __funnelshift_r(tbw, (uint32_t)a, 4);    // the nibble of step st lands at bits 4 (st & 7) once the word is full
        if ((st & 7) == 7) tbs[(st >> 3) * NT + i] = tbw;   // (word 0 holds steps 1..7: step 0 has no predecessor)
        __syncthreads();
        const float xt = xs[st];
        const float4 *mr = reinterpret_cast<const float4 *>(&ms[par][g * TQ_T4_MROW]);
        const float4 m0 = mr[0], m1 = mr[1], m2 = mr[2], m3 = mr[3];
        const float mm[16] = {m0.x, m0.y, m0.z, m0.w, m1.x, m1.y, m1.z, m1.w,
                              m2.x, m2.y, m2.z, m2.w, m3.x, m3.y, m3.z, m3.w};
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            const float d = __fsub_rn(xt, cbv[j]);
            cost[j] = __fadd_rn(mm[j], __fmul_rn(d, d));
        }
    }
    float v;
    int a;
    tq_t4_min16(cost, v, a);
    int s = (a << SH) | i;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float v2 = __shfl_xor_sync(0xffffffffu, v, o);
        const int s2 = __shfl_xor_sync(0xffffffffu, s, o);
        if (v2 < v || (v2 == v && s2 < s)) { v = v2; s = s2; }
    }
    __syncthreads();                                   // rv is reused below
    if (lane == 0) { rv[warp] = v; rs[warp] = s; }
    __syncthreads();
    if (i == 0) {
        for (int k = 1; k < NW; ++k) {
            const float v2 = rv[k];
            const int s2 = rs[k];
            if (v2 < v || (v2 == v && s2 < s)) { v = v2; s = s2; }
        }
        // the walk: 128 code bytes (symbol pairs, low nibble first) + the side word (prefix | fp16 scale << 8)
        uint8_t *codes = (kv ? jb.dv[fl] : jb.dk[fl]) + (drow * jb.nkv + h) * 128;
        uint8_t *side = (kv ? jb.sv[fl] : jb.sk[fl]) + (drow * jb.nkv + h) * 4;
        uint32_t byte = 0;
        for (int st = 255; st > 0; --st) {
            const uint32_t b = (uint32_t)(s & 15);
            if (st & 1) byte = b << 4; else codes[st >> 1] = (uint8_t)(byte | b);
            const int c = s >> 4;
            const uint32_t w = tbs[(st >> 3) * NT + c];
            const int nib = (int)((w >> ((st & 7) * 4)) & 15u);
            s = c | (nib << SH);
        }
        codes[0] = (uint8_t)(byte | (uint32_t)(s & 15));                       // b_1 << 4 | b_0
        const uint32_t prefix = (uint32_t)((s >> 8) & 15) | ((uint32_t)((s >> 4) & 15) << 4);
        *(uint32_t *)side = prefix | ((uint32_t)__half_as_ushort(__float2half(rms)) << 8);
    }
}
__global__ void __launch_bounds__(TQ_T4_NT, 2)
k_t4_encode(const tq_t4_job jb, const float *__restrict__ cb) { t4_encode_body(jb, cb); }
__global__ void __launch_bounds__(TQ_T4_NT, 2)
k_t4_encode_dev(const tq_t4_job *__restrict__ jbp, const float *__restrict__ cb) { t4_encode_body(*jbp, cb); }

// T4 = 1 (TQ_KV_T4): rows of complete pages are trellis codes (pr_chunk >= 0), rows of the slot's open
// page FP4 in the staging arrays (stored as pr_chunk = -1 - staging row).
template <int VQ4, int T4 = 0>
__global__ void k_tq_paged_attn_q4(
    float *out, const float *q_proj_base, const uint16_t *q_norm_w,
    const uint8_t *k4_pool, const float *kq4s_pool, const uint8_t *v8_pool, const float *vscale_pool,
    const int *positions, const int *slot_ids,
    const int *block_table, int max_blocks, int page, int page_log,
    int nh, int nkv, int hd, int q_m, int attn_m, float eps, float rope_theta,
    const tq_t4_pool t4p = tq_t4_pool()) {
    __shared__ float partial[8];
    __shared__ float q_sum_shared;
    __shared__ float qn[256];
    __shared__ float s_chunk[TQ_WIDE_ATTN_CHUNK];
    __shared__ int   pr_chunk[TQ_WIDE_ATTN_CHUNK];   // physical row per logical row in chunk
    __shared__ float red[256];
    __shared__ float m_run, l_run;
    __shared__ __align__(16) uint16_t t4tab[T4 ? 4096 : 8];
    int head = blockIdx.x;
    int col = blockIdx.y;
    int tid = threadIdx.x;
    if (head >= nh || tid >= hd) return;
    if constexpr (T4) tq_t4_load_table(t4p.table, t4tab, tid);
    int pos = positions[col];
    int slot = slot_ids[col];
    int group = nh / nkv, kv_head = head / group;
    int rotary_dim = 64;
    const float *q_proj = q_proj_base + (size_t)col * q_m;
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
        float c = cosf(angle), s = sinf(angle);
        float q_pair = q_proj[head * (2 * hd) + ((tid < 32) ? tid + 32 : tid - 32)];
        q_pair = q_pair * rsqrtf(q_sum_shared / (float)hd + eps) * (1.0f + tq_bf16_to_float(q_norm_w[(tid < 32) ? tid + 32 : tid - 32]));
        float q_rot = (tid < 32) ? -q_pair : q_pair;
        qv = qv * c + q_rot * s;
    }
    qn[tid] = qv;
    __syncthreads();
    tq_fwht256(qn, tid);
    if (tid == 0) { m_run = -3.402823466e38f; l_run = 0.0f; }
    __syncthreads();
    size_t bt_base = (size_t)slot * max_blocks;
    size_t k4_pp = (size_t)(nkv * hd) >> 1, kq4s_pp = (size_t)nkv * 16, v8_pp = (size_t)nkv * hd;
    int warp = tid >> 5, lane = tid & 31;
    float scale = rsqrtf((float)hd);
    float acc = 0.0f;
    int total = pos + 1;
    for (int c0 = 0; c0 < total; c0 += TQ_WIDE_ATTN_CHUNK) {
        int clen = total - c0;
        if (clen > TQ_WIDE_ATTN_CHUNK) clen = TQ_WIDE_ATTN_CHUNK;
        for (int j = tid; j < clen; j += 256) {        // physical rows for this chunk
            int t = c0 + j;
            int phys = block_table[bt_base + (t >> page_log)];
            if constexpr (T4) {        // trellis row (pool) for t < pos, the column's own key (t == pos) from the fp16 staging
                pr_chunk[j] = (t < pos) ? phys * page + (t & (page - 1))
                                        : -1 - (slot * page + (t & (page - 1)));
            } else {
                pr_chunk[j] = phys * page + (t & (page - 1));
            }
        }
        __syncthreads();
        for (int j = warp; j < clen; j += 8) {
            float dot = 0.0f;
            if (T4 && pr_chunk[j] >= 0) {                  // trellis row
                size_t pr = (size_t)pr_chunk[j];
                const uint8_t *krow = k4_pool + pr * k4_pp + (size_t)kv_head * (hd >> 1);
                const uint32_t sw = tq_t4_side(t4p.kside, pr, nkv, kv_head);
                #pragma unroll
                for (int e = 0; e < 8; e++) {
                    int d = lane + 32 * e;
                    dot += qn[d] * tq_t4_dq(krow, sw, d, t4tab);
                }
            } else if (T4) {                               // in-flight row: fp16 rotated K
                size_t pr = (size_t)(-1 - pr_chunk[j]);
                const uint16_t *kf = t4p.f16s + ((pr * nkv + kv_head) << 9);
                #pragma unroll
                for (int e = 0; e < 8; e++) {
                    int d = lane + 32 * e;
                    dot += qn[d] * __half2float(__ushort_as_half(kf[d]));
                }
            } else {
                size_t pr = (size_t)pr_chunk[j];
                const uint8_t *krow = k4_pool + pr * k4_pp + (size_t)kv_head * (hd >> 1);
                const float *ksz = q4s_adv(kq4s_pool, pr * kq4s_pp + (size_t)kv_head * 16);
                #pragma unroll
                for (int e = 0; e < 8; e++) {
                    int d = lane + 32 * e;
                    uint8_t byte = krow[d >> 1];
                    float code = (float)((d & 1) ? (byte >> 4) : (byte & 15));
                    dot += qn[d] * ((code - q4s_at(ksz, 2 * e + 1)) * q4s_at(ksz, 2 * e));
                }
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) dot += __shfl_down_sync(0xffffffffu, dot, off);
            if (lane == 0) s_chunk[j] = dot * scale;
        }
        __syncthreads();
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
        float m_old = m_run, m_new = fmaxf(m_old, m_chunk), alpha = expf(m_old - m_new);
        float lsum = 0.0f;
        for (int j = tid; j < clen; j += 256) {
            float p = expf(s_chunk[j] - m_new);
            s_chunk[j] = p; lsum += p;
        }
        red[tid] = lsum;
        __syncthreads();
        for (int s = 128; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            __syncthreads();
        }
        float l_chunk = red[0];
        acc *= alpha;
        for (int j = 0; j < clen; j++) {
            if (T4 && pr_chunk[j] >= 0) {                  // trellis row
                size_t pr = (size_t)pr_chunk[j];
                const uint8_t *vrow = v8_pool + pr * k4_pp + (size_t)kv_head * (hd >> 1);
                const uint32_t sw = tq_t4_side(t4p.vside, pr, nkv, kv_head);
                acc += s_chunk[j] * tq_t4_dq(vrow, sw, tid, t4tab);
                continue;
            }
            if (T4) {                                      // in-flight row: fp16 rotated V
                size_t pr = (size_t)(-1 - pr_chunk[j]);
                acc += s_chunk[j] * __half2float(__ushort_as_half(t4p.f16s[((pr * nkv + kv_head) << 9) + 256 + tid]));
                continue;
            }
            size_t pr = (size_t)pr_chunk[j];
            if constexpr (VQ4) {
                uint8_t byte = v8_pool[pr * k4_pp + (size_t)kv_head * (hd >> 1) + (tid >> 1)];
                float2 z = q4s_pair(q4s_adv(vscale_pool, pr * kq4s_pp + (size_t)kv_head * 16), 2 * (tid >> 5));
                acc += s_chunk[j] * (((float)((tid & 1) ? (byte >> 4) : (byte & 15)) - z.y) * z.x);
            } else {
                float vcode = tq_e4m3_dec_fast(v8_pool[pr * v8_pp + (size_t)kv_head * hd + tid]);
                float vscl = vscale_pool[pr * nkv + kv_head];
                acc += s_chunk[j] * vcode * vscl;
            }
        }
        if (tid == 0) { l_run = l_run * alpha + l_chunk; m_run = m_new; }
        __syncthreads();
    }
    float ratio = acc / l_run;
    if constexpr (VQ4) {                                 // rotated V space -> back (red is free here)
        red[tid] = ratio;
        __syncthreads();
        tq_fwht256(red, tid);
        ratio = red[tid];
    }
    float gate = q_proj[head * (2 * hd) + hd + tid];
    out[(size_t)col * attn_m + head * hd + tid] = ratio * (1.0f / (1.0f + expf(-gate)));
}

// --- milestone 3.5: split-K flash-decoding paged Q4 attention ---
// Measure-first found the single-block kernel above is occupancy-bound at the few-client
// long-ctx regime (N=4 -> only nh*N=96 blocks; attn BW 31.7 GB/s = 1.7% of peak, linear in
// block count). This SPLITs each (head,col)'s [0..pos] walk across grid.z=S partial blocks
// (S*nh*N blocks -> fills the 170 SMs); each computes a partial online-softmax (m,l,acc) over
// its position range. k_tq_paged_attn_q4_merge combines the S partials. Default-off below a
// block threshold (S=1 -> the single kernel above; no short-ctx / high-N regression).
// VQ4: partials stay in the rotated V space; k_tq_paged_attn_q4_merge<1> rotates back.
template <int VQ4, int T4 = 0>
__global__ void k_tq_paged_attn_q4_split(
    const float *q_proj_base, const uint16_t *q_norm_w,
    const uint8_t *k4_pool, const float *kq4s_pool, const uint8_t *v8_pool, const float *vscale_pool,
    const int *positions, const int *slot_ids,
    const int *block_table, int max_blocks, int page, int page_log,
    int nh, int nkv, int hd, int q_m, int S,
    float eps, float rope_theta, float *part_acc, float *part_ml, const tq_t4_pool t4p = tq_t4_pool()) {
    __shared__ float partial[8];
    __shared__ float q_sum_shared;
    __shared__ float qn[256];
    __shared__ float s_chunk[TQ_WIDE_ATTN_CHUNK];
    __shared__ int   pr_chunk[TQ_WIDE_ATTN_CHUNK];
    __shared__ float red[256];
    __shared__ float m_run, l_run;
    __shared__ __align__(16) uint16_t t4tab[T4 ? 4096 : 8];
    int head = blockIdx.x, col = blockIdx.y, split = blockIdx.z, tid = threadIdx.x;
    if (head >= nh || tid >= hd) return;
    if constexpr (T4) tq_t4_load_table(t4p.table, t4tab, tid);
    int pos = positions[col];
    int slot = slot_ids[col];
    int group = nh / nkv, kv_head = head / group;
    int rotary_dim = 64;
    const float *q_proj = q_proj_base + (size_t)col * q_m;
    float qv = q_proj[head * (2 * hd) + tid];
    float qsum = qv * qv;
    for (int offset = 16; offset > 0; offset >>= 1) qsum += __shfl_down_sync(0xffffffff, qsum, offset);
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
        float c = cosf(angle), s = sinf(angle);
        float q_pair = q_proj[head * (2 * hd) + ((tid < 32) ? tid + 32 : tid - 32)];
        q_pair = q_pair * rsqrtf(q_sum_shared / (float)hd + eps) * (1.0f + tq_bf16_to_float(q_norm_w[(tid < 32) ? tid + 32 : tid - 32]));
        float q_rot = (tid < 32) ? -q_pair : q_pair;
        qv = qv * c + q_rot * s;
    }
    qn[tid] = qv;
    __syncthreads();
    tq_fwht256(qn, tid);
    if (tid == 0) { m_run = -3.402823466e38f; l_run = 0.0f; }
    __syncthreads();
    size_t bt_base = (size_t)slot * max_blocks;
    size_t k4_pp = (size_t)(nkv * hd) >> 1, kq4s_pp = (size_t)nkv * 16, v8_pp = (size_t)nkv * hd;
    int warp = tid >> 5, lane = tid & 31;
    float scale = rsqrtf((float)hd);
    float acc = 0.0f;
    int total = pos + 1;
    // this split's contiguous slice of [0..pos]
    int per = (total + S - 1) / S;
    int lo = split * per;
    int hi = lo + per; if (hi > total) hi = total;
    for (int c0 = lo; c0 < hi; c0 += TQ_WIDE_ATTN_CHUNK) {
        int clen = hi - c0;
        if (clen > TQ_WIDE_ATTN_CHUNK) clen = TQ_WIDE_ATTN_CHUNK;
        for (int j = tid; j < clen; j += 256) {
            int t = c0 + j;
            int phys = block_table[bt_base + (t >> page_log)];
            if constexpr (T4) {        // trellis row (pool) for t < pos, the column's own key (t == pos) from the fp16 staging
                pr_chunk[j] = (t < pos) ? phys * page + (t & (page - 1))
                                        : -1 - (slot * page + (t & (page - 1)));
            } else {
                pr_chunk[j] = phys * page + (t & (page - 1));
            }
        }
        __syncthreads();
        for (int j = warp; j < clen; j += 8) {
            float dot = 0.0f;
            if (T4 && pr_chunk[j] >= 0) {                  // trellis row
                size_t pr = (size_t)pr_chunk[j];
                const uint8_t *krow = k4_pool + pr * k4_pp + (size_t)kv_head * (hd >> 1);
                const uint32_t sw = tq_t4_side(t4p.kside, pr, nkv, kv_head);
                #pragma unroll
                for (int e = 0; e < 8; e++) {
                    int d = lane + 32 * e;
                    dot += qn[d] * tq_t4_dq(krow, sw, d, t4tab);
                }
            } else if (T4) {                               // in-flight row: fp16 rotated K
                size_t pr = (size_t)(-1 - pr_chunk[j]);
                const uint16_t *kf = t4p.f16s + ((pr * nkv + kv_head) << 9);
                #pragma unroll
                for (int e = 0; e < 8; e++) {
                    int d = lane + 32 * e;
                    dot += qn[d] * __half2float(__ushort_as_half(kf[d]));
                }
            } else {
                size_t pr = (size_t)pr_chunk[j];
                const uint8_t *krow = k4_pool + pr * k4_pp + (size_t)kv_head * (hd >> 1);
                const float *ksz = q4s_adv(kq4s_pool, pr * kq4s_pp + (size_t)kv_head * 16);
                #pragma unroll
                for (int e = 0; e < 8; e++) {
                    int d = lane + 32 * e;
                    uint8_t byte = krow[d >> 1];
                    float code = (float)((d & 1) ? (byte >> 4) : (byte & 15));
                    dot += qn[d] * ((code - q4s_at(ksz, 2 * e + 1)) * q4s_at(ksz, 2 * e));
                }
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) dot += __shfl_down_sync(0xffffffffu, dot, off);
            if (lane == 0) s_chunk[j] = dot * scale;
        }
        __syncthreads();
        float m = -3.402823466e38f;
        for (int j = tid; j < clen; j += 256) if (s_chunk[j] > m) m = s_chunk[j];
        red[tid] = m;
        __syncthreads();
        for (int s = 128; s > 0; s >>= 1) { if (tid < s && red[tid + s] > red[tid]) red[tid] = red[tid + s]; __syncthreads(); }
        float m_chunk = red[0];
        __syncthreads();
        float m_old = m_run, m_new = fmaxf(m_old, m_chunk), alpha = expf(m_old - m_new);
        float lsum = 0.0f;
        for (int j = tid; j < clen; j += 256) { float p = expf(s_chunk[j] - m_new); s_chunk[j] = p; lsum += p; }
        red[tid] = lsum;
        __syncthreads();
        for (int s = 128; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid + s]; __syncthreads(); }
        float l_chunk = red[0];
        acc *= alpha;
        for (int j = 0; j < clen; j++) {
            if (T4 && pr_chunk[j] >= 0) {                  // trellis row
                size_t pr = (size_t)pr_chunk[j];
                const uint8_t *vrow = v8_pool + pr * k4_pp + (size_t)kv_head * (hd >> 1);
                const uint32_t sw = tq_t4_side(t4p.vside, pr, nkv, kv_head);
                acc += s_chunk[j] * tq_t4_dq(vrow, sw, tid, t4tab);
                continue;
            }
            if (T4) {                                      // in-flight row: fp16 rotated V
                size_t pr = (size_t)(-1 - pr_chunk[j]);
                acc += s_chunk[j] * __half2float(__ushort_as_half(t4p.f16s[((pr * nkv + kv_head) << 9) + 256 + tid]));
                continue;
            }
            size_t pr = (size_t)pr_chunk[j];
            if constexpr (VQ4) {
                uint8_t byte = v8_pool[pr * k4_pp + (size_t)kv_head * (hd >> 1) + (tid >> 1)];
                float2 z = q4s_pair(q4s_adv(vscale_pool, pr * kq4s_pp + (size_t)kv_head * 16), 2 * (tid >> 5));
                acc += s_chunk[j] * (((float)((tid & 1) ? (byte >> 4) : (byte & 15)) - z.y) * z.x);
            } else {
                float vcode = tq_e4m3_dec_fast(v8_pool[pr * v8_pp + (size_t)kv_head * hd + tid]);
                float vscl = vscale_pool[pr * nkv + kv_head];
                acc += s_chunk[j] * vcode * vscl;
            }
        }
        if (tid == 0) { l_run = l_run * alpha + l_chunk; m_run = m_new; }
        __syncthreads();
    }
    // write this split's partial (unnormalized acc rel. to m_run; merge rescales + normalizes)
    size_t pidx = ((size_t)(head * gridDim.y + col) * S + split);
    part_acc[pidx * hd + tid] = acc;
    if (tid == 0) { part_ml[pidx * 2 + 0] = m_run; part_ml[pidx * 2 + 1] = l_run; }
}

// combine S split partials per (head,col): m*=max m_s; acc*=sum exp(m_s-m*) acc_s; l* likewise;
// out = (acc*/l*) * sigmoid(gate). One block per (head,col), thread tid = output channel.
// VQ4: the partials are in the rotated V space; one FWHT-256 of acc*/l* before the gate.
template <int VQ4>
__global__ void k_tq_paged_attn_q4_merge(
    float *out, const float *q_proj_base, const int *positions, int nh, int hd, int q_m, int attn_m,
    int S, const float *part_acc, const float *part_ml, const int *col_S) {
    tq_pdl_enter();
    __shared__ float rbuf[VQ4 ? 256 : 1];
    int head = blockIdx.x, col = blockIdx.y, tid = threadIdx.x;
    if (head >= nh || tid >= hd) return;
    const int Sc = col_S ? col_S[col] : S;              // per-column split count (verify waves); layout stride S
    if (Sc <= 1) return;                                // the kernel wrote this column's rows directly
    size_t base = (size_t)(head * gridDim.y + col) * S;
    float mstar = -3.402823466e38f;
    for (int s = 0; s < Sc; s++) mstar = fmaxf(mstar, part_ml[(base + s) * 2 + 0]);
    float acc = 0.0f, l = 0.0f;
    for (int s = 0; s < Sc; s++) {
        float ms = part_ml[(base + s) * 2 + 0];
        float ls = part_ml[(base + s) * 2 + 1];
        float sc = (ms <= -3.0e38f) ? 0.0f : expf(ms - mstar);
        acc += sc * part_acc[(base + s) * hd + tid];
        l += sc * ls;
    }
    const float *q_proj = q_proj_base + (size_t)col * q_m;
    float gate = q_proj[head * (2 * hd) + hd + tid];
    float ratio = (l > 0.0f) ? acc / l : 0.0f;
    if constexpr (VQ4) {
        rbuf[tid] = ratio;
        __syncthreads();
        tq_fwht256(rbuf, tid);
        ratio = rbuf[tid];
    }
    out[(size_t)col * attn_m + head * hd + tid] = ratio * (1.0f / (1.0f + expf(-gate)));
}

// --- paged DeltaNet: IO indexed by column (blockIdx.y), STATE by slot_ids[column] ---
// TQ_DSTATE_BF16 (read by qwn_paged_init): the storage format of the per-slot recurrent state g_pg_recur --
// 0 = fp32 (default, main's layout), 1 = bf16, 2 = fp16 (see tq_ds_ld / tq_ds_st). g_pg_ds_es = bytes per element.
static int g_pg_dstate = 0;
static size_t g_pg_ds_es = sizeof(float);
__global__ void k_tq_linear_conv_update_paged(float *out, float *state, const float *x,
                                              const uint16_t *conv_weight, int channels, int kernel_size,
                                              const int *slot_ids, int state_stride) {
    tq_pdl_enter();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= channels) return;
    int col = blockIdx.y;
    int slot = slot_ids[col];
    float *s = state + (size_t)slot * state_stride + (size_t)i * kernel_size;
    const float *xn = x + (size_t)col * channels;
    float *outn = out + (size_t)col * channels;
    float sum = 0.0f;
    for (int j = 0; j < kernel_size - 1; j++) {
        s[j] = s[j + 1];
        sum += s[j] * tq_bf16_to_float(conv_weight[i * kernel_size + j]);
    }
    s[kernel_size - 1] = xn[i];
    sum += xn[i] * tq_bf16_to_float(conv_weight[i * kernel_size + (kernel_size - 1)]);
    outn[i] = sum / (1.0f + expf(-sum));
}

// Sum of squares of a 128-vector by one warp, BIT FOR BIT the value of tq_block_reduce_sum_sh(red = x[i]^2, 128, t):
// that tree adds red[t] += red[t + 64] (t < 64), then += red[t + 32] (t < 32), then += red[t + 16], 8, 4, 2, 1 and reads
// red[0]. With lane l holding the squares of x[l], x[l + 32], x[l + 64], x[l + 96], the first two levels are in-lane
// ((x[l]^2 + x[l+64]^2) + (x[l+32]^2 + x[l+96]^2) = red[l] after level 2), the five others are shfl_down pairs (t, t +
// off), and lane 0 ends with red[0]; the result is broadcast from lane 0. Explicit __fmul_rn / __fadd_rn: the tree's
// own operations (a plain multiply stored to shared memory, plain adds) cannot be contracted, and neither can these.
// Session 17: it replaces the 7-step shared-memory tree (7 __syncthreads each, two per column) in the DeltaNet decode
// core AND the verify's sequential core; the outputs of both are the previous kernels' bit for bit.
static __device__ __forceinline__ float tq_warp_sumsq128(const float *x, int lane) {
    const float s0 = __fmul_rn(x[lane], x[lane]), s1 = __fmul_rn(x[lane + 32], x[lane + 32]);
    const float s2 = __fmul_rn(x[lane + 64], x[lane + 64]), s3 = __fmul_rn(x[lane + 96], x[lane + 96]);
    float v = __fadd_rn(__fadd_rn(s0, s2), __fadd_rn(s1, s3));      // levels 64 and 32: red[lane] of the old tree
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_down_sync(0xffffffffu, v, o));   // levels 16 .. 1: red[t] += red[t + o]
    return __shfl_sync(0xffffffffu, v, 0);                             // red[0]
}

// DS: storage format of recurrent_state (tq_ds_ld / tq_ds_st; 0 = fp32, main's kernel bit for bit).
// Session 17: the q / k norms' sums come from tq_warp_sumsq128 (every warp computes them, no shared memory, no
// barrier) and only the qn / kn write needs one __syncthreads (18 per column before: two 7-step trees + 4).
template <int G, int HS, int DS>
__global__ void k_tq_linear_decode_core_hs_paged(float *core_out, void *recurrent_state,
                                             const float *qkv_conv,
                                             const float *b_proj, const float *a_proj,
                                             const float *A_log, const uint16_t *dt_bias,
                                             int value_heads, int key_heads, int dim,
                                             const int *slot_ids, int recur_stride,
                                             int conv_dim, int core_stride, const int *positions) {
    tq_pdl_enter();
    extern __shared__ float smem[];
    float *qn = smem, *kn = smem + dim;
    int col = blockIdx.y;
    int slot = slot_ids[col];
    float *core_o = core_out + (size_t)col * core_stride;
    qkv_conv += (size_t)col * conv_dim;
    b_proj += (size_t)col * value_heads;
    a_proj += (size_t)col * value_heads;
    int blk = blockIdx.x, head = blk / HS, part = blk % HS;
    if (head >= value_heads) return;
    int t = threadIdx.x, bs = blockDim.x;
    int tids_per = dim / HS, tid_local = t / G, klane = t % G;
    int tid = part * tids_per + tid_local;
    int group = value_heads / key_heads, key_head = head / group, key_dim = key_heads * dim;
    const float *q = qkv_conv + key_head * dim;
    const float *k = qkv_conv + key_dim + key_head * dim;
    const float *v = qkv_conv + 2 * key_dim + head * dim;
    const int lane = t & 31;
    const float qsum = tq_warp_sumsq128(q, lane), ksum = tq_warp_sumsq128(k, lane);   // dim == 128 (the launcher checks)
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
    tq_ds_acc<DS> state = tq_ds_make<DS>(recurrent_state, (size_t)slot, (size_t)recur_stride, (size_t)head * dim * dim);
    if constexpr (DS == 4) state.pos = (uint32_t)positions[col];
    // dim == 128 (checked by the launcher): constant-trip loops keep the state slice in registers (the runtime-bound
    // form put st[] in local memory: STACK 64); same kk order -> same bits.
    constexpr int NKK = 128 / G;
    float st[NKK];
    float psum = 0.0f;
    #pragma unroll
    for (int j = 0; j < NKK; j++) {
        int kk = klane + j * G;
        float sval = state.ld((size_t)kk * dim + tid);
        st[j] = sval;
        psum += sval * kn[kk];
    }
    #pragma unroll
    for (int off = G >> 1; off > 0; off >>= 1) psum += __shfl_down_sync(0xffffffffu, psum, off, G);
    float kv_mem = __shfl_sync(0xffffffffu, psum, 0, G) * decay;
    float delta = (v[tid] - kv_mem) * beta;
    float pcore = 0.0f;
    #pragma unroll
    for (int j = 0; j < NKK; j++) {
        int kk = klane + j * G;
        float newv = st[j] * decay + kn[kk] * delta;
        state.st((size_t)kk * dim + tid, newv);
        pcore += newv * qn[kk];
    }
    #pragma unroll
    for (int off = G >> 1; off > 0; off >>= 1) pcore += __shfl_down_sync(0xffffffffu, pcore, off, G);
    float core = __shfl_sync(0xffffffffu, pcore, 0, G);
    if (klane == 0) core_o[head * dim + tid] = core;
}

// paged DeltaNet core + gated-norm over N columns (state by slot, IO + norm by column).
static int paged_linear_core(tq_layer_t *l, float *core_base, void *recur_base,
                             const float *qkv_base, const float *z_base,
                             const float *bp_base, const float *ap_base,
                             const int *slot_ids, const int *iota, int N,
                             int recur_f, int conv_dim, int value_dim, const int *positions) {
    int heads = g_qwen.linear_num_value_heads, key_heads = g_qwen.linear_num_key_heads;
    int dim = g_qwen.linear_value_head_dim;
    // 8 threads per state element, each head split over 4 blocks (fills 170 SMs); the cores unroll over dim == 128
    const int lc_g = 8, lc_hs = 4;
    if (dim != 128) return -1;
    size_t shA = (size_t)2 * dim * sizeof(float), shB = (size_t)dim * sizeof(float);   // qn, kn
    int blkA = (dim / lc_hs) * lc_g;
    dim3 gridA(heads * lc_hs, N);
    #define PAGED_CORE_HS1(GG, HH, DS)                                                 \
        tq_launch(g_pdl_small, k_tq_linear_decode_core_hs_paged<GG, HH, DS>, gridA, dim3(blkA), shA, \
            g_qwen.stream, core_base, recur_base, qkv_base, bp_base, ap_base,          \
            l->d_linear_A_log, l->d_linear_dt_bias, heads, key_heads, dim,             \
            slot_ids, recur_f, conv_dim, value_dim, positions)
    #define PAGED_CORE_HS(GG, HH) do {                                                 \
        if (g_pg_dstate == 1) PAGED_CORE_HS1(GG, HH, 1);                               \
        else if (g_pg_dstate == 2) PAGED_CORE_HS1(GG, HH, 2);                          \
        else if (g_pg_dstate == 3) PAGED_CORE_HS1(GG, HH, 3);                          \
        else if (g_pg_dstate == 4) PAGED_CORE_HS1(GG, HH, 4);                          \
        else PAGED_CORE_HS1(GG, HH, 0); } while (0)
    PAGED_CORE_HS(8, 4);
    #undef PAGED_CORE_HS
    #undef PAGED_CORE_HS1
    // gated norm: IO is column-major -> identity ids (the merged _b kernel reads node=ids[col]).
    dim3 gridN(heads, N);
    tq_launch(g_pdl_small, k_tq_linear_decode_gated_norm_b, gridN, dim3(dim), shB, g_qwen.stream,
              core_base, z_base, l->d_linear_norm, heads, dim, g_qwen.eps, iota, value_dim);
    return 0;
}

// split-K flash-decoding partial scratch (lazy; sized nh*N*S) + cached SM count.
static float *g_attn_pacc = NULL, *g_attn_pml = NULL;
static size_t g_attn_pacc_n = 0, g_attn_pml_n = 0;
static int g_sm_count = 0;
static int paged_sm_count(void) {
    if (g_sm_count == 0) {
        cudaDeviceProp p; int dev = 0; cudaGetDevice(&dev);
        g_sm_count = (cudaGetDeviceProperties(&p, dev) == cudaSuccess) ? p.multiProcessorCount : 170;
    }
    return g_sm_count;
}

// One 128-key super-tile of the GQA decode kernel. Force-inlined into both the dense walk and
// the page-list walk so the dense instantiation (USE_TOPK=0) does not carry the list path.
// `t0` is the logical position of the tile's first key; `phys` is its physical block.
// VQ4: vcb8 / vscb point at int4 V codes / (scale, zp) pairs (K's layout, cstride4 / szstride).
template <int VQ4>
static __device__ __forceinline__ void tq_gqa_mma_supertile(
    float oacc[4][4], uint32_t *sm, float *wmax, float *wsum, float *mst, float *lst, float *alph,
    int tid, int warp, int lane, int gr, int t4,
    int t0, int phys, int pos, int page, int group, int nkv, float rsc,
    const uint8_t *kcb4, size_t cstride4, const float *ksz4, size_t szstride,
    const uint8_t *vcb8, size_t cstride, const float *vscb) {
    const size_t pr0 = (size_t)phys * page + (size_t)(t0 & (page - 1));
#define TQ_GROW(lp) (pr0 + (size_t)(min((lp), pos) - t0))
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
                const uint32_t *a = &sm[TQ_AMM_QA + (c * 32 + lane) * 4];
                int bi = (8 * c + t4) & 15;
                uint8_t a0 = b0[bi], a8 = b0[bi + 4], e0 = b1[bi], e8 = b1[bi + 4];
                uint32_t bk00 = tq_pack_bf16(((float)(a0 & 15) - zp0) * sc0, ((float)(a0 >> 4) - zp0) * sc0);
                uint32_t bk08 = tq_pack_bf16(((float)(a8 & 15) - zp0) * sc0, ((float)(a8 >> 4) - zp0) * sc0);
                uint32_t bk10 = tq_pack_bf16(((float)(e0 & 15) - zp1) * sc1, ((float)(e0 >> 4) - zp1) * sc1);
                uint32_t bk18 = tq_pack_bf16(((float)(e8 & 15) - zp1) * sc1, ((float)(e8 >> 4) - zp1) * sc1);
                tq_mma_bf16_m16n8k16(s0, a, bk00, bk08);
                tq_mma_bf16_m16n8k16(s1, a, bk10, bk18);
            }
        }
    }
    // scale + mask: keys > pos and inert rows (>= group) go to -inf
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
    if constexpr (!VQ4) {   // fold the per-key V scale into p (the denominator keeps the unscaled p)
        int p0c = t0 + warp * 16 + 2 * t4, p1c = p0c + 8;
        float v00 = vscb[TQ_GROW(p0c) * nkv], v01 = vscb[TQ_GROW(p0c + 1) * nkv];
        float v10 = vscb[TQ_GROW(p1c) * nkv], v11 = vscb[TQ_GROW(p1c + 1) * nkv];
        p00 *= v00; p01 *= v01; p02 *= v00; p03 *= v01;
        p10 *= v10; p11 *= v11; p12 *= v10; p13 *= v11;
    }
    uint32_t *pst = &sm[TQ_AMM_PST + (warp * 32 + lane) * 4];
    pst[0] = tq_pack_bf16(p00, p01); pst[1] = tq_pack_bf16(p02, p03);
    pst[2] = tq_pack_bf16(p10, p11); pst[3] = tq_pack_bf16(p12, p13);
    __syncthreads();
    if (warp == 0 && lane < 16) {
        float l = lst[lane] * alph[lane];
        for (int w = 0; w < 8; w++) l += wsum[w * 16 + lane];
        lst[lane] = l;
    }
    float a_lo = alph[gr], a_hi = alph[8 + gr];
    #pragma unroll
    for (int f = 0; f < 4; f++) { oacc[f][0] *= a_lo; oacc[f][1] *= a_lo; oacc[f][2] *= a_hi; oacc[f][3] *= a_hi; }
    for (int tt = 0; tt < 8; tt++) {
        const uint32_t *pa = &sm[TQ_AMM_PST + (tt * 32 + lane) * 4];
        uint32_t a[4] = {pa[0], pa[1], pa[2], pa[3]};
        if constexpr (VQ4) {
            // int4 V: the warp's 32 output channels are one (scale, zp) group (g = warp). MMA f's
            // column gr carries channel warp*32 + 4*gr + f, so a thread's four columns are two
            // adjacent code bytes of a key row: one 16-bit load per row.
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
                tq_mma_bf16_m16n8k16(oacc[f], a, b0, b1);
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
            tq_mma_bf16_m16n8k16(oacc[f], a, b0, b1);
        }
    }
#undef TQ_GROW
    __syncthreads();
}

// Trellis (TQ_KV_T4) twin of tq_gqa_mma_supertile<1>: K and V rows are 128 trellis code bytes + a side
// word (prefix byte, fp16 scale); dequant through the 8 KB fp16 table in smem (tab). Softmax / P.V /
// mask math identical to the int4-V super-tile, so the two formats mix freely across pages.
static __device__ __forceinline__ void tq_gqa_mma_supertile_t4(
    float oacc[4][4], uint32_t *sm, float *wmax, float *wsum, float *mst, float *lst, float *alph,
    int tid, int warp, int lane, int gr, int t4,
    int t0, int phys, int pos, int page, int group, int nkv, float rsc,
    const uint8_t *kcb, const uint8_t *kside, const uint8_t *vcb, const uint8_t *vside, int kv_head,
    const uint16_t *tab) {
    const size_t pr0 = (size_t)phys * page + (size_t)(t0 & (page - 1));
    const size_t cstride4 = (size_t)nkv * 128;
#define TQ_GROW(lp) (pr0 + (size_t)(min((lp), pos) - t0))
    float s0[4], s1[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) { s0[j] = 0.0f; s1[j] = 0.0f; }
    const int tp = t0 + warp * 16;
    const int p0 = tp + gr, p1 = tp + 8 + gr;
    {
        const size_t r0 = TQ_GROW(p0), r1 = TQ_GROW(p1);
        const uint8_t *kr0 = kcb + r0 * cstride4;
        const uint8_t *kr1 = kcb + r1 * cstride4;
        const uint32_t sw0 = tq_t4_side(kside, r0, nkv, kv_head), sw1 = tq_t4_side(kside, r1, nkv, kv_head);
        const float sc0 = tq_t4_scale(sw0), sc1 = tq_t4_scale(sw1);
        uint32_t prev0 = sw0 & 0xFFu, prev1 = sw1 & 0xFFu;
        #pragma unroll
        for (int g = 0; g < 8; g++) {
            const uint4 kp0 = *(const uint4 *)(kr0 + 16 * g);
            const uint4 kp1 = *(const uint4 *)(kr1 + 16 * g);
            uint32_t bk0[2], bk8[2], ek0[2], ek8[2];
            tq_t4_kfrags(kp0, prev0, t4, sc0, tab, bk0, bk8);
            tq_t4_kfrags(kp1, prev1, t4, sc1, tab, ek0, ek8);
            prev0 = kp0.w >> 24; prev1 = kp1.w >> 24;
            #pragma unroll
            for (int cc = 0; cc < 2; cc++) {
                const int c = 2 * g + cc;
                const uint32_t *a = &sm[TQ_AMM_QA + (c * 32 + lane) * 4];
                tq_mma_bf16_m16n8k16(s0, a, bk0[cc], bk8[cc]);
                tq_mma_bf16_m16n8k16(s1, a, ek0[cc], ek8[cc]);
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
    __syncthreads();
    if (warp == 0 && lane < 16) {
        float l = lst[lane] * alph[lane];
        for (int w = 0; w < 8; w++) l += wsum[w * 16 + lane];
        lst[lane] = l;
    }
    float a_lo = alph[gr], a_hi = alph[8 + gr];
    #pragma unroll
    for (int f = 0; f < 4; f++) { oacc[f][0] *= a_lo; oacc[f][1] *= a_lo; oacc[f][2] *= a_hi; oacc[f][3] *= a_hi; }
    const int cb = 16 * warp + 2 * gr;                     // stream byte of the thread's first channel
    for (int tt = 0; tt < 8; tt++) {
        const uint32_t *pa = &sm[TQ_AMM_PST + (tt * 32 + lane) * 4];
        uint32_t a[4] = {pa[0], pa[1], pa[2], pa[3]};
        const int r0 = t0 + tt * 16 + 2 * t4;
        const size_t q0 = TQ_GROW(r0), q1 = TQ_GROW(r0 + 1), q8 = TQ_GROW(r0 + 8), q9 = TQ_GROW(r0 + 9);
        const uint32_t z0 = tq_t4_side(vside, q0, nkv, kv_head), z1 = tq_t4_side(vside, q1, nkv, kv_head);
        const uint32_t z8 = tq_t4_side(vside, q8, nkv, kv_head), z9 = tq_t4_side(vside, q9, nkv, kv_head);
        const uint32_t c0 = tq_t4_vbits(vcb + q0 * cstride4, z0, cb), c1 = tq_t4_vbits(vcb + q1 * cstride4, z1, cb);
        const uint32_t c8 = tq_t4_vbits(vcb + q8 * cstride4, z8, cb), c9 = tq_t4_vbits(vcb + q9 * cstride4, z9, cb);
        const float v0 = tq_t4_scale(z0), v1 = tq_t4_scale(z1), v8 = tq_t4_scale(z8), v9 = tq_t4_scale(z9);
        #pragma unroll
        for (int f = 0; f < 4; f++) {
            uint32_t b0 = tq_pack_bf16(tq_t4_val(c0, f, tab, v0), tq_t4_val(c1, f, tab, v1));
            uint32_t b1 = tq_pack_bf16(tq_t4_val(c8, f, tab, v8), tq_t4_val(c9, f, tab, v9));
            tq_mma_bf16_m16n8k16(oacc[f], a, b0, b1);
        }
    }
#undef TQ_GROW
    __syncthreads();
}

// Mixed super-tile (TQ_KV_T4, the super-tile holding the column's own key): page-local rows < nenc are trellis
// codes in the pool block (kcb / vcb + side words), rows >= nenc (the column's own key) exact fp16 from the slot's
// staging rows. Per-thread row format, warp-uniform MMAs.
static __device__ __forceinline__ void tq_gqa_mma_supertile_mix(
    float oacc[4][4], uint32_t *sm, float *wmax, float *wsum, float *mst, float *lst, float *alph,
    int tid, int warp, int lane, int gr, int t4,
    int t0, int phys, int stg, int nenc, int pos, int page, int group, int nkv, float rsc,
    const uint8_t *kcb, const uint8_t *kside, const uint8_t *vcb, const uint8_t *vside, int kv_head,
    const uint16_t *tab, const uint16_t *f16s) {
    const size_t pr0 = (size_t)phys * page + (size_t)(t0 & (page - 1));          // pool rows
    const size_t sr0 = (size_t)stg * page + (size_t)(t0 & (page - 1));           // staging rows
    const size_t cstride4 = (size_t)nkv * 128;
    const int diag = pos - t0;                          // the column's own key: exact (fp16), never the code
#define TQ_GOFF(lp) ((size_t)(min((lp), pos) - t0))
    float s0[4], s1[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) { s0[j] = 0.0f; s1[j] = 0.0f; }
    const int tp = t0 + warp * 16;
    const int p0 = tp + gr, p1 = tp + 8 + gr;
    {
        const size_t o0 = TQ_GOFF(p0), o1 = TQ_GOFF(p1);
        const bool tr0 = (int)o0 < nenc && (int)o0 != diag, tr1 = (int)o1 < nenc && (int)o1 != diag;
        // trellis rows: 16 code bytes per group; in-flight rows: their fp16 rotated K (exact), 4 halves per group
        const uint8_t *kr0 = tr0 ? kcb + (pr0 + o0) * cstride4 : NULL;
        const uint8_t *kr1 = tr1 ? kcb + (pr0 + o1) * cstride4 : NULL;
        const uint16_t *kf0 = tr0 ? NULL : f16s + (((sr0 + o0) * nkv + kv_head) << 9);
        const uint16_t *kf1 = tr1 ? NULL : f16s + (((sr0 + o1) * nkv + kv_head) << 9);
        const uint32_t sw0 = tr0 ? tq_t4_side(kside, pr0 + o0, nkv, kv_head) : 0u;
        const uint32_t sw1 = tr1 ? tq_t4_side(kside, pr0 + o1, nkv, kv_head) : 0u;
        const float sc0 = tq_t4_scale(sw0), sc1 = tq_t4_scale(sw1);
        uint32_t prev0 = sw0 & 0xFFu, prev1 = sw1 & 0xFFu;
        #pragma unroll
        for (int g = 0; g < 8; g++) {
            uint32_t bk0[2], bk8[2], ek0[2], ek8[2];
            if (tr0) {
                const uint4 kp0 = *(const uint4 *)(kr0 + 16 * g);
                tq_t4_kfrags(kp0, prev0, t4, sc0, tab, bk0, bk8);
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
                tq_t4_kfrags(kp1, prev1, t4, sc1, tab, ek0, ek8);
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
                const uint32_t *a = &sm[TQ_AMM_QA + (c * 32 + lane) * 4];
                tq_mma_bf16_m16n8k16(s0, a, bk0[cc], bk8[cc]);
                tq_mma_bf16_m16n8k16(s1, a, ek0[cc], ek8[cc]);
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
    __syncthreads();
    if (warp == 0 && lane < 16) {
        float l = lst[lane] * alph[lane];
        for (int w = 0; w < 8; w++) l += wsum[w * 16 + lane];
        lst[lane] = l;
    }
    float a_lo = alph[gr], a_hi = alph[8 + gr];
    #pragma unroll
    for (int f = 0; f < 4; f++) { oacc[f][0] *= a_lo; oacc[f][1] *= a_lo; oacc[f][2] *= a_hi; oacc[f][3] *= a_hi; }
    const int cb = 16 * warp + 2 * gr;
    for (int tt = 0; tt < 8; tt++) {
        const uint32_t *pa = &sm[TQ_AMM_PST + (tt * 32 + lane) * 4];
        uint32_t a[4] = {pa[0], pa[1], pa[2], pa[3]};
        const int r0 = t0 + tt * 16 + 2 * t4;
        float v[4][4];                                  // [row q0,q1,q8,q9][f]
        #pragma unroll
        for (int rr = 0; rr < 4; rr++) {
            const size_t o = TQ_GOFF(r0 + (rr >> 1) * 8 + (rr & 1));
            if ((int)o < nenc && (int)o != diag) {
                const uint32_t z = tq_t4_side(vside, pr0 + o, nkv, kv_head);
                const uint32_t c = tq_t4_vbits(vcb + (pr0 + o) * cstride4, z, cb);
                const float vs = tq_t4_scale(z);
                #pragma unroll
                for (int f = 0; f < 4; f++) v[rr][f] = tq_t4_val(c, f, tab, vs);
            } else {                                    // in-flight row: fp16 rotated V, channels 2cb .. 2cb+3
                const uint2 w = *(const uint2 *)(f16s + (((sr0 + o) * nkv + kv_head) << 9) + 256 + 2 * cb);
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
        }
    }
#undef TQ_GOFF
    __syncthreads();
}
