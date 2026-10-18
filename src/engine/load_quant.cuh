// load-time conversion kernels (E2M3 packing, embedding quantizers) and weight helpers.
// Part of the forward_qwen.cu unity build: included in order, not a standalone header.

// ===== GPU-side BF16 -> dense E2M3 quant+pack (fast on-load conversion) =====

static __device__ __forceinline__ uint32_t tq_quant_e2m3_code(float v) {
    float a = fminf(fabsf(v), 7.5f);
    uint32_t code;
    if (a < 1.0f) {
        int m = (int)(a * 8.0f + 0.5f);     // subnormal m/8, m in 0..8
        if (m < 8) code = (uint32_t)m;
        else code = (1u << 3);              // rounded up to normal e=1, m=0 (= 1.0)
    } else {
        int e = (a < 2.0f) ? 1 : (a < 4.0f ? 2 : 3);
        float sc = (float)(1 << (e - 1));
        int m = (int)((a / sc - 1.0f) * 8.0f + 0.5f);
        if (m >= 8) {                        // mantissa carry: bump exponent
            m = 0;
            e += 1;
            if (e > 3) { e = 3; m = 7; }     // clamp at max normal (7.5)
        }
        code = ((uint32_t)e << 3) | (uint32_t)m;
    }
    if (v < 0.0f) code |= 0x20u;
    return code;
}

static __device__ __forceinline__ float tq_e2m3_decode(uint32_t c) {
    float m = (float)(c & 7u);
    uint32_t e = (c >> 3) & 3u;
    float v = (e == 0) ? m * 0.125f : (1.0f + m * 0.125f) * (float)(1 << (e - 1));
    return (c & 0x20u) ? -v : v;
}

// ===== TQ_EMBED_FP8 load-time quantizers (one block per token row) =====
// Per-row absmax block reduce shared by both encoders.
static __device__ __forceinline__ float tq_embed_row_absmax(const uint16_t *src, int H, float *sm) {
    float amax = 0.0f;
    for (int i = threadIdx.x; i < H; i += blockDim.x)
        amax = fmaxf(amax, fabsf(tq_bf16_to_float(src[i])));
    for (int off = 16; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, off));
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) sm[warp] = amax;
    __syncthreads();
    if (warp == 0) {
        amax = (lane < (int)((blockDim.x + 31) / 32)) ? sm[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, off));
        if (lane == 0) sm[0] = amax;
    }
    __syncthreads();
    return sm[0];
}

// E4M3 per-row: scale = absmax/448 so the row max maps to the E4M3 max; lookup
// returns E4M3(x/scale) * scale. -1.18 GiB vs bf16.
__global__ void k_tq_embed_quant_e4m3(const uint16_t *embed, uint8_t *out8, float *scale, int H) {
    int row = blockIdx.x;
    const uint16_t *src = embed + (size_t)row * H;
    __shared__ float sm[32];
    float amax = tq_embed_row_absmax(src, H, sm);
    float sc = amax * (1.0f / 448.0f);
    if (!(sc > 0.0f)) sc = 1.0f;
    if (threadIdx.x == 0) scale[row] = sc;
    float inv = 1.0f / sc;
    uint8_t *dst = out8 + (size_t)row * H;
    for (int i = threadIdx.x; i < H; i += blockDim.x)
        dst[i] = tq_float_to_e4m3(tq_bf16_to_float(src[i]) * inv);
}

// E2M3 per-row packed 6-bit: scale = absmax/7.5; 4 codes packed into 3 bytes.
// Requires H % 4 == 0 (5120 holds). -1.48 GiB vs bf16.
__global__ void k_tq_embed_quant_e2m3(const uint16_t *embed, uint8_t *out6, float *scale, int H) {
    int row = blockIdx.x;
    const uint16_t *src = embed + (size_t)row * H;
    __shared__ float sm[32];
    float amax = tq_embed_row_absmax(src, H, sm);
    float sc = amax * (1.0f / 7.5f);
    if (!(sc > 0.0f)) sc = 1.0f;
    if (threadIdx.x == 0) scale[row] = sc;
    float inv = 1.0f / sc;
    uint8_t *dst = out6 + (size_t)row * ((H * 6) / 8);
    for (int g = threadIdx.x; g < H / 4; g += blockDim.x) {
        uint32_t c0 = tq_quant_e2m3_code(tq_bf16_to_float(src[4 * g + 0]) * inv);
        uint32_t c1 = tq_quant_e2m3_code(tq_bf16_to_float(src[4 * g + 1]) * inv);
        uint32_t c2 = tq_quant_e2m3_code(tq_bf16_to_float(src[4 * g + 2]) * inv);
        uint32_t c3 = tq_quant_e2m3_code(tq_bf16_to_float(src[4 * g + 3]) * inv);
        uint32_t packed = c0 | (c1 << 6) | (c2 << 12) | (c3 << 18);
        dst[g * 3 + 0] = (uint8_t)(packed & 0xff);
        dst[g * 3 + 1] = (uint8_t)((packed >> 8) & 0xff);
        dst[g * 3 + 2] = (uint8_t)((packed >> 16) & 0xff);
    }
}

// Nearest-grid E2M1 code ([s][ee][m], grid 0,.5,1,1.5,2,3,4,6).
static __device__ __forceinline__ uint32_t tq_quant_e2m1_code(float v) {
    float a = fminf(fabsf(v), 6.0f);
    uint32_t code;
    if (a < 0.75f) {
        code = (a < 0.25f) ? 0u : 1u;
    } else {
        int e = (a < 2.0f) ? 1 : (a < 4.0f ? 2 : 3);
        float sc = (float)(1 << (e - 1));
        int m = (int)((a / sc - 1.0f) * 2.0f + 0.5f);
        if (m >= 2) { m = 0; e += 1; if (e > 3) { e = 3; m = 1; } }
        code = ((uint32_t)e << 1) | (uint32_t)m;
    }
    if (v < 0.0f) code |= 0x8u;
    return code;
}

// Decode one E2M3 code from a dense-payload tile, given (row,col) inside the
// weight: inverse of the k_tq_pack_bf16_to_e2m3_dense fragment mapping.
static __device__ __forceinline__ uint32_t tq_e2m3_code_at(const uint32_t *a_payload,
                                                           int Kt, int row, int col) {
    int mt = row >> 4, kt = col >> 5;
    const uint32_t *W = a_payload + ((size_t)mt * Kt + kt) * 96;
    int g = (row & 15) >> 1, s = (col & 31) >> 3;
    int j = (((col >> 2) & 1) << 3) | ((row & 1) << 2) | (col & 3);
    int lane = (g << 2) | s;
    int bitpos = 6 * j;
    int w0 = lane * 3 + (bitpos >> 5);
    int sh = bitpos & 31;
    uint32_t lo = W[w0] >> sh;
    if (sh > 26) lo |= W[w0 + 1] << (32 - sh);
    return lo & 0x3Fu;
}

// Per 128x128 block: absmax of the DECODED E2M3 values -> pow2 scale targeting
// the E2M1 max (6.0). Used when repacking the (already quantized) lm_head.
__global__ void k_tq_block_absmax_e2m1_from_e2m3(const uint32_t *a_payload, int Kt,
                                                 int M, int K, int scale_cols,
                                                 const float *scale_old, float *scale_new) {
    int br = blockIdx.x, bc = blockIdx.y;
    int r0 = br * 128, c0 = bc * 128;
    float so = scale_old[br * scale_cols + bc];
    float amax = 0.0f;
    for (int i = threadIdx.x; i < 128 * 128; i += blockDim.x) {
        int r = r0 + (i >> 7), c = c0 + (i & 127);
        if (r < M && c < K)
            amax = fmaxf(amax, fabsf(tq_e2m3_decode(tq_e2m3_code_at(a_payload, Kt, r, c)) * so));
    }
    __shared__ float sm[32];
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, o));
    if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = amax;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = (threadIdx.x < (blockDim.x + 31) / 32) ? sm[threadIdx.x] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, o));
        if (threadIdx.x == 0)
            scale_new[br * scale_cols + bc] = (v > 0.0f) ? exp2f(ceilf(log2f(v / 6.0f))) : 1.0f;
    }
}

// Repack dense E2M3 payload -> dense E2M1 (64 words/tile, 2 per lane). One thread
// per output uint32; same fragment order as the E2M3 pack (8 nibbles per word).
__global__ void k_tq_repack_e2m3_to_e2m1(const uint32_t *src, int M, int K,
                                         int scale_cols, const float *scale_old,
                                         const float *scale_new, uint32_t *out, int Kt) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)(M / 16) * Kt * 64;
    if (idx >= total) return;
    int wl = (int)(idx & 1);
    long tmp = idx >> 1;
    int lane = (int)(tmp % 32);
    long tile = tmp / 32;
    int kt = (int)(tile % Kt);
    int mt = (int)(tile / Kt);
    int g = lane >> 2, s = lane & 3;
    uint32_t val = 0;
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        int bitpos = 4 * j - 32 * wl;
        if (bitpos < 0 || bitpos >= 32) continue;
        int half = j >> 3, row01 = (j >> 2) & 1, col4 = j & 3;
        int row = mt * 16 + ((g << 1) | row01);
        int col = kt * 32 + ((s << 3) | (half << 2) | col4);
        int scol = (scale_cols == (K + 31) / 32) ? (col >> 5) : (col >> 7);
        float so = scale_old[(row >> 7) * scale_cols + scol];
        float sn = scale_new[(row >> 7) * scale_cols + scol];
        float w = tq_e2m3_decode(tq_e2m3_code_at(src, Kt, row, col)) * so;
        val |= tq_quant_e2m1_code(w / sn) << bitpos;
    }
    out[idx] = val;
}

// One block per 128x128 weight block -> pow2 ue8m0 scale_inv targeting E2M3 (7.5).
__global__ void k_tq_block_absmax_pow2_e2m3(const uint16_t *bf16, int M, int K,
                                            int scale_cols, float *scale_inv) {
    int br = blockIdx.x, bc = blockIdx.y;
    int r0 = br * 128, c0 = bc * 128;
    float amax = 0.0f;
    for (int i = threadIdx.x; i < 128 * 128; i += blockDim.x) {
        int r = r0 + (i >> 7), c = c0 + (i & 127);
        if (r < M && c < K) amax = fmaxf(amax, fabsf(tq_bf16_to_float(bf16[(size_t)r * K + c])));
    }
    __shared__ float sm[32];
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, o));
    if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = amax;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = (threadIdx.x < (blockDim.x + 31) / 32) ? sm[threadIdx.x] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, o));
        if (threadIdx.x == 0)
            scale_inv[br * scale_cols + bc] = (v > 0.0f) ? exp2f(ceilf(log2f(v / 7.5f))) : 1.0f;
    }
}

// One thread per output uint32 of the dense E2M3 A payload. Each thread owns its
// uint32 and gathers the (up to 6) E2M3 codes whose 6-bit fields overlap it,
// mapping each back to its source (row,col) via the verified FP8 fragment order.
__global__ void k_tq_pack_bf16_to_e2m3_dense(const uint16_t *bf16, int M, int K,
                                             int scale_cols, const float *scale_inv,
                                             uint32_t *out, int Kt) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)(M / 16) * Kt * 96;
    if (idx >= total) return;
    int wl = (int)(idx % 3);
    long tmp = idx / 3;
    int lane = (int)(tmp % 32);
    long tile = tmp / 32;
    int kt = (int)(tile % Kt);
    int mt = (int)(tile / Kt);
    int g = lane >> 2, s = lane & 3;
    uint32_t val = 0;
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        int bitpos = 6 * j - 32 * wl;
        if (bitpos <= -6 || bitpos >= 32) continue;
        int half = j >> 3, row01 = (j >> 2) & 1, col4 = j & 3;
        int row = mt * 16 + ((g << 1) | row01);
        int col = kt * 32 + ((s << 3) | (half << 2) | col4);
        float w = tq_bf16_to_float(bf16[(size_t)row * K + col]);
        float si = scale_inv[(row >> 7) * scale_cols + (col >> 7)];
        uint32_t code = tq_quant_e2m3_code(w / si);
        if (bitpos >= 0) val |= code << bitpos;
        else val |= code >> (-bitpos);
    }
    out[idx] = val;
}

__global__ void k_tq_add_vec(float *out, const float *a, const float *b, int N) {
    tq_pdl_enter();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    out[i] = a[i] + b[i];
}

// Batched add over `nodes` contiguous vectors (vector v at v*N), node = blockIdx.y.
__global__ void k_tq_add_vec_b(float *out, const float *a, const float *b, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    size_t off = (size_t)blockIdx.y * N;
    out[off + i] = a[off + i] + b[off + i];
}

__global__ void k_tq_silu_mul(float *out, const float *gate, const float *up, int N) {
    tq_pdl_enter();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float g = gate[i];
    out[i] = (g / (1.0f + expf(-g))) * up[i];
}

// Batched silu*up over `nodes` contiguous vectors (vector v at v*N), node = blockIdx.y.
__global__ void k_tq_silu_mul_b(float *out, const float *gate, const float *up, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    size_t off = (size_t)blockIdx.y * N;
    float g = gate[off + i];
    out[off + i] = (g / (1.0f + expf(-g))) * up[off + i];
}

__global__ void k_tq_lm_head_bf16(float *logits, const float *x, const uint16_t *embed, int H, int V) {
    __shared__ float smem[256];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= V) return;
    float sum = 0.0f;
    const uint16_t *w = embed + (size_t)row * H;
    for (int i = tid; i < H; i += blockDim.x) {
        sum += tq_bf16_to_float(w[i]) * x[i];
    }
    smem[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    if (tid == 0) logits[row] = smem[0];
}

__global__ void k_tq_argmax_stage1(float *block_vals, int *block_ids, const float *logits, int V) {
    __shared__ float vals[256];
    __shared__ int ids[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    float best = -3.402823466e38f;
    int best_id = 0;
    for (int i = idx; i < V; i += gridDim.x * blockDim.x) {
        float v = logits[i];
        if (v > best) {
            best = v;
            best_id = i;
        }
    }
    vals[tid] = best;
    ids[tid] = best_id;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride && vals[tid + stride] > vals[tid]) {
            vals[tid] = vals[tid + stride];
            ids[tid] = ids[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        block_vals[blockIdx.x] = vals[0];
        block_ids[blockIdx.x] = ids[0];
    }
}

__global__ void __launch_bounds__(1024, 1)
k_tq_argmax_stage2(float *out_val, int *out_id, const float *vals_in, const int *ids_in, int N) {
    __shared__ float vals[1024];
    __shared__ int ids[1024];
    int tid = threadIdx.x;
    float best = tid < N ? vals_in[tid] : -3.402823466e38f;
    int best_id = tid < N ? ids_in[tid] : 0;
    vals[tid] = best;
    ids[tid] = best_id;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride && vals[tid + stride] > vals[tid]) {
            vals[tid] = vals[tid + stride];
            ids[tid] = ids[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out_val[0] = vals[0];
        out_id[0] = ids[0];
    }
}

// Optimistic top-k: record the current argmax (held in arg_val/arg_id[0]) into
// slot `slot`, then mask that vocab entry out of the logits so the next argmax
// pass returns the next-best token. Single thread; pure device (graph-safe).
__global__ void k_tq_topk_record_and_mask(float *logits, float *topk_vals, int *topk_ids,
                                          const float *arg_val, const int *arg_id, int slot) {
    int id = arg_id[0];
    topk_vals[slot] = arg_val[0];
    topk_ids[slot] = id;
    logits[id] = -3.402823466e38f;
}

static __device__ __forceinline__ uint64_t tq_splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

static __device__ __forceinline__ float tq_u01(uint64_t seed, uint32_t a, uint32_t b, uint32_t c) {
    uint64_t h = tq_splitmix64(seed ^ ((uint64_t)a << 40) ^ ((uint64_t)b << 20) ^ (uint64_t)c);
    return (float)((h >> 11) * (1.0 / 9007199254740992.0));   // [0,1)
}

// Draft sampling (Gumbel-top-k == sampling-without-replacement from
// softmax(l/tau)): scratch[v] = l[v]/tau + Gumbel(0,1). The peel-off then runs
// over SCRATCH while recorded vals stay the ORIGINAL logits (margin/eligibility
// policy keeps real-confidence semantics; candidate ORDER follows the sample).
__global__ void k_tq_gumbel_scratch(float *scratch, const float *logits, float inv_tau,
                                    uint64_t seed, uint32_t ctr, uint32_t tag, int V) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= V) return;
    float u = tq_u01(seed, ctr, tag, (uint32_t)v);
    float g = -__logf(-__logf(fmaxf(u, 1.0e-20f)));
    scratch[v] = logits[v] * inv_tau + g;
}

__global__ void k_tq_topk_record_and_mask_sampled(const float *orig_logits, float *scratch,
                                                  float *topk_vals, int *topk_ids,
                                                  const float *win_val, const int *win_id,
                                                  int slot) {
    (void)win_val;
    int id = *win_id;
    topk_ids[slot] = id;
    topk_vals[slot] = orig_logits[id];
    scratch[id] = -3.402823466e38f;
}

// Optimistic winner selection. For the v1 guard the winner is top-1 (so the
// committed token is bit-identical to the plain argmax => lossless), and we also
// expose the confidence margin (top1 - top2) which the precision-aware guard
// will gate on in a later step.
__global__ void k_tq_optimistic_winner(float *out_val, int *out_id, float *margin_out,
                                       const float *topk_vals, const int *topk_ids, int k) {
    int winner_id = topk_ids[0];
    float winner_val = topk_vals[0];
    float second = (k > 1) ? topk_vals[1] : -3.402823466e38f;
    out_id[0] = winner_id;
    out_val[0] = winner_val;
    if (margin_out) margin_out[0] = winner_val - second;
}

__global__ void k_tq_decode_graph_advance(int *token, int *pos, int *step,
                                          int *generated, int max_generated,
                                          const int *arg_id) {
    int id = arg_id[0];
    int s = step[0];
    if (s < max_generated) generated[s] = id;
    token[0] = id;
    pos[0] = pos[0] + 1;
    step[0] = s + 1;
}

__global__ void k_tq_full_attn_first_token(float *out, const float *q_proj, const float *v_proj,
                                           int nh, int nkv, int hd) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nh * hd;
    if (i >= total) return;
    int head = i / hd;
    int d = i - head * hd;
    int group = nh / nkv;
    int kv_head = head / group;
    float gate = q_proj[head * (2 * hd) + hd + d];
    float v = v_proj[kv_head * hd + d];
    out[i] = v * (1.0f / (1.0f + expf(-gate)));
}

__global__ void k_tq_full_attn_decode(float *out, const float *q_proj, const float *k_proj,
                                      const float *v_proj, const uint16_t *q_norm_w,
                                      const uint16_t *k_norm_w, float *k_cache,
                                      float *v_cache,
                                      uint8_t *k_cache8, float *k_scale,
                                      uint8_t *v_cache8, float *v_scale,
                                      uint8_t *k_cache4, float *k_q4s,
                                      int pos, int nh, int nkv,
                                      int hd, float eps, float rope_theta,
                                           float *scores_g, int sstride) {
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

__global__ void k_tq_full_attn_decode_ptr(float *out, const float *q_proj, const float *k_proj,
                                          const float *v_proj, const uint16_t *q_norm_w,
                                          const uint16_t *k_norm_w, float *k_cache,
                                          float *v_cache,
                                          uint8_t *k_cache8, float *k_scale,
                                          uint8_t *v_cache8, float *v_scale,
                                          uint8_t *k_cache4, float *k_q4s,
                                          const int *pos_ptr, int nh, int nkv,
                                          int hd, float eps, float rope_theta,
                                           float *scores_g, int sstride) {
    int pos = pos_ptr[0];
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

static int read_exact(FILE *f, void *dst, size_t nbytes, const char *what) {
    if (nbytes == 0) return 0;
    size_t got = fread(dst, 1, nbytes, f);
    if (got != nbytes) {
        fprintf(stderr, "TQF read failed for %s: expected %zu bytes, got %zu\n", what, nbytes, got);
        return -1;
    }
    return 0;
}

static int skip_exact(FILE *f, size_t nbytes, const char *what) {
    const size_t chunk = 1 << 20;
    uint8_t *buf = (uint8_t *)malloc(chunk);
    if (!buf) {
        fprintf(stderr, "malloc failed while skipping %s\n", what);
        return -1;
    }
    size_t left = nbytes;
    while (left > 0) {
        size_t now = left < chunk ? left : chunk;
        if (read_exact(f, buf, now, what) != 0) {
            free(buf);
            return -1;
        }
        left -= now;
    }
    free(buf);
    return 0;
}

static int upload_payload(FILE *f, void **dst, size_t nbytes, const char *what, int upload) {
    if (!upload) {
        return skip_exact(f, nbytes, what);
    }
    void *host = malloc(nbytes);
    if (!host) {
        fprintf(stderr, "malloc failed for %s (%zu bytes)\n", what, nbytes);
        return -1;
    }
    if (read_exact(f, host, nbytes, what) != 0) {
        free(host);
        return -1;
    }
    cudaError_t err = cudaMalloc(dst, nbytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for %s (%zu bytes): %s\n", what, nbytes, cudaGetErrorString(err));
        free(host);
        return -1;
    }
    err = cudaMemcpy(*dst, host, nbytes, cudaMemcpyHostToDevice);
    free(host);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for %s: %s\n", what, cudaGetErrorString(err));
        cudaFree(*dst);
        *dst = NULL;
        return -1;
    }
    g_qwen.device_bytes += nbytes;
    return 0;
}

static int read_non_quant(FILE *f, uint16_t **dst, size_t count, const char *what, int upload) {
    size_t nbytes = count * sizeof(uint16_t);
    g_qwen.non_quant_count++;
    return upload_payload(f, (void **)dst, nbytes, what, upload);
}

// TQ_EMBED_HOST: the bf16 token-embedding table stays in pinned host memory mapped into the
// device address space, so it costs no device memory and no quantization error; every lookup
// kernel reads its row (H bf16 = 10 KB) over PCIe. Lookup-only, so tie_word_embeddings=FALSE.
static int read_embed_host(FILE *f, size_t count) {
    size_t nbytes = count * sizeof(uint16_t);
    g_qwen.non_quant_count++;
    void *h = NULL, *d = NULL;
    if (cudaHostAlloc(&h, nbytes, cudaHostAllocMapped) != cudaSuccess) {
        fprintf(stderr, "cudaHostAlloc failed for embed (%zu bytes)\n", nbytes);
        return -1;
    }
    if (read_exact(f, h, nbytes, "embed") != 0 || cudaHostGetDevicePointer(&d, h, 0) != cudaSuccess) {
        cudaFreeHost(h);
        return -1;
    }
    g_qwen.h_embed = (uint16_t *)h;
    g_qwen.d_embed = (uint16_t *)d;
    printf("TQ_EMBED_HOST=1: bf16 embed table in pinned host memory, %.0f MiB\n", nbytes / 1048576.0);
    return 0;
}

static int read_f32(FILE *f, float **dst, size_t count, const char *what, int upload) {
    size_t nbytes = count * sizeof(float);
    g_qwen.f32_count++;
    return upload_payload(f, (void **)dst, nbytes, what, upload);
}

// [kt][nt][NW] -> [nt][kt][NW] (uint32 words), one thread per word.
__global__ void k_tq_exl3_repack_tiles(uint32_t *dst, const uint32_t *src, int KT, int NT, int NW) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)KT * NT * NW;
    if (i >= total) return;
    int wd = (int)(i % NW);
    long tile = i / NW;
    int nt = (int)(tile % NT), kt = (int)(tile / NT);
    dst[((long)nt * KT + kt) * NW + wd] = src[i];
}

// EXL3 record (TQ_FLAG_EXL3_WEIGHTS): <IIII> kbits, K, M, codebook; kbits > 0: suh fp16[K],
// svh fp16[M], trellis int16 [K/16][M/16][16*kbits] (re-ordered to [M/16][K/16] on the device
// at load); kbits == 0: bf16 [M][K].
static int read_exl3(FILE *f, tq_qmma_weight_t *w, int M, int K, const char *what, int upload) {
    uint32_t hdr[4];
    if (read_exact(f, hdr, sizeof(hdr), what) != 0) return -1;
    uint32_t kbits = hdr[0], rk = hdr[1], rm = hdr[2], cb = hdr[3];
    if (rk != (uint32_t)K || rm != (uint32_t)M) {
        fprintf(stderr, "%s: exl3 record shape K=%u M=%u, expected K=%d M=%d\n", what, rk, rm, K, M);
        return -1;
    }
    w->M = M; w->K = K; w->Mt = M / 16; w->Kt = K / 32;
    w->exl3 = 1;
    w->exl3_kbits = (int)kbits;
    w->exl3_codebook = (int)cb;
    g_qwen.qmma_count++;
    if (kbits == 0) {
        return upload_payload(f, (void **)&w->d_exl3_bf16, (size_t)M * K * 2, what, upload);
    }
    if (kbits > 8 || (M % 128) != 0 || (K % 128) != 0) {
        fprintf(stderr, "%s: exl3 kbits=%u shape %dx%d unsupported (need 1..8 bits, 128-aligned)\n",
                what, kbits, M, K);
        return -1;
    }
    if (cb != 2) {
        fprintf(stderr, "%s: exl3 codebook %u not implemented (mul1 only)\n", what, cb);
        return -1;
    }
    if (upload_payload(f, (void **)&w->d_exl3_suh, (size_t)K * 2, what, upload) != 0) return -1;
    if (upload_payload(f, (void **)&w->d_exl3_svh, (size_t)M * 2, what, upload) != 0) return -1;
    size_t tbytes = (size_t)(K / 16) * (size_t)(M / 16) * (size_t)(16 * kbits) * 2;
    if (upload_payload(f, (void **)&w->d_exl3_trellis, tbytes, what, upload) != 0) return -1;
    if (upload) {
        // exllamav3 stores tiles [kt][nt]; the streaming GEMM walks k for a fixed n-tile, so
        // re-order to [nt][kt] (each warp then reads one contiguous run of tiles).
        uint16_t *dst = NULL;
        if (cudaMalloc(&dst, tbytes) != cudaSuccess) { fprintf(stderr, "%s: exl3 repack alloc failed\n", what); return -1; }
        int KT = K / 16, NT = M / 16, NW = 8 * (int)kbits;
        long ntiles = (long)KT * NT;
        k_tq_exl3_repack_tiles<<<(unsigned)((ntiles * NW + 255) / 256), 256>>>(
            (uint32_t *)dst, (const uint32_t *)w->d_exl3_trellis, KT, NT, NW);
        if (cudaDeviceSynchronize() != cudaSuccess) { fprintf(stderr, "%s: exl3 repack failed\n", what); return -1; }
        cudaFree(w->d_exl3_trellis);
        w->d_exl3_trellis = dst;
    }
    return 0;
}

static int read_qmma(FILE *f, tq_qmma_weight_t *w, int M, int K, const char *what, int upload,
                     int want_sparse) {
    if ((M % 16) != 0 || (K % 32) != 0) {
        fprintf(stderr, "%s: unaligned QMMA shape %dx%d\n", what, M, K);
        return -1;
    }
    if (g_qwen.flags & TQ_FLAG_EXL3_WEIGHTS) return read_exl3(f, w, M, K, what, upload);
    w->M = M;
    w->K = K;
    w->Mt = M / 16;
    w->Kt = K / 32;
    w->block_scaled = (g_qwen.flags & TQ_FLAG_BLOCK_SCALED_QMMA) != 0;
    w->row_major = (g_qwen.flags & TQ_FLAG_BLOCK_SCALED_ROWMAJOR) != 0;
    w->word_major = (g_qwen.flags & TQ_FLAG_BLOCK_SCALED_QMMA_WORDMAJOR) != 0;
    w->e2m3 = (g_qwen.flags & TQ_FLAG_BLOCK_SCALED_E2M3) != 0;
    if (w->block_scaled) {
        uint32_t sr = 0, sc = 0;
        if (read_exact(f, &sr, sizeof(uint32_t), what) != 0) return -1;
        if (read_exact(f, &sc, sizeof(uint32_t), what) != 0) return -1;
        uint32_t expected_sr = (uint32_t)((M + 127) / 128);
        uint32_t expected_sc = (uint32_t)((K + 127) / 128);
        uint32_t expected_sc_k32 = (uint32_t)((K + 31) / 32);   // per-k32 weight scales (MX-style E2M1)
        if (sr != expected_sr || (sc != expected_sc && sc != expected_sc_k32)) {
            fprintf(stderr, "%s: invalid block scale shape %ux%u expected %ux%u\n",
                    what, sr, sc, expected_sr, expected_sc);
            return -1;
        }
        w->scale_rows = (int)sr;
        w->scale_cols = (int)sc;
        size_t scale_bytes = (size_t)sr * (size_t)sc * sizeof(float);
        if (upload_payload(f, (void **)&w->d_block_scale_inv, scale_bytes, what, upload) != 0) return -1;
        w->inv_scale = 1.0f;
    } else {
        float scale = 0.0f;
        if (read_exact(f, &scale, sizeof(float), what) != 0) return -1;
        if (scale == 0.0f) {
            fprintf(stderr, "%s: invalid zero scale\n", what);
            return -1;
        }
        w->inv_scale = 1.0f / scale;
    }
    size_t nbytes = (size_t)w->Mt * (size_t)w->Kt * (w->e2m3 ? 384u : 512u);
    g_qwen.qmma_count++;
    if (upload_payload(f, (void **)&w->d_A, nbytes, what, upload) != 0) return -1;

    // Optimistic sparse-FP4 draft record (appended by the converter for the
    // calibrated linear-attn projections when TQ_FLAG_SPARSE_24_E2M3 is set).
    if (want_sparse && w->e2m3 && (g_qwen.flags & TQ_FLAG_SPARSE_24_E2M3) && (K % 64) == 0) {
        uint32_t na = 0, nm = 0;
        if (read_exact(f, &na, sizeof(uint32_t), what) != 0) return -1;
        if (read_exact(f, &nm, sizeof(uint32_t), what) != 0) return -1;
        w->Kt64 = K / 64;
        uint32_t exp_na = (uint32_t)w->Mt * (uint32_t)w->Kt64 * 96u;
        uint32_t exp_nm = (uint32_t)w->Mt * (uint32_t)w->Kt64 * 32u;
        if (na != exp_na || nm != exp_nm) {
            fprintf(stderr, "%s: invalid sparse record sizes %u/%u expected %u/%u\n",
                    what, na, nm, exp_na, exp_nm);
            return -1;
        }
        if (upload_payload(f, (void **)&w->d_sparse_a, (size_t)na * 4u, what, upload) != 0) return -1;
        if (upload_payload(f, (void **)&w->d_sparse_meta, (size_t)nm * 4u, what, upload) != 0) return -1;
        w->has_sparse = 1;
    }
    return 0;
}

static void free_qmma(tq_qmma_weight_t *w) {
    if (w->d_A) cudaFree(w->d_A);
    if (w->d_block_scale_inv) cudaFree(w->d_block_scale_inv);
    if (w->d_sparse_a) cudaFree(w->d_sparse_a);
    if (w->d_sparse_meta) cudaFree(w->d_sparse_meta);
    if (w->d_exl3_trellis) cudaFree(w->d_exl3_trellis);
    if (w->d_exl3_suh) cudaFree(w->d_exl3_suh);
    if (w->d_exl3_svh) cudaFree(w->d_exl3_svh);
    if (w->d_exl3_bf16) cudaFree(w->d_exl3_bf16);
    memset(w, 0, sizeof(*w));
}
