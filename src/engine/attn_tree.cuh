// batched tree attention of the single-stream spec verify.
// Part of the forward_qwen.cu unity build: included in order, not a standalone header.

// ===== Batched tree attention (one launch per full-attn layer, all N nodes) =====
// Replaces the per-node k_tq_full_attn_decode loop + per-node KV-archive copies.
// Split in two launches so sibling/ancestor KV has no intra-grid ordering:
//   (1) kv_prep writes every node's normed+roped k and v straight into the
//       per-(layer,node) archive (the verify never touches the real KV cache;
//       the 1-read commit re-places the accepted path's KV from the archive).
//   (2) attn_tree attends: positions <= base come from the committed cache,
//       tree positions come from the node's ancestor chain in the archive.
// Bit-identical floats and reduction order vs k_tq_full_attn_decode (the cache
// round-trip is replaced by the archive holding the same values).

__global__ void k_tq_spec_attn_kv_prep(float *k_arch, float *v_arch,
                                       const float *k_proj_base, const float *v_proj_base,
                                       const uint16_t *k_norm_w, const int *pos_arr,
                                       int nkv, int hd, int kv_m,
                                       float eps, float rope_theta, const int *slot_map,
                                       uint8_t *k_dst8, float *k_dscale,
                                       uint8_t *v_dst8, float *v_dscale,
                                       uint8_t *k_dst4, float *k_dq4s, int kv_q4) {
    __shared__ float partial[8];
    __shared__ float k_sum_shared;
    __shared__ float rot[256];
    __shared__ float gsz[16];
    int kv_head = blockIdx.x;
    int node = blockIdx.y;
    int tid = threadIdx.x;
    if (kv_head >= nkv || tid >= hd) return;
    int pos = pos_arr[node];
    int arow = slot_map ? slot_map[node] : node;   // archive row (batched tree remap)
    const float *k_proj = k_proj_base + (size_t)node * kv_m;
    const float *v_proj = v_proj_base + (size_t)node * kv_m;
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
    if (kv_q4 && hd == 256) {
        // TQ_KV_Q4: K leaves this kernel rotated -- the fp32 archive, the int4
        // trunk write below and the later commit all live in the rotated space.
        rot[tid] = kv;
        __syncthreads();
        tq_fwht256(rot, tid);
        kv = rot[tid];
        __syncthreads();
    }
    if (k_dst4) {
        // int4-K destination (MTP trunk cache writes, TQ_KV_Q4): V row stays E4M3.
        float vv = v_proj[kv_head * hd + tid];
        __syncthreads();
        float vscl = tq_kv_block_scale(vv, partial, &k_sum_shared);
        v_dst8[(size_t)arow * kv_m + kv_head * hd + tid] = tq_float_to_e4m3(vv / vscl);
        if (tid == 0) v_dscale[(size_t)arow * nkv + kv_head] = vscl;
        uint8_t code = tq_k_q4_quant(kv, tid, gsz);
        unsigned hi4 = __shfl_down_sync(0xffffffffu, (unsigned)code, 1);
        if ((tid & 1) == 0)
            k_dst4[(size_t)arow * (kv_m >> 1) + kv_head * (hd >> 1) + (tid >> 1)] =
                (uint8_t)(code | ((hi4 & 15u) << 4));
        if (tid < 16)
            q4s_put(k_dq4s, ((size_t)arow * nkv + kv_head) * 16 + tid, gsz[tid]);
        return;
    }
    if (k_dst8) {
        // FP8 destination (MTP trunk cache writes): quantize instead of the fp32
        // archive store; arow is the absolute cache slot via slot_map.
        float vv = v_proj[kv_head * hd + tid];
        __syncthreads();
        float kscl = tq_kv_block_scale(kv, partial, &k_sum_shared);
        k_dst8[(size_t)arow * kv_m + kv_head * hd + tid] = tq_float_to_e4m3(kv / kscl);
        __syncthreads();
        float vscl = tq_kv_block_scale(vv, partial, &k_sum_shared);
        v_dst8[(size_t)arow * kv_m + kv_head * hd + tid] = tq_float_to_e4m3(vv / vscl);
        if (tid == 0) {
            k_dscale[(size_t)arow * nkv + kv_head] = kscl;
            v_dscale[(size_t)arow * nkv + kv_head] = vscl;
        }
        return;
    }
    k_arch[(size_t)arow * kv_m + kv_head * hd + tid] = kv;
    v_arch[(size_t)arow * kv_m + kv_head * hd + tid] = v_proj[kv_head * hd + tid];
}

// FP8 commit: quantize archived fp32 KV rows into the E4M3 prefix cache at the
// accepted path positions (replaces the fp32 cudaMemcpyAsync row copies; a kernel
// keeps the path CUDA-graph-capture-safe). rows[2j]=arch node, rows[2j+1]=cache pos.
// Grid (nkv, m), block hd: one (kv_head, path row) section per block.
__global__ void k_tq_spec_commit_kv_fp8(uint8_t *k8, float *ksc, uint8_t *v8, float *vsc,
                                        const float *k_arch, const float *v_arch,
                                        const int *rows, int kv_m, int nkv, int hd) {
    __shared__ float sm8[8];
    __shared__ float s_scale;
    int kv_head = blockIdx.x, j = blockIdx.y, tid = threadIdx.x;
    if (kv_head >= nkv || tid >= hd) return;
    int node = rows[2 * j], pos = rows[2 * j + 1];
    float kvv = k_arch[(size_t)node * kv_m + kv_head * hd + tid];
    float kscl = tq_kv_block_scale(kvv, sm8, &s_scale);
    k8[((size_t)pos * nkv + kv_head) * hd + tid] = tq_float_to_e4m3(kvv / kscl);
    __syncthreads();
    float vvv = v_arch[(size_t)node * kv_m + kv_head * hd + tid];
    float vscl = tq_kv_block_scale(vvv, sm8, &s_scale);
    v8[((size_t)pos * nkv + kv_head) * hd + tid] = tq_float_to_e4m3(vvv / vscl);
    if (tid == 0) {
        ksc[(size_t)pos * nkv + kv_head] = kscl;
        vsc[(size_t)pos * nkv + kv_head] = vscl;
    }
}

// int4-K commit (TQ_KV_Q4): the archived K rows are ALREADY rotated (kv_prep),
// so this just group-quantizes them into the int4 prefix cache; V keeps the
// E4M3 path. Same grid/layout contract as k_tq_spec_commit_kv_fp8.
__global__ void k_tq_spec_commit_kv_q4(uint8_t *k4, float *kq4s, uint8_t *v8, float *vsc,
                                       const float *k_arch, const float *v_arch,
                                       const int *rows, int kv_m, int nkv, int hd) {
    __shared__ float sm8[8];
    __shared__ float s_scale;
    __shared__ float gsz[16];
    int kv_head = blockIdx.x, j = blockIdx.y, tid = threadIdx.x;
    if (kv_head >= nkv || tid >= hd) return;
    int node = rows[2 * j], pos = rows[2 * j + 1];
    float vvv = v_arch[(size_t)node * kv_m + kv_head * hd + tid];
    float vscl = tq_kv_block_scale(vvv, sm8, &s_scale);
    v8[((size_t)pos * nkv + kv_head) * hd + tid] = tq_float_to_e4m3(vvv / vscl);
    if (tid == 0) vsc[(size_t)pos * nkv + kv_head] = vscl;
    float kvv = k_arch[(size_t)node * kv_m + kv_head * hd + tid];
    __syncthreads();
    uint8_t code = tq_k_q4_quant(kvv, tid, gsz);
    unsigned hi4 = __shfl_down_sync(0xffffffffu, (unsigned)code, 1);
    if ((tid & 1) == 0)
        k4[((size_t)pos * nkv + kv_head) * (hd >> 1) + (tid >> 1)] =
            (uint8_t)(code | ((hi4 & 15u) << 4));
    if (tid < 16)
        q4s_put(kq4s, ((size_t)pos * nkv + kv_head) * 16 + tid, gsz[tid]);
}

__global__ void k_tq_spec_attn_tree(float *out, const float *q_proj_base,
                                    const uint16_t *q_norm_w,
                                    const float *k_cache, const float *v_cache,
                                    const uint8_t *k_cache8, const float *k_scale,
                                    const uint8_t *v_cache8, const float *v_scale,
                                    const uint8_t *k_cache4, const float *k_q4s,
                                    const float *k_arch, const float *v_arch,
                                    const int *pos_arr, const int *anc,
                                    int nh, int nkv, int hd, int kv_m,
                                    int q_stride, int out_stride,
                                    float eps, float rope_theta, int base_in,
                                           float *scores_g, int sstride) {
    __shared__ float partial[8];
    __shared__ float q_sum_shared;
    float *scores = scores_g + (size_t)((size_t)blockIdx.y * gridDim.x + blockIdx.x) * sstride;
    __shared__ float max_score_shared;
    __shared__ float denom_shared;
    __shared__ float qn[256];
    int head = blockIdx.x;
    int node = blockIdx.y;
    int tid = threadIdx.x;
    if (head >= nh || tid >= hd) return;
    int pos = pos_arr[node];
    // committed prefix is 0..base; tree position p maps to anc[node][p-base-1].
    // base_in < 0 derives it from node 0 (the verify tree root at base+1, graph-safe).
    int base = base_in >= 0 ? base_in : pos_arr[0] - 1;
    const float *q_proj = q_proj_base + (size_t)node * q_stride;
    const int *anc_n = anc + node * TQ_SPEC_MAX_N;
    int group = nh / nkv;
    int kv_head = head / group;
    int rotary_dim = 64;
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
    __syncthreads();
    if (k_cache4 && hd == 256)
        tq_fwht256(qn, tid);   // q4 score space is rotated (arch K rows too)

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
                                k_arch + (size_t)kv_head * hd, v_arch + (size_t)kv_head * hd,
                                (size_t)kv_m, anc_n, base, pos, hd, tid);
    } else {
        float max_score = -3.402823466e38f;
        for (int t = 0; t <= pos; t++) {
            const float *krow = (t <= base)
                ? k_cache + ((size_t)t * nkv + kv_head) * hd
                : k_arch + (size_t)anc_n[t - base - 1] * kv_m + kv_head * hd;
            float dot = qn[tid] * krow[tid];
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
            const float *vrow = (t <= base)
                ? v_cache + ((size_t)t * nkv + kv_head) * hd
                : v_arch + (size_t)anc_n[t - base - 1] * kv_m + kv_head * hd;
            float p = expf(scores[t] - max_score) / denom_shared;
            acc += p * vrow[tid];
        }
        ratio = acc;
    }
    float gate = q_proj[head * (2 * hd) + hd + tid];
    out[(size_t)node * out_stride + head * hd + tid] = ratio * (1.0f / (1.0f + expf(-gate)));
}

// __launch_bounds__(1024, 1) on the whole stripe-reduce rmsnorm/argmax family below:
// every kernel is always launched at exactly 1024 threads/block, and on nvcc 12.9
// ptxas allocates up to 80 regs/thread for these -- 80 x 1024 exceeds the 64K-register
// SM budget, so each launch failed at runtime (spec decode died after the first
// token). The bound forces ptxas into the 64-reg budget on any toolchain; excess
// pressure becomes spills instead of dead launches.
__global__ void __launch_bounds__(1024, 1)
k_tq_qwen_rmsnorm(float *out, const float *x, const uint16_t *weight, int N, float eps) {
    __shared__ float smem[32];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;

    float sum_sq = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float v = x[i];
        sum_sq += v * v;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }
    if (lane == 0) smem[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        sum_sq = (lane < (blockDim.x + 31) / 32) ? smem[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        }
        if (lane == 0) smem[0] = rsqrtf(sum_sq / (float)N + eps);
    }
    __syncthreads();

    float rms_inv = smem[0];
    // Each block computes the full (identical) sum above, then scales only its
    // own output stripe. This spreads the normalize pass across gridDim.x SMs
    // instead of one, while keeping the result bit-identical to the grid=1 path.
    int stripe = (N + gridDim.x - 1) / gridDim.x;
    int begin = blockIdx.x * stripe;
    int end = begin + stripe;
    if (end > N) end = N;
    for (int i = begin + tid; i < end; i += blockDim.x) {
        float w = 1.0f + tq_bf16_to_float(weight[i]);
        out[i] = x[i] * rms_inv * w;
    }
}

// Programmatic dependent launch (paged decode step, TQ_PDL): the kernels of the step call trigger + wait first
// (or, the EXL3 GEMM, wait before its first dependent access); both are no-ops for a launch without the attribute.
// wait = the predecessor grid has completed and its writes are visible; trigger = dependents may launch now.
__device__ __forceinline__ void tq_pdl_wait(void) { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void tq_pdl_trigger(void) { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ void tq_pdl_enter(void) { tq_pdl_trigger(); tq_pdl_wait(); }

// Host side: the paged decode step sets the flags. TQ_PDL=1 launches bfrag and the EXL3 GEMM with the attribute
// (g_pdl_on); TQ_PDL=2 also the step's other kernels (g_pdl_small), which measured slower at 32K: their CTAs,
// resident early and waiting, cost more than the launch latency they hide.
static int g_pdl_on = 0, g_pdl_small = 0;
static int tq_pdl_enabled(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("TQ_PDL"); v = e ? atoi(e) : 0; if (v < 0 || v > 2) v = 0; }
    return v;
}
template <typename... KArgs, typename... Args>
static void tq_launch(bool pdl, void (*kern)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t st,
                      Args... args) {
    if (!pdl) { kern<<<grid, block, smem, st>>>(args...); return; }
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid; cfg.blockDim = block; cfg.dynamicSmemBytes = smem; cfg.stream = st;
    cfg.attrs = attr; cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, kern, static_cast<KArgs>(args)...);
}

// CUDA graphs for the paged decode step (TQ_GRAPH=1, session 10 "decode-graph"): the step's kernel sequence is
// captured once per shape class (clients N, GQA key-split count S, MTP columns, mode flags) and replayed; the
// per-step inputs (tokens, slot ids, positions, block tables, encoded lengths, MTP control rows) reach the kernels
// through device buffers uploaded before the replay, so the replayed kernels run with exactly the eager step's
// parameters (bitwise identical). Every device buffer a captured step may reference is owned by the engine and
// (re)allocated only in a handful of places; each of them bumps g_graph_gen, which drops every cached graph.
// TQ_GRAPH=1: the paged decode step as a graph; TQ_GRAPH=2: also the verify wave and the MTP draft (measured: no
// gain over the eager spec round, and the verify key varies with the wave's column count -> more captures).
// DEFAULT 1 since 2026-10-17 (owner's decision (g), with TQ_GEMM_FORK: -0.35 ms per step at every N / context,
// bitwise); TQ_GRAPH=0 = the eager launch sequence.
static int tq_graph_enabled(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("TQ_GRAPH"); v = (e && *e) ? atoi(e) : 1; if (v < 0) v = 0; }
    return v;
}
static int tq_graph_debug(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("TQ_GRAPH_DEBUG"); v = (e && *e) ? atoi(e) : 0; }
    return v;
}
static unsigned g_graph_gen = 1;
static void graph_cache_clear(void);
static void graph_invalidate(void) { g_graph_gen++; graph_cache_clear(); }

// Batched rmsnorm over `nodes` contiguous vectors (vector v at v*N). Grid = (stripes,
// nodes), node = blockIdx.y. Bit-identical to k_tq_qwen_rmsnorm per node (same .x stripe
// count); one launch replaces N for the Step-5 batched verify (cuts launch overhead).
__global__ void __launch_bounds__(1024, 1)
k_tq_qwen_rmsnorm_b(float *out, const float *x, const uint16_t *weight, int N, float eps) {
    tq_pdl_enter();
    __shared__ float smem[32];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    const float *xn = x + (size_t)blockIdx.y * N;
    float *outn = out + (size_t)blockIdx.y * N;
    float sum_sq = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) { float v = xn[i]; sum_sq += v * v; }
    for (int offset = 16; offset > 0; offset >>= 1) sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    if (lane == 0) smem[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        sum_sq = (lane < (blockDim.x + 31) / 32) ? smem[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        if (lane == 0) smem[0] = rsqrtf(sum_sq / (float)N + eps);
    }
    __syncthreads();
    float rms_inv = smem[0];
    int stripe = (N + gridDim.x - 1) / gridDim.x;
    int begin = blockIdx.x * stripe;
    int end = begin + stripe;
    if (end > N) end = N;
    for (int i = begin + tid; i < end; i += blockDim.x) {
        float w = 1.0f + tq_bf16_to_float(weight[i]);
        outn[i] = xn[i] * rms_inv * w;
    }
}

// Fused add-residual + RMSNorm: computes resid[i] = a[i] + b[i], then
// out[i] = rmsnorm(resid, weight). Replaces separate add_vec_b + rmsnorm_b.
__global__ void __launch_bounds__(1024, 1)
k_tq_add_rmsnorm_b(float *out, float *resid, const float *a, const float *b,
                                   const uint16_t *weight, int N, float eps) {
    tq_pdl_enter();
    __shared__ float smem[32];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    const float *an = a + (size_t)blockIdx.y * N;
    const float *bn = b + (size_t)blockIdx.y * N;
    float *rn = resid + (size_t)blockIdx.y * N;
    float *outn = out + (size_t)blockIdx.y * N;
    float sum_sq = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float v = an[i] + bn[i];
        rn[i] = v;
        sum_sq += v * v;
    }
    for (int offset = 16; offset > 0; offset >>= 1) sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    if (lane == 0) smem[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        sum_sq = (lane < (blockDim.x + 31) / 32) ? smem[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        if (lane == 0) smem[0] = rsqrtf(sum_sq / (float)N + eps);
    }
    __syncthreads();
    float rms_inv = smem[0];
    int stripe = (N + gridDim.x - 1) / gridDim.x;
    int begin = blockIdx.x * stripe;
    int end = begin + stripe;
    if (end > N) end = N;
    for (int i = begin + tid; i < end; i += blockDim.x) {
        float w = 1.0f + tq_bf16_to_float(weight[i]);
        outn[i] = rn[i] * rms_inv * w;
    }
}

// rmsnorm_b variant that also leaves each block's stripe absmax of the output in
// amax[node * gridDim.x + blockIdx.x] (max over stripes = column absmax, exact) so
// the persistent layer kernel can skip its absmax phase. Same norm arithmetic.
__global__ void __launch_bounds__(1024, 1)
k_tq_qwen_rmsnorm_b_amax(float *out, const float *x, const uint16_t *weight,
                                         int N, float eps, float *amax) {
    __shared__ float smem[32];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    const float *xn = x + (size_t)blockIdx.y * N;
    float *outn = out + (size_t)blockIdx.y * N;
    float sum_sq = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) { float v = xn[i]; sum_sq += v * v; }
    for (int offset = 16; offset > 0; offset >>= 1) sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    if (lane == 0) smem[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        sum_sq = (lane < (blockDim.x + 31) / 32) ? smem[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        if (lane == 0) smem[0] = rsqrtf(sum_sq / (float)N + eps);
    }
    __syncthreads();
    float rms_inv = smem[0];
    int stripe = (N + gridDim.x - 1) / gridDim.x;
    int begin = blockIdx.x * stripe;
    int end = begin + stripe;
    if (end > N) end = N;
    float local_max = 0.0f;
    for (int i = begin + tid; i < end; i += blockDim.x) {
        float w = 1.0f + tq_bf16_to_float(weight[i]);
        float v = xn[i] * rms_inv * w;
        outn[i] = v;
        float a = fabsf(v);
        if (a > local_max) local_max = a;
    }
    __syncthreads();
    for (int offset = 16; offset > 0; offset >>= 1) {
        float o = __shfl_down_sync(0xffffffff, local_max, offset);
        if (o > local_max) local_max = o;
    }
    if (lane == 0) smem[warp_id] = local_max;
    __syncthreads();
    if (warp_id == 0) {
        local_max = (lane < (blockDim.x + 31) / 32) ? smem[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            float o = __shfl_down_sync(0xffffffff, local_max, offset);
            if (o > local_max) local_max = o;
        }
        if (lane == 0) amax[blockIdx.y * gridDim.x + blockIdx.x] = local_max;
    }
}

// add_rmsnorm_b variant with the stripe-absmax side output AND the b operand read
// as an inline k-split reduction of GEMV partials (sum over splits / factors[node]),
// folding the persistent kernel's out-projection reduce phase into this kernel.
__global__ void __launch_bounds__(1024, 1)
k_tq_add_rmsnorm_b_amax_red(float *out, float *resid, const float *a,
                                            const float *partials, const float *factors,
                                            int ks, const uint16_t *weight, int N, float eps,
                                            float *amax) {
    __shared__ float smem[32];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    const float *an = a + (size_t)blockIdx.y * N;
    float *rn = resid + (size_t)blockIdx.y * N;
    float *outn = out + (size_t)blockIdx.y * N;
    size_t total = (size_t)gridDim.y * N;
    size_t bbase = (size_t)blockIdx.y * N;
    float fac = factors[blockIdx.y];
    float sum_sq = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float bsum = 0.0f;
        for (int s = 0; s < ks; s++) bsum += partials[(size_t)s * total + bbase + i];
        float v = an[i] + bsum / fac;
        rn[i] = v;
        sum_sq += v * v;
    }
    for (int offset = 16; offset > 0; offset >>= 1) sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    if (lane == 0) smem[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        sum_sq = (lane < (blockDim.x + 31) / 32) ? smem[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        if (lane == 0) smem[0] = rsqrtf(sum_sq / (float)N + eps);
    }
    __syncthreads();
    float rms_inv = smem[0];
    int stripe = (N + gridDim.x - 1) / gridDim.x;
    int begin = blockIdx.x * stripe;
    int end = begin + stripe;
    if (end > N) end = N;
    float local_max = 0.0f;
    for (int i = begin + tid; i < end; i += blockDim.x) {
        float w = 1.0f + tq_bf16_to_float(weight[i]);
        float v = rn[i] * rms_inv * w;
        outn[i] = v;
        float av = fabsf(v);
        if (av > local_max) local_max = av;
    }
    __syncthreads();
    for (int offset = 16; offset > 0; offset >>= 1) {
        float o = __shfl_down_sync(0xffffffff, local_max, offset);
        if (o > local_max) local_max = o;
    }
    if (lane == 0) smem[warp_id] = local_max;
    __syncthreads();
    if (warp_id == 0) {
        local_max = (lane < (blockDim.x + 31) / 32) ? smem[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            float o = __shfl_down_sync(0xffffffff, local_max, offset);
            if (o > local_max) local_max = o;
        }
        if (lane == 0) amax[blockIdx.y * gridDim.x + blockIdx.x] = local_max;
    }
}

// add_rmsnorm_b variant with the same stripe-absmax side output.
__global__ void __launch_bounds__(1024, 1)
k_tq_add_rmsnorm_b_amax(float *out, float *resid, const float *a, const float *b,
                                        const uint16_t *weight, int N, float eps, float *amax) {
    __shared__ float smem[32];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    const float *an = a + (size_t)blockIdx.y * N;
    const float *bn = b + (size_t)blockIdx.y * N;
    float *rn = resid + (size_t)blockIdx.y * N;
    float *outn = out + (size_t)blockIdx.y * N;
    float sum_sq = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float v = an[i] + bn[i];
        rn[i] = v;
        sum_sq += v * v;
    }
    for (int offset = 16; offset > 0; offset >>= 1) sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    if (lane == 0) smem[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        sum_sq = (lane < (blockDim.x + 31) / 32) ? smem[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        if (lane == 0) smem[0] = rsqrtf(sum_sq / (float)N + eps);
    }
    __syncthreads();
    float rms_inv = smem[0];
    int stripe = (N + gridDim.x - 1) / gridDim.x;
    int begin = blockIdx.x * stripe;
    int end = begin + stripe;
    if (end > N) end = N;
    float local_max = 0.0f;
    for (int i = begin + tid; i < end; i += blockDim.x) {
        float w = 1.0f + tq_bf16_to_float(weight[i]);
        float v = rn[i] * rms_inv * w;
        outn[i] = v;
        float av = fabsf(v);
        if (av > local_max) local_max = av;
    }
    __syncthreads();
    for (int offset = 16; offset > 0; offset >>= 1) {
        float o = __shfl_down_sync(0xffffffff, local_max, offset);
        if (o > local_max) local_max = o;
    }
    if (lane == 0) smem[warp_id] = local_max;
    __syncthreads();
    if (warp_id == 0) {
        local_max = (lane < (blockDim.x + 31) / 32) ? smem[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            float o = __shfl_down_sync(0xffffffff, local_max, offset);
            if (o > local_max) local_max = o;
        }
        if (lane == 0) amax[blockIdx.y * gridDim.x + blockIdx.x] = local_max;
    }
}

// Simple BF16-weight x FP32-activation GEMV for the full-precision MTP head
// projections. Row-major weight [M, K]: out[m] = sum_k bf16->f32(w[m*K+k]) * x[k].
// One block per output row; correctness over speed (single layer, not perf-critical).
__global__ void k_tq_bf16_gemv(float *out, const uint16_t *w, const float *x, int M, int K) {
    __shared__ float smem[256];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= M) return;
    const uint16_t *wr = w + (size_t)row * K;
    float sum = 0.0f;
    for (int k = tid; k < K; k += blockDim.x) sum += tq_bf16_to_float(wr[k]) * x[k];
    smem[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out[row] = smem[0];
}

__global__ void k_tq_debug_qmma_gemv(float *out, const uint8_t *a, int M, int K, int Kt,
                                     float inv_scale, const float *x) {
    __shared__ float smem[256];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= M) return;

    float sum = 0.0f;
    for (int col = tid; col < K; col += blockDim.x) {
        uint8_t q = tq_qmma_a_byte(a, Kt, row, col);
        sum += tq_e4m3_to_float(q) * inv_scale * x[col];
    }
    smem[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out[row] = smem[0];
}

__global__ void k_tq_debug_qmma_gemv_block_scaled(float *out, const uint8_t *a, int M, int K, int Kt,
                                                  const float *scale_inv, int scale_cols,
                                                  const float *x) {
    __shared__ float smem[256];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= M) return;

    int scale_row = row >> 7;
    float sum = 0.0f;
    for (int col = tid; col < K; col += blockDim.x) {
        uint8_t q = tq_qmma_a_byte(a, Kt, row, col);
        float s = scale_inv[scale_row * scale_cols + (col >> 7)];
        sum += tq_e4m3_to_float(q) * s * x[col];
    }
    smem[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out[row] = smem[0];
}

__global__ void k_tq_debug_qmma_gemv_block_scaled_rowmajor(float *out, const uint8_t *a, int M, int K,
                                                           const float *scale_inv, int scale_cols,
                                                           const float *x) {
    __shared__ float smem[256];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= M) return;

    int scale_row = row >> 7;
    const uint8_t *row_ptr = a + (size_t)row * K;
    float sum = 0.0f;
    for (int col = tid; col < K; col += blockDim.x) {
        float s = scale_inv[scale_row * scale_cols + (col >> 7)];
        sum += tq_e4m3_to_float(row_ptr[col]) * s * x[col];
    }
    smem[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out[row] = smem[0];
}

__global__ void k_tq_lm_head_argmax_block_scaled(float *block_vals, int *block_ids,
                                                 const uint8_t *a, int V, int K, int Kt,
                                                 const float *scale_inv, int scale_cols,
                                                 const float *x) {
    __shared__ float smem[256];
    __shared__ float best_val_shared;
    __shared__ int best_id_shared;
    int tid = threadIdx.x;
    if (tid == 0) {
        best_val_shared = -3.402823466e38f;
        best_id_shared = 0;
    }
    __syncthreads();

    for (int row = blockIdx.x; row < V; row += gridDim.x) {
        int scale_row = row >> 7;
        float sum = 0.0f;
        for (int col = tid; col < K; col += blockDim.x) {
            uint8_t q = tq_qmma_a_byte(a, Kt, row, col);
            float s = scale_inv[scale_row * scale_cols + (col >> 7)];
            sum += tq_e4m3_to_float(q) * s * x[col];
        }
        smem[tid] = sum;
        __syncthreads();

        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) smem[tid] += smem[tid + stride];
            __syncthreads();
        }
        if (tid == 0 && smem[0] > best_val_shared) {
            best_val_shared = smem[0];
            best_id_shared = row;
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_vals[blockIdx.x] = best_val_shared;
        block_ids[blockIdx.x] = best_id_shared;
    }
}

__global__ void k_tq_lm_head_argmax_block_scaled_rowmajor(float *block_vals, int *block_ids,
                                                          const uint8_t *a, int V, int K,
                                                          const float *scale_inv, int scale_cols,
                                                          const float *x) {
    __shared__ float smem[256];
    __shared__ float best_val_shared;
    __shared__ int best_id_shared;
    int tid = threadIdx.x;
    if (tid == 0) {
        best_val_shared = -3.402823466e38f;
        best_id_shared = 0;
    }
    __syncthreads();

    for (int row = blockIdx.x; row < V; row += gridDim.x) {
        int scale_row = row >> 7;
        const uint8_t *row_ptr = a + (size_t)row * K;
        float sum = 0.0f;
        for (int col = tid; col < K; col += blockDim.x) {
            float s = scale_inv[scale_row * scale_cols + (col >> 7)];
            sum += tq_e4m3_to_float(row_ptr[col]) * s * x[col];
        }
        smem[tid] = sum;
        __syncthreads();

        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) smem[tid] += smem[tid + stride];
            __syncthreads();
        }
        if (tid == 0 && smem[0] > best_val_shared) {
            best_val_shared = smem[0];
            best_id_shared = row;
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_vals[blockIdx.x] = best_val_shared;
        block_ids[blockIdx.x] = best_id_shared;
    }
}

__global__ void k_tq_absmax_scale(float *d_scale, const float *x, int K) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    float local = 0.0f;
    for (int i = tid; i < K; i += blockDim.x) {
        float v = fabsf(x[i]);
        if (v > local) local = v;
    }
    smem[tid] = local;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride && smem[tid + stride] > smem[tid]) smem[tid] = smem[tid + stride];
        __syncthreads();
    }
    if (tid == 0) d_scale[0] = smem[0] > 0.0f ? 448.0f / smem[0] : 1.0f;
}

__device__ __forceinline__ uint8_t tq_float_to_e4m3(float val) {
    // E4M3FN activation encoder: full normal range up to 448 (biased exponent up to
    // 15), round-to-nearest-even (rintf), subnormals in exponent field 0, and
    // saturation to 448 (never emits the NaN code S.1111.111). Mirrors the fixed
    // weight encoder in tools/convert.py:float_to_e4m3. (Previously the biased
    // exponent was clipped to 14 -> max ~240, clipping large QMMA.SF activations.)
    float ax = fabsf(val);
    uint8_t sign = val < 0.0f ? 0x80 : 0;
    if (!isfinite(ax)) ax = 448.0f;
    if (ax < 0x1p-6f) {                          // subnormal: step 2^-9, mantissa 0..8
        int m = (int)rintf(ax * 512.0f);         // ax / 2^-9, round-half-to-even
        if (m < 0) m = 0;
        if (m > 8) m = 8;                        // m==8 carries to smallest normal (0x08)
        return sign | (uint8_t)m;
    }
    int e = (int)floorf(log2f(ax));              // unbiased exponent
    int biased = e + 7;
    int m = (int)rintf(ax / ldexpf(1.0f, e) * 8.0f) - 8;   // RNE mantissa, 0..8
    if (m >= 8) { m = 0; biased += 1; }          // mantissa overflow -> bump exponent
    if (biased > 15 || (biased == 15 && m > 6)) { biased = 15; m = 6; }   // saturate to 448
    if (biased < 1) biased = 1;
    if (m < 0) m = 0;
    else if (m > 7) m = 7;
    return sign | ((uint8_t)biased << 3) | (uint8_t)m;
}

__device__ __forceinline__ uint8_t tq_float_to_e5m2(float val) {
    // E5M2 activation encoder: 1 sign + 5 exponent + 2 mantissa, bias 15.
    // Max normal = 57344 (= 1.75 × 2^15). Saturates to ±57344 (no NaN).
    float ax = fabsf(val);
    uint8_t sign = val < 0.0f ? 0x80 : 0;
    if (!isfinite(ax)) ax = 57344.0f;
    if (ax < 0x1p-14f) {                          // subnormal: step 2^-16, mantissa 0..3
        int m = (int)rintf(ax * 65536.0f);        // ax / 2^-16
        if (m < 0) m = 0;
        if (m > 3) m = 3;                         // max subnormal = 0.75 × 2^-14
        return sign | (uint8_t)m;
    }
    int e = (int)floorf(log2f(ax));               // unbiased exponent
    int biased = e + 15;
    int m = (int)rintf(ax / ldexpf(1.0f, e) * 4.0f) - 4;   // RNE mantissa, 0..4
    if (m >= 4) { m = 0; biased += 1; }           // mantissa overflow -> bump exponent
    if (biased > 30 || (biased == 30 && m > 3)) { biased = 30; m = 3; }   // saturate to 57344
    if (biased < 1) biased = 1;
    if (m < 0) m = 0;
    else if (m > 3) m = 3;
    return sign | ((uint8_t)biased << 2) | (uint8_t)m;
}

__global__ void k_tq_quantize_bfrag(uint8_t *b, const float *x, int K, const float *d_scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K) return;
    int tile = idx >> 5;
    int k_in_tile = idx & 31;
    b[tile * 256 + k_in_tile] = tq_float_to_e4m3(x[idx] * d_scale[0]);
}

__global__ void k_tq_absmax_scale_inv_blocks(float *scale_inv, const float *x, int K) {
    __shared__ float smem[128];
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
        scale_inv[block] = exp2f(ceilf(log2f(raw)));
    }
}

__global__ void k_tq_quantize_bfrag_pow2(uint8_t *b, const float *x, int K, const float *scale_inv) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K) return;
    int tile = idx >> 5;
    int k_in_tile = idx & 31;
    float s = scale_inv[idx >> 7];
    b[tile * 256 + k_in_tile] = tq_float_to_e4m3(x[idx] / s);
}

// Residual-lo of the dense B-fragment (same scale as the hi prepare): writes
// e4m3(x/s - decode(e4m3(x/s))) into the activation bytes (rest stays zeroed by
// the prior hi prepare). Lets a dense GEMV run a 2xe4m3 (~BF16) activation pass.
__global__ void k_tq_quantize_bfrag_residual_lo(uint8_t *b, const float *x, int K,
                                                const float *scale_inv) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K) return;
    int tile = idx >> 5;
    int k_in_tile = idx & 31;
    float s = scale_inv[idx >> 7];
    float xs = x[idx] / s;
    uint8_t hi = tq_float_to_e4m3(xs);
    b[tile * 256 + k_in_tile] = tq_float_to_e4m3(xs - tq_e4m3_to_float(hi));
}

__global__ void k_tq_quantize_bfrag_pow2_fused(uint8_t *b, float *scale_inv, const float *x, int K) {
    __shared__ float smem[128];
    __shared__ float block_scale_inv;
    int block = blockIdx.x;
    int tid = threadIdx.x;
    int base = block * 128;
    int base_tile = base >> 5;
    int remaining = K - base;
    int live_tiles = remaining >= 128 ? 4 : (remaining + 31) / 32;

    for (int i = tid; i < live_tiles * 256; i += blockDim.x) {
        b[base_tile * 256 + i] = 0;
    }

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
        int tile = idx >> 5;
        int k_in_tile = idx & 31;
        b[tile * 256 + k_in_tile] = tq_float_to_e4m3(x[idx] / block_scale_inv);
    }
}

// N=k B-fragment: pack `nvar` activation vectors (x laid out as nvar contiguous
// K-length vectors) into the first nvar columns of the m16n8 B operand. Column n
// lives at byte 32*n within each 256-byte K-tile (lanes 4n..4n+3 carry it).
// Per-column global absmax for nk prescaling. One block per column, 256 threads.
// Writes col_absmax[col] = max(|x[col*K .. col*K+K-1]|).
__global__ void k_tq_nk_col_absmax(const float *x, float *col_absmax, int K, int nvar) {
    int col = blockIdx.x;
    if (col >= nvar) return;
    int tid = threadIdx.x;
    __shared__ float smem[256];
    const float *xv = x + (size_t)col * K;
    float local_max = 0.0f;
    for (int i = tid; i < K; i += blockDim.x) {
        float a = fabsf(xv[i]);
        if (a > local_max) local_max = a;
    }
    smem[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s && smem[tid + s] > smem[tid]) smem[tid] = smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) col_absmax[col] = smem[0];
}

// Prescale nk columns in-place so all have the same global absmax. On entry
// factors[col] = col_absmax (from k_tq_nk_col_absmax); on exit factors[col] =
// global_max / col_absmax (the multiplicative factor applied). One block per column.
__global__ void k_tq_nk_prescale(float *x, float *factors, int K, int nvar) {
    int col = blockIdx.x;
    if (col >= nvar) return;
    int tid = threadIdx.x;
    float global_max = 0.0f;
    for (int v = 0; v < nvar; v++) {
        float a = factors[v];
        if (a > global_max) global_max = a;
    }
    if (global_max == 0.0f) {
        if (tid == 0) factors[col] = 1.0f;
        return;
    }
    float my_absmax = factors[col];
    float f = (my_absmax > 0.0f) ? global_max / my_absmax : 1.0f;
    float *xv = x + (size_t)col * K;
    for (int i = tid; i < K; i += blockDim.x)
        xv[i] *= f;
    if (tid == 0) factors[col] = f;
}

// Post-GEMV correction: divide each column's output by its prescale factor so the
// final result equals the un-prescaled GEMV. out is column-major [nvar x M].
__global__ void k_tq_nk_postcorrect(float *out, const float *factors, int M, int nvar) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nvar * M) return;
    int col = idx / M;
    out[idx] /= factors[col];
}

// E4M3 quantization with shared pow2 scale (max across columns), matching the
// per-token path. After RMSNorm the per-column absmax values are similar (~3-5),
// so the shared scale loses at most 1 pow2 step; with E4M3's 3 mantissa bits
// this still provides better precision than E5M2's 2 mantissa bits.
__global__ void k_tq_quantize_bfrag_pow2_fused_nk(uint8_t *b, float *scale_inv,
                                                  const float *x, int K, int nvar) {
    __shared__ float smem[128];
    __shared__ float block_scale_inv;
    int block = blockIdx.x;
    int tid = threadIdx.x;
    int base = block * 128;
    int base_tile = base >> 5;
    int remaining = K - base;
    int live_tiles = remaining >= 128 ? 4 : (remaining + 31) / 32;

    size_t gstride = (size_t)((K + 31) / 32) * 256;
    for (int i = tid; i < live_tiles * 256; i += blockDim.x) {
        b[base_tile * 256 + i] = 0;
        if (nvar > 8) b[gstride + base_tile * 256 + i] = 0;
    }

    float local = 0.0f;
    for (int v = 0; v < nvar; v++) {
        const float *xv = x + (size_t)v * K;
        for (int i = tid; i < 128 && base + i < K; i += blockDim.x) {
            float a = fabsf(xv[base + i]);
            if (a > local) local = a;
        }
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

    for (int v = 0; v < nvar; v++) {
        const float *xv = x + (size_t)v * K;
        uint8_t *bg = b + (v >> 3) * gstride;
        int vc = v & 7;
        for (int i = tid; i < 128 && base + i < K; i += blockDim.x) {
            int idx = base + i;
            int tile = idx >> 5;
            int k_in_tile = idx & 31;
            bg[tile * 256 + 32 * vc + k_in_tile] = tq_float_to_e4m3(xv[idx] / block_scale_inv);
        }
    }
}
