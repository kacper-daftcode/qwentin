// speculative decoding on the paged pool: verify, commit and the paged MTP drafter.
// Part of the forward_qwen.cu unity build: included in order, not a standalone header.

// ===================== Speculative decoding on the paged pool: verify + commit =====================
// qwn_paged_verify runs ONE ragged wave of K segments (slot i: its K_i draft tokens after the last
// committed token plus the bonus column, T = sum(K_i + 1) <= TQ_VF_MAX_T columns) through the paged
// DECODE-step kernels (not the prefill ones), with lm_head logits + argmax for EVERY column:
//  - full-attention layers: all T columns' KV go into the pool first (the decode kv-write kernel), then
//    the decode GQA kernel attends per column over [0..pos], so column t of a segment sees the segment's
//    columns < t exactly as K_i + 1 sequential decode steps would. A rejected position's KV is simply
//    overwritten by a later wave (the server owns the slot length; nothing reads past a column's pos).
//  - DeltaNet layers: g_pg_conv / g_pg_recur are READ, not advanced. k_tq_linear_conv_seq_paged and
//    k_tq_linear_core_seq_paged carry the conv window / the recurrent state through the segment in
//    registers, column by column, with the per-token kernels' exact FP order (bit-identical to running
//    k_tq_linear_conv_update_paged + k_tq_linear_decode_core_hs_paged K_i + 1 times). The pre-conv
//    in_proj_qkv output and (b, a) of every column are kept per layer (g_vf_x/b/a, ~40 KB per column
//    per layer) so qwn_paged_commit can replay the accepted prefix into the slot state: one read + one
//    write of the slot's state per commit whatever the accepted length. No per-slot scratch state.
//  - GEMMs run at T columns with 32-token units at most (g_exl3_ng_max = 4: fp32 accumulators, the
//    decode step's regime) and the in_proj_b/a pair kernel at any T.
// With the split pickers pinned (TQ_QMMA_SF_K_SPLIT, TQ_PAGED_GQA_SPLIT) the column logits are
// bit-identical to the decode step's (analysis/spec_paged_check.py); unpinned, a column's k-splits
// follow T instead of N (the same "not batch-invariant" band as decode at different N).
// Per linear layer: the kept verify inputs and the slot states (device table for the fused commit replay).
struct tq_vf_lp { float *x; const float *b; const float *a; void *recur; float *conv;     // recur: g_pg_dstate format
                  const uint16_t *conv_w; const float *A_log; const uint16_t *dt_bias; };
#define TQ_VF_MAX_T 64
#define TQ_VF_MAX_SEG 256
static int    g_vf_cap = 0;                          // columns the kept-input buffers hold
static float *g_vf_x[TQ_MAX_LAYERS] = {0};           // per linear layer: pre-conv in_proj_qkv output [cap][conv_dim]
static float *g_vf_b[TQ_MAX_LAYERS] = {0};           // [cap][lvh]
static float *g_vf_a[TQ_MAX_LAYERS] = {0};
static float *g_vf_logits = NULL;                    // [cap][V] every column's logits (fp32)
static int   *g_vf_argmax = NULL;                    // [cap]
static float *g_vf_hidden = NULL;                    // [cap][H] pre-final-norm hidden of every column (MTP seeding)
static int   *g_vf_iota = NULL;                      // 0..cap-1 (gated-norm column ids)
static int   *g_vf_colS = NULL;                      // [cap] per-column GQA key-split count (= the decode step's at that pos)
static int   *g_vf_segc0 = NULL, *g_vf_segn = NULL;  // [cap] sub-segments (col0, n) for the segment-aware attention
static tq_vf_lp *g_vf_lp = NULL; static int g_vf_nlin = 0;   // per-linear-layer pointer table (fused commit replay)
static int   *g_vf_seg_d = NULL;                     // device: [3][TQ_VF_MAX_SEG] slot/off/len of the verify wave
static int   *g_vf_cseg_d = NULL;                    // device: [3][TQ_VF_MAX_SEG] slot/off/len of a commit list
static int   *g_vf_pos_d = NULL;                     // device: [cap] the pending wave's column positions (the commit replay's DS = 4 seeds)
static int    g_vf_K = 0, g_vf_T = 0;                // the pending verify wave (0 = none)
static int    h_vf_seg_slot[TQ_VF_MAX_SEG], h_vf_seg_off[TQ_VF_MAX_SEG], h_vf_seg_len[TQ_VF_MAX_SEG];
static int    h_vf_done[TQ_VF_MAX_SEG];
static int    h_vf_cseg[3][TQ_VF_MAX_SEG];
static int    h_vf_tok[TQ_VF_MAX_T], h_vf_pos[TQ_VF_MAX_T];   // the pending wave's tokens / positions (MTP trunk)

static void vf_free_all(void) {
    for (int L = 0; L < TQ_MAX_LAYERS; L++) {
        if (g_vf_x[L]) { cudaFree(g_vf_x[L]); g_vf_x[L] = NULL; }
        if (g_vf_b[L]) { cudaFree(g_vf_b[L]); g_vf_b[L] = NULL; }
        if (g_vf_a[L]) { cudaFree(g_vf_a[L]); g_vf_a[L] = NULL; }
    }
    if (g_vf_logits) { cudaFree(g_vf_logits); g_vf_logits = NULL; }
    if (g_vf_argmax) { cudaFree(g_vf_argmax); g_vf_argmax = NULL; }
    if (g_vf_hidden) { cudaFree(g_vf_hidden); g_vf_hidden = NULL; }
    if (g_vf_iota) { cudaFree(g_vf_iota); g_vf_iota = NULL; }
    if (g_vf_colS) { cudaFree(g_vf_colS); g_vf_colS = NULL; }
    if (g_vf_segc0) { cudaFree(g_vf_segc0); g_vf_segc0 = NULL; }
    if (g_vf_segn) { cudaFree(g_vf_segn); g_vf_segn = NULL; }
    if (g_vf_lp) { cudaFree(g_vf_lp); g_vf_lp = NULL; } g_vf_nlin = 0;
    if (g_vf_seg_d) { cudaFree(g_vf_seg_d); g_vf_seg_d = NULL; }
    if (g_vf_cseg_d) { cudaFree(g_vf_cseg_d); g_vf_cseg_d = NULL; }
    if (g_vf_pos_d) { cudaFree(g_vf_pos_d); g_vf_pos_d = NULL; }
    g_vf_cap = 0; g_vf_K = 0; g_vf_T = 0;
}

// Kept-input + logits buffers for waves of up to T columns (grown on demand, never shrunk).
static int vf_ensure(int T) {
    if (g_vf_cap >= T) return 0;
    graph_invalidate();
    int cap = (T + 15) & ~15;
    if (cap > TQ_VF_MAX_T) cap = TQ_VF_MAX_T;
    if (cap < T) return -1;
    int lkh = g_qwen.linear_num_key_heads, lvh = g_qwen.linear_num_value_heads, ld = g_qwen.linear_value_head_dim;
    int conv_dim = 2 * lkh * ld + lvh * ld;
    vf_free_all();
    for (int L = 0; L < g_qwen.L; L++) {
        if (g_qwen.layer_types[L] != TQ_LAYER_LINEAR_ATTENTION) continue;
        if (cudaMalloc(&g_vf_x[L], (size_t)cap * conv_dim * sizeof(float)) != cudaSuccess) { vf_free_all(); return -2; }
        if (cudaMalloc(&g_vf_b[L], (size_t)cap * lvh * sizeof(float)) != cudaSuccess) { vf_free_all(); return -2; }
        if (cudaMalloc(&g_vf_a[L], (size_t)cap * lvh * sizeof(float)) != cudaSuccess) { vf_free_all(); return -2; }
    }
    if (cudaMalloc(&g_vf_logits, (size_t)cap * g_qwen.V * sizeof(float)) != cudaSuccess) { vf_free_all(); return -3; }
    if (cudaMalloc(&g_vf_argmax, (size_t)cap * sizeof(int)) != cudaSuccess) { vf_free_all(); return -3; }
    if (cudaMalloc(&g_vf_hidden, (size_t)cap * g_qwen.H * sizeof(float)) != cudaSuccess) { vf_free_all(); return -3; }
    if (cudaMalloc(&g_vf_iota, (size_t)cap * sizeof(int)) != cudaSuccess) { vf_free_all(); return -3; }
    if (cudaMalloc(&g_vf_colS, (size_t)cap * sizeof(int)) != cudaSuccess) { vf_free_all(); return -3; }
    if (cudaMalloc(&g_vf_segc0, (size_t)cap * sizeof(int)) != cudaSuccess) { vf_free_all(); return -3; }
    if (cudaMalloc(&g_vf_segn, (size_t)cap * sizeof(int)) != cudaSuccess) { vf_free_all(); return -3; }
    if (cudaMalloc(&g_vf_seg_d, (size_t)3 * TQ_VF_MAX_SEG * sizeof(int)) != cudaSuccess) { vf_free_all(); return -3; }
    if (cudaMalloc(&g_vf_cseg_d, (size_t)3 * TQ_VF_MAX_SEG * sizeof(int)) != cudaSuccess) { vf_free_all(); return -3; }
    if (cudaMalloc(&g_vf_pos_d, (size_t)cap * sizeof(int)) != cudaSuccess) { vf_free_all(); return -3; }
    int hi[TQ_VF_MAX_T];
    for (int i = 0; i < cap; i++) hi[i] = i;
    cudaMemcpyAsync(g_vf_iota, hi, (size_t)cap * sizeof(int), cudaMemcpyHostToDevice, g_qwen.stream);
    {   // the per-layer pointer table of the fused commit replay
        tq_vf_lp h_lp[TQ_MAX_LAYERS];
        int n = 0;
        for (int L = 0; L < g_qwen.L; L++) {
            if (g_qwen.layer_types[L] != TQ_LAYER_LINEAR_ATTENTION) continue;
            tq_layer_t *l = &g_qwen.layers[L];
            h_lp[n].x = g_vf_x[L]; h_lp[n].b = g_vf_b[L]; h_lp[n].a = g_vf_a[L];
            h_lp[n].recur = g_pg_recur[L]; h_lp[n].conv = g_pg_conv[L];
            h_lp[n].conv_w = l->d_linear_conv1d; h_lp[n].A_log = l->d_linear_A_log; h_lp[n].dt_bias = l->d_linear_dt_bias;
            n++;
        }
        if (cudaMalloc(&g_vf_lp, (size_t)n * sizeof(tq_vf_lp)) != cudaSuccess) { vf_free_all(); return -3; }
        cudaMemcpyAsync(g_vf_lp, h_lp, (size_t)n * sizeof(tq_vf_lp), cudaMemcpyHostToDevice, g_qwen.stream);
        g_vf_nlin = n;
    }
    cudaStreamSynchronize(g_qwen.stream);
    g_vf_cap = cap;
    return 0;
}

// Causal conv over the columns of each segment (grid.y), window loaded from the slot's conv state;
// same shift + append + sum order as k_tq_linear_conv_update_paged per column. write_state=0 leaves
// the slot state untouched (verify), 1 stores the final window (commit replay).
static __device__ __forceinline__ void tq_linear_conv_seq_body(
        float *out, float *state_base, const float *x, const uint16_t *conv_weight, int channels, int kernel_size,
        const int *seg_slot, const int *seg_off, const int *seg_len, int state_stride, int write_state, int seg) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= channels) return;
    int slot = seg_slot[seg], off = seg_off[seg], n = seg_len[seg];
    float *s = state_base + (size_t)slot * state_stride + (size_t)i * kernel_size;
    const uint16_t *wt = conv_weight + (size_t)i * kernel_size;
    float win[8];
    for (int j = 0; j < kernel_size; j++) win[j] = s[j];
    for (int t = 0; t < n; t++) {
        const float *xn = x + (size_t)(off + t) * channels;
        float sum = 0.0f;
        for (int j = 0; j < kernel_size - 1; j++) { win[j] = win[j + 1]; sum += win[j] * tq_bf16_to_float(wt[j]); }
        win[kernel_size - 1] = xn[i];
        sum += xn[i] * tq_bf16_to_float(wt[kernel_size - 1]);
        out[(size_t)(off + t) * channels + i] = sum / (1.0f + expf(-sum));
    }
    if (write_state) for (int j = 0; j < kernel_size; j++) s[j] = win[j];
}
__global__ void k_tq_linear_conv_seq_paged(float *out, float *state_base, const float *x,
                                           const uint16_t *conv_weight, int channels, int kernel_size,
                                           const int *seg_slot, const int *seg_off, const int *seg_len,
                                           int state_stride, int write_state) {
    tq_pdl_enter();
    tq_linear_conv_seq_body(out, state_base, x, conv_weight, channels, kernel_size, seg_slot, seg_off, seg_len,
                            state_stride, write_state, blockIdx.y);
}
// Commit replay of ALL linear layers in one launch (grid.z = layer): the conv runs IN PLACE over the kept pre-conv
// input (per channel the read of column t precedes the write of column t) and stores the window.
__global__ void k_tq_linear_conv_seq_paged_all(const tq_vf_lp *lp, int channels, int kernel_size,
                                               const int *seg_slot, const int *seg_off, const int *seg_len, int state_stride) {
    tq_pdl_enter();
    const tq_vf_lp p = lp[blockIdx.z];
    tq_linear_conv_seq_body(p.x, p.conv, p.x, p.conv_w, channels, kernel_size, seg_slot, seg_off, seg_len,
                            state_stride, 1, blockIdx.y);
}

// The per-token DeltaNet recurrence of k_tq_linear_decode_core_hs_paged<G, HS> carried through the
// columns of a segment (grid.y) with the state slice in registers: same thread layout, same per-column
// math and reduction order, so the column outputs and the final state equal n sequential launches of
// the per-token kernel. core_out = NULL skips the core (commit replay); write_state stores the state.
template <int G, int HS, int DS>
static __device__ __forceinline__ void tq_linear_core_seq_body(
        float *core_out, void *recurrent_state, const float *qkv_conv, const float *b_proj, const float *a_proj,
        const float *A_log, const uint16_t *dt_bias, int value_heads, int key_heads, int dim,
        const int *seg_slot, const int *seg_off, const int *seg_len,
        int recur_stride, int conv_dim, int core_stride, int write_state, int seg, const int *col_pos) {
    extern __shared__ float smem[];
    float *qn = smem, *kn = smem + dim, *red = smem + 2 * dim;
    int slot = seg_slot[seg], off = seg_off[seg], n = seg_len[seg];
    int blk = blockIdx.x, head = blk / HS, part = blk % HS;
    if (head >= value_heads) return;
    int t = threadIdx.x, bs = blockDim.x;
    int tids_per = dim / HS, tid_local = t / G, klane = t % G;
    int tid = part * tids_per + tid_local;
    int group = value_heads / key_heads, key_head = head / group, key_dim = key_heads * dim;
    tq_ds_acc<DS> state = tq_ds_make<DS>(recurrent_state, (size_t)slot, (size_t)recur_stride, (size_t)head * dim * dim);
    constexpr int NKK = 128 / G;                      // dim == 128: constant-trip loops keep st[] in registers
    float st[NKK];
    int nkk = 0;
    // The segment's inputs are all known up front: the next column's q/k/v/b/a are loaded while this column runs
    // (software pipelining; the per-column math and reduction order are unchanged -> the state stays bitwise).
    // bs >= dim for every valid (G, HS) config, so thread t owns q[t]/k[t] (t < dim) and v[tid].
    float q_c = 0.0f, k_c = 0.0f, v_c = 0.0f, b_c = 0.0f, a_c = 0.0f;
    {
        const float *qc = qkv_conv + (size_t)off * conv_dim;
        if (t < dim) { q_c = qc[key_head * dim + t]; k_c = qc[key_dim + key_head * dim + t]; }
        v_c = qc[2 * key_dim + head * dim + tid];
        b_c = b_proj[(size_t)off * value_heads + head]; a_c = a_proj[(size_t)off * value_heads + head];
    }
    #pragma unroll
    for (int j = 0; j < NKK; j++) st[j] = state.ld((size_t)(klane + j * G) * dim + tid);
    (void)nkk;
    for (int s = 0; s < n; s++) {
        int col = off + s;
        if constexpr (DS == 4) state.pos = (uint32_t)col_pos[col];
        float q_n = 0.0f, k_n = 0.0f, v_n = 0.0f, b_n = 0.0f, a_n = 0.0f;
        if (s + 1 < n) {
            const float *qc = qkv_conv + (size_t)(col + 1) * conv_dim;
            if (t < dim) { q_n = qc[key_head * dim + t]; k_n = qc[key_dim + key_head * dim + t]; }
            v_n = qc[2 * key_dim + head * dim + tid];
            b_n = b_proj[(size_t)(col + 1) * value_heads + head]; a_n = a_proj[(size_t)(col + 1) * value_heads + head];
        }
        if (t < dim) red[t] = q_c * q_c;
        __syncthreads();
        float qsum = tq_block_reduce_sum_sh(red, dim, t);
        __syncthreads();
        if (t < dim) red[t] = k_c * k_c;
        __syncthreads();
        float ksum = tq_block_reduce_sum_sh(red, dim, t);
        __syncthreads();
        if (t < dim) {
            qn[t] = q_c * rsqrtf(qsum + 1.0e-6f) * rsqrtf((float)dim);
            kn[t] = k_c * rsqrtf(ksum + 1.0e-6f);
        }
        __syncthreads();
        float beta = 1.0f / (1.0f + expf(-b_c));
        float softplus_arg = a_c + tq_bf16_to_float(dt_bias[head]);
        float softplus = log1pf(expf(-fabsf(softplus_arg))) + fmaxf(softplus_arg, 0.0f);
        float gg = -expf(A_log[head]) * softplus;
        float decay = expf(gg);
        float psum = 0.0f;
        #pragma unroll
        for (int j = 0; j < NKK; j++) psum += st[j] * kn[klane + j * G];
        #pragma unroll
        for (int o = G >> 1; o > 0; o >>= 1) psum += __shfl_down_sync(0xffffffffu, psum, o, G);
        float kv_mem = __shfl_sync(0xffffffffu, psum, 0, G) * decay;
        float delta = (v_c - kv_mem) * beta;
        q_c = q_n; k_c = k_n; v_c = v_n; b_c = b_n; a_c = a_n;
        float pcore = 0.0f;
        #pragma unroll
        for (int j = 0; j < NKK; j++) {
            int kk = klane + j * G;
            float newv = st[j] * decay + kn[kk] * delta;
            st[j] = state.rnd((size_t)kk * dim + tid, newv);   // what the next decode step would load back (fp32: newv)
            pcore += newv * qn[kk];
        }
        #pragma unroll
        for (int o = G >> 1; o > 0; o >>= 1) pcore += __shfl_down_sync(0xffffffffu, pcore, o, G);
        float core = __shfl_sync(0xffffffffu, pcore, 0, G);
        if (core_out && klane == 0) core_out[(size_t)col * core_stride + head * dim + tid] = core;
        __syncthreads();                                  // qn/kn/red are rebuilt for the next column
    }
    if (write_state) {
        // the carried values are already rounded (rnd == st's rounding; with DS = 4 the same seed: the last column's)
        if constexpr (DS == 4) state.pos = (uint32_t)col_pos[off + n - 1];
        #pragma unroll
        for (int j = 0; j < NKK; j++) state.st((size_t)(klane + j * G) * dim + tid, st[j]);
    }
}
template <int G, int HS, int DS>
__global__ void k_tq_linear_core_seq_paged(float *core_out, void *recurrent_state,
                                           const float *qkv_conv, const float *b_proj, const float *a_proj,
                                           const float *A_log, const uint16_t *dt_bias,
                                           int value_heads, int key_heads, int dim,
                                           const int *seg_slot, const int *seg_off, const int *seg_len,
                                           int recur_stride, int conv_dim, int core_stride, int write_state,
                                           const int *col_pos) {
    tq_pdl_enter();
    tq_linear_core_seq_body<G, HS, DS>(core_out, recurrent_state, qkv_conv, b_proj, a_proj, A_log, dt_bias, value_heads,
                                       key_heads, dim, seg_slot, seg_off, seg_len, recur_stride, conv_dim, core_stride,
                                       write_state, blockIdx.y, col_pos);
}
template <int G, int HS, int DS>
__global__ void k_tq_linear_core_seq_paged_all(const tq_vf_lp *lp, int value_heads, int key_heads, int dim,
                                               const int *seg_slot, const int *seg_off, const int *seg_len,
                                               int recur_stride, int conv_dim, const int *col_pos) {
    tq_pdl_enter();
    const tq_vf_lp p = lp[blockIdx.z];
    tq_linear_core_seq_body<G, HS, DS>(NULL, p.recur, p.x, p.b, p.a, p.A_log, p.dt_bias, value_heads, key_heads, dim,
                                       seg_slot, seg_off, seg_len, recur_stride, conv_dim, 0, 1, blockIdx.y, col_pos);
}

// Launch the sequential core over nseg segments (device seg arrays). core_out NULL = state replay only.
// col_pos: the wave's column positions (the verify's g_wide_pos; the commit's kept copy g_vf_pos_d) -- DS = 4's seeds.
static int vf_linear_core_seq(tq_layer_t *l, float *core_out, void *recur_base, const float *qkv_conv,
                              const float *bp, const float *ap, const int *seg_slot, const int *seg_off,
                              const int *seg_len, int nseg, int recur_f, int conv_dim, int value_dim, int write_state,
                              const int *col_pos) {
    int heads = g_qwen.linear_num_value_heads, key_heads = g_qwen.linear_num_key_heads;
    int dim = g_qwen.linear_value_head_dim;
    int lc_g = linear_core_par_g(), lc_hs = linear_core_hs();
    bool lc_par_ok = lc_g > 0 && dim == 128 && dim * lc_g <= 1024;      // the paged cores unroll over dim == 128
    bool lc_hs_ok = lc_par_ok && lc_hs >= 2 && (dim % lc_hs) == 0 && lc_g >= lc_hs &&
                    ((lc_g == 8 && (lc_hs == 2 || lc_hs == 4 || lc_hs == 8)) ||
                     (lc_g == 4 && (lc_hs == 2 || lc_hs == 4)));
    if (!lc_hs_ok) return -1;
    size_t shA = (size_t)3 * dim * sizeof(float);
    int blkA = (dim / lc_hs) * lc_g;
    dim3 gridA(heads * lc_hs, nseg);
    #define VF_CORE_HS1(GG, HH, DS)                                                               \
        tq_launch(g_pdl_small, k_tq_linear_core_seq_paged<GG, HH, DS>, gridA, dim3(blkA), shA, g_qwen.stream, \
                  core_out, recur_base, qkv_conv, bp, ap, l->d_linear_A_log, l->d_linear_dt_bias,   \
                  heads, key_heads, dim, seg_slot, seg_off, seg_len, recur_f, conv_dim, value_dim, write_state, col_pos)
    #define VF_CORE_HS(GG, HH) do {                                                               \
        if (g_pg_dstate == 1) VF_CORE_HS1(GG, HH, 1);                                             \
        else if (g_pg_dstate == 2) VF_CORE_HS1(GG, HH, 2);                                        \
        else if (g_pg_dstate == 3) VF_CORE_HS1(GG, HH, 3);                                        \
        else if (g_pg_dstate == 4) VF_CORE_HS1(GG, HH, 4);                                        \
        else VF_CORE_HS1(GG, HH, 0); } while (0)
    if (lc_g == 8 && lc_hs == 8) VF_CORE_HS(8, 8);
    else if (lc_g == 8 && lc_hs == 4) VF_CORE_HS(8, 4);
    else if (lc_g == 8 && lc_hs == 2) VF_CORE_HS(8, 2);
    else if (lc_g == 4 && lc_hs == 4) VF_CORE_HS(4, 4);
    else VF_CORE_HS(4, 2);
    #undef VF_CORE_HS
    #undef VF_CORE_HS1
    return 0;
}

// Host + upload prework of a verify wave: buffers, blocks, the kept host copies, block tables / encoded lengths,
// the column / segment arrays, the per-column key-split counts and the sub-segment list (g_attn_* for the
// launcher). graph_mode = 1 (TQ_GRAPH): the caller's arrays go through the pinned staging rows (tokens into
// g_tok_d), the wave's trellis encode jobs are built and uploaded (g_t4_job_mode), and the attention partials of
// the wave's S are allocated here, so the captured body allocates nothing. Returns 0, or < 0 (-90/-91 as before;
// -92 = not representable as a graph (graph_mode only)); *tiles_out = the T4 job tiles per layer launch.
static int paged_verify_prep(const int *tokens, const int *col_slot, const int *col_pos,
                             const int *seg_slot, const int *seg_off, const int *seg_len, int K, int T,
                             int graph_mode, int *vf_maxpos_out, int *tiles_out) {
    if (ensure_wide_prefill_buffers(T) != 0) return -90;
    if (ensure_pf_colslot(T) != 0) return -90;
    if (vf_ensure(T) != 0) return -90;
    int vf_maxpos = 0;
    for (int k = 0; k < K; k++) {
        int last_pos = col_pos[seg_off[k] + seg_len[k] - 1];
        if (paged_ensure_blocks(seg_slot[k], last_pos) != 0) return -91;     // pool full
        if (last_pos > vf_maxpos) vf_maxpos = last_pos;
    }
    for (int k = 0; k < K; k++) { h_vf_seg_slot[k] = seg_slot[k]; h_vf_seg_off[k] = seg_off[k]; h_vf_seg_len[k] = seg_len[k]; }
    for (int j = 0; j < T; j++) { h_vf_tok[j] = tokens[j]; h_vf_pos[j] = col_pos[j]; }
    paged_upload_tables(g_qwen.stream);
    if (g_pg_t4) paged_t4_upload_wave_encl(seg_slot, seg_off, seg_len, col_pos, K, g_qwen.stream);
    *tiles_out = 0;
    if (graph_mode) {
        if (!g_tok_d || !h_pin_ctl || T > g_pin_cols) return -92;
        int *pt = h_pin_ctl, *ps = h_pin_ctl + g_pin_cols, *pp = h_pin_ctl + 2 * g_pin_cols;
        for (int j = 0; j < T; j++) { pt[j] = tokens[j]; ps[j] = col_slot[j]; pp[j] = col_pos[j]; }
        cudaMemcpyAsync(g_tok_d, pt, (size_t)T * sizeof(int), cudaMemcpyHostToDevice, g_qwen.stream);
        cudaMemcpyAsync(g_pf_colslot, ps, (size_t)T * sizeof(int), cudaMemcpyHostToDevice, g_qwen.stream);
        cudaMemcpyAsync(g_wide_pos, pp, (size_t)T * sizeof(int), cudaMemcpyHostToDevice, g_qwen.stream);
        if (g_pg_t4) {
            int tiles = paged_t4_prep_wave_jobs(seg_slot, seg_off, seg_len, col_pos, K, g_qwen.stream);
            if (tiles <= 0) return -92;                   // nothing to encode / not representable: eager
            *tiles_out = tiles;
        }
    } else {
        cudaMemcpyAsync(g_pf_colslot, col_slot, (size_t)T * sizeof(int), cudaMemcpyHostToDevice, g_qwen.stream);
        cudaMemcpyAsync(g_wide_pos, col_pos, (size_t)T * sizeof(int), cudaMemcpyHostToDevice, g_qwen.stream);
    }
    cudaMemcpyAsync(g_vf_pos_d, h_vf_pos, (size_t)T * sizeof(int), cudaMemcpyHostToDevice, g_qwen.stream);   // kept for the commit (DS = 4 seeds); both modes
    cudaMemcpyAsync(g_vf_seg_d, h_vf_seg_slot, (size_t)K * sizeof(int), cudaMemcpyHostToDevice, g_qwen.stream);
    cudaMemcpyAsync(g_vf_seg_d + TQ_VF_MAX_SEG, h_vf_seg_off, (size_t)K * sizeof(int), cudaMemcpyHostToDevice, g_qwen.stream);
    cudaMemcpyAsync(g_vf_seg_d + 2 * TQ_VF_MAX_SEG, h_vf_seg_len, (size_t)K * sizeof(int), cudaMemcpyHostToDevice, g_qwen.stream);
    // each column's GQA key-split count = the one a decode step would pick at ITS position (a wave can span a
    // tile boundary where the context clamp changes S), so the attention bits match the step's column by column
    // the full-attention layers stream each slot's K/V once per sub-segment of up to 32/group columns
    // (k_tq_paged_attn_q4_gqa_seg) instead of once per column, when the group fits one m16 tile
    int vf_qt = 1;
    if (g_qwen.nh % g_qwen.nkv == 0 && g_qwen.nh / g_qwen.nkv <= 16) {
        int group = g_qwen.nh / g_qwen.nkv, nmax = 0;
        for (int k = 0; k < K; k++) if (seg_len[k] > nmax) nmax = seg_len[k];
        vf_qt = (nmax * group <= 16) ? 1 : 2;
    }
    {
        int h_colS[TQ_VF_MAX_T], smax = 1;
        for (int j = 0; j < T; j++) { h_colS[j] = paged_attn_split_S(K, col_pos[j], vf_qt); if (h_colS[j] > smax) smax = h_colS[j]; }
        cudaMemcpyAsync(g_vf_colS, h_colS, (size_t)T * sizeof(int), cudaMemcpyHostToDevice, g_qwen.stream);
        g_attn_col_S = g_vf_colS; g_attn_col_S_max = smax;
    }
    g_attn_seg_cnt = 0;
    if (g_qwen.nh % g_qwen.nkv == 0 && g_qwen.nh / g_qwen.nkv <= 16) {
        // sub-segments of <= QT*16/group columns; QT = 1 when every segment fits one m16 tile, else 2
        int group = g_qwen.nh / g_qwen.nkv, nmax = 0;
        for (int k = 0; k < K; k++) if (seg_len[k] > nmax) nmax = seg_len[k];
        int qt = (nmax * group <= 16) ? 1 : 2, chunk = (qt * 16) / group;
        int h_c0[TQ_VF_MAX_T], h_n[TQ_VF_MAX_T], cnt = 0;
        for (int k = 0; k < K; k++)
            for (int t = 0; t < seg_len[k]; t += chunk) {
                h_c0[cnt] = seg_off[k] + t;
                h_n[cnt] = (seg_len[k] - t < chunk) ? seg_len[k] - t : chunk;
                cnt++;
            }
        cudaMemcpyAsync(g_vf_segc0, h_c0, (size_t)cnt * sizeof(int), cudaMemcpyHostToDevice, g_qwen.stream);
        cudaMemcpyAsync(g_vf_segn, h_n, (size_t)cnt * sizeof(int), cudaMemcpyHostToDevice, g_qwen.stream);
        g_attn_seg_c0 = g_vf_segc0; g_attn_seg_n = g_vf_segn; g_attn_seg_cnt = cnt; g_attn_seg_qt = qt;
    }
    if (graph_mode && g_attn_col_S_max > 1 &&          // the launcher's partials, before the capture
        ensure_attn_partials((size_t)g_qwen.nh * T * g_attn_col_S_max) != 0) return -92;
    *vf_maxpos_out = vf_maxpos;
    return 0;
}
// The wave's device work (the kernel sequence; graph_body = 1: tokens from g_tok_d in one launch, the trellis
// encode of the wave's columns from the prepared device jobs -- see paged_verify_prep).
static int paged_verify_body(const int *tokens, const int *col_slot, const int *col_pos,
                             const int *seg_slot, const int *seg_off, const int *seg_len, int K, int T,
                             int vf_maxpos, int graph_body) {
    int H = g_qwen.H, attn_m = g_qwen.nh * g_qwen.hd;
    int lkh = g_qwen.linear_num_key_heads, lvh = g_qwen.linear_num_value_heads, ld = g_qwen.linear_value_head_dim;
    int value_dim = lvh * ld, key_dim = lkh * ld, conv_dim = 2 * key_dim + value_dim;
    int recur_f = batched_recur_floats(), conv_f = batched_conv_floats();
    int ret;
    const int *d_sslot = g_vf_seg_d, *d_soff = g_vf_seg_d + TQ_VF_MAX_SEG, *d_slen = g_vf_seg_d + 2 * TQ_VF_MAX_SEG;
    if (graph_body)
        k_tq_embed_lookup_nk<<<dim3((H + 255) / 256, T), 256, 0, g_qwen.stream>>>(g_wide_h, g_qwen.d_embed, g_tok_d, H);
    else
        for (int j = 0; j < T; j++)
            k_tq_embed_lookup<<<(H + 255) / 256, 256, 0, g_qwen.stream>>>(
                g_wide_h + (size_t)j * H, g_qwen.d_embed, tokens[j], H);
    const int fuse = (g_qwen.flags & TQ_FLAG_EXL3_WEIGHTS) != 0;
    int norm_ready = 0;                       // the previous layer's fused tail produced this layer's input norm
    for (int L = 0; L < g_qwen.L; L++) {
        tq_layer_t *l = &g_qwen.layers[L];
        const uint16_t *next_ln = (L + 1 < g_qwen.L) ? g_qwen.layers[L + 1].d_input_ln : g_qwen.d_norm;
        if (g_qwen.layer_types[L] == TQ_LAYER_FULL_ATTENTION) {
            if (!(tq_wide_fmt_ok(&l->q_proj) && tq_wide_fmt_ok(&l->o_proj) &&
                  tq_wide_fmt_ok(&l->mlp_gate) && tq_wide_fmt_ok(&l->mlp_down))) return -63;
            if (!norm_ready)
                tq_launch(g_pdl_small, k_tq_qwen_rmsnorm_b, dim3(8, T), dim3(1024), 0, g_qwen.stream,
                          g_wide_norm, g_wide_h, l->d_input_ln, H, g_qwen.eps);
            if ((ret = wide_quant_input(g_wide_norm, H, T)) != 0) return -64;
            const tq_qmma_weight_t *qkv[3] = {&l->q_proj, &l->k_proj, &l->v_proj};
            float *qkv_out[3] = {g_wide_q, g_wide_k, g_wide_v};
            if ((ret = wide_proj_group(qkv, qkv_out, 3, T)) != 0) return -64;
            if (launch_paged_attn_q4(g_wide_core, g_wide_q, l->d_q_norm, g_wide_k, g_wide_v, l->d_k_norm,
                                     g_pool_k4[L], g_pool_kq4s[L], g_pool_v8[L], g_pool_vscale[L],
                                     g_wide_pos, g_pf_colslot, g_block_table, g_pg_maxblk, g_pg_page, g_pg_plog,
                                     T, vf_maxpos, L, col_pos, col_slot, 1, g_qwen.stream,
                                     g_pg_t4 ? L : -1, g_pg_t4 ? seg_slot : NULL, seg_off, seg_len, K) != 0) return -67;
            if ((ret = wide_quant_input(g_wide_core, attn_m, T)) != 0) return -68;
            if ((ret = wide_proj(&l->o_proj, g_wide_o, T)) != 0) return -68;
            if (fuse) {
                if ((ret = wide_mlp(l, g_wide_resid, g_wide_h, T, g_wide_h, g_wide_o, next_ln, g_wide_norm)) != 0) return -69;
                norm_ready = 1;
            } else {
                tq_launch(g_pdl_small, k_tq_add_vec, dim3(((size_t)T * H + 255) / 256), dim3(256), 0, g_qwen.stream,
                          g_wide_resid, g_wide_h, g_wide_o, (int)((size_t)T * H));
                if ((ret = wide_mlp(l, g_wide_resid, g_wide_h, T)) != 0) return -69;
            }
        } else {
            if (!(tq_wide_fmt_ok(&l->linear_in_qkv) && tq_wide_fmt_ok(&l->linear_in_z) &&
                  tq_wide_fmt_ok(&l->linear_in_b) && tq_wide_fmt_ok(&l->linear_in_a) &&
                  tq_wide_fmt_ok(&l->linear_out) && tq_wide_fmt_ok(&l->mlp_gate) &&
                  tq_wide_fmt_ok(&l->mlp_down))) return -70;
            if (!norm_ready)
                tq_launch(g_pdl_small, k_tq_qwen_rmsnorm_b, dim3(8, T), dim3(1024), 0, g_qwen.stream,
                          g_wide_norm, g_wide_h, l->d_input_ln, H, g_qwen.eps);
            if ((ret = wide_quant_input(g_wide_norm, H, T)) != 0) return -71;
            const tq_qmma_weight_t *qz[2] = {&l->linear_in_qkv, &l->linear_in_z};
            float *qz_out[2] = {g_vf_x[L], g_wide_z};                  // pre-conv input kept for the commit
            if ((ret = wide_proj_group(qz, qz_out, 2, T)) != 0) return -71;
            if ((ret = wide_proj_pair_seq(&l->linear_in_b, g_vf_b[L], &l->linear_in_a, g_vf_a[L], T)) != 0) return -73;
            tq_launch(g_pdl_small, k_tq_linear_conv_seq_paged, dim3((conv_dim + 255) / 256, K), dim3(256), 0,
                      g_qwen.stream, g_wide_qkv, g_pg_conv[L], (const float *)g_vf_x[L], l->d_linear_conv1d, conv_dim,
                      g_qwen.linear_conv_kernel_dim, d_sslot, d_soff, d_slen, conv_f, 0);
            if (vf_linear_core_seq(l, g_wide_lcore, g_pg_recur[L], g_wide_qkv, g_vf_b[L], g_vf_a[L],
                                   d_sslot, d_soff, d_slen, K, recur_f, conv_dim, value_dim, 0, g_wide_pos) != 0) return -75;
            tq_launch(g_pdl_small, k_tq_linear_decode_gated_norm_b, dim3(lvh, T), dim3(ld), (size_t)ld * sizeof(float),
                      g_qwen.stream, g_wide_lcore, g_wide_z, l->d_linear_norm, lvh, ld, g_qwen.eps, g_vf_iota, value_dim);
            if ((ret = wide_quant_input(g_wide_lcore, value_dim, T)) != 0) return -76;
            if ((ret = wide_proj(&l->linear_out, g_wide_o, T)) != 0) return -76;
            if (fuse) {
                if ((ret = wide_mlp(l, g_wide_resid, g_wide_h, T, g_wide_h, g_wide_o, next_ln, g_wide_norm)) != 0) return -77;
                norm_ready = 1;
            } else {
                tq_launch(g_pdl_small, k_tq_add_vec, dim3(((size_t)T * H + 255) / 256), dim3(256), 0, g_qwen.stream,
                          g_wide_resid, g_wide_h, g_wide_o, (int)((size_t)T * H));
                if ((ret = wide_mlp(l, g_wide_resid, g_wide_h, T)) != 0) return -77;
            }
        }
    }
    cudaMemcpyAsync(g_vf_hidden, g_wide_h, (size_t)T * H * sizeof(float), cudaMemcpyDeviceToDevice, g_qwen.stream);
    if (!norm_ready)
        tq_launch(g_pdl_small, k_tq_qwen_rmsnorm_b, dim3(8, T), dim3(1024), 0, g_qwen.stream,
                  g_wide_norm, g_wide_h, g_qwen.d_norm, H, g_qwen.eps);
    if ((ret = wide_quant_input(g_wide_norm, H, T)) != 0) return -78;
    if ((ret = wide_proj(&g_qwen.lm_head, g_vf_logits, T)) != 0) return -78;
    tq_launch(g_pdl_small, k_tq_batched_argmax, dim3(T), dim3(256), 0, g_qwen.stream,
              g_vf_argmax, g_vf_logits, g_qwen.V, T);
    if (g_pg_sink) {
        if (T > g_pg_sink_cap) return -81;
        cudaMemcpyAsync(g_pg_sink, g_vf_logits, (size_t)T * g_qwen.V * sizeof(float),
                        cudaMemcpyDeviceToDevice, g_qwen.stream);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : -80;
}
// The eager verify wave (the stream sequence of 9da2d3b).
static int run_paged_verify_core(const int *tokens, const int *col_slot, const int *col_pos,
                                 const int *seg_slot, const int *seg_off, const int *seg_len, int K, int T) {
    int vf_maxpos = 0, tiles = 0;
    int ret = paged_verify_prep(tokens, col_slot, col_pos, seg_slot, seg_off, seg_len, K, T, 0, &vf_maxpos, &tiles);
    if (ret != 0) return ret;
    return paged_verify_body(tokens, col_slot, col_pos, seg_slot, seg_off, seg_len, K, T, vf_maxpos, 0);
}
// TQ_GRAPH: the verify wave as a graph per (K, T, encode tiles, key-split max, sub-segment count / QT, flags).
// Returns 1 when not covered (the caller runs the eager wave; the prework is idempotent).
static int run_paged_verify_graph(const int *tokens, const int *col_slot, const int *col_pos,
                                  const int *seg_slot, const int *seg_off, const int *seg_len, int K, int T) {
    if (g_graph_disabled || paged_topk_pages() > 0) return 1;
    if (!(paged_attn_gqa_mma_enabled() && (g_pg_page & 127) == 0 && g_qwen.hd == 256 &&
          g_qwen.nh % g_qwen.nkv == 0 && g_qwen.nh / g_qwen.nkv <= 16)) return 1;
    int vf_maxpos = 0, tiles = 0;
    int ret = paged_verify_prep(tokens, col_slot, col_pos, seg_slot, seg_off, seg_len, K, T, 1, &vf_maxpos, &tiles);
    if (ret == -92) return 1;
    if (ret != 0) return ret;
    const int fuse = (g_qwen.flags & TQ_FLAG_EXL3_WEIGHTS) != 0;
    unsigned long long key = 2ull;                                   // kind 2 = verify wave
    key = (key << 8) | (unsigned)K;
    key = (key << 8) | (unsigned)T;
    key = (key << 12) | (unsigned)tiles;
    key = (key << 8) | (unsigned)g_attn_col_S_max;
    key = (key << 8) | (unsigned)g_attn_seg_cnt;
    key = (key << 4) | (unsigned)(g_attn_seg_qt & 3) | ((unsigned)(g_exl3_ng_max == 8) << 2) | ((unsigned)(g_split_n_override > 0) << 3);
    key = (key << 4) | (unsigned)(fuse | (g_pg_t4 << 1) | (g_pg_v4 << 2) | ((g_pg_sink != NULL) << 3));
    key = (key << 4) | (unsigned)(paged_attn_st_enabled() & 3) | ((unsigned)(g_pdl_on | (g_pdl_small << 1)) << 2);
    key = (key << 8) | (g_graph_gen & 0xff);
    cudaGraphExec_t exec = graph_cache_get(key);
    if (!exec) {
        int nodes = 0;
        g_t4_job_mode = g_pg_t4 ? 1 : 0; g_t4_job_tiles = tiles;
        exec = graph_capture("verify", [&]() {
            return paged_verify_body(tokens, col_slot, col_pos, seg_slot, seg_off, seg_len, K, T, vf_maxpos, 1); }, &nodes);
        g_t4_job_mode = 0; g_t4_job_tiles = 0;
        if (!exec) return 1;
        key = (key & ~0xffull) | (g_graph_gen & 0xff);
        graph_cache_put(key, exec, nodes);
        if (tq_graph_debug())
            fprintf(stderr, "[graph] verify: captured K=%d T=%d tiles=%d Smax=%d segs=%d qt=%d maxpos=%d (%d cached)\n",
                    K, T, tiles, g_attn_col_S_max, g_attn_seg_cnt, g_attn_seg_qt, vf_maxpos, g_graph_n);
    }
    cudaError_t e = cudaGraphLaunch(exec, g_qwen.stream);
    if (e != cudaSuccess) {
        fprintf(stderr, "[graph] verify: cudaGraphLaunch failed: %s\n", cudaGetErrorString(e));
        return -84;
    }
    return 0;
}

// ABI. tokens/col_slot/col_pos: T concatenated columns; seg_slot/seg_off/seg_len: K segments covering
// them in order (one segment per slot, columns of a segment at consecutive positions). out_argmax[T]
// = greedy token after every column. The DeltaNet slot states are left untouched: call
// qwn_paged_commit for the accepted prefix of every segment before anything else touches these slots.
// Needs the wide (EXL3) lm_head; T <= 64.
extern "C" int qwn_paged_verify(const int *tokens, const int *col_slot, const int *col_pos,
                                const int *seg_slot, const int *seg_off, const int *seg_len,
                                int K, int T, int *out_argmax) {
    if (!g_pg_ready) return -1;
    if (!g_qwen.kv_q4) return -2;
    if (!paged_lm_head_wide()) return -3;
    if (K < 1 || K > TQ_VF_MAX_SEG || K > g_pg_maxslots || T < 1 || T > TQ_VF_MAX_T || K > T) return -4;
    int next = 0;
    for (int k = 0; k < K; k++) {
        int slot = seg_slot[k], off = seg_off[k], n = seg_len[k];
        if (slot < 0 || slot >= g_pg_maxslots) return -5;
        if (n < 1 || off != next || off + n > T) return -6;
        for (int j = 0; j < k; j++) if (seg_slot[j] == slot) return -7;      // one segment per slot
        for (int t = 0; t < n; t++) {
            if (col_slot[off + t] != slot || col_pos[off + t] != col_pos[off] + t) return -8;
            if (col_pos[off + t] < 0 || col_pos[off + t] >= g_qwen.max_seq) return -9;
        }
        next = off + n;
    }
    if (next != T) return -6;
    int pdl = tq_pdl_enabled();
    g_pdl_on = pdl != 0;
    g_pdl_small = pdl == 2;
    // 32-token GEMM units, and the GEMM k-splits / attention key-splits of the slot count rather than the column
    // count: the committed tokens equal plain decode's at that client count bit for bit (except a wave crossing a
    // 128-key tile boundary)
    g_exl3_ng_max = 4;
    g_split_n_override = K;
    g_vf_K = 0;
    int ret = 1;
    if (tq_graph_enabled() >= 2) ret = run_paged_verify_graph(tokens, col_slot, col_pos, seg_slot, seg_off, seg_len, K, T);
    if (ret == 1) ret = run_paged_verify_core(tokens, col_slot, col_pos, seg_slot, seg_off, seg_len, K, T);
    g_attn_col_S = NULL; g_attn_col_S_max = 0;
    g_attn_seg_c0 = g_attn_seg_n = NULL; g_attn_seg_cnt = 0; g_attn_seg_qt = 0;
    g_split_n_override = 0;
    g_exl3_ng_max = 16;
    g_pdl_on = g_pdl_small = 0;
    if (ret != 0) { cudaStreamSynchronize(g_qwen.stream); return ret; }
    g_vf_K = K; g_vf_T = T;
    for (int k = 0; k < K; k++) h_vf_done[k] = 0;
    if (out_argmax)
        cudaMemcpyAsync(out_argmax, g_vf_argmax, (size_t)T * sizeof(int), cudaMemcpyDeviceToHost, g_qwen.stream);
    return cudaStreamSynchronize(g_qwen.stream) == cudaSuccess ? 0 : -10;
}

// Commit the first n_accept[i] columns of segment seg_idx[i] of the pending verify wave into the slot's
// DeltaNet state (conv window + recurrence replayed from the kept inputs; the KV is already in the
// pool). n_accept = 0 discards the segment. Each segment commits at most once; the wave stays pending
// until the next qwn_paged_verify.
extern "C" int qwn_paged_commit(const int *seg_idx, const int *n_accept, int M) {
    if (!g_pg_ready) return -1;
    if (g_vf_K <= 0) return -2;
    if (M < 1 || M > g_vf_K) return -3;
    int m = 0;
    for (int i = 0; i < M; i++) {
        int k = seg_idx[i], n = n_accept[i];
        if (k < 0 || k >= g_vf_K) return -4;
        if (h_vf_done[k]) return -5;
        if (n < 0 || n > h_vf_seg_len[k]) return -6;
        for (int j = 0; j < i; j++) if (seg_idx[j] == k) return -5;
    }
    for (int i = 0; i < M; i++) {
        int k = seg_idx[i], n = n_accept[i];
        h_vf_done[k] = 1;
        if (n == 0) continue;
        h_vf_cseg[0][m] = h_vf_seg_slot[k]; h_vf_cseg[1][m] = h_vf_seg_off[k]; h_vf_cseg[2][m] = n;
        m++;
    }
    if (m == 0) return 0;
    int lkh = g_qwen.linear_num_key_heads, lvh = g_qwen.linear_num_value_heads, ld = g_qwen.linear_value_head_dim;
    int value_dim = lvh * ld, key_dim = lkh * ld, conv_dim = 2 * key_dim + value_dim;
    int recur_f = batched_recur_floats(), conv_f = batched_conv_floats();
    for (int c = 0; c < 3; c++)
        cudaMemcpyAsync(g_vf_cseg_d + c * TQ_VF_MAX_SEG, h_vf_cseg[c], (size_t)m * sizeof(int),
                        cudaMemcpyHostToDevice, g_qwen.stream);
    const int *d_cslot = g_vf_cseg_d, *d_coff = g_vf_cseg_d + TQ_VF_MAX_SEG, *d_clen = g_vf_cseg_d + 2 * TQ_VF_MAX_SEG;
    int pdl = tq_pdl_enabled();
    g_pdl_small = pdl == 2;
    if (g_vf_lp) {
        // all 48 linear layers in two launches (grid.z = layer): 48 x 8 x 192 blocks fill the GPU instead of 48 x 1.13
        // waves of a 1536-block launch; the per-thread math (and so the state) is the per-layer kernels' bit for bit
        int heads = lvh, dim = ld, lc_g = linear_core_par_g(), lc_hs = linear_core_hs();
        bool ok = lc_g > 0 && dim == 128 && lc_hs >= 2 && lc_g >= lc_hs &&
                  ((lc_g == 8 && (lc_hs == 2 || lc_hs == 4 || lc_hs == 8)) || (lc_g == 4 && (lc_hs == 2 || lc_hs == 4)));
        if (!ok) { g_pdl_small = 0; return -7; }
        tq_launch(g_pdl_small, k_tq_linear_conv_seq_paged_all, dim3((conv_dim + 255) / 256, m, g_vf_nlin), dim3(256), 0,
                  g_qwen.stream, (const tq_vf_lp *)g_vf_lp, conv_dim, g_qwen.linear_conv_kernel_dim, d_cslot, d_coff, d_clen, conv_f);
        size_t shA = (size_t)3 * dim * sizeof(float);
        dim3 gridA(heads * lc_hs, m, g_vf_nlin), blkA((dim / lc_hs) * lc_g);
        #define VF_CORE_ALL1(GG, HH, DS) \
            tq_launch(g_pdl_small, k_tq_linear_core_seq_paged_all<GG, HH, DS>, gridA, blkA, shA, g_qwen.stream, \
                      (const tq_vf_lp *)g_vf_lp, heads, lkh, dim, d_cslot, d_coff, d_clen, recur_f, conv_dim, (const int *)g_vf_pos_d)
        #define VF_CORE_ALL(GG, HH) do {                                                          \
            if (g_pg_dstate == 1) VF_CORE_ALL1(GG, HH, 1);                                        \
            else if (g_pg_dstate == 2) VF_CORE_ALL1(GG, HH, 2);                                   \
            else if (g_pg_dstate == 3) VF_CORE_ALL1(GG, HH, 3);                                   \
            else if (g_pg_dstate == 4) VF_CORE_ALL1(GG, HH, 4);                                   \
            else VF_CORE_ALL1(GG, HH, 0); } while (0)
        if (lc_g == 8 && lc_hs == 8) VF_CORE_ALL(8, 8);
        else if (lc_g == 8 && lc_hs == 4) VF_CORE_ALL(8, 4);
        else if (lc_g == 8 && lc_hs == 2) VF_CORE_ALL(8, 2);
        else if (lc_g == 4 && lc_hs == 4) VF_CORE_ALL(4, 4);
        else VF_CORE_ALL(4, 2);
        #undef VF_CORE_ALL
        #undef VF_CORE_ALL1
    } else {
        for (int L = 0; L < g_qwen.L; L++) {
            if (g_qwen.layer_types[L] != TQ_LAYER_LINEAR_ATTENTION) continue;
            tq_layer_t *l = &g_qwen.layers[L];
            tq_launch(g_pdl_small, k_tq_linear_conv_seq_paged, dim3((conv_dim + 255) / 256, m), dim3(256), 0,
                      g_qwen.stream, g_wide_conv, g_pg_conv[L], (const float *)g_vf_x[L], l->d_linear_conv1d, conv_dim,
                      g_qwen.linear_conv_kernel_dim, d_cslot, d_coff, d_clen, conv_f, 1);
            if (vf_linear_core_seq(l, NULL, g_pg_recur[L], g_wide_conv, g_vf_b[L], g_vf_a[L],
                                   d_cslot, d_coff, d_clen, m, recur_f, conv_dim, value_dim, 1, g_vf_pos_d) != 0) {
                g_pdl_small = 0;
                return -7;
            }
        }
    }
    g_pdl_small = 0;
    if (g_mtp_on && mtp_hook_commit(m) != 0) return -10;      // MTP trunk over the accepted columns
    if (g_pg_t4)                                            // the accepted columns were encoded per layer by the wave
        for (int c = 0; c < m; c++) {
            int slot = h_vf_cseg[0][c], len = h_vf_pos[h_vf_cseg[1][c]] + h_vf_cseg[2][c];
            if (len > h_slot_encl[slot]) h_slot_encl[slot] = len;
        }
    if (cudaGetLastError() != cudaSuccess) return -8;
    return cudaStreamSynchronize(g_qwen.stream) == cudaSuccess ? 0 : -9;
}

// Device pointer to the pending wave's pre-final-norm hidden states ([T][H] fp32; column c of the
// last qwn_paged_verify) for the MTP drafter; NULL when no wave is pending.
extern "C" const float *qwn_paged_verify_hidden(int *out_T) {
    if (out_T) *out_T = g_vf_K > 0 ? g_vf_T : 0;
    return g_vf_K > 0 ? g_vf_hidden : NULL;
}

// ===================== Paged MTP drafter (stage 3) =====================
// The MTP head (fc + one full-attention layer + mtp.norm, the shared lm_head) runs over its own paged
// KV pool on the slots' block tables (one more "layer" of pool rows, FP4 like the main pool). MTP KV
// slot p of a slot <- (h_p, x_{p+1}): the main model's pre-final-norm hidden at p and the token that
// followed (the single-stream trunk convention); the MTP forward at slot p predicts x_{p+2}.
// Trunk maintenance, only with qwn_paged_mtp_init(1): prefill waves write the in-wave pairs (col t,
// col t+1) of every segment plus the cross-wave pair from the slot's remembered last hidden; the decode
// step writes (last hidden, token) -> slot pos-1; the commit writes the pairs over the accepted columns.
// Each path then stores the slot's new last hidden (g_mtp_last_h). qwn_paged_mtp_draft runs K chained
// MTP steps per slot from (last hidden, next token), each step's post-norm hidden + argmax feeding the
// next; it writes MTP slots pos..pos+K-1 on the way (re-written from the main hiddens at the next
// commit). A stale or missing MTP slot (prefix-cache truncation, position 0) only costs acceptance,
// never correctness: the verify decides every token.
#define TQ_MTP_CTL_ROWS 24
#define TQ_MTP_CTL_COLS 128
#define TQ_MTP_MAX_K 16
static float   *g_mtp_fc_in = NULL, *g_mtp_x = NULL, *g_mtp_hid = NULL, *g_mtp_e = NULL;   // [128][2H], [128][H] x 3
static int     *g_mtp_ctl = NULL;                    // device ints [TQ_MTP_CTL_ROWS][TQ_MTP_CTL_COLS]
static int     *g_mtp_draft_d = NULL;                // [TQ_MTP_MAX_K][max_slots]
static int      h_mtp_ctl[TQ_MTP_CTL_ROWS][TQ_MTP_CTL_COLS];

static void mtp_paged_free(void) {
    graph_invalidate();
    if (g_mtp_k4) { cudaFree(g_mtp_k4); g_mtp_k4 = NULL; }
    if (g_mtp_kq4s) { cudaFree(g_mtp_kq4s); g_mtp_kq4s = NULL; }
    if (g_mtp_v8) { cudaFree(g_mtp_v8); g_mtp_v8 = NULL; }
    if (g_mtp_vscale) { cudaFree(g_mtp_vscale); g_mtp_vscale = NULL; }
    if (g_mtp_last_h) { cudaFree(g_mtp_last_h); g_mtp_last_h = NULL; }
    if (g_mtp_fc_in) { cudaFree(g_mtp_fc_in); g_mtp_fc_in = NULL; }
    if (g_mtp_x) { cudaFree(g_mtp_x); g_mtp_x = NULL; }
    if (g_mtp_hid) { cudaFree(g_mtp_hid); g_mtp_hid = NULL; }
    if (g_mtp_e) { cudaFree(g_mtp_e); g_mtp_e = NULL; }
    if (g_mtp_ctl) { cudaFree(g_mtp_ctl); g_mtp_ctl = NULL; }
    if (g_mtp_draft_d) { cudaFree(g_mtp_draft_d); g_mtp_draft_d = NULL; }
    g_mtp_on = 0;
}

// Bytes of one MTP pool row set (K codes + scales + V codes + scales) per position = one full layer's.
extern "C" int qwn_paged_mtp_init(int on) {
    if (!g_pg_ready) return -1;
    mtp_paged_free();
    if (!on) return 0;
    if (!g_qwen.has_mtp_section) return -2;
    if (!g_qwen.mtp_exl3) return -3;                     // needs a --mtp-exl3 tqf (EXL3 MTP projections)
    if (!paged_lm_head_wide()) return -4;
    if (g_pg_maxslots > TQ_MTP_CTL_COLS) return -9;
    int kv_m = g_qwen.nkv * g_qwen.hd, nkv = g_qwen.nkv, H = g_qwen.H;
    size_t qe = batched_q4s_elem_bytes();
    size_t bp = (size_t)g_pg_nblocks * g_pg_page;
    size_t k4b = bp * (kv_m >> 1), kq4sb = bp * nkv * 16 * qe;
    size_t v8b = g_pg_v4 ? k4b : bp * kv_m, vscb = g_pg_v4 ? kq4sb : bp * nkv * sizeof(float);
    if (cudaMalloc(&g_mtp_k4, k4b) != cudaSuccess || cudaMalloc(&g_mtp_kq4s, kq4sb) != cudaSuccess ||
        cudaMalloc(&g_mtp_v8, v8b) != cudaSuccess || cudaMalloc(&g_mtp_vscale, vscb) != cudaSuccess) {
        mtp_paged_free(); return -5;
    }
    cudaMemsetAsync(g_mtp_k4, 0, k4b, g_qwen.stream); cudaMemsetAsync(g_mtp_kq4s, 0, kq4sb, g_qwen.stream);
    cudaMemsetAsync(g_mtp_v8, 0, v8b, g_qwen.stream); cudaMemsetAsync(g_mtp_vscale, 0, vscb, g_qwen.stream);
    if (cudaMalloc(&g_mtp_last_h, (size_t)g_pg_maxslots * H * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&g_mtp_fc_in, (size_t)TQ_MTP_CTL_COLS * 2 * H * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&g_mtp_x, (size_t)TQ_MTP_CTL_COLS * H * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&g_mtp_hid, (size_t)TQ_MTP_CTL_COLS * H * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&g_mtp_e, (size_t)TQ_MTP_CTL_COLS * H * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&g_mtp_ctl, (size_t)TQ_MTP_CTL_ROWS * TQ_MTP_CTL_COLS * sizeof(int)) != cudaSuccess ||
        cudaMalloc(&g_mtp_draft_d, (size_t)TQ_MTP_MAX_K * g_pg_maxslots * sizeof(int)) != cudaSuccess) {
        mtp_paged_free(); return -6;
    }
    cudaMemsetAsync(g_mtp_last_h, 0, (size_t)g_pg_maxslots * H * sizeof(float), g_qwen.stream);
    if (ensure_wide_prefill_buffers(g_pg_maxslots) != 0) { mtp_paged_free(); return -7; }
    if (cudaStreamSynchronize(g_qwen.stream) != cudaSuccess) { mtp_paged_free(); return -8; }
    g_mtp_on = 1;
    fprintf(stderr, "[paged-mtp] MTP KV pool %.1f MB (%zu rows), drafter on\n",
            (double)(k4b + kq4sb + v8b + vscb) / 1048576.0, bp);
    return 0;
}

static int *mtp_ctl_row(int r) { return g_mtp_ctl + (size_t)r * TQ_MTP_CTL_COLS; }
static void mtp_ctl_upload(int rows) {
    cudaMemcpyAsync(g_mtp_ctl, h_mtp_ctl, (size_t)rows * TQ_MTP_CTL_COLS * sizeof(int), cudaMemcpyHostToDevice, g_qwen.stream);
}

__global__ void k_tq_scatter_rows_nk(float *dst, const int *dst_idx, const float *src, const int *src_idx, int H) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= H) return;
    int s = src_idx ? src_idx[blockIdx.y] : (int)blockIdx.y;
    dst[(size_t)dst_idx[blockIdx.y] * H + i] = src[(size_t)s * H + i];
}

// fc_in = [norm_emb(embed(tok)), norm_hidden(hidden)] -> x = fc(fc_in) for m columns (device arrays:
// tokens d_tok, hidden rows d_hidx of hsrc, or hsrc rows 0..m-1 when d_hidx == NULL).
static int mtp_fc_forward(const float *hsrc, const int *d_hidx, const int *d_tok, int m) {
    int H = g_qwen.H;
    cudaStream_t st = g_qwen.stream;
    k_tq_embed_lookup_nk<<<dim3((H + 255) / 256, m), 256, 0, st>>>(g_mtp_e, g_qwen.d_embed, d_tok, H);
    k_tq_qwen_rmsnorm_b_out<<<dim3(8, m), 1024, 0, st>>>(g_mtp_fc_in, 2 * H, 0, g_mtp_e, g_qwen.d_mtp_pre_fc_norm_emb, H, g_qwen.eps);
    const float *hid = hsrc;
    if (d_hidx) {
        k_tq_gather_rows_nk<<<dim3((H + 255) / 256, m), 256, 0, st>>>(g_mtp_hid, hsrc, d_hidx, H);
        hid = g_mtp_hid;
    }
    k_tq_qwen_rmsnorm_b_out<<<dim3(8, m), 1024, 0, st>>>(g_mtp_fc_in, 2 * H, H, hid, g_qwen.d_mtp_pre_fc_norm_hidden, H, g_qwen.eps);
    int ret = wide_quant_input(g_mtp_fc_in, 2 * H, m);
    if (ret != 0) return ret;
    return wide_proj(&g_qwen.mtp_fc_x, g_mtp_x, m);
}

// MTP KV slot d_pos[j] of slot d_slot[j] <- (hidden, token) pairs, j < m (KV write only, no attention).
static int mtp_trunk_write(const float *hsrc, const int *d_hidx, const int *d_tok, const int *d_pos, const int *d_slot, int m) {
    if (m < 1) return 0;
    int H = g_qwen.H;
    tq_layer_t *mt = &g_qwen.mtp_layer;
    if (ensure_wide_prefill_buffers(m) != 0) return -1;
    int ret = mtp_fc_forward(hsrc, d_hidx, d_tok, m);
    if (ret != 0) return ret;
    k_tq_qwen_rmsnorm_b<<<dim3(8, m), 1024, 0, g_qwen.stream>>>(g_wide_norm, g_mtp_x, mt->d_input_ln, H, g_qwen.eps);
    if ((ret = wide_quant_input(g_wide_norm, H, m)) != 0) return ret;
    const tq_qmma_weight_t *kv[2] = {&mt->k_proj, &mt->v_proj};
    float *kv_out[2] = {g_wide_k, g_wide_v};
    if ((ret = wide_proj_group(kv, kv_out, 2, m)) != 0) return ret;
    paged_kv_write(m, g_mtp_k4, g_mtp_kq4s, g_mtp_v8, g_mtp_vscale, g_wide_k, g_wide_v, mt->d_k_norm,
                   d_pos, d_slot, g_block_table, g_pg_maxblk, g_pg_page, g_pg_plog, g_qwen.stream);
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

// Decode step: (last hidden of the slot, tokens[j]) -> MTP slot positions[j]-1; then last hidden = g_wide_h[j].
// Split in two so the graph path (TQ_GRAPH) can upload the control rows before the replay and capture the kernels:
// prep = host rows + upload (returns m = the columns with a position > 0), kernels = the device work for that m.
static int mtp_hook_decode_prep(const int *tokens, const int *slot_ids_h, const int *positions, int N) {
    int m = 0;
    for (int j = 0; j < N; j++) {
        if (positions[j] <= 0) continue;
        h_mtp_ctl[0][m] = tokens[j]; h_mtp_ctl[1][m] = slot_ids_h[j]; h_mtp_ctl[2][m] = positions[j] - 1;
        h_mtp_ctl[3][m] = slot_ids_h[j]; m++;
    }
    for (int j = 0; j < N; j++) h_mtp_ctl[4][j] = slot_ids_h[j];
    mtp_ctl_upload(5);
    return m;
}
static int mtp_hook_decode_kernels(int m, int N) {
    int ret = mtp_trunk_write(g_mtp_last_h, mtp_ctl_row(1), mtp_ctl_row(0), mtp_ctl_row(2), mtp_ctl_row(3), m);
    if (ret != 0) return ret;
    k_tq_scatter_rows_nk<<<dim3((g_qwen.H + 255) / 256, N), 256, 0, g_qwen.stream>>>(
        g_mtp_last_h, mtp_ctl_row(4), g_wide_h, NULL, g_qwen.H);
    return cudaGetLastError() == cudaSuccess ? 0 : -3;
}
static int mtp_hook_decode(const int *tokens, const int *slot_ids_h, const int *positions, int N) {
    int m = mtp_hook_decode_prep(tokens, slot_ids_h, positions, N);
    return mtp_hook_decode_kernels(m, N);
}

// Prefill wave: in-wave pairs (col t -> token of col t+1) of every segment, the cross-wave pair of each
// continuing segment from the slot's last hidden, then last hidden = the segment's last column.
static int mtp_hook_prefill(const int *tokens, const int *col_pos, const int *seg_slot, const int *seg_off,
                            const int *seg_len, int K, int T) {
    if (T > TQ_MTP_CTL_COLS || K > TQ_MTP_CTL_COLS) return -1;
    int m = 0, c = 0;
    for (int k = 0; k < K; k++) {
        int off = seg_off[k], n = seg_len[k], slot = seg_slot[k];
        for (int t = 0; t + 1 < n; t++) {
            int col = off + t;
            h_mtp_ctl[0][m] = tokens[col + 1]; h_mtp_ctl[1][m] = col; h_mtp_ctl[2][m] = col_pos[col]; h_mtp_ctl[3][m] = slot; m++;
        }
        if (col_pos[off] > 0) {
            h_mtp_ctl[4][c] = tokens[off]; h_mtp_ctl[5][c] = slot; h_mtp_ctl[6][c] = col_pos[off] - 1; h_mtp_ctl[7][c] = slot; c++;
        }
        h_mtp_ctl[8][k] = slot; h_mtp_ctl[9][k] = off + n - 1;
    }
    mtp_ctl_upload(10);
    int ret = mtp_trunk_write(g_wide_h, mtp_ctl_row(1), mtp_ctl_row(0), mtp_ctl_row(2), mtp_ctl_row(3), m);
    if (ret != 0) return ret;
    ret = mtp_trunk_write(g_mtp_last_h, mtp_ctl_row(5), mtp_ctl_row(4), mtp_ctl_row(6), mtp_ctl_row(7), c);
    if (ret != 0) return ret;
    k_tq_scatter_rows_nk<<<dim3((g_qwen.H + 255) / 256, K), 256, 0, g_qwen.stream>>>(
        g_mtp_last_h, mtp_ctl_row(8), g_wide_h, mtp_ctl_row(9), g_qwen.H);
    return cudaGetLastError() == cudaSuccess ? 0 : -3;
}

// Commit: over the m committed segments of the pending wave (h_vf_cseg): pairs (col t -> token of col t+1)
// for the accepted columns but the last, hiddens from g_vf_hidden; last hidden = the last accepted column.
static int mtp_hook_commit(int m) {
    int p = 0;
    for (int i = 0; i < m; i++) {
        int slot = h_vf_cseg[0][i], off = h_vf_cseg[1][i], n = h_vf_cseg[2][i];
        for (int t = 0; t + 1 < n; t++) {
            int col = off + t;
            h_mtp_ctl[0][p] = h_vf_tok[col + 1]; h_mtp_ctl[1][p] = col; h_mtp_ctl[2][p] = h_vf_pos[col]; h_mtp_ctl[3][p] = slot; p++;
        }
        h_mtp_ctl[4][i] = slot; h_mtp_ctl[5][i] = off + n - 1;
    }
    mtp_ctl_upload(6);
    int ret = mtp_trunk_write(g_vf_hidden, mtp_ctl_row(1), mtp_ctl_row(0), mtp_ctl_row(2), mtp_ctl_row(3), p);
    if (ret != 0) return ret;
    k_tq_scatter_rows_nk<<<dim3((g_qwen.H + 255) / 256, m), 256, 0, g_qwen.stream>>>(
        g_mtp_last_h, mtp_ctl_row(4), g_vf_hidden, mtp_ctl_row(5), g_qwen.H);
    return cudaGetLastError() == cudaSuccess ? 0 : -3;
}

// K chained MTP draft steps for N slots. positions[i] = the slot's last committed position (its hidden is
// the slot's last hidden), seeds[i] = the token that follows it (the next token to feed). Step k runs at
// MTP slot positions[i] + k on (hidden_k, token_k): hidden_0 = last hidden, token_0 = seed, then the
// step's post-norm hidden and argmax. drafts[i*K + k] = the argmax of step k (the guess for position
// positions[i] + 2 + k). The verify wave for the round is [seed, drafts...] at positions[i]+1 ...
extern "C" int qwn_paged_mtp_draft(const int *slot_ids, const int *positions, const int *seeds, int N, int K, int *drafts) {
    if (!g_pg_ready || !g_mtp_on) return -1;
    if (N < 1 || N > g_pg_maxslots || N > TQ_MTP_CTL_COLS || K < 1 || K > TQ_MTP_MAX_K) return -2;
    int H = g_qwen.H, attn_m = g_qwen.nh * g_qwen.hd;
    tq_layer_t *mt = &g_qwen.mtp_layer;
    int max_pos = 0;
    for (int i = 0; i < N; i++) {
        int slot = slot_ids[i], pos = positions[i];
        if (slot < 0 || slot >= g_pg_maxslots) return -3;
        if (pos < 0 || pos + K + 1 >= g_qwen.max_seq) return -4;
        if (seeds[i] < 0 || seeds[i] >= g_qwen.V) return -5;
        for (int j = 0; j < i; j++) if (slot_ids[j] == slot) return -6;
        if (paged_ensure_blocks(slot, pos + K - 1) != 0) return -7;
        if (pos > max_pos) max_pos = pos;
    }
    for (int i = 0; i < N; i++) { h_mtp_ctl[0][i] = slot_ids[i]; h_mtp_ctl[1][i] = seeds[i]; h_mtp_ctl[2][i] = slot_ids[i]; }
    for (int k = 0; k < K; k++) for (int i = 0; i < N; i++) h_mtp_ctl[3 + k][i] = positions[i] + k;
    paged_upload_tables(g_qwen.stream);
    mtp_ctl_upload(3 + K);
    int pdl = tq_pdl_enabled();
    g_pdl_on = pdl != 0; g_pdl_small = pdl == 2;
    int ret = 0;
    // TQ_GRAPH: the K chained steps as one graph per (N, K, the key-split counts of the K steps: S is monotone in
    // the position and moves at most once inside K <= 16 consecutive positions -> (S_0, first step with S_0 + 1)).
    int use_graph = 0;
    if (tq_graph_enabled() >= 2 && !g_graph_disabled && paged_topk_pages() == 0 &&
        paged_attn_gqa_mma_enabled() && (g_pg_page & 127) == 0 && g_qwen.hd == 256 &&
        g_qwen.nh % g_qwen.nkv == 0 && g_qwen.nh / g_qwen.nkv <= 16) {
        int S0 = paged_attn_split_S(N, max_pos, 1), jump = K, Smax = S0, ok = 1;
        for (int k = 1; k < K; k++) {
            int Sk = paged_attn_split_S(N, max_pos + k, 1);
            if (Sk > Smax) Smax = Sk;
            if (Sk != S0 && jump == K) jump = k;
            if (Sk != (k < jump ? S0 : S0 + 1)) ok = 0;
        }
        if (ok && (Smax <= 1 || ensure_attn_partials((size_t)g_qwen.nh * N * Smax) == 0)) {
            unsigned long long key = 3ull;                           // kind 3 = MTP draft
            key = (key << 8) | (unsigned)N;
            key = (key << 8) | (unsigned)K;
            key = (key << 8) | (unsigned)S0;
            key = (key << 8) | (unsigned)jump;
            key = (key << 4) | (unsigned)((g_pg_t4) | (g_pg_v4 << 1) | ((g_pg_sink != NULL) << 2));
            key = (key << 4) | (unsigned)(paged_attn_st_enabled() & 3) | ((unsigned)(g_pdl_on | (g_pdl_small << 1)) << 2);
            key = (key << 16) | (g_graph_gen & 0xffff);
            cudaGraphExec_t exec = graph_cache_get(key);
            if (!exec) {
                int nodes = 0;
                exec = graph_capture("draft", [&]() { return mtp_draft_body(N, K, max_pos); }, &nodes);
                if (exec) {
                    key = (key & ~0xffffull) | (g_graph_gen & 0xffff);
                    graph_cache_put(key, exec, nodes);
                    if (tq_graph_debug())
                        fprintf(stderr, "[graph] draft: captured N=%d K=%d S0=%d jump=%d maxpos=%d (%d cached)\n", N, K, S0, jump, max_pos, g_graph_n);
                }
            }
            if (exec) {
                use_graph = 1;
                if (cudaGraphLaunch(exec, g_qwen.stream) != cudaSuccess) {
                    fprintf(stderr, "[graph] draft: cudaGraphLaunch failed\n"); ret = -18;
                }
            }
        }
    }
    if (!use_graph && ret == 0) ret = mtp_draft_body(N, K, max_pos);
    g_pdl_on = g_pdl_small = 0;
    if (ret != 0) { cudaStreamSynchronize(g_qwen.stream); return ret; }
    int tmp[TQ_MTP_MAX_K * TQ_MTP_CTL_COLS];
    for (int k = 0; k < K; k++)
        cudaMemcpyAsync(tmp + k * N, g_mtp_draft_d + (size_t)k * g_pg_maxslots, (size_t)N * sizeof(int),
                        cudaMemcpyDeviceToHost, g_qwen.stream);
    if (cudaStreamSynchronize(g_qwen.stream) != cudaSuccess) return -16;
    for (int i = 0; i < N; i++) for (int k = 0; k < K; k++) drafts[i * K + k] = tmp[k * N + i];
    return cudaGetLastError() == cudaSuccess ? 0 : -17;
}
// The drafter's K chained MTP steps (device work after the control rows are uploaded; the same sequence eager
// or captured).
static int mtp_draft_body(int N, int K, int max_pos) {
    int H = g_qwen.H, attn_m = g_qwen.nh * g_qwen.hd;
    tq_layer_t *mt = &g_qwen.mtp_layer;
    const int *d_slot = mtp_ctl_row(0);
    int ret = 0;
    for (int k = 0; k < K && ret == 0; k++) {
        const int *d_tok = k == 0 ? mtp_ctl_row(1) : g_mtp_draft_d + (size_t)(k - 1) * g_pg_maxslots;
        const int *d_pos = mtp_ctl_row(3 + k);
        if ((ret = mtp_fc_forward(k == 0 ? g_mtp_last_h : g_mtp_hid, k == 0 ? mtp_ctl_row(2) : NULL, d_tok, N)) != 0) { ret = -10; break; }
        tq_launch(g_pdl_small, k_tq_qwen_rmsnorm_b, dim3(8, N), dim3(1024), 0, g_qwen.stream,
                  g_wide_norm, g_mtp_x, mt->d_input_ln, H, g_qwen.eps);
        if ((ret = wide_quant_input(g_wide_norm, H, N)) != 0) { ret = -11; break; }
        const tq_qmma_weight_t *qkv[3] = {&mt->q_proj, &mt->k_proj, &mt->v_proj};
        float *qkv_out[3] = {g_wide_q, g_wide_k, g_wide_v};
        if ((ret = wide_proj_group(qkv, qkv_out, 3, N)) != 0) { ret = -11; break; }
        if (launch_paged_attn_q4(g_wide_core, g_wide_q, mt->d_q_norm, g_wide_k, g_wide_v, mt->d_k_norm,
                                 g_mtp_k4, g_mtp_kq4s, g_mtp_v8, g_mtp_vscale,
                                 d_pos, d_slot, g_block_table, g_pg_maxblk, g_pg_page, g_pg_plog,
                                 N, max_pos + k, 0, NULL, NULL, 0, g_qwen.stream) != 0) { ret = -12; break; }
        if ((ret = wide_quant_input(g_wide_core, attn_m, N)) != 0) { ret = -13; break; }
        if ((ret = wide_proj(&mt->o_proj, g_wide_o, N)) != 0) { ret = -13; break; }
        tq_launch(g_pdl_small, k_tq_add_vec, dim3(((size_t)N * H + 255) / 256), dim3(256), 0, g_qwen.stream,
                  g_wide_resid, g_mtp_x, g_wide_o, (int)((size_t)N * H));
        if ((ret = wide_mlp(mt, g_wide_resid, g_wide_h, N)) != 0) { ret = -14; break; }
        tq_launch(g_pdl_small, k_tq_qwen_rmsnorm_b, dim3(8, N), dim3(1024), 0, g_qwen.stream,
                  g_mtp_hid, g_wide_h, g_qwen.d_mtp_norm, H, g_qwen.eps);         // post-norm hidden: next step + head
        if ((ret = wide_quant_input(g_mtp_hid, H, N)) != 0) { ret = -15; break; }
        if ((ret = wide_proj(&g_qwen.lm_head, g_pg_logits, N)) != 0) { ret = -15; break; }
        tq_launch(g_pdl_small, k_tq_batched_argmax, dim3(N), dim3(256), 0, g_qwen.stream,
                  g_mtp_draft_d + (size_t)k * g_pg_maxslots, g_pg_logits, g_qwen.V, N);
    }
    return ret;
}
