// the single-stream persistent warp-specialized layer kernel (device side).
// Part of the forward_qwen.cu unity build: included in order, not a standalone header.

// ============== Persistent warp-specialized layer kernel (spec verify) ==============
// One launch covers half a linear layer (part 0: pack -> in-projections -> DeltaNet ->
// gated norm -> pack -> out-projection; part 1: pack -> gate/up -> silu -> pack -> down
// -> residual add). CTAs are persistent: phases are separated by global software
// barriers, GEMV m-tiles are pulled from an atomic work queue, and any CTA that waits
// (DeltaNet depth gate, phase barrier) streams upcoming weight bytes into L2 instead
// of idling. Every phase body replicates the exact arithmetic + reduction order of the
// standalone kernels (max-reductions are exact and may be reshaped), so the persistent
// path is bit-identical to the sequential fused-pack path.

// scratch layout (ints): [0..15] phase barriers (two 8-slot pools: fused-layer
// halves get disjoint barrier ids via args.bar_base), [16] final arrive counter,
// [18..41] work queues (two 12-slot pools via args.q_base), [42] deltanet depth
// counter, [44..47] prefetch cursors (two programs per half), [48..63] per-node
// completion, [64..64+16*16) two-level barrier group counters (self-resetting)
#define TQ_PERSIST_BAR0 0
#define TQ_PERSIST_BARS 16
#define TQ_PERSIST_FINAL 16
#define TQ_PERSIST_Q0 18
#define TQ_PERSIST_DEPTH 42
#define TQ_PERSIST_PF 44
#define TQ_PERSIST_PF1 45
#define TQ_PERSIST_PF_B 46
#define TQ_PERSIST_PF1_B 47
#define TQ_PERSIST_NODE_DONE 48   // [TQ_SPEC_MAX_N] per-node completion (chain-segment gates)
#define TQ_PERSIST_SCRATCH_INTS 64
// Per-128-K-block quant->GEMV readiness flags (phase-tagged: quant1/quant2 x
// fused halves use distinct tags, so no mid-launch reset is needed).
#define TQ_PERSIST_QFLAGS_OFF (TQ_PERSIST_SCRATCH_INTS + TQ_PERSIST_BARS * 16)
#define TQ_PERSIST_QFLAGS_N 256
// Two-level arrival groups for node_done (fork publication): 128 single-counter
// RMWs serialized ~2.5us on the chain critical path; 16 groups cut that ~8x.
#define TQ_PERSIST_NODE_GRP (TQ_PERSIST_QFLAGS_OFF + TQ_PERSIST_QFLAGS_N)
#define TQ_PERSIST_NODE_GRP_W 16
// TQ_PERSIST_P2P per-m-tile completion counters (one per gemv2 m-tile, >= max
// m-tiles: down M=5120 -> 320, gate/up I=17408 -> 1088). Appended after node_grp;
// zeroed with the rest of the scratch by the last CTA at exit.
#define TQ_PERSIST_P2P_OFF (TQ_PERSIST_SCRATCH_INTS + TQ_PERSIST_BARS * 16 + TQ_PERSIST_QFLAGS_N + TQ_SPEC_MAX_N * TQ_PERSIST_NODE_GRP_W)
#define TQ_PERSIST_P2P_N 1152
// d_spec_segs tail slot holding the round's segment count: kernels read it from
// DEVICE memory (not a kernel arg) so a captured CUDA graph stays valid across
// rounds with different tree topologies.
#define TQ_SEGS_NSEGS_SLOT (3 * TQ_SPEC_MAX_N + 1 + TQ_SPEC_MAX_N * 8)
#define TQ_PERSIST_SCRATCH_TOTAL (TQ_PERSIST_SCRATCH_INTS + TQ_PERSIST_BARS * 16 + TQ_PERSIST_QFLAGS_N + TQ_SPEC_MAX_N * TQ_PERSIST_NODE_GRP_W + TQ_PERSIST_P2P_N)
#define TQ_PERSIST_CLK_SLOTS 32   // [0..15] phase/barrier-exit marks, [16..23] barrier-entry marks (prof only)

typedef struct {
    const uint32_t *a;            // weight payload
    const float *scale_inv;       // weight block scales
    int scale_cols;
    int M, Kt, ks;                // rows, K-tiles, k-splits (same auto split as spec_gemv)
    int abyte;                    // payload is e2m3_byte (128 words/tile, no unpack)
    int ae2m1;                    // payload is e2m1 4-bit (64 words/tile): half the
                                  // weight bytes of dense E2M3; lossy E2M3->E2M1 repack
    float *partials;              // [ks][N*M]
    float *out;                   // reduced [N x M] (col-major by node)
} tq_pw_t;

typedef struct {
    int N, ncta;
    int *scratch;                 // TQ_PERSIST_SCRATCH_INTS ints, zeroed before first use
    int *bar_groups;              // [12 x TQ_PERSIST_BAR_GROUPS] two-level arrival counters
    // pack 1 (input activations) + pack 2 (mid activations, produced in-kernel)
    const float *pack1_in; int pack1_K;
    float *pack2_in; int pack2_K;
    float *absmax, *factors;      // d_spec_nk_factors [0..8) and [8..16)
    const float *pack1_amax;      // per-column absmax partials for pack1, [N x stride]
    int pack1_amax_stride;        // (host norm kernels leave 8 stripe-maxes per node)
    uint8_t *bfrag; float *bscale;
    float *dnprep;                // [N][key_heads][2][dim] qn/kn + [N][value_dim] v (seg-fused DeltaNet)
    // group-1 GEMVs consume pack1; group-2 GEMV consumes pack2
    tq_pw_t w1[4]; int nw1;
    tq_pw_t w2;
    int part;                     // 0 = attn half (DeltaNet), 1 = mlp half (silu)
    // ---- part 0: DeltaNet ----
    float *recur_base; float *conv_state_base;
    const float *qkv_in;          // = w1[0].out
    const float *bp, *ap;         // = w1[2].out, w1[3].out
    const float *A_log; const uint16_t *dt_bias; const uint16_t *conv_w;
    const float *committed_recur, *committed_conv;
    const float *z;               // = w1[1].out
    const float *gated_norm_w;
    int heads, key_heads, dim, kernel_size;
    int recur_f, conv_f, conv_dim, value_dim;
    const int *group_ids, *parent_ids, *depth_offsets; int n_depths;
    const int *segs; int n_segs;  // chain segments: nodes[N] then off[n_segs+1] (0 = use depth waves)
    int state_skip;               // 1 = chain items write recur slabs only at fork nodes (commit replays)
    float eps;
    // ---- part 1: silu + final residual ----
    const float *gate_in, *up_in; // = w1[0].out, w1[1].out
    float *final_out;             // d_spec_h
    const float *final_resid;     // d_spec_resid
    // ---- part 2: full attention ----
    const uint16_t *attn_qnorm, *attn_knorm;
    const float *attn_kcache, *attn_vcache;
    const uint8_t *attn_kcache8, *attn_vcache8;   // FP8 prefix cache (TQ_KV_FP8)
    const float *attn_kscale, *attn_vscale;
    const uint8_t *attn_kcache4;                  // int4 K prefix cache (TQ_KV_Q4)
    const float *attn_kq4s;                       // (scale, zp) per 32-ch group
    float *attn_karch, *attn_varch;
    float *attn_scores;           // [ncta x max_seq] gmem score slabs (part 2)
    int max_seq;
    const int *attn_pos, *attn_anc;
    int nh, nkv, hd, kv_m, q_m, attn_m, attn_base;
    int attn_mma;                 // TQ_ATTN_MMA: ph3 items become per-head MMA blocks
    int attn_nchunks;             // position chunks per head (host-derived from base)
    int attn_pair;                // two q-heads of a kv group per item (N <= 8)
    float rope_theta;
    // ---- L2 prefetch programs (consumed while waiting) ----
    // program 1 (cursor 27): group-1 weights, streamed during the pack-1 barrier;
    // program 2 (cursor 28): downstream weights, streamed during DeltaNet waits.
    const uint32_t *pf_ptr[3]; size_t pf_words[3]; int n_pf;
    const uint32_t *pf1_ptr[2]; size_t pf1_words[2]; int n_pf1;
    int pf_bars;                  // also prefetch in generic phase barriers
    long long *clk;               // per-phase clock64 marks (prof mode; NULL = off)
    int recur_bf16;               // tree recurrent-state archive stored as BF16
    int gemv_warps;               // active GEMV warps per CTA (0 = all 8); MSHR-pressure knob
    // ---- fused-layer launch (PART pair in one kernel): disjoint sync pools ----
    int bar_base;                 // barrier-id offset for this half (0 or 8)
    int q_base;                   // work-queue offset for this half (0 or 12)
    int pf_slot, pf1_slot;        // prefetch cursor scratch slots for this half
    // ---- in-kernel norm phases (replace the standalone host norm kernels) ----
    const uint16_t *ln_in;        // input rmsnorm weight (norm_in phase; NULL = skip)
    const float *norm_h_in;       // residual stream read by norm_in ([N x H])
    float *norm_out;              // normed output (= pack1_in of this half)
    float *norm_amax;             // stripe absmax output (= pack1_amax buffer)
    const uint16_t *ln_post;      // post-attn rmsnorm weight (norm_out phase)
    const float *out_partials;    // first half's w2 GEMV partials ([ks][N*M])
    int out_ks;                   // first half's w2 k-splits
    const float *out_factors;     // first half's nk prescale factors (per node)
    float *resid_out;             // residual stream written by norm_out ([N x H])
} tq_spec_persist_args_t;

// Prefetch one 32KB chunk (256 threads x 128B) from a prefetch program.
// Returns 0 when the program is exhausted. s_tmp is a shared int for the cursor.
static __device__ int tq_persist_prefetch_prog(const tq_spec_persist_args_t *A, int *s_tmp,
                                               int cursor, const uint32_t *const *ptrs,
                                               const size_t *words, int n) {
    if (n == 0) return 0;
    if (threadIdx.x == 0) *s_tmp = atomicAdd(&A->scratch[cursor], 1);
    __syncthreads();
    int chunk = *s_tmp;
    __syncthreads();
    size_t word_off = (size_t)chunk * 8192;   // 32KB = 8192 words
    for (int r = 0; r < n; r++) {
        if (word_off < words[r]) {
            size_t my_off = word_off + (size_t)threadIdx.x * 32;
            if (my_off < words[r]) {
                const uint32_t *p = ptrs[r] + my_off;
                asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
            }
            return 1;
        }
        word_off -= words[r];
    }
    return 0;
}

static __device__ int tq_persist_prefetch_step(const tq_spec_persist_args_t *A, int *s_tmp) {
    return tq_persist_prefetch_prog(A, s_tmp, A->pf_slot, A->pf_ptr, A->pf_words, A->n_pf);
}

static __device__ int tq_persist_prefetch_step1(const tq_spec_persist_args_t *A, int *s_tmp) {
    return tq_persist_prefetch_prog(A, s_tmp, A->pf1_slot, A->pf1_ptr, A->pf1_words, A->n_pf1);
}

// Global software barrier across the persistent grid. Two-level arrival: CTAs
// add to one of 16 strided group counters (cutting same-address atomic
// serialization 16x); the group's last arrival promotes the group count to the
// root counter. allow_pf streams the prefetch program into L2 while waiting --
// only enabled for barriers whose stragglers are compute-bound (DeltaNet), so
// prefetch traffic never competes with a memory-bound phase tail.
#define TQ_PERSIST_BAR_GROUPS 16
static __device__ void tq_persist_bar_pf(const tq_spec_persist_args_t *A, int b_local, int allow_pf) {
    __shared__ int s_done, s_pf;
    // prof: CTA0 records arrival time so the python can split phase work (entry -
    // prev exit) from the barrier tail+latency (exit - entry). b_local <= 7.
    if (A->clk && blockIdx.x == 0 && threadIdx.x == 0 && b_local < 8) A->clk[16 + b_local] = clock64();
    int b = A->bar_base + b_local;
    int *root = &A->scratch[TQ_PERSIST_BAR0 + b];
    int *grp = A->bar_groups + b * TQ_PERSIST_BAR_GROUPS + (blockIdx.x % TQ_PERSIST_BAR_GROUPS);
    int grp_total = A->ncta / TQ_PERSIST_BAR_GROUPS +
                    ((int)(blockIdx.x % TQ_PERSIST_BAR_GROUPS) < (A->ncta % TQ_PERSIST_BAR_GROUPS) ? 1 : 0);
    __syncthreads();
    __threadfence();
    if (threadIdx.x == 0) {
        int prev = atomicAdd(grp, 1);
        if (prev == grp_total - 1) {
            atomicAdd(root, grp_total);
            atomicExch(grp, 0);   // reset for the next launch
        }
    }
    int pf_alive = allow_pf;
    while (1) {
        // poll with a plain volatile L2 read: the arrive side uses atomics +
        // threadfence, so a non-atomic read observes the final value; this cuts
        // the L2 read-modify-write storm of 340 CTAs polling one counter and
        // makes a shorter backoff affordable (barrier exit latency -> ~us less)
        if (threadIdx.x == 0)
            s_done = (*(volatile int *)root >= A->ncta);
        __syncthreads();
        if (s_done) break;
        if (pf_alive == 2) {
            if (!tq_persist_prefetch_step1(A, &s_pf)) pf_alive = 0;
        } else if (pf_alive == 1) {
            if (!tq_persist_prefetch_step(A, &s_pf)) pf_alive = 0;
        } else {
            __nanosleep(64);
        }
        __syncthreads();
    }
    __threadfence();
    if (A->clk && blockIdx.x == 0 && threadIdx.x == 0) A->clk[b_local + 1] = clock64();
}

static __device__ void tq_persist_bar(const tq_spec_persist_args_t *A, int b) {
    tq_persist_bar_pf(A, b, A->pf_bars);
}

// Phase work queue: returns the next item index or -1 when the phase is drained.
static __device__ int tq_persist_grab(const tq_spec_persist_args_t *A, int q, int n_items,
                                      int *s_tmp) {
    if (threadIdx.x == 0) *s_tmp = atomicAdd(&A->scratch[TQ_PERSIST_Q0 + A->q_base + q], 1);
    __syncthreads();
    int item = *s_tmp;
    __syncthreads();
    return item < n_items ? item : -1;
}

// Quantize one 128-K block of all N columns into the B-fragment, applying the
// prescale factors inline (identical math to k_tq_quantize_bfrag_pow2_fused_nk_f;
// the block absmax is an exact max so the 256-thread shape change is bit-safe).
// Column absmax = exact max over `amax_stride` partial maxes per column (stride 1
// = plain per-column absmax; stride 8 = stripe partials from the host norm kernel).
static __device__ void tq_persist_quant_item(const tq_spec_persist_args_t *A, int block,
                                             const float *x, int K,
                                             const float *amax, int amax_stride, float *sm,
                                             int qtag) {
    int tid = threadIdx.x;
    int nvar = A->N;
    int base = block * 128;
    int base_tile = base >> 5;
    int remaining = K - base;
    int live_tiles = remaining >= 128 ? 4 : (remaining + 31) / 32;
    // Parallel factor pre-scan: every thread used to walk all
    // nvar*amax_stride partial maxes through global memory serially (~64
    // strided reads on the item's critical path). Now one thread per partial
    // loads it once; the column maxes and global max fold through smem. Max
    // is exact under any order, the division operands are unchanged -> the
    // factors are bit-identical.
    float *s_pre = sm + 128;            // [nvar*amax_stride] raw partials
    float *s_cmax = sm + 128 + TQ_SPEC_MAX_N * 8;   // [nvar] column maxes
    int npre = nvar * amax_stride;
    if (tid < npre) s_pre[tid] = amax[tid];
    __syncthreads();
    if (tid < nvar) {
        float a = 0.0f;
        for (int s = 0; s < amax_stride; s++) {
            float p = s_pre[tid * amax_stride + s];
            if (p > a) a = p;
        }
        s_cmax[tid] = a;
    }
    __syncthreads();
    __shared__ float s_gmax;
    if (tid == 0) {
        float g = 0.0f;
        for (int v = 0; v < nvar; v++)
            if (s_cmax[v] > g) g = s_cmax[v];
        s_gmax = g;
    }
    __syncthreads();
    float gmax = s_gmax;
    float f[TQ_SPEC_MAX_N];
    for (int v = 0; v < nvar; v++) {
        float a = s_cmax[v];
        f[v] = (gmax == 0.0f) ? 1.0f : ((a > 0.0f) ? gmax / a : 1.0f);
    }
    if (block == 0 && tid < nvar) A->factors[tid] = f[tid];
    size_t gstride = (size_t)((K + 31) / 32) * 256;
    // Zeroing is only load-bearing for a partial K-tail block: stale bytes at
    // k >= K would corrupt every column's dot product. For a full 128-block
    // every consumed (k, column) byte is overwritten below, and bytes of
    // column slots beyond nvar feed MMA lanes whose results are discarded at
    // the partials write -- so the pre-zero of 1-2KB/item is dead work.
    if (remaining < 128) {
        for (int i = tid; i < live_tiles * 256; i += blockDim.x) {
            A->bfrag[base_tile * 256 + i] = 0;
            if (nvar > 8) A->bfrag[gstride + base_tile * 256 + i] = 0;
        }
    }
    float local = 0.0f;
    for (int v = 0; v < nvar; v++) {
        const float *xv = x + (size_t)v * K;
        for (int i = tid; i < 128 && base + i < K; i += blockDim.x) {
            float a = fabsf(xv[base + i] * f[v]);
            if (a > local) local = a;
        }
    }
    // warp-shuffle absmax (exact max, any order): replaces a 7-level smem tree
    // with 7 syncthreads by one shuffle reduce + one syncthreads
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        float t2 = __shfl_down_sync(0xffffffffu, local, o);
        if (t2 > local) local = t2;
    }
    if ((tid & 31) == 0) sm[tid >> 5] = local;
    __syncthreads();
    __shared__ float s_bsi;
    if (tid == 0) {
        float m = sm[0];
        for (int w8 = 1; w8 < (int)(blockDim.x >> 5); w8++)
            if (sm[w8] > m) m = sm[w8];
        float raw = m > 0.0f ? m / 448.0f : 1.0f;
        s_bsi = exp2f(ceilf(log2f(raw)));
        A->bscale[block] = s_bsi;
    }
    __syncthreads();
    float bsi = s_bsi;
    for (int v = 0; v < nvar; v++) {
        const float *xv = x + (size_t)v * K;
        uint8_t *bg = A->bfrag + (v >> 3) * gstride;
        int vc = v & 7;
        for (int i = tid; i < 128 && base + i < K; i += blockDim.x) {
            int idx = base + i;
            int tile = idx >> 5;
            int k_in_tile = idx & 31;
            bg[tile * 256 + 32 * vc + k_in_tile] = tq_float_to_e4m3((xv[idx] * f[v]) / bsi);
        }
    }
    __syncthreads();
    // Publish this K-block: GEMV items gate on exactly the blocks of their
    // k-slice instead of a full grid barrier, so the GEMV stream starts while
    // the quant tail is still running. syncthreads above ordered the block's
    // writes before thread 0's release fence.
    if (qtag && threadIdx.x == 0) {
        __threadfence();
        atomicExch(&A->scratch[TQ_PERSIST_QFLAGS_OFF + block], qtag);
    }
}

// One GEMV work item = (m16 tile, k-split slice) of one weight, executed by one warp.
// Verbatim inner loop of k_tq_qmma_sf_gemv_tiled_nk_split.
template <bool USE_LDSM = false>
static __device__ void tq_persist_gemv_item(const tq_spec_persist_args_t *A,
                                            const tq_pw_t *w, int tile_m, int split,
                                            int qtag, uint8_t *ldsm_stage = nullptr,
                                            int *p2p_done = nullptr) {
    int lane = threadIdx.x & 31;
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
    float c4 = 0.0f, c5 = 0.0f, c6 = 0.0f, c7 = 0.0f;   // column group 2 (cols 8..15)
    const uint32_t *b_base = (const uint32_t *)A->bfrag;
    const float *act_scale_inv = A->bscale;
    int scale_row = tile_m >> 3;
    int kt_per = w->Kt / w->ks;
    int tk_begin = split * kt_per;
    int tk_end = tk_begin + kt_per;
    if (qtag) {
        // quant->GEMV pipelining: wait only for the 128-K quant blocks this
        // k-slice reads (tags ascend across phases/halves -> no flag resets
        // mid-launch). Trailing threadfence = acquire for the bfrag reads.
        int b0 = tk_begin >> 2;
        int b1 = (tk_end + 3) >> 2;
        volatile int *qf = (volatile int *)&A->scratch[TQ_PERSIST_QFLAGS_OFF];
        int ok = 0;
        while (!ok) {
            int mine = 1;
            for (int b = b0 + lane; b < b1; b += 32) mine &= (qf[b] >= qtag);
            ok = __all_sync(0xffffffffu, mine);
            if (!ok) __nanosleep(64);
        }
        __threadfence();
    }
    int nvar = A->N;
    int M = w->M, Kt = w->Kt;
    int g2 = nvar > 8;                       // second m16n8 column group active
    const uint32_t *b_base2 = b_base + (size_t)Kt * 64;

    if (USE_LDSM) {
        // SM120 hardware FP6 unpack (PROBE, TQ_LDSM): weights are in LDSM-natural
        // order (load-time repack). Per k-tile: cp.async the 384 B tile into a
        // 16 B-aligned-row SMEM stage, ldmatrix.U6 into the operand-A fragment
        // (bit-identical to the LOP3 path), then the same e2m3.e4m3 QMMA. The
        // staging adds SMEM traffic + sync but reads the SAME 384 B/tile -> tests
        // whether HW unpack frees issue slots in the latency-bound persistent item.
        uint8_t *stg = ldsm_stage + (size_t)(threadIdx.x >> 5) * 512;  // 32 blocks x 16 B
        const uint32_t *a_base = w->a + (size_t)tile_m * Kt * 96;
        uint32_t blk_addr;
        asm volatile("{ .reg .u64 u; cvta.to.shared.u64 u, %1; cvt.u32.u64 %0, u; }"
                     : "=r"(blk_addr) : "l"(stg + lane * 16));
        *(uint32_t *)(stg + lane * 16 + 12) = 0;   // zero the 4 B/row pad once
        for (int tk = tk_begin; tk < tk_end; tk++) {
            const uint32_t *atile = a_base + (size_t)tk * 96;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(blk_addr + 0), "l"(&atile[lane * 3 + 0]));
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(blk_addr + 4), "l"(&atile[lane * 3 + 1]));
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(blk_addr + 8), "l"(&atile[lane * 3 + 2]));
            asm volatile("cp.async.commit_group;\n");
            asm volatile("cp.async.wait_group 0;\n");
            __syncwarp();
            uint32_t a0, a1, a2, a3;
            asm volatile("ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b6x16_p32 {%0,%1,%2,%3}, [%4];\n"
                         : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(blk_addr));
            int b_idx = tk * 64 + lane * 2;
            uint32_t b0 = __ldg(&b_base[b_idx + 0]);
            uint32_t b1 = __ldg(&b_base[b_idx + 1]);
            uint32_t sfa = tq_ue8m0_pack4(w->scale_inv[scale_row * w->scale_cols + (w->scale_cols == Kt ? tk : (tk >> 2))]);
            uint32_t sfb = tq_ue8m0_pack4(act_scale_inv[tk >> 2]);
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X"
                ".f32.e2m3.e4m3.f32.ue8m0 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, {0,1}, %15, {0,1};"
                : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                  "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(sfa), "r"(sfb));
            if (g2) {
                uint32_t bb0 = __ldg(&b_base2[b_idx + 0]);
                uint32_t bb1 = __ldg(&b_base2[b_idx + 1]);
                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X"
                    ".f32.e2m3.e4m3.f32.ue8m0 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, {0,1}, %15, {0,1};"
                    : "+f"(c4), "+f"(c5), "+f"(c6), "+f"(c7)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(bb0), "r"(bb1),
                      "f"(c4), "f"(c5), "f"(c6), "f"(c7), "r"(sfa), "r"(sfb));
            }
            __syncwarp();   // stage buffer reused by the next k-tile
        }
    } else if (w->abyte) {
        // HW-direct E2M3 byte containers (128 words/tile): the entire 6-bit
        // decode ALU block disappears -- loads feed the e2m3 QMMA directly.
        // Same 4-tile load burst as the dense path (MSHR batching), bit-exact
        // (same codes, scales, MMA order); +33% weight bytes.
        const uint32_t *a_base = w->a + (size_t)tile_m * Kt * 128;
        int tk = tk_begin;
        for (; tk + 3 < tk_end; tk += 4) {
            int a_idx = tk * 128 + lane * 4;
            int b_idx = tk * 64 + lane * 2;
            uint32_t AA[4][4];
            uint32_t BB[4][2];
            uint32_t BB2[4][2];
            #pragma unroll
            for (int u = 0; u < 4; u++) {
                AA[u][0] = __ldg(&a_base[a_idx + 128 * u + 0]);
                AA[u][1] = __ldg(&a_base[a_idx + 128 * u + 1]);
                AA[u][2] = __ldg(&a_base[a_idx + 128 * u + 2]);
                AA[u][3] = __ldg(&a_base[a_idx + 128 * u + 3]);
                BB[u][0] = __ldg(&b_base[b_idx + 64 * u + 0]);
                BB[u][1] = __ldg(&b_base[b_idx + 64 * u + 1]);
                if (g2) {
                    BB2[u][0] = __ldg(&b_base2[b_idx + 64 * u + 0]);
                    BB2[u][1] = __ldg(&b_base2[b_idx + 64 * u + 1]);
                }
            }
            #pragma unroll
            for (int half = 0; half < 4; half++) {
                int tkh = tk + half;
                uint32_t sfa = tq_ue8m0_pack4(w->scale_inv[scale_row * w->scale_cols + (w->scale_cols == Kt ? tkh : (tkh >> 2))]);
                uint32_t sfb = tq_ue8m0_pack4(act_scale_inv[tkh >> 2]);
                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X"
                    ".f32.e2m3.e4m3.f32.ue8m0 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, {0,1}, %15, {0,1};"
                    : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                    : "r"(AA[half][0]), "r"(AA[half][1]), "r"(AA[half][2]), "r"(AA[half][3]),
                      "r"(BB[half][0]), "r"(BB[half][1]),
                      "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(sfa), "r"(sfb));
                if (g2) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X"
                        ".f32.e2m3.e4m3.f32.ue8m0 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, {0,1}, %15, {0,1};"
                        : "+f"(c4), "+f"(c5), "+f"(c6), "+f"(c7)
                        : "r"(AA[half][0]), "r"(AA[half][1]), "r"(AA[half][2]), "r"(AA[half][3]),
                          "r"(BB2[half][0]), "r"(BB2[half][1]),
                          "f"(c4), "f"(c5), "f"(c6), "f"(c7), "r"(sfa), "r"(sfb));
                }
            }
        }
        for (; tk < tk_end; tk++) {
            int a_idx = tk * 128 + lane * 4;
            int b_idx = tk * 64 + lane * 2;
            uint32_t a0 = __ldg(&a_base[a_idx + 0]);
            uint32_t a1 = __ldg(&a_base[a_idx + 1]);
            uint32_t a2 = __ldg(&a_base[a_idx + 2]);
            uint32_t a3 = __ldg(&a_base[a_idx + 3]);
            uint32_t b0 = __ldg(&b_base[b_idx + 0]);
            uint32_t b1 = __ldg(&b_base[b_idx + 1]);
            uint32_t sfa = tq_ue8m0_pack4(w->scale_inv[scale_row * w->scale_cols + (w->scale_cols == Kt ? tk : (tk >> 2))]);
            uint32_t sfb = tq_ue8m0_pack4(act_scale_inv[tk >> 2]);
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X"
                ".f32.e2m3.e4m3.f32.ue8m0 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, {0,1}, %15, {0,1};"
                : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                  "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(sfa), "r"(sfb));
            if (g2) {
                uint32_t bb0 = __ldg(&b_base2[b_idx + 0]);
                uint32_t bb1 = __ldg(&b_base2[b_idx + 1]);
                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X"
                    ".f32.e2m3.e4m3.f32.ue8m0 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, {0,1}, %15, {0,1};"
                    : "+f"(c4), "+f"(c5), "+f"(c6), "+f"(c7)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(bb0), "r"(bb1),
                      "f"(c4), "f"(c5), "f"(c6), "f"(c7), "r"(sfa), "r"(sfb));
            }
        }
    } else if (w->ae2m1) {
        // 4-bit E2M1 payload (64 words/tile): nibble<<2 embeds each code as its
        // exact 6-bit E2M3 twin for the same QMMA (see draft-head notes). HALF
        // the weight bytes of dense E2M3 -- the unpack ALU rides under DRAM.
        const uint32_t *a_base = w->a + (size_t)tile_m * Kt * 64;
        int tk = tk_begin;
        for (; tk + 3 < tk_end; tk += 4) {
            int a_idx = tk * 64 + lane * 2;
            int b_idx = tk * 64 + lane * 2;
            uint32_t QQ[4][2];
            uint32_t BB[4][2];
            uint32_t BB2[4][2];
            #pragma unroll
            for (int u = 0; u < 4; u++) {
                QQ[u][0] = __ldg(&a_base[a_idx + 64 * u + 0]);
                QQ[u][1] = __ldg(&a_base[a_idx + 64 * u + 1]);
                BB[u][0] = __ldg(&b_base[b_idx + 64 * u + 0]);
                BB[u][1] = __ldg(&b_base[b_idx + 64 * u + 1]);
                if (g2) {
                    BB2[u][0] = __ldg(&b_base2[b_idx + 64 * u + 0]);
                    BB2[u][1] = __ldg(&b_base2[b_idx + 64 * u + 1]);
                }
            }
            #pragma unroll
            for (int half = 0; half < 4; half++) {
                int tkh = tk + half;
                uint32_t a0, a1, a2, a3;
                tq_unpack_e2m1_16(QQ[half][0], QQ[half][1], a0, a1, a2, a3);
                uint32_t sfa = tq_ue8m0_pack4(w->scale_inv[scale_row * w->scale_cols + (w->scale_cols == Kt ? tkh : (tkh >> 2))]);
                uint32_t sfb = tq_ue8m0_pack4(act_scale_inv[tkh >> 2]);
                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X"
                    ".f32.e2m3.e4m3.f32.ue8m0 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, {0,1}, %15, {0,1};"
                    : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(BB[half][0]), "r"(BB[half][1]),
                      "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(sfa), "r"(sfb));
                if (g2) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X"
                        ".f32.e2m3.e4m3.f32.ue8m0 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, {0,1}, %15, {0,1};"
                        : "+f"(c4), "+f"(c5), "+f"(c6), "+f"(c7)
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(BB2[half][0]), "r"(BB2[half][1]),
                          "f"(c4), "f"(c5), "f"(c6), "f"(c7), "r"(sfa), "r"(sfb));
                }
            }
        }
        for (; tk < tk_end; tk++) {
            int a_idx = tk * 64 + lane * 2;
            int b_idx = tk * 64 + lane * 2;
            uint32_t Q0 = __ldg(&a_base[a_idx + 0]);
            uint32_t Q1 = __ldg(&a_base[a_idx + 1]);
            uint32_t a0, a1, a2, a3;
            tq_unpack_e2m1_16(Q0, Q1, a0, a1, a2, a3);
            uint32_t b0 = __ldg(&b_base[b_idx + 0]);
            uint32_t b1 = __ldg(&b_base[b_idx + 1]);
            uint32_t sfa = tq_ue8m0_pack4(w->scale_inv[scale_row * w->scale_cols + (w->scale_cols == Kt ? tk : (tk >> 2))]);
            uint32_t sfb = tq_ue8m0_pack4(act_scale_inv[tk >> 2]);
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X"
                ".f32.e2m3.e4m3.f32.ue8m0 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, {0,1}, %15, {0,1};"
                : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                  "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(sfa), "r"(sfb));
            if (g2) {
                uint32_t bb0 = __ldg(&b_base2[b_idx + 0]);
                uint32_t bb1 = __ldg(&b_base2[b_idx + 1]);
                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X"
                    ".f32.e2m3.e4m3.f32.ue8m0 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, {0,1}, %15, {0,1};"
                    : "+f"(c4), "+f"(c5), "+f"(c6), "+f"(c7)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(bb0), "r"(bb1),
                      "f"(c4), "f"(c5), "f"(c6), "f"(c7), "r"(sfa), "r"(sfb));
            }
        }
    } else if (gc_weight_e2m3) {
        // Software-pipelined pair: both K-tiles' A/B words are loaded before the
        // first unpack so memory latency overlaps the 6-bit decode ALU. Pure load
        // reordering (per-tile unpack + MMA accumulation order unchanged). The
        // second column group reuses the unpacked A registers (the N=16 payoff).
        const uint32_t *a_base = w->a + (size_t)tile_m * Kt * 96;
        int tk = tk_begin;
        for (; tk + 3 < tk_end; tk += 4) {
            int a_idx = tk * 96 + lane * 3;
            int b_idx = tk * 64 + lane * 2;
            uint32_t PP[4][3];
            uint32_t BB[4][2];
            uint32_t BB2[4][2];
            #pragma unroll
            for (int u = 0; u < 4; u++) {
                PP[u][0] = __ldg(&a_base[a_idx + 96 * u + 0]);
                PP[u][1] = __ldg(&a_base[a_idx + 96 * u + 1]);
                PP[u][2] = __ldg(&a_base[a_idx + 96 * u + 2]);
                BB[u][0] = __ldg(&b_base[b_idx + 64 * u + 0]);
                BB[u][1] = __ldg(&b_base[b_idx + 64 * u + 1]);
                if (g2) {
                    BB2[u][0] = __ldg(&b_base2[b_idx + 64 * u + 0]);
                    BB2[u][1] = __ldg(&b_base2[b_idx + 64 * u + 1]);
                }
            }
            #pragma unroll
            for (int half = 0; half < 4; half++) {
                const uint32_t *P = PP[half];
                uint32_t bb0 = BB[half][0];
                uint32_t bb1 = BB[half][1];
                int tkh = tk + half;
                uint32_t a0, a1, a2, a3;
                tq_unpack_e2m3_16(P[0], P[1], P[2], a0, a1, a2, a3);
                uint32_t sfa = tq_ue8m0_pack4(w->scale_inv[scale_row * w->scale_cols + (w->scale_cols == Kt ? tkh : (tkh >> 2))]);
                uint32_t sfb = tq_ue8m0_pack4(act_scale_inv[tkh >> 2]);
                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X"
                    ".f32.e2m3.e4m3.f32.ue8m0 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, {0,1}, %15, {0,1};"
                    : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(bb0), "r"(bb1),
                      "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(sfa), "r"(sfb));
                if (g2) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X"
                        ".f32.e2m3.e4m3.f32.ue8m0 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, {0,1}, %15, {0,1};"
                        : "+f"(c4), "+f"(c5), "+f"(c6), "+f"(c7)
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(BB2[half][0]), "r"(BB2[half][1]),
                          "f"(c4), "f"(c5), "f"(c6), "f"(c7), "r"(sfa), "r"(sfb));
                }
            }
        }
        for (; tk < tk_end; tk++) {
            int a_idx = tk * 96 + lane * 3;
            uint32_t P[3] = {
                __ldg(&a_base[a_idx + 0]), __ldg(&a_base[a_idx + 1]), __ldg(&a_base[a_idx + 2]),
            };
            uint32_t a0, a1, a2, a3;
            tq_unpack_e2m3_16(P[0], P[1], P[2], a0, a1, a2, a3);
            int b_idx = tk * 64 + lane * 2;
            uint32_t b0 = __ldg(&b_base[b_idx + 0]);
            uint32_t b1 = __ldg(&b_base[b_idx + 1]);
            uint32_t sfa = tq_ue8m0_pack4(w->scale_inv[scale_row * w->scale_cols + (w->scale_cols == Kt ? tk : (tk >> 2))]);
            uint32_t sfb = tq_ue8m0_pack4(act_scale_inv[tk >> 2]);
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X"
                ".f32.e2m3.e4m3.f32.ue8m0 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, {0,1}, %15, {0,1};"
                : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                  "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(sfa), "r"(sfb));
            if (g2) {
                uint32_t bb0 = __ldg(&b_base2[b_idx + 0]);
                uint32_t bb1 = __ldg(&b_base2[b_idx + 1]);
                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X"
                    ".f32.e2m3.e4m3.f32.ue8m0 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, {0,1}, %15, {0,1};"
                    : "+f"(c4), "+f"(c5), "+f"(c6), "+f"(c7)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(bb0), "r"(bb1),
                      "f"(c4), "f"(c5), "f"(c6), "f"(c7), "r"(sfa), "r"(sfb));
            }
        }
    } else {
        const uint32_t *a_base = w->a + (size_t)tile_m * Kt * 128;
        for (int tk = tk_begin; tk < tk_end; tk++) {
            int a_idx = tk * 128 + lane * 4;
            int b_idx = tk * 64 + lane * 2;
            uint32_t a0 = __ldg(&a_base[a_idx + 0]);
            uint32_t a1 = __ldg(&a_base[a_idx + 1]);
            uint32_t a2 = __ldg(&a_base[a_idx + 2]);
            uint32_t a3 = __ldg(&a_base[a_idx + 3]);
            uint32_t b0 = __ldg(&b_base[b_idx + 0]);
            uint32_t b1 = __ldg(&b_base[b_idx + 1]);
            uint32_t sfa = tq_ue8m0_pack4(w->scale_inv[scale_row * w->scale_cols + (w->scale_cols == Kt ? tk : (tk >> 2))]);
            uint32_t sfb = tq_ue8m0_pack4(act_scale_inv[tk >> 2]);
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X"
                ".f32.e4m3.e4m3.f32.ue8m0 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, {0,1}, %15, {0,1};"
                : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                  "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(sfa), "r"(sfb));
        }
    }
    int cc = lane & 3;
    int g = lane >> 2;
    int cola = 2 * cc;
    int colb = 2 * cc + 1;
    size_t pbase = (size_t)split * nvar * M;
    if (cola < nvar) {
        w->partials[pbase + (size_t)cola * M + tile_m * 16 + 2 * g] = c0;
        w->partials[pbase + (size_t)cola * M + tile_m * 16 + 2 * g + 1] = c2;
    }
    if (colb < nvar) {
        w->partials[pbase + (size_t)colb * M + tile_m * 16 + 2 * g] = c1;
        w->partials[pbase + (size_t)colb * M + tile_m * 16 + 2 * g + 1] = c3;
    }
    if (g2) {
        int cola2 = 8 + 2 * cc;
        int colb2 = 9 + 2 * cc;
        if (cola2 < nvar) {
            w->partials[pbase + (size_t)cola2 * M + tile_m * 16 + 2 * g] = c4;
            w->partials[pbase + (size_t)cola2 * M + tile_m * 16 + 2 * g + 1] = c6;
        }
        if (colb2 < nvar) {
            w->partials[pbase + (size_t)colb2 * M + tile_m * 16 + 2 * g] = c5;
            w->partials[pbase + (size_t)colb2 * M + tile_m * 16 + 2 * g + 1] = c7;
        }
    }
    // TQ_PERSIST_P2P: publish this (m-tile, k-split) completion so reduce_final
    // can start on it without waiting for the whole grid. Release fence (every
    // lane) orders the partials writes before the counter bump.
    if (p2p_done) {
        __threadfence();
        __syncwarp();
        if (lane == 0) atomicAdd(&p2p_done[tile_m], 1);
    }
}

// Inline reduced read of one element of a group-1 GEMV output: (sum of split
// partials in ascending order) / factors[node]. Bit-identical to materializing
// the reduce, so consumer phases can read projections straight from partials.
static __device__ __forceinline__ float tq_persist_red_read(const tq_spec_persist_args_t *A,
                                                            const tq_pw_t *w, int idx) {
    int total = A->N * w->M;
    float sum = 0.0f;
    for (int s = 0; s < w->ks; s++)
        sum += w->partials[(size_t)s * total + idx];
    return sum / A->factors[idx / w->M];
}

// Reduce one TQ_RED_CHUNK-element chunk of a weight's [N x M] output: out =
// (sum of split partials) / factors[col]. Same split order and divide as the
// pc-reduce kernel. 512 chunks: the 1024 chunking used at most
// ~1/3 of the 340 CTAs; doubling the item count stays inside one wave and
// halves the per-item latency chain (pure re-chunk of element-wise work).
#define TQ_RED_CHUNK 512
static __device__ void tq_persist_reduce_item(const tq_spec_persist_args_t *A,
                                              const tq_pw_t *w, int chunk) {
    int total = A->N * w->M;
    int idx0 = chunk * TQ_RED_CHUNK + threadIdx.x;
    for (int idx = idx0; idx < total && idx < chunk * TQ_RED_CHUNK + TQ_RED_CHUNK; idx += blockDim.x) {
        float sum = 0.0f;
        for (int s = 0; s < w->ks; s++)
            sum += w->partials[(size_t)s * total + idx];
        w->out[idx] = sum / A->factors[idx / w->M];
    }
}

// Gated RMSNorm of one (head, node) on the core, in place. Identical arithmetic
// to k_tq_linear_decode_gated_norm_b (sum tree over dim entries is unchanged).
static __device__ void tq_persist_gated_norm_item(const tq_spec_persist_args_t *A,
                                                  int item, float *sm) {
    int head = item % A->heads;
    int slot = item / A->heads;
    int node = A->group_ids[slot];
    int dim = A->dim;
    int tid = threadIdx.x;
    float *outn = A->pack2_in + (size_t)node * A->value_dim;
    float core = 0.0f;
    if (tid < dim) {
        core = outn[head * dim + tid];
        sm[tid] = core * core;
    }
    __syncthreads();
    for (int stride = dim >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) sm[tid] += sm[tid + stride];
        __syncthreads();
    }
    float core_sum = sm[0];
    __syncthreads();
    float aval = 0.0f;
    if (tid < dim) {
        float normed = core * rsqrtf(core_sum / (float)dim + A->eps) * A->gated_norm_w[tid];
        float gate = tq_persist_red_read(A, &A->w1[1], node * A->value_dim + head * dim + tid);
        float val = normed * (gate / (1.0f + expf(-gate)));
        outn[head * dim + tid] = val;
        aval = fabsf(val);
    }
    // absmax fold (exact max): one atomic per item instead of a column-scan phase
    sm[tid] = aval;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride && sm[tid + stride] > sm[tid]) sm[tid] = sm[tid + stride];
        __syncthreads();
    }
    if (tid == 0) atomicMax((unsigned int *)&A->absmax[node], __float_as_uint(sm[0]));
    __syncthreads();
}

// Two-head gated RMSNorm item: N*heads single-head items used only
// 128 of 256 threads AND overflowed the grid (384 items on 340 CTAs = 2 waves).
// Each 128-thread half runs the IDENTICAL per-head sumsq tree / norm / gate math
// as the single-head item, on its own smem half -> bit-exact; the absmax fold
// merges the two heads' maxes (exact max, same atomicMax target) before one RMW.
// Caller gates on dim*2 == blockDim and even heads.
static __device__ void tq_persist_gated_norm_pair_item(const tq_spec_persist_args_t *A,
                                                       int item, float *sm) {
    int hpairs = A->heads >> 1;
    int pair = item % hpairs;
    int slot = item / hpairs;
    int node = A->group_ids[slot];
    int dim = A->dim;
    int tid = threadIdx.x;
    int l = tid & (dim - 1);
    int h = tid >= dim;
    int head = 2 * pair + h;
    float *outn = A->pack2_in + (size_t)node * A->value_dim;
    float core = outn[head * dim + l];
    sm[tid] = core * core;
    __syncthreads();
    float *smh = sm + h * dim;
    for (int stride = dim >> 1; stride > 0; stride >>= 1) {
        if (l < stride) smh[l] += smh[l + stride];
        __syncthreads();
    }
    float core_sum = smh[0];
    __syncthreads();
    float normed = core * rsqrtf(core_sum / (float)dim + A->eps) * A->gated_norm_w[l];
    float gate = tq_persist_red_read(A, &A->w1[1], node * A->value_dim + head * dim + l);
    float val = normed * (gate / (1.0f + expf(-gate)));
    outn[head * dim + l] = val;
    // absmax fold via warp shuffle (exact max; both heads share the node slot)
    float amax_l = fabsf(val);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        float t2 = __shfl_down_sync(0xffffffffu, amax_l, o);
        if (t2 > amax_l) amax_l = t2;
    }
    if ((tid & 31) == 0) sm[tid >> 5] = amax_l;
    __syncthreads();
    if (tid == 0) {
        float m = sm[0];
        for (int w8 = 1; w8 < (int)(blockDim.x >> 5); w8++)
            if (sm[w8] > m) m = sm[w8];
        atomicMax((unsigned int *)&A->absmax[node], __float_as_uint(m));
    }
    __syncthreads();
}

// silu(gate)*up for one TQ_SILU_CHUNK-element chunk into pack2_in, with the
// per-column absmax folded in (a chunk spans at most 2 columns; exact max via
// atomicMax). 512-element chunks: N*I/1024 = 136 items left 204 of
// 340 CTAs idle for the whole phase; 272 half-size items still fit one wave at
// half the per-item latency. Pure re-chunking of element-wise work + exact-max
// folds -> bit-exact by construction.
#define TQ_SILU_CHUNK 512
static __device__ void tq_persist_silu_item(const tq_spec_persist_args_t *A, int chunk,
                                            float *sm) {
    int total = A->N * A->pack2_K;
    int K = A->pack2_K;
    int tid = threadIdx.x;
    int idx0 = chunk * TQ_SILU_CHUNK + tid;
    int last = chunk * TQ_SILU_CHUNK + TQ_SILU_CHUNK - 1;
    if (last > total - 1) last = total - 1;
    int col_lo = (chunk * TQ_SILU_CHUNK) / K;
    int col_hi = last / K;   // uniform: a chunk spans at most 2 columns
    float m_lo = 0.0f, m_hi = 0.0f;
    for (int idx = idx0; idx < total && idx < chunk * TQ_SILU_CHUNK + TQ_SILU_CHUNK; idx += blockDim.x) {
        float g = tq_persist_red_read(A, &A->w1[0], idx);
        float val = (g / (1.0f + expf(-g))) * tq_persist_red_read(A, &A->w1[1], idx);
        A->pack2_in[idx] = val;
        float a = fabsf(val);
        if (idx / K == col_lo) { if (a > m_lo) m_lo = a; }
        else { if (a > m_hi) m_hi = a; }
    }
    // both column folds via one warp-shuffle reduce + one syncthreads (the two
    // 8-level smem trees cost 16 syncthreads per item); max is exact under any
    // order, the atomicMax targets and >0 guards are unchanged
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        float a = __shfl_down_sync(0xffffffffu, m_lo, o);
        float b = __shfl_down_sync(0xffffffffu, m_hi, o);
        if (a > m_lo) m_lo = a;
        if (b > m_hi) m_hi = b;
    }
    if ((tid & 31) == 0) { sm[tid >> 5] = m_lo; sm[8 + (tid >> 5)] = m_hi; }
    __syncthreads();
    if (tid == 0) {
        float lo = sm[0], hi = sm[8];
        for (int w8 = 1; w8 < (int)(blockDim.x >> 5); w8++) {
            if (sm[w8] > lo) lo = sm[w8];
            if (sm[8 + w8] > hi) hi = sm[8 + w8];
        }
        if (lo > 0.0f)
            atomicMax((unsigned int *)&A->absmax[col_lo], __float_as_uint(lo));
        if (col_hi != col_lo && hi > 0.0f)
            atomicMax((unsigned int *)&A->absmax[col_hi], __float_as_uint(hi));
    }
    __syncthreads();
}

// Final reduce of the down-projection + residual add: h = resid + (sum / f).
static __device__ void tq_persist_reduce_final_item(const tq_spec_persist_args_t *A,
                                                    const tq_pw_t *w, int chunk,
                                                    const int *p2p_done = nullptr) {
    int total = A->N * w->M;
    // TQ_PERSIST_P2P: gate on exactly the m-tiles this 512-element slice reads
    // (a strict subset) instead of bar6. Requires M % TQ_RED_CHUNK == 0 so the
    // chunk stays within one node -> a contiguous 32 m-tile window.
    if (p2p_done) {
        int M = w->M;
        int m_start = (chunk * TQ_RED_CHUNK) % M;
        int mt0 = m_start / 16;
        int mt1 = (m_start + TQ_RED_CHUNK - 1) / 16;
        int tid = threadIdx.x;
        __shared__ int s_wait;
        do {
            if (tid == 0) s_wait = 0;
            __syncthreads();
            for (int mt = mt0 + tid; mt <= mt1; mt += blockDim.x)
                if (((volatile int *)p2p_done)[mt] < w->ks) atomicOr(&s_wait, 1);
            __syncthreads();
            if (s_wait) { if (tid == 0) __nanosleep(64); __syncthreads(); }
        } while (s_wait);
        __threadfence();   // acquire: the awaited partials are now visible
    }
    int idx0 = chunk * TQ_RED_CHUNK + threadIdx.x;
    for (int idx = idx0; idx < total && idx < chunk * TQ_RED_CHUNK + TQ_RED_CHUNK; idx += blockDim.x) {
        float sum = 0.0f;
        for (int s = 0; s < w->ks; s++)
            sum += w->partials[(size_t)s * total + idx];
        A->final_out[idx] = A->final_resid[idx] + sum / A->factors[idx / w->M];
    }
}


// Full-attn kv-prep for one (kv_head, node) item; verbatim k_tq_spec_attn_kv_prep
// (256-thread block shape == CTA, same reductions -> bit-identical).
static __device__ void tq_persist_kvprep_item(const tq_spec_persist_args_t *A, int item,
                                              float *sm) {
    int nkv = A->nkv, hd = A->hd, kv_m = A->kv_m;
    int kv_head = item % nkv;
    int node = item / nkv;
    int tid = threadIdx.x;
    float *s_partial = sm;          // [8]
    float *s_ksum = sm + 8;         // [1]
    if (tid < hd) {
        int pos = A->attn_pos[node];
        const float *k_proj = A->w1[1].out + (size_t)node * kv_m;
        const float *v_proj = A->w1[2].out + (size_t)node * kv_m;
        int rotary_dim = 64;
        float kv = k_proj[kv_head * hd + tid];
        float ksum = kv * kv;
        for (int offset = 16; offset > 0; offset >>= 1)
            ksum += __shfl_down_sync(0xffffffff, ksum, offset);
        if ((tid & 31) == 0) s_partial[tid >> 5] = ksum;
        __syncthreads();
        if (tid < 8) {
            float vtmp = s_partial[tid];
            for (int offset = 4; offset > 0; offset >>= 1) vtmp += __shfl_down_sync(0xff, vtmp, offset);
            if (tid == 0) s_ksum[0] = vtmp;
        }
        __syncthreads();
        float k_sum_shared = s_ksum[0];
        kv = kv * rsqrtf(k_sum_shared / (float)hd + A->eps) * (1.0f + tq_bf16_to_float(A->attn_knorm[tid]));
        if (tid < rotary_dim) {
            int idx = tid & 31;
            float freq = powf(A->rope_theta, -((float)(2 * idx) / (float)rotary_dim));
            float angle = (float)pos * freq;
            float c = cosf(angle);
            float s = sinf(angle);
            float k_pair = k_proj[kv_head * hd + ((tid < 32) ? tid + 32 : tid - 32)];
            k_pair = k_pair * rsqrtf(k_sum_shared / (float)hd + A->eps) * (1.0f + tq_bf16_to_float(A->attn_knorm[(tid < 32) ? tid + 32 : tid - 32]));
            float k_rot = (tid < 32) ? -k_pair : k_pair;
            kv = kv * c + k_rot * s;
        }
        if (A->attn_kcache4 && hd == 256) {
            // TQ_KV_Q4: the K archive lives in the rotated space (the commit
            // quantizes it to int4 without re-rotating; scores use rotated q)
            float *rot = sm + 16;          // 256-float scratch (smem is 384)
            rot[tid] = kv;
            __syncthreads();
            tq_fwht256(rot, tid);
            kv = rot[tid];
        }
        A->attn_karch[(size_t)node * kv_m + kv_head * hd + tid] = kv;
        A->attn_varch[(size_t)node * kv_m + kv_head * hd + tid] = v_proj[kv_head * hd + tid];
    }
    __syncthreads();
}

// Full-attn tree attention for one (head, node) item; verbatim k_tq_spec_attn_tree.
// Scores live in a per-CTA gmem slab (A->attn_scores); smem keeps qn|partial|scalars.
static __device__ void tq_persist_attn_item(const tq_spec_persist_args_t *A, int item,
                                            float *sm) {
    int nh = A->nh, nkv = A->nkv, hd = A->hd, kv_m = A->kv_m;
    int head = item % nh;
    int node = item / nh;
    int tid = threadIdx.x;
    float *scores = A->attn_scores + (size_t)blockIdx.x * A->max_seq;
    float *qn = sm;
    float *s_partial = sm + 256;
    float *s_scal = sm + 256 + 8;   // [0]=qsum/denom [1]=max
    if (tid >= hd) { __syncthreads(); __syncthreads(); __syncthreads(); }
    else {
    int pos = A->attn_pos[node];
    int base = A->attn_base >= 0 ? A->attn_base : A->attn_pos[0] - 1;
    const float *q_proj = A->w1[0].out + (size_t)node * A->q_m;
    const int *anc_n = A->attn_anc + node * TQ_SPEC_MAX_N;
    int group = nh / nkv;
    int kv_head = head / group;
    int rotary_dim = 64;
    float qv = q_proj[head * (2 * hd) + tid];
    float qsum = qv * qv;
    for (int offset = 16; offset > 0; offset >>= 1)
        qsum += __shfl_down_sync(0xffffffff, qsum, offset);
    if ((tid & 31) == 0) s_partial[tid >> 5] = qsum;
    __syncthreads();
    if (tid < 8) {
        float vtmp = s_partial[tid];
        for (int offset = 4; offset > 0; offset >>= 1) vtmp += __shfl_down_sync(0xff, vtmp, offset);
        if (tid == 0) s_scal[0] = vtmp;
    }
    __syncthreads();
    float q_sum_shared = s_scal[0];
    qv = qv * rsqrtf(q_sum_shared / (float)hd + A->eps) * (1.0f + tq_bf16_to_float(A->attn_qnorm[tid]));
    if (tid < rotary_dim) {
        int idx = tid & 31;
        float freq = powf(A->rope_theta, -((float)(2 * idx) / (float)rotary_dim));
        float angle = (float)pos * freq;
        float c = cosf(angle);
        float s = sinf(angle);
        float q_pair = q_proj[head * (2 * hd) + ((tid < 32) ? tid + 32 : tid - 32)];
        q_pair = q_pair * rsqrtf(q_sum_shared / (float)hd + A->eps) * (1.0f + tq_bf16_to_float(A->attn_qnorm[(tid < 32) ? tid + 32 : tid - 32]));
        float q_rot = (tid < 32) ? -q_pair : q_pair;
        qv = qv * c + q_rot * s;
    }
    qn[tid] = qv;
    __syncthreads();
    if (A->attn_kcache4 && hd == 256)
        tq_fwht256(qn, tid);   // q4 score space is rotated (arch K rows too)

    float ratio = tq_attn_tail_pp(qn, scores, qn,
                                  A->attn_kcache + (size_t)kv_head * hd,
                                  A->attn_vcache + (size_t)kv_head * hd,
                                  (size_t)nkv * hd,
                                  A->attn_kcache8 ? A->attn_kcache8 + (size_t)kv_head * hd : NULL,
                                  A->attn_kscale + kv_head,
                                  A->attn_vcache8 ? A->attn_vcache8 + (size_t)kv_head * hd : NULL,
                                  A->attn_vscale + kv_head, nkv,
                                  A->attn_kcache4 ? A->attn_kcache4 + (size_t)kv_head * (hd >> 1) : NULL,
                                  A->attn_kcache4 ? q4s_adv(A->attn_kq4s, (size_t)kv_head * 16) : NULL,
                                  A->attn_karch + (size_t)kv_head * hd,
                                  A->attn_varch + (size_t)kv_head * hd,
                                  (size_t)kv_m, anc_n, base, pos, hd, tid);
    float gate = q_proj[head * (2 * hd) + hd + tid];
    float val = ratio * (1.0f / (1.0f + expf(-gate)));
    A->pack2_in[(size_t)node * A->attn_m + head * hd + tid] = val;
    __syncthreads();
    // absmax fold for the pack2 quantizer (exact max, one atomic per item);
    // warp-shuffle reduce instead of an 8-level smem tree with 8 syncthreads
    float amax_l = fabsf(val);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        float t2 = __shfl_down_sync(0xffffffffu, amax_l, o);
        if (t2 > amax_l) amax_l = t2;
    }
    if ((tid & 31) == 0) qn[tid >> 5] = amax_l;
    __syncthreads();
    if (tid == 0) {
        float m = qn[0];
        for (int w8 = 1; w8 < (int)(blockDim.x >> 5); w8++)
            if (qn[w8] > m) m = qn[w8];
        atomicMax((unsigned int *)&A->absmax[node], __float_as_uint(m));
    }
    }
    __syncthreads();
}

// TQ_ATTN_MMA variant of the attention phase: one item = one (q-head, pos-chunk),
// all N nodes in m16n8 tiles (16x fewer K/V passes than per-(head,node) items).
// attn_pair halves the head grid: one item covers heads (2b, 2b+1) of a kv group
// with both m16 tile halves live (same K/V bytes once for two heads).
// attn_nchunks == 1 keeps the full epilogue in the item (same pack2 + absmax-fold
// side effects as tq_persist_attn_item); > 1 writes flash partials to the scores
// slab, combined by the merge items below. __noinline__ so the MMA codegen cannot
// perturb the default path's registers (fp8-tail precedent).
static __device__ __noinline__ void tq_persist_attn_mma_item(const tq_spec_persist_args_t *A,
                                                             int item, uint32_t *sm) {
    int nc = A->attn_nchunks;
    int pair = A->attn_pair;
    int nheads = pair ? A->nh >> 1 : A->nh;
    int head = pair ? 2 * (item % nheads) : item % nheads;
    int chunk = item / nheads;
    float *part = nc > 1 ? A->attn_scores : NULL;
    float *outp = nc > 1 ? NULL : A->pack2_in;
    float *am = nc > 1 ? NULL : A->absmax;
    if (A->attn_mma == 3)
        tq_attn_mma_block<3>(A->w1[0].out, (size_t)A->q_m, A->attn_qnorm,
                             A->attn_kcache, A->attn_vcache,
                             A->attn_kcache8, A->attn_kscale, A->attn_vcache8, A->attn_vscale,
                             A->attn_kcache4, A->attn_kq4s,
                             A->attn_karch, A->attn_varch,
                             A->attn_pos, A->attn_anc, A->attn_base, A->N,
                             head, pair, A->nh, A->nkv, A->hd, A->kv_m, A->eps, A->rope_theta,
                             chunk, nc, outp, (size_t)A->attn_m, am, part, sm);
    else if (A->attn_mma == 2)
        tq_attn_mma_block<2>(A->w1[0].out, (size_t)A->q_m, A->attn_qnorm,
                             A->attn_kcache, A->attn_vcache,
                             A->attn_kcache8, A->attn_kscale, A->attn_vcache8, A->attn_vscale,
                             A->attn_kcache4, A->attn_kq4s,
                             A->attn_karch, A->attn_varch,
                             A->attn_pos, A->attn_anc, A->attn_base, A->N,
                             head, pair, A->nh, A->nkv, A->hd, A->kv_m, A->eps, A->rope_theta,
                             chunk, nc, outp, (size_t)A->attn_m, am, part, sm);
    else
        tq_attn_mma_block<1>(A->w1[0].out, (size_t)A->q_m, A->attn_qnorm,
                             A->attn_kcache, A->attn_vcache,
                             A->attn_kcache8, A->attn_kscale, A->attn_vcache8, A->attn_vscale,
                             A->attn_kcache4, A->attn_kq4s,
                             A->attn_karch, A->attn_varch,
                             A->attn_pos, A->attn_anc, A->attn_base, A->N,
                             head, pair, A->nh, A->nkv, A->hd, A->kv_m, A->eps, A->rope_theta,
                             chunk, nc, outp, (size_t)A->attn_m, am, part, sm);
}

static __device__ __noinline__ void tq_persist_attn_mma_merge_item(
    const tq_spec_persist_args_t *A, int head, float *sm) {
    tq_attn_mma_merge_block(A->w1[0].out, (size_t)A->q_m, head, 0, A->N, A->hd,
                            A->attn_nchunks, A->attn_scores,
                            A->pack2_in, (size_t)A->attn_m, A->absmax, sm);
}

// DeltaNet conv+core for one (slot, blk) item; verbatim k_tq_deltanet_fused_depths
// body with the depth gate counting completed items (identical protocol). While the
// gate is closed the CTA prefetches upcoming weights into L2.
template <int G, int HS>
static __device__ void tq_persist_deltanet_item(const tq_spec_persist_args_t *A,
                                                int item, float *smem) {
    int blocks_x = A->heads * HS;
    int slot = item / blocks_x;
    int blk = item % blocks_x;
    int dim = A->dim;
    float *qn = smem;
    float *kn = smem + dim;
    float *red = smem + 2 * dim;
    int *depth_done = &A->scratch[TQ_PERSIST_DEPTH];

    int my_depth = A->n_depths - 1;
    for (int d = 0; d < A->n_depths; d++) {
        if (slot < A->depth_offsets[d + 1]) { my_depth = d; break; }
    }
    int items_before = A->depth_offsets[my_depth] * blocks_x;
    if (items_before > 0) {
        __shared__ int s_done, s_pf;
        int pf_alive = 1;
        while (1) {
            if (threadIdx.x == 0) s_done = (*(volatile int *)depth_done >= items_before);
            __syncthreads();
            if (s_done) break;
            if (pf_alive) pf_alive = tq_persist_prefetch_step(A, &s_pf);
            else __nanosleep(64);
            __syncthreads();
        }
        __threadfence();
    }

    int node = A->group_ids[slot];
    int par = A->parent_ids[node];
    int kernel_size = A->kernel_size;
    int value_heads = A->heads, key_heads = A->key_heads;
    int conv_dim = A->conv_dim;
    const float *src_conv = (par >= 0) ? A->conv_state_base + (size_t)par * A->conv_f
                                       : A->committed_conv;
    float *dst_conv = A->conv_state_base + (size_t)node * A->conv_f;
    // archive slabs may be BF16 (recur_bf16); the committed live state is FP32
    int src_bf16 = A->recur_bf16 && par >= 0;
    const void *src_rstate_v = (par >= 0)
        ? (A->recur_bf16 ? (const void *)((const uint16_t *)A->recur_base + (size_t)par * A->recur_f)
                         : (const void *)(A->recur_base + (size_t)par * A->recur_f))
        : (const void *)A->committed_recur;
    void *dst_rstate_v = A->recur_bf16
        ? (void *)((uint16_t *)A->recur_base + (size_t)node * A->recur_f)
        : (void *)(A->recur_base + (size_t)node * A->recur_f);
    int xn_base = node * conv_dim;   // qkv read inline from GEMV partials (reduce folded)
    float *core_o = A->pack2_in + (size_t)node * A->value_dim;
    const uint16_t *conv_weight = A->conv_w;

    int head = blk / HS;
    int part = blk % HS;
    int t = threadIdx.x;
    int bs = blockDim.x;

    int ch_per_block = (conv_dim + blocks_x - 1) / blocks_x;
    int ch_start = blk * ch_per_block;
    int ch_end = ch_start + ch_per_block;
    if (ch_end > conv_dim) ch_end = conv_dim;
    const float *xn = A->qkv_in + (size_t)xn_base;
    for (int ch = ch_start + t; ch < ch_end; ch += bs) {
        const float *s_src = src_conv + (size_t)ch * kernel_size;
        float *s_dst = dst_conv + (size_t)ch * kernel_size;
        for (int j = 0; j < kernel_size - 1; j++)
            s_dst[j] = s_src[j + 1];
        s_dst[kernel_size - 1] = xn[ch];
    }

    int group = value_heads / key_heads;
    int key_head = head / group;
    int key_dim_total = key_heads * dim;
    int q_off = key_head * dim;
    int k_off = key_dim_total + key_head * dim;
    for (int i = t; i < dim; i += bs) {
        int qch = q_off + i;
        const float *qs = src_conv + (size_t)qch * kernel_size;
        float qval = 0.0f;
        for (int j = 0; j < kernel_size - 1; j++)
            qval += qs[j + 1] * tq_bf16_to_float(conv_weight[qch * kernel_size + j]);
        qval += xn[qch] * tq_bf16_to_float(conv_weight[qch * kernel_size + (kernel_size - 1)]);
        qn[i] = qval / (1.0f + expf(-qval));

        int kch = k_off + i;
        const float *ks = src_conv + (size_t)kch * kernel_size;
        float kval = 0.0f;
        for (int j = 0; j < kernel_size - 1; j++)
            kval += ks[j + 1] * tq_bf16_to_float(conv_weight[kch * kernel_size + j]);
        kval += xn[kch] * tq_bf16_to_float(conv_weight[kch * kernel_size + (kernel_size - 1)]);
        kn[i] = kval / (1.0f + expf(-kval));
    }
    __syncthreads();

    int tids_per = dim / HS;
    int tid_local = t / G;
    int klane = t % G;
    int tid = part * tids_per + tid_local;
    int v_off = 2 * key_dim_total + head * dim;
    int vch = v_off + tid;
    const float *vs = src_conv + (size_t)vch * kernel_size;
    float vsum = 0.0f;
    for (int j = 0; j < kernel_size - 1; j++)
        vsum += vs[j + 1] * tq_bf16_to_float(conv_weight[vch * kernel_size + j]);
    vsum += xn[vch] * tq_bf16_to_float(conv_weight[vch * kernel_size + (kernel_size - 1)]);
    float v_val = vsum / (1.0f + expf(-vsum));

    for (int i = t; i < dim; i += bs) red[i] = qn[i] * qn[i];
    __syncthreads();
    float qsum = tq_block_reduce_sum_sh(red, dim, t);
    __syncthreads();
    for (int i = t; i < dim; i += bs) red[i] = kn[i] * kn[i];
    __syncthreads();
    float ksum = tq_block_reduce_sum_sh(red, dim, t);
    __syncthreads();
    for (int i = t; i < dim; i += bs) {
        qn[i] = qn[i] * rsqrtf(qsum + 1.0e-6f) * rsqrtf((float)dim);
        kn[i] = kn[i] * rsqrtf(ksum + 1.0e-6f);
    }
    __syncthreads();

    float beta = 1.0f / (1.0f + expf(-A->bp[node * value_heads + head]));
    float softplus_arg = A->ap[node * value_heads + head] + tq_bf16_to_float(A->dt_bias[head]);
    float softplus = log1pf(expf(-fabsf(softplus_arg))) + fmaxf(softplus_arg, 0.0f);
    float gg = -expf(A->A_log[head]) * softplus;
    float decay = expf(gg);
    const float *src32 = (const float *)src_rstate_v + (size_t)head * dim * dim;
    const uint16_t *src16 = (const uint16_t *)src_rstate_v + (size_t)head * dim * dim;
    float *dst32 = (float *)dst_rstate_v + (size_t)head * dim * dim;
    uint16_t *dst16 = (uint16_t *)dst_rstate_v + (size_t)head * dim * dim;

    float st[(128 + G - 1) / G];
    int nkk = 0;
    float psum = 0.0f;
    for (int kk = klane; kk < dim; kk += G) {
        size_t sidx = (size_t)kk * dim + tid;
        float s = src_bf16 ? tq_bf16_to_float(src16[sidx]) : src32[sidx];
        st[nkk++] = s;
        psum += s * kn[kk];
    }
    #pragma unroll
    for (int off = G >> 1; off > 0; off >>= 1)
        psum += __shfl_down_sync(0xffffffffu, psum, off, G);
    float kv_mem = __shfl_sync(0xffffffffu, psum, 0, G) * decay;
    float delta = (v_val - kv_mem) * beta;

    float pcore = 0.0f;
    nkk = 0;
    for (int kk = klane; kk < dim; kk += G) {
        float newv = st[nkk] * decay + kn[kk] * delta;
        size_t sidx = (size_t)kk * dim + tid;
        if (A->recur_bf16) dst16[sidx] = tq_float_to_bf16(newv);
        else dst32[sidx] = newv;
        pcore += newv * qn[kk];
        nkk++;
    }
    #pragma unroll
    for (int off = G >> 1; off > 0; off >>= 1)
        pcore += __shfl_down_sync(0xffffffffu, pcore, off, G);
    float core = __shfl_sync(0xffffffffu, pcore, 0, G);

    if (klane == 0) core_o[head * dim + tid] = core;

    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(depth_done, 1);
}

// --- chain-segment fused DeltaNet (two phases) ---
//
// Everything that does NOT depend on the recurrent state is hoisted into a fully
// parallel prep phase over all nodes: per-node conv slab writes, q/k conv+silu+
// norms, v conv+silu. Conv windows are gathered STATELESSLY via a host-built
// per-node window table (win[n][m] = tree node feeding slot m, or -(s+1) for
// committed-slab slot s), which reproduces the shift-chain values bit-exactly.
// The sequential phase then walks each chain segment with the state slice in
// registers, doing only the recurrence math (dot, FMA, slab/core writes).

// Prep item: same (node, head, part) blk decomposition, loops, and block-reduce
// as the monolithic depth-gated item, so every float is bit-identical to it.
// One designated block per (node, key_head) publishes the normalized qn/kn.
template <int G, int HS>
static __device__ void tq_persist_dnprep_item(const tq_spec_persist_args_t *A,
                                              int item, float *smem) {
    int blocks_x = A->heads * HS;
    int node = item / blocks_x;
    int blk = item % blocks_x;
    int dim = A->dim;
    float *qn = smem;
    float *kn = smem + dim;
    float *red = smem + 2 * dim;

    int kernel_size = A->kernel_size;
    int value_heads = A->heads, key_heads = A->key_heads;
    int conv_dim = A->conv_dim;
    const uint16_t *conv_weight = A->conv_w;
    const int *win = A->segs + 2 * A->N + 1 + node * kernel_size;
    const float *xn = A->qkv_in + (size_t)node * conv_dim;
    float *dst_conv = A->conv_state_base + (size_t)node * A->conv_f;

    int head = blk / HS;
    int part = blk % HS;
    int t = threadIdx.x;
    int bs = blockDim.x;

    // window value at slot m: a tree node's qkv row or a committed conv slot
    #define TQ_WIN_V(m_, ch_)  \
        (win[m_] >= 0 ? A->qkv_in[(size_t)win[m_] * conv_dim + (ch_)]  \
                      : A->committed_conv[(size_t)(ch_) * kernel_size + (-win[m_] - 1)])

    int ch_per_block = (conv_dim + blocks_x - 1) / blocks_x;
    int ch_start = blk * ch_per_block;
    int ch_end = ch_start + ch_per_block;
    if (ch_end > conv_dim) ch_end = conv_dim;
    // flattened (ch, tap) units: one gather + one store per thread (~1 memory
    // round-trip on 4x more threads) instead of a kernel_size-deep serial
    // chain on ch_per_block threads; identical values to identical slots
    int slab_units = (ch_end - ch_start) * kernel_size;
    for (int u = t; u < slab_units; u += bs) {
        int ch = ch_start + u / kernel_size;
        int m = u % kernel_size;
        dst_conv[(size_t)ch * kernel_size + m] =
            (m == kernel_size - 1) ? xn[ch] : TQ_WIN_V(m, ch);
    }

    int group = value_heads / key_heads;
    int key_head = head / group;
    int key_dim_total = key_heads * dim;

    int tids_per = dim / HS;
    int tid_local = t / G;
    int klane = t % G;
    int tid = part * tids_per + tid_local;
    int v_off = 2 * key_dim_total + head * dim;
    int vch = v_off + tid;
    float vsum = 0.0f;
    if (kernel_size == 4) {
        // unrolled taps (the runtime-bound loop serializes each gather behind
        // the previous FFMA on the in-order SM); identical FMA order -> the
        // same float as the generic loop
        float t0 = TQ_WIN_V(0, vch), t1 = TQ_WIN_V(1, vch), t2 = TQ_WIN_V(2, vch);
        float tx = xn[vch];
        float w0 = tq_bf16_to_float(conv_weight[vch * 4 + 0]);
        float w1 = tq_bf16_to_float(conv_weight[vch * 4 + 1]);
        float w2 = tq_bf16_to_float(conv_weight[vch * 4 + 2]);
        float w3 = tq_bf16_to_float(conv_weight[vch * 4 + 3]);
        vsum += t0 * w0; vsum += t1 * w1; vsum += t2 * w2; vsum += tx * w3;
    } else {
        for (int m = 0; m < kernel_size - 1; m++)
            vsum += TQ_WIN_V(m, vch) * tq_bf16_to_float(conv_weight[vch * kernel_size + m]);
        vsum += xn[vch] * tq_bf16_to_float(conv_weight[vch * kernel_size + (kernel_size - 1)]);
    }
    float v_val = vsum / (1.0f + expf(-vsum));

    float *qkn = A->dnprep + (size_t)node * key_heads * 2 * dim + (size_t)key_head * 2 * dim;
    float *vbuf = A->dnprep + (size_t)TQ_SPEC_MAX_N * key_heads * 2 * dim + (size_t)node * A->value_dim;
    if (klane == 0) vbuf[head * dim + tid] = v_val;

    // q/k conv + silu + norms ONLY in the two designated items per (node,
    // key_head): part 0 publishes qn, part 1 publishes kn (split --
    // one item carrying both q and k was the longest pole of the phase; the
    // halves are independent per-channel convs + their own block-reduce on
    // their own vector, so the split publishes identical floats). The other
    // sibling items only do their conv-slab slice + v.
    if ((part == 0 || part == 1) && (head % group) == 0) {
        int q_off = part == 0 ? key_head * dim : key_dim_total + key_head * dim;
        float *vn = part == 0 ? qn : kn;
        for (int i = t; i < dim; i += bs) {
            int qch = q_off + i;
            float qval = 0.0f;
            if (kernel_size == 4) {
                float t0 = TQ_WIN_V(0, qch), t1 = TQ_WIN_V(1, qch), t2 = TQ_WIN_V(2, qch);
                float tx = xn[qch];
                float w0 = tq_bf16_to_float(conv_weight[qch * 4 + 0]);
                float w1 = tq_bf16_to_float(conv_weight[qch * 4 + 1]);
                float w2 = tq_bf16_to_float(conv_weight[qch * 4 + 2]);
                float w3 = tq_bf16_to_float(conv_weight[qch * 4 + 3]);
                qval += t0 * w0; qval += t1 * w1; qval += t2 * w2; qval += tx * w3;
            } else {
                for (int m = 0; m < kernel_size - 1; m++)
                    qval += TQ_WIN_V(m, qch) * tq_bf16_to_float(conv_weight[qch * kernel_size + m]);
                qval += xn[qch] * tq_bf16_to_float(conv_weight[qch * kernel_size + (kernel_size - 1)]);
            }
            vn[i] = qval / (1.0f + expf(-qval));
        }
        __syncthreads();
        for (int i = t; i < dim; i += bs) red[i] = vn[i] * vn[i];
        __syncthreads();
        float vsum2 = tq_block_reduce_sum_sh(red, dim, t);
        __syncthreads();
        if (part == 0) {
            for (int i = t; i < dim; i += bs)
                qkn[i] = vn[i] * rsqrtf(vsum2 + 1.0e-6f) * rsqrtf((float)dim);
        } else {
            for (int i = t; i < dim; i += bs)
                qkn[dim + i] = vn[i] * rsqrtf(vsum2 + 1.0e-6f);
        }
        __syncthreads();
    }
    #undef TQ_WIN_V
    if (part == 0 && t == 0) {
        float *bd = A->dnprep + (size_t)TQ_SPEC_MAX_N * (key_heads * 2 * dim + A->value_dim) +
                    (size_t)(node * value_heads + head) * 2;
        bd[0] = 1.0f / (1.0f + expf(-A->bp[node * value_heads + head]));
        float softplus_arg = A->ap[node * value_heads + head] + tq_bf16_to_float(A->dt_bias[head]);
        float softplus = log1pf(expf(-fabsf(softplus_arg))) + fmaxf(softplus_arg, 0.0f);
        bd[1] = expf(-expf(A->A_log[head]) * softplus);
    }
}

// Sequential phase: one item per (segment, head, part); the recurrent-state
// slice lives in registers across the whole chain. Per node: load prebaked
// qn/kn/v from dnprep, dot+FMA, write the slab (commit can land mid-chain) and
// the core. Side segments gate on node_done[fork parent] only. With recur_bf16
// the in-register chain skips the parent's BF16 rounding (numerically tighter
// than the slab round-trip, so not bit-identical to the unfused bf16 path).
template <int G, int HS>
static __device__ void tq_persist_deltanet_seg_item(const tq_spec_persist_args_t *A,
                                                    int item) {
    int blocks_x = A->heads * HS;
    int seg = item / blocks_x;
    int blk = item % blocks_x;
    // spec_persist_layer_ok gates dim == 128: compile-time bound keeps st[] in
    // registers (a runtime dim makes the indexing dynamic -> local-memory spills,
    // ~16 LDL+STL per thread per chain step)
    const int dim = 128;
    const int *seg_off = A->segs + A->N;
    const int *chain = A->segs + seg_off[seg];
    const int *pub = A->segs + 2 * A->N + 1 + A->N * A->kernel_size;
    int seg_len = seg_off[seg + 1] - seg_off[seg];
    int par0 = A->parent_ids[chain[0]];
    int *node_done = &A->scratch[TQ_PERSIST_NODE_DONE];

    if (par0 >= 0) {
        __shared__ int s_done, s_pf;
        int pf_alive = 1;
        while (1) {
            // lane-parallel volatile poll over the 16 arrival groups (one 64B
            // sector, one L2 round-trip per iteration; a serial 16-read loop
            // adds ~2us of detection latency per poll and chains across gates)
            if ((threadIdx.x >> 5) == 0) {
                int j = threadIdx.x & 31;
                volatile int *g = (volatile int *)&A->scratch[TQ_PERSIST_NODE_GRP +
                                                              par0 * TQ_PERSIST_NODE_GRP_W];
                int v = (j < TQ_PERSIST_NODE_GRP_W) ? g[j] : 0;
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
                if (j == 0) s_done = (v >= blocks_x);
            }
            __syncthreads();
            if (s_done) break;
            if (pf_alive) pf_alive = tq_persist_prefetch_step(A, &s_pf);
            else __nanosleep(64);
            __syncthreads();
        }
        __threadfence();
    }

    int value_heads = A->heads, key_heads = A->key_heads;
    int head = blk / HS;
    int part = blk % HS;
    int t = threadIdx.x;
    int group = value_heads / key_heads;
    int key_head = head / group;
    int tids_per = dim / HS;
    int tid = part * tids_per + t / G;
    int klane = t % G;

    int src_bf16 = A->recur_bf16 && par0 >= 0;
    const void *src_rstate_v = (par0 >= 0)
        ? (A->recur_bf16 ? (const void *)((const uint16_t *)A->recur_base + (size_t)par0 * A->recur_f)
                         : (const void *)(A->recur_base + (size_t)par0 * A->recur_f))
        : (const void *)A->committed_recur;
    const float *src32 = (const float *)src_rstate_v + (size_t)head * dim * dim;
    const uint16_t *src16 = (const uint16_t *)src_rstate_v + (size_t)head * dim * dim;
    float st[128 / G];
    #pragma unroll
    for (int j = 0; j < 128 / G; j++) {
        size_t sidx = (size_t)(klane + j * G) * dim + tid;
        st[j] = src_bf16 ? tq_bf16_to_float(src16[sidx]) : src32[sidx];
    }

    const float *vbase = A->dnprep + (size_t)TQ_SPEC_MAX_N * key_heads * 2 * dim;
    for (int ci = 0; ci < seg_len; ci++) {
        int node = chain[ci];
        const float *qkn = A->dnprep + (size_t)node * key_heads * 2 * dim + (size_t)key_head * 2 * dim;
        const float *qn = qkn, *kn = qkn + dim;
        float v_val = vbase[(size_t)node * A->value_dim + head * dim + tid];
        float *core_o = A->pack2_in + (size_t)node * A->value_dim;
        void *dst_rstate_v = A->recur_bf16
            ? (void *)((uint16_t *)A->recur_base + (size_t)node * A->recur_f)
            : (void *)(A->recur_base + (size_t)node * A->recur_f);
        float *dst32 = (float *)dst_rstate_v + (size_t)head * dim * dim;
        uint16_t *dst16 = (uint16_t *)dst_rstate_v + (size_t)head * dim * dim;

        const float *bd = A->dnprep + (size_t)TQ_SPEC_MAX_N * (key_heads * 2 * dim + A->value_dim) +
                          (size_t)(node * value_heads + head) * 2;
        float beta = bd[0];
        float decay = bd[1];

        float psum = 0.0f;
        #pragma unroll
        for (int j = 0; j < 128 / G; j++)
            psum += st[j] * kn[klane + j * G];
        #pragma unroll
        for (int off = G >> 1; off > 0; off >>= 1)
            psum += __shfl_down_sync(0xffffffffu, psum, off, G);
        float kv_mem = __shfl_sync(0xffffffffu, psum, 0, G) * decay;
        float delta = (v_val - kv_mem) * beta;

        // state-skip: mid-chain slabs are write-skipped (nobody re-reads them --
        // children use registers, forks are published, commit REPLAYS from the
        // nearest fork using the per-layer dnprep archive)
        int write_slab = !A->state_skip || pub[node];
        float pcore = 0.0f;
        #pragma unroll
        for (int j = 0; j < 128 / G; j++) {
            int kk = klane + j * G;
            float newv = st[j] * decay + kn[kk] * delta;
            size_t sidx = (size_t)kk * dim + tid;
            if (write_slab) {
                if (A->recur_bf16) dst16[sidx] = tq_float_to_bf16(newv);
                else dst32[sidx] = newv;
            }
            st[j] = newv;
            pcore += newv * qn[kk];
        }
        #pragma unroll
        for (int off = G >> 1; off > 0; off >>= 1)
            pcore += __shfl_down_sync(0xffffffffu, pcore, off, G);
        float core = __shfl_sync(0xffffffffu, pcore, 0, G);
        if (klane == 0) core_o[head * dim + tid] = core;

        // publish only fork nodes (side segments gate on them); the phase-end
        // barrier provides visibility for everything else
        if (pub[node]) {
            __threadfence();
            __syncthreads();
            if (threadIdx.x == 0)
                atomicAdd(&A->scratch[TQ_PERSIST_NODE_GRP +
                                      node * TQ_PERSIST_NODE_GRP_W +
                                      (blockIdx.x & (TQ_PERSIST_NODE_GRP_W - 1))], 1);
        }
    }
}

// Commit-time recurrence replay (state-skip mode): rebuild the accepted leaf's
// DeltaNet state for every linear layer from the nearest PUBLISHED ancestor slab
// (fork) or the committed live state, re-running the recurrence on the per-layer
// dnprep archive (kn/v/beta/decay). Identical float ops in identical order as the
// verify chain items -> bit-exact with the write-everything path.
// grid = (heads*HS, n_lin), block = 256. replay header: [0]=len, [1]=src_node
// (-1 = live), [2..] = chain nodes in depth-ascending order.
template <int G, int HS>
__global__ void k_tq_spec_commit_replay(
    float *const *live_ptrs, float *recur_base0, size_t lin_stride, size_t recur_f,
    const float *dnprep0, size_t dn_stride,
    const int *replay, int heads, int key_heads, int dim, int value_dim, int n_alloc) {
    int lin = blockIdx.y;
    int blk = blockIdx.x;
    int head = blk / HS;
    int part = blk % HS;
    int t = threadIdx.x;
    int group = heads / key_heads;
    int key_head = head / group;
    int tids_per = dim / HS;
    int tid = part * tids_per + t / G;
    int klane = t % G;
    int chain_len = replay[0];
    int src_node = replay[1];
    const int *chain = replay + 2;

    float *live = live_ptrs[lin];
    const float *src = (src_node >= 0)
        ? recur_base0 + lin * lin_stride + (size_t)src_node * recur_f
        : live;
    const float *src_h = src + (size_t)head * dim * dim;
    float st[128 / G];
    #pragma unroll
    for (int j = 0; j < 128 / G; j++)
        st[j] = src_h[(size_t)(klane + j * G) * dim + tid];

    const float *dnp = dnprep0 + lin * dn_stride;
    const float *vbase = dnp + (size_t)n_alloc * key_heads * 2 * dim;
    for (int ci = 0; ci < chain_len; ci++) {
        int node = chain[ci];
        const float *qkn = dnp + (size_t)node * key_heads * 2 * dim + (size_t)key_head * 2 * dim;
        const float *kn = qkn + dim;
        float v_val = vbase[(size_t)node * value_dim + head * dim + tid];
        const float *bd = dnp + (size_t)n_alloc * (key_heads * 2 * dim + value_dim) +
                          (size_t)(node * heads + head) * 2;
        float beta = bd[0];
        float decay = bd[1];
        float psum = 0.0f;
        #pragma unroll
        for (int j = 0; j < 128 / G; j++)
            psum += st[j] * kn[klane + j * G];
        #pragma unroll
        for (int off = G >> 1; off > 0; off >>= 1)
            psum += __shfl_down_sync(0xffffffffu, psum, off, G);
        float kv_mem = __shfl_sync(0xffffffffu, psum, 0, G) * decay;
        float delta = (v_val - kv_mem) * beta;
        #pragma unroll
        for (int j = 0; j < 128 / G; j++)
            st[j] = st[j] * decay + kn[klane + j * G] * delta;
    }
    float *dst_h = live + (size_t)head * dim * dim;
    #pragma unroll
    for (int j = 0; j < 128 / G; j++)
        dst_h[(size_t)(klane + j * G) * dim + tid] = st[j];
}

// Bit-exact in-kernel replica of k_tq_qwen_rmsnorm_b_amax: one (node, stripe)
// item. 256 threads emulate the 1024-thread block -- virtual thread
// vt = tid + 256k lands in virtual warp (warp + 8k) at the same lane, so the
// shuffle reduction tree (hence the float sum) is bit-identical. The absmax
// side outputs are exact maxes (order-free).
static __device__ void tq_persist_norm_in_item(const tq_spec_persist_args_t *A, int item,
                                               float *sm) {
    int ns = A->pack1_amax_stride;
    int node = item / ns, bx = item % ns;
    int H = A->pack1_K;
    const float *xn = A->norm_h_in + (size_t)node * H;
    float *outn = A->norm_out + (size_t)node * H;
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    float part[4];
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        float ss = 0.0f;
        for (int i = tid + 256 * k; i < H; i += 1024) { float v = xn[i]; ss += v * v; }
        part[k] = ss;
    }
    #pragma unroll
    for (int k = 0; k < 4; k++)
        for (int o = 16; o > 0; o >>= 1) part[k] += __shfl_down_sync(0xffffffffu, part[k], o);
    __syncthreads();
    if (lane == 0) {
        #pragma unroll
        for (int k = 0; k < 4; k++) sm[warp + 8 * k] = part[k];
    }
    __syncthreads();
    if (warp == 0) {
        float ss = sm[lane];
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, o);
        if (lane == 0) sm[0] = rsqrtf(ss / (float)H + A->eps);
    }
    __syncthreads();
    float rms_inv = sm[0];
    int stripe = (H + ns - 1) / ns;
    int begin = bx * stripe;
    int end = begin + stripe;
    if (end > H) end = H;
    float lmax = 0.0f;
    for (int i = begin + tid; i < end; i += 256) {
        float w = 1.0f + tq_bf16_to_float(A->ln_in[i]);
        float v = xn[i] * rms_inv * w;
        outn[i] = v;
        float a = fabsf(v);
        if (a > lmax) lmax = a;
    }
    __syncthreads();
    for (int o = 16; o > 0; o >>= 1) {
        float t = __shfl_down_sync(0xffffffffu, lmax, o);
        if (t > lmax) lmax = t;
    }
    if (lane == 0) sm[warp] = lmax;
    __syncthreads();
    if (warp == 0) {
        float m = (lane < 8) ? sm[lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) {
            float t = __shfl_down_sync(0xffffffffu, m, o);
            if (t > m) m = t;
        }
        if (lane == 0) A->norm_amax[node * ns + bx] = m;
    }
    __syncthreads();
}

// Wide residual phase (fused v2): rn[i] = h[i] + (k-split reduced out-proj)/fac,
// each element computed exactly ONCE (v1 recomputed the ks-strided partials sum
// 8x per stripe item -- the measured 13 us hot spot). Float math identical to
// k_tq_add_rmsnorm_b_amax_red's resid expression.
static __device__ void tq_persist_resid_item(const tq_spec_persist_args_t *A, int item) {
    int H = A->pack1_K;
    int cpn = (H + 1023) >> 10;
    int node = item / cpn, chunk = item % cpn;
    int begin = chunk << 10;
    int end = begin + 1024;
    if (end > H) end = H;
    const float *an = A->norm_h_in + (size_t)node * H;
    float *rn = A->resid_out + (size_t)node * H;
    size_t total = (size_t)A->N * H;
    size_t bbase = (size_t)node * H;
    float fac = A->out_factors[node];
    int ks = A->out_ks;
    for (int i = begin + threadIdx.x; i < end; i += blockDim.x) {
        float bsum = 0.0f;
        for (int sp = 0; sp < ks; sp++) bsum += A->out_partials[(size_t)sp * total + bbase + i];
        rn[i] = an[i] + bsum / fac;
    }
}

// Bit-exact in-kernel replica of k_tq_add_rmsnorm_b_amax_red's NORM part: the
// resid values come pre-materialized (tq_persist_resid_item) -- identical f32
// values, so the 1024-thread-virtualized sumsq tree gives the identical sum.
static __device__ void tq_persist_norm_out_item(const tq_spec_persist_args_t *A, int item,
                                                float *sm) {
    int ns = A->pack1_amax_stride;
    int node = item / ns, bx = item % ns;
    int H = A->pack1_K;
    const float *rn = A->resid_out + (size_t)node * H;
    float *outn = A->norm_out + (size_t)node * H;
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    float part[4];
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        float ss = 0.0f;
        for (int i = tid + 256 * k; i < H; i += 1024) {
            float v = rn[i];
            ss += v * v;
        }
        part[k] = ss;
    }
    #pragma unroll
    for (int k = 0; k < 4; k++)
        for (int o = 16; o > 0; o >>= 1) part[k] += __shfl_down_sync(0xffffffffu, part[k], o);
    __syncthreads();
    if (lane == 0) {
        #pragma unroll
        for (int k = 0; k < 4; k++) sm[warp + 8 * k] = part[k];
    }
    __syncthreads();
    if (warp == 0) {
        float ss = sm[lane];
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, o);
        if (lane == 0) sm[0] = rsqrtf(ss / (float)H + A->eps);
    }
    __syncthreads();
    float rms_inv = sm[0];
    int stripe = (H + ns - 1) / ns;
    int begin = bx * stripe;
    int end = begin + stripe;
    if (end > H) end = H;
    float lmax = 0.0f;
    for (int i = begin + tid; i < end; i += 256) {
        float w = 1.0f + tq_bf16_to_float(A->ln_post[i]);
        float v = rn[i] * rms_inv * w;
        outn[i] = v;
        float a = fabsf(v);
        if (a > lmax) lmax = a;
    }
    __syncthreads();
    for (int o = 16; o > 0; o >>= 1) {
        float t = __shfl_down_sync(0xffffffffu, lmax, o);
        if (t > lmax) lmax = t;
    }
    if (lane == 0) sm[warp] = lmax;
    __syncthreads();
    if (warp == 0) {
        float m = (lane < 8) ? sm[lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) {
            float t = __shfl_down_sync(0xffffffffu, m, o);
            if (t > m) m = t;
        }
        if (lane == 0) A->norm_amax[node * ns + bx] = m;
    }
    __syncthreads();
}

// One full persistent half-layer schedule (quant1 -> GEMV1 -> middle(PART) ->
// quant2 -> GEMV2 [-> final reduce]). Factored out of the kernel so a fused
// launch can run two halves (disjoint barrier/queue pools via A.bar_base /
// A.q_base) without exiting the grid.
template <int G, int HS, int PART, bool USE_LDSM = false>
static __device__ void tq_persist_layer_body(const tq_spec_persist_args_t &A, float *smem,
                                             int *s_grab_p, uint8_t *ldsm_stage = nullptr) {
#define s_grab (*s_grab_p)
    int item;

    // ph0: quantize pack1 into the B-fragment (+ publish factors); the column
    // absmax comes pre-reduced as stripe partials from the host norm kernel.
    // (also zero absmax for the later gated-norm/silu folds)
    if (blockIdx.x == 0 && threadIdx.x < TQ_SPEC_MAX_N) A.absmax[threadIdx.x] = 0.0f;
    // NOTE: per-block quant->GEMV flag pipelining (no bar0, items
    // gate on their k-slice's flags) measured -5.5 tok/s: ~2700 spinning GEMV
    // warps x 10-40 flag reads cost more L2 than the two-level barrier with 340
    // pollers on one root. Mechanism kept inert (qtag=0) for future regimes.
    int q1_items = (A.pack1_K + 127) / 128;
    while ((item = tq_persist_grab(&A, 0, q1_items, &s_grab)) >= 0)
        tq_persist_quant_item(&A, item, A.pack1_in, A.pack1_K,
                              A.pack1_amax, A.pack1_amax_stride, smem, 0);
    tq_persist_bar_pf(&A, 0, 2);   // waiters stream group-1 weights into L2

    // ph1: group-1 GEMVs (per-warp m16-tile x k-split items over all weights).
    // The reduce/postcorrect is NOT materialized -- consumer phases read the
    // partials inline (tq_persist_red_read), saving a phase + barrier.
    {
        int off[5];
        off[0] = 0;
        for (int w = 0; w < A.nw1; w++) {
            int mt = (A.w1[w].M + 15) / 16;
            off[w + 1] = off[w] + mt * A.w1[w].ks;
        }
        int total = off[A.nw1];
        // MSHR-pressure knob: with all 16 warps/SM streaming 4-deep bursts the
        // outstanding-load count oversubscribes the SM's MSHRs (the standalone
        // 32-thread GEMV runs ~90% peak at ~6 warps/SM); idle warps cost nothing
        while (A.gemv_warps == 0 || (threadIdx.x >> 5) < A.gemv_warps) {
            int it;
            if ((threadIdx.x & 31) == 0) it = atomicAdd(&A.scratch[TQ_PERSIST_Q0 + A.q_base + 1], 1);
            it = __shfl_sync(0xffffffffu, it, 0);
            if (it >= total) break;
            int w = 0;
            while (it >= off[w + 1]) w++;
            int local = it - off[w];
            int ks = A.w1[w].ks;
            tq_persist_gemv_item<USE_LDSM>(&A, &A.w1[w], local / ks, local % ks, 0, ldsm_stage);
        }
    }
    tq_persist_bar(&A, 1);

    if constexpr (PART == 0) {
        // ph1b: materialize the DeltaNet inputs (qkv conv stream, b, a) -- they are
        // read several times with latency-critical access patterns, so a parallel
        // reduce beats inline partial sums there. z / gate / up stay inline.
        // NOTE: inlining b/a into the dnprep bd-publisher (red_read)
        // measured NEUTRAL: reduce1 -0.6us but dnprep +0.5us -- the partials'
        // L2 round-trips just move to the publisher's thread 0, the queue
        // drains were never the cost. Keep the materialized reduce.
        for (int wsel = 0; wsel < 3; wsel++) {
            int w = (wsel == 0) ? 0 : wsel + 1;
            int chunks = (A.N * A.w1[w].M + TQ_RED_CHUNK - 1) / TQ_RED_CHUNK;
            while ((item = tq_persist_grab(&A, 2 + wsel, chunks, &s_grab)) >= 0)
                tq_persist_reduce_item(&A, &A.w1[w], item);
        }
        tq_persist_bar(&A, 2);
        // ph2: DeltaNet. Fused path (n_segs > 0): parallel prep over all nodes
        // (conv slabs, qn/kn/v) -> barrier -> register-resident chain segments.
        // Fallback (n_segs == 0): monolithic depth-gated items.
        // n_segs comes from DEVICE memory (graph-replay safe across topologies).
        int n_segs_now = A.segs ? A.segs[TQ_SEGS_NSEGS_SLOT] : 0;
        if (n_segs_now > 0) {
            int prep_items = A.N * A.heads * HS;
            while ((item = tq_persist_grab(&A, 5, prep_items, &s_grab)) >= 0)
                tq_persist_dnprep_item<G, HS>(&A, item, smem);
            tq_persist_bar(&A, 7);
            int dn_items = n_segs_now * A.heads * HS;
            while ((item = tq_persist_grab(&A, 6, dn_items, &s_grab)) >= 0)
                tq_persist_deltanet_seg_item<G, HS>(&A, item);
        } else {
            int dn_items = A.N * A.heads * HS;
            while ((item = tq_persist_grab(&A, 6, dn_items, &s_grab)) >= 0)
                tq_persist_deltanet_item<G, HS>(&A, item, smem);
        }
        tq_persist_bar_pf(&A, 3, 1);
        // ph3: gated RMSNorm of the core (+ per-node absmax fold); two heads
        // per item when the block can hold both halves (single wave at N=8)
        if (2 * A.dim == (int)blockDim.x && (A.heads & 1) == 0) {
            int gn_items = A.N * (A.heads >> 1);
            while ((item = tq_persist_grab(&A, 7, gn_items, &s_grab)) >= 0)
                tq_persist_gated_norm_pair_item(&A, item, smem);
        } else {
            int gn_items = A.N * A.heads;
            while ((item = tq_persist_grab(&A, 7, gn_items, &s_grab)) >= 0)
                tq_persist_gated_norm_item(&A, item, smem);
        }
        tq_persist_bar_pf(&A, 4, 1);
    } else if constexpr (PART == 2) {
        // ph1b: materialize q/k/v
        for (int wsel = 0; wsel < 3; wsel++) {
            int chunks = (A.N * A.w1[wsel].M + TQ_RED_CHUNK - 1) / TQ_RED_CHUNK;
            while ((item = tq_persist_grab(&A, 2 + wsel, chunks, &s_grab)) >= 0)
                tq_persist_reduce_item(&A, &A.w1[wsel], item);
        }
        tq_persist_bar(&A, 2);
        // ph2: k/v norm+rope into the KV archive
        while ((item = tq_persist_grab(&A, 6, A.N * A.nkv, &s_grab)) >= 0)
            tq_persist_kvprep_item(&A, item, smem);
        tq_persist_bar(&A, 3);
        // ph3: tree attention (+ absmax folds via the writes happening before quant2's
        // dedicated absmax... attention writes pack2; fold its absmax like gnorm).
        // TQ_ATTN_MMA: per-(head, pos-chunk) MMA items in the dynamic-smem slab;
        // multi-chunk rounds add a merge phase (bar 7 / queue 5, both otherwise
        // unused by the part-2 body and the fused half-A glue).
        if (A.attn_mma) {
            extern __shared__ uint32_t tq_dyn_sm[];
            // The two MMA items are the ONLY __noinline__ callees that take &A of
            // the by-value kernel parameter. nvcc 13.1 / sm_120f lowers that escaped
            // param address to a generic "param window" pointer (c[0x0][0x348]) and
            // the callee's field reads become LDL through it; the first one
            // (A->attn_nchunks, +0x310) faults with "Invalid __local__ read" -> every
            // round with attn_mma active died (the pos-1024 spec crash: MMA turns on
            // at TQ_ATTN_MMA_MIN_BASE while the persistent path still owns attention
            // below TQ_SPEC_ATTN_LEGACY_MIN). Stage the args in shared memory once
            // per CTA (plain ld.param copy, immediate offsets) and hand the callees
            // that pointer; the inlined body keeps reading the parameter directly.
            __shared__ __align__(16) tq_spec_persist_args_t s_mma_args;
            if (threadIdx.x == 0) s_mma_args = A;
            __syncthreads();
            int nheads = A.attn_pair ? A.nh >> 1 : A.nh;
            while ((item = tq_persist_grab(&A, 7, nheads * A.attn_nchunks, &s_grab)) >= 0)
                tq_persist_attn_mma_item(&s_mma_args, item, tq_dyn_sm);
            if (A.attn_nchunks > 1) {
                tq_persist_bar(&A, 7);
                while ((item = tq_persist_grab(&A, 5, A.nh, &s_grab)) >= 0)
                    tq_persist_attn_mma_merge_item(&s_mma_args, item, smem);
            }
        } else {
            while ((item = tq_persist_grab(&A, 7, A.N * A.nh, &s_grab)) >= 0)
                tq_persist_attn_item(&A, item, smem);
        }
        tq_persist_bar(&A, 4);
    } else {
        // ph2 (mlp): silu(gate)*up (+ per-column absmax fold)
        int si_items = (A.N * A.pack2_K + TQ_SILU_CHUNK - 1) / TQ_SILU_CHUNK;
        while ((item = tq_persist_grab(&A, 6, si_items, &s_grab)) >= 0)
            tq_persist_silu_item(&A, item, smem);
        tq_persist_bar(&A, 4);
    }

    // ph4: quantize pack2 (column absmax was folded into the producer phase)
    int q2_items = (A.pack2_K + 127) / 128;
    while ((item = tq_persist_grab(&A, 8, q2_items, &s_grab)) >= 0)
        tq_persist_quant_item(&A, item, A.pack2_in, A.pack2_K, A.absmax, 1, smem, 0);
    // part-1: stream mlp_down into L2 during the pack2 drain so ph5's GEMV
    // reads from L2 instead of DRAM.
    if constexpr (PART == 1) tq_persist_bar_pf(&A, 5, 1);
    else tq_persist_bar(&A, 5);

    // ph5: group-2 GEMV. TQ_PERSIST_P2P (part 1 only): publish per-m-tile
    // completion so reduce_final can gate on its subset instead of bar6.
    int *p2p2 = nullptr;
    if constexpr (PART == 1) {
        if (gc_use_p2p && (A.w2.M % TQ_RED_CHUNK) == 0 && (A.w2.M + 15) / 16 <= TQ_PERSIST_P2P_N)
            p2p2 = &A.scratch[TQ_PERSIST_P2P_OFF];
    }
    {
        int mt = (A.w2.M + 15) / 16;
        int total = mt * A.w2.ks;
        while (A.gemv_warps == 0 || (threadIdx.x >> 5) < A.gemv_warps) {
            int it;
            if ((threadIdx.x & 31) == 0) it = atomicAdd(&A.scratch[TQ_PERSIST_Q0 + A.q_base + 9], 1);
            it = __shfl_sync(0xffffffffu, it, 0);
            if (it >= total) break;
            tq_persist_gemv_item<USE_LDSM>(&A, &A.w2, it / A.w2.ks, it % A.w2.ks, 0, ldsm_stage, p2p2);
        }
    }

    // part 0 ends here: the host residual kernel reduces the out-projection
    // partials inline. part 1 reduces down + adds the residual itself.
    if constexpr (PART == 1) {
        if (!p2p2) tq_persist_bar(&A, 6);   // P2P replaces bar6 with per-m-tile gating
        int chunks = (A.N * A.w2.M + TQ_RED_CHUNK - 1) / TQ_RED_CHUNK;
        while ((item = tq_persist_grab(&A, 10, chunks, &s_grab)) >= 0)
            tq_persist_reduce_final_item(&A, &A.w2, item, p2p2);
    }

#undef s_grab
}

// Final arrive-and-exit: the last CTA re-zeros the scratch for the next launch
// (no one reads the counters after their final arrive).
static __device__ void tq_persist_exit(const tq_spec_persist_args_t &A) {
    __syncthreads();
    __threadfence();
    if (A.clk && blockIdx.x == 0 && threadIdx.x == 0) A.clk[15] = clock64();
    __shared__ int s_last;
    if (threadIdx.x == 0) {
        int prev = atomicAdd(&A.scratch[TQ_PERSIST_FINAL], 1);
        s_last = (prev == A.ncta - 1);
    }
    __syncthreads();
    if (s_last) {
        // last CTA re-zeros the whole scratch region with the full block
        // (bar groups are already self-reset; re-zeroing them is harmless)
        for (int i = threadIdx.x; i < TQ_PERSIST_SCRATCH_TOTAL; i += blockDim.x)
            A.scratch[i] = 0;
        __threadfence();
    }
}

template <int G, int HS, int PART, bool USE_LDSM = false>
__global__ void __launch_bounds__(256, 2) k_tq_spec_persistent_layer(tq_spec_persist_args_t A) {
    __shared__ float smem[3 * 128];
    __shared__ int s_grab;
    // LDSM staging: 8 warps x 32 blocks x 16 B = 4 KB, only in the TQ_LDSM
    // instantiation (size 1 otherwise -> the default kernel is byte-identical).
    constexpr int STG = USE_LDSM ? (8 * 32 * 16) : 1;
    __shared__ __align__(16) uint8_t ldsm_stage[STG];
    if (A.clk && blockIdx.x == 0 && threadIdx.x == 0) {
        for (int i = 1; i < 15; i++) A.clk[i] = 0;
        A.clk[0] = clock64();
    }
    tq_persist_layer_body<G, HS, PART, USE_LDSM>(A, smem, &s_grab, USE_LDSM ? ldsm_stage : nullptr);
    tq_persist_exit(A);
}

// Fused full-layer launch: [input norm -> PARTA (DeltaNet or full-attn) ->
// residual norm (k-split reduce folded) -> MLP] in ONE persistent grid. Kills
// two standalone norm launches and one cooperative-launch ramp per layer. The
// halves use disjoint barrier/queue pools (A1.bar_base/q_base); the two norm
// phases replicate the standalone kernels bit-exactly.
template <int G, int HS, int PARTA>
__global__ void __launch_bounds__(256, 2) k_tq_spec_persistent_layer_fused(
    tq_spec_persist_args_t A0, tq_spec_persist_args_t A1) {
    __shared__ float smem[3 * 128];
    __shared__ int s_grab;
    int item;
    if (A0.clk && blockIdx.x == 0 && threadIdx.x == 0) {
        for (int i = 1; i < 15; i++) A0.clk[i] = 0;
        A0.clk[0] = clock64();
    }
    if (A1.clk && blockIdx.x == 0 && threadIdx.x == 0) {
        for (int i = 1; i < 15; i++) A1.clk[i] = 0;
        A1.clk[0] = clock64();
    }
    // ph_norm_in (half-A queue 11 / barrier 6 -- both unused by the part-0/2
    // bodies): input rmsnorm of the residual stream.
    while ((item = tq_persist_grab(&A0, 11, A0.N * A0.pack1_amax_stride, &s_grab)) >= 0)
        tq_persist_norm_in_item(&A0, item, smem);
    tq_persist_bar(&A0, 6);
    tq_persist_layer_body<G, HS, PARTA>(A0, smem, &s_grab);
    // w2 (out-projection) partials must be complete before the residual phase
    // reads them (half-B pool id 7; the part-1 body never uses it).
    tq_persist_bar_pf(&A1, 7, 1);
    // ph_resid (half-B queue 11): wide one-pass residual + k-split reduce.
    {
        int cpn = (A1.pack1_K + 1023) >> 10;
        while ((item = tq_persist_grab(&A1, 11, A1.N * cpn, &s_grab)) >= 0)
            tq_persist_resid_item(&A1, item);
    }
    tq_persist_bar(&A1, 2);
    // ph_norm_out (half-A queue 10 -- free in part-0/2 bodies; half-B bar 3):
    // post rmsnorm over the materialized resid.
    while ((item = tq_persist_grab(&A0, 10, A1.N * A1.pack1_amax_stride, &s_grab)) >= 0)
        tq_persist_norm_out_item(&A1, item, smem);
    tq_persist_bar(&A1, 3);
    tq_persist_layer_body<G, HS, 1>(A1, smem, &s_grab);
    tq_persist_exit(A0);
}

// Finalises the head-split core: gated RMSNorm over the head's dim, in place.
__global__ void k_tq_linear_decode_gated_norm(float *out, const float *z,
                                              const float *norm_weight,
                                              int value_heads, int dim, float eps) {
    extern __shared__ float smem[];
    int head = blockIdx.x;
    int tid = threadIdx.x;
    if (head >= value_heads || tid >= dim) return;
    float core = out[head * dim + tid];
    smem[tid] = core * core;
    __syncthreads();
    float core_sum = tq_block_reduce_sum_sh(smem, dim, tid);
    __syncthreads();
    float normed = core * rsqrtf(core_sum / (float)dim + eps) * norm_weight[tid];
    float gate = z[head * dim + tid];
    out[head * dim + tid] = normed * (gate / (1.0f + expf(-gate)));
}

// Batched gated RMSNorm over a group of sibling nodes. grid=(value_heads, gc);
// node = group_ids[blockIdx.y]. out/z have stride core_stride per node. norm shared.
__global__ void k_tq_linear_decode_gated_norm_b(float *out, const float *z,
                                                const float *norm_weight,
                                                int value_heads, int dim, float eps,
                                                const int *group_ids, int core_stride) {
    tq_pdl_enter();
    extern __shared__ float smem[];
    int head = blockIdx.x;
    int tid = threadIdx.x;
    if (head >= value_heads || tid >= dim) return;
    int node = group_ids[blockIdx.y];
    float *outn = out + (size_t)node * core_stride;
    const float *zn = z + (size_t)node * core_stride;
    float core = outn[head * dim + tid];
    smem[tid] = core * core;
    __syncthreads();
    float core_sum = tq_block_reduce_sum_sh(smem, dim, tid);
    __syncthreads();
    float normed = core * rsqrtf(core_sum / (float)dim + eps) * norm_weight[tid];
    float gate = zn[head * dim + tid];
    outn[head * dim + tid] = normed * (gate / (1.0f + expf(-gate)));
}
