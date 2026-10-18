// the persistent layer kernel's host side.
// Part of the forward_qwen.cu unity build: included in order, not a standalone header.

// =============== Persistent layer kernel: host side ===============

// Persistent path (env TQ_SPEC_PERSIST): one kernel per linear-layer half instead
// of ~30 launches; phases are software barriers inside k_tq_spec_persistent_layer.
static int spec_persist_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("TQ_SPEC_PERSIST");
        cached = env ? atoi(env) : 1;
    }
    return cached;
}

static int spec_persist_pf_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("TQ_SPEC_PERSIST_PF");
        cached = env ? atoi(env) : 0;
    }
    return cached;
}

// BF16 tree-state archive (default on with the persistent path): halves the
// DeltaNet fork-state write traffic (~50 -> 25 MB/layer) and the archive VRAM.
// Only the IN-TREE state handoffs are BF16-rounded; the committed live state
// stays FP32 (the commit decompresses the accepted leaf slab). Decided at first
// buffer allocation; TQ_SPEC_STATE_FP32=1 or the non-persistent fallbacks
// (TQ_SPEC_PERSIST=0 / TQ_SPEC_NOBATCH_RECUR) keep the FP32 archive.
// Pointer-swap commit (default ON; TQ_SPEC_COMMIT_SWAP=0 disables). Disabled
// under the CUDA-graph paths: graphs capture archive pointers, half-rotation
// would replay against the wrong half.
static int spec_commit_swap_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("TQ_SPEC_COMMIT_SWAP");
        cached = (e && *e == '0') ? 0 : 1;
        if (getenv("TQ_SPEC_GRAPH") || getenv("TQ_SPEC_GRAPH_CACHED")) cached = 0;
    }
    return cached;
}

static int spec_lin_recur_floats(void);
static int spec_lin_conv_floats(void);

// Copy archive-resident live slabs back into the owned buffers. Needed before
// any round that writes BOTH archive halves (N > 8), since such a round would
// clobber the slab the live pointers reference.
static void spec_commit_unswap(void) {
    if (!g_spec_live_in_archive) return;
    int recur_f = spec_lin_recur_floats(), conv_f = spec_lin_conv_floats();
    for (int L = 0; L < g_qwen.L; L++) {
        tq_layer_t *l = &g_qwen.layers[L];
        if (g_qwen.layer_types[L] != TQ_LAYER_LINEAR_ATTENTION) continue;
        if (l->d_linear_recurrent_state != l->d_linear_recurrent_state_owned) {
            cudaMemcpyAsync(l->d_linear_recurrent_state_owned, l->d_linear_recurrent_state,
                            (size_t)recur_f * sizeof(float), cudaMemcpyDeviceToDevice, g_qwen.stream);
            l->d_linear_recurrent_state = l->d_linear_recurrent_state_owned;
        }
        if (l->d_linear_conv_state != l->d_linear_conv_state_owned) {
            cudaMemcpyAsync(l->d_linear_conv_state_owned, l->d_linear_conv_state,
                            (size_t)conv_f * sizeof(float), cudaMemcpyDeviceToDevice, g_qwen.stream);
            l->d_linear_conv_state = l->d_linear_conv_state_owned;
        }
    }
    g_spec_live_in_archive = 0;
}

static int spec_recur_bf16(void) {
    static int cached = -1;
    if (cached < 0) {
        // Measured neutral-to-slightly-negative e2e (the DeltaNet phase is
        // wave-latency-bound, not write-bound; accept dips ~0.05): OPT-IN, kept
        // for the VRAM headroom it buys (archive halves; enables N=32 prefill).
        const char *env = getenv("TQ_SPEC_STATE_BF16");
        cached = (env && atoi(env)) && spec_persist_enabled() &&
                 !getenv("TQ_SPEC_NOBATCH_RECUR");
    }
    return cached;
}

static int ensure_spec_persist(void) {
    if (g_spec_persist_scratch) return 0;
    size_t bytes = TQ_PERSIST_SCRATCH_TOTAL * sizeof(int) +
                   (size_t)TQ_SPEC_MAX_N * 64 * sizeof(float) +   // fused norm: up to 64 stripes
                   TQ_PERSIST_CLK_SLOTS * sizeof(long long);
    if (cudaMalloc(&g_spec_persist_scratch, bytes) != cudaSuccess)
        return -1;
    if (cudaMemset(g_spec_persist_scratch, 0, bytes) != cudaSuccess)
        return -2;
    g_spec_persist_amax = (float *)(g_spec_persist_scratch + TQ_PERSIST_SCRATCH_TOTAL);
    g_spec_persist_clk = (long long *)(g_spec_persist_amax + TQ_SPEC_MAX_N * 64);
    int dev = 0, sms = 0, occ0 = 0, occ1 = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ0, k_tq_spec_persistent_layer<8, 4, 0>,
                                                      256, 0) != cudaSuccess || occ0 < 1)
        return -3;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ1, k_tq_spec_persistent_layer<8, 4, 1>,
                                                      256, 0) != cudaSuccess || occ1 < 1)
        return -3;
    // TQ_LDSM part-1 instantiation carries a 4 KB staging buffer; its occupancy
    // gates the grid size for the actually-launched kernel (cooperative launch
    // fails if the CTAs are not co-resident).
    int occ1l = occ1;
    if (g_use_ldsm) {
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ1l, k_tq_spec_persistent_layer<8, 4, 1, true>,
                                                          256, 0) != cudaSuccess || occ1l < 1)
            return -3;
        fprintf(stderr, "TQ_LDSM: part1 occupancy default=%d/SM, ldmatrix=%d/SM\n", occ1, occ1l);
        occ1 = occ1l;
    }
    int occ2 = 0;
    // part 2 occupancy must account for the TQ_ATTN_MMA dynamic-smem slab (the
    // cooperative launch fails if ncta CTAs cannot be co-resident)
    size_t dyn2 = g_qwen.attn_mma ? TQ_ATTN_MMA_SMEM : 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ2, k_tq_spec_persistent_layer<8, 4, 2>,
                                                      256, dyn2) != cudaSuccess || occ2 < 1)
        return -3;
    g_spec_persist_ncta = sms * occ0;
    g_spec_persist_ncta1 = sms * occ1;
    g_spec_persist_ncta2 = sms * (occ2 < 2 ? occ2 : 2);
    const char *env = getenv("TQ_SPEC_PERSIST_CTAS");
    if (env) {
        int v = atoi(env);
        if (v >= 1 && v <= sms * occ0) g_spec_persist_ncta = v;
    }
    const char *env1 = getenv("TQ_SPEC_PERSIST_CTAS1");
    if (env1) {
        int v = atoi(env1);
        if (v >= 1 && v <= sms * occ1) g_spec_persist_ncta1 = v;
    }
    fprintf(stderr, "spec persistent kernel: %d SMs, part0 %d CTAs/SM -> %d, part1 %d CTAs/SM -> %d\n",
            sms, occ0, g_spec_persist_ncta, occ1, g_spec_persist_ncta1);
    return 0;
}

// Weight descriptor for the persistent kernel; same auto k-split as spec_gemv.
static int spec_persist_pw(const tq_qmma_weight_t *w, tq_pw_t *pw, float *out,
                           float *partials_base, size_t *partials_off, int N) {
    if (!can_use_qmma_sf_weight(w) || w->row_major || w->word_major) return -1;
    if (N > 8 && !(w->e2m3 || w->e2m1)) return -1;   // g2 column group: e2m3/e2m1/byte items handle it
    pw->a = (const uint32_t *)w->d_A;
    pw->scale_inv = w->d_block_scale_inv;
    pw->scale_cols = w->scale_cols;
    pw->M = w->M;
    pw->Kt = w->Kt;
    pw->abyte = w->e2m3_byte;
    pw->ae2m1 = w->e2m1;
    int ks = qmma_sf_k_split_auto(w->Mt, w->Kt);
    if (ks < 1 || (w->Kt % ks) != 0) ks = 1;
    // persist-only finer-split override (queue items, no grid-size concerns):
    // probes the phase-tail/imbalance hypothesis without touching standalone GEMVs
    {
        static int fks = -1;
        if (fks < 0) {
            const char *e = getenv("TQ_SPEC_PERSIST_KSPLIT");
            fks = e ? atoi(e) : 0;
            if (fks < 0 || fks > 16) fks = 0;
        }
        if (fks > ks && (w->Kt % fks) == 0) ks = fks;
    }
    pw->ks = ks;
    pw->out = out;
    pw->partials = partials_base + *partials_off;
    *partials_off += (size_t)ks * N * w->M;
    return 0;
}

// Can this layer's MLP half run through the persistent kernel?
static int spec_persist_mlp_ok(tq_layer_t *l) {
    const tq_qmma_weight_t *ws[3] = {&l->mlp_gate, &l->mlp_up, &l->mlp_down};
    for (int i = 0; i < 3; i++)
        if (!can_use_qmma_sf_weight(ws[i]) || ws[i]->row_major || ws[i]->word_major) return 0;
    return 1;
}

// Decompose the tree into maximal single-child chains ("segments") for the fused
// DeltaNet items: a node starts a new segment unless it is the FIRST child of its
// parent; the segment then follows first-child links. Node ids are topological
// (parent < child), so segment heads ascend and a side segment always waits on an
// earlier segment -> the monotonic work queue cannot deadlock.
// Also builds the per-node conv-window table used by the stateless prep gathers:
// win[n][m] = tree node whose qkv feeds slot m of node n's window, or -(s+1) for
// slot s of the committed conv slab (window reaches behind the tree root).
// Layout written to out: nodes[N] | off[N+1 region, n_segs+1 used] | win[N*ks] |
// pub[N] (1 = some side segment gates on this fork node -> the chain item must
// fence+count it; non-fork nodes skip the per-node publication entirely).
// Returns n_segs.
static int spec_build_segments(const int *parent, int N, int ks, int *out) {
    int first_child[TQ_SPEC_MAX_N];
    for (int i = 0; i < N; i++) first_child[i] = -1;
    for (int i = N - 1; i >= 0; i--)
        if (parent[i] >= 0) first_child[parent[i]] = i;
    int *off = out + N;
    int pos = 0, n_segs = 0;
    for (int i = 0; i < N; i++) {
        if (parent[i] >= 0 && first_child[parent[i]] == i) continue;
        off[n_segs++] = pos;
        for (int c = i; c >= 0; c = first_child[c]) out[pos++] = c;
    }
    off[n_segs] = pos;
    int *win = out + 2 * N + 1;
    for (int n = 0; n < N; n++) {
        for (int m = 0; m < ks; m++) {
            int a = ks - 1 - m;          // ancestor distance feeding slot m
            int cur = n, steps = 0;
            while (steps < a && cur >= 0) { cur = parent[cur]; steps++; }
            // ran out of tree ancestors after `steps` hops -> committed slot m+steps
            win[n * ks + m] = (cur >= 0) ? cur : -((m + steps) + 1);
        }
    }
    int *pub = win + N * ks;
    for (int n = 0; n < N; n++) pub[n] = 0;
    for (int s = 0; s < n_segs; s++) {
        int p0 = parent[out[off[s]]];
        if (p0 >= 0) pub[p0] = 1;
    }
    return n_segs;
}

// Fused full-layer persistent launches (in-norm + partA + residual-norm + MLP in
// one grid): kills 2 standalone norm launches + 1 cooperative ramp per layer.
// Opt-in while gating: TQ_SPEC_FUSE_LAYER=1.
static int spec_fuse_layer_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("TQ_SPEC_FUSE_LAYER");
        cached = (e && *e == '1') ? 1 : 0;
    }
    return cached;
}

// Chain-segment fusion for the persistent DeltaNet phase (default ON; =0 for A/B).
static int spec_seg_fuse_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("TQ_SPEC_SEG_FUSE");
        cached = (e && *e == '0') ? 0 : 1;
    }
    return cached;
}

// State-write skip + commit replay (compute-for-bandwidth: mid-chain recur slabs
// are never written; commit replays the accepted path from the nearest fork using
// the per-layer dnprep archive). Gated bit-exact (identical committed tokens);
// decode +0.9%, prefill +3.0%. Default ON; TQ_SPEC_STATE_SKIP=0 disables.
static int spec_state_skip_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("TQ_SPEC_STATE_SKIP");
        cached = (e && *e == '0') ? 0 : 1;
    }
    return cached;
}

static int g_spec_persist_nsegs = 0;   // segments uploaded for this round (0 = depth waves)

// Can this full-attention layer's attn half run through the persistent kernel?
// CONTEXT GATE: past the threshold the standalone-kernel (legacy) branch beats
// the persistent part2 -- its attention kernel gets its own occupancy instead
// of sharing the 128-reg __launch_bounds__(256,2) budget (already spilling,
// STACK:192), and the branch is what unlocks the group/GROUP2 attention items.
// MEASURED: on the 5090 legacy wins at every base from 32k; on
// the PRO 6000 the persistent overlap won below ~96k against PAIR, but with
// GROUP2 in the standalone branch legacy+group2 wins from 32k
// there too (32k 22.7 vs 23.8, 64k 23.5 vs 25.8, 98k 25.8 vs 27.1 ms/round)
// -> flat 32768 default. Free-run trajectories are identical across the flip
// at matched kernels (scheduling only). TQ_SPEC_ATTN_LEGACY=1/0 forces
// always/never; TQ_SPEC_ATTN_LEGACY_MIN moves the auto threshold.
static int spec_persist_attn_ok(tq_layer_t *l) {
    static int legacy = -2, legacy_min = -1;
    if (legacy == -2) {
        const char *e = getenv("TQ_SPEC_ATTN_LEGACY");
        legacy = (e && *e) ? (atoi(e) != 0 ? 1 : 0) : -1;      // -1 = auto
        const char *m = getenv("TQ_SPEC_ATTN_LEGACY_MIN");
        legacy_min = (m && *m) ? atoi(m) : 8192;               // measured with GROUP2
    }
    if (legacy == 1) return 0;
    if (legacy == -1 && g_spec_attn_base_host >= legacy_min) return 0;
    if (g_qwen.hd != 256) return 0;
    const tq_qmma_weight_t *ws[4] = {&l->q_proj, &l->k_proj, &l->v_proj, &l->o_proj};
    for (int i = 0; i < 4; i++)
        if (!can_use_qmma_sf_weight(ws[i]) || ws[i]->row_major || ws[i]->word_major) return 0;
    return spec_persist_mlp_ok(l);
}

// Can this linear layer run through the persistent kernel?
static int spec_persist_layer_ok(tq_layer_t *l) {
    int dim = g_qwen.linear_value_head_dim;
    if (dim != 128 || linear_core_par_g() != 8 || linear_core_hs() != 4) return 0;
    if (getenv("TQ_SPEC_NOBATCH_RECUR")) return 0;
    const tq_qmma_weight_t *ws[5] = {&l->linear_in_qkv, &l->linear_in_z, &l->linear_in_b,
                                     &l->linear_in_a, &l->linear_out};
    for (int i = 0; i < 5; i++)
        if (!can_use_qmma_sf_weight(ws[i]) || ws[i]->row_major || ws[i]->word_major) return 0;
    return spec_persist_mlp_ok(l);
}

// Launch one half of a linear layer through the persistent kernel.
// part 0: pack(norm) -> qkv/z/b/a GEMVs -> DeltaNet -> gated norm -> pack(core) -> out GEMV -> d_spec_lo
// part 1: pack(post) -> gate/up GEMVs -> silu -> pack(mlp) -> down GEMV -> h = resid + down
static int spec_persist_fill(tq_layer_t *l, int N, int lin_idx, int part, int n_depths,
                             tq_spec_persist_args_t *pA) {
    int ret = ensure_spec_persist();
    if (ret != 0) return ret;
    int H = g_qwen.H, I = g_qwen.I;
    int dim = g_qwen.linear_value_head_dim;
    int heads = g_qwen.linear_num_value_heads, key_heads = g_qwen.linear_num_key_heads;
    int key_dim = key_heads * dim, value_dim = heads * dim;
    int conv_dim = key_dim * 2 + value_dim;
    int recur_f = spec_lin_recur_floats(), conv_f = spec_lin_conv_floats();
    int n_alloc = TQ_SPEC_MAX_N;

    tq_spec_persist_args_t A;
    memset(&A, 0, sizeof(A));
    A.N = N;
    A.ncta = (part == 0) ? g_spec_persist_ncta
           : (part == 2) ? g_spec_persist_ncta2 : g_spec_persist_ncta1;
    A.scratch = g_spec_persist_scratch;
    A.bar_groups = g_spec_persist_scratch + TQ_PERSIST_SCRATCH_INTS;
    A.absmax = g_qwen.d_spec_nk_factors;
    A.factors = g_qwen.d_spec_nk_factors + TQ_SPEC_MAX_N;
    A.bfrag = g_qwen.d_spec_b;
    A.bscale = g_qwen.d_spec_bscale;
    A.part = part;
    A.eps = g_qwen.eps;
    A.pf_slot = TQ_PERSIST_PF;
    A.pf1_slot = TQ_PERSIST_PF1;
    size_t poff = 0;

    A.pack1_amax = g_spec_persist_amax;
    A.pack1_amax_stride = 8;
    if (part == 0) {
        A.pack1_in = g_qwen.d_spec_norm; A.pack1_K = H;
        A.pack2_in = g_qwen.d_spec_core; A.pack2_K = value_dim;
        if (spec_persist_pw(&l->linear_in_qkv, &A.w1[0], g_qwen.d_spec_qkv, g_qwen.d_spec_part, &poff, N)) return -70;
        if (spec_persist_pw(&l->linear_in_z, &A.w1[1], g_qwen.d_spec_z, g_qwen.d_spec_part, &poff, N)) return -70;
        if (spec_persist_pw(&l->linear_in_b, &A.w1[2], g_qwen.d_spec_bp, g_qwen.d_spec_part, &poff, N)) return -70;
        if (spec_persist_pw(&l->linear_in_a, &A.w1[3], g_qwen.d_spec_ap, g_qwen.d_spec_part, &poff, N)) return -70;
        A.nw1 = 4;
        if (spec_persist_pw(&l->linear_out, &A.w2, g_qwen.d_spec_lo, g_qwen.d_spec_part, &poff, N)) return -70;
        A.recur_base = (float *)((char *)g_qwen.d_spec_recur +
            ((size_t)lin_idx * n_alloc + g_spec_slot_base) * recur_f * (spec_recur_bf16() ? 2 : 4));
        A.conv_state_base = g_qwen.d_spec_conv + ((size_t)lin_idx * n_alloc + g_spec_slot_base) * conv_f;
        A.qkv_in = g_qwen.d_spec_qkv;
        A.bp = g_qwen.d_spec_bp;
        A.ap = g_qwen.d_spec_ap;
        A.A_log = l->d_linear_A_log;
        A.dt_bias = l->d_linear_dt_bias;
        A.conv_w = l->d_linear_conv1d;
        A.committed_recur = l->d_linear_recurrent_state;
        A.committed_conv = l->d_linear_conv_state;
        A.z = g_qwen.d_spec_z;
        A.gated_norm_w = l->d_linear_norm;
        A.heads = heads; A.key_heads = key_heads; A.dim = dim;
        A.kernel_size = g_qwen.linear_conv_kernel_dim;
        A.recur_f = recur_f; A.conv_f = conv_f;
        A.conv_dim = conv_dim; A.value_dim = value_dim;
        A.group_ids = g_qwen.d_spec_group_ids;
        A.parent_ids = g_qwen.d_spec_parent;
        A.depth_offsets = g_qwen.d_spec_depth_offsets;
        A.n_depths = n_depths;
        A.segs = g_qwen.d_spec_segs;
        A.n_segs = g_spec_persist_nsegs;
        A.dnprep = g_qwen.d_spec_dnprep +
                   (size_t)lin_idx * n_alloc * (key_heads * 2 * dim + value_dim + heads * 2);
        A.state_skip = g_spec_state_skip_round;
        A.recur_bf16 = spec_recur_bf16();
        g_spec_persist_out_partials = A.w2.partials;
        g_spec_persist_out_ks = A.w2.ks;
        // late prefetch program: this layer's downstream weights (out, then mlp)
        A.pf_ptr[0] = (const uint32_t *)l->linear_out.d_A;
        A.pf_words[0] = (size_t)l->linear_out.Mt * l->linear_out.Kt * tq_w_tile_words(&l->linear_out);
        A.pf_ptr[1] = (const uint32_t *)l->mlp_gate.d_A;
        A.pf_words[1] = (size_t)l->mlp_gate.Mt * l->mlp_gate.Kt * tq_w_tile_words(&l->mlp_gate);
        A.pf_ptr[2] = (const uint32_t *)l->mlp_up.d_A;
        A.pf_words[2] = (size_t)l->mlp_up.Mt * l->mlp_up.Kt * tq_w_tile_words(&l->mlp_up);
        A.n_pf = 3;
        // early prefetch program: the group-1 weights about to be streamed
        A.pf1_ptr[0] = (const uint32_t *)l->linear_in_qkv.d_A;
        A.pf1_words[0] = (size_t)l->linear_in_qkv.Mt * l->linear_in_qkv.Kt * tq_w_tile_words(&l->linear_in_qkv);
        A.pf1_ptr[1] = (const uint32_t *)l->linear_in_z.d_A;
        A.pf1_words[1] = (size_t)l->linear_in_z.Mt * l->linear_in_z.Kt * tq_w_tile_words(&l->linear_in_z);
        A.n_pf1 = 2;
        // Prefetch DEFAULT OFF: measured -1.4 ms/verify. The batched 8-node DeltaNet
        // moves ~50 MB of state per layer, so there is no idle DRAM window to fill --
        // prefetch only steals bandwidth from the recurrence (TQ_SPEC_PERSIST_PF=1
        // re-enables for experiments).
        if (!spec_persist_pf_enabled()) { A.n_pf = 0; A.n_pf1 = 0; }
    } else if (part == 2) {
        tq_layer_t *fl = l;
        A.pack1_in = g_qwen.d_spec_norm; A.pack1_K = H;
        A.pack2_in = g_qwen.d_spec_core; A.pack2_K = g_qwen.nh * g_qwen.hd;
        if (spec_persist_pw(&fl->q_proj, &A.w1[0], g_qwen.d_spec_qkv, g_qwen.d_spec_part, &poff, N)) return -70;
        if (spec_persist_pw(&fl->k_proj, &A.w1[1], g_qwen.d_spec_kbuf, g_qwen.d_spec_part, &poff, N)) return -70;
        if (spec_persist_pw(&fl->v_proj, &A.w1[2], g_qwen.d_spec_z, g_qwen.d_spec_part, &poff, N)) return -70;
        A.nw1 = 3;
        if (spec_persist_pw(&fl->o_proj, &A.w2, g_qwen.d_spec_lo, g_qwen.d_spec_part, &poff, N)) return -70;
        g_spec_persist_out_partials = A.w2.partials;
        g_spec_persist_out_ks = A.w2.ks;
        int kv_m = g_qwen.nkv * g_qwen.hd;
        A.attn_qnorm = fl->d_q_norm;
        A.attn_knorm = fl->d_k_norm;
        A.attn_kcache = fl->d_k_cache;
        A.attn_vcache = fl->d_v_cache;
        A.attn_kcache8 = fl->d_k_cache8;
        A.attn_vcache8 = fl->d_v_cache8;
        A.attn_kscale = fl->d_k_scale;
        A.attn_vscale = fl->d_v_scale;
        A.attn_kcache4 = fl->d_k_cache4;
        A.attn_kq4s = fl->d_k_q4s;
        A.attn_karch = g_qwen.d_spec_k_arch + (size_t)lin_idx * TQ_SPEC_MAX_N * kv_m;
        A.attn_varch = g_qwen.d_spec_v_arch + (size_t)lin_idx * TQ_SPEC_MAX_N * kv_m;
        A.attn_pos = g_qwen.d_spec_positions;
        A.attn_anc = g_qwen.d_spec_anc;
        A.attn_mma = g_spec_attn_mma_round;
        A.attn_pair = attn_mma_pair_ok(N);
        A.attn_nchunks = attn_mma_nchunks(g_spec_attn_base_host, A.attn_pair);
        if (ensure_attn_scores() != 0) return -97;
        A.attn_scores = g_qwen.d_attn_scores;
        A.max_seq = g_qwen.max_seq;
        A.nh = g_qwen.nh; A.nkv = g_qwen.nkv; A.hd = g_qwen.hd;
        A.kv_m = kv_m; A.q_m = g_qwen.nh * g_qwen.hd * 2; A.attn_m = g_qwen.nh * g_qwen.hd;
        A.attn_base = n_depths;   // caller passes base via n_depths slot (-1 = device-derived)
        A.rope_theta = g_qwen.rope_theta;
        A.heads = heads; A.dim = dim; A.value_dim = value_dim;
    } else {
        A.pack1_in = g_qwen.d_spec_post; A.pack1_K = H;
        A.pack2_in = g_qwen.d_spec_mlp; A.pack2_K = I;
        if (spec_persist_pw(&l->mlp_gate, &A.w1[0], g_qwen.d_spec_gate, g_qwen.d_spec_part, &poff, N)) return -70;
        if (spec_persist_pw(&l->mlp_up, &A.w1[1], g_qwen.d_spec_up, g_qwen.d_spec_part, &poff, N)) return -70;
        A.nw1 = 2;
        if (spec_persist_pw(&l->mlp_down, &A.w2, g_qwen.d_spec_lo, g_qwen.d_spec_part, &poff, N)) return -70;
        A.gate_in = g_qwen.d_spec_gate;
        A.up_in = g_qwen.d_spec_up;
        A.final_out = g_qwen.d_spec_h;
        A.final_resid = g_qwen.d_spec_resid;
        A.heads = heads; A.dim = dim; A.value_dim = value_dim;
        A.pf_ptr[0] = (const uint32_t *)l->mlp_down.d_A;
        A.pf_words[0] = (size_t)l->mlp_down.Mt * l->mlp_down.Kt * tq_w_tile_words(&l->mlp_down);
        A.n_pf = 1;
        A.pf1_ptr[0] = (const uint32_t *)l->mlp_gate.d_A;
        A.pf1_words[0] = (size_t)l->mlp_gate.Mt * l->mlp_gate.Kt * tq_w_tile_words(&l->mlp_gate);
        A.pf1_ptr[1] = (const uint32_t *)l->mlp_up.d_A;
        A.pf1_words[1] = (size_t)l->mlp_up.Mt * l->mlp_up.Kt * tq_w_tile_words(&l->mlp_up);
        A.n_pf1 = 2;
        if (!spec_persist_pf_enabled()) { A.n_pf = 0; A.n_pf1 = 0; }
    }
    {
        static int pf_bars = -1;
        if (pf_bars < 0) {
            const char *e = getenv("TQ_SPEC_PERSIST_PF_BARS");
            pf_bars = e ? atoi(e) : 1;
        }
        A.pf_bars = pf_bars;
    }
    {
        static int prof = -1;
        if (prof < 0) {
            const char *e = getenv("TQ_SPEC_PERSIST_PROF");
            prof = e ? atoi(e) : 0;
        }
        // prof=1: clocks hold the LAST launch (a part-1); prof=2: part-0 only;
        // prof=3: part-2 only.
        A.clk = (prof == 1 || (prof == 2 && part == 0) || (prof == 3 && part == 2))
                    ? g_spec_persist_clk : NULL;
    }
    {
        static int gw = -1;
        if (gw < 0) {
            const char *e = getenv("TQ_SPEC_GEMV_WARPS");
            gw = e ? atoi(e) : 0;
            if (gw < 0 || gw > 8) gw = 0;
        }
        A.gemv_warps = gw;
    }
    size_t pcap = (size_t)16 * TQ_SPEC_MAX_N * (size_t)(I > conv_dim ? I : conv_dim);
    if (poff > pcap) return -71;   // partials scratch too small for this shape
    *pA = A;
    return 0;
}

static int run_spec_persist_part(tq_layer_t *l, int N, int lin_idx, int part, int n_depths) {
    tq_spec_persist_args_t A;
    int ret = spec_persist_fill(l, N, lin_idx, part, n_depths, &A);
    if (ret != 0) return ret;
    // Cooperative launch: guarantees all ncta CTAs are co-resident (the software
    // barriers deadlock if even one CTA is not scheduled, e.g. when a zombie
    // kernel squats on SMs) -- turns that failure mode into a launch error.
    void *kargs[] = {&A};
    void *kfn = (part == 0) ? (void *)k_tq_spec_persistent_layer<8, 4, 0>
              : (part == 2) ? (void *)k_tq_spec_persistent_layer<8, 4, 2>
              : (g_use_ldsm) ? (void *)k_tq_spec_persistent_layer<8, 4, 1, true>
                            : (void *)k_tq_spec_persistent_layer<8, 4, 1>;
    size_t dynsm = (part == 2 && g_qwen.attn_mma) ? TQ_ATTN_MMA_SMEM : 0;
    cudaError_t e = cudaLaunchCooperativeKernel(kfn, dim3(A.ncta), dim3(256), kargs, dynsm, g_qwen.stream);
    if (e != cudaSuccess) {
        fprintf(stderr, "spec persistent launch failed: %s\n", cudaGetErrorString(e));
        return -72;
    }
    return 0;
}

// Fused full-layer launch (TQ_SPEC_FUSE_LAYER): [in-norm -> partA -> residual
// norm -> MLP] in one persistent grid. Halves get disjoint barrier/queue pools;
// the in-kernel norm phases replicate the standalone kernels bit-exactly.
static int run_spec_persist_fused(tq_layer_t *l, int N, int lin_idx, int parta, int n_depths) {
    tq_spec_persist_args_t A0, A1;
    int ret = spec_persist_fill(l, N, lin_idx, parta, n_depths, &A0);
    if (ret != 0) return ret;
    ret = spec_persist_fill(l, N, lin_idx, 1, n_depths, &A1);
    if (ret != 0) return ret;
    if (A0.ncta != A1.ncta) return -73;   // halves must share one grid shape
    // NOTE: wider norm stripes (64) measured WORSE (-1.7 ms): the per-item sumsq
    // is recomputed redundantly, so traffic scales with stripe count. Stay at 8.
    A0.ln_in = l->d_input_ln;
    A0.norm_h_in = g_qwen.d_spec_h;
    A0.norm_out = g_qwen.d_spec_norm;     // = A0.pack1_in
    A0.norm_amax = g_spec_persist_amax;   // = pack1_amax buffer
    A1.bar_base = 8;
    A1.q_base = 12;
    A1.pf_slot = TQ_PERSIST_PF_B;
    A1.pf1_slot = TQ_PERSIST_PF1_B;
    A1.ln_post = l->d_post_ln;
    A1.norm_h_in = g_qwen.d_spec_h;
    A1.norm_out = g_qwen.d_spec_post;     // = A1.pack1_in
    A1.norm_amax = g_spec_persist_amax;
    A1.resid_out = g_qwen.d_spec_resid;
    A1.out_partials = A0.w2.partials;
    A1.out_ks = A0.w2.ks;
    A1.out_factors = A0.factors;          // pack2 prescale factors of half A
    void *kargs[] = {&A0, &A1};
    void *kfn = (parta == 0) ? (void *)k_tq_spec_persistent_layer_fused<8, 4, 0>
                             : (void *)k_tq_spec_persistent_layer_fused<8, 4, 2>;
    size_t dynsm = (parta == 2 && g_qwen.attn_mma) ? TQ_ATTN_MMA_SMEM : 0;
    cudaError_t e = cudaLaunchCooperativeKernel(kfn, dim3(A0.ncta), dim3(256), kargs, dynsm, g_qwen.stream);
    if (e != cudaSuccess) {
        fprintf(stderr, "spec persistent fused launch failed: %s\n", cudaGetErrorString(e));
        return -72;
    }
    return 0;
}

// Grid-stride L2 prefetch of a weight byte range (side-effect free).
__global__ void k_tq_l2_prefetch(const uint32_t *p, size_t words) {
    size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 32;
    size_t stride = (size_t)gridDim.x * blockDim.x * 32;
    for (; i < words; i += stride)
        asm volatile("prefetch.global.L2 [%0];" :: "l"(p + i));
}

// Sequential-path DeltaNet/GEMV overlap (TQ_SPEC_PF_STREAM=1): while the main
// stream runs the (compute-bound) DeltaNet recurrence, a second stream streams the
// layer's downstream weights (out_proj + budgeted gate/up slice) into L2 so the
// following GEMVs read from L2 instead of DRAM. Numerics untouched.
static cudaStream_t g_spec_pf_stream = NULL;
static cudaEvent_t g_spec_pf_event = NULL;

static int spec_pf_stream_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("TQ_SPEC_PF_STREAM");
        cached = env ? atoi(env) : 0;
    }
    return cached;
}

static size_t spec_pf_budget_words(void) {
    static size_t cached = 0;
    if (!cached) {
        const char *env = getenv("TQ_SPEC_PF_MB");
        int mb = env ? atoi(env) : 48;
        if (mb < 1) mb = 1;
        cached = (size_t)mb * 1024 * 1024 / 4;
    }
    return cached;
}

static void spec_pf_stream_launch(const tq_qmma_weight_t **ws, int nw) {
    if (!g_spec_pf_stream) {
        if (cudaStreamCreateWithFlags(&g_spec_pf_stream, cudaStreamNonBlocking) != cudaSuccess) return;
        if (cudaEventCreateWithFlags(&g_spec_pf_event, cudaEventDisableTiming) != cudaSuccess) return;
    }
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(g_qwen.stream, &cap);
    if (cap != cudaStreamCaptureStatusNone) return;   // keep graph capture single-stream
    cudaEventRecord(g_spec_pf_event, g_qwen.stream);
    cudaStreamWaitEvent(g_spec_pf_stream, g_spec_pf_event, 0);
    size_t budget = spec_pf_budget_words();
    for (int i = 0; i < nw && budget > 0; i++) {
        const tq_qmma_weight_t *w = ws[i];
        size_t words = (size_t)w->Mt * w->Kt * tq_w_tile_words(w);
        if (words > budget) words = budget;
        k_tq_l2_prefetch<<<64, 256, 0, g_spec_pf_stream>>>((const uint32_t *)w->d_A, words);
        budget -= words;
    }
}


// Layout microbenchmark: is the ~1 TB/s E2M3 A-stream ceiling an address-pattern
// artifact? Emulates the persistent GEMV phase (340 CTAs x 8 warp-items pulling
// (m-tile, k-slice) work) over a synthetic weight in two layouts:
//   0: production tile-sequential  A[(mt * Kt + tk) * 96]
//   1: pair-interleaved            A[((mt/2) * Kt + tk) * 192 + (mt&1) * 96]
// Layout 1 halves the stream count: a warp-item owns an m-tile PAIR and reads
// 768 B contiguously per K-tile step.
__global__ void k_tq_layout_bench(float *out, const uint32_t *a, const uint32_t *b,
                                  int Mt, int Kt, int ks, int layout, int *queue) {
    int lane = threadIdx.x & 31;
    int kt_per = Kt / ks;
    int items = (layout ? Mt / 2 : Mt) * ks;
    while (1) {
        int it;
        if (lane == 0) it = atomicAdd(queue, 1);
        it = __shfl_sync(0xffffffffu, it, 0);
        if (it >= items) break;
        int tile_m = it / ks, split = it % ks;
        int tk0 = split * kt_per, tk1 = tk0 + kt_per;
        float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
        float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
        if (layout == 0) {
            const uint32_t *a_base = a + (size_t)tile_m * Kt * 96;
            for (int tk = tk0; tk + 3 < tk1; tk += 4) {
                int a_idx = tk * 96 + lane * 3;
                uint32_t PP[4][3];
                uint32_t BB[4][2];
                #pragma unroll
                for (int u = 0; u < 4; u++) {
                    PP[u][0] = __ldg(&a_base[a_idx + 96 * u + 0]);
                    PP[u][1] = __ldg(&a_base[a_idx + 96 * u + 1]);
                    PP[u][2] = __ldg(&a_base[a_idx + 96 * u + 2]);
                    BB[u][0] = __ldg(&b[(tk + u) * 64 + lane * 2 + 0]);
                    BB[u][1] = __ldg(&b[(tk + u) * 64 + lane * 2 + 1]);
                }
                #pragma unroll
                for (int u = 0; u < 4; u++) {
                    uint32_t a0, a1, a2, a3;
                    tq_unpack_e2m3_16(PP[u][0], PP[u][1], PP[u][2], a0, a1, a2, a3);
                    uint32_t sfa = 0x7f7f7f7fu, sfb = 0x7f7f7f7fu;
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X"
                        ".f32.e2m3.e4m3.f32.ue8m0 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, {0,1}, %15, {0,1};"
                        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(BB[u][0]), "r"(BB[u][1]),
                          "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(sfa), "r"(sfb));
                }
            }
        } else {
            const uint32_t *a_base = a + (size_t)tile_m * Kt * 192;
            for (int tk = tk0; tk + 1 < tk1; tk += 2) {
                int a_idx = tk * 192 + lane * 3;
                uint32_t PP[4][3];
                uint32_t BB[2][2];
                #pragma unroll
                for (int u = 0; u < 2; u++) {
                    #pragma unroll
                    for (int h = 0; h < 2; h++) {
                        PP[u * 2 + h][0] = __ldg(&a_base[a_idx + 192 * u + 96 * h + 0]);
                        PP[u * 2 + h][1] = __ldg(&a_base[a_idx + 192 * u + 96 * h + 1]);
                        PP[u * 2 + h][2] = __ldg(&a_base[a_idx + 192 * u + 96 * h + 2]);
                    }
                    BB[u][0] = __ldg(&b[(tk + u) * 64 + lane * 2 + 0]);
                    BB[u][1] = __ldg(&b[(tk + u) * 64 + lane * 2 + 1]);
                }
                #pragma unroll
                for (int u = 0; u < 2; u++) {
                    uint32_t a0, a1, a2, a3;
                    tq_unpack_e2m3_16(PP[u * 2][0], PP[u * 2][1], PP[u * 2][2], a0, a1, a2, a3);
                    uint32_t sfa = 0x7f7f7f7fu, sfb = 0x7f7f7f7fu;
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X"
                        ".f32.e2m3.e4m3.f32.ue8m0 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, {0,1}, %15, {0,1};"
                        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(BB[u][0]), "r"(BB[u][1]),
                          "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(sfa), "r"(sfb));
                    uint32_t e0, e1, e2, e3;
                    tq_unpack_e2m3_16(PP[u * 2 + 1][0], PP[u * 2 + 1][1], PP[u * 2 + 1][2], e0, e1, e2, e3);
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X"
                        ".f32.e2m3.e4m3.f32.ue8m0 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, {0,1}, %15, {0,1};"
                        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                        : "r"(e0), "r"(e1), "r"(e2), "r"(e3), "r"(BB[u][0]), "r"(BB[u][1]),
                          "f"(d0), "f"(d1), "f"(d2), "f"(d3), "r"(sfa), "r"(sfb));
                }
            }
        }
        if (lane == 0) out[it & 1023] = c0 + c1 + c2 + c3 + d0 + d1 + d2 + d3;
    }
}

// Mode 2: the REAL persistent GEMV item (g2 branch, scale loads, partials
// epilogue) in a clean dedicated kernel -- isolates item-code cost from the
// persistent-kernel context (register union, launch bounds).
__global__ void k_tq_layout_bench_real(const tq_spec_persist_args_t A, int *queue) {
    int kt_per = A.w1[0].Kt / A.w1[0].ks;
    (void)kt_per;
    int items = ((A.w1[0].M + 15) / 16) * A.w1[0].ks;
    while (1) {
        int it;
        if ((threadIdx.x & 31) == 0) it = atomicAdd(queue, 1);
        it = __shfl_sync(0xffffffffu, it, 0);
        if (it >= items) break;
        tq_persist_gemv_item(&A, &A.w1[0], it / A.w1[0].ks, it % A.w1[0].ks, 0);
    }
}

// Dedicated queued GEMV kernel: the persistent work-queue scheme (multi-weight
// m-tile x k-slice items, one warp each) in a clean kernel. The same item code
// runs at ~80% of DRAM peak here vs ~57% inlined in the persistent mega-kernel
// (codegen pays for the union of phases there), so the GEMV phases are launched
// separately and the persistent kernel keeps only the fusion-friendly phases.
__global__ void __launch_bounds__(256, 2) k_tq_spec_gemv_queue(tq_spec_persist_args_t A) {
    int off[5];
    off[0] = 0;
    for (int w = 0; w < A.nw1; w++) {
        int mt = (A.w1[w].M + 15) / 16;
        off[w + 1] = off[w] + mt * A.w1[w].ks;
    }
    int total = off[A.nw1];
    while (1) {
        int it;
        if ((threadIdx.x & 31) == 0) it = atomicAdd(&A.scratch[28], 1);
        it = __shfl_sync(0xffffffffu, it, 0);
        if (it >= total) break;
        int w = 0;
        while (it >= off[w + 1]) w++;
        int local = it - off[w];
        int ks = A.w1[w].ks;
        tq_persist_gemv_item(&A, &A.w1[w], local / ks, local % ks, 0);
    }
    // last CTA to exit resets the queue for the next launch (spare slots 28/29)
    __syncthreads();
    if (threadIdx.x == 0) {
        int prev = atomicAdd(&A.scratch[29], 1);
        if (prev == (int)gridDim.x - 1) {
            A.scratch[28] = 0;
            A.scratch[29] = 0;
            __threadfence();
        }
    }
}

// Bench both layouts over a gate+up-sized synthetic weight. Returns 0; fills
// ms_out[2] with per-iteration kernel times.
extern "C" int qwn_layout_bench(int Mt, int Kt, int ks, int iters, float *ms_out) {
    size_t words = (size_t)Mt * Kt * 96;
    uint32_t *d_a = NULL, *d_b = NULL;
    float *d_out = NULL;
    int *d_q = NULL;
    if (cudaMalloc(&d_a, words * 4) != cudaSuccess) return -1;
    if (cudaMalloc(&d_b, (size_t)Kt * 64 * 4) != cudaSuccess) return -1;
    if (cudaMalloc(&d_out, 1024 * 4) != cudaSuccess) return -1;
    if (cudaMalloc(&d_q, 4) != cudaSuccess) return -1;
    cudaMemset(d_a, 0x35, words * 4);
    cudaMemset(d_b, 0x35, (size_t)Kt * 64 * 4);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    if (iters < 0) {   // mode 2: real item, |iters| iterations
        iters = -iters;
        int one = 1;   // standalone bench: the item branches on gc_weight_e2m3
        cudaMemcpyToSymbol(gc_weight_e2m3, &one, sizeof(int));
        tq_spec_persist_args_t A;
        memset(&A, 0, sizeof(A));
        float *d_scale = NULL, *d_part = NULL, *d_fac = NULL, *d_bs = NULL;
        cudaMalloc(&d_scale, (size_t)((Mt * 16 + 127) / 128) * ((Kt * 32 + 127) / 128) * 4);
        cudaMalloc(&d_part, (size_t)ks * 8 * Mt * 16 * 4);
        cudaMalloc(&d_fac, 64);
        cudaMalloc(&d_bs, (size_t)((Kt * 32 + 127) / 128) * 4);
        cudaMemset(d_scale, 0x3f, (size_t)((Mt * 16 + 127) / 128) * ((Kt * 32 + 127) / 128) * 4);
        cudaMemset(d_bs, 0x3f, (size_t)((Kt * 32 + 127) / 128) * 4);
        A.N = 8;
        A.bfrag = (uint8_t *)d_b;
        A.bscale = d_bs;
        A.factors = d_fac;
        A.absmax = d_fac;
        A.w1[0].a = d_a;
        A.w1[0].scale_inv = d_scale;
        A.w1[0].scale_cols = (Kt * 32 + 127) / 128;
        A.w1[0].M = Mt * 16;
        A.w1[0].Kt = Kt;
        A.w1[0].ks = ks;
        A.w1[0].partials = d_part;
        A.w1[0].out = d_part;
        cudaMemset(d_q, 0, 4);
        k_tq_layout_bench_real<<<340, 256>>>(A, d_q);
        cudaDeviceSynchronize();
        cudaEventRecord(e0);
        for (int i = 0; i < iters; i++) {
            cudaMemset(d_q, 0, 4);
            k_tq_layout_bench_real<<<340, 256>>>(A, d_q);
        }
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, e0, e1);
        ms_out[0] = ms / iters;
        ms_out[1] = 0.0f;
        cudaFree(d_scale); cudaFree(d_part); cudaFree(d_fac); cudaFree(d_bs);
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_out); cudaFree(d_q);
        return cudaGetLastError() == cudaSuccess ? 0 : -2;
    }
    for (int layout = 0; layout < 2; layout++) {
        cudaMemset(d_q, 0, 4);
        k_tq_layout_bench<<<340, 256>>>(d_out, d_a, d_b, Mt, Kt, ks, layout, d_q);  // warmup
        cudaDeviceSynchronize();
        cudaEventRecord(e0);
        for (int i = 0; i < iters; i++) {
            cudaMemset(d_q, 0, 4);
            k_tq_layout_bench<<<340, 256>>>(d_out, d_a, d_b, Mt, Kt, ks, layout, d_q);
        }
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, e0, e1);
        ms_out[layout] = ms / iters;
    }
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out); cudaFree(d_q);
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

// Debug: copy the last persistent launch's phase clocks (prof mode).
extern "C" int qwn_spec_persist_clocks(long long *out) {
    if (!g_spec_persist_clk) return -1;
    cudaError_t e = cudaMemcpy(out, g_spec_persist_clk, TQ_PERSIST_CLK_SLOTS * sizeof(long long),
                               cudaMemcpyDeviceToHost);
    return e == cudaSuccess ? 0 : -2;
}

// Debug: dump the persistent-kernel scratch counters (use from a watchdog while the
// stream is hung to see which barrier/queue the grid is stuck on).
extern "C" int qwn_spec_persist_dump(int *out32) {
    if (!g_spec_persist_scratch) return -1;
    cudaStream_t s;
    if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess) return -2;
    cudaError_t e = cudaMemcpyAsync(out32, g_spec_persist_scratch,
                                    TQ_PERSIST_SCRATCH_INTS * sizeof(int),
                                    cudaMemcpyDeviceToHost, s);
    if (e == cudaSuccess) e = cudaStreamSynchronize(s);
    cudaStreamDestroy(s);
    return e == cudaSuccess ? 0 : -3;
}

static int spec_split_gemv_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("TQ_SPEC_SPLIT_GEMV");
        cached = env ? atoi(env) : 0;
    }
    return cached;
}

// MLP half with the GEMV phases as dedicated queued launches (the persistent
// mega-kernel's codegen costs the GEMV ~23 points of DRAM efficiency); the
// small phases stay as the proven standalone kernels.
static int run_spec_mlp_split(tq_layer_t *l, int N) {
    int ret = ensure_spec_persist();
    if (ret != 0) return ret;
    int H = g_qwen.H, I = g_qwen.I;
    tq_spec_persist_args_t A;
    memset(&A, 0, sizeof(A));
    A.N = N;
    A.ncta = g_spec_persist_ncta1;
    A.scratch = g_spec_persist_scratch;
    A.bar_groups = g_spec_persist_scratch + TQ_PERSIST_SCRATCH_INTS;
    A.absmax = g_qwen.d_spec_nk_factors;
    A.factors = g_qwen.d_spec_nk_factors + TQ_SPEC_MAX_N;
    A.bfrag = g_qwen.d_spec_b;
    A.bscale = g_qwen.d_spec_bscale;
    size_t poff = 0;
    // pack(post) with the stripe absmax from the host norm kernel
    k_tq_quantize_bfrag_pow2_fused_nk_f8<<<(H + 127) / 128, 128, 0, g_qwen.stream>>>(
        g_qwen.d_spec_b, g_qwen.d_spec_bscale, g_qwen.d_spec_post, H, N,
        g_spec_persist_amax, g_qwen.d_spec_nk_factors + TQ_SPEC_MAX_N);
    if (spec_persist_pw(&l->mlp_gate, &A.w1[0], g_qwen.d_spec_gate, g_qwen.d_spec_part, &poff, N)) return -70;
    if (spec_persist_pw(&l->mlp_up, &A.w1[1], g_qwen.d_spec_up, g_qwen.d_spec_part, &poff, N)) return -70;
    A.nw1 = 2;
    k_tq_spec_gemv_queue<<<g_spec_persist_ncta1, 256, 0, g_qwen.stream>>>(A);
    // silu reads the partials inline (reduce folded), writes mlp + absmax fold:
    // reuse the persistent item via a thin standalone kernel? simpler: materialize
    // with the pc-reduce kernels, then the proven silu+quant kernels.
    for (int w = 0; w < 2; w++) {
        int total = N * A.w1[w].M;
        k_tq_reduce_qmma_sf_splits_pc<<<(total + 255) / 256, 256, 0, g_qwen.stream>>>(
            A.w1[w].out, A.w1[w].partials, A.factors, A.w1[w].M, N, A.w1[w].ks);
    }
    k_tq_silu_mul_b<<<dim3((I + 255) / 256, N), 256, 0, g_qwen.stream>>>(
        g_qwen.d_spec_mlp, g_qwen.d_spec_gate, g_qwen.d_spec_up, I);
    k_tq_nk_col_absmax<<<N, 256, 0, g_qwen.stream>>>(
        g_qwen.d_spec_mlp, g_qwen.d_spec_nk_factors, I, N);
    k_tq_quantize_bfrag_pow2_fused_nk_f<<<(I + 127) / 128, 128, 0, g_qwen.stream>>>(
        g_qwen.d_spec_b, g_qwen.d_spec_bscale, g_qwen.d_spec_mlp, I, N,
        g_qwen.d_spec_nk_factors, g_qwen.d_spec_nk_factors + TQ_SPEC_MAX_N);
    memset(&A.w1[0], 0, sizeof(A.w1[0]) * 4);
    poff = 0;
    if (spec_persist_pw(&l->mlp_down, &A.w1[0], g_qwen.d_spec_lo, g_qwen.d_spec_part, &poff, N)) return -70;
    A.nw1 = 1;
    k_tq_spec_gemv_queue<<<g_spec_persist_ncta1, 256, 0, g_qwen.stream>>>(A);
    // final residual with the down reduce folded (same as the persist part-1 tail)
    k_tq_reduce_final_split<<<dim3((H * N + 255) / 256), 256, 0, g_qwen.stream>>>(
        g_qwen.d_spec_h, g_qwen.d_spec_resid, A.w1[0].partials,
        g_qwen.d_spec_nk_factors + TQ_SPEC_MAX_N, H, N, A.w1[0].ks);
    return 0;
}

// Batched tree attention (default on): one kv-prep + one attend launch per
// full-attn layer instead of N sequential per-node kernels + 2N archive copies.
static int spec_tree_attn_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("TQ_SPEC_TREE_ATTN");
        cached = env ? atoi(env) : 1;
    }
    return cached;
}

// Ancestor-at-depth table for the tree attention: anc[i][d] = the node occupying
// tree position base+1+d on node i's branch (anc[i][depth[i]] = i). DFS order
// guarantees parent[i] < i.
static void spec_build_anc(const int *parent, const int *depth, int N,
                           int anc[][TQ_SPEC_MAX_N]) {
    for (int i = 0; i < N; i++) {
        for (int d = 0; d < TQ_SPEC_MAX_N; d++) anc[i][d] = 0;
        if (parent[i] >= 0)
            for (int d = 0; d < TQ_SPEC_MAX_N; d++) anc[i][d] = anc[parent[i]][d];
        anc[i][depth[i]] = i;
    }
}

// Debug: compare the unfused (absmax -> prescale -> quantize) and fused
// (absmax -> quantize-with-factors) pack paths on host-provided x [N x K].
// Returns max byte diff of the packed B, max abs diff of block scales/factors.
extern "C" int qwn_spec_pack_check(const float *x_host, int K, int N,
                                   int *b_diff, float *s_diff, float *f_diff) {
    int Kt = (K + 31) / 32;
    int nblocks = (K + 127) / 128;
    float *d_x1 = NULL, *d_x2 = NULL, *d_s1 = NULL, *d_s2 = NULL, *d_f = NULL;
    uint8_t *d_b1 = NULL, *d_b2 = NULL;
    cudaMalloc(&d_x1, (size_t)N * K * sizeof(float));
    cudaMalloc(&d_x2, (size_t)N * K * sizeof(float));
    cudaMalloc(&d_s1, (size_t)nblocks * sizeof(float));
    cudaMalloc(&d_s2, (size_t)nblocks * sizeof(float));
    cudaMalloc(&d_f, (size_t)2 * TQ_SPEC_MAX_N * sizeof(float));
    cudaMalloc(&d_b1, (size_t)Kt * 256);
    cudaMalloc(&d_b2, (size_t)Kt * 256);
    if (!d_x1 || !d_x2 || !d_s1 || !d_s2 || !d_f || !d_b1 || !d_b2) return -1;
    cudaMemcpy(d_x1, x_host, (size_t)N * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x2, x_host, (size_t)N * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_b1, 0xee, (size_t)Kt * 256);
    cudaMemset(d_b2, 0xee, (size_t)Kt * 256);
    // unfused (mutates d_x1)
    k_tq_nk_col_absmax<<<N, 256>>>(d_x1, d_f, K, N);
    k_tq_nk_prescale<<<N, 256>>>(d_x1, d_f, K, N);
    k_tq_quantize_bfrag_pow2_fused_nk<<<nblocks, 128>>>(d_b1, d_s1, d_x1, K, N);
    // fused
    k_tq_nk_col_absmax<<<N, 256>>>(d_x2, d_f + TQ_SPEC_MAX_N, K, N);
    k_tq_quantize_bfrag_pow2_fused_nk_f<<<nblocks, 128>>>(d_b2, d_s2, d_x2, K, N,
        d_f + TQ_SPEC_MAX_N, d_f + TQ_SPEC_MAX_N);
    if (cudaDeviceSynchronize() != cudaSuccess) return -2;
    uint8_t *b1 = (uint8_t *)malloc((size_t)Kt * 256), *b2 = (uint8_t *)malloc((size_t)Kt * 256);
    float *s1 = (float *)malloc(nblocks * sizeof(float)), *s2 = (float *)malloc(nblocks * sizeof(float));
    float f1[TQ_SPEC_MAX_N], f2[TQ_SPEC_MAX_N];
    cudaMemcpy(b1, d_b1, (size_t)Kt * 256, cudaMemcpyDeviceToHost);
    cudaMemcpy(b2, d_b2, (size_t)Kt * 256, cudaMemcpyDeviceToHost);
    cudaMemcpy(s1, d_s1, nblocks * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(s2, d_s2, nblocks * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(f1, d_f, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(f2, d_f + TQ_SPEC_MAX_N, N * sizeof(float), cudaMemcpyDeviceToHost);
    int bd = 0; float sd = 0.0f, fd = 0.0f;
    for (size_t i = 0; i < (size_t)Kt * 256; i++) { int d = abs((int)b1[i] - (int)b2[i]); if (d > bd) bd = d; }
    for (int i = 0; i < nblocks; i++) { float d = fabsf(s1[i] - s2[i]); if (d > sd) sd = d; }
    for (int i = 0; i < N; i++) { float d = fabsf(f1[i] - f2[i]); if (d > fd) fd = d; }
    *b_diff = bd; *s_diff = sd; *f_diff = fd;
    free(b1); free(b2); free(s1); free(s2);
    cudaFree(d_x1); cudaFree(d_x2); cudaFree(d_s1); cudaFree(d_s2); cudaFree(d_f);
    cudaFree(d_b1); cudaFree(d_b2);
    return 0;
}
