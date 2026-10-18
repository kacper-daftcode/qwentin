// the single-stream spec-forward graph, n-gram drafter, spec round driver and wide prefill forward.
// Part of the forward_qwen.cu unity build: included in order, not a standalone header.

// =================== Spec-forward CUDA graph ===================
// Graph-compatible spec_forward: reads token IDs from d_spec_tokens and positions
// from d_spec_positions (device arrays) instead of host-side arguments. The tree
// topology (parent/depth/N) is FIXED across replays.

static int ensure_spec_graph_params(int N) {
    if (!g_qwen.d_spec_tokens) {
        cudaError_t err = cudaMalloc(&g_qwen.d_spec_tokens, (size_t)TQ_SPEC_MAX_N * sizeof(int));
        if (err != cudaSuccess) return -1;
        g_qwen.device_bytes += (size_t)TQ_SPEC_MAX_N * sizeof(int);
    }
    if (!g_qwen.d_spec_positions) {
        cudaError_t err = cudaMalloc(&g_qwen.d_spec_positions, (size_t)TQ_SPEC_MAX_N * sizeof(int));
        if (err != cudaSuccess) return -2;
        g_qwen.device_bytes += (size_t)TQ_SPEC_MAX_N * sizeof(int);
    }
    return 0;
}

static int set_spec_graph_params(const int *tok, const int *posv, int N) {
    int ret = ensure_spec_graph_params(N);
    if (ret != 0) return ret;
    cudaError_t err = cudaMemcpyAsync(g_qwen.d_spec_tokens, tok, (size_t)N * sizeof(int),
                                      cudaMemcpyHostToDevice, g_qwen.stream);
    if (err != cudaSuccess) return -3;
    err = cudaMemcpyAsync(g_qwen.d_spec_positions, posv, (size_t)N * sizeof(int),
                          cudaMemcpyHostToDevice, g_qwen.stream);
    if (err != cudaSuccess) return -4;
    return 0;
}

extern "C" int qwn_capture_spec_forward(const int *tok, const int *parent, const int *depth,
                                        const int *posv, int N) {
    if (!g_qwen.initialized) return -1;
    if (N < 1 || N > TQ_SPEC_MAX_N) return -2;
    destroy_spec_graph();
    int ret = ensure_spec_buffers();
    if (ret != 0) return ret;

    // warmup + capture the EAGER verify itself: one implementation for eager and
    // graph (bit-identical numerics); all its H2D sources are static staging.
    ret = run_spec_forward_n(tok, parent, depth, posv, N, g_qwen.L);
    if (ret != 0) return ret;
    cudaError_t err = cudaStreamSynchronize(g_qwen.stream);
    if (err != cudaSuccess) return -20;

    err = cudaStreamBeginCapture(g_qwen.stream, cudaStreamCaptureModeGlobal);
    if (err != cudaSuccess) {
        fprintf(stderr, "spec cudaStreamBeginCapture failed: %s\n", cudaGetErrorString(err));
        return -21;
    }
    ret = run_spec_forward_n(tok, parent, depth, posv, N, g_qwen.L);
    if (ret != 0) {
        cudaStreamEndCapture(g_qwen.stream, &g_qwen.spec_graph);
        destroy_spec_graph();
        return ret;
    }
    err = cudaStreamEndCapture(g_qwen.stream, &g_qwen.spec_graph);
    if (err != cudaSuccess) {
        fprintf(stderr, "spec cudaStreamEndCapture failed: %s\n", cudaGetErrorString(err));
        destroy_spec_graph();
        return -22;
    }
    err = cudaGraphInstantiate(&g_qwen.spec_graph_exec, g_qwen.spec_graph, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "spec cudaGraphInstantiate failed: %s\n", cudaGetErrorString(err));
        destroy_spec_graph();
        return -23;
    }
    g_qwen.spec_graph_ready = 1;
    g_qwen.spec_graph_N = N;
    return 0;
}

extern "C" int qwn_spec_forward_graph(const int *tok, const int *parent, const int *depth,
                                      const int *posv, int N, int *out_argmax, float *out_hidden) {
    if (!g_qwen.initialized) return -1;
    if (N < 1 || N > TQ_SPEC_MAX_N) return -2;
    if (!g_qwen.spec_graph_ready || !g_qwen.spec_graph_exec) {
        int ret = qwn_capture_spec_forward(tok, parent, depth, posv, N);
        if (ret != 0) return ret;
    }
    if (N != g_qwen.spec_graph_N) return -3;
    int ret = set_spec_graph_params(tok, posv, N);
    if (ret != 0) return ret;
    cudaError_t err = cudaGraphLaunch(g_qwen.spec_graph_exec, g_qwen.stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "spec cudaGraphLaunch failed: %s\n", cudaGetErrorString(err));
        return -4;
    }
    if (out_hidden) {
        if (cudaStreamSynchronize(g_qwen.stream) != cudaSuccess) return -5;
        cudaMemcpy(out_hidden, g_qwen.d_spec_norm, (size_t)N * g_qwen.H * sizeof(float), cudaMemcpyDeviceToHost);
    }
    return spec_argmax(N, out_argmax);
}

// 1-read commit: after a verify run_spec_forward_n over the tree, advance the REAL
// committed state to the accepted path WITHOUT a 2nd forward. Copies the accepted
// leaf's archived per-(layer,node) DeltaNet recur/conv into the live state, and
// re-places the accepted path nodes' archived attn KV into their cache slots (DFS
// verify left each shared slot holding the last branch, not the accepted one).
//   path_nodes[j] = verify DFS node index of the j-th path node (0=root .. m-1=leaf)
//   path_pos[j]   = that node's absolute KV slot (base_pos+1+j)
// The caller then snapshots live -> slot0 (qwn_delta_state_save(0)).
extern "C" int qwn_spec_commit(const int *path_nodes, const int *path_pos, int m) {
    if (!g_qwen.initialized) return -1;
    if (m < 1 || m > TQ_SPEC_MAX_N) return -2;
    if (!g_qwen.d_spec_recur || !g_qwen.d_spec_k_arch) return -3;
    int n_alloc = TQ_SPEC_MAX_N;
    int recur_f = spec_lin_recur_floats(), conv_f = spec_lin_conv_floats();
    int kv_m = g_qwen.nkv * g_qwen.hd;
    int leaf = path_nodes[m - 1];
    if (leaf < 0 || leaf >= n_alloc) return -5;
    if (g_qwen.kv_fp8) {
        // stage the accepted path's [arch node, cache pos] pairs once; the per-layer
        // quant-copy kernel re-places + re-quantizes the rows (capture-safe).
        if (!g_spec_commit_rows) {
            if (cudaMalloc(&g_spec_commit_rows, 2 * TQ_SPEC_MAX_N * sizeof(int)) != cudaSuccess)
                return -7;
            g_qwen.device_bytes += 2 * TQ_SPEC_MAX_N * sizeof(int);
        }
        int h_rows[2 * TQ_SPEC_MAX_N];
        for (int j = 0; j < m; j++) {
            if (path_nodes[j] < 0 || path_nodes[j] >= n_alloc) return -6;
            h_rows[2 * j] = path_nodes[j];
            h_rows[2 * j + 1] = path_pos[j];
        }
        cudaMemcpyAsync(g_spec_commit_rows, h_rows, (size_t)(2 * m) * sizeof(int),
                        cudaMemcpyHostToDevice, g_qwen.stream);
    }
    int lin_idx = 0, full_idx = 0;
    for (int L = 0; L < g_qwen.L; L++) {
        tq_layer_t *l = &g_qwen.layers[L];
        if (g_qwen.layer_types[L] == TQ_LAYER_LINEAR_ATTENTION) {
            if (g_spec_state_skip_round) {
                // recur state comes from the replay kernel (launched once below);
                // conv slabs are fully written by the prep phase -> copy as usual
                cudaMemcpyAsync(l->d_linear_conv_state,
                    g_qwen.d_spec_conv + ((size_t)lin_idx * n_alloc + leaf) * conv_f,
                    (size_t)conv_f * sizeof(float), cudaMemcpyDeviceToDevice, g_qwen.stream);
                lin_idx++;
                continue;
            }
            if (g_spec_commit_swap) {
                // zero-copy commit: the leaf slab BECOMES the live state; the next
                // round writes the other archive half so it cannot clobber it
                l->d_linear_recurrent_state = g_qwen.d_spec_recur +
                    ((size_t)lin_idx * n_alloc + g_spec_slot_base + leaf) * recur_f;
                l->d_linear_conv_state = g_qwen.d_spec_conv +
                    ((size_t)lin_idx * n_alloc + g_spec_slot_base + leaf) * conv_f;
                lin_idx++;
                continue;
            }
            if (spec_recur_bf16()) {
                // bf16 archive: decompress the accepted leaf slab into the live state
                const uint16_t *slab = (const uint16_t *)g_qwen.d_spec_recur +
                                       ((size_t)lin_idx * n_alloc + leaf) * recur_f;
                k_tq_bf16_to_f32<<<(recur_f + 255) / 256, 256, 0, g_qwen.stream>>>(
                    l->d_linear_recurrent_state, slab, recur_f);
            } else {
                cudaMemcpyAsync(l->d_linear_recurrent_state,
                    g_qwen.d_spec_recur + ((size_t)lin_idx * n_alloc + leaf) * recur_f,
                    (size_t)recur_f * sizeof(float), cudaMemcpyDeviceToDevice, g_qwen.stream);
            }
            cudaMemcpyAsync(l->d_linear_conv_state,
                g_qwen.d_spec_conv + ((size_t)lin_idx * n_alloc + leaf) * conv_f,
                (size_t)conv_f * sizeof(float), cudaMemcpyDeviceToDevice, g_qwen.stream);
            lin_idx++;
        } else {
            if (g_qwen.kv_fp8) {
                if (g_qwen.kv_q4)
                    k_tq_spec_commit_kv_q4<<<dim3(g_qwen.nkv, m), g_qwen.hd, 0, g_qwen.stream>>>(
                        l->d_k_cache4, l->d_k_q4s, l->d_v_cache8, l->d_v_scale,
                        g_qwen.d_spec_k_arch + (size_t)full_idx * n_alloc * kv_m,
                        g_qwen.d_spec_v_arch + (size_t)full_idx * n_alloc * kv_m,
                        g_spec_commit_rows, kv_m, g_qwen.nkv, g_qwen.hd);
                else
                    k_tq_spec_commit_kv_fp8<<<dim3(g_qwen.nkv, m), g_qwen.hd, 0, g_qwen.stream>>>(
                        l->d_k_cache8, l->d_k_scale, l->d_v_cache8, l->d_v_scale,
                        g_qwen.d_spec_k_arch + (size_t)full_idx * n_alloc * kv_m,
                        g_qwen.d_spec_v_arch + (size_t)full_idx * n_alloc * kv_m,
                        g_spec_commit_rows, kv_m, g_qwen.nkv, g_qwen.hd);
                full_idx++;
                continue;
            }
            for (int j = 0; j < m; j++) {
                int node = path_nodes[j];
                if (node < 0 || node >= n_alloc) return -6;
                cudaMemcpyAsync(l->d_k_cache + (size_t)path_pos[j] * kv_m,
                    g_qwen.d_spec_k_arch + ((size_t)full_idx * n_alloc + node) * kv_m,
                    (size_t)kv_m * sizeof(float), cudaMemcpyDeviceToDevice, g_qwen.stream);
                cudaMemcpyAsync(l->d_v_cache + (size_t)path_pos[j] * kv_m,
                    g_qwen.d_spec_v_arch + ((size_t)full_idx * n_alloc + node) * kv_m,
                    (size_t)kv_m * sizeof(float), cudaMemcpyDeviceToDevice, g_qwen.stream);
            }
            full_idx++;
        }
    }
    if (g_spec_state_skip_round) {
        // one replay launch reconstructs the accepted leaf's recur state for ALL
        // linear layers from the nearest published fork (or live), bit-exactly
        if (!g_qwen.live_recur_ptrs_ready) {
            float *h_ptrs[64];
            int li = 0;
            for (int L2 = 0; L2 < g_qwen.L && li < 64; L2++)
                if (g_qwen.layer_types[L2] == TQ_LAYER_LINEAR_ATTENTION)
                    h_ptrs[li++] = g_qwen.layers[L2].d_linear_recurrent_state_owned;
            cudaMemcpyAsync(g_qwen.d_live_recur_ptrs, h_ptrs, (size_t)li * sizeof(float *),
                            cudaMemcpyHostToDevice, g_qwen.stream);
            g_qwen.live_recur_ptrs_ready = 1;
        }
        // replay chain = accepted-path suffix below the last published (fork) node
        int h_replay[TQ_SPEC_MAX_N + 2];
        int src = -1, start = 0;
        for (int j = m - 1; j >= 0; j--)
            if (g_spec_pub_h[path_nodes[j]]) { src = path_nodes[j]; start = j + 1; break; }
        int cl = 0;
        for (int j = start; j < m; j++) h_replay[2 + cl++] = path_nodes[j];
        h_replay[0] = cl;
        h_replay[1] = src;
        cudaMemcpyAsync(g_qwen.d_spec_replay, h_replay, (size_t)(cl + 2) * sizeof(int),
                        cudaMemcpyHostToDevice, g_qwen.stream);
        int heads = g_qwen.linear_num_value_heads, key_heads = g_qwen.linear_num_key_heads;
        int dim = g_qwen.linear_value_head_dim, value_dim = heads * dim;
        size_t dn_stride = (size_t)n_alloc * (key_heads * 2 * dim + value_dim + heads * 2);
        int n_lin_total = 0;
        for (int L2 = 0; L2 < g_qwen.L; L2++)
            if (g_qwen.layer_types[L2] == TQ_LAYER_LINEAR_ATTENTION) n_lin_total++;
        k_tq_spec_commit_replay<8, 4><<<dim3(heads * 4, n_lin_total), 256, 0, g_qwen.stream>>>(
            (float *const *)g_qwen.d_live_recur_ptrs, g_qwen.d_spec_recur,
            (size_t)n_alloc * recur_f, (size_t)recur_f,
            g_qwen.d_spec_dnprep, dn_stride, g_qwen.d_spec_replay,
            heads, key_heads, dim, value_dim, n_alloc);
    }
    if (g_spec_commit_swap) { g_spec_live_in_archive = 1; g_spec_commit_swap = 0; }
    return cudaStreamSynchronize(g_qwen.stream) == cudaSuccess ? 0 : -4;
}

// After spec_commit: take node `node_idx`'s pre-final-norm hidden from spec buffers,
// run final_norm + lm_head + argmax (NO full decode, NO weight read beyond lm_head).
// Sets d_debug_x (for MTP) and last_argmax_id/logit. Returns argmax token id.
extern "C" int qwn_spec_lmhead_from_node(int node_idx) {
    if (!g_qwen.initialized || !g_qwen.d_spec_h) return -1;
    if (node_idx < 0 || node_idx >= TQ_SPEC_MAX_N) return -2;
    cudaMemcpyAsync(g_qwen.d_debug_x,
                    g_qwen.d_spec_h + (size_t)node_idx * g_qwen.H,
                    (size_t)g_qwen.H * sizeof(float),
                    cudaMemcpyDeviceToDevice, g_qwen.stream);
    int ret = run_final_norm_device();
    if (ret != 0) return -3;
    ret = run_lm_head_argmax_device();
    if (ret != 0) return -4;
    return copy_forward_argmax_result(NULL);
}

// ====================== n-gram copy-drafter (CPU-side) ======================
// Suffix-index copy-drafting (probe-validated: +27-35% accept on copy-heavy
// code, neutral elsewhere; bench/calib/NGRAM_REPORT.md). The engine keeps a
// host-side history of (prompt + committed) tokens; before each round it looks
// up the longest recent n-gram match (n=4 -> 2, most-recent, min match 2) and
// grafts a <=TQ_NGRAM_CHAIN-token copy chain into the MTP tree as a second
// branch (dedup: follows existing matching nodes first). TQ_NGRAM_DRAFT=1
// enables (opt-in until the C-path e2e gate).
#define TQ_NGRAM_CHAIN 8
static int g_hist[TQ_MAX_SEQ];
static int g_hist_len = 0;

static void hist_reset(void) { g_hist_len = 0; }

static void hist_append(const int *toks, int n) {
    for (int i = 0; i < n && g_hist_len < TQ_MAX_SEQ; i++) g_hist[g_hist_len++] = toks[i];
}

static int ngram_draft_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("TQ_NGRAM_DRAFT");
        cached = (e && atoi(e) != 0) ? 1 : 0;
    }
    return cached;
}

// Longest-context (4 -> 2), most-recent suffix match against the history;
// returns the number of continuation tokens written to out (0 = no match).
static int ngram_lookup(int *out, int max_out) {
    if (g_hist_len < 3) return 0;
    for (int n = 4; n >= 2; n--) {
        if (g_hist_len < n + 1) continue;
        const int *suf = g_hist + g_hist_len - n;
        for (int s = g_hist_len - n - 1; s >= 0; s--) {   // most recent first
            int ok = 1;
            for (int j = 0; j < n; j++)
                if (g_hist[s + j] != suf[j]) { ok = 0; break; }
            if (!ok) continue;
            int cont = s + n;                              // continuation start
            int m = 0;
            while (m < max_out && cont + m < g_hist_len - n)
                out[m] = g_hist[cont + m], m++;
            if (m >= 1) return m;
        }
    }
    return 0;
}

// ====================== C-side spec-decode round driver ======================
// One call = one full round: build the MTP tree, DFS-order it, run the batched
// TQ_TEMP: target-side sampling temperature for the spec round (0 = greedy,
// bit-exact default path untouched). TQ_SEED: RNG seed (counter-based, replayable).


// Per-request sampling control (OpenAI server path): overrides the env values
// and resets the round counter so equal seeds reproduce equal streams.
extern "C" int qwn_set_sampling(float temp, unsigned long long seed) {
    g_spec_temp_v = (temp < 0.0f) ? 0.0f : temp;
    g_spec_seed_v = seed;
    g_spec_seed_init = 1;
    g_spec_round_ctr = 0;
    return 0;
}

// Prefix-cache support: zero the positional state rows [lo, hi) -- full-attn
// KV (+fp8 scales) and the MTP trunk cache -- exactly what qwn_reset_state
// zeroes there. A continuation that reuses rows [0, lo) and clears [lo, hi)
// is bitwise indistinguishable from reset+prefill at every row, so requests
// whose state extended PAST a later request's prompt cannot leak into it
// (measured: stale rows above the new prompt flip near-tie argmaxes).
extern "C" int qwn_kv_clear_rows(int lo, int hi) {
    if (!g_qwen.initialized) return -1;
    int ret = ensure_decode_state();
    if (ret != 0) return ret;
    if (lo < 0) lo = 0;
    if (hi < 0 || hi > g_qwen.max_seq) hi = g_qwen.max_seq;
    if (lo >= hi) return 0;
    size_t rows = (size_t)(hi - lo);
    int full_kv = g_qwen.nkv * g_qwen.hd;
    tq_layer_t *clear_set[TQ_MAX_LAYERS + 1];
    int n_clear = 0;
    for (int i = 0; i < g_qwen.L; i++)
        if (g_qwen.layer_types[i] == TQ_LAYER_FULL_ATTENTION)
            clear_set[n_clear++] = &g_qwen.layers[i];
    if (g_qwen.has_mtp_section)
        clear_set[n_clear++] = &g_qwen.mtp_layer;   // trunk cache rows
    for (int c = 0; c < n_clear; c++) {
        tq_layer_t *l = clear_set[c];
        if (g_qwen.kv_q4 && l->d_k_cache4) {
            cudaMemsetAsync(l->d_k_cache4 + ((size_t)lo * full_kv >> 1), 0,
                            (rows * full_kv) >> 1, g_qwen.stream);
            cudaMemsetAsync((char *)l->d_k_q4s +
                                (size_t)lo * g_qwen.nkv * 16 * (g_qwen.kv_q4_fp32s ? sizeof(float) : sizeof(__half)), 0,
                            rows * g_qwen.nkv * 16 * (g_qwen.kv_q4_fp32s ? sizeof(float) : sizeof(__half)), g_qwen.stream);
            cudaMemsetAsync(l->d_v_cache8 + (size_t)lo * full_kv, 0, rows * full_kv, g_qwen.stream);
            cudaMemsetAsync(l->d_v_scale + (size_t)lo * g_qwen.nkv, 0,
                            rows * g_qwen.nkv * sizeof(float), g_qwen.stream);
        } else if (g_qwen.kv_fp8 && l->d_k_cache8) {
            cudaMemsetAsync(l->d_k_cache8 + (size_t)lo * full_kv, 0, rows * full_kv, g_qwen.stream);
            cudaMemsetAsync(l->d_v_cache8 + (size_t)lo * full_kv, 0, rows * full_kv, g_qwen.stream);
            cudaMemsetAsync(l->d_k_scale + (size_t)lo * g_qwen.nkv, 0,
                            rows * g_qwen.nkv * sizeof(float), g_qwen.stream);
            cudaMemsetAsync(l->d_v_scale + (size_t)lo * g_qwen.nkv, 0,
                            rows * g_qwen.nkv * sizeof(float), g_qwen.stream);
        } else if (l->d_k_cache) {
            cudaMemsetAsync(l->d_k_cache + (size_t)lo * full_kv, 0,
                            rows * full_kv * sizeof(float), g_qwen.stream);
            cudaMemsetAsync(l->d_v_cache + (size_t)lo * full_kv, 0,
                            rows * full_kv * sizeof(float), g_qwen.stream);
        }
    }
    return cudaStreamSynchronize(g_qwen.stream) == cudaSuccess ? 0 : -5;
}

// Debug/proto: copy fp32 KV cache rows [lo,hi) of one FULL-ATTENTION layer to
// host (k_out/v_out: (hi-lo) x nkv*hd floats). fp32-KV mode only (the int4
// quantizer prototype needs the exact engine-side K/V distribution, including
// RoPE and k-norm). Returns rows copied, <0 on error.
extern "C" int qwn_kv_dump(int layer, int lo, int hi, float *k_out, float *v_out) {
    if (!g_qwen.initialized) return -1;
    if (layer < 0 || layer >= g_qwen.L) return -2;
    if (g_qwen.layer_types[layer] != TQ_LAYER_FULL_ATTENTION) return -3;
    tq_layer_t *l = &g_qwen.layers[layer];
    if (!l->d_k_cache || !l->d_v_cache) return -4;       // fp8 mode or not allocated
    if (lo < 0 || hi > g_qwen.max_seq || lo >= hi) return -5;
    size_t kv_m = (size_t)g_qwen.nkv * g_qwen.hd;
    size_t off = (size_t)lo * kv_m, n = (size_t)(hi - lo) * kv_m;
    if (cudaMemcpy(k_out, l->d_k_cache + off, n * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
        return -6;
    if (cudaMemcpy(v_out, l->d_v_cache + off, n * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
        return -7;
    return hi - lo;
}

// Commit-stop set (prefix-cache support): tokens at which qwn_spec_round stops
// EXTENDING the accepted path. The stop token itself is returned as the pending
// seed (io_state[0]) and is NOT committed, so the engine's committed state ends
// exactly at the last pre-stop token -- a clean, cacheable prefix (no post-EOS
// junk landing in KV/DeltaNet/trunk). Empty set (default) = behavior unchanged.
#define TQ_COMMIT_STOP_MAX 8
static int g_commit_stop[TQ_COMMIT_STOP_MAX];
static int g_commit_stop_n = 0;

extern "C" int qwn_set_commit_stop(const int *toks, int n) {
    if (n < 0 || n > TQ_COMMIT_STOP_MAX) return -1;
    if (n > 0 && !toks) return -2;
    for (int i = 0; i < n; i++) g_commit_stop[i] = toks[i];
    g_commit_stop_n = n;
    return 0;
}

// Truncate the accepted path at the first stop token: it becomes accepted[m-1]
// (the pending seed), nothing beyond it is committed. m stays >= 1.
static inline void commit_stop_truncate(const int *accepted, int *m) {
    if (g_commit_stop_n == 0) return;
    for (int j = 0; j < *m; j++)
        for (int s = 0; s < g_commit_stop_n; s++)
            if (accepted[j] == g_commit_stop[s]) { *m = j + 1; return; }
}

// verify, descend the argmax chain, 1-read commit, extend the MTP trunk for the
// accepted tokens. Replaces ~6.5 ms/round of Python/ctypes orchestration.
// Outputs the accepted chain (seed + accepted tokens, <= max_nodes+1) and the
// updated seed/base_pos. The caller keeps the dense-equivalence semantics of
// tools/mtp_spec_smoke.py's lmhead-only loop.
extern "C" int qwn_mtp_tree_build(int seed_token, int pos, int depth, int k, float tau,
                                  int max_nodes, int *out_tokens, int *out_parents,
                                  int *out_depths, float *out_margins);
static int mtp_advance_wave(const float *hidden_src, const int *node_idx, const int *toks, int pos0, int m);

extern "C" int qwn_spec_round(int seed, int base_pos, int depth, int k, float tau,
                              int max_nodes, int *out_chain, int *io_state) {
    if (!g_qwen.initialized) return -1;
    if (max_nodes < 1 || max_nodes > TQ_SPEC_MAX_N) return -2;
    int tree_tok[TQ_SPEC_MAX_N], tree_par[TQ_SPEC_MAX_N], tree_dep[TQ_SPEC_MAX_N];
    float tree_mrg[TQ_SPEC_MAX_N];
    int n = qwn_mtp_tree_build(seed, base_pos, depth, k, tau, max_nodes - 1,
                               tree_tok, tree_par, tree_dep, tree_mrg);
    if (n < 0) return n;
    // merged n-gram policy: graft a copy chain (dedup walk first: descend along
    // existing nodes whose tokens already match the chain)
    if (ngram_draft_enabled()) {
        int chain[TQ_NGRAM_CHAIN];
        int cl = ngram_lookup(chain, TQ_NGRAM_CHAIN);
        int cur = -1, ci = 0;                       // -1 = root
        while (ci < cl) {
            int found = -1;
            for (int i = 0; i < n; i++)
                if (tree_par[i] == cur && tree_tok[i] == chain[ci]) { found = i; break; }
            if (found < 0) break;
            cur = found; ci++;
        }
        int dep = (cur < 0) ? 0 : tree_dep[cur];
        while (ci < cl && n + 1 < TQ_SPEC_MAX_N && dep + 1 <= 15) {
            tree_tok[n] = chain[ci];
            tree_par[n] = cur;
            tree_dep[n] = ++dep;
            tree_mrg[n] = 0.0f;
            cur = n; n++; ci++;
        }
    }

    // verify node list: node 0 = seed root, tree nodes shifted by 1, DFS order
    int vtok[TQ_SPEC_MAX_N], vpar[TQ_SPEC_MAX_N], vdep[TQ_SPEC_MAX_N];
    vtok[0] = seed; vpar[0] = -1; vdep[0] = 0;
    for (int i = 0; i < n; i++) {
        vtok[i + 1] = tree_tok[i];
        vpar[i + 1] = tree_par[i] < 0 ? 0 : tree_par[i] + 1;
        vdep[i + 1] = tree_dep[i];
    }
    int nv = n + 1;
    // iterative DFS (children in recorded order; matches the Python dfs_order)
    int order[TQ_SPEC_MAX_N], stack[TQ_SPEC_MAX_N], pos_of[TQ_SPEC_MAX_N];
    int sp = 0, no = 0;
    stack[sp++] = 0;
    while (sp > 0) {
        int x = stack[--sp];
        order[no++] = x;
        for (int ci = nv - 1; ci >= 1; ci--)   // reversed push = recorded order pop
            if (vpar[ci] == x) stack[sp++] = ci;
    }
    int dtok[TQ_SPEC_MAX_N], dpar[TQ_SPEC_MAX_N], ddep[TQ_SPEC_MAX_N], dpos[TQ_SPEC_MAX_N];
    for (int i = 0; i < nv; i++) pos_of[order[i]] = i;
    for (int i = 0; i < nv; i++) {
        int o = order[i];
        dtok[i] = vtok[o];
        dpar[i] = vpar[o] < 0 ? -1 : pos_of[vpar[o]];
        ddep[i] = vdep[o];
        dpos[i] = base_pos + 1 + vdep[o];
    }
    int ret;
    // Cached-graph verify (TQ_C_GRAPH=1): one capture per N, then replay -- the
    // per-round topology/tokens are re-uploaded OUTSIDE the graph and read by the
    // kernels from device buffers. Trees are PADDED to a fixed N (= max_nodes+1)
    // with extra chain drafts under the deepest leaf: pad nodes are legitimate
    // (if cheap) guesses, so descent/commit semantics are unchanged.
    {
        static int c_graph = -1;
        if (c_graph < 0) {
            const char *e = getenv("TQ_C_GRAPH");
            c_graph = (e && atoi(e) != 0) ? 1 : 0;
        }
        if (c_graph && max_nodes <= TQ_SPEC_MAX_N && !ngram_draft_enabled()) {
            // verify nodes = root + (max_nodes-1) drafts = max_nodes; pad only
            // the rounds whose tau-pruned tree came up short
            int NG = max_nodes;
            int deepest = nv - 1;
            for (int i = 0; i < nv; i++)
                if (ddep[i] > ddep[deepest]) deepest = i;
            int par = deepest, dep = ddep[deepest];
            for (int i = nv; i < NG; i++) {        // pad: token-0 chain under the leaf
                dtok[i] = 0; dpar[i] = par; ddep[i] = ++dep;
                dpos[i] = base_pos + 1 + ddep[i];
                par = i;
            }
            if (dep <= 15) {
                if (!g_qwen.spec_graph_ready || g_qwen.spec_graph_N != NG) {
                    ret = qwn_capture_spec_forward(dtok, dpar, ddep, dpos, NG);
                    if (ret != 0) return ret;
                }
                spec_stage_round(dtok, dpar, ddep, dpos, NG, 1);
                cudaError_t eg = cudaGraphLaunch(g_qwen.spec_graph_exec, g_qwen.stream);
                if (eg != cudaSuccess) {
                    fprintf(stderr, "c-round graph launch failed: %s\n", cudaGetErrorString(eg));
                    return -95;
                }
                int path[TQ_SPEC_MAX_N], accepted[TQ_SPEC_MAX_N];
                int m = 0;
                if (spec_temp() > 0.0f) {
                    ret = spec_sample_descend(nv, dtok, dpar, spec_temp(), spec_seed(),
                                              g_spec_round_ctr++, path, accepted, &m);
                    if (ret != 0) return ret;
                } else {
                    int amg[TQ_SPEC_MAX_N];
                    ret = spec_argmax(NG, amg);
                    if (ret != 0) return ret;
                    int am[TQ_SPEC_MAX_N];
                    for (int i = 0; i < nv; i++) am[i] = amg[i];
                    int cur = 0, a = am[0];
                    path[m] = 0; accepted[m] = a; m++;
                    while (m < TQ_SPEC_MAX_N) {
                        int nxt = -1;
                        for (int ci = 0; ci < nv; ci++)
                            if (dpar[ci] == cur && dtok[ci] == a) { nxt = ci; break; }
                        if (nxt < 0) break;
                        cur = nxt; a = am[cur];
                        path[m] = cur; accepted[m] = a; m++;
                    }
                }
                commit_stop_truncate(accepted, &m);
                {
                    int path_pos[TQ_SPEC_MAX_N];
                    for (int j = 0; j < m; j++) path_pos[j] = base_pos + 1 + j;
                    ret = qwn_spec_commit(path, path_pos, m);
                    if (ret != 0) return ret;
                    ret = qwn_spec_set_root_from(path[m - 1]);
                    if (ret != 0) return ret;
                    if (g_qwen.mtp_quant_ready && mtp_quant_enabled() && !getenv("TQ_MTP_SEQ_ADVANCE")) {
                        ret = mtp_advance_wave(g_qwen.d_spec_h, path, accepted, base_pos + 1, m);
                        if (ret != 0) return ret;
                    } else {
                        for (int j = 0; j < m; j++) {
                            ret = qwn_mtp_advance_from_spec(path[j], accepted[j], base_pos + 1 + j);
                            if (ret != 0) return ret;
                        }
                    }
                    out_chain[0] = seed;
                    for (int j = 0; j < m; j++) out_chain[j + 1] = accepted[j];
                    hist_append(accepted, m);
                    io_state[0] = accepted[m - 1];
                    io_state[1] = base_pos + m;
                    return m + 1;
                }
            }
        }
    }
    ret = run_spec_forward_n(dtok, dpar, ddep, dpos, nv, g_qwen.L);
    if (ret != 0) return ret;
    int path[TQ_SPEC_MAX_N], accepted[TQ_SPEC_MAX_N];
    int m = 0;
    if (spec_temp() > 0.0f) {
        ret = spec_sample_descend(nv, dtok, dpar, spec_temp(), spec_seed(),
                                  g_spec_round_ctr++, path, accepted, &m);
        if (ret != 0) return ret;
    } else {
        int am[TQ_SPEC_MAX_N];
        ret = spec_argmax(nv, am);
        if (ret != 0) return ret;
        // greedy descent: follow the argmax chain through matching children
        int cur = 0, a = am[0];
        path[m] = 0; accepted[m] = a; m++;
        while (m < TQ_SPEC_MAX_N) {
            int nxt = -1;
            for (int ci = 0; ci < nv; ci++)
                if (dpar[ci] == cur && dtok[ci] == a) { nxt = ci; break; }
            if (nxt < 0) break;
            cur = nxt; a = am[cur];
            path[m] = cur; accepted[m] = a; m++;
        }
    }
    commit_stop_truncate(accepted, &m);
    int path_pos[TQ_SPEC_MAX_N];
    for (int j = 0; j < m; j++) path_pos[j] = base_pos + 1 + j;
    ret = qwn_spec_commit(path, path_pos, m);
    if (ret != 0) return ret;
    ret = qwn_spec_set_root_from(path[m - 1]);
    if (ret != 0) return ret;
    if (g_qwen.mtp_quant_ready && mtp_quant_enabled() && !getenv("TQ_MTP_SEQ_ADVANCE")) {
        ret = mtp_advance_wave(g_qwen.d_spec_h, path, accepted, base_pos + 1, m);
        if (ret != 0) return ret;
    } else {
        for (int j = 0; j < m; j++) {
            ret = qwn_mtp_advance_from_spec(path[j], accepted[j], base_pos + 1 + j);
            if (ret != 0) return ret;
        }
    }
    // chain = seed + accepted; new seed = last accepted; base_pos += path length
    out_chain[0] = seed;
    for (int j = 0; j < m; j++) out_chain[j + 1] = accepted[j];
    hist_append(accepted, m);
    io_state[0] = accepted[m - 1];      // next seed
    io_state[1] = base_pos + m;         // next base_pos
    return m + 1;                        // chain length
}

// Batched kv-only MTP trunk advances: m independent (hidden, token) pairs in one
// fused quantized wave; knorm+rope written straight into the trunk cache slots.
// hidden j = hidden_src[node_idx[j] * H]; position j = pos0 + j. hidden_src is the
// per-token pre-final-norm hidden buffer: d_spec_h (baseline prefill/verify path) or
// g_wide_h (wide prefill -- the trunk bridge that lets spec-decode run after a wide
// prefill chunk; same row layout, equally-valid forward up to float-eps).
static int mtp_advance_wave(const float *hidden_src, const int *node_idx, const int *toks, int pos0, int m) {
    int H = g_qwen.H, kv_m = g_qwen.nkv * g_qwen.hd;
    tq_layer_t *mt = &g_qwen.mtp_layer;
    int ret = ensure_mtp_batch_buffers();
    if (ret != 0) return ret;
    int hctl[16 + 16 + 16];
    for (int j = 0; j < m; j++) hctl[j] = toks[j];
    for (int j = 0; j < m; j++) hctl[16 + j] = node_idx[j];
    for (int j = 0; j < m; j++) hctl[32 + j] = pos0 + j;
    cudaMemcpyAsync(g_mtpb_dev, hctl, sizeof(hctl), cudaMemcpyHostToDevice, g_qwen.stream);
    int *d_tok = g_mtpb_dev, *d_gidx = g_mtpb_dev + 16, *d_pos = g_mtpb_dev + 32;
    k_tq_embed_lookup_nk<<<dim3((H + 255) / 256, m), 256, 0, g_qwen.stream>>>(
        g_mtpb_x, g_qwen.d_embed, d_tok, H);
    k_tq_qwen_rmsnorm_b_out<<<dim3(8, m), 1024, 0, g_qwen.stream>>>(
        g_mtpb_fc_in, 2 * H, 0, g_mtpb_x, g_qwen.d_mtp_pre_fc_norm_emb, H, g_qwen.eps);
    k_tq_gather_rows_nk<<<dim3((H + 255) / 256, m), 256, 0, g_qwen.stream>>>(
        g_mtpb_x, hidden_src, d_gidx, H);
    k_tq_qwen_rmsnorm_b_out<<<dim3(8, m), 1024, 0, g_qwen.stream>>>(
        g_mtpb_fc_in, 2 * H, H, g_mtpb_x, g_qwen.d_mtp_pre_fc_norm_hidden, H, g_qwen.eps);
    if (mtp_wave_gemv(0, &g_qwen.mtp_fc, g_mtpb_fc_in, 2 * H, g_mtpb_x, m, 1) != 0) return -61;
    k_tq_qwen_rmsnorm_b<<<dim3(8, m), 1024, 0, g_qwen.stream>>>(
        g_qwen.d_spec_norm, g_mtpb_x, mt->d_input_ln, H, g_qwen.eps);
    if (mtp_wave_gemv(2, &g_qwen.mtp_k_proj, g_qwen.d_spec_norm, H, g_qwen.d_spec_kbuf, m, 1) != 0) return -61;
    if (mtp_wave_gemv(3, &g_qwen.mtp_v_proj, g_qwen.d_spec_norm, H, g_qwen.d_spec_z, m, 0) != 0) return -61;
    k_tq_spec_attn_kv_prep<<<dim3(g_qwen.nkv, m), g_qwen.hd, 0, g_qwen.stream>>>(
        mt->d_k_cache, mt->d_v_cache, g_qwen.d_spec_kbuf, g_qwen.d_spec_z,
        mt->d_k_norm, d_pos, g_qwen.nkv, g_qwen.hd, kv_m, g_qwen.eps, g_qwen.rope_theta, d_pos,
        mt->d_k_cache8, mt->d_k_scale, mt->d_v_cache8, mt->d_v_scale,
        mt->d_k_cache4, mt->d_k_q4s, g_qwen.kv_q4);
    return cudaStreamSynchronize(g_qwen.stream) == cudaSuccess ? 0 : -62;
}

// Exported: trunk advances for a prefill chunk (nodes are chunk-local hidden rows).
extern "C" int qwn_mtp_advance_wave(const int *node_idx, const int *toks, int pos0, int m) {
    if (!g_qwen.has_mtp_section) return -1;
    if (m < 1 || m > 16) return -2;
    if (!(g_qwen.mtp_quant_ready && mtp_quant_enabled())) {
        int ret = ensure_mtp_quant();
        if (ret != 0) return ret;
    }
    return mtp_advance_wave(g_qwen.d_spec_h, node_idx, toks, pos0, m);
}

// ===================== D.1 (2): wide prefill forward (needle wiring) =====================
// A self-contained N-wide prefill chunk forward that drives the new k_tq_wide_attn at the
// chunk width with NO 16-slot archive. Full-attention layers run BATCHED: per-token
// rmsnorm + raw q/k/v projections into [token][proj] buffers, then k_tq_wide_attn_kv_write
// + k_tq_wide_attn straight to the live fp32 KV cache, then per-token o_proj(+residual) +
// MLP. Linear-attention layers reuse the proven per-token decode path verbatim (the
// DeltaNet recurrence is sequential, so per-token IS the chunk recurrence). This does NOT
// touch the shipping qwn_prefill_chunk / run_spec_forward_n path (which stays capped at
// TQ_SPEC_MAX_N). fp32 KV cache ONLY (TQ_KV_FP8 / TQ_KV_Q4 rejected -- the wide K/V store
// is a deliberate TODO). The point is to validate that wide chunks no longer corrupt the
// long-context KV state (the 8k needle that the 16-slot archive failed at N>16).
static float *g_wide_h = NULL, *g_wide_q = NULL, *g_wide_k = NULL, *g_wide_v = NULL, *g_wide_core = NULL;
static float *g_wide_norm = NULL, *g_wide_resid = NULL, *g_wide_o = NULL, *g_wide_pn = NULL;
static float *g_wide_gate = NULL, *g_wide_up = NULL, *g_wide_mh = NULL, *g_wide_down = NULL;
static float *g_wide_qkv = NULL, *g_wide_z = NULL, *g_wide_b = NULL, *g_wide_a = NULL, *g_wide_lcore = NULL;
static float *g_wide_conv = NULL;      // conv outputs (chunk-parallel conv must not run in place)
static int   *g_wide_pos = NULL;
static int    g_wide_cap = 0;
static int    g_wide_capture_mode = 0;   // when set: skip the pos-setup host sync (graph capture)

static int ensure_wide_prefill_buffers(int N) {
    if (g_wide_cap >= N) return 0;
    graph_invalidate();
    float **bufs[] = {&g_wide_h, &g_wide_q, &g_wide_k, &g_wide_v, &g_wide_core, &g_wide_norm,
                      &g_wide_resid, &g_wide_o, &g_wide_pn, &g_wide_gate, &g_wide_up,
                      &g_wide_mh, &g_wide_down, &g_wide_qkv, &g_wide_z, &g_wide_b,
                      &g_wide_a, &g_wide_lcore, &g_wide_conv};
    for (size_t i = 0; i < sizeof(bufs) / sizeof(bufs[0]); i++) {
        if (*bufs[i]) cudaFree(*bufs[i]);
        *bufs[i] = NULL;
    }
    if (g_wide_pos) cudaFree(g_wide_pos);
    g_wide_pos = NULL;
    int H = g_qwen.H, I = g_qwen.I;
    int q_m = g_qwen.nh * g_qwen.hd * 2, kv_m = g_qwen.nkv * g_qwen.hd, attn_m = g_qwen.nh * g_qwen.hd;
    if (cudaMalloc(&g_wide_h,    (size_t)N * H * sizeof(float)) != cudaSuccess) return -1;
    if (cudaMalloc(&g_wide_q,    (size_t)N * q_m * sizeof(float)) != cudaSuccess) return -1;
    if (cudaMalloc(&g_wide_k,    (size_t)N * kv_m * sizeof(float)) != cudaSuccess) return -1;
    if (cudaMalloc(&g_wide_v,    (size_t)N * kv_m * sizeof(float)) != cudaSuccess) return -1;
    if (cudaMalloc(&g_wide_core, (size_t)N * attn_m * sizeof(float)) != cudaSuccess) return -1;
    if (cudaMalloc(&g_wide_norm, (size_t)N * H * sizeof(float)) != cudaSuccess) return -1;
    if (cudaMalloc(&g_wide_resid,(size_t)N * H * sizeof(float)) != cudaSuccess) return -1;
    if (cudaMalloc(&g_wide_o,    (size_t)N * H * sizeof(float)) != cudaSuccess) return -1;
    if (cudaMalloc(&g_wide_pn,   (size_t)N * H * sizeof(float)) != cudaSuccess) return -1;
    if (cudaMalloc(&g_wide_gate, (size_t)N * I * sizeof(float)) != cudaSuccess) return -1;
    if (cudaMalloc(&g_wide_up,   (size_t)N * I * sizeof(float)) != cudaSuccess) return -1;
    if (cudaMalloc(&g_wide_mh,   (size_t)N * I * sizeof(float)) != cudaSuccess) return -1;
    if (cudaMalloc(&g_wide_down, (size_t)N * H * sizeof(float)) != cudaSuccess) return -1;
    int lkh = g_qwen.linear_num_key_heads, lvh = g_qwen.linear_num_value_heads, ld = g_qwen.linear_value_head_dim;
    int value_dim = lvh * ld, key_dim = lkh * ld, conv_dim = 2 * key_dim + value_dim;
    if (cudaMalloc(&g_wide_qkv,  (size_t)N * conv_dim * sizeof(float)) != cudaSuccess) return -1;
    if (cudaMalloc(&g_wide_conv, (size_t)N * conv_dim * sizeof(float)) != cudaSuccess) return -1;
    if (cudaMalloc(&g_wide_z,    (size_t)N * value_dim * sizeof(float)) != cudaSuccess) return -1;
    if (cudaMalloc(&g_wide_b,    (size_t)N * lvh * sizeof(float)) != cudaSuccess) return -1;
    if (cudaMalloc(&g_wide_a,    (size_t)N * lvh * sizeof(float)) != cudaSuccess) return -1;
    if (cudaMalloc(&g_wide_lcore,(size_t)N * value_dim * sizeof(float)) != cudaSuccess) return -1;
    if (cudaMalloc(&g_wide_pos,  (size_t)N * sizeof(int)) != cudaSuccess) return -1;
    g_wide_cap = N;
    return 0;
}

static int wide_quant_input(const float *d_x, int K, int N);
static int wide_proj(const tq_qmma_weight_t *w, float *out, int N);
static const float *g_exl3_x2 = NULL;          // optional SiLU gate partner of the recorded EXL3 input (set by wide_mlp)
static int wide_proj_group(const tq_qmma_weight_t *const *ws, float *const *outs, int n, int N);
// Q4(K)+E4M3(V) batched/wide attention (defined in the roadmap-E section below). With
// {k4,kq4s,v8,vscale}_stride = 0 every column shares ONE cache -> wide-prefill Q4 attention.
static int launch_batched_attn_q4(float *out, const float *q_proj, const uint16_t *q_norm_w,
                                  const float *k_proj, const float *v_proj, const uint16_t *k_norm_w,
                                  uint8_t *k4_base, float *kq4s_base, uint8_t *v8_base, float *vscale_base,
                                  const int *positions, int N, int max_pos,
                                  size_t k4_stride, size_t kq4s_stride, size_t v8_stride, size_t vscale_stride,
                                  cudaStream_t st);

// Wide MLP for a chunk of n tokens: post-rmsnorm(resid) -> gate/up (wide, shared B-frag)
// -> silu*mul -> down (wide) -> layer_out = resid + down. resid and layer_out are [n x H]
// (token-major); may NOT alias. Returns 0; -1 if mlp weights aren't e2m3-SF (caller falls back).
// TQ_FUSE_NORM (default 1): the decode/verify steps fuse (a) the residual add after o_proj / out_proj with the
// post-attention norm, (b) SiLU * up into the down projection's fragment build, (c) the layer's final residual add
// with the NEXT layer's input norm (the final norm after the last layer). Same expressions, same reduction order ->
// bitwise identical to the separate kernels (k_tq_add_rmsnorm_b has k_tq_qwen_rmsnorm_b's reduction); -128 launches
// per step. TQ_FUSE_NORM=0 reverts to the separate kernels.
static int tq_fuse_norm(void) {
    static int c = -1;
    if (c < 0) { const char *e = getenv("TQ_FUSE_NORM"); c = (e && *e) ? (atoi(e) != 0) : 1; }
    return c;
}
// add_a/add_b (optional): resid = add_a + add_b is computed here, fused with the post norm. next_ln/next_norm
// (optional): the final residual add also writes next_norm = rmsnorm(layer_out, next_ln).
static int wide_mlp(tq_layer_t *l, float *resid, float *layer_out, int n,
                    const float *add_a = NULL, const float *add_b = NULL,
                    const uint16_t *next_ln = NULL, float *next_norm = NULL) {
    int H = g_qwen.H, I = g_qwen.I;
    int exl3_ok = l->mlp_gate.exl3 && l->mlp_up.exl3 && l->mlp_down.exl3;
    if (!exl3_ok &&
        !((l->mlp_gate.e2m3 || l->mlp_gate.e2m1) && (l->mlp_up.e2m3 || l->mlp_up.e2m1) &&
          (l->mlp_down.e2m3 || l->mlp_down.e2m1) &&
          can_use_qmma_sf_weight(&l->mlp_gate) && !l->mlp_gate.row_major &&
          can_use_qmma_sf_weight(&l->mlp_down) && !l->mlp_down.row_major)) return -1;
    if (add_a) {
        tq_launch(g_pdl_small, k_tq_add_rmsnorm_b, dim3(8, n), dim3(1024), 0, g_qwen.stream,
                  g_wide_pn, resid, add_a, add_b, l->d_post_ln, H, g_qwen.eps);
    } else {
        tq_launch(g_pdl_small, k_tq_qwen_rmsnorm_b, dim3(8, n), dim3(1024), 0, g_qwen.stream,
                  g_wide_pn, resid, l->d_post_ln, H, g_qwen.eps);
    }
    int ret = wide_quant_input(g_wide_pn, H, n);
    if (ret != 0) return ret;
    const tq_qmma_weight_t *gu[2] = {&l->mlp_gate, &l->mlp_up};
    float *gu_out[2] = {g_wide_gate, g_wide_up};
    if ((ret = wide_proj_group(gu, gu_out, 2, n)) != 0) return ret;
    if (exl3_ok && tq_fuse_norm()) {
        // the down projection's fragment build applies silu(gate) * up itself
        if ((ret = wide_quant_input(g_wide_gate, I, n)) != 0) return ret;
        g_exl3_x2 = g_wide_up;
    } else {
        tq_launch(g_pdl_small, k_tq_silu_mul, dim3(((size_t)n * I + 255) / 256), dim3(256), 0, g_qwen.stream,
                  g_wide_mh, g_wide_gate, g_wide_up, (int)((size_t)n * I));
        if ((ret = wide_quant_input(g_wide_mh, I, n)) != 0) return ret;
    }
    if ((ret = wide_proj(&l->mlp_down, g_wide_down, n)) != 0) return ret;
    g_exl3_x2 = NULL;
    if (next_ln) {
        tq_launch(g_pdl_small, k_tq_add_rmsnorm_b, dim3(8, n), dim3(1024), 0, g_qwen.stream,
                  next_norm, layer_out, resid, g_wide_down, next_ln, H, g_qwen.eps);
    } else {
        tq_launch(g_pdl_small, k_tq_add_vec, dim3(((size_t)n * H + 255) / 256), dim3(256), 0, g_qwen.stream,
                  layer_out, resid, g_wide_down, (int)((size_t)n * H));
    }
    return 0;
}
