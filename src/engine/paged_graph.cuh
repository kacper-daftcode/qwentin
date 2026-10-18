// the paged decode step (one CUDA graph per shape class, TQ_GRAPH) and the paged prefill waves.
// Part of the forward_qwen.cu unity build: included in order, not a standalone header.

// ===== TQ_GRAPH: the decode step as a CUDA graph per shape class =====
// Key = everything the captured launch sequence depends on besides the device buffers: N, the GQA key-split
// count S (a function of N and the step's max position -- the split picker's cap changes every 256 positions
// below ~16K at N = 1), the MTP column count, the mode flags. A small LRU of instantiated graphs; a miss captures
// the body (cudaStreamCaptureModeRelaxed: a lazily grown buffer may be allocated during the capture, which drops
// the older graphs through graph_invalidate) and instantiates it.
#define TQ_GRAPH_CACHE 32
struct tq_graph_ent { unsigned long long key; cudaGraphExec_t exec; unsigned tick; int nodes; };
static tq_graph_ent g_graph_cache[TQ_GRAPH_CACHE];
static int g_graph_n = 0;
static unsigned g_graph_tick = 0;
static long g_graph_hits = 0, g_graph_captures = 0, g_graph_evictions = 0;
static double g_graph_capture_ms = 0.0;             // total host time spent capturing + instantiating
static int g_graph_disabled = 0;                    // a capture failed: stay eager (logged once)
static void graph_cache_clear(void) {
    for (int i = 0; i < g_graph_n; i++) if (g_graph_cache[i].exec) cudaGraphExecDestroy(g_graph_cache[i].exec);
    g_graph_n = 0;
}
static cudaGraphExec_t graph_cache_get(unsigned long long key) {
    for (int i = 0; i < g_graph_n; i++)
        if (g_graph_cache[i].key == key) { g_graph_cache[i].tick = ++g_graph_tick; g_graph_hits++; return g_graph_cache[i].exec; }
    return NULL;
}
static void graph_cache_put(unsigned long long key, cudaGraphExec_t exec, int nodes) {
    int slot = g_graph_n;
    if (g_graph_n == TQ_GRAPH_CACHE) {              // evict the least recently used
        slot = 0;
        for (int i = 1; i < g_graph_n; i++) if (g_graph_cache[i].tick < g_graph_cache[slot].tick) slot = i;
        if (g_graph_cache[slot].exec) cudaGraphExecDestroy(g_graph_cache[slot].exec);
        g_graph_evictions++;
    } else g_graph_n++;
    g_graph_cache[slot].key = key; g_graph_cache[slot].exec = exec; g_graph_cache[slot].tick = ++g_graph_tick;
    g_graph_cache[slot].nodes = nodes;
}
static double tq_now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}
// Capture `body` on g_qwen.stream and instantiate it. Returns NULL (and disables the graph path) on failure.
template <typename Body>
static cudaGraphExec_t graph_capture(const char *what, Body body, int *nodes_out) {
    cudaStream_t st = g_qwen.stream;
    double t0 = tq_now_ms();
    cudaError_t e = cudaStreamBeginCapture(st, cudaStreamCaptureModeRelaxed);
    if (e != cudaSuccess) {
        fprintf(stderr, "[graph] %s: cudaStreamBeginCapture failed: %s -- staying eager\n", what, cudaGetErrorString(e));
        g_graph_disabled = 1; return NULL;
    }
    unsigned gen0 = g_graph_gen;
    int r = body();
    cudaGraph_t graph = NULL;
    cudaError_t e2 = cudaStreamEndCapture(st, &graph);
    if (r != 0 || e2 != cudaSuccess || !graph) {
        fprintf(stderr, "[graph] %s: capture failed (body %d, end %s) -- staying eager\n", what, r, cudaGetErrorString(e2));
        if (graph) cudaGraphDestroy(graph);
        cudaGetLastError();
        g_graph_disabled = 1; return NULL;
    }
    if (g_graph_gen != gen0) {
        // A lazily sized buffer grew (cudaFree + cudaMalloc) inside the capture: nodes recorded before the growth
        // still name the old allocation. Drop this graph, run the step eagerly (the buffers are now sized), and
        // capture again on the next call with this key (the generation is part of every key).
        if (tq_graph_debug())
            fprintf(stderr, "[graph] %s: a buffer grew inside the capture (gen %u -> %u) -- eager this step, recapture next\n",
                    what, gen0, g_graph_gen);
        cudaGraphDestroy(graph);
        return NULL;
    }
    size_t nn = 0; cudaGraphGetNodes(graph, NULL, &nn);
    if (tq_graph_debug()) {
        size_t ne = 0; cudaGraphGetEdges(graph, NULL, NULL, NULL, &ne);
        size_t npdl = 0;
        if (ne > 0) {
            cudaGraphNode_t *from = (cudaGraphNode_t *)malloc(ne * sizeof(cudaGraphNode_t));
            cudaGraphNode_t *to = (cudaGraphNode_t *)malloc(ne * sizeof(cudaGraphNode_t));
            cudaGraphEdgeData *ed = (cudaGraphEdgeData *)malloc(ne * sizeof(cudaGraphEdgeData));
            size_t ne2 = ne;
            if (from && to && ed && cudaGraphGetEdges(graph, from, to, ed, &ne2) == cudaSuccess)
                for (size_t i = 0; i < ne2; i++) if (ed[i].type == cudaGraphDependencyTypeProgrammatic) npdl++;
            free(from); free(to); free(ed);
        }
        fprintf(stderr, "[graph] %s: %zu nodes, %zu edges (%zu programmatic), capture %.2f ms\n", what, nn, ne, npdl,
                tq_now_ms() - t0);
    }
    double t1 = tq_now_ms();
    cudaGraphExec_t exec = NULL;
    e = cudaGraphInstantiate(&exec, graph, 0);
    cudaGraphDestroy(graph);
    if (e != cudaSuccess || !exec) {
        fprintf(stderr, "[graph] %s: cudaGraphInstantiate failed: %s -- staying eager\n", what, cudaGetErrorString(e));
        cudaGetLastError();
        g_graph_disabled = 1; return NULL;
    }
    double t2 = tq_now_ms();
    if (tq_graph_debug()) fprintf(stderr, "[graph] %s: instantiate %.2f ms\n", what, t2 - t1);
    g_graph_capture_ms += t2 - t0;
    g_graph_captures++;
    if (nodes_out) *nodes_out = (int)nn;
    return exec;
}
// Stats for the tools / notes: graph replays, captures, evictions, cached graphs, capture time.
extern "C" int qwn_graph_stats(long *hits, long *captures, long *evictions, int *cached, double *capture_ms) {
    if (hits) *hits = g_graph_hits;
    if (captures) *captures = g_graph_captures;
    if (evictions) *evictions = g_graph_evictions;
    if (cached) *cached = g_graph_n;
    if (capture_ms) *capture_ms = g_graph_capture_ms;
    return g_graph_disabled ? 1 : 0;
}

// The graph decode step. Returns 1 when the step is not covered (the caller runs the eager step), else the step's
// status. Prework (eager, stream-ordered before the replay): block tables + encoded lengths, tokens / slot ids /
// positions through the pinned staging rows into g_tok_d / g_slot_ids / g_wide_pos, the MTP control rows, the
// attention partials for the step's S. Then the replay, then the T4 tail.
static int run_paged_decode_step_graph(const int *tokens, const int *slot_ids_h, const int *positions, int N) {
    if (g_graph_disabled || !g_tok_d || !h_pin_ctl) return 1;
    if (!paged_lm_head_wide() || paged_topk_pages() > 0) return 1;             // paths with host-side per-step work
    if (!(paged_attn_gqa_mma_enabled() && (g_pg_page & 127) == 0 && g_qwen.hd == 256 &&
          g_qwen.nh % g_qwen.nkv == 0 && g_qwen.nh / g_qwen.nkv <= 16)) return 1;   // the scalar attention picker
    int dec_maxpos = 0, ret;
    if ((ret = paged_decode_prep(slot_ids_h, positions, N, &dec_maxpos)) != 0) return ret;
    cudaStream_t st = g_qwen.stream;
    paged_upload_tables(st);
    int *pt = h_pin_ctl, *ps = h_pin_ctl + g_pin_cols, *pp = h_pin_ctl + 2 * g_pin_cols;
    for (int j = 0; j < N; j++) { pt[j] = tokens[j]; ps[j] = slot_ids_h[j]; pp[j] = positions[j]; }
    cudaMemcpyAsync(g_tok_d, pt, (size_t)N * sizeof(int), cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(g_slot_ids, ps, (size_t)N * sizeof(int), cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(g_wide_pos, pp, (size_t)N * sizeof(int), cudaMemcpyHostToDevice, st);
    int mtp_m = g_mtp_on ? mtp_hook_decode_prep(tokens, slot_ids_h, positions, N) : 0;
    // the GQA key-split count the launcher will pick for this step (and its partials, allocated before the capture)
    int S = paged_attn_split_S(N, dec_maxpos, 1);
    if (S > 1 && ensure_attn_partials((size_t)g_qwen.nh * N * S) != 0) S = 1;
    const int fuse = tq_fuse_norm() && (g_qwen.flags & TQ_FLAG_EXL3_WEIGHTS);
    unsigned long long key = 1ull;                                   // kind 1 = decode step
    key = (key << 8) | (unsigned)N;
    key = (key << 8) | (unsigned)S;
    key = (key << 8) | (unsigned)mtp_m;
    key = (key << 4) | (unsigned)(fuse | (g_mtp_on << 1) | (g_pg_t4 << 2) | (g_pg_v4 << 3));
    key = (key << 4) | (unsigned)(paged_attn_st_enabled() & 3) | ((unsigned)(g_pdl_on | (g_pdl_small << 1)) << 2);
    key = (key << 4) | (unsigned)(g_pg_sink != NULL);
    key = (key << 16) | (g_graph_gen & 0xffff);
    cudaGraphExec_t exec = graph_cache_get(key);
    if (!exec) {
        int nodes = 0;
        exec = graph_capture("decode", [&]() { return paged_decode_body(tokens, slot_ids_h, positions, N, dec_maxpos, 1, mtp_m); }, &nodes);
        if (!exec) return 1;                                         // eager fallback (the prework is idempotent)
        key = (key & ~0xffffull) | (g_graph_gen & 0xffff);           // a buffer grown inside the capture bumped it
        graph_cache_put(key, exec, nodes);
        if (tq_graph_debug())
            fprintf(stderr, "[graph] decode: captured N=%d S=%d mtp_m=%d maxpos=%d (%d cached)\n", N, S, mtp_m, dec_maxpos, g_graph_n);
    }
    cudaError_t e = cudaGraphLaunch(exec, st);
    if (e != cudaSuccess) {
        fprintf(stderr, "[graph] decode: cudaGraphLaunch failed: %s\n", cudaGetErrorString(e));
        return -84;
    }
    return paged_decode_t4_tail(slot_ids_h, positions, N);
}

extern "C" int qwn_paged_decode_step(const int *tokens, const int *slot_ids, const int *positions,
                                     int N, int *out_argmax) {
    if (!g_pg_ready) return -1;
    if (N < 1 || N > g_pg_maxslots) return -2;
    if (!g_qwen.kv_q4) return -3;
    int pdl = tq_pdl_enabled();
    g_pdl_on = pdl != 0;
    g_pdl_small = pdl == 2;
    int ret = 1;
    if (tq_graph_enabled()) ret = run_paged_decode_step_graph(tokens, slot_ids, positions, N);
    if (ret == 1) ret = run_paged_decode_step_core(tokens, slot_ids, positions, N);
    g_pdl_on = g_pdl_small = 0;
    if (ret != 0) return ret;
    if (out_argmax)
        cudaMemcpyAsync(out_argmax, g_pg_argmax, (size_t)N * sizeof(int), cudaMemcpyDeviceToHost, g_qwen.stream);
    return cudaStreamSynchronize(g_qwen.stream) == cudaSuccess ? 0 : -9;
}

// Pool occupancy for capacity reporting: free blocks / total blocks / page / max blocks-per-seq.
extern "C" int qwn_paged_stats(int *free_blocks, int *total_blocks, int *page, int *max_blocks_per_seq) {
    if (!g_pg_ready) return -1;
    if (free_blocks) *free_blocks = g_pg_nfree;
    if (total_blocks) *total_blocks = g_pg_nblocks;
    if (page) *page = g_pg_page;
    if (max_blocks_per_seq) *max_blocks_per_seq = g_pg_maxblk;
    return 0;
}

// Test ABI (analysis/qwentin_e2e.py): while set, every paged prefill wave also writes the logits
// of ALL its T columns, and every paged decode step those of its N columns, into dev_out
// ([cols][V] fp32, caller-owned device memory with cap_cols rows). The wave's head runs head_cols
// columns at a time: served lm_head calls are decode-sized (f32-accumulate units), a 128-column
// head would take the fp16-accumulate EXL3 units instead. dev_out = NULL turns it off.
extern "C" int qwn_paged_logits_sink(float *dev_out, int cap_cols, int head_cols) {
    if (!g_pg_ready) return -1;
    graph_invalidate();                                      // the sink copy is a node of a captured step
    if (!dev_out) { g_pg_sink = NULL; g_pg_sink_cap = 0; g_pg_sink_head = 0; return 0; }
    if (cap_cols < 1 || head_cols < 1) return -2;
    if (!paged_lm_head_wide()) return -3;
    g_pg_sink = dev_out; g_pg_sink_cap = cap_cols; g_pg_sink_head = head_cols;
    return 0;
}

// --- milestone 3.2a: native ragged batched prefill (into paged slots) ---
// Chunked causal conv over ONE client's segment of n tokens, carrying the slot conv state.
// Bit-identical to the per-token k_tq_linear_conv_update loop (same shift+append+sum order).
__global__ void k_tq_linear_conv_chunk(float *out, float *state, const float *x,
                                       const uint16_t *conv_weight, int channels, int kernel_size, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= channels) return;
    float *s = state + (size_t)i * kernel_size;
    const uint16_t *wt = conv_weight + (size_t)i * kernel_size;
    float win[8];
    for (int j = 0; j < kernel_size; j++) win[j] = s[j];
    for (int t = 0; t < n; t++) {
        float sum = 0.0f;
        for (int j = 0; j < kernel_size - 1; j++) { win[j] = win[j + 1]; sum += win[j] * tq_bf16_to_float(wt[j]); }
        win[kernel_size - 1] = x[(size_t)t * channels + i];
        sum += win[kernel_size - 1] * tq_bf16_to_float(wt[kernel_size - 1]);
        out[(size_t)t * channels + i] = sum / (1.0f + expf(-sum));
    }
    for (int j = 0; j < kernel_size; j++) s[j] = win[j];
}

// Session 11 test ABI (analysis/gdn_ab.py): the paged prefill's DeltaNet path as served (k_tq_linear_conv_chunk +
// k_tq_deltanet_chunk_hs<8,128,2,8,DS> + k_tq_deltanet_norm) vs the gdn path (k_tq_gdn_prep + k_tq_gdn_chunk<KT,DS> +
// norm, TQ_GDN_PREFILL = mode) from identical synthetic inputs (pre-conv qkv, z, b, a, the slot's conv window, the
// recurrent state in format ds, positions pos0..pos0+N-1): bitwise comparison of core_out, of the recurrent state's
// bytes and of the conv window, then `iters` timed runs of each. mism[3] = differing elements (core, state bytes, conv
// window), ms[2] = ms per segment (old, new). Returns 0 on success.
extern "C" int qwn_gdn_prefill_ab(int layer, int N, int ds, int mode, int iters, int pos0, int *mism, float *ms) {
    if (!g_qwen.initialized) return -1;
    if (layer < 0 || layer >= g_qwen.L) return -2;
    if (g_qwen.layer_types[layer] != TQ_LAYER_LINEAR_ATTENTION) return -3;
    if (N <= 0 || N > 256 || ds < 0 || ds > 4 || mode < 1 || mode > 3) return -4;
    tq_layer_t *l = &g_qwen.layers[layer];
    int heads = g_qwen.linear_num_value_heads, key_heads = g_qwen.linear_num_key_heads, D = g_qwen.linear_value_head_dim;
    int KS = g_qwen.linear_conv_kernel_dim;
    if (D != 128) return -5;
    int key_dim = key_heads * D, value_dim = heads * D, conv_dim = 2 * key_dim + value_dim;
    size_t state_f = (size_t)heads * D * D, conv_f = (size_t)conv_dim * KS;
    size_t es = tq_ds_elem_bytes(ds), state_b = state_f * es;
    cudaStream_t st = g_qwen.stream;
    size_t nx = (size_t)N * conv_dim, nz = (size_t)N * value_dim, nba = (size_t)N * heads;
    float *d_xa = 0, *d_xb = 0, *d_z = 0, *d_b = 0, *d_a = 0, *d_s32 = 0, *d_ca = 0, *d_cb = 0, *d_coa = 0, *d_cob = 0;
    void *d_sa = 0, *d_sb = 0; int *d_pos = 0;
    int rc = 0;
    if (cudaMalloc(&d_xa, nx * 4) || cudaMalloc(&d_xb, nx * 4) || cudaMalloc(&d_z, nz * 4) || cudaMalloc(&d_b, nba * 4) ||
        cudaMalloc(&d_a, nba * 4) || cudaMalloc(&d_s32, state_f * 4) || cudaMalloc(&d_ca, conv_f * 4) || cudaMalloc(&d_cb, conv_f * 4) ||
        cudaMalloc(&d_coa, nz * 4) || cudaMalloc(&d_cob, nz * 4) || cudaMalloc(&d_sa, state_b) || cudaMalloc(&d_sb, state_b) ||
        cudaMalloc(&d_pos, (size_t)N * 4)) rc = -6;
    size_t hmax = nx > state_f ? nx : state_f;
    float *h = (float *)malloc(hmax * 4);
    int *hp = (int *)malloc((size_t)N * 4);
    unsigned char *ha = (unsigned char *)malloc(state_b), *hb = (unsigned char *)malloc(state_b);
    float *hca = (float *)malloc(conv_f * 4), *hcb = (float *)malloc(conv_f * 4);
    float *hoa = (float *)malloc(nz * 4), *hob = (float *)malloc(nz * 4);
    if (rc == 0) {
        deltanet_synth_fill(h, nx, 0); cudaMemcpyAsync(d_xa, h, nx * 4, cudaMemcpyHostToDevice, st); cudaMemcpyAsync(d_xb, h, nx * 4, cudaMemcpyHostToDevice, st);
        cudaStreamSynchronize(st);
        deltanet_synth_fill(h, nz, 1); cudaMemcpyAsync(d_z, h, nz * 4, cudaMemcpyHostToDevice, st); cudaStreamSynchronize(st);
        deltanet_synth_fill(h, nba, 2); cudaMemcpyAsync(d_b, h, nba * 4, cudaMemcpyHostToDevice, st); cudaStreamSynchronize(st);
        deltanet_synth_fill(h, nba, 3); cudaMemcpyAsync(d_a, h, nba * 4, cudaMemcpyHostToDevice, st); cudaStreamSynchronize(st);
        // a recurrent state of a realistic magnitude (|S| up to ~2, the layers' p99.99 is ~1): 100x the harness fill
        deltanet_synth_fill(h, state_f, 4); for (size_t i = 0; i < state_f; i++) h[i] *= 100.0f;
        cudaMemcpyAsync(d_s32, h, state_f * 4, cudaMemcpyHostToDevice, st); cudaStreamSynchronize(st);
        deltanet_synth_fill(h, conv_f, 0); cudaMemcpyAsync(d_ca, h, conv_f * 4, cudaMemcpyHostToDevice, st); cudaMemcpyAsync(d_cb, h, conv_f * 4, cudaMemcpyHostToDevice, st);
        for (int t = 0; t < N; t++) hp[t] = pos0 + t;
        cudaMemcpyAsync(d_pos, hp, (size_t)N * 4, cudaMemcpyHostToDevice, st);
        unsigned g = (unsigned)((state_f + 255) / 256);
        switch (ds) {
            case 1: k_tq_ds_from_f32<1><<<g, 256, 0, st>>>(d_sa, d_s32, state_f); break;
            case 2: k_tq_ds_from_f32<2><<<g, 256, 0, st>>>(d_sa, d_s32, state_f); break;
            case 3: k_tq_ds_from_f32<3><<<g, 256, 0, st>>>(d_sa, d_s32, state_f); break;
            case 4: k_tq_ds_from_f32<4><<<g, 256, 0, st>>>(d_sa, d_s32, state_f); break;
            default: cudaMemcpyAsync(d_sa, d_s32, state_f * 4, cudaMemcpyDeviceToDevice, st); break;
        }
        cudaMemcpyAsync(d_sb, d_sa, state_b, cudaMemcpyDeviceToDevice, st);
        cudaStreamSynchronize(st);
        // old path
        k_tq_linear_conv_chunk<<<(conv_dim + 255) / 256, 256, 0, st>>>(d_xa, d_ca, d_xa, l->d_linear_conv1d, conv_dim, KS, N);
        rc = launch_deltanet_chunk_hs_ds(ds, d_coa, d_sa, d_xa, d_z, d_b, d_a, l->d_linear_A_log, l->d_linear_dt_bias,
                                         l->d_linear_norm, N, heads, key_heads, D, g_qwen.eps, st, d_pos);
        // gdn path
        if (rc == 0)
            rc = launch_gdn_prefill(mode, ds, d_cob, d_sb, d_cb, d_xb, d_z, d_b, d_a, l->d_linear_A_log, l->d_linear_dt_bias,
                                    l->d_linear_conv1d, l->d_linear_norm, N, heads, key_heads, D, KS, g_qwen.eps, st, d_pos);
        if (cudaStreamSynchronize(st) != cudaSuccess || cudaGetLastError() != cudaSuccess) rc = rc ? rc : -7;
    }
    if (rc == 0) {
        cudaMemcpy(hoa, d_coa, nz * 4, cudaMemcpyDeviceToHost); cudaMemcpy(hob, d_cob, nz * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(ha, d_sa, state_b, cudaMemcpyDeviceToHost); cudaMemcpy(hb, d_sb, state_b, cudaMemcpyDeviceToHost);
        cudaMemcpy(hca, d_ca, conv_f * 4, cudaMemcpyDeviceToHost); cudaMemcpy(hcb, d_cb, conv_f * 4, cudaMemcpyDeviceToHost);
        int m0 = 0, m1 = 0, m2 = 0;
        for (size_t i = 0; i < nz; i++) m0 += memcmp(&hoa[i], &hob[i], 4) != 0;
        for (size_t i = 0; i < state_b; i++) m1 += ha[i] != hb[i];
        for (size_t i = 0; i < conv_f; i++) m2 += memcmp(&hca[i], &hcb[i], 4) != 0;
        if (m0 || m1) {   // first differing core element, for the log
            for (size_t i = 0; i < nz; i++) if (memcmp(&hoa[i], &hob[i], 4) != 0) {
                fprintf(stderr, "[gdn_ab] first core diff at tok %zu col %zu: old %.9g new %.9g\n", i / value_dim, i % value_dim, hoa[i], hob[i]); break; }
        }
        mism[0] = m0; mism[1] = m1; mism[2] = m2;
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0, st);
        for (int it = 0; it < iters; it++) {
            k_tq_linear_conv_chunk<<<(conv_dim + 255) / 256, 256, 0, st>>>(d_xa, d_ca, d_xa, l->d_linear_conv1d, conv_dim, KS, N);
            launch_deltanet_chunk_hs_ds(ds, d_coa, d_sa, d_xa, d_z, d_b, d_a, l->d_linear_A_log, l->d_linear_dt_bias,
                                        l->d_linear_norm, N, heads, key_heads, D, g_qwen.eps, st, d_pos);
        }
        cudaEventRecord(e1, st); cudaEventSynchronize(e1);
        float ms_a = 0; cudaEventElapsedTime(&ms_a, e0, e1);
        cudaEventRecord(e0, st);
        for (int it = 0; it < iters; it++)
            launch_gdn_prefill(mode, ds, d_cob, d_sb, d_cb, d_xb, d_z, d_b, d_a, l->d_linear_A_log, l->d_linear_dt_bias,
                               l->d_linear_conv1d, l->d_linear_norm, N, heads, key_heads, D, KS, g_qwen.eps, st, d_pos);
        cudaEventRecord(e1, st); cudaEventSynchronize(e1);
        float ms_b = 0; cudaEventElapsedTime(&ms_b, e0, e1);
        ms[0] = iters > 0 ? ms_a / iters : 0.0f; ms[1] = iters > 0 ? ms_b / iters : 0.0f;
        cudaEventDestroy(e0); cudaEventDestroy(e1);
        if (cudaGetLastError() != cudaSuccess) rc = -8;
    }
    free(h); free(hp); free(ha); free(hb); free(hca); free(hcb); free(hoa); free(hob);
    cudaFree(d_xa); cudaFree(d_xb); cudaFree(d_z); cudaFree(d_b); cudaFree(d_a); cudaFree(d_s32); cudaFree(d_ca); cudaFree(d_cb);
    cudaFree(d_coa); cudaFree(d_cob); cudaFree(d_sa); cudaFree(d_sb); cudaFree(d_pos);
    return rc;
}

static int ensure_pf_colslot(int T) {
    if (g_pf_colslot_cap >= T) return 0;
    graph_invalidate();
    if (g_pf_colslot) cudaFree(g_pf_colslot);
    if (cudaMalloc(&g_pf_colslot, (size_t)T * sizeof(int)) != cudaSuccess) { g_pf_colslot = NULL; g_pf_colslot_cap = 0; return -1; }
    g_pf_colslot_cap = T;
    return 0;
}

// Prefill K client segments in ONE wave: T concatenated columns (col j -> slot col_slot[j] at
// intra-position col_pos[j]). Projections + paged attention run batched over ALL T columns (one
// weight read); the conv + chunkwise DeltaNet run PER-CLIENT over each segment (advancing that
// slot's state; carries across waves for long prompts). seg_final[k]=1 -> compute that client's
// seed (greedy argmax at its last token) into out_seed[k]; else out_seed[k]=-1. The caller must
// have reset (or be continuing) each slot and allocated nothing -- blocks are ensured here.
static int run_paged_prefill_wave_core(const int *tokens, const int *col_slot, const int *col_pos,
                                       const int *seg_slot, const int *seg_off, const int *seg_len,
                                       const int *seg_final, int K, int T, int *out_seed) {
    int H = g_qwen.H, attn_m = g_qwen.nh * g_qwen.hd;
    int lkh = g_qwen.linear_num_key_heads, lvh = g_qwen.linear_num_value_heads, ld = g_qwen.linear_value_head_dim;
    int value_dim = lvh * ld, key_dim = lkh * ld, conv_dim = 2 * key_dim + value_dim;
    int conv_f = batched_conv_floats();
    int ret;
    if (ensure_wide_prefill_buffers(T) != 0) return -90;
    if (ensure_pf_colslot(T) != 0) return -90;
    int pf_maxpos = 0;
    for (int k = 0; k < K; k++) {
        int last_pos = col_pos[seg_off[k] + seg_len[k] - 1];
        if (paged_ensure_blocks(seg_slot[k], last_pos) != 0) return -91;     // pool full
        if (last_pos > pf_maxpos) pf_maxpos = last_pos;
    }
    paged_upload_tables(g_qwen.stream);
    if (g_pg_t4 && !g_t4_async) paged_t4_upload_wave_encl(seg_slot, seg_off, seg_len, col_pos, K, g_qwen.stream);
    cudaMemcpyAsync(g_pf_colslot, col_slot, (size_t)T * sizeof(int), cudaMemcpyHostToDevice, g_qwen.stream);
    cudaMemcpyAsync(g_wide_pos, col_pos, (size_t)T * sizeof(int), cudaMemcpyHostToDevice, g_qwen.stream);
    for (int j = 0; j < T; j++)
        k_tq_embed_lookup<<<(H + 255) / 256, 256, 0, g_qwen.stream>>>(
            g_wide_h + (size_t)j * H, g_qwen.d_embed, tokens[j], H);
    for (int L = 0; L < g_qwen.L; L++) {
        tq_layer_t *l = &g_qwen.layers[L];
        if (g_qwen.layer_types[L] == TQ_LAYER_FULL_ATTENTION) {
            if (!(tq_wide_fmt_ok(&l->q_proj) && tq_wide_fmt_ok(&l->o_proj) &&
                  tq_wide_fmt_ok(&l->mlp_gate) && tq_wide_fmt_ok(&l->mlp_down))) return -92;
            k_tq_qwen_rmsnorm_b<<<dim3(8, T), 1024, 0, g_qwen.stream>>>(g_wide_norm, g_wide_h, l->d_input_ln, H, g_qwen.eps);
            if ((ret = wide_quant_input(g_wide_norm, H, T)) != 0) return -93;
            if ((ret = wide_proj(&l->q_proj, g_wide_q, T)) != 0) return -93;
            if ((ret = wide_proj(&l->k_proj, g_wide_k, T)) != 0) return -93;
            if ((ret = wide_proj(&l->v_proj, g_wide_v, T)) != 0) return -93;
            if (wide_attn_mma_enabled() && g_pg_page >= 128) {
                if (launch_paged_attn_q4_mma_prefill(g_wide_core, g_wide_q, l->d_q_norm, g_wide_k, g_wide_v, l->d_k_norm,
                                     g_pool_k4[L], g_pool_kq4s[L], g_pool_v8[L], g_pool_vscale[L],
                                     g_wide_pos, g_pf_colslot, g_block_table, g_pg_maxblk, g_pg_page, g_pg_plog,
                                     seg_slot, seg_off, seg_len, col_pos, col_slot, L, K, T, g_qwen.stream,
                                     g_pg_t4 ? L : -1) != 0) return -94;
            } else if (launch_paged_attn_q4(g_wide_core, g_wide_q, l->d_q_norm, g_wide_k, g_wide_v, l->d_k_norm,
                                     g_pool_k4[L], g_pool_kq4s[L], g_pool_v8[L], g_pool_vscale[L],
                                     g_wide_pos, g_pf_colslot, g_block_table, g_pg_maxblk, g_pg_page, g_pg_plog,
                                     T, pf_maxpos, L, col_pos, col_slot, 0, g_qwen.stream, g_pg_t4 ? L : -1,
                                     g_pg_t4 ? seg_slot : NULL, seg_off, seg_len, K) != 0) return -94;
            if ((ret = wide_quant_input(g_wide_core, attn_m, T)) != 0) return -95;
            if ((ret = wide_proj(&l->o_proj, g_wide_o, T)) != 0) return -95;
            k_tq_add_vec<<<((size_t)T * H + 255) / 256, 256, 0, g_qwen.stream>>>(g_wide_resid, g_wide_h, g_wide_o, (size_t)T * H);
            if ((ret = wide_mlp(l, g_wide_resid, g_wide_h, T)) != 0) return -96;
        } else {
            if (!(tq_wide_fmt_ok(&l->linear_in_qkv) && tq_wide_fmt_ok(&l->linear_in_z) &&
                  tq_wide_fmt_ok(&l->linear_in_b) && tq_wide_fmt_ok(&l->linear_in_a) &&
                  tq_wide_fmt_ok(&l->linear_out) && tq_wide_fmt_ok(&l->mlp_gate) &&
                  tq_wide_fmt_ok(&l->mlp_down))) return -97;
            k_tq_qwen_rmsnorm_b<<<dim3(8, T), 1024, 0, g_qwen.stream>>>(g_wide_norm, g_wide_h, l->d_input_ln, H, g_qwen.eps);
            if ((ret = wide_quant_input(g_wide_norm, H, T)) != 0) return -98;
            if ((ret = wide_proj(&l->linear_in_qkv, g_wide_qkv, T)) != 0) return -98;
            if ((ret = wide_proj(&l->linear_in_z, g_wide_z, T)) != 0) return -98;
            if ((ret = wide_proj_pair(&l->linear_in_b, g_wide_b, &l->linear_in_a, g_wide_a, T)) != 0) return -98;
            for (int k = 0; k < K; k++) {     // per-client conv + chunkwise DeltaNet over the segment
                int off = seg_off[k], n = seg_len[k], slot = seg_slot[k];
                if (tq_gdn_prefill_mode() != 0) {   // session 11: hoisted prep + register-resident chunk kernel (bitwise)
                    int rg = launch_gdn_prefill(tq_gdn_prefill_mode(), g_pg_dstate, g_wide_lcore + (size_t)off * value_dim,
                            pg_recur_slot(L, slot), g_pg_conv[L] + (size_t)slot * conv_f, g_wide_qkv + (size_t)off * conv_dim,
                            g_wide_z + (size_t)off * value_dim, g_wide_b + (size_t)off * lvh, g_wide_a + (size_t)off * lvh,
                            l->d_linear_A_log, l->d_linear_dt_bias, l->d_linear_conv1d, l->d_linear_norm, n, lvh, lkh, ld,
                            g_qwen.linear_conv_kernel_dim, g_qwen.eps, g_qwen.stream, g_wide_pos + off);
                    if (rg == 0) continue;
                    if (rg != -100) return -99;     // -100 = a geometry the gdn kernels do not cover: the kernels below
                }
                k_tq_linear_conv_chunk<<<(conv_dim + 255) / 256, 256, 0, g_qwen.stream>>>(
                    g_wide_qkv + (size_t)off * conv_dim, g_pg_conv[L] + (size_t)slot * conv_f,
                    g_wide_qkv + (size_t)off * conv_dim, l->d_linear_conv1d, conv_dim, g_qwen.linear_conv_kernel_dim, n);
                if (launch_deltanet_chunk_hs_ds(g_pg_dstate, g_wide_lcore + (size_t)off * value_dim,
                        pg_recur_slot(L, slot), g_wide_qkv + (size_t)off * conv_dim,
                        g_wide_z + (size_t)off * value_dim, g_wide_b + (size_t)off * lvh, g_wide_a + (size_t)off * lvh,
                        l->d_linear_A_log, l->d_linear_dt_bias, l->d_linear_norm, n, lvh, lkh, ld,
                        g_qwen.eps, g_qwen.stream, g_wide_pos + off) != 0) return -99;
            }
            if ((ret = wide_quant_input(g_wide_lcore, value_dim, T)) != 0) return -100;
            if ((ret = wide_proj(&l->linear_out, g_wide_o, T)) != 0) return -100;
            k_tq_add_vec<<<((size_t)T * H + 255) / 256, 256, 0, g_qwen.stream>>>(g_wide_resid, g_wide_h, g_wide_o, (size_t)T * H);
            if ((ret = wide_mlp(l, g_wide_resid, g_wide_h, T)) != 0) return -101;
        }
    }
    if (g_pg_sink) {     // every column's logits; the head in decode-size chunks (f32 accumulators)
        if (T > g_pg_sink_cap) return -106;
        k_tq_qwen_rmsnorm_b<<<dim3(8, T), 1024, 0, g_qwen.stream>>>(g_wide_norm, g_wide_h, g_qwen.d_norm, H, g_qwen.eps);
        for (int c0 = 0; c0 < T; c0 += g_pg_sink_head) {
            int nc = T - c0 < g_pg_sink_head ? T - c0 : g_pg_sink_head;
            if ((ret = wide_quant_input(g_wide_norm + (size_t)c0 * H, H, nc)) != 0) return -107;
            if ((ret = wide_proj(&g_qwen.lm_head, g_pg_sink + (size_t)c0 * g_qwen.V, nc)) != 0) return -107;
        }
    }
    // seeds: gather each final client's last-token hidden -> final norm -> wide lm_head -> argmax.
    int Kf = 0;
    int fk[256];
    if (K > 256) return -105;
    for (int k = 0; k < K; k++) {
        out_seed[k] = -1;
        if (!seg_final[k]) continue;
        int last_col = seg_off[k] + seg_len[k] - 1;
        cudaMemcpyAsync(g_wide_resid + (size_t)Kf * H, g_wide_h + (size_t)last_col * H,
                        (size_t)H * sizeof(float), cudaMemcpyDeviceToDevice, g_qwen.stream);
        fk[Kf++] = k;
    }
    if (Kf > 0) {
        k_tq_qwen_rmsnorm_b<<<dim3(8, Kf), 1024, 0, g_qwen.stream>>>(g_wide_norm, g_wide_resid, g_qwen.d_norm, H, g_qwen.eps);
        if (paged_lm_head_wide()) {
            if ((ret = wide_quant_input(g_wide_norm, H, Kf)) != 0) return -102;
            if ((ret = wide_proj(&g_qwen.lm_head, g_pg_logits, Kf)) != 0) return -102;
            k_tq_batched_argmax<<<Kf, 256, 0, g_qwen.stream>>>(g_pg_argmax, g_pg_logits, g_qwen.V, Kf);
        } else {
            for (int f = 0; f < Kf; f++) {
                cudaMemcpyAsync(g_qwen.d_debug_norm, g_wide_norm + (size_t)f * H, (size_t)H * sizeof(float),
                                cudaMemcpyDeviceToDevice, g_qwen.stream);
                if (run_lm_head_argmax_device() != 0) return -103;
                cudaMemcpyAsync(g_pg_argmax + f, g_qwen.d_debug_arg_ids + TQ_ARGMAX_BLOCKS, sizeof(int),
                                cudaMemcpyDeviceToDevice, g_qwen.stream);
            }
        }
        int htmp[256];
        cudaMemcpyAsync(htmp, g_pg_argmax, (size_t)Kf * sizeof(int), cudaMemcpyDeviceToHost, g_qwen.stream);
        cudaStreamSynchronize(g_qwen.stream);
        for (int f = 0; f < Kf; f++) out_seed[fk[f]] = htmp[f];
    }
    if (g_mtp_on && mtp_hook_prefill(tokens, col_pos, seg_slot, seg_off, seg_len, K, T) != 0) return -108;
    if (g_pg_t4) {                                           // the wave's columns are committed (encoded, or being encoded)
        paged_t4_wave_done(seg_slot, seg_off, seg_len, col_pos, K);
        if (g_t4_async) cudaEventRecord(g_t4_done, g_t4_stream);
    }
    return cudaStreamSynchronize(g_qwen.stream) == cudaSuccess ? 0 : -104;
}

// ABI: prefill K client segments in one ragged wave. tokens/col_slot/col_pos are T concatenated
// columns; seg_* are per-client (K). Returns 0; out_seed[k] = seed for final segments, -1 else.
extern "C" int qwn_paged_prefill_batch(const int *tokens, const int *col_slot, const int *col_pos,
                                       const int *seg_slot, const int *seg_off, const int *seg_len,
                                       const int *seg_final, int K, int T, int *out_seed) {
    if (!g_pg_ready) return -1;
    if (!g_qwen.kv_q4) return -2;
    if (K < 1 || T < 1) return -3;
    // Wave cap: 128 columns = the FP6 path's fast activation quantizer smem limit (the EXL3 path builds its fragments
    // per Linear and has no such limit). TQ_WAVE_MAX (session 10, prefill thread; <= 256) lifts it for the A/B of
    // 256-column waves; the trellis pool keeps one in-flight staging page per slot, so it stays at <= page there.
    static int wave_max = -1;
    if (wave_max < 0) {
        const char *e = getenv("TQ_WAVE_MAX");
        wave_max = (e && *e) ? atoi(e) : 128;
        if (wave_max < 1) wave_max = 1;
        if (wave_max > 256) wave_max = 256;
        if (!(g_qwen.flags & TQ_FLAG_EXL3_WEIGHTS) && wave_max > 128) wave_max = 128;
    }
    if (T > wave_max) return -6;
    if (g_pg_t4 && T > g_pg_page) return -6;
    for (int k = 0; k < K; k++) {
        if (seg_slot[k] < 0 || seg_slot[k] >= g_pg_maxslots) return -4;
        if (seg_len[k] < 1) return -5;
    }
    return run_paged_prefill_wave_core(tokens, col_slot, col_pos, seg_slot, seg_off, seg_len, seg_final, K, T, out_seed);
}
