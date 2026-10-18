// the paged pool's init / free, the trellis pool's host side (column encoder, bookkeeping, the built-in codebook) and B3 page top-k.
// Part of the forward_qwen.cu unity build: included in order, not a standalone header.

// ===== TQ_KV_T4 host side: encoded length per slot, column encoder, bookkeeping =====
// Positions < h_slot_encl[slot] of a slot are trellis codes in its pool blocks; the columns of the wave / step in
// flight are FP4 + fp16 in staging page 2*slot + (lb & 1) until paged_t4_encode_ranges moves them into the pool
// (prefill: per layer before the layer's attention -> every key the attention sees is trellis, exactly the
// simulation; decode / verify: after the step / at commit -> the step's own columns are attended as FP4 once).

// Upload the block table (always the raw pool blocks) and the per-slot encoded lengths for the next kernels.
static void paged_upload_tables(cudaStream_t st) {
    paged_t4_sync(st);                                        // pending side-stream encodes of the last prefill wave
    size_t n = (size_t)g_pg_maxslots * g_pg_maxblk;
    cudaMemcpyAsync(g_block_table, h_block_table, n * sizeof(int), cudaMemcpyHostToDevice, st);
    if (g_pg_t4)
        cudaMemcpyAsync(g_t4_encl_d, h_slot_encl, (size_t)g_pg_maxslots * sizeof(int), cudaMemcpyHostToDevice, st);
}

// A batch of position ranges (staging rows -> pool rows), one launch set for every full-attention layer
// (only_layer < 0) or one layer, K and V. Asynchronous on g_qwen.stream.
struct tq_t4_ranges { unsigned src[TQ_T4_MAXR], dst[TQ_T4_MAXR]; int n[TQ_T4_MAXR]; int nr; };
// The encode job of `rg` for one full-attention layer (only_layer >= 0) or all of them; returns the tile count (< 0 = error).
static int paged_t4_build_job(const tq_t4_ranges &rg, int only_layer, tq_t4_job *jb) {
    memset(jb, 0, sizeof(*jb));
    int nf = 0;
    for (int L = 0; L < g_qwen.L; L++) {
        if (g_qwen.layer_types[L] != TQ_LAYER_FULL_ATTENTION) continue;
        if (only_layer >= 0 && L != only_layer) continue;
        if (nf >= TQ_T4_MAXL) return -3;
        jb->src[nf] = g_t4_stg_f16[L];
        jb->dk[nf] = g_pool_k4[L]; jb->dv[nf] = g_pool_v8[L];
        jb->sk[nf] = g_t4_kside[L]; jb->sv[nf] = g_t4_vside[L];
        nf++;
    }
    int tot = 0;
    for (int r = 0; r < rg.nr; r++) { jb->src_row0[r] = rg.src[r]; jb->dst_row0[r] = rg.dst[r]; jb->pre[r] = tot; tot += rg.n[r]; }
    jb->pre[rg.nr] = tot;
    jb->nr = rg.nr; jb->npos = tot; jb->nkv = g_qwen.nkv; jb->nfull = nf;
    return nf * tot * g_qwen.nkv * 2;
}
// TQ_GRAPH verify wave: the wave's per-layer encode jobs are built by the prework into the pinned h_t4_jobs
// (one per full-attention layer, ordinal order) and uploaded to g_t4_jobs_d before the replay; while
// g_t4_job_mode is set, paged_t4_encode_wave_layer launches k_t4_encode_dev on them (g_t4_job_tiles tiles).
static tq_t4_job *h_t4_jobs = NULL, *g_t4_jobs_d = NULL;
static int g_t4_job_mode = 0, g_t4_job_tiles = 0;
static int paged_t4_ordinal(int L) {
    int o = 0;
    for (int i = 0; i < L; i++) if (g_qwen.layer_types[i] == TQ_LAYER_FULL_ATTENTION) o++;
    return o;
}
static int paged_t4_encode_ranges(const tq_t4_ranges &rg, int only_layer, cudaStream_t st = NULL) {
    if (!st) st = g_qwen.stream;
    if (!g_pg_t4 || !g_t4_cb) return -1;
    if (rg.nr <= 0) return 0;
    tq_t4_job jb;
    const int ntiles = paged_t4_build_job(rg, only_layer, &jb);
    if (ntiles < 0) return ntiles;
    for (int t0 = 0; t0 < ntiles; t0 += TQ_T4_CHUNK) {
        int n = ntiles - t0 < TQ_T4_CHUNK ? ntiles - t0 : TQ_T4_CHUNK;
        jb.tile0 = t0;
        k_t4_encode<<<n, TQ_T4_NT, 0, st>>>(jb, g_t4_cb);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : -4;
}
// Append positions [p0, p1) of `slot` to the batch (split at page boundaries); flushes when full.
static int paged_t4_add_cols(tq_t4_ranges &rg, int slot, int p0, int p1, int only_layer, cudaStream_t st = NULL) {
    while (p0 < p1) {
        int lb = p0 >> g_pg_plog, pend = (lb + 1) << g_pg_plog;
        int n = (p1 < pend ? p1 : pend) - p0;
        int phys = h_block_table[(size_t)slot * g_pg_maxblk + lb];
        if (phys < 0) return -2;
        if (rg.nr == TQ_T4_MAXR) { int rc = paged_t4_encode_ranges(rg, only_layer, st); if (rc != 0) return rc; rg.nr = 0; }
        rg.src[rg.nr] = (unsigned)((size_t)slot * g_pg_page + (p0 & (g_pg_page - 1)));
        rg.dst[rg.nr] = (unsigned)((size_t)phys * g_pg_page + (p0 & (g_pg_page - 1)));
        rg.n[rg.nr] = n;
        rg.nr++;
        p0 += n;
    }
    return 0;
}
// Prefill waves: before layer L's attention, move the wave's columns (positions encl.. last) of every segment
// into the pool for that layer, so every key the attention reads is trellis (the simulation's semantics).
static int paged_t4_encode_wave_layer(const int *seg_slot, const int *seg_off, const int *seg_len, const int *col_pos,
                                      int K, int L, cudaStream_t st) {
    if (g_t4_job_mode) {                                  // graph body: this layer's job was prepared by the prework
        if (!g_t4_jobs_d || g_t4_job_tiles <= 0) return -5;
        k_t4_encode_dev<<<g_t4_job_tiles, TQ_T4_NT, 0, st>>>(g_t4_jobs_d + paged_t4_ordinal(L), g_t4_cb);
        return cudaGetLastError() == cudaSuccess ? 0 : -4;
    }
    tq_t4_ranges rg; rg.nr = 0;
    for (int k = 0; k < K; k++) {
        int slot = seg_slot[k], first = col_pos[seg_off[k]], last = col_pos[seg_off[k] + seg_len[k] - 1];
        int from = first < h_slot_encl[slot] ? h_slot_encl[slot] : first;
        int rc = paged_t4_add_cols(rg, slot, from, last + 1, L, st);
        if (rc != 0) return rc;
    }
    return paged_t4_encode_ranges(rg, L, st);
}
// Graph prework of a verify wave: the wave's ranges (as paged_t4_encode_wave_layer builds them, without the flush
// path) -> one job per full-attention layer into h_t4_jobs, uploaded to g_t4_jobs_d on `st`. Returns the tiles per
// layer launch (0 = nothing to encode), < 0 = not representable (the caller stays eager).
static int paged_t4_prep_wave_jobs(const int *seg_slot, const int *seg_off, const int *seg_len, const int *col_pos,
                                   int K, cudaStream_t st) {
    if (!h_t4_jobs || !g_t4_jobs_d) return -1;
    tq_t4_ranges rg; rg.nr = 0;
    for (int k = 0; k < K; k++) {
        int slot = seg_slot[k], first = col_pos[seg_off[k]], last = col_pos[seg_off[k] + seg_len[k] - 1];
        int p0 = first < h_slot_encl[slot] ? h_slot_encl[slot] : first, p1 = last + 1;
        while (p0 < p1) {
            int lb = p0 >> g_pg_plog, pend = (lb + 1) << g_pg_plog;
            int n = (p1 < pend ? p1 : pend) - p0;
            int phys = h_block_table[(size_t)slot * g_pg_maxblk + lb];
            if (phys < 0) return -2;
            if (rg.nr == TQ_T4_MAXR) return -3;
            rg.src[rg.nr] = (unsigned)((size_t)slot * g_pg_page + (p0 & (g_pg_page - 1)));
            rg.dst[rg.nr] = (unsigned)((size_t)phys * g_pg_page + (p0 & (g_pg_page - 1)));
            rg.n[rg.nr] = n;
            rg.nr++;
            p0 += n;
        }
    }
    if (rg.nr == 0) return 0;
    int tiles = -1, nf = 0;
    for (int L = 0; L < g_qwen.L; L++) {
        if (g_qwen.layer_types[L] != TQ_LAYER_FULL_ATTENTION) continue;
        if (nf >= TQ_T4_MAXL) return -4;
        int t = paged_t4_build_job(rg, L, &h_t4_jobs[nf++]);
        if (t < 0) return t;
        if (tiles >= 0 && t != tiles) return -5;
        tiles = t;
    }
    if (tiles > TQ_T4_CHUNK) return -6;                   // one launch per layer (a wave has <= 64 x nkv x 2 tiles)
    cudaMemcpyAsync(g_t4_jobs_d, h_t4_jobs, (size_t)nf * sizeof(tq_t4_job), cudaMemcpyHostToDevice, st);
    return tiles;
}
// The wave's columns are committed as written: advance encl (their trellis codes were produced per layer).
static void paged_t4_wave_done(const int *seg_slot, const int *seg_off, const int *seg_len, const int *col_pos, int K) {
    for (int k = 0; k < K; k++) {
        int slot = seg_slot[k], len = col_pos[seg_off[k] + seg_len[k] - 1] + 1;
        if (len > h_slot_encl[slot]) h_slot_encl[slot] = len;
    }
}
// Device view for the wave's attention: the segments' columns count as encoded (temporary encl = last + 1).
static void paged_t4_upload_wave_encl(const int *seg_slot, const int *seg_off, const int *seg_len, const int *col_pos,
                                      int K, cudaStream_t st) {
    memcpy(h_t4_encl_tmp, h_slot_encl, (size_t)g_pg_maxslots * sizeof(int));
    for (int k = 0; k < K; k++) {
        int slot = seg_slot[k], len = col_pos[seg_off[k] + seg_len[k] - 1] + 1;
        h_t4_encl_tmp[slot] = len > h_slot_encl[slot] ? len : h_slot_encl[slot];
    }
    cudaMemcpyAsync(g_t4_encl_d, h_t4_encl_tmp, (size_t)g_pg_maxslots * sizeof(int), cudaMemcpyHostToDevice, st);
}

// Full-page encode from staging page `stg` into pool block `phys` (the test ABI below).
static int paged_t4_encode_into(int stg, int phys, int only_layer) {
    if (phys < 0) return -2;
    tq_t4_ranges rg; rg.nr = 1;
    rg.src[0] = (unsigned)((size_t)stg * g_pg_page); rg.dst[0] = (unsigned)((size_t)phys * g_pg_page); rg.n[0] = g_pg_page;
    return paged_t4_encode_ranges(rg, only_layer);
}

// Test ABI (analysis/kv_trellis_cuda, verification ladder 4a): n <= page*nkv*2 rows of fp16 rotated
// values go through the engine's staging -> encoder -> pool path (staging page 0 of slot 0, the pool's
// last block, full-attention layer 0); returns the 129-byte code streams [n][129] (prefix byte, 128
// symbol bytes) and the fp16 scale bits [n]. Destroys the contents of those staging / pool rows.
extern "C" int qwn_paged_t4_test_encode(const uint16_t *f16_rows, int n, uint8_t *codes_out, uint16_t *scale_out) {
    if (!g_pg_ready || !g_pg_t4) return -1;
    int L0 = -1;
    for (int L = 0; L < g_qwen.L; L++) if (g_qwen.layer_types[L] == TQ_LAYER_FULL_ATTENTION) { L0 = L; break; }
    if (L0 < 0) return -2;
    const int nkv = g_qwen.nkv, cap = g_pg_page * nkv * 2;
    if (n < 1 || n > cap) return -3;
    const int phys = g_pg_nblocks - 1;
    // staging rows [t][h][kv][256] are contiguous in tile order (t * nkv + h) * 2 + kv
    cudaMemcpyAsync(g_t4_stg_f16[L0], f16_rows, (size_t)n * 256 * sizeof(uint16_t), cudaMemcpyHostToDevice, g_qwen.stream);
    if (n < cap) cudaMemsetAsync(g_t4_stg_f16[L0] + (size_t)n * 256, 0, (size_t)(cap - n) * 256 * sizeof(uint16_t), g_qwen.stream);
    int rc = paged_t4_encode_into(0, phys, L0);
    if (rc != 0) return -4;
    if (cudaStreamSynchronize(g_qwen.stream) != cudaSuccess) return -5;
    uint8_t *kc = (uint8_t *)malloc((size_t)g_pg_page * nkv * 128), *vc = (uint8_t *)malloc((size_t)g_pg_page * nkv * 128);
    uint32_t *ks = (uint32_t *)malloc((size_t)g_pg_page * nkv * 4), *vs = (uint32_t *)malloc((size_t)g_pg_page * nkv * 4);
    size_t bo = (size_t)phys * g_pg_page;
    cudaMemcpy(kc, g_pool_k4[L0] + bo * nkv * 128, (size_t)g_pg_page * nkv * 128, cudaMemcpyDeviceToHost);
    cudaMemcpy(vc, g_pool_v8[L0] + bo * nkv * 128, (size_t)g_pg_page * nkv * 128, cudaMemcpyDeviceToHost);
    cudaMemcpy(ks, g_t4_kside[L0] + bo * nkv * 4, (size_t)g_pg_page * nkv * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(vs, g_t4_vside[L0] + bo * nkv * 4, (size_t)g_pg_page * nkv * 4, cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) {
        int vec = i >> 1, kv = i & 1;                    // vec = t * nkv + h
        const uint8_t *c = (kv ? vc : kc) + (size_t)vec * 128;
        uint32_t sw = (kv ? vs : ks)[vec];
        codes_out[(size_t)i * 129] = (uint8_t)(sw & 0xFFu);
        memcpy(codes_out + (size_t)i * 129 + 1, c, 128);
        scale_out[i] = (uint16_t)((sw >> 8) & 0xFFFFu);
    }
    free(kc); free(vc); free(ks); free(vs);
    return cudaGetLastError() == cudaSuccess ? 0 : -6;
}

static void paged_t4_free(void) {
    for (int L = 0; L < TQ_MAX_LAYERS; L++) {
        if (g_t4_kside[L]) { cudaFree(g_t4_kside[L]); g_t4_kside[L] = NULL; }
        if (g_t4_vside[L]) { cudaFree(g_t4_vside[L]); g_t4_vside[L] = NULL; }
        if (g_t4_stg_f16[L]) { cudaFree(g_t4_stg_f16[L]); g_t4_stg_f16[L] = NULL; }
    }
    if (g_t4_cb) { cudaFree(g_t4_cb); g_t4_cb = NULL; }
    if (g_t4_tab) { cudaFree(g_t4_tab); g_t4_tab = NULL; }
    if (g_t4_encl_d) { cudaFree(g_t4_encl_d); g_t4_encl_d = NULL; }
    if (g_t4_jobs_d) { cudaFree(g_t4_jobs_d); g_t4_jobs_d = NULL; }
    if (h_t4_jobs) { cudaFreeHost(h_t4_jobs); h_t4_jobs = NULL; }
    g_t4_job_mode = 0; g_t4_job_tiles = 0;
    if (g_t4_stream) { cudaStreamSynchronize(g_t4_stream); cudaStreamDestroy(g_t4_stream); g_t4_stream = NULL; }
    if (g_t4_ev) { cudaEventDestroy(g_t4_ev); g_t4_ev = NULL; }
    if (g_t4_done) { cudaEventDestroy(g_t4_done); g_t4_done = NULL; }
    if (h_slot_encl) { free(h_slot_encl); h_slot_encl = NULL; }
    if (h_t4_encl_tmp) { free(h_t4_encl_tmp); h_t4_encl_tmp = NULL; }
    g_t4_stg_rows = 0; g_pg_t4 = 0;
}

#include "kv_t4_codebook.inc"

// Codebook: 4096 fp32 in the analysis-side order -- the built-in Lloyd-trained lloyd_kv12a (src/engine/kv_t4_codebook.inc), or
// TQ_KV_T4_CODEBOOK = path of 4096 raw little-endian fp32. The fp16 dequant table permutes the nibble
// order (table[e] = cb[swap nibbles 0 and 2 of e]).
static int paged_t4_load_codebook(void) {
    const char *path = getenv("TQ_KV_T4_CODEBOOK");
    float *cb = (float *)malloc(4096 * sizeof(float));
    if (!path || !*path) {
        memcpy(cb, k_t4_codebook_lloyd_kv12a, sizeof(k_t4_codebook_lloyd_kv12a));
        path = "built-in lloyd_kv12a";
    } else {
        FILE *f = fopen(path, "rb");
        if (!f) { free(cb); fprintf(stderr, "[paged] TQ_KV_T4: cannot open codebook %s\n", path); return -1; }
        size_t got = fread(cb, sizeof(float), 4096, f);
        fclose(f);
        if (got != 4096) { free(cb); fprintf(stderr, "[paged] TQ_KV_T4: codebook %s has %zu floats, want 4096\n", path, got); return -2; }
    }
    uint16_t *tab = (uint16_t *)malloc(4096 * sizeof(uint16_t));
    for (int e = 0; e < 4096; e++) {
        int s = ((e & 15) << 8) | (e & 0xF0) | (e >> 8);
        tab[e] = __half_as_ushort(__float2half(cb[s]));
    }
    cudaMemcpyAsync(g_t4_cb, cb, 4096 * sizeof(float), cudaMemcpyHostToDevice, g_qwen.stream);
    cudaMemcpyAsync(g_t4_tab, tab, 4096 * sizeof(uint16_t), cudaMemcpyHostToDevice, g_qwen.stream);
    cudaStreamSynchronize(g_qwen.stream);
    free(cb); free(tab);
    fprintf(stderr, "[paged] TQ_KV_T4: trellis-coded K+V (L=12, 4 bit, codebook %s); in-flight columns exact (fp16 staging), prefill encodes %s\n",
            path, g_t4_async ? "on a side stream" : "synchronous per layer");
    return 0;
}

// Allocations of the trellis mode (called by qwn_paged_init after the pool's code arrays exist).
static int paged_t4_alloc(int max_slots, int num_blocks, int page) {
    int kv_m = g_qwen.nkv * g_qwen.hd, nkv = g_qwen.nkv;
    size_t qe = batched_q4s_elem_bytes();
    size_t bp = (size_t)num_blocks * page;
    // staging: one page of fp16 rotated K | V per slot (a wave's <= 128 consecutive columns map injectively onto
    // pos & (page - 1); a verify wave has <= 64, a decode step 1)
    g_t4_stg_rows = max_slots * page;
    size_t sr = (size_t)g_t4_stg_rows;
    (void)kv_m; (void)qe;
    for (int L = 0; L < g_qwen.L; L++) {
        if (g_qwen.layer_types[L] != TQ_LAYER_FULL_ATTENTION) continue;
        size_t sideb = bp * nkv * 4, f16b = sr * nkv * 512 * 2;
        if (cudaMalloc(&g_t4_kside[L], sideb) != cudaSuccess || cudaMalloc(&g_t4_vside[L], sideb) != cudaSuccess ||
            cudaMalloc(&g_t4_stg_f16[L], f16b) != cudaSuccess) return -1;
        cudaMemsetAsync(g_t4_kside[L], 0, sideb, g_qwen.stream); cudaMemsetAsync(g_t4_vside[L], 0, sideb, g_qwen.stream);
        cudaMemsetAsync(g_t4_stg_f16[L], 0, f16b, g_qwen.stream);
    }
    if (cudaMalloc(&g_t4_cb, 4096 * sizeof(float)) != cudaSuccess || cudaMalloc(&g_t4_tab, 4096 * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc(&g_t4_encl_d, (size_t)max_slots * sizeof(int)) != cudaSuccess) return -2;
    cudaMemsetAsync(g_t4_encl_d, 0, (size_t)max_slots * sizeof(int), g_qwen.stream);
    if (cudaMalloc(&g_t4_jobs_d, (size_t)TQ_T4_MAXL * sizeof(tq_t4_job)) != cudaSuccess ||
        cudaHostAlloc(&h_t4_jobs, (size_t)TQ_T4_MAXL * sizeof(tq_t4_job), cudaHostAllocDefault) != cudaSuccess) {
        h_t4_jobs = NULL; return -2;
    }
    { const char *e = getenv("TQ_KV_T4_ASYNC"); g_t4_async = (e && *e) ? (atoi(e) != 0) : 1; }
    int prio_lo = 0, prio_hi = 0;                           // the side stream gets the highest priority: its CTAs
    cudaDeviceGetStreamPriorityRange(&prio_lo, &prio_hi);   // are scheduled ahead of the attention's and are done
    if (cudaStreamCreateWithPriority(&g_t4_stream, cudaStreamNonBlocking, prio_hi) != cudaSuccess ||   // before the GEMMs
        cudaEventCreateWithFlags(&g_t4_ev, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&g_t4_done, cudaEventDisableTiming) != cudaSuccess) return -4;
    cudaEventRecord(g_t4_done, g_qwen.stream);
    h_slot_encl = (int *)calloc((size_t)max_slots, sizeof(int));
    h_t4_encl_tmp = (int *)calloc((size_t)max_slots, sizeof(int));
    if (!h_slot_encl || !h_t4_encl_tmp) return -3;
    return paged_t4_load_codebook();
}

// Bytes of one (token, kv head) K + V pair: pool rows (trellis) / staging rows (FP4 + fp16)
static size_t paged_t4_pool_vec_bytes(void) { return 2 * (128 + 4); }

// ===== B3 page top-k (Quest) =====
// Per physical block, per KV head: element-wise min and max of the dequantized rotated K
// (the values the GQA kernel actually dots). fp32 so the bound stays valid; ~8 KB/block
// (~4% of a 128-token page). Refreshed from stored int4 K after every KV write of a page,
// and rebuilt from K on load/restore (not part of the parking blob — a pure function of K).
// TQ_PAGED_TOPK_PAGES=0 (default) leaves the dense path bit-for-bit, no extra allocs.
#define TQ_KMM_SCRATCH 4096
static int h_kmm_slot[TQ_KMM_SCRATCH], h_kmm_pos[TQ_KMM_SCRATCH];

static int paged_topk_pages(void) {
    static int c = -1;
    if (c < 0) {
        const char *e = getenv("TQ_PAGED_TOPK_PAGES");
        c = (e && *e) ? atoi(e) : 0;
        if (c < 0) c = 0;
        if (c > 4096) c = 4096;
    }
    return c;
}
static int paged_topk_min_ctx(void) {
    static int c = -1;
    if (c < 0) {
        const char *e = getenv("TQ_PAGED_TOPK_MIN_CTX");
        c = (e && *e) ? atoi(e) : 16384;
        if (c < 0) c = 0;
    }
    return c;
}

static int paged_kmm_ensure(void) {
    if (g_kmm_ready) return 0;
    if (!g_pg_ready) return -1;
    int nkv = g_qwen.nkv, hd = g_qwen.hd;
    size_t bytes = (size_t)g_pg_nblocks * nkv * 2 * hd * sizeof(float);
    int nfull = 0;
    for (int L = 0; L < g_qwen.L; L++) {
        if (g_qwen.layer_types[L] != TQ_LAYER_FULL_ATTENTION) continue;
        if (!g_pool_kmm[L]) {
            if (cudaMalloc(&g_pool_kmm[L], bytes) != cudaSuccess) {
                g_pool_kmm[L] = NULL;
                return -2;
            }
            cudaMemsetAsync(g_pool_kmm[L], 0, bytes, g_qwen.stream);
        }
        nfull++;
    }
    if (!g_kmm_slot_d && cudaMalloc(&g_kmm_slot_d, (size_t)TQ_KMM_SCRATCH * sizeof(int)) != cudaSuccess) return -3;
    if (!g_kmm_pos_d && cudaMalloc(&g_kmm_pos_d, (size_t)TQ_KMM_SCRATCH * sizeof(int)) != cudaSuccess) return -3;
    g_kmm_ready = 1;
    fprintf(stderr, "[paged-topk] K min/max %.1f MB (%d layers)\n",
            (double)nfull * (double)bytes / (1024.0 * 1024.0), nfull);
    return 0;
}

static int paged_topk_bufs(int k) {
    if (g_topk_k_alloc >= k && g_topk_list && g_topk_count) return 0;
    int cap = k + 3;
    if (g_topk_list) { cudaFree(g_topk_list); g_topk_list = NULL; }
    if (g_topk_count) { cudaFree(g_topk_count); g_topk_count = NULL; }
    size_t rows = (size_t)g_pg_maxslots * g_qwen.nkv;
    if (cudaMalloc(&g_topk_list, rows * (size_t)cap * 2 * sizeof(int)) != cudaSuccess) return -1;
    if (cudaMalloc(&g_topk_count, rows * sizeof(int)) != cudaSuccess) return -1;
    g_topk_k_alloc = k;
    g_topk_cap = cap;
    return 0;
}

// Recompute min/max for each (slot, page) in the list. `pos` is the last valid
// logical position in that page (the scan covers rows [0 .. pos & (page-1)]).
__global__ void k_tq_paged_kmm_pages(
    float *kmm, const uint8_t *k4, const float *kq4s,
    const int *slots, const int *maxpos, int npages,
    const int *block_table, int max_blocks, int page, int page_log,
    int nkv, int hd) {
    int kv_head = blockIdx.x;
    int pi = blockIdx.y;
    int tid = threadIdx.x;
    if (pi >= npages || kv_head >= nkv || tid >= hd) return;
    int slot = slots[pi];
    int pos = maxpos[pi];
    int lb = pos >> page_log;
    int phys = block_table[(size_t)slot * max_blocks + lb];
    if (phys < 0) return;
    int valid = (pos & (page - 1)) + 1;
    if (valid > page) valid = page;
    size_t kv_m = (size_t)nkv * hd;
    float mn = 3.402823466e38f, mx = -3.402823466e38f;
    for (int r = 0; r < valid; r++) {
        size_t pr = (size_t)phys * page + r;
        const uint8_t *krow = k4 + pr * (kv_m >> 1) + (size_t)kv_head * (hd >> 1);
        const float *ksz = q4s_adv(kq4s, pr * ((size_t)nkv * 16) + (size_t)kv_head * 16);
        int g = tid >> 5;
        uint8_t byte = krow[tid >> 1];
        float code = (float)((tid & 1) ? (byte >> 4) : (byte & 15));
        float v = (code - q4s_at(ksz, (size_t)(2 * g + 1))) * q4s_at(ksz, (size_t)(2 * g));
        mn = fminf(mn, v);
        mx = fmaxf(mx, v);
    }
    float *dst = kmm + (((size_t)phys * nkv + kv_head) * 2) * hd;
    dst[tid] = mn;
    dst[hd + tid] = mx;
}

static int paged_kmm_launch(int layer, int nu, cudaStream_t st) {
    if (nu < 1) return 0;
    k_tq_paged_kmm_pages<<<dim3(g_qwen.nkv, nu), g_qwen.hd, 0, st>>>(
        g_pool_kmm[layer], g_pool_k4[layer], g_pool_kq4s[layer],
        g_kmm_slot_d, g_kmm_pos_d, nu,
        g_block_table, g_pg_maxblk, g_pg_page, g_pg_plog, g_qwen.nkv, g_qwen.hd);
    return 0;
}

static int paged_kmm_upload(int nu, cudaStream_t st) {
    cudaMemcpyAsync(g_kmm_slot_d, h_kmm_slot, (size_t)nu * sizeof(int), cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(g_kmm_pos_d, h_kmm_pos, (size_t)nu * sizeof(int), cudaMemcpyHostToDevice, st);
    return 0;
}

static int paged_kmm_refresh(int layer, const int *pos_h, const int *slot_h, int n, cudaStream_t st) {
    if (n < 1) return 0;
    if (n > TQ_KMM_SCRATCH) return -2;
    if (paged_kmm_ensure() != 0) return -1;
    if (!g_pool_kmm[layer]) return 0;          // not a full-attn layer
    int nu = 0, plog = g_pg_plog;
    for (int i = 0; i < n; i++) {
        int slot = slot_h[i], pos = pos_h[i];
        int lb = pos >> plog, found = -1;
        for (int j = 0; j < nu; j++) {
            if (h_kmm_slot[j] == slot && (h_kmm_pos[j] >> plog) == lb) { found = j; break; }
        }
        if (found < 0) {
            h_kmm_slot[nu] = slot;
            h_kmm_pos[nu] = pos;
            nu++;
        } else if (pos > h_kmm_pos[found]) {
            h_kmm_pos[found] = pos;
        }
    }
    paged_kmm_upload(nu, st);
    return paged_kmm_launch(layer, nu, st);
}

// Every logical page of one slot, after a KV copy that bypassed the write kernel
// (qwn_paged_load_client / qwn_paged_slot_restore).
static int paged_kmm_rebuild_slot(int slot, int upto_pos, cudaStream_t st) {
    if (paged_topk_pages() <= 0) return 0;
    if (paged_kmm_ensure() != 0) return -1;
    int plog = g_pg_plog;
    int nb = (upto_pos >> plog) + 1;
    if (nb > TQ_KMM_SCRATCH) return -2;
    cudaMemcpyAsync(g_block_table, h_block_table,
                    (size_t)g_pg_maxslots * g_pg_maxblk * sizeof(int), cudaMemcpyHostToDevice, st);
    for (int lb = 0; lb < nb; lb++) {
        int last = ((lb + 1) << plog) - 1;
        if (last > upto_pos) last = upto_pos;
        h_kmm_slot[lb] = slot;
        h_kmm_pos[lb] = last;
    }
    paged_kmm_upload(nb, st);
    for (int L = 0; L < g_qwen.L; L++) {
        if (g_qwen.layer_types[L] != TQ_LAYER_FULL_ATTENTION) continue;
        if (paged_kmm_launch(L, nb, st) != 0) return -3;
    }
    return cudaStreamSynchronize(st) == cudaSuccess ? 0 : -4;
}

// Quest rank: one block per (kv head, client). Rotates q the same way the GQA kernel
// does, scores each page by max over the group's q-heads of Σ_d max(q_d*min_d, q_d*max_d),
// then keeps the top-k pages plus the first page and the last two. count 0 = dense
// (context below TQ_PAGED_TOPK_MIN_CTX, or k already covers every page).
__global__ void __launch_bounds__(256) k_tq_paged_topk_rank(
    int *list, int *counts, int cap, int kpages, int min_ctx, int max_nblk,
    const float *kmm, const float *q_proj_base, const uint16_t *q_norm_w,
    const int *positions, const int *slot_ids, const int *block_table, int max_blocks,
    int nkv, int nh, int hd, int q_m, int page_log,
    float eps, float rope_theta) {
    extern __shared__ __align__(16) char tk_dyn[];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int kv_head = blockIdx.x, col = blockIdx.y;
    const int group = nh / nkv;
    const int pos = positions[col];
    const int slot = slot_ids[col];
    int *cntp = counts + ((size_t)col * nkv + kv_head);
    if (tid == 0) *cntp = 0;
    if (pos + 1 < min_ctx || group > 16 || group < 1) return;
    const int nblk = (pos >> page_log) + 1;
    if (nblk < 1 || nblk > max_nblk) return;
    // max_nblk is a power of two >= this batch's page count (bitonic sort length).
    float *qn = (float *)tk_dyn;
    float *scores = qn + (size_t)group * hd;
    int *ord = (int *)(scores + max_nblk);
    unsigned char *sel = (unsigned char *)(ord + max_nblk);
    __shared__ float qtmp[256];
    __shared__ float red[16];
    const float *q_proj = q_proj_base + (size_t)col * q_m;
    for (int r = 0; r < group; r++) {
        const int hh = kv_head * group + r;
        float qv = q_proj[hh * (2 * hd) + tid];
        float qsum = qv * qv;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) qsum += __shfl_down_sync(0xffffffffu, qsum, o);
        if (lane == 0) red[warp] = qsum;
        __syncthreads();
        if (tid < 8) {
            float v = red[tid];
            for (int o = 4; o > 0; o >>= 1) v += __shfl_down_sync(0xffu, v, o);
            if (tid == 0) red[8] = v;
        }
        __syncthreads();
        float qss = red[8];
        qv = qv * rsqrtf(qss / (float)hd + eps) * (1.0f + tq_bf16_to_float(q_norm_w[tid]));
        if (tid < 64) {
            int idx = tid & 31;
            float freq = powf(rope_theta, -((float)(2 * idx) / 64.0f));
            float angle = (float)pos * freq, c = cosf(angle), s = sinf(angle);
            float q_pair = q_proj[hh * (2 * hd) + ((tid < 32) ? tid + 32 : tid - 32)];
            q_pair = q_pair * rsqrtf(qss / (float)hd + eps) * (1.0f + tq_bf16_to_float(q_norm_w[(tid < 32) ? tid + 32 : tid - 32]));
            float q_rot = (tid < 32) ? -q_pair : q_pair;
            qv = qv * c + q_rot * s;
        }
        __syncthreads();
        qtmp[tid] = qv;
        __syncthreads();
        tq_fwht256(qtmp, tid);
        qn[r * hd + tid] = qtmp[tid];
        __syncthreads();
    }
    for (int b = warp; b < nblk; b += 8) {
        int phys = block_table[(size_t)slot * max_blocks + b];
        const float *mn = kmm + (((size_t)phys * nkv + kv_head) * 2) * hd;
        const float *mx = mn + hd;
        float best = -3.402823466e38f;
        for (int h = 0; h < group; h++) {
            const float *q = qn + h * hd;
            float s = 0.0f;
            for (int d0 = 0; d0 < hd; d0 += 32) {
                int d = d0 + lane;
                float qd = q[d];
                s += fmaxf(qd * mn[d], qd * mx[d]);
            }
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
            best = fmaxf(best, s);
        }
        if (lane == 0) scores[b] = best;
    }
    __syncthreads();
    for (int b = tid; b < max_nblk; b += 256) {
        if (b >= nblk) scores[b] = -3.402823466e38f;
        ord[b] = (b < nblk) ? b : -1;
        sel[b] = 0;
    }
    __syncthreads();
    // Ascending bitonic sort of page ids by Quest score. Padded -inf sinks to the front,
    // so the best pages sit at the end of ord[].
    for (int size = 2; size <= max_nblk; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = tid; i < max_nblk; i += 256) {
                int j = i ^ stride;
                if (j > i) {
                    bool ascending = ((i & size) == 0);
                    if ((scores[i] > scores[j]) == ascending) {
                        float ts = scores[i]; scores[i] = scores[j]; scores[j] = ts;
                        int ti = ord[i]; ord[i] = ord[j]; ord[j] = ti;
                    }
                }
            }
            __syncthreads();
        }
    }
    if (tid == 0) {
        int taken = 0;
        for (int i = max_nblk - 1; i >= 0 && taken < kpages; --i) {
            int b = ord[i];
            if (b < 0 || b >= nblk) continue;
            sel[b] = 1;
            taken++;
        }
        sel[0] = 1;
        if (nblk >= 2) sel[nblk - 1] = 1;
        if (nblk >= 3) sel[nblk - 2] = 1;
    }
    __syncthreads();
    if (tid == 0) {
        int *out = list + ((size_t)col * nkv + kv_head) * (size_t)cap * 2;
        int w = 0;
        for (int b = 0; b < nblk && w < cap; b++) {
            if (!sel[b]) continue;
            int phys = block_table[(size_t)slot * max_blocks + b];
            out[w * 2] = phys;
            out[w * 2 + 1] = b;
            w++;
        }
        *cntp = (w >= nblk) ? 0 : w;
    }
}

static int g_topk_logged = 0;
static int paged_topk_rank(int layer, const float *q_proj, const uint16_t *q_norm_w,
                           const int *positions, const int *slot_ids, int N, int max_pos,
                           int kpages, cudaStream_t st) {
    if (paged_topk_bufs(kpages) != 0) return -1;
    if (!g_kmm_ready && paged_kmm_ensure() != 0) return -1;
    if (!g_pool_kmm[layer]) return -2;
    int nh = g_qwen.nh, nkv = g_qwen.nkv, hd = g_qwen.hd;
    int group = nh / nkv;
    int max_nblk = (max_pos >> g_pg_plog) + 1;
    if (max_nblk < 1) max_nblk = 1;
    int pad = 1;
    while (pad < max_nblk) pad <<= 1;
    if (pad > 8192) return -3;
    size_t smem = ((size_t)group * hd + (size_t)pad) * sizeof(float)
                + (size_t)pad * sizeof(int) + (size_t)pad;
    if (!g_topk_logged) {
        fprintf(stderr, "[paged-topk] pages=%d min_ctx=%d (count 0 = dense)\n",
                kpages, paged_topk_min_ctx());
        g_topk_logged = 1;
    }
    k_tq_paged_topk_rank<<<dim3(nkv, N), 256, smem, st>>>(
        g_topk_list, g_topk_count, g_topk_cap, kpages, paged_topk_min_ctx(), pad,
        g_pool_kmm[layer], q_proj, q_norm_w, positions, slot_ids, g_block_table, g_pg_maxblk,
        nkv, nh, hd, nh * hd * 2, g_pg_plog, g_qwen.eps, g_qwen.rope_theta);
    // TQ_PAGED_TOPK_DEBUG=1: once per process, count and logical page ids for client 0 / kv head 0.
    static int debug = -1, dumped = 0;
    if (debug < 0) {
        const char *e = getenv("TQ_PAGED_TOPK_DEBUG");
        debug = (e && e[0] == '1') ? 1 : 0;
    }
    if (debug && !dumped) {
        int c = -1;
        cudaMemcpyAsync(&c, g_topk_count, sizeof(int), cudaMemcpyDeviceToHost, st);
        cudaStreamSynchronize(st);
        fprintf(stderr, "[paged-topk] first-rank client0 pages_kept=%d (0 = dense walk)\n", c);
        if (c > 0 && c <= g_topk_cap) {
            int nshow = c < 128 ? c : 128;
            int buf[256];
            cudaMemcpy(buf, g_topk_list, (size_t)nshow * 2 * sizeof(int), cudaMemcpyDeviceToHost);
            fprintf(stderr, "[paged-topk] logical pages:");
            for (int i = 0; i < nshow; i++) fprintf(stderr, " %d", buf[i * 2 + 1]);
            if (c > nshow) fprintf(stderr, " ...");
            fprintf(stderr, "\n");
        }
        dumped = 1;
    }
    return 0;
}

// Allocate the shared paged pool: num_blocks blocks of `page` positions (Q4 KV for the 16
// full-attn layers) + per-slot DeltaNet state for max_slots. page must be a power of two.
// TQ_KV_V4=1 stores V like K (int4 codes + (scale, zp) pairs) instead of E4M3 + a per-head scale.
extern "C" int qwn_paged_init(int max_slots, int num_blocks, int page) {
    if (!g_qwen.initialized) return -1;
    if (!g_qwen.kv_q4) return -2;                          // Q4(K) + E4M3(V) or Q4(V) tiers only
    if (max_slots < 1 || num_blocks < 1 || page < 1 || (page & (page - 1))) return -3;
    paged_free_all();
    { const char *e = getenv("TQ_KV_V4"); g_pg_v4 = (e && atoi(e) != 0) ? 1 : 0; }
    { const char *e = getenv("TQ_KV_T4"); g_pg_t4 = (e && atoi(e) != 0) ? 1 : 0; }
    {   // TQ_DSTATE_BF16: 0 = fp32 recurrent state, 1 = bf16, 2 = fp16, 3 = bf16 + 8 bits (24-bit), 4 = fp16 with deterministic
        // stochastic rounding (tq_ds_acc) -- DEFAULT 4 since 2026-10-17 (owner's decision (c): e2e-neutral at 4K-64K in prefill
        // and pure decode, 80 instead of 155 MiB per slot = +400 blocks at the served footprint); 0 = the fp32 state (bitwise the
        // 10-16 line). notes/session10-dstate.md.
        const char *e = getenv("TQ_DSTATE_BF16"); int m = (e && *e) ? atoi(e) : 4;
        if (m < 0 || m > 4) return -13;
        g_pg_dstate = m; g_pg_ds_es = tq_ds_elem_bytes(m);
    }
    if (g_pg_t4 && !g_pg_v4) return -10;                   // trellis pages stage their open page as FP4 K+V
    if (g_pg_t4 && (page & 127)) return -10;                // whole 128-token pages are encoded
    if (g_pg_t4 && paged_topk_pages() > 0) return -11;      // page summaries read FP4 K; not ported
    int plog = 0; while ((1 << plog) < page) plog++;
    g_pg_page = page; g_pg_plog = plog; g_pg_nblocks = num_blocks; g_pg_maxslots = max_slots;
    g_pg_maxblk = (g_qwen.max_seq + page - 1) / page;
    int kv_m = g_qwen.nkv * g_qwen.hd, nkv = g_qwen.nkv;
    size_t qe = batched_q4s_elem_bytes();
    size_t recur_f = (size_t)batched_recur_floats(), conv_f = (size_t)batched_conv_floats();
    size_t bp = (size_t)num_blocks * page;                // pool rows
    for (int L = 0; L < g_qwen.L; L++) {
        if (g_qwen.layer_types[L] == TQ_LAYER_FULL_ATTENTION) {
            size_t k4b = bp * (kv_m >> 1), kq4sb = bp * nkv * 16 * qe;
            size_t v8b = g_pg_v4 ? k4b : bp * kv_m, vscb = g_pg_v4 ? kq4sb : bp * nkv * sizeof(float);
            if (g_pg_t4) kq4sb = vscb = 0;                // trellis pages: side words instead (paged_t4_alloc)
            if (cudaMalloc(&g_pool_k4[L], k4b) != cudaSuccess) { paged_free_all(); return -4; }
            if (kq4sb && cudaMalloc(&g_pool_kq4s[L], kq4sb) != cudaSuccess) { paged_free_all(); return -4; }
            if (cudaMalloc(&g_pool_v8[L], v8b) != cudaSuccess) { paged_free_all(); return -4; }
            if (vscb && cudaMalloc(&g_pool_vscale[L], vscb) != cudaSuccess) { paged_free_all(); return -4; }
            cudaMemsetAsync(g_pool_k4[L], 0, k4b, g_qwen.stream);
            if (kq4sb) cudaMemsetAsync(g_pool_kq4s[L], 0, kq4sb, g_qwen.stream);
            cudaMemsetAsync(g_pool_v8[L], 0, v8b, g_qwen.stream);
            if (vscb) cudaMemsetAsync(g_pool_vscale[L], 0, vscb, g_qwen.stream);
        } else if (g_qwen.layer_types[L] == TQ_LAYER_LINEAR_ATTENTION) {
            size_t rb = (size_t)max_slots * recur_f * g_pg_ds_es, cb = (size_t)max_slots * conv_f * sizeof(float);
            if (cudaMalloc(&g_pg_recur[L], rb) != cudaSuccess) { paged_free_all(); return -5; }
            if (cudaMalloc(&g_pg_conv[L], cb) != cudaSuccess) { paged_free_all(); return -5; }
            cudaMemsetAsync(g_pg_recur[L], 0, rb, g_qwen.stream);   // 0x0000 is +0 in every format
            cudaMemsetAsync(g_pg_conv[L], 0, cb, g_qwen.stream);
        }
    }
    size_t btn = (size_t)max_slots * g_pg_maxblk;
    if (cudaMalloc(&g_block_table, btn * sizeof(int)) != cudaSuccess) { paged_free_all(); return -6; }
    if (cudaMalloc(&g_slot_ids, (size_t)max_slots * sizeof(int)) != cudaSuccess) { paged_free_all(); return -6; }
    {   // TQ_GRAPH staging: tokens / slot ids / positions of a decode step (N <= max_slots) or a verify wave (T <= 64)
        int cols = max_slots > 64 ? max_slots : 64;
        if (cudaMalloc(&g_tok_d, (size_t)cols * sizeof(int)) != cudaSuccess) { paged_free_all(); return -6; }
        if (cudaHostAlloc(&h_pin_ctl, (size_t)3 * cols * sizeof(int), cudaHostAllocDefault) != cudaSuccess) {
            h_pin_ctl = NULL; paged_free_all(); return -6;
        }
        g_pin_cols = cols;
    }
    if (cudaMalloc(&g_iota, (size_t)max_slots * sizeof(int)) != cudaSuccess) { paged_free_all(); return -6; }
    if (cudaMalloc(&g_ones, (size_t)max_slots * sizeof(int)) != cudaSuccess) { paged_free_all(); return -6; }
    if (cudaMalloc(&g_pg_logits, (size_t)max_slots * g_qwen.V * sizeof(float)) != cudaSuccess) { paged_free_all(); return -6; }
    if (cudaMalloc(&g_pg_argmax, (size_t)max_slots * sizeof(int)) != cudaSuccess) { paged_free_all(); return -6; }
    h_block_table = (int *)malloc(btn * sizeof(int));
    h_free = (int *)malloc((size_t)num_blocks * sizeof(int));
    h_slot_nb = (int *)malloc((size_t)max_slots * sizeof(int));
    if (!h_block_table || !h_free || !h_slot_nb) { paged_free_all(); return -7; }
    for (size_t i = 0; i < btn; i++) h_block_table[i] = -1;
    for (int i = 0; i < num_blocks; i++) h_free[i] = num_blocks - 1 - i;   // pop low ids first
    g_pg_nfree = num_blocks;
    for (int i = 0; i < max_slots; i++) h_slot_nb[i] = 0;
    int *hi = (int *)malloc((size_t)max_slots * sizeof(int));
    for (int i = 0; i < max_slots; i++) hi[i] = i;
    cudaMemcpyAsync(g_iota, hi, (size_t)max_slots * sizeof(int), cudaMemcpyHostToDevice, g_qwen.stream);
    for (int i = 0; i < max_slots; i++) hi[i] = 1;
    cudaMemcpyAsync(g_ones, hi, (size_t)max_slots * sizeof(int), cudaMemcpyHostToDevice, g_qwen.stream);
    cudaMemcpyAsync(g_block_table, h_block_table, btn * sizeof(int), cudaMemcpyHostToDevice, g_qwen.stream);
    if (g_pg_t4 && paged_t4_alloc(max_slots, num_blocks, page) != 0) { free(hi); paged_free_all(); return -12; }
    if (ensure_wide_prefill_buffers(max_slots) != 0) { free(hi); paged_free_all(); return -8; }
    cudaStreamSynchronize(g_qwen.stream);
    free(hi);
    g_pg_ready = 1;
    if (g_pg_v4) fprintf(stderr, "[paged] TQ_KV_V4: V stored as int4 like K\n");
    if (g_pg_dstate) {
        int nlin = 0;
        for (int L = 0; L < g_qwen.L; L++) nlin += g_qwen.layer_types[L] == TQ_LAYER_LINEAR_ATTENTION;
        static const char *names[5] = {"fp32", "bf16", "fp16", "bf16+8 (24-bit)", "fp16 + stochastic rounding"};
        fprintf(stderr, "[paged] TQ_DSTATE_BF16=%d: DeltaNet recurrent state stored as %s (%.1f MiB per slot instead of %.1f; "
                "conv window fp32 %.1f MiB; compute fp32)\n", g_pg_dstate, names[g_pg_dstate],
                (double)nlin * recur_f * g_pg_ds_es / 1048576.0, (double)nlin * recur_f * sizeof(float) / 1048576.0,
                (double)nlin * conv_f * sizeof(float) / 1048576.0);
    }
    fprintf(stderr, "[paged] TQ_PAGED_ATTN_ST=%d%s\n", paged_attn_st_enabled(),
            g_pg_t4 ? " (no-op with TQ_KV_T4: the trellis pool uses its own GQA kernels)"
            : (paged_attn_st_enabled() >= 2 && g_pg_v4) ? " (fp16 int4 GQA decode / verify kernel)"
            : (paged_attn_st_enabled() >= 2) ? " (E4M3 V: the fp16 int4 path needs TQ_KV_V4=1; base kernel, bitwise = e1ecb3a's)"
            : " (bitwise = e1ecb3a's kernel)");
    return 0;
}

extern "C" int qwn_paged_free(void) { paged_free_all(); return 0; }

// Free a slot's blocks back to the pool and zero its DeltaNet state (slot is now reusable).
extern "C" int qwn_paged_reset_slot(int slot) {
    if (!g_pg_ready || slot < 0 || slot >= g_pg_maxslots) return -1;
    int *row = h_block_table + (size_t)slot * g_pg_maxblk;
    for (int lb = 0; lb < h_slot_nb[slot]; lb++) {
        if (row[lb] >= 0) h_free[g_pg_nfree++] = row[lb];
        row[lb] = -1;
    }
    h_slot_nb[slot] = 0;
    if (g_pg_t4) { h_slot_encl[slot] = 0; paged_t4_sync(g_qwen.stream); }
    size_t conv_f = (size_t)batched_conv_floats();
    for (int L = 0; L < g_qwen.L; L++) {
        if (g_qwen.layer_types[L] != TQ_LAYER_LINEAR_ATTENTION) continue;
        cudaMemsetAsync(pg_recur_slot(L, slot), 0, pg_recur_slot_bytes(), g_qwen.stream);
        cudaMemsetAsync(g_pg_conv[L] + (size_t)slot * conv_f, 0, conv_f * sizeof(float), g_qwen.stream);
    }
    return cudaStreamSynchronize(g_qwen.stream) == cudaSuccess ? 0 : -2;
}

// fp32 -> the dstate format (qwn_paged_load_client: the single-stream state into a half-precision slot; dst = the slot).
template <int DS>
__global__ void k_tq_ds_from_f32(void *dst, const float *src, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) tq_ds_st_elem<DS>(dst, n, i, src[i]);
}
static void paged_ds_from_f32(void *dst_slot, const float *src, size_t n, cudaStream_t st) {
    unsigned g = (unsigned)((n + 255) / 256);
    switch (g_pg_dstate) {
        case 1: k_tq_ds_from_f32<1><<<g, 256, 0, st>>>(dst_slot, src, n); break;
        case 2: k_tq_ds_from_f32<2><<<g, 256, 0, st>>>(dst_slot, src, n); break;
        case 3: k_tq_ds_from_f32<3><<<g, 256, 0, st>>>(dst_slot, src, n); break;
        case 4: k_tq_ds_from_f32<4><<<g, 256, 0, st>>>(dst_slot, src, n); break;
        default: cudaMemcpyAsync(dst_slot, src, n * sizeof(float), cudaMemcpyDeviceToDevice, st); break;
    }
}

// Snapshot the engine's current single-stream Q4 state into paged slot (parity / bring-up):
// copy KV rows [0..upto_pos] into the slot's (scattered) pool blocks + DeltaNet state.
extern "C" int qwn_paged_load_client(int slot, int upto_pos) {
    if (!g_pg_ready || slot < 0 || slot >= g_pg_maxslots) return -1;
    if (upto_pos < 0) return -2;
    if (!g_qwen.kv_q4) return -3;
    if (g_pg_v4 || g_pg_t4) return -9;                           // the single-stream cache holds E4M3 V
    if (paged_ensure_blocks(slot, upto_pos) != 0) return -4;     // pool exhausted
    int kv_m = g_qwen.nkv * g_qwen.hd, nkv = g_qwen.nkv, page = g_pg_page;
    size_t qe = batched_q4s_elem_bytes();
    size_t recur_f = (size_t)batched_recur_floats(), conv_f = (size_t)batched_conv_floats();
    int *row = h_block_table + (size_t)slot * g_pg_maxblk;
    int need = (upto_pos >> g_pg_plog) + 1;
    for (int L = 0; L < g_qwen.L; L++) {
        tq_layer_t *l = &g_qwen.layers[L];
        if (g_qwen.layer_types[L] == TQ_LAYER_FULL_ATTENTION) {
            if (!l->d_k_cache4 || !l->d_k_q4s || !l->d_v_cache8 || !l->d_v_scale) return -5;
            for (int lb = 0; lb < need; lb++) {
                int phys = row[lb], t0 = lb * page, cnt = (upto_pos + 1 - t0);
                if (cnt > page) cnt = page;
                if (cnt <= 0) break;
                size_t bo = (size_t)phys * page;     // block base row in pool
                cudaMemcpyAsync(g_pool_k4[L] + bo * (kv_m >> 1), l->d_k_cache4 + (size_t)t0 * (kv_m >> 1),
                                (size_t)cnt * (kv_m >> 1), cudaMemcpyDeviceToDevice, g_qwen.stream);
                cudaMemcpyAsync((char *)g_pool_kq4s[L] + bo * nkv * 16 * qe,
                                (char *)l->d_k_q4s + (size_t)t0 * nkv * 16 * qe,
                                (size_t)cnt * nkv * 16 * qe, cudaMemcpyDeviceToDevice, g_qwen.stream);
                cudaMemcpyAsync(g_pool_v8[L] + bo * kv_m, l->d_v_cache8 + (size_t)t0 * kv_m,
                                (size_t)cnt * kv_m, cudaMemcpyDeviceToDevice, g_qwen.stream);
                cudaMemcpyAsync(g_pool_vscale[L] + bo * nkv, l->d_v_scale + (size_t)t0 * nkv,
                                (size_t)cnt * nkv * sizeof(float), cudaMemcpyDeviceToDevice, g_qwen.stream);
            }
        } else if (g_qwen.layer_types[L] == TQ_LAYER_LINEAR_ATTENTION) {
            if (!l->d_linear_recurrent_state || !l->d_linear_conv_state) return -6;
            paged_ds_from_f32(pg_recur_slot(L, slot), l->d_linear_recurrent_state, recur_f, g_qwen.stream);
            cudaMemcpyAsync(g_pg_conv[L] + (size_t)slot * conv_f, l->d_linear_conv_state,
                            conv_f * sizeof(float), cudaMemcpyDeviceToDevice, g_qwen.stream);
        }
    }
    if (paged_topk_pages() > 0) {
        if (paged_kmm_rebuild_slot(slot, upto_pos, g_qwen.stream) != 0) return -8;
    }
    return cudaStreamSynchronize(g_qwen.stream) == cudaSuccess ? 0 : -7;
}
