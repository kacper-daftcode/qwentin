// session parking: slot save / restore through host memory and the pinned ring.
// Part of the forward_qwen.cu unity build: included in order, not a standalone header.

// ===== Session parking: a slot's paged KV + DeltaNet state <-> a host buffer =====
// Host layout (bytes), logical-block order so a restore may land in different physical blocks:
//   [for each FULL layer L][for each logical block lb < nb]: K4 rows | kq4s rows | V8 rows | vscale rows
//   (TQ_KV_V4: V4 rows | vq4s rows in K's row sizes)
//   [for each LINEAR layer L]: recur floats | conv floats
// Rows per block = page except the last (cnt = n - lb*page). n = upto_pos + 1 consumed positions.
// Pageable host memory is fine for v1 (the runtime stages it; ~1 GB per 40K tokens moves in
// ~0.2-0.4 s); pinned buffers turn the copies async later.
static size_t paged_layer_token_bytes(void) {             // one full-attention layer, one position (pool rows)
    int kv_m = g_qwen.nkv * g_qwen.hd, nkv = g_qwen.nkv;
    if (g_pg_t4) return (size_t)nkv * paged_t4_pool_vec_bytes();
    size_t k_tok = (size_t)(kv_m >> 1) + (size_t)nkv * 16 * batched_q4s_elem_bytes();
    return k_tok + (g_pg_v4 ? k_tok : (size_t)kv_m + (size_t)nkv * sizeof(float));
}
static size_t paged_fp4_layer_token_bytes(void) {         // the FP4 K + V row footprint (MTP pool, T4 staging)
    int kv_m = g_qwen.nkv * g_qwen.hd, nkv = g_qwen.nkv;
    size_t k_tok = (size_t)(kv_m >> 1) + (size_t)nkv * 16 * batched_q4s_elem_bytes();
    return 2 * k_tok;
}

// TQ_DSTATE_BF16: a parking blob written with a half-precision recurrent state starts with this 16-byte header (the
// fp32 mode keeps main's header-less layout byte for byte); qwn_paged_slot_restore rejects a blob whose header is
// missing or names another state format (-11) instead of scattering fp32 bytes into a bf16 state (or vice versa; an
// fp32 blob offered to a half-precision pool also fails the size check only in the other direction).
struct tq_park_hdr { char magic[4]; uint8_t version, dstate, es, pad; uint32_t recur_f; uint32_t conv_f; };
static size_t paged_park_hdr_bytes(void) { return g_pg_dstate ? sizeof(tq_park_hdr) : 0; }
static void paged_park_hdr_fill(tq_park_hdr *h) {
    memcpy(h->magic, "QDS1", 4); h->version = 1; h->dstate = (uint8_t)g_pg_dstate; h->es = (uint8_t)g_pg_ds_es; h->pad = 0;
    h->recur_f = (uint32_t)batched_recur_floats(); h->conv_f = (uint32_t)batched_conv_floats();
}
static int paged_park_hdr_ok(const tq_park_hdr *h) {
    tq_park_hdr w; paged_park_hdr_fill(&w);
    return memcmp(h, &w, sizeof(tq_park_hdr)) == 0;
}

static size_t paged_slot_bytes(int upto_pos) {
    int n = upto_pos + 1;
    size_t per_tok = paged_layer_token_bytes();
    size_t bytes = paged_park_hdr_bytes();
    int nfull = 0, nlin = 0;
    for (int L = 0; L < g_qwen.L; L++) {
        if (g_qwen.layer_types[L] == TQ_LAYER_FULL_ATTENTION) nfull++;
        else if (g_qwen.layer_types[L] == TQ_LAYER_LINEAR_ATTENTION) nlin++;
    }
    bytes += (size_t)nfull * (size_t)n * per_tok;             // T4: every committed position is a trellis row
    bytes += (size_t)nlin * (pg_recur_slot_bytes() + (size_t)batched_conv_floats() * sizeof(float));
    if (g_mtp_on) bytes += (size_t)n * (g_pg_t4 ? paged_fp4_layer_token_bytes() : per_tok) + (size_t)g_qwen.H * sizeof(float);
    return bytes;
}

extern "C" size_t qwn_paged_slot_bytes(int upto_pos) {
    if (!g_pg_ready || upto_pos < 0) return 0;
    return paged_slot_bytes(upto_pos);
}

// KV bytes per resident token over all full-attention layers (pool sizing / reporting).
extern "C" size_t qwn_paged_token_bytes(void) {
    if (!g_pg_ready) return 0;
    int nfull = 0;
    for (int L = 0; L < g_qwen.L; L++) nfull += g_qwen.layer_types[L] == TQ_LAYER_FULL_ATTENTION;
    return (size_t)nfull * paged_layer_token_bytes();
}

// The pool's recurrent-state storage format (TQ_DSTATE_BF16 as read by qwn_paged_init): 0 fp32, 1 bf16, 2 fp16;
// -1 before qwn_paged_init. qwn_paged_state_bytes: the per-slot DeltaNet state (recurrent + conv window) in bytes.
extern "C" int qwn_paged_dstate(void) { return g_pg_ready ? g_pg_dstate : -1; }
extern "C" size_t qwn_paged_state_bytes(void) {
    if (!g_pg_ready) return 0;
    int nlin = 0;
    for (int L = 0; L < g_qwen.L; L++) nlin += g_qwen.layer_types[L] == TQ_LAYER_LINEAR_ATTENTION;
    return (size_t)nlin * (pg_recur_slot_bytes() + (size_t)batched_conv_floats() * sizeof(float));
}

// dir 0 = save (device -> host), 1 = restore (host -> device; the slot must already own the blocks)
// ===== Parking through a pinned staging ring (B4) =====
// paged_slot_xfer issued one pageable cudaMemcpyAsync per block-row span (~65K of <= 64 KB for a 128K session) ->
// 5.5 GB/s out, ~16 GB/s in. Here the spans of a chunk (<= TQ_PARK_CHUNK bytes) are gathered on the device into a
// staging buffer by one kernel, moved as ONE DMA to/from a pinned ring slot, and copied between the ring slot and the
// caller's pageable blob by a few host threads -- the host memcpy of chunk k overlaps the DMA of chunk k+1 (save) /
// k-1 (restore). Same host layout and span order as paged_slot_xfer (blobs are interchangeable). TQ_PARK_RING=0
// restores the old path. M5 (notes): a concurrent 53 GB/s DMA costs other streams' small copies <= 0.1 ms.
struct tq_span { const uint8_t *dev; size_t off; size_t bytes; };
#define TQ_PARK_CHUNK ((size_t)32 << 20)
#define TQ_PARK_RING 3
#define TQ_PARK_MAX_SPANS 4096
#define TQ_PARK_THREADS 8
static uint8_t *g_park_pin[TQ_PARK_RING] = {0};
static tq_span *g_park_spans_h[TQ_PARK_RING] = {0}, *g_park_spans_d[TQ_PARK_RING] = {0};
static uint8_t *g_park_stage[2] = {0};
static cudaEvent_t g_park_ev[TQ_PARK_RING] = {0};
static int g_park_ready = 0;

static void park_ring_free(void) {
    for (int r = 0; r < TQ_PARK_RING; r++) {
        if (g_park_pin[r]) { cudaFreeHost(g_park_pin[r]); g_park_pin[r] = NULL; }
        if (g_park_spans_h[r]) { cudaFreeHost(g_park_spans_h[r]); g_park_spans_h[r] = NULL; }
        if (g_park_spans_d[r]) { cudaFree(g_park_spans_d[r]); g_park_spans_d[r] = NULL; }
        if (g_park_ev[r]) { cudaEventDestroy(g_park_ev[r]); g_park_ev[r] = NULL; }
    }
    for (int i = 0; i < 2; i++) if (g_park_stage[i]) { cudaFree(g_park_stage[i]); g_park_stage[i] = NULL; }
    g_park_ready = 0;
}
static int g_park_ring_mode = -1;                  // -1 = from TQ_PARK_RING (default 1), else qwn_park_ring_set
static int park_ring_enabled(void) {
    static int c = -1;
    if (g_park_ring_mode >= 0) return g_park_ring_mode;
    if (c < 0) { const char *e = getenv("TQ_PARK_RING"); c = (e && *e) ? (atoi(e) != 0) : 1; }
    return c;
}
// Test ABI: 1 = pinned staging ring, 0 = the per-span pageable path (analysis/park_bench.py A/B).
extern "C" int qwn_park_ring_set(int on) { g_park_ring_mode = on ? 1 : 0; return 0; }
static int park_ring_ensure(void) {
    if (g_park_ready) return 0;
    for (int r = 0; r < TQ_PARK_RING; r++) {
        if (cudaHostAlloc((void **)&g_park_pin[r], TQ_PARK_CHUNK, cudaHostAllocDefault) != cudaSuccess) { park_ring_free(); return -1; }
        if (cudaHostAlloc((void **)&g_park_spans_h[r], TQ_PARK_MAX_SPANS * sizeof(tq_span), cudaHostAllocDefault) != cudaSuccess) { park_ring_free(); return -1; }
        if (cudaMalloc(&g_park_spans_d[r], TQ_PARK_MAX_SPANS * sizeof(tq_span)) != cudaSuccess) { park_ring_free(); return -2; }
        if (cudaEventCreateWithFlags(&g_park_ev[r], cudaEventDisableTiming) != cudaSuccess) { park_ring_free(); return -3; }
    }
    for (int i = 0; i < 2; i++)
        if (cudaMalloc(&g_park_stage[i], TQ_PARK_CHUNK) != cudaSuccess) { park_ring_free(); return -2; }
    g_park_ready = 1;
    return 0;
}
// span blockIdx.x: device rows -> stage + off (gather) / stage + off -> device rows (scatter); 16-B units.
__global__ void k_tq_park_gather(const tq_span *spans, uint8_t *stage) {
    const tq_span s = spans[blockIdx.x];
    const uint4 *src = (const uint4 *)s.dev;
    uint4 *dst = (uint4 *)(stage + s.off);
    for (size_t i = threadIdx.x; i < s.bytes >> 4; i += blockDim.x) dst[i] = src[i];
}
__global__ void k_tq_park_scatter(const tq_span *spans, const uint8_t *stage) {
    const tq_span s = spans[blockIdx.x];
    const uint4 *src = (const uint4 *)(stage + s.off);
    uint4 *dst = (uint4 *)const_cast<uint8_t *>(s.dev);
    for (size_t i = threadIdx.x; i < s.bytes >> 4; i += blockDim.x) dst[i] = src[i];
}
static void park_memcpy_mt(void *dst, const void *src, size_t bytes) {
    std::thread th[TQ_PARK_THREADS];
    size_t per = ((bytes + TQ_PARK_THREADS - 1) / TQ_PARK_THREADS + 4095) & ~(size_t)4095;
    int nt = 0;
    for (size_t o = 0; o < bytes; o += per, nt++) {
        size_t b = bytes - o < per ? bytes - o : per;
        th[nt] = std::thread(memcpy, (char *)dst + o, (const char *)src + o, b);
    }
    for (int i = 0; i < nt; i++) th[i].join();
}
// The slot's spans in the host-layout order (device pointer, blob offset, bytes); also the total.
static size_t park_spans(int slot, int upto_pos, std::vector<tq_span> &sp) {
    int kv_m = g_qwen.nkv * g_qwen.hd, nkv = g_qwen.nkv, page = g_pg_page;
    size_t qe = batched_q4s_elem_bytes();
    size_t recur_f = (size_t)batched_recur_floats(), conv_f = (size_t)batched_conv_floats();
    int *row = h_block_table + (size_t)slot * g_pg_maxblk;
    int n = upto_pos + 1, nb = (upto_pos >> g_pg_plog) + 1;
    size_t off = 0;
    auto add = [&](const void *dev, size_t bytes) { sp.push_back({(const uint8_t *)dev, off, bytes}); off += bytes; };
    for (int L = 0; L < g_qwen.L; L++) {
        if (g_qwen.layer_types[L] == TQ_LAYER_FULL_ATTENTION) {
            for (int lb = 0; lb < nb; lb++) {
                int phys = row[lb], t0 = lb * page, cnt = n - t0;
                if (cnt > page) cnt = page;
                if (cnt <= 0) break;
                size_t bo = (size_t)phys * page;
                if (g_pg_t4) {                                // TQ_KV_T4: trellis rows + side words (same order as paged_slot_xfer)
                    add(g_pool_k4[L] + bo * (kv_m >> 1), (size_t)cnt * (kv_m >> 1));
                    add(g_t4_kside[L] + bo * nkv * 4, (size_t)cnt * nkv * 4);
                    add(g_pool_v8[L] + bo * (kv_m >> 1), (size_t)cnt * (kv_m >> 1));
                    add(g_t4_vside[L] + bo * nkv * 4, (size_t)cnt * nkv * 4);
                    continue;
                }
                add(g_pool_k4[L] + bo * (kv_m >> 1), (size_t)cnt * (kv_m >> 1));
                add((char *)g_pool_kq4s[L] + bo * nkv * 16 * qe, (size_t)cnt * nkv * 16 * qe);
                if (g_pg_v4) {
                    add(g_pool_v8[L] + bo * (kv_m >> 1), (size_t)cnt * (kv_m >> 1));
                    add((char *)g_pool_vscale[L] + bo * nkv * 16 * qe, (size_t)cnt * nkv * 16 * qe);
                } else {
                    add(g_pool_v8[L] + bo * kv_m, (size_t)cnt * kv_m);
                    add(g_pool_vscale[L] + bo * nkv, (size_t)cnt * nkv * sizeof(float));
                }
            }
        } else if (g_qwen.layer_types[L] == TQ_LAYER_LINEAR_ATTENTION) {
            add(pg_recur_slot(L, slot), pg_recur_slot_bytes());
            add(g_pg_conv[L] + (size_t)slot * conv_f, conv_f * sizeof(float));
        }
    }
    (void)recur_f;
    if (g_mtp_on) {
        for (int lb = 0; lb < nb; lb++) {
            int phys = row[lb], t0 = lb * page, cnt = n - t0;
            if (cnt > page) cnt = page;
            if (cnt <= 0) break;
            size_t bo = (size_t)phys * page;
            add(g_mtp_k4 + bo * (kv_m >> 1), (size_t)cnt * (kv_m >> 1));
            add((char *)g_mtp_kq4s + bo * nkv * 16 * qe, (size_t)cnt * nkv * 16 * qe);
            if (g_pg_v4) {
                add(g_mtp_v8 + bo * (kv_m >> 1), (size_t)cnt * (kv_m >> 1));
                add((char *)g_mtp_vscale + bo * nkv * 16 * qe, (size_t)cnt * nkv * 16 * qe);
            } else {
                add(g_mtp_v8 + bo * kv_m, (size_t)cnt * kv_m);
                add(g_mtp_vscale + bo * nkv, (size_t)cnt * nkv * sizeof(float));
            }
        }
        add(g_mtp_last_h + (size_t)slot * g_qwen.H, (size_t)g_qwen.H * sizeof(float));
    }
    return off;
}
// dir 0 = save (device -> host blob), 1 = restore. Every span is a multiple of 16 B at 16-B aligned addresses.
static int paged_slot_xfer_ring(int slot, int upto_pos, char *host, int dir) {
    if (park_ring_ensure() != 0) return -8;
    std::vector<tq_span> sp;
    sp.reserve(80000);
    size_t total = park_spans(slot, upto_pos, sp);
    for (const tq_span &x : sp)
        if ((x.bytes & 15) || ((uintptr_t)x.dev & 15) || x.bytes > TQ_PARK_CHUNK) return -9;
    // chunks: [a, b) span ranges of <= TQ_PARK_CHUNK bytes and <= TQ_PARK_MAX_SPANS spans
    std::vector<int> ca, cb; std::vector<size_t> coff, cbytes;
    for (size_t i = 0; i < sp.size();) {
        size_t base = sp[i].off, bytes = 0; size_t j = i;
        while (j < sp.size() && bytes + sp[j].bytes <= TQ_PARK_CHUNK && (int)(j - i) < TQ_PARK_MAX_SPANS) { bytes += sp[j].bytes; j++; }
        ca.push_back((int)i); cb.push_back((int)j); coff.push_back(base); cbytes.push_back(bytes);
        i = j;
    }
    int nchunks = (int)ca.size();
    cudaStream_t st = g_qwen.stream;
    auto issue = [&](int k) -> int {
        int r = k % TQ_PARK_RING, sg = k % 2, nsp = cb[k] - ca[k];
        tq_span *hs = g_park_spans_h[r];
        for (int i = 0; i < nsp; i++) { hs[i] = sp[ca[k] + i]; hs[i].off -= coff[k]; }
        if (cudaMemcpyAsync(g_park_spans_d[r], hs, (size_t)nsp * sizeof(tq_span), cudaMemcpyHostToDevice, st) != cudaSuccess) return -4;
        if (dir == 0) {
            k_tq_park_gather<<<nsp, 256, 0, st>>>(g_park_spans_d[r], g_park_stage[sg]);
            if (cudaMemcpyAsync(g_park_pin[r], g_park_stage[sg], cbytes[k], cudaMemcpyDeviceToHost, st) != cudaSuccess) return -4;
        } else {
            if (cudaMemcpyAsync(g_park_stage[sg], g_park_pin[r], cbytes[k], cudaMemcpyHostToDevice, st) != cudaSuccess) return -4;
            k_tq_park_scatter<<<nsp, 256, 0, st>>>(g_park_spans_d[r], g_park_stage[sg]);
        }
        return cudaEventRecord(g_park_ev[r], st) == cudaSuccess ? 0 : -4;
    };
    int rc = 0;
    if (dir == 0) {
        // save: GPU gathers + DMAs chunk k+1/k+2 while the host copies chunk k out of its ring slot
        for (int k = 0; k < nchunks && k < 2; k++) if ((rc = issue(k)) != 0) return rc;
        for (int k = 0; k < nchunks; k++) {
            if (cudaEventSynchronize(g_park_ev[k % TQ_PARK_RING]) != cudaSuccess) return -5;
            park_memcpy_mt(host + coff[k], g_park_pin[k % TQ_PARK_RING], cbytes[k]);
            if (k + 2 < nchunks && (rc = issue(k + 2)) != 0) return rc;
        }
    } else {
        // restore: the host fills ring slot k+1 while the GPU moves + scatters chunk k
        park_memcpy_mt(g_park_pin[0], host + coff[0], cbytes[0]);
        if ((rc = issue(0)) != 0) return rc;
        for (int k = 0; k < nchunks; k++) {
            if (k + 1 < nchunks) {
                int r = (k + 1) % TQ_PARK_RING;
                if (k + 1 >= TQ_PARK_RING && cudaEventSynchronize(g_park_ev[r]) != cudaSuccess) return -5;   // slot's DMA done
                park_memcpy_mt(g_park_pin[r], host + coff[k + 1], cbytes[k + 1]);
                if ((rc = issue(k + 1)) != 0) return rc;
            }
        }
    }
    (void)total;
    return cudaStreamSynchronize(st) == cudaSuccess ? 0 : -5;
}

static int paged_slot_xfer(int slot, int upto_pos, char *host, int dir) {
    int kv_m = g_qwen.nkv * g_qwen.hd, nkv = g_qwen.nkv, page = g_pg_page;
    size_t qe = batched_q4s_elem_bytes();
    size_t recur_f = (size_t)batched_recur_floats(), conv_f = (size_t)batched_conv_floats();
    int *row = h_block_table + (size_t)slot * g_pg_maxblk;
    int n = upto_pos + 1, nb = (upto_pos >> g_pg_plog) + 1;
    if (nb > h_slot_nb[slot]) return -3;                         // slot does not own that many blocks
    paged_t4_sync(g_qwen.stream);                                // TQ_KV_T4: the side-stream encodes of this slot are in the pool
    if (park_ring_enabled()) {
        int rc = paged_slot_xfer_ring(slot, upto_pos, host, dir);
        if (rc != -8 && rc != -9) return rc;                     // -8/-9: ring unavailable -> the per-span path
    }
    cudaMemcpyKind kind = dir ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
    char *p = host;
    #define TQ_XFER(dev, bytes) do { \
        void *d_ = (void *)(dev); size_t b_ = (bytes); \
        cudaError_t e_ = dir ? cudaMemcpyAsync(d_, p, b_, kind, g_qwen.stream) \
                             : cudaMemcpyAsync(p, d_, b_, kind, g_qwen.stream); \
        if (e_ != cudaSuccess) return -4; p += b_; } while (0)
    for (int L = 0; L < g_qwen.L; L++) {
        tq_layer_t *l = &g_qwen.layers[L];
        (void)l;
        if (g_qwen.layer_types[L] == TQ_LAYER_FULL_ATTENTION) {
            for (int lb = 0; lb < nb; lb++) {
                int phys = row[lb], t0 = lb * page, cnt = n - t0;
                if (cnt > page) cnt = page;
                if (cnt <= 0) break;
                size_t bo = (size_t)phys * page;
                if (g_pg_t4) {                                // every committed position is a trellis row
                    TQ_XFER(g_pool_k4[L] + bo * (kv_m >> 1), (size_t)cnt * (kv_m >> 1));
                    TQ_XFER(g_t4_kside[L] + bo * nkv * 4, (size_t)cnt * nkv * 4);
                    TQ_XFER(g_pool_v8[L] + bo * (kv_m >> 1), (size_t)cnt * (kv_m >> 1));
                    TQ_XFER(g_t4_vside[L] + bo * nkv * 4, (size_t)cnt * nkv * 4);
                    continue;
                }
                TQ_XFER(g_pool_k4[L] + bo * (kv_m >> 1), (size_t)cnt * (kv_m >> 1));
                TQ_XFER((char *)g_pool_kq4s[L] + bo * nkv * 16 * qe, (size_t)cnt * nkv * 16 * qe);
                if (g_pg_v4) {
                    TQ_XFER(g_pool_v8[L] + bo * (kv_m >> 1), (size_t)cnt * (kv_m >> 1));
                    TQ_XFER((char *)g_pool_vscale[L] + bo * nkv * 16 * qe, (size_t)cnt * nkv * 16 * qe);
                } else {
                    TQ_XFER(g_pool_v8[L] + bo * kv_m, (size_t)cnt * kv_m);
                    TQ_XFER(g_pool_vscale[L] + bo * nkv, (size_t)cnt * nkv * sizeof(float));
                }
            }
        } else if (g_qwen.layer_types[L] == TQ_LAYER_LINEAR_ATTENTION) {
            TQ_XFER(pg_recur_slot(L, slot), pg_recur_slot_bytes());
            TQ_XFER(g_pg_conv[L] + (size_t)slot * conv_f, conv_f * sizeof(float));
        }
    }
    (void)recur_f;
    if (g_mtp_on) {                                  // the MTP layer's KV rows + the slot's last hidden
        for (int lb = 0; lb < nb; lb++) {
            int phys = row[lb], t0 = lb * page, cnt = n - t0;
            if (cnt > page) cnt = page;
            if (cnt <= 0) break;
            size_t bo = (size_t)phys * page;
            TQ_XFER(g_mtp_k4 + bo * (kv_m >> 1), (size_t)cnt * (kv_m >> 1));
            TQ_XFER((char *)g_mtp_kq4s + bo * nkv * 16 * qe, (size_t)cnt * nkv * 16 * qe);
            if (g_pg_v4) {
                TQ_XFER(g_mtp_v8 + bo * (kv_m >> 1), (size_t)cnt * (kv_m >> 1));
                TQ_XFER((char *)g_mtp_vscale + bo * nkv * 16 * qe, (size_t)cnt * nkv * 16 * qe);
            } else {
                TQ_XFER(g_mtp_v8 + bo * kv_m, (size_t)cnt * kv_m);
                TQ_XFER(g_mtp_vscale + bo * nkv, (size_t)cnt * nkv * sizeof(float));
            }
        }
        TQ_XFER(g_mtp_last_h + (size_t)slot * g_qwen.H, (size_t)g_qwen.H * sizeof(float));
    }
    #undef TQ_XFER
    return cudaStreamSynchronize(g_qwen.stream) == cudaSuccess ? 0 : -5;
}

// Save positions [0..upto_pos] of a slot (KV blocks + DeltaNet state) into `host` (>= qwn_paged_slot_bytes).
extern "C" int qwn_paged_slot_save(int slot, int upto_pos, void *host, size_t host_bytes) {
    if (!g_pg_ready || slot < 0 || slot >= g_pg_maxslots || upto_pos < 0 || !host) return -1;
    if (host_bytes < paged_slot_bytes(upto_pos)) return -2;
    size_t hb = paged_park_hdr_bytes();
    if (hb) { tq_park_hdr h; paged_park_hdr_fill(&h); memcpy(host, &h, sizeof(h)); }
    return paged_slot_xfer(slot, upto_pos, (char *)host + hb, 0);
}

// Restore a saved session into a (reset, empty) slot: allocates its blocks, copies KV + state back.
extern "C" int qwn_paged_slot_restore(int slot, int upto_pos, const void *host, size_t host_bytes) {
    if (!g_pg_ready || slot < 0 || slot >= g_pg_maxslots || upto_pos < 0 || !host) return -1;
    if (host_bytes < paged_slot_bytes(upto_pos)) return -2;
    size_t hb = paged_park_hdr_bytes();
    if (hb && !paged_park_hdr_ok((const tq_park_hdr *)host)) return -11;   // blob of another recurrent-state format
    if (paged_ensure_blocks(slot, upto_pos) != 0) return -6;     // pool exhausted
    int rc = paged_slot_xfer(slot, upto_pos, (char *)host + hb, 1);
    if (rc != 0) return rc;
    if (g_pg_t4) h_slot_encl[slot] = upto_pos + 1;
    // Summaries are a function of K and are not in the parking blob. The restored
    // blocks are new physical ids, so rebuild before the next decode step ranks them.
    if (paged_topk_pages() > 0) {
        if (paged_kmm_rebuild_slot(slot, upto_pos, g_qwen.stream) != 0) return -7;
    }
    return 0;
}

// The paged paths hold every column's logits (g_pg_logits) only when the lm_head runs on the
// wide path (EXL3, or E2M3 in the QMMA scale-factor layout); other layouts argmax per column.
static inline int paged_lm_head_wide(void) {
    return !g_qwen.tie_word_embeddings &&
           (g_qwen.lm_head.exl3 ||
            (g_qwen.lm_head.e2m3 &&
             !g_qwen.lm_head.word_major && can_use_qmma_sf_weight(&g_qwen.lm_head) && !g_qwen.lm_head.row_major));
}

// One paged decode step over N active columns. column j -> slot_ids[j] (persistent slot);
// tokens[j] is decoded at positions[j] (its own KV history via the block table).
// Host prework of a decode step: validation, block allocation, the step's max position.
static int paged_decode_prep(const int *slot_ids_h, const int *positions, int N, int *dec_maxpos_out) {
    int dec_maxpos = 0;
    for (int j = 0; j < N; j++) {
        int slot = slot_ids_h[j];
        if (slot < 0 || slot >= g_pg_maxslots) return -60;
        if (positions[j] < 0 || positions[j] >= g_qwen.max_seq) return -61;
        if (paged_ensure_blocks(slot, positions[j]) != 0) return -62;     // admission / pool full
        if (positions[j] > dec_maxpos) dec_maxpos = positions[j];
    }
    *dec_maxpos_out = dec_maxpos;
    return 0;
}
// TQ_KV_T4 tail of a decode step: the step's columns -> trellis for the next step (host-side ranges, eager in
// both the stream and the graph mode: the ranges change every step).
static int paged_decode_t4_tail(const int *slot_ids_h, const int *positions, int N) {
    if (!g_pg_t4) return 0;
    tq_t4_ranges rg; rg.nr = 0;
    for (int j = 0; j < N; j++) {
        int slot = slot_ids_h[j], len = positions[j] + 1;
        if (len <= h_slot_encl[slot]) continue;
        if (paged_t4_add_cols(rg, slot, h_slot_encl[slot], len, -1) != 0) return -83;
        h_slot_encl[slot] = len;
    }
    return paged_t4_encode_ranges(rg, -1) != 0 ? -83 : 0;
}
// The step's device work once the block tables, slot ids and positions are on the device. graph_body = 0: the
// eager step (tokens from the host, one embed launch per column; the launch sequence of 9da2d3b); graph_body = 1:
// the sequence captured into a CUDA graph (TQ_GRAPH): the tokens come from g_tok_d in one launch, the MTP control
// rows were uploaded by the prework (mtp_m = its column count) -- every kernel then runs with the eager step's
// grid and parameters, so the replay is bitwise the eager step.
static int paged_decode_body(const int *tokens, const int *slot_ids_h, const int *positions, int N,
                             int dec_maxpos, int graph_body, int mtp_m) {
    int H = g_qwen.H, attn_m = g_qwen.nh * g_qwen.hd;
    int lkh = g_qwen.linear_num_key_heads, lvh = g_qwen.linear_num_value_heads, ld = g_qwen.linear_value_head_dim;
    int value_dim = lvh * ld, key_dim = lkh * ld, conv_dim = 2 * key_dim + value_dim;
    int recur_f = batched_recur_floats(), conv_f = batched_conv_floats();
    int ret;
    if (graph_body)
        k_tq_embed_lookup_nk<<<dim3((H + 255) / 256, N), 256, 0, g_qwen.stream>>>(g_wide_h, g_qwen.d_embed, g_tok_d, H);
    else
        for (int j = 0; j < N; j++)
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
                tq_launch(g_pdl_small, k_tq_qwen_rmsnorm_b, dim3(8, N), dim3(1024), 0, g_qwen.stream,
                          g_wide_norm, g_wide_h, l->d_input_ln, H, g_qwen.eps);
            if ((ret = wide_quant_input(g_wide_norm, H, N)) != 0) return -64;
            const tq_qmma_weight_t *qkv[3] = {&l->q_proj, &l->k_proj, &l->v_proj};
            float *qkv_out[3] = {g_wide_q, g_wide_k, g_wide_v};
            if ((ret = wide_proj_group(qkv, qkv_out, 3, N)) != 0) return -64;
            if (launch_paged_attn_q4(g_wide_core, g_wide_q, l->d_q_norm, g_wide_k, g_wide_v, l->d_k_norm,
                                     g_pool_k4[L], g_pool_kq4s[L], g_pool_v8[L], g_pool_vscale[L],
                                     g_wide_pos, g_slot_ids, g_block_table, g_pg_maxblk, g_pg_page, g_pg_plog,
                                     N, dec_maxpos, L, positions, slot_ids_h, 1, g_qwen.stream,
                                     g_pg_t4 ? L : -1) != 0) return -67;
            if ((ret = wide_quant_input(g_wide_core, attn_m, N)) != 0) return -68;
            if ((ret = wide_proj(&l->o_proj, g_wide_o, N)) != 0) return -68;
            if (fuse) {
                if ((ret = wide_mlp(l, g_wide_resid, g_wide_h, N, g_wide_h, g_wide_o, next_ln, g_wide_norm)) != 0) return -69;
                norm_ready = 1;
            } else {
                tq_launch(g_pdl_small, k_tq_add_vec, dim3(((size_t)N * H + 255) / 256), dim3(256), 0, g_qwen.stream,
                          g_wide_resid, g_wide_h, g_wide_o, (int)((size_t)N * H));
                if ((ret = wide_mlp(l, g_wide_resid, g_wide_h, N)) != 0) return -69;
            }
        } else {
            if (!(tq_wide_fmt_ok(&l->linear_in_qkv) && tq_wide_fmt_ok(&l->linear_in_z) &&
                  tq_wide_fmt_ok(&l->linear_in_b) && tq_wide_fmt_ok(&l->linear_in_a) &&
                  tq_wide_fmt_ok(&l->linear_out) && tq_wide_fmt_ok(&l->mlp_gate) &&
                  tq_wide_fmt_ok(&l->mlp_down))) return -70;
            if (!norm_ready)
                tq_launch(g_pdl_small, k_tq_qwen_rmsnorm_b, dim3(8, N), dim3(1024), 0, g_qwen.stream,
                          g_wide_norm, g_wide_h, l->d_input_ln, H, g_qwen.eps);
            if ((ret = wide_quant_input(g_wide_norm, H, N)) != 0) return -71;
            const tq_qmma_weight_t *qz[2] = {&l->linear_in_qkv, &l->linear_in_z};
            float *qz_out[2] = {g_wide_qkv, g_wide_z};
            if ((ret = wide_proj_group(qz, qz_out, 2, N)) != 0) return -71;
            if ((ret = wide_proj_pair(&l->linear_in_b, g_wide_b, &l->linear_in_a, g_wide_a, N)) != 0) return -73;
            tq_launch(g_pdl_small, k_tq_linear_conv_update_paged, dim3((conv_dim + 255) / 256, N), dim3(256), 0,
                      g_qwen.stream, g_wide_qkv, g_pg_conv[L], g_wide_qkv, l->d_linear_conv1d, conv_dim,
                      g_qwen.linear_conv_kernel_dim, g_slot_ids, conv_f);
            if (paged_linear_core(l, g_wide_lcore, g_pg_recur[L], g_wide_qkv, g_wide_z,
                                  g_wide_b, g_wide_a, g_slot_ids, g_iota, N, recur_f, conv_dim, value_dim, g_wide_pos) != 0)
                return -75;
            if ((ret = wide_quant_input(g_wide_lcore, value_dim, N)) != 0) return -76;
            if ((ret = wide_proj(&l->linear_out, g_wide_o, N)) != 0) return -76;
            if (fuse) {
                if ((ret = wide_mlp(l, g_wide_resid, g_wide_h, N, g_wide_h, g_wide_o, next_ln, g_wide_norm)) != 0) return -77;
                norm_ready = 1;
            } else {
                tq_launch(g_pdl_small, k_tq_add_vec, dim3(((size_t)N * H + 255) / 256), dim3(256), 0, g_qwen.stream,
                          g_wide_resid, g_wide_h, g_wide_o, (int)((size_t)N * H));
                if ((ret = wide_mlp(l, g_wide_resid, g_wide_h, N)) != 0) return -77;
            }
        }
    }
    if (!norm_ready)
        tq_launch(g_pdl_small, k_tq_qwen_rmsnorm_b, dim3(8, N), dim3(1024), 0, g_qwen.stream,
                  g_wide_norm, g_wide_h, g_qwen.d_norm, H, g_qwen.eps);
    if (paged_lm_head_wide()) {
        if ((ret = wide_quant_input(g_wide_norm, H, N)) != 0) return -78;
        if ((ret = wide_proj(&g_qwen.lm_head, g_pg_logits, N)) != 0) return -78;
        tq_launch(g_pdl_small, k_tq_batched_argmax, dim3(N), dim3(256), 0, g_qwen.stream,
                  g_pg_argmax, g_pg_logits, g_qwen.V, N);
        if (g_pg_sink) {
            if (N > g_pg_sink_cap) return -81;
            cudaMemcpyAsync(g_pg_sink, g_pg_logits, (size_t)N * g_qwen.V * sizeof(float),
                            cudaMemcpyDeviceToDevice, g_qwen.stream);
        }
    } else {
        for (int j = 0; j < N; j++) {
            cudaMemcpyAsync(g_qwen.d_debug_norm, g_wide_norm + (size_t)j * H, (size_t)H * sizeof(float),
                            cudaMemcpyDeviceToDevice, g_qwen.stream);
            if (run_lm_head_argmax_device() != 0) return -79;
            cudaMemcpyAsync(g_pg_argmax + j, g_qwen.d_debug_arg_ids + TQ_ARGMAX_BLOCKS, sizeof(int),
                            cudaMemcpyDeviceToDevice, g_qwen.stream);
        }
    }
    if (g_mtp_on) {                                         // MTP trunk slot pos-1
        if (graph_body) { if (mtp_hook_decode_kernels(mtp_m, N) != 0) return -82; }
        else if (mtp_hook_decode(tokens, slot_ids_h, positions, N) != 0) return -82;
    }
    return cudaGetLastError() == cudaSuccess ? 0 : -80;
}

// The eager decode step (the stream sequence of 9da2d3b: tables, slot ids, positions, N embed launches, the body,
// the MTP hook, the T4 tail).
static int run_paged_decode_step_core(const int *tokens, const int *slot_ids_h, const int *positions, int N) {
    int dec_maxpos = 0, ret;
    if ((ret = paged_decode_prep(slot_ids_h, positions, N, &dec_maxpos)) != 0) return ret;
    paged_upload_tables(g_qwen.stream);
    cudaMemcpyAsync(g_slot_ids, slot_ids_h, (size_t)N * sizeof(int), cudaMemcpyHostToDevice, g_qwen.stream);
    cudaMemcpyAsync(g_wide_pos, positions, (size_t)N * sizeof(int), cudaMemcpyHostToDevice, g_qwen.stream);
    if ((ret = paged_decode_body(tokens, slot_ids_h, positions, N, dec_maxpos, 0, 0)) != 0) return ret;
    return paged_decode_t4_tail(slot_ids_h, positions, N);
}
