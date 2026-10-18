// the single-stream forward: DeltaNet state snapshots, projection dispatch, decode and debug paths.
// Part of the forward_qwen.cu unity build: included in order, not a standalone header.

// ===================== DeltaNet state save/restore (Step 3) ==================
// The dense tree verify forks the DeltaNet (linear-attention) recurrent + conv state
// per branch. Full-attention layers fork implicitly through the causal KV cache +
// position (a branch writes slots pos+1.. and never reads stale deeper slots, like
// the Step-2 MTP tree), so only the DeltaNet single-state-per-layer needs explicit
// snapshot/restore. qwn_delta_state_save(slot) snapshots the CURRENT (committed)
// recurrent + conv state of every linear layer; qwn_delta_state_restore(slot) writes
// it back, so a branch can be replayed from the committed state via qwn_decode and a
// different branch restored afterward. ADDITIVE: never touched by the default decode.

static size_t delta_state_floats(void) {
    int lin_key_dim = g_qwen.linear_num_key_heads * g_qwen.linear_key_head_dim;
    int lin_value_dim = g_qwen.linear_num_value_heads * g_qwen.linear_value_head_dim;
    int lin_conv_dim = lin_key_dim * 2 + lin_value_dim;
    size_t conv_floats = (size_t)lin_conv_dim * g_qwen.linear_conv_kernel_dim;
    size_t recur_floats = (size_t)g_qwen.linear_num_value_heads *
                          g_qwen.linear_key_head_dim * g_qwen.linear_value_head_dim;
    size_t total = 0;
    for (int i = 0; i < g_qwen.L; i++)
        if (g_qwen.layer_types[i] == TQ_LAYER_LINEAR_ATTENTION)
            total += recur_floats + conv_floats;
    return total;
}

static int ensure_delta_snap(int slot) {
    if (slot < 0 || slot >= TQ_DELTA_SNAP_SLOTS) return -1;
    size_t floats = delta_state_floats();
    g_qwen.delta_snap_floats = floats;
    if (floats == 0) return 0;   // model has no linear-attention layers
    if (!g_qwen.d_delta_snap[slot]) {
        if (cudaMalloc(&g_qwen.d_delta_snap[slot], floats * sizeof(float)) != cudaSuccess) return -2;
        g_qwen.device_bytes += floats * sizeof(float);
    }
    return 0;
}

// dir 0 = save (live state -> snapshot slot), dir 1 = restore (slot -> live state).
static int delta_state_copy(int slot, int dir) {
    if (slot < 0 || slot >= TQ_DELTA_SNAP_SLOTS) return -1;
    if (!g_qwen.initialized) return -10;
    int ret = ensure_decode_state();
    if (ret != 0) return ret;
    ret = ensure_delta_snap(slot);
    if (ret != 0) return ret;
    if (!g_qwen.d_delta_snap[slot]) return 0;   // no linear layers -> nothing to do
    int lin_key_dim = g_qwen.linear_num_key_heads * g_qwen.linear_key_head_dim;
    int lin_value_dim = g_qwen.linear_num_value_heads * g_qwen.linear_value_head_dim;
    int lin_conv_dim = lin_key_dim * 2 + lin_value_dim;
    size_t conv_floats = (size_t)lin_conv_dim * g_qwen.linear_conv_kernel_dim;
    size_t recur_floats = (size_t)g_qwen.linear_num_value_heads *
                          g_qwen.linear_key_head_dim * g_qwen.linear_value_head_dim;
    float *snap = g_qwen.d_delta_snap[slot];
    size_t off = 0;
    for (int i = 0; i < g_qwen.L; i++) {
        if (g_qwen.layer_types[i] != TQ_LAYER_LINEAR_ATTENTION) continue;
        tq_layer_t *l = &g_qwen.layers[i];
        float *r_dst = (dir == 0) ? (snap + off) : l->d_linear_recurrent_state;
        float *r_src = (dir == 0) ? l->d_linear_recurrent_state : (snap + off);
        float *c_dst = (dir == 0) ? (snap + off + recur_floats) : l->d_linear_conv_state;
        float *c_src = (dir == 0) ? l->d_linear_conv_state : (snap + off + recur_floats);
        cudaMemcpyAsync(r_dst, r_src, recur_floats * sizeof(float),
                        cudaMemcpyDeviceToDevice, g_qwen.stream);
        cudaMemcpyAsync(c_dst, c_src, conv_floats * sizeof(float),
                        cudaMemcpyDeviceToDevice, g_qwen.stream);
        off += recur_floats + conv_floats;
    }
    cudaError_t err = cudaStreamSynchronize(g_qwen.stream);
    return err == cudaSuccess ? 0 : -3;
}

// Snapshot the current DeltaNet recurrent + conv state of all linear layers into a
// slot (the committed state a tree branch forks from). Returns 0 on success.
extern "C" int qwn_delta_state_save(int slot) { return delta_state_copy(slot, 0); }

// Restore the DeltaNet state previously saved into a slot. Returns 0 on success.
extern "C" int qwn_delta_state_restore(int slot) { return delta_state_copy(slot, 1); }

static int use_act_fp8_gemv(void) {
    static int cached = -1;
    if (cached < 0) cached = getenv("TQ_USE_ACT_FP8") != NULL && getenv("TQ_USE_DEQUANT_GEMV") == NULL;
    return cached;
}

static int use_fused_lm_head(void) {
    static int cached = -1;
    if (cached < 0) cached = getenv("TQ_FUSED_LM_HEAD") != NULL;
    return cached;
}

// Number of optimistic candidate tokens to extract at the lm_head (0 = off).
// k=1 is equivalent to plain argmax, so it is treated as off.
static int optimistic_topk(void) {
    static int cached = -2;
    if (cached == -2) {
        const char *e = getenv("TQ_OPTIMISTIC_TOPK");
        int k = e ? atoi(e) : 0;
        if (k <= 1) k = 0;
        if (k > TQ_OPTIMISTIC_TOPK_MAX) k = TQ_OPTIMISTIC_TOPK_MAX;
        cached = k;
    }
    return cached;
}

static int use_qmma_sf(void) {
    static int env_cached = -1;
    if (env_cached < 0) env_cached = getenv("TQ_USE_QMMA_SF") != NULL;
    // E2M3 (6-bit dense) weights are ONLY correct via the QMMA.SF path. The
    // non-SF block-scaled fallback (k_tq_debug_qmma_gemv_block_scaled) dequants
    // as E4M3 and addresses 512-byte tiles, so on 6-bit E2M3 payloads it reads
    // wrong bytes — silent garbage at small dims (0.8B "pains"), out-of-bounds
    // IMA at 27B dims. Force the SF path whenever the model is E2M3.
    return env_cached || (g_qwen.flags & TQ_FLAG_BLOCK_SCALED_E2M3) != 0;
}

static int qmma_sf_multi_warps_per_block(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("TQ_QMMA_SF_MULTI_WARPS");
        cached = env ? atoi(env) : 1;
        if (cached < 1) cached = 1;
        if (cached > 4) cached = 4;
    }
    return cached;
}

static int qmma_sf_multi_tiles_per_warp(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("TQ_QMMA_SF_MULTI_TILES_PER_WARP");
        cached = env ? atoi(env) : 1;
        if (cached != 2) cached = 1;
    }
    return cached;
}

static int qmma_sf_prefetch_distance(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("TQ_QMMA_SF_PREFETCH_DISTANCE");
        cached = env ? atoi(env) : 0;
        if (cached < 0) cached = 0;
        if (cached > 16) cached = 16;
    }
    return cached;
}

static int qmma_sf_k_split(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("TQ_QMMA_SF_K_SPLIT");
        cached = env ? atoi(env) : 0; // 0 => auto (per-projection)
        if (env && cached < 1) cached = 1;
        if (cached > 8) cached = 8;
    }
    return cached;
}

// Per-projection K-split for single (non-fused) QMMA.SF GEMVs. Small-Mt
// projections (o_proj/down_proj, Mt=320 => only ~2 blocks/SM unsplit) need more
// splits to fill the GPU, while large ones (lm_head, Mt=15520) need none and
// would only pay extra reduction cost. A fixed TQ_QMMA_SF_K_SPLIT overrides this.
static int qmma_sf_k_split_auto(int Mt, int Kt) {
    int fixed = qmma_sf_k_split();
    if (fixed >= 1) return fixed;
    int best = 1;
    for (int s = 8; s >= 1; s >>= 1) {
        if (Kt % s != 0) continue;       // split must divide the K-tile count
        if (Kt / s < 16) continue;       // keep enough K-work per block
        if ((long)Mt * s > 2560) continue; // ~15 blocks/SM ceiling on 170 SMs
        best = s;
        break;
    }
    return best;
}

// Thread-per-element parallelism factor for the linear-attention decode core.
// 0 selects the original one-block-per-head kernel; otherwise G threads split
// the contraction loop. Default 8 fully occupies the active SMs.
static int linear_core_par_g(void) {
    static int cached = -2;
    if (cached == -2) {
        const char *env = getenv("TQ_LINEAR_CORE_PAR");
        cached = env ? atoi(env) : 8;
        if (cached != 0 && cached != 2 && cached != 4 && cached != 8) cached = 8;
    }
    return cached;
}

// Head-split factor for the linear decode core: spreads each head over HS blocks
// so all SMs (not just value_heads of them) issue recurrent-state traffic.
// 1 disables head-split (single-kernel core_par path). Default 4 fills 170 SMs.
static int linear_core_hs(void) {
    static int cached = -2;
    if (cached == -2) {
        const char *env = getenv("TQ_LINEAR_CORE_HS");
        cached = env ? atoi(env) : 4;
        if (cached != 1 && cached != 2 && cached != 4 && cached != 8) cached = 4;
    }
    return cached;
}

static int qmma_sf_multi_k_split(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("TQ_QMMA_SF_MULTI_K_SPLIT");
        if (env) {
            cached = atoi(env);
        } else {
            // E2M3 weights need extra warps per fused GEMV to hide the 6-bit
            // unpack ALU (mid-grid kernels were ~55% of peak at split 1). For
            // E4M3 the fused GEMV is already bandwidth-bound, so keep 1.
            cached = (g_qwen.flags & TQ_FLAG_BLOCK_SCALED_E2M3) ? 8 : 1;
        }
        if (cached < 1) cached = 1;
        if (cached > 8) cached = 8;
    }
    return cached;
}

static int qmma_sf_multi_k_split_atomic(void) {
    static int cached = -1;
    if (cached < 0) cached = getenv("TQ_QMMA_SF_MULTI_K_SPLIT_ATOMIC") != NULL;
    return cached;
}

static int use_qmma_sf_multi_persistent(void) {
    static int cached = -1;
    if (cached < 0) cached = getenv("TQ_QMMA_SF_MULTI_PERSISTENT") != NULL;
    return cached;
}

static int qmma_sf_multi_persistent_blocks(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("TQ_QMMA_SF_MULTI_PERSISTENT_BLOCKS");
        cached = env ? atoi(env) : 680;
        if (cached < 1) cached = 1;
        if (cached > 4096) cached = 4096;
    }
    return cached;
}

// Optimistic sparse-FP4 draft: route calibrated linear-attn projections through
// the 2:4 sparse residual GEMV (full-precision activations via 2xe4m3). Disables
// projection fusion so every projection hits the per-projection sparse hook.
static int sparse_draft_enabled(void) {
    static int cached = -1;
    if (cached < 0) cached = getenv("TQ_SPARSE_DRAFT") != NULL;
    return cached;
}

// High-precision (residual 2xe4m3, ~BF16) activations on the DENSE path — used so
// the coverage reference matches the draft's activation precision (offline uses
// BF16 acts for both). Env-gated; default path unchanged.
static int dense_residual_enabled(void) {
    static int cached = -1;
    if (cached < 0) cached = getenv("TQ_DENSE_RESIDUAL") != NULL;
    return cached;
}

static int use_fused_residual_add(void) {
    static int cached = -1;
    if (cached < 0) cached = getenv("TQ_FUSE_RESIDUAL_ADD") != NULL;
    if (sparse_draft_enabled()) return 0;
    return cached;
}

static int ensure_qmma_work_counter(void) {
    if (g_qwen.d_qmma_work_counter) return 0;
    cudaError_t err = cudaMalloc(&g_qwen.d_qmma_work_counter, sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_qmma_work_counter failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    g_qwen.device_bytes += sizeof(int);
    return 0;
}

static int can_use_qmma_sf_weight(const tq_qmma_weight_t *w) {
    return use_qmma_sf() && w->block_scaled;
}

static int can_share_qmma_sf_input(const tq_qmma_weight_t *base, const tq_qmma_weight_t *w) {
    return can_use_qmma_sf_weight(base) && can_use_qmma_sf_weight(w) &&
           w->K == base->K && w->Kt == base->Kt;
}

// TQ_NO_FUSE forces the single (non-multi-fused) SF GEMV path so the per-token
// decode uses the SAME GEMV kernel the batched spec verify uses (the nk GEMV matches
// the single sf_prepared path bit-exactly, but NOT the fused multi path). This makes
// the spec path bit-identical to dense greedy for gate-a. Default (unset) is unchanged.
static int no_fuse_qmma(void) {
    static int cached = -1;
    if (cached < 0) cached = getenv("TQ_NO_FUSE") != NULL;
    return cached;
}

static int can_fuse_qmma_sf_tiled_input(const tq_qmma_weight_t *base, const tq_qmma_weight_t *w) {
    if (no_fuse_qmma()) return 0;
    if (sparse_draft_enabled() && (base->has_sparse || w->has_sparse)) return 0;
    return can_share_qmma_sf_input(base, w) && !base->row_major && !w->row_major &&
           base->word_major == w->word_major;
}

static int prepare_qmma_sf_input(const tq_qmma_weight_t *w, const float *input) {
    int ret = ensure_qmma_debug_scratch(w);
    if (ret != 0) return ret;
    int act_scale_count = (w->K + 127) / 128;
    if (ensure_float_buffer(&g_qwen.d_debug_act_scale_inv, &g_qwen.debug_act_scale_count,
                            act_scale_count, "d_debug_act_scale_inv") != 0) return -20;
    k_tq_quantize_bfrag_pow2_fused<<<act_scale_count, 128, 0, g_qwen.stream>>>(
        g_qwen.d_debug_b, g_qwen.d_debug_act_scale_inv, input, w->K);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "QMMA.SF input quantize launch failed: %s\n", cudaGetErrorString(err));
        return -21;
    }
    return 0;
}

static int prepare_qmma_sf_input_silu_mul(const tq_qmma_weight_t *w, const float *gate, const float *up) {
    int ret = ensure_qmma_debug_scratch(w);
    if (ret != 0) return ret;
    int act_scale_count = (w->K + 127) / 128;
    if (ensure_float_buffer(&g_qwen.d_debug_act_scale_inv, &g_qwen.debug_act_scale_count,
                            act_scale_count, "d_debug_act_scale_inv") != 0) return -20;
    k_tq_silu_mul_quantize_bfrag_pow2_fused<<<act_scale_count, 128, 0, g_qwen.stream>>>(
        g_qwen.d_debug_b, g_qwen.d_debug_act_scale_inv, gate, up, w->K);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "QMMA.SF silu input quantize launch failed: %s\n", cudaGetErrorString(err));
        return -21;
    }
    return 0;
}

__global__ __launch_bounds__(128, 1)
void k_tq_qmma_sf_gemv_tiled_multi_tile2(float *out0, const uint32_t *a0, int mt0,
                                         const float *scale0, int scale_cols0,
                                         float *out1, const uint32_t *a1, int mt1,
                                         const float *scale1, int scale_cols1,
                                         float *out2, const uint32_t *a2, int mt2,
                                         const float *scale2, int scale_cols2,
                                         float *out3, const uint32_t *a3, int mt3,
                                         const float *scale3, int scale_cols3,
                                         const uint32_t *b, int Kt,
                                         const float *act_scale_inv,
                                         int warps_per_block, int word_major,
                                         int prefetch_distance) {
    int warp = threadIdx.x >> 5;
    if (warp >= warps_per_block) return;
    int pair = blockIdx.x * warps_per_block + warp;
    int pairs0 = (mt0 + 1) >> 1;
    int pairs1 = (mt1 + 1) >> 1;
    int pairs2 = (mt2 + 1) >> 1;
    int tile = pair * 2;
    if (tile < mt0) {
        if (tile + 1 < mt0) {
            tq_qmma_sf_gemv_tiled_tile_pair(out0, a0, b, Kt, scale0, scale_cols0,
                                            act_scale_inv, tile, word_major, prefetch_distance);
        } else {
            tq_qmma_sf_gemv_tiled_tile(out0, a0, b, Kt, scale0, scale_cols0,
                                       act_scale_inv, tile, word_major, prefetch_distance, 0);
        }
        return;
    }
    pair -= pairs0;
    tile = pair * 2;
    if (tile < mt1) {
        if (tile + 1 < mt1) {
            tq_qmma_sf_gemv_tiled_tile_pair(out1, a1, b, Kt, scale1, scale_cols1,
                                            act_scale_inv, tile, word_major, prefetch_distance);
        } else {
            tq_qmma_sf_gemv_tiled_tile(out1, a1, b, Kt, scale1, scale_cols1,
                                       act_scale_inv, tile, word_major, prefetch_distance, 0);
        }
        return;
    }
    pair -= pairs1;
    tile = pair * 2;
    if (tile < mt2) {
        if (tile + 1 < mt2) {
            tq_qmma_sf_gemv_tiled_tile_pair(out2, a2, b, Kt, scale2, scale_cols2,
                                            act_scale_inv, tile, word_major, prefetch_distance);
        } else {
            tq_qmma_sf_gemv_tiled_tile(out2, a2, b, Kt, scale2, scale_cols2,
                                       act_scale_inv, tile, word_major, prefetch_distance, 0);
        }
        return;
    }
    pair -= pairs2;
    tile = pair * 2;
    if (tile < mt3) {
        if (tile + 1 < mt3) {
            tq_qmma_sf_gemv_tiled_tile_pair(out3, a3, b, Kt, scale3, scale_cols3,
                                            act_scale_inv, tile, word_major, prefetch_distance);
        } else {
            tq_qmma_sf_gemv_tiled_tile(out3, a3, b, Kt, scale3, scale_cols3,
                                       act_scale_inv, tile, word_major, prefetch_distance, 0);
        }
    }
}

static int run_qmma_sf_prepared_multi_to_buffers(const tq_qmma_weight_t *w0, float *dst0, int dst_count0,
                                                 const tq_qmma_weight_t *w1, float *dst1, int dst_count1,
                                                 const tq_qmma_weight_t *w2, float *dst2, int dst_count2,
                                                 const tq_qmma_weight_t *w3, float *dst3, int dst_count3) {
    if (!w0 || !w1 || !dst0 || !dst1) return -30;
    if (!can_fuse_qmma_sf_tiled_input(w0, w1)) return -31;
    if (w2 && !can_fuse_qmma_sf_tiled_input(w0, w2)) return -32;
    if (w3 && !can_fuse_qmma_sf_tiled_input(w0, w3)) return -33;
    // One a_byte flag drives all four matrices of a fused launch; TQ_W_BYTE
    // converts whole classes so groups stay uniform -- enforce it.
    if (w1->e2m3_byte != w0->e2m3_byte || (w2 && w2->e2m3_byte != w0->e2m3_byte) ||
        (w3 && w3->e2m3_byte != w0->e2m3_byte)) {
        fprintf(stderr, "QMMA.SF multi: mixed e2m3_byte layouts in fused group\n");
        return -36;
    }
    if (dst_count0 < w0->M || dst_count1 < w1->M ||
        (w2 && dst_count2 < w2->M) || (w3 && dst_count3 < w3->M)) {
        return -34;
    }
    int act_scale_count = (w0->K + 127) / 128;
    if (!g_qwen.d_debug_b || g_qwen.debug_b_bytes < w0->Kt * 256 ||
        !g_qwen.d_debug_act_scale_inv || g_qwen.debug_act_scale_count < act_scale_count) {
        fprintf(stderr, "QMMA.SF prepared multi input is missing or too small\n");
        return -35;
    }

    int mt2 = w2 ? w2->Mt : 0;
    int mt3 = w3 ? w3->Mt : 0;
    int total_mt = w0->Mt + w1->Mt + mt2 + mt3;
    int warps_per_block = qmma_sf_multi_warps_per_block();
    int tiles_per_warp = qmma_sf_multi_tiles_per_warp();
    int tile2_work = ((w0->Mt + 1) >> 1) + ((w1->Mt + 1) >> 1) +
                     ((mt2 + 1) >> 1) + ((mt3 + 1) >> 1);
    int blocks = tiles_per_warp == 2
        ? (tile2_work + warps_per_block - 1) / warps_per_block
        : (total_mt + warps_per_block - 1) / warps_per_block;
    int k_splits = qmma_sf_multi_k_split();
    int total_m = w0->M + w1->M + (w2 ? w2->M : 0) + (w3 ? w3->M : 0);
    int prefetch_distance = qmma_sf_prefetch_distance();
    if (use_qmma_sf_multi_persistent() && k_splits == 1) {
        if (tiles_per_warp != 1) return -39;
        int ret_counter = ensure_qmma_work_counter();
        if (ret_counter != 0) return -38;
        cudaMemsetAsync(g_qwen.d_qmma_work_counter, 0, sizeof(int), g_qwen.stream);
        int pblocks = qmma_sf_multi_persistent_blocks();
        k_tq_qmma_sf_gemv_tiled_multi_persistent<<<pblocks, warps_per_block * 32, 0, g_qwen.stream>>>(
            dst0, (const uint32_t *)w0->d_A, w0->Mt, w0->d_block_scale_inv, w0->scale_cols,
            dst1, (const uint32_t *)w1->d_A, w1->Mt, w1->d_block_scale_inv, w1->scale_cols,
            w2 ? dst2 : dst0, w2 ? (const uint32_t *)w2->d_A : (const uint32_t *)w0->d_A, mt2,
            w2 ? w2->d_block_scale_inv : w0->d_block_scale_inv, w2 ? w2->scale_cols : w0->scale_cols,
            w3 ? dst3 : dst0, w3 ? (const uint32_t *)w3->d_A : (const uint32_t *)w0->d_A, mt3,
            w3 ? w3->d_block_scale_inv : w0->d_block_scale_inv, w3 ? w3->scale_cols : w0->scale_cols,
            (const uint32_t *)g_qwen.d_debug_b, w0->Kt, g_qwen.d_debug_act_scale_inv,
            warps_per_block, total_mt, w0->word_major, w0->e2m3_byte,
            g_qwen.d_qmma_work_counter);
    } else if (k_splits > 1 && (w0->Kt % k_splits) == 0) {
        if (tiles_per_warp != 1) return -39;
        if (qmma_sf_multi_k_split_atomic()) {
            cudaMemsetAsync(dst0, 0, (size_t)w0->M * sizeof(float), g_qwen.stream);
            cudaMemsetAsync(dst1, 0, (size_t)w1->M * sizeof(float), g_qwen.stream);
            if (w2) cudaMemsetAsync(dst2, 0, (size_t)w2->M * sizeof(float), g_qwen.stream);
            if (w3) cudaMemsetAsync(dst3, 0, (size_t)w3->M * sizeof(float), g_qwen.stream);
            k_tq_qmma_sf_gemv_tiled_multi_split_atomic<<<dim3(blocks, k_splits, 1), warps_per_block * 32, 0, g_qwen.stream>>>(
                dst0, (const uint32_t *)w0->d_A, w0->Mt, w0->d_block_scale_inv, w0->scale_cols,
                dst1, (const uint32_t *)w1->d_A, w1->Mt, w1->d_block_scale_inv, w1->scale_cols,
                w2 ? dst2 : dst0, w2 ? (const uint32_t *)w2->d_A : (const uint32_t *)w0->d_A, mt2,
                w2 ? w2->d_block_scale_inv : w0->d_block_scale_inv, w2 ? w2->scale_cols : w0->scale_cols,
                w3 ? dst3 : dst0, w3 ? (const uint32_t *)w3->d_A : (const uint32_t *)w0->d_A, mt3,
                w3 ? w3->d_block_scale_inv : w0->d_block_scale_inv, w3 ? w3->scale_cols : w0->scale_cols,
                (const uint32_t *)g_qwen.d_debug_b, w0->Kt, g_qwen.d_debug_act_scale_inv,
                warps_per_block, k_splits, w0->word_major);
        } else {
            if (ensure_float_buffer(&g_qwen.d_debug_qmma_raw, &g_qwen.debug_raw_count,
                                    total_m * k_splits, "d_debug_qmma_raw") != 0) return -37;
            k_tq_qmma_sf_gemv_tiled_multi_split<<<dim3(blocks, k_splits, 1), warps_per_block * 32, 0, g_qwen.stream>>>(
                g_qwen.d_debug_qmma_raw,
                (const uint32_t *)w0->d_A, w0->Mt, w0->d_block_scale_inv, w0->scale_cols,
                (const uint32_t *)w1->d_A, w1->Mt, w1->d_block_scale_inv, w1->scale_cols,
                w2 ? (const uint32_t *)w2->d_A : (const uint32_t *)w0->d_A, mt2,
                w2 ? w2->d_block_scale_inv : w0->d_block_scale_inv, w2 ? w2->scale_cols : w0->scale_cols,
                w3 ? (const uint32_t *)w3->d_A : (const uint32_t *)w0->d_A, mt3,
                w3 ? w3->d_block_scale_inv : w0->d_block_scale_inv, w3 ? w3->scale_cols : w0->scale_cols,
                (const uint32_t *)g_qwen.d_debug_b, w0->Kt, g_qwen.d_debug_act_scale_inv,
                warps_per_block, k_splits, total_m, w0->word_major, w0->e2m1 ? 1 : (w0->e2m3_byte ? 2 : 0));
            k_tq_reduce_qmma_sf_multi_splits<<<(total_m + 255) / 256, 256, 0, g_qwen.stream>>>(
                dst0, w0->M, dst1, w1->M, w2 ? dst2 : dst0, w2 ? w2->M : 0,
                w3 ? dst3 : dst0, w3 ? w3->M : 0, g_qwen.d_debug_qmma_raw, total_m, k_splits);
        }
    } else if (tiles_per_warp == 2) {
        k_tq_qmma_sf_gemv_tiled_multi_tile2<<<blocks, warps_per_block * 32, 0, g_qwen.stream>>>(
            dst0, (const uint32_t *)w0->d_A, w0->Mt, w0->d_block_scale_inv, w0->scale_cols,
            dst1, (const uint32_t *)w1->d_A, w1->Mt, w1->d_block_scale_inv, w1->scale_cols,
            w2 ? dst2 : dst0, w2 ? (const uint32_t *)w2->d_A : (const uint32_t *)w0->d_A, mt2,
            w2 ? w2->d_block_scale_inv : w0->d_block_scale_inv, w2 ? w2->scale_cols : w0->scale_cols,
            w3 ? dst3 : dst0, w3 ? (const uint32_t *)w3->d_A : (const uint32_t *)w0->d_A, mt3,
            w3 ? w3->d_block_scale_inv : w0->d_block_scale_inv, w3 ? w3->scale_cols : w0->scale_cols,
            (const uint32_t *)g_qwen.d_debug_b, w0->Kt, g_qwen.d_debug_act_scale_inv,
            warps_per_block, w0->word_major, prefetch_distance);
    } else {
        k_tq_qmma_sf_gemv_tiled_multi<<<blocks, warps_per_block * 32, 0, g_qwen.stream>>>(
            dst0, (const uint32_t *)w0->d_A, w0->Mt, w0->d_block_scale_inv, w0->scale_cols,
            dst1, (const uint32_t *)w1->d_A, w1->Mt, w1->d_block_scale_inv, w1->scale_cols,
            w2 ? dst2 : dst0, w2 ? (const uint32_t *)w2->d_A : (const uint32_t *)w0->d_A, mt2,
            w2 ? w2->d_block_scale_inv : w0->d_block_scale_inv, w2 ? w2->scale_cols : w0->scale_cols,
            w3 ? dst3 : dst0, w3 ? (const uint32_t *)w3->d_A : (const uint32_t *)w0->d_A, mt3,
            w3 ? w3->d_block_scale_inv : w0->d_block_scale_inv, w3 ? w3->scale_cols : w0->scale_cols,
            (const uint32_t *)g_qwen.d_debug_b, w0->Kt, g_qwen.d_debug_act_scale_inv,
            warps_per_block, w0->word_major, prefetch_distance, w0->e2m1 ? 1 : (w0->e2m3_byte ? 2 : 0));
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "QMMA.SF multi GEMV launch failed: %s\n", cudaGetErrorString(err));
        return -36;
    }
    return 0;
}

static int run_qmma_sf_prepared_to_debug_out(const tq_qmma_weight_t *w) {
    int ret = ensure_qmma_debug_scratch(w);
    if (ret != 0) return ret;
    int act_scale_count = (w->K + 127) / 128;
    if (!g_qwen.d_debug_b || g_qwen.debug_b_bytes < w->Kt * 256 ||
        !g_qwen.d_debug_act_scale_inv || g_qwen.debug_act_scale_count < act_scale_count) {
        fprintf(stderr, "QMMA.SF prepared input is missing or too small\n");
        return -22;
    }
    if (w->row_major) {
        k_tq_qmma_sf_gemv_rowmajor<<<w->Mt, 32, 0, g_qwen.stream>>>(
            g_qwen.d_debug_qmma_out, w->d_A, g_qwen.d_debug_b, w->K, w->Kt,
            w->d_block_scale_inv, w->scale_cols, g_qwen.d_debug_act_scale_inv);
    } else {
        int k_splits = qmma_sf_k_split_auto(w->Mt, w->Kt);
        if (k_splits > 1 && (w->Kt % k_splits) == 0 &&
            g_qwen.debug_raw_count >= w->M * k_splits) {
            k_tq_qmma_sf_gemv_tiled_split<<<dim3(w->Mt, k_splits, 1), 32, 0, g_qwen.stream>>>(
                g_qwen.d_debug_qmma_raw, (const uint32_t *)w->d_A,
                (const uint32_t *)g_qwen.d_debug_b, w->M, w->Kt,
                w->d_block_scale_inv, w->scale_cols, g_qwen.d_debug_act_scale_inv,
                k_splits, w->word_major, w->e2m1 ? 1 : (w->e2m3_byte ? 2 : 0));
            k_tq_reduce_qmma_sf_splits<<<(w->M + 255) / 256, 256, 0, g_qwen.stream>>>(
                g_qwen.d_debug_qmma_out, g_qwen.d_debug_qmma_raw, w->M, k_splits);
        } else {
            k_tq_qmma_sf_gemv_tiled<<<w->Mt, 32, 0, g_qwen.stream>>>(
                g_qwen.d_debug_qmma_out, (const uint32_t *)w->d_A, (const uint32_t *)g_qwen.d_debug_b,
                w->Kt, w->d_block_scale_inv, w->scale_cols, g_qwen.d_debug_act_scale_inv,
                w->word_major, 0, w->e2m1 ? 1 : (w->e2m3_byte ? 2 : 0));
        }
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "QMMA.SF GEMV launch failed: %s\n", cudaGetErrorString(err));
        return -23;
    }
    return 0;
}

static int run_qmma_projection_from_input_to_debug_out(const tq_qmma_weight_t *w, const float *input);

static int run_qmma_sf_prepared_to_buffer_add(const tq_qmma_weight_t *w, const float *residual,
                                              float *dst, int dst_count) {
    if (dst_count < w->M) return -30;
    if (!can_use_qmma_sf_weight(w) || w->row_major) return -31;
    int ret = ensure_qmma_debug_scratch(w);
    if (ret != 0) return ret;
    int act_scale_count = (w->K + 127) / 128;
    if (!g_qwen.d_debug_b || g_qwen.debug_b_bytes < w->Kt * 256 ||
        !g_qwen.d_debug_act_scale_inv || g_qwen.debug_act_scale_count < act_scale_count) {
        return -32;
    }
    int k_splits = qmma_sf_k_split_auto(w->Mt, w->Kt);
    if (k_splits > 1 && (w->Kt % k_splits) == 0 &&
        g_qwen.debug_raw_count >= w->M * k_splits) {
        k_tq_qmma_sf_gemv_tiled_split<<<dim3(w->Mt, k_splits, 1), 32, 0, g_qwen.stream>>>(
            g_qwen.d_debug_qmma_raw, (const uint32_t *)w->d_A,
            (const uint32_t *)g_qwen.d_debug_b, w->M, w->Kt,
            w->d_block_scale_inv, w->scale_cols, g_qwen.d_debug_act_scale_inv,
            k_splits, w->word_major, w->e2m1 ? 1 : (w->e2m3_byte ? 2 : 0));
        k_tq_reduce_qmma_sf_splits_add<<<(w->M + 255) / 256, 256, 0, g_qwen.stream>>>(
            dst, g_qwen.d_debug_qmma_raw, residual, w->M, k_splits);
    } else {
        ret = run_qmma_sf_prepared_to_debug_out(w);
        if (ret != 0) return ret;
        k_tq_add_vec<<<(w->M + 255) / 256, 256, 0, g_qwen.stream>>>(
            dst, residual, g_qwen.d_debug_qmma_out, w->M);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "QMMA.SF add projection launch failed: %s\n", cudaGetErrorString(err));
        return -33;
    }
    return 0;
}

static int run_qmma_projection_from_input_add_to_buffer(const tq_qmma_weight_t *w, const float *input,
                                                        const float *residual, float *dst,
                                                        int dst_count) {
    if (use_fused_residual_add() && can_use_qmma_sf_weight(w) && !w->row_major) {
        int ret = prepare_qmma_sf_input(w, input);
        if (ret != 0) return ret;
        ret = run_qmma_sf_prepared_to_buffer_add(w, residual, dst, dst_count);
        if (ret == 0) return 0;
    }
    int ret = run_qmma_projection_from_input_to_debug_out(w, input);
    if (ret != 0) return ret;
    if (dst_count < w->M) return -3;
    k_tq_add_vec<<<(w->M + 255) / 256, 256, 0, g_qwen.stream>>>(
        dst, residual, g_qwen.d_debug_qmma_out, w->M);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -4;
    return 0;
}

// Sparse-FP4 draft GEMV into d_debug_qmma_out: residual 2xe4m3 activation (full
// precision) on the 2:4 kept weights. Two MMA passes (hi, lo) on the same weights.
static int run_sparse_draft_to_debug_out(const tq_qmma_weight_t *w, const float *input) {
    int ret = ensure_qmma_debug_scratch(w);
    if (ret != 0) return ret;
    int Kt64 = w->K / 64;
    int sc = w->scale_cols;
    int bcount = Kt64 * 128;  // uint32 per hi/lo buffer
    if (ensure_float_buffer(&g_qwen.d_sparse_b_hi, &g_qwen.sparse_bhi_count, bcount, "d_sparse_b_hi") != 0) return -40;
    if (ensure_float_buffer(&g_qwen.d_sparse_b_lo, &g_qwen.sparse_blo_count, bcount, "d_sparse_b_lo") != 0) return -40;
    if (ensure_float_buffer(&g_qwen.d_sparse_ascale, &g_qwen.sparse_ascale_count, sc, "d_sparse_ascale") != 0) return -40;
    if (ensure_float_buffer(&g_qwen.d_sparse_out_lo, &g_qwen.sparse_outlo_count, w->M, "d_sparse_out_lo") != 0) return -40;
    cudaMemsetAsync(g_qwen.d_sparse_b_hi, 0, (size_t)bcount * 4, g_qwen.stream);
    cudaMemsetAsync(g_qwen.d_sparse_b_lo, 0, (size_t)bcount * 4, g_qwen.stream);
    k_tq_pack_b_k64_residual_scaled<<<sc, 128, 0, g_qwen.stream>>>(
        (uint8_t *)g_qwen.d_sparse_b_hi, (uint8_t *)g_qwen.d_sparse_b_lo,
        g_qwen.d_sparse_ascale, input, w->K);
    k_tq_sparse_gemv_e2m3<<<w->Mt, 32, 0, g_qwen.stream>>>(
        g_qwen.d_debug_qmma_out, w->d_sparse_a, (const uint32_t *)g_qwen.d_sparse_b_hi,
        w->d_sparse_meta, w->d_block_scale_inv, g_qwen.d_sparse_ascale, w->M, Kt64, sc);
    // Residual lo pass (full-precision activation). TQ_SPARSE_NO_RESIDUAL=1 skips it
    // (single-e4m3 draft) — used to A/B the activation-precision effect on coverage.
    static int no_resid = -1;
    if (no_resid < 0) no_resid = getenv("TQ_SPARSE_NO_RESIDUAL") != NULL;
    if (!no_resid) {
        k_tq_sparse_gemv_e2m3<<<w->Mt, 32, 0, g_qwen.stream>>>(
            g_qwen.d_sparse_out_lo, w->d_sparse_a, (const uint32_t *)g_qwen.d_sparse_b_lo,
            w->d_sparse_meta, w->d_block_scale_inv, g_qwen.d_sparse_ascale, w->M, Kt64, sc);
        k_tq_add_vec<<<(w->M + 255) / 256, 256, 0, g_qwen.stream>>>(
            g_qwen.d_debug_qmma_out, g_qwen.d_debug_qmma_out, g_qwen.d_sparse_out_lo, w->M);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "sparse draft GEMV launch failed: %s\n", cudaGetErrorString(err));
        return -41;
    }
    return 0;
}

// Dense E2M3 GEMV with residual 2xe4m3 (~BF16) activation: hi pass + lo pass on
// the SAME dense weights, summed. The high-precision reference for coverage.
static int run_dense_residual_to_debug_out(const tq_qmma_weight_t *w, const float *input) {
    int ret = prepare_qmma_sf_input(w, input);  // d_debug_b = hi codes, scale set
    if (ret != 0) return ret;
    ret = run_qmma_sf_prepared_to_debug_out(w);  // d_debug_qmma_out = out_hi
    if (ret != 0) return ret;
    if (ensure_float_buffer(&g_qwen.d_sparse_out_lo, &g_qwen.sparse_outlo_count, w->M, "d_sparse_out_lo") != 0) return -42;
    cudaMemcpyAsync(g_qwen.d_sparse_out_lo, g_qwen.d_debug_qmma_out, (size_t)w->M * sizeof(float),
                    cudaMemcpyDeviceToDevice, g_qwen.stream);
    k_tq_quantize_bfrag_residual_lo<<<(w->K + 255) / 256, 256, 0, g_qwen.stream>>>(
        g_qwen.d_debug_b, input, w->K, g_qwen.d_debug_act_scale_inv);  // d_debug_b = lo codes
    ret = run_qmma_sf_prepared_to_debug_out(w);  // d_debug_qmma_out = out_lo
    if (ret != 0) return ret;
    k_tq_add_vec<<<(w->M + 255) / 256, 256, 0, g_qwen.stream>>>(
        g_qwen.d_debug_qmma_out, g_qwen.d_debug_qmma_out, g_qwen.d_sparse_out_lo, w->M);
    return 0;
}

static int run_qmma_projection_from_input_to_debug_out(const tq_qmma_weight_t *w, const float *input) {
    int ret = ensure_qmma_debug_scratch(w);
    if (ret != 0) return ret;
    if (sparse_draft_enabled() && w->has_sparse) {
        return run_sparse_draft_to_debug_out(w, input);
    }
    if (dense_residual_enabled() && can_use_qmma_sf_weight(w) && !w->row_major) {
        return run_dense_residual_to_debug_out(w, input);
    }
    if (can_use_qmma_sf_weight(w)) {
        ret = prepare_qmma_sf_input(w, input);
        if (ret != 0) return ret;
        return run_qmma_sf_prepared_to_debug_out(w);
    }
    if (w->block_scaled) {
        if (w->row_major) {
            k_tq_debug_qmma_gemv_block_scaled_rowmajor<<<w->M, 256, 0, g_qwen.stream>>>(
                g_qwen.d_debug_qmma_out, w->d_A, w->M, w->K,
                w->d_block_scale_inv, w->scale_cols, input);
        } else {
            k_tq_debug_qmma_gemv_block_scaled<<<w->M, 256, 0, g_qwen.stream>>>(
                g_qwen.d_debug_qmma_out, w->d_A, w->M, w->K, w->Kt,
                w->d_block_scale_inv, w->scale_cols, input);
        }
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "block-scaled dequant GEMV launch failed: %s\n", cudaGetErrorString(err));
            return -2;
        }
        return 0;
    }
    if (!use_act_fp8_gemv()) {
        k_tq_debug_qmma_gemv<<<w->M, 256, 0, g_qwen.stream>>>(
            g_qwen.d_debug_qmma_out, w->d_A, w->M, w->K, w->Kt, w->inv_scale, input);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "dequant GEMV launch failed: %s\n", cudaGetErrorString(err));
            return -2;
        }
        return 0;
    }
    k_tq_absmax_scale<<<1, 256, 0, g_qwen.stream>>>(g_qwen.d_debug_scale, input, w->K);
    cudaMemsetAsync(g_qwen.d_debug_b, 0, (size_t)w->Kt * 256, g_qwen.stream);
    k_tq_quantize_bfrag<<<(w->K + 255) / 256, 256, 0, g_qwen.stream>>>(g_qwen.d_debug_b, input,
                                                     w->K, g_qwen.d_debug_scale);
    cudaMemsetAsync(g_qwen.d_debug_qmma_raw, 0, (size_t)w->Mt * 128 * sizeof(float), g_qwen.stream);
    if (w->Kt == 32) {
        k_tq_qmma_gemv_k1024<<<w->Mt, 32, 0, g_qwen.stream>>>(
            g_qwen.d_debug_qmma_raw, (const uint32_t *)w->d_A,
            (const uint32_t *)g_qwen.d_debug_b);
    } else {
        k_tq_qmma_gemv_fp8<<<w->Mt, 32, 0, g_qwen.stream>>>(
            g_qwen.d_debug_qmma_raw, (const uint32_t *)w->d_A,
            (const uint32_t *)g_qwen.d_debug_b, w->Kt);
    }
    k_tq_extract_descale<<<(w->M + 255) / 256, 256, 0, g_qwen.stream>>>(g_qwen.d_debug_qmma_out,
                                                      g_qwen.d_debug_qmma_raw, w->M,
                                                      w->inv_scale, g_qwen.d_debug_scale);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "QMMA projection debug launch failed: %s\n", cudaGetErrorString(err));
        return -2;
    }
    return 0;
}

static int run_qmma_projection_to_debug_out(const tq_qmma_weight_t *w) {
    return run_qmma_projection_from_input_to_debug_out(w, g_qwen.d_debug_norm);
}

static int run_qmma_projection_from_input_to_buffer(const tq_qmma_weight_t *w, const float *input,
                                                    float *dst, int dst_count) {
    int ret = run_qmma_projection_from_input_to_debug_out(w, input);
    if (ret != 0) return ret;
    if (dst_count < w->M) return -3;
    cudaError_t err = cudaMemcpyAsync(dst, g_qwen.d_debug_qmma_out, (size_t)w->M * sizeof(float),
                                      cudaMemcpyDeviceToDevice, g_qwen.stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "projection device copy failed: %s\n", cudaGetErrorString(err));
        return -4;
    }
    return 0;
}

// Dispatch the N=k batched GEMV with optional K-split (Step 5 sub-step 1). When
// k_splits>1 (and divides Kt), the K dimension is split across blockIdx.y warps
// (partials -> reduce), so large-K / small-Mt weights (mlp_down/o_proj) fill the GPU
// instead of one warp serially streaming all of K. out is [nvar*M] col-major;
// partials (>= k_splits*nvar*M floats) is required when k_splits>1.
// pc_factors != NULL folds the nk post-correction (divide column col by
// pc_factors[col]) into the split reduction (k-split path only; the caller must
// run k_tq_nk_postcorrect itself on the non-split path).
static int run_nk_gemv(float *out, float *partials, const tq_qmma_weight_t *w,
                       const uint8_t *b, const float *act_scale, int nvar, int k_splits,
                       const float *pc_factors) {
    int M = w->M, Kt = w->Kt, Mt = w->Mt;
    if (k_splits > 1 && (Kt % k_splits) == 0 && partials) {
        k_tq_qmma_sf_gemv_tiled_nk_split<<<dim3(Mt, k_splits), 32, 0, g_qwen.stream>>>(
            partials, (const uint32_t *)w->d_A, (const uint32_t *)b, M, Kt,
            w->d_block_scale_inv, w->scale_cols, act_scale, nvar, k_splits,
            w->e2m1 ? 1 : (w->e2m3_byte ? 2 : 0));
        if (pc_factors)
            k_tq_reduce_qmma_sf_splits_pc<<<((size_t)nvar * M + 255) / 256, 256, 0, g_qwen.stream>>>(
                out, partials, pc_factors, M, nvar, k_splits);
        else
            k_tq_reduce_qmma_sf_splits<<<((size_t)nvar * M + 255) / 256, 256, 0, g_qwen.stream>>>(
                out, partials, nvar * M, k_splits);
    } else {
        k_tq_qmma_sf_gemv_tiled_nk<<<Mt, 32, 0, g_qwen.stream>>>(
            out, (const uint32_t *)w->d_A, (const uint32_t *)b, M, Kt,
            w->d_block_scale_inv, w->scale_cols, act_scale, nvar,
            w->e2m1 ? 1 : (w->e2m3_byte ? 2 : 0));
    }
    return 0;
}

// Step-5 feasibility probe: does the validated N=k batched GEMV (nvar=1) produce
// BIT-IDENTICAL output to the per-token SF GEMV path that qwn_decode uses? The
// batched verify forward relies on the nk GEMV; if its accumulation order differs
// from the per-token path (k-split / persistent / fused), the verify argmax can
// flip on near-ties and break the token-perfect gate. which: 0 = lm_head, 1 = the
// first full-attention layer's q_proj (both K=H, fed the current d_debug_norm).
extern "C" int qwn_gemv_nk_check(int which, float *max_abs_diff, double *rel,
                                 int *argmax_pertoken, int *argmax_nk) {
    if (!g_qwen.initialized || !g_qwen.d_debug_norm) return -1;
    const tq_qmma_weight_t *w = NULL;
    if (which == 0) {
        w = &g_qwen.lm_head;
    } else if (which == 1) {
        for (int i = 0; i < g_qwen.L; i++)
            if (g_qwen.layer_types[i] == TQ_LAYER_FULL_ATTENTION) { w = &g_qwen.layers[i].q_proj; break; }
    } else if (which == 2) {
        for (int i = 0; i < g_qwen.L; i++)
            if (g_qwen.layer_types[i] == TQ_LAYER_LINEAR_ATTENTION) { w = &g_qwen.layers[i].linear_in_qkv; break; }
    } else {
        for (int i = 0; i < g_qwen.L; i++)
            if (g_qwen.layer_types[i] == TQ_LAYER_LINEAR_ATTENTION) { w = &g_qwen.layers[i].mlp_gate; break; }
    }
    if (!w || !w->d_A) return -2;
    fprintf(stderr, "[nk_check which=%d] M=%d K=%d Kt=%d row_major=%d word_major=%d block_scaled=%d e2m3=%d\n",
            which, w->M, w->K, w->Kt, w->row_major, w->word_major, w->block_scaled, w->e2m3);
    if (!can_use_qmma_sf_weight(w) || w->row_major) return -3;   // nk path is column-major E2M3 SF
    int K = w->K, M = w->M, Kt = w->Kt;
    // per-token path (exactly what qwn_decode uses for this weight).
    int ret = run_qmma_projection_from_input_to_debug_out(w, g_qwen.d_debug_norm);
    if (ret != 0) return -4;
    if (cudaStreamSynchronize(g_qwen.stream) != cudaSuccess) return -5;
    float *A = (float *)malloc((size_t)M * sizeof(float));
    if (!A) return -6;
    cudaMemcpy(A, g_qwen.d_debug_qmma_out, (size_t)M * sizeof(float), cudaMemcpyDeviceToHost);
    // nk path (nvar=1) on the same input, WITH the auto K-split (sub-step 1).
    int nblocks = (K + 127) / 128;
    int k_splits = qmma_sf_k_split_auto(w->Mt, w->Kt);
    uint8_t *d_b = NULL; float *d_scale = NULL, *d_out = NULL, *d_part = NULL;
    cudaMalloc(&d_b, (size_t)Kt * 256);
    cudaMalloc(&d_scale, (size_t)nblocks * sizeof(float));
    cudaMalloc(&d_out, (size_t)M * sizeof(float));
    if (k_splits > 1) cudaMalloc(&d_part, (size_t)k_splits * M * sizeof(float));
    if (!d_b || !d_scale || !d_out) { free(A); if (d_b) cudaFree(d_b); if (d_scale) cudaFree(d_scale); if (d_out) cudaFree(d_out); if (d_part) cudaFree(d_part); return -7; }
    cudaMemsetAsync(d_b, 0, (size_t)Kt * 256, g_qwen.stream);
    k_tq_quantize_bfrag_pow2_fused_nk<<<nblocks, 128, 0, g_qwen.stream>>>(d_b, d_scale, g_qwen.d_debug_norm, K, 1);
    run_nk_gemv(d_out, d_part, w, d_b, d_scale, 1, k_splits, NULL);
    if (cudaStreamSynchronize(g_qwen.stream) != cudaSuccess) { free(A); cudaFree(d_b); cudaFree(d_scale); cudaFree(d_out); if (d_part) cudaFree(d_part); return -8; }
    float *B = (float *)malloc((size_t)M * sizeof(float));
    if (!B) { free(A); cudaFree(d_b); cudaFree(d_scale); cudaFree(d_out); return -9; }
    cudaMemcpy(B, d_out, (size_t)M * sizeof(float), cudaMemcpyDeviceToHost);
    float md = 0.0f, mag = 0.0f; int amA = 0, amB = 0;
    for (int i = 0; i < M; i++) {
        float d = fabsf(A[i] - B[i]); if (d > md) md = d;
        float a = fabsf(A[i]); if (a > mag) mag = a;
        if (A[i] > A[amA]) amA = i;
        if (B[i] > B[amB]) amB = i;
    }
    if (max_abs_diff) *max_abs_diff = md;
    if (rel) *rel = (mag > 0.0f) ? (double)md / (double)mag : 0.0;
    if (argmax_pertoken) *argmax_pertoken = amA;
    if (argmax_nk) *argmax_nk = amB;
    free(A); free(B); cudaFree(d_b); cudaFree(d_scale); cudaFree(d_out); if (d_part) cudaFree(d_part);
    return 0;
}

// Step-5 amortization probe: time the validated N=k batched GEMV (1 weight read,
// N columns) vs N per-token GEMVs (N weight reads), for a representative weight.
// This is the memory-bound speedup mechanism: if t_nk(N) ~ t_pertoken (one read),
// then a spec round verifying N tree nodes costs ~1 weight read instead of N.
// which: 0 = lm_head, 1 = first full-attn q_proj, 2 = first linear-layer mlp_down.
extern "C" int qwn_gemv_nk_bench(int which, int N, int iters,
                                 double *t_nk_ms, double *t_pertoken_ms) {
    if (!g_qwen.initialized || !g_qwen.d_debug_norm) return -1;
    if (N < 1) N = 1; if (N > 8) N = 8; if (iters < 1) iters = 1;
    const tq_qmma_weight_t *w = NULL;
    int lin = -1, full = -1;
    for (int i = 0; i < g_qwen.L; i++) {
        if (full < 0 && g_qwen.layer_types[i] == TQ_LAYER_FULL_ATTENTION) full = i;
        if (lin < 0 && g_qwen.layer_types[i] == TQ_LAYER_LINEAR_ATTENTION) lin = i;
    }
    if (which == 0) w = &g_qwen.lm_head;
    else if (which == 1 && full >= 0) w = &g_qwen.layers[full].q_proj;
    else if (which == 2 && lin >= 0) w = &g_qwen.layers[lin].mlp_down;
    else if (which == 3 && lin >= 0) w = &g_qwen.layers[lin].mlp_gate;
    else if (which == 4 && lin >= 0) w = &g_qwen.layers[lin].linear_in_qkv;
    if (!w || !w->d_A) return -2;
    if (!can_use_qmma_sf_weight(w) || w->row_major) return -3;
    int K = w->K, M = w->M, Kt = w->Kt, Mt = w->Mt;
    // Per-token input is d_debug_norm when K==H; for K!=H reuse a zero-safe slice
    // (timing only depends on shapes/reads, not values).
    int act_scale_count = (K + 127) / 128;
    int nblocks = (K + 127) / 128;
    int k_splits = qmma_sf_k_split_auto(Mt, Kt);
    float *d_x = NULL, *d_out_nk = NULL, *d_bscale = NULL, *d_part = NULL;
    uint8_t *d_b = NULL;
    cudaMalloc(&d_x, (size_t)N * K * sizeof(float));
    cudaMalloc(&d_out_nk, (size_t)N * M * sizeof(float));
    cudaMalloc(&d_b, (size_t)Kt * 256);
    cudaMalloc(&d_bscale, (size_t)nblocks * sizeof(float));
    if (k_splits > 1) cudaMalloc(&d_part, (size_t)k_splits * N * M * sizeof(float));
    if (!d_x || !d_out_nk || !d_b || !d_bscale) { if(d_x)cudaFree(d_x); if(d_out_nk)cudaFree(d_out_nk); if(d_b)cudaFree(d_b); if(d_bscale)cudaFree(d_bscale); if(d_part)cudaFree(d_part); return -4; }
    for (int v = 0; v < N; v++)   // fill columns with the current hidden (tiled if K>H)
        cudaMemcpyAsync(d_x + (size_t)v * K, g_qwen.d_debug_norm,
                        (size_t)(K < g_qwen.H ? K : g_qwen.H) * sizeof(float),
                        cudaMemcpyDeviceToDevice, g_qwen.stream);
    cudaStreamSynchronize(g_qwen.stream);

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    // (1) batched nk GEMV: ONE weight read, N columns (auto K-split), per iter.
    cudaEventRecord(e0, g_qwen.stream);
    for (int it = 0; it < iters; it++) {
        cudaMemsetAsync(d_b, 0, (size_t)Kt * 256, g_qwen.stream);
        k_tq_quantize_bfrag_pow2_fused_nk<<<nblocks, 128, 0, g_qwen.stream>>>(d_b, d_bscale, d_x, K, N);
        run_nk_gemv(d_out_nk, d_part, w, d_b, d_bscale, N, k_splits, NULL);
    }
    cudaEventRecord(e1, g_qwen.stream);
    cudaEventSynchronize(e1);
    float ms_nk = 0.0f; cudaEventElapsedTime(&ms_nk, e0, e1);
    // (2) N per-token GEMVs: N weight reads per iter (the baseline's per-token cost).
    cudaEventRecord(e0, g_qwen.stream);
    for (int it = 0; it < iters; it++) {
        for (int v = 0; v < N; v++) {
            run_qmma_projection_from_input_to_debug_out(w, d_x + (size_t)v * K);
        }
    }
    cudaEventRecord(e1, g_qwen.stream);
    cudaEventSynchronize(e1);
    float ms_pt = 0.0f; cudaEventElapsedTime(&ms_pt, e0, e1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    if (t_nk_ms) *t_nk_ms = (double)ms_nk / iters;
    if (t_pertoken_ms) *t_pertoken_ms = (double)ms_pt / iters;
    (void)act_scale_count;
    cudaFree(d_x); cudaFree(d_out_nk); cudaFree(d_b); cudaFree(d_bscale); if (d_part) cudaFree(d_part);
    return 0;
}

// D.1 wide projection GEMM selector: which 0=lm_head 1=q_proj 2=mlp_down 3=mlp_gate
// 4=linear_in_qkv. e2m3 (FP6) weights only.
static const tq_qmma_weight_t *wide_gemm_pick(int which) {
    int lin = -1, full = -1;
    for (int i = 0; i < g_qwen.L; i++) {
        if (full < 0 && g_qwen.layer_types[i] == TQ_LAYER_FULL_ATTENTION) full = i;
        if (lin < 0 && g_qwen.layer_types[i] == TQ_LAYER_LINEAR_ATTENTION) lin = i;
    }
    if (which == 0) return &g_qwen.lm_head;
    if (which == 1 && full >= 0) return &g_qwen.layers[full].q_proj;
    if (which == 2 && lin >= 0) return &g_qwen.layers[lin].mlp_down;
    if (which == 3 && lin >= 0) return &g_qwen.layers[lin].mlp_gate;
    if (which == 4 && lin >= 0) return &g_qwen.layers[lin].linear_in_qkv;
    return NULL;
}

// Bit-sanity: with N IDENTICAL columns, every wide-GEMM output column must equal the
// per-token projection of that column (shared pow2 scale == per-token scale when the
// columns match). Use a weight with K==H (which 0/1/3/4) so the input copy is full.
extern "C" int qwn_wide_gemm_check(int which, int N, float *max_abs_diff) {
    if (!g_qwen.initialized || !g_qwen.d_debug_norm) return -1;
    if (N < 1) N = 1; if (N > 256) N = 256;
    const tq_qmma_weight_t *w = wide_gemm_pick(which);
    if (!w || !w->d_A) return -2;
    if (!can_use_qmma_sf_weight(w) || w->row_major || !w->e2m3) return -3;
    int K = w->K, M = w->M, Kt = w->Kt, G = (N + 7) / 8, nblocks = (K + 127) / 128;
    int k_splits = qmma_sf_k_split_auto(w->Mt, w->Kt);
    if (run_qmma_projection_from_input_to_debug_out(w, g_qwen.d_debug_norm) != 0) return -4;
    if (cudaStreamSynchronize(g_qwen.stream) != cudaSuccess) return -5;
    float *ref = (float *)malloc((size_t)M * sizeof(float));
    if (!ref) return -6;
    cudaMemcpy(ref, g_qwen.d_debug_qmma_out, (size_t)M * sizeof(float), cudaMemcpyDeviceToHost);
    float *d_x = NULL, *d_out = NULL, *d_bscale = NULL, *d_part = NULL; uint8_t *d_b = NULL;
    cudaMalloc(&d_x, (size_t)N * K * sizeof(float));
    cudaMalloc(&d_out, (size_t)N * M * sizeof(float));
    cudaMalloc(&d_b, (size_t)G * Kt * 256);
    cudaMalloc(&d_bscale, (size_t)nblocks * sizeof(float));
    if (k_splits > 1) cudaMalloc(&d_part, (size_t)k_splits * N * M * sizeof(float));
    if (!d_x || !d_out || !d_b || !d_bscale || (k_splits > 1 && !d_part)) {
        free(ref); cudaFree(d_x); cudaFree(d_out); cudaFree(d_b); cudaFree(d_bscale); cudaFree(d_part); return -7;
    }
    for (int v = 0; v < N; v++)
        cudaMemcpyAsync(d_x + (size_t)v * K, g_qwen.d_debug_norm, (size_t)K * sizeof(float),
                        cudaMemcpyDeviceToDevice, g_qwen.stream);
    cudaMemsetAsync(d_b, 0, (size_t)G * Kt * 256, g_qwen.stream);
    k_tq_quantize_bfrag_pow2_fused_nk<<<nblocks, 128, 0, g_qwen.stream>>>(d_b, d_bscale, d_x, K, N);
    launch_fp6_wide_gemm(d_out, w, d_b, d_bscale, N, d_part, k_splits, g_qwen.stream);
    if (cudaStreamSynchronize(g_qwen.stream) != cudaSuccess) {
        free(ref); cudaFree(d_x); cudaFree(d_out); cudaFree(d_b); cudaFree(d_bscale); cudaFree(d_part); return -8;
    }
    float *wide = (float *)malloc((size_t)N * M * sizeof(float));
    if (!wide) { free(ref); cudaFree(d_x); cudaFree(d_out); cudaFree(d_b); cudaFree(d_bscale); cudaFree(d_part); return -9; }
    cudaMemcpy(wide, d_out, (size_t)N * M * sizeof(float), cudaMemcpyDeviceToHost);
    float md = 0.0f;
    for (int v = 0; v < N; v++)
        for (int r = 0; r < M; r++) {
            float d = fabsf(wide[(size_t)v * M + r] - ref[r]);
            if (d > md) md = d;
        }
    if (max_abs_diff) *max_abs_diff = md;
    free(ref); free(wide);
    cudaFree(d_x); cudaFree(d_out); cudaFree(d_b); cudaFree(d_bscale); cudaFree(d_part);
    return 0;
}

// N-scaling timing: wide GEMM (one weight read, N columns) vs N per-token GEMVs
// (the prefill baseline = one weight read PER token). The wide ms staying ~flat as
// N grows = memory-bound (weight amortized); ms growing ~linearly = compute-bound.
extern "C" int qwn_wide_gemm_bench(int which, int N, int iters,
                                   double *t_wide_ms, double *t_pertoken_ms) {
    if (!g_qwen.initialized || !g_qwen.d_debug_norm) return -1;
    if (N < 1) N = 1; if (N > 256) N = 256; if (iters < 1) iters = 1;
    const tq_qmma_weight_t *w = wide_gemm_pick(which);
    if (!w || !w->d_A) return -2;
    if (!can_use_qmma_sf_weight(w) || w->row_major || !w->e2m3) return -3;
    int K = w->K, M = w->M, Kt = w->Kt, G = (N + 7) / 8, nblocks = (K + 127) / 128;
    int k_splits = qmma_sf_k_split_auto(w->Mt, w->Kt);
    float *d_x = NULL, *d_out = NULL, *d_bscale = NULL, *d_part = NULL; uint8_t *d_b = NULL;
    cudaMalloc(&d_x, (size_t)N * K * sizeof(float));
    cudaMalloc(&d_out, (size_t)N * M * sizeof(float));
    cudaMalloc(&d_b, (size_t)G * Kt * 256);
    cudaMalloc(&d_bscale, (size_t)nblocks * sizeof(float));
    if (k_splits > 1) cudaMalloc(&d_part, (size_t)k_splits * N * M * sizeof(float));
    if (!d_x || !d_out || !d_b || !d_bscale || (k_splits > 1 && !d_part)) {
        cudaFree(d_x); cudaFree(d_out); cudaFree(d_b); cudaFree(d_bscale); cudaFree(d_part); return -4;
    }
    for (int v = 0; v < N; v++)
        cudaMemcpyAsync(d_x + (size_t)v * K, g_qwen.d_debug_norm,
                        (size_t)(K < g_qwen.H ? K : g_qwen.H) * sizeof(float),
                        cudaMemcpyDeviceToDevice, g_qwen.stream);
    cudaStreamSynchronize(g_qwen.stream);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0, g_qwen.stream);
    for (int it = 0; it < iters; it++) {
        cudaMemsetAsync(d_b, 0, (size_t)G * Kt * 256, g_qwen.stream);
        k_tq_quantize_bfrag_pow2_fused_nk<<<nblocks, 128, 0, g_qwen.stream>>>(d_b, d_bscale, d_x, K, N);
        launch_fp6_wide_gemm(d_out, w, d_b, d_bscale, N, d_part, k_splits, g_qwen.stream);
    }
    cudaEventRecord(e1, g_qwen.stream); cudaEventSynchronize(e1);
    float ms_w = 0.0f; cudaEventElapsedTime(&ms_w, e0, e1);
    cudaEventRecord(e0, g_qwen.stream);
    for (int it = 0; it < iters; it++)
        for (int v = 0; v < N; v++)
            run_qmma_projection_from_input_to_debug_out(w, d_x + (size_t)v * K);
    cudaEventRecord(e1, g_qwen.stream); cudaEventSynchronize(e1);
    float ms_pt = 0.0f; cudaEventElapsedTime(&ms_pt, e0, e1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    if (t_wide_ms) *t_wide_ms = (double)ms_w / iters;
    if (t_pertoken_ms) *t_pertoken_ms = (double)ms_pt / iters;
    cudaFree(d_x); cudaFree(d_out); cudaFree(d_b); cudaFree(d_bscale); cudaFree(d_part);
    return 0;
}

// Deterministic, bounded synthetic projection fill (debug only): a smooth sinusoid so
// the wide-attn check has non-degenerate, reproducible q/k/v across the N chunk tokens.
__global__ void k_tq_wide_attn_synth_fill(float *buf, int total, float scale, float phase,
                                          unsigned salt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    unsigned h = (unsigned)idx * 2654435761u + salt;
    float u = (float)(h & 0xffffu) / 65536.0f - 0.5f;            // [-0.5, 0.5)
    buf[idx] = scale * (u + 0.4f * sinf(phase * (float)idx + 0.37f * (float)salt));
}

extern "C" int qwn_prefill_chunk(const int *tokens, int base_pos, int n, int *out_argmax);

// D.1 (2) bit-sanity gate: the N-wide attention (k_tq_wide_attn[_kv_write]) MUST equal
// the per-token reference (k_tq_full_attn_decode_arr) applied to each chunk token over
// the SAME prefix cache and the SAME synthetic raw q/k/v. Prefills base_pos tokens
// normally (fp32 KV) to lay down a realistic prefix [0..base_pos-1], synthesizes N
// chunk tokens at positions base_pos..base_pos+N-1, runs BOTH paths, and reports
// max|wide - ref| over [N*nh*hd]. The per-token path is the source of truth; target
// ~0 / float-eps. fp32 KV ONLY (rejects TQ_KV_FP8 / TQ_KV_Q4).
extern "C" int qwn_wide_attn_check(int layer, int base_pos, int N, float *max_abs_diff) {
    if (!g_qwen.initialized) return -1;
    if (layer < 0 || layer >= g_qwen.L) return -2;
    if (g_qwen.layer_types[layer] != TQ_LAYER_FULL_ATTENTION) return -3;
    if (g_qwen.kv_fp8 || g_qwen.kv_q4) return -4;      // fp32 KV cache only (scope limit)
    if (N < 1) N = 1;
    if (base_pos < 0) base_pos = 0;
    if (base_pos + N + 2 > g_qwen.max_seq) return -5;
    int nh = g_qwen.nh, nkv = g_qwen.nkv, hd = g_qwen.hd;
    int q_m = nh * hd * 2, kv_m = nkv * hd, attn_m = nh * hd;
    tq_layer_t *l = &g_qwen.layers[layer];

    // (1) populate prefix rows [0..base_pos-1] via the normal fp32 chunked prefill.
    if (qwn_reset_state() != 0) return -6;
    for (int t = 0; t < base_pos; ) {
        int n = base_pos - t; if (n > TQ_SPEC_MAX_N) n = TQ_SPEC_MAX_N;
        int chunk[TQ_SPEC_MAX_N];
        for (int j = 0; j < n; j++)
            chunk[j] = (int)(((size_t)(t + j) * 1103515245u + 12345u) % (unsigned)g_qwen.V);
        int amx = 0;
        if (qwn_prefill_chunk(chunk, t - 1, n, &amx) != 0) return -7;
        t += n;
    }

    // (2) synthetic raw q/k/v for the chunk + their absolute positions.
    float *d_q = NULL, *d_k = NULL, *d_v = NULL, *d_ref = NULL, *d_wide = NULL;
    int *d_pos = NULL; float *hr = NULL, *hw = NULL; int rc = 0;
    cudaMalloc(&d_q, (size_t)N * q_m * sizeof(float));
    cudaMalloc(&d_k, (size_t)N * kv_m * sizeof(float));
    cudaMalloc(&d_v, (size_t)N * kv_m * sizeof(float));
    cudaMalloc(&d_ref, (size_t)N * attn_m * sizeof(float));
    cudaMalloc(&d_wide, (size_t)N * attn_m * sizeof(float));
    cudaMalloc(&d_pos, (size_t)N * sizeof(int));
    if (!d_q || !d_k || !d_v || !d_ref || !d_wide || !d_pos) { rc = -8; goto cleanup; }
    k_tq_wide_attn_synth_fill<<<((size_t)N * q_m + 255) / 256, 256, 0, g_qwen.stream>>>(d_q, N * q_m, 0.7f, 0.0011f, 1u);
    k_tq_wide_attn_synth_fill<<<((size_t)N * kv_m + 255) / 256, 256, 0, g_qwen.stream>>>(d_k, N * kv_m, 0.7f, 0.0017f, 2u);
    k_tq_wide_attn_synth_fill<<<((size_t)N * kv_m + 255) / 256, 256, 0, g_qwen.stream>>>(d_v, N * kv_m, 0.7f, 0.0023f, 3u);
    {
        int *hp = (int *)malloc((size_t)N * sizeof(int));
        if (!hp) { rc = -9; goto cleanup; }
        for (int i = 0; i < N; i++) hp[i] = base_pos + i;
        cudaMemcpyAsync(d_pos, hp, (size_t)N * sizeof(int), cudaMemcpyHostToDevice, g_qwen.stream);
        cudaStreamSynchronize(g_qwen.stream);
        free(hp);
    }
    if (ensure_attn_scores() != 0) { rc = -10; goto cleanup; }

    // (3) per-token reference: each token writes its K/V then attends [0..pos], in
    //     ascending order (same-stream serialization => token i sees tokens 0..i-1).
    for (int i = 0; i < N; i++) {
        k_tq_full_attn_decode_arr<<<nh, hd, 0, g_qwen.stream>>>(
            d_ref + (size_t)i * attn_m, d_q + (size_t)i * q_m, d_k + (size_t)i * kv_m,
            d_v + (size_t)i * kv_m, l->d_q_norm, l->d_k_norm, l->d_k_cache, l->d_v_cache,
            NULL, NULL, NULL, NULL, NULL, NULL,
            d_pos, i, nh, nkv, hd, g_qwen.eps, g_qwen.rope_theta,
            g_qwen.d_attn_scores, g_qwen.max_seq);
    }

    // (4) wide path: re-write the chunk's K/V (identical bytes) then attend all N at once.
    if (launch_wide_attn(d_wide, d_q, l->d_q_norm, d_k, d_v, l->d_k_norm,
                         l->d_k_cache, l->d_v_cache, d_pos, N, base_pos + N - 1,
                         g_qwen.stream) != 0) { rc = -11; goto cleanup; }
    if (cudaStreamSynchronize(g_qwen.stream) != cudaSuccess) { rc = -12; goto cleanup; }

    // (5) compare.
    hr = (float *)malloc((size_t)N * attn_m * sizeof(float));
    hw = (float *)malloc((size_t)N * attn_m * sizeof(float));
    if (!hr || !hw) { rc = -13; goto cleanup; }
    cudaMemcpy(hr, d_ref, (size_t)N * attn_m * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hw, d_wide, (size_t)N * attn_m * sizeof(float), cudaMemcpyDeviceToHost);
    {
        float md = 0.0f;
        for (size_t j = 0; j < (size_t)N * attn_m; j++) {
            float dlt = fabsf(hw[j] - hr[j]);
            if (dlt > md) md = dlt;
        }
        if (max_abs_diff) *max_abs_diff = md;
    }
cleanup:
    free(hr); free(hw);
    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v);
    cudaFree(d_ref); cudaFree(d_wide); cudaFree(d_pos);
    return rc;
}

static int run_debug_embed_norm(int token_id, int layer) {
    if (!g_qwen.initialized) {
        fprintf(stderr, "debug embed/norm: model is not initialized\n");
        return -1;
    }
    if (g_qwen.non_quant_dtype != TQ_DTYPE_BF16) {
        fprintf(stderr, "debug embed/norm: only BF16 non-quant tensors are supported for now\n");
        return -2;
    }
    if (token_id < 0 || token_id >= g_qwen.V || layer < 0 || layer >= g_qwen.L) {
        fprintf(stderr, "debug embed/norm: invalid token/layer token=%d layer=%d\n", token_id, layer);
        return -3;
    }
    int err_debug = ensure_debug_vectors(0);
    if (err_debug != 0) return -4;

    int blocks = (g_qwen.H + 255) / 256;
    k_tq_embed_lookup<<<blocks, 256, 0, g_qwen.stream>>>(g_qwen.d_debug_x, g_qwen.d_embed, token_id, g_qwen.H);
    k_tq_qwen_rmsnorm<<<8, 1024, 0, g_qwen.stream>>>(g_qwen.d_debug_norm, g_qwen.d_debug_x,
                                   g_qwen.layers[layer].d_input_ln, g_qwen.H, g_qwen.eps);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "debug kernels failed to launch: %s\n", cudaGetErrorString(err));
        return -5;
    }
    return 0;
}

extern "C" int qwn_debug_embed_input_norm(int token_id, int layer, float *out, int max_count) {
    if (!out || max_count <= 0) {
        fprintf(stderr, "qwn_debug_embed_input_norm: invalid output buffer\n");
        return -4;
    }
    int ret = run_debug_embed_norm(token_id, layer);
    if (ret != 0) return ret;
    int count = max_count < g_qwen.H ? max_count : g_qwen.H;
    cudaError_t err = cudaMemcpy(out, g_qwen.d_debug_norm, (size_t)count * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "debug output copy failed: %s\n", cudaGetErrorString(err));
        return -8;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "debug synchronize failed: %s\n", cudaGetErrorString(err));
        return -9;
    }
    return count;
}

extern "C" int qwn_debug_linear_in_a(int token_id, int layer, float *out, int max_count) {
    if (!out || max_count <= 0) {
        fprintf(stderr, "qwn_debug_linear_in_a: invalid output buffer\n");
        return -4;
    }
    if (layer < 0 || layer >= g_qwen.L || g_qwen.layer_types[layer] != TQ_LAYER_LINEAR_ATTENTION) {
        fprintf(stderr, "qwn_debug_linear_in_a: layer %d is not linear_attention\n", layer);
        return -10;
    }
    int ret = run_debug_embed_norm(token_id, layer);
    if (ret != 0) return ret;
    tq_qmma_weight_t *w = &g_qwen.layers[layer].linear_in_a;
    ret = ensure_debug_vectors(w->M);
    if (ret != 0) return -11;
    k_tq_debug_qmma_gemv<<<w->M, 256, 0, g_qwen.stream>>>(g_qwen.d_debug_proj, w->d_A, w->M, w->K,
                                        w->Kt, w->inv_scale, g_qwen.d_debug_norm);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "linear_in_a debug GEMV launch failed: %s\n", cudaGetErrorString(err));
        return -12;
    }
    int count = max_count < w->M ? max_count : w->M;
    err = cudaMemcpy(out, g_qwen.d_debug_proj, (size_t)count * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "linear_in_a debug output copy failed: %s\n", cudaGetErrorString(err));
        return -13;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "linear_in_a debug synchronize failed: %s\n", cudaGetErrorString(err));
        return -14;
    }
    return count;
}

static tq_qmma_weight_t *select_norm_input_projection(int layer, int projection_id) {
    tq_layer_t *l = &g_qwen.layers[layer];
    switch (projection_id) {
        case 1: return &l->linear_in_a;
        case 2: return &l->linear_in_b;
        case 3: return &l->linear_in_qkv;
        case 4: return &l->linear_in_z;
        default: return NULL;
    }
}

extern "C" int qwn_debug_linear_projection_qmma(int token_id, int layer, int projection_id,
                                                float *out, int max_count) {
    if (!out || max_count <= 0) {
        fprintf(stderr, "qwn_debug_linear_projection_qmma: invalid output buffer\n");
        return -4;
    }
    if (layer < 0 || layer >= g_qwen.L || g_qwen.layer_types[layer] != TQ_LAYER_LINEAR_ATTENTION) {
        fprintf(stderr, "qwn_debug_linear_projection_qmma: layer %d is not linear_attention\n", layer);
        return -10;
    }
    tq_qmma_weight_t *w = select_norm_input_projection(layer, projection_id);
    if (!w || !w->d_A) {
        fprintf(stderr, "qwn_debug_linear_projection_qmma: unsupported projection_id=%d\n", projection_id);
        return -15;
    }
    int ret = run_debug_embed_norm(token_id, layer);
    if (ret != 0) return ret;
    ret = run_qmma_projection_to_debug_out(w);
    if (ret != 0) return -12;
    int count = max_count < w->M ? max_count : w->M;
    cudaError_t err = cudaMemcpy(out, g_qwen.d_debug_qmma_out, (size_t)count * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "linear_in_a QMMA debug output copy failed: %s\n", cudaGetErrorString(err));
        return -13;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "linear_in_a QMMA debug synchronize failed: %s\n", cudaGetErrorString(err));
        return -14;
    }
    return count;
}

extern "C" int qwn_debug_linear_in_a_qmma(int token_id, int layer, float *out, int max_count) {
    return qwn_debug_linear_projection_qmma(token_id, layer, 1, out, max_count);
}

extern "C" int qwn_debug_linear_qkv_conv_first(int token_id, int layer, float *out, int max_count) {
    if (!out || max_count <= 0) {
        fprintf(stderr, "qwn_debug_linear_qkv_conv_first: invalid output buffer\n");
        return -4;
    }
    if (layer < 0 || layer >= g_qwen.L || g_qwen.layer_types[layer] != TQ_LAYER_LINEAR_ATTENTION) {
        fprintf(stderr, "qwn_debug_linear_qkv_conv_first: layer %d is not linear_attention\n", layer);
        return -10;
    }
    int ret = run_debug_embed_norm(token_id, layer);
    if (ret != 0) return ret;
    tq_qmma_weight_t *w = &g_qwen.layers[layer].linear_in_qkv;
    ret = run_qmma_projection_to_debug_out(w);
    if (ret != 0) return -11;
    ret = ensure_debug_conv(w->M);
    if (ret != 0) return -12;
    k_tq_linear_conv_first_token<<<(w->M + 255) / 256, 256, 0, g_qwen.stream>>>(
        g_qwen.d_debug_conv, g_qwen.d_debug_qmma_out, g_qwen.layers[layer].d_linear_conv1d,
        w->M, g_qwen.linear_conv_kernel_dim);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "linear qkv conv debug launch failed: %s\n", cudaGetErrorString(err));
        return -13;
    }
    int count = max_count < w->M ? max_count : w->M;
    err = cudaMemcpy(out, g_qwen.d_debug_conv, (size_t)count * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "linear qkv conv debug output copy failed: %s\n", cudaGetErrorString(err));
        return -14;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "linear qkv conv debug synchronize failed: %s\n", cudaGetErrorString(err));
        return -15;
    }
    return count;
}

static int run_linear_first_core_from_norm_to_debug_core(int layer) {
    if (layer < 0 || layer >= g_qwen.L || g_qwen.layer_types[layer] != TQ_LAYER_LINEAR_ATTENTION) {
        fprintf(stderr, "linear first core: layer %d is not linear_attention\n", layer);
        return -10;
    }
    int heads = g_qwen.linear_num_value_heads;
    int key_heads = g_qwen.linear_num_key_heads;
    int dim = g_qwen.linear_value_head_dim;
    int key_dim = key_heads * dim;
    int value_dim = heads * dim;
    int conv_dim = key_dim * 2 + value_dim;
    tq_layer_t *l = &g_qwen.layers[layer];

    int ret = 0;
    if (ensure_float_buffer(&g_qwen.d_debug_qkv, &g_qwen.debug_qkv_count, conv_dim, "d_debug_qkv") != 0) return -11;
    if (ensure_float_buffer(&g_qwen.d_debug_z, &g_qwen.debug_z_count, value_dim, "d_debug_z") != 0) return -12;
    if (ensure_float_buffer(&g_qwen.d_debug_core, &g_qwen.debug_core_count, value_dim, "d_debug_core") != 0) return -13;
    if (ensure_debug_vectors(heads) != 0) return -14;

    if (can_fuse_qmma_sf_tiled_input(&l->linear_in_qkv, &l->linear_in_z) &&
        can_fuse_qmma_sf_tiled_input(&l->linear_in_qkv, &l->linear_in_b)) {
        ret = prepare_qmma_sf_input(&l->linear_in_qkv, g_qwen.d_debug_norm);
        if (ret != 0) return -15;
        ret = run_qmma_sf_prepared_multi_to_buffers(
            &l->linear_in_qkv, g_qwen.d_debug_qkv, conv_dim,
            &l->linear_in_z, g_qwen.d_debug_z, value_dim,
            &l->linear_in_b, g_qwen.d_debug_proj, heads,
            NULL, NULL, 0);
        if (ret != 0) return -15;
    } else {
        ret = run_qmma_projection_from_input_to_buffer(&l->linear_in_qkv, g_qwen.d_debug_norm,
                                                       g_qwen.d_debug_qkv, conv_dim);
        if (ret != 0) return -15;
    }
    k_tq_linear_conv_first_token<<<(conv_dim + 255) / 256, 256, 0, g_qwen.stream>>>(
        g_qwen.d_debug_qkv, g_qwen.d_debug_qkv, l->d_linear_conv1d, conv_dim,
        g_qwen.linear_conv_kernel_dim);
    if (!(can_fuse_qmma_sf_tiled_input(&l->linear_in_qkv, &l->linear_in_z) &&
          can_fuse_qmma_sf_tiled_input(&l->linear_in_qkv, &l->linear_in_b))) {
        ret = run_qmma_projection_from_input_to_buffer(&l->linear_in_z, g_qwen.d_debug_norm,
                                                       g_qwen.d_debug_z, value_dim);
        if (ret != 0) return -16;
        ret = run_qmma_projection_from_input_to_buffer(&l->linear_in_b, g_qwen.d_debug_norm,
                                                       g_qwen.d_debug_proj, heads);
        if (ret != 0) return -17;
    }

    k_tq_linear_first_core_gated<<<heads, dim, 0, g_qwen.stream>>>(g_qwen.d_debug_core, g_qwen.d_debug_qkv,
                                                 g_qwen.d_debug_z, g_qwen.d_debug_proj,
                                                 l->d_linear_norm, heads, key_heads, dim, g_qwen.eps);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "linear first core launch failed: %s\n", cudaGetErrorString(err));
        return -18;
    }
    return 0;
}

static int run_linear_first_core_to_debug_core(int token_id, int layer) {
    int ret = run_debug_embed_norm(token_id, layer);
    if (ret != 0) return ret;
    return run_linear_first_core_from_norm_to_debug_core(layer);
}

static int run_mlp_from_resid_to_layer_out_w(tq_layer_t *l) {
    if (ensure_float_buffer(&g_qwen.d_debug_post_norm, &g_qwen.debug_post_norm_count, g_qwen.H, "d_debug_post_norm") != 0) return -2;
    if (ensure_float_buffer(&g_qwen.d_debug_gate, &g_qwen.debug_gate_count, g_qwen.I, "d_debug_gate") != 0) return -3;
    if (ensure_float_buffer(&g_qwen.d_debug_up, &g_qwen.debug_up_count, g_qwen.I, "d_debug_up") != 0) return -4;
    if (ensure_float_buffer(&g_qwen.d_debug_mlp_hidden, &g_qwen.debug_mlp_hidden_count, g_qwen.I, "d_debug_mlp_hidden") != 0) return -5;
    if (ensure_float_buffer(&g_qwen.d_debug_layer_out, &g_qwen.debug_layer_out_count, g_qwen.H, "d_debug_layer_out") != 0) return -6;

    k_tq_qwen_rmsnorm<<<8, 1024, 0, g_qwen.stream>>>(g_qwen.d_debug_post_norm, g_qwen.d_debug_resid,
                                   l->d_post_ln, g_qwen.H, g_qwen.eps);
    int ret;
    if (can_fuse_qmma_sf_tiled_input(&l->mlp_gate, &l->mlp_up)) {
        ret = prepare_qmma_sf_input(&l->mlp_gate, g_qwen.d_debug_post_norm);
        if (ret != 0) return -7;
        ret = run_qmma_sf_prepared_multi_to_buffers(
            &l->mlp_gate, g_qwen.d_debug_gate, g_qwen.I,
            &l->mlp_up, g_qwen.d_debug_up, g_qwen.I,
            NULL, NULL, 0,
            NULL, NULL, 0);
        if (ret != 0) return -7;
    } else {
        ret = run_qmma_projection_from_input_to_buffer(&l->mlp_gate, g_qwen.d_debug_post_norm,
                                                       g_qwen.d_debug_gate, g_qwen.I);
        if (ret != 0) return -7;
        ret = run_qmma_projection_from_input_to_buffer(&l->mlp_up, g_qwen.d_debug_post_norm,
                                                       g_qwen.d_debug_up, g_qwen.I);
        if (ret != 0) return -8;
    }
    if (use_fused_residual_add() && can_use_qmma_sf_weight(&l->mlp_down) && !l->mlp_down.row_major) {
        ret = prepare_qmma_sf_input_silu_mul(&l->mlp_down, g_qwen.d_debug_gate, g_qwen.d_debug_up);
        if (ret != 0) return -9;
        ret = run_qmma_sf_prepared_to_buffer_add(&l->mlp_down, g_qwen.d_debug_resid,
                                                 g_qwen.d_debug_layer_out, g_qwen.H);
        if (ret != 0) return -9;
    } else {
        k_tq_silu_mul<<<(g_qwen.I + 255) / 256, 256, 0, g_qwen.stream>>>(g_qwen.d_debug_mlp_hidden,
                                                       g_qwen.d_debug_gate, g_qwen.d_debug_up,
                                                       g_qwen.I);
        ret = run_qmma_projection_from_input_to_debug_out(&l->mlp_down, g_qwen.d_debug_mlp_hidden);
        if (ret != 0) return -9;
        k_tq_add_vec<<<(g_qwen.H + 255) / 256, 256, 0, g_qwen.stream>>>(g_qwen.d_debug_layer_out,
                                                      g_qwen.d_debug_resid,
                                                      g_qwen.d_debug_qmma_out, g_qwen.H);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "MLP/residual layer output launch failed: %s\n", cudaGetErrorString(err));
        return -10;
    }
    return 0;
}

static int run_mlp_from_resid_to_layer_out(int layer) {
    return run_mlp_from_resid_to_layer_out_w(&g_qwen.layers[layer]);
}

static int run_linear_layer_from_current_to_layer_out(int layer) {
    tq_layer_t *l = &g_qwen.layers[layer];
    if (ensure_float_buffer(&g_qwen.d_debug_resid, &g_qwen.debug_resid_count, g_qwen.H, "d_debug_resid") != 0) return -4;
    k_tq_qwen_rmsnorm<<<8, 1024, 0, g_qwen.stream>>>(g_qwen.d_debug_norm, g_qwen.d_debug_x,
                                   l->d_input_ln, g_qwen.H, g_qwen.eps);
    int ret = run_linear_first_core_from_norm_to_debug_core(layer);
    if (ret != 0) return -1;
    ret = run_qmma_projection_from_input_add_to_buffer(&l->linear_out, g_qwen.d_debug_core,
                                                       g_qwen.d_debug_x, g_qwen.d_debug_resid,
                                                       g_qwen.H);
    if (ret != 0) return -2;
    ret = run_mlp_from_resid_to_layer_out(layer);
    if (ret != 0) return -3;
    return 0;
}

static int run_full_layer_from_current_to_layer_out(int layer) {
    tq_layer_t *l = &g_qwen.layers[layer];
    int q_m = g_qwen.nh * g_qwen.hd * 2;
    int v_m = g_qwen.nkv * g_qwen.hd;
    int attn_m = g_qwen.nh * g_qwen.hd;
    if (ensure_float_buffer(&g_qwen.d_debug_qkv, &g_qwen.debug_qkv_count, q_m, "d_debug_qkv") != 0) return -1;
    if (ensure_float_buffer(&g_qwen.d_debug_z, &g_qwen.debug_z_count, v_m, "d_debug_z") != 0) return -2;
    if (ensure_float_buffer(&g_qwen.d_debug_core, &g_qwen.debug_core_count, attn_m, "d_debug_core") != 0) return -3;
    if (ensure_float_buffer(&g_qwen.d_debug_resid, &g_qwen.debug_resid_count, g_qwen.H, "d_debug_resid") != 0) return -8;
    k_tq_qwen_rmsnorm<<<8, 1024, 0, g_qwen.stream>>>(g_qwen.d_debug_norm, g_qwen.d_debug_x,
                                   l->d_input_ln, g_qwen.H, g_qwen.eps);
    int ret;
    if (can_fuse_qmma_sf_tiled_input(&l->q_proj, &l->v_proj)) {
        ret = prepare_qmma_sf_input(&l->q_proj, g_qwen.d_debug_norm);
        if (ret != 0) return -4;
        ret = run_qmma_sf_prepared_multi_to_buffers(
            &l->q_proj, g_qwen.d_debug_qkv, q_m,
            &l->v_proj, g_qwen.d_debug_z, v_m,
            NULL, NULL, 0,
            NULL, NULL, 0);
        if (ret != 0) return -4;
    } else {
        ret = run_qmma_projection_from_input_to_buffer(&l->q_proj, g_qwen.d_debug_norm,
                                                       g_qwen.d_debug_qkv, q_m);
        if (ret != 0) return -4;
        ret = run_qmma_projection_from_input_to_buffer(&l->v_proj, g_qwen.d_debug_norm,
                                                       g_qwen.d_debug_z, v_m);
        if (ret != 0) return -5;
    }
    k_tq_full_attn_first_token<<<(attn_m + 255) / 256, 256, 0, g_qwen.stream>>>(
        g_qwen.d_debug_core, g_qwen.d_debug_qkv, g_qwen.d_debug_z,
        g_qwen.nh, g_qwen.nkv, g_qwen.hd);
    ret = run_qmma_projection_from_input_add_to_buffer(&l->o_proj, g_qwen.d_debug_core,
                                                       g_qwen.d_debug_x, g_qwen.d_debug_resid,
                                                       g_qwen.H);
    if (ret != 0) return -6;
    ret = run_mlp_from_resid_to_layer_out(layer);
    if (ret != 0) return -7;
    return 0;
}

static int run_linear_layer_decode_from_current(int layer) {
    tq_layer_t *l = &g_qwen.layers[layer];
    int heads = g_qwen.linear_num_value_heads;
    int key_heads = g_qwen.linear_num_key_heads;
    int dim = g_qwen.linear_value_head_dim;
    int key_dim = key_heads * dim;
    int value_dim = heads * dim;
    int conv_dim = key_dim * 2 + value_dim;
    k_tq_qwen_rmsnorm<<<8, 1024, 0, g_qwen.stream>>>(g_qwen.d_debug_norm, g_qwen.d_debug_x,
                                                       l->d_input_ln, g_qwen.H, g_qwen.eps);
    if (ensure_float_buffer(&g_qwen.d_debug_qkv, &g_qwen.debug_qkv_count, conv_dim, "d_debug_qkv") != 0) return -1;
    if (ensure_float_buffer(&g_qwen.d_debug_z, &g_qwen.debug_z_count, value_dim, "d_debug_z") != 0) return -2;
    if (ensure_float_buffer(&g_qwen.d_debug_core, &g_qwen.debug_core_count, value_dim, "d_debug_core") != 0) return -3;
    if (ensure_float_buffer(&g_qwen.d_debug_a_proj, &g_qwen.debug_a_count, heads, "d_debug_a_proj") != 0) return -4;
    if (ensure_float_buffer(&g_qwen.d_debug_resid, &g_qwen.debug_resid_count, g_qwen.H, "d_debug_resid") != 0) return -12;
    if (ensure_debug_vectors(heads) != 0) return -5;
    int ret;
    if (can_fuse_qmma_sf_tiled_input(&l->linear_in_qkv, &l->linear_in_z) &&
        can_fuse_qmma_sf_tiled_input(&l->linear_in_qkv, &l->linear_in_b) &&
        can_fuse_qmma_sf_tiled_input(&l->linear_in_qkv, &l->linear_in_a)) {
        ret = prepare_qmma_sf_input(&l->linear_in_qkv, g_qwen.d_debug_norm);
        if (ret != 0) return -6;
        ret = run_qmma_sf_prepared_multi_to_buffers(
            &l->linear_in_qkv, g_qwen.d_debug_qkv, conv_dim,
            &l->linear_in_z, g_qwen.d_debug_z, value_dim,
            &l->linear_in_b, g_qwen.d_debug_proj, heads,
            &l->linear_in_a, g_qwen.d_debug_a_proj, heads);
        if (ret != 0) return -6;
    } else {
        ret = run_qmma_projection_from_input_to_buffer(&l->linear_in_qkv, g_qwen.d_debug_norm,
                                                       g_qwen.d_debug_qkv, conv_dim);
        if (ret != 0) return -6;
    }
    k_tq_linear_conv_update<<<(conv_dim + 255) / 256, 256, 0, g_qwen.stream>>>(
        g_qwen.d_debug_qkv, l->d_linear_conv_state, g_qwen.d_debug_qkv,
        l->d_linear_conv1d, conv_dim, g_qwen.linear_conv_kernel_dim);
    if (!(can_fuse_qmma_sf_tiled_input(&l->linear_in_qkv, &l->linear_in_z) &&
          can_fuse_qmma_sf_tiled_input(&l->linear_in_qkv, &l->linear_in_b) &&
          can_fuse_qmma_sf_tiled_input(&l->linear_in_qkv, &l->linear_in_a))) {
        ret = run_qmma_projection_from_input_to_buffer(&l->linear_in_z, g_qwen.d_debug_norm,
                                                       g_qwen.d_debug_z, value_dim);
        if (ret != 0) return -7;
        ret = run_qmma_projection_from_input_to_buffer(&l->linear_in_b, g_qwen.d_debug_norm,
                                                       g_qwen.d_debug_proj, heads);
        if (ret != 0) return -8;
        ret = run_qmma_projection_from_input_to_buffer(&l->linear_in_a, g_qwen.d_debug_norm,
                                                       g_qwen.d_debug_a_proj, heads);
        if (ret != 0) return -9;
    }
    int lc_g = linear_core_par_g();
    int lc_hs = linear_core_hs();
    bool lc_par_ok = lc_g > 0 && dim > 0 && dim <= 128 &&
                     (dim & (dim - 1)) == 0 && dim * lc_g <= 1024;
    bool lc_hs_ok = lc_par_ok && lc_hs >= 2 && (dim % lc_hs) == 0 && lc_g >= lc_hs &&
                    ((lc_g == 8 && (lc_hs == 2 || lc_hs == 4 || lc_hs == 8)) ||
                     (lc_g == 4 && (lc_hs == 2 || lc_hs == 4)));
    if (lc_hs_ok) {
        size_t shA = (size_t)3 * dim * sizeof(float);
        size_t shB = (size_t)dim * sizeof(float);
        int blkA = (dim / lc_hs) * lc_g;
        dim3 gridA(heads * lc_hs);
        #define TQ_LAUNCH_CORE_HS(GG, HH)                                              \
            k_tq_linear_decode_core_hs<GG, HH><<<gridA, blkA, shA, g_qwen.stream>>>(    \
                g_qwen.d_debug_core, l->d_linear_recurrent_state, g_qwen.d_debug_qkv,   \
                g_qwen.d_debug_proj, g_qwen.d_debug_a_proj,                             \
                l->d_linear_A_log, l->d_linear_dt_bias, heads, key_heads, dim)
        if (lc_g == 8 && lc_hs == 8) TQ_LAUNCH_CORE_HS(8, 8);
        else if (lc_g == 8 && lc_hs == 4) TQ_LAUNCH_CORE_HS(8, 4);
        else if (lc_g == 8 && lc_hs == 2) TQ_LAUNCH_CORE_HS(8, 2);
        else if (lc_g == 4 && lc_hs == 4) TQ_LAUNCH_CORE_HS(4, 4);
        else TQ_LAUNCH_CORE_HS(4, 2);
        #undef TQ_LAUNCH_CORE_HS
        k_tq_linear_decode_gated_norm<<<heads, dim, shB, g_qwen.stream>>>(
            g_qwen.d_debug_core, g_qwen.d_debug_z, l->d_linear_norm, heads, dim, g_qwen.eps);
    } else if (lc_par_ok) {
        size_t lc_shbytes = (size_t)3 * dim * sizeof(float);
        if (lc_g == 8)
            k_tq_linear_decode_core_gated_par<8><<<heads, dim * 8, lc_shbytes, g_qwen.stream>>>(
                g_qwen.d_debug_core, l->d_linear_recurrent_state, g_qwen.d_debug_qkv,
                g_qwen.d_debug_z, g_qwen.d_debug_proj, g_qwen.d_debug_a_proj,
                l->d_linear_A_log, l->d_linear_dt_bias, l->d_linear_norm,
                heads, key_heads, dim, g_qwen.eps);
        else if (lc_g == 4)
            k_tq_linear_decode_core_gated_par<4><<<heads, dim * 4, lc_shbytes, g_qwen.stream>>>(
                g_qwen.d_debug_core, l->d_linear_recurrent_state, g_qwen.d_debug_qkv,
                g_qwen.d_debug_z, g_qwen.d_debug_proj, g_qwen.d_debug_a_proj,
                l->d_linear_A_log, l->d_linear_dt_bias, l->d_linear_norm,
                heads, key_heads, dim, g_qwen.eps);
        else
            k_tq_linear_decode_core_gated_par<2><<<heads, dim * 2, lc_shbytes, g_qwen.stream>>>(
                g_qwen.d_debug_core, l->d_linear_recurrent_state, g_qwen.d_debug_qkv,
                g_qwen.d_debug_z, g_qwen.d_debug_proj, g_qwen.d_debug_a_proj,
                l->d_linear_A_log, l->d_linear_dt_bias, l->d_linear_norm,
                heads, key_heads, dim, g_qwen.eps);
    } else {
        k_tq_linear_decode_core_gated<<<heads, dim, 0, g_qwen.stream>>>(
            g_qwen.d_debug_core, l->d_linear_recurrent_state, g_qwen.d_debug_qkv,
            g_qwen.d_debug_z, g_qwen.d_debug_proj, g_qwen.d_debug_a_proj,
            l->d_linear_A_log, l->d_linear_dt_bias, l->d_linear_norm,
            heads, key_heads, dim, g_qwen.eps);
    }
    ret = run_qmma_projection_from_input_add_to_buffer(&l->linear_out, g_qwen.d_debug_core,
                                                       g_qwen.d_debug_x, g_qwen.d_debug_resid,
                                                       g_qwen.H);
    if (ret != 0) return -10;
    ret = run_mlp_from_resid_to_layer_out(layer);
    if (ret != 0) return -11;
    return 0;
}

static int run_full_layer_decode_from_current(int layer, int pos) {
    tq_layer_t *l = &g_qwen.layers[layer];
    int q_m = g_qwen.nh * g_qwen.hd * 2;
    int kv_m = g_qwen.nkv * g_qwen.hd;
    int attn_m = g_qwen.nh * g_qwen.hd;
    if (ensure_float_buffer(&g_qwen.d_debug_qkv, &g_qwen.debug_qkv_count, q_m, "d_debug_qkv") != 0) return -1;
    if (ensure_float_buffer(&g_qwen.d_debug_a_proj, &g_qwen.debug_a_count, kv_m, "d_debug_a_proj") != 0) return -2;
    if (ensure_float_buffer(&g_qwen.d_debug_z, &g_qwen.debug_z_count, kv_m, "d_debug_z") != 0) return -3;
    if (ensure_float_buffer(&g_qwen.d_debug_core, &g_qwen.debug_core_count, attn_m, "d_debug_core") != 0) return -4;
    if (ensure_float_buffer(&g_qwen.d_debug_resid, &g_qwen.debug_resid_count, g_qwen.H, "d_debug_resid") != 0) return -10;
    if (ensure_float_buffer(&g_qwen.d_debug_resid, &g_qwen.debug_resid_count, g_qwen.H, "d_debug_resid") != 0) return -10;
    k_tq_qwen_rmsnorm<<<8, 1024, 0, g_qwen.stream>>>(g_qwen.d_debug_norm, g_qwen.d_debug_x,
                                                       l->d_input_ln, g_qwen.H, g_qwen.eps);
    int ret;
    if (can_fuse_qmma_sf_tiled_input(&l->q_proj, &l->k_proj) &&
        can_fuse_qmma_sf_tiled_input(&l->q_proj, &l->v_proj)) {
        ret = prepare_qmma_sf_input(&l->q_proj, g_qwen.d_debug_norm);
        if (ret != 0) return -5;
        ret = run_qmma_sf_prepared_multi_to_buffers(
            &l->q_proj, g_qwen.d_debug_qkv, q_m,
            &l->k_proj, g_qwen.d_debug_a_proj, kv_m,
            &l->v_proj, g_qwen.d_debug_z, kv_m,
            NULL, NULL, 0);
        if (ret != 0) return -5;
    } else {
        ret = run_qmma_projection_from_input_to_buffer(&l->q_proj, g_qwen.d_debug_norm,
                                                       g_qwen.d_debug_qkv, q_m);
        if (ret != 0) return -5;
        ret = run_qmma_projection_from_input_to_buffer(&l->k_proj, g_qwen.d_debug_norm,
                                                       g_qwen.d_debug_a_proj, kv_m);
        if (ret != 0) return -6;
        ret = run_qmma_projection_from_input_to_buffer(&l->v_proj, g_qwen.d_debug_norm,
                                                       g_qwen.d_debug_z, kv_m);
        if (ret != 0) return -7;
    }
    if (ensure_attn_scores() != 0) return -97;
    k_tq_full_attn_decode<<<g_qwen.nh, g_qwen.hd, 0, g_qwen.stream>>>(
        g_qwen.d_debug_core, g_qwen.d_debug_qkv, g_qwen.d_debug_a_proj,
        g_qwen.d_debug_z, l->d_q_norm, l->d_k_norm, l->d_k_cache, l->d_v_cache,
        l->d_k_cache8, l->d_k_scale, l->d_v_cache8, l->d_v_scale,
        l->d_k_cache4, l->d_k_q4s,
        pos, g_qwen.nh, g_qwen.nkv, g_qwen.hd, g_qwen.eps, g_qwen.rope_theta,
        g_qwen.d_attn_scores, g_qwen.max_seq);
    ret = run_qmma_projection_from_input_add_to_buffer(&l->o_proj, g_qwen.d_debug_core,
                                                       g_qwen.d_debug_x, g_qwen.d_debug_resid,
                                                       g_qwen.H);
    if (ret != 0) return -8;
    ret = run_mlp_from_resid_to_layer_out(layer);
    if (ret != 0) return -9;
    return 0;
}

static int run_full_layer_decode_from_current_graph(int layer) {
    tq_layer_t *l = &g_qwen.layers[layer];
    int q_m = g_qwen.nh * g_qwen.hd * 2;
    int kv_m = g_qwen.nkv * g_qwen.hd;
    int attn_m = g_qwen.nh * g_qwen.hd;
    if (ensure_float_buffer(&g_qwen.d_debug_qkv, &g_qwen.debug_qkv_count, q_m, "d_debug_qkv") != 0) return -1;
    if (ensure_float_buffer(&g_qwen.d_debug_a_proj, &g_qwen.debug_a_count, kv_m, "d_debug_a_proj") != 0) return -2;
    if (ensure_float_buffer(&g_qwen.d_debug_z, &g_qwen.debug_z_count, kv_m, "d_debug_z") != 0) return -3;
    if (ensure_float_buffer(&g_qwen.d_debug_core, &g_qwen.debug_core_count, attn_m, "d_debug_core") != 0) return -4;
    k_tq_qwen_rmsnorm<<<8, 1024, 0, g_qwen.stream>>>(g_qwen.d_debug_norm, g_qwen.d_debug_x,
                                                       l->d_input_ln, g_qwen.H, g_qwen.eps);
    int ret;
    if (can_fuse_qmma_sf_tiled_input(&l->q_proj, &l->k_proj) &&
        can_fuse_qmma_sf_tiled_input(&l->q_proj, &l->v_proj)) {
        ret = prepare_qmma_sf_input(&l->q_proj, g_qwen.d_debug_norm);
        if (ret != 0) return -5;
        ret = run_qmma_sf_prepared_multi_to_buffers(
            &l->q_proj, g_qwen.d_debug_qkv, q_m,
            &l->k_proj, g_qwen.d_debug_a_proj, kv_m,
            &l->v_proj, g_qwen.d_debug_z, kv_m,
            NULL, NULL, 0);
        if (ret != 0) return -5;
    } else {
        ret = run_qmma_projection_from_input_to_buffer(&l->q_proj, g_qwen.d_debug_norm,
                                                       g_qwen.d_debug_qkv, q_m);
        if (ret != 0) return -5;
        ret = run_qmma_projection_from_input_to_buffer(&l->k_proj, g_qwen.d_debug_norm,
                                                       g_qwen.d_debug_a_proj, kv_m);
        if (ret != 0) return -6;
        ret = run_qmma_projection_from_input_to_buffer(&l->v_proj, g_qwen.d_debug_norm,
                                                       g_qwen.d_debug_z, kv_m);
        if (ret != 0) return -7;
    }
    if (ensure_attn_scores() != 0) return -97;
    k_tq_full_attn_decode_ptr<<<g_qwen.nh, g_qwen.hd, 0, g_qwen.stream>>>(
        g_qwen.d_debug_core, g_qwen.d_debug_qkv, g_qwen.d_debug_a_proj,
        g_qwen.d_debug_z, l->d_q_norm, l->d_k_norm, l->d_k_cache, l->d_v_cache,
        l->d_k_cache8, l->d_k_scale, l->d_v_cache8, l->d_v_scale,
        l->d_k_cache4, l->d_k_q4s,
        g_qwen.d_decode_pos, g_qwen.nh, g_qwen.nkv, g_qwen.hd, g_qwen.eps, g_qwen.rope_theta,
        g_qwen.d_attn_scores, g_qwen.max_seq);
    ret = run_qmma_projection_from_input_add_to_buffer(&l->o_proj, g_qwen.d_debug_core,
                                                       g_qwen.d_debug_x, g_qwen.d_debug_resid,
                                                       g_qwen.H);
    if (ret != 0) return -8;
    ret = run_mlp_from_resid_to_layer_out(layer);
    if (ret != 0) return -9;
    return 0;
}

extern "C" int qwn_debug_linear_first_core(int token_id, int layer, float *out, int max_count) {
    if (!out || max_count <= 0) {
        fprintf(stderr, "qwn_debug_linear_first_core: invalid output buffer\n");
        return -4;
    }
    int ret = run_linear_first_core_to_debug_core(token_id, layer);
    if (ret != 0) return ret;
    int value_dim = g_qwen.linear_num_value_heads * g_qwen.linear_value_head_dim;
    int count = max_count < value_dim ? max_count : value_dim;
    cudaError_t err = cudaMemcpy(out, g_qwen.d_debug_core, (size_t)count * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "linear first core output copy failed: %s\n", cudaGetErrorString(err));
        return -19;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "linear first core synchronize failed: %s\n", cudaGetErrorString(err));
        return -20;
    }
    return count;
}

extern "C" int qwn_debug_linear_first_output(int token_id, int layer, float *out, int max_count) {
    if (!out || max_count <= 0) {
        fprintf(stderr, "qwn_debug_linear_first_output: invalid output buffer\n");
        return -4;
    }
    if (layer < 0 || layer >= g_qwen.L || g_qwen.layer_types[layer] != TQ_LAYER_LINEAR_ATTENTION) {
        fprintf(stderr, "qwn_debug_linear_first_output: layer %d is not linear_attention\n", layer);
        return -10;
    }
    int ret = run_linear_first_core_to_debug_core(token_id, layer);
    if (ret != 0) return -11;
    tq_layer_t *l = &g_qwen.layers[layer];
    ret = run_qmma_projection_from_input_to_debug_out(&l->linear_out, g_qwen.d_debug_core);
    if (ret != 0) return -12;
    int count = max_count < g_qwen.H ? max_count : g_qwen.H;
    cudaError_t err = cudaMemcpy(out, g_qwen.d_debug_qmma_out, (size_t)count * sizeof(float),
                                 cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "linear first output copy failed: %s\n", cudaGetErrorString(err));
        return -13;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "linear first output synchronize failed: %s\n", cudaGetErrorString(err));
        return -14;
    }
    return count;
}

extern "C" int qwn_debug_linear_layer_output(int token_id, int layer, float *out, int max_count) {
    if (!out || max_count <= 0) {
        fprintf(stderr, "qwn_debug_linear_layer_output: invalid output buffer\n");
        return -4;
    }
    if (layer < 0 || layer >= g_qwen.L || g_qwen.layer_types[layer] != TQ_LAYER_LINEAR_ATTENTION) {
        fprintf(stderr, "qwn_debug_linear_layer_output: layer %d is not linear_attention\n", layer);
        return -10;
    }
    tq_layer_t *l = &g_qwen.layers[layer];
    int ret = run_debug_embed_norm(token_id, layer);
    if (ret != 0) return ret;
    ret = run_linear_first_core_to_debug_core(token_id, layer);
    if (ret != 0) return -11;
    ret = run_qmma_projection_from_input_to_debug_out(&l->linear_out, g_qwen.d_debug_core);
    if (ret != 0) return -12;

    if (ensure_float_buffer(&g_qwen.d_debug_resid, &g_qwen.debug_resid_count, g_qwen.H, "d_debug_resid") != 0) return -13;
    if (ensure_float_buffer(&g_qwen.d_debug_post_norm, &g_qwen.debug_post_norm_count, g_qwen.H, "d_debug_post_norm") != 0) return -14;
    if (ensure_float_buffer(&g_qwen.d_debug_gate, &g_qwen.debug_gate_count, g_qwen.I, "d_debug_gate") != 0) return -15;
    if (ensure_float_buffer(&g_qwen.d_debug_up, &g_qwen.debug_up_count, g_qwen.I, "d_debug_up") != 0) return -16;
    if (ensure_float_buffer(&g_qwen.d_debug_mlp_hidden, &g_qwen.debug_mlp_hidden_count, g_qwen.I, "d_debug_mlp_hidden") != 0) return -17;
    if (ensure_float_buffer(&g_qwen.d_debug_layer_out, &g_qwen.debug_layer_out_count, g_qwen.H, "d_debug_layer_out") != 0) return -18;

    k_tq_add_vec<<<(g_qwen.H + 255) / 256, 256, 0, g_qwen.stream>>>(g_qwen.d_debug_resid, g_qwen.d_debug_x,
                                                  g_qwen.d_debug_qmma_out, g_qwen.H);
    k_tq_qwen_rmsnorm<<<8, 1024, 0, g_qwen.stream>>>(g_qwen.d_debug_post_norm, g_qwen.d_debug_resid,
                                   l->d_post_ln, g_qwen.H, g_qwen.eps);
    ret = run_qmma_projection_from_input_to_buffer(&l->mlp_gate, g_qwen.d_debug_post_norm,
                                                   g_qwen.d_debug_gate, g_qwen.I);
    if (ret != 0) return -19;
    ret = run_qmma_projection_from_input_to_buffer(&l->mlp_up, g_qwen.d_debug_post_norm,
                                                   g_qwen.d_debug_up, g_qwen.I);
    if (ret != 0) return -20;
    k_tq_silu_mul<<<(g_qwen.I + 255) / 256, 256, 0, g_qwen.stream>>>(g_qwen.d_debug_mlp_hidden,
                                                   g_qwen.d_debug_gate, g_qwen.d_debug_up,
                                                   g_qwen.I);
    ret = run_qmma_projection_from_input_to_debug_out(&l->mlp_down, g_qwen.d_debug_mlp_hidden);
    if (ret != 0) return -21;
    k_tq_add_vec<<<(g_qwen.H + 255) / 256, 256, 0, g_qwen.stream>>>(g_qwen.d_debug_layer_out,
                                                  g_qwen.d_debug_resid,
                                                  g_qwen.d_debug_qmma_out, g_qwen.H);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "linear layer output launch failed: %s\n", cudaGetErrorString(err));
        return -22;
    }
    int count = max_count < g_qwen.H ? max_count : g_qwen.H;
    err = cudaMemcpy(out, g_qwen.d_debug_layer_out, (size_t)count * sizeof(float),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "linear layer output copy failed: %s\n", cudaGetErrorString(err));
        return -23;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "linear layer output synchronize failed: %s\n", cudaGetErrorString(err));
        return -24;
    }
    return count;
}

static int run_forward_layers_to_debug_x(int token_id, int n_layers) {
    if (!g_qwen.initialized) {
        fprintf(stderr, "forward layers: model is not initialized\n");
        return -1;
    }
    if (token_id < 0 || token_id >= g_qwen.V) {
        fprintf(stderr, "forward layers: invalid token=%d\n", token_id);
        return -2;
    }
    if (n_layers < 0 || n_layers > g_qwen.L) {
        fprintf(stderr, "forward layers: invalid n_layers=%d\n", n_layers);
        return -3;
    }
    int ret = ensure_debug_vectors(0);
    if (ret != 0) return -4;
    k_tq_embed_lookup<<<(g_qwen.H + 255) / 256, 256, 0, g_qwen.stream>>>(g_qwen.d_debug_x, g_qwen.d_embed,
                                                       token_id, g_qwen.H);
    for (int layer = 0; layer < n_layers; layer++) {
        if (g_qwen.layer_types[layer] == TQ_LAYER_LINEAR_ATTENTION) {
            ret = run_linear_layer_from_current_to_layer_out(layer);
        } else if (g_qwen.layer_types[layer] == TQ_LAYER_FULL_ATTENTION) {
            ret = run_full_layer_from_current_to_layer_out(layer);
        } else {
            fprintf(stderr, "forward layers: unsupported layer type %u at layer %d\n",
                    g_qwen.layer_types[layer], layer);
            return -5;
        }
        if (ret != 0) {
            fprintf(stderr, "forward layers: layer %d failed ret=%d\n", layer, ret);
            return -6;
        }
        cudaError_t err = cudaMemcpyAsync(g_qwen.d_debug_x, g_qwen.d_debug_layer_out,
                                          (size_t)g_qwen.H * sizeof(float),
                                          cudaMemcpyDeviceToDevice, g_qwen.stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "forward layers: hidden copy failed: %s\n", cudaGetErrorString(err));
            return -7;
        }
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "forward layers launch error: %s\n", cudaGetErrorString(err));
        return -8;
    }
    return 0;
}

extern "C" int qwn_debug_forward_layers(int token_id, int n_layers, float *out, int max_count) {
    if (!out || max_count <= 0) {
        fprintf(stderr, "qwn_debug_forward_layers: invalid output buffer\n");
        return -4;
    }
    int ret = run_forward_layers_to_debug_x(token_id, n_layers);
    if (ret != 0) return ret;
    int count = max_count < g_qwen.H ? max_count : g_qwen.H;
    cudaError_t err = cudaMemcpy(out, g_qwen.d_debug_x, (size_t)count * sizeof(float),
                                 cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "forward layers output copy failed: %s\n", cudaGetErrorString(err));
        return -9;
    }
    err = cudaStreamSynchronize(g_qwen.stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "forward layers synchronize failed: %s\n", cudaGetErrorString(err));
        return -10;
    }
    return count;
}

extern "C" int qwn_debug_forward_final_norm(int token_id, float *out, int max_count) {
    if (!out || max_count <= 0) {
        fprintf(stderr, "qwn_debug_forward_final_norm: invalid output buffer\n");
        return -4;
    }
    int ret = run_forward_layers_to_debug_x(token_id, g_qwen.L);
    if (ret != 0) return ret;
    k_tq_qwen_rmsnorm<<<8, 1024, 0, g_qwen.stream>>>(g_qwen.d_debug_norm, g_qwen.d_debug_x,
                                   g_qwen.d_norm, g_qwen.H, g_qwen.eps);
    int count = max_count < g_qwen.H ? max_count : g_qwen.H;
    cudaError_t err = cudaMemcpy(out, g_qwen.d_debug_norm, (size_t)count * sizeof(float),
                                 cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "forward final norm output copy failed: %s\n", cudaGetErrorString(err));
        return -9;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "forward final norm synchronize failed: %s\n", cudaGetErrorString(err));
        return -10;
    }
    return count;
}

static int run_final_norm_device(void) {
    k_tq_qwen_rmsnorm<<<8, 1024, 0, g_qwen.stream>>>(g_qwen.d_debug_norm, g_qwen.d_debug_x,
                                   g_qwen.d_norm, g_qwen.H, g_qwen.eps);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "final norm launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

static int run_lm_head_argmax_device(void) {
    if (ensure_float_buffer(&g_qwen.d_debug_arg_vals, &g_qwen.debug_arg_count, TQ_ARGMAX_BLOCKS + 1, "d_debug_arg_vals") != 0) return -3;
    if (ensure_int_buffer(&g_qwen.d_debug_arg_ids, &g_qwen.debug_arg_ids_count, TQ_ARGMAX_BLOCKS + 1, "d_debug_arg_ids") != 0) return -4;
    int topk = optimistic_topk();
    g_qwen.optimistic_topk_k = topk;
    // Allocate the small top-k scratch once, on the first (eager) call, so that
    // no cudaMalloc happens later inside the captured decode graph.
    if (topk && !g_qwen.d_optimistic_topk_vals) {
        cudaMalloc(&g_qwen.d_optimistic_topk_vals, (size_t)TQ_OPTIMISTIC_TOPK_MAX * sizeof(float));
        cudaMalloc(&g_qwen.d_optimistic_topk_ids, (size_t)TQ_OPTIMISTIC_TOPK_MAX * sizeof(int));
        cudaMalloc(&g_qwen.d_optimistic_margin, sizeof(float));
    }
    // The optimistic top-k path needs materialized logits to mask candidates,
    // so it always takes the non-fused branch (the fused argmax never writes
    // d_debug_logits). The default (topk==0) path is unchanged.
    if (!topk && use_fused_lm_head() && !g_qwen.tie_word_embeddings && g_qwen.lm_head.block_scaled) {
        if (g_qwen.lm_head.row_major) {
            k_tq_lm_head_argmax_block_scaled_rowmajor<<<TQ_ARGMAX_BLOCKS, 256, 0, g_qwen.stream>>>(
                g_qwen.d_debug_arg_vals, g_qwen.d_debug_arg_ids, g_qwen.lm_head.d_A,
                g_qwen.V, g_qwen.H, g_qwen.lm_head.d_block_scale_inv,
                g_qwen.lm_head.scale_cols, g_qwen.d_debug_norm);
        } else {
            k_tq_lm_head_argmax_block_scaled<<<TQ_ARGMAX_BLOCKS, 256, 0, g_qwen.stream>>>(
                g_qwen.d_debug_arg_vals, g_qwen.d_debug_arg_ids, g_qwen.lm_head.d_A,
                g_qwen.V, g_qwen.H, g_qwen.lm_head.Kt, g_qwen.lm_head.d_block_scale_inv,
                g_qwen.lm_head.scale_cols, g_qwen.d_debug_norm);
        }
    } else {
        if (ensure_float_buffer(&g_qwen.d_debug_logits, &g_qwen.debug_logits_count, g_qwen.V, "d_debug_logits") != 0) return -2;
        if (g_qwen.tie_word_embeddings) {
        k_tq_lm_head_bf16<<<g_qwen.V, 256, 0, g_qwen.stream>>>(g_qwen.d_debug_logits, g_qwen.d_debug_norm,
                                             g_qwen.d_embed, g_qwen.H, g_qwen.V);
        } else {
            int ret = run_qmma_projection_from_input_to_buffer(&g_qwen.lm_head, g_qwen.d_debug_norm,
                                                               g_qwen.d_debug_logits, g_qwen.V);
            if (ret != 0) return -6;
        }
        k_tq_argmax_stage1<<<TQ_ARGMAX_BLOCKS, 256, 0, g_qwen.stream>>>(g_qwen.d_debug_arg_vals, g_qwen.d_debug_arg_ids,
                                                      g_qwen.d_debug_logits, g_qwen.V);
    }
    k_tq_argmax_stage2<<<1, 1024, 0, g_qwen.stream>>>(g_qwen.d_debug_arg_vals + TQ_ARGMAX_BLOCKS,
                                    g_qwen.d_debug_arg_ids + TQ_ARGMAX_BLOCKS,
                                    g_qwen.d_debug_arg_vals, g_qwen.d_debug_arg_ids, TQ_ARGMAX_BLOCKS);
    if (topk) {
        // Pass 0: the just-computed argmax is candidate 0. Then repeatedly mask
        // the winner and re-run the argmax reduction to peel off the next best.
        k_tq_topk_record_and_mask<<<1, 1, 0, g_qwen.stream>>>(
            g_qwen.d_debug_logits, g_qwen.d_optimistic_topk_vals, g_qwen.d_optimistic_topk_ids,
            g_qwen.d_debug_arg_vals + TQ_ARGMAX_BLOCKS, g_qwen.d_debug_arg_ids + TQ_ARGMAX_BLOCKS, 0);
        for (int i = 1; i < topk; i++) {
            k_tq_argmax_stage1<<<TQ_ARGMAX_BLOCKS, 256, 0, g_qwen.stream>>>(
                g_qwen.d_debug_arg_vals, g_qwen.d_debug_arg_ids, g_qwen.d_debug_logits, g_qwen.V);
            k_tq_argmax_stage2<<<1, 1024, 0, g_qwen.stream>>>(
                g_qwen.d_debug_arg_vals + TQ_ARGMAX_BLOCKS, g_qwen.d_debug_arg_ids + TQ_ARGMAX_BLOCKS,
                g_qwen.d_debug_arg_vals, g_qwen.d_debug_arg_ids, TQ_ARGMAX_BLOCKS);
            k_tq_topk_record_and_mask<<<1, 1, 0, g_qwen.stream>>>(
                g_qwen.d_debug_logits, g_qwen.d_optimistic_topk_vals, g_qwen.d_optimistic_topk_ids,
                g_qwen.d_debug_arg_vals + TQ_ARGMAX_BLOCKS, g_qwen.d_debug_arg_ids + TQ_ARGMAX_BLOCKS, i);
        }
        // Winner = top-1 (lossless); also publishes the top1-top2 margin.
        k_tq_optimistic_winner<<<1, 1, 0, g_qwen.stream>>>(
            g_qwen.d_debug_arg_vals + TQ_ARGMAX_BLOCKS, g_qwen.d_debug_arg_ids + TQ_ARGMAX_BLOCKS,
            g_qwen.d_optimistic_margin, g_qwen.d_optimistic_topk_vals, g_qwen.d_optimistic_topk_ids, topk);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "lm head/argmax launch failed: %s\n", cudaGetErrorString(err));
        return -5;
    }
    return 0;
}

static int run_forward_argmax_device(int token_id) {
    int ret = run_forward_layers_to_debug_x(token_id, g_qwen.L);
    if (ret != 0) return -1;
    ret = run_final_norm_device();
    if (ret != 0) return -2;
    ret = run_lm_head_argmax_device();
    if (ret != 0) return -3;
    return 0;
}

static int copy_forward_argmax_result(float *logit_out) {
    int h_id = 0;
    float h_val = 0.0f;
    cudaError_t err = cudaStreamSynchronize(g_qwen.stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "forward argmax stream sync failed: %s\n", cudaGetErrorString(err));
        return -8;
    }
    err = cudaMemcpy(&h_id, g_qwen.d_debug_arg_ids + TQ_ARGMAX_BLOCKS, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "forward argmax id copy failed: %s\n", cudaGetErrorString(err));
        return -6;
    }
    err = cudaMemcpy(&h_val, g_qwen.d_debug_arg_vals + TQ_ARGMAX_BLOCKS, sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "forward argmax value copy failed: %s\n", cudaGetErrorString(err));
        return -7;
    }
    if (logit_out) *logit_out = h_val;
    g_qwen.last_argmax_id = h_id;
    g_qwen.last_argmax_logit = h_val;
    return h_id;
}

extern "C" int qwn_debug_forward_argmax(int token_id, float *logit_out) {
    int ret = run_forward_argmax_device(token_id);
    if (ret != 0) return ret;
    return copy_forward_argmax_result(logit_out);
}

extern "C" int qwn_forward_first(int token_id) {
    return qwn_debug_forward_argmax(token_id, NULL);
}

static int run_decode_layers_to_debug_x(int token_id, int pos, int n_layers) {
    if (pos < 0 || pos >= g_qwen.max_seq) {
        fprintf(stderr, "decode layers: invalid pos=%d\n", pos);
        return -1;
    }
    if (n_layers < 0 || n_layers > g_qwen.L) {
        fprintf(stderr, "decode layers: invalid n_layers=%d\n", n_layers);
        return -8;
    }
    if (pos == 0) {
        int r = qwn_reset_state();
        if (r != 0) return r;
    } else {
        int r = ensure_decode_state();
        if (r != 0) return r;
    }
    int scratch = ensure_debug_vectors(0);
    if (scratch != 0) return -7;
    k_tq_embed_lookup<<<(g_qwen.H + 255) / 256, 256, 0, g_qwen.stream>>>(
        g_qwen.d_debug_x, g_qwen.d_embed, token_id, g_qwen.H);
    for (int layer = 0; layer < n_layers; layer++) {
        int ret;
        rmap_hook_after_layer(layer - 1, g_qwen.d_debug_x, 1);
        if (g_qwen.layer_types[layer] == TQ_LAYER_LINEAR_ATTENTION) {
            ret = run_linear_layer_decode_from_current(layer);
        } else if (g_qwen.layer_types[layer] == TQ_LAYER_FULL_ATTENTION) {
            ret = run_full_layer_decode_from_current(layer, pos);
        } else {
            return -2;
        }
        if (ret != 0) {
            fprintf(stderr, "decode layers: layer %d failed ret=%d\n", layer, ret);
            return -3;
        }
        cudaError_t err = cudaMemcpyAsync(g_qwen.d_debug_x, g_qwen.d_debug_layer_out,
                                          (size_t)g_qwen.H * sizeof(float),
                                          cudaMemcpyDeviceToDevice, g_qwen.stream);
        if (err != cudaSuccess) return -4;
    }
    rmap_hook_after_layer(n_layers - 1, g_qwen.d_debug_x, 1);
    return 0;
}

static int ensure_decode_graph_params(void) {
    if (!g_qwen.d_decode_token) {
        cudaError_t err = cudaMalloc(&g_qwen.d_decode_token, sizeof(int));
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMalloc d_decode_token failed: %s\n", cudaGetErrorString(err));
            return -1;
        }
        g_qwen.device_bytes += sizeof(int);
    }
    if (!g_qwen.d_decode_pos) {
        cudaError_t err = cudaMalloc(&g_qwen.d_decode_pos, sizeof(int));
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMalloc d_decode_pos failed: %s\n", cudaGetErrorString(err));
            return -2;
        }
        g_qwen.device_bytes += sizeof(int);
    }
    if (!g_qwen.d_decode_step) {
        cudaError_t err = cudaMalloc(&g_qwen.d_decode_step, sizeof(int));
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMalloc d_decode_step failed: %s\n", cudaGetErrorString(err));
            return -5;
        }
        g_qwen.device_bytes += sizeof(int);
    }
    return 0;
}

static int ensure_decode_generated(int count) {
    if (count <= 0) count = 1;
    if (g_qwen.decode_generated_count >= count) return 0;
    if (g_qwen.d_decode_generated) {
        cudaFree(g_qwen.d_decode_generated);
        g_qwen.device_bytes -= (size_t)g_qwen.decode_generated_count * sizeof(int);
    }
    cudaError_t err = cudaMalloc(&g_qwen.d_decode_generated, (size_t)count * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_decode_generated failed: %s\n", cudaGetErrorString(err));
        g_qwen.d_decode_generated = NULL;
        g_qwen.decode_generated_count = 0;
        return -1;
    }
    g_qwen.decode_generated_count = count;
    g_qwen.device_bytes += (size_t)count * sizeof(int);
    destroy_decode_graph();
    return 0;
}

static int set_decode_graph_params(int token_id, int pos) {
    int ret = ensure_decode_graph_params();
    if (ret != 0) return ret;
    int step = 0;
    cudaError_t err = cudaMemcpyAsync(g_qwen.d_decode_token, &token_id, sizeof(int),
                                      cudaMemcpyHostToDevice, g_qwen.stream);
    if (err != cudaSuccess) return -3;
    err = cudaMemcpyAsync(g_qwen.d_decode_pos, &pos, sizeof(int),
                          cudaMemcpyHostToDevice, g_qwen.stream);
    if (err != cudaSuccess) return -4;
    err = cudaMemcpyAsync(g_qwen.d_decode_step, &step, sizeof(int),
                          cudaMemcpyHostToDevice, g_qwen.stream);
    if (err != cudaSuccess) return -5;
    return 0;
}

static int run_decode_layers_to_debug_x_graph_params(int n_layers) {
    if (n_layers < 0 || n_layers > g_qwen.L) return -8;
    int r = ensure_decode_state();
    if (r != 0) return r;
    int scratch = ensure_debug_vectors(0);
    if (scratch != 0) return -7;
    k_tq_embed_lookup_ptr<<<(g_qwen.H + 255) / 256, 256, 0, g_qwen.stream>>>(
        g_qwen.d_debug_x, g_qwen.d_embed, g_qwen.d_decode_token, g_qwen.H);
    for (int layer = 0; layer < n_layers; layer++) {
        int ret;
        rmap_hook_after_layer(layer - 1, g_qwen.d_debug_x, 1);
        if (g_qwen.layer_types[layer] == TQ_LAYER_LINEAR_ATTENTION) {
            ret = run_linear_layer_decode_from_current(layer);
        } else if (g_qwen.layer_types[layer] == TQ_LAYER_FULL_ATTENTION) {
            ret = run_full_layer_decode_from_current_graph(layer);
        } else {
            return -2;
        }
        if (ret != 0) return -3;
        cudaError_t err = cudaMemcpyAsync(g_qwen.d_debug_x, g_qwen.d_debug_layer_out,
                                          (size_t)g_qwen.H * sizeof(float),
                                          cudaMemcpyDeviceToDevice, g_qwen.stream);
        if (err != cudaSuccess) return -4;
    }
    rmap_hook_after_layer(n_layers - 1, g_qwen.d_debug_x, 1);
    return 0;
}

extern "C" int qwn_debug_decode_layers(int token_id, int pos, int n_layers, float *out, int max_count) {
    if (!out || max_count <= 0) {
        fprintf(stderr, "qwn_debug_decode_layers: invalid output buffer\n");
        return -9;
    }
    int ret = run_decode_layers_to_debug_x(token_id, pos, n_layers);
    if (ret != 0) return ret;
    int count = max_count < g_qwen.H ? max_count : g_qwen.H;
    cudaError_t err = cudaStreamSynchronize(g_qwen.stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "decode layers synchronize failed: %s\n", cudaGetErrorString(err));
        return -11;
    }
    err = cudaMemcpy(out, g_qwen.d_debug_x, (size_t)count * sizeof(float),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "decode layers output copy failed: %s\n", cudaGetErrorString(err));
        return -10;
    }
    return count;
}

extern "C" int qwn_decode(int token_id, int pos) {
    int ret = run_decode_layers_to_debug_x(token_id, pos, g_qwen.L);
    if (ret != 0) return ret;
    ret = run_final_norm_device();
    if (ret != 0) return -5;
    ret = run_lm_head_argmax_device();
    if (ret != 0) return -6;
    return copy_forward_argmax_result(NULL);
}
