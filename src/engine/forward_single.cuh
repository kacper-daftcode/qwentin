// the single-stream decoder qwn_decode (the test oracle of serve_batched.py --selftest and tools/paged_smoke.py) and the QMMA projection dispatch it shares with the batched path.
// Part of the forward_qwen.cu unity build: included in order, not a standalone header.

static int use_qmma_sf(void) {
    // E2M3 (6-bit dense) weights are ONLY correct via the QMMA.SF path. The
    // non-SF block-scaled fallback (k_tq_debug_qmma_gemv_block_scaled) dequants
    // as E4M3 and addresses 512-byte tiles, so on 6-bit E2M3 payloads it reads
    // wrong bytes — silent garbage at small dims (0.8B "pains"), out-of-bounds
    // IMA at 27B dims.
    return (g_qwen.flags & TQ_FLAG_BLOCK_SCALED_E2M3) != 0;
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

static int qmma_sf_multi_k_split(void) {
    // E2M3 weights need extra warps per fused GEMV to hide the 6-bit
    // unpack ALU (mid-grid kernels were ~55% of peak at split 1). For
    // E4M3 the fused GEMV is already bandwidth-bound, so keep 1.
    return (g_qwen.flags & TQ_FLAG_BLOCK_SCALED_E2M3) ? 8 : 1;
}

static int can_use_qmma_sf_weight(const tq_qmma_weight_t *w) {
    return use_qmma_sf() && w->block_scaled;
}

static int can_share_qmma_sf_input(const tq_qmma_weight_t *base, const tq_qmma_weight_t *w) {
    return can_use_qmma_sf_weight(base) && can_use_qmma_sf_weight(w) &&
           w->K == base->K && w->Kt == base->Kt;
}

static int can_fuse_qmma_sf_tiled_input(const tq_qmma_weight_t *base, const tq_qmma_weight_t *w) {
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

static int run_qmma_sf_prepared_multi_to_buffers(const tq_qmma_weight_t *w0, float *dst0, int dst_count0,
                                                 const tq_qmma_weight_t *w1, float *dst1, int dst_count1,
                                                 const tq_qmma_weight_t *w2, float *dst2, int dst_count2,
                                                 const tq_qmma_weight_t *w3, float *dst3, int dst_count3) {
    if (!w0 || !w1 || !dst0 || !dst1) return -30;
    if (!can_fuse_qmma_sf_tiled_input(w0, w1)) return -31;
    if (w2 && !can_fuse_qmma_sf_tiled_input(w0, w2)) return -32;
    if (w3 && !can_fuse_qmma_sf_tiled_input(w0, w3)) return -33;
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
    const int warps_per_block = 1, prefetch_distance = 0;
    int blocks = total_mt;
    int k_splits = qmma_sf_multi_k_split();
    int total_m = w0->M + w1->M + (w2 ? w2->M : 0) + (w3 ? w3->M : 0);
    if (k_splits > 1 && (w0->Kt % k_splits) == 0) {
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
            warps_per_block, k_splits, total_m, w0->word_major, 0);
        k_tq_reduce_qmma_sf_multi_splits<<<(total_m + 255) / 256, 256, 0, g_qwen.stream>>>(
            dst0, w0->M, dst1, w1->M, w2 ? dst2 : dst0, w2 ? w2->M : 0,
            w3 ? dst3 : dst0, w3 ? w3->M : 0, g_qwen.d_debug_qmma_raw, total_m, k_splits);
    } else {
        k_tq_qmma_sf_gemv_tiled_multi<<<blocks, warps_per_block * 32, 0, g_qwen.stream>>>(
            dst0, (const uint32_t *)w0->d_A, w0->Mt, w0->d_block_scale_inv, w0->scale_cols,
            dst1, (const uint32_t *)w1->d_A, w1->Mt, w1->d_block_scale_inv, w1->scale_cols,
            w2 ? dst2 : dst0, w2 ? (const uint32_t *)w2->d_A : (const uint32_t *)w0->d_A, mt2,
            w2 ? w2->d_block_scale_inv : w0->d_block_scale_inv, w2 ? w2->scale_cols : w0->scale_cols,
            w3 ? dst3 : dst0, w3 ? (const uint32_t *)w3->d_A : (const uint32_t *)w0->d_A, mt3,
            w3 ? w3->d_block_scale_inv : w0->d_block_scale_inv, w3 ? w3->scale_cols : w0->scale_cols,
            (const uint32_t *)g_qwen.d_debug_b, w0->Kt, g_qwen.d_debug_act_scale_inv,
            warps_per_block, w0->word_major, prefetch_distance, 0);
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
                k_splits, w->word_major, 0);
            k_tq_reduce_qmma_sf_splits<<<(w->M + 255) / 256, 256, 0, g_qwen.stream>>>(
                g_qwen.d_debug_qmma_out, g_qwen.d_debug_qmma_raw, w->M, k_splits);
        } else {
            k_tq_qmma_sf_gemv_tiled<<<w->Mt, 32, 0, g_qwen.stream>>>(
                g_qwen.d_debug_qmma_out, (const uint32_t *)w->d_A, (const uint32_t *)g_qwen.d_debug_b,
                w->Kt, w->d_block_scale_inv, w->scale_cols, g_qwen.d_debug_act_scale_inv,
                w->word_major, 0, 0);
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

static int run_qmma_projection_from_input_add_to_buffer(const tq_qmma_weight_t *w, const float *input,
                                                        const float *residual, float *dst,
                                                        int dst_count) {
    int ret = run_qmma_projection_from_input_to_debug_out(w, input);
    if (ret != 0) return ret;
    if (dst_count < w->M) return -3;
    k_tq_add_vec<<<(w->M + 255) / 256, 256, 0, g_qwen.stream>>>(
        dst, residual, g_qwen.d_debug_qmma_out, w->M);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -4;
    return 0;
}

static int run_qmma_projection_from_input_to_debug_out(const tq_qmma_weight_t *w, const float *input) {
    int ret = ensure_qmma_debug_scratch(w);
    if (ret != 0) return ret;
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
    k_tq_debug_qmma_gemv<<<w->M, 256, 0, g_qwen.stream>>>(
        g_qwen.d_debug_qmma_out, w->d_A, w->M, w->K, w->Kt, w->inv_scale, input);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "dequant GEMV launch failed: %s\n", cudaGetErrorString(err));
        return -2;
    }
    return 0;
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
    k_tq_silu_mul<<<(g_qwen.I + 255) / 256, 256, 0, g_qwen.stream>>>(g_qwen.d_debug_mlp_hidden,
                                                   g_qwen.d_debug_gate, g_qwen.d_debug_up,
                                                   g_qwen.I);
    ret = run_qmma_projection_from_input_to_debug_out(&l->mlp_down, g_qwen.d_debug_mlp_hidden);
    if (ret != 0) return -9;
    k_tq_add_vec<<<(g_qwen.H + 255) / 256, 256, 0, g_qwen.stream>>>(g_qwen.d_debug_layer_out,
                                                  g_qwen.d_debug_resid,
                                                  g_qwen.d_debug_qmma_out, g_qwen.H);
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
    const int lc_g = 8, lc_hs = 4;                                      // paged_linear_core's split
    bool lc_par_ok = dim > 0 && dim <= 128 && (dim & (dim - 1)) == 0;
    bool lc_hs_ok = lc_par_ok && (dim % lc_hs) == 0;
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
        TQ_LAUNCH_CORE_HS(8, 4);
        #undef TQ_LAUNCH_CORE_HS
        k_tq_linear_decode_gated_norm<<<heads, dim, shB, g_qwen.stream>>>(
            g_qwen.d_debug_core, g_qwen.d_debug_z, l->d_linear_norm, heads, dim, g_qwen.eps);
    } else if (lc_par_ok) {
        size_t lc_shbytes = (size_t)3 * dim * sizeof(float);
        k_tq_linear_decode_core_gated_par<8><<<heads, dim * 8, lc_shbytes, g_qwen.stream>>>(
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
    k_tq_argmax_stage2<<<1, 1024, 0, g_qwen.stream>>>(g_qwen.d_debug_arg_vals + TQ_ARGMAX_BLOCKS,
                                    g_qwen.d_debug_arg_ids + TQ_ARGMAX_BLOCKS,
                                    g_qwen.d_debug_arg_vals, g_qwen.d_debug_arg_ids, TQ_ARGMAX_BLOCKS);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "lm head/argmax launch failed: %s\n", cudaGetErrorString(err));
        return -5;
    }
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
    g_qwen.last_argmax_logit = h_val;
    return h_id;
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
    return 0;
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
