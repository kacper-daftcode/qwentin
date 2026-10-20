// the wide-column activation buffers and the projection / MLP helpers of the prefill waves, the decode step and the verify.
// Part of the forward_qwen.cu unity build: included in order, not a standalone header.

// Per-column activation buffers of the wide forward (prefill waves, the decode step, the verify), grown on demand.
static float *g_wide_h = NULL, *g_wide_q = NULL, *g_wide_k = NULL, *g_wide_v = NULL, *g_wide_core = NULL;
static float *g_wide_norm = NULL, *g_wide_resid = NULL, *g_wide_o = NULL, *g_wide_pn = NULL;
static float *g_wide_gate = NULL, *g_wide_up = NULL, *g_wide_mh = NULL, *g_wide_down = NULL;
static float *g_wide_qkv = NULL, *g_wide_z = NULL, *g_wide_b = NULL, *g_wide_a = NULL, *g_wide_lcore = NULL;
static float *g_wide_conv = NULL;      // conv outputs (chunk-parallel conv must not run in place)
static int   *g_wide_pos = NULL;
static int    g_wide_cap = 0;
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
// Wide MLP for a chunk of n tokens: post-rmsnorm(resid) -> gate/up (wide, shared B-frag)
// -> silu*mul -> down (wide) -> layer_out = resid + down. resid and layer_out are [n x H]
// (token-major); may NOT alias. Returns 0; -1 if mlp weights aren't e2m3-SF (caller falls back).
// With EXL3 weights the decode/verify steps fuse (a) the residual add after o_proj / out_proj with the post-attention
// norm, (b) SiLU * up into the down projection's fragment build, (c) the layer's final residual add with the NEXT
// layer's input norm (the final norm after the last layer). Same expressions, same reduction order -> bitwise identical
// to the separate kernels (k_tq_add_rmsnorm_b has k_tq_qwen_rmsnorm_b's reduction); -128 launches per step. The other
// weight formats take the separate kernels.
// add_a/add_b (optional): resid = add_a + add_b is computed here, fused with the post norm. next_ln/next_norm
// (optional): the final residual add also writes next_norm = rmsnorm(layer_out, next_ln).
static int wide_mlp(tq_layer_t *l, float *resid, float *layer_out, int n,
                    const float *add_a = NULL, const float *add_b = NULL,
                    const uint16_t *next_ln = NULL, float *next_norm = NULL) {
    int H = g_qwen.H, I = g_qwen.I;
    int exl3_ok = l->mlp_gate.exl3 && l->mlp_up.exl3 && l->mlp_down.exl3;
    if (!exl3_ok &&
        !(l->mlp_gate.e2m3 && l->mlp_up.e2m3 &&
          l->mlp_down.e2m3 &&
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
    if (exl3_ok) {
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
