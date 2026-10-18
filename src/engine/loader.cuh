// the residual-stream steering hook, the .tqf loader (parse_tqf), qwn_init / qwn_free and the model getters.
// Part of the forward_qwen.cu unity build: included in order, not a standalone header.

// ================== Residual-stream steering hook ==================
// One rank-2 affine map per hooked decoder layer on the residual stream (tools/residual_map.py loads a maps file and
// arms it; serve_openai.py --residual-map). Applied in fp32 on the engine's own residual buffers at the loop-top seam of
// every forward path (per-node spec verify, wide prefill, dense/graph decode): iteration L first applies layer L-1's
// map -- the post-layer seam -- plus one last call after the final layer.
//   y  = P^T (x - mu)                      (rank-2 latent; P is H x 2 row-major)
//   z  = (A - I)^T y                       (z_j = sum_c y_c * A_minus_I[j][c])
//   x += scale * (shift + P z)
#define TQ_RMAP_MAX_HOOKS 16
static struct {
    int layer;
    float scale, a00, a01, a10, a11;
    float *d_P, *d_mu, *d_ms;
} g_rmap_hooks[TQ_RMAP_MAX_HOOKS];
static int g_rmap_nhooks = 0;

// One block per residual row. Phase 1: block-reduce the two 1xH dot products.
// Phase 2: in-place rank-2 update. ~25k FLOPs/token/layer -- immeasurable.
__global__ void k_tq_rmap_hook2(float *x, int H,
                                const float *P, const float *mu, const float *shift,
                                float a00, float a01, float a10, float a11, float scale) {
    __shared__ float red[2 * 256];
    __shared__ float z0s, z1s;
    int tid = threadIdx.x;
    x += (size_t)blockIdx.x * H;
    float y0 = 0.0f, y1 = 0.0f;
    for (int i = tid; i < H; i += 256) {
        float c = x[i] - mu[i];
        y0 += c * P[2 * i];
        y1 += c * P[2 * i + 1];
    }
    red[tid] = y0;
    red[tid + 256] = y1;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            red[tid] += red[tid + s];
            red[tid + 256] += red[tid + 256 + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        z0s = a00 * red[0] + a01 * red[256];
        z1s = a10 * red[0] + a11 * red[256];
    }
    __syncthreads();
    float z0 = z0s, z1 = z1s;
    for (int i = tid; i < H; i += 256)
        x[i] += scale * (shift[i] + z0 * P[2 * i] + z1 * P[2 * i + 1]);
}

extern "C" int qwn_rmap_hook_add(int layer, float scale, const float *P, const float *A_minus_I,
                                 const float *mu, const float *shift) {
    if (!g_qwen.initialized) return -1;
    if (layer < 0 || layer >= g_qwen.L) return -2;
    if (g_rmap_nhooks >= TQ_RMAP_MAX_HOOKS) return -3;
    for (int s = 0; s < g_rmap_nhooks; s++)
        if (g_rmap_hooks[s].layer == layer) return -4;
    int sl = g_rmap_nhooks;
    size_t hsz = (size_t)g_qwen.H * sizeof(float);
    if (cudaMalloc(&g_rmap_hooks[sl].d_P, 2 * hsz) != cudaSuccess) return -5;
    if (cudaMalloc(&g_rmap_hooks[sl].d_mu, hsz) != cudaSuccess) return -6;
    if (cudaMalloc(&g_rmap_hooks[sl].d_ms, hsz) != cudaSuccess) return -7;
    cudaMemcpy(g_rmap_hooks[sl].d_P, P, 2 * hsz, cudaMemcpyHostToDevice);
    cudaMemcpy(g_rmap_hooks[sl].d_mu, mu, hsz, cudaMemcpyHostToDevice);
    cudaMemcpy(g_rmap_hooks[sl].d_ms, shift, hsz, cudaMemcpyHostToDevice);
    g_rmap_hooks[sl].layer = layer;
    g_rmap_hooks[sl].scale = scale;
    g_rmap_hooks[sl].a00 = A_minus_I[0];
    g_rmap_hooks[sl].a01 = A_minus_I[1];
    g_rmap_hooks[sl].a10 = A_minus_I[2];
    g_rmap_hooks[sl].a11 = A_minus_I[3];
    g_qwen.device_bytes += 4 * hsz;
    g_rmap_nhooks++;
    return 0;
}

extern "C" int qwn_rmap_hook_count(void) { return g_rmap_nhooks; }

// Retune without re-arming: scale rides as a kernel argument, so this is a
// host-only slot update taking effect at the next launch (schedules / sweeps).
extern "C" int qwn_rmap_set_scale(int layer, float scale) {
    for (int s = 0; s < g_rmap_nhooks; s++) {
        if (g_rmap_hooks[s].layer != layer) continue;
        g_rmap_hooks[s].scale = scale;
        return 0;
    }
    return -1;
}

// Fire the hook armed for `layer` on a [rows x H] fp32 residual buffer. Returns
// how many kernels launched (forward-path callers ignore it). No-op when
// unarmed -- zero cost in the default config.
static int rmap_hook_after_layer(int layer, float *x_rows, int rows) {
    int n = 0;
    if (layer < 0 || rows <= 0) return 0;
    for (int s = 0; s < g_rmap_nhooks; s++) {
        if (g_rmap_hooks[s].layer != layer) continue;
        k_tq_rmap_hook2<<<rows, 256, 0, g_qwen.stream>>>(
            x_rows, g_qwen.H, g_rmap_hooks[s].d_P, g_rmap_hooks[s].d_mu, g_rmap_hooks[s].d_ms,
            g_rmap_hooks[s].a00, g_rmap_hooks[s].a01, g_rmap_hooks[s].a10, g_rmap_hooks[s].a11,
            g_rmap_hooks[s].scale);
        n++;
    }
    return n;
}

static void rmap_hooks_free(void) {
    for (int s = 0; s < g_rmap_nhooks; s++) {
        cudaFree(g_rmap_hooks[s].d_P);
        cudaFree(g_rmap_hooks[s].d_mu);
        cudaFree(g_rmap_hooks[s].d_ms);
    }
    g_rmap_nhooks = 0;
}

// tools/residual_map_check.py driving surface: run the armed hook on host-supplied
// rows and read them back (kernel-vs-numpy fp32 parity gate).
extern "C" int qwn_rmap_apply_debug(int layer, int rows, float *io) {
    if (!g_qwen.initialized) return -1;
    if (rows < 1 || rows > 512) return -2;
    static float *d_buf = NULL;
    static size_t d_buf_floats = 0;
    size_t need = (size_t)rows * g_qwen.H;
    if (need > d_buf_floats) {
        if (d_buf) cudaFree(d_buf);
        if (cudaMalloc(&d_buf, need * sizeof(float)) != cudaSuccess) {
            d_buf = NULL;
            d_buf_floats = 0;
            return -3;
        }
        d_buf_floats = need;
    }
    size_t bytes = need * sizeof(float);
    if (cudaMemcpy(d_buf, io, bytes, cudaMemcpyHostToDevice) != cudaSuccess) return -4;
    if (rmap_hook_after_layer(layer, d_buf, rows) == 0) return -7;   // nothing armed here
    if (cudaDeviceSynchronize() != cudaSuccess) return -5;
    if (cudaMemcpy(io, d_buf, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) return -6;
    return 0;
}

// The 2026-08 names of the entry points, kept as aliases for one release (removed together with the server's
// --bark-all-day / --ot-* flag aliases).
extern "C" int qwn_ot_hook_add(int layer, float scale, const float *P, const float *A_k_minus_I,
                               const float *mu_H, const float *mean_shift) {
    return qwn_rmap_hook_add(layer, scale, P, A_k_minus_I, mu_H, mean_shift);
}
extern "C" int qwn_ot_hook_count(void) { return qwn_rmap_hook_count(); }
extern "C" int qwn_ot_set_scale(int layer, float scale) { return qwn_rmap_set_scale(layer, scale); }
extern "C" int qwn_ot_apply_debug(int layer, int rows, float *io) { return qwn_rmap_apply_debug(layer, rows, io); }

extern "C" void qwn_free(void) {
    rmap_hooks_free();
    destroy_forward_graph();
    destroy_decode_graph();
    destroy_spec_graph();
    if (g_qwen.stream) cudaStreamSynchronize(g_qwen.stream);
    if (g_qwen.d_debug_x) cudaFree(g_qwen.d_debug_x);
    if (g_qwen.d_debug_norm) cudaFree(g_qwen.d_debug_norm);
    if (g_qwen.d_debug_proj) cudaFree(g_qwen.d_debug_proj);
    if (g_qwen.d_debug_qmma_out) cudaFree(g_qwen.d_debug_qmma_out);
    if (g_qwen.d_debug_qmma_raw) cudaFree(g_qwen.d_debug_qmma_raw);
    if (g_qwen.d_debug_conv) cudaFree(g_qwen.d_debug_conv);
    if (g_qwen.d_debug_qkv) cudaFree(g_qwen.d_debug_qkv);
    if (g_qwen.d_debug_z) cudaFree(g_qwen.d_debug_z);
    if (g_qwen.d_debug_a_proj) cudaFree(g_qwen.d_debug_a_proj);
    if (g_qwen.d_debug_b_proj) cudaFree(g_qwen.d_debug_b_proj);
    if (g_qwen.d_debug_core) cudaFree(g_qwen.d_debug_core);
    if (g_qwen.d_debug_resid) cudaFree(g_qwen.d_debug_resid);
    if (g_qwen.d_debug_post_norm) cudaFree(g_qwen.d_debug_post_norm);
    if (g_qwen.d_debug_gate) cudaFree(g_qwen.d_debug_gate);
    if (g_qwen.d_debug_up) cudaFree(g_qwen.d_debug_up);
    if (g_qwen.d_debug_mlp_hidden) cudaFree(g_qwen.d_debug_mlp_hidden);
    if (g_qwen.d_debug_layer_out) cudaFree(g_qwen.d_debug_layer_out);
    if (g_qwen.d_debug_logits) cudaFree(g_qwen.d_debug_logits);
    if (g_qwen.d_debug_arg_vals) cudaFree(g_qwen.d_debug_arg_vals);
    if (g_qwen.d_debug_arg_ids) cudaFree(g_qwen.d_debug_arg_ids);
    if (g_qwen.d_optimistic_topk_vals) cudaFree(g_qwen.d_optimistic_topk_vals);
    if (g_qwen.d_optimistic_topk_ids) cudaFree(g_qwen.d_optimistic_topk_ids);
    if (g_qwen.d_optimistic_margin) cudaFree(g_qwen.d_optimistic_margin);
    if (g_qwen.d_debug_scale) cudaFree(g_qwen.d_debug_scale);
    if (g_qwen.d_debug_act_scale_inv) cudaFree(g_qwen.d_debug_act_scale_inv);
    if (g_qwen.d_sparse_b_hi) cudaFree(g_qwen.d_sparse_b_hi);
    if (g_qwen.d_sparse_b_lo) cudaFree(g_qwen.d_sparse_b_lo);
    if (g_qwen.d_sparse_ascale) cudaFree(g_qwen.d_sparse_ascale);
    if (g_qwen.d_sparse_out_lo) cudaFree(g_qwen.d_sparse_out_lo);
    if (g_qwen.d_debug_b) cudaFree(g_qwen.d_debug_b);
    if (g_qwen.d_decode_token) cudaFree(g_qwen.d_decode_token);
    if (g_qwen.d_decode_pos) cudaFree(g_qwen.d_decode_pos);
    if (g_qwen.d_decode_step) cudaFree(g_qwen.d_decode_step);
    if (g_qwen.d_decode_generated) cudaFree(g_qwen.d_decode_generated);
    if (g_qwen.d_qmma_work_counter) cudaFree(g_qwen.d_qmma_work_counter);
    if (g_qwen.h_embed) cudaFreeHost(g_qwen.h_embed);
    else if (g_qwen.d_embed) cudaFree(g_qwen.d_embed);
    if (g_qwen.d_embed8) cudaFree(g_qwen.d_embed8);
    if (g_qwen.d_embed_scale) cudaFree(g_qwen.d_embed_scale);
    g_qwen.d_embed = nullptr; g_qwen.h_embed = nullptr; g_qwen.d_embed8 = nullptr; g_qwen.d_embed_scale = nullptr;
    g_qwen.embed_fp8 = 0;
    if (g_qwen.d_norm) cudaFree(g_qwen.d_norm);
    free_qmma(&g_qwen.lm_head);
    free_qmma(&g_qwen.lm_head_fp4);
    g_qwen.lm_head_fp4_ready = 0;
    // MTP head buffers.
    {
        tq_layer_t *m = &g_qwen.mtp_layer;
        if (m->d_input_ln) cudaFree(m->d_input_ln);
        if (m->d_post_ln) cudaFree(m->d_post_ln);
        if (m->d_q_norm) cudaFree(m->d_q_norm);
        if (m->d_k_norm) cudaFree(m->d_k_norm);
        if (m->d_k_cache) cudaFree(m->d_k_cache);
        if (m->d_v_cache) cudaFree(m->d_v_cache);
        if (m->d_k_cache8) cudaFree(m->d_k_cache8);
        if (m->d_v_cache8) cudaFree(m->d_v_cache8);
        if (m->d_k_scale) cudaFree(m->d_k_scale);
        if (m->d_v_scale) cudaFree(m->d_v_scale);
        if (m->d_k_cache4) cudaFree(m->d_k_cache4);
        if (m->d_k_q4s) cudaFree(m->d_k_q4s);
        if (g_qwen.mtp_fc.d_w) cudaFree(g_qwen.mtp_fc.d_w);
        if (g_qwen.mtp_q_proj.d_w) cudaFree(g_qwen.mtp_q_proj.d_w);
        if (g_qwen.mtp_k_proj.d_w) cudaFree(g_qwen.mtp_k_proj.d_w);
        if (g_qwen.mtp_v_proj.d_w) cudaFree(g_qwen.mtp_v_proj.d_w);
        if (g_qwen.mtp_o_proj.d_w) cudaFree(g_qwen.mtp_o_proj.d_w);
        if (g_qwen.mtp_gate.d_w) cudaFree(g_qwen.mtp_gate.d_w);
        if (g_qwen.mtp_up.d_w) cudaFree(g_qwen.mtp_up.d_w);
        if (g_qwen.mtp_down.d_w) cudaFree(g_qwen.mtp_down.d_w);
        if (g_qwen.mtp_exl3) {                     // EXL3 MTP projections (paged drafter)
            free_qmma(&g_qwen.mtp_fc_x);
            free_qmma(&m->q_proj); free_qmma(&m->k_proj); free_qmma(&m->v_proj); free_qmma(&m->o_proj);
            free_qmma(&m->mlp_gate); free_qmma(&m->mlp_up); free_qmma(&m->mlp_down);
            g_qwen.mtp_exl3 = 0;
        }
        if (g_qwen.d_mtp_pre_fc_norm_emb) cudaFree(g_qwen.d_mtp_pre_fc_norm_emb);
        if (g_qwen.d_mtp_pre_fc_norm_hidden) cudaFree(g_qwen.d_mtp_pre_fc_norm_hidden);
        if (g_qwen.d_mtp_norm) cudaFree(g_qwen.d_mtp_norm);
        if (g_qwen.d_mtp_e) cudaFree(g_qwen.d_mtp_e);
        if (g_qwen.d_mtp_fc_in) cudaFree(g_qwen.d_mtp_fc_in);
        if (g_qwen.d_mtp_x) cudaFree(g_qwen.d_mtp_x);
        if (g_qwen.d_mtp_normout) cudaFree(g_qwen.d_mtp_normout);
        if (g_qwen.d_mtp_hidden_in) cudaFree(g_qwen.d_mtp_hidden_in);
        if (g_qwen.d_mtp_logits) cudaFree(g_qwen.d_mtp_logits);
        if (g_qwen.d_mtp_topk_vals) cudaFree(g_qwen.d_mtp_topk_vals);
        if (g_qwen.d_mtp_topk_ids) cudaFree(g_qwen.d_mtp_topk_ids);
        if (g_qwen.d_mtp_root_hidden) cudaFree(g_qwen.d_mtp_root_hidden);
        for (int di = 0; di < TQ_MTP_MAX_DEPTH; di++) {
            if (g_qwen.d_mtp_post[di]) cudaFree(g_qwen.d_mtp_post[di]);
        }
        for (int di = 0; di < TQ_DELTA_SNAP_SLOTS; di++) {
            if (g_qwen.d_delta_snap[di]) cudaFree(g_qwen.d_delta_snap[di]);
        }
        float *spec_bufs[] = {g_qwen.d_spec_h, g_qwen.d_spec_norm, g_qwen.d_spec_resid,
            g_qwen.d_spec_post, g_qwen.d_spec_lo, g_qwen.d_spec_qkv, g_qwen.d_spec_kbuf,
            g_qwen.d_spec_z, g_qwen.d_spec_bp, g_qwen.d_spec_ap, g_qwen.d_spec_core,
            g_qwen.d_spec_gate, g_qwen.d_spec_up, g_qwen.d_spec_mlp, g_qwen.d_spec_logits,
            g_qwen.d_spec_bscale, g_qwen.d_spec_part, g_qwen.d_spec_recur, g_qwen.d_spec_conv,
            g_qwen.d_spec_argv, g_qwen.d_spec_nk_factors};
        for (size_t si = 0; si < sizeof(spec_bufs) / sizeof(spec_bufs[0]); si++)
            if (spec_bufs[si]) cudaFree(spec_bufs[si]);
        if (g_qwen.d_spec_b) cudaFree(g_qwen.d_spec_b);
        if (g_qwen.d_spec_argi) cudaFree(g_qwen.d_spec_argi);
        if (g_qwen.d_spec_group_ids) cudaFree(g_qwen.d_spec_group_ids);
        if (g_qwen.d_spec_segs) cudaFree(g_qwen.d_spec_segs);
        if (g_qwen.d_spec_dnprep) cudaFree(g_qwen.d_spec_dnprep);
        if (g_qwen.d_live_recur_ptrs) cudaFree(g_qwen.d_live_recur_ptrs);
        if (g_qwen.d_spec_replay) cudaFree(g_qwen.d_spec_replay);
        if (g_qwen.d_spec_parent) cudaFree(g_qwen.d_spec_parent);
        if (g_qwen.d_spec_depth_offsets) cudaFree(g_qwen.d_spec_depth_offsets);
        if (g_qwen.d_spec_depth_done) cudaFree(g_qwen.d_spec_depth_done);
        if (g_qwen.d_spec_anc) cudaFree(g_qwen.d_spec_anc);
        if (g_qwen.d_spec_tokens) cudaFree(g_qwen.d_spec_tokens);
        if (g_qwen.d_spec_positions) cudaFree(g_qwen.d_spec_positions);
        if (g_spec_persist_scratch) { cudaFree(g_spec_persist_scratch); g_spec_persist_scratch = NULL; }
        if (g_spec_commit_rows) { cudaFree(g_spec_commit_rows); g_spec_commit_rows = NULL; }
    }
    for (int i = 0; i < g_qwen.L && i < TQ_MAX_LAYERS; i++) {
        tq_layer_t *l = &g_qwen.layers[i];
        if (l->d_input_ln) cudaFree(l->d_input_ln);
        if (l->d_post_ln) cudaFree(l->d_post_ln);
        free_qmma(&l->mlp_gate);
        free_qmma(&l->mlp_up);
        free_qmma(&l->mlp_down);
        if (l->d_q_norm) cudaFree(l->d_q_norm);
        if (l->d_k_norm) cudaFree(l->d_k_norm);
        free_qmma(&l->q_proj);
        free_qmma(&l->k_proj);
        free_qmma(&l->v_proj);
        free_qmma(&l->o_proj);
        if (l->d_linear_A_log) cudaFree(l->d_linear_A_log);
        if (l->d_linear_dt_bias) cudaFree(l->d_linear_dt_bias);
        if (l->d_linear_norm) cudaFree(l->d_linear_norm);
        if (l->d_linear_conv1d) cudaFree(l->d_linear_conv1d);
        if (l->d_linear_conv_state_owned) cudaFree(l->d_linear_conv_state_owned);
        if (l->d_linear_recurrent_state_owned) cudaFree(l->d_linear_recurrent_state_owned);
        if (l->d_k_cache) cudaFree(l->d_k_cache);
        if (l->d_v_cache) cudaFree(l->d_v_cache);
        if (l->d_k_cache8) cudaFree(l->d_k_cache8);
        if (l->d_v_cache8) cudaFree(l->d_v_cache8);
        if (l->d_k_scale) cudaFree(l->d_k_scale);
        if (l->d_v_scale) cudaFree(l->d_v_scale);
        if (l->d_k_cache4) cudaFree(l->d_k_cache4);
        if (l->d_k_q4s) cudaFree(l->d_k_q4s);
        free_qmma(&l->linear_in_a);
        free_qmma(&l->linear_in_b);
        free_qmma(&l->linear_in_qkv);
        free_qmma(&l->linear_in_z);
        free_qmma(&l->linear_out);
    }
    if (g_qwen.stream) cudaStreamDestroy(g_qwen.stream);
    memset(&g_qwen, 0, sizeof(g_qwen));
}

static int parse_tqf(const char *path, int upload) {
    if (upload) qwn_free();
    else memset(&g_qwen, 0, sizeof(g_qwen));
    if (upload) {
        cudaError_t serr = cudaStreamCreateWithFlags(&g_qwen.stream, cudaStreamNonBlocking);
        if (serr != cudaSuccess) {
            fprintf(stderr, "cudaStreamCreateWithFlags failed: %s\n", cudaGetErrorString(serr));
            return -1;
        }
    }

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open %s\n", path);
        return -1;
    }

    char magic[4];
    if (read_exact(f, magic, 4, "magic") != 0) { fclose(f); return -1; }
    if (memcmp(magic, "TQF1", 4) != 0) {
        fprintf(stderr, "Bad TQF magic\n");
        fclose(f);
        return -1;
    }

    uint32_t header_bytes = 0;
    if (read_exact(f, &header_bytes, 4, "header_bytes") != 0) { fclose(f); return -1; }
    uint32_t u[16];
    float fl[3];
    uint32_t tail[8];
    if (read_exact(f, u, sizeof(u), "fixed uint header") != 0) { fclose(f); return -1; }
    if (read_exact(f, fl, sizeof(fl), "fixed float header") != 0) { fclose(f); return -1; }
    if (read_exact(f, tail, sizeof(tail), "fixed tail header") != 0) { fclose(f); return -1; }

    g_qwen.flags = u[0];
    g_qwen.model_family = u[1];
    g_qwen.non_quant_dtype = u[2];
    g_qwen.H = (int)u[3];
    g_qwen.I = (int)u[4];
    g_qwen.L = (int)u[5];
    g_qwen.V = (int)u[6];
    g_qwen.nh = (int)u[7];
    g_qwen.nkv = (int)u[8];
    g_qwen.hd = (int)u[9];
    g_qwen.linear_num_key_heads = (int)u[10];
    g_qwen.linear_key_head_dim = (int)u[11];
    g_qwen.linear_num_value_heads = (int)u[12];
    g_qwen.linear_value_head_dim = (int)u[13];
    g_qwen.linear_conv_kernel_dim = (int)u[14];
    g_qwen.max_position_embeddings = (int)u[15];
    g_qwen.eps = fl[0];
    g_qwen.rope_theta = fl[1];
    g_qwen.partial_rotary_factor = fl[2];
    g_qwen.mrope_interleaved = (int)tail[0];
    g_qwen.mrope_section[0] = (int)tail[1];
    g_qwen.mrope_section[1] = (int)tail[2];
    g_qwen.mrope_section[2] = (int)tail[3];
    g_qwen.tie_word_embeddings = (int)tail[4];
    g_qwen.text_only = (int)tail[5];
    g_qwen.has_vision_config = (int)tail[6];
    g_qwen.has_mtp = (int)tail[7];
    g_qwen.has_mtp_section = (g_qwen.flags & TQ_FLAG_HAS_MTP) != 0;

    if (g_qwen.L <= 0 || g_qwen.L > TQ_MAX_LAYERS) {
        fprintf(stderr, "Unsupported layer count: %d\n", g_qwen.L);
        fclose(f);
        return -1;
    }
    uint32_t expected_header = 4u + 4u + 16u * 4u + 3u * 4u + 8u * 4u + (uint32_t)g_qwen.L;
    if (header_bytes != expected_header) {
        fprintf(stderr, "Unexpected header_bytes=%u expected=%u\n", header_bytes, expected_header);
        fclose(f);
        return -1;
    }
    if (read_exact(f, g_qwen.layer_types, (size_t)g_qwen.L, "layer types") != 0) {
        fclose(f);
        return -1;
    }

    if (upload) {
        int e2m3_flag = (g_qwen.flags & TQ_FLAG_BLOCK_SCALED_E2M3) != 0;
        cudaError_t cerr = cudaMemcpyToSymbol(gc_weight_e2m3, &e2m3_flag, sizeof(int));
        if (cerr != cudaSuccess) {
            fprintf(stderr, "cudaMemcpyToSymbol(gc_weight_e2m3) failed: %s\n", cudaGetErrorString(cerr));
            fclose(f);
            return -1;
        }
    }

    printf("TQF1: H=%d I=%d L=%d V=%d nh=%d nkv=%d hd=%d dtype=%u flags=0x%x\n",
           g_qwen.H, g_qwen.I, g_qwen.L, g_qwen.V, g_qwen.nh, g_qwen.nkv, g_qwen.hd,
           g_qwen.non_quant_dtype, g_qwen.flags);

    int lin_key_dim = g_qwen.linear_num_key_heads * g_qwen.linear_key_head_dim;
    int lin_value_dim = g_qwen.linear_num_value_heads * g_qwen.linear_value_head_dim;
    int lin_conv_dim = lin_key_dim * 2 + lin_value_dim;

    const char *eh = getenv("TQ_EMBED_HOST");
    if (upload && eh && eh[0] && eh[0] != '0' && !g_qwen.tie_word_embeddings) {
        if (read_embed_host(f, (size_t)g_qwen.V * g_qwen.H) != 0) { fclose(f); return -1; }
    } else if (read_non_quant(f, &g_qwen.d_embed, (size_t)g_qwen.V * g_qwen.H, "embed", upload) != 0) { fclose(f); return -1; }
    if (read_non_quant(f, &g_qwen.d_norm, (size_t)g_qwen.H, "norm", upload) != 0) { fclose(f); return -1; }
    if (!g_qwen.tie_word_embeddings) {
        if (read_qmma(f, &g_qwen.lm_head, g_qwen.V, g_qwen.H, "lm_head", upload, 0) != 0) { fclose(f); return -1; }
    }

    // TQ_EMBED_FP8: quantize the bf16 embedding table at load (E4M3=1, E2M3-6bit=2)
    // and free the bf16 copy. Lookup-only -> requires tie_word_embeddings=FALSE.
    g_qwen.embed_fp8 = 0;
    g_qwen.d_embed8 = nullptr;
    g_qwen.d_embed_scale = nullptr;
    if (upload) {
        const char *ef = getenv("TQ_EMBED_FP8");
        int emode = 0;
        if (ef && ef[0] && ef[0] != '0')
            emode = (ef[0] == '2' || ef[0] == 'e' || ef[0] == 'E') ? 2 : 1;
        if (emode && g_qwen.tie_word_embeddings) {
            fprintf(stderr, "TQ_EMBED_FP8 ignored: tie_word_embeddings=TRUE (embed feeds lm_head)\n");
            emode = 0;
        }
        if (emode && g_qwen.h_embed) {
            printf("TQ_EMBED_FP8 ignored: TQ_EMBED_HOST keeps the bf16 table in host memory\n");
            emode = 0;
        }
        if (emode && g_qwen.d_embed) {
            size_t rows = (size_t)g_qwen.V, Hh = (size_t)g_qwen.H;
            size_t qbytes = (emode == 2) ? rows * ((Hh * 6) / 8) : rows * Hh;
            uint8_t *d_q = nullptr; float *d_sc = nullptr;
            cudaDeviceSynchronize();   // embed upload must be visible to the quant kernel
            if (cudaMalloc(&d_q, qbytes) != cudaSuccess ||
                cudaMalloc(&d_sc, rows * sizeof(float)) != cudaSuccess) {
                fprintf(stderr, "TQ_EMBED_FP8 alloc failed -- staying bf16\n");
                if (d_q) cudaFree(d_q);
            } else {
                if (emode == 2)
                    k_tq_embed_quant_e2m3<<<(unsigned)rows, 256>>>(g_qwen.d_embed, d_q, d_sc, g_qwen.H);
                else
                    k_tq_embed_quant_e4m3<<<(unsigned)rows, 256>>>(g_qwen.d_embed, d_q, d_sc, g_qwen.H);
                cudaError_t qe = cudaDeviceSynchronize();
                if (qe != cudaSuccess) {
                    fprintf(stderr, "TQ_EMBED_FP8 quant kernel failed: %s -- staying bf16\n",
                            cudaGetErrorString(qe));
                    cudaFree(d_q); cudaFree(d_sc);
                } else {
                    cudaFree(g_qwen.d_embed);
                    g_qwen.d_embed = nullptr;
                    g_qwen.d_embed8 = d_q;
                    g_qwen.d_embed_scale = d_sc;
                    g_qwen.embed_fp8 = emode;
                    g_qwen.device_bytes += qbytes + rows * sizeof(float);
                    g_qwen.device_bytes -= rows * Hh * sizeof(uint16_t);   // the freed bf16 upload
                    printf("TQ_EMBED_FP8=%d: embed table -> %s, %.0f MiB (+%.0f MiB scale) "
                           "vs %.0f MiB bf16\n", emode,
                           emode == 2 ? "E2M3-6bit per-row" : "E4M3 per-row",
                           qbytes / 1048576.0, rows * sizeof(float) / 1048576.0,
                           rows * Hh * 2 / 1048576.0);
                }
            }
        }
        cudaMemcpyToSymbol(gc_embed_mode, &g_qwen.embed_fp8, sizeof(int));
        cudaMemcpyToSymbol(gc_embed8, &g_qwen.d_embed8, sizeof(void *));
        cudaMemcpyToSymbol(gc_embed_scale, &g_qwen.d_embed_scale, sizeof(void *));
    }

    for (int i = 0; i < g_qwen.L; i++) {
        tq_layer_t *l = &g_qwen.layers[i];
        char what[128];
        // mlp.* sparse records exist only on linear-attention layers (the
        // calibrated FP4 set); full-attention layers' mlp stays dense.
        int lin_layer = (g_qwen.layer_types[i] == TQ_LAYER_LINEAR_ATTENTION);
        snprintf(what, sizeof(what), "layer%d.input_ln", i);
        if (read_non_quant(f, &l->d_input_ln, g_qwen.H, what, upload) != 0) { fclose(f); return -1; }
        snprintf(what, sizeof(what), "layer%d.post_ln", i);
        if (read_non_quant(f, &l->d_post_ln, g_qwen.H, what, upload) != 0) { fclose(f); return -1; }
        snprintf(what, sizeof(what), "layer%d.mlp_gate", i);
        if (read_qmma(f, &l->mlp_gate, g_qwen.I, g_qwen.H, what, upload, lin_layer) != 0) { fclose(f); return -1; }
        snprintf(what, sizeof(what), "layer%d.mlp_up", i);
        if (read_qmma(f, &l->mlp_up, g_qwen.I, g_qwen.H, what, upload, lin_layer) != 0) { fclose(f); return -1; }
        snprintf(what, sizeof(what), "layer%d.mlp_down", i);
        if (read_qmma(f, &l->mlp_down, g_qwen.H, g_qwen.I, what, upload, lin_layer) != 0) { fclose(f); return -1; }

        if (g_qwen.layer_types[i] == TQ_LAYER_FULL_ATTENTION) {
            snprintf(what, sizeof(what), "layer%d.q_norm", i);
            if (read_non_quant(f, &l->d_q_norm, g_qwen.hd, what, upload) != 0) { fclose(f); return -1; }
            snprintf(what, sizeof(what), "layer%d.k_norm", i);
            if (read_non_quant(f, &l->d_k_norm, g_qwen.hd, what, upload) != 0) { fclose(f); return -1; }
            snprintf(what, sizeof(what), "layer%d.q_proj", i);
            if (read_qmma(f, &l->q_proj, g_qwen.nh * g_qwen.hd * 2, g_qwen.H, what, upload, 0) != 0) { fclose(f); return -1; }
            snprintf(what, sizeof(what), "layer%d.k_proj", i);
            if (read_qmma(f, &l->k_proj, g_qwen.nkv * g_qwen.hd, g_qwen.H, what, upload, 0) != 0) { fclose(f); return -1; }
            snprintf(what, sizeof(what), "layer%d.v_proj", i);
            if (read_qmma(f, &l->v_proj, g_qwen.nkv * g_qwen.hd, g_qwen.H, what, upload, 0) != 0) { fclose(f); return -1; }
            snprintf(what, sizeof(what), "layer%d.o_proj", i);
            if (read_qmma(f, &l->o_proj, g_qwen.H, g_qwen.nh * g_qwen.hd, what, upload, 0) != 0) { fclose(f); return -1; }
        } else if (g_qwen.layer_types[i] == TQ_LAYER_LINEAR_ATTENTION) {
            snprintf(what, sizeof(what), "layer%d.linear_A_log", i);
            if (read_f32(f, &l->d_linear_A_log, g_qwen.linear_num_value_heads, what, upload) != 0) { fclose(f); return -1; }
            snprintf(what, sizeof(what), "layer%d.linear_dt_bias", i);
            if (read_non_quant(f, &l->d_linear_dt_bias, g_qwen.linear_num_value_heads, what, upload) != 0) { fclose(f); return -1; }
            snprintf(what, sizeof(what), "layer%d.linear_norm", i);
            if (read_f32(f, &l->d_linear_norm, g_qwen.linear_value_head_dim, what, upload) != 0) { fclose(f); return -1; }
            snprintf(what, sizeof(what), "layer%d.linear_conv1d", i);
            if (read_non_quant(f, &l->d_linear_conv1d, (size_t)lin_conv_dim * g_qwen.linear_conv_kernel_dim, what, upload) != 0) { fclose(f); return -1; }
            snprintf(what, sizeof(what), "layer%d.linear_in_a", i);
            if (read_qmma(f, &l->linear_in_a, g_qwen.linear_num_value_heads, g_qwen.H, what, upload, 0) != 0) { fclose(f); return -1; }
            snprintf(what, sizeof(what), "layer%d.linear_in_b", i);
            if (read_qmma(f, &l->linear_in_b, g_qwen.linear_num_value_heads, g_qwen.H, what, upload, 0) != 0) { fclose(f); return -1; }
            snprintf(what, sizeof(what), "layer%d.linear_in_qkv", i);
            if (read_qmma(f, &l->linear_in_qkv, lin_conv_dim, g_qwen.H, what, upload, 1) != 0) { fclose(f); return -1; }
            snprintf(what, sizeof(what), "layer%d.linear_in_z", i);
            if (read_qmma(f, &l->linear_in_z, lin_value_dim, g_qwen.H, what, upload, 1) != 0) { fclose(f); return -1; }
            snprintf(what, sizeof(what), "layer%d.linear_out", i);
            if (read_qmma(f, &l->linear_out, g_qwen.H, lin_value_dim, what, upload, 1) != 0) { fclose(f); return -1; }
        } else {
            fprintf(stderr, "Unsupported layer type byte at layer %d: %u\n", i, g_qwen.layer_types[i]);
            fclose(f);
            return -1;
        }
        if (i % 6 == 0) printf("  layer %d/%d loaded\n", i, g_qwen.L);
    }

    // Native MTP head section (appended after the main payload when the converter
    // ran with TQ_EMIT_MTP=1). Read order MUST match mtp_tensor_specs() in
    // tools/convert_qwen_tqf.py. The MTP layer is one full_attention decoder layer.
    // TQ_NO_MTP=1 skips it (servers without speculative decoding, e.g. serve_batched.py):
    // the bytes are read past and the engine behaves as for a TQF without the section.
    const char *nm = getenv("TQ_NO_MTP");
    int no_mtp = upload && nm && nm[0] && nm[0] != '0';
    if (g_qwen.has_mtp_section) {
        int mup = upload && !no_mtp;
        tq_layer_t *m = &g_qwen.mtp_layer;
        int q_m = g_qwen.nh * g_qwen.hd * 2;
        int kv_m = g_qwen.nkv * g_qwen.hd;
        int attn_m = g_qwen.nh * g_qwen.hd;
        // All 8 MTP projections are FULL-PRECISION BF16 (non_quant), or EXL3 records with
        // TQ_FLAG_MTP_EXL3 (--mtp-exl3). Order MUST match mtp_tensor_specs() in tools/convert_qwen_tqf.py.
        int mtp_exl3 = (g_qwen.flags & TQ_FLAG_MTP_EXL3) != 0;
        #define TQ_READ_MTP_BF16(W, MM, KK, NAME)                                              \
            do { (W).M = (MM); (W).K = (KK);                                                   \
                 if (read_non_quant(f, &(W).d_w, (size_t)(MM) * (KK), NAME, mup) != 0) {        \
                     fclose(f); return -1; } } while (0)
        #define TQ_READ_MTP_PROJ(W, XW, MM, KK, NAME)                                          \
            do { if (mtp_exl3) { if (read_exl3(f, &(XW), (MM), (KK), NAME, mup) != 0) { fclose(f); return -1; } } \
                 else TQ_READ_MTP_BF16(W, MM, KK, NAME); } while (0)
        if (read_non_quant(f, &g_qwen.d_mtp_pre_fc_norm_emb, g_qwen.H, "mtp.pre_fc_norm_emb", mup) != 0) { fclose(f); return -1; }
        if (read_non_quant(f, &g_qwen.d_mtp_pre_fc_norm_hidden, g_qwen.H, "mtp.pre_fc_norm_hidden", mup) != 0) { fclose(f); return -1; }
        TQ_READ_MTP_PROJ(g_qwen.mtp_fc, g_qwen.mtp_fc_x, g_qwen.H, 2 * g_qwen.H, "mtp.fc");
        if (read_non_quant(f, &m->d_input_ln, g_qwen.H, "mtp.input_ln", mup) != 0) { fclose(f); return -1; }
        if (read_non_quant(f, &m->d_post_ln, g_qwen.H, "mtp.post_ln", mup) != 0) { fclose(f); return -1; }
        if (read_non_quant(f, &m->d_q_norm, g_qwen.hd, "mtp.q_norm", mup) != 0) { fclose(f); return -1; }
        if (read_non_quant(f, &m->d_k_norm, g_qwen.hd, "mtp.k_norm", mup) != 0) { fclose(f); return -1; }
        TQ_READ_MTP_PROJ(g_qwen.mtp_q_proj, m->q_proj, q_m, g_qwen.H, "mtp.q_proj");
        TQ_READ_MTP_PROJ(g_qwen.mtp_k_proj, m->k_proj, kv_m, g_qwen.H, "mtp.k_proj");
        TQ_READ_MTP_PROJ(g_qwen.mtp_v_proj, m->v_proj, kv_m, g_qwen.H, "mtp.v_proj");
        TQ_READ_MTP_PROJ(g_qwen.mtp_o_proj, m->o_proj, g_qwen.H, attn_m, "mtp.o_proj");
        TQ_READ_MTP_PROJ(g_qwen.mtp_gate, m->mlp_gate, g_qwen.I, g_qwen.H, "mtp.mlp_gate");
        TQ_READ_MTP_PROJ(g_qwen.mtp_up, m->mlp_up, g_qwen.I, g_qwen.H, "mtp.mlp_up");
        TQ_READ_MTP_PROJ(g_qwen.mtp_down, m->mlp_down, g_qwen.H, g_qwen.I, "mtp.mlp_down");
        if (read_non_quant(f, &g_qwen.d_mtp_norm, g_qwen.H, "mtp.norm", mup) != 0) { fclose(f); return -1; }
        #undef TQ_READ_MTP_PROJ
        #undef TQ_READ_MTP_BF16
        g_qwen.mtp_exl3 = mup && mtp_exl3;
        if (mup) printf(mtp_exl3 ? "  MTP head section loaded (EXL3 projections: paged drafter only)\n"
                                 : "  MTP head section loaded (BF16 projections)\n");
        else if (no_mtp) {
            g_qwen.has_mtp_section = 0;
            printf("  MTP head section skipped (TQ_NO_MTP)\n");
        }
        // Calibrated-head side-load (TQ_MTP_WEIGHTS=<dir>): overwrite the MTP
        // buffers from raw BF16 .bin files named by HF tensor (the output of
        // tools/mtp_calib_train.py). Happens before the lazy E2M3 repack, so the
        // quantized wave path automatically uses the calibrated weights.
        const char *mwd = getenv("TQ_MTP_WEIGHTS");
        if (mup && mwd && *mwd && !mtp_exl3) {
            int bad = 0;
            #define TQ_MTP_SIDELOAD(NAME, DST, COUNT)                                            \
                do {                                                                             \
                    char p_[1024];                                                               \
                    snprintf(p_, sizeof(p_), "%s/%s.bin", mwd, NAME);                            \
                    FILE *fp_ = fopen(p_, "rb");                                                 \
                    if (!fp_) { fprintf(stderr, "mtp sideload missing: %s\n", p_); bad = 1; }    \
                    else {                                                                       \
                        size_t cnt_ = (size_t)(COUNT);                                           \
                        fseek(fp_, 0, SEEK_END);                                                 \
                        long sz_ = ftell(fp_);                                                   \
                        fseek(fp_, 0, SEEK_SET);                                                 \
                        if ((size_t)sz_ != cnt_ * 2) {                                           \
                            fprintf(stderr, "mtp sideload size mismatch %s: %ld != %zu\n",       \
                                    p_, sz_, cnt_ * 2);                                          \
                            bad = 1;                                                             \
                        } else {                                                                 \
                            uint16_t *h_ = (uint16_t *)malloc(cnt_ * 2);                         \
                            if (fread(h_, 2, cnt_, fp_) != cnt_ ||                               \
                                cudaMemcpy(DST, h_, cnt_ * 2, cudaMemcpyHostToDevice)            \
                                    != cudaSuccess) bad = 1;                                     \
                            free(h_);                                                            \
                        }                                                                        \
                        fclose(fp_);                                                             \
                    }                                                                            \
                } while (0)
            TQ_MTP_SIDELOAD("mtp.pre_fc_norm_embedding.weight", g_qwen.d_mtp_pre_fc_norm_emb, g_qwen.H);
            TQ_MTP_SIDELOAD("mtp.pre_fc_norm_hidden.weight", g_qwen.d_mtp_pre_fc_norm_hidden, g_qwen.H);
            TQ_MTP_SIDELOAD("mtp.fc.weight", g_qwen.mtp_fc.d_w, (size_t)g_qwen.H * 2 * g_qwen.H);
            TQ_MTP_SIDELOAD("mtp.layers.0.input_layernorm.weight", m->d_input_ln, g_qwen.H);
            TQ_MTP_SIDELOAD("mtp.layers.0.post_attention_layernorm.weight", m->d_post_ln, g_qwen.H);
            TQ_MTP_SIDELOAD("mtp.layers.0.self_attn.q_norm.weight", m->d_q_norm, g_qwen.hd);
            TQ_MTP_SIDELOAD("mtp.layers.0.self_attn.k_norm.weight", m->d_k_norm, g_qwen.hd);
            TQ_MTP_SIDELOAD("mtp.layers.0.self_attn.q_proj.weight", g_qwen.mtp_q_proj.d_w, (size_t)q_m * g_qwen.H);
            TQ_MTP_SIDELOAD("mtp.layers.0.self_attn.k_proj.weight", g_qwen.mtp_k_proj.d_w, (size_t)kv_m * g_qwen.H);
            TQ_MTP_SIDELOAD("mtp.layers.0.self_attn.v_proj.weight", g_qwen.mtp_v_proj.d_w, (size_t)kv_m * g_qwen.H);
            TQ_MTP_SIDELOAD("mtp.layers.0.self_attn.o_proj.weight", g_qwen.mtp_o_proj.d_w, (size_t)g_qwen.H * attn_m);
            TQ_MTP_SIDELOAD("mtp.layers.0.mlp.gate_proj.weight", g_qwen.mtp_gate.d_w, (size_t)g_qwen.I * g_qwen.H);
            TQ_MTP_SIDELOAD("mtp.layers.0.mlp.up_proj.weight", g_qwen.mtp_up.d_w, (size_t)g_qwen.I * g_qwen.H);
            TQ_MTP_SIDELOAD("mtp.layers.0.mlp.down_proj.weight", g_qwen.mtp_down.d_w, (size_t)g_qwen.H * g_qwen.I);
            TQ_MTP_SIDELOAD("mtp.norm.weight", g_qwen.d_mtp_norm, g_qwen.H);
            #undef TQ_MTP_SIDELOAD
            if (bad) { fclose(f); return -97; }
            printf("  MTP head OVERRIDDEN from %s (calibrated)\n", mwd);
        }
    }

    long pos = ftell(f);
    long end = -1;
    if (pos >= 0 && fseek(f, 0, SEEK_END) == 0) {
        end = ftell(f);
    }
    fclose(f);
    if (pos >= 0 && end >= 0 && pos != end) {
        fprintf(stderr, "TQF payload mismatch: stopped at %ld of %ld\n", pos, end);
        return -1;
    }
    g_qwen.initialized = upload ? 1 : 0;
    printf("TQF1 %s complete: tensors non_quant=%d qm8=%d f32=%d device_bytes=%zu\n",
           upload ? "load" : "probe", g_qwen.non_quant_count, g_qwen.qmma_count,
           g_qwen.f32_count, g_qwen.device_bytes);
    return 0;
}

// Validation/utility ABI: quantize+pack one BF16 weight to dense E2M3 on the GPU
// and copy the dense d_A payload (Mt*Kt*384 bytes) and pow2 scale_inv back to host.
extern "C" int qwn_gpu_pack_e2m3_test(const uint16_t *h_bf16, int M, int K,
                                      uint8_t *h_out_dA, float *h_out_scale) {
    if ((M % 16) != 0 || (K % 32) != 0) return -1;
    int Mt = M / 16, Kt = K / 32;
    int sr = (M + 127) / 128, sc = (K + 127) / 128;
    size_t bf_bytes = (size_t)M * K * sizeof(uint16_t);
    size_t da_bytes = (size_t)Mt * Kt * 384u;
    size_t sc_bytes = (size_t)sr * sc * sizeof(float);
    uint16_t *d_bf16 = NULL;
    uint32_t *d_dA = NULL;
    float *d_scale = NULL;
    if (cudaMalloc(&d_bf16, bf_bytes) != cudaSuccess) return -2;
    if (cudaMalloc(&d_dA, da_bytes) != cudaSuccess) { cudaFree(d_bf16); return -2; }
    if (cudaMalloc(&d_scale, sc_bytes) != cudaSuccess) { cudaFree(d_bf16); cudaFree(d_dA); return -2; }
    cudaMemcpy(d_bf16, h_bf16, bf_bytes, cudaMemcpyHostToDevice);
    dim3 g1(sr, sc);
    k_tq_block_absmax_pow2_e2m3<<<g1, 256>>>(d_bf16, M, K, sc, d_scale);
    long total = (long)Mt * Kt * 96;
    int threads = 256;
    long blocks = (total + threads - 1) / threads;
    k_tq_pack_bf16_to_e2m3_dense<<<(unsigned)blocks, threads>>>(d_bf16, M, K, sc, d_scale, d_dA, Kt);
    cudaError_t err = cudaDeviceSynchronize();
    if (err == cudaSuccess) {
        cudaMemcpy(h_out_dA, d_dA, da_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_out_scale, d_scale, sc_bytes, cudaMemcpyDeviceToHost);
    }
    cudaFree(d_bf16);
    cudaFree(d_dA);
    cudaFree(d_scale);
    return (err == cudaSuccess) ? 0 : -3;
}

static int ensure_attn_scores(void) {
    if (g_qwen.d_attn_scores) return 0;
    if (g_qwen.max_seq <= 0) g_qwen.max_seq = 8192;
    size_t bytes = (size_t)TQ_ATTN_SCORE_SLOTS * g_qwen.max_seq * sizeof(float);
    if (cudaMalloc(&g_qwen.d_attn_scores, bytes) != cudaSuccess) {
        fprintf(stderr, "attn scores scratch alloc failed (%zu MB)\n", bytes >> 20);
        return -1;
    }
    return 0;
}

extern "C" int qwn_probe(const char *path) {
    return parse_tqf(path, 0);
}

// TQ_W_BYTE: convert selected dense E2M3 layer weights to HW-direct byte
// containers at load time (one code per byte -> zero-unpack GEMV loops).
// Classes: "mlp", "delta", "attn" (substring match); "1"/"all" = all three.
// lm_head/embed/MTP stay dense: their GEMVs are DRAM-bound, where 6-bit
// packing still wins. Bit-exact: same codes, same scales, same MMA order.
static void maybe_byte_convert_weight(tq_qmma_weight_t *w, const char *what) {
    if (!w->d_A || !w->e2m3 || w->e2m1 || w->e2m3_byte || w->row_major || w->word_major) return;
    long ntiles = (long)w->Mt * w->Kt;
    size_t nbytes = (size_t)ntiles * 512u;
    uint32_t *d_new = NULL;
    if (cudaMalloc(&d_new, nbytes) != cudaSuccess) {
        fprintf(stderr, "TQ_W_BYTE: OOM on %s (%zu MB), keeping dense\n", what, nbytes >> 20);
        return;
    }
    long total = ntiles * 32;
    k_tq_repack_e2m3_dense_to_byte<<<(unsigned)((total + 255) / 256), 256>>>(
        (const uint32_t *)w->d_A, d_new, ntiles);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "TQ_W_BYTE: repack failed on %s\n", what);
        cudaFree(d_new);
        return;
    }
    cudaFree(w->d_A);
    w->d_A = (uint8_t *)d_new;
    w->e2m3_byte = 1;
    g_qwen.device_bytes += (size_t)ntiles * 128u;   // 512 - 384 B/tile
}

static void maybe_byte_convert_all(void) {
    const char *env = getenv("TQ_W_BYTE");
    if (!env || !env[0] || !strcmp(env, "0")) return;
    int mlp = 0, delta = 0, attn = 0;
    if (!strcmp(env, "1") || strstr(env, "all")) mlp = delta = attn = 1;
    if (strstr(env, "mlp")) mlp = 1;
    if (strstr(env, "delta")) delta = 1;
    if (strstr(env, "attn")) attn = 1;
    size_t before = g_qwen.device_bytes;
    for (int i = 0; i < g_qwen.L; i++) {
        tq_layer_t *l = &g_qwen.layers[i];
        if (mlp) {
            maybe_byte_convert_weight(&l->mlp_gate, "mlp_gate");
            maybe_byte_convert_weight(&l->mlp_up, "mlp_up");
            maybe_byte_convert_weight(&l->mlp_down, "mlp_down");
        }
        if (delta) {
            maybe_byte_convert_weight(&l->linear_in_qkv, "linear_in_qkv");
            maybe_byte_convert_weight(&l->linear_in_z, "linear_in_z");
            maybe_byte_convert_weight(&l->linear_in_b, "linear_in_b");
            maybe_byte_convert_weight(&l->linear_in_a, "linear_in_a");
            maybe_byte_convert_weight(&l->linear_out, "linear_out");
        }
        if (attn) {
            maybe_byte_convert_weight(&l->q_proj, "q_proj");
            maybe_byte_convert_weight(&l->k_proj, "k_proj");
            maybe_byte_convert_weight(&l->v_proj, "v_proj");
            maybe_byte_convert_weight(&l->o_proj, "o_proj");
        }
    }
    fprintf(stderr, "TQ_W_BYTE=%s (mlp=%d delta=%d attn=%d): byte-container weights, +%zu MB device\n",
            env, mlp, delta, attn, (g_qwen.device_bytes - before) >> 20);
}

// TQ_W_E2M1: LOSSY load-time repack of selected dense E2M3 weights to 4-bit
// E2M1 (64 words/tile = HALF the dense weight bytes). Same class selectors as
// TQ_W_BYTE ("mlp"/"delta"/"attn"/"all"). Block scales are recomputed per block
// (absmax rescale, k_tq_block_absmax_e2m1_from_e2m3) -- this is the speed/
// quality PROBE rung of the ladder; GPTQ-compensated offline E2M1 is the
// quality follow-up. lm_head/embed/MTP stay dense E2M3.
static int g_e2m1_k32only = 0;   // TQ_W_E2M1=k32: convert exactly the weights
                                  // whose tqf scales are per-k32 (offline-marked)

static void maybe_e2m1_convert_weight(tq_qmma_weight_t *w, const char *what) {
    if (!w->d_A || !w->e2m3 || w->e2m1 || w->e2m3_byte || w->row_major || w->word_major) return;
    if (g_e2m1_k32only && w->scale_cols != w->Kt) return;
    if (w->has_sparse) { fprintf(stderr, "TQ_W_E2M1: %s has sparse payload, skipped\n", what); return; }
    int M = w->M, Kt = w->Kt;
    int sr = w->scale_rows, sc = w->scale_cols;
    int k32_scales = (sc == Kt);   // MX-style container: per-k32 scales already
                                   // E2M1-native -> pure recode, scales kept
    uint32_t *d_A = NULL;
    float *d_scale = NULL;
    size_t abytes = (size_t)w->Mt * Kt * 256u;
    if (cudaMalloc(&d_A, abytes) != cudaSuccess) { fprintf(stderr, "TQ_W_E2M1: OOM on %s\n", what); return; }
    if (k32_scales) {
        d_scale = w->d_block_scale_inv;
    } else {
        if (cudaMalloc(&d_scale, (size_t)sr * sc * sizeof(float)) != cudaSuccess) { cudaFree(d_A); return; }
        k_tq_block_absmax_e2m1_from_e2m3<<<dim3(sr, sc), 256>>>(
            (const uint32_t *)w->d_A, Kt, M, w->K, sc, w->d_block_scale_inv, d_scale);
    }
    long total = (long)w->Mt * Kt * 64;
    k_tq_repack_e2m3_to_e2m1<<<(unsigned)((total + 255) / 256), 256>>>(
        (const uint32_t *)w->d_A, M, w->K, sc, w->d_block_scale_inv, d_scale, d_A, Kt);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "TQ_W_E2M1: repack failed on %s\n", what);
        cudaFree(d_A);
        if (!k32_scales) cudaFree(d_scale);
        return;
    }
    cudaFree(w->d_A);
    if (!k32_scales) {
        cudaFree(w->d_block_scale_inv);
        w->d_block_scale_inv = d_scale;
    }
    w->d_A = (uint8_t *)d_A;
    w->e2m3 = 0;
    w->e2m1 = 1;
    g_qwen.device_bytes -= (size_t)w->Mt * Kt * 128u;   // 384 -> 256 B/tile
}

// TQ_LDSM (PROBE, default OFF). 1 once the MLP weights are in LDSM-natural order
// and the part-1 kernel should launch its ldmatrix.U6 instantiation.
static int g_use_ldsm = 0;

// Repack one dense E2M3 weight into LDSM-natural order (same 384 B/tile, same
// codes). Bit-identical MMA inputs (verified on hardware: test_ldsm_mma_map.cu).
static void maybe_ldsm_convert_weight(tq_qmma_weight_t *w, const char *what) {
    if (!w->d_A || !w->e2m3 || w->e2m1 || w->e2m3_byte || w->row_major || w->word_major) return;
    long ntiles = (long)w->Mt * w->Kt;
    if (ntiles <= 0) return;
    size_t words = (size_t)ntiles * 96u;
    uint32_t *d_new = NULL;
    if (cudaMalloc(&d_new, words * 4u) != cudaSuccess) {
        fprintf(stderr, "TQ_LDSM: OOM on %s (%zu MB), keeping dense\n", what, (words * 4u) >> 20);
        return;
    }
    k_tq_repack_e2m3_frag_to_ldsm<<<(unsigned)ntiles, 32>>>(
        (const uint32_t *)w->d_A, d_new, ntiles);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "TQ_LDSM: repack failed on %s\n", what);
        cudaFree(d_new);
        return;
    }
    cudaFree(w->d_A);
    w->d_A = (uint8_t *)d_new;   // same size: device_bytes unchanged
}

// TQ_LDSM: repack the MLP dense-E2M3 weights (the part-1 GEMV that dominates
// verify) so the persistent kernel can use the SM120 hardware FP6 unpack. The
// default path (and parts 0/2) are untouched -> bit-exact when TQ_LDSM=0.
static void maybe_ldsm_convert_all(void) {
    const char *env = getenv("TQ_LDSM");
    if (!env || !env[0] || !strcmp(env, "0")) return;
    for (int i = 0; i < g_qwen.L; i++) {
        tq_layer_t *l = &g_qwen.layers[i];
        maybe_ldsm_convert_weight(&l->mlp_gate, "mlp_gate");
        maybe_ldsm_convert_weight(&l->mlp_up, "mlp_up");
        maybe_ldsm_convert_weight(&l->mlp_down, "mlp_down");
    }
    g_use_ldsm = 1;
    int one = 1;
    cudaMemcpyToSymbol(gc_use_ldsm, &one, sizeof(int));
    fprintf(stderr, "TQ_LDSM=1 (PROBE): MLP dense-E2M3 repacked to LDSM-natural; "
                    "part-1 GEMV uses ldmatrix.U6 hardware unpack (default path untouched)\n");
}

static void maybe_e2m1_convert_all(void) {
    const char *env = getenv("TQ_W_E2M1");
    if (!env || !env[0] || !strcmp(env, "0")) return;
    int mlp = 0, delta = 0, attn = 0;
    if (!strcmp(env, "1") || strstr(env, "all")) mlp = delta = attn = 1;
    if (!strcmp(env, "k32")) {
        // tqf-driven rung: the offline pipeline marks E2M1-intended weights
        // with per-k32 scales; convert exactly those, leave the rest FP6.
        g_e2m1_k32only = 1;
        mlp = delta = attn = 1;
    }
    if (strstr(env, "mlp")) mlp = 1;
    {
        // "deltain" alone must NOT imply "delta" (substring trap): accept a
        // "delta" match only when not followed by "in".
        const char *dm = env;
        while ((dm = strstr(dm, "delta")) != NULL) {
            if (strncmp(dm, "deltain", 7) != 0) { delta = 1; break; }
            dm += 7;
        }
    }
    if (strstr(env, "attn")) attn = 1;
    // fine-grained MLP selectors for hetero rungs (e.g. "gate,up,delta,attn"
    // keeps down_proj FP6); "mlp"/"all" still mean all three.
    int gate = mlp || strstr(env, "gate") != NULL;
    int up = mlp || strstr(env, "up") != NULL;
    int down = mlp || strstr(env, "down") != NULL;
    size_t before = g_qwen.device_bytes;
    for (int i = 0; i < g_qwen.L; i++) {
        tq_layer_t *l = &g_qwen.layers[i];
        {
            if (gate) maybe_e2m1_convert_weight(&l->mlp_gate, "mlp_gate");
            if (up) maybe_e2m1_convert_weight(&l->mlp_up, "mlp_up");
            if (down) maybe_e2m1_convert_weight(&l->mlp_down, "mlp_down");
        }
        {
            // "delta" = all five; "deltain" = the four in_* only (out_proj
            // stays FP6 -- hetero rung; SNR ranks linear_out 2nd-worst kind)
            int din = delta || strstr(getenv("TQ_W_E2M1"), "deltain") != NULL;
            if (din) {
                maybe_e2m1_convert_weight(&l->linear_in_qkv, "linear_in_qkv");
                maybe_e2m1_convert_weight(&l->linear_in_z, "linear_in_z");
                maybe_e2m1_convert_weight(&l->linear_in_b, "linear_in_b");
                maybe_e2m1_convert_weight(&l->linear_in_a, "linear_in_a");
            }
            if (delta) maybe_e2m1_convert_weight(&l->linear_out, "linear_out");
        }
        if (attn) {
            maybe_e2m1_convert_weight(&l->q_proj, "q_proj");
            maybe_e2m1_convert_weight(&l->k_proj, "k_proj");
            maybe_e2m1_convert_weight(&l->v_proj, "v_proj");
            maybe_e2m1_convert_weight(&l->o_proj, "o_proj");
        }
    }
    fprintf(stderr, "TQ_W_E2M1=%s (mlp=%d delta=%d attn=%d): 4-bit E2M1 weights (LOSSY probe), -%zu MB device\n",
            env, mlp, delta, attn, (before - g_qwen.device_bytes) >> 20);
}

extern "C" int qwn_init(const char *path) {
    int r = parse_tqf(path, 1);
    if (r != 0) return r;
    // AFTER parse_tqf: it memsets g_qwen, so the runtime context limit must be
    // set here (and before any ensure path allocates KV/scratch).
    {
        const char *e = getenv("TQ_CTX");
        int v = e ? atoi(e) : 8192;
        if (v < 2048) v = 2048;
        if (v > TQ_MAX_SEQ) v = TQ_MAX_SEQ;
        g_qwen.max_seq = v;
        fprintf(stderr, "context limit: %d tokens (TQ_CTX, cap %d)\n", v, TQ_MAX_SEQ);
    }
    {
        const char *e = getenv("TQ_KV_FP8");
        g_qwen.kv_fp8 = (e && atoi(e) != 0) ? 1 : 0;
        const char *e4 = getenv("TQ_KV_Q4");
        g_qwen.kv_q4 = (e4 && atoi(e4) != 0) ? 1 : 0;
        if (g_qwen.kv_q4 && g_qwen.hd != 256) {
            fprintf(stderr, "TQ_KV_Q4=1 requires hd==256 (got %d) -- disabled\n", g_qwen.hd);
            g_qwen.kv_q4 = 0;
        }
        if (g_qwen.kv_q4)
            g_qwen.kv_fp8 = 1;   // V rows + scales ride the FP8 machinery
        const char *efs = getenv("TQ_KV_Q4_FP32S");
        g_qwen.kv_q4_fp32s = (efs && atoi(efs) != 0) ? 1 : 0;
        cudaMemcpyToSymbol(gc_kv_q4_fp32s, &g_qwen.kv_q4_fp32s, sizeof(int));
        if (g_qwen.kv_fp8 && g_qwen.hd != 256) {
            fprintf(stderr, "TQ_KV_FP8=1 requires hd==256 (got %d) -- disabled\n", g_qwen.hd);
            g_qwen.kv_fp8 = 0;
        }
        if (g_qwen.kv_q4)
            fprintf(stderr, "KV prefix cache: K int4 asym g32 + Hadamard rotation, "
                            "V E4M3 + pow2 scale (TQ_KV_Q4=1); (scale,zp) groups %s\n",
                    g_qwen.kv_q4_fp32s ? "fp32 (TQ_KV_Q4_FP32S)" : "fp16");
        else if (g_qwen.kv_fp8)
            fprintf(stderr, "KV prefix cache: FP8 E4M3 + pow2 scale per (pos, kv_head) (TQ_KV_FP8=1)\n");
    }
    {
        // TQ_ATTN_MMA: tensor-core attention + online softmax. Default ON since
        // Short-ctx neutral (111.7 vs 111.8 tok/s @1k), long-ctx
        // +52..87% (maraton @10k 57->87, @16k 45->84), quality gates A-F PASS.
        // TQ_ATTN_MMA=0 is the escape hatch. The prefix-cache format picks the
        // flavor: fp32 cache -> bf16 HMMA (1), E4M3 cache (TQ_KV_FP8) -> QMMA (2),
        // int4 K cache (TQ_KV_Q4) -> dequant + bf16 HMMA (3).
        const char *e = getenv("TQ_ATTN_MMA");
        int v = e ? atoi(e) : 1;
        g_qwen.attn_mma = 0;
        if (v != 0) {
            if (g_qwen.hd != 256) {
                fprintf(stderr, "TQ_ATTN_MMA requires hd==256 (got %d) -- disabled\n", g_qwen.hd);
            } else {
                g_qwen.attn_mma = g_qwen.kv_q4 ? 3 : (g_qwen.kv_fp8 ? 2 : 1);
                if (v != g_qwen.attn_mma)
                    fprintf(stderr, "TQ_ATTN_MMA=%d adjusted to %d (TQ_KV_FP8=%d/TQ_KV_Q4=%d pick the flavor)\n",
                            v, g_qwen.attn_mma, g_qwen.kv_fp8, g_qwen.kv_q4);
                fprintf(stderr, "attention: tensor-core MMA + online softmax (TQ_ATTN_MMA=%d)\n",
                        g_qwen.attn_mma);
            }
        }
    }
    maybe_byte_convert_all();
    maybe_e2m1_convert_all();
    maybe_ldsm_convert_all();
    {
        // Default ON: bit-exact by construction (deadlock-free
        // per-m-tile gating), measured +0.5-0.6% e2e on both 5090 and PRO 6000.
        // TQ_PERSIST_P2P=0 reverts to the plain bar6 grid barrier.
        const char *e = getenv("TQ_PERSIST_P2P");
        int p2p = (e && e[0]) ? (strcmp(e, "0") != 0 ? 1 : 0) : 1;
        cudaMemcpyToSymbol(gc_use_p2p, &p2p, sizeof(int));
        if (p2p)
            fprintf(stderr, "TQ_PERSIST_P2P=1: part-1 gemv2->reduce_final uses "
                            "per-m-tile point-to-point readiness (bar6 replaced; =0 reverts)\n");
    }
    return 0;
}

extern "C" int qwn_hidden_size(void) { return g_qwen.H; }
extern "C" int qwn_intermediate_size(void) { return g_qwen.I; }
extern "C" int qwn_vocab_size(void) { return g_qwen.V; }
extern "C" int qwn_num_layers(void) { return g_qwen.L; }
extern "C" int qwn_num_attention_heads(void) { return g_qwen.nh; }
extern "C" int qwn_num_key_value_heads(void) { return g_qwen.nkv; }
extern "C" int qwn_head_dim(void) { return g_qwen.hd; }
extern "C" int qwn_max_seq(void) { return g_qwen.max_seq; }
extern "C" int qwn_layer_type(int layer) {
    if (layer < 0 || layer >= g_qwen.L) return 0;
    return (int)g_qwen.layer_types[layer];
}
extern "C" size_t qwn_device_bytes(void) { return g_qwen.device_bytes; }
extern "C" int qwn_non_quant_count(void) { return g_qwen.non_quant_count; }
extern "C" int qwn_qmma_count(void) { return g_qwen.qmma_count; }
extern "C" int qwn_f32_count(void) { return g_qwen.f32_count; }
extern "C" int qwn_last_argmax_id(void) { return g_qwen.last_argmax_id; }
extern "C" float qwn_last_argmax_logit(void) { return g_qwen.last_argmax_logit; }

// Optimistic guard inspection (no-op unless TQ_OPTIMISTIC_TOPK is set).
extern "C" float qwn_last_topk_margin(void) {
    if (!g_qwen.d_optimistic_margin) return -1.0f;
    float m = -1.0f;
    cudaMemcpy(&m, g_qwen.d_optimistic_margin, sizeof(float), cudaMemcpyDeviceToHost);
    return m;
}
extern "C" int qwn_last_topk(int *ids_out, float *vals_out, int k) {
    int n = g_qwen.optimistic_topk_k;
    if (n <= 0 || !g_qwen.d_optimistic_topk_ids) return 0;
    if (k > n) k = n;
    if (k <= 0) return 0;
    if (ids_out) cudaMemcpy(ids_out, g_qwen.d_optimistic_topk_ids, (size_t)k * sizeof(int), cudaMemcpyDeviceToHost);
    if (vals_out) cudaMemcpy(vals_out, g_qwen.d_optimistic_topk_vals, (size_t)k * sizeof(float), cudaMemcpyDeviceToHost);
    return k;
}

// Foundation sanity for the N=k lookahead GEMV: feed nvar identical copies of the
// current hidden (d_debug_norm; requires a prior qwn_forward_first) into the
// nvar-column lm_head GEMV. All nvar output columns must be bit-identical -- this
// validates the B-fragment column packing + the k-column C-fragment extraction
// independently of any orchestration. Returns max |col_n - col_0| over the vocab.
extern "C" int qwn_nk_lmhead_sanity(int nvar, float *max_col_diff) {
    if (!g_qwen.initialized || !g_qwen.lm_head.block_scaled || !g_qwen.d_debug_norm) return -1;
    if (nvar < 1) nvar = 1;
    if (nvar > 8) nvar = 8;
    int H = g_qwen.H, V = g_qwen.V;
    int Kt = g_qwen.lm_head.Kt;
    int Mt = g_qwen.lm_head.Mt;
    int nblocks = (H + 127) / 128;

    float *d_x = NULL, *d_scale = NULL, *d_out = NULL;
    uint8_t *d_b = NULL;
    cudaMalloc(&d_x, (size_t)nvar * H * sizeof(float));
    cudaMalloc(&d_b, (size_t)Kt * 256);
    cudaMalloc(&d_scale, (size_t)nblocks * sizeof(float));
    cudaMalloc(&d_out, (size_t)nvar * V * sizeof(float));
    if (!d_x || !d_b || !d_scale || !d_out) {
        if (d_x) cudaFree(d_x);
        if (d_b) cudaFree(d_b);
        if (d_scale) cudaFree(d_scale);
        if (d_out) cudaFree(d_out);
        return -2;
    }
    for (int v = 0; v < nvar; v++) {
        cudaMemcpy(d_x + (size_t)v * H, g_qwen.d_debug_norm,
                   (size_t)H * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    k_tq_quantize_bfrag_pow2_fused_nk<<<nblocks, 128>>>(d_b, d_scale, d_x, H, nvar);
    k_tq_qmma_sf_gemv_tiled_nk<<<Mt, 32>>>(
        d_out, (const uint32_t *)g_qwen.lm_head.d_A, (const uint32_t *)d_b, V, Kt,
        g_qwen.lm_head.d_block_scale_inv, g_qwen.lm_head.scale_cols, d_scale, nvar, 0);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "nk sanity launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_x); cudaFree(d_b); cudaFree(d_scale); cudaFree(d_out);
        return -3;
    }
    float *h = (float *)malloc((size_t)nvar * V * sizeof(float));
    cudaMemcpy(h, d_out, (size_t)nvar * V * sizeof(float), cudaMemcpyDeviceToHost);
    float md = 0.0f;
    for (int c = 1; c < nvar; c++) {
        for (int i = 0; i < V; i++) {
            float d = fabsf(h[(size_t)c * V + i] - h[i]);
            if (d > md) md = d;
        }
    }
    if (max_col_diff) *max_col_diff = md;
    free(h);
    cudaFree(d_x); cudaFree(d_b); cudaFree(d_scale); cudaFree(d_out);
    return 0;
}

// Copy the last final-norm hidden state (the lm_head input) to host. Valid after
// qwn_decode / qwn_forward_first. Used to measure the 4-bit-vs-full coverage curve
// offline against a reference lm_head.
extern "C" int qwn_copy_last_norm(float *out, int max_count) {
    if (!g_qwen.d_debug_norm || !out) return -1;
    int n = g_qwen.H < max_count ? g_qwen.H : max_count;
    cudaMemcpy(out, g_qwen.d_debug_norm, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost);
    return n;
}

// Sanity for the optimistic sparse-FP4 draft. Runs the engine sparse 2:4 GEMV on
// a loaded linear-attn projection (proj_id 0=mlp_gate,1=mlp_up,2=mlp_down,
// 3=in_qkv,4=in_z,5=out) with host activation x[K] (ascale=1.0), returns out[M].
// A host reference recomputed from the original weight then validates the full
// chain converter->TQF->loader->kernel. Returns M, or <0 on error.
extern "C" int qwn_sparse_gemv_sanity(int layer, int proj_id,
                                      const float *x_host, float *out_host) {
    if (!g_qwen.initialized || layer < 0 || layer >= g_qwen.L) return -1;
    tq_layer_t *l = &g_qwen.layers[layer];
    tq_qmma_weight_t *w = NULL;
    switch (proj_id) {
        case 0: w = &l->mlp_gate; break;
        case 1: w = &l->mlp_up; break;
        case 2: w = &l->mlp_down; break;
        case 3: w = &l->linear_in_qkv; break;
        case 4: w = &l->linear_in_z; break;
        case 5: w = &l->linear_out; break;
        default: return -2;
    }
    if (!w->has_sparse || !w->d_sparse_a || !w->d_sparse_meta) return -3;
    int M = w->M, K = w->K, Kt64 = w->Kt64, sc = w->scale_cols;
    float *d_x = NULL, *d_out = NULL, *d_ascale = NULL;
    uint8_t *d_b = NULL;
    cudaMalloc(&d_x, (size_t)K * sizeof(float));
    cudaMalloc(&d_b, (size_t)Kt64 * 512);
    cudaMalloc(&d_out, (size_t)M * sizeof(float));
    cudaMalloc(&d_ascale, (size_t)sc * sizeof(float));
    if (!d_x || !d_b || !d_out || !d_ascale) {
        if (d_x) cudaFree(d_x); if (d_b) cudaFree(d_b);
        if (d_out) cudaFree(d_out); if (d_ascale) cudaFree(d_ascale);
        return -4;
    }
    cudaMemcpy(d_x, x_host, (size_t)K * sizeof(float), cudaMemcpyHostToDevice);
    float *ones = (float *)malloc((size_t)sc * sizeof(float));
    for (int i = 0; i < sc; i++) ones[i] = 1.0f;
    cudaMemcpy(d_ascale, ones, (size_t)sc * sizeof(float), cudaMemcpyHostToDevice);
    free(ones);
    cudaMemset(d_b, 0, (size_t)Kt64 * 512);
    k_tq_pack_b_k64<<<(K + 255) / 256, 256>>>(d_b, d_x, K);
    k_tq_sparse_gemv_e2m3<<<w->Mt, 32>>>(
        d_out, w->d_sparse_a, (const uint32_t *)d_b, w->d_sparse_meta,
        w->d_block_scale_inv, d_ascale, M, Kt64, sc);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "sparse_gemv_sanity: %s\n", cudaGetErrorString(err));
        cudaFree(d_x); cudaFree(d_b); cudaFree(d_out); cudaFree(d_ascale);
        return -5;
    }
    cudaMemcpy(out_host, d_out, (size_t)M * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_x); cudaFree(d_b); cudaFree(d_out); cudaFree(d_ascale);
    return M;
}

// Full-precision-activation draft (option B): residual 2xe4m3. Runs the sparse
// GEMV twice on the SAME loaded weights (hi then lo activation), sums -> out[M].
// Activation ~ BF16 (~7 mantissa bits) vs single-e4m3 (~3). ascale=1.0 (x is O(1)).
extern "C" int qwn_sparse_gemv_residual(int layer, int proj_id,
                                        const float *x_host, float *out_host) {
    if (!g_qwen.initialized || layer < 0 || layer >= g_qwen.L) return -1;
    tq_layer_t *l = &g_qwen.layers[layer];
    tq_qmma_weight_t *w = NULL;
    switch (proj_id) {
        case 0: w = &l->mlp_gate; break;
        case 1: w = &l->mlp_up; break;
        case 2: w = &l->mlp_down; break;
        case 3: w = &l->linear_in_qkv; break;
        case 4: w = &l->linear_in_z; break;
        case 5: w = &l->linear_out; break;
        default: return -2;
    }
    if (!w->has_sparse) return -3;
    int M = w->M, K = w->K, Kt64 = w->Kt64, sc = w->scale_cols;
    float *d_x = NULL, *d_out_hi = NULL, *d_out_lo = NULL, *d_ascale = NULL;
    uint8_t *d_b_hi = NULL, *d_b_lo = NULL;
    cudaMalloc(&d_x, (size_t)K * sizeof(float));
    cudaMalloc(&d_b_hi, (size_t)Kt64 * 512);
    cudaMalloc(&d_b_lo, (size_t)Kt64 * 512);
    cudaMalloc(&d_out_hi, (size_t)M * sizeof(float));
    cudaMalloc(&d_out_lo, (size_t)M * sizeof(float));
    cudaMalloc(&d_ascale, (size_t)sc * sizeof(float));
    if (!d_x || !d_b_hi || !d_b_lo || !d_out_hi || !d_out_lo || !d_ascale) {
        if (d_x) cudaFree(d_x); if (d_b_hi) cudaFree(d_b_hi); if (d_b_lo) cudaFree(d_b_lo);
        if (d_out_hi) cudaFree(d_out_hi); if (d_out_lo) cudaFree(d_out_lo);
        if (d_ascale) cudaFree(d_ascale);
        return -4;
    }
    cudaMemcpy(d_x, x_host, (size_t)K * sizeof(float), cudaMemcpyHostToDevice);
    float *ones = (float *)malloc((size_t)sc * sizeof(float));
    for (int i = 0; i < sc; i++) ones[i] = 1.0f;
    cudaMemcpy(d_ascale, ones, (size_t)sc * sizeof(float), cudaMemcpyHostToDevice);
    free(ones);
    cudaMemset(d_b_hi, 0, (size_t)Kt64 * 512);
    cudaMemset(d_b_lo, 0, (size_t)Kt64 * 512);
    k_tq_pack_b_k64_residual<<<(K + 255) / 256, 256>>>(d_b_hi, d_b_lo, d_x, K);
    k_tq_sparse_gemv_e2m3<<<w->Mt, 32>>>(d_out_hi, w->d_sparse_a, (const uint32_t *)d_b_hi,
        w->d_sparse_meta, w->d_block_scale_inv, d_ascale, M, Kt64, sc);
    k_tq_sparse_gemv_e2m3<<<w->Mt, 32>>>(d_out_lo, w->d_sparse_a, (const uint32_t *)d_b_lo,
        w->d_sparse_meta, w->d_block_scale_inv, d_ascale, M, Kt64, sc);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "sparse_gemv_residual: %s\n", cudaGetErrorString(err));
        cudaFree(d_x); cudaFree(d_b_hi); cudaFree(d_b_lo);
        cudaFree(d_out_hi); cudaFree(d_out_lo); cudaFree(d_ascale);
        return -5;
    }
    float *hi = (float *)malloc((size_t)M * sizeof(float));
    cudaMemcpy(hi, d_out_hi, (size_t)M * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_host, d_out_lo, (size_t)M * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < M; i++) out_host[i] += hi[i];
    free(hi);
    cudaFree(d_x); cudaFree(d_b_hi); cudaFree(d_b_lo);
    cudaFree(d_out_hi); cudaFree(d_out_lo); cudaFree(d_ascale);
    return M;
}

static int ensure_debug_vectors(int proj_count) {
    if (!g_qwen.d_debug_x) {
        cudaError_t err = cudaMalloc(&g_qwen.d_debug_x, (size_t)g_qwen.H * sizeof(float));
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMalloc d_debug_x failed: %s\n", cudaGetErrorString(err));
            return -1;
        }
        g_qwen.device_bytes += (size_t)g_qwen.H * sizeof(float);
    }
    if (!g_qwen.d_debug_norm) {
        cudaError_t err = cudaMalloc(&g_qwen.d_debug_norm, (size_t)g_qwen.H * sizeof(float));
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMalloc d_debug_norm failed: %s\n", cudaGetErrorString(err));
            return -2;
        }
        g_qwen.device_bytes += (size_t)g_qwen.H * sizeof(float);
    }
    if (proj_count > 0 && g_qwen.debug_proj_count < proj_count) {
        if (g_qwen.d_debug_proj) {
            cudaFree(g_qwen.d_debug_proj);
            g_qwen.device_bytes -= (size_t)g_qwen.debug_proj_count * sizeof(float);
        }
        cudaError_t err = cudaMalloc(&g_qwen.d_debug_proj, (size_t)proj_count * sizeof(float));
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMalloc d_debug_proj failed: %s\n", cudaGetErrorString(err));
            g_qwen.d_debug_proj = NULL;
            g_qwen.debug_proj_count = 0;
            return -3;
        }
        g_qwen.debug_proj_count = proj_count;
        g_qwen.device_bytes += (size_t)proj_count * sizeof(float);
    }
    return 0;
}

static int ensure_qmma_debug_scratch(const tq_qmma_weight_t *w) {
    int ret = ensure_debug_vectors(w->M);
    if (ret != 0) return ret;
    int b_bytes = w->Kt * 256;
    if (g_qwen.debug_b_bytes < b_bytes) {
        if (g_qwen.d_debug_b) {
            cudaFree(g_qwen.d_debug_b);
            g_qwen.device_bytes -= (size_t)g_qwen.debug_b_bytes;
        }
        cudaError_t err = cudaMalloc(&g_qwen.d_debug_b, (size_t)b_bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMalloc d_debug_b failed: %s\n", cudaGetErrorString(err));
            g_qwen.d_debug_b = NULL;
            g_qwen.debug_b_bytes = 0;
            return -10;
        }
        g_qwen.debug_b_bytes = b_bytes;
        g_qwen.device_bytes += (size_t)b_bytes;
    }
    int raw_count = w->Mt * 128;
    if (g_qwen.debug_raw_count < raw_count) {
        if (g_qwen.d_debug_qmma_raw) {
            cudaFree(g_qwen.d_debug_qmma_raw);
            g_qwen.device_bytes -= (size_t)g_qwen.debug_raw_count * sizeof(float);
        }
        cudaError_t err = cudaMalloc(&g_qwen.d_debug_qmma_raw, (size_t)raw_count * sizeof(float));
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMalloc d_debug_qmma_raw failed: %s\n", cudaGetErrorString(err));
            g_qwen.d_debug_qmma_raw = NULL;
            g_qwen.debug_raw_count = 0;
            return -11;
        }
        g_qwen.debug_raw_count = raw_count;
        g_qwen.device_bytes += (size_t)raw_count * sizeof(float);
    }
    if (g_qwen.debug_qmma_out_count < w->M) {
        if (g_qwen.d_debug_qmma_out) {
            cudaFree(g_qwen.d_debug_qmma_out);
            g_qwen.device_bytes -= (size_t)g_qwen.debug_qmma_out_count * sizeof(float);
        }
        cudaError_t err = cudaMalloc(&g_qwen.d_debug_qmma_out, (size_t)w->M * sizeof(float));
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMalloc d_debug_qmma_out failed: %s\n", cudaGetErrorString(err));
            g_qwen.d_debug_qmma_out = NULL;
            g_qwen.debug_qmma_out_count = 0;
            return -12;
        }
        g_qwen.debug_qmma_out_count = w->M;
        g_qwen.device_bytes += (size_t)w->M * sizeof(float);
    }
    if (!g_qwen.d_debug_scale) {
        cudaError_t err = cudaMalloc(&g_qwen.d_debug_scale, sizeof(float));
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMalloc d_debug_scale failed: %s\n", cudaGetErrorString(err));
            return -13;
        }
        g_qwen.device_bytes += sizeof(float);
    }
    return 0;
}

static int ensure_debug_conv(int count) {
    if (g_qwen.debug_conv_count < count) {
        if (g_qwen.d_debug_conv) {
            cudaFree(g_qwen.d_debug_conv);
            g_qwen.device_bytes -= (size_t)g_qwen.debug_conv_count * sizeof(float);
        }
        cudaError_t err = cudaMalloc(&g_qwen.d_debug_conv, (size_t)count * sizeof(float));
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMalloc d_debug_conv failed: %s\n", cudaGetErrorString(err));
            g_qwen.d_debug_conv = NULL;
            g_qwen.debug_conv_count = 0;
            return -1;
        }
        g_qwen.debug_conv_count = count;
        g_qwen.device_bytes += (size_t)count * sizeof(float);
    }
    return 0;
}

static int ensure_float_buffer(float **ptr, int *capacity, int count, const char *name) {
    if (*capacity >= count) return 0;
    if (*ptr) {
        cudaFree(*ptr);
        g_qwen.device_bytes -= (size_t)(*capacity) * sizeof(float);
    }
    cudaError_t err = cudaMalloc(ptr, (size_t)count * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc %s failed: %s\n", name, cudaGetErrorString(err));
        *ptr = NULL;
        *capacity = 0;
        return -1;
    }
    *capacity = count;
    g_qwen.device_bytes += (size_t)count * sizeof(float);
    return 0;
}

static int ensure_int_buffer(int **ptr, int *capacity, int count, const char *name) {
    if (*capacity >= count) return 0;
    if (*ptr) {
        cudaFree(*ptr);
        g_qwen.device_bytes -= (size_t)(*capacity) * sizeof(int);
    }
    cudaError_t err = cudaMalloc(ptr, (size_t)count * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc %s failed: %s\n", name, cudaGetErrorString(err));
        *ptr = NULL;
        *capacity = 0;
        return -1;
    }
    *capacity = count;
    g_qwen.device_bytes += (size_t)count * sizeof(int);
    return 0;
}

static int ensure_decode_state(void) {
    int lin_key_dim = g_qwen.linear_num_key_heads * g_qwen.linear_key_head_dim;
    int lin_value_dim = g_qwen.linear_num_value_heads * g_qwen.linear_value_head_dim;
    int lin_conv_dim = lin_key_dim * 2 + lin_value_dim;
    int full_kv_dim = g_qwen.nkv * g_qwen.hd;
    for (int i = 0; i < g_qwen.L; i++) {
        tq_layer_t *l = &g_qwen.layers[i];
        if (g_qwen.layer_types[i] == TQ_LAYER_LINEAR_ATTENTION) {
            size_t conv_bytes = (size_t)lin_conv_dim * g_qwen.linear_conv_kernel_dim * sizeof(float);
            size_t state_bytes = (size_t)g_qwen.linear_num_value_heads *
                                 g_qwen.linear_key_head_dim *
                                 g_qwen.linear_value_head_dim * sizeof(float);
            if (!l->d_linear_conv_state) {
                cudaError_t err = cudaMalloc(&l->d_linear_conv_state, conv_bytes);
                if (err != cudaSuccess) return -1;
                g_qwen.device_bytes += conv_bytes;
                l->d_linear_conv_state_owned = l->d_linear_conv_state;
            }
            if (!l->d_linear_recurrent_state) {
                cudaError_t err = cudaMalloc(&l->d_linear_recurrent_state, state_bytes);
                if (err != cudaSuccess) return -2;
                g_qwen.device_bytes += state_bytes;
                l->d_linear_recurrent_state_owned = l->d_linear_recurrent_state;
            }
        } else if (g_qwen.layer_types[i] == TQ_LAYER_FULL_ATTENTION) {
            if (g_qwen.kv_fp8) {
                // FP8 prefix cache: u8 codes + per-(pos, kv_head) scales; the fp32
                // cache is never allocated on this path (-75% KV bytes).
                size_t code_bytes = (size_t)g_qwen.max_seq * full_kv_dim;
                size_t scale_bytes = (size_t)g_qwen.max_seq * g_qwen.nkv * sizeof(float);
                if (g_qwen.kv_q4) {
                    // K side: int4 codes (half the bytes) + per-group (scale, zp)
                    size_t k4_bytes = code_bytes >> 1;
                    size_t k4s_bytes = (size_t)g_qwen.max_seq * g_qwen.nkv * 16 * (g_qwen.kv_q4_fp32s ? sizeof(float) : sizeof(__half));
                    if (!l->d_k_cache4) {
                        if (cudaMalloc(&l->d_k_cache4, k4_bytes) != cudaSuccess) return -3;
                        g_qwen.device_bytes += k4_bytes;
                    }
                    if (!l->d_k_q4s) {
                        if (cudaMalloc(&l->d_k_q4s, k4s_bytes) != cudaSuccess) return -3;
                        g_qwen.device_bytes += k4s_bytes;
                    }
                } else {
                    if (!l->d_k_cache8) {
                        if (cudaMalloc(&l->d_k_cache8, code_bytes) != cudaSuccess) return -3;
                        g_qwen.device_bytes += code_bytes;
                    }
                    if (!l->d_k_scale) {
                        if (cudaMalloc(&l->d_k_scale, scale_bytes) != cudaSuccess) return -3;
                        g_qwen.device_bytes += scale_bytes;
                    }
                }
                if (!l->d_v_cache8) {
                    if (cudaMalloc(&l->d_v_cache8, code_bytes) != cudaSuccess) return -4;
                    g_qwen.device_bytes += code_bytes;
                }
                if (!l->d_v_scale) {
                    if (cudaMalloc(&l->d_v_scale, scale_bytes) != cudaSuccess) return -4;
                    g_qwen.device_bytes += scale_bytes;
                }
                continue;
            }
            size_t cache_bytes = (size_t)g_qwen.max_seq * full_kv_dim * sizeof(float);
            if (!l->d_k_cache) {
                cudaError_t err = cudaMalloc(&l->d_k_cache, cache_bytes);
                if (err != cudaSuccess) return -3;
                g_qwen.device_bytes += cache_bytes;
            }
            if (!l->d_v_cache) {
                cudaError_t err = cudaMalloc(&l->d_v_cache, cache_bytes);
                if (err != cudaSuccess) return -4;
                g_qwen.device_bytes += cache_bytes;
            }
        }
    }
    return 0;
}

extern "C" int qwn_reset_state(void) {
    int ret = ensure_decode_state();
    if (ret != 0) return ret;
    hist_reset();
    int lin_key_dim = g_qwen.linear_num_key_heads * g_qwen.linear_key_head_dim;
    int lin_value_dim = g_qwen.linear_num_value_heads * g_qwen.linear_value_head_dim;
    int lin_conv_dim = lin_key_dim * 2 + lin_value_dim;
    int full_kv_dim = g_qwen.nkv * g_qwen.hd;
    for (int i = 0; i < g_qwen.L; i++) {
        tq_layer_t *l = &g_qwen.layers[i];
        if (g_qwen.layer_types[i] == TQ_LAYER_LINEAR_ATTENTION) {
            // a swap-commit may have parked the live pointers in the archive;
            // a reset re-canonicalizes them to the owned buffers
            l->d_linear_conv_state = l->d_linear_conv_state_owned;
            l->d_linear_recurrent_state = l->d_linear_recurrent_state_owned;
            g_spec_live_in_archive = 0;
            g_spec_slot_base = 0;
            cudaMemsetAsync(l->d_linear_conv_state, 0,
                            (size_t)lin_conv_dim * g_qwen.linear_conv_kernel_dim * sizeof(float),
                            g_qwen.stream);
            cudaMemsetAsync(l->d_linear_recurrent_state, 0,
                            (size_t)g_qwen.linear_num_value_heads *
                            g_qwen.linear_key_head_dim *
                            g_qwen.linear_value_head_dim * sizeof(float),
                            g_qwen.stream);
        } else if (g_qwen.layer_types[i] == TQ_LAYER_FULL_ATTENTION) {
            if (g_qwen.kv_fp8) {
                if (g_qwen.kv_q4) {
                    cudaMemsetAsync(l->d_k_cache4, 0, ((size_t)g_qwen.max_seq * full_kv_dim) >> 1, g_qwen.stream);
                    cudaMemsetAsync(l->d_k_q4s, 0, (size_t)g_qwen.max_seq * g_qwen.nkv * 16 * (g_qwen.kv_q4_fp32s ? sizeof(float) : sizeof(__half)), g_qwen.stream);
                } else {
                    cudaMemsetAsync(l->d_k_cache8, 0, (size_t)g_qwen.max_seq * full_kv_dim, g_qwen.stream);
                    cudaMemsetAsync(l->d_k_scale, 0, (size_t)g_qwen.max_seq * g_qwen.nkv * sizeof(float), g_qwen.stream);
                }
                cudaMemsetAsync(l->d_v_cache8, 0, (size_t)g_qwen.max_seq * full_kv_dim, g_qwen.stream);
                cudaMemsetAsync(l->d_v_scale, 0, (size_t)g_qwen.max_seq * g_qwen.nkv * sizeof(float), g_qwen.stream);
                continue;
            }
            cudaMemsetAsync(l->d_k_cache, 0, (size_t)g_qwen.max_seq * full_kv_dim * sizeof(float), g_qwen.stream);
            cudaMemsetAsync(l->d_v_cache, 0, (size_t)g_qwen.max_seq * full_kv_dim * sizeof(float), g_qwen.stream);
        }
    }
    cudaError_t err = cudaStreamSynchronize(g_qwen.stream);
    return err == cudaSuccess ? 0 : -5;
}
