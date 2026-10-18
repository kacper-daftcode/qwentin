// the native MTP head, MTP draft trees and the batched spec-decode verify forward.
// Part of the forward_qwen.cu unity build: included in order, not a standalone header.

// ============================ Native MTP head =============================
// The MTP draft forward (matches tools/mtp_accept.py exactly):
//   e = embed_tokens(token_{t+1})                          (shared embedding)
//   x = fc(concat[pre_fc_norm_embedding(e), pre_fc_norm_hidden(h)])   ([2H]->[H])
//   x = ONE full_attention decoder layer (mtp.layers.0.*, own KV cache, RoPE@pos)
//   x = mtp.norm(x)
//   logits = lm_head(x)  -> top-k
// h is the main model PRE-final-norm last hidden (the residual into model.norm),
// which is left in d_debug_x by the preceding qwn_decode (run_final_norm reads but
// preserves it). RoPE is relative-shift-invariant, so the engine drives the MTP
// layer with pos == cache slot == MTP step index; this is bit-equivalent (in exact
// arithmetic) to the reference's t+1 positions and reproduces its causal attention.

static int ensure_mtp_buffers(void) {
    // d_debug_x / d_debug_norm scratch (the BF16 MTP path no longer goes through the
    // qmma scratch allocator that used to create them).
    if (ensure_debug_vectors(0) != 0) return -1;
    if (ensure_float_buffer(&g_qwen.d_mtp_e, &g_qwen.mtp_e_count, g_qwen.H, "d_mtp_e") != 0) return -1;
    if (ensure_float_buffer(&g_qwen.d_mtp_fc_in, &g_qwen.mtp_fc_in_count, 2 * g_qwen.H, "d_mtp_fc_in") != 0) return -2;
    if (ensure_float_buffer(&g_qwen.d_mtp_x, &g_qwen.mtp_x_count, g_qwen.H, "d_mtp_x") != 0) return -3;
    if (ensure_float_buffer(&g_qwen.d_mtp_normout, &g_qwen.mtp_normout_count, g_qwen.H, "d_mtp_normout") != 0) return -4;
    if (!g_qwen.d_mtp_topk_vals) {
        if (cudaMalloc(&g_qwen.d_mtp_topk_vals, (size_t)TQ_MTP_TOPK_MAX * sizeof(float)) != cudaSuccess) return -5;
        g_qwen.device_bytes += (size_t)TQ_MTP_TOPK_MAX * sizeof(float);
    }
    if (!g_qwen.d_mtp_topk_ids) {
        if (cudaMalloc(&g_qwen.d_mtp_topk_ids, (size_t)TQ_MTP_TOPK_MAX * sizeof(int)) != cudaSuccess) return -6;
        g_qwen.device_bytes += (size_t)TQ_MTP_TOPK_MAX * sizeof(int);
    }
    tq_layer_t *m = &g_qwen.mtp_layer;
    if (g_qwen.kv_fp8) {
        size_t code_bytes = (size_t)g_qwen.max_seq * g_qwen.nkv * g_qwen.hd;
        size_t scale_bytes = (size_t)g_qwen.max_seq * g_qwen.nkv * sizeof(float);
        if (g_qwen.kv_q4) {
            size_t k4_bytes = code_bytes >> 1;
            size_t k4s_bytes = (size_t)g_qwen.max_seq * g_qwen.nkv * 16 * (g_qwen.kv_q4_fp32s ? sizeof(float) : sizeof(__half));
            if (!m->d_k_cache4) {
                if (cudaMalloc(&m->d_k_cache4, k4_bytes) != cudaSuccess) return -7;
                g_qwen.device_bytes += k4_bytes;
                cudaMemsetAsync(m->d_k_cache4, 0, k4_bytes, g_qwen.stream);
            }
            if (!m->d_k_q4s) {
                if (cudaMalloc(&m->d_k_q4s, k4s_bytes) != cudaSuccess) return -7;
                g_qwen.device_bytes += k4s_bytes;
                cudaMemsetAsync(m->d_k_q4s, 0, k4s_bytes, g_qwen.stream);
            }
        } else {
            if (!m->d_k_cache8) {
                if (cudaMalloc(&m->d_k_cache8, code_bytes) != cudaSuccess) return -7;
                g_qwen.device_bytes += code_bytes;
                cudaMemsetAsync(m->d_k_cache8, 0, code_bytes, g_qwen.stream);
            }
            if (!m->d_k_scale) {
                if (cudaMalloc(&m->d_k_scale, scale_bytes) != cudaSuccess) return -7;
                g_qwen.device_bytes += scale_bytes;
                cudaMemsetAsync(m->d_k_scale, 0, scale_bytes, g_qwen.stream);
            }
        }
        if (!m->d_v_cache8) {
            if (cudaMalloc(&m->d_v_cache8, code_bytes) != cudaSuccess) return -8;
            g_qwen.device_bytes += code_bytes;
            cudaMemsetAsync(m->d_v_cache8, 0, code_bytes, g_qwen.stream);
        }
        if (!m->d_v_scale) {
            if (cudaMalloc(&m->d_v_scale, scale_bytes) != cudaSuccess) return -8;
            g_qwen.device_bytes += scale_bytes;
            cudaMemsetAsync(m->d_v_scale, 0, scale_bytes, g_qwen.stream);
        }
        return 0;
    }
    size_t cache_bytes = (size_t)g_qwen.max_seq * g_qwen.nkv * g_qwen.hd * sizeof(float);
    if (!m->d_k_cache) {
        if (cudaMalloc(&m->d_k_cache, cache_bytes) != cudaSuccess) return -7;
        g_qwen.device_bytes += cache_bytes;
        cudaMemsetAsync(m->d_k_cache, 0, cache_bytes, g_qwen.stream);
    }
    if (!m->d_v_cache) {
        if (cudaMalloc(&m->d_v_cache, cache_bytes) != cudaSuccess) return -8;
        g_qwen.device_bytes += cache_bytes;
        cudaMemsetAsync(m->d_v_cache, 0, cache_bytes, g_qwen.stream);
    }
    return 0;
}

extern "C" int qwn_has_mtp(void) {
    return (g_qwen.initialized && g_qwen.has_mtp_section) ? 1 : 0;
}

// Zero the MTP KV cache so a fresh causal MTP sequence can start at slot 0.
extern "C" int qwn_mtp_reset(void) {
    if (!g_qwen.has_mtp_section) return -1;
    int ret = ensure_mtp_buffers();
    if (ret != 0) return ret;
    tq_layer_t *m = &g_qwen.mtp_layer;
    if (g_qwen.kv_fp8) {
        size_t code_bytes = (size_t)g_qwen.max_seq * g_qwen.nkv * g_qwen.hd;
        if (g_qwen.kv_q4) {
            cudaMemsetAsync(m->d_k_cache4, 0, code_bytes >> 1, g_qwen.stream);
            cudaMemsetAsync(m->d_k_q4s, 0, (size_t)g_qwen.max_seq * g_qwen.nkv * 16 * (g_qwen.kv_q4_fp32s ? sizeof(float) : sizeof(__half)), g_qwen.stream);
        } else {
            cudaMemsetAsync(m->d_k_cache8, 0, code_bytes, g_qwen.stream);
            cudaMemsetAsync(m->d_k_scale, 0, (size_t)g_qwen.max_seq * g_qwen.nkv * sizeof(float), g_qwen.stream);
        }
        cudaMemsetAsync(m->d_v_cache8, 0, code_bytes, g_qwen.stream);
        cudaMemsetAsync(m->d_v_scale, 0, (size_t)g_qwen.max_seq * g_qwen.nkv * sizeof(float), g_qwen.stream);
    } else {
    size_t cache_bytes = (size_t)g_qwen.max_seq * g_qwen.nkv * g_qwen.hd * sizeof(float);
    cudaMemsetAsync(m->d_k_cache, 0, cache_bytes, g_qwen.stream);
    cudaMemsetAsync(m->d_v_cache, 0, cache_bytes, g_qwen.stream);
    }
    cudaError_t err = cudaStreamSynchronize(g_qwen.stream);
    return err == cudaSuccess ? 0 : -9;
}

// BF16-weight GEMV into a destination buffer for the full-precision MTP projections.
static int run_mtp_bf16_gemv(const tq_bf16_weight_t *w, const float *x, float *dst) {
    k_tq_bf16_gemv<<<w->M, 256, 0, g_qwen.stream>>>(dst, w->d_w, x, w->M, w->K);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "MTP BF16 GEMV launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

// One MTP full_attention decoder layer over d_mtp_x (fc output) at slot `pos`,
// leaving the result in d_debug_layer_out. Mirrors run_full_layer_decode_from_current
// but all 8 projections run in BF16 (full precision) and use the MTP KV cache;
// attention/rmsnorm/RoPE/silu kernels are reused as-is. Residual = the fc output.
static int run_mtp_full_layer(int pos) {
    tq_layer_t *m = &g_qwen.mtp_layer;     // norms + KV cache only
    int q_m = g_qwen.nh * g_qwen.hd * 2;
    int kv_m = g_qwen.nkv * g_qwen.hd;
    int attn_m = g_qwen.nh * g_qwen.hd;
    if (ensure_float_buffer(&g_qwen.d_debug_qkv, &g_qwen.debug_qkv_count, q_m, "d_debug_qkv") != 0) return -1;
    if (ensure_float_buffer(&g_qwen.d_debug_a_proj, &g_qwen.debug_a_count, kv_m, "d_debug_a_proj") != 0) return -2;
    if (ensure_float_buffer(&g_qwen.d_debug_z, &g_qwen.debug_z_count, kv_m, "d_debug_z") != 0) return -3;
    if (ensure_float_buffer(&g_qwen.d_debug_core, &g_qwen.debug_core_count, attn_m, "d_debug_core") != 0) return -4;
    if (ensure_float_buffer(&g_qwen.d_debug_resid, &g_qwen.debug_resid_count, g_qwen.H, "d_debug_resid") != 0) return -5;
    if (ensure_float_buffer(&g_qwen.d_debug_qmma_out, &g_qwen.debug_qmma_out_count, g_qwen.H, "d_debug_qmma_out") != 0) return -6;
    if (ensure_float_buffer(&g_qwen.d_debug_post_norm, &g_qwen.debug_post_norm_count, g_qwen.H, "d_debug_post_norm") != 0) return -7;
    if (ensure_float_buffer(&g_qwen.d_debug_gate, &g_qwen.debug_gate_count, g_qwen.I, "d_debug_gate") != 0) return -8;
    if (ensure_float_buffer(&g_qwen.d_debug_up, &g_qwen.debug_up_count, g_qwen.I, "d_debug_up") != 0) return -9;
    if (ensure_float_buffer(&g_qwen.d_debug_mlp_hidden, &g_qwen.debug_mlp_hidden_count, g_qwen.I, "d_debug_mlp_hidden") != 0) return -10;
    if (ensure_float_buffer(&g_qwen.d_debug_layer_out, &g_qwen.debug_layer_out_count, g_qwen.H, "d_debug_layer_out") != 0) return -11;
    // self-attention: input_layernorm -> q/k/v (BF16) -> attn -> o_proj (BF16) + residual
    k_tq_qwen_rmsnorm<<<8, 1024, 0, g_qwen.stream>>>(g_qwen.d_debug_norm, g_qwen.d_mtp_x,
                                                     m->d_input_ln, g_qwen.H, g_qwen.eps);
    int ret = run_mtp_bf16_gemv(&g_qwen.mtp_q_proj, g_qwen.d_debug_norm, g_qwen.d_debug_qkv);
    if (ret != 0) return -12;
    ret = run_mtp_bf16_gemv(&g_qwen.mtp_k_proj, g_qwen.d_debug_norm, g_qwen.d_debug_a_proj);
    if (ret != 0) return -13;
    ret = run_mtp_bf16_gemv(&g_qwen.mtp_v_proj, g_qwen.d_debug_norm, g_qwen.d_debug_z);
    if (ret != 0) return -14;
    if (ensure_attn_scores() != 0) return -97;
    k_tq_full_attn_decode<<<g_qwen.nh, g_qwen.hd, 0, g_qwen.stream>>>(
        g_qwen.d_debug_core, g_qwen.d_debug_qkv, g_qwen.d_debug_a_proj,
        g_qwen.d_debug_z, m->d_q_norm, m->d_k_norm, m->d_k_cache, m->d_v_cache,
        m->d_k_cache8, m->d_k_scale, m->d_v_cache8, m->d_v_scale,
        m->d_k_cache4, m->d_k_q4s,
        pos, g_qwen.nh, g_qwen.nkv, g_qwen.hd, g_qwen.eps, g_qwen.rope_theta,
        g_qwen.d_attn_scores, g_qwen.max_seq);
    ret = run_mtp_bf16_gemv(&g_qwen.mtp_o_proj, g_qwen.d_debug_core, g_qwen.d_debug_qmma_out);
    if (ret != 0) return -15;
    k_tq_add_vec<<<(g_qwen.H + 255) / 256, 256, 0, g_qwen.stream>>>(
        g_qwen.d_debug_resid, g_qwen.d_mtp_x, g_qwen.d_debug_qmma_out, g_qwen.H);
    // MLP: post_attention_layernorm -> gate/up (BF16) -> silu*up -> down (BF16) + residual
    k_tq_qwen_rmsnorm<<<8, 1024, 0, g_qwen.stream>>>(g_qwen.d_debug_post_norm, g_qwen.d_debug_resid,
                                                     m->d_post_ln, g_qwen.H, g_qwen.eps);
    ret = run_mtp_bf16_gemv(&g_qwen.mtp_gate, g_qwen.d_debug_post_norm, g_qwen.d_debug_gate);
    if (ret != 0) return -16;
    ret = run_mtp_bf16_gemv(&g_qwen.mtp_up, g_qwen.d_debug_post_norm, g_qwen.d_debug_up);
    if (ret != 0) return -17;
    k_tq_silu_mul<<<(g_qwen.I + 255) / 256, 256, 0, g_qwen.stream>>>(
        g_qwen.d_debug_mlp_hidden, g_qwen.d_debug_gate, g_qwen.d_debug_up, g_qwen.I);
    ret = run_mtp_bf16_gemv(&g_qwen.mtp_down, g_qwen.d_debug_mlp_hidden, g_qwen.d_debug_qmma_out);
    if (ret != 0) return -18;
    k_tq_add_vec<<<(g_qwen.H + 255) / 256, 256, 0, g_qwen.stream>>>(
        g_qwen.d_debug_layer_out, g_qwen.d_debug_resid, g_qwen.d_debug_qmma_out, g_qwen.H);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "MTP layer launch failed: %s\n", cudaGetErrorString(err));
        return -19;
    }
    return 0;
}

// KV-only MTP layer pass for trunk advances: a trunk extension only needs the
// MTP KV cache slot written (the layer output, q/attention/o_proj and the whole
// MLP are never consumed). Runs input_ln -> k/v projections -> knorm+rope ->
// cache write; identical k/v floats to the full layer's attention kernel.
// Cuts the advance from ~849 MB of BF16 weight reads to ~126 MB.
__global__ void k_tq_mtp_kv_write(float *k_cache, float *v_cache,
                                  uint8_t *k_cache8, float *k_scale,
                                  uint8_t *v_cache8, float *v_scale,
                                  uint8_t *k_cache4, float *k_q4s,
                                  const float *k_proj, const float *v_proj,
                                  const uint16_t *k_norm_w, int pos,
                                  int nkv, int hd, float eps, float rope_theta) {
    __shared__ float partial[8];
    __shared__ float k_sum_shared;
    __shared__ float rot[256];
    __shared__ float gsz[16];
    int kv_head = blockIdx.x;
    int tid = threadIdx.x;
    if (kv_head >= nkv || tid >= hd) return;
    int rotary_dim = 64;
    float kv = k_proj[kv_head * hd + tid];
    float ksum = kv * kv;
    for (int offset = 16; offset > 0; offset >>= 1)
        ksum += __shfl_down_sync(0xffffffff, ksum, offset);
    if ((tid & 31) == 0) partial[tid >> 5] = ksum;
    __syncthreads();
    if (tid < 8) {
        float vtmp = partial[tid];
        for (int offset = 4; offset > 0; offset >>= 1) vtmp += __shfl_down_sync(0xff, vtmp, offset);
        if (tid == 0) k_sum_shared = vtmp;
    }
    __syncthreads();
    kv = kv * rsqrtf(k_sum_shared / (float)hd + eps) * (1.0f + tq_bf16_to_float(k_norm_w[tid]));
    if (tid < rotary_dim) {
        int idx = tid & 31;
        float freq = powf(rope_theta, -((float)(2 * idx) / (float)rotary_dim));
        float angle = (float)pos * freq;
        float c = cosf(angle);
        float s = sinf(angle);
        float k_pair = k_proj[kv_head * hd + ((tid < 32) ? tid + 32 : tid - 32)];
        k_pair = k_pair * rsqrtf(k_sum_shared / (float)hd + eps) * (1.0f + tq_bf16_to_float(k_norm_w[(tid < 32) ? tid + 32 : tid - 32]));
        float k_rot = (tid < 32) ? -k_pair : k_pair;
        kv = kv * c + k_rot * s;
    }
    if (k_cache4) {
        // TQ_KV_Q4 trunk write: V row E4M3, K row rotated + int4-group-quantized
        float vv = v_proj[kv_head * hd + tid];
        __syncthreads();
        float vscl = tq_kv_block_scale(vv, partial, &k_sum_shared);
        v_cache8[((size_t)pos * nkv + kv_head) * hd + tid] = tq_float_to_e4m3(vv / vscl);
        if (tid == 0) v_scale[(size_t)pos * nkv + kv_head] = vscl;
        rot[tid] = kv;
        __syncthreads();
        tq_fwht256(rot, tid);
        uint8_t code = tq_k_q4_quant(rot[tid], tid, gsz);
        unsigned hi4 = __shfl_down_sync(0xffffffffu, (unsigned)code, 1);
        if ((tid & 1) == 0)
            k_cache4[((size_t)pos * nkv + kv_head) * (hd >> 1) + (tid >> 1)] =
                (uint8_t)(code | ((hi4 & 15u) << 4));
        if (tid < 16)
            q4s_put(k_q4s, ((size_t)pos * nkv + kv_head) * 16 + tid, gsz[tid]);
        return;
    }
    if (k_cache8) {
        float vv = v_proj[kv_head * hd + tid];
        __syncthreads();
        float kscl = tq_kv_block_scale(kv, partial, &k_sum_shared);
        k_cache8[((size_t)pos * nkv + kv_head) * hd + tid] = tq_float_to_e4m3(kv / kscl);
        __syncthreads();
        float vscl = tq_kv_block_scale(vv, partial, &k_sum_shared);
        v_cache8[((size_t)pos * nkv + kv_head) * hd + tid] = tq_float_to_e4m3(vv / vscl);
        if (tid == 0) {
            k_scale[(size_t)pos * nkv + kv_head] = kscl;
            v_scale[(size_t)pos * nkv + kv_head] = vscl;
        }
        return;
    }
    k_cache[((size_t)pos * nkv + kv_head) * hd + tid] = kv;
    v_cache[((size_t)pos * nkv + kv_head) * hd + tid] = v_proj[kv_head * hd + tid];
}

static int mtp_fast_advance_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("TQ_MTP_FAST_ADVANCE");
        cached = env ? atoi(env) : 1;
    }
    return cached;
}

static int run_mtp_kv_only(int pos) {
    tq_layer_t *m = &g_qwen.mtp_layer;
    int kv_m = g_qwen.nkv * g_qwen.hd;
    if (ensure_float_buffer(&g_qwen.d_debug_a_proj, &g_qwen.debug_a_count, kv_m, "d_debug_a_proj") != 0) return -2;
    if (ensure_float_buffer(&g_qwen.d_debug_z, &g_qwen.debug_z_count, kv_m, "d_debug_z") != 0) return -3;
    k_tq_qwen_rmsnorm<<<8, 1024, 0, g_qwen.stream>>>(g_qwen.d_debug_norm, g_qwen.d_mtp_x,
                                                     m->d_input_ln, g_qwen.H, g_qwen.eps);
    int ret = run_mtp_bf16_gemv(&g_qwen.mtp_k_proj, g_qwen.d_debug_norm, g_qwen.d_debug_a_proj);
    if (ret != 0) return -13;
    ret = run_mtp_bf16_gemv(&g_qwen.mtp_v_proj, g_qwen.d_debug_norm, g_qwen.d_debug_z);
    if (ret != 0) return -14;
    k_tq_mtp_kv_write<<<g_qwen.nkv, g_qwen.hd, 0, g_qwen.stream>>>(
        m->d_k_cache, m->d_v_cache,
        m->d_k_cache8, m->d_k_scale, m->d_v_cache8, m->d_v_scale,
        m->d_k_cache4, m->d_k_q4s,
        g_qwen.d_debug_a_proj, g_qwen.d_debug_z,
        m->d_k_norm, pos, g_qwen.nkv, g_qwen.hd, g_qwen.eps, g_qwen.rope_theta);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "MTP kv-only launch failed: %s\n", cudaGetErrorString(err));
        return -19;
    }
    return 0;
}

// Core MTP forward from a PRE-final-norm hidden (device ptr), fed token token_id,
// RoPE/cache slot pos. Leaves the top-k in d_mtp_topk_{ids,vals}.
static int run_mtp_forward(const float *d_hidden, int token_id, int pos, int k) {
    if (!g_qwen.has_mtp_section) {
        fprintf(stderr, "qwn_mtp: model has no MTP section\n");
        return -1;
    }
    if (g_qwen.mtp_exl3) return -40;             // EXL3 MTP projections: paged drafter only
    if (token_id < 0 || token_id >= g_qwen.V) { fprintf(stderr, "qwn_mtp: bad token %d\n", token_id); return -2; }
    if (pos < 0 || pos >= g_qwen.max_seq) { fprintf(stderr, "qwn_mtp: bad pos %d\n", pos); return -3; }
    int ret = ensure_mtp_buffers();
    if (ret != 0) return ret;
    // fc_in[0:H] = pre_fc_norm_embedding(embed(token)); fc_in[H:2H] = pre_fc_norm_hidden(hidden)
    k_tq_embed_lookup<<<(g_qwen.H + 255) / 256, 256, 0, g_qwen.stream>>>(
        g_qwen.d_mtp_e, g_qwen.d_embed, token_id, g_qwen.H);
    k_tq_qwen_rmsnorm<<<8, 1024, 0, g_qwen.stream>>>(
        g_qwen.d_mtp_fc_in, g_qwen.d_mtp_e, g_qwen.d_mtp_pre_fc_norm_emb, g_qwen.H, g_qwen.eps);
    k_tq_qwen_rmsnorm<<<8, 1024, 0, g_qwen.stream>>>(
        g_qwen.d_mtp_fc_in + g_qwen.H, d_hidden, g_qwen.d_mtp_pre_fc_norm_hidden, g_qwen.H, g_qwen.eps);
    // x = fc(concat)   (BF16 weight [H, 2H] x FP32 activation)
    ret = run_mtp_bf16_gemv(&g_qwen.mtp_fc, g_qwen.d_mtp_fc_in, g_qwen.d_mtp_x);
    if (ret != 0) return -11;
    ret = run_mtp_full_layer(pos);
    if (ret != 0) return -12;
    k_tq_qwen_rmsnorm<<<8, 1024, 0, g_qwen.stream>>>(
        g_qwen.d_mtp_normout, g_qwen.d_debug_layer_out, g_qwen.d_mtp_norm, g_qwen.H, g_qwen.eps);
    if (ensure_float_buffer(&g_qwen.d_debug_logits, &g_qwen.debug_logits_count, g_qwen.V, "d_debug_logits") != 0) return -13;
    if (ensure_float_buffer(&g_qwen.d_debug_arg_vals, &g_qwen.debug_arg_count, TQ_ARGMAX_BLOCKS + 1, "d_debug_arg_vals") != 0) return -14;
    if (ensure_int_buffer(&g_qwen.d_debug_arg_ids, &g_qwen.debug_arg_ids_count, TQ_ARGMAX_BLOCKS + 1, "d_debug_arg_ids") != 0) return -15;
    if (g_qwen.tie_word_embeddings) {
        k_tq_lm_head_bf16<<<g_qwen.V, 256, 0, g_qwen.stream>>>(
            g_qwen.d_debug_logits, g_qwen.d_mtp_normout, g_qwen.d_embed, g_qwen.H, g_qwen.V);
    } else {
        ret = run_qmma_projection_from_input_to_buffer(&g_qwen.lm_head, g_qwen.d_mtp_normout,
                                                       g_qwen.d_debug_logits, g_qwen.V);
        if (ret != 0) return -16;
    }
    // Snapshot the full logits before the top-k mask destroys them (validation).
    if (ensure_float_buffer(&g_qwen.d_mtp_logits, &g_qwen.mtp_logits_count, g_qwen.V, "d_mtp_logits") == 0) {
        cudaMemcpyAsync(g_qwen.d_mtp_logits, g_qwen.d_debug_logits, (size_t)g_qwen.V * sizeof(float),
                        cudaMemcpyDeviceToDevice, g_qwen.stream);
    }
    if (k < 1) k = 1;
    if (k > TQ_MTP_TOPK_MAX) k = TQ_MTP_TOPK_MAX;
    // Peel off top-k: argmax, record+mask, repeat (reuses the optimistic-guard kernels).
    for (int i = 0; i < k; i++) {
        k_tq_argmax_stage1<<<TQ_ARGMAX_BLOCKS, 256, 0, g_qwen.stream>>>(
            g_qwen.d_debug_arg_vals, g_qwen.d_debug_arg_ids, g_qwen.d_debug_logits, g_qwen.V);
        k_tq_argmax_stage2<<<1, 1024, 0, g_qwen.stream>>>(
            g_qwen.d_debug_arg_vals + TQ_ARGMAX_BLOCKS, g_qwen.d_debug_arg_ids + TQ_ARGMAX_BLOCKS,
            g_qwen.d_debug_arg_vals, g_qwen.d_debug_arg_ids, TQ_ARGMAX_BLOCKS);
        k_tq_topk_record_and_mask<<<1, 1, 0, g_qwen.stream>>>(
            g_qwen.d_debug_logits, g_qwen.d_mtp_topk_vals, g_qwen.d_mtp_topk_ids,
            g_qwen.d_debug_arg_vals + TQ_ARGMAX_BLOCKS, g_qwen.d_debug_arg_ids + TQ_ARGMAX_BLOCKS, i);
    }
    g_qwen.mtp_topk_count = k;
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "MTP forward launch failed: %s\n", cudaGetErrorString(err));
        return -17;
    }
    return 0;
}

// ============== Batched MTP tree build (depth waves, KV archive) ==============
// The sequential builder replays the whole path per expansion (O(depth) full MTP
// forwards each, 849 MB BF16 + lm_head). The batched builder expands a whole
// depth wave in ONE pass over the MTP weights: batched BF16 GEMVs (weights read
// once for <=16 columns), per-node KV archived (the tree attention reads trunk
// from the real MTP cache + ancestors from the archive -- same machinery as the
// verify), batched lm_head through the spec nk path.

static void spec_pack(const float *in_nk, int K, int N);
static int spec_gemv(const tq_qmma_weight_t *w, float *out, int N);
static int spec_fused_pack_enabled(void);
static int ensure_spec_buffers(void);
static int spec_recur_bf16(void);
static int spec_persist_enabled(void);

// Batched BF16 GEMV: out[col*M+row] = sum_k w[row,k] * x[col*K+k]. One block per
// row; per-thread partials accumulate all N columns while streaming the weight
// row once; the 256-wide shared tree reduce per column matches k_tq_bf16_gemv's
// order exactly (bit-identical per column).
__global__ void k_tq_bf16_gemv_nk(float *out, const uint16_t *w, const float *x,
                                  int M, int K, int N) {
    __shared__ float smem[256];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= M) return;
    const uint16_t *wr = w + (size_t)row * K;
    float acc[16];
    #pragma unroll
    for (int c = 0; c < 16; c++) acc[c] = 0.0f;
    for (int k = tid; k < K; k += blockDim.x) {
        float wv = tq_bf16_to_float(wr[k]);
        #pragma unroll
        for (int c = 0; c < 16; c++)
            if (c < N) acc[c] += wv * x[(size_t)c * K + k];
    }
    for (int c = 0; c < N; c++) {
        smem[tid] = acc[c];
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) smem[tid] += smem[tid + stride];
            __syncthreads();
        }
        if (tid == 0) out[(size_t)c * M + row] = smem[0];
        __syncthreads();
    }
}

__global__ void k_tq_embed_lookup_nk(float *out, const uint16_t *embed,
                                     const int *tokens, int H) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= H) return;
    int tok = tokens[blockIdx.y];
    out[(size_t)blockIdx.y * H + i] = tq_embed_lookup_one(embed, (size_t)tok, i, H);
}

// Gather rows: out[n] = src[idx[n]] (both [.. x H]).
__global__ void k_tq_gather_rows_nk(float *out, const float *src, const int *idx, int H) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= H) return;
    out[(size_t)blockIdx.y * H + i] = src[(size_t)idx[blockIdx.y] * H + i];
}

// rmsnorm_b with an output stride/offset (for packing norms into fc_in[N x 2H]).
__global__ void __launch_bounds__(1024, 1)
k_tq_qwen_rmsnorm_b_out(float *out, int out_stride, int out_off,
                                        const float *x, const uint16_t *weight,
                                        int N, float eps) {
    __shared__ float smem[32];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    const float *xn = x + (size_t)blockIdx.y * N;
    float *outn = out + (size_t)blockIdx.y * out_stride + out_off;
    float sum_sq = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) { float v = xn[i]; sum_sq += v * v; }
    for (int offset = 16; offset > 0; offset >>= 1) sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    if (lane == 0) smem[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        sum_sq = (lane < (blockDim.x + 31) / 32) ? smem[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        if (lane == 0) smem[0] = rsqrtf(sum_sq / (float)N + eps);
    }
    __syncthreads();
    float rms_inv = smem[0];
    int stripe = (N + gridDim.x - 1) / gridDim.x;
    int begin = blockIdx.x * stripe;
    int end = begin + stripe;
    if (end > N) end = N;
    for (int i = begin + tid; i < end; i += blockDim.x) {
        float w = 1.0f + tq_bf16_to_float(weight[i]);
        outn[i] = xn[i] * rms_inv * w;
    }
}

// Batched-tree scratch (allocated lazily; tree build never overlaps the verify so
// the big d_spec_* buffers are reused for projections/logits).
static float *g_mtpb_fc_in = NULL;     // [16 x 2H]
static float *g_mtpb_x = NULL;         // [16 x H]
static float *g_mtpb_norm[2] = {NULL, NULL};   // wave normout ping-pong [16 x H]
static float *g_mtpb_karch = NULL, *g_mtpb_varch = NULL;   // [16 x kv_m]
static int *g_mtpb_dev = NULL;         // tokens[16] | gather idx[16] | pos[16] | anc[16*16]

static int ensure_mtp_batch_buffers(void) {
    if (g_mtpb_fc_in) return 0;
    int H = g_qwen.H, kv_m = g_qwen.nkv * g_qwen.hd;
    if (cudaMalloc(&g_mtpb_fc_in, (size_t)16 * 2 * H * sizeof(float)) != cudaSuccess) return -1;
    if (cudaMalloc(&g_mtpb_x, (size_t)16 * H * sizeof(float)) != cudaSuccess) return -1;
    if (cudaMalloc(&g_mtpb_norm[0], (size_t)16 * H * sizeof(float)) != cudaSuccess) return -1;
    if (cudaMalloc(&g_mtpb_norm[1], (size_t)16 * H * sizeof(float)) != cudaSuccess) return -1;
    if (cudaMalloc(&g_mtpb_karch, (size_t)16 * kv_m * sizeof(float)) != cudaSuccess) return -1;
    if (cudaMalloc(&g_mtpb_varch, (size_t)16 * kv_m * sizeof(float)) != cudaSuccess) return -1;
    // zero once: the MMA path's masked-slot p=0 x garbage(NaN) protection
    cudaMemset(g_mtpb_karch, 0, (size_t)16 * kv_m * sizeof(float));
    cudaMemset(g_mtpb_varch, 0, (size_t)16 * kv_m * sizeof(float));
    if (cudaMalloc(&g_mtpb_dev, (size_t)(16 + 16 + 16 + 16 * 16 + 16) * sizeof(int)) != cudaSuccess) return -1;
    return 0;
}

// Quantize one BF16 MTP projection to the dense E2M3 QMMA format on the GPU
// (same pack policy as the main weights: 128x128 pow2 block scales).
static int mtp_quant_one(const tq_bf16_weight_t *src, tq_qmma_weight_t *dst) {
    int M = src->M, K = src->K;
    if (!src->d_w || (M % 16) != 0 || (K % 32) != 0) return -1;
    int Mt = M / 16, Kt = K / 32;
    int sr = (M + 127) / 128, sc = (K + 127) / 128;
    memset(dst, 0, sizeof(*dst));
    uint32_t *d_A = NULL;
    float *d_scale = NULL;
    if (cudaMalloc(&d_A, (size_t)Mt * Kt * 384u) != cudaSuccess) return -2;
    if (cudaMalloc(&d_scale, (size_t)sr * sc * sizeof(float)) != cudaSuccess) { cudaFree(d_A); return -2; }
    g_qwen.device_bytes += (size_t)Mt * Kt * 384u + (size_t)sr * sc * sizeof(float);
    k_tq_block_absmax_pow2_e2m3<<<dim3(sr, sc), 256, 0, g_qwen.stream>>>(src->d_w, M, K, sc, d_scale);
    long total = (long)Mt * Kt * 96;
    k_tq_pack_bf16_to_e2m3_dense<<<(unsigned)((total + 255) / 256), 256, 0, g_qwen.stream>>>(
        src->d_w, M, K, sc, d_scale, d_A, Kt);
    dst->M = M; dst->K = K; dst->Mt = Mt; dst->Kt = Kt;
    dst->inv_scale = 1.0f;
    dst->d_A = (uint8_t *)d_A;
    dst->d_block_scale_inv = d_scale;
    dst->block_scaled = 1;
    dst->row_major = 0;
    dst->word_major = 0;
    dst->e2m3 = 1;
    dst->scale_rows = sr;
    dst->scale_cols = sc;
    return 0;
}

static int mtp_quant_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("TQ_MTP_QUANT");
        cached = env ? atoi(env) : 1;
    }
    return cached;
}

// Draft-side lm_head vocab shortlist (TREE BUILD ONLY -- verify always reads the
// full head, so committed output stays exactly the full-FP6 greedy). MEASURED
// NEGATIVE, default OFF: although a 98304-row BPE prefix covers 99.8% of final
// draft tokens, accept drops 4.01 -> 3.63 and e2e 126 -> 116 tok/s. The draft
// tree is far more sensitive than token coverage suggests: one out-of-prefix
// top-2/tau distortion poisons the conditioning of every deeper wave in that
// branch. Kept as TQ_MTP_VOCAB_CAP=<rows> for experiments (0 = off).
static int mtp_vocab_cap(void) {
    static int cached = -2;
    if (cached == -2) {
        const char *e = getenv("TQ_MTP_VOCAB_CAP");
        int v = e ? atoi(e) : 0;
        v &= ~15;                          // m16 row-tile granularity
        if (v <= 0 || v >= g_qwen.V) v = 0;
        cached = v;
    }
    return cached;
}

// Pack all 8 MTP projections (one-time, ~0.3 s) so tree waves stream 0.75 B/elem
// instead of 2 B/elem. Requires the global e2m3 weight format (gc_weight_e2m3).
static int ensure_mtp_quant(void) {
    if (g_qwen.mtp_quant_ready) return 0;
    if (g_qwen.mtp_exl3) return -40;             // EXL3 MTP projections: paged drafter only
    const tq_bf16_weight_t *srcs[8] = {&g_qwen.mtp_fc, &g_qwen.mtp_q_proj, &g_qwen.mtp_k_proj,
                                       &g_qwen.mtp_v_proj, &g_qwen.mtp_o_proj, &g_qwen.mtp_gate,
                                       &g_qwen.mtp_up, &g_qwen.mtp_down};
    for (int i = 0; i < 8; i++) {
        int ret = mtp_quant_one(srcs[i], &g_qwen.mtp_qw[i]);
        if (ret != 0) {
            fprintf(stderr, "mtp head quantize failed at %d: %d\n", i, ret);
            return ret;
        }
    }
    if (cudaStreamSynchronize(g_qwen.stream) != cudaSuccess) return -3;
    fprintf(stderr, "mtp head quantized to E2M3 (8 projections)\n");
    g_qwen.mtp_quant_ready = 1;
    return 0;
}

// FP4 draft lm_head (TQ_MTP_HEAD_FP4=1, opt-in until probe-validated): repack
// the E2M3 lm_head to E2M1 (1.03 -> 0.69 GB read per tree wave). Double
// quantization (BF16->E2M3->E2M1) -- a draft-only quality trade, gated by the
// teacher-forced accept probe. Verify always uses the E2M3 head.
static int mtp_head_fp4_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("TQ_MTP_HEAD_FP4");
        cached = (e && atoi(e) != 0) ? 1 : 0;
    }
    return cached;
}

static int ensure_lmhead_fp4(void) {
    if (g_qwen.lm_head_fp4_ready) return 0;
    const tq_qmma_weight_t *src = &g_qwen.lm_head;
    if (!src->d_A || !src->e2m3 || src->row_major || src->word_major) return -1;
    int M = src->M, Kt = src->Kt;
    int sr = src->scale_rows, sc = src->scale_cols;
    tq_qmma_weight_t *dst = &g_qwen.lm_head_fp4;
    memset(dst, 0, sizeof(*dst));
    uint32_t *d_A = NULL;
    float *d_scale = NULL;
    size_t abytes = (size_t)src->Mt * Kt * 256u;
    if (cudaMalloc(&d_A, abytes) != cudaSuccess) return -2;
    if (cudaMalloc(&d_scale, (size_t)sr * sc * sizeof(float)) != cudaSuccess) { cudaFree(d_A); return -2; }
    g_qwen.device_bytes += abytes + (size_t)sr * sc * sizeof(float);
    k_tq_block_absmax_e2m1_from_e2m3<<<dim3(sr, sc), 256, 0, g_qwen.stream>>>(
        (const uint32_t *)src->d_A, Kt, M, src->K, sc, src->d_block_scale_inv, d_scale);
    long total = (long)src->Mt * Kt * 64;
    k_tq_repack_e2m3_to_e2m1<<<(unsigned)((total + 255) / 256), 256, 0, g_qwen.stream>>>(
        (const uint32_t *)src->d_A, M, src->K, sc, src->d_block_scale_inv, d_scale, d_A, Kt);
    if (cudaStreamSynchronize(g_qwen.stream) != cudaSuccess) { cudaFree(d_A); cudaFree(d_scale); return -3; }
    *dst = *src;
    dst->d_A = (uint8_t *)d_A;
    dst->d_block_scale_inv = d_scale;
    dst->e2m3 = 0;
    dst->e2m1 = 1;
    dst->d_sparse_a = NULL; dst->d_sparse_meta = NULL; dst->has_sparse = 0;
    fprintf(stderr, "lm_head repacked to E2M1 (%.0f MB)\n", abytes / 1.0e6);
    g_qwen.lm_head_fp4_ready = 1;
    return 0;
}

// One MTP wave projection: quantized nk GEMV (weights at 0.75 B/elem, one read
// per wave) when the head is packed, BF16 fallback otherwise. do_pack packs the
// shared input B-fragment (q/k/v and gate/up share one pack).
static int mtp_wave_gemv(int qi, const tq_bf16_weight_t *bw, const float *in, int K,
                         float *out, int nw, int do_pack) {
    if (g_qwen.mtp_quant_ready && mtp_quant_enabled()) {
        if (do_pack) spec_pack(in, K, nw);
        return spec_gemv(&g_qwen.mtp_qw[qi], out, nw);
    }
    k_tq_bf16_gemv_nk<<<bw->M, 256, 0, g_qwen.stream>>>(out, bw->d_w, in, bw->M, K, nw);
    return 0;
}

// Expand one wave of `nw` tree nodes (all at depth `d`, KV slot c->pos + d) in one
// batched MTP pass. hidden_src = parent post-norm hiddens (gathered by hsrc_idx
// from prev_norm; for the root wave hsrc_idx[0] = -1 -> d_mtp_root_hidden).
// arch_slot[i] = this node's KV-archive slot; anc rows give the slots of the
// node's ancestors at depths 0..d. Leaves per-node top-k in out_ids/out_vals
// (k each) and the wave's post-norm hiddens in norm_out.
static int mtp_batch_expand_wave(int nw, int d, int pos_base,
                                 const int *tokens, const int *hsrc_idx,
                                 const float *prev_norm, float *norm_out,
                                 const int *arch_slots, const int (*anc)[16],
                                 int k, int *out_ids, float *out_vals) {
    int H = g_qwen.H, I = g_qwen.I, V = g_qwen.V;
    int q_m = g_qwen.nh * g_qwen.hd * 2, kv_m = g_qwen.nkv * g_qwen.hd, attn_m = g_qwen.nh * g_qwen.hd;
    tq_layer_t *m = &g_qwen.mtp_layer;
    int pos = pos_base + d;
    // device control data: tokens[16] | gather idx | pos | anc[16][16] (anc rows by
    // WAVE-LOCAL node index; entries are archive slots)
    int hctl[16 + 16 + 16 + 16 * 16 + 16];
    for (int i = 0; i < nw; i++) hctl[i] = tokens[i];
    for (int i = 0; i < nw; i++) hctl[16 + i] = hsrc_idx[i] >= 0 ? hsrc_idx[i] : 0;
    for (int i = 0; i < nw; i++) hctl[32 + i] = pos;
    for (int i = 0; i < nw; i++)
        for (int dd = 0; dd < 16; dd++) hctl[48 + i * 16 + dd] = anc[i][dd];
    for (int i = 0; i < nw; i++) hctl[48 + 16 * 16 + i] = arch_slots[i];
    cudaMemcpyAsync(g_mtpb_dev, hctl, sizeof(hctl), cudaMemcpyHostToDevice, g_qwen.stream);
    int *d_tok = g_mtpb_dev, *d_gidx = g_mtpb_dev + 16, *d_pos = g_mtpb_dev + 32, *d_anc = g_mtpb_dev + 48;
    int *d_slots = g_mtpb_dev + 48 + 16 * 16;

    // fc_in[n] = [ norm(emb(tok_n)) | norm(hidden_n) ]
    k_tq_embed_lookup_nk<<<dim3((H + 255) / 256, nw), 256, 0, g_qwen.stream>>>(
        g_mtpb_x, g_qwen.d_embed, d_tok, H);
    k_tq_qwen_rmsnorm_b_out<<<dim3(8, nw), 1024, 0, g_qwen.stream>>>(
        g_mtpb_fc_in, 2 * H, 0, g_mtpb_x, g_qwen.d_mtp_pre_fc_norm_emb, H, g_qwen.eps);
    if (d == 0) {
        k_tq_qwen_rmsnorm_b_out<<<dim3(8, 1), 1024, 0, g_qwen.stream>>>(
            g_mtpb_fc_in, 2 * H, H, g_qwen.d_mtp_root_hidden, g_qwen.d_mtp_pre_fc_norm_hidden, H, g_qwen.eps);
    } else {
        k_tq_gather_rows_nk<<<dim3((H + 255) / 256, nw), 256, 0, g_qwen.stream>>>(
            g_mtpb_x, prev_norm, d_gidx, H);
        k_tq_qwen_rmsnorm_b_out<<<dim3(8, nw), 1024, 0, g_qwen.stream>>>(
            g_mtpb_fc_in, 2 * H, H, g_mtpb_x, g_qwen.d_mtp_pre_fc_norm_hidden, H, g_qwen.eps);
    }
    // x = fc(fc_in)
    if (mtp_wave_gemv(0, &g_qwen.mtp_fc, g_mtpb_fc_in, 2 * H, g_mtpb_x, nw, 1) != 0) return -61;
    // decoder layer: input_ln -> q/k/v -> tree attention -> o + resid
    k_tq_qwen_rmsnorm_b<<<dim3(8, nw), 1024, 0, g_qwen.stream>>>(
        g_qwen.d_spec_norm, g_mtpb_x, m->d_input_ln, H, g_qwen.eps);
    if (mtp_wave_gemv(1, &g_qwen.mtp_q_proj, g_qwen.d_spec_norm, H, g_qwen.d_spec_qkv, nw, 1) != 0) return -61;
    if (mtp_wave_gemv(2, &g_qwen.mtp_k_proj, g_qwen.d_spec_norm, H, g_qwen.d_spec_kbuf, nw, 0) != 0) return -61;
    if (mtp_wave_gemv(3, &g_qwen.mtp_v_proj, g_qwen.d_spec_norm, H, g_qwen.d_spec_z, nw, 0) != 0) return -61;
    // k/v (knorm+rope) -> archive slots; q attends trunk cache + ancestor archive
    k_tq_spec_attn_kv_prep<<<dim3(g_qwen.nkv, nw), g_qwen.hd, 0, g_qwen.stream>>>(
        g_mtpb_karch, g_mtpb_varch, g_qwen.d_spec_kbuf, g_qwen.d_spec_z,
        m->d_k_norm, d_pos, g_qwen.nkv, g_qwen.hd, kv_m, g_qwen.eps, g_qwen.rope_theta, d_slots,
        NULL, NULL, NULL, NULL, NULL, NULL, g_qwen.kv_q4);
    // kv_prep wrote arch rows by WAVE index; attention reads slots via anc rows --
    // remap: this wave's node i sits at arch row i, ancestors at their stored rows.
    int wave_mma = attn_mma_active(pos_base - 1);
    if (wave_mma) {
        launch_attn_tree_mma(g_qwen.d_spec_core, g_qwen.d_spec_qkv, m->d_q_norm,
            m->d_k_cache, m->d_v_cache,
            m->d_k_cache8, m->d_k_scale, m->d_v_cache8, m->d_v_scale,
            m->d_k_cache4, m->d_k_q4s,
            g_mtpb_karch, g_mtpb_varch, d_pos, d_anc, nw, q_m, attn_m,
            pos_base - 1, pos_base - 1, wave_mma);
    } else {
    if (ensure_attn_scores() != 0) return -97;
    k_tq_spec_attn_tree<<<dim3(g_qwen.nh, nw), g_qwen.hd, 0, g_qwen.stream>>>(
        g_qwen.d_spec_core, g_qwen.d_spec_qkv, m->d_q_norm,
        m->d_k_cache, m->d_v_cache,
        m->d_k_cache8, m->d_k_scale, m->d_v_cache8, m->d_v_scale,
        m->d_k_cache4, m->d_k_q4s,
        g_mtpb_karch, g_mtpb_varch,
        d_pos, d_anc, g_qwen.nh, g_qwen.nkv, g_qwen.hd, kv_m, q_m, attn_m,
        g_qwen.eps, g_qwen.rope_theta, pos_base - 1,
        g_qwen.d_attn_scores, g_qwen.max_seq);
    }
    if (mtp_wave_gemv(4, &g_qwen.mtp_o_proj, g_qwen.d_spec_core, attn_m, g_qwen.d_spec_lo, nw, 1) != 0) return -61;
    k_tq_add_vec_b<<<dim3((H + 255) / 256, nw), 256, 0, g_qwen.stream>>>(
        g_qwen.d_spec_resid, g_mtpb_x, g_qwen.d_spec_lo, H);
    // MLP
    k_tq_qwen_rmsnorm_b<<<dim3(8, nw), 1024, 0, g_qwen.stream>>>(
        g_qwen.d_spec_post, g_qwen.d_spec_resid, m->d_post_ln, H, g_qwen.eps);
    if (mtp_wave_gemv(5, &g_qwen.mtp_gate, g_qwen.d_spec_post, H, g_qwen.d_spec_gate, nw, 1) != 0) return -61;
    if (mtp_wave_gemv(6, &g_qwen.mtp_up, g_qwen.d_spec_post, H, g_qwen.d_spec_up, nw, 0) != 0) return -61;
    k_tq_silu_mul_b<<<dim3((I + 255) / 256, nw), 256, 0, g_qwen.stream>>>(
        g_qwen.d_spec_mlp, g_qwen.d_spec_gate, g_qwen.d_spec_up, I);
    if (mtp_wave_gemv(7, &g_qwen.mtp_down, g_qwen.d_spec_mlp, I, g_qwen.d_spec_lo, nw, 1) != 0) return -61;
    k_tq_add_vec_b<<<dim3((H + 255) / 256, nw), 256, 0, g_qwen.stream>>>(
        g_mtpb_x, g_qwen.d_spec_resid, g_qwen.d_spec_lo, H);
    // final mtp.norm -> lm_head (one quantized weight read for the whole wave)
    k_tq_qwen_rmsnorm_b<<<dim3(8, nw), 1024, 0, g_qwen.stream>>>(
        norm_out, g_mtpb_x, g_qwen.d_mtp_norm, H, g_qwen.eps);
    spec_pack(norm_out, H, nw);
    // draft shortlist: prefix rows are contiguous in the row-tile-major layout
    // (a_base = a + tile_m*Kt*words), so a shallow copy with clamped M/Mt works
    tq_qmma_weight_t head = (g_qwen.lm_head_fp4_ready && mtp_head_fp4_enabled())
                                ? g_qwen.lm_head_fp4 : g_qwen.lm_head;
    int vcap = mtp_vocab_cap();
    if (vcap > 0 && vcap < head.M) { head.M = vcap; head.Mt = vcap / 16; }
    int Vd = head.M;                 // logits stride/extent for this wave
    int ret = spec_gemv(&head, g_qwen.d_spec_logits, nw);
    if (ret != 0) return ret;
    // per-node top-k (argmax + mask, k rounds each). TQ_DRAFT_SAMPLE=1 (default
    // OFF) samples candidates without replacement via Gumbel-top-k instead.
    // Measured at temp=1.0: accept 2.26 -> 2.01, i.e. WORSE -- our sequential
    // acceptance uses deterministic proposals (accept child at p_tgt/(1-removed)),
    // for which greedy top-k is already the optimal candidate set. The vLLM
    // #41044 gain lives in their q-corrected rejection regime and does not
    // transfer. Kept env-gated for future p/q-corrected experiments.
    int draft_sample = 0;
    {
        static int ds = -1;
        if (ds < 0) {
            const char *e = getenv("TQ_DRAFT_SAMPLE");
            ds = e ? atoi(e) : 0;
        }
        draft_sample = ds && (spec_temp() > 0.0f);
    }
    float *gscr = NULL;
    if (draft_sample) {
        if (ensure_float_buffer(&g_qwen.d_debug_logits, &g_qwen.debug_logits_count, Vd,
                                "d_debug_logits") != 0)
            return -61;
        gscr = g_qwen.d_debug_logits;
    }
    for (int i = 0; i < nw; i++) {
        float *logits = g_qwen.d_spec_logits + (size_t)i * Vd;
        if (draft_sample) {
            uint32_t tag = 0x40000000u | ((uint32_t)d << 16) | (uint32_t)i;
            k_tq_gumbel_scratch<<<(Vd + 255) / 256, 256, 0, g_qwen.stream>>>(
                gscr, logits, 1.0f / spec_temp(), spec_seed(), g_spec_round_ctr, tag, Vd);
        }
        for (int j = 0; j < k; j++) {
            k_tq_argmax_stage1<<<TQ_ARGMAX_BLOCKS, 256, 0, g_qwen.stream>>>(
                g_qwen.d_spec_argv, g_qwen.d_spec_argi, draft_sample ? gscr : logits, Vd);
            k_tq_argmax_stage2<<<1, 1024, 0, g_qwen.stream>>>(
                g_qwen.d_spec_argv + TQ_ARGMAX_BLOCKS, g_qwen.d_spec_argi + TQ_ARGMAX_BLOCKS,
                g_qwen.d_spec_argv, g_qwen.d_spec_argi, TQ_ARGMAX_BLOCKS);
            if (draft_sample) {
                k_tq_topk_record_and_mask_sampled<<<1, 1, 0, g_qwen.stream>>>(
                    logits, gscr, g_qwen.d_mtp_topk_vals, g_qwen.d_mtp_topk_ids,
                    g_qwen.d_spec_argv + TQ_ARGMAX_BLOCKS, g_qwen.d_spec_argi + TQ_ARGMAX_BLOCKS, j);
            } else {
                k_tq_topk_record_and_mask<<<1, 1, 0, g_qwen.stream>>>(
                    logits, g_qwen.d_mtp_topk_vals, g_qwen.d_mtp_topk_ids,
                    g_qwen.d_spec_argv + TQ_ARGMAX_BLOCKS, g_qwen.d_spec_argi + TQ_ARGMAX_BLOCKS, j);
            }
        }
        cudaMemcpyAsync(out_ids + (size_t)i * k, g_qwen.d_mtp_topk_ids, (size_t)k * sizeof(int),
                        cudaMemcpyDeviceToHost, g_qwen.stream);
        cudaMemcpyAsync(out_vals + (size_t)i * k, g_qwen.d_mtp_topk_vals, (size_t)k * sizeof(float),
                        cudaMemcpyDeviceToHost, g_qwen.stream);
    }
    cudaError_t err = cudaStreamSynchronize(g_qwen.stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "mtp batch wave failed: %s\n", cudaGetErrorString(err));
        return -60;
    }
    return 0;
}

static int copy_mtp_topk(int *topk_ids, float *topk_vals, int k) {
    cudaError_t err = cudaStreamSynchronize(g_qwen.stream);
    if (err != cudaSuccess) { fprintf(stderr, "MTP sync failed: %s\n", cudaGetErrorString(err)); return -20; }
    int n = g_qwen.mtp_topk_count < k ? g_qwen.mtp_topk_count : k;
    if (n < 0) n = 0;
    if (topk_ids && n > 0) cudaMemcpy(topk_ids, g_qwen.d_mtp_topk_ids, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost);
    if (topk_vals && n > 0) cudaMemcpy(topk_vals, g_qwen.d_mtp_topk_vals, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost);
    int top1 = -1;
    if (g_qwen.mtp_topk_count > 0) cudaMemcpy(&top1, g_qwen.d_mtp_topk_ids, sizeof(int), cudaMemcpyDeviceToHost);
    return top1;
}

// Copy an MTP intermediate to host for stage-by-stage validation:
//   0 = fc_in (concat[norm(emb),norm(hidden)], 2H)   1 = fc output / layer input (H)
//   2 = decoder-layer output (H)                      3 = post mtp.norm (H)
extern "C" int qwn_copy_mtp_debug(int which, float *out, int max_count) {
    if (!out) return -1;
    const float *src = NULL;
    int count = g_qwen.H;
    switch (which) {
        case 0: src = g_qwen.d_mtp_fc_in; count = 2 * g_qwen.H; break;
        case 1: src = g_qwen.d_mtp_x; break;
        case 2: src = g_qwen.d_debug_layer_out; break;
        case 3: src = g_qwen.d_mtp_normout; break;
        case 4: src = g_qwen.d_debug_x; break;   // pre-final-norm trunk hidden (MTP root input)
        default: return -2;
    }
    if (!src) return -3;
    int n = count < max_count ? count : max_count;
    if (cudaStreamSynchronize(g_qwen.stream) != cudaSuccess) return -4;
    cudaMemcpy(out, src, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost);
    return n;
}

// Copy the last MTP forward's full logits to host (validation: lets the smoke test
// compare engine-vs-reference MTP logit vectors directly, not just argmax).
extern "C" int qwn_copy_last_mtp_logits(float *out, int max_count) {
    if (!g_qwen.d_mtp_logits || !out) return -1;
    int n = g_qwen.V < max_count ? g_qwen.V : max_count;
    cudaError_t err = cudaStreamSynchronize(g_qwen.stream);
    if (err != cudaSuccess) return -2;
    cudaMemcpy(out, g_qwen.d_mtp_logits, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost);
    return n;
}

// Public MTP step: uses the engine's current PRE-final-norm hidden (d_debug_x from
// the preceding qwn_decode). Fills topk_ids/topk_vals (k entries) and returns top-1.
extern "C" int qwn_mtp_step(int token_id, int pos, int *topk_ids, float *topk_vals, int k) {
    if (!g_qwen.d_debug_x) {
        fprintf(stderr, "qwn_mtp_step: no current hidden (call qwn_decode first)\n");
        return -30;
    }
    int ret = run_mtp_forward(g_qwen.d_debug_x, token_id, pos, k);
    if (ret != 0) return ret;
    return copy_mtp_topk(topk_ids, topk_vals, k);
}

// Validation entry: inject an external PRE-final-norm hidden (e.g. the PyTorch
// reference's BF16 hidden) so the MTP forward is comparable apples-to-apples.
extern "C" int qwn_mtp_step_hidden(const float *hidden, int token_id, int pos,
                                   int *topk_ids, float *topk_vals, int k) {
    if (!hidden) return -30;
    if (!g_qwen.has_mtp_section) return -1;
    if (ensure_float_buffer(&g_qwen.d_mtp_hidden_in, &g_qwen.mtp_hidden_in_count, g_qwen.H, "d_mtp_hidden_in") != 0) return -31;
    cudaError_t err = cudaMemcpyAsync(g_qwen.d_mtp_hidden_in, hidden, (size_t)g_qwen.H * sizeof(float),
                                      cudaMemcpyHostToDevice, g_qwen.stream);
    if (err != cudaSuccess) return -32;
    int ret = run_mtp_forward(g_qwen.d_mtp_hidden_in, token_id, pos, k);
    if (ret != 0) return ret;
    return copy_mtp_topk(topk_ids, topk_vals, k);
}

// ========================= MTP covering tree (Step 2) ========================
// See docs/MTP_SPECDEC_PLAN.md Step 2. These entry points are ADDITIVE: they are
// invoked only by tools/mtp_tree_smoke.py, never by the default decode path (which
// stays byte-identical). They reuse the validated Step-1 MTP forward
// (run_mtp_forward) over the single MTP full-attention KV cache.
//
// KEY IDEA (per-branch KV forking without physical duplication): a tree path of
// depth d occupies MTP cache slots pos..pos+d; expanding a node at depth d runs ONE
// MTP forward at slot pos+d, which writes that slot's K/V (from the node's token +
// its parent's post-norm hidden) and attends CAUSALLY over slots 0..pos+d -- the
// shared trunk (0..pos-1, prebuilt by qwn_mtp_advance) plus this path's prefix. A
// depth-first walk makes the fork implicit: each expansion overwrites only its own
// depth's slot, siblings reuse the unchanged parent prefix, and stale deeper slots
// are never read (causal mask attends only to 0..slot). The MTP layer is ONE layer
// so this is cheap (unlike the 64-layer forking-DeltaNet verify of Step 3).

static int ensure_mtp_tree_buffers(void) {
    if (!g_qwen.d_mtp_root_hidden) {
        if (cudaMalloc(&g_qwen.d_mtp_root_hidden, (size_t)g_qwen.H * sizeof(float)) != cudaSuccess) return -1;
        g_qwen.device_bytes += (size_t)g_qwen.H * sizeof(float);
    }
    for (int d = 0; d < TQ_MTP_MAX_DEPTH; d++) {
        if (!g_qwen.d_mtp_post[d]) {
            if (cudaMalloc(&g_qwen.d_mtp_post[d], (size_t)g_qwen.H * sizeof(float)) != cudaSuccess) return -2;
            g_qwen.device_bytes += (size_t)g_qwen.H * sizeof(float);
        }
    }
    return 0;
}

// Cheap MTP "advance": write the MTP KV cache at slot `pos` for the committed token
// using the engine's current pre-final-norm hidden (d_debug_x from qwn_decode), but
// WITHOUT the mtp.norm / lm_head / top-k. Used to build the shared causal trunk
// (MTP slots 0..pos-1) before rooting a tree at pos -- exactly the per-position MTP
// state the validated on-policy path (tools/mtp_engine_smoke.py) builds, minus the
// lm_head cost the trunk does not need.
extern "C" int qwn_mtp_advance(int token_id, int pos) {
    if (!g_qwen.has_mtp_section) return -1;
    if (g_qwen.mtp_exl3) return -40;
    if (!g_qwen.d_debug_x) return -30;
    if (token_id < 0 || token_id >= g_qwen.V) return -2;
    if (pos < 0 || pos >= g_qwen.max_seq) return -3;
    int ret = ensure_mtp_buffers();
    if (ret != 0) return ret;
    k_tq_embed_lookup<<<(g_qwen.H + 255) / 256, 256, 0, g_qwen.stream>>>(
        g_qwen.d_mtp_e, g_qwen.d_embed, token_id, g_qwen.H);
    k_tq_qwen_rmsnorm<<<8, 1024, 0, g_qwen.stream>>>(
        g_qwen.d_mtp_fc_in, g_qwen.d_mtp_e, g_qwen.d_mtp_pre_fc_norm_emb, g_qwen.H, g_qwen.eps);
    k_tq_qwen_rmsnorm<<<8, 1024, 0, g_qwen.stream>>>(
        g_qwen.d_mtp_fc_in + g_qwen.H, g_qwen.d_debug_x, g_qwen.d_mtp_pre_fc_norm_hidden, g_qwen.H, g_qwen.eps);
    ret = run_mtp_bf16_gemv(&g_qwen.mtp_fc, g_qwen.d_mtp_fc_in, g_qwen.d_mtp_x);
    if (ret != 0) return -11;
    // trunk advance only needs the KV slot; skip attention/o_proj/MLP (~723 MB)
    ret = mtp_fast_advance_enabled() ? run_mtp_kv_only(pos) : run_mtp_full_layer(pos);
    if (ret != 0) return -12;
    cudaError_t err = cudaStreamSynchronize(g_qwen.stream);
    return err == cudaSuccess ? 0 : -13;
}

// MTP "advance" (write KV slot `pos`) from an arbitrary device hidden (no lm_head).
static int mtp_advance_with_hidden(const float *hidden, int token_id, int pos) {
    if (!g_qwen.has_mtp_section) return -1;
    if (g_qwen.mtp_exl3) return -40;
    if (token_id < 0 || token_id >= g_qwen.V) return -2;
    if (pos < 0 || pos >= g_qwen.max_seq) return -3;
    int ret = ensure_mtp_buffers();
    if (ret != 0) return ret;
    k_tq_embed_lookup<<<(g_qwen.H + 255) / 256, 256, 0, g_qwen.stream>>>(
        g_qwen.d_mtp_e, g_qwen.d_embed, token_id, g_qwen.H);
    k_tq_qwen_rmsnorm<<<8, 1024, 0, g_qwen.stream>>>(
        g_qwen.d_mtp_fc_in, g_qwen.d_mtp_e, g_qwen.d_mtp_pre_fc_norm_emb, g_qwen.H, g_qwen.eps);
    k_tq_qwen_rmsnorm<<<8, 1024, 0, g_qwen.stream>>>(
        g_qwen.d_mtp_fc_in + g_qwen.H, hidden, g_qwen.d_mtp_pre_fc_norm_hidden, g_qwen.H, g_qwen.eps);
    ret = run_mtp_bf16_gemv(&g_qwen.mtp_fc, g_qwen.d_mtp_fc_in, g_qwen.d_mtp_x);
    if (ret != 0) return -11;
    // trunk advance only needs the KV slot; skip attention/o_proj/MLP (~723 MB)
    ret = mtp_fast_advance_enabled() ? run_mtp_kv_only(pos) : run_mtp_full_layer(pos);
    return ret != 0 ? -12 : 0;
}

// Spec-loop handoff: set the MTP tree root hidden from spec-forward node `node_idx`'s
// pre-final-norm hidden (d_spec_h), so the next round's tree roots at the just-
// committed position without an extra dense decode.
extern "C" int qwn_spec_set_root_from(int node_idx) {
    if (!g_qwen.d_spec_h || !g_qwen.d_mtp_root_hidden) return -1;
    cudaMemcpyAsync(g_qwen.d_mtp_root_hidden, g_qwen.d_spec_h + (size_t)node_idx * g_qwen.H,
                    (size_t)g_qwen.H * sizeof(float), cudaMemcpyDeviceToDevice, g_qwen.stream);
    return cudaStreamSynchronize(g_qwen.stream) == cudaSuccess ? 0 : -2;
}

// Spec-loop handoff: extend the MTP trunk (KV slot `pos`) for a committed token,
// using spec-forward node `node_idx`'s pre-final-norm hidden as h_pos.
extern "C" int qwn_mtp_advance_from_spec(int node_idx, int token, int pos) {
    if (!g_qwen.d_spec_h) return -1;
    int ret = mtp_advance_with_hidden(g_qwen.d_spec_h + (size_t)node_idx * g_qwen.H, token, pos);
    if (ret != 0) return ret;
    return cudaStreamSynchronize(g_qwen.stream) == cudaSuccess ? 0 : -13;
}

// Snapshot the current pre-final-norm hidden (d_debug_x from the last qwn_decode)
// into a stable buffer so a tree/path can be rooted at it AFTER the engine advances
// its dense decode (free-run), which overwrites d_debug_x.
extern "C" int qwn_mtp_snapshot_root(void) {
    if (!g_qwen.has_mtp_section) return -1;
    if (!g_qwen.d_debug_x) return -30;
    int ret = ensure_mtp_tree_buffers();
    if (ret != 0) return ret;
    cudaError_t err = cudaMemcpyAsync(g_qwen.d_mtp_root_hidden, g_qwen.d_debug_x,
                                      (size_t)g_qwen.H * sizeof(float),
                                      cudaMemcpyDeviceToDevice, g_qwen.stream);
    if (err != cudaSuccess) return -31;
    err = cudaStreamSynchronize(g_qwen.stream);
    return err == cudaSuccess ? 0 : -32;
}

// Expand one MTP node: run the Step-1 MTP forward (hidden_in, token) at cache slot
// `slot`, stash the post-norm hidden in d_mtp_post[post_depth] (consumed by this
// node's children), and copy the top-k token ids/vals to host. One stream sync.
static int mtp_expand_node(const float *d_hidden, int token, int slot, int k,
                           int *out_ids, float *out_vals, int post_depth) {
    int ret = run_mtp_forward(d_hidden, token, slot, k);
    if (ret != 0) return ret;
    // Save POST-norm hidden BEFORE the next forward overwrites d_mtp_normout.
    // Stream-ordered, so it lands before any subsequent expansion reads it.
    cudaMemcpyAsync(g_qwen.d_mtp_post[post_depth], g_qwen.d_mtp_normout,
                    (size_t)g_qwen.H * sizeof(float), cudaMemcpyDeviceToDevice, g_qwen.stream);
    cudaError_t err = cudaStreamSynchronize(g_qwen.stream);
    if (err != cudaSuccess) return -40;
    int n = g_qwen.mtp_topk_count < k ? g_qwen.mtp_topk_count : k;
    if (n < 0) n = 0;
    if (out_ids && n > 0) cudaMemcpy(out_ids, g_qwen.d_mtp_topk_ids, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost);
    if (out_vals && n > 0) cudaMemcpy(out_vals, g_qwen.d_mtp_topk_vals, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost);
    return 0;
}

typedef struct {
    int pos;
    int depth;       // max tree depth
    int k;           // RECORD width: top-k children recorded at every expanded node
    float tau;       // beam threshold (logit gap): a child is EXPANDABLE iff it is
                     // the top-1 OR (top1_logit - child_logit) < tau. Flat (low-margin)
                     // nodes -> many expandable (branch wide); peaked (confident)
                     // nodes -> only the top-1 (narrow). This is the margin shaping.
    int max_nodes;   // hard cap on recorded nodes (memory-bound column budget)
    int seed;        // root (depth-0) token = the main-model free token tok_{pos+1}
    int *tokens;
    int *parents;    // parent node index, -1 = root (the seed/free token)
    int *depths;     // 1..depth
    float *margins;  // top1-top2 logit margin of the expansion that PRODUCED this node
    int n;
} mtp_tree_ctx_t;

// Replay a node's ancestor path (root..parent) to repopulate the MTP KV cache slots
// pos..pos+node_depth-1 and the POST-norm hidden stack, then expand the node itself
// at slot pos+node_depth, returning its top-k children to host. Best-first expansion
// visits nodes out of DFS order, so each expansion re-walks its path; the MTP layer
// is ONE cheap layer, so O(depth) replay forwards per expansion is affordable. This
// is the explicit per-branch KV fork: each path gets its own slots pos..pos+d written
// from its own tokens before the node is expanded.
static int mtp_replay_expand(mtp_tree_ctx_t *c, int node_index, int node_token,
                             int node_depth, int *out_ids, float *out_vals) {
    int anc[TQ_MTP_MAX_DEPTH];
    int cur = (node_index >= 0) ? c->parents[node_index] : -1;
    for (int dd = node_depth - 1; dd >= 1; dd--) {
        anc[dd] = c->tokens[cur];
        cur = c->parents[cur];
    }
    int tmp_ids[TQ_MTP_TOPK_MAX];
    float tmp_vals[TQ_MTP_TOPK_MAX];
    // depth 0 (root): feed the seed at slot pos -> POST[0] (shared trunk attended).
    int ret = mtp_expand_node(g_qwen.d_mtp_root_hidden, c->seed, c->pos, c->k,
                              (node_depth == 0) ? out_ids : tmp_ids,
                              (node_depth == 0) ? out_vals : tmp_vals, 0);
    if (ret != 0) return ret;
    for (int dd = 1; dd <= node_depth - 1; dd++) {
        ret = mtp_expand_node(g_qwen.d_mtp_post[dd - 1], anc[dd], c->pos + dd, c->k,
                              tmp_ids, tmp_vals, dd);
        if (ret != 0) return ret;
    }
    if (node_depth >= 1) {
        ret = mtp_expand_node(g_qwen.d_mtp_post[node_depth - 1], node_token,
                              c->pos + node_depth, c->k, out_ids, out_vals, node_depth);
    }
    return ret;
}

// Build a margin-shaped covering tree rooted at the snapshot hidden + seed token.
// COVERING: every expanded node records its full top-k children (so the dense path
// is covered to top-k breadth at each level). SPARSE: only EXPAND (descend into)
// nodes worth it -- best-first by cumulative path logit, gated by the beam threshold
// tau, until the node budget is hit. Fills out_* (each sized >= max_nodes) and
// returns node count N (>=0), or <0 on error. The root (seed/free token) is implicit:
// depth-1 nodes have parent index -1.
extern "C" int qwn_mtp_tree_build(int seed_token, int pos, int depth, int k, float tau,
                                  int max_nodes, int *out_tokens, int *out_parents,
                                  int *out_depths, float *out_margins) {
    if (!g_qwen.has_mtp_section) return -1;
    if (!g_qwen.d_mtp_root_hidden) return -30;   // call qwn_mtp_snapshot_root first
    if (seed_token < 0 || seed_token >= g_qwen.V) return -2;
    if (depth < 1) depth = 1;
    if (depth > TQ_MTP_MAX_DEPTH) depth = TQ_MTP_MAX_DEPTH;
    if (k < 1) k = 1;
    if (k > TQ_MTP_TOPK_MAX) k = TQ_MTP_TOPK_MAX;
    if (pos < 0 || pos + depth > g_qwen.max_seq) return -3;
    if (max_nodes < 1) max_nodes = 1;
    int ret = ensure_mtp_tree_buffers();
    if (ret != 0) return ret;

    mtp_tree_ctx_t c;
    c.pos = pos; c.depth = depth; c.k = k; c.tau = tau; c.max_nodes = max_nodes;
    c.seed = seed_token;
    c.tokens = out_tokens; c.parents = out_parents; c.depths = out_depths;
    c.margins = out_margins; c.n = 0;

    // Batched depth-wave builder (default; TQ_MTP_TREE_BATCH=0 falls back to the
    // sequential best-first replay path). One batched MTP pass per depth instead
    // of O(depth) full forwards per expansion.
    {
        static int batch_en = -1;
        if (batch_en < 0) {
            const char *envb = getenv("TQ_MTP_TREE_BATCH");
            batch_en = envb ? atoi(envb) : 1;
        }
        // Wave batching pays when trees are deep/wide (one MTP+lm_head read per
        // WAVE instead of per expansion): 16-node deep trees 14.2 -> 4.9 ms. For
        // shallow narrow trees (max_nodes <= 8) the sequential replay is cheaper.
        int batch_quant = mtp_quant_enabled() && g_qwen.lm_head.e2m3;
        if (batch_en && k <= 16 && (max_nodes > 8 || batch_quant) && max_nodes <= 256 &&
            spec_fused_pack_enabled()) {
            ret = ensure_mtp_batch_buffers();
            if (ret != 0) return ret;
            ret = ensure_spec_buffers();   // tree build reuses the spec scratch
            if (ret != 0) return ret;
            if (mtp_quant_enabled() && g_qwen.lm_head.e2m3)
                ensure_mtp_quant();   // best-effort: BF16 fallback if packing fails
            if (mtp_head_fp4_enabled() && g_qwen.lm_head.e2m3)
                ensure_lmhead_fp4();  // best-effort: E2M3 head fallback
            float score[TQ_SPEC_MAX_N * TQ_MTP_TOPK_MAX];
            char eligible[TQ_SPEC_MAX_N * TQ_MTP_TOPK_MAX];
            int child_wavepos[TQ_SPEC_MAX_N * TQ_MTP_TOPK_MAX];   // parent's wave-local idx
            int wave_tok[16], wave_hsrc[16], wave_arch[16], wave_node[16];
            int wave_anc[16][16];
            int ids[16 * TQ_MTP_TOPK_MAX];
            float vals[16 * TQ_MTP_TOPK_MAX];
            int n_arch = 0;   // archive slots used (total expansions, hard cap 16)

            // wave 0: the root
            int nw = 1;
            wave_tok[0] = seed_token; wave_hsrc[0] = -1; wave_node[0] = -1;
            wave_arch[0] = n_arch++;
            for (int dd = 0; dd < 16; dd++) wave_anc[0][dd] = 0;
            wave_anc[0][0] = wave_arch[0];
            for (int d = 0; d < depth && nw > 0 && c.n < c.max_nodes; d++) {
                ret = mtp_batch_expand_wave(nw, d, pos, wave_tok, wave_hsrc,
                                            g_mtpb_norm[(d + 1) & 1], g_mtpb_norm[d & 1],
                                            wave_arch, wave_anc, c.k, ids, vals);
                if (ret != 0) return ret;
                // record children (same margin/score/eligibility rules as sequential)
                int rec_lo = c.n;
                for (int i = 0; i < nw && c.n < c.max_nodes; i++) {
                    const int *iv = ids + (size_t)i * c.k;
                    const float *vv = vals + (size_t)i * c.k;
                    float _m = (c.k >= 2) ? (vv[0] - vv[1]) : 3.0e38f;
                    float base_score = (wave_node[i] >= 0) ? score[wave_node[i]] : 0.0f;
                    for (int j = 0; j < c.k && c.n < c.max_nodes; j++) {
                        int ci = c.n++;
                        c.tokens[ci] = iv[j];
                        c.parents[ci] = wave_node[i];
                        c.depths[ci] = d + 1;
                        c.margins[ci] = _m;
                        score[ci] = base_score + (vv[j] - vv[0]);
                        eligible[ci] = (j == 0 || (vv[0] - vv[j]) < c.tau) ? 1 : 0;
                        child_wavepos[ci] = i;
                    }
                }
                // next wave: eligible depth-(d+1) children, best score first
                int cand[TQ_SPEC_MAX_N * TQ_MTP_TOPK_MAX], ncand = 0;
                for (int ci = rec_lo; ci < c.n; ci++)
                    if (eligible[ci] && c.depths[ci] < depth) cand[ncand++] = ci;
                for (int a = 0; a < ncand; a++)
                    for (int b = a + 1; b < ncand; b++)
                        if (score[cand[b]] > score[cand[a]]) { int t = cand[a]; cand[a] = cand[b]; cand[b] = t; }
                int prev_arch[16];
                int prev_anc[16][16];
                for (int i = 0; i < nw; i++) {
                    prev_arch[i] = wave_arch[i];
                    for (int dd = 0; dd < 16; dd++) prev_anc[i][dd] = wave_anc[i][dd];
                }
                int nw2 = 0;
                for (int a = 0; a < ncand && nw2 < 16 && n_arch < 16 && c.n + 1 <= c.max_nodes; a++) {
                    int ci = cand[a];
                    if (c.n >= c.max_nodes) break;
                    wave_tok[nw2] = c.tokens[ci];
                    wave_node[nw2] = ci;
                    wave_hsrc[nw2] = child_wavepos[ci];
                    wave_arch[nw2] = n_arch++;
                    int p = child_wavepos[ci];
                    for (int dd = 0; dd < 16; dd++) wave_anc[nw2][dd] = prev_anc[p][dd];
                    wave_anc[nw2][d + 1] = wave_arch[nw2];
                    nw2++;
                }
                (void)prev_arch;
                nw = nw2;
            }
            return c.n;
        }
    }

    // Host scratch: per-node cumulative path score (sum of logits) + flags.
    float *score = (float *)malloc((size_t)max_nodes * sizeof(float));
    char *expanded = (char *)calloc((size_t)max_nodes, 1);
    char *eligible = (char *)calloc((size_t)max_nodes, 1);
    if (!score || !expanded || !eligible) {
        free(score); free(expanded); free(eligible);
        return -50;
    }
    int ids[TQ_MTP_TOPK_MAX];
    float vals[TQ_MTP_TOPK_MAX];

    // Record the children of an expansion (margin/score/eligibility bookkeeping).
    // (inlined twice below -- once for the root, once per popped frontier node.)
    #define TQ_TREE_RECORD_CHILDREN(parent_idx, child_depth, base_score)              \
        do {                                                                          \
            float _m = (c.k >= 2) ? (vals[0] - vals[1]) : 3.0e38f;                    \
            for (int j = 0; j < c.k && c.n < c.max_nodes; j++) {                      \
                int _ci = c.n++;                                                      \
                c.tokens[_ci] = ids[j];                                               \
                c.parents[_ci] = (parent_idx);                                        \
                c.depths[_ci] = (child_depth);                                        \
                c.margins[_ci] = _m;                                                  \
                score[_ci] = (base_score) + (vals[j] - vals[0]);                      \
                expanded[_ci] = 0;                                                    \
                eligible[_ci] = (j == 0 || (vals[0] - vals[j]) < c.tau) ? 1 : 0;      \
            }                                                                         \
        } while (0)

    // Expand the root (depth 0) -> depth-1 children. Trunk slots 0..pos-1 are assumed
    // prebuilt (qwn_mtp_advance); the root writes slot pos.
    ret = mtp_replay_expand(&c, -1, seed_token, 0, ids, vals);
    if (ret != 0) { free(score); free(expanded); free(eligible); return ret; }
    TQ_TREE_RECORD_CHILDREN(-1, 1, 0.0f);

    // Best-first expansion: repeatedly expand the highest-score eligible, unexpanded,
    // non-leaf node while ANY recording room remains (children are truncated to the
    // remaining slots). The old `n + k <= max_nodes` guard made k=8 trees stop at
    // depth 1 for max_nodes<=16 -- the accept-length was capped by tree shape, not
    // by model uncertainty.
    while (c.n < c.max_nodes) {
        int best = -1;
        float best_score = -3.0e38f;
        for (int i = 0; i < c.n; i++) {
            if (expanded[i] || !eligible[i]) continue;
            if (c.depths[i] >= c.depth) continue;      // leaf level -> nothing to expand
            if (score[i] > best_score) { best_score = score[i]; best = i; }
        }
        if (best < 0) break;
        expanded[best] = 1;
        ret = mtp_replay_expand(&c, best, c.tokens[best], c.depths[best], ids, vals);
        if (ret != 0) { free(score); free(expanded); free(eligible); return ret; }
        TQ_TREE_RECORD_CHILDREN(best, c.depths[best] + 1, score[best]);
    }

    #undef TQ_TREE_RECORD_CHILDREN
    free(score); free(expanded); free(eligible);
    return c.n;
}

// Teacher-forced linear MTP rollout along a GIVEN path of fed tokens, returning the
// top-k ids/vals at each step (out arrays sized n_fed*k, row-major per step). Step i
// feeds fed[i] at MTP slot pos+i, chaining the post-norm hidden (step 0 uses the
// snapshot root hidden, step i>0 uses d_mtp_post[i-1]). Feeding the dense-greedy
// path measures the covering tree's per-depth coverage upper bound (the verify
// descends the true path so the prefix is correct -> "teacher-forced top-8").
// Returns the number of steps written, or <0 on error.
extern "C" int qwn_mtp_path_topk(const int *fed_tokens, int n_fed, int pos, int k,
                                 int *out_topk_ids, float *out_topk_vals) {
    if (!g_qwen.has_mtp_section) return -1;
    if (!g_qwen.d_mtp_root_hidden) return -30;
    if (!fed_tokens) return -2;
    if (n_fed < 1) return -2;
    if (n_fed > TQ_MTP_MAX_DEPTH) n_fed = TQ_MTP_MAX_DEPTH;
    if (k < 1) k = 1;
    if (k > TQ_MTP_TOPK_MAX) k = TQ_MTP_TOPK_MAX;
    if (pos < 0 || pos + n_fed > g_qwen.max_seq) return -3;
    int ret = ensure_mtp_tree_buffers();
    if (ret != 0) return ret;
    for (int i = 0; i < n_fed; i++) {
        if (fed_tokens[i] < 0 || fed_tokens[i] >= g_qwen.V) return -4;
        const float *hidden = (i == 0) ? g_qwen.d_mtp_root_hidden : g_qwen.d_mtp_post[i - 1];
        ret = mtp_expand_node(hidden, fed_tokens[i], pos + i, k,
                              out_topk_ids ? out_topk_ids + (size_t)i * k : NULL,
                              out_topk_vals ? out_topk_vals + (size_t)i * k : NULL, i);
        if (ret != 0) return ret;
    }
    return n_fed;
}

// ===================== Batched spec-decode verify forward (Step 5) ===========
// One dense FP6 forward over N<=8 verify nodes (a covering-tree path set), reading
// each weight ONCE via the k-split nk GEMV (run_nk_gemv) while attention/recurrence
// run per node. Nodes MUST be in DFS order (parent index < child index) so the full-
// attention shared KV cache holds each node's branch when it attends, and DeltaNet
// forks via per-node saved recurrent+conv state (child seeds from parent). The
// committed prefix (full-attn KV[0..base_pos], DeltaNet state) must be live on entry.
// Additive + env-gated (only reached through qwn_decode_spec); default decode intact.

static int spec_n_linear_layers(void) {
    int n = 0;
    for (int i = 0; i < g_qwen.L; i++)
        if (g_qwen.layer_types[i] == TQ_LAYER_LINEAR_ATTENTION) n++;
    return n;
}
static int spec_n_full_layers(void) { return g_qwen.L - spec_n_linear_layers(); }

static int spec_fused_delta_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("TQ_SPEC_FUSED_DELTA");
        cached = env ? atoi(env) : 1;
    }
    return cached;
}

static int spec_lin_recur_floats(void) {
    return g_qwen.linear_num_value_heads * g_qwen.linear_key_head_dim * g_qwen.linear_value_head_dim;
}
static int spec_lin_conv_floats(void) {
    int lkd = g_qwen.linear_num_key_heads * g_qwen.linear_key_head_dim;
    int lvd = g_qwen.linear_num_value_heads * g_qwen.linear_value_head_dim;
    return (lkd * 2 + lvd) * g_qwen.linear_conv_kernel_dim;
}

static int ensure_spec_buffers(void) {
    int N = TQ_SPEC_MAX_N;
    if (g_qwen.spec_alloc_n >= N) return 0;
    int H = g_qwen.H, I = g_qwen.I, V = g_qwen.V;
    int q_m = g_qwen.nh * g_qwen.hd * 2, kv_m = g_qwen.nkv * g_qwen.hd, attn_m = g_qwen.nh * g_qwen.hd;
    int dim = g_qwen.linear_value_head_dim;
    int heads = g_qwen.linear_num_value_heads, key_heads = g_qwen.linear_num_key_heads;
    int key_dim = key_heads * dim, value_dim = heads * dim;
    int conv_dim = key_dim * 2 + value_dim;
    int proj_max = q_m > conv_dim ? q_m : conv_dim;
    int zd_max = value_dim > kv_m ? value_dim : kv_m;
    int core_max = value_dim > attn_m ? value_dim : attn_m;
    int kmax = H > I ? H : I;
    int mmax = I; if (conv_dim > mmax) mmax = conv_dim; if (q_m > mmax) mmax = q_m; if (H > mmax) mmax = H;
    #define TQ_SPEC_ALLOC(p, count) do { if (cudaMalloc(&g_qwen.p, (size_t)(count) * sizeof(float)) != cudaSuccess) return -1; g_qwen.device_bytes += (size_t)(count)*sizeof(float); } while (0)
    TQ_SPEC_ALLOC(d_spec_h, (size_t)N * H);   TQ_SPEC_ALLOC(d_spec_norm, (size_t)N * H);
    TQ_SPEC_ALLOC(d_spec_resid, (size_t)N * H); TQ_SPEC_ALLOC(d_spec_post, (size_t)N * H);
    TQ_SPEC_ALLOC(d_spec_lo, (size_t)N * H);
    TQ_SPEC_ALLOC(d_spec_qkv, (size_t)N * proj_max); TQ_SPEC_ALLOC(d_spec_kbuf, (size_t)N * kv_m);
    TQ_SPEC_ALLOC(d_spec_z, (size_t)N * zd_max); TQ_SPEC_ALLOC(d_spec_bp, (size_t)N * heads);
    TQ_SPEC_ALLOC(d_spec_ap, (size_t)N * heads); TQ_SPEC_ALLOC(d_spec_core, (size_t)N * core_max);
    TQ_SPEC_ALLOC(d_spec_gate, (size_t)N * I); TQ_SPEC_ALLOC(d_spec_up, (size_t)N * I);
    TQ_SPEC_ALLOC(d_spec_mlp, (size_t)N * I); TQ_SPEC_ALLOC(d_spec_logits, (size_t)N * V);
    // per-LAYER dnprep archive (not per-launch scratch): the commit-time state
    // replay needs each layer's kn/v/beta/decay for the accepted path. 20 MB
    // buys skipping ~225 MB/round of mid-chain state-slab writes.
    {
        int n_lin0 = 0;
        for (int i = 0; i < g_qwen.L; i++)
            if (g_qwen.layer_types[i] == TQ_LAYER_LINEAR_ATTENTION) n_lin0++;
        if (n_lin0 < 1) n_lin0 = 1;
        TQ_SPEC_ALLOC(d_spec_dnprep,
                      (size_t)n_lin0 * N * (key_heads * 2 * dim + value_dim + heads * 2));
    }
    if (cudaMalloc(&g_qwen.d_live_recur_ptrs, 64 * sizeof(float *)) != cudaSuccess) return -1;
    if (cudaMalloc(&g_qwen.d_spec_replay, (size_t)(TQ_SPEC_MAX_N + 2) * sizeof(int)) != cudaSuccess) return -1;
    g_qwen.live_recur_ptrs_ready = 0;
    TQ_SPEC_ALLOC(d_spec_bscale, (size_t)(kmax / 128 + 2));
    TQ_SPEC_ALLOC(d_spec_part, (size_t)16 * N * mmax);   // headroom for forced k-splits up to 8
    int n_lin = spec_n_linear_layers(), n_full = spec_n_full_layers();
    // per-(layer,node) DeltaNet + per-(full-layer,node) KV archive: the verify forward
    // saves every node's per-layer state so the commit advances to the accepted leaf
    // (and re-places the path's attn KV) WITHOUT a second forward (Step-5 1-read commit).
    TQ_SPEC_ALLOC(d_spec_recur, (size_t)n_lin * N * spec_lin_recur_floats() /
                                (spec_recur_bf16() ? 2 : 1));
    TQ_SPEC_ALLOC(d_spec_conv, (size_t)n_lin * N * spec_lin_conv_floats());
    TQ_SPEC_ALLOC(d_spec_k_arch, (size_t)n_full * N * kv_m);
    TQ_SPEC_ALLOC(d_spec_v_arch, (size_t)n_full * N * kv_m);
    TQ_SPEC_ALLOC(d_spec_argv, (size_t)(TQ_ARGMAX_BLOCKS + 1));
    #undef TQ_SPEC_ALLOC
    // Zero the KV archives once: the MMA attention path streams whole 16-slot
    // tiles and multiplies never-written slots by p=0 -- malloc garbage could be
    // NaN/Inf and 0*NaN would poison the tile. (Written rows are always finite.)
    cudaMemset(g_qwen.d_spec_k_arch, 0, (size_t)n_full * N * kv_m * sizeof(float));
    cudaMemset(g_qwen.d_spec_v_arch, 0, (size_t)n_full * N * kv_m * sizeof(float));
    if (cudaMalloc(&g_qwen.d_spec_b, (size_t)TQ_SPEC_MAX_GROUPS * (kmax / 32 + 1) * 256) != cudaSuccess) return -1;
    if (cudaMalloc(&g_qwen.d_spec_argi, (size_t)(TQ_ARGMAX_BLOCKS + 1) * sizeof(int)) != cudaSuccess) return -1;
    if (cudaMalloc(&g_qwen.d_spec_group_ids, (size_t)N * sizeof(int)) != cudaSuccess) return -1;
    if (cudaMalloc(&g_qwen.d_spec_parent, (size_t)N * sizeof(int)) != cudaSuccess) return -1;
    if (cudaMalloc(&g_qwen.d_spec_depth_offsets, (size_t)(TQ_SPEC_MAX_N + 1) * sizeof(int)) != cudaSuccess) return -1;
    if (cudaMalloc(&g_qwen.d_spec_depth_done, (size_t)2 * sizeof(int)) != cudaSuccess) return -1;
    if (cudaMalloc(&g_qwen.d_spec_anc, (size_t)N * TQ_SPEC_MAX_N * sizeof(int)) != cudaSuccess) return -1;
    if (cudaMalloc(&g_qwen.d_spec_segs,
                   (size_t)(TQ_SEGS_NSEGS_SLOT + 1) * sizeof(int)) != cudaSuccess) return -1;
    // [0..N) = per-column absmax, [N..2N) = prescale factors (fused pack writes both)
    if (cudaMalloc(&g_qwen.d_spec_nk_factors, (size_t)2 * N * sizeof(float)) != cudaSuccess) return -1;
    g_qwen.device_bytes += (size_t)(kmax / 32 + 1) * 256 + (size_t)(TQ_ARGMAX_BLOCKS + 1) * sizeof(int) + (size_t)2 * N * sizeof(int) + (size_t)N * sizeof(float) + (size_t)(TQ_SPEC_MAX_N + 3) * sizeof(int);
    g_qwen.spec_alloc_n = N;
    return 0;
}

// Fused pack/postcorrect path (default on): the prescale is applied inline in the
// quantize kernel (no x rewrite, one launch less per pack) and the post-GEMV
// correction is folded into the k-split reduce. Bit-identical to the unfused path.
static int spec_fused_pack_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("TQ_SPEC_FUSED_PACK");
        cached = env ? atoi(env) : 1;
    }
    return cached;
}

// Pack N column activations [N x K] (col n at n*K) into the shared nk B-fragment.
// When N>1, pre-normalizes columns to have the same global absmax before quantizing
// (factors stored in d_spec_nk_factors[N..2N) for post-GEMV correction in spec_gemv).
static void spec_pack(const float *in_nk, int K, int N) {
    if (N > 1 && spec_fused_pack_enabled()) {
        k_tq_nk_col_absmax<<<N, 256, 0, g_qwen.stream>>>(
            in_nk, g_qwen.d_spec_nk_factors, K, N);
        k_tq_quantize_bfrag_pow2_fused_nk_f<<<(K + 127) / 128, 128, 0, g_qwen.stream>>>(
            g_qwen.d_spec_b, g_qwen.d_spec_bscale, in_nk, K, N,
            g_qwen.d_spec_nk_factors, g_qwen.d_spec_nk_factors + TQ_SPEC_MAX_N);
        return;
    }
    if (N > 1) {
        k_tq_nk_col_absmax<<<N, 256, 0, g_qwen.stream>>>(
            in_nk, g_qwen.d_spec_nk_factors, K, N);
        k_tq_nk_prescale<<<N, 256, 0, g_qwen.stream>>>(
            (float *)in_nk, g_qwen.d_spec_nk_factors, K, N);
    }
    k_tq_quantize_bfrag_pow2_fused_nk<<<(K + 127) / 128, 128, 0, g_qwen.stream>>>(
        g_qwen.d_spec_b, g_qwen.d_spec_bscale, in_nk, K, N);
}

// Projection over the packed B-fragment (call spec_pack first) into out [N x w->M].
// When N>1, applies post-correction (divide by prescale factors) so the output
// matches the un-prescaled GEMV result.
static int spec_gemv(const tq_qmma_weight_t *w, float *out, int N) {
    if (!can_use_qmma_sf_weight(w) || w->row_major) return -77;   // v1: nk needs col-major SF
    if (N > 8 && !(w->e2m3 || w->e2m1)) return -76;   // column group 2: e2m3/e2m1 paths only
    int ks = qmma_sf_k_split_auto(w->Mt, w->Kt);
    int fused = N > 1 && spec_fused_pack_enabled();
    const float *factors = fused ? g_qwen.d_spec_nk_factors + TQ_SPEC_MAX_N
                                 : g_qwen.d_spec_nk_factors;
    int split = ks > 1 && (w->Kt % ks) == 0 && !getenv("TQ_SPEC_NO_PC_FOLD");
    int ret = run_nk_gemv(out, g_qwen.d_spec_part, w, g_qwen.d_spec_b, g_qwen.d_spec_bscale,
                          N, ks, (fused && split) ? factors : NULL);
    if (ret != 0) return ret;
    if (N > 1 && !(fused && split)) {
        int total = N * w->M;
        k_tq_nk_postcorrect<<<(total + 255) / 256, 256, 0, g_qwen.stream>>>(
            out, factors, w->M, N);
    }
    return 0;
}

// Per-node DeltaNet recurrence core. Replicates the EXACT dispatch qwn_decode uses
// (hs / par / simple) so the batched forward is bit-identical to the per-token path
// (the simple kernel alone differs in reduction order -> ~1e-3 drift over 48 layers).
static void spec_linear_core(tq_layer_t *l, float *core, const float *qkv, const float *z,
                             const float *bp, const float *ap) {
    int heads = g_qwen.linear_num_value_heads, key_heads = g_qwen.linear_num_key_heads;
    int dim = g_qwen.linear_value_head_dim;
    int lc_g = linear_core_par_g();
    int lc_hs = linear_core_hs();
    bool lc_par_ok = lc_g > 0 && dim > 0 && dim <= 128 &&
                     (dim & (dim - 1)) == 0 && dim * lc_g <= 1024;
    bool lc_hs_ok = lc_par_ok && lc_hs >= 2 && (dim % lc_hs) == 0 && lc_g >= lc_hs &&
                    ((lc_g == 8 && (lc_hs == 2 || lc_hs == 4 || lc_hs == 8)) ||
                     (lc_g == 4 && (lc_hs == 2 || lc_hs == 4)));
    if (lc_hs_ok) {
        size_t shA = (size_t)3 * dim * sizeof(float), shB = (size_t)dim * sizeof(float);
        int blkA = (dim / lc_hs) * lc_g; dim3 gridA(heads * lc_hs);
        #define SPEC_CORE_HS(GG, HH)                                                    \
            k_tq_linear_decode_core_hs<GG, HH><<<gridA, blkA, shA, g_qwen.stream>>>(     \
                core, l->d_linear_recurrent_state, qkv, bp, ap,                         \
                l->d_linear_A_log, l->d_linear_dt_bias, heads, key_heads, dim)
        if (lc_g == 8 && lc_hs == 8) SPEC_CORE_HS(8, 8);
        else if (lc_g == 8 && lc_hs == 4) SPEC_CORE_HS(8, 4);
        else if (lc_g == 8 && lc_hs == 2) SPEC_CORE_HS(8, 2);
        else if (lc_g == 4 && lc_hs == 4) SPEC_CORE_HS(4, 4);
        else SPEC_CORE_HS(4, 2);
        #undef SPEC_CORE_HS
        k_tq_linear_decode_gated_norm<<<heads, dim, shB, g_qwen.stream>>>(
            core, z, l->d_linear_norm, heads, dim, g_qwen.eps);
    } else if (lc_par_ok) {
        size_t shb = (size_t)3 * dim * sizeof(float);
        if (lc_g == 8)
            k_tq_linear_decode_core_gated_par<8><<<heads, dim * 8, shb, g_qwen.stream>>>(
                core, l->d_linear_recurrent_state, qkv, z, bp, ap,
                l->d_linear_A_log, l->d_linear_dt_bias, l->d_linear_norm, heads, key_heads, dim, g_qwen.eps);
        else if (lc_g == 4)
            k_tq_linear_decode_core_gated_par<4><<<heads, dim * 4, shb, g_qwen.stream>>>(
                core, l->d_linear_recurrent_state, qkv, z, bp, ap,
                l->d_linear_A_log, l->d_linear_dt_bias, l->d_linear_norm, heads, key_heads, dim, g_qwen.eps);
        else
            k_tq_linear_decode_core_gated_par<2><<<heads, dim * 2, shb, g_qwen.stream>>>(
                core, l->d_linear_recurrent_state, qkv, z, bp, ap,
                l->d_linear_A_log, l->d_linear_dt_bias, l->d_linear_norm, heads, key_heads, dim, g_qwen.eps);
    } else {
        k_tq_linear_decode_core_gated<<<heads, dim, 0, g_qwen.stream>>>(
            core, l->d_linear_recurrent_state, qkv, z, bp, ap,
            l->d_linear_A_log, l->d_linear_dt_bias, l->d_linear_norm, heads, key_heads, dim, g_qwen.eps);
    }
}

// Batched DeltaNet core+gated-norm over `gc` sibling nodes (Step-5 verify). Same hs
// dispatch as spec_linear_core but with the node dimension batched (grid.y=gc, node =
// group_dev[.]), operating in place on the per-(layer,node) archive (recur_base/core_base
// + node*stride). Bit-identical per node. Returns 0 on the hs path, -1 otherwise (only
// the hs path is batched; default lc_g=8/lc_hs=4 hits it).
static int spec_linear_core_b(tq_layer_t *l, float *core_base, float *recur_base,
                              const float *qkv_base, const float *z_base,
                              const float *bp_base, const float *ap_base,
                              const int *group_dev, int gc,
                              int recur_f, int conv_dim, int value_dim) {
    int heads = g_qwen.linear_num_value_heads, key_heads = g_qwen.linear_num_key_heads;
    int dim = g_qwen.linear_value_head_dim;
    int lc_g = linear_core_par_g();
    int lc_hs = linear_core_hs();
    bool lc_par_ok = lc_g > 0 && dim > 0 && dim <= 128 && (dim & (dim - 1)) == 0 && dim * lc_g <= 1024;
    bool lc_hs_ok = lc_par_ok && lc_hs >= 2 && (dim % lc_hs) == 0 && lc_g >= lc_hs &&
                    ((lc_g == 8 && (lc_hs == 2 || lc_hs == 4 || lc_hs == 8)) ||
                     (lc_g == 4 && (lc_hs == 2 || lc_hs == 4)));
    if (!lc_hs_ok) return -1;
    size_t shA = (size_t)3 * dim * sizeof(float), shB = (size_t)dim * sizeof(float);
    int blkA = (dim / lc_hs) * lc_g;
    dim3 gridA(heads * lc_hs, gc);
    #define SPEC_CORE_HS_B(GG, HH)                                                        \
        k_tq_linear_decode_core_hs_b<GG, HH><<<gridA, blkA, shA, g_qwen.stream>>>(         \
            core_base, recur_base, qkv_base, bp_base, ap_base,                            \
            l->d_linear_A_log, l->d_linear_dt_bias, heads, key_heads, dim,                \
            group_dev, recur_f, conv_dim, value_dim)
    if (lc_g == 8 && lc_hs == 8) SPEC_CORE_HS_B(8, 8);
    else if (lc_g == 8 && lc_hs == 4) SPEC_CORE_HS_B(8, 4);
    else if (lc_g == 8 && lc_hs == 2) SPEC_CORE_HS_B(8, 2);
    else if (lc_g == 4 && lc_hs == 4) SPEC_CORE_HS_B(4, 4);
    else SPEC_CORE_HS_B(4, 2);
    #undef SPEC_CORE_HS_B
    dim3 gridN(heads, gc);
    k_tq_linear_decode_gated_norm_b<<<gridN, dim, shB, g_qwen.stream>>>(
        core_base, z_base, l->d_linear_norm, heads, dim, g_qwen.eps, group_dev, value_dim);
    return 0;
}

// Fused-tree DeltaNet core launch: all N nodes in one kernel, with global atomic
// depth synchronisation. Grid: (heads*HS, N). Bit-identical to the per-depth loop.
static int spec_linear_core_fused_tree(tq_layer_t *l, float *core_base, float *recur_base,
                                       const float *qkv_base,
                                       const float *bp_base, const float *ap_base,
                                       const int *group_dev, const int *parent_dev, int N,
                                       int recur_f, int conv_dim, int value_dim,
                                       const float *committed_recur,
                                       const int *depth_offsets, int n_depths,
                                       int *depth_done) {
    int heads = g_qwen.linear_num_value_heads, key_heads = g_qwen.linear_num_key_heads;
    int dim = g_qwen.linear_value_head_dim;
    int lc_g = linear_core_par_g();
    int lc_hs = linear_core_hs();
    bool lc_par_ok = lc_g > 0 && dim > 0 && dim <= 128 && (dim & (dim - 1)) == 0 && dim * lc_g <= 1024;
    bool lc_hs_ok = lc_par_ok && lc_hs >= 2 && (dim % lc_hs) == 0 && lc_g >= lc_hs &&
                    ((lc_g == 8 && (lc_hs == 2 || lc_hs == 4 || lc_hs == 8)) ||
                     (lc_g == 4 && (lc_hs == 2 || lc_hs == 4)));
    if (!lc_hs_ok) return -1;
    size_t shA = (size_t)3 * dim * sizeof(float);
    int blkA = (dim / lc_hs) * lc_g;
    dim3 gridA(heads * lc_hs, N);
    #define SPEC_CORE_FUSED_TREE(GG, HH)                                                       \
        k_tq_linear_decode_core_hs_fused_tree<GG, HH><<<gridA, blkA, shA, g_qwen.stream>>>(    \
            core_base, recur_base, qkv_base, bp_base, ap_base,                                 \
            l->d_linear_A_log, l->d_linear_dt_bias, heads, key_heads, dim,                     \
            group_dev, parent_dev, recur_f, conv_dim, value_dim, committed_recur,               \
            depth_offsets, n_depths, depth_done)
    if (lc_g == 8 && lc_hs == 8) SPEC_CORE_FUSED_TREE(8, 8);
    else if (lc_g == 8 && lc_hs == 4) SPEC_CORE_FUSED_TREE(8, 4);
    else if (lc_g == 8 && lc_hs == 2) SPEC_CORE_FUSED_TREE(8, 2);
    else if (lc_g == 4 && lc_hs == 4) SPEC_CORE_FUSED_TREE(4, 4);
    else SPEC_CORE_FUSED_TREE(4, 2);
    #undef SPEC_CORE_FUSED_TREE
    return 0;
}

// Fully-fused DeltaNet launch: conv + core in ONE kernel for all N nodes, with
// global atomic depth sync. Saves one kernel launch per linear layer vs the
// separate conv-fused-tree + core-fused-tree pair. Returns 0 on success.
static int spec_linear_deltanet_fused(tq_layer_t *l, float *core_base, float *recur_base,
                                      float *conv_base, const float *qkv_base,
                                      const float *bp_base, const float *ap_base,
                                      const int *group_dev, const int *parent_dev, int N,
                                      int recur_f, int conv_dim, int value_dim, int conv_f,
                                      const float *committed_recur,
                                      const float *committed_conv,
                                      const int *depth_offsets, int n_depths,
                                      int *depth_done) {
    int heads = g_qwen.linear_num_value_heads, key_heads = g_qwen.linear_num_key_heads;
    int dim = g_qwen.linear_value_head_dim;
    int lc_g = linear_core_par_g();
    int lc_hs = linear_core_hs();
    bool lc_par_ok = lc_g > 0 && dim > 0 && dim <= 128 && (dim & (dim - 1)) == 0 && dim * lc_g <= 1024;
    bool lc_hs_ok = lc_par_ok && lc_hs >= 2 && (dim % lc_hs) == 0 && lc_g >= lc_hs &&
                    ((lc_g == 8 && (lc_hs == 2 || lc_hs == 4 || lc_hs == 8)) ||
                     (lc_g == 4 && (lc_hs == 2 || lc_hs == 4)));
    if (!lc_hs_ok) return -1;
    size_t shA = (size_t)3 * dim * sizeof(float);
    int blkA = (dim / lc_hs) * lc_g;
    dim3 gridA(heads * lc_hs, N);
    int ks = g_qwen.linear_conv_kernel_dim;
    #define SPEC_DELTANET_FUSED(GG, HH)                                                          \
        k_tq_deltanet_fused_depths<GG, HH><<<gridA, blkA, shA, g_qwen.stream>>>(                \
            core_base, recur_base, conv_base, qkv_base, bp_base, ap_base,                        \
            l->d_linear_A_log, l->d_linear_dt_bias, l->d_linear_conv1d,                          \
            heads, key_heads, dim, ks, group_dev, parent_dev,                                    \
            recur_f, conv_dim, value_dim, conv_f, committed_recur, committed_conv,               \
            depth_offsets, n_depths, depth_done)
    if (lc_g == 8 && lc_hs == 8) SPEC_DELTANET_FUSED(8, 8);
    else if (lc_g == 8 && lc_hs == 4) SPEC_DELTANET_FUSED(8, 4);
    else if (lc_g == 8 && lc_hs == 2) SPEC_DELTANET_FUSED(8, 2);
    else if (lc_g == 4 && lc_hs == 4) SPEC_DELTANET_FUSED(4, 4);
    else SPEC_DELTANET_FUSED(4, 2);
    #undef SPEC_DELTANET_FUSED
    return 0;
}

static int ensure_spec_graph_params(int N);
