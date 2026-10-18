// batched multi-client decode (the pre-paged batched path).
// Part of the forward_qwen.cu unity build: included in order, not a standalone header.

// ====================== Roadmap E: batched / multi-client DECODE ======================
// Serve N clients in parallel through ONE engine, NO speculative decoding. The wide FP6
// GEMM (k_tq_fp6_wide_gemm) reads each weight ONCE and multiplies it by N client-columns,
// so the projections (qkv/o/gate/up/down/lm_head) amortize the weight reads across clients
// -- the exact wide-prefill mechanic with N = clients instead of N = prefill tokens. Unlike
// wide prefill (one sequence, one KV cache, one DeltaNet state, consecutive positions), each
// client sits at its OWN position with its OWN KV history and its OWN DeltaNet recurrent+conv
// state. Default-off; the single-stream interactive path is untouched. KV store follows the
// engine: TQ_KV_Q4=1 -> Q4(K rotated int4 + Hadamard)+E4M3(V) (the @128k VRAM unlock), else fp32.
//
// Reuses: the g_wide_* per-column activation buffers, wide_quant_input / wide_proj / wide_mlp,
// the batched DeltaNet kernels (k_tq_linear_conv_update_b, spec_linear_core_b -- both
// bit-identical per node to the single-stream recurrence). The only new kernels are the
// batched attention pair below: the wide-attn kernels with a per-client KV-slab offset.

// Per-client KV write: identical to k_tq_wide_attn_kv_write, but column i is a distinct
// client, so its KV cache base is offset by i*kv_stride (kv_stride = bat_ctx*nkv*hd).
__global__ void k_tq_batched_attn_kv_write(
    float *k_cache, float *v_cache,
    const float *k_proj_base, const float *v_proj_base,
    const uint16_t *k_norm_w, const int *positions,
    int nkv, int hd, int kv_m, size_t kv_stride, float eps, float rope_theta) {
    __shared__ float partial[8];
    __shared__ float k_sum_shared;
    int kv_head = blockIdx.x;
    int i = blockIdx.y;          // client index
    int tid = threadIdx.x;
    if (kv_head >= nkv || tid >= hd) return;
    int pos = positions[i];
    const float *k_proj = k_proj_base + (size_t)i * kv_m;
    const float *v_proj = v_proj_base + (size_t)i * kv_m;
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
    float *kc = k_cache + (size_t)i * kv_stride;
    float *vc = v_cache + (size_t)i * kv_stride;
    kc[((size_t)pos * nkv + kv_head) * hd + tid] = kv;
    vc[((size_t)pos * nkv + kv_head) * hd + tid] = v_proj[kv_head * hd + tid];
}

// Per-client flash attention: identical to k_tq_wide_attn, but column i (a client) reads
// ONLY its own KV slab (base + i*kv_stride) and walks its own causal prefix [0..positions[i]].
__global__ void k_tq_batched_attn(
    float *out, const float *q_proj_base, const uint16_t *q_norm_w,
    const float *k_cache, const float *v_cache,
    const int *positions, int nh, int nkv, int hd, int q_m, int attn_m,
    size_t kv_stride, float eps, float rope_theta) {
    __shared__ float partial[8];
    __shared__ float q_sum_shared;
    __shared__ float qn[256];
    __shared__ float s_chunk[TQ_WIDE_ATTN_CHUNK];
    __shared__ float red[256];
    __shared__ float m_run, l_run;
    int head = blockIdx.x;
    int i = blockIdx.y;          // client index
    int tid = threadIdx.x;
    if (head >= nh || tid >= hd) return;
    int pos = positions[i];
    int group = nh / nkv;
    int kv_head = head / group;
    int rotary_dim = 64;
    const float *q_proj = q_proj_base + (size_t)i * q_m;
    float qv = q_proj[head * (2 * hd) + tid];
    float qsum = qv * qv;
    for (int offset = 16; offset > 0; offset >>= 1)
        qsum += __shfl_down_sync(0xffffffff, qsum, offset);
    if ((tid & 31) == 0) partial[tid >> 5] = qsum;
    __syncthreads();
    if (tid < 8) {
        float vtmp = partial[tid];
        for (int offset = 4; offset > 0; offset >>= 1) vtmp += __shfl_down_sync(0xff, vtmp, offset);
        if (tid == 0) q_sum_shared = vtmp;
    }
    __syncthreads();
    qv = qv * rsqrtf(q_sum_shared / (float)hd + eps) * (1.0f + tq_bf16_to_float(q_norm_w[tid]));
    if (tid < rotary_dim) {
        int idx = tid & 31;
        float freq = powf(rope_theta, -((float)(2 * idx) / (float)rotary_dim));
        float angle = (float)pos * freq;
        float c = cosf(angle);
        float s = sinf(angle);
        float q_pair = q_proj[head * (2 * hd) + ((tid < 32) ? tid + 32 : tid - 32)];
        q_pair = q_pair * rsqrtf(q_sum_shared / (float)hd + eps) * (1.0f + tq_bf16_to_float(q_norm_w[(tid < 32) ? tid + 32 : tid - 32]));
        float q_rot = (tid < 32) ? -q_pair : q_pair;
        qv = qv * c + q_rot * s;
    }
    qn[tid] = qv;
    if (tid == 0) { m_run = -3.402823466e38f; l_run = 0.0f; }
    __syncthreads();
    const float *kc = k_cache + (size_t)i * kv_stride + (size_t)kv_head * hd;
    const float *vc = v_cache + (size_t)i * kv_stride + (size_t)kv_head * hd;
    size_t cstride = (size_t)nkv * hd;
    int warp = tid >> 5, lane = tid & 31;
    float scale = rsqrtf((float)hd);
    float acc = 0.0f;
    int total = pos + 1;
    for (int c0 = 0; c0 < total; c0 += TQ_WIDE_ATTN_CHUNK) {
        int clen = total - c0;
        if (clen > TQ_WIDE_ATTN_CHUNK) clen = TQ_WIDE_ATTN_CHUNK;
        for (int j = warp; j < clen; j += 8) {
            const float *krow = kc + (size_t)(c0 + j) * cstride;
            float dot = 0.0f;
            #pragma unroll
            for (int e = 0; e < 8; e++) { int d = lane + 32 * e; dot += qn[d] * krow[d]; }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) dot += __shfl_down_sync(0xffffffffu, dot, off);
            if (lane == 0) s_chunk[j] = dot * scale;
        }
        __syncthreads();
        float m = -3.402823466e38f;
        for (int j = tid; j < clen; j += 256) if (s_chunk[j] > m) m = s_chunk[j];
        red[tid] = m;
        __syncthreads();
        for (int s = 128; s > 0; s >>= 1) {
            if (tid < s && red[tid + s] > red[tid]) red[tid] = red[tid + s];
            __syncthreads();
        }
        float m_chunk = red[0];
        __syncthreads();
        float m_old = m_run;
        float m_new = fmaxf(m_old, m_chunk);
        float alpha = expf(m_old - m_new);
        float lsum = 0.0f;
        for (int j = tid; j < clen; j += 256) {
            float p = expf(s_chunk[j] - m_new);
            s_chunk[j] = p;
            lsum += p;
        }
        red[tid] = lsum;
        __syncthreads();
        for (int s = 128; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            __syncthreads();
        }
        float l_chunk = red[0];
        acc *= alpha;
        for (int j = 0; j < clen; j++)
            acc += s_chunk[j] * vc[(size_t)(c0 + j) * cstride + tid];
        if (tid == 0) { l_run = l_run * alpha + l_chunk; m_run = m_new; }
        __syncthreads();
    }
    float ratio = acc / l_run;
    float gate = q_proj[head * (2 * hd) + hd + tid];
    out[(size_t)i * attn_m + head * hd + tid] = ratio * (1.0f / (1.0f + expf(-gate)));
}

static int launch_batched_attn(float *out, const float *q_proj, const uint16_t *q_norm_w,
                               const float *k_proj, const float *v_proj, const uint16_t *k_norm_w,
                               float *k_cache_base, float *v_cache_base, const int *positions,
                               int N, size_t kv_stride, cudaStream_t st) {
    int nh = g_qwen.nh, nkv = g_qwen.nkv, hd = g_qwen.hd;
    int q_m = nh * hd * 2, kv_m = nkv * hd, attn_m = nh * hd;
    k_tq_batched_attn_kv_write<<<dim3(nkv, N), hd, 0, st>>>(
        k_cache_base, v_cache_base, k_proj, v_proj, k_norm_w, positions, nkv, hd, kv_m,
        kv_stride, g_qwen.eps, g_qwen.rope_theta);
    k_tq_batched_attn<<<dim3(nh, N), hd, 0, st>>>(
        out, q_proj, q_norm_w, k_cache_base, v_cache_base, positions, nh, nkv, hd, q_m, attn_m,
        kv_stride, g_qwen.eps, g_qwen.rope_theta);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

// ---- Q4(K)+E4M3(V) batched attention (the @128k VRAM unlock) ----
// Ports the production single-stream TQ_KV_Q4 scheme (k_tq_full_attn_decode q4 branch +
// tq_attn_tail_pp_q4) into the per-client batched layout: K rows are RMSNorm+RoPE'd, then
// FWHT-256 rotated and int4-quantized per 32-channel group ((code-zp)*scale); V is E4M3
// with a per-(pos,kv_head) block scale; the query is rotated once at read so q'.k' == q.k.
// Each column i is a distinct client; its Q4 slabs are offset by i*{k4,kq4s,v8,vscale}_stride.

__global__ void k_tq_batched_attn_kv_write_q4(
    uint8_t *k4_base, float *kq4s_base, uint8_t *v8_base, float *vscale_base,
    const float *k_proj_base, const float *v_proj_base,
    const uint16_t *k_norm_w, const int *positions,
    int nkv, int hd, int kv_m,
    size_t k4_stride, size_t kq4s_stride, size_t v8_stride, size_t vscale_stride,
    float eps, float rope_theta) {
    __shared__ float partial[8];
    __shared__ float k_sum_shared;
    __shared__ float gsz[16];
    __shared__ float rot[256];
    int kv_head = blockIdx.x;
    int i = blockIdx.y;          // client index
    int tid = threadIdx.x;
    if (kv_head >= nkv || tid >= hd) return;
    int pos = positions[i];
    const float *k_proj = k_proj_base + (size_t)i * kv_m;
    const float *v_proj = v_proj_base + (size_t)i * kv_m;
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
    // V: E4M3 + per-(pos,kv_head) block scale (into client i's slab).
    float vv = v_proj[kv_head * hd + tid];
    __syncthreads();
    float vscl = tq_kv_block_scale(vv, partial, &k_sum_shared);
    v8_base[(size_t)i * v8_stride + (size_t)(pos * nkv + kv_head) * hd + tid] = tq_float_to_e4m3(vv / vscl);
    if (tid == 0) vscale_base[(size_t)i * vscale_stride + (size_t)pos * nkv + kv_head] = vscl;
    // K: FWHT-256 rotate, then group-asymmetric int4 (2 codes/byte) + (scale,zp) groups.
    rot[tid] = kv;
    __syncthreads();
    tq_fwht256(rot, tid);
    float krot = rot[tid];
    __syncthreads();
    uint8_t code = tq_k_q4_quant(krot, tid, gsz);
    unsigned hi4 = __shfl_down_sync(0xffffffffu, (unsigned)code, 1);
    if ((tid & 1) == 0)
        k4_base[(size_t)i * k4_stride + (size_t)(pos * nkv + kv_head) * (hd >> 1) + (tid >> 1)] =
            (uint8_t)(code | ((hi4 & 15u) << 4));
    if (tid < 16)
        q4s_put(kq4s_base, (size_t)i * kq4s_stride + ((size_t)pos * nkv + kv_head) * 16 + tid, gsz[tid]);
}

__global__ void k_tq_batched_attn_q4(
    float *out, const float *q_proj_base, const uint16_t *q_norm_w,
    const uint8_t *k4_base, const float *kq4s_base, const uint8_t *v8_base, const float *vscale_base,
    const int *positions, int nh, int nkv, int hd, int q_m, int attn_m,
    size_t k4_stride, size_t kq4s_stride, size_t v8_stride, size_t vscale_stride,
    float eps, float rope_theta) {
    __shared__ float partial[8];
    __shared__ float q_sum_shared;
    __shared__ float qn[256];                       // rotated query, persists across chunks
    __shared__ float s_chunk[TQ_WIDE_ATTN_CHUNK];
    __shared__ float red[256];
    __shared__ float m_run, l_run;
    int head = blockIdx.x;
    int i = blockIdx.y;          // client index
    int tid = threadIdx.x;
    if (head >= nh || tid >= hd) return;
    int pos = positions[i];
    int group = nh / nkv;
    int kv_head = head / group;
    int rotary_dim = 64;
    const float *q_proj = q_proj_base + (size_t)i * q_m;
    float qv = q_proj[head * (2 * hd) + tid];
    float qsum = qv * qv;
    for (int offset = 16; offset > 0; offset >>= 1)
        qsum += __shfl_down_sync(0xffffffff, qsum, offset);
    if ((tid & 31) == 0) partial[tid >> 5] = qsum;
    __syncthreads();
    if (tid < 8) {
        float vtmp = partial[tid];
        for (int offset = 4; offset > 0; offset >>= 1) vtmp += __shfl_down_sync(0xff, vtmp, offset);
        if (tid == 0) q_sum_shared = vtmp;
    }
    __syncthreads();
    qv = qv * rsqrtf(q_sum_shared / (float)hd + eps) * (1.0f + tq_bf16_to_float(q_norm_w[tid]));
    if (tid < rotary_dim) {
        int idx = tid & 31;
        float freq = powf(rope_theta, -((float)(2 * idx) / (float)rotary_dim));
        float angle = (float)pos * freq;
        float c = cosf(angle);
        float s = sinf(angle);
        float q_pair = q_proj[head * (2 * hd) + ((tid < 32) ? tid + 32 : tid - 32)];
        q_pair = q_pair * rsqrtf(q_sum_shared / (float)hd + eps) * (1.0f + tq_bf16_to_float(q_norm_w[(tid < 32) ? tid + 32 : tid - 32]));
        float q_rot = (tid < 32) ? -q_pair : q_pair;
        qv = qv * c + q_rot * s;
    }
    qn[tid] = qv;
    __syncthreads();
    tq_fwht256(qn, tid);                             // rotated-space q matches rotated int4 K
    if (tid == 0) { m_run = -3.402823466e38f; l_run = 0.0f; }
    __syncthreads();
    // per-client + per-kv_head Q4 slab pointers
    const uint8_t *kc4 = k4_base + (size_t)i * k4_stride + (size_t)kv_head * (hd >> 1);
    const float *ks4 = q4s_adv(kq4s_base, (size_t)i * kq4s_stride + (size_t)kv_head * 16);
    const uint8_t *vc8 = v8_base + (size_t)i * v8_stride + (size_t)kv_head * hd;
    const float *vsc = vscale_base + (size_t)i * vscale_stride + kv_head;
    size_t cstride4 = (size_t)(nkv * hd) >> 1;       // int4 K row stride (bytes)
    size_t sstride16 = (size_t)nkv * 16;             // K scale row stride (elements)
    size_t vstride = (size_t)nkv * hd;               // E4M3 V row stride (bytes)
    int vsc_stride = nkv;                            // V scale per-pos stride
    int warp = tid >> 5, lane = tid & 31;
    float scale = rsqrtf((float)hd);
    float acc = 0.0f;
    int total = pos + 1;
    for (int c0 = 0; c0 < total; c0 += TQ_WIDE_ATTN_CHUNK) {
        int clen = total - c0;
        if (clen > TQ_WIDE_ATTN_CHUNK) clen = TQ_WIDE_ATTN_CHUNK;
        // (1) scores: one warp per row, int4-K dequant (group g == e), warp-reduced dot.
        for (int j = warp; j < clen; j += 8) {
            int t = c0 + j;
            const uint8_t *krow = kc4 + (size_t)t * cstride4;
            const float *ksz = q4s_adv(ks4, (size_t)t * sstride16);
            float dot = 0.0f;
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                int d = lane + 32 * e;
                uint8_t byte = krow[d >> 1];
                float code = (float)((d & 1) ? (byte >> 4) : (byte & 15));
                dot += qn[d] * ((code - q4s_at(ksz, 2 * e + 1)) * q4s_at(ksz, 2 * e));
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) dot += __shfl_down_sync(0xffffffffu, dot, off);
            if (lane == 0) s_chunk[j] = dot * scale;
        }
        __syncthreads();
        float m = -3.402823466e38f;
        for (int j = tid; j < clen; j += 256) if (s_chunk[j] > m) m = s_chunk[j];
        red[tid] = m;
        __syncthreads();
        for (int s = 128; s > 0; s >>= 1) {
            if (tid < s && red[tid + s] > red[tid]) red[tid] = red[tid + s];
            __syncthreads();
        }
        float m_chunk = red[0];
        __syncthreads();
        float m_old = m_run;
        float m_new = fmaxf(m_old, m_chunk);
        float alpha = expf(m_old - m_new);
        float lsum = 0.0f;
        for (int j = tid; j < clen; j += 256) {
            float p = expf(s_chunk[j] - m_new);
            s_chunk[j] = p;
            lsum += p;
        }
        red[tid] = lsum;
        __syncthreads();
        for (int s = 128; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            __syncthreads();
        }
        float l_chunk = red[0];
        // (2) accumulate this thread's output dim: E4M3 V dequant * prob.
        acc *= alpha;
        for (int j = 0; j < clen; j++) {
            int t = c0 + j;
            float vcode = tq_e4m3_dec_fast(vc8[(size_t)t * vstride + tid]);
            float vscl = vsc[(size_t)t * vsc_stride];
            acc += s_chunk[j] * vcode * vscl;
        }
        if (tid == 0) { l_run = l_run * alpha + l_chunk; m_run = m_new; }
        __syncthreads();
    }
    float ratio = acc / l_run;
    float gate = q_proj[head * (2 * hd) + hd + tid];
    out[(size_t)i * attn_m + head * hd + tid] = ratio * (1.0f / (1.0f + expf(-gate)));
}

static int launch_batched_attn_q4(float *out, const float *q_proj, const uint16_t *q_norm_w,
                                  const float *k_proj, const float *v_proj, const uint16_t *k_norm_w,
                                  uint8_t *k4_base, float *kq4s_base, uint8_t *v8_base, float *vscale_base,
                                  const int *positions, int N, int max_pos,
                                  size_t k4_stride, size_t kq4s_stride, size_t v8_stride, size_t vscale_stride,
                                  cudaStream_t st) {
    int nh = g_qwen.nh, nkv = g_qwen.nkv, hd = g_qwen.hd;
    int q_m = nh * hd * 2, kv_m = nkv * hd, attn_m = nh * hd;
    k_tq_batched_attn_kv_write_q4<<<dim3(nkv, N), hd, 0, st>>>(
        k4_base, kq4s_base, v8_base, vscale_base, k_proj, v_proj, k_norm_w, positions,
        nkv, hd, kv_m, k4_stride, kq4s_stride, v8_stride, vscale_stride, g_qwen.eps, g_qwen.rope_theta);
    // Wide-prefill (shared cache, strides==0) + MMA enabled -> tensor-core Q4 attention (compute-bound).
    // Per-client decode (strides>0) keeps the scalar memory-bound path.
    bool wide_shared = (k4_stride == 0 && kq4s_stride == 0 && v8_stride == 0 && vscale_stride == 0);
    if (wide_shared && wide_attn_mma_enabled()) {
        int qr = wide_attn_qrows();
        int nqt = (N + qr - 1) / qr;
        int S = wide_attn_split_S(max_pos);
        int part_f = 2 * qr + qr * 256;
        if (S > 1 && ensure_wide_attn_part((size_t)nh * nqt * S * part_f) != 0)
            S = 1;
        if (qr == 32)
            k_tq_wide_attn_mma<3, 2><<<dim3(nh, nqt, S), 256, 0, st>>>(
                out, q_proj, q_norm_w, NULL, NULL, k4_base, kq4s_base, v8_base, vscale_base, positions,
                0, NULL, 0, 0, 0, N, nh, nkv, hd, q_m, attn_m, g_qwen.eps, g_qwen.rope_theta,
                g_wide_attn_part);
        else
            k_tq_wide_attn_mma<3, 1><<<dim3(nh, nqt, S), 256, 0, st>>>(
                out, q_proj, q_norm_w, NULL, NULL, k4_base, kq4s_base, v8_base, vscale_base, positions,
                0, NULL, 0, 0, 0, N, nh, nkv, hd, q_m, attn_m, g_qwen.eps, g_qwen.rope_theta,
                g_wide_attn_part);
        if (S > 1)
            if (qr == 32)
                k_tq_wide_attn_mma_merge<2><<<dim3(nh, nqt), 256, 0, st>>>(
                    out, q_proj, N, nh, hd, q_m, attn_m, S, g_wide_attn_part);
            else
                k_tq_wide_attn_mma_merge<1><<<dim3(nh, nqt), 256, 0, st>>>(
                    out, q_proj, N, nh, hd, q_m, attn_m, S, g_wide_attn_part);
    } else {
        k_tq_batched_attn_q4<<<dim3(nh, N), hd, 0, st>>>(
            out, q_proj, q_norm_w, k4_base, kq4s_base, v8_base, vscale_base, positions,
            nh, nkv, hd, q_m, attn_m, k4_stride, kq4s_stride, v8_stride, vscale_stride,
            g_qwen.eps, g_qwen.rope_theta);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

// Per-client greedy argmax over [N x V] logits (one block/client, lowest-index tie-break).
__global__ void k_tq_batched_argmax(int *out_ids, const float *logits, int V, int N) {
    tq_pdl_enter();
    int c = blockIdx.x;
    if (c >= N) return;
    const float *row = logits + (size_t)c * V;
    int tid = threadIdx.x, bs = blockDim.x;
    float best = -3.402823466e38f; int besti = 0;
    for (int i = tid; i < V; i += bs) {
        float v = row[i];
        if (v > best) { best = v; besti = i; }
    }
    __shared__ float sv[256];
    __shared__ int si[256];
    sv[tid] = best; si[tid] = besti;
    __syncthreads();
    for (int s = bs >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            if (sv[tid + s] > sv[tid] || (sv[tid + s] == sv[tid] && si[tid + s] < si[tid])) {
                sv[tid] = sv[tid + s]; si[tid] = si[tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) out_ids[c] = si[0];
}

// Per-client persistent state (lives across decode steps). g_wide_* hold the per-step
// activation columns (reused). Full-attn layers: per-client fp32 KV slab. Linear layers:
// per-client DeltaNet recurrent + conv state. Indexed by layer; NULL for the other type.
static float *g_bat_k[TQ_MAX_LAYERS] = {0};       // fp32 KV mode (g_bat_q4==0)
static float *g_bat_v[TQ_MAX_LAYERS] = {0};
static uint8_t *g_bat_k4[TQ_MAX_LAYERS] = {0};    // Q4 mode: int4 rotated K (2 codes/byte)
static float *g_bat_kq4s[TQ_MAX_LAYERS] = {0};    // Q4 mode: K (scale,zp) groups (tagged fp16/fp32)
static uint8_t *g_bat_v8[TQ_MAX_LAYERS] = {0};    // Q4 mode: E4M3 V
static float *g_bat_vscale[TQ_MAX_LAYERS] = {0};  // Q4 mode: V block scale per (pos,kv_head)
static float *g_bat_recur[TQ_MAX_LAYERS] = {0};
static float *g_bat_conv[TQ_MAX_LAYERS] = {0};
static int   *g_bat_ids = NULL;        // device identity group-ids [N] = {0..N-1}
static float *g_bat_logits = NULL;     // [N x V] wide lm_head output
static int   *g_bat_argmax = NULL;     // [N] device per-client argmax
static int    g_bat_N = 0;             // allocated client count
static int    g_bat_ctx = 0;           // per-client context capacity (KV slab rows)
static int    g_bat_q4 = 0;            // 1 = Q4(K)+E4M3(V) KV, 0 = fp32 KV

// (scale,zp) element size for the Q4 K scale store: fp16 default, fp32 escape hatch.
static size_t batched_q4s_elem_bytes(void) { return g_qwen.kv_q4_fp32s ? sizeof(float) : sizeof(__half); }

static void batched_free_buffers(void) {
    for (int L = 0; L < TQ_MAX_LAYERS; L++) {
        if (g_bat_k[L]) { cudaFree(g_bat_k[L]); g_bat_k[L] = NULL; }
        if (g_bat_v[L]) { cudaFree(g_bat_v[L]); g_bat_v[L] = NULL; }
        if (g_bat_k4[L]) { cudaFree(g_bat_k4[L]); g_bat_k4[L] = NULL; }
        if (g_bat_kq4s[L]) { cudaFree(g_bat_kq4s[L]); g_bat_kq4s[L] = NULL; }
        if (g_bat_v8[L]) { cudaFree(g_bat_v8[L]); g_bat_v8[L] = NULL; }
        if (g_bat_vscale[L]) { cudaFree(g_bat_vscale[L]); g_bat_vscale[L] = NULL; }
        if (g_bat_recur[L]) { cudaFree(g_bat_recur[L]); g_bat_recur[L] = NULL; }
        if (g_bat_conv[L]) { cudaFree(g_bat_conv[L]); g_bat_conv[L] = NULL; }
    }
    if (g_bat_ids) { cudaFree(g_bat_ids); g_bat_ids = NULL; }
    if (g_bat_logits) { cudaFree(g_bat_logits); g_bat_logits = NULL; }
    if (g_bat_argmax) { cudaFree(g_bat_argmax); g_bat_argmax = NULL; }
    g_bat_N = 0; g_bat_ctx = 0; g_bat_q4 = 0;
}

static int batched_recur_floats(void) {
    return g_qwen.linear_num_value_heads * g_qwen.linear_key_head_dim * g_qwen.linear_value_head_dim;
}
static int batched_conv_floats(void) {
    int key_dim = g_qwen.linear_num_key_heads * g_qwen.linear_key_head_dim;
    int value_dim = g_qwen.linear_num_value_heads * g_qwen.linear_value_head_dim;
    int conv_dim = 2 * key_dim + value_dim;
    return conv_dim * g_qwen.linear_conv_kernel_dim;
}

static int ensure_batched_buffers(int N, int bat_ctx) {
    if (N < 1 || bat_ctx < 1) return -1;
    if (g_bat_N == N && g_bat_ctx == bat_ctx) return 0;   // already sized
    batched_free_buffers();
    int kv_m = g_qwen.nkv * g_qwen.hd;
    int nkv = g_qwen.nkv;
    int q4 = g_qwen.kv_q4 ? 1 : 0;
    size_t recur_f = (size_t)batched_recur_floats();
    size_t conv_f = (size_t)batched_conv_floats();
    for (int L = 0; L < g_qwen.L; L++) {
        if (g_qwen.layer_types[L] == TQ_LAYER_FULL_ATTENTION) {
            if (q4) {
                size_t k4b = (size_t)N * bat_ctx * (kv_m >> 1);                       // int4 K bytes
                size_t kq4sb = (size_t)N * bat_ctx * nkv * 16 * batched_q4s_elem_bytes(); // (scale,zp)
                size_t v8b = (size_t)N * bat_ctx * kv_m;                              // E4M3 V bytes
                size_t vscb = (size_t)N * bat_ctx * nkv * sizeof(float);              // V scales
                if (cudaMalloc(&g_bat_k4[L], k4b) != cudaSuccess) { batched_free_buffers(); return -2; }
                if (cudaMalloc(&g_bat_kq4s[L], kq4sb) != cudaSuccess) { batched_free_buffers(); return -2; }
                if (cudaMalloc(&g_bat_v8[L], v8b) != cudaSuccess) { batched_free_buffers(); return -2; }
                if (cudaMalloc(&g_bat_vscale[L], vscb) != cudaSuccess) { batched_free_buffers(); return -2; }
                cudaMemsetAsync(g_bat_k4[L], 0, k4b, g_qwen.stream);
                cudaMemsetAsync(g_bat_kq4s[L], 0, kq4sb, g_qwen.stream);
                cudaMemsetAsync(g_bat_v8[L], 0, v8b, g_qwen.stream);
                cudaMemsetAsync(g_bat_vscale[L], 0, vscb, g_qwen.stream);
            } else {
                size_t bytes = (size_t)N * bat_ctx * kv_m * sizeof(float);
                if (cudaMalloc(&g_bat_k[L], bytes) != cudaSuccess) { batched_free_buffers(); return -2; }
                if (cudaMalloc(&g_bat_v[L], bytes) != cudaSuccess) { batched_free_buffers(); return -2; }
                cudaMemsetAsync(g_bat_k[L], 0, bytes, g_qwen.stream);
                cudaMemsetAsync(g_bat_v[L], 0, bytes, g_qwen.stream);
            }
        } else if (g_qwen.layer_types[L] == TQ_LAYER_LINEAR_ATTENTION) {
            size_t rb = (size_t)N * recur_f * sizeof(float);
            size_t cb = (size_t)N * conv_f * sizeof(float);
            if (cudaMalloc(&g_bat_recur[L], rb) != cudaSuccess) { batched_free_buffers(); return -3; }
            if (cudaMalloc(&g_bat_conv[L], cb) != cudaSuccess) { batched_free_buffers(); return -3; }
            cudaMemsetAsync(g_bat_recur[L], 0, rb, g_qwen.stream);
            cudaMemsetAsync(g_bat_conv[L], 0, cb, g_qwen.stream);
        }
    }
    if (cudaMalloc(&g_bat_ids, (size_t)N * sizeof(int)) != cudaSuccess) { batched_free_buffers(); return -4; }
    int *hids = (int *)malloc((size_t)N * sizeof(int));
    if (!hids) { batched_free_buffers(); return -5; }
    for (int i = 0; i < N; i++) hids[i] = i;
    cudaMemcpyAsync(g_bat_ids, hids, (size_t)N * sizeof(int), cudaMemcpyHostToDevice, g_qwen.stream);
    if (cudaMalloc(&g_bat_logits, (size_t)N * g_qwen.V * sizeof(float)) != cudaSuccess) {
        free(hids); batched_free_buffers(); return -6;
    }
    if (cudaMalloc(&g_bat_argmax, (size_t)N * sizeof(int)) != cudaSuccess) {
        free(hids); batched_free_buffers(); return -7;
    }
    if (ensure_wide_prefill_buffers(N) != 0) { free(hids); batched_free_buffers(); return -8; }
    cudaStreamSynchronize(g_qwen.stream);
    free(hids);
    g_bat_N = N; g_bat_ctx = bat_ctx; g_bat_q4 = q4;
    return 0;
}

// Allocate per-client batched state for up to N clients, each with bat_ctx context. KV mode
// follows the engine: TQ_KV_Q4=1 -> Q4(K)+E4M3(V) slabs (the @128k VRAM unlock), else fp32.
// Pure-FP8 (kv_fp8 without kv_q4) is not ported -> rejected.
extern "C" int qwn_batched_init(int N, int bat_ctx) {
    if (!g_qwen.initialized) return -1;
    if (N < 1 || bat_ctx < 1) return -2;
    if (g_qwen.kv_fp8 && !g_qwen.kv_q4) return -3;
    if (bat_ctx > g_qwen.max_seq) return -4;
    return ensure_batched_buffers(N, bat_ctx);
}

extern "C" int qwn_batched_free(void) {
    batched_free_buffers();
    return 0;
}

// Snapshot the engine's CURRENT single-stream state into client slot c: copy each full-attn
// layer's KV rows [0..upto_pos] and each linear layer's DeltaNet recurrent+conv state. Lets a
// test prefill each client through the proven single-stream path and stash it into a batched
// slot. The next batched step feeds a token at position upto_pos+1.
extern "C" int qwn_batched_load_client(int c, int upto_pos) {
    if (!g_qwen.initialized) return -1;
    if (c < 0 || c >= g_bat_N) return -2;
    if (upto_pos < 0 || upto_pos + 1 >= g_bat_ctx) return -3;
    int kv_m = g_qwen.nkv * g_qwen.hd, nkv = g_qwen.nkv;
    size_t recur_f = (size_t)batched_recur_floats();
    size_t conv_f = (size_t)batched_conv_floats();
    size_t rows = (size_t)upto_pos + 1;
    size_t qe = batched_q4s_elem_bytes();
    for (int L = 0; L < g_qwen.L; L++) {
        tq_layer_t *l = &g_qwen.layers[L];
        if (g_qwen.layer_types[L] == TQ_LAYER_FULL_ATTENTION) {
            if (g_bat_q4) {
                if (!l->d_k_cache4 || !l->d_k_q4s || !l->d_v_cache8 || !l->d_v_scale) return -5;
                uint8_t *k4d = g_bat_k4[L] + (size_t)c * g_bat_ctx * (kv_m >> 1);
                char *kq4sd = (char *)g_bat_kq4s[L] + (size_t)c * g_bat_ctx * nkv * 16 * qe;
                uint8_t *v8d = g_bat_v8[L] + (size_t)c * g_bat_ctx * kv_m;
                float *vscd = g_bat_vscale[L] + (size_t)c * g_bat_ctx * nkv;
                cudaMemcpyAsync(k4d, l->d_k_cache4, rows * (kv_m >> 1), cudaMemcpyDeviceToDevice, g_qwen.stream);
                cudaMemcpyAsync(kq4sd, l->d_k_q4s, rows * nkv * 16 * qe, cudaMemcpyDeviceToDevice, g_qwen.stream);
                cudaMemcpyAsync(v8d, l->d_v_cache8, rows * kv_m, cudaMemcpyDeviceToDevice, g_qwen.stream);
                cudaMemcpyAsync(vscd, l->d_v_scale, rows * nkv * sizeof(float), cudaMemcpyDeviceToDevice, g_qwen.stream);
            } else {
                if (!l->d_k_cache || !l->d_v_cache) return -5;
                float *kd = g_bat_k[L] + (size_t)c * g_bat_ctx * kv_m;
                float *vd = g_bat_v[L] + (size_t)c * g_bat_ctx * kv_m;
                cudaMemcpyAsync(kd, l->d_k_cache, rows * kv_m * sizeof(float), cudaMemcpyDeviceToDevice, g_qwen.stream);
                cudaMemcpyAsync(vd, l->d_v_cache, rows * kv_m * sizeof(float), cudaMemcpyDeviceToDevice, g_qwen.stream);
            }
        } else if (g_qwen.layer_types[L] == TQ_LAYER_LINEAR_ATTENTION) {
            if (!l->d_linear_recurrent_state || !l->d_linear_conv_state) return -6;
            cudaMemcpyAsync(g_bat_recur[L] + (size_t)c * recur_f, l->d_linear_recurrent_state,
                            recur_f * sizeof(float), cudaMemcpyDeviceToDevice, g_qwen.stream);
            cudaMemcpyAsync(g_bat_conv[L] + (size_t)c * conv_f, l->d_linear_conv_state,
                            conv_f * sizeof(float), cudaMemcpyDeviceToDevice, g_qwen.stream);
        }
    }
    return cudaStreamSynchronize(g_qwen.stream) == cudaSuccess ? 0 : -7;
}

// One batched decode step: feed tokens[c] at positions[c] for every client in ONE pass over
// the weights (wide FP6 GEMM for all projections), batched attention (each client attends its
// own KV), batched DeltaNet (N independent states advance one token). Mirrors
// run_wide_chunk_forward, but N = clients each at their own pos / KV / DeltaNet state.
static int run_batched_decode_step_core(const int *tokens, const int *positions, int N) {
    int H = g_qwen.H;
    int attn_m = g_qwen.nh * g_qwen.hd;
    int kv_m = g_qwen.nkv * g_qwen.hd;
    int lkh = g_qwen.linear_num_key_heads, lvh = g_qwen.linear_num_value_heads, ld = g_qwen.linear_value_head_dim;
    int value_dim = lvh * ld, key_dim = lkh * ld, conv_dim = 2 * key_dim + value_dim;
    int recur_f = batched_recur_floats(), conv_f = batched_conv_floats();
    int nkv = g_qwen.nkv;
    size_t kv_stride = (size_t)g_bat_ctx * kv_m;                 // fp32 KV client stride
    size_t k4_stride = (size_t)g_bat_ctx * (kv_m >> 1);          // Q4: int4 K bytes/client
    size_t kq4s_stride = (size_t)g_bat_ctx * nkv * 16;           // Q4: K scale elems/client
    size_t v8_stride = (size_t)g_bat_ctx * kv_m;                 // Q4: E4M3 V bytes/client
    size_t vscale_stride = (size_t)g_bat_ctx * nkv;             // Q4: V scale floats/client
    int ret;
    // positions -> device (reuse g_wide_pos), embed N tokens into g_wide_h.
    cudaMemcpyAsync(g_wide_pos, positions, (size_t)N * sizeof(int), cudaMemcpyHostToDevice, g_qwen.stream);
    for (int i = 0; i < N; i++)
        k_tq_embed_lookup<<<(H + 255) / 256, 256, 0, g_qwen.stream>>>(
            g_wide_h + (size_t)i * H, g_qwen.d_embed, tokens[i], H);
    for (int L = 0; L < g_qwen.L; L++) {
        tq_layer_t *l = &g_qwen.layers[L];
        if (g_qwen.layer_types[L] == TQ_LAYER_FULL_ATTENTION) {
            if (!(tq_wide_fmt_ok(&l->q_proj) && tq_wide_fmt_ok(&l->o_proj) &&
                  tq_wide_fmt_ok(&l->mlp_gate) && tq_wide_fmt_ok(&l->mlp_down))) return -20;
            k_tq_qwen_rmsnorm_b<<<dim3(8, N), 1024, 0, g_qwen.stream>>>(
                g_wide_norm, g_wide_h, l->d_input_ln, H, g_qwen.eps);
            if ((ret = wide_quant_input(g_wide_norm, H, N)) != 0) return -21;
            if ((ret = wide_proj(&l->q_proj, g_wide_q, N)) != 0) return -21;
            if ((ret = wide_proj(&l->k_proj, g_wide_k, N)) != 0) return -22;
            if ((ret = wide_proj(&l->v_proj, g_wide_v, N)) != 0) return -23;
            if (g_bat_q4) {
                if (launch_batched_attn_q4(g_wide_core, g_wide_q, l->d_q_norm, g_wide_k, g_wide_v, l->d_k_norm,
                                           g_bat_k4[L], g_bat_kq4s[L], g_bat_v8[L], g_bat_vscale[L],
                                           g_wide_pos, N, -1, k4_stride, kq4s_stride, v8_stride, vscale_stride,
                                           g_qwen.stream) != 0) return -24;
            } else {
                if (launch_batched_attn(g_wide_core, g_wide_q, l->d_q_norm, g_wide_k, g_wide_v, l->d_k_norm,
                                        g_bat_k[L], g_bat_v[L], g_wide_pos, N, kv_stride, g_qwen.stream) != 0)
                    return -24;
            }
            if ((ret = wide_quant_input(g_wide_core, attn_m, N)) != 0) return -25;
            if ((ret = wide_proj(&l->o_proj, g_wide_o, N)) != 0) return -25;
            k_tq_add_vec<<<((size_t)N * H + 255) / 256, 256, 0, g_qwen.stream>>>(
                g_wide_resid, g_wide_h, g_wide_o, (size_t)N * H);
            if ((ret = wide_mlp(l, g_wide_resid, g_wide_h, N)) != 0) return -26;
        } else {   // linear-attention (gated DeltaNet)
            if (!(tq_wide_fmt_ok(&l->linear_in_qkv) && tq_wide_fmt_ok(&l->linear_in_z) &&
                  tq_wide_fmt_ok(&l->linear_in_b) && tq_wide_fmt_ok(&l->linear_in_a) &&
                  tq_wide_fmt_ok(&l->linear_out) && tq_wide_fmt_ok(&l->mlp_gate) &&
                  tq_wide_fmt_ok(&l->mlp_down))) return -30;
            k_tq_qwen_rmsnorm_b<<<dim3(8, N), 1024, 0, g_qwen.stream>>>(
                g_wide_norm, g_wide_h, l->d_input_ln, H, g_qwen.eps);
            if ((ret = wide_quant_input(g_wide_norm, H, N)) != 0) return -31;
            if ((ret = wide_proj(&l->linear_in_qkv, g_wide_qkv, N)) != 0) return -31;
            if ((ret = wide_proj(&l->linear_in_z, g_wide_z, N)) != 0) return -32;
            if ((ret = wide_proj(&l->linear_in_b, g_wide_b, N)) != 0) return -33;
            if ((ret = wide_proj(&l->linear_in_a, g_wide_a, N)) != 0) return -34;
            // batched conv update (N clients, each own conv state slab) in place on g_wide_qkv
            k_tq_linear_conv_update_b<<<dim3((conv_dim + 255) / 256, N), 256, 0, g_qwen.stream>>>(
                g_wide_qkv, g_bat_conv[L], g_wide_qkv, l->d_linear_conv1d, conv_dim,
                g_qwen.linear_conv_kernel_dim, g_bat_ids, conv_f);
            // batched DeltaNet core+gated-norm (N clients, each own recurrent state, one token)
            if (spec_linear_core_b(l, g_wide_lcore, g_bat_recur[L], g_wide_qkv, g_wide_z,
                                   g_wide_b, g_wide_a, g_bat_ids, N, recur_f, conv_dim, value_dim) != 0)
                return -35;
            if ((ret = wide_quant_input(g_wide_lcore, value_dim, N)) != 0) return -36;
            if ((ret = wide_proj(&l->linear_out, g_wide_o, N)) != 0) return -36;
            k_tq_add_vec<<<((size_t)N * H + 255) / 256, 256, 0, g_qwen.stream>>>(
                g_wide_resid, g_wide_h, g_wide_o, (size_t)N * H);
            if ((ret = wide_mlp(l, g_wide_resid, g_wide_h, N)) != 0) return -37;
        }
    }
    // final norm (wide) -> lm_head. Wide lm_head when it is e2m3-SF (one read for all N);
    // otherwise per-client fallback through the canonical single-stream lm_head argmax.
    k_tq_qwen_rmsnorm_b<<<dim3(8, N), 1024, 0, g_qwen.stream>>>(
        g_wide_norm, g_wide_h, g_qwen.d_norm, H, g_qwen.eps);
    int per_client_lm = getenv("TQ_BATCHED_LMHEAD_PERCLIENT") != NULL;
    if (!per_client_lm && !g_qwen.tie_word_embeddings &&
        (g_qwen.lm_head.exl3 ||
         (g_qwen.lm_head.e2m3 && !g_qwen.lm_head.e2m3_byte && !g_qwen.lm_head.word_major &&
          can_use_qmma_sf_weight(&g_qwen.lm_head) && !g_qwen.lm_head.row_major))) {
        if ((ret = wide_quant_input(g_wide_norm, H, N)) != 0) return -40;
        if ((ret = wide_proj(&g_qwen.lm_head, g_bat_logits, N)) != 0) return -40;
        k_tq_batched_argmax<<<N, 256, 0, g_qwen.stream>>>(g_bat_argmax, g_bat_logits, g_qwen.V, N);
    } else {
        for (int c = 0; c < N; c++) {
            cudaMemcpyAsync(g_qwen.d_debug_norm, g_wide_norm + (size_t)c * H,
                            (size_t)H * sizeof(float), cudaMemcpyDeviceToDevice, g_qwen.stream);
            if (run_lm_head_argmax_device() != 0) return -41;
            cudaMemcpyAsync(g_bat_argmax + c, g_qwen.d_debug_arg_ids + TQ_ARGMAX_BLOCKS,
                            sizeof(int), cudaMemcpyDeviceToDevice, g_qwen.stream);
        }
    }
    return cudaGetLastError() == cudaSuccess ? 0 : -42;
}

extern "C" int qwn_batched_decode_step(const int *tokens, const int *positions, int N, int *out_argmax) {
    if (!g_qwen.initialized) return -1;
    if (N < 1 || N > g_bat_N) return -2;
    if (g_qwen.kv_fp8 && !g_qwen.kv_q4) return -3;
    for (int i = 0; i < N; i++)
        if (positions[i] < 0 || positions[i] >= g_bat_ctx) return -4;
    int ret = run_batched_decode_step_core(tokens, positions, N);
    if (ret != 0) return ret;
    if (out_argmax)
        cudaMemcpyAsync(out_argmax, g_bat_argmax, (size_t)N * sizeof(int),
                        cudaMemcpyDeviceToHost, g_qwen.stream);
    return cudaStreamSynchronize(g_qwen.stream) == cudaSuccess ? 0 : -9;
}

// Diagnostic: copy client c's batched lm_head logits [V] to host (valid only after a
// decode_step that used the WIDE lm_head path; returns V on success). Lets the test
// quantify mismatches as near-ties (logit gap between two candidate tokens).
extern "C" int qwn_batched_logits(int c, float *out_host) {
    if (!g_qwen.initialized || !g_bat_logits) return -1;
    if (c < 0 || c >= g_bat_N) return -2;
    if (cudaMemcpy(out_host, g_bat_logits + (size_t)c * g_qwen.V,
                   (size_t)g_qwen.V * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) return -3;
    return g_qwen.V;
}
