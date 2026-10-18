// chunkwise gated DeltaNet for prefill, incl. the register-resident kernels (TQ_GDN_PREFILL).
// Part of the forward_qwen.cu unity build: included in order, not a standalone header.

// =================== Chunkwise-parallel gated DeltaNet (wide prefill) =========
//
// k_tq_deltanet_chunk computes, for a chunk of N prompt tokens, the SAME result
// as running the per-token kernel k_tq_linear_decode_core_gated N times (gated
// RMSNorm'd core [N, value_dim] + an advanced recurrent_state [value_heads,D,D]),
// but with intra-chunk parallelism instead of N serial latency-bound launches.
//
// Per value head h (key head kh = h/group, group = value_heads/key_heads), the
// per-token recurrence the reference runs is, with L2-normalised q/k
//   qn = q*rsqrt(|q|^2+1e-6)/sqrt(D),  kn = k*rsqrt(|k|^2+1e-6),
// scalar gate beta = sigmoid(b), and scalar log-decay
//   g = -exp(A_log)*softplus(a + dt_bias)   (g <= 0 always, decay = e^g in (0,1]):
//   S_t   = e^{g_t} S_{t-1} + kn_t (delta_t)^T            [D(key) x D(value)]
//   delta_t = beta_t ( v_t - e^{g_t} S_{t-1}^T kn_t )      [D(value)]
//   core_t  = S_t^T qn_t                                   [D(value)]
// followed by gated RMSNorm over the value dim: out_t = RMSNorm(core_t)*silu(z_t).
//
// Within a sub-chunk of CK tokens (entering state S0, local cumulative log-decay
// la_j = sum_{m=0..j} g_{s+m}, gamma_j = e^{la_j}) this is algebraically:
//   (I + L) Delta = R       (forward-substitution solve, parallel over value col)
//     L[j,m] = beta_j e^{la_j-la_m} (kn_j . kn_m)   for m < j  (strictly lower)
//     R[j]   = beta_j ( v_j - gamma_j (S0^T kn_j) )
//   core_j   = gamma_j (S0^T qn_j) + sum_{i<=j} e^{la_j-la_i}(qn_j . kn_i) Delta_i
//   S0'      = gamma_{CK-1} S0 + sum_i e^{la_{CK-1}-la_i} kn_i (Delta_i)^T
// Sub-chunks are carried sequentially through S0 (only N/CK serial steps, all in
// one launch); everything inside a sub-chunk is dense parallel work. Every e^{.}
// argument is <= 0 so all decay weights are in (0,1] (numerically stable).
//
// Launch: grid = value_heads blocks, block = D (=128) threads (thread = value
// column). Static shared per block (CK=32, D=128) ~ 41 KB (< 48 KB, no opt-in).
// ADDITIVE: not wired into any shipping path; validated by qwn_deltanet_chunk_check.
template<int CK, int D>
__global__ void k_tq_deltanet_chunk(float *core_out, float *recurrent_state,
                                    const float *qkv_conv, const float *z,
                                    const float *b_proj, const float *a_proj,
                                    const float *A_log, const uint16_t *dt_bias,
                                    const float *norm_weight,
                                    int N, int value_heads, int key_heads, int dim,
                                    float eps) {
    int head = blockIdx.x;
    if (head >= value_heads) return;
    int v = threadIdx.x;                 // value-dim column owned by this thread
    int group = value_heads / key_heads;
    int key_head = head / group;
    int key_dim = key_heads * D;
    int value_dim = value_heads * D;
    int conv_dim = 2 * key_dim + value_dim;

    __shared__ float kn_sh[CK][D];       // L2-normalised keys      (token j, dim d)
    __shared__ float qn_sh[CK][D];       // normalised queries (* 1/sqrt(D))
    __shared__ float Lmat[CK][CK];       // beta_j e^{la_j-la_m}(kn_j.kn_m), m<j
    __shared__ float Amat[CK][CK];       // e^{la_j-la_i}(qn_j.kn_i), i<=j
    __shared__ float la_sh[CK];          // cumulative log-decay within sub-chunk
    __shared__ float beta_sh[CK];
    __shared__ float qfac[CK];           // per-token q/k inverse-L2 scale factors
    __shared__ float kfac[CK];
    __shared__ float red[D];             // block-reduction scratch (RMSNorm)

    float Alog = A_log[head];                       // per-head, constant over chunk
    float dtb  = tq_bf16_to_float(dt_bias[head]);
    float *state = recurrent_state + (size_t)head * D * D;   // S[k][v] = state[k*D+v]

    for (int s = 0; s < N; s += CK) {
        int L = (N - s) < CK ? (N - s) : CK;        // sub-chunk length (uniform/block)

        // --- 1. build kn_sh / qn_sh (per-token L2 norm over D) + beta/la ---
        // 1a: thread j (j<L) computes its token's whole q/k sum-of-squares (no
        // cross-thread reduction -> avoids ~2*L block reductions per sub-chunk).
        if (v < L) {
            int t = s + v;
            const float *q = qkv_conv + (size_t)t * conv_dim + (size_t)key_head * D;
            const float *k = qkv_conv + (size_t)t * conv_dim + key_dim + (size_t)key_head * D;
            float qss = 0.0f, kss = 0.0f;
            for (int d = 0; d < D; d++) { float a = q[d]; qss += a * a; float b = k[d]; kss += b * b; }
            qfac[v] = rsqrtf(qss + 1.0e-6f) * rsqrtf((float)D);
            kfac[v] = rsqrtf(kss + 1.0e-6f);
        }
        __syncthreads();
        // 1b: all D threads write column v of every token's normalised q/k.
        for (int j = 0; j < L; j++) {
            int t = s + j;
            const float *q = qkv_conv + (size_t)t * conv_dim + (size_t)key_head * D;
            const float *k = qkv_conv + (size_t)t * conv_dim + key_dim + (size_t)key_head * D;
            qn_sh[j][v] = q[v] * qfac[j];
            kn_sh[j][v] = k[v] * kfac[j];
        }
        if (v == 0) {
            float cum = 0.0f;
            for (int j = 0; j < L; j++) {
                int t = s + j;
                float bb = b_proj[(size_t)t * value_heads + head];
                float beta = 1.0f / (1.0f + expf(-bb));
                float sp_arg = a_proj[(size_t)t * value_heads + head] + dtb;
                float sp = log1pf(expf(-fabsf(sp_arg))) + fmaxf(sp_arg, 0.0f);
                cum += -expf(Alog) * sp;            // g_j <= 0
                beta_sh[j] = beta;
                la_sh[j] = cum;
            }
        }
        __syncthreads();

        // --- 2. decayed inner-product matrices L (beta*K K^T) and A (Q K^T) ---
        for (int idx = v; idx < L * L; idx += D) {
            int j = idx / L, m = idx % L;
            float dkk = 0.0f, dqk = 0.0f;
            #pragma unroll 4
            for (int d = 0; d < D; d++) {
                float knm = kn_sh[m][d];
                dkk += kn_sh[j][d] * knm;
                dqk += qn_sh[j][d] * knm;
            }
            float decay = (m <= j) ? expf(la_sh[j] - la_sh[m]) : 0.0f;
            Lmat[j][m] = (m < j)  ? beta_sh[j] * decay * dkk : 0.0f;
            Amat[j][m] = (m <= j) ? decay * dqk : 0.0f;
        }
        __syncthreads();

        // --- 3. initial read-outs kS0[j]=S0^T kn_j, qS0[j]=S0^T qn_j (col v) ---
        float kS0[CK], qS0[CK];
        #pragma unroll
        for (int j = 0; j < CK; j++) { kS0[j] = 0.0f; qS0[j] = 0.0f; }
        for (int k = 0; k < D; k++) {
            float sval = state[(size_t)k * D + v];
            for (int j = 0; j < L; j++) {
                kS0[j] += kn_sh[j][k] * sval;
                qS0[j] += qn_sh[j][k] * sval;
            }
        }

        // --- 4. RHS R[j] = beta_j (v_j - gamma_j kS0[j]) ---
        float Dl[CK];
        for (int j = 0; j < L; j++) {
            int t = s + j;
            float vv = qkv_conv[(size_t)t * conv_dim + 2 * key_dim + (size_t)head * D + v];
            float gamma = expf(la_sh[j]);
            Dl[j] = beta_sh[j] * (vv - gamma * kS0[j]);
        }
        // --- 5. forward substitution: Delta[j] = R[j] - sum_{m<j} L[j,m] Delta[m] ---
        for (int j = 1; j < L; j++) {
            float acc = Dl[j];
            for (int m = 0; m < j; m++) acc -= Lmat[j][m] * Dl[m];
            Dl[j] = acc;
        }

        // --- 6. core[j] = gamma_j qS0[j] + sum_{i<=j} A[j,i] Delta[i] ---
        float core[CK];
        for (int j = 0; j < L; j++) {
            float acc = expf(la_sh[j]) * qS0[j];
            for (int i = 0; i <= j; i++) acc += Amat[j][i] * Dl[i];
            core[j] = acc;
        }

        // --- 7. gated RMSNorm over value dim, write out ---
        for (int j = 0; j < L; j++) {
            int t = s + j;
            red[v] = core[j] * core[j]; __syncthreads();
            float ss = tq_block_reduce_sum_sh(red, D, v); __syncthreads();
            float normed = core[j] * rsqrtf(ss / (float)D + eps) * norm_weight[v];
            float gate = z[(size_t)t * value_dim + (size_t)head * D + v];
            core_out[(size_t)t * value_dim + (size_t)head * D + v] =
                normed * (gate / (1.0f + expf(-gate)));
        }

        // --- 8. state carry S0' = gamma_last S0 + sum_i e^{la_last-la_i} kn_i Delta_i ---
        float la_last = la_sh[L - 1];
        float gamma_last = expf(la_last);
        float dec[CK];
        for (int i = 0; i < L; i++) dec[i] = expf(la_last - la_sh[i]) * Dl[i];
        for (int k = 0; k < D; k++) {
            float acc = gamma_last * state[(size_t)k * D + v];
            for (int i = 0; i < L; i++) acc += dec[i] * kn_sh[i][k];
            state[(size_t)k * D + v] = acc;
        }
        __syncthreads();   // all threads done with kn_sh/qn_sh before next sub-chunk rebuilds them
    }
}

// Host launcher: dispatch the sub-chunk size CK (template) used by the kernel.
static int launch_deltanet_chunk(int ck, float *core_out, float *recurrent_state,
                                 const float *qkv_conv, const float *z,
                                 const float *b_proj, const float *a_proj,
                                 const float *A_log, const uint16_t *dt_bias,
                                 const float *norm_weight, int N, int value_heads,
                                 int key_heads, int dim, float eps, cudaStream_t st) {
    if (dim != 128) return -100;            // kernel specialised to D=128
    dim3 grid(value_heads), blk(128);
    #define TQ_LAUNCH_DNC(CKV)                                                      \
        k_tq_deltanet_chunk<CKV, 128><<<grid, blk, 0, st>>>(                        \
            core_out, recurrent_state, qkv_conv, z, b_proj, a_proj,                 \
            A_log, dt_bias, norm_weight, N, value_heads, key_heads, dim, eps)
    switch (ck) {
        case 8:  TQ_LAUNCH_DNC(8);  break;
        case 16: TQ_LAUNCH_DNC(16); break;
        case 32: TQ_LAUNCH_DNC(32); break;
        default: return -101;
    }
    #undef TQ_LAUNCH_DNC
    return 0;
}

// ---- Occupancy-optimised variant: value-column head-split + G-lane split ----
// The single-block-per-head kernel above engages only value_heads(48) of 170 SMs
// at ~4 warps/SM (latency-bound, ~5% compute). Here each head's D value columns
// are split into NSTRIPE blocks and each value column is handled by G lanes that
// split the head-dim contractions (state read in step 3, state write in step 8),
// so grid = value_heads*NSTRIPE blocks of SV*G threads engage many more SMs with
// more warps each. The cross-value gated RMSNorm cannot be done per stripe, so
// this writes the RAW core and k_tq_deltanet_norm finalises it. Numerically
// identical to k_tq_deltanet_chunk (same per-column math, same FP order per lane
// group up to the G-split reduction order).
// DS (dstate): storage format of recurrent_state (0 fp32 = the original kernel; 1 bf16 / 2 fp16 for the paged pool
// under TQ_DSTATE_BF16: the state is read once and written once per CK-token sub-chunk, converted at the access).
template<int CK, int D, int NSTRIPE, int G, int DS = 0>
__global__ void k_tq_deltanet_chunk_hs(float *core_raw, void *recurrent_state,
                                       const float *qkv_conv,
                                       const float *b_proj, const float *a_proj,
                                       const float *A_log, const uint16_t *dt_bias,
                                       int N, int value_heads, int key_heads, int dim,
                                       const int *tok_pos = NULL) {   // DS = 4: the N tokens' positions
    const int SV = D / NSTRIPE;                  // value columns per stripe
    const int BS = SV * G;                       // threads per block
    int head = blockIdx.x / NSTRIPE;
    int stripe = blockIdx.x % NSTRIPE;
    if (head >= value_heads) return;
    int tid = threadIdx.x;
    int col_local = tid / G;
    int klane = tid % G;
    int v = stripe * SV + col_local;             // global value column
    int group = value_heads / key_heads;
    int key_head = head / group;
    int key_dim = key_heads * D;
    int value_dim = value_heads * D;
    int conv_dim = 2 * key_dim + value_dim;

    __shared__ float kn_sh[CK][D];
    __shared__ float qn_sh[CK][D];
    __shared__ float Lmat[CK][CK];
    __shared__ float Amat[CK][CK];
    __shared__ float la_sh[CK];
    __shared__ float beta_sh[CK];
    __shared__ float qfac[CK];
    __shared__ float kfac[CK];

    float Alog = A_log[head];
    float dtb  = tq_bf16_to_float(dt_bias[head]);
    // the head's state slice in the DS format (recurrent_state = the slot's base; the slot holds value_heads x D x D)
    tq_ds_acc<DS> acc = tq_ds_make<DS>(recurrent_state, 0, (size_t)value_heads * D * D, (size_t)head * D * D);
    const unsigned FULL = 0xffffffffu;

    for (int s = 0; s < N; s += CK) {
        int L = (N - s) < CK ? (N - s) : CK;

        // 1a. per-token q/k inverse-L2 factors (thread tid<L owns token tid).
        if (tid < L) {
            int t = s + tid;
            const float *q = qkv_conv + (size_t)t * conv_dim + (size_t)key_head * D;
            const float *k = qkv_conv + (size_t)t * conv_dim + key_dim + (size_t)key_head * D;
            float qss = 0.0f, kss = 0.0f;
            for (int d = 0; d < D; d++) { float a = q[d]; qss += a * a; float b = k[d]; kss += b * b; }
            qfac[tid] = rsqrtf(qss + 1.0e-6f) * rsqrtf((float)D);
            kfac[tid] = rsqrtf(kss + 1.0e-6f);
        }
        if (tid == 0) {
            float cum = 0.0f;
            for (int j = 0; j < L; j++) {
                int t = s + j;
                float bb = b_proj[(size_t)t * value_heads + head];
                float sp_arg = a_proj[(size_t)t * value_heads + head] + dtb;
                float sp = log1pf(expf(-fabsf(sp_arg))) + fmaxf(sp_arg, 0.0f);
                cum += -expf(Alog) * sp;
                beta_sh[j] = 1.0f / (1.0f + expf(-bb));
                la_sh[j] = cum;
            }
        }
        __syncthreads();
        // 1b. normalised q/k for every (token,dim) -- flat over all BS threads.
        for (int idx = tid; idx < L * D; idx += BS) {
            int j = idx / D, d = idx % D;
            int t = s + j;
            qn_sh[j][d] = qkv_conv[(size_t)t * conv_dim + (size_t)key_head * D + d] * qfac[j];
            kn_sh[j][d] = qkv_conv[(size_t)t * conv_dim + key_dim + (size_t)key_head * D + d] * kfac[j];
        }
        __syncthreads();
        // 2. decayed inner-product matrices L and A.
        for (int idx = tid; idx < L * L; idx += BS) {
            int j = idx / L, m = idx % L;
            float dkk = 0.0f, dqk = 0.0f;
            #pragma unroll 4
            for (int d = 0; d < D; d++) {
                float knm = kn_sh[m][d];
                dkk += kn_sh[j][d] * knm;
                dqk += qn_sh[j][d] * knm;
            }
            float decay = (m <= j) ? expf(la_sh[j] - la_sh[m]) : 0.0f;
            Lmat[j][m] = (m < j)  ? beta_sh[j] * decay * dkk : 0.0f;
            Amat[j][m] = (m <= j) ? decay * dqk : 0.0f;
        }
        __syncthreads();

        // 3. kS0[j], qS0[j] for column v -- G lanes split the head-dim contraction.
        float kS0[CK], qS0[CK];
        #pragma unroll
        for (int j = 0; j < CK; j++) { kS0[j] = 0.0f; qS0[j] = 0.0f; }
        for (int k = klane; k < D; k += G) {
            float sval = acc.ld((size_t)k * D + v);
            for (int j = 0; j < L; j++) {
                kS0[j] += kn_sh[j][k] * sval;
                qS0[j] += qn_sh[j][k] * sval;
            }
        }
        #pragma unroll
        for (int off = G >> 1; off > 0; off >>= 1)
            for (int j = 0; j < L; j++) {
                kS0[j] += __shfl_down_sync(FULL, kS0[j], off, G);
                qS0[j] += __shfl_down_sync(FULL, qS0[j], off, G);
            }

        float Dl[CK];
        if (klane == 0) {
            for (int j = 0; j < L; j++) {
                int t = s + j;
                float vv = qkv_conv[(size_t)t * conv_dim + 2 * key_dim + (size_t)head * D + v];
                Dl[j] = beta_sh[j] * (vv - expf(la_sh[j]) * kS0[j]);
            }
            for (int j = 1; j < L; j++) {
                float acc = Dl[j];
                for (int m = 0; m < j; m++) acc -= Lmat[j][m] * Dl[m];
                Dl[j] = acc;
            }
            for (int j = 0; j < L; j++) {
                float acc = expf(la_sh[j]) * qS0[j];
                for (int i = 0; i <= j; i++) acc += Amat[j][i] * Dl[i];
                // raw (un-normed) core for column v, token s+j
                core_raw[(size_t)(s + j) * value_dim + (size_t)head * D + v] = acc;
            }
        }
        // 8. state carry -- broadcast Delta from klane 0, lanes split the k write.
        for (int i = 0; i < L; i++) Dl[i] = __shfl_sync(FULL, Dl[i], 0, G);
        float la_last = la_sh[L - 1];
        float gamma_last = expf(la_last);
        float dec[CK];
        for (int i = 0; i < L; i++) dec[i] = expf(la_last - la_sh[i]) * Dl[i];
        if constexpr (DS == 4) acc.pos = (uint32_t)tok_pos[s + L - 1];   // the sub-chunk's last token seeds its rounding
        for (int k = klane; k < D; k += G) {
            float sacc = gamma_last * acc.ld((size_t)k * D + v);
            for (int i = 0; i < L; i++) sacc += dec[i] * kn_sh[i][k];
            acc.st((size_t)k * D + v, sacc);
        }
        __syncthreads();
    }
}

// Batched gated RMSNorm finaliser for the head-split path: one block per
// (token, value head) normalises the D raw core values, scales by norm_weight,
// and applies the silu(z) gate -- in place on core_raw -> core_out.
__global__ void k_tq_deltanet_norm(float *core_out, const float *core_raw, const float *z,
                                   const float *norm_weight, int value_heads, int dim, float eps) {
    int blk = blockIdx.x;                 // = token * value_heads + head
    int v = threadIdx.x;
    int value_dim = value_heads * dim;
    int head = blk % value_heads;
    size_t base = (size_t)(blk / value_heads) * value_dim + (size_t)head * dim;
    __shared__ float red[256];
    float c = core_raw[base + v];
    red[v] = c * c;
    __syncthreads();
    for (int stride = dim >> 1; stride > 0; stride >>= 1) {
        if (v < stride) red[v] += red[v + stride];
        __syncthreads();
    }
    float normed = c * rsqrtf(red[0] / (float)dim + eps) * norm_weight[v];
    float gate = z[base + v];
    core_out[base + v] = normed * (gate / (1.0f + expf(-gate)));
}

// Launch the head-split chunk kernel + the norm finaliser. block = (D/nstripe)*g
// threads, grid = value_heads*nstripe blocks. BEST config measured on RTX 5090
// (N=128): ck=8, nstripe=2, g=8 -> ~260us = 15.3x vs the 128x per-token loop
// (~3995us). Returns 0 on success, -101 if (ck,nstripe,g) was not instantiated.
static int launch_deltanet_chunk_hs(int ck, int nstripe, int g,
                                    float *core_out, float *recurrent_state,
                                    const float *qkv_conv, const float *z,
                                    const float *b_proj, const float *a_proj,
                                    const float *A_log, const uint16_t *dt_bias,
                                    const float *norm_weight, int N, int value_heads,
                                    int key_heads, int dim, float eps, cudaStream_t st) {
    if (dim != 128) return -100;
    int sv = dim / nstripe;
    dim3 grid(value_heads * nstripe), blk(sv * g);
    #define TQ_LAUNCH_DNC_HS(CKV, NS, GG)                                              \
        k_tq_deltanet_chunk_hs<CKV, 128, NS, GG><<<grid, blk, 0, st>>>(                 \
            core_out, recurrent_state, qkv_conv, b_proj, a_proj,                        \
            A_log, dt_bias, N, value_heads, key_heads, dim)
    int ok = 0;
    if (ck == 16) {
        if      (nstripe == 1 && g == 4) TQ_LAUNCH_DNC_HS(16, 1, 4);
        else if (nstripe == 1 && g == 8) TQ_LAUNCH_DNC_HS(16, 1, 8);
        else if (nstripe == 2 && g == 4) TQ_LAUNCH_DNC_HS(16, 2, 4);
        else if (nstripe == 2 && g == 8) TQ_LAUNCH_DNC_HS(16, 2, 8);
        else if (nstripe == 4 && g == 4) TQ_LAUNCH_DNC_HS(16, 4, 4);
        else if (nstripe == 4 && g == 8) TQ_LAUNCH_DNC_HS(16, 4, 8);
        else if (nstripe == 8 && g == 4) TQ_LAUNCH_DNC_HS(16, 8, 4);
        else if (nstripe == 8 && g == 8) TQ_LAUNCH_DNC_HS(16, 8, 8);
        else ok = -101;
    } else if (ck == 8) {
        if      (nstripe == 2 && g == 4) TQ_LAUNCH_DNC_HS(8, 2, 4);
        else if (nstripe == 4 && g == 4) TQ_LAUNCH_DNC_HS(8, 4, 4);
        else if (nstripe == 2 && g == 8) TQ_LAUNCH_DNC_HS(8, 2, 8);
        else if (nstripe == 1 && g == 8) TQ_LAUNCH_DNC_HS(8, 1, 8);
        else if (nstripe == 4 && g == 8) TQ_LAUNCH_DNC_HS(8, 4, 8);
        else if (nstripe == 8 && g == 8) TQ_LAUNCH_DNC_HS(8, 8, 8);
        else if (nstripe == 2 && g == 16) TQ_LAUNCH_DNC_HS(8, 2, 16);
        else if (nstripe == 4 && g == 16) TQ_LAUNCH_DNC_HS(8, 4, 16);
        else ok = -101;
    } else if (ck == 4) {
        if      (nstripe == 2 && g == 8) TQ_LAUNCH_DNC_HS(4, 2, 8);
        else if (nstripe == 2 && g == 16) TQ_LAUNCH_DNC_HS(4, 2, 16);
        else if (nstripe == 4 && g == 8) TQ_LAUNCH_DNC_HS(4, 4, 8);
        else ok = -101;
    } else if (ck == 32) {
        if      (nstripe == 2 && g == 4) TQ_LAUNCH_DNC_HS(32, 2, 4);
        else if (nstripe == 4 && g == 8) TQ_LAUNCH_DNC_HS(32, 4, 8);
        else ok = -101;
    } else ok = -101;
    #undef TQ_LAUNCH_DNC_HS
    if (ok != 0) return ok;
    k_tq_deltanet_norm<<<value_heads * N, dim, 0, st>>>(
        core_out, core_out, z, norm_weight, value_heads, dim, eps);
    return 0;
}

// The paged prefill waves' call (ck = 8, nstripe = 2, g = 8) with the recurrent state in the dstate format ds
// (0 = the launcher above, unchanged; 1 bf16 / 2 fp16: the same kernel instantiated on the half storage).
static int launch_deltanet_chunk_hs_ds(int ds, float *core_out, void *recurrent_state,
                                       const float *qkv_conv, const float *z,
                                       const float *b_proj, const float *a_proj,
                                       const float *A_log, const uint16_t *dt_bias,
                                       const float *norm_weight, int N, int value_heads,
                                       int key_heads, int dim, float eps, cudaStream_t st, const int *tok_pos) {
    if (ds == 0)
        return launch_deltanet_chunk_hs(8, 2, 8, core_out, (float *)recurrent_state, qkv_conv, z, b_proj, a_proj,
                                        A_log, dt_bias, norm_weight, N, value_heads, key_heads, dim, eps, st);
    if (dim != 128) return -100;
    dim3 grid(value_heads * 2), blk((dim / 2) * 8);
    #define TQ_DNC_DS(DSV) k_tq_deltanet_chunk_hs<8, 128, 2, 8, DSV><<<grid, blk, 0, st>>>( \
        core_out, recurrent_state, qkv_conv, b_proj, a_proj, A_log, dt_bias, N, value_heads, key_heads, dim, tok_pos)
    if (ds == 1) TQ_DNC_DS(1);
    else if (ds == 2) TQ_DNC_DS(2);
    else if (ds == 3) TQ_DNC_DS(3);
    else if (ds == 4) TQ_DNC_DS(4);
    else return -102;
    #undef TQ_DNC_DS
    k_tq_deltanet_norm<<<value_heads * N, dim, 0, st>>>(
        core_out, core_out, z, norm_weight, value_heads, dim, eps);
    return 0;
}

// =================== Session 11 (gdn-prefill): hoisted prep + register-resident chunkwise DeltaNet ===================
// TQ_GDN_PREFILL=1 replaces, for the paged prefill waves only, k_tq_linear_conv_chunk + k_tq_deltanet_chunk_hs<8,128,2,8,DS>
// by two kernels that compute bit for bit the same core_raw, recurrent state (in every DS format, same SR seeds and
// decisions) and conv window, then the unchanged k_tq_deltanet_norm:
//  * k_tq_gdn_prep (per segment): everything that does not depend on the recurrent state, once instead of per sub-chunk
//    and per stripe -- the causal conv + SiLU (same window / same FMA chain as k_tq_linear_conv_chunk), the per-token
//    q/k L2 factors (the same sequential sum of squares) and the normalised qn / kn written "lane-permuted"
//    ([key head][token][klane 0..7][16], element (klane, i) = dim klane + 8 i, so a warp's chain reads are 16-byte
//    broadcasts), the conv'd v, the 8x8 decayed inner products (dkk, dqk) of every sub-chunk (the same d-sequential
//    chains), and the gates beta / cumulative log-decay la / e^la / e^(la_last - la) of every (head, token).
//  * k_tq_gdn_chunk<KT, DS>: grid = value_heads x 4 stripes of 32 value columns, block = 8/KT warps (128 threads at
//    KT = 2). Warp w's lanes are the 32 columns; the warp owns the head-dim chains k = w + (8/KT) h (h < KT) of the
//    chunk kernel's 8-lane split and keeps those 16 KT state values per column in registers for the whole segment
//    (read once, written once; between sub-chunks the register is rounded exactly as a store + load would round it,
//    tq_ds_acc::rnd_fast with the same (element index, position) seed). The 16 sub-chunks run without any
//    recomputation: per sub-chunk the read-outs S0^T kn_j / S0^T qn_j are the same per-chain FMA chains, the 8-lane
//    shuffle tree ((l0+l4)+(l2+l6))+((l1+l5)+(l3+l7)) is reproduced as in-thread adds (chains w, w+4) + one shared-memory
//    exchange between the warps, every warp then solves the 8x8 forward substitution and the core redundantly (warp 0
//    stores core_raw) and applies the state carry to its own chains. kn / qn / v / gates of 16 tokens are staged in
//    shared memory per stage (35 KB -> 2 blocks per SM).
// Also the conv window of the slot is advanced (first blocks of the chunk kernel, after the prep consumed the old one).
// Everything below is additive; TQ_GDN_PREFILL=0 (default) runs the kernels above unchanged.

// grid = nsub * (key_heads + value_dim / 256) (an 8-token sub-chunk x a 256-channel group: the q+k channels of one key
// head, or 256 v channels) + ceil(value_heads * nsub / 256) (role c: gates); block = 256. KS = 4 (the window of the
// sub-chunk's 8 tokens = 11 rows, loaded once into registers).
template <int D>
__global__ void __launch_bounds__(256) k_tq_gdn_prep(
        const float *__restrict__ x, const float *__restrict__ conv_state, const uint16_t *__restrict__ conv_w,
        const float *__restrict__ b_proj, const float *__restrict__ a_proj,
        const float *__restrict__ A_log, const uint16_t *__restrict__ dt_bias,
        float *__restrict__ qn_out, float *__restrict__ kn_out,       // [key_heads][T][D] lane-permuted
        float2 *__restrict__ dd_out,                                   // [key_heads][nsub][8][8] (dkk, dqk)
        float *__restrict__ v_out,                                     // [T][value_dim]
        float4 *__restrict__ gates,                                    // [value_heads][T] {beta, la, e^la, e^(la_last - la)}
        int T, int value_heads, int key_heads) {
    constexpr int KS = 4;
    const int key_dim = key_heads * D, value_dim = value_heads * D, conv_dim = 2 * key_dim + value_dim;
    const int nsub = (T + 7) / 8, ngrp = key_heads + value_dim / 256;
    int blk = blockIdx.x, tid = threadIdx.x;
    if (blk < nsub * ngrp) {
        __shared__ float qk[2][8][D + 1];          // normalised q / k of the sub-chunk's 8 tokens (post conv + SiLU)
        __shared__ float fac[2][8];
        int sc = blk / ngrp, g = blk % ngrp, s = sc * 8, L = min(8, T - s);
        const bool isqk = g < key_heads;
        int kh = g, d = tid & (D - 1), which = tid >> 7;
        int c = isqk ? ((tid < D) ? kh * D + tid : key_dim + kh * D + (tid - D)) : 2 * key_dim + (g - key_heads) * 256 + tid;
        // the window rows s-3 .. s+7 of channel c (rows < 0 from the slot's conv window, rows >= T unused)
        float xr[8 + KS - 1];
        const float *cs = conv_state + (size_t)c * KS;
        #pragma unroll
        for (int jj = 0; jj < 8 + KS - 1; jj++) {
            int u = s - (KS - 1) + jj;
            xr[jj] = (u < 0) ? cs[KS + u] : (u < T) ? x[(size_t)u * conv_dim + c] : 0.0f;
        }
        const uint16_t *wt = conv_w + (size_t)c * KS;
        float wf[KS];
        #pragma unroll
        for (int j = 0; j < KS; j++) wf[j] = tq_bf16_to_float(wt[j]);
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            if (j < L) {
                // k_tq_linear_conv_chunk's order: sum = 0; += win[0] w0; += win[1] w1; += win[2] w2; += win[3] w3
                float sum = 0.0f;
                #pragma unroll
                for (int jj = 0; jj < KS; jj++) sum = __fmaf_rn(xr[j + jj], wf[jj], sum);
                float y = sum / (1.0f + expf(-sum));
                if (isqk) qk[which][j][d] = y;
                else v_out[(size_t)(s + j) * value_dim + (g - key_heads) * 256 + tid] = y;
            }
        }
        if (!isqk) return;
        __syncthreads();
        if (tid < 16) {                              // the sequential sum of squares of token tid/2 (q: even, k: odd)
            int j = tid >> 1, wq = tid & 1;
            if (j < L) {
                const float *q = qk[wq][j];
                float ss = 0.0f;
                for (int dd = 0; dd < D; dd++) { float a = q[dd]; ss = __fmaf_rn(a, a, ss); }
                fac[wq][j] = (wq == 0) ? rsqrtf(ss + 1.0e-6f) * rsqrtf((float)D) : rsqrtf(ss + 1.0e-6f);
            }
        }
        __syncthreads();
        for (int j = 0; j < L; j++) qk[which][j][d] = qk[which][j][d] * fac[which][j];
        __syncthreads();
        // lane-permuted rows, written coalesced: position d holds dim (d & 15) * 8 + (d >> 4) (the inverse of (dim & 7) * 16 + (dim >> 3))
        int src = (d & 15) * 8 + (d >> 4);
        float *out = which ? kn_out : qn_out;
        for (int j = 0; j < L; j++) out[((size_t)kh * T + s + j) * D + d] = qk[which][j][src];
        if (tid < 64) {                              // the 8x8 decayed inner products' dot parts (d-sequential chains)
            int j = tid >> 3, m = tid & 7;
            if (j < L && m < L) {
                float dkk = 0.0f, dqk = 0.0f;
                #pragma unroll 4
                for (int dd = 0; dd < D; dd++) {
                    float knm = qk[1][m][dd];
                    dkk = __fmaf_rn(qk[1][j][dd], knm, dkk);
                    dqk = __fmaf_rn(qk[0][j][dd], knm, dqk);
                }
                dd_out[((size_t)kh * nsub + sc) * 64 + tid] = make_float2(dkk, dqk);
            }
        }
        return;
    }
    blk -= nsub * ngrp;
    int g = blk * 256 + tid;
    if (g >= value_heads * nsub) return;
    int head = g / nsub, sc = g % nsub, s = sc * 8, L = min(8, T - s);
    float Alog = A_log[head];
    float dtb  = tq_bf16_to_float(dt_bias[head]);
    float beta[8], la[8];
    float cum = 0.0f;
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        if (j < L) {
            int t = s + j;
            float bb = b_proj[(size_t)t * value_heads + head];
            float sp_arg = a_proj[(size_t)t * value_heads + head] + dtb;
            float sp = log1pf(expf(-fabsf(sp_arg))) + fmaxf(sp_arg, 0.0f);
            cum = __fmaf_rn(-expf(Alog), sp, cum);
            beta[j] = 1.0f / (1.0f + expf(-bb));
            la[j] = cum;
        } else { beta[j] = 0.0f; la[j] = cum; }
    }
    float la_last = la[L - 1];
    #pragma unroll
    for (int j = 0; j < 8; j++)
        if (j < L) gates[(size_t)head * T + s + j] = make_float4(beta[j], la[j], expf(la[j]), expf(la_last - la[j]));
}

// One sub-chunk of the register-resident chunk kernel (FULL: L == 8, no guards). S[h][i] = state[k][col], k = kk(h) + 8 i.
template <int KT, int DS, bool FULL>
static __device__ __forceinline__ void tq_gdn_subchunk(
        float (&S)[KT][16], const float (*kn_s)[128], const float (*qn_s)[128], const float (*v_s)[32],
        float (*Lm)[8], float (*Am)[8], const float4 *g_s, float *part /* [NW][16][32] */, float2 d2,
        float *__restrict__ core_raw, int Lrt, int w, int lane, int value_dim, int head_col_base, int t_base,
        const tq_ds_acc<DS> &acc, int col) {
    constexpr int NW = 8 / KT, D = 128;
    const int L = FULL ? 8 : Lrt;
    const int tid = w * 32 + lane;
    // 3. read-outs: per chain h the k-sequential FMA chains kS0_h[j] / qS0_h[j] (k = kk + 8 i), then the in-thread levels
    //    of the shuffle tree (chains kk, kk + NW, ... pair at distance 4, then 2, ...).
    float accK[KT][8], accQ[KT][8];
    #pragma unroll
    for (int h = 0; h < KT; h++) {
        const int kk = w + NW * h;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float ak = 0.0f, aq = 0.0f;
            if (FULL || j < L) {
                const float4 *kp = (const float4 *)(kn_s[j] + kk * 16);
                const float4 *qp = (const float4 *)(qn_s[j] + kk * 16);
                #pragma unroll
                for (int q = 0; q < 4; q++) {
                    float4 kv = kp[q], qv = qp[q];
                    ak = __fmaf_rn(kv.x, S[h][4 * q + 0], ak); aq = __fmaf_rn(qv.x, S[h][4 * q + 0], aq);
                    ak = __fmaf_rn(kv.y, S[h][4 * q + 1], ak); aq = __fmaf_rn(qv.y, S[h][4 * q + 1], aq);
                    ak = __fmaf_rn(kv.z, S[h][4 * q + 2], ak); aq = __fmaf_rn(qv.z, S[h][4 * q + 2], aq);
                    ak = __fmaf_rn(kv.w, S[h][4 * q + 3], ak); aq = __fmaf_rn(qv.w, S[h][4 * q + 3], aq);
                }
            }
            accK[h][j] = ak; accQ[h][j] = aq;
        }
    }
    #pragma unroll
    for (int off = KT / 2; off > 0; off >>= 1)
        #pragma unroll
        for (int h = 0; h < off; h++)
            #pragma unroll
            for (int j = 0; j < 8; j++) { accK[h][j] += accK[h + off][j]; accQ[h][j] += accQ[h + off][j]; }
    // exchange the warps' partials: part[w][kq * 8 + j][lane]
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        part[(w * 16 + j) * 32 + lane] = accK[0][j];
        part[(w * 16 + 8 + j) * 32 + lane] = accQ[0][j];
    }
    // 2. L / A of this sub-chunk from the prep's (dkk, dqk) + the gates (threads 0..63, one (j, m) pair each; published
    //    by the barrier below, read by the solve after it)
    if (tid < 64) {
        const int j = tid >> 3, m = tid & 7;
        if (FULL || (j < L && m < L)) {
            float laj = g_s[j].y, lam = g_s[m].y, betaj = g_s[j].x;
            float decay = (m <= j) ? expf(laj - lam) : 0.0f;
            Lm[j][m] = (m < j)  ? betaj * decay * d2.x : 0.0f;
            Am[j][m] = (m <= j) ? decay * d2.y : 0.0f;
        }
    }
    __syncthreads();
    float kS0[8], qS0[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        float pk[NW], pq[NW];
        #pragma unroll
        for (int ww = 0; ww < NW; ww++) { pk[ww] = part[(ww * 16 + j) * 32 + lane]; pq[ww] = part[(ww * 16 + 8 + j) * 32 + lane]; }
        #pragma unroll
        for (int off = NW / 2; off > 0; off >>= 1)
            #pragma unroll
            for (int ww = 0; ww < off; ww++) { pk[ww] += pk[ww + off]; pq[ww] += pq[ww + off]; }
        kS0[j] = pk[0]; qS0[j] = pq[0];
    }
    if constexpr (KT == 1) __syncthreads();   // single-buffered partials at 8 warps (static smem limit)
    // 4-6. RHS, forward substitution, core (every warp redundantly; warp 0 stores). Explicit __fmul_rn / __fmaf_rn: the
    // original kernel's loops (runtime bounds) materialise the products gamma * qS0 and gamma_last * S as FMULs before the
    // FFMA chains; fully unrolled, nvcc would otherwise be free to fuse the other product of "a*b + c*d" (not bitwise).
    float Dl[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        if (FULL || j < L) {
            float vv = v_s[j][lane];
            Dl[j] = __fmul_rn(g_s[j].x, __fmaf_rn(-g_s[j].z, kS0[j], vv));
        } else Dl[j] = 0.0f;
    }
    #pragma unroll
    for (int j = 1; j < 8; j++) {
        if (FULL || j < L) {
            float a = Dl[j];
            #pragma unroll
            for (int m = 0; m < j; m++) a = __fmaf_rn(-Lm[j][m], Dl[m], a);
            Dl[j] = a;
        }
    }
    if (w == 0) {
        // the original's SASS: core_j = fma(gamma_j, qS0_j, A[j][0] * Dl[0]) then += A[j][i] * Dl[i] for i = 1..j
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            if (FULL || j < L) {
                float a = __fmaf_rn(g_s[j].z, qS0[j], __fmul_rn(Am[j][0], Dl[0]));
                #pragma unroll
                for (int i = 1; i <= j; i++) a = __fmaf_rn(Am[j][i], Dl[i], a);
                core_raw[(size_t)(t_base + j) * value_dim + head_col_base + col] = a;
            }
        }
    }
    // 8. state carry on the warp's chains: S = gamma_last * S; += dec[i] * kn_i[k] in token order; then the storage rounding.
    const float gamma_last = g_s[L - 1].z;
    float dec[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) dec[i] = (FULL || i < L) ? __fmul_rn(g_s[i].w, Dl[i]) : 0.0f;
    // The original's SASS differs per format: the half / 24-bit instantiations (a conversion precedes the multiply) do
    // S = gamma_last * S (FMUL) then the FFMA chain; the fp32 instantiation fuses the scaling into the first term:
    // S = fma(gamma_last, S, dec[0] * kn_0) then the chain from i = 1.
    if constexpr (DS != 0) {
        #pragma unroll
        for (int h = 0; h < KT; h++)
            #pragma unroll
            for (int i = 0; i < 16; i++) S[h][i] = __fmul_rn(gamma_last, S[h][i]);
    } else {
        #pragma unroll
        for (int h = 0; h < KT; h++) {
            const int kk = w + NW * h;
            const float4 *kp = (const float4 *)(kn_s[0] + kk * 16);
            #pragma unroll
            for (int q = 0; q < 4; q++) {
                float4 kv = kp[q];
                S[h][4 * q + 0] = __fmaf_rn(gamma_last, S[h][4 * q + 0], __fmul_rn(dec[0], kv.x));
                S[h][4 * q + 1] = __fmaf_rn(gamma_last, S[h][4 * q + 1], __fmul_rn(dec[0], kv.y));
                S[h][4 * q + 2] = __fmaf_rn(gamma_last, S[h][4 * q + 2], __fmul_rn(dec[0], kv.z));
                S[h][4 * q + 3] = __fmaf_rn(gamma_last, S[h][4 * q + 3], __fmul_rn(dec[0], kv.w));
            }
        }
    }
    #pragma unroll
    for (int i = (DS != 0) ? 0 : 1; i < 8; i++) {
        if (FULL || i < L) {
            #pragma unroll
            for (int h = 0; h < KT; h++) {
                const int kk = w + NW * h;
                const float4 *kp = (const float4 *)(kn_s[i] + kk * 16);
                #pragma unroll
                for (int q = 0; q < 4; q++) {
                    float4 kv = kp[q];
                    S[h][4 * q + 0] = __fmaf_rn(dec[i], kv.x, S[h][4 * q + 0]);
                    S[h][4 * q + 1] = __fmaf_rn(dec[i], kv.y, S[h][4 * q + 1]);
                    S[h][4 * q + 2] = __fmaf_rn(dec[i], kv.z, S[h][4 * q + 2]);
                    S[h][4 * q + 3] = __fmaf_rn(dec[i], kv.w, S[h][4 * q + 3]);
                }
            }
        }
    }
    if constexpr (DS == 4) {
        // tq_ds_hash(gbase + idx, pos) with idx = (kk + 8 i) * D + col: idx * K1 = (gbase + kk * D + col) * K1 + i * (8 D K1)
        // (mod 2^32); the asm barrier keeps the compiler from hoisting the 32 per-element sums into registers.
        const uint32_t posterm = (acc.pos + 0x7F4A7C15u) * 0x85EBCA77u, pc = posterm ^ (posterm >> 16);
        #pragma unroll
        for (int h = 0; h < KT; h++) {
            const int kk = w + NW * h;
            uint32_t hk = (acc.gbase + (uint32_t)(kk * D + col)) * 0x9E3779B1u;
            asm volatile("" : "+r"(hk));
            #pragma unroll
            for (int i = 0; i < 16; i++)
                S[h][i] = tq_f16_sr_round(S[h][i], tq_ds_hash_mix(hk + (uint32_t)(i * 8 * D) * 0x9E3779B1u, pc));
        }
    } else if constexpr (DS != 0) {
        #pragma unroll
        for (int h = 0; h < KT; h++) {
            const int kk = w + NW * h;
            #pragma unroll
            for (int i = 0; i < 16; i++) S[h][i] = acc.rnd(( size_t)(kk + 8 * i) * D + col, S[h][i]);
        }
    }
}

template <int KT, int DS>
__global__ void __launch_bounds__(256 / KT) k_tq_gdn_chunk(
        float *__restrict__ core_raw, void *recurrent_state,
        const float *__restrict__ qn, const float *__restrict__ kn, const float2 *__restrict__ dd,
        const float *__restrict__ vconv, const float4 *__restrict__ gates, const int *__restrict__ tok_pos,
        float *__restrict__ conv_state, const float *__restrict__ x,
        int T, int value_heads, int key_heads, int conv_dim, int KS) {
    constexpr int D = 128, CK = 8, NW = 8 / KT, BS = 32 * NW;
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, tid = threadIdx.x;
    const int head = blockIdx.x >> 2, stripe = blockIdx.x & 3;
    const int group = value_heads / key_heads, key_head = head / group;
    const int value_dim = value_heads * D, nsub = (T + 7) / 8;
    const int col = stripe * 32 + lane;

    // the slot's conv window: win[j] = x[T - 4 + j] (or the old window shifted by T if T < 4); after k_tq_gdn_prep.
    if (blockIdx.x * BS < conv_dim) {
        int c = blockIdx.x * BS + tid;
        if (c < conv_dim) {
            float *s = conv_state + (size_t)c * 4;
            float old[4], nw[4];
            #pragma unroll
            for (int j = 0; j < 4; j++) old[j] = s[j];
            #pragma unroll
            for (int j = 0; j < 4; j++) { int u = j + T - 4; nw[j] = (u >= 0) ? x[(size_t)u * conv_dim + c] : 0.0f; }
            #pragma unroll
            for (int j = 0; j < 4; j++) { int u = j + T - 4; s[j] = (u >= 0) ? nw[j] : old[min(j + T, 3)]; }
        }
    }

    // per-sub-chunk staging (cp.async, double-buffered): kn / qn rows (lane-permuted), the stripe's v, the gates
    __shared__ __align__(16) float kn_s[2][CK][D];
    __shared__ __align__(16) float qn_s[2][CK][D];
    __shared__ __align__(16) float v_s[2][CK][32];
    __shared__ __align__(16) float4 g_s[2][CK];
    __shared__ float Lm[8][8], Am[8][8];
    __shared__ float part[KT == 1 ? 1 : 2][NW * 16 * 32];   // the warps' partial read-outs (double-buffered: 1 barrier per sub-chunk)
    __shared__ int pos_s[32];                                // the sub-chunks' seed positions (their last token), T <= 256

    tq_ds_acc<DS> acc = tq_ds_make<DS>(recurrent_state, 0, (size_t)value_heads * D * D, (size_t)head * D * D);
    if (tid < nsub && tid < 32) pos_s[tid] = tok_pos[min(tid * CK + CK - 1, T - 1)];
    const float *kn_src = kn + (size_t)key_head * T * D, *qn_src = qn + (size_t)key_head * T * D;
    const float *v_src = vconv + (size_t)head * D + stripe * 32;
    const float4 *g_src = gates + (size_t)head * T;
    // issue the copies of sub-chunk sc into buffer b: L rows x 32 16-byte chunks of kn and qn (BS threads), L x 8 chunks of v
    // (threads < 64), L chunks of gates (threads < 8); every thread commits a group (possibly empty)
    auto stage = [&](int sc, int b) {
        const int s = sc * CK, L = min(CK, T - s);
        const uint32_t kd = (uint32_t)__cvta_generic_to_shared(&kn_s[b][0][0]), qd = (uint32_t)__cvta_generic_to_shared(&qn_s[b][0][0]);
        for (int i = tid; i < L * (D / 4); i += BS) {
            tq_cp_async16(kd + (uint32_t)i * 16, kn_src + (size_t)s * D + (size_t)i * 4);
            tq_cp_async16(qd + (uint32_t)i * 16, qn_src + (size_t)s * D + (size_t)i * 4);
        }
        if (tid < L * 8) {
            int t = tid >> 3, q = tid & 7;
            tq_cp_async16((uint32_t)__cvta_generic_to_shared(&v_s[b][t][q * 4]), v_src + (size_t)(s + t) * value_dim + q * 4);
        }
        if (tid < L) tq_cp_async16((uint32_t)__cvta_generic_to_shared(&g_s[b][tid]), g_src + s + tid);
        asm volatile("cp.async.commit_group;\n");
    };
    auto dd_of = [&](int sc) -> float2 {                     // this thread's (dkk, dqk) pair of sub-chunk sc (threads < 64)
        return (tid < 64 && sc < nsub) ? dd[((size_t)key_head * nsub + sc) * 64 + tid] : make_float2(0.0f, 0.0f);
    };
    stage(0, 0);
    float2 d2 = dd_of(0);
    float S[KT][16];
    #pragma unroll
    for (int h = 0; h < KT; h++) {
        const int kk = w + NW * h;
        #pragma unroll
        for (int i = 0; i < 16; i++) S[h][i] = acc.ld((size_t)(kk + 8 * i) * D + col);
    }
    for (int sc = 0; sc < nsub; sc++) {
        const int b = sc & 1, s = sc * CK, L = min(CK, T - s);
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        __syncthreads();                                      // sub-chunk sc's staging complete; everyone done with sc - 1
        float2 d2n = make_float2(0.0f, 0.0f);
        if (sc + 1 < nsub) { stage(sc + 1, b ^ 1); d2n = dd_of(sc + 1); }
        if constexpr (DS == 4) acc.pos = (uint32_t)pos_s[sc];
        if (L == CK)
            tq_gdn_subchunk<KT, DS, true>(S, kn_s[b], qn_s[b], v_s[b], Lm, Am, g_s[b], part[KT == 1 ? 0 : b], d2,
                                          core_raw, L, w, lane, value_dim, head * D, s, acc, col);
        else
            tq_gdn_subchunk<KT, DS, false>(S, kn_s[b], qn_s[b], v_s[b], Lm, Am, g_s[b], part[KT == 1 ? 0 : b], d2,
                                           core_raw, L, w, lane, value_dim, head * D, s, acc, col);
        d2 = d2n;
    }
    #pragma unroll
    for (int h = 0; h < KT; h++) {
        const int kk = w + NW * h;
        #pragma unroll
        for (int i = 0; i < 16; i++) acc.st((size_t)(kk + 8 * i) * D + col, S[h][i]);   // a no-op rounding: S is representable
    }
}

// Scratch of the gdn prefill path, sized by the wave's T (per layer the buffers are reused).
static float  *g_gdn_qn = NULL, *g_gdn_kn = NULL, *g_gdn_v = NULL;
static float2 *g_gdn_dd = NULL;
static float4 *g_gdn_gates = NULL;
static int     g_gdn_cap = 0;
static int ensure_gdn_buffers(int T) {
    if (g_gdn_cap >= T) return 0;
    int lkh = g_qwen.linear_num_key_heads, lvh = g_qwen.linear_num_value_heads, D = g_qwen.linear_value_head_dim;
    int nsub = (T + 7) / 8;
    if (g_gdn_qn) cudaFree(g_gdn_qn); if (g_gdn_kn) cudaFree(g_gdn_kn); if (g_gdn_v) cudaFree(g_gdn_v);
    if (g_gdn_dd) cudaFree(g_gdn_dd); if (g_gdn_gates) cudaFree(g_gdn_gates);
    g_gdn_qn = g_gdn_kn = g_gdn_v = NULL; g_gdn_dd = NULL; g_gdn_gates = NULL; g_gdn_cap = 0;
    if (cudaMalloc(&g_gdn_qn, (size_t)lkh * T * D * sizeof(float)) != cudaSuccess) return -1;
    if (cudaMalloc(&g_gdn_kn, (size_t)lkh * T * D * sizeof(float)) != cudaSuccess) return -1;
    if (cudaMalloc(&g_gdn_v, (size_t)T * lvh * D * sizeof(float)) != cudaSuccess) return -1;
    if (cudaMalloc(&g_gdn_dd, (size_t)lkh * nsub * 64 * sizeof(float2)) != cudaSuccess) return -1;
    if (cudaMalloc(&g_gdn_gates, (size_t)lvh * T * sizeof(float4)) != cudaSuccess) return -1;
    g_gdn_cap = T;
    return 0;
}

// TQ_GDN_PREFILL: 0 = the kernels above (the 10-16 path, bit-identical); 1 = the gdn path with KT = 2 (4 warps per block) --
// DEFAULT since 2026-10-17 (session 11: DeltaNet 14.95 -> 2.76 ms per 128-column wave, bitwise on every gate,
// notes/session11-gdn.md); 2 = KT = 1 (8 warps); 3 = KT = 4 (2 warps). Read once.
static int tq_gdn_prefill_mode(void) {
    static int mode = -1;
    if (mode < 0) {
        const char *e = getenv("TQ_GDN_PREFILL");
        mode = (e && *e) ? atoi(e) : 1;
        if (mode < 0 || mode > 3) mode = 0;
    }
    return mode;
}

// The gdn path for one segment of n tokens (conv + chunkwise DeltaNet + norm): x_qkv = the segment's pre-conv qkv (NOT
// written: the conv output stays in the scratch), conv_slot = the slot's conv window (advanced), recur_slot = the slot's
// recurrent state in format ds (advanced), core_out = the normed core [n][value_dim], tok_pos = the n positions.
static int launch_gdn_prefill(int mode, int ds, float *core_out, void *recur_slot, float *conv_slot, const float *x_qkv,
                              const float *z, const float *b_proj, const float *a_proj, const float *A_log,
                              const uint16_t *dt_bias, const uint16_t *conv_w, const float *norm_weight,
                              int n, int value_heads, int key_heads, int dim, int KS, float eps, cudaStream_t st,
                              const int *tok_pos) {
    if (dim != 128 || KS != 4 || n > 256 || (value_heads * dim) % 256 != 0 || value_heads % key_heads != 0) return -100;
    if (ensure_gdn_buffers(n) != 0) return -90;
    int key_dim = key_heads * dim, value_dim = value_heads * dim, conv_dim = 2 * key_dim + value_dim;
    int nsub = (n + 7) / 8;
    int grid_p = nsub * (key_heads + value_dim / 256) + (value_heads * nsub + 255) / 256;
    k_tq_gdn_prep<128><<<grid_p, 256, 0, st>>>(x_qkv, conv_slot, conv_w, b_proj, a_proj, A_log, dt_bias,
                                                g_gdn_qn, g_gdn_kn, g_gdn_dd, g_gdn_v, g_gdn_gates, n, value_heads, key_heads);
    int kt = (mode == 2) ? 1 : (mode == 3) ? 4 : 2;
    int grid_c = value_heads * 4;
    if (grid_c * (256 / kt) < conv_dim) return -103;          // the window update rides on the first blocks
    #define TQ_GDN_LAUNCH(KTV, DSV) k_tq_gdn_chunk<KTV, DSV><<<grid_c, 256 / KTV, 0, st>>>(                           \
        core_out, recur_slot, g_gdn_qn, g_gdn_kn, g_gdn_dd, g_gdn_v, g_gdn_gates, tok_pos, conv_slot, x_qkv,            \
        n, value_heads, key_heads, conv_dim, KS)
    #define TQ_GDN_LAUNCH_DS(KTV)                                                                                      \
        switch (ds) {                                                                                                  \
            case 0: TQ_GDN_LAUNCH(KTV, 0); break; case 1: TQ_GDN_LAUNCH(KTV, 1); break;                                \
            case 2: TQ_GDN_LAUNCH(KTV, 2); break; case 3: TQ_GDN_LAUNCH(KTV, 3); break;                                \
            case 4: TQ_GDN_LAUNCH(KTV, 4); break; default: return -102;                                                \
        }
    if (kt == 1) { TQ_GDN_LAUNCH_DS(1) } else if (kt == 4) { TQ_GDN_LAUNCH_DS(4) } else { TQ_GDN_LAUNCH_DS(2) }
    #undef TQ_GDN_LAUNCH_DS
    #undef TQ_GDN_LAUNCH
    k_tq_deltanet_norm<<<value_heads * n, dim, 0, st>>>(core_out, core_out, z, norm_weight, value_heads, dim, eps);
    return 0;
}

// Deterministic synthetic fill for the check/bench harness (same values feed the
// reference per-token loop and the chunk kernel, so any diff is purely the two
// kernels' numerics). Bounded ~O(1) q/k/v so the L2 norm + gating are exercised.
static void deltanet_synth_fill(float *h, size_t n, int kind) {
    for (size_t i = 0; i < n; i++) {
        double x = (double)i;
        switch (kind) {
            case 0: // qkv (post-conv): ~O(1)
                h[i] = (float)(0.6 * sin(0.0131 * (x + 1.0)) + 0.4 * cos(0.0007 * (x * 7.0 + 3.0))); break;
            case 1: // z gate: ~[-0.9,0.9]
                h[i] = (float)(0.8 * sin(0.0193 * x + 1.3)); break;
            case 2: // b proj -> beta = sigmoid(b), spread over (0,1)
                h[i] = (float)(2.5 * sin(0.211 * x + 0.4)); break;
            case 3: // a proj -> softplus(a+dt_bias)
                h[i] = (float)(1.0 * sin(0.137 * x) - 0.3); break;
            case 4: // initial recurrent state: small nonzero (exercises S0 carry)
                h[i] = (float)(0.02 * sin(0.0051 * (x + 1.0) + 0.31)); break;
            default: h[i] = 0.0f; break;
        }
    }
}

// Correctness gate: build synthetic post-conv qkv/z/b/a for N tokens + a synthetic
// initial recurrent state, run the per-token reference loop AND k_tq_deltanet_chunk
// (sub-chunk CK) from the SAME inputs/initial state, and report the max abs diff
// over BOTH the core outputs [N,value_dim] and the final recurrent state.
// Returns 0 on success; *max_abs_diff = max(core_diff, state_diff). Target < 1e-3.
static int deltanet_chunk_check_impl(int layer, int N, int ck, int nstripe, int g,
                                     float *max_abs_diff) {
    if (!g_qwen.initialized) return -1;
    if (layer < 0 || layer >= g_qwen.L) return -2;
    if (g_qwen.layer_types[layer] != TQ_LAYER_LINEAR_ATTENTION) return -3;
    if (N <= 0 || N > 8192) return -4;
    tq_layer_t *l = &g_qwen.layers[layer];
    int heads = g_qwen.linear_num_value_heads;
    int key_heads = g_qwen.linear_num_key_heads;
    int D = g_qwen.linear_value_head_dim;
    if (D != 128) return -5;
    int key_dim = key_heads * D, value_dim = heads * D, conv_dim = 2 * key_dim + value_dim;
    size_t state_floats = (size_t)heads * D * D;

    float *d_qkv = 0, *d_z = 0, *d_b = 0, *d_a = 0;
    float *d_s0 = 0, *d_sref = 0, *d_schk = 0, *d_cref = 0, *d_cchk = 0;
    cudaMalloc(&d_qkv, (size_t)N * conv_dim * sizeof(float));
    cudaMalloc(&d_z,   (size_t)N * value_dim * sizeof(float));
    cudaMalloc(&d_b,   (size_t)N * heads * sizeof(float));
    cudaMalloc(&d_a,   (size_t)N * heads * sizeof(float));
    cudaMalloc(&d_s0,  state_floats * sizeof(float));
    cudaMalloc(&d_sref, state_floats * sizeof(float));
    cudaMalloc(&d_schk, state_floats * sizeof(float));
    cudaMalloc(&d_cref, (size_t)N * value_dim * sizeof(float));
    cudaMalloc(&d_cchk, (size_t)N * value_dim * sizeof(float));
    if (!d_qkv || !d_z || !d_b || !d_a || !d_s0 || !d_sref || !d_schk || !d_cref || !d_cchk) {
        cudaFree(d_qkv); cudaFree(d_z); cudaFree(d_b); cudaFree(d_a); cudaFree(d_s0);
        cudaFree(d_sref); cudaFree(d_schk); cudaFree(d_cref); cudaFree(d_cchk);
        return -6;
    }
    float *h_qkv = (float *)malloc((size_t)N * conv_dim * sizeof(float));
    float *h_z   = (float *)malloc((size_t)N * value_dim * sizeof(float));
    float *h_b   = (float *)malloc((size_t)N * heads * sizeof(float));
    float *h_a   = (float *)malloc((size_t)N * heads * sizeof(float));
    float *h_s0  = (float *)malloc(state_floats * sizeof(float));
    deltanet_synth_fill(h_qkv, (size_t)N * conv_dim, 0);
    deltanet_synth_fill(h_z,   (size_t)N * value_dim, 1);
    deltanet_synth_fill(h_b,   (size_t)N * heads, 2);
    deltanet_synth_fill(h_a,   (size_t)N * heads, 3);
    deltanet_synth_fill(h_s0,  state_floats, 4);
    // All copies on g_qwen.stream (it is cudaStreamNonBlocking): a plain cudaMemcpy
    // runs on the legacy default stream which does NOT order against a non-blocking
    // stream, so the kernels below would race the (async) D2D state-init otherwise.
    cudaStream_t st = g_qwen.stream;
    cudaMemcpyAsync(d_qkv, h_qkv, (size_t)N * conv_dim * sizeof(float), cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(d_z,   h_z,   (size_t)N * value_dim * sizeof(float), cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(d_b,   h_b,   (size_t)N * heads * sizeof(float), cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(d_a,   h_a,   (size_t)N * heads * sizeof(float), cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(d_s0,  h_s0,  state_floats * sizeof(float), cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(d_sref, d_s0, state_floats * sizeof(float), cudaMemcpyDeviceToDevice, st);
    cudaMemcpyAsync(d_schk, d_s0, state_floats * sizeof(float), cudaMemcpyDeviceToDevice, st);

    // reference: per-token loop (the ground truth), advancing d_sref.
    for (int t = 0; t < N; t++) {
        k_tq_linear_decode_core_gated<<<heads, D, 0, g_qwen.stream>>>(
            d_cref + (size_t)t * value_dim, d_sref,
            d_qkv + (size_t)t * conv_dim, d_z + (size_t)t * value_dim,
            d_b + (size_t)t * heads, d_a + (size_t)t * heads,
            l->d_linear_A_log, l->d_linear_dt_bias, l->d_linear_norm,
            heads, key_heads, D, g_qwen.eps);
    }
    // chunk kernel on the same inputs + initial state, advancing d_schk.
    int lret;
    if (nstripe > 0)
        lret = launch_deltanet_chunk_hs(ck, nstripe, g, d_cchk, d_schk, d_qkv, d_z, d_b, d_a,
                                        l->d_linear_A_log, l->d_linear_dt_bias, l->d_linear_norm,
                                        N, heads, key_heads, D, g_qwen.eps, g_qwen.stream);
    else
        lret = launch_deltanet_chunk(ck, d_cchk, d_schk, d_qkv, d_z, d_b, d_a,
                                     l->d_linear_A_log, l->d_linear_dt_bias, l->d_linear_norm,
                                     N, heads, key_heads, D, g_qwen.eps, g_qwen.stream);
    cudaError_t err = cudaStreamSynchronize(g_qwen.stream);
    cudaError_t kerr = cudaGetLastError();
    int rc = 0;
    if (lret != 0) rc = lret;
    else if (err != cudaSuccess || kerr != cudaSuccess) rc = -7;
    float cmax = 0.0f, smax = 0.0f;
    if (rc == 0) {
        float *h_cref = (float *)malloc((size_t)N * value_dim * sizeof(float));
        float *h_cchk = (float *)malloc((size_t)N * value_dim * sizeof(float));
        float *h_sref = (float *)malloc(state_floats * sizeof(float));
        float *h_schk = (float *)malloc(state_floats * sizeof(float));
        cudaMemcpy(h_cref, d_cref, (size_t)N * value_dim * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_cchk, d_cchk, (size_t)N * value_dim * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_sref, d_sref, state_floats * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_schk, d_schk, state_floats * sizeof(float), cudaMemcpyDeviceToHost);
        int cmt = -1, cmc = -1;
        for (size_t i = 0; i < (size_t)N * value_dim; i++) {
            float d = fabsf(h_cref[i] - h_cchk[i]);
            if (d > cmax) { cmax = d; cmt = (int)(i / value_dim); cmc = (int)(i % value_dim); }
        }
        for (size_t i = 0; i < state_floats; i++) {
            float d = fabsf(h_sref[i] - h_schk[i]); if (d > smax) smax = d;
        }
        // first token whose core diverges past 1e-3 (causality / carry diagnostics)
        int firstbad = -1;
        for (int t = 0; t < N && firstbad < 0; t++)
            for (int c = 0; c < value_dim; c++)
                if (fabsf(h_cref[(size_t)t*value_dim+c] - h_cchk[(size_t)t*value_dim+c]) > 1e-3f) { firstbad = t; break; }
        fprintf(stderr, "[deltanet_chunk_check] layer=%d N=%d CK=%d core_max=%.3e state_max=%.3e "
                "@tok=%d ch=%d ref=%.4f chk=%.4f firstbad_tok=%d\n",
                layer, N, ck, cmax, smax, cmt, cmc,
                cmt>=0 ? h_cref[(size_t)cmt*value_dim+cmc] : 0.0f,
                cmt>=0 ? h_cchk[(size_t)cmt*value_dim+cmc] : 0.0f, firstbad);
        free(h_cref); free(h_cchk); free(h_sref); free(h_schk);
    }
    free(h_qkv); free(h_z); free(h_b); free(h_a); free(h_s0);
    cudaFree(d_qkv); cudaFree(d_z); cudaFree(d_b); cudaFree(d_a); cudaFree(d_s0);
    cudaFree(d_sref); cudaFree(d_schk); cudaFree(d_cref); cudaFree(d_cchk);
    if (rc == 0 && max_abs_diff) *max_abs_diff = fmaxf(cmax, smax);
    return rc;
}

extern "C" int qwn_deltanet_chunk_check(int layer, int N, float *max_abs_diff) {
    return deltanet_chunk_check_impl(layer, N, 16, 0, 0, max_abs_diff);   // CK=16, simple kernel
}

// CK-parameterised variant for the sub-chunk sweep (ck in {8,16,32}).
extern "C" int qwn_deltanet_chunk_check_ck(int layer, int N, int ck, float *max_abs_diff) {
    return deltanet_chunk_check_impl(layer, N, ck, 0, 0, max_abs_diff);
}

// Head-split (occupancy) variant: nstripe value-column stripes, g lanes/column.
extern "C" int qwn_deltanet_chunk_check_hs(int layer, int N, int ck, int nstripe, int g,
                                           float *max_abs_diff) {
    return deltanet_chunk_check_impl(layer, N, ck, nstripe, g, max_abs_diff);
}

// Benchmark: time the chunk kernel (sub-chunk ck; nstripe>0 -> head-split kernel)
// vs the N-step per-token loop for the SAME chunk, both averaged over `iters`.
static int deltanet_chunk_bench_impl(int layer, int N, int ck, int nstripe, int g, int iters,
                                     float *ms_chunk, float *ms_pertoken) {
    if (!g_qwen.initialized) return -1;
    if (layer < 0 || layer >= g_qwen.L) return -2;
    if (g_qwen.layer_types[layer] != TQ_LAYER_LINEAR_ATTENTION) return -3;
    if (N <= 0 || N > 8192 || iters <= 0) return -4;
    tq_layer_t *l = &g_qwen.layers[layer];
    int heads = g_qwen.linear_num_value_heads;
    int key_heads = g_qwen.linear_num_key_heads;
    int D = g_qwen.linear_value_head_dim;
    if (D != 128) return -5;
    int key_dim = key_heads * D, value_dim = heads * D, conv_dim = 2 * key_dim + value_dim;
    size_t state_floats = (size_t)heads * D * D;

    float *d_qkv = 0, *d_z = 0, *d_b = 0, *d_a = 0, *d_s = 0, *d_c = 0;
    cudaMalloc(&d_qkv, (size_t)N * conv_dim * sizeof(float));
    cudaMalloc(&d_z,   (size_t)N * value_dim * sizeof(float));
    cudaMalloc(&d_b,   (size_t)N * heads * sizeof(float));
    cudaMalloc(&d_a,   (size_t)N * heads * sizeof(float));
    cudaMalloc(&d_s,   state_floats * sizeof(float));
    cudaMalloc(&d_c,   (size_t)N * value_dim * sizeof(float));
    if (!d_qkv || !d_z || !d_b || !d_a || !d_s || !d_c) {
        cudaFree(d_qkv); cudaFree(d_z); cudaFree(d_b); cudaFree(d_a); cudaFree(d_s); cudaFree(d_c);
        return -6;
    }
    cudaStream_t st = g_qwen.stream;
    float *h = (float *)malloc((size_t)N * conv_dim * sizeof(float));
    deltanet_synth_fill(h, (size_t)N * conv_dim, 0);
    cudaMemcpyAsync(d_qkv, h, (size_t)N * conv_dim * sizeof(float), cudaMemcpyHostToDevice, st);
    deltanet_synth_fill(h, (size_t)N * value_dim, 1);
    cudaMemcpyAsync(d_z, h, (size_t)N * value_dim * sizeof(float), cudaMemcpyHostToDevice, st);
    deltanet_synth_fill(h, (size_t)N * heads, 2);
    cudaMemcpyAsync(d_b, h, (size_t)N * heads * sizeof(float), cudaMemcpyHostToDevice, st);
    deltanet_synth_fill(h, (size_t)N * heads, 3);
    cudaMemcpyAsync(d_a, h, (size_t)N * heads * sizeof(float), cudaMemcpyHostToDevice, st);
    deltanet_synth_fill(h, state_floats, 4);
    cudaMemcpyAsync(d_s, h, state_floats * sizeof(float), cudaMemcpyHostToDevice, st);
    cudaStreamSynchronize(st);   // 'h' reused per fill; ensure each copy is consumed
    free(h);

    #define TQ_BENCH_CHUNK_LAUNCH()                                                        \
        (nstripe > 0                                                                       \
         ? launch_deltanet_chunk_hs(ck, nstripe, g, d_c, d_s, d_qkv, d_z, d_b, d_a,         \
               l->d_linear_A_log, l->d_linear_dt_bias, l->d_linear_norm,                   \
               N, heads, key_heads, D, g_qwen.eps, g_qwen.stream)                          \
         : launch_deltanet_chunk(ck, d_c, d_s, d_qkv, d_z, d_b, d_a,                        \
               l->d_linear_A_log, l->d_linear_dt_bias, l->d_linear_norm,                   \
               N, heads, key_heads, D, g_qwen.eps, g_qwen.stream))
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    // warmup
    TQ_BENCH_CHUNK_LAUNCH();
    k_tq_linear_decode_core_gated<<<heads, D, 0, g_qwen.stream>>>(
        d_c, d_s, d_qkv, d_z, d_b, d_a, l->d_linear_A_log, l->d_linear_dt_bias,
        l->d_linear_norm, heads, key_heads, D, g_qwen.eps);
    cudaStreamSynchronize(g_qwen.stream);

    cudaEventRecord(e0, g_qwen.stream);
    for (int it = 0; it < iters; it++)
        TQ_BENCH_CHUNK_LAUNCH();
    cudaEventRecord(e1, g_qwen.stream);
    cudaEventSynchronize(e1);
    float ms_c = 0.0f; cudaEventElapsedTime(&ms_c, e0, e1);

    cudaEventRecord(e0, g_qwen.stream);
    for (int it = 0; it < iters; it++)
        for (int t = 0; t < N; t++)
            k_tq_linear_decode_core_gated<<<heads, D, 0, g_qwen.stream>>>(
                d_c + (size_t)t * value_dim, d_s,
                d_qkv + (size_t)t * conv_dim, d_z + (size_t)t * value_dim,
                d_b + (size_t)t * heads, d_a + (size_t)t * heads,
                l->d_linear_A_log, l->d_linear_dt_bias, l->d_linear_norm,
                heads, key_heads, D, g_qwen.eps);
    cudaEventRecord(e1, g_qwen.stream);
    cudaEventSynchronize(e1);
    float ms_p = 0.0f; cudaEventElapsedTime(&ms_p, e0, e1);

    #undef TQ_BENCH_CHUNK_LAUNCH
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaFree(d_qkv); cudaFree(d_z); cudaFree(d_b); cudaFree(d_a); cudaFree(d_s); cudaFree(d_c);
    if (ms_chunk) *ms_chunk = ms_c / iters;
    if (ms_pertoken) *ms_pertoken = ms_p / iters;
    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? 0 : -7;
}

extern "C" int qwn_deltanet_chunk_bench(int layer, int N, int ck, int iters,
                                        float *ms_chunk, float *ms_pertoken) {
    return deltanet_chunk_bench_impl(layer, N, ck, 0, 0, iters, ms_chunk, ms_pertoken);
}

// Head-split bench: nstripe value-column stripes, g lanes/column.
extern "C" int qwn_deltanet_chunk_bench_hs(int layer, int N, int ck, int nstripe, int g,
                                           int iters, float *ms_chunk, float *ms_pertoken) {
    return deltanet_chunk_bench_impl(layer, N, ck, nstripe, g, iters, ms_chunk, ms_pertoken);
}

// Run the batched verify forward. tok/parent/depth/pos arrays length N (DFS order,
// parent[i]<i, parent[0]==-1). Fills d_spec_logits [N x V]. Returns 0 on success.
// ---- round-input staging (file scope: CUDA-graph replays re-read these) ----
static int s_tok[TQ_SPEC_MAX_N], s_par[TQ_SPEC_MAX_N], s_pos[TQ_SPEC_MAX_N];
static int s_order[TQ_SPEC_MAX_N], s_doff[TQ_SPEC_MAX_N + 1];
static int s_anc[TQ_SPEC_MAX_N][TQ_SPEC_MAX_N];
static int s_segs[TQ_SEGS_NSEGS_SLOT + 1];

// HOST-only staging of one round's verify inputs (no CUDA calls): fills the
// static staging arrays + the commit-side host state (pub mask, n_segs,
// state-skip demotions). Returns n_depths. Used by both the eager verify (which
// then uploads from the statics) and the graph-replay path (upload nodes inside
// the captured graph re-read the statics).
static int spec_stage_round(const int *tok, const int *parent, const int *depth,
                            const int *posv, int N, int use_batch_recur) {
    int n_depths = 0;
    for (int i = 0; i < N; i++) { s_tok[i] = tok[i]; s_par[i] = parent[i]; s_pos[i] = posv[i]; }
    if (use_batch_recur) {
        int maxd = 0;
        for (int i = 0; i < N; i++) if (depth[i] > maxd) maxd = depth[i];
        int pos = 0;
        for (int d = 0; d <= maxd; d++) {
            s_doff[d] = pos;
            for (int i = 0; i < N; i++) if (depth[i] == d) s_order[pos++] = i;
        }
        n_depths = maxd + 1;
        s_doff[n_depths] = N;
        g_spec_persist_nsegs = 0;
        g_spec_npub_nodes = 0;
        if (spec_seg_fuse_enabled() && g_qwen.linear_conv_kernel_dim <= 8) {
            int ks = g_qwen.linear_conv_kernel_dim;
            g_spec_persist_nsegs = spec_build_segments(s_par, N, ks, s_segs);
            const int *pub = s_segs + 2 * N + 1 + N * ks;
            for (int i = 0; i < N; i++) g_spec_pub_h[i] = pub[i];
            g_spec_npub_nodes = N;
        }
        s_segs[TQ_SEGS_NSEGS_SLOT] = g_spec_persist_nsegs;
        if (g_spec_persist_nsegs == 0) g_spec_state_skip_round = 0;
    } else {
        g_spec_state_skip_round = 0;
    }
    if (spec_tree_attn_enabled()) spec_build_anc(s_par, depth, N, s_anc);
    return n_depths;
}

static int run_spec_forward_n(const int *tok, const int *parent, const int *depth,
                              const int *posv, int N, int n_layers) {
    int ret = ensure_spec_buffers();
    if (ret != 0) return ret;
    if (getenv("TQ_SPEC_ZERO")) {   // debug: zero GEMV-output scratch (rule out non-16-M garbage)
        int q_m0 = g_qwen.nh * g_qwen.hd * 2, kv_m0 = g_qwen.nkv * g_qwen.hd;
        int dim0 = g_qwen.linear_value_head_dim;
        int conv_dim0 = (g_qwen.linear_num_key_heads * dim0) * 2 + g_qwen.linear_num_value_heads * dim0;
        int proj0 = q_m0 > conv_dim0 ? q_m0 : conv_dim0;
        int zd0 = (g_qwen.linear_num_value_heads * dim0) > kv_m0 ? (g_qwen.linear_num_value_heads * dim0) : kv_m0;
        cudaMemsetAsync(g_qwen.d_spec_qkv, 0, (size_t)N * proj0 * sizeof(float), g_qwen.stream);
        cudaMemsetAsync(g_qwen.d_spec_z, 0, (size_t)N * zd0 * sizeof(float), g_qwen.stream);
        cudaMemsetAsync(g_qwen.d_spec_kbuf, 0, (size_t)N * kv_m0 * sizeof(float), g_qwen.stream);
        cudaMemsetAsync(g_qwen.d_spec_bp, 0, (size_t)N * g_qwen.linear_num_value_heads * sizeof(float), g_qwen.stream);
        cudaMemsetAsync(g_qwen.d_spec_ap, 0, (size_t)N * g_qwen.linear_num_value_heads * sizeof(float), g_qwen.stream);
        cudaMemsetAsync(g_qwen.d_spec_core, 0, (size_t)N * proj0 * sizeof(float), g_qwen.stream);
        cudaMemsetAsync(g_qwen.d_spec_lo, 0, (size_t)N * g_qwen.H * sizeof(float), g_qwen.stream);
    }
    int H = g_qwen.H, I = g_qwen.I;
    int q_m = g_qwen.nh * g_qwen.hd * 2, kv_m = g_qwen.nkv * g_qwen.hd, attn_m = g_qwen.nh * g_qwen.hd;
    int dim = g_qwen.linear_value_head_dim;
    int heads = g_qwen.linear_num_value_heads, key_heads = g_qwen.linear_num_key_heads;
    int key_dim = key_heads * dim, value_dim = heads * dim;
    int conv_dim = key_dim * 2 + value_dim;
    int recur_f = spec_lin_recur_floats(), conv_f = spec_lin_conv_floats();
    int n_alloc = TQ_SPEC_MAX_N;            // per-layer node stride of the archives
    int lin_idx = 0, full_idx = 0;          // running linear / full-attn layer counters
    g_spec_attn_mma_round = attn_mma_active(posv[0] - 1);   // TQ_ATTN_MMA round gate
    g_spec_attn_base_host = posv[0] - 1;
    // root (parent<0) forks from the COMMITTED state (slot 0) at EVERY layer, not the
    // stale live leftover of the previous layer's last node (that contaminates the seed
    // and, through it, all branches). Default on; TQ_SPEC_NO_ROOTFIX restores old A/B.
    // Depth-major node grouping for the batched recurrence: a node forks from its
    // parent's THIS-layer state, so depth groups 0,1,2,.. must run in order, but nodes
    // within a depth are independent and run batched (one launch over the siblings).
    int use_batch_recur = !getenv("TQ_SPEC_NOBATCH_RECUR");
    int use_fused_delta = spec_fused_delta_enabled();
    // commit strategy for this round (mutually exclusive):
    //   state-skip > pointer-swap > copy.
    // Skip needs the seg-fused persistent path (dnprep archive + pub mask) and
    // always keeps the live state in the OWNED buffers (no half-rotation).
    g_spec_state_skip_round = 0;
    if (spec_state_skip_enabled() && spec_seg_fuse_enabled() && use_batch_recur &&
        !spec_recur_bf16() && spec_persist_enabled()) {
        spec_commit_unswap();
        g_spec_slot_base = 0;
        g_spec_commit_swap = 0;
        g_spec_state_skip_round = 1;
    } else if (spec_commit_swap_enabled() && use_batch_recur && N <= 8 &&
               !spec_recur_bf16() && TQ_SPEC_MAX_N >= 16) {
        // archive half-rotation for the pointer-swap commit; rounds that write both
        // halves (N > 8) must first re-materialize an archive-resident live state
        g_spec_slot_base = g_spec_slot_base ? 0 : 8;
        g_spec_commit_swap = 1;
    } else {
        spec_commit_unswap();
        g_spec_slot_base = 0;
        g_spec_commit_swap = 0;
    }
    int n_depths = spec_stage_round(tok, parent, depth, posv, N, use_batch_recur);
    if ((ret = ensure_spec_graph_params(N)) != 0) return ret;   // d_spec_tokens/positions
    if (use_batch_recur) {
        cudaMemcpyAsync(g_qwen.d_spec_group_ids, s_order, (size_t)N * sizeof(int),
                        cudaMemcpyHostToDevice, g_qwen.stream);
        cudaMemcpyAsync(g_qwen.d_spec_parent, s_par, (size_t)N * sizeof(int),
                        cudaMemcpyHostToDevice, g_qwen.stream);
        cudaMemcpyAsync(g_qwen.d_spec_depth_offsets, s_doff,
                        (size_t)(n_depths + 1) * sizeof(int), cudaMemcpyHostToDevice, g_qwen.stream);
        cudaMemcpyAsync(g_qwen.d_spec_segs, s_segs,
                        (size_t)(TQ_SEGS_NSEGS_SLOT + 1) * sizeof(int),
                        cudaMemcpyHostToDevice, g_qwen.stream);
    }
    int use_tree_attn = spec_tree_attn_enabled();
    if (use_tree_attn) {
        cudaMemcpyAsync(g_qwen.d_spec_positions, s_pos, (size_t)N * sizeof(int),
                        cudaMemcpyHostToDevice, g_qwen.stream);
        cudaMemcpyAsync(g_qwen.d_spec_anc, s_anc, (size_t)N * TQ_SPEC_MAX_N * sizeof(int),
                        cudaMemcpyHostToDevice, g_qwen.stream);
    } else if (g_qwen.kv_fp8) {
        // the fp8 fallback's archive dequant-gather reads positions on device
        cudaMemcpyAsync(g_qwen.d_spec_positions, s_pos, (size_t)N * sizeof(int),
                        cudaMemcpyHostToDevice, g_qwen.stream);
    }
    cudaMemcpyAsync(g_qwen.d_spec_tokens, s_tok, (size_t)N * sizeof(int),
                    cudaMemcpyHostToDevice, g_qwen.stream);
    int use_graph = !!getenv("TQ_SPEC_GRAPH") && n_layers >= g_qwen.L;
    if (use_graph) {
        cudaError_t eg = cudaStreamSynchronize(g_qwen.stream);
        if (eg != cudaSuccess) return -80;
        eg = cudaStreamBeginCapture(g_qwen.stream, cudaStreamCaptureModeGlobal);
        if (eg != cudaSuccess) return -81;
    }

    // embed all N node tokens (device-read token ids: graph-replay safe).
    for (int i = 0; i < N; i++)
        k_tq_embed_lookup_arr<<<(H + 255) / 256, 256, 0, g_qwen.stream>>>(
            g_qwen.d_spec_h + (size_t)i * H, g_qwen.d_embed, g_qwen.d_spec_tokens, i, H);

    #define TQ_PERSIST_ATTN_BASE (-1)
    int use_persist = spec_persist_enabled() && use_batch_recur;
    for (int L = 0; L < n_layers; L++) {
        tq_layer_t *l = &g_qwen.layers[L];
        rmap_hook_after_layer(L - 1, g_qwen.d_spec_h, N);
        if (use_persist && g_qwen.layer_types[L] == TQ_LAYER_LINEAR_ATTENTION &&
            spec_persist_layer_ok(l)) {
            if ((ret = ensure_spec_persist()) != 0) return ret;
            if (spec_fuse_layer_enabled() && !spec_split_gemv_enabled()) {
                if ((ret = run_spec_persist_fused(l, N, lin_idx, 0, n_depths)) != 0) return ret;
                lin_idx++;
                continue;
            }
            k_tq_qwen_rmsnorm_b_amax<<<dim3(8, N), 1024, 0, g_qwen.stream>>>(
                g_qwen.d_spec_norm, g_qwen.d_spec_h, l->d_input_ln, H, g_qwen.eps,
                g_spec_persist_amax);
            if ((ret = run_spec_persist_part(l, N, lin_idx, 0, n_depths)) != 0) return ret;
            k_tq_add_rmsnorm_b_amax_red<<<dim3(8, N), 1024, 0, g_qwen.stream>>>(
                g_qwen.d_spec_post, g_qwen.d_spec_resid, g_qwen.d_spec_h,
                g_spec_persist_out_partials, g_qwen.d_spec_nk_factors + TQ_SPEC_MAX_N,
                g_spec_persist_out_ks, l->d_post_ln, H, g_qwen.eps, g_spec_persist_amax);
            if (spec_split_gemv_enabled()) {
                if ((ret = run_spec_mlp_split(l, N)) != 0) return ret;
            } else {
                if ((ret = run_spec_persist_part(l, N, lin_idx, 1, n_depths)) != 0) return ret;
            }
            lin_idx++;
            continue;
        }
        if (use_persist && g_qwen.layer_types[L] != TQ_LAYER_LINEAR_ATTENTION &&
            spec_tree_attn_enabled() && spec_persist_attn_ok(l)) {
            if ((ret = ensure_spec_persist()) != 0) return ret;
            if (spec_fuse_layer_enabled() && !spec_split_gemv_enabled()) {
                if ((ret = run_spec_persist_fused(l, N, full_idx, 2, TQ_PERSIST_ATTN_BASE)) != 0) return ret;
                full_idx++;
                continue;
            }
            k_tq_qwen_rmsnorm_b_amax<<<dim3(8, N), 1024, 0, g_qwen.stream>>>(
                g_qwen.d_spec_norm, g_qwen.d_spec_h, l->d_input_ln, H, g_qwen.eps,
                g_spec_persist_amax);
            if ((ret = run_spec_persist_part(l, N, full_idx, 2, TQ_PERSIST_ATTN_BASE)) != 0) return ret;
            k_tq_add_rmsnorm_b_amax_red<<<dim3(8, N), 1024, 0, g_qwen.stream>>>(
                g_qwen.d_spec_post, g_qwen.d_spec_resid, g_qwen.d_spec_h,
                g_spec_persist_out_partials, g_qwen.d_spec_nk_factors + TQ_SPEC_MAX_N,
                g_spec_persist_out_ks, l->d_post_ln, H, g_qwen.eps, g_spec_persist_amax);
            if (spec_split_gemv_enabled()) {
                if ((ret = run_spec_mlp_split(l, N)) != 0) return ret;
            } else {
                if ((ret = run_spec_persist_part(l, N, full_idx, 1, n_depths)) != 0) return ret;
            }
            full_idx++;
            continue;
        }
        k_tq_qwen_rmsnorm_b<<<dim3(8, N), 1024, 0, g_qwen.stream>>>(   // input rmsnorm (batched)
            g_qwen.d_spec_norm, g_qwen.d_spec_h, l->d_input_ln, H, g_qwen.eps);

        if (g_qwen.layer_types[L] == TQ_LAYER_LINEAR_ATTENTION) {
            spec_pack(g_qwen.d_spec_norm, H, N);
            if ((ret = spec_gemv(&l->linear_in_qkv, g_qwen.d_spec_qkv, N)) != 0) return ret;
            if ((ret = spec_gemv(&l->linear_in_z, g_qwen.d_spec_z, N)) != 0) return ret;
            if ((ret = spec_gemv(&l->linear_in_b, g_qwen.d_spec_bp, N)) != 0) return ret;
            if ((ret = spec_gemv(&l->linear_in_a, g_qwen.d_spec_ap, N)) != 0) return ret;
            if (spec_pf_stream_enabled()) {   // overlap DeltaNet with downstream weight prefetch
                const tq_qmma_weight_t *pf_ws[3] = {&l->linear_out, &l->mlp_gate, &l->mlp_up};
                spec_pf_stream_launch(pf_ws, 3);
            }
            // per-(layer,node) DeltaNet archive (lin_idx slab): the recurrence runs IN
            // PLACE here so the committed live state (l->d_linear_*) stays untouched (the
            // root forks from it, correct at every layer) and the 1-read commit reads the
            // accepted leaf's slab.
            float *recur_base = g_qwen.d_spec_recur + ((size_t)lin_idx * n_alloc + g_spec_slot_base) * recur_f;
            float *conv_base  = g_qwen.d_spec_conv  + ((size_t)lin_idx * n_alloc + g_spec_slot_base) * conv_f;
            if (spec_recur_bf16()) return -90;   // bf16 archive requires the persistent path
            g_spec_state_skip_round = 0;         // legacy path writes all slabs, no dnprep archive
            if (use_batch_recur && use_fused_delta) {
                cudaMemsetAsync(g_qwen.d_spec_depth_done, 0, sizeof(int), g_qwen.stream);
                if (spec_linear_deltanet_fused(l, g_qwen.d_spec_core, recur_base,
                                              conv_base, g_qwen.d_spec_qkv,
                                              g_qwen.d_spec_bp, g_qwen.d_spec_ap,
                                              g_qwen.d_spec_group_ids, g_qwen.d_spec_parent, N,
                                              recur_f, conv_dim, value_dim, conv_f,
                                              l->d_linear_recurrent_state,
                                              l->d_linear_conv_state,
                                              g_qwen.d_spec_depth_offsets, n_depths,
                                              g_qwen.d_spec_depth_done) != 0) return -89;
                {
                    size_t shB = (size_t)dim * sizeof(float);
                    k_tq_linear_decode_gated_norm_b<<<dim3(heads, N), dim, shB, g_qwen.stream>>>(
                        g_qwen.d_spec_core, g_qwen.d_spec_z, l->d_linear_norm, heads, dim, g_qwen.eps,
                        g_qwen.d_spec_group_ids, value_dim);
                }
            } else if (use_batch_recur) {
                cudaMemsetAsync(g_qwen.d_spec_depth_done, 0, 2 * sizeof(int), g_qwen.stream);
                k_tq_linear_conv_update_fused_tree<<<dim3((conv_dim + 255) / 256, N), 256, 0, g_qwen.stream>>>(
                    g_qwen.d_spec_qkv, conv_base, g_qwen.d_spec_qkv, l->d_linear_conv1d,
                    conv_dim, g_qwen.linear_conv_kernel_dim, g_qwen.d_spec_group_ids, g_qwen.d_spec_parent,
                    conv_f, l->d_linear_conv_state,
                    g_qwen.d_spec_depth_offsets, n_depths, g_qwen.d_spec_depth_done);
                if (spec_linear_core_fused_tree(l, g_qwen.d_spec_core, recur_base, g_qwen.d_spec_qkv,
                                                g_qwen.d_spec_bp, g_qwen.d_spec_ap,
                                                g_qwen.d_spec_group_ids, g_qwen.d_spec_parent, N,
                                                recur_f, conv_dim, value_dim,
                                                l->d_linear_recurrent_state,
                                                g_qwen.d_spec_depth_offsets, n_depths,
                                                g_qwen.d_spec_depth_done + 1) != 0) return -88;
                {
                    size_t shB = (size_t)dim * sizeof(float);
                    k_tq_linear_decode_gated_norm_b<<<dim3(heads, N), dim, shB, g_qwen.stream>>>(
                        g_qwen.d_spec_core, g_qwen.d_spec_z, l->d_linear_norm, heads, dim, g_qwen.eps,
                        g_qwen.d_spec_group_ids, value_dim);
                }
            } else {
                for (int i = 0; i < N; i++) {   // per-node fallback (A/B: TQ_SPEC_NOBATCH_RECUR)
                    float *r_dst = recur_base + (size_t)i * recur_f, *c_dst = conv_base + (size_t)i * conv_f;
                    const float *r_src = (parent[i] >= 0) ? recur_base + (size_t)parent[i] * recur_f
                                                          : l->d_linear_recurrent_state;
                    const float *c_src = (parent[i] >= 0) ? conv_base + (size_t)parent[i] * conv_f
                                                          : l->d_linear_conv_state;
                    cudaMemcpyAsync(l->d_linear_recurrent_state, r_src, (size_t)recur_f * sizeof(float), cudaMemcpyDeviceToDevice, g_qwen.stream);
                    cudaMemcpyAsync(l->d_linear_conv_state, c_src, (size_t)conv_f * sizeof(float), cudaMemcpyDeviceToDevice, g_qwen.stream);
                    float *qkv_i = g_qwen.d_spec_qkv + (size_t)i * conv_dim;
                    k_tq_linear_conv_update<<<(conv_dim + 255) / 256, 256, 0, g_qwen.stream>>>(
                        qkv_i, l->d_linear_conv_state, qkv_i, l->d_linear_conv1d, conv_dim, g_qwen.linear_conv_kernel_dim);
                    spec_linear_core(l, g_qwen.d_spec_core + (size_t)i * value_dim, qkv_i,
                                     g_qwen.d_spec_z + (size_t)i * value_dim,
                                     g_qwen.d_spec_bp + (size_t)i * heads, g_qwen.d_spec_ap + (size_t)i * heads);
                    cudaMemcpyAsync(r_dst, l->d_linear_recurrent_state, (size_t)recur_f * sizeof(float), cudaMemcpyDeviceToDevice, g_qwen.stream);
                    cudaMemcpyAsync(c_dst, l->d_linear_conv_state, (size_t)conv_f * sizeof(float), cudaMemcpyDeviceToDevice, g_qwen.stream);
                }
            }
            spec_pack(g_qwen.d_spec_core, value_dim, N);
            if ((ret = spec_gemv(&l->linear_out, g_qwen.d_spec_lo, N)) != 0) return ret;
            lin_idx++;
        } else {   // full attention
            spec_pack(g_qwen.d_spec_norm, H, N);
            if ((ret = spec_gemv(&l->q_proj, g_qwen.d_spec_qkv, N)) != 0) return ret;
            if ((ret = spec_gemv(&l->k_proj, g_qwen.d_spec_kbuf, N)) != 0) return ret;
            if ((ret = spec_gemv(&l->v_proj, g_qwen.d_spec_z, N)) != 0) return ret;
            // per-node attention in DFS order: write node i's k/v to slot pos[i], attend
            // its branch, then archive that slot's k/v (full_idx slab) so the 1-read commit
            // can re-place the accepted path's KV (DFS leaves the shared slots holding the
            // last branch, not the accepted one).
            float *k_arch = g_qwen.d_spec_k_arch + (size_t)full_idx * n_alloc * kv_m;
            float *v_arch = g_qwen.d_spec_v_arch + (size_t)full_idx * n_alloc * kv_m;
            if (use_tree_attn) {
                // batched: k/v go straight to the archive (never the live cache); the
                // attend kernel reads tree positions from the node's ancestor chain.
                k_tq_spec_attn_kv_prep<<<dim3(g_qwen.nkv, N), g_qwen.hd, 0, g_qwen.stream>>>(
                    k_arch, v_arch, g_qwen.d_spec_kbuf, g_qwen.d_spec_z,
                    l->d_k_norm, g_qwen.d_spec_positions, g_qwen.nkv, g_qwen.hd, kv_m,
                    g_qwen.eps, g_qwen.rope_theta, NULL,
                    NULL, NULL, NULL, NULL, NULL, NULL, g_qwen.kv_q4);
                if (g_spec_attn_mma_round) {
                    launch_attn_tree_mma(g_qwen.d_spec_core, g_qwen.d_spec_qkv, l->d_q_norm,
                        l->d_k_cache, l->d_v_cache,
                        l->d_k_cache8, l->d_k_scale, l->d_v_cache8, l->d_v_scale,
                        l->d_k_cache4, l->d_k_q4s,
                        k_arch, v_arch, g_qwen.d_spec_positions, g_qwen.d_spec_anc,
                        N, q_m, attn_m, -1, g_spec_attn_base_host, g_spec_attn_mma_round);
                } else {
                if (ensure_attn_scores() != 0) return -97;
    k_tq_spec_attn_tree<<<dim3(g_qwen.nh, N), g_qwen.hd, 0, g_qwen.stream>>>(
                    g_qwen.d_spec_core, g_qwen.d_spec_qkv, l->d_q_norm,
                    l->d_k_cache, l->d_v_cache,
                    l->d_k_cache8, l->d_k_scale, l->d_v_cache8, l->d_v_scale,
                    l->d_k_cache4, l->d_k_q4s,
                    k_arch, v_arch,
                    g_qwen.d_spec_positions, g_qwen.d_spec_anc,
                    g_qwen.nh, g_qwen.nkv, g_qwen.hd, kv_m, q_m, attn_m,
                    g_qwen.eps, g_qwen.rope_theta, -1,
        g_qwen.d_attn_scores, g_qwen.max_seq);
                }
            } else {
            for (int i = 0; i < N; i++) {
                if (ensure_attn_scores() != 0) return -97;
    k_tq_full_attn_decode<<<g_qwen.nh, g_qwen.hd, 0, g_qwen.stream>>>(
                    g_qwen.d_spec_core + (size_t)i * attn_m, g_qwen.d_spec_qkv + (size_t)i * q_m,
                    g_qwen.d_spec_kbuf + (size_t)i * kv_m, g_qwen.d_spec_z + (size_t)i * kv_m,
                    l->d_q_norm, l->d_k_norm, l->d_k_cache, l->d_v_cache,
                    l->d_k_cache8, l->d_k_scale, l->d_v_cache8, l->d_v_scale,
                    l->d_k_cache4, l->d_k_q4s,
                    posv[i], g_qwen.nh, g_qwen.nkv, g_qwen.hd, g_qwen.eps, g_qwen.rope_theta,
        g_qwen.d_attn_scores, g_qwen.max_seq);
                if (g_qwen.kv_fp8) {
                    // fp8/q4 cache -> fp32 archive: dequant gather (the memcpy below
                    // would copy raw codes); same kernel as the _arr fallback.
                    k_tq_spec_archive_kv<<<(kv_m + 255) / 256, 256, 0, g_qwen.stream>>>(
                        k_arch + (size_t)i * kv_m, NULL, g_qwen.d_spec_positions, i, kv_m,
                        l->d_k_cache8, l->d_k_scale, g_qwen.nkv, g_qwen.hd,
                        l->d_k_cache4, l->d_k_q4s);
                    k_tq_spec_archive_kv<<<(kv_m + 255) / 256, 256, 0, g_qwen.stream>>>(
                        v_arch + (size_t)i * kv_m, NULL, g_qwen.d_spec_positions, i, kv_m,
                        l->d_v_cache8, l->d_v_scale, g_qwen.nkv, g_qwen.hd,
                        NULL, NULL);
                    continue;
                }
                cudaMemcpyAsync(k_arch + (size_t)i * kv_m, l->d_k_cache + (size_t)posv[i] * kv_m,
                                (size_t)kv_m * sizeof(float), cudaMemcpyDeviceToDevice, g_qwen.stream);
                cudaMemcpyAsync(v_arch + (size_t)i * kv_m, l->d_v_cache + (size_t)posv[i] * kv_m,
                                (size_t)kv_m * sizeof(float), cudaMemcpyDeviceToDevice, g_qwen.stream);
            }
            }
            spec_pack(g_qwen.d_spec_core, attn_m, N);
            if ((ret = spec_gemv(&l->o_proj, g_qwen.d_spec_lo, N)) != 0) return ret;
            full_idx++;
            if (use_persist && spec_persist_mlp_ok(l)) {
                // full-attn layer MLP half through the persistent kernel
                if ((ret = ensure_spec_persist()) != 0) return ret;
                k_tq_add_rmsnorm_b_amax<<<dim3(8, N), 1024, 0, g_qwen.stream>>>(
                    g_qwen.d_spec_post, g_qwen.d_spec_resid, g_qwen.d_spec_h, g_qwen.d_spec_lo,
                    l->d_post_ln, H, g_qwen.eps, g_spec_persist_amax);
                if ((ret = run_spec_persist_part(l, N, 0, 1, n_depths)) != 0) return ret;
                continue;
            }
        }
        // fused attn/recurrence residual + post-attn norm (saves 1 kernel launch)
        k_tq_add_rmsnorm_b<<<dim3(8, N), 1024, 0, g_qwen.stream>>>(
            g_qwen.d_spec_post, g_qwen.d_spec_resid, g_qwen.d_spec_h, g_qwen.d_spec_lo,
            l->d_post_ln, H, g_qwen.eps);
        spec_pack(g_qwen.d_spec_post, H, N);
        if ((ret = spec_gemv(&l->mlp_gate, g_qwen.d_spec_gate, N)) != 0) return ret;
        if ((ret = spec_gemv(&l->mlp_up, g_qwen.d_spec_up, N)) != 0) return ret;
        k_tq_silu_mul_b<<<dim3((I + 255) / 256, N), 256, 0, g_qwen.stream>>>(
            g_qwen.d_spec_mlp, g_qwen.d_spec_gate, g_qwen.d_spec_up, I);
        spec_pack(g_qwen.d_spec_mlp, I, N);
        if ((ret = spec_gemv(&l->mlp_down, g_qwen.d_spec_lo, N)) != 0) return ret;
        k_tq_add_vec_b<<<dim3((H + 255) / 256, N), 256, 0, g_qwen.stream>>>(   // final residual (batched)
            g_qwen.d_spec_h, g_qwen.d_spec_resid, g_qwen.d_spec_lo, H);
    }
#undef TQ_PERSIST_ATTN_BASE
    rmap_hook_after_layer(n_layers - 1, g_qwen.d_spec_h, N);
    if (n_layers < g_qwen.L) {   // debug: stop early, leave pre-final-norm hidden in d_spec_h
        cudaError_t e = cudaStreamSynchronize(g_qwen.stream);
        return e == cudaSuccess ? 0 : -79;
    }
    // final norm + lm_head (batched) -> d_spec_logits [N x V].
    k_tq_qwen_rmsnorm_b<<<dim3(8, N), 1024, 0, g_qwen.stream>>>(
        g_qwen.d_spec_norm, g_qwen.d_spec_h, g_qwen.d_norm, H, g_qwen.eps);
    if (g_qwen.tie_word_embeddings) {
        for (int i = 0; i < N; i++)
            k_tq_lm_head_bf16<<<g_qwen.V, 256, 0, g_qwen.stream>>>(
                g_qwen.d_spec_logits + (size_t)i * g_qwen.V, g_qwen.d_spec_norm + (size_t)i * H,
                g_qwen.d_embed, H, g_qwen.V);
    } else {
        spec_pack(g_qwen.d_spec_norm, H, N);
        if ((ret = spec_gemv(&g_qwen.lm_head, g_qwen.d_spec_logits, N)) != 0) return ret;
    }
    if (use_graph) {
        cudaGraph_t graph;
        cudaError_t eg = cudaStreamEndCapture(g_qwen.stream, &graph);
        if (eg != cudaSuccess || !graph) {
            if (graph) cudaGraphDestroy(graph);
            fprintf(stderr, "spec forward graph capture failed: %s\n", cudaGetErrorString(eg));
            return -82;
        }
        cudaGraphExec_t exec;
        eg = cudaGraphInstantiate(&exec, graph, 0);
        cudaGraphDestroy(graph);
        if (eg != cudaSuccess) {
            fprintf(stderr, "spec forward graph instantiate failed: %s\n", cudaGetErrorString(eg));
            return -83;
        }
        eg = cudaGraphLaunch(exec, g_qwen.stream);
        cudaGraphExecDestroy(exec);
        if (eg != cudaSuccess) {
            fprintf(stderr, "spec forward graph launch failed: %s\n", cudaGetErrorString(eg));
            return -84;
        }
    } else {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) { fprintf(stderr, "spec forward launch failed: %s\n", cudaGetErrorString(err)); return -78; }
    }
    return 0;
}

// --- temperature sampling for the spec round (apples-to-apples vs vLLM) ---
//
// Lossless speculative SAMPLING over the deterministically-built tree: at each
// path node the children are treated as an ordered proposal LIST. Child c is
// accepted with probability p_tau(tok_c) / (1 - sum of previously REJECTED
// siblings' p) -- the exact sequential-residual scheme for deterministic
// proposals (each rejection zeroes that token's mass and renormalizes). When
// every child is rejected (or none exist), the bonus emission is drawn from
// the residual distribution via Gumbel-max with rejected tokens masked. The
// tau->0 limit reproduces the greedy argmax descent exactly (handled as an
// explicit argmax mode so the gate can compare bit-for-bit).
//
// Counter-based RNG (splitmix64): every uniform is a pure function of
// (seed, round_ctr, step, slot) -- replayable, stream-order independent.
// out layout: [0]=m, [1+j]=path node j, [1+TQ_SPEC_MAX_N+j]=accepted token j.
__global__ void __launch_bounds__(1024, 1)
k_tq_spec_sample_descend(const float *logits, int V, int nv,
                         const int *dtok, const int *dpar,
                         float inv_tau, float logp_min, uint64_t seed, uint32_t round_ctr,
                         int *out) {
    __shared__ float s_red[32];
    __shared__ float s_max, s_sum;
    __shared__ int s_cur, s_m, s_done, s_pick;
    __shared__ float s_removed;
    __shared__ int s_child[TQ_SPEC_MAX_N];
    __shared__ float s_gbest[32];
    __shared__ int s_gbestv[32];
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int argmax_mode = !(inv_tau < 1.0e6f);
    if (tid == 0) { s_cur = 0; s_m = 0; s_done = 0; }
    __syncthreads();

    for (int step = 0; step < TQ_SPEC_MAX_N && !s_done; step++) {
        int cur = s_cur;
        const float *lg = logits + (size_t)cur * V;
        // pass 1: max
        float mx = -3.4e38f;
        for (int v = tid; v < V; v += blockDim.x) { float x = lg[v]; if (x > mx) mx = x; }
        for (int o = 16; o > 0; o >>= 1) { float t = __shfl_down_sync(0xffffffffu, mx, o); if (t > mx) mx = t; }
        if (lane == 0) s_red[warp] = mx;
        __syncthreads();
        if (warp == 0) {
            float t = (lane < 32) ? s_red[lane] : -3.4e38f;
            for (int o = 16; o > 0; o >>= 1) { float u = __shfl_down_sync(0xffffffffu, t, o); if (u > t) t = u; }
            if (lane == 0) s_max = t;
        }
        __syncthreads();
        float vmax = s_max;
        float sum = 0.0f;
        if (!argmax_mode) {
            // pass 2: sum of exp((l - max)/tau) over the min-p-kept set. Tokens with
            // p/p_max < min_p (i.e. (l-max)/tau < log(min_p)) are dropped here so the
            // residual sampler below normalizes over the truncated head only -- this
            // is what stops temp>0 from drawing the inflated low-prob tail (the
            // multilingual-junk tokens seen mid-generation).
            for (int v = tid; v < V; v += blockDim.x) {
                float lp = (lg[v] - vmax) * inv_tau;
                if (lp >= logp_min) sum += __expf(lp);
            }
            for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, o);
            if (lane == 0) s_red[warp] = sum;
            __syncthreads();
            if (warp == 0) {
                float t = (lane < 32) ? s_red[lane] : 0.0f;
                for (int o = 16; o > 0; o >>= 1) t += __shfl_down_sync(0xffffffffu, t, o);
                if (lane == 0) s_sum = t;
            }
            __syncthreads();
        }
        // children of cur in draft order (nv <= 16: thread 0 scans)
        if (tid == 0) {
            int nc = 0;
            for (int ci = 0; ci < nv; ci++)
                if (dpar[ci] == cur) s_child[nc++] = ci;
            for (int j = nc; j < TQ_SPEC_MAX_N; j++) s_child[j] = -1;
            s_pick = -1;
            s_removed = 0.0f;
            for (int j = 0; j < nc; j++) {
                int ci = s_child[j];
                int tk = dtok[ci];
                if (tk < 0 || tk >= V) continue;
                if (argmax_mode) {
                    // exact greedy: accept iff the child IS the argmax token
                    // (resolved below by the Gumbel/argmax pass over V; defer)
                    continue;
                }
                float lp_tk = (lg[tk] - vmax) * inv_tau;
                if (lp_tk < logp_min) continue;   // child below min-p: q(tk)=0, not a candidate
                float p = __expf(lp_tk) / s_sum;
                float denom = 1.0f - s_removed;
                float pa = (denom > 0.0f) ? fminf(1.0f, p / denom) : 0.0f;
                float r = tq_u01(seed, round_ctr, (uint32_t)step, (uint32_t)j);
                if (r < pa) { s_pick = ci; break; }
                s_removed += p;
            }
        }
        __syncthreads();
        if (!argmax_mode && s_pick >= 0) {
            // child accepted: emit its token, descend
            if (tid == 0) {
                int m = s_m;
                out[1 + m] = s_cur;
                out[1 + TQ_SPEC_MAX_N + m] = dtok[s_pick];
                s_m = m + 1;
                s_cur = s_pick;
            }
            __syncthreads();
            continue;
        }
        // bonus emission: argmax (greedy mode) or Gumbel-max over the residual
        // (rejected children masked); then stop. In greedy mode a child match
        // still descends (exact argmax-walk semantics).
        float best = -3.4e38f; int bestv = -1;
        for (int v = tid; v < V; v += blockDim.x) {
            int masked = 0;
            if (!argmax_mode) {
                for (int j = 0; j < TQ_SPEC_MAX_N && j < nv; j++) {
                    int ci = s_child[j];
                    if (ci >= 0 && dpar[ci] == cur && dtok[ci] == v) { masked = 1; break; }
                }
            }
            if (masked) continue;
            float x = (lg[v] - vmax) * (argmax_mode ? 1.0f : inv_tau);
            if (!argmax_mode) {
                if (x < logp_min) continue;   // min-p: drop tail tokens from the residual draw
                float u = tq_u01(seed, round_ctr, (uint32_t)(64 + step), (uint32_t)v);
                x += -__logf(-__logf(fmaxf(u, 1.0e-20f)));   // Gumbel(0,1)
            }
            if (x > best || (x == best && (unsigned)v < (unsigned)bestv)) { best = x; bestv = v; }
        }
        for (int o = 16; o > 0; o >>= 1) {
            float tb = __shfl_down_sync(0xffffffffu, best, o);
            int tv = __shfl_down_sync(0xffffffffu, bestv, o);
            if (tb > best || (tb == best && (unsigned)tv < (unsigned)bestv)) { best = tb; bestv = tv; }
        }
        if (lane == 0) { s_gbest[warp] = best; s_gbestv[warp] = bestv; }
        __syncthreads();
        if (warp == 0) {
            float tb = (lane < 32) ? s_gbest[lane] : -3.4e38f;
            int tv = (lane < 32) ? s_gbestv[lane] : -1;
            for (int o = 16; o > 0; o >>= 1) {
                float ub = __shfl_down_sync(0xffffffffu, tb, o);
                int uv = __shfl_down_sync(0xffffffffu, tv, o);
                if (ub > tb || (ub == tb && (unsigned)uv < (unsigned)tv)) { tb = ub; tv = uv; }
            }
            if (lane == 0) { s_gbest[0] = tb; s_gbestv[0] = tv; }
        }
        __syncthreads();
        int emit = s_gbestv[0];
        if (argmax_mode) {
            // greedy semantics: if a child carries the argmax token, descend
            if (tid == 0) {
                int nxt = -1;
                for (int ci = 0; ci < nv; ci++)
                    if (dpar[ci] == cur && dtok[ci] == emit) { nxt = ci; break; }
                int m = s_m;
                out[1 + m] = s_cur;
                out[1 + TQ_SPEC_MAX_N + m] = emit;
                s_m = m + 1;
                if (nxt >= 0) s_cur = nxt; else s_done = 1;
            }
            __syncthreads();
            continue;
        }
        if (tid == 0) {
            int m = s_m;
            out[1 + m] = s_cur;
            out[1 + TQ_SPEC_MAX_N + m] = emit;
            s_m = m + 1;
            s_done = 1;
        }
        __syncthreads();
    }
    if (tid == 0) out[0] = s_m;
}

// Min-p nucleus floor for the spec sampler. Drops candidates whose probability is
// below min_p * p_max BEFORE the residual draw, so temp>0 cannot emit the inflated
// low-prob tail (the multilingual-junk tokens). 0 disables (exact-but-leaky old
// behavior). Default 0.02; tune via TQ_MIN_P. argmax/greedy (temp==0) is untouched.
static float spec_min_p(void) {
    static float mp = -1.0f;
    if (mp < 0.0f) {
        const char *e = getenv("TQ_MIN_P");
        mp = e ? (float)atof(e) : 0.02f;
        if (mp < 0.0f) mp = 0.0f;
        if (mp > 1.0f) mp = 1.0f;
    }
    return mp;
}

// Host wrapper: sampled (or tau->0 greedy) descent over the verified tree.
// Replaces spec_argmax + the host argmax walk when TQ_TEMP > 0.
static int spec_sample_descend(int nv, const int *dtok, const int *dpar,
                               float temp, uint64_t seed, uint32_t round_ctr,
                               int *path, int *accepted, int *out_m) {
    if (!g_qwen.d_spec_sample_io) {
        if (cudaMalloc(&g_qwen.d_spec_sample_io,
                       (1 + 2 * TQ_SPEC_MAX_N) * sizeof(int) + 2 * TQ_SPEC_MAX_N * sizeof(int)) != cudaSuccess)
            return -1;
    }
    int *d_io = g_qwen.d_spec_sample_io;
    int *d_tok = d_io + 1 + 2 * TQ_SPEC_MAX_N;
    int *d_par = d_tok + TQ_SPEC_MAX_N;
    if (cudaMemcpyAsync(d_tok, dtok, nv * sizeof(int), cudaMemcpyHostToDevice, g_qwen.stream) != cudaSuccess)
        return -2;
    if (cudaMemcpyAsync(d_par, dpar, nv * sizeof(int), cudaMemcpyHostToDevice, g_qwen.stream) != cudaSuccess)
        return -2;
    float inv_tau = (temp > 1.0e-6f) ? (1.0f / temp) : 2.0e6f;
    float mp = spec_min_p();
    float logp_min = (mp > 0.0f) ? logf(mp) : -3.0e38f;
    k_tq_spec_sample_descend<<<1, 1024, 0, g_qwen.stream>>>(
        g_qwen.d_spec_logits, g_qwen.V, nv, d_tok, d_par, inv_tau, logp_min, seed, round_ctr, d_io);
    int h_io[1 + 2 * TQ_SPEC_MAX_N];
    if (cudaMemcpyAsync(h_io, d_io, sizeof(h_io), cudaMemcpyDeviceToHost, g_qwen.stream) != cudaSuccess)
        return -3;
    if (cudaStreamSynchronize(g_qwen.stream) != cudaSuccess) return -4;
    int m = h_io[0];
    if (m < 1 || m > TQ_SPEC_MAX_N) return -5;
    for (int j = 0; j < m; j++) {
        path[j] = h_io[1 + j];
        accepted[j] = h_io[1 + TQ_SPEC_MAX_N + j];
    }
    *out_m = m;
    return 0;
}

// Per-node full-vocab argmax of d_spec_logits into host out_argmax[N].
static int spec_argmax(int N, int *out_argmax) {
    for (int i = 0; i < N; i++) {
        k_tq_argmax_stage1<<<TQ_ARGMAX_BLOCKS, 256, 0, g_qwen.stream>>>(
            g_qwen.d_spec_argv, g_qwen.d_spec_argi, g_qwen.d_spec_logits + (size_t)i * g_qwen.V, g_qwen.V);
        k_tq_argmax_stage2<<<1, 1024, 0, g_qwen.stream>>>(
            g_qwen.d_spec_argv + TQ_ARGMAX_BLOCKS, g_qwen.d_spec_argi + TQ_ARGMAX_BLOCKS,
            g_qwen.d_spec_argv, g_qwen.d_spec_argi, TQ_ARGMAX_BLOCKS);
        if (cudaMemcpyAsync(&out_argmax[i], g_qwen.d_spec_argi + TQ_ARGMAX_BLOCKS, sizeof(int),
                            cudaMemcpyDeviceToHost, g_qwen.stream) != cudaSuccess) return -1;
    }
    return cudaStreamSynchronize(g_qwen.stream) == cudaSuccess ? 0 : -2;
}

// Validation entry: run the batched verify forward over N nodes and return per-node
// argmax (and optionally the per-node final-norm hidden). Used by mtp_verify_smoke
// to check the batched forward bit-matches the proven per-token replay.
// Copy the PRE-final-norm trunk hiddens of the last spec verify ([n x H] from
// d_spec_h). This is the exact tensor the MTP head consumes as its root input
// (snapshot_root semantics) -- used by the draft-head calibration dumper.
extern "C" int qwn_spec_copy_hidden_pre(float *out, int n) {
    if (!g_qwen.initialized || !g_qwen.d_spec_h || !out) return -1;
    if (n < 1 || n > TQ_SPEC_MAX_N) return -2;
    if (cudaStreamSynchronize(g_qwen.stream) != cudaSuccess) return -3;
    if (cudaMemcpy(out, g_qwen.d_spec_h, (size_t)n * g_qwen.H * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) return -4;
    return 0;
}

extern "C" int qwn_spec_forward_test(const int *tok, const int *parent, const int *depth,
                                     const int *posv, int N, int *out_argmax, float *out_hidden) {
    if (!g_qwen.initialized) return -1;
    if (N < 1 || N > TQ_SPEC_MAX_N) return -2;
    int ret = run_spec_forward_n(tok, parent, depth, posv, N, g_qwen.L);
    if (ret != 0) return ret;
    if (out_hidden) {
        if (cudaStreamSynchronize(g_qwen.stream) != cudaSuccess) return -3;
        cudaMemcpy(out_hidden, g_qwen.d_spec_norm, (size_t)N * g_qwen.H * sizeof(float), cudaMemcpyDeviceToHost);
    }
    return spec_argmax(N, out_argmax);
}
