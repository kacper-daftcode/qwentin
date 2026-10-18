// unit-test entry points.
// Part of the forward_qwen.cu unity build: included in order, not a standalone header.

// ===== TQ_ATTN_MMA unit-test entry (gate B): legacy tree kernel vs MMA kernel on
// host-provided synthetic tensors (no model load; explicit dims). fp8=1 first
// quantizes the prefix to the E4M3 cache (the shipped TQ_KV_FP8 format) and runs
// the legacy fp8 tail vs the QMMA path. Returns 0 on success.
extern "C" int qwn_attn_mma_unit(const float *q_proj, const uint16_t *qnorm,
                                 const float *kcache, const float *vcache,
                                 const float *karch, const float *varch,
                                 const int *posv, const int *anc,
                                 int N, int base, int nh, int nkv, int hd,
                                 float eps, float rope_theta, int fp8,
                                 float *out_legacy, float *out_mma) {
    if (N < 1 || N > TQ_SPEC_MAX_N || hd != 256) return -1;
    int q_m = nh * hd * 2, kv_m = nkv * hd, attn_m = nh * hd;
    int rows = base + 1 > 0 ? base + 1 : 1;
    int sst = base + 32;
    float *d_q = NULL, *d_kc = NULL, *d_vc = NULL, *d_ka = NULL, *d_va = NULL;
    float *d_out = NULL, *d_scores = NULL, *d_ksc = NULL, *d_vsc = NULL;
    uint16_t *d_qn = NULL;
    uint8_t *d_k8 = NULL, *d_v8 = NULL;
    int *d_pos = NULL, *d_anc = NULL, *d_rows = NULL;
    int ret = -2;
    if (cudaMalloc(&d_q, (size_t)N * q_m * sizeof(float)) != cudaSuccess) goto done;
    if (cudaMalloc(&d_kc, (size_t)rows * kv_m * sizeof(float)) != cudaSuccess) goto done;
    if (cudaMalloc(&d_vc, (size_t)rows * kv_m * sizeof(float)) != cudaSuccess) goto done;
    if (cudaMalloc(&d_ka, (size_t)16 * kv_m * sizeof(float)) != cudaSuccess) goto done;
    if (cudaMalloc(&d_va, (size_t)16 * kv_m * sizeof(float)) != cudaSuccess) goto done;
    if (cudaMalloc(&d_out, (size_t)N * attn_m * sizeof(float)) != cudaSuccess) goto done;
    if (cudaMalloc(&d_scores, (size_t)nh * N * sst * sizeof(float)) != cudaSuccess) goto done;
    if (cudaMalloc(&d_qn, (size_t)hd * sizeof(uint16_t)) != cudaSuccess) goto done;
    if (cudaMalloc(&d_pos, (size_t)N * sizeof(int)) != cudaSuccess) goto done;
    if (cudaMalloc(&d_anc, (size_t)N * TQ_SPEC_MAX_N * sizeof(int)) != cudaSuccess) goto done;
    cudaMemcpy(d_q, q_proj, (size_t)N * q_m * sizeof(float), cudaMemcpyHostToDevice);
    if (base >= 0) {
        cudaMemcpy(d_kc, kcache, (size_t)rows * kv_m * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vc, vcache, (size_t)rows * kv_m * sizeof(float), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(d_ka, karch, (size_t)16 * kv_m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_va, varch, (size_t)16 * kv_m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qn, qnorm, (size_t)hd * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos, posv, (size_t)N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_anc, anc, (size_t)N * TQ_SPEC_MAX_N * sizeof(int), cudaMemcpyHostToDevice);
    if (fp8) {
        enum { TQ_UNIT_MAX_ROWS = 8192 };
        static int h_rows[2 * TQ_UNIT_MAX_ROWS];
        if (rows > TQ_UNIT_MAX_ROWS) { ret = -3; goto done; }
        if (cudaMalloc(&d_k8, (size_t)rows * kv_m) != cudaSuccess) goto done;
        if (cudaMalloc(&d_v8, (size_t)rows * kv_m) != cudaSuccess) goto done;
        if (cudaMalloc(&d_ksc, (size_t)rows * nkv * sizeof(float)) != cudaSuccess) goto done;
        if (cudaMalloc(&d_vsc, (size_t)rows * nkv * sizeof(float)) != cudaSuccess) goto done;
        if (cudaMalloc(&d_rows, (size_t)2 * rows * sizeof(int)) != cudaSuccess) goto done;
        for (int j = 0; j < rows; j++) { h_rows[2 * j] = j; h_rows[2 * j + 1] = j; }
        cudaMemcpy(d_rows, h_rows, (size_t)2 * rows * sizeof(int), cudaMemcpyHostToDevice);
        // quantize prefix rows through the shipped commit kernel (arch layout ==
        // [row][kv_m], exactly the cache layout here)
        k_tq_spec_commit_kv_fp8<<<dim3(nkv, rows), hd>>>(d_k8, d_ksc, d_v8, d_vsc,
                                                         d_kc, d_vc, d_rows, kv_m, nkv, hd);
    }
    // legacy reference (the shipped scalar path; fp8 variant when fp8=1)
    k_tq_spec_attn_tree<<<dim3(nh, N), hd>>>(
        d_out, d_q, d_qn, d_kc, d_vc,
        fp8 ? d_k8 : NULL, d_ksc, fp8 ? d_v8 : NULL, d_vsc,
        NULL, NULL,
        d_ka, d_va, d_pos, d_anc, nh, nkv, hd, kv_m, q_m, attn_m,
        eps, rope_theta, base, d_scores, sst);
    if (cudaDeviceSynchronize() != cudaSuccess) { ret = -4; goto done; }
    cudaMemcpy(out_legacy, d_out, (size_t)N * attn_m * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemset(d_out, 0, (size_t)N * attn_m * sizeof(float));
    {
        // exercise the production chunking (flash partials + merge when the base
        // crosses the chunk size); partial scratch allocated here, standalone-style
        int nchunks = attn_mma_nchunks(base, 0);
        float *d_part = NULL;
        if (cudaMalloc(&d_part, (size_t)nh * nchunks * TQ_AMM_PART_F * sizeof(float)) != cudaSuccess) {
            ret = -6; goto done;
        }
        dim3 grid(nh, nchunks);
        if (fp8)
            k_tq_spec_attn_tree_mma<2><<<grid, 256, TQ_ATTN_MMA_SMEM>>>(
                d_out, d_q, d_qn, d_kc, d_vc, d_k8, d_ksc, d_v8, d_vsc,
                NULL, NULL,
                d_ka, d_va, d_pos, d_anc, N, 0, nh, nkv, hd, kv_m, q_m, attn_m,
                eps, rope_theta, base, d_part);
        else
            k_tq_spec_attn_tree_mma<1><<<grid, 256, TQ_ATTN_MMA_SMEM>>>(
                d_out, d_q, d_qn, d_kc, d_vc, NULL, NULL, NULL, NULL,
                NULL, NULL,
                d_ka, d_va, d_pos, d_anc, N, 0, nh, nkv, hd, kv_m, q_m, attn_m,
                eps, rope_theta, base, d_part);
        if (nchunks > 1)
            k_tq_spec_attn_tree_mma_merge<<<dim3(nh, N), 256>>>(
                d_out, d_q, N, nh, hd, q_m, attn_m, nchunks, d_part);
        cudaError_t se = cudaDeviceSynchronize();
        cudaFree(d_part);
        if (se != cudaSuccess) { ret = -5; goto done; }
    }
    cudaMemcpy(out_mma, d_out, (size_t)N * attn_m * sizeof(float), cudaMemcpyDeviceToHost);
    ret = 0;
done:
    cudaFree(d_q); cudaFree(d_kc); cudaFree(d_vc); cudaFree(d_ka); cudaFree(d_va);
    cudaFree(d_out); cudaFree(d_scores); cudaFree(d_qn); cudaFree(d_pos); cudaFree(d_anc);
    cudaFree(d_k8); cudaFree(d_v8); cudaFree(d_ksc); cudaFree(d_vsc); cudaFree(d_rows);
    return ret;
}
