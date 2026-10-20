// per-column embedding lookup, row gather and RMSNorm kernels of the batched forward.
// Part of the forward_qwen.cu unity build: included in order, not a standalone header.

// Embedding lookup per column: out[n] = embed[tokens[n]] ([N x H]).
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
