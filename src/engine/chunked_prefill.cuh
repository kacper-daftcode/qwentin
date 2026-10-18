// single-stream chunked prefill, decode-graph capture and benches.
// Part of the forward_qwen.cu unity build: included in order, not a standalone header.

// ============================ Chunked prefill ============================
// A prompt chunk of n consecutive tokens is a degenerate verify tree: a CHAIN
// (parent = i-1, depth = i). The batched verify machinery then does everything
// a prefill needs in ONE pass over the weights: the depth-gated fused DeltaNet
// advances the recurrent state sequentially through the chunk, the tree
// attention provides causality (ancestors = the chunk prefix), and the 1-read
// commit lands the whole chain's DeltaNet/conv/KV state. Weights are read once
// per 16 tokens instead of once per token (~11x prefill speedup).
// Leaves the last token's pre-final-norm hidden in d_spec_h[node n-1] and its
// argmax in *out_argmax (for the greedy continuation / MTP root).
extern "C" int qwn_prefill_chunk(const int *tokens, int base_pos, int n, int *out_argmax) {
    if (!g_qwen.initialized) return -1;
    if (n < 1 || n > TQ_SPEC_MAX_N) return -2;
    hist_append(tokens, n);
    int tok[TQ_SPEC_MAX_N], par[TQ_SPEC_MAX_N], dep[TQ_SPEC_MAX_N], posv[TQ_SPEC_MAX_N];
    int path[TQ_SPEC_MAX_N], path_pos[TQ_SPEC_MAX_N];
    for (int i = 0; i < n; i++) {
        tok[i] = tokens[i];
        par[i] = i - 1;
        dep[i] = i;
        posv[i] = base_pos + 1 + i;
        path[i] = i;
        path_pos[i] = base_pos + 1 + i;
    }
    int ret = run_spec_forward_n(tok, par, dep, posv, n, g_qwen.L);
    if (ret != 0) return ret;
    ret = qwn_spec_commit(path, path_pos, n);
    if (ret != 0) return ret;
    if (out_argmax) {
        // argmax of the LAST chunk position only (the greedy continuation)
        k_tq_argmax_stage1<<<TQ_ARGMAX_BLOCKS, 256, 0, g_qwen.stream>>>(
            g_qwen.d_spec_argv, g_qwen.d_spec_argi,
            g_qwen.d_spec_logits + (size_t)(n - 1) * g_qwen.V, g_qwen.V);
        k_tq_argmax_stage2<<<1, 1024, 0, g_qwen.stream>>>(
            g_qwen.d_spec_argv + TQ_ARGMAX_BLOCKS, g_qwen.d_spec_argi + TQ_ARGMAX_BLOCKS,
            g_qwen.d_spec_argv, g_qwen.d_spec_argi, TQ_ARGMAX_BLOCKS);
        if (cudaMemcpyAsync(out_argmax, g_qwen.d_spec_argi + TQ_ARGMAX_BLOCKS, sizeof(int),
                            cudaMemcpyDeviceToHost, g_qwen.stream) != cudaSuccess) return -3;
    }
    return cudaStreamSynchronize(g_qwen.stream) == cudaSuccess ? 0 : -4;
}

// Debug: run the batched forward (N=1 seed) AND the per-token decode for `maxL`
// layers, return both pre-final-norm hiddens [H] so a smoke can binary-search which
// layer the batched vs per-token drift begins. Requires slot 0 = committed state.
extern "C" int qwn_spec_dbg(int seed, int pos, int maxL, float *out_spec, float *out_dec) {
    if (!g_qwen.initialized) return -1;
    int tok[1] = {seed}, par[1] = {-1}, dep[1] = {0}, pv[1] = {pos + 1};
    int ret = qwn_delta_state_restore(0);   // start from the CLEAN committed state (slot 0)
    if (ret != 0) return ret;
    ret = run_spec_forward_n(tok, par, dep, pv, 1, maxL);
    if (ret != 0) return ret;
    if (cudaStreamSynchronize(g_qwen.stream) != cudaSuccess) return -2;
    cudaMemcpy(out_spec, g_qwen.d_spec_h, (size_t)g_qwen.H * sizeof(float), cudaMemcpyDeviceToHost);
    ret = qwn_delta_state_restore(0);
    if (ret != 0) return ret;
    ret = run_decode_layers_to_debug_x(seed, pos + 1, maxL);
    if (ret != 0) return ret;
    if (cudaStreamSynchronize(g_qwen.stream) != cudaSuccess) return -3;
    cudaMemcpy(out_dec, g_qwen.d_debug_x, (size_t)g_qwen.H * sizeof(float), cudaMemcpyDeviceToHost);
    return 0;
}

static int run_decode_argmax_device_graph_params(void) {
    int ret = run_decode_layers_to_debug_x_graph_params(g_qwen.L);
    if (ret != 0) return ret;
    ret = run_final_norm_device();
    if (ret != 0) return -5;
    ret = run_lm_head_argmax_device();
    if (ret != 0) return -6;
    k_tq_decode_graph_advance<<<1, 1, 0, g_qwen.stream>>>(
        g_qwen.d_decode_token, g_qwen.d_decode_pos, g_qwen.d_decode_step,
        g_qwen.d_decode_generated, g_qwen.decode_graph_max_tokens,
        g_qwen.d_debug_arg_ids + TQ_ARGMAX_BLOCKS);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -7;
    return 0;
}

static int capture_decode_graph(int token_id, int pos, int max_tokens) {
    if (max_tokens <= 0) max_tokens = 1;
    if (g_qwen.decode_graph_ready && g_qwen.decode_graph_exec &&
        g_qwen.decode_graph_max_tokens == max_tokens) return 0;
    destroy_decode_graph();
    int gen_ret = ensure_decode_generated(max_tokens);
    if (gen_ret != 0) return gen_ret;
    g_qwen.decode_graph_max_tokens = max_tokens;
    int ret = set_decode_graph_params(token_id, pos);
    if (ret != 0) return ret;

    /* Allocate all lazy scratch outside capture. This mutates decode state; callers
       that need semantic generation should reset and prefill after capture. */
    ret = run_decode_argmax_device_graph_params();
    if (ret != 0) return ret;
    cudaError_t err = cudaStreamSynchronize(g_qwen.stream);
    if (err != cudaSuccess) return -20;

    err = cudaStreamBeginCapture(g_qwen.stream, cudaStreamCaptureModeGlobal);
    if (err != cudaSuccess) {
        fprintf(stderr, "decode cudaStreamBeginCapture failed: %s\n", cudaGetErrorString(err));
        return -21;
    }
    for (int i = 0; i < max_tokens; i++) {
        ret = run_decode_argmax_device_graph_params();
        if (ret != 0) {
            cudaStreamEndCapture(g_qwen.stream, &g_qwen.decode_graph);
            destroy_decode_graph();
            return ret;
        }
    }
    err = cudaStreamEndCapture(g_qwen.stream, &g_qwen.decode_graph);
    if (err != cudaSuccess) {
        fprintf(stderr, "decode cudaStreamEndCapture failed: %s\n", cudaGetErrorString(err));
        destroy_decode_graph();
        return -22;
    }
    err = cudaGraphInstantiate(&g_qwen.decode_graph_exec, g_qwen.decode_graph, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "decode cudaGraphInstantiate failed: %s\n", cudaGetErrorString(err));
        destroy_decode_graph();
        return -23;
    }
    g_qwen.decode_graph_ready = 1;
    return 0;
}

extern "C" int qwn_capture_decode(int token_id, int pos) {
    return capture_decode_graph(token_id, pos, 1);
}

extern "C" int qwn_decode_graph(int token_id, int pos) {
    if (!g_qwen.decode_graph_ready || !g_qwen.decode_graph_exec) {
        int ret = capture_decode_graph(token_id, pos, 1);
        if (ret != 0) return ret;
    }
    int ret = set_decode_graph_params(token_id, pos);
    if (ret != 0) return ret;
    cudaError_t err = cudaGraphLaunch(g_qwen.decode_graph_exec, g_qwen.stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "decode cudaGraphLaunch failed: %s\n", cudaGetErrorString(err));
        return -2;
    }
    return copy_forward_argmax_result(NULL);
}

extern "C" int qwn_capture_decode_chain(int token_id, int pos, int max_tokens) {
    return capture_decode_graph(token_id, pos, max_tokens);
}

extern "C" int qwn_bench_decode_graph_chain(int token_id, int pos, int steps,
                                            float *avg_ms_out, int *out_ids, int max_out,
                                            int *final_id_out, float *logit_out) {
    if (steps <= 0) return -1;
    int ret = capture_decode_graph(token_id, pos, steps);
    if (ret != 0) return ret;
    ret = set_decode_graph_params(token_id, pos);
    if (ret != 0) return ret;
    cudaError_t err = cudaStreamSynchronize(g_qwen.stream);
    if (err != cudaSuccess) return -2;
    cudaEvent_t start, stop;
    err = cudaEventCreate(&start);
    if (err != cudaSuccess) return -3;
    err = cudaEventCreate(&stop);
    if (err != cudaSuccess) {
        cudaEventDestroy(start);
        return -4;
    }
    cudaEventRecord(start, g_qwen.stream);
    err = cudaGraphLaunch(g_qwen.decode_graph_exec, g_qwen.stream);
    if (err != cudaSuccess) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -5;
    }
    cudaEventRecord(stop, g_qwen.stream);
    err = cudaEventSynchronize(stop);
    if (err != cudaSuccess) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -6;
    }
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (avg_ms_out) *avg_ms_out = elapsed_ms / (float)steps;
    int copy_count = steps < max_out ? steps : max_out;
    if (out_ids && copy_count > 0) {
        err = cudaMemcpy(out_ids, g_qwen.d_decode_generated, (size_t)copy_count * sizeof(int),
                         cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) return -7;
    }
    int h_id = copy_forward_argmax_result(logit_out);
    if (h_id < 0) return h_id;
    if (final_id_out) *final_id_out = h_id;
    return copy_count;
}

extern "C" int qwn_bench_forward_first(int token_id, int warmup, int iters,
                                        float *avg_ms_out, int *id_out, float *logit_out) {
    if (iters <= 0) {
        fprintf(stderr, "qwn_bench_forward_first: iters must be > 0\n");
        return -1;
    }
    if (warmup < 0) warmup = 0;

    for (int i = 0; i < warmup; i++) {
        int ret = run_forward_argmax_device(token_id);
        if (ret != 0) return ret;
    }
    cudaError_t err = cudaStreamSynchronize(g_qwen.stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "bench warmup synchronize failed: %s\n", cudaGetErrorString(err));
        return -2;
    }

    cudaEvent_t start, stop;
    err = cudaEventCreate(&start);
    if (err != cudaSuccess) return -3;
    err = cudaEventCreate(&stop);
    if (err != cudaSuccess) {
        cudaEventDestroy(start);
        return -4;
    }

    cudaEventRecord(start, g_qwen.stream);
    for (int i = 0; i < iters; i++) {
        int ret = run_forward_argmax_device(token_id);
        if (ret != 0) {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return ret;
        }
    }
    cudaEventRecord(stop, g_qwen.stream);
    err = cudaEventSynchronize(stop);
    if (err != cudaSuccess) {
        fprintf(stderr, "bench event synchronize failed: %s\n", cudaGetErrorString(err));
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -5;
    }
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    int h_id = copy_forward_argmax_result(logit_out);
    if (h_id < 0) return h_id;
    if (avg_ms_out) *avg_ms_out = elapsed_ms / (float)iters;
    if (id_out) *id_out = h_id;
    return 0;
}

extern "C" int qwn_bench_forward_first_breakdown(int token_id, int warmup, int iters,
                                                 float *layers_ms_out,
                                                 float *final_norm_ms_out,
                                                 float *lm_head_ms_out,
                                                 float *total_ms_out,
                                                 int *id_out,
                                                 float *logit_out) {
    if (iters <= 0) return -1;
    if (warmup < 0) warmup = 0;

    /* Prewarm once so lazy scratch allocation is outside timing. */
    int ret = run_forward_argmax_device(token_id);
    if (ret != 0) return ret;
    for (int i = 0; i < warmup; i++) {
        ret = run_forward_argmax_device(token_id);
        if (ret != 0) return ret;
    }
    cudaError_t err = cudaStreamSynchronize(g_qwen.stream);
    if (err != cudaSuccess) return -2;

    cudaEvent_t e0, e1, e2, e3;
    if (cudaEventCreate(&e0) != cudaSuccess) return -3;
    if (cudaEventCreate(&e1) != cudaSuccess) { cudaEventDestroy(e0); return -4; }
    if (cudaEventCreate(&e2) != cudaSuccess) { cudaEventDestroy(e0); cudaEventDestroy(e1); return -5; }
    if (cudaEventCreate(&e3) != cudaSuccess) {
        cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2); return -6;
    }

    float layers_sum = 0.0f, norm_sum = 0.0f, lm_sum = 0.0f, total_sum = 0.0f;
    for (int i = 0; i < iters; i++) {
        cudaEventRecord(e0, g_qwen.stream);
        ret = run_forward_layers_to_debug_x(token_id, g_qwen.L);
        if (ret != 0) goto fail;
        cudaEventRecord(e1, g_qwen.stream);
        ret = run_final_norm_device();
        if (ret != 0) goto fail;
        cudaEventRecord(e2, g_qwen.stream);
        ret = run_lm_head_argmax_device();
        if (ret != 0) goto fail;
        cudaEventRecord(e3, g_qwen.stream);
        err = cudaEventSynchronize(e3);
        if (err != cudaSuccess) { ret = -7; goto fail; }
        float layers_ms = 0.0f, norm_ms = 0.0f, lm_ms = 0.0f, total_ms = 0.0f;
        cudaEventElapsedTime(&layers_ms, e0, e1);
        cudaEventElapsedTime(&norm_ms, e1, e2);
        cudaEventElapsedTime(&lm_ms, e2, e3);
        cudaEventElapsedTime(&total_ms, e0, e3);
        layers_sum += layers_ms;
        norm_sum += norm_ms;
        lm_sum += lm_ms;
        total_sum += total_ms;
    }

    cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2); cudaEventDestroy(e3);
    {
        int h_id = copy_forward_argmax_result(logit_out);
        if (h_id < 0) return h_id;
        if (layers_ms_out) *layers_ms_out = layers_sum / (float)iters;
        if (final_norm_ms_out) *final_norm_ms_out = norm_sum / (float)iters;
        if (lm_head_ms_out) *lm_head_ms_out = lm_sum / (float)iters;
        if (total_ms_out) *total_ms_out = total_sum / (float)iters;
        if (id_out) *id_out = h_id;
    }
    return 0;

fail:
    cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2); cudaEventDestroy(e3);
    return ret;
}

extern "C" int qwn_bench_forward_layer_breakdown(int token_id, int warmup, int iters,
                                                 float *layer_ms_out, int max_layers) {
    if (!layer_ms_out || max_layers < g_qwen.L) return -1;
    if (iters <= 0) return -2;
    if (warmup < 0) warmup = 0;
    for (int i = 0; i < max_layers; i++) layer_ms_out[i] = 0.0f;

    int ret = run_forward_layers_to_debug_x(token_id, g_qwen.L);
    if (ret != 0) return ret;
    for (int i = 0; i < warmup; i++) {
        ret = run_forward_layers_to_debug_x(token_id, g_qwen.L);
        if (ret != 0) return ret;
    }
    cudaError_t err = cudaStreamSynchronize(g_qwen.stream);
    if (err != cudaSuccess) return -3;

    cudaEvent_t start, stop;
    if (cudaEventCreate(&start) != cudaSuccess) return -4;
    if (cudaEventCreate(&stop) != cudaSuccess) {
        cudaEventDestroy(start);
        return -5;
    }

    for (int iter = 0; iter < iters; iter++) {
        int prep = ensure_debug_vectors(0);
        if (prep != 0) { ret = prep; goto fail_layer; }
        k_tq_embed_lookup<<<(g_qwen.H + 255) / 256, 256, 0, g_qwen.stream>>>(
            g_qwen.d_debug_x, g_qwen.d_embed, token_id, g_qwen.H);
        for (int layer = 0; layer < g_qwen.L; layer++) {
            cudaEventRecord(start, g_qwen.stream);
            if (g_qwen.layer_types[layer] == TQ_LAYER_LINEAR_ATTENTION) {
                ret = run_linear_layer_from_current_to_layer_out(layer);
            } else if (g_qwen.layer_types[layer] == TQ_LAYER_FULL_ATTENTION) {
                ret = run_full_layer_from_current_to_layer_out(layer);
            } else {
                ret = -6;
            }
            if (ret != 0) goto fail_layer;
            err = cudaMemcpyAsync(g_qwen.d_debug_x, g_qwen.d_debug_layer_out,
                                  (size_t)g_qwen.H * sizeof(float),
                                  cudaMemcpyDeviceToDevice, g_qwen.stream);
            if (err != cudaSuccess) { ret = -7; goto fail_layer; }
            cudaEventRecord(stop, g_qwen.stream);
            err = cudaEventSynchronize(stop);
            if (err != cudaSuccess) { ret = -8; goto fail_layer; }
            float elapsed = 0.0f;
            cudaEventElapsedTime(&elapsed, start, stop);
            layer_ms_out[layer] += elapsed;
        }
    }
    for (int i = 0; i < g_qwen.L; i++) layer_ms_out[i] /= (float)iters;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return g_qwen.L;

fail_layer:
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ret;
}

static int capture_forward_graph(int token_id) {
    if (g_qwen.forward_graph_ready && g_qwen.forward_graph_token_id == token_id) return 0;
    destroy_forward_graph();

    /* Allocate all lazy scratch outside capture. */
    int ret = run_forward_argmax_device(token_id);
    if (ret != 0) return ret;
    cudaError_t err = cudaStreamSynchronize(g_qwen.stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "graph prewarm synchronize failed: %s\n", cudaGetErrorString(err));
        return -20;
    }

    err = cudaStreamBeginCapture(g_qwen.stream, cudaStreamCaptureModeGlobal);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaStreamBeginCapture failed: %s\n", cudaGetErrorString(err));
        return -21;
    }
    ret = run_forward_argmax_device(token_id);
    if (ret != 0) {
        cudaStreamEndCapture(g_qwen.stream, &g_qwen.forward_graph);
        destroy_forward_graph();
        return ret;
    }
    err = cudaStreamEndCapture(g_qwen.stream, &g_qwen.forward_graph);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaStreamEndCapture failed: %s\n", cudaGetErrorString(err));
        destroy_forward_graph();
        return -22;
    }
    err = cudaGraphInstantiate(&g_qwen.forward_graph_exec, g_qwen.forward_graph, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGraphInstantiate failed: %s\n", cudaGetErrorString(err));
        destroy_forward_graph();
        return -23;
    }
    g_qwen.forward_graph_ready = 1;
    g_qwen.forward_graph_token_id = token_id;
    return 0;
}

extern "C" int qwn_capture_forward_first(int token_id) {
    return capture_forward_graph(token_id);
}

extern "C" int qwn_forward_first_graph(void) {
    if (!g_qwen.forward_graph_ready || !g_qwen.forward_graph_exec) {
        fprintf(stderr, "qwn_forward_first_graph: graph is not captured\n");
        return -1;
    }
    cudaError_t err = cudaGraphLaunch(g_qwen.forward_graph_exec, g_qwen.stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGraphLaunch failed: %s\n", cudaGetErrorString(err));
        return -2;
    }
    return copy_forward_argmax_result(NULL);
}

extern "C" int qwn_bench_forward_first_graph(int token_id, int warmup, int iters,
                                             float *avg_ms_out, int *id_out, float *logit_out) {
    if (iters <= 0) return -1;
    int ret = capture_forward_graph(token_id);
    if (ret != 0) return ret;
    if (warmup < 0) warmup = 0;
    for (int i = 0; i < warmup; i++) {
        cudaError_t err = cudaGraphLaunch(g_qwen.forward_graph_exec, g_qwen.stream);
        if (err != cudaSuccess) return -2;
    }
    cudaError_t err = cudaStreamSynchronize(g_qwen.stream);
    if (err != cudaSuccess) return -3;

    cudaEvent_t start, stop;
    err = cudaEventCreate(&start);
    if (err != cudaSuccess) return -4;
    err = cudaEventCreate(&stop);
    if (err != cudaSuccess) {
        cudaEventDestroy(start);
        return -5;
    }
    cudaEventRecord(start, g_qwen.stream);
    for (int i = 0; i < iters; i++) {
        err = cudaGraphLaunch(g_qwen.forward_graph_exec, g_qwen.stream);
        if (err != cudaSuccess) {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return -6;
        }
    }
    cudaEventRecord(stop, g_qwen.stream);
    err = cudaEventSynchronize(stop);
    if (err != cudaSuccess) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -7;
    }
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    int h_id = copy_forward_argmax_result(logit_out);
    if (h_id < 0) return h_id;
    if (avg_ms_out) *avg_ms_out = elapsed_ms / (float)iters;
    if (id_out) *id_out = h_id;
    return 0;
}
