/* qwentin.h -- the C ABI of libforward_qwen.so, generated from src/engine/ by tools/gen_qwentin_h.py; do not
 * edit. The int entry points return 0 or a negative error code, except the getters (sizes, counts, modes);
 * the definitions in src/engine/ say which. */
#ifndef QWENTIN_H
#define QWENTIN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* engine/quant_bfrag.cuh */
int qwn_quant_bfrag_check(int K, int N, float *max_abs_diff_bytes, float *max_abs_diff_scale);
int qwn_quant_bfrag_bench(int K, int N, int iters, double *t_ref_ms, double *t_fast_ms, double *gbps_ref, double *gbps_fast);

/* engine/loader.cuh */
int qwn_rmap_hook_add(int layer, float scale, const float *P, const float *A_minus_I, const float *mu, const float *shift);
int qwn_rmap_hook_count(void);
int qwn_rmap_set_scale(int layer, float scale);
int qwn_rmap_apply_debug(int layer, int rows, float *io);
int qwn_ot_hook_add(int layer, float scale, const float *P, const float *A_k_minus_I, const float *mu_H, const float *mean_shift);
int qwn_ot_hook_count(void);
int qwn_ot_set_scale(int layer, float scale);
int qwn_ot_apply_debug(int layer, int rows, float *io);
void qwn_free(void);
int qwn_gpu_pack_e2m3_test(const uint16_t *h_bf16, int M, int K, uint8_t *h_out_dA, float *h_out_scale);
int qwn_probe(const char *path);
int qwn_init(const char *path);
int qwn_hidden_size(void);
int qwn_intermediate_size(void);
int qwn_vocab_size(void);
int qwn_num_layers(void);
int qwn_num_attention_heads(void);
int qwn_num_key_value_heads(void);
int qwn_head_dim(void);
int qwn_max_seq(void);
int qwn_layer_type(int layer);
size_t qwn_device_bytes(void);
int qwn_non_quant_count(void);
int qwn_qmma_count(void);
int qwn_f32_count(void);
int qwn_last_argmax_id(void);
float qwn_last_argmax_logit(void);
float qwn_last_topk_margin(void);
int qwn_last_topk(int *ids_out, float *vals_out, int k);
int qwn_nk_lmhead_sanity(int nvar, float *max_col_diff);
int qwn_copy_last_norm(float *out, int max_count);
int qwn_sparse_gemv_sanity(int layer, int proj_id, const float *x_host, float *out_host);
int qwn_sparse_gemv_residual(int layer, int proj_id, const float *x_host, float *out_host);
int qwn_reset_state(void);

/* engine/forward_single.cuh */
int qwn_delta_state_save(int slot);
int qwn_delta_state_restore(int slot);
int qwn_gemv_nk_check(int which, float *max_abs_diff, double *rel, int *argmax_pertoken, int *argmax_nk);
int qwn_gemv_nk_bench(int which, int N, int iters, double *t_nk_ms, double *t_pertoken_ms);
int qwn_wide_gemm_check(int which, int N, float *max_abs_diff);
int qwn_wide_gemm_bench(int which, int N, int iters, double *t_wide_ms, double *t_pertoken_ms);
int qwn_wide_attn_check(int layer, int base_pos, int N, float *max_abs_diff);
int qwn_debug_embed_input_norm(int token_id, int layer, float *out, int max_count);
int qwn_debug_linear_in_a(int token_id, int layer, float *out, int max_count);
int qwn_debug_linear_projection_qmma(int token_id, int layer, int projection_id, float *out, int max_count);
int qwn_debug_linear_in_a_qmma(int token_id, int layer, float *out, int max_count);
int qwn_debug_linear_qkv_conv_first(int token_id, int layer, float *out, int max_count);
int qwn_debug_linear_first_core(int token_id, int layer, float *out, int max_count);
int qwn_debug_linear_first_output(int token_id, int layer, float *out, int max_count);
int qwn_debug_linear_layer_output(int token_id, int layer, float *out, int max_count);
int qwn_debug_forward_layers(int token_id, int n_layers, float *out, int max_count);
int qwn_debug_forward_final_norm(int token_id, float *out, int max_count);
int qwn_debug_forward_argmax(int token_id, float *logit_out);
int qwn_forward_first(int token_id);
int qwn_debug_decode_layers(int token_id, int pos, int n_layers, float *out, int max_count);
int qwn_decode(int token_id, int pos);

/* engine/mtp.cuh */
int qwn_has_mtp(void);
int qwn_mtp_reset(void);
int qwn_copy_mtp_debug(int which, float *out, int max_count);
int qwn_copy_last_mtp_logits(float *out, int max_count);
int qwn_mtp_step(int token_id, int pos, int *topk_ids, float *topk_vals, int k);
int qwn_mtp_step_hidden(const float *hidden, int token_id, int pos, int *topk_ids, float *topk_vals, int k);
int qwn_mtp_advance(int token_id, int pos);
int qwn_spec_set_root_from(int node_idx);
int qwn_mtp_advance_from_spec(int node_idx, int token, int pos);
int qwn_mtp_snapshot_root(void);
int qwn_mtp_tree_build(int seed_token, int pos, int depth, int k, float tau, int max_nodes, int *out_tokens, int *out_parents, int *out_depths, float *out_margins);
int qwn_mtp_path_topk(const int *fed_tokens, int n_fed, int pos, int k, int *out_topk_ids, float *out_topk_vals);

/* engine/spec_persistent_host.cuh */
int qwn_layout_bench(int Mt, int Kt, int ks, int iters, float *ms_out);
int qwn_spec_persist_clocks(long long *out);
int qwn_spec_persist_dump(int *out32);
int qwn_spec_pack_check(const float *x_host, int K, int N, int *b_diff, float *s_diff, float *f_diff);

/* engine/deltanet_prefill.cuh */
int qwn_deltanet_chunk_check(int layer, int N, float *max_abs_diff);
int qwn_deltanet_chunk_check_ck(int layer, int N, int ck, float *max_abs_diff);
int qwn_deltanet_chunk_check_hs(int layer, int N, int ck, int nstripe, int g, float *max_abs_diff);
int qwn_deltanet_chunk_bench(int layer, int N, int ck, int iters, float *ms_chunk, float *ms_pertoken);
int qwn_deltanet_chunk_bench_hs(int layer, int N, int ck, int nstripe, int g, int iters, float *ms_chunk, float *ms_pertoken);
int qwn_spec_copy_hidden_pre(float *out, int n);
int qwn_spec_forward_test(const int *tok, const int *parent, const int *depth, const int *posv, int N, int *out_argmax, float *out_hidden);

/* engine/spec_driver.cuh */
int qwn_capture_spec_forward(const int *tok, const int *parent, const int *depth, const int *posv, int N);
int qwn_spec_forward_graph(const int *tok, const int *parent, const int *depth, const int *posv, int N, int *out_argmax, float *out_hidden);
int qwn_spec_commit(const int *path_nodes, const int *path_pos, int m);
int qwn_spec_lmhead_from_node(int node_idx);
int qwn_set_sampling(float temp, unsigned long long seed);
int qwn_kv_clear_rows(int lo, int hi);
int qwn_kv_dump(int layer, int lo, int hi, float *k_out, float *v_out);
int qwn_set_commit_stop(const int *toks, int n);
int qwn_spec_round(int seed, int base_pos, int depth, int k, float tau, int max_nodes, int *out_chain, int *io_state);
int qwn_mtp_advance_wave(const int *node_idx, const int *toks, int pos0, int m);

/* engine/exl3.cuh */
int qwn_wide_linear_test(int layer, int which, const float *h_x, int N, float *h_out, int *out_M, int *out_K, int *out_kbits);
int qwn_wide_linear_bench(int layer, int which, int N, int iters, float *out_ms, long long *out_wbytes);
int qwn_prefill_wide(const int *tokens, int base_pos, int n, int *out_argmax);
int qwn_wide_advance_trunk(const int *toks, int pos0, int m);
int qwn_wide_prefill_graph_bench(int n, int iters, double *eager_ms, double *graph_ms);

/* engine/batched_decode.cuh */
int qwn_batched_init(int N, int bat_ctx);
int qwn_batched_free(void);
int qwn_batched_load_client(int c, int upto_pos);
int qwn_batched_decode_step(const int *tokens, const int *positions, int N, int *out_argmax);
int qwn_batched_logits(int c, float *out_host);

/* engine/paged_attn.cuh */
int qwn_paged_attn_st(void);

/* engine/paged_t4.cuh */
int qwn_paged_t4_test_encode(const uint16_t *f16_rows, int n, uint8_t *codes_out, uint16_t *scale_out);
int qwn_paged_init(int max_slots, int num_blocks, int page);
int qwn_paged_free(void);
int qwn_paged_reset_slot(int slot);
int qwn_paged_load_client(int slot, int upto_pos);

/* engine/parking.cuh */
size_t qwn_paged_slot_bytes(int upto_pos);
size_t qwn_paged_token_bytes(void);
int qwn_paged_dstate(void);
size_t qwn_paged_state_bytes(void);
int qwn_park_ring_set(int on);
int qwn_paged_slot_save(int slot, int upto_pos, void *host, size_t host_bytes);
int qwn_paged_slot_restore(int slot, int upto_pos, const void *host, size_t host_bytes);

/* engine/paged_graph.cuh */
int qwn_graph_stats(long *hits, long *captures, long *evictions, int *cached, double *capture_ms);
int qwn_paged_decode_step(const int *tokens, const int *slot_ids, const int *positions, int N, int *out_argmax);
int qwn_paged_stats(int *free_blocks, int *total_blocks, int *page, int *max_blocks_per_seq);
int qwn_paged_logits_sink(float *dev_out, int cap_cols, int head_cols);
int qwn_gdn_prefill_ab(int layer, int N, int ds, int mode, int iters, int pos0, int *mism, float *ms);
int qwn_paged_prefill_batch(const int *tokens, const int *col_slot, const int *col_pos, const int *seg_slot, const int *seg_off, const int *seg_len, const int *seg_final, int K, int T, int *out_seed);

/* engine/paged_spec.cuh */
int qwn_paged_verify(const int *tokens, const int *col_slot, const int *col_pos, const int *seg_slot, const int *seg_off, const int *seg_len, int K, int T, int *out_argmax);
int qwn_paged_commit(const int *seg_idx, const int *n_accept, int M);
const float *qwn_paged_verify_hidden(int *out_T);
int qwn_paged_mtp_init(int on);
int qwn_paged_mtp_draft(const int *slot_ids, const int *positions, const int *seeds, int N, int K, int *drafts);

/* engine/chunked_prefill.cuh */
int qwn_prefill_chunk(const int *tokens, int base_pos, int n, int *out_argmax);
int qwn_spec_dbg(int seed, int pos, int maxL, float *out_spec, float *out_dec);
int qwn_capture_decode(int token_id, int pos);
int qwn_decode_graph(int token_id, int pos);
int qwn_capture_decode_chain(int token_id, int pos, int max_tokens);
int qwn_bench_decode_graph_chain(int token_id, int pos, int steps, float *avg_ms_out, int *out_ids, int max_out, int *final_id_out, float *logit_out);
int qwn_bench_forward_first(int token_id, int warmup, int iters, float *avg_ms_out, int *id_out, float *logit_out);
int qwn_bench_forward_first_breakdown(int token_id, int warmup, int iters, float *layers_ms_out, float *final_norm_ms_out, float *lm_head_ms_out, float *total_ms_out, int *id_out, float *logit_out);
int qwn_bench_forward_layer_breakdown(int token_id, int warmup, int iters, float *layer_ms_out, int max_layers);
int qwn_capture_forward_first(int token_id);
int qwn_forward_first_graph(void);
int qwn_bench_forward_first_graph(int token_id, int warmup, int iters, float *avg_ms_out, int *id_out, float *logit_out);

/* engine/tests.cuh */
int qwn_attn_mma_unit(const float *q_proj, const uint16_t *qnorm, const float *kcache, const float *vcache, const float *karch, const float *varch, const int *posv, const int *anc, int N, int base, int nh, int nkv, int hd, float eps, float rope_theta, int fp8, float *out_legacy, float *out_mma);

#ifdef __cplusplus
}
#endif
#endif
