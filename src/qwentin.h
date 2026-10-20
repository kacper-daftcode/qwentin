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
void qwn_free(void);
int qwn_tqf_blake2b256(const void *data, size_t n, uint8_t *out);
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
int qwn_reset_state(void);

/* engine/decode_oracle.cuh */
int qwn_wide_gemm_check(int which, int N, float *max_abs_diff);
int qwn_wide_gemm_bench(int which, int N, int iters, double *t_wide_ms, double *t_pertoken_ms);
int qwn_decode(int token_id, int pos);

/* engine/deltanet_prefill.cuh */
int qwn_deltanet_chunk_check(int layer, int N, float *max_abs_diff);
int qwn_deltanet_chunk_check_ck(int layer, int N, int ck, float *max_abs_diff);
int qwn_deltanet_chunk_check_hs(int layer, int N, int ck, int nstripe, int g, float *max_abs_diff);
int qwn_deltanet_chunk_bench(int layer, int N, int ck, int iters, float *ms_chunk, float *ms_pertoken);
int qwn_deltanet_chunk_bench_hs(int layer, int N, int ck, int nstripe, int g, int iters, float *ms_chunk, float *ms_pertoken);

/* engine/exl3.cuh */
int qwn_wide_linear_test(int layer, int which, const float *h_x, int N, float *h_out, int *out_M, int *out_K, int *out_kbits);
int qwn_wide_linear_bench(int layer, int which, int N, int iters, float *out_ms, long long *out_wbytes);

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

#ifdef __cplusplus
}
#endif
#endif
