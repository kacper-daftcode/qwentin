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

/* engine/config.cuh: the engine's configuration, one field per TQ_* switch (named in each comment; docs/FLAGS.md).
 * qwn_config_from_env fills it the way qwn_init reads the environment; qwn_init_with takes it instead. */
typedef struct qwn_config {
    /* the configuration a deployment sets */
    int ctx;                    /* TQ_CTX */
    int kv_q4;                  /* TQ_KV_Q4 */
    int kv_v4;                  /* TQ_KV_V4 */
    int kv_t4;                  /* TQ_KV_T4 */
    int dstate;                 /* TQ_DSTATE_BF16 */
    int embed_host;             /* TQ_EMBED_HOST */
    int embed_fp8;              /* TQ_EMBED_FP8 */
    int no_mtp;                 /* TQ_NO_MTP */
    int draft_vocab;            /* TQ_DRAFT_VOCAB */
    int verify_tqf;             /* TQ_VERIFY_TQF */
    int pdl;                    /* TQ_PDL */
    int wide_attn_mma;          /* TQ_WIDE_ATTN_MMA */
    int paged_attn_mma;         /* TQ_PAGED_ATTN_MMA */
    int gdn_prefill;            /* TQ_GDN_PREFILL */
    /* off-switches and opt-in experiments */
    int t4_attn_f16;            /* TQ_T4_ATTN_F16 */
    int paged_attn_st;          /* TQ_PAGED_ATTN_ST */
    int graph;                  /* TQ_GRAPH */
    int gemm_fork;              /* TQ_GEMM_FORK */
    int gemm_fork_maxn;         /* TQ_GEMM_FORK_MAXN */
    int exl3_wide_ng16;         /* TQ_EXL3_WIDE_NG16 */
    int exl3_bf16_r4;           /* TQ_EXL3_BF16_R4 */
    int pattn_gqa;              /* TQ_PATTN_GQA */
    int wave_max;               /* TQ_WAVE_MAX */
    int wide_attn_qrows;        /* TQ_WIDE_ATTN_QROWS */
    int kv_t4_async;            /* TQ_KV_T4_ASYNC */
    int paged_topk_pages;       /* TQ_PAGED_TOPK_PAGES */
    int paged_topk_min_ctx;     /* TQ_PAGED_TOPK_MIN_CTX */
    /* pins for tools and tests */
    int park_ring;              /* TQ_PARK_RING */
    int qmma_sf_k_split;        /* TQ_QMMA_SF_K_SPLIT */
    int paged_gqa_split;        /* TQ_PAGED_GQA_SPLIT */
    /* diagnostics */
    int pattn_debug;            /* TQ_PATTN_DEBUG */
    int exl3_debug;             /* TQ_EXL3_DEBUG */
    int graph_debug;            /* TQ_GRAPH_DEBUG */
    int paged_topk_debug;       /* TQ_PAGED_TOPK_DEBUG */
    const char *kv_t4_codebook;   /* TQ_KV_T4_CODEBOOK: a file of 4096 fp32 entries; NULL = the built-in codebook */
} qwn_config;

/* engine/config.cuh */
void qwn_config_defaults(qwn_config *c);
void qwn_config_from_env(qwn_config *c);
void qwn_config_get(qwn_config *c);

/* engine/quant_bfrag.cuh */
int qwn_quant_bfrag_check(int K, int N, float *max_abs_diff_bytes, float *max_abs_diff_scale);
int qwn_quant_bfrag_bench(int K, int N, int iters, double *t_ref_ms, double *t_fast_ms, double *gbps_ref, double *gbps_fast);

/* engine/loader.cuh */
void qwn_free(void);
int qwn_tqf_blake2b256(const void *data, size_t n, uint8_t *out);
int qwn_gpu_pack_e2m3_test(const uint16_t *h_bf16, int M, int K, uint8_t *h_out_dA, float *h_out_scale);
int qwn_probe(const char *path);
int qwn_init_with(const char *path, const qwn_config *cfg);
int qwn_init(const char *path);
int qwn_hidden_size(void);
int qwn_intermediate_size(void);
int qwn_vocab_size(void);
int qwn_draft_vocab_size(void);
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
