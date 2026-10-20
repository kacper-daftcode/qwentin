// includes, limits and file flags, the engine state (g_qwen), numeric helpers.
// Part of the forward_qwen.cu unity build: included in order, not a standalone header.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <thread>
#include <vector>

#define TQ_MAX_LAYERS 64
#define TQ_LAYER_LINEAR_ATTENTION 1
#define TQ_LAYER_FULL_ATTENTION 2
#define TQ_DTYPE_FP16 1
#define TQ_DTYPE_BF16 2
#define TQ_ARGMAX_BLOCKS 1024
// Hard CAP for sequence length (static host scratch, clamp for TQ_CTX).
// Runtime limit lives in g_qwen.max_seq (default 8192): attention scores moved
// from smem to a global scratch (bit-exact: same loops/order, other storage),
// so the old 48KB-smem ceiling is gone. KV fp32: 16k=+2.1GB, 32k=+4.3GB; long
// contexts need 4-bit KV (TQ_KV_Q4). The cap below is the model's
// native max_position_embeddings (Qwen3.6: 262144); the real working limit is
// still TQ_CTX (g_qwen.max_seq), which gates every cache/scratch allocation.
#define TQ_MAX_SEQ 262144
#define TQ_FLAG_BLOCK_SCALED_QMMA (1u << 5)
#define TQ_FLAG_BLOCK_SCALED_ROWMAJOR (1u << 6)
#define TQ_FLAG_BLOCK_SCALED_QMMA_WORDMAJOR (1u << 7)
#define TQ_FLAG_BLOCK_SCALED_E2M3 (1u << 8)
#define TQ_FLAG_SPARSE_24_E2M3 (1u << 9)
#define TQ_FLAG_HAS_MTP (1u << 10)
// Linear weights are EXL3 trellis records copied from an exllamav3 checkpoint (see
// tools/convert_qwen_tqf.py write_exl3_record); dequantized per 16x16 tile at run time.
#define TQ_FLAG_EXL3_WEIGHTS (1u << 11)
// The MTP section's 8 projections are EXL3 records (convert_qwen_tqf.py --mtp-exl3) instead of bf16.
#define TQ_FLAG_MTP_EXL3 (1u << 12)

// Commit-by-pointer-swap: rounds with N <= 8 alternate archive halves (slots
// 0-7 / 8-15), so the accepted leaf's state slab can BECOME the live state
// (zero-copy commit); the next round writes the other half and cannot clobber it.
static int g_spec_slot_base = 0;        // archive half (0 or 8) used by the current round
static int g_spec_live_in_archive = 0;  // live DeltaNet pointers parked in the archive
#ifndef TQ_SPEC_MAX_N
#define TQ_SPEC_MAX_N 16       // max batched verify nodes/round (16 = 2 m16n8 column groups).
                               // MEASURED dead-end (roadmap D stopgap):
                               // -DTQ_SPEC_MAX_N=32 + TQ_PREFILL_CHUNK=32 gives +50% prefill
                               // (695->1032 tok/s @2k; lever is REAL) BUT the tree-attention
                               // verify (k_tq_spec_attn_tree_mma) uses a single 16-slot m16
                               // archive tile -> chains >~20 overflow it and CORRUPT the
                               // long-context state (needle 0/3 garbage @8k; CH<=20 still OK).
                               // Not a transparent cap; also perturbs branching-tree decode
                               // numerics (max_abs 19) and scales every d_spec_* buffer (prod
                               // GPU6 @256k has ~0.45 GiB margin). Wide prefill needs a DEDICATED
                               // N-wide kernel (D.1), not this bump. Keep the default at 16.
#endif

// Set once at load time: 1 if QMMA weights are dense 6-bit E2M3 (mixed-format
// E2M3 weights x E4M3 activations), 0 for the default E4M3 weight path.
__constant__ int gc_weight_e2m3 = 0;

// Unread. Deleting them moves the bank-3 offsets of the constants below and with them the SASS of every kernel that
// reads those; remove them together with the q4s_* fp32 branch (gc_kv_q4_fp32s) under a bitwise gate.
__constant__ int gc_use_ldsm = 0;
__constant__ int gc_use_p2p = 0;

// TQ_EMBED_FP8: token-embedding table stored quantized at load (lookup-only --
// valid ONLY when tie_word_embeddings=FALSE, so the table never feeds the lm_head
// matmul). Published once at load; the lookup kernels branch on gc_embed_mode.
//   0 = off (d_embed stays bf16, bit-exact default path)
//   1 = per-row-scaled E4M3 (1 B/elem, -1.18 GiB)
//   2 = per-row-scaled E2M3 packed 6-bit (4 codes / 3 bytes, -1.48 GiB)
__constant__ int gc_embed_mode = 0;
__constant__ const uint8_t *gc_embed8 = nullptr;
__constant__ const float *gc_embed_scale = nullptr;

// TQ_KV_Q4 K-cache (scale, zp) group params: fp16 storage. Nothing sets this to 1 any more (the TQ_KV_Q4_FP32S
// switch is gone); the q4s_* helpers keep their fp32 branch until a bitwise-gated change removes it from every kernel
// that inlines them.
__constant__ int gc_kv_q4_fp32s = 0;

typedef struct {
    int M, K, Mt, Kt;
    float inv_scale;
    uint8_t *d_A;
    float *d_block_scale_inv;
    int block_scaled;
    int row_major;
    int word_major;
    int e2m3;
    int scale_rows;
    int scale_cols;
    // Optimistic sparse-FP4 draft (TQ_FLAG_SPARSE_24_E2M3): tile-level WANDA 2:4
    // record appended after the dense E2M3 payload. d_sparse_a = compressed E2M3
    // (3 u32/lane per k64-tile, stride Mt*Kt64*96), d_sparse_meta = ordered
    // metadata (1 u32/lane per k64-tile). Kept values are a sub-read of d_A.
    uint32_t *d_sparse_a;
    uint32_t *d_sparse_meta;
    int has_sparse;
    int Kt64;
    // EXL3 trellis weights (TQ_FLAG_EXL3_WEIGHTS). W[m][k] = svh[m] * (H128 W_hat H128)[k][m] * suh[k]
    // per 128x128 tile, W_hat dequantized from the 16x16 trellis tiles (mul1 codebook). The wide
    // path reads these; the E2M3 payload fields above stay NULL. exl3_kbits == 0: the source kept
    // this Linear unquantized -> d_exl3_bf16 holds bf16 [M][K] row-major.
    int exl3;
    int exl3_kbits;
    int exl3_codebook;       // 0 3inst, 1 mcg, 2 mul1 (only mul1 is implemented)
    uint16_t *d_exl3_trellis; // int16 [K/16][M/16][16*kbits]
    uint16_t *d_exl3_suh;     // fp16 [K]
    uint16_t *d_exl3_svh;     // fp16 [M]
    uint16_t *d_exl3_bf16;    // kbits == 0 only
} tq_qmma_weight_t;

typedef struct {
    uint16_t *d_input_ln;
    uint16_t *d_post_ln;

    tq_qmma_weight_t mlp_gate;
    tq_qmma_weight_t mlp_up;
    tq_qmma_weight_t mlp_down;

    uint16_t *d_q_norm;
    uint16_t *d_k_norm;
    tq_qmma_weight_t q_proj;
    tq_qmma_weight_t k_proj;
    tq_qmma_weight_t v_proj;
    tq_qmma_weight_t o_proj;

    float *d_linear_A_log;
    uint16_t *d_linear_dt_bias;
    float *d_linear_norm;
    uint16_t *d_linear_conv1d;
    tq_qmma_weight_t linear_in_a;
    tq_qmma_weight_t linear_in_b;
    tq_qmma_weight_t linear_in_qkv;
    tq_qmma_weight_t linear_in_z;
    tq_qmma_weight_t linear_out;

    float *d_linear_conv_state;       // LIVE state (may point into the spec archive after a swap-commit)
    float *d_linear_recurrent_state;
    float *d_linear_conv_state_owned;     // original allocations (free target,
    float *d_linear_recurrent_state_owned; // un-swap destination)
    float *d_k_cache;
    float *d_v_cache;
    // E4M3 rows + pow2 scale per (pos, kv_head) (TQ_KV_Q4: the V rows; K goes to d_k_cache4 below). When active the
    // fp32 cache above is NOT allocated. NULL when TQ_KV_Q4 is off.
    uint8_t *d_k_cache8;
    uint8_t *d_v_cache8;
    float *d_k_scale;
    float *d_v_scale;
    // 4-bit K prefix-cache (TQ_KV_Q4=1, implies the FP8 machinery for V): K rows
    // are Hadamard-rotated (FWHT-256, self-inverse) at write and stored as int4
    // codes (2 channels/byte) with an asymmetric (scale, zero-point) pair per
    // 32-channel group: 8 groups per (pos, kv_head). Everything downstream of
    // the write (archives, commits, scores) lives in the rotated space; V and
    // the attention epilogue are untouched (proto: pure-int4 V fails the cos
    // gate at 0.9958, K-int4+Hadamard passes at 0.99917 -- see
    // bench/kv_q4_proto_results.txt). d_k_cache8 stays NULL in this mode.
    uint8_t *d_k_cache4;     // [pos x nkv*hd/2] int4 codes
    float *d_k_q4s;          // [pos x nkv x 8 x 2] (scale, zp) per group
} tq_layer_t;

typedef struct {
    int initialized;
    uint32_t flags;
    uint32_t model_family;
    uint32_t non_quant_dtype;
    int H, I, L, V;
    int nh, nkv, hd;
    int linear_num_key_heads;
    int linear_key_head_dim;
    int linear_num_value_heads;
    int linear_value_head_dim;
    int linear_conv_kernel_dim;
    int max_position_embeddings;
    float eps;
    float rope_theta;
    float partial_rotary_factor;
    int mrope_interleaved;
    int mrope_section[3];
    int tie_word_embeddings;
    int text_only;
    int has_vision_config;
    int has_mtp;
    int has_mtp_section;  // MTP head weights present in this TQF (TQ_FLAG_HAS_MTP)
    uint8_t layer_types[TQ_MAX_LAYERS];

    uint16_t *d_embed;
    uint16_t *h_embed;       // TQ_EMBED_HOST: the bf16 table in mapped pinned host memory (d_embed aliases it)
    uint8_t *d_embed8;       // TQ_EMBED_FP8: quantized embed table (E4M3 mode 1, E2M3-6bit mode 2)
    float *d_embed_scale;    // per-row (per-token) absmax scale [V] for d_embed8
    int embed_fp8;           // 0=off (bf16), 1=E4M3, 2=E2M3-6bit (lookup dequant mode)
    uint16_t *d_norm;
    tq_qmma_weight_t lm_head;
    tq_layer_t layers[TQ_MAX_LAYERS];

    // The native MTP (multi-token-prediction) head of a --mtp-exl3 tqf (TQ_FLAG_MTP_EXL3), read by the paged drafter
    // (qwn_paged_mtp_*) through the wide path: the 8 projections are EXL3 records in mtp_layer.{q,k,v,o}_proj /
    // mlp_{gate,up,down} and mtp_fc_x, the norms in mtp_layer.d_{input,post}_ln / d_{q,k}_norm and the three below.
    // mtp_exl3 = 1 when it is loaded (a bf16 head is never loaded: nothing drafts with it).
    tq_layer_t mtp_layer;
    tq_qmma_weight_t mtp_fc_x;
    int mtp_exl3;
    uint16_t *d_mtp_pre_fc_norm_emb;
    uint16_t *d_mtp_pre_fc_norm_hidden;
    uint16_t *d_mtp_norm;
    float *d_attn_scores;                  // [512 x max_seq] gmem attention score slabs
    int max_seq;                           // runtime context limit (TQ_CTX, <= TQ_MAX_SEQ)
    int kv_fp8;                            // = kv_q4: the V rows + scales ride the E4M3 machinery
    int kv_q4;                             // TQ_KV_Q4=1: K int4+Hadamard, V E4M3 (sets kv_fp8)


    size_t device_bytes;
    int non_quant_count;
    int qmma_count;
    int f32_count;
    float last_argmax_logit;
    cudaStream_t stream;

    float *d_debug_x;
    float *d_debug_norm;
    float *d_debug_proj;
    float *d_debug_qmma_out;
    float *d_debug_qmma_raw;
    float *d_debug_qkv;
    float *d_debug_z;
    float *d_debug_a_proj;
    float *d_debug_core;
    float *d_debug_resid;
    float *d_debug_post_norm;
    float *d_debug_gate;
    float *d_debug_up;
    float *d_debug_mlp_hidden;
    float *d_debug_layer_out;
    float *d_debug_logits;
    float *d_debug_arg_vals;
    int *d_debug_arg_ids;
    float *d_debug_scale;
    float *d_debug_act_scale_inv;
    uint8_t *d_debug_b;
    int debug_proj_count;
    int debug_qmma_out_count;
    int debug_qkv_count;
    int debug_z_count;
    int debug_a_count;
    int debug_core_count;
    int debug_resid_count;
    int debug_post_norm_count;
    int debug_gate_count;
    int debug_up_count;
    int debug_mlp_hidden_count;
    int debug_layer_out_count;
    int debug_logits_count;
    int debug_arg_count;
    int debug_arg_ids_count;
    int debug_act_scale_count;
    int debug_b_bytes;
    int debug_raw_count;
} tq_model_t;

static tq_model_t g_qwen;

static __device__ __forceinline__ uint16_t tq_float_to_bf16(float x) {
    uint32_t u = __float_as_uint(x);
    uint32_t r = (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;   // round-to-nearest-even
    return (uint16_t)r;
}

static __device__ __forceinline__ float tq_bf16_to_float(uint16_t bits) {
    union {
        uint32_t u;
        float f;
    } v;
    v.u = ((uint32_t)bits) << 16;
    return v.f;
}

// --- TQ_DSTATE_BF16 (session 10, dstate): storage format of the paged DeltaNet RECURRENT state (g_pg_recur) ---
// DS = 0: fp32 (main's layout and arithmetic, bitwise); 1: bf16 (round-to-nearest-even); 2: fp16 (RNE, clamped to
// +-65504 so one outlier cannot poison a head with inf); 3: "bf16 + 8": the top 24 bits of the fp32 pattern (fp32's
// exponent, a 16-bit significand) stored as a bf16 array plus a byte array of the next 8 mantissa bits (3 bytes per
// element; per slot the hi array is followed by the lo array); 4: fp16 with deterministic STOCHASTIC rounding (the
// probability of rounding up = the fractional distance to the upper fp16 neighbour, drawn from a hash of the element's
// index in the slot and the token position, so the same token always rounds the same way: verify == decode stays
// bitwise and the expected stored value is the exact one -- no dead zone for updates below half an ulp).
// The compute stays fp32 in every kernel: load -> convert -> update -> round -> store; only the bits that reach HBM
// change. The conv window (g_pg_conv, 5 % of the state) stays fp32.
template <int DS> struct tq_ds_elem { typedef float T; };          // the element type of each format (3: the hi array)
template <> struct tq_ds_elem<1> { typedef uint16_t T; };
template <> struct tq_ds_elem<2> { typedef uint16_t T; };
template <> struct tq_ds_elem<3> { typedef uint16_t T; };
template <> struct tq_ds_elem<4> { typedef uint16_t T; };
static __host__ __device__ __forceinline__ size_t tq_ds_elem_bytes(int ds) { return ds == 0 ? 4 : ds == 3 ? 3 : 2; }
static __device__ __forceinline__ uint32_t tq_ds_hash(uint32_t idx, uint32_t pos) {   // murmur3 fmix of (idx, pos)
    uint32_t h = idx * 0x9E3779B1u ^ (pos + 0x7F4A7C15u) * 0x85EBCA77u;
    h ^= h >> 16; h *= 0x85EBCA6Bu; h ^= h >> 13; h *= 0xC2B2AE35u; h ^= h >> 16;
    return h;
}
static __device__ __forceinline__ uint16_t tq_f16_sr(float v, uint32_t r) {   // fp16 with stochastic rounding (r uniform)
    v = fminf(fmaxf(v, -65504.0f), 65504.0f);
    __half lo = __float2half_rd(v), hi = __float2half_ru(v);
    float flo = __half2float(lo), fhi = __half2float(hi);
    if (fhi == flo) return __half_as_ushort(lo);
    float p = (v - flo) / (fhi - flo);                         // in [0, 1)
    return __half_as_ushort(((float)r * (1.0f / 4294967296.0f) < p) ? hi : lo);
}
// The same decision without the two directed F2F, the two F2F back and the fp32 divide (session 11, gdn-prefill): the fp16
// neighbours of |v| are the fp32 bit pattern with the sub-grid mantissa bits cleared (grid = 2^13 mantissa units in the
// normal range, 2^(-1-e) in the subnormal range, |v| itself below 2^-24), the fractional distance p = m / grid is exact, and
// for v < 0 the distance from the lower neighbour is 1 - p (exact too). Bit-identical to tq_f16_sr on every fp32 input and
// every r (400 M random cases incl. NaN / inf / subnormals / the clamp, /tmp/gdn_sr_test.cu). Returns the fp16 bits and
// writes the rounded value as fp32 (= __half2float of the bits).
static __device__ __forceinline__ uint16_t tq_f16_sr_fast(float v, uint32_t r, float *rounded) {
    v = fminf(fmaxf(v, -65504.0f), 65504.0f);
    uint32_t u = __float_as_uint(v);
    uint32_t a = u & 0x7FFFFFFFu, sign = u & 0x80000000u;
    uint32_t lo_a, hi_a; float p;
    if (a >= 0x38800000u) {                                    // |v| >= 2^-14: fp16 normal range
        uint32_t m = a & 0x1FFFu;
        lo_a = a - m; hi_a = lo_a + 0x2000u;
        p = (__uint_as_float(0x4B000000u | m) - 8388608.0f) * (1.0f / 8192.0f);   // (float)m exactly, no I2F
    } else if (a >= 0x33800000u) {                             // [2^-24, 2^-14): the fp16 subnormal grid 2^-24
        int sh = -1 - ((int)(a >> 23) - 127);                  // 14..23 dropped mantissa bits
        uint32_t m = a & ((1u << sh) - 1u);
        lo_a = a - m; hi_a = lo_a + (1u << sh);
        p = (__uint_as_float(0x4B000000u | m) - 8388608.0f) * __uint_as_float((uint32_t)(127 - sh) << 23);
    } else {                                                   // |v| < 2^-24: neighbours 0 and 2^-24
        lo_a = 0u; hi_a = 0x33800000u;
        p = __uint_as_float(a) * 16777216.0f;
    }
    if (a == lo_a) { *rounded = v; return __half_as_ushort(__float2half_rn(v)); }   // representable (incl. +-0)
    float pp = sign ? (1.0f - p) : p;
    bool up = (float)r * (1.0f / 4294967296.0f) < pp;
    uint32_t pick = sign ? (up ? lo_a : hi_a) : (up ? hi_a : lo_a);
    float f = __uint_as_float(pick | sign);
    *rounded = f;
    return __half_as_ushort(__float2half_rn(f));
}
// The same rounding, branch-free (selects instead of the three range branches, so the compiler can interleave the
// independent elements of a thread) and without the final F2F: returns the rounded value as fp32. Bit-identical to
// tq_f16_sr followed by __half2float (the unit test covers it too). r = the hash of (element index, position).
static __device__ __forceinline__ float tq_f16_sr_round(float v, uint32_t r) {
    v = fminf(fmaxf(v, -65504.0f), 65504.0f);
    const uint32_t u = __float_as_uint(v);
    const uint32_t a = u & 0x7FFFFFFFu, sign = u & 0x80000000u;
    const bool normal = a >= 0x38800000u, tiny = a < 0x33800000u;
    const int ebits = (int)(a >> 23);
    const int sh = normal ? 13 : min(126 - ebits, 31);              // dropped mantissa bits (-1 - e in the subnormal range)
    const uint32_t step = 1u << sh;
    const uint32_t lo_a = tiny ? 0u : (a & ~(step - 1u));
    const uint32_t hi_a = tiny ? 0x33800000u : lo_a + step;
    // the comparison (float)r * 2^-32 < p with both sides scaled by 2^32: p32 = (|v| - lo) / grid * 2^32 = diff * scale,
    // diff exact (same binade / Sterbenz), scale = 2^(42 - e) in the normal range (m * 2^19), 2^56 below 2^-14 (grid 2^-24);
    // for v < 0 the distance from the lower neighbour is 1 - p -> 2^32 - p32 (exact as well)
    const float diff = __uint_as_float(a) - __uint_as_float(lo_a);
    const float scale = normal ? __uint_as_float((uint32_t)(296 - ebits) << 23) : 72057594037927936.0f;
    const float p32 = diff * scale;
    const float pp32 = sign ? (4294967296.0f - p32) : p32;
    const bool up = (float)r < pp32;
    uint32_t pick = (up != (sign != 0u)) ? hi_a : lo_a;
    pick = (a == lo_a) ? a : pick;
    return __uint_as_float(pick | sign);
}
// tq_ds_hash(idx, pos) = fmix(idx * K1 ^ (pos + C) * K2) with the first fmix xor folded: h ^= h >> 16 of (x ^ p) equals
// x ^ (x >> 16) ^ pc with pc = p ^ (p >> 16) (p = the per-sub-chunk position term).
static __device__ __forceinline__ uint32_t tq_ds_hash_mix(uint32_t x, uint32_t pc) {
    uint32_t h = x ^ (x >> 16) ^ pc;
    h *= 0x85EBCA6Bu; h ^= h >> 13; h *= 0xC2B2AE35u; h ^= h >> 16;
    return h;
}
// The state slice of one head of one slot in format DS: ld(i) / st(i, v) with i the element index inside the head
// (kk * dim + tid); pos is the token position (DS = 4's rounding seed; set per column by the kernels).
template <int DS> struct tq_ds_acc {
    typename tq_ds_elem<DS>::T *p;
    uint8_t *lo;                                               // DS = 3 only
    uint32_t gbase;                                            // the head's first element index in the slot (DS = 4)
    uint32_t pos;
    __device__ __forceinline__ float ld(size_t i) const {
        if constexpr (DS == 1) return tq_bf16_to_float(p[i]);
        else if constexpr (DS == 2 || DS == 4) return __half2float(__ushort_as_half(p[i]));
        else if constexpr (DS == 3) return __uint_as_float(((uint32_t)p[i] << 16) | ((uint32_t)lo[i] << 8));
        else return p[i];
    }
    __device__ __forceinline__ void st(size_t i, float v) const {
        if constexpr (DS == 1) p[i] = tq_float_to_bf16(v);
        else if constexpr (DS == 2) p[i] = __half_as_ushort(__float2half_rn(fminf(fmaxf(v, -65504.0f), 65504.0f)));
        else if constexpr (DS == 3) {
            uint32_t u = __float_as_uint(v);
            uint32_t r = (u + 0x7Fu + ((u >> 8) & 1u)) >> 8;        // RNE to 24 bits
            p[i] = (uint16_t)(r >> 8); lo[i] = (uint8_t)(r & 0xFFu);
        } else if constexpr (DS == 4) p[i] = tq_f16_sr(v, tq_ds_hash(gbase + (uint32_t)i, pos));
        else p[i] = v;
    }
    // The value a store + load round trip would give (the sequential verify / commit kernel applies it to the state it
    // carries in registers between columns, so its columns equal per-token decode steps bit for bit in every format).
    __device__ __forceinline__ float rnd(size_t i, float v) const {
        if constexpr (DS == 1) return tq_bf16_to_float(tq_float_to_bf16(v));
        else if constexpr (DS == 2) return __half2float(__float2half_rn(fminf(fmaxf(v, -65504.0f), 65504.0f)));
        else if constexpr (DS == 3) { uint32_t u = __float_as_uint(v); return __uint_as_float(((u + 0x7Fu + ((u >> 8) & 1u)) >> 8) << 8); }
        else if constexpr (DS == 4) return __half2float(__ushort_as_half(tq_f16_sr(v, tq_ds_hash(gbase + (uint32_t)i, pos))));
        else return v;
    }
    // rnd() with the fast fp16-SR (the register-resident prefill kernel rounds 16x per element per wave).
    __device__ __forceinline__ float rnd_fast(size_t i, float v) const {
        if constexpr (DS == 4) { float f; tq_f16_sr_fast(v, tq_ds_hash(gbase + (uint32_t)i, pos), &f); return f; }
        else return rnd(i, v);
    }
};
// The accessor of (slot, head) inside a per-layer state buffer: slot stride recur_stride elements, head offset head_off.
template <int DS> static __device__ __forceinline__ tq_ds_acc<DS> tq_ds_make(void *base, size_t slot, size_t recur_stride, size_t head_off) {
    tq_ds_acc<DS> a;
    if constexpr (DS == 3) {
        uint8_t *sb = (uint8_t *)base + slot * recur_stride * 3;
        a.p = (uint16_t *)sb + head_off; a.lo = sb + 2 * recur_stride + head_off;
    } else {
        a.p = (typename tq_ds_elem<DS>::T *)base + slot * recur_stride + head_off; a.lo = NULL;
    }
    a.gbase = (uint32_t)head_off; a.pos = 0;
    return a;
}
// Plain element access (qwn_paged_load_client's conversion): element i of a slot in format DS.
template <int DS> static __device__ __forceinline__ void tq_ds_st_elem(void *slot_base, size_t recur_stride, size_t i, float v) {
    tq_ds_acc<DS> a = tq_ds_make<DS>(slot_base, 0, recur_stride, 0);
    a.st(i, v);
}

static __device__ __forceinline__ float tq_e4m3_to_float(uint8_t v) {
    int sign = (v >> 7) & 1;
    int exp = (v >> 3) & 0x0f;
    int man = v & 0x07;
    if ((v & 0x7f) == 0) return 0.0f;

    float out;
    if (exp == 0) {
        out = ldexpf((float)man / 8.0f, -6);
    } else {
        out = ldexpf(1.0f + (float)man / 8.0f, exp - 7);
    }
    return sign ? -out : out;
}

static __device__ __forceinline__ uint8_t tq_qmma_a_byte(const uint8_t *a, int Kt, int row, int col) {
    int mt = row >> 4;
    int r = row & 15;
    int kt = col >> 5;
    int c = col & 31;
    int g = r >> 1;
    int pair = r & 1;
    int sub = c >> 3;
    int rem = c & 7;
    int half = rem >> 2;
    int elem = rem & 3;
    int lane = g * 4 + sub;
    int byte_idx = half * 8 + pair * 4 + elem;
    return a[((mt * Kt + kt) * 512) + lane * 16 + byte_idx];
}

// One embedding element for token `row`, channel `i` (TQ_EMBED_FP8). mode 0 is the
// bit-exact bf16 default; mode 1 dequants per-row-scaled E4M3; mode 2 dequants
// per-row-scaled E2M3 packed 6-bit (4 codes / 3 bytes, requires H % 4 == 0).
static __device__ __forceinline__ float tq_embed_lookup_one(const uint16_t *embed,
                                                            size_t row, int i, int H) {
    int mode = gc_embed_mode;
    if (mode == 0) return tq_bf16_to_float(embed[row * (size_t)H + i]);
    if (mode == 1) {
        uint8_t v = gc_embed8[row * (size_t)H + i];
        uint32_t bits = ((uint32_t)(v & 0x80u) << 24) | ((uint32_t)(v & 0x7fu) << 20);
        return __uint_as_float(bits) * 0x1p+120f * gc_embed_scale[row];   // E4M3 decode * row scale
    }
    // mode 2: E2M3 6-bit packed
    size_t rb = row * (size_t)((H * 6) / 8);
    int g = i >> 2, sub = i & 3;
    const uint8_t *p = gc_embed8 + rb + (size_t)g * 3;
    uint32_t packed = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16);
    uint32_t c = (packed >> (6 * sub)) & 0x3fu;
    float mm = (float)(c & 7u);
    uint32_t e = (c >> 3) & 3u;
    float val = (e == 0) ? mm * 0.125f : (1.0f + mm * 0.125f) * (float)(1u << (e - 1));
    return ((c & 0x20u) ? -val : val) * gc_embed_scale[row];
}

__global__ void k_tq_embed_lookup(float *out, const uint16_t *embed, int token_id, int H) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= H) return;
    out[i] = tq_embed_lookup_one(embed, (size_t)token_id, i, H);
}

