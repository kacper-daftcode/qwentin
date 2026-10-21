// the .tqf loader (parse_tqf), qwn_init / qwn_free and the model getters.
// Part of the forward_qwen.cu unity build: included in order, not a standalone header.

extern "C" void qwn_free(void) {
    if (g_qwen.stream) cudaStreamSynchronize(g_qwen.stream);
    if (g_qwen.d_debug_x) cudaFree(g_qwen.d_debug_x);
    if (g_qwen.d_debug_norm) cudaFree(g_qwen.d_debug_norm);
    if (g_qwen.d_debug_proj) cudaFree(g_qwen.d_debug_proj);
    if (g_qwen.d_debug_qmma_out) cudaFree(g_qwen.d_debug_qmma_out);
    if (g_qwen.d_debug_qmma_raw) cudaFree(g_qwen.d_debug_qmma_raw);
    if (g_qwen.d_debug_qkv) cudaFree(g_qwen.d_debug_qkv);
    if (g_qwen.d_debug_z) cudaFree(g_qwen.d_debug_z);
    if (g_qwen.d_debug_a_proj) cudaFree(g_qwen.d_debug_a_proj);
    if (g_qwen.d_debug_core) cudaFree(g_qwen.d_debug_core);
    if (g_qwen.d_debug_resid) cudaFree(g_qwen.d_debug_resid);
    if (g_qwen.d_debug_post_norm) cudaFree(g_qwen.d_debug_post_norm);
    if (g_qwen.d_debug_gate) cudaFree(g_qwen.d_debug_gate);
    if (g_qwen.d_debug_up) cudaFree(g_qwen.d_debug_up);
    if (g_qwen.d_debug_mlp_hidden) cudaFree(g_qwen.d_debug_mlp_hidden);
    if (g_qwen.d_debug_layer_out) cudaFree(g_qwen.d_debug_layer_out);
    if (g_qwen.d_debug_logits) cudaFree(g_qwen.d_debug_logits);
    if (g_qwen.d_debug_arg_vals) cudaFree(g_qwen.d_debug_arg_vals);
    if (g_qwen.d_debug_arg_ids) cudaFree(g_qwen.d_debug_arg_ids);
    if (g_qwen.d_debug_scale) cudaFree(g_qwen.d_debug_scale);
    if (g_qwen.d_debug_act_scale_inv) cudaFree(g_qwen.d_debug_act_scale_inv);
    if (g_qwen.d_debug_b) cudaFree(g_qwen.d_debug_b);
    if (g_qwen.h_embed) cudaFreeHost(g_qwen.h_embed);
    else if (g_qwen.d_embed) cudaFree(g_qwen.d_embed);
    if (g_qwen.d_embed8) cudaFree(g_qwen.d_embed8);
    if (g_qwen.d_embed_scale) cudaFree(g_qwen.d_embed_scale);
    g_qwen.d_embed = nullptr; g_qwen.h_embed = nullptr; g_qwen.d_embed8 = nullptr; g_qwen.d_embed_scale = nullptr;
    g_qwen.embed_fp8 = 0;
    if (g_qwen.d_norm) cudaFree(g_qwen.d_norm);
    free_qmma(&g_qwen.lm_head);
    // MTP head buffers.
    {
        tq_layer_t *m = &g_qwen.mtp_layer;
        if (m->d_input_ln) cudaFree(m->d_input_ln);
        if (m->d_post_ln) cudaFree(m->d_post_ln);
        if (m->d_q_norm) cudaFree(m->d_q_norm);
        if (m->d_k_norm) cudaFree(m->d_k_norm);
        if (m->d_k_cache) cudaFree(m->d_k_cache);
        if (m->d_v_cache) cudaFree(m->d_v_cache);
        if (m->d_k_cache8) cudaFree(m->d_k_cache8);
        if (m->d_v_cache8) cudaFree(m->d_v_cache8);
        if (m->d_k_scale) cudaFree(m->d_k_scale);
        if (m->d_v_scale) cudaFree(m->d_v_scale);
        if (m->d_k_cache4) cudaFree(m->d_k_cache4);
        if (m->d_k_q4s) cudaFree(m->d_k_q4s);
        if (g_qwen.mtp_exl3) {                     // EXL3 MTP projections (paged drafter)
            free_qmma(&g_qwen.mtp_fc_x);
            free_qmma(&m->q_proj); free_qmma(&m->k_proj); free_qmma(&m->v_proj); free_qmma(&m->o_proj);
            free_qmma(&m->mlp_gate); free_qmma(&m->mlp_up); free_qmma(&m->mlp_down);
            g_qwen.mtp_exl3 = 0;
        }
        if (g_qwen.draft_on) { free_qmma(&g_qwen.draft_head); g_qwen.draft_on = 0; }
        if (g_qwen.d_draft_d2t) cudaFree(g_qwen.d_draft_d2t);
        if (g_qwen.d_mtp_pre_fc_norm_emb) cudaFree(g_qwen.d_mtp_pre_fc_norm_emb);
        if (g_qwen.d_mtp_pre_fc_norm_hidden) cudaFree(g_qwen.d_mtp_pre_fc_norm_hidden);
        if (g_qwen.d_mtp_norm) cudaFree(g_qwen.d_mtp_norm);
    }
    for (int i = 0; i < g_qwen.L && i < TQ_MAX_LAYERS; i++) {
        tq_layer_t *l = &g_qwen.layers[i];
        if (l->d_input_ln) cudaFree(l->d_input_ln);
        if (l->d_post_ln) cudaFree(l->d_post_ln);
        free_qmma(&l->mlp_gate);
        free_qmma(&l->mlp_up);
        free_qmma(&l->mlp_down);
        if (l->d_q_norm) cudaFree(l->d_q_norm);
        if (l->d_k_norm) cudaFree(l->d_k_norm);
        free_qmma(&l->q_proj);
        free_qmma(&l->k_proj);
        free_qmma(&l->v_proj);
        free_qmma(&l->o_proj);
        if (l->d_linear_A_log) cudaFree(l->d_linear_A_log);
        if (l->d_linear_dt_bias) cudaFree(l->d_linear_dt_bias);
        if (l->d_linear_norm) cudaFree(l->d_linear_norm);
        if (l->d_linear_conv1d) cudaFree(l->d_linear_conv1d);
        if (l->d_linear_conv_state) cudaFree(l->d_linear_conv_state);
        if (l->d_linear_recurrent_state) cudaFree(l->d_linear_recurrent_state);
        if (l->d_k_cache) cudaFree(l->d_k_cache);
        if (l->d_v_cache) cudaFree(l->d_v_cache);
        if (l->d_k_cache8) cudaFree(l->d_k_cache8);
        if (l->d_v_cache8) cudaFree(l->d_v_cache8);
        if (l->d_k_scale) cudaFree(l->d_k_scale);
        if (l->d_v_scale) cudaFree(l->d_v_scale);
        if (l->d_k_cache4) cudaFree(l->d_k_cache4);
        if (l->d_k_q4s) cudaFree(l->d_k_q4s);
        free_qmma(&l->linear_in_a);
        free_qmma(&l->linear_in_b);
        free_qmma(&l->linear_in_qkv);
        free_qmma(&l->linear_in_z);
        free_qmma(&l->linear_out);
    }
    if (g_qwen.stream) cudaStreamDestroy(g_qwen.stream);
    memset(&g_qwen, 0, sizeof(g_qwen));
}

// BLAKE2b (RFC 7693, unkeyed), for the .tqf header and payload hashes.
typedef struct { uint64_t h[8], t[2]; uint8_t b[128]; size_t c, outlen; } tq_b2b_ctx;
static const uint64_t tq_b2b_iv[8] = {
    0x6A09E667F3BCC908ULL, 0xBB67AE8584CAA73BULL, 0x3C6EF372FE94F82BULL, 0xA54FF53A5F1D36F1ULL,
    0x510E527FADE682D1ULL, 0x9B05688C2B3E6C1FULL, 0x1F83D9ABFB41BD6BULL, 0x5BE0CD19137E2179ULL};
static const uint8_t tq_b2b_sigma[12][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
    {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4}, {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
    {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13}, {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
    {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11}, {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
    {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5}, {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3}};
static inline uint64_t tq_rotr64(uint64_t x, int n) { return (x >> n) | (x << (64 - n)); }
static void tq_b2b_compress(tq_b2b_ctx *c, int last) {
    uint64_t v[16], m[16];
    for (int i = 0; i < 8; i++) { v[i] = c->h[i]; v[i + 8] = tq_b2b_iv[i]; }
    v[12] ^= c->t[0]; v[13] ^= c->t[1];
    if (last) v[14] = ~v[14];
    for (int i = 0; i < 16; i++) {
        uint64_t w = 0;
        for (int j = 7; j >= 0; j--) w = (w << 8) | c->b[8 * i + j];
        m[i] = w;
    }
    static const int G[8][4] = {{0, 4, 8, 12}, {1, 5, 9, 13}, {2, 6, 10, 14}, {3, 7, 11, 15},
                                {0, 5, 10, 15}, {1, 6, 11, 12}, {2, 7, 8, 13}, {3, 4, 9, 14}};
    for (int r = 0; r < 12; r++) {
        const uint8_t *s = tq_b2b_sigma[r];
        for (int g = 0; g < 8; g++) {
            int a = G[g][0], b = G[g][1], cc = G[g][2], d = G[g][3];
            v[a] = v[a] + v[b] + m[s[2 * g]];     v[d] = tq_rotr64(v[d] ^ v[a], 32);
            v[cc] = v[cc] + v[d];                 v[b] = tq_rotr64(v[b] ^ v[cc], 24);
            v[a] = v[a] + v[b] + m[s[2 * g + 1]]; v[d] = tq_rotr64(v[d] ^ v[a], 16);
            v[cc] = v[cc] + v[d];                 v[b] = tq_rotr64(v[b] ^ v[cc], 63);
        }
    }
    for (int i = 0; i < 8; i++) c->h[i] ^= v[i] ^ v[i + 8];
}
static void tq_b2b_init(tq_b2b_ctx *c, size_t outlen) {
    for (int i = 0; i < 8; i++) c->h[i] = tq_b2b_iv[i];
    c->h[0] ^= 0x01010000ULL ^ (uint64_t)outlen;
    c->t[0] = c->t[1] = 0; c->c = 0; c->outlen = outlen;
}
static void tq_b2b_update(tq_b2b_ctx *c, const void *in, size_t n) {
    const uint8_t *p = (const uint8_t *)in;
    while (n > 0) {
        if (c->c == 128) {                     // a full block is compressed only once more input follows it
            c->t[0] += 128; if (c->t[0] < 128) c->t[1]++;
            tq_b2b_compress(c, 0); c->c = 0;
        }
        size_t k = 128 - c->c < n ? 128 - c->c : n;
        memcpy(c->b + c->c, p, k); c->c += k; p += k; n -= k;
    }
}
static void tq_b2b_final(tq_b2b_ctx *c, uint8_t *out) {
    c->t[0] += c->c; if (c->t[0] < c->c) c->t[1]++;
    memset(c->b + c->c, 0, 128 - c->c);
    tq_b2b_compress(c, 1);
    for (size_t i = 0; i < c->outlen; i++) out[i] = (uint8_t)(c->h[i >> 3] >> (8 * (i & 7)));
}
extern "C" int qwn_tqf_blake2b256(const void *data, size_t n, uint8_t *out) {
    tq_b2b_ctx c; tq_b2b_init(&c, 32); tq_b2b_update(&c, data, n); tq_b2b_final(&c, out);
    return 0;
}

// .tqf format version 2 (docs/TQF_FORMAT.md): an 80-byte extension after the layer types records the payload size,
// a BLAKE2b-256 of the payload and one of the header (with both hash fields zeroed). A version-1 file (no extension)
// loads as before. Refused: a newer format (-40), a payload size that differs from the record (-41: truncated or
// extended file), a header that does not match its hash (-42), with TQ_VERIFY_TQF=1 a payload that does not (-43; a
// full extra read of the file), unknown flag bits in a version-2 file (-44; a warning in version 1), a malformed
// extension (-45). Leaves the file positioned at the payload.
#define TQF_FORMAT_VERSION 2
#define TQF_EXT_BYTES 80
#define TQF_KNOWN_FLAGS 0x1FFFu
static int tqf_check_integrity(FILE *f, const char *path, uint32_t header_bytes, uint32_t base_header) {
    uint32_t unknown = g_qwen.flags & ~TQF_KNOWN_FLAGS;
    if (header_bytes == base_header) {
        if (unknown) fprintf(stderr, "[tqf] WARNING: %s sets unknown flag bits 0x%x\n", path, unknown);
        printf("TQF1 format v1: no recorded payload size or hashes (a re-conversion writes v2)\n");
        return 0;
    }
    uint8_t *hdr = (uint8_t *)malloc(header_bytes);
    if (!hdr || fseeko(f, 0, SEEK_SET) != 0 || fread(hdr, 1, header_bytes, f) != header_bytes) {
        free(hdr); fprintf(stderr, "[tqf] %s: cannot re-read the %u-byte header\n", path, header_bytes); return -1;
    }
    uint8_t *ext = hdr + base_header, payload_hash[32], header_hash[32], h[32];
    uint32_t version, ext_bytes; uint64_t payload_bytes;
    memcpy(&version, ext, 4); memcpy(&ext_bytes, ext + 4, 4); memcpy(&payload_bytes, ext + 8, 8);
    memcpy(payload_hash, ext + 16, 32); memcpy(header_hash, ext + 48, 32);
    int rc = 0;
    if (version > TQF_FORMAT_VERSION) {
        fprintf(stderr, "[tqf] %s: format v%u needs a newer engine (this one reads v1..v%d)\n", path, version, TQF_FORMAT_VERSION);
        rc = -40;
    } else if (version < 2 || ext_bytes < TQF_EXT_BYTES || base_header + ext_bytes != header_bytes) {
        fprintf(stderr, "[tqf] %s: malformed header extension (format_version %u, extension_bytes %u)\n", path, version, ext_bytes);
        rc = -45;
    } else if (unknown) {
        fprintf(stderr, "[tqf] %s: unknown flag bits 0x%x in a format-v2 file\n", path, unknown);
        rc = -44;
    } else {
        memset(ext + 16, 0, 64);
        qwn_tqf_blake2b256(hdr, header_bytes, h);
        if (memcmp(h, header_hash, 32) != 0) {
            fprintf(stderr, "[tqf] %s: the header does not match its recorded hash (corrupted header)\n", path);
            rc = -42;
        }
    }
    free(hdr);
    if (rc != 0) return rc;
    if (fseeko(f, 0, SEEK_END) != 0) return -1;
    uint64_t on_disk = (uint64_t)ftello(f) - header_bytes;
    if (on_disk != payload_bytes) {
        fprintf(stderr, "[tqf] %s: the payload is %llu bytes, the header records %llu (truncated or extended file)\n",
                path, (unsigned long long)on_disk, (unsigned long long)payload_bytes);
        return -41;
    }
    if (g_cfg.verify_tqf) {
        if (fseeko(f, (off_t)header_bytes, SEEK_SET) != 0) return -1;
        const size_t cap = (size_t)1 << 24;
        uint8_t *buf = (uint8_t *)malloc(cap);
        if (!buf) return -1;
        tq_b2b_ctx c; tq_b2b_init(&c, 32);
        size_t n;
        while ((n = fread(buf, 1, cap, f)) > 0) tq_b2b_update(&c, buf, n);
        free(buf);
        tq_b2b_final(&c, h);
        if (memcmp(h, payload_hash, 32) != 0) {
            fprintf(stderr, "[tqf] %s: the payload does not match its recorded hash (corrupted file)\n", path);
            return -43;
        }
    }
    if (fseeko(f, (off_t)header_bytes, SEEK_SET) != 0) return -1;
    printf("TQF1 format v%u: payload %llu bytes as recorded, header hash OK, payload hash %s\n", version,
           (unsigned long long)payload_bytes, g_cfg.verify_tqf ? "OK (TQ_VERIFY_TQF)" : "not checked (TQ_VERIFY_TQF=1)");
    return 0;
}

static int parse_tqf(const char *path, int upload) {
    if (upload) qwn_free();
    else memset(&g_qwen, 0, sizeof(g_qwen));
    if (upload) {
        cudaError_t serr = cudaStreamCreateWithFlags(&g_qwen.stream, cudaStreamNonBlocking);
        if (serr != cudaSuccess) {
            fprintf(stderr, "cudaStreamCreateWithFlags failed: %s\n", cudaGetErrorString(serr));
            return -1;
        }
    }

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open %s\n", path);
        return -1;
    }

    char magic[4];
    if (read_exact(f, magic, 4, "magic") != 0) { fclose(f); return -1; }
    if (memcmp(magic, "TQF1", 4) != 0) {
        fprintf(stderr, "Bad TQF magic\n");
        fclose(f);
        return -1;
    }

    uint32_t header_bytes = 0;
    if (read_exact(f, &header_bytes, 4, "header_bytes") != 0) { fclose(f); return -1; }
    uint32_t u[16];
    float fl[3];
    uint32_t tail[8];
    if (read_exact(f, u, sizeof(u), "fixed uint header") != 0) { fclose(f); return -1; }
    if (read_exact(f, fl, sizeof(fl), "fixed float header") != 0) { fclose(f); return -1; }
    if (read_exact(f, tail, sizeof(tail), "fixed tail header") != 0) { fclose(f); return -1; }

    g_qwen.flags = u[0];
    g_qwen.model_family = u[1];
    g_qwen.non_quant_dtype = u[2];
    g_qwen.H = (int)u[3];
    g_qwen.I = (int)u[4];
    g_qwen.L = (int)u[5];
    g_qwen.V = (int)u[6];
    g_qwen.nh = (int)u[7];
    g_qwen.nkv = (int)u[8];
    g_qwen.hd = (int)u[9];
    g_qwen.linear_num_key_heads = (int)u[10];
    g_qwen.linear_key_head_dim = (int)u[11];
    g_qwen.linear_num_value_heads = (int)u[12];
    g_qwen.linear_value_head_dim = (int)u[13];
    g_qwen.linear_conv_kernel_dim = (int)u[14];
    g_qwen.max_position_embeddings = (int)u[15];
    g_qwen.eps = fl[0];
    g_qwen.rope_theta = fl[1];
    g_qwen.partial_rotary_factor = fl[2];
    g_qwen.mrope_interleaved = (int)tail[0];
    g_qwen.mrope_section[0] = (int)tail[1];
    g_qwen.mrope_section[1] = (int)tail[2];
    g_qwen.mrope_section[2] = (int)tail[3];
    g_qwen.tie_word_embeddings = (int)tail[4];
    g_qwen.text_only = (int)tail[5];
    g_qwen.has_vision_config = (int)tail[6];
    g_qwen.has_mtp = (int)tail[7];
    g_qwen.has_mtp_section = (g_qwen.flags & TQ_FLAG_HAS_MTP) != 0;

    if (g_qwen.L <= 0 || g_qwen.L > TQ_MAX_LAYERS) {
        fprintf(stderr, "Unsupported layer count: %d\n", g_qwen.L);
        fclose(f);
        return -1;
    }
    uint32_t base_header = 4u + 4u + 16u * 4u + 3u * 4u + 8u * 4u + (uint32_t)g_qwen.L;
    if (header_bytes != base_header && header_bytes < base_header + TQF_EXT_BYTES) {
        fprintf(stderr, "Unexpected header_bytes=%u expected=%u (format v1) or >= %u (v2)\n", header_bytes, base_header,
                base_header + TQF_EXT_BYTES);
        fclose(f);
        return -1;
    }
    if (read_exact(f, g_qwen.layer_types, (size_t)g_qwen.L, "layer types") != 0) {
        fclose(f);
        return -1;
    }
    {
        int irc = tqf_check_integrity(f, path, header_bytes, base_header);
        if (irc != 0) { fclose(f); return irc; }
    }

    if (upload) {
        int e2m3_flag = (g_qwen.flags & TQ_FLAG_BLOCK_SCALED_E2M3) != 0;
        cudaError_t cerr = cudaMemcpyToSymbol(gc_weight_e2m3, &e2m3_flag, sizeof(int));
        if (cerr != cudaSuccess) {
            fprintf(stderr, "cudaMemcpyToSymbol(gc_weight_e2m3) failed: %s\n", cudaGetErrorString(cerr));
            fclose(f);
            return -1;
        }
    }

    printf("TQF1: H=%d I=%d L=%d V=%d nh=%d nkv=%d hd=%d dtype=%u flags=0x%x\n",
           g_qwen.H, g_qwen.I, g_qwen.L, g_qwen.V, g_qwen.nh, g_qwen.nkv, g_qwen.hd,
           g_qwen.non_quant_dtype, g_qwen.flags);

    int lin_key_dim = g_qwen.linear_num_key_heads * g_qwen.linear_key_head_dim;
    int lin_value_dim = g_qwen.linear_num_value_heads * g_qwen.linear_value_head_dim;
    int lin_conv_dim = lin_key_dim * 2 + lin_value_dim;

    if (upload && g_cfg.embed_host && !g_qwen.tie_word_embeddings) {
        if (read_embed_host(f, (size_t)g_qwen.V * g_qwen.H) != 0) { fclose(f); return -1; }
    } else if (read_non_quant(f, &g_qwen.d_embed, (size_t)g_qwen.V * g_qwen.H, "embed", upload) != 0) { fclose(f); return -1; }
    if (read_non_quant(f, &g_qwen.d_norm, (size_t)g_qwen.H, "norm", upload) != 0) { fclose(f); return -1; }
    if (!g_qwen.tie_word_embeddings) {
        if (read_qmma(f, &g_qwen.lm_head, g_qwen.V, g_qwen.H, "lm_head", upload, 0) != 0) { fclose(f); return -1; }
    }

    // TQ_EMBED_FP8: quantize the bf16 embedding table at load (E4M3=1, E2M3-6bit=2)
    // and free the bf16 copy. Lookup-only -> requires tie_word_embeddings=FALSE.
    g_qwen.embed_fp8 = 0;
    g_qwen.d_embed8 = nullptr;
    g_qwen.d_embed_scale = nullptr;
    if (upload) {
        int emode = g_cfg.embed_fp8;
        if (emode && g_qwen.tie_word_embeddings) {
            fprintf(stderr, "TQ_EMBED_FP8 ignored: tie_word_embeddings=TRUE (embed feeds lm_head)\n");
            emode = 0;
        }
        if (emode && g_qwen.h_embed) {
            printf("TQ_EMBED_FP8 ignored: TQ_EMBED_HOST keeps the bf16 table in host memory\n");
            emode = 0;
        }
        if (emode && g_qwen.d_embed) {
            size_t rows = (size_t)g_qwen.V, Hh = (size_t)g_qwen.H;
            size_t qbytes = (emode == 2) ? rows * ((Hh * 6) / 8) : rows * Hh;
            uint8_t *d_q = nullptr; float *d_sc = nullptr;
            cudaDeviceSynchronize();   // embed upload must be visible to the quant kernel
            if (cudaMalloc(&d_q, qbytes) != cudaSuccess ||
                cudaMalloc(&d_sc, rows * sizeof(float)) != cudaSuccess) {
                fprintf(stderr, "TQ_EMBED_FP8 alloc failed -- staying bf16\n");
                if (d_q) cudaFree(d_q);
            } else {
                if (emode == 2)
                    k_tq_embed_quant_e2m3<<<(unsigned)rows, 256>>>(g_qwen.d_embed, d_q, d_sc, g_qwen.H);
                else
                    k_tq_embed_quant_e4m3<<<(unsigned)rows, 256>>>(g_qwen.d_embed, d_q, d_sc, g_qwen.H);
                cudaError_t qe = cudaDeviceSynchronize();
                if (qe != cudaSuccess) {
                    fprintf(stderr, "TQ_EMBED_FP8 quant kernel failed: %s -- staying bf16\n",
                            cudaGetErrorString(qe));
                    cudaFree(d_q); cudaFree(d_sc);
                } else {
                    cudaFree(g_qwen.d_embed);
                    g_qwen.d_embed = nullptr;
                    g_qwen.d_embed8 = d_q;
                    g_qwen.d_embed_scale = d_sc;
                    g_qwen.embed_fp8 = emode;
                    g_qwen.device_bytes += qbytes + rows * sizeof(float);
                    g_qwen.device_bytes -= rows * Hh * sizeof(uint16_t);   // the freed bf16 upload
                    printf("TQ_EMBED_FP8=%d: embed table -> %s, %.0f MiB (+%.0f MiB scale) "
                           "vs %.0f MiB bf16\n", emode,
                           emode == 2 ? "E2M3-6bit per-row" : "E4M3 per-row",
                           qbytes / 1048576.0, rows * sizeof(float) / 1048576.0,
                           rows * Hh * 2 / 1048576.0);
                }
            }
        }
        cudaMemcpyToSymbol(gc_embed_mode, &g_qwen.embed_fp8, sizeof(int));
        cudaMemcpyToSymbol(gc_embed8, &g_qwen.d_embed8, sizeof(void *));
        cudaMemcpyToSymbol(gc_embed_scale, &g_qwen.d_embed_scale, sizeof(void *));
    }

    for (int i = 0; i < g_qwen.L; i++) {
        tq_layer_t *l = &g_qwen.layers[i];
        char what[128];
        // mlp.* sparse records exist only on linear-attention layers (the
        // calibrated FP4 set); full-attention layers' mlp stays dense.
        int lin_layer = (g_qwen.layer_types[i] == TQ_LAYER_LINEAR_ATTENTION);
        snprintf(what, sizeof(what), "layer%d.input_ln", i);
        if (read_non_quant(f, &l->d_input_ln, g_qwen.H, what, upload) != 0) { fclose(f); return -1; }
        snprintf(what, sizeof(what), "layer%d.post_ln", i);
        if (read_non_quant(f, &l->d_post_ln, g_qwen.H, what, upload) != 0) { fclose(f); return -1; }
        snprintf(what, sizeof(what), "layer%d.mlp_gate", i);
        if (read_qmma(f, &l->mlp_gate, g_qwen.I, g_qwen.H, what, upload, lin_layer) != 0) { fclose(f); return -1; }
        snprintf(what, sizeof(what), "layer%d.mlp_up", i);
        if (read_qmma(f, &l->mlp_up, g_qwen.I, g_qwen.H, what, upload, lin_layer) != 0) { fclose(f); return -1; }
        snprintf(what, sizeof(what), "layer%d.mlp_down", i);
        if (read_qmma(f, &l->mlp_down, g_qwen.H, g_qwen.I, what, upload, lin_layer) != 0) { fclose(f); return -1; }

        if (g_qwen.layer_types[i] == TQ_LAYER_FULL_ATTENTION) {
            snprintf(what, sizeof(what), "layer%d.q_norm", i);
            if (read_non_quant(f, &l->d_q_norm, g_qwen.hd, what, upload) != 0) { fclose(f); return -1; }
            snprintf(what, sizeof(what), "layer%d.k_norm", i);
            if (read_non_quant(f, &l->d_k_norm, g_qwen.hd, what, upload) != 0) { fclose(f); return -1; }
            snprintf(what, sizeof(what), "layer%d.q_proj", i);
            if (read_qmma(f, &l->q_proj, g_qwen.nh * g_qwen.hd * 2, g_qwen.H, what, upload, 0) != 0) { fclose(f); return -1; }
            snprintf(what, sizeof(what), "layer%d.k_proj", i);
            if (read_qmma(f, &l->k_proj, g_qwen.nkv * g_qwen.hd, g_qwen.H, what, upload, 0) != 0) { fclose(f); return -1; }
            snprintf(what, sizeof(what), "layer%d.v_proj", i);
            if (read_qmma(f, &l->v_proj, g_qwen.nkv * g_qwen.hd, g_qwen.H, what, upload, 0) != 0) { fclose(f); return -1; }
            snprintf(what, sizeof(what), "layer%d.o_proj", i);
            if (read_qmma(f, &l->o_proj, g_qwen.H, g_qwen.nh * g_qwen.hd, what, upload, 0) != 0) { fclose(f); return -1; }
        } else if (g_qwen.layer_types[i] == TQ_LAYER_LINEAR_ATTENTION) {
            snprintf(what, sizeof(what), "layer%d.linear_A_log", i);
            if (read_f32(f, &l->d_linear_A_log, g_qwen.linear_num_value_heads, what, upload) != 0) { fclose(f); return -1; }
            snprintf(what, sizeof(what), "layer%d.linear_dt_bias", i);
            if (read_non_quant(f, &l->d_linear_dt_bias, g_qwen.linear_num_value_heads, what, upload) != 0) { fclose(f); return -1; }
            snprintf(what, sizeof(what), "layer%d.linear_norm", i);
            if (read_f32(f, &l->d_linear_norm, g_qwen.linear_value_head_dim, what, upload) != 0) { fclose(f); return -1; }
            snprintf(what, sizeof(what), "layer%d.linear_conv1d", i);
            if (read_non_quant(f, &l->d_linear_conv1d, (size_t)lin_conv_dim * g_qwen.linear_conv_kernel_dim, what, upload) != 0) { fclose(f); return -1; }
            snprintf(what, sizeof(what), "layer%d.linear_in_a", i);
            if (read_qmma(f, &l->linear_in_a, g_qwen.linear_num_value_heads, g_qwen.H, what, upload, 0) != 0) { fclose(f); return -1; }
            snprintf(what, sizeof(what), "layer%d.linear_in_b", i);
            if (read_qmma(f, &l->linear_in_b, g_qwen.linear_num_value_heads, g_qwen.H, what, upload, 0) != 0) { fclose(f); return -1; }
            snprintf(what, sizeof(what), "layer%d.linear_in_qkv", i);
            if (read_qmma(f, &l->linear_in_qkv, lin_conv_dim, g_qwen.H, what, upload, 1) != 0) { fclose(f); return -1; }
            snprintf(what, sizeof(what), "layer%d.linear_in_z", i);
            if (read_qmma(f, &l->linear_in_z, lin_value_dim, g_qwen.H, what, upload, 1) != 0) { fclose(f); return -1; }
            snprintf(what, sizeof(what), "layer%d.linear_out", i);
            if (read_qmma(f, &l->linear_out, g_qwen.H, lin_value_dim, what, upload, 1) != 0) { fclose(f); return -1; }
        } else {
            fprintf(stderr, "Unsupported layer type byte at layer %d: %u\n", i, g_qwen.layer_types[i]);
            fclose(f);
            return -1;
        }
        if (i % 6 == 0) printf("  layer %d/%d loaded\n", i, g_qwen.L);
    }

    // Native MTP head section (appended after the main payload by the converter; read order MUST match
    // mtp_tensor_specs() in tools/convert_qwen_tqf.py). The MTP layer is one full_attention decoder layer. Only a
    // head with EXL3 projections (TQ_FLAG_MTP_EXL3, convert_qwen_tqf.py --mtp-exl3) is loaded: the paged drafter
    // reads it. A bf16 head (FP6 files, the older -mtp.tqf) is read past -- nothing drafts with it, and
    // qwn_paged_mtp_init refuses the file (-3). TQ_NO_MTP=1 reads past any head, and the engine then behaves as for a
    // TQF without the section.
    int no_mtp = upload && g_cfg.no_mtp;
    if (g_qwen.has_mtp_section) {
        int mtp_exl3 = (g_qwen.flags & TQ_FLAG_MTP_EXL3) != 0;
        int mup = upload && !no_mtp && mtp_exl3;
        tq_layer_t *m = &g_qwen.mtp_layer;
        int q_m = g_qwen.nh * g_qwen.hd * 2;
        int kv_m = g_qwen.nkv * g_qwen.hd;
        int attn_m = g_qwen.nh * g_qwen.hd;
        uint16_t *bf16_unread = NULL;
        #define TQ_READ_MTP_PROJ(XW, MM, KK, NAME)                                                                    \
            do { if (mtp_exl3) { if (read_exl3(f, &(XW), (MM), (KK), NAME, mup) != 0) { fclose(f); return -1; } }      \
                 else if (read_non_quant(f, &bf16_unread, (size_t)(MM) * (KK), NAME, 0) != 0) { fclose(f); return -1; } \
            } while (0)
        if (read_non_quant(f, &g_qwen.d_mtp_pre_fc_norm_emb, g_qwen.H, "mtp.pre_fc_norm_emb", mup) != 0) { fclose(f); return -1; }
        if (read_non_quant(f, &g_qwen.d_mtp_pre_fc_norm_hidden, g_qwen.H, "mtp.pre_fc_norm_hidden", mup) != 0) { fclose(f); return -1; }
        TQ_READ_MTP_PROJ(g_qwen.mtp_fc_x, g_qwen.H, 2 * g_qwen.H, "mtp.fc");
        if (read_non_quant(f, &m->d_input_ln, g_qwen.H, "mtp.input_ln", mup) != 0) { fclose(f); return -1; }
        if (read_non_quant(f, &m->d_post_ln, g_qwen.H, "mtp.post_ln", mup) != 0) { fclose(f); return -1; }
        if (read_non_quant(f, &m->d_q_norm, g_qwen.hd, "mtp.q_norm", mup) != 0) { fclose(f); return -1; }
        if (read_non_quant(f, &m->d_k_norm, g_qwen.hd, "mtp.k_norm", mup) != 0) { fclose(f); return -1; }
        TQ_READ_MTP_PROJ(m->q_proj, q_m, g_qwen.H, "mtp.q_proj");
        TQ_READ_MTP_PROJ(m->k_proj, kv_m, g_qwen.H, "mtp.k_proj");
        TQ_READ_MTP_PROJ(m->v_proj, kv_m, g_qwen.H, "mtp.v_proj");
        TQ_READ_MTP_PROJ(m->o_proj, g_qwen.H, attn_m, "mtp.o_proj");
        TQ_READ_MTP_PROJ(m->mlp_gate, g_qwen.I, g_qwen.H, "mtp.mlp_gate");
        TQ_READ_MTP_PROJ(m->mlp_up, g_qwen.I, g_qwen.H, "mtp.mlp_up");
        TQ_READ_MTP_PROJ(m->mlp_down, g_qwen.H, g_qwen.I, "mtp.mlp_down");
        if (read_non_quant(f, &g_qwen.d_mtp_norm, g_qwen.H, "mtp.norm", mup) != 0) { fclose(f); return -1; }
        #undef TQ_READ_MTP_PROJ
        g_qwen.mtp_exl3 = mup;
        if (mup) printf("  MTP head section loaded (EXL3 projections: the paged drafter)\n");
        else if (no_mtp) {
            g_qwen.has_mtp_section = 0;
            printf("  MTP head section skipped (TQ_NO_MTP)\n");
        } else if (upload) printf("  MTP head section skipped (bf16 projections: the paged drafter needs a --mtp-exl3 file)\n");
    }
    // The DRAFT lm_head section (TQ_FLAG_DRAFT_HEAD, convert_qwen_tqf.py --draft-head): read with the MTP head when
    // TQ_DRAFT_VOCAB=1 (the drafter then argmaxes over its V' rows and maps the row to the token id), read past otherwise.
    if (g_qwen.flags & TQ_FLAG_DRAFT_HEAD) {
        int dup = g_qwen.mtp_exl3 && g_cfg.draft_vocab;
        float meta[4];
        if (read_exact(f, meta, sizeof(meta), "draft.meta") != 0) { fclose(f); return -1; }
        g_qwen.f32_count++;
        int dv = (int)meta[0];
        if (dv <= 0 || (dv % 128) != 0 || dv > g_qwen.V) {
            fprintf(stderr, "draft.meta: bad draft vocabulary size %d (V=%d)\n", dv, g_qwen.V); fclose(f); return -1;
        }
        if (read_f32(f, &g_qwen.d_draft_d2t, (size_t)dv, "draft.d2t", dup) != 0) { fclose(f); return -1; }
        if (read_exl3(f, &g_qwen.draft_head, dv, g_qwen.H, "draft.lm_head", dup) != 0) { fclose(f); return -1; }
        g_qwen.draft_V = dv;
        g_qwen.draft_on = dup;
        if (dup) printf("  DRAFT lm_head section loaded: the drafter argmaxes over %d token ids (%d-bit EXL3; TQ_DRAFT_VOCAB=0 = the full lm_head)\n",
                        dv, g_qwen.draft_head.exl3_kbits);
        else if (upload) printf("  DRAFT lm_head section skipped (%s)\n", g_qwen.mtp_exl3 ? "TQ_DRAFT_VOCAB=0" : "no MTP head loaded");
    }

    long pos = ftell(f);
    long end = -1;
    if (pos >= 0 && fseek(f, 0, SEEK_END) == 0) {
        end = ftell(f);
    }
    fclose(f);
    if (pos >= 0 && end >= 0 && pos != end) {
        fprintf(stderr, "TQF payload mismatch: stopped at %ld of %ld\n", pos, end);
        return -1;
    }
    g_qwen.initialized = upload ? 1 : 0;
    printf("TQF1 %s complete: tensors non_quant=%d qm8=%d f32=%d device_bytes=%zu\n",
           upload ? "load" : "probe", g_qwen.non_quant_count, g_qwen.qmma_count,
           g_qwen.f32_count, g_qwen.device_bytes);
    return 0;
}

// Validation/utility ABI: quantize+pack one BF16 weight to dense E2M3 on the GPU
// and copy the dense d_A payload (Mt*Kt*384 bytes) and pow2 scale_inv back to host.
extern "C" int qwn_gpu_pack_e2m3_test(const uint16_t *h_bf16, int M, int K,
                                      uint8_t *h_out_dA, float *h_out_scale) {
    if ((M % 16) != 0 || (K % 32) != 0) return -1;
    int Mt = M / 16, Kt = K / 32;
    int sr = (M + 127) / 128, sc = (K + 127) / 128;
    size_t bf_bytes = (size_t)M * K * sizeof(uint16_t);
    size_t da_bytes = (size_t)Mt * Kt * 384u;
    size_t sc_bytes = (size_t)sr * sc * sizeof(float);
    uint16_t *d_bf16 = NULL;
    uint32_t *d_dA = NULL;
    float *d_scale = NULL;
    if (cudaMalloc(&d_bf16, bf_bytes) != cudaSuccess) return -2;
    if (cudaMalloc(&d_dA, da_bytes) != cudaSuccess) { cudaFree(d_bf16); return -2; }
    if (cudaMalloc(&d_scale, sc_bytes) != cudaSuccess) { cudaFree(d_bf16); cudaFree(d_dA); return -2; }
    cudaMemcpy(d_bf16, h_bf16, bf_bytes, cudaMemcpyHostToDevice);
    dim3 g1(sr, sc);
    k_tq_block_absmax_pow2_e2m3<<<g1, 256>>>(d_bf16, M, K, sc, d_scale);
    long total = (long)Mt * Kt * 96;
    int threads = 256;
    long blocks = (total + threads - 1) / threads;
    k_tq_pack_bf16_to_e2m3_dense<<<(unsigned)blocks, threads>>>(d_bf16, M, K, sc, d_scale, d_dA, Kt);
    cudaError_t err = cudaDeviceSynchronize();
    if (err == cudaSuccess) {
        cudaMemcpy(h_out_dA, d_dA, da_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_out_scale, d_scale, sc_bytes, cudaMemcpyDeviceToHost);
    }
    cudaFree(d_bf16);
    cudaFree(d_dA);
    cudaFree(d_scale);
    return (err == cudaSuccess) ? 0 : -3;
}

static int ensure_attn_scores(void) {
    if (g_qwen.d_attn_scores) return 0;
    if (g_qwen.max_seq <= 0) g_qwen.max_seq = 8192;
    size_t bytes = (size_t)TQ_ATTN_SCORE_SLOTS * g_qwen.max_seq * sizeof(float);
    if (cudaMalloc(&g_qwen.d_attn_scores, bytes) != cudaSuccess) {
        fprintf(stderr, "attn scores scratch alloc failed (%zu MB)\n", bytes >> 20);
        return -1;
    }
    return 0;
}

extern "C" int qwn_probe(const char *path) {
    qwn_config c;
    qwn_config_from_env(&c);
    tq_config_apply(&c);
    return parse_tqf(path, 0);
}

// qwn_init with this configuration instead of the TQ_* environment (src/engine/config.cuh).
extern "C" int qwn_init_with(const char *path, const qwn_config *cfg) {
    if (!cfg) return -1;
    tq_config_apply(cfg);
    tq_config_print(stderr, &g_cfg);
    int r = parse_tqf(path, 1);
    if (r != 0) return r;
    // AFTER parse_tqf: it memsets g_qwen, so the runtime context limit must be
    // set here (and before any ensure path allocates KV/scratch).
    g_qwen.max_seq = g_cfg.ctx;
    fprintf(stderr, "context limit: %d tokens (TQ_CTX, cap %d)\n", g_qwen.max_seq, TQ_MAX_SEQ);
    g_qwen.kv_q4 = g_cfg.kv_q4;
    if (g_qwen.kv_q4 && g_qwen.hd != 256) {
        fprintf(stderr, "TQ_KV_Q4=1 requires hd==256 (got %d) -- disabled\n", g_qwen.hd);
        g_qwen.kv_q4 = 0;
    }
    g_qwen.kv_fp8 = g_qwen.kv_q4;   // V rows + scales ride the FP8 machinery
    if (g_qwen.kv_q4)
        fprintf(stderr, "KV prefix cache: K int4 asym g32 + Hadamard rotation, "
                        "V E4M3 + pow2 scale (TQ_KV_Q4=1); (scale,zp) groups fp16\n");
    return 0;
}

extern "C" int qwn_init(const char *path) {
    qwn_config c;
    qwn_config_from_env(&c);
    return qwn_init_with(path, &c);
}

extern "C" int qwn_hidden_size(void) { return g_qwen.H; }
extern "C" int qwn_intermediate_size(void) { return g_qwen.I; }
extern "C" int qwn_vocab_size(void) { return g_qwen.V; }
extern "C" int qwn_num_layers(void) { return g_qwen.L; }
extern "C" int qwn_num_attention_heads(void) { return g_qwen.nh; }
extern "C" int qwn_num_key_value_heads(void) { return g_qwen.nkv; }
extern "C" int qwn_head_dim(void) { return g_qwen.hd; }
extern "C" int qwn_max_seq(void) { return g_qwen.max_seq; }
extern "C" int qwn_layer_type(int layer) {
    if (layer < 0 || layer >= g_qwen.L) return 0;
    return (int)g_qwen.layer_types[layer];
}
extern "C" size_t qwn_device_bytes(void) { return g_qwen.device_bytes; }
extern "C" int qwn_non_quant_count(void) { return g_qwen.non_quant_count; }
extern "C" int qwn_qmma_count(void) { return g_qwen.qmma_count; }
extern "C" int qwn_f32_count(void) { return g_qwen.f32_count; }
static int ensure_debug_vectors(int proj_count) {
    if (!g_qwen.d_debug_x) {
        cudaError_t err = cudaMalloc(&g_qwen.d_debug_x, (size_t)g_qwen.H * sizeof(float));
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMalloc d_debug_x failed: %s\n", cudaGetErrorString(err));
            return -1;
        }
        g_qwen.device_bytes += (size_t)g_qwen.H * sizeof(float);
    }
    if (!g_qwen.d_debug_norm) {
        cudaError_t err = cudaMalloc(&g_qwen.d_debug_norm, (size_t)g_qwen.H * sizeof(float));
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMalloc d_debug_norm failed: %s\n", cudaGetErrorString(err));
            return -2;
        }
        g_qwen.device_bytes += (size_t)g_qwen.H * sizeof(float);
    }
    if (proj_count > 0 && g_qwen.debug_proj_count < proj_count) {
        if (g_qwen.d_debug_proj) {
            cudaFree(g_qwen.d_debug_proj);
            g_qwen.device_bytes -= (size_t)g_qwen.debug_proj_count * sizeof(float);
        }
        cudaError_t err = cudaMalloc(&g_qwen.d_debug_proj, (size_t)proj_count * sizeof(float));
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMalloc d_debug_proj failed: %s\n", cudaGetErrorString(err));
            g_qwen.d_debug_proj = NULL;
            g_qwen.debug_proj_count = 0;
            return -3;
        }
        g_qwen.debug_proj_count = proj_count;
        g_qwen.device_bytes += (size_t)proj_count * sizeof(float);
    }
    return 0;
}

static int ensure_qmma_debug_scratch(const tq_qmma_weight_t *w) {
    int ret = ensure_debug_vectors(w->M);
    if (ret != 0) return ret;
    int b_bytes = w->Kt * 256;
    if (g_qwen.debug_b_bytes < b_bytes) {
        if (g_qwen.d_debug_b) {
            cudaFree(g_qwen.d_debug_b);
            g_qwen.device_bytes -= (size_t)g_qwen.debug_b_bytes;
        }
        cudaError_t err = cudaMalloc(&g_qwen.d_debug_b, (size_t)b_bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMalloc d_debug_b failed: %s\n", cudaGetErrorString(err));
            g_qwen.d_debug_b = NULL;
            g_qwen.debug_b_bytes = 0;
            return -10;
        }
        g_qwen.debug_b_bytes = b_bytes;
        g_qwen.device_bytes += (size_t)b_bytes;
    }
    int raw_count = w->Mt * 128;
    if (g_qwen.debug_raw_count < raw_count) {
        if (g_qwen.d_debug_qmma_raw) {
            cudaFree(g_qwen.d_debug_qmma_raw);
            g_qwen.device_bytes -= (size_t)g_qwen.debug_raw_count * sizeof(float);
        }
        cudaError_t err = cudaMalloc(&g_qwen.d_debug_qmma_raw, (size_t)raw_count * sizeof(float));
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMalloc d_debug_qmma_raw failed: %s\n", cudaGetErrorString(err));
            g_qwen.d_debug_qmma_raw = NULL;
            g_qwen.debug_raw_count = 0;
            return -11;
        }
        g_qwen.debug_raw_count = raw_count;
        g_qwen.device_bytes += (size_t)raw_count * sizeof(float);
    }
    if (g_qwen.debug_qmma_out_count < w->M) {
        if (g_qwen.d_debug_qmma_out) {
            cudaFree(g_qwen.d_debug_qmma_out);
            g_qwen.device_bytes -= (size_t)g_qwen.debug_qmma_out_count * sizeof(float);
        }
        cudaError_t err = cudaMalloc(&g_qwen.d_debug_qmma_out, (size_t)w->M * sizeof(float));
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMalloc d_debug_qmma_out failed: %s\n", cudaGetErrorString(err));
            g_qwen.d_debug_qmma_out = NULL;
            g_qwen.debug_qmma_out_count = 0;
            return -12;
        }
        g_qwen.debug_qmma_out_count = w->M;
        g_qwen.device_bytes += (size_t)w->M * sizeof(float);
    }
    if (!g_qwen.d_debug_scale) {
        cudaError_t err = cudaMalloc(&g_qwen.d_debug_scale, sizeof(float));
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMalloc d_debug_scale failed: %s\n", cudaGetErrorString(err));
            return -13;
        }
        g_qwen.device_bytes += sizeof(float);
    }
    return 0;
}

static int ensure_float_buffer(float **ptr, int *capacity, int count, const char *name) {
    if (*capacity >= count) return 0;
    if (*ptr) {
        cudaFree(*ptr);
        g_qwen.device_bytes -= (size_t)(*capacity) * sizeof(float);
    }
    cudaError_t err = cudaMalloc(ptr, (size_t)count * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc %s failed: %s\n", name, cudaGetErrorString(err));
        *ptr = NULL;
        *capacity = 0;
        return -1;
    }
    *capacity = count;
    g_qwen.device_bytes += (size_t)count * sizeof(float);
    return 0;
}

static int ensure_int_buffer(int **ptr, int *capacity, int count, const char *name) {
    if (*capacity >= count) return 0;
    if (*ptr) {
        cudaFree(*ptr);
        g_qwen.device_bytes -= (size_t)(*capacity) * sizeof(int);
    }
    cudaError_t err = cudaMalloc(ptr, (size_t)count * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc %s failed: %s\n", name, cudaGetErrorString(err));
        *ptr = NULL;
        *capacity = 0;
        return -1;
    }
    *capacity = count;
    g_qwen.device_bytes += (size_t)count * sizeof(int);
    return 0;
}

static int ensure_decode_state(void) {
    int lin_key_dim = g_qwen.linear_num_key_heads * g_qwen.linear_key_head_dim;
    int lin_value_dim = g_qwen.linear_num_value_heads * g_qwen.linear_value_head_dim;
    int lin_conv_dim = lin_key_dim * 2 + lin_value_dim;
    int full_kv_dim = g_qwen.nkv * g_qwen.hd;
    for (int i = 0; i < g_qwen.L; i++) {
        tq_layer_t *l = &g_qwen.layers[i];
        if (g_qwen.layer_types[i] == TQ_LAYER_LINEAR_ATTENTION) {
            size_t conv_bytes = (size_t)lin_conv_dim * g_qwen.linear_conv_kernel_dim * sizeof(float);
            size_t state_bytes = (size_t)g_qwen.linear_num_value_heads *
                                 g_qwen.linear_key_head_dim *
                                 g_qwen.linear_value_head_dim * sizeof(float);
            if (!l->d_linear_conv_state) {
                cudaError_t err = cudaMalloc(&l->d_linear_conv_state, conv_bytes);
                if (err != cudaSuccess) return -1;
                g_qwen.device_bytes += conv_bytes;
            }
            if (!l->d_linear_recurrent_state) {
                cudaError_t err = cudaMalloc(&l->d_linear_recurrent_state, state_bytes);
                if (err != cudaSuccess) return -2;
                g_qwen.device_bytes += state_bytes;
            }
        } else if (g_qwen.layer_types[i] == TQ_LAYER_FULL_ATTENTION) {
            if (g_qwen.kv_fp8) {
                // FP8 prefix cache: u8 codes + per-(pos, kv_head) scales; the fp32
                // cache is never allocated on this path (-75% KV bytes).
                size_t code_bytes = (size_t)g_qwen.max_seq * full_kv_dim;
                size_t scale_bytes = (size_t)g_qwen.max_seq * g_qwen.nkv * sizeof(float);
                if (g_qwen.kv_q4) {
                    // K side: int4 codes (half the bytes) + per-group (scale, zp)
                    size_t k4_bytes = code_bytes >> 1;
                    size_t k4s_bytes = (size_t)g_qwen.max_seq * g_qwen.nkv * 16 * sizeof(__half);
                    if (!l->d_k_cache4) {
                        if (cudaMalloc(&l->d_k_cache4, k4_bytes) != cudaSuccess) return -3;
                        g_qwen.device_bytes += k4_bytes;
                    }
                    if (!l->d_k_q4s) {
                        if (cudaMalloc(&l->d_k_q4s, k4s_bytes) != cudaSuccess) return -3;
                        g_qwen.device_bytes += k4s_bytes;
                    }
                } else {
                    if (!l->d_k_cache8) {
                        if (cudaMalloc(&l->d_k_cache8, code_bytes) != cudaSuccess) return -3;
                        g_qwen.device_bytes += code_bytes;
                    }
                    if (!l->d_k_scale) {
                        if (cudaMalloc(&l->d_k_scale, scale_bytes) != cudaSuccess) return -3;
                        g_qwen.device_bytes += scale_bytes;
                    }
                }
                if (!l->d_v_cache8) {
                    if (cudaMalloc(&l->d_v_cache8, code_bytes) != cudaSuccess) return -4;
                    g_qwen.device_bytes += code_bytes;
                }
                if (!l->d_v_scale) {
                    if (cudaMalloc(&l->d_v_scale, scale_bytes) != cudaSuccess) return -4;
                    g_qwen.device_bytes += scale_bytes;
                }
                continue;
            }
            size_t cache_bytes = (size_t)g_qwen.max_seq * full_kv_dim * sizeof(float);
            if (!l->d_k_cache) {
                cudaError_t err = cudaMalloc(&l->d_k_cache, cache_bytes);
                if (err != cudaSuccess) return -3;
                g_qwen.device_bytes += cache_bytes;
            }
            if (!l->d_v_cache) {
                cudaError_t err = cudaMalloc(&l->d_v_cache, cache_bytes);
                if (err != cudaSuccess) return -4;
                g_qwen.device_bytes += cache_bytes;
            }
        }
    }
    return 0;
}

extern "C" int qwn_reset_state(void) {
    int ret = ensure_decode_state();
    if (ret != 0) return ret;
    int lin_key_dim = g_qwen.linear_num_key_heads * g_qwen.linear_key_head_dim;
    int lin_value_dim = g_qwen.linear_num_value_heads * g_qwen.linear_value_head_dim;
    int lin_conv_dim = lin_key_dim * 2 + lin_value_dim;
    int full_kv_dim = g_qwen.nkv * g_qwen.hd;
    for (int i = 0; i < g_qwen.L; i++) {
        tq_layer_t *l = &g_qwen.layers[i];
        if (g_qwen.layer_types[i] == TQ_LAYER_LINEAR_ATTENTION) {
            cudaMemsetAsync(l->d_linear_conv_state, 0,
                            (size_t)lin_conv_dim * g_qwen.linear_conv_kernel_dim * sizeof(float),
                            g_qwen.stream);
            cudaMemsetAsync(l->d_linear_recurrent_state, 0,
                            (size_t)g_qwen.linear_num_value_heads *
                            g_qwen.linear_key_head_dim *
                            g_qwen.linear_value_head_dim * sizeof(float),
                            g_qwen.stream);
        } else if (g_qwen.layer_types[i] == TQ_LAYER_FULL_ATTENTION) {
            if (g_qwen.kv_fp8) {
                if (g_qwen.kv_q4) {
                    cudaMemsetAsync(l->d_k_cache4, 0, ((size_t)g_qwen.max_seq * full_kv_dim) >> 1, g_qwen.stream);
                    cudaMemsetAsync(l->d_k_q4s, 0, (size_t)g_qwen.max_seq * g_qwen.nkv * 16 * sizeof(__half), g_qwen.stream);
                } else {
                    cudaMemsetAsync(l->d_k_cache8, 0, (size_t)g_qwen.max_seq * full_kv_dim, g_qwen.stream);
                    cudaMemsetAsync(l->d_k_scale, 0, (size_t)g_qwen.max_seq * g_qwen.nkv * sizeof(float), g_qwen.stream);
                }
                cudaMemsetAsync(l->d_v_cache8, 0, (size_t)g_qwen.max_seq * full_kv_dim, g_qwen.stream);
                cudaMemsetAsync(l->d_v_scale, 0, (size_t)g_qwen.max_seq * g_qwen.nkv * sizeof(float), g_qwen.stream);
                continue;
            }
            cudaMemsetAsync(l->d_k_cache, 0, (size_t)g_qwen.max_seq * full_kv_dim * sizeof(float), g_qwen.stream);
            cudaMemsetAsync(l->d_v_cache, 0, (size_t)g_qwen.max_seq * full_kv_dim * sizeof(float), g_qwen.stream);
        }
    }
    cudaError_t err = cudaStreamSynchronize(g_qwen.stream);
    return err == cudaSuccess ? 0 : -5;
}
