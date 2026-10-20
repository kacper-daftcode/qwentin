// the engine's configuration: every TQ_* environment switch, parsed in one place when qwn_init (or qwn_probe) runs.
// Part of the forward_qwen.cu unity build: included in order, not a standalone header.
//
// The other parts read g_cfg and never the environment, so a process configures the engine by setting its environment
// before qwn_init; nothing re-reads it later. One rule for every switch: unset or empty = the default; otherwise the
// integer value (atoi), on / off for a boolean switch (any non-zero value is on); an integer outside a switch's range
// is clamped, or replaced by the value in the table's last column. docs/FLAGS.md describes the same switches in the
// same four groups; tests/test_config.py checks both lists against each other and these rules.

#include <limits.h>

typedef struct {
    // the configuration a deployment sets (tools/serve_batched.py sets several of these from its arguments)
    int ctx, kv_q4, kv_v4, kv_t4, dstate, embed_host, embed_fp8, no_mtp, verify_tqf, pdl, wide_attn_mma, paged_attn_mma,
        gdn_prefill;
    char *kv_t4_codebook;   // TQ_KV_T4_CODEBOOK: a file of 4096 fp32 that replaces the built-in codebook; NULL = built-in
    // off-switches and opt-in experiments: a non-default value restores an earlier served line or runs a measured variant
    int t4_attn_f16, paged_attn_st, graph, gemm_fork, gemm_fork_maxn, exl3_wide_ng16, exl3_bf16_r4, pattn_gqa, wave_max,
        wide_attn_qrows, kv_t4_async, paged_topk_pages, paged_topk_min_ctx;
    // pins for tools and tests
    int park_ring, qmma_sf_k_split, paged_gqa_split;
    // diagnostics
    int pattn_debug, exl3_debug, graph_debug, paged_topk_debug;
} tq_config_t;

enum { TQ_SW_INT, TQ_SW_BOOL };
#define TQ_SW_CLAMP INT_MIN
typedef struct {
    const char *name;
    int tq_config_t::*field;
    int kind, dflt;
    int lo, hi, out;        // TQ_SW_INT: a value outside [lo, hi] becomes `out` (TQ_SW_CLAMP: the nearer bound)
} tq_switch_t;

static const tq_switch_t k_tq_switches[] = {
    // name                    field                               kind        default  lo    hi          out of range
    {"TQ_CTX",                 &tq_config_t::ctx,                 TQ_SW_INT,  8192,    2048, TQ_MAX_SEQ, TQ_SW_CLAMP},
    {"TQ_KV_Q4",               &tq_config_t::kv_q4,               TQ_SW_BOOL, 0},
    {"TQ_KV_V4",               &tq_config_t::kv_v4,               TQ_SW_BOOL, 0},
    {"TQ_KV_T4",               &tq_config_t::kv_t4,               TQ_SW_BOOL, 0},
    {"TQ_DSTATE_BF16",         &tq_config_t::dstate,              TQ_SW_INT,  4,       INT_MIN, INT_MAX, TQ_SW_CLAMP},   // qwn_paged_init refuses all but 0..4
    {"TQ_EMBED_HOST",          &tq_config_t::embed_host,          TQ_SW_BOOL, 0},
    {"TQ_EMBED_FP8",           &tq_config_t::embed_fp8,           TQ_SW_INT,  0,       0,    2,          1},
    {"TQ_NO_MTP",              &tq_config_t::no_mtp,              TQ_SW_BOOL, 0},
    {"TQ_VERIFY_TQF",          &tq_config_t::verify_tqf,          TQ_SW_BOOL, 0},
    {"TQ_PDL",                 &tq_config_t::pdl,                 TQ_SW_INT,  0,       0,    2,          0},
    {"TQ_WIDE_ATTN_MMA",       &tq_config_t::wide_attn_mma,       TQ_SW_BOOL, 0},
    {"TQ_PAGED_ATTN_MMA",      &tq_config_t::paged_attn_mma,      TQ_SW_BOOL, 1},
    {"TQ_GDN_PREFILL",         &tq_config_t::gdn_prefill,         TQ_SW_INT,  1,       0,    3,          0},
    // off-switches and opt-in experiments
    {"TQ_T4_ATTN_F16",         &tq_config_t::t4_attn_f16,         TQ_SW_BOOL, 1},
    {"TQ_PAGED_ATTN_ST",       &tq_config_t::paged_attn_st,       TQ_SW_INT,  3,       0,    3,          3},
    {"TQ_GRAPH",               &tq_config_t::graph,               TQ_SW_INT,  1,       0,    INT_MAX,    TQ_SW_CLAMP},
    {"TQ_GEMM_FORK",           &tq_config_t::gemm_fork,           TQ_SW_BOOL, 1},
    {"TQ_GEMM_FORK_MAXN",      &tq_config_t::gemm_fork_maxn,      TQ_SW_INT,  64,      0,    INT_MAX,    TQ_SW_CLAMP},
    {"TQ_EXL3_WIDE_NG16",      &tq_config_t::exl3_wide_ng16,      TQ_SW_BOOL, 1},
    {"TQ_EXL3_BF16_R4",        &tq_config_t::exl3_bf16_r4,        TQ_SW_BOOL, 1},
    {"TQ_PATTN_GQA",           &tq_config_t::pattn_gqa,           TQ_SW_INT,  1,       INT_MIN, INT_MAX, TQ_SW_CLAMP},
    {"TQ_WAVE_MAX",            &tq_config_t::wave_max,            TQ_SW_INT,  128,     1,    256,        TQ_SW_CLAMP},
    {"TQ_WIDE_ATTN_QROWS",     &tq_config_t::wide_attn_qrows,     TQ_SW_INT,  32,      16,   16,         32},    // 16 or 32
    {"TQ_KV_T4_ASYNC",         &tq_config_t::kv_t4_async,         TQ_SW_BOOL, 1},
    {"TQ_PAGED_TOPK_PAGES",    &tq_config_t::paged_topk_pages,    TQ_SW_INT,  0,       0,    4096,       TQ_SW_CLAMP},
    {"TQ_PAGED_TOPK_MIN_CTX",  &tq_config_t::paged_topk_min_ctx,  TQ_SW_INT,  16384,   0,    INT_MAX,    TQ_SW_CLAMP},
    // pins for tools and tests
    {"TQ_PARK_RING",           &tq_config_t::park_ring,           TQ_SW_BOOL, 1},
    {"TQ_QMMA_SF_K_SPLIT",     &tq_config_t::qmma_sf_k_split,     TQ_SW_INT,  0,       1,    8,          TQ_SW_CLAMP},   // 0 = auto
    {"TQ_PAGED_GQA_SPLIT",     &tq_config_t::paged_gqa_split,     TQ_SW_INT,  -1,      INT_MIN, INT_MAX, TQ_SW_CLAMP},   // < 1 = auto
    // diagnostics
    {"TQ_PATTN_DEBUG",         &tq_config_t::pattn_debug,         TQ_SW_BOOL, 0},
    {"TQ_EXL3_DEBUG",          &tq_config_t::exl3_debug,          TQ_SW_BOOL, 0},
    {"TQ_GRAPH_DEBUG",         &tq_config_t::graph_debug,         TQ_SW_BOOL, 0},
    {"TQ_PAGED_TOPK_DEBUG",    &tq_config_t::paged_topk_debug,    TQ_SW_BOOL, 0},
};

static int tq_switch_parse(const tq_switch_t *s, const char *e) {
    if (!e || !*e) return s->dflt;
    int v = atoi(e);
    if (s->kind == TQ_SW_BOOL) return v != 0;
    if (v < s->lo || v > s->hi) return s->out != TQ_SW_CLAMP ? s->out : (v < s->lo ? s->lo : s->hi);
    return v;
}

static tq_config_t tq_config_defaults(void) {
    tq_config_t c;
    memset(&c, 0, sizeof(c));
    for (const tq_switch_t &s : k_tq_switches) c.*s.field = s.dflt;
    return c;
}

// c must hold a configuration (the defaults or an earlier parse): its codebook string is replaced.
static void tq_config_from_env(tq_config_t *c) {
    for (const tq_switch_t &s : k_tq_switches) c->*s.field = tq_switch_parse(&s, getenv(s.name));
    const char *e = getenv("TQ_KV_T4_CODEBOOK");
    free(c->kv_t4_codebook);
    c->kv_t4_codebook = (e && *e) ? strdup(e) : NULL;
}

// The switches that differ from their defaults, as one line ("every switch at its default" when none does).
static void tq_config_print(FILE *f, const tq_config_t *c) {
    const int total = (int)(sizeof(k_tq_switches) / sizeof(k_tq_switches[0])) + 1;
    int n = 0;
    fprintf(f, "[config]");
    for (const tq_switch_t &s : k_tq_switches)
        if (c->*s.field != s.dflt) { fprintf(f, " %s=%d", s.name, c->*s.field); n++; }
    if (c->kv_t4_codebook) { fprintf(f, " TQ_KV_T4_CODEBOOK=%s", c->kv_t4_codebook); n++; }
    if (n) fprintf(f, " (the other %d TQ_* switches at their defaults)\n", total - n);
    else fprintf(f, " every TQ_* switch at its default\n");
}

static tq_config_t g_cfg = tq_config_defaults();
static void tq_config_load(void) { tq_config_from_env(&g_cfg); }
