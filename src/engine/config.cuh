// the engine's configuration: every TQ_* switch is one field of qwn_config (src/qwentin.h), filled from the environment
// or by the host program. Part of the forward_qwen.cu unity build: included in order, not a standalone header.
//
// qwn_init(path) = qwn_config_from_env + qwn_init_with(path, cfg): the environment is read once, there, and the other
// parts read g_cfg, never the environment. A host program may fill a qwn_config itself (qwn_config_defaults, then its
// own values) and call qwn_init_with. One rule for every switch: unset or empty = the default; otherwise the integer
// value (atoi), on / off for a boolean switch (any non-zero value is on); an integer outside a switch's range is clamped,
// or replaced by the value in the table's last column. qwn_init_with applies the same rule to the fields it receives,
// except that a field holding its switch's default is taken as is.
// The defaults are the served line's (docs/FLAGS.md lists the earlier ones). tools/gen_qwentin_h.py generates qwn_config
// and its ctypes twin from the table below (field order and "group:" comments); docs/FLAGS.md describes the same
// switches in the same four groups; tests/test_config.py checks the lists against each other and pins the rule.

#include <limits.h>

enum { TQ_SW_INT, TQ_SW_BOOL };
#define TQ_SW_CLAMP INT_MIN
typedef struct {
    const char *name;
    int qwn_config::*field;
    int kind, dflt;
    int lo, hi, out;        // TQ_SW_INT: a value outside [lo, hi] becomes `out` (TQ_SW_CLAMP: the nearer bound)
} tq_switch_t;

static const tq_switch_t k_tq_switches[] = {
    // name                    field                              kind        default  lo    hi          out of range
    // group: the configuration a deployment sets
    {"TQ_CTX",                 &qwn_config::ctx,                 TQ_SW_INT,  262144,  2048, TQ_MAX_SEQ, TQ_SW_CLAMP},
    {"TQ_KV_Q4",               &qwn_config::kv_q4,               TQ_SW_BOOL, 1},
    {"TQ_KV_V4",               &qwn_config::kv_v4,               TQ_SW_BOOL, 1},
    {"TQ_KV_T4",               &qwn_config::kv_t4,               TQ_SW_BOOL, 1},
    {"TQ_DSTATE_BF16",         &qwn_config::dstate,              TQ_SW_INT,  4,       INT_MIN, INT_MAX, TQ_SW_CLAMP},   // qwn_paged_init refuses all but 0..4
    {"TQ_EMBED_HOST",          &qwn_config::embed_host,          TQ_SW_BOOL, 1},
    {"TQ_EMBED_FP8",           &qwn_config::embed_fp8,           TQ_SW_INT,  0,       0,    2,          1},
    {"TQ_NO_MTP",              &qwn_config::no_mtp,              TQ_SW_BOOL, 0},
    {"TQ_VERIFY_TQF",          &qwn_config::verify_tqf,          TQ_SW_BOOL, 0},
    {"TQ_PDL",                 &qwn_config::pdl,                 TQ_SW_INT,  1,       0,    2,          0},
    {"TQ_WIDE_ATTN_MMA",       &qwn_config::wide_attn_mma,       TQ_SW_BOOL, 1},
    {"TQ_PAGED_ATTN_MMA",      &qwn_config::paged_attn_mma,      TQ_SW_BOOL, 1},
    {"TQ_GDN_PREFILL",         &qwn_config::gdn_prefill,         TQ_SW_INT,  1,       0,    3,          0},
    // group: off-switches and opt-in experiments
    {"TQ_T4_ATTN_F16",         &qwn_config::t4_attn_f16,         TQ_SW_BOOL, 1},
    {"TQ_PAGED_ATTN_ST",       &qwn_config::paged_attn_st,       TQ_SW_INT,  3,       0,    3,          3},
    {"TQ_GRAPH",               &qwn_config::graph,               TQ_SW_INT,  1,       0,    INT_MAX,    TQ_SW_CLAMP},
    {"TQ_GEMM_FORK",           &qwn_config::gemm_fork,           TQ_SW_BOOL, 1},
    {"TQ_GEMM_FORK_MAXN",      &qwn_config::gemm_fork_maxn,      TQ_SW_INT,  64,      0,    INT_MAX,    TQ_SW_CLAMP},
    {"TQ_EXL3_WIDE_NG16",      &qwn_config::exl3_wide_ng16,      TQ_SW_BOOL, 1},
    {"TQ_EXL3_BF16_R4",        &qwn_config::exl3_bf16_r4,        TQ_SW_BOOL, 1},
    {"TQ_PATTN_GQA",           &qwn_config::pattn_gqa,           TQ_SW_INT,  1,       INT_MIN, INT_MAX, TQ_SW_CLAMP},
    {"TQ_WAVE_MAX",            &qwn_config::wave_max,            TQ_SW_INT,  128,     1,    256,        TQ_SW_CLAMP},
    {"TQ_WIDE_ATTN_QROWS",     &qwn_config::wide_attn_qrows,     TQ_SW_INT,  32,      16,   16,         32},    // 16 or 32
    {"TQ_KV_T4_ASYNC",         &qwn_config::kv_t4_async,         TQ_SW_BOOL, 1},
    {"TQ_PAGED_TOPK_PAGES",    &qwn_config::paged_topk_pages,    TQ_SW_INT,  0,       0,    4096,       TQ_SW_CLAMP},
    {"TQ_PAGED_TOPK_MIN_CTX",  &qwn_config::paged_topk_min_ctx,  TQ_SW_INT,  16384,   0,    INT_MAX,    TQ_SW_CLAMP},
    // group: pins for tools and tests
    {"TQ_PARK_RING",           &qwn_config::park_ring,           TQ_SW_BOOL, 1},
    {"TQ_QMMA_SF_K_SPLIT",     &qwn_config::qmma_sf_k_split,     TQ_SW_INT,  0,       1,    8,          TQ_SW_CLAMP},   // 0 = auto
    {"TQ_PAGED_GQA_SPLIT",     &qwn_config::paged_gqa_split,     TQ_SW_INT,  -1,      INT_MIN, INT_MAX, TQ_SW_CLAMP},   // < 1 = auto
    // group: diagnostics
    {"TQ_PATTN_DEBUG",         &qwn_config::pattn_debug,         TQ_SW_BOOL, 0},
    {"TQ_EXL3_DEBUG",          &qwn_config::exl3_debug,          TQ_SW_BOOL, 0},
    {"TQ_GRAPH_DEBUG",         &qwn_config::graph_debug,         TQ_SW_BOOL, 0},
    {"TQ_PAGED_TOPK_DEBUG",    &qwn_config::paged_topk_debug,    TQ_SW_BOOL, 0},
};

static int tq_switch_norm(const tq_switch_t *s, int v) {
    if (s->kind == TQ_SW_BOOL) return v != 0;
    if (v < s->lo || v > s->hi) return s->out != TQ_SW_CLAMP ? s->out : (v < s->lo ? s->lo : s->hi);
    return v;
}

static int tq_switch_parse(const tq_switch_t *s, const char *e) {
    return (!e || !*e) ? s->dflt : tq_switch_norm(s, atoi(e));
}

// Every switch at its default; kv_t4_codebook NULL (the built-in codebook).
extern "C" void qwn_config_defaults(qwn_config *c) {
    memset(c, 0, sizeof(*c));
    for (const tq_switch_t &s : k_tq_switches) c->*s.field = s.dflt;
}

// The environment as qwn_init reads it. kv_t4_codebook points into the environment (valid while it is unchanged;
// qwn_init_with copies it).
extern "C" void qwn_config_from_env(qwn_config *c) {
    qwn_config_defaults(c);
    for (const tq_switch_t &s : k_tq_switches) c->*s.field = tq_switch_parse(&s, getenv(s.name));
    const char *e = getenv("TQ_KV_T4_CODEBOOK");
    c->kv_t4_codebook = (e && *e) ? e : NULL;
}

static qwn_config tq_config_default_value(void) { qwn_config c; qwn_config_defaults(&c); return c; }
static qwn_config g_cfg = tq_config_default_value();
static char *g_cfg_codebook = NULL;              // g_cfg's own copy of the codebook path

// g_cfg := c, each field normalized by its switch's rule unless it holds the default (TQ_QMMA_SF_K_SPLIT's 0 = auto lies
// outside its 1..8 range), the codebook path copied.
static void tq_config_apply(const qwn_config *c) {
    char *cb = (c->kv_t4_codebook && *c->kv_t4_codebook) ? strdup(c->kv_t4_codebook) : NULL;
    qwn_config n = *c;
    for (const tq_switch_t &s : k_tq_switches)
        if (c->*s.field != s.dflt) n.*s.field = tq_switch_norm(&s, c->*s.field);
    free(g_cfg_codebook);
    g_cfg_codebook = cb;
    n.kv_t4_codebook = cb;
    g_cfg = n;
}

// The configuration the engine runs with; kv_t4_codebook is the engine's copy (valid until the next qwn_init*).
extern "C" void qwn_config_get(qwn_config *c) { *c = g_cfg; }

// The switches that differ from their defaults, as one line ("every switch at its default" when none does).
static void tq_config_print(FILE *f, const qwn_config *c) {
    const int total = (int)(sizeof(k_tq_switches) / sizeof(k_tq_switches[0])) + 1;
    int n = 0;
    fprintf(f, "[config]");
    for (const tq_switch_t &s : k_tq_switches)
        if (c->*s.field != s.dflt) { fprintf(f, " %s=%d", s.name, c->*s.field); n++; }
    if (c->kv_t4_codebook) { fprintf(f, " TQ_KV_T4_CODEBOOK=%s", c->kv_t4_codebook); n++; }
    if (n) fprintf(f, " (the other %d TQ_* switches at their defaults)\n", total - n);
    else fprintf(f, " every TQ_* switch at its default\n");
}
