/*
 * forward_qwen.cu - the qwentin engine: Qwen3.6/3.8 27B (gated DeltaNet + GQA) on SM120 GPUs, one CUDA translation
 * unit behind the qwn_* C ABI, built from the parts in src/engine/ in the order below (a unity build: the parts
 * share statics and are not standalone headers). It loads .tqf model files (docs/TQF_FORMAT.md) and is driven by
 * tools/serve_batched.py (paged KV pool, continuous batching, speculative decoding) and the legacy single-stream
 * tools/serve_openai.py. How it works: docs/ARCHITECTURE.md; environment flags: docs/FLAGS.md.
 */
#include "qwentin.h"
#include "engine/base.cuh"
#include "engine/kv_cache.cuh"
#include "engine/t4_kv.cuh"
#include "engine/attn_group.cuh"
#include "engine/attn_prefill.cuh"
#include "engine/attn_tree.cuh"
#include "engine/quant_bfrag.cuh"
#include "engine/gemm_wide.cuh"
#include "engine/spec_persistent.cuh"
#include "engine/load_quant.cuh"
#include "engine/loader.cuh"
#include "engine/forward_single.cuh"
#include "engine/mtp.cuh"
#include "engine/spec_persistent_host.cuh"
#include "engine/deltanet_prefill.cuh"
#include "engine/spec_driver.cuh"
#include "engine/exl3.cuh"
#include "engine/batched_decode.cuh"
#include "engine/paged_pool.cuh"
#include "engine/paged_attn.cuh"
#include "engine/paged_t4.cuh"
#include "engine/parking.cuh"
#include "engine/paged_graph.cuh"
#include "engine/paged_spec.cuh"
#include "engine/chunked_prefill.cuh"
#include "engine/tests.cuh"
