# Flags — the engine's environment variables and the server's arguments (full reference)

The full reference. Every default is the served line (the README's "Quick start" runs it with no flags). Current as of
`main`: the v0.10 served line (engine `b2e7a3f`, library md5 `484bdb825c36`) plus the changes under "Unreleased" in
[`CHANGELOG.md`](../CHANGELOG.md) (`TQ_PATTN_GQA`, the built-in trellis-KV codebook). Also here: the server's endpoints
and request-side behaviour (thinking, per-request stats, logging), the exposure guards, and the off-switches that
restore each previous served line bit for bit.

## Engine environment flags

Environment flags read by the engine (`src/forward_qwen.cu` and its `src/engine/` parts; the server sets several of them
from its arguments unless they are already in the environment):

| Flag | Default | Meaning |
|------|---------|---------|
| `TQ_CTX` | 262144 | max context per session (the engine cap) |
| `TQ_KV_Q4=1` | — | int4 K paged pool (required by the batched server) |
| `TQ_KV_V4` | server: 1 | V stored like K: FWHT-256-rotated int4, asymmetric per 32 channels (20.0 KiB/token); required by `TQ_KV_T4` (`--kv-v4`) |
| `TQ_KV_T4` | server: 1 | 4-bit trellis-coded K+V for every committed token, 16.5 KiB/token; the open page is staged exact (`--kv-t4`). `TQ_KV_T4_ASYNC` (1): prefill waves attend their own columns exact and encode on a side stream (=0 reproduces the exllamav3 simulation exactly) |
| `TQ_KV_T4_CODEBOOK` | unset | a file of 4096 raw fp32 entries that replaces the built-in codebook (`--kv-t4-codebook`); unset = the Lloyd-trained `lloyd_kv12a` compiled into the engine (`src/engine/kv_t4_codebook.inc`, generated from `tools/codebooks/lloyd_kv12a.f32`) |
| `TQ_VERIFY_TQF` | 0 | 1 = also hash the whole payload of a format-v2 `.tqf` at load and refuse a mismatch (one extra read, ~20 s for 21 GB); the header hash and the recorded payload size are checked on every load (docs/TQF_FORMAT.md) |
| `TQ_T4_ATTN_F16` | 1 | the trellis pool's decode step and verify wave run the fp16-operand segment kernel with cp.async-staged V codes (one kernel for both; −1.0 ms/step at 32K N = 8); 0 = the bf16 per-column + segment kernels (the 10-15 path, bit for bit) |
| `TQ_PAGED_ATTN_ST` | 3 | FP4 / E4M3 pools only (a no-op under `TQ_KV_T4`): 0 base GQA decode kernel, 1 staged-V kernel (bitwise = base), 2 fp16 P·V, 3 also fp16 K dequant and Q (−1.5 ms/step at 32K N = 8, verify == decode) |
| `TQ_GRAPH` | 1 | the paged decode step as a CUDA graph (captured per N / key split / MTP columns / flags, LRU of instantiated graphs; a buffer grown inside a capture drops the graph and recaptures); 2 = also the verify wave and the MTP draft (measured: no gain, keep off); 0 = eager |
| `TQ_GEMM_FORK` | 1 | the 2nd/3rd Linear of an input-sharing group (k/v after q, z after qkv, up after gate) on side streams concurrently with the first; same kernels, grids, k-splits → bitwise. With `TQ_GRAPH=1`: −0.35 ms/step at every N and context |
| `TQ_GEMM_FORK_MAXN` | 64 | the GEMM fork only up to N columns: decode steps and verify waves (N ≤ 64) keep it, the 128-column prefill wave runs its gate / up groups on one stream (bitwise, neutral on the wave); 1000000 = fork at every N, the previous behaviour |
| `TQ_DSTATE_BF16` | 4 | DeltaNet recurrent state storage: 0 fp32 (155 MiB/slot, the 10-16 line bitwise), 1 bf16 (fails the yardstick), 2 fp16 RNE (fails long pure decode), 3 bf16 + 8 bits (24-bit, neutral), 4 fp16 with deterministic stochastic rounding (80 MiB/slot, neutral everywhere measured; `--dstate fp16sr`) |
| `TQ_EXL3_WIDE_NG16` | 1 | 128-token EXL3 GEMM units at one block per SM for every Linear with ≥ 32 of them (k/v_proj keep 32-token units): −5.1 % per 32K prefill wave, −2.6 % at 128K, bitwise; 0 = 64-token units |
| `TQ_EXL3_BF16_R4` | 1 | the bf16 DeltaNet `in_proj_b/a` at prefill N: 4 rows per block, shuffle tree (same partial sums → bitwise), −1.1 % per wave; 0 = the one-row kernel |
| `TQ_GDN_PREFILL` | 1 | the 48 gated-DeltaNet layers of a paged prefill wave through a hoisted prep kernel + a register-resident chunk kernel (`--gdn-prefill`; engine `b2e7a3f`, session 11): the state-independent work once per wave instead of per sub-chunk, the recurrent state in registers across the 16 sub-chunks, same FMA chains / contractions / SR seeds → bitwise with the 10-16 kernels on every gate. DeltaNet 14.95 → 2.76 ms per 128-column wave, the wave 65.7 → 55.0 ms at 32K and 116.5 → 106.0 ms at 128K; served-line TTFT 15.9 → 13.3 s @32K, 86.6 → 77.0 s @128K (`docs/evidence/cand8-all.log`); decode / verify / commit untouched. 0 = the 10-16 kernels (`--no-gdn-prefill`); 2 / 3 = the 8-warp / 2-warp variants (bitwise, slower) |
| `TQ_PATTN_GQA` | 1 | the trellis pool's paged prefill attention as one GQA-grouped tensor-core kernel with fp32 accumulation (since v0.10, main `9c0a046`): attention 17.0 → 9.3 ms per 128-column wave at 32K, 67.6 → 34.2 ms at 128K; served-line TTFT 13.3 → 11.8 s @32K, 77.0 → 60.4 s @128K. Not bitwise (a different tiling of the same fp32 online softmax): e2e within the 95 % CI at 4K and 32K, spec exactness ALL PASS, needle 6/6 (`docs/evidence/cand9-all.log`). 0 = the per-head bf16 kernel of v0.10 bit for bit; 3 / 4 / 5 = fp16-accumulate variants (opt-in, +0.00005 KL at ≥ 24K); 2 = instrumentation |
| `TQ_WAVE_MAX` | 128 | prefill wave column cap (≤ 256, EXL3 only, for the 256-column A/B; not bitwise by design — half the k-splits). The trellis pool keeps one in-flight staging page per slot, so it stays ≤ 128 there |
| `TQ_PDL` | server: 1 | programmatic dependent launch of the fragment build + EXL3 GEMM in the decode step (`--pdl`; bitwise, −0.55…−0.61 ms/step); 2 = also the step's other kernels (measured slower at 32K) |
| `TQ_EMBED_HOST` | server: 1 | the bf16 embedding table in pinned host RAM, rows read over PCIe (`--embed-host`; no device copy, no quantization error). Overrides `TQ_EMBED_FP8` |
| `TQ_NO_MTP` | server: 1 unless `--spec`/`--mtp` | skip the tqf's MTP section (the bf16 head is 0.85 GB; the EXL3 one 0.21 GB) |
| `TQ_EMBED_FP8=2` | legacy | 6-bit (E2M3) embed table on the device, −1.5 GiB (needed for the full 256k on 32 GB on the single-stream path; irrelevant with `TQ_EMBED_HOST=1`) |
| `TQ_WIDE_PREFILL=1`, `TQ_WIDE_ATTN_MMA=1`, `TQ_PAGED_ATTN_MMA=1` | server: on | wide / ragged tensor-core prefill paths (the single-stream server defaults both ON; the batched server sets `TQ_WIDE_ATTN_MMA=1` for `--page >= 128`) |
| `TQ_ATTN_MMA=1`, `TQ_ATTN_MMA_PAIR=0`, `TQ_ATTN_MMA_GROUP_MIN` / `TQ_SPEC_ATTN_LEGACY_MIN`, `TQ_ATTN_MMA_GROUP2=0`, `TQ_WIDE_CONV=0` | legacy | the single-stream path's attention / conv variants of the 2026-08 README (all default on; the =0 forms are bit-identical fallbacks) |
| `TQ_PAGED_TOPK_PAGES` | 0 | Quest-style page top-k for the decode attention (built, measured to fail the FP8 gate below ~75 % of the pages; keep 0; not ported to the trellis pool) |

<sub>Provenance: flag names and defaults are those of `src/forward_qwen.cu` at engine `b2e7a3f` (library md5 `484bdb825c36`; the kernels of `2cf671c` / `e454a29c3caf` plus the `TQ_GDN_PREFILL` default flip, CUDA 13.1 nvcc `cuda_13.1.r13.1/compiler.37061995_0`, driver 610.57.04); the measured effects quoted per row come from the landing gates `docs/evidence/cand4-all.log` … `cand8-all.log` and `ASSUMPTIONS.md` (research repo), 2026-10-16/17, torch 2.14.0+cu130, transformers 5.17.0.</sub>

## Server arguments

`tools/serve_batched.py --help` has the full text:

| Argument | Default | Meaning |
|----------|---------|---------|
| `--max-slots N` | 12 | concurrent sessions resident on the GPU (DeltaNet state per slot: 80 MiB with `--dstate fp16sr`) |
| `--num-blocks N` / `--page P` | 1024 / 128 | the shared paged KV pool: N pages of P tokens. Served line 4500 × 128 = 576K tokens (29 894 MiB with 12 slots and spec); 5000 without spec (≈ 640K tokens; memory model, not a gated run); 400 (2 slots) for the 16 GB model; size another configuration from the startup line's memory |
| `--kv-v4` / `--no-kv-v4` | on | int4 V (`TQ_KV_V4`); `--no-kv-v4` = E4M3 V (26.2 KiB/token), incompatible with `--kv-t4` |
| `--kv-t4` / `--no-kv-t4`, `--kv-t4-codebook FILE` | on, built-in | the trellis-coded KV (`TQ_KV_T4`); `--kv-t4-codebook` sets `TQ_KV_T4_CODEBOOK` (default: the engine's built-in codebook); `--no-kv-t4` = FP4 K+V, the pre-2026-10 format |
| `--spec` / `--no-spec`, `--spec-k auto\|1..16` | on, auto | speculative decoding with the paged MTP drafter in decode-only phases; needs a `--mtp-exl3` tqf (else a warning and plain decode); auto = K 4 at ≤ 2 active slots, 3 at ≤ 8, 2 above (measured through the server: K = 6 ties 4 at N = 1 and loses at N = 2; 3 beats 2 and 4 at N = 4). The drafter's KV rows and the EXL3 head take ~11 % of the pool (4500 vs 5000 blocks) |
| `--dstate fp32\|bf16\|fp16\|bf16+8\|fp16sr` | fp16sr | the DeltaNet recurrent-state format (`TQ_DSTATE_BF16` 0–4); `fp32` = the 10-16 line bit for bit |
| `--gdn-prefill` / `--no-gdn-prefill` | on | `TQ_GDN_PREFILL` 1 / 0: the session-11 DeltaNet prefill kernels for the paged prefill waves (bitwise; TTFT 15.9 → 13.3 s @32K, 86.6 → 77.0 s @128K); `--no-gdn-prefill` = the 10-16 kernels |
| `--decode-steps-per-wave k` | 1 | extra pure decode steps after each mixed prefill wave: decoders get 1 + k tokens (speculative rounds with `--spec`, 3–5 tokens each) per wave; 0 = one token per wave. 1 lifted decoders under prefill load 3–4× in the 8-session mix at the price of ~+16 s cold TTFT for a 96K prompt |
| `--mixed-prefill` / `--no-mixed-prefill`, `--prefill-waves-per-step k` | on, 2 | a long prompt's ≤ 128-column waves carry the active slots' decode steps as final segments; `--no-mixed-prefill` alternates waves and steps; `--prefill-waves-per-step 0` = the blocking whole-prompt prefill |
| `--prefix-cache` / `--no-prefix-cache` | on | slot-level prefix cache: a finished request leaves its slot parked (KV pages + state + history); a follow-up that extends the history prefills only the new tokens |
| `--park-ram-gb GB`, `--park-ram-ttl s` | 64, 0 | host-RAM tier for sessions evicted from the GPU (pinned staging ring, 37–45 GB/s; LRU, optional TTL); 0 disables parking |
| `--embed-host` / `--no-embed-host`, `--mtp` | on, off | `TQ_EMBED_HOST`; `--mtp` loads the MTP head without `--spec` |
| `--pdl` / `--no-pdl` | on | `TQ_PDL=1` |
| `--wide-attn-mma` / `--no-wide-attn-mma` | on | tensor-core ragged prefill attention for the waves (`TQ_WIDE_ATTN_MMA`; needs `--page >= 128`); without it the waves use the scalar decode kernel per column (~6× slower prefill) |
| `--timeout s`, `--model-name`, `--host`, `--port` | 300, qwentin-qwen3.6-27b-batched, 127.0.0.1, 8100 | admission wait when all slots are busy, the `model` name clients send, bind address (loopback by default), port |
| `--api-key KEY`, `--max-body-bytes N`, `--max-tokens-cap N`, `--socket-timeout s` | unset, 32 MiB, 16384, 600 | request-side hardening, shared with `serve_openai.py` (`tools/serve_common.py`): with a key every path but `/health` needs `Authorization: Bearer KEY` (401 otherwise); bodies above the limit get 413 before being read (411 without `Content-Length`); a request's `max_tokens` above the cap is clamped and the clamp reported as `x_qwentin.max_tokens_clamped_from` / `max_tokens_cap` (finish_reason `length`); a client socket that stops sending or stops reading is closed after the timeout |
| `--selftest`, `--bench ...` | — | staggered join/leave correctness vs single-stream; steady-state paged decode bench (`--bench-p` prompt tokens, `--bench-ns` slot counts) — both run with spec off |

<sub>Provenance: `tools/serve_batched.py` at this tag (`--help` is authoritative); the defaults are the ones served on 2026-10-17 with engine `b2e7a3f` / library md5 `484bdb825c36` (CUDA 13.1 nvcc `cuda_13.1.r13.1/compiler.37061995_0`, driver 610.57.04, torch 2.14.0+cu130, transformers 5.17.0) and gated in `docs/evidence/cand8-all.log` (every default but `--gdn-prefill` already in `cand7-all.log`).</sub>


## Server behaviour: endpoints, thinking, per-request stats, logging

Endpoints: `/v1/chat/completions` and `/v1/completions` (the OpenAI API, SSE streaming), `/health` (free / total
blocks, active, queued, parked, prefilling, prefix-cache hits, spec rounds / tokens per round / ms per round /
fallbacks, parking-tier stats) and `/spec?on=0|1&k=N` (switches speculative decoding at run time; `k` = the draft
depth). The startup line prints the KV format and the pooled tokens (`576000` on the served line), the DeltaNet state
format and the spec / attention settings.

Thinking is off unless the request body carries `"enable_thinking": true` (top level; the agents'
harness sends it off explicitly). Per-request stats are returned under `x_qwentin`
(`prefix_cached_tokens`, `slot`, `server_ttft_s`) and `usage.prompt_tokens_details.cached_tokens`;
`/v1/completions` is served too, tool calls pass through the chat template. The server logs one `[admit]` line
per admitted request (slot, prompt tokens, cached, new, tier, free blocks) and one `[req]` line
per request (`TQ_SERVE_LOG_ADMITS=0` silences the former). `--timeout` is the admission wait when
all slots are busy; `--park-ram-gb` the host-RAM budget for parked sessions (16.5 KiB/token + 80
MiB of DeltaNet state each; a 117K-token session restores in well under a second at 37–45 GB/s).

## Exposure

Both servers bind `127.0.0.1` by default and carry the same request-side guards
(`tools/serve_common.py`, unit-tested without a GPU in `tests/test_serve_common.py`): `--api-key KEY`
makes every path except `/health` require `Authorization: Bearer KEY` (401 otherwise; `/spec` and
`/v1/*` included), `--max-body-bytes` (32 MiB) refuses larger bodies with 413 before reading them,
`--max-tokens-cap` (16384) clamps a request's `max_tokens` and reports the clamp in
`x_qwentin.max_tokens_clamped_from` / `max_tokens_cap` (the response's `finish_reason` is then
`length`), and `--socket-timeout` (600 s) closes client connections that stop sending a request or
stop reading a stream. Serving on `--host 0.0.0.0` should always come with `--api-key` (and a reverse
proxy for TLS — the servers speak plain HTTP). The engine itself is unchanged by any of these.

## Off-switches and the previous served lines

```bash
# The v0.10 served line without the DeltaNet prefill lever (gate 1 of docs/evidence/cand8-all.log proved it bitwise
# in both directions): one engine flag or the server flag
TQ_GDN_PREFILL=0 ... tools/serve_batched.py ...            # or: tools/serve_batched.py ... --no-gdn-prefill
# Off-switches that restore the 10-16 line bit for bit (gate 1 of docs/evidence/cand7-all.log proved both directions
# bitwise; 525K tokens): the four engine flags in the env + the two server flags + the smaller pool
TQ_T4_ATTN_F16=0 TQ_GRAPH=0 TQ_GEMM_FORK=0 ... tools/serve_batched.py ... --dstate fp32 --decode-steps-per-wave 0 --num-blocks 4100
# The pre-2026-10 stack (FP4 K+V, no spec; e.g. the opt-6.25 model at 422K tokens with --num-blocks 3300):
TQ_PAGED_ATTN_ST=1 ... tools/serve_batched.py ... --no-kv-t4 --no-spec
```
