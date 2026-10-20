# Changelog

Releases are git tags; the numbers of a release are pinned at its tag and backed by a log in
[`docs/evidence/`](docs/evidence/README.md) (every gate log names the engine commit and the library md5 it ran).
Measurement dates are UTC. "Bitwise" means the lever reproduces the previous kernels' outputs bit for bit
(`tools/paged_logit_ab.py`, max |d| 0), so it was landed on speed alone; everything else went through the quality
yardstick of [`docs/QUALITY.md`](docs/QUALITY.md).

## [Unreleased]

- **The single-stream path is retired.** Gone: `tools/serve_openai.py` with Reservoir Dogs and the residual-stream
  steering hook (`qwn_rmap_*` and the `qwn_ot_*` aliases), the engine's single-stream speculative decoding (MTP
  covering tree, batched FP6 verify, persistent layer kernel, n-gram drafter, sampling), the pre-paged batched decode
  (`qwn_batched_*`), the single-stream wide prefill, the FP4 (E2M1) weight path and the never-called debug / bench
  entry points, with the 17 tools that drove them (`serve_smoke.py`, `bench_coding.py`, `mtp_*.py`,
  `bench_rounds.py`, `persist_phases.py`, `accept_probe.py`, `needle_check.py`, `residual_map*.py`,
  `batched_smoke.py`). 138 → 55 entry points in `src/qwentin.h`, 30k → 15k engine lines, 398 → 211 kernels,
  26 → 20 engine parts. **v0.11 is the last release with any of it.** What stays: FP6 E2M3 files on the batched
  server (`--no-kv-t4 --no-spec`), the converter's FP6 recipe, and the single-stream decoder `qwn_decode` as the test
  oracle of `serve_batched.py --selftest` and `tools/paged_smoke.py` (docs/legacy-fp6.md).
- **Engine switches: 34 left** (119 at v0.10), every one in docs/FLAGS.md: the served line and its documented
  off-switches, three tool / test pins, four debug switches, `TQ_PAGED_TOPK_*` and `TQ_WIDE_ATTN_QROWS`. Besides the
  single-stream path's own switches, `TQ_LINEAR_CORE_PAR` / `TQ_LINEAR_CORE_HS` are gone: the DeltaNet decode / verify /
  commit cores run their served split (8 threads per element, 4 blocks per head) and only that template instance.
- Gate (`docs/evidence/cand17-all.log`): of the 211 kernels left, 210 keep their SASS bit for bit; the one that changed
  is the single-stream decoder's attention, whose non-inlined helper lost its other callers and was
  specialized by the compiler — the decoder's tokens, its KV and the paged logits after it are bitwise unchanged.
- **The engine reads its environment once, in one place.** `src/engine/config.cuh` parses the 34 switches into one
  struct when `qwn_init` (or `qwn_probe`) runs; no other part calls `getenv`, so nothing changes if the environment
  does later. One parse rule for every switch: unset or empty = the default, otherwise the integer value, a boolean on
  for any non-zero value, ranges clamped or replaced as the table says. Every documented value parses as before; odd
  inputs now follow the rule too — `TQ_EXL3_DEBUG=0` / `TQ_PATTN_DEBUG=0` are off (presence used to turn them on),
  `TQ_EMBED_HOST` / `TQ_NO_MTP` / `TQ_EMBED_FP8` read the number instead of the first character (`TQ_EMBED_FP8=e` is
  no longer an alias of 2), and an empty `TQ_CTX` / `TQ_QMMA_SF_K_SPLIT` is the default instead of 2048 / 1.
  `qwn_init` prints the switches that differ from their defaults as one `[config]` line on stderr. docs/FLAGS.md
  lists the switches in the struct's four groups; `tests/test_config.py` keeps the two in step and pins the rule.
- **A bf16 MTP head is no longer loaded.** Nothing drafts with it since the single-stream path went (the paged drafter
  needs a `--mtp-exl3` file), so FP6 files and older `-mtp.tqf` files skip it: 0.85 GB less device memory with
  `TQ_NO_MTP` unset.
- Four engine parts renamed to what they hold (`embed_rows`, `wide_mlp`, `norm_launch`, `decode_oracle`), `config.cuh`
  added: 21 parts.
- Gate (`docs/evidence/cand18-all.log`, `cand18-matrix.log`): host code only — all 211 kernels keep their SASS bit
  for bit; the served line, spec exactness, the test decoder and an FP6 file as before, and 25 cells with switches
  off their defaults bitwise against the previous library under the same environment.

## [v0.11] — 2026-10-20

The v0.10 served stack with a faster prefill and a tidier tree: **TTFT 13.3 → 11.6 s @32K and 77.0 → 59.7 s @128K**
(the GQA-grouped prefill attention), the decode path bit for bit v0.10's, the engine in 26 parts behind a generated C
header, `.tqf` format version 2 (recorded payload size, BLAKE2b-256 hashes), the trellis-KV codebook compiled in, ten
experiment switches gone. Engine `dd6d21e`, library `build-qwen/libforward_qwen.so` md5 `158edd067533`, served line
`--max-slots 12 --num-blocks 4500` with no flags on a version-2 `.tqf` (the v0.10 model re-converted: the same payload
byte for byte), CUDA 13.1, driver 610.57.04, RTX 5090 at 550 W.

### Numbers pinned at this tag

| | value | log |
|---|---|---|
| Pooled KV capacity | **576 000 tokens** (4500 × 128-token pages, 12 slots × up to 256K) at **29 894 MiB**, unchanged | `docs/evidence/cand16-all.log` gate 2 |
| Needle-in-a-repository | **6/6** at 32K and 128K (3 depths × 3 decoys) | `cand16-all.log` gate 2 |
| Time to first token, cold | **11.6 s @32K · 59.7 s @128K** (v0.10: 13.3 / 77.0 s) | `cand16-all.log` gate 2; the same in `cand14` / `cand15` |
| Decode per stream @32K, N = 1 / 2 / 4 / 8, spec on | **177 / 153 / 106 / 79 tok/s** — one prompt per cell; the gates since v0.10 read 175–177 / 78–82 at N = 1 / 8 | `cand16-all.log` gate 2 |
| Decode per stream @32K, spec off | 66 / 62 / 57 / 48 tok/s | `cand16-all.log` gate 2 |
| Decode per stream @128K, N = 1 / 2, spec on | **115 / 99 tok/s** (v0.10: 133 / 111, see below) | `cand16-all.log` gate 2 |
| Quality vs the bf16 original | KL 0.0022–0.0027, top-1 99.0–99.1 %, unchanged | v0.10's rows; `cand9-all.log` gate 2 for the new prefill attention |

The decode path is v0.10's bit for bit: the v0.11 library reproduces the v0.10 library's paged-decode logits exactly
(401 decode steps at N = 3 and 8, with the default flags and with `TQ_PATTN_GQA=0`;
`docs/evidence/rel11-v010-bitwise.log`), and a round costs what it did (~21 ms @32K, ~25 ms @128K at N = 1). The
spec-on cells moved with the accepted draft length of their one prompt, which the new prefill attention (not bitwise)
shifts: 3.78 instead of 3.56 tokens per round @32K N = 1, 2.85 instead of 3.29 @128K. With `TQ_PATTN_GQA=0` the same
library reads v0.10's 167 tok/s @32K and 130 @128K in v0.10's 84 / 91 rounds (`diag-128k-pattn.log`). The quality rows
carry over: the decode path is the one they were measured on, and the new attention's prefill rows are within the 95 %
CI of v0.10's (`cand9-all.log` gate 2).

### Changes

- **Prefill attention on the trellis pool** (`TQ_PATTN_GQA=1`, default since 2026-10-17, main `9c0a046`): one
  GQA-grouped tensor-core kernel with fp32 accumulation instead of the per-head bf16 kernel — attention 17.0 → 9.3 ms
  per 128-column wave at 32K, 67.6 → 34.2 ms at 128K; served-line **TTFT 13.3 → 11.8 s @32K, 77.0 → 60.4 s @128K**.
  Not bitwise (a different tiling of the same fp32 softmax): e2e within the 95 % CI at 4K and 32K, spec exactness ALL
  PASS, needle 6/6, A/B 10/10 (`docs/evidence/cand9-all.log`); `TQ_PATTN_GQA=0` = the v0.10 kernel bit for bit.
- **The trellis-KV codebook is compiled into the engine** (`src/engine/kv_t4_codebook.inc`, generated from
  `tools/codebooks/lloyd_kv12a.f32` by `tools/gen_codebook_inc.py` and checked by `tests/test_codebook_inc.py`).
  `TQ_KV_T4_CODEBOOK` / `--kv-t4-codebook` only override it; the engine's fallback to the measurement box's path is
  gone. Same values, bitwise (`docs/evidence/cand10-all.log`).
- README cut to the pitch, the table, a quick start and the docs index; its details moved to `docs/FLAGS.md`
  (endpoints, the startup line) and `docs/evidence/README.md` (`bench/reproduce.sh` options, build reproducibility).
- **The engine in 26 parts**: `src/forward_qwen.cu` includes `src/engine/*.cuh` in order — still one translation unit,
  per-kernel SASS identical to the unsplit file. Tools that use the engine source as prompt text read it through
  `tools/engine_text.py`; `tests/test_unity_build.py` checks the part list.
- **`src/qwentin.h`**: the 137 `qwn_*` entry points as a C header (generated by `tools/gen_qwentin_h.py`), included
  first by the translation unit, so a definition that drifts from its declaration does not compile; the same
  signatures for ctypes in `tools/qwentin_abi.py` (`bind(lib)`). `tests/test_qwentin_abi.py` checks both are current
  and that the 210 hand-written ctypes declarations in `tools/` agree with the header (it found `qwn_free` declared
  as returning int in three tools; it is void).
- `TQ_GEMM_FORK_MAXN` (default 64): the concurrent GEMM groups only up to N = 64 — decode steps and verify waves keep
  them, the 128-column prefill wave runs its gate / up groups on one stream (bitwise, neutral on the wave;
  `TQ_GEMM_FORK_MAXN=1000000` = the previous behaviour).
- The residual-stream hook's entry points are `qwn_rmap_hook_add` / `_hook_count` / `_set_scale` / `_apply_debug`; the
  2026-08 `qwn_ot_*` names stay as aliases for one release. Stale comments (the file header, June/July dates) and
  never-called code (the single-stream spec path's graph experiment, an unused fork kernel) removed.
- **`.tqf` format version 2**: the converter appends an 80-byte header extension with the payload size and BLAKE2b-256
  hashes of the payload and of the header (sealed last, so an interrupted conversion is refused). The engine checks
  the header hash and the size on every load, the payload hash with `TQ_VERIFY_TQF=1` (~20 s for 21 GB), and refuses
  a newer format, a truncated copy, a corrupted header or payload and unknown flags with distinct codes (-40 … -45);
  `tools/inspect_tqf.py --verify` does the same offline. Version-1 files load as before; re-converting the served
  `-mtp4.tqf` gives the same payload byte for byte plus the extension (docs/TQF_FORMAT.md).
- Ten experiment switches of the served path removed, their defaults hard-coded (bitwise: today's behaviour is the
  default branch): the sweep overrides `TQ_PATTN_S`, `TQ_WIDE_ATTN_SPLIT`, `TQ_EXL3_WIDE_NG`, `TQ_PAGED_SPLIT`,
  `TQ_BFRAG_BDIM`; the never-adopted opt-in `TQ_VF_NG8`; `TQ_VF_SPLIT_BY_SLOTS` (its other value broke the
  spec-exactness guarantee); and the bitwise-either-way `TQ_VF_COMMIT_FUSED`, `TQ_VF_SEG_ATTN`, `TQ_FUSE_NORM`. None was
  documented or used by a tool. Debug switches are now listed in docs/FLAGS.md.
- Tools: the default `--lib` is this repository's `build-qwen/libforward_qwen.so` (was a path on the development box),
  and both servers require `--tqf` / `--model-dir` instead of defaulting to files that do not exist.

## [v0.10] — 2026-10-17

The 2026-10 served stack: **Qwen3.8-27B on one RTX 5090 (32 GB), 576K pooled context tokens across 12 sessions at FP8
quality, speculative decoding on every stream**, through the batched OpenAI-compatible server (`tools/serve_batched.py`).
Engine `b2e7a3f` (`src/forward_qwen.cu`; the DeltaNet prefill kernels on by default, otherwise the kernels of
`2cf671c`), library `build-qwen/libforward_qwen.so` md5 `484bdb825c36`, served line `--max-slots 12 --num-blocks 4500`
with no flags, CUDA 13.1, driver 610.57.04, RTX 5090 at 550 W.

### Numbers pinned at this tag

| | value | log |
|---|---|---|
| Pooled KV capacity | **576 000 tokens** (4500 × 128-token pages, 12 slots × up to 256K) at **29 894 MiB** | `docs/evidence/cand8-all.log` gate 3 |
| Needle-in-a-repository | **6/6** at 32K and 128K (3 depths × 3 decoys) | `cand8-all.log` gate 3 |
| Time to first token, cold | **13.3 s @32K · 77.0 s @128K** (15.9 / 86.6 s before the DeltaNet prefill lever) | `cand8-all.log` gate 3 (`cand7-all.log` for the previous values) |
| Decode per stream @32K, N = 1 / 4 / 8, spec on | **167 / 109 / 81 tok/s** — the gate runs read 157–169 / 74–81 (run-to-run ±2 %; the accepted draft length depends on the text: 3.4–3.6 tokens per round on the vLLM-repository corpus, 2.7 on this repository's code) | `final-speed.log` (R), `cand7-all.log` gate 3, `cand8-all.log` gate 3 |
| Decode per stream @32K, spec off | 65 / 56 / 47 tok/s | `final-speed.log` (R) |
| Decode per stream @128K, N = 1 | **133 tok/s** | `cand7-all.log` gate 3, `final-speed.log` (R) |
| Decode under a concurrent long prefill (8 sessions, 32K / 96K / 160K) | 40 / 25 / 43 tok/s per client | `final-t5.log` g1 |
| Quality vs the bf16 original | **KL 0.0022–0.0027, top-1 99.0–99.1 %** (the FP8 checkpoint: 0.0037 / 98.7 %); T1 EvalPlus = the FP8 reference | `final-e2e.log`, `e2e/*.json`, `evalplus/` |
| 16 GB variant (4.25 bpw, 2 slots) | 51K tokens at 16 006 MiB, 184 tok/s @32K, KL 0.0110 (weights) | `p16-check.log` |
| 24 GB variant (1 slot) | a full 256K single session at FP8 quality, 161 tok/s @32K, 113 @240K | `b2eng-t14-spec.log` |

Decode and quality were measured on engine `2cf671c` (library `e454a29c3caf`, `cand7-all.log`); `b2e7a3f` changes only
the prefill waves and reproduces the decode path bit for bit (`cand8-all.log` gate 1), so those rows carry over. The
same-day stock-vLLM baselines (NVFP4 on one card: 147–179K tokens, 71.5 / 51.4 tok/s, TTFT 3.1 / 25.2 s; FP8 on two
cards: 493K, 76.8 / 60.3 / 50.4, 4.4 / 24.1 s) are in `docs/evidence/baselines-vllm-20261017.md`.

### Added since the 2026-08 build

- **EXL3 trellis weights** (2026-09-25 → 10-14; `--exl3`, header flag `0x800`). exllamav3 checkpoints copied verbatim
  into the `.tqf` and dequantized on the fly into fp16 HMMA GEMMs (prefill) and GEMVs (decode): the served opt-5.75
  recipe (`tools/recipes/qwen3_8-27b-exl3-opt5.75.yaml`: K = 5 on 34 % of the weights, 6 on 55 %, 7 on 10 %, 6-bit head)
  is ~18.5 GiB at FP8 quality where the FP8 checkpoint itself (27.5 GiB) does not fit the card; the 4.25 bpw t45-s425
  recipe is the 16 GB model. Decode GEMV: KB = 7 at 2 blocks/SM, k-split picker, programmatic dependent launch
  (`TQ_PDL`, −0.55…−0.61 ms per step, bitwise), one fragment build per input-sharing group, L2 prefetch (−3.6 % GEMM).
- **Trellis-coded KV cache** (`--kv-t4`, 2026-10-15/16). Every committed (token, KV head) K and V vector as a
  12-bit-state bitshift trellis code (4 bits per value) over the Hadamard-rotated head with an fp16 RMS scale: 132 bytes
  per vector, **16.5 KiB/token** (FP4 K+V 20.0, int4 K + E4M3 V 26.2). Lloyd-trained codebook shipped
  (`tools/codebooks/lloyd_kv12a.f32`); a fused CUDA Viterbi encoder on a side stream (in-flight columns attended exact
  from an fp16 staging page); decode inside the attention kernel. Passes the FP8 gate at opt-5.75 where FP4 K+V does
  not; +39 % pooled tokens over the 10-13 stack (opt-6.25 + FP4 V, 422K).
- **EXL3 MTP speculative decoding in the batched server** (`--spec`, 2026-10-15; `--mtp-exl3` in the converter, flag
  `0x1000`). The checkpoint's own MTP layer as 4-bpw EXL3 records (0.25 GB) drafts K tokens per slot
  (`qwn_paged_mtp_draft`), one ragged verify wave scores all slots' drafts through the decode-step kernels with
  per-column key splits (`qwn_paged_verify`), the accepted prefix is committed by replaying the DeltaNet recurrence
  (`qwn_paged_commit`). ~3.4 accepted tokens per round at K = 4; 2.5–2.7× per stream at N = 1, 1.9× at N = 4, 1.7× at
  N = 8 @32K; outputs bit for bit the plain decode's (`tools/spec_paged_check.py` ALL PASS, `tools/spec_server_ab.py`
  10/10). `--spec-k auto` = 4 / 3 / 2 by active slots.
- **DeltaNet recurrent state in fp16 with deterministic stochastic rounding** (`--dstate fp16sr`, `TQ_DSTATE_BF16=4`,
  2026-10-16/17). 155 → 80 MiB per slot, **+51K pooled tokens per card** (4100 → 4500 blocks, 525K → 576K), −0.38 ms per
  step at 32K N = 8; neutral on the yardstick at 4K / 32K / 64K where bf16 and fp16 round-to-nearest fail.
- **Decode step as one CUDA graph with concurrent GEMM groups** (`TQ_GRAPH=1 TQ_GEMM_FORK=1`, 2026-10-16/17). The
  paged decode step captured per (N, key split, MTP columns, flags) into an LRU of graphs; the 2nd/3rd Linear of an
  input-sharing group on side streams. −0.35 ms per step at every N and context, bitwise.
- **fp16 staged trellis attention** (`TQ_T4_ATTN_F16=1`, 2026-10-16/17). One fp16-operand kernel for the decode step and
  the verify wave over the trellis pool, cp.async-staged V codes: 335 → 277 µs per layer at 32K N = 8, −1.0 ms per step;
  the opt-in set it leads is +20–28 % at 128K; same quality.
- **Prefill GEMM units** (2026-10-16/17, bitwise). 128-token EXL3 GEMM units at one block per SM (`TQ_EXL3_WIDE_NG16`),
  paired codeword decode, a 4-row bf16 kernel for the DeltaNet `in_proj_b/a` (`TQ_EXL3_BF16_R4`): −5.1 % per 32K prefill
  wave, −2.6 % at 128K; served-line TTFT 32K 17.1 → 16.0 s, 128K 93–98 → 87.4 s (`cand5-all.log`).
- **Gated-DeltaNet prefill kernels** (`TQ_GDN_PREFILL=1` / `--gdn-prefill`, 2026-10-17, session 11, bitwise). A
  hoisted prep kernel does the state-independent work once per wave; a register-resident chunk kernel keeps the
  recurrent state in registers across the 16 sub-chunks with the same FMA chains, contractions and SR seeds as the
  10-16 kernels. DeltaNet 14.95 → 2.76 ms per 128-column wave (22.8 % → 5.0 % of the wave); the wave 65.7 → 55.0 ms at
  32K, 116.5 → 106.0 ms at 128K; **served-line TTFT 15.9 → 13.3 s @32K, 86.6 → 77.0 s @128K**; decode unchanged
  (`cand8-all.log`).
- **Host-RAM parking of idle sessions** (2026-09-22 → 10-16). Evicted sessions' KV pages + DeltaNet state saved and
  restored through a pinned staging ring at 37–45 GB/s (`--park-ram-gb`, `--park-ram-ttl`; a 32K session leaves or
  returns in ~16–19 ms); the slot-level prefix cache continues a returning conversation from its parked state; the
  pool-exhaustion bug (every active request failing at ~95 % occupancy) fixed.
- **The batched server itself** (`tools/serve_batched.py`, 2026-09-21 → 10-17): paged KV + continuous batching over 12
  slots, long-prompt prefill in ≤ 128-column ragged waves that carry the active slots' decode steps, SRPT prefill order,
  slot-level prefix cache, token streaming, `--decode-steps-per-wave 1` (one speculative round between mixed waves:
  decoders under a long prefill 13 / 8 / 11 → 43 / 28 / 43 tok/s per client, `final-t5.log`), GQA tensor-core decode
  attention (`TQ_PAGED_ATTN_ST=3` fp16 kernels for the FP4 / E4M3 pools), FP4 V (`--kv-v4`), the bf16 embedding table
  in pinned host RAM (`--embed-host`), `[admit]` / `[req]` log lines, `/health` and `/spec`.
- **Request-side hardening** (2026-10-17, both servers, `tools/serve_common.py`): loopback bind by default, `--api-key`,
  `--max-body-bytes` (32 MiB), `--max-tokens-cap` (16384), `--socket-timeout` (600 s); unit-tested without a GPU.
- **Evidence pack and reproduction** (2026-10-17): `docs/evidence/` (the gate logs `cand4` … `cand8-all.log`,
  `final-*.log`, `p16-check.log`, `b2eng-t14-spec.log`, the scripts that produced them, the vLLM baseline note, the
  EvalPlus logs, the e2e run files, a cell-by-cell map from each README number to its log line); the quality yardstick
  made public (`tools/qwentin_e2e.py`, `exl3_diff.py`, `e2e_frontier.py`, `dequant_fp8_to_bf16.py`, `evalplus.sh`,
  `docs/QUALITY.md`); `bench/reproduce.sh` (build → model check → served line → needle / A-B / speed → results JSON
  with the environment block, `--quality`, `--dry-run`); the verification tools (`code_needle.py`, `spec_speed.py`,
  `spec_server_ab.py`, `spec_paged_check.py`, `paged_logit_ab.py`); CI (compile check of the engine TU, ruff, pytest,
  the dry run) and the disabled nightly 5090 gate; `pyproject.toml` / `requirements.txt` with the measured pins;
  `docs/TQF_FORMAT.md`; `tools/inspect_tqf.py` PASS / FAIL verdicts and the EXL3-aware payload walk.

### Changed

- Served defaults (`serve_batched.py` + engine): `--kv-v4 --kv-t4 --spec --dstate fp16sr --decode-steps-per-wave 1
  --gdn-prefill --pdl --embed-host --park-ram-gb 64`; `TQ_PAGED_ATTN_ST=3 TQ_T4_ATTN_F16=1 TQ_GRAPH=1 TQ_GEMM_FORK=1
  TQ_DSTATE_BF16=4 TQ_GDN_PREFILL=1 TQ_EXL3_WIDE_NG16=1 TQ_EXL3_BF16_R4=1`. Each previous served line is one set of
  off-switches away, bit for bit (`docs/FLAGS.md`).
- `tools/convert_qwen_tqf.py`: the production FP6 recipe (`--block-scaled always --block-layout qmma-e2m3
  --block-scale-policy pow2` + the MTP section) is the default for a HF checkpoint; the old `auto` (which silently fell
  back to plain FP8 without the MTP head) is behind `--legacy-fp8`; a WARNING whenever the output is neither production
  format; `--exl3` / `--mtp-exl3` for EXL3 checkpoints.
- `serve_openai.py`: `--residual-map` (a neutral affine residual-stream map after layer L) replaces `--bark-all-day` /
  `--ot-*`; the 2026-08 spellings are hidden aliases for one more release. The server binds `127.0.0.1` by default (was
  `0.0.0.0`). `tools/residual_map_check.py` (was `ot_hook_check.py`) is the generic kernel-vs-numpy parity check.
- README: a release page (pitch, one table, the reproduction); the narrative, architecture, flags, variants and the
  FP6 path moved intact to `docs/NOTES.md`, `docs/ARCHITECTURE.md`, `docs/FLAGS.md`, `docs/VARIANTS.md`,
  `docs/legacy-fp6.md`.

### Moved out of this repository

- `tools/bark_autotune.py`, the LLobotomy demo and the scale sweep of the steering hook (to the LLobotomy repositories).

### Not re-measured

- The FP6 single-stream path (`serve_openai.py`, FP6 E2M3 + int4/E4M3 KV + MTP tree, Reservoir Dogs) still runs, but
  its numbers are the 2026-08 ones (CUDA 13.3, driver 595) — `docs/legacy-fp6.md`. Quest page top-k
  (`TQ_PAGED_TOPK_PAGES`) is built, measured to fail the FP8 gate below ~75 % of the pages, and kept off.

## 2026-08 build (untagged; the README of 2026-08-18, commit `95952dd`)

FP6 E2M3 block-scaled weights (`mma.sync ...kind::mxf8f6f4.block_scale`), int4-K / E4M3-V KV cache, the single-stream
server with the MTP covering tree, Reservoir Dogs and `--bark-all-day`; a 256K single session on one 32 GB card
(FP6 quality: KL 0.0139 / top-1 97.4 % on the 2026-10 yardstick), 142 tok/s @32K cold / 212 on prefix-cached turns,
2 059 prompt tok/s, needle 4/4 @239K; the batched server of that build held 196K pooled tokens (FP6). Kept as
`docs/legacy-fp6.md`.

[Unreleased]: https://github.com/kacper-daftcode/qwentin/compare/v0.10...HEAD
[v0.10]: https://github.com/kacper-daftcode/qwentin/releases/tag/v0.10
