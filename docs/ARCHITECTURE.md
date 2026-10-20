# Architecture — what is in the engine and how it works

The engine's feature list ("Highlights"), the pipeline from a HuggingFace checkpoint to a served token ("How it
works"), the conversion details and the full repository layout — moved from the 2026-10-17 README at v0.10. Flags:
[`FLAGS.md`](FLAGS.md); the numbers and where they come from: the [README](../README.md) and
[`evidence/README.md`](evidence/README.md); the narrative of the 2026-10 build: [`NOTES.md`](NOTES.md); the 2026-08
single-stream FP6 path: [`legacy-fp6.md`](legacy-fp6.md).

qwentin is a from-scratch CUDA inference engine for the 27B hybrid-attention Qwen3.6/3.8 tower
on one consumer Blackwell GPU (SM120), with hand-written tensor-core kernels and an
OpenAI-compatible API, so any client — including an `opencode`/`continue`-style coding agent —
can use it. Since 2026-10 the primary path is the **batched server** (`tools/serve_batched.py`):
paged KV + continuous batching, EXL3 weights straight from an exllamav3 checkpoint, the
trellis-coded KV cache, an EXL3 MTP drafter for speculative decoding, host-RAM parking of idle
sessions. The 2026-08 **FP6 single-stream path** (`tools/serve_openai.py`, FP6 E2M3 weights +
int4/E4M3 KV + MTP tree, the residual-stream steering hook, Reservoir Dogs) is still in the tree
and documented in [`legacy-fp6.md`](legacy-fp6.md) as the legacy single-stream path. Qwen3.6-27B shares the
text-tower layout and runs identically on the FP6 path (the EXL3 path was measured on Qwen3.8 only).

## Highlights

- **FP8 quality in 32 GB, with 576K tokens of KV.** EXL3 trellis weights at 5.75 bpw from an
  exllamav3 checkpoint (per-tensor bitrate recipe, 4–8 bits by sensitivity) + the 4-bit
  trellis-coded KV cache. The whole stack — weights, KV, embedding — is measured end to end
  against the bf16 original and is not worse than the FP8 checkpoint.
- **4-bit trellis-coded KV cache (`--kv-t4`).** Every committed (token, KV head) K and V vector is
  a 12-bit-state bitshift trellis code (4 bits per value) over the Hadamard-rotated head, with an
  fp16 RMS scale: 132 bytes per vector, 16.5 KiB per token (FP4 K+V: 20.0, int4 K + E4M3 V: 26.2).
  The codebook is Lloyd-trained on real K/V rows and compiled into the engine (`src/engine/kv_t4_codebook.inc`,
  generated from `tools/codebooks/lloyd_kv12a.f32`; `TQ_KV_T4_CODEBOOK=<file>` loads another).
  Prefill and verify waves encode their columns per layer on a side stream; decode steps encode
  their new column after the step; the in-flight columns are attended exactly from an fp16
  staging ring, so the engine matches its own exllamav3 simulation within ±0.0001 KL.
- **Speculative decoding on the batched server (`--spec`).** A paged MTP drafter (the
  checkpoint's own MTP layer, 4-bpw EXL3 records in the `.tqf`) drafts K tokens per slot, one
  ragged verify wave scores all slots' drafts through the decode-step kernels, the longest
  matching prefix + 1 is committed by replaying the DeltaNet recurrence over the accepted
  columns. ~3.4 accepted tokens per round at K = 4; 2.5–2.7× per stream at N = 1, 1.9× at N = 4,
  1.7× at N = 8 @32K; outputs bit-for-bit plain decode's.
- **DeltaNet recurrent state in fp16 with deterministic stochastic rounding (`--dstate fp16sr`).**
  155 → 80 MiB per slot; bf16 and fp16 round-to-nearest both fail the yardstick (dead zone /
  rounding bias on long pure decode), fp16 + SR is neutral everywhere measured and buys +51K
  pooled tokens per card.
- **Decode step as a CUDA graph with concurrent GEMM groups (`TQ_GRAPH=1 TQ_GEMM_FORK=1`).**
  −0.35 ms per step at every N and context, bitwise.
- **fp16 staged trellis attention (`TQ_T4_ATTN_F16=1`).** One kernel for decode and verify over
  the trellis pool, cp.async-staged V codes: 335 → 277 µs per layer at 32K N = 8, −1.0 ms per
  step vs the bf16 trellis kernels, same quality.
- **Prefill GEMM units.** 128-token EXL3 GEMM units at one block per SM, paired codeword decode,
  a 4-row bf16 kernel for the DeltaNet `in_proj_b/a`, L2 prefetch: −5 % per 32K prefill wave,
  bitwise; on the served line TTFT 32K 17.1 → 16.0 s, 128K 93–98 → 87.4 s (`logs/cand5-all.log`).
- **Host-RAM parking of idle sessions.** A pinned staging ring saves and restores a session's KV
  pages + DeltaNet state at 37–45 GB/s (`--park-ram-gb`, `--park-ram-ttl`); the slot-level prefix
  cache continues a returning conversation from its parked state.
- **One speculative round per mixed prefill wave (`--decode-steps-per-wave 1`).** Decoders that
  share the GPU with a long prefill produce 1 + ~3.3 tokens per 70-ms wave instead of one:
  13 / 8 / 11 → 43 / 28 / 43 tok/s per client in the 8-session mix (`logs/final-t5.log`).
- **Hybrid attention.** Qwen3.6/3.8 mixes 48 gated-DeltaNet *linear-attention* layers (O(1) state)
  with 16 *full-attention* layers — the engine implements both, on the single-stream and the paged
  path.
- **Custom SM120 kernels, stock toolchain.** Inline-PTX tensor-core kernels (block-scaled
  `mma.sync ...kind::mxf8f6f4.block_scale` for FP6, fp16 HMMA with on-the-fly trellis dequant for
  EXL3, GQA decode attention, chunkwise DeltaNet, the trellis Viterbi encoder), compiled by the
  stock CUDA 13 `nvcc`/`ptxas` — no external assembler or precompiled cubins.
- **Legacy single-stream path (FP6 E2M3 + int4 KV + MTP tree, `serve_openai.py`).**   256K in one
  session on 32 GB, 137–157 tok/s at short context (200+ on prefix-cached turns), the
  residual-stream steering hook (`--residual-map`), Reservoir Dogs — [`legacy-fp6.md`](legacy-fp6.md).

> Research engine. SM120-only, wired for the Qwen3.6/3.8-27B layout (64 layers, 5120 hidden,
> 16 full-attn / 48 gated-DeltaNet). Every 2026-10 number here was measured on one RTX 5090 at a
> 550 W power cap with the exact line in the README's "Quick start"; the 16 GB and 24 GB variants were emulated on
> the same card with a capped KV pool and a process-memory budget, not on those cards ([`VARIANTS.md`](VARIANTS.md)).

## How it works

```
HuggingFace Qwen3.8-27B (bf16)
   │ exllamav3 convert.py -rcp recipe (EXL3 trellis, 5.75 bpw opt / 4.25 bpw t45)
   ▼
EXL3 checkpoint ──convert_qwen_tqf.py --mtp-exl3──▶  model-mtp4.tqf  (EXL3 Linears verbatim + EXL3 MTP layer)
                                                         │
                                libforward_qwen.so  ◀────┘   (one CUDA translation unit, SM120)
                                         │
                       tools/serve_batched.py  ──▶  OpenAI /v1/chat/completions, /health, /spec
```

- **Weights — EXL3 trellis (exllamav3), per-tensor bitrates.** The tqf carries exllamav3's
  trellis records verbatim (header flag `0x800`); the engine dequantizes tiles on the fly into
  fp16 HMMA GEMMs (prefill, 128-token units) and GEMVs (decode, k-split with programmatic
  dependent launch and one fragment build per input-sharing group). opt-5.75 = 5.75 bpw average
  from a sensitivity measurement: K = 5 on 34 % of the weights, 6 on 55 %, 7 on 10 %, head 6; the
  16 GB model t45-s425 = two tiers {4, 5} at 4.25 bpw. The bf16 embedding table stays in pinned
  host RAM (`TQ_EMBED_HOST=1`, rows read over PCIe).
- **KV cache — 4-bit trellis-coded K and V** (above). The pool is paged (128-token pages); a
  slot's open page is staged exact in fp16 while its columns are in flight (on the `--kv-v4` pool
  layout, hence that requirement) and becomes trellis rows when they commit, so a parked slot's
  blob is pure trellis rows plus its DeltaNet state.
  Dequant is a 4096-entry fp16 table lookup in shared memory (`table[(stream >> 4t) & 0xFFF] ×
  scale`); the remaining 0.6 ms/step behind the FP4 kernels is that table lookup (LSU-bound).
- **Speculative decoding — EXL3 MTP drafter + ragged verify + commit replay.** `qwn_paged_mtp_draft`
  runs K chained MTP steps for all active slots; `qwn_paged_verify` scores every slot's draft
  segment in one wave with the decode-step kernels (segment-aware GQA attention: one K/V stream
  per slot segment, per-column key splits so the logits equal the sequential decode steps'
  bitwise); `qwn_paged_commit` replays the per-token DeltaNet recurrence over the accepted prefix
  from the kept inputs (no scratch state). `--spec-k auto` = 4 at ≤ 2 active slots, 3 at ≤ 8, 2
  above. A tqf without the EXL3 MTP section runs with spec off after a warning.
- **DeltaNet state — fp16 + deterministic stochastic rounding.** The recurrent S of the 48
  linear-attention layers is stored in fp16 with SR seeded by (element index, position), so
  decode, verify and the commit replay agree bitwise; the conv window stays fp32.
- **Prefill — ragged 128-column waves.** Short prompts share one wave; a long prompt is prefilled
  in ≤ 128-column waves that carry the active slots' decode steps (or one speculative round) as
  final segments, so decoders keep producing while a 128K prompt prefills at ~1500 tok/s.
  Projections are one wide EXL3 GEMM per wave (128-token units at one block per SM), the 48
  DeltaNet layers a chunkwise-parallel gated-delta kernel, the 16 attention layers a tensor-core
  ragged attention against the trellis pages (the wave's own columns exact).
- **Decode step — one CUDA graph.** The paged decode step is captured per (N, key split, MTP
  columns, flags) into an LRU of instantiated graphs with the step's inputs uploaded through
  device buffers; the 2nd/3rd Linear of an input-sharing group (k/v after q, z after qkv, up after
  gate) run on side streams concurrently with the first. Same kernels, grids and splits → bitwise.
- **Scheduler.** Continuous batching over 12 slots; admission = free slot + enough free pages;
  SRPT prefill order; slot-level prefix cache (a follow-up whose prompt extends a slot's history
  prefills only the suffix); LRU parking of evicted sessions to pinned host RAM at 37–45 GB/s.
- **Legacy FP6 path.** FP6 E2M3 block-scaled weights (`mma.sync ...mxf8f6f4.block_scale`, FP6
  `ldmatrix ...b6x16` unpack), int4-K/E4M3-V KV, the MTP covering tree + batched FP6 verify, the
  wide prefill — as described in [`legacy-fp6.md`](legacy-fp6.md); `serve_openai.py` still needs an
  E2M3 tqf (it does not load EXL3 weights).

## Converting a model — the annotated recipe (EXL3, the served format)

The README's "Quick start" has the short form of this; the FP6 E2M3 format of the single-stream path is in
[`legacy-fp6.md`](legacy-fp6.md). The quantizer is stock upstream
[exllamav3](https://github.com/turboderp-org/exllamav3) at commit `02aef45` or newer (it carries `convert.py -rcp`
and the `sc_measure.py` / `sc_optimize.py` sensitivity pipeline); its input must be the **original bf16**
`Qwen/Qwen3.8-27B` — quantizing from the FP8 checkpoint costs +0.003 KL (double quantization). The FP6 legacy path
converts either checkpoint. Python: 3.10 or newer (measured with 3.14) with `torch`, `transformers`, `numpy`,
`safetensors`, `tokenizers`; the HTTP verification tools also need `httpx` — the measured versions are pinned in
`requirements.txt` / `pyproject.toml`.

```bash
# 1) EXL3 quantization with exllamav3 (stock upstream, commit 02aef45 or newer), from the ORIGINAL bf16
#    checkpoint. The per-tensor bitrate recipe is tools/recipes/qwen3_8-27b-exl3-opt5.75.yaml (400 tensors;
#    sc_optimize.py output for 5.75 bpw on the sensitivity table of the 27B: K = 4 on 0.7 % of the weights,
#    5 on 34 %, 6 on 55 %, 7 on 10.5 %, 8 on 0.04 %). -hb 6 = 6-bit lm_head, -mb 4 = 4-bit MTP layer (the
#    drafter below), -vb 16 = vision tower unquantized (it is not served). ~40 min on one 5090, 250 x 2048
#    calibration rows (the defaults), mul1 codebook, output scales always.
cd exllamav3
python convert.py -i /path/to/Qwen3.8-27B -w /path/to/work-27b-opt5.75 -o /path/to/Qwen3.8-27B-exl3-opt5.75 \
    -rcp /path/to/qwentin/tools/recipes/qwen3_8-27b-exl3-opt5.75.yaml -hb 6 -mb 4 -vb 16 -d 0 -cpi 300
# The 16 GB model: the two-tier {4,5} recipe at 4.25 bpw (no bpw header in the file, so -b 4.25 is for reporting)
python convert.py -i /path/to/Qwen3.8-27B -w /path/to/work-27b-t45 -o /path/to/Qwen3.8-27B-exl3-t45-s425 \
    -rcp /path/to/qwentin/tools/recipes/qwen3_8-27b-exl3-t45-s425.yaml -b 4.25 -hb 6 -mb 4 -vb 16 -d 0 -cpi 300
# (A new recipe for another bitrate: sc_measure.py on the bf16 model -> measure.json, then
#  `sc_optimize.py -m measure.json -b <bpw> -hb 6 -o recipe.yaml`. The bf16 27B needs ~54 GB of GPU memory
#  for the measurement; our fork added a layer-split option for several 32 GB cards, upstream needs one big GPU.)

# 2) EXL3 checkpoint -> .tqf (CPU only, ~3 min). The Linears are copied verbatim as trellis records
#    (header flag 0x800, --exl3 auto reads quantization_config.json); --mtp-exl3 stores the checkpoint's own
#    4-bpw MTP layer as EXL3 records (flag 0x1000) -- the paged drafter behind --spec reads exactly that
#    (TQ_EMIT_MTP=1 is the default since 2026-10 and may be omitted; the converter warns when an EXL3 source
#    is converted without --mtp-exl3). Without it the server runs with speculative decoding OFF after a warning.
cd qwentin
TQ_EMIT_MTP=1 python3 tools/convert_qwen_tqf.py /path/to/Qwen3.8-27B-exl3-opt5.75 \
    -o /path/to/qwen3_8-27b-exl3-opt5.75-mtp4.tqf --mtp-exl3          # 21.3 GB
TQ_EMIT_MTP=1 python3 tools/convert_qwen_tqf.py /path/to/Qwen3.8-27B-exl3-t45-s425 \
    -o /path/to/qwen3_8-27b-exl3-t45-s425-mtp4.tqf --mtp-exl3         # 16.7 GB, the 16 GB variant
```

`python3 tools/inspect_tqf.py model.tqf` reports the flags, walks the payload (EXL3 records included:
per-bitrate histogram, byte-exact size check) and prints two verdict lines — `verdict EXL3 + MTP ...
PASS` is the file `--spec` needs (bits `0x800`, `0x400` and `0x1000` set; `--expect exl3-mtp` makes
that an exit code). `--mtp-source /path/to/Qwen3.8-27B` instead of `--mtp-exl3` writes a bf16 MTP
head (`-mtp.tqf`) for the single-stream MTP tools; the batched server's drafter needs the EXL3 one
(the converter warns when an EXL3 source is converted without either). Only `serve_batched.py` runs
EXL3 files; `serve_openai.py` needs the FP6 format ([`legacy-fp6.md`](legacy-fp6.md)). The format itself:
[`TQF_FORMAT.md`](TQF_FORMAT.md).

## Repository layout, in full

```
src/forward_qwen.cu          the engine's one CUDA translation unit: it includes qwentin.h, then the src/engine/ parts in order
src/qwentin.h                the C ABI (all qwn_* entry points), generated by tools/gen_qwentin_h.py; a definition that
                             drifts from it does not compile
src/engine/*.cuh             the engine in 26 parts (~30k lines; each part's first line says what it holds): FP6
                             single-stream path, paged pool (int4 / FP4 / trellis KV), EXL3 GEMM/GEMV, paged verify /
                             commit / MTP drafter, DeltaNet state formats, CUDA-graph decode step, parking ring; the
                             .tqf loader and qwn_init in loader.cuh, the paged pool's init in paged_t4.cuh
src/engine/kv_t4_codebook.inc   the trellis-KV codebook compiled into the engine (generated by tools/gen_codebook_inc.py)
tools/serve_batched.py       THE SERVED SERVER: multi-client OpenAI API, paged KV + continuous batching, spec decode,
                             prefix cache, host-RAM parking (/health, /spec)
tools/serve_openai.py        legacy single-stream OpenAI server (FP6 + MTP tree, prefix cache, tools, reasoning split,
                             residual-map hook, Dogs)
tools/convert_qwen_tqf.py    HuggingFace / EXL3 checkpoint -> .tqf (--exl3, --mtp-exl3, the FP6 block-scaled formats)
tools/inspect_tqf.py         inspect a .tqf model file (flags, payload walk incl. EXL3 records, format-v2 size and hashes
                             with --verify, production FP6 / EXL3 + MTP verdicts); the format's reference parser
tools/codebooks/lloyd_kv12a.f32   the codebook's source: 4096 fp32, Lloyd-trained (TQ_KV_T4_CODEBOOK loads another file)
tools/gen_codebook_inc.py    lloyd_kv12a.f32 -> src/engine/kv_t4_codebook.inc (tests/test_codebook_inc.py checks the two agree)
tools/engine_text.py         the engine source in include order as prompt text (needle haystacks, prefill prompts, benches)
tools/qwentin_abi.py         the same ABI as ctypes signatures (bind(lib)), generated with qwentin.h
tools/recipes/*.yaml         the exllamav3 per-tensor bitrate recipes of the served models (opt-5.75, t45-s425)
tools/code_needle.py         needle-in-a-repository retrieval matrix through the OpenAI API
tools/spec_speed.py          per-stream decode tok/s at N streams x context, spec on vs off
tools/spec_server_ab.py      spec on vs off on the same prompts: identical-output count
tools/spec_paged_check.py    exactness of the paged verify / commit path vs plain decode (bare engine)
tools/paged_logit_ab.py      raw paged-decode logits of a build for bitwise A/B between builds
tools/harness_common.py      shared helpers of the HTTP tools (tokenizer, code-corpus snapshots, streaming client) and the
                             quality yardstick's held-out sampler (calibration_ids)
tools/serve_common.py        request-side guards of both servers (--api-key, body limit, max_tokens cap, socket timeout)
tools/qwentin_e2e.py         the quality yardstick on the engine: KL / top-1 vs the bf16 original (docs/QUALITY.md)
tools/exl3_diff.py           the same for a checkpoint through exllamav3 (the FP8 reference row, the EXL3 ladder)
tools/e2e_frontier.py        the gate table: paired dKL / dtop-1 with the 95 % block-bootstrap CI
tools/dequant_fp8_to_bf16.py FP8 checkpoint -> bf16 containers (the FP8 reference row's input)
tools/evalplus.sh            T1: EvalPlus HumanEval+ / MBPP+ against any OpenAI endpoint
tools/paged_smoke.py         batched/paged decode parity + throughput harness (legacy)
tools/mtp_spec_smoke.py      single-stream spec-decode harness (also exports prefill() used by serve_openai)
tools/bench_rounds.py        single-stream decode-only spec-round benchmark (ms/round, net tok/s; nsys hook)
tools/accept_probe.py        teacher-forced accept probe (degeneration-free draft-quality A/B)
tools/persist_phases.py      phase-clock breakdown of the persistent kernel (work vs barrier tails)
tools/serve_smoke.py         2-turn cold + prefix-cache E2E smoke against a running single-stream server
tools/needle_check.py        single-stream long-context retrieval gate
tools/residual_map.py        the residual-stream steering hook's maps-file loader, server flags (+ deprecated aliases),
                             schedule parser and numpy reference (CPU-tested in tests/test_residual_map.py)
tools/residual_map_check.py  kernel-vs-numpy parity of the hook on a maps file (GPU)
tools/mtp_*.py               MTP draft-head training / accept-length eval / dump (single-stream)
tools/convert.py, tools/sparse_pack.py           older converter pieces
bench/reproduce.sh           build -> model check -> served line -> needle / spec A-B / speed -> results-<date>-<commit>.json
tests/                       CPU pytest: the .tqf header parser / payload walk / format-v2 integrity, the servers' request guards, the residual-map
                             flags / file format, the built-in codebook vs its source file, the unity build's part list,
                             qwentin.h / qwentin_abi.py current and the tools' ctypes declarations consistent with them
docs/NOTES.md                the 2026-10 build's narrative: where the memory went, DeepSeek-V4.1, the served stack's performance
docs/ARCHITECTURE.md         this file: highlights, how it works, the annotated conversion recipe, this layout
docs/FLAGS.md                every engine environment flag and server argument, the off-switches
docs/VARIANTS.md             the 16 GB and 24 GB variants (tables + serve lines)
docs/legacy-fp6.md           the 2026-08 single-stream FP6 path (serve_openai.py, Dogs, the residual-map hook, its tables)
docs/QUALITY.md, docs/TQF_FORMAT.md, docs/evidence/   the quality protocol, the file format, the gate logs behind the numbers
CHANGELOG.md                 what landed in each release (v0.10 = the 2026-10 served stack, v0.11 = its prefill + hygiene)
.github/workflows/           CI (compile check, ruff, pytest) and the disabled nightly 5090 gate
```
