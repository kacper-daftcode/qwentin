# Changelog

Releases are git tags; the numbers of a release are pinned at its tag and backed by a log in
[`docs/evidence/`](docs/evidence/README.md) (every gate log names the engine commit and the library md5 it ran).
Measurement dates are UTC. "Bitwise" means the lever reproduces the previous kernels' outputs bit for bit
(`tools/paged_logit_ab.py`, max |d| 0), so it was landed on speed alone; everything else went through the quality
yardstick of [`docs/QUALITY.md`](docs/QUALITY.md).

## [Unreleased]

### Changes

- **The DeltaNet recurrence's q / k norms without per-column block reductions (bitwise).** The decode core and the
  verify's sequential core (`k_tq_linear_decode_core_hs_paged`, `k_tq_linear_core_seq_paged[_all]`) computed each column's
  two norms with a 7-step shared-memory tree each (14 `__syncthreads`) plus 4 more barriers per column; the verify
  carried that chain through every column of every segment in 48 launches per wave and once more in the commit replay —
  6.1 ms of a 33 ms round at 8 streams, 7.5 of 46 at 12. Now one warp sums the 128 squares in the tree's exact
  association (`tq_warp_sumsq128`), the decode core needs one barrier per column and the sequential core computes the
  q / k rows of up to 8 columns behind one barrier and runs the column loop with none (`__launch_bounds__(256, 2)`,
  ≤ 128 registers). Every normalizer is the same float as before, so the library is bitwise against v0.13's (paged
  logits at N = 3 / 8 and with the drafter's hooks: max |d| 0) and speculative decoding stays plain decode's bit for
  bit. The spec round on the frozen code corpus at 32K (60 rounds, two A/B pairs): **21.5 → 21.2 ms at N = 1 (−1.2 %),
  32.9 → 32.0 at N = 8 (−2.7 %), 45.0 → 43.7 at N = 12 (−2.9 %)**; the commit replay 2.66 → 2.26 ms at N = 8, the
  sequential core 69.8 → 45.5 µs per launch. Prefill and the attention path are untouched (204 / 219 kernels
  SASS-identical; the 15 that differ are the three DeltaNet-core families × 5 state formats). Gate:
  `docs/evidence/cand31-all.log`.

## [v0.13] — 2026-10-22

Speculative decoding faster at every stream count, all of it lossless (outputs are plain decode's bit for bit at the
same client count): the drafter argmaxes over a reduced vocabulary (a 0.25 GB head instead of the 0.95 GB lm_head), the
draft goes 6 deep at ≤ 2 streams, the verify wave of 33..64 columns runs in one pass over the weights (12 streams no
longer pay a second weight pass per round), `--spec-k auto` adapts the draft depth to the recent acceptance at 3–6
streams, and the prefill wave's norms run one block per row. Engine `e3be83d`, library `build-qwen/libforward_qwen.so`
md5 `b2e93405f8a6`, the served line `--max-slots 12 --num-blocks 4500` with no flags and no environment on the opt-5.75
version-2 `.tqf` **with the DRAFT lm_head section** (`...-mtp4-dv64k.tqf`: `tools/draft_head_build.py` + `convert_qwen_tqf.py
--draft-head`; a file without the section serves as before with the full-head drafts); the 16 GB line is v0.12's (`t45v4`,
unchanged). CUDA 13.1, driver 610.57.04, RTX 5090 at 550 W.

### Numbers pinned at this tag

| | value | log |
|---|---|---|
| Pooled KV capacity | **576 000 tokens** (4500 × 128-token pages, 12 slots × up to 256K) at **30 138 MiB** at start (+240 MiB for the draft head; 30 586 after a 128K prefill and the 12-stream runs) | `docs/evidence/rel13-table.log` |
| Needle-in-a-repository | **6/6** at 32K and 128K (3 depths × 3 decoys) | `rel13-table.log`; `cand29-all.log` gate 3 |
| Time to first token, cold | **11.4 s @32K · 58.5 s @128K** (v0.12: 11.6 / 60.1; 11.36–11.38 / 58.4–58.5 in the table run, 11.35 / 58.5–58.7 in the cand29 gate on this library) | `rel13-table.log`, `cand29-all.log` |
| Decode per stream @32K, N = 1 / 2 / 4 / 8 / 12, spec on | **213 / 166 / 120 / 88 / 64 tok/s** — the mean of two readings of one prompt per stream (212.6 / 212.5, 165.9 / 165.7, 120.9 / 118.7, 88.2 / 88.5, 63.3 / 65.2); 780 tok/s aggregate at 12 streams (v0.12: 179 / 153 / 114 / 82 / —) | `rel13-table.log` |
| Decode per stream @32K, spec off | 66 / 62 / 57 / 48 / 42 tok/s | `rel13-table.log` |
| Decode per stream @128K, N = 1 / 2, spec on (off) | **123 / 102 tok/s** (58 / 53) (v0.12: 114 / 100) | `rel13-table.log` |
| Spec A/B | 10/10 identical texts, and the same texts as every gate since `cand12` | `rel13-table.log`, `cand27/28/29-all.log` |
| T1 EvalPlus with speculative decoding on | HumanEval/+ **93.9 / 91.5** (= the spec-off reference) on the one-pass wave, the adaptive K and the norm grid | `cand27-all.log`, `cand28-all.log`, `cand29-all.log` |
| Quality vs the bf16 original | KL 0.0021–0.0027, top-1 99.0–99.1 % — unchanged: the trunk, the verify and the decode kernels compute what v0.12's did (the drafter's reduced vocabulary cannot change an output; 212 / 212 kernels SASS-identical through the engine landings, 7 new) | `cand25/27/29-all.log` gate 1 |
| Full occupancy | **800 / 800 tok/s aggregate at 12 streams** in the one-pass gate vs **666 / 680** on the v0.12 + reduced-vocabulary library in the same gate (the table run read 774 / 788) | `cand27-all.log` |
| Code continuation at 3–6 streams | 586 → 659 tok/s aggregate at 4 streams, 735 → 803 at 6 (the adaptive draft depth); prose answers unchanged | `cand28-all.log` |
| Prefill wave | 47.9 → 47.1 ms per 128-column wave at 32K, 73.0 → 72.3 at 128K; the wave logits bitwise | `cand29-wave-ab.log` |
| Wikitext-2 KL (exllamav3 `model_diff.py`, 100 × 2048) | opt-5.75 **0.00207 / 98.35 %** vs the bf16 original; t45v4 0.01123 / 96.01, t45-s425 0.01048 / 96.10 | `docs/evidence/wiki2/`, QUALITY.md § 7 |

The spec-on cells follow the accepted draft length of one prompt per stream (4.62 tokens per round @32K N = 1 at K = 6,
2.86 @128K at K = 4); a round costs ~22 ms @32K and ~23 ms @128K at one stream, 33 ms at 8 and 40–46 ms at 12 (K 2 / 3).

### Changes

- **The drafter's reduced vocabulary (a DRAFT lm_head section in the `.tqf`, `TQ_DRAFT_VOCAB`).** Every draft step
  read the full 248 320-row 6-bit lm_head (0.95 GB, 585 µs — 72 % of the step; 2.3 ms of a 21.3 ms round at one
  stream). The drafter now argmaxes over the 65 536 most frequent token ids through a 0.25 GB EXL3 head built from
  the checkpoint's own lm_head rows (`tools/draft_head_build.py`, the id list in `tools/draft_vocab/`,
  `convert_qwen_tqf.py --draft-head`, flag `0x2000`; docs/TQF_FORMAT.md); the verify keeps the full head, so the
  outputs are the same tokens by construction. Measured on the served line: **191 tok/s at N = 1 @32K (179 before),
  670 aggregate at N = 8 (646), 122 @128K (114)**; 60 traced prompts: outputs 60/60 identical, tokens per round
  −0.5 % (99.7 % of the accepted drafts lie in the vocabulary; the 6-bit requantization agrees with the full head's
  argmax on 99.6 % of the covered positions), tok/s +8.9 %. `TQ_DRAFT_VOCAB=0` or a file without the section = the
  full lm_head as before (the switch-off cell reproduces the stock drafts round for round). +240 MiB on the device.
  Gate: `docs/evidence/cand25-all.log` (211 / 211 kernels SASS-identical + one new argmax kernel, paged logits bitwise
  with the drafter's hooks on and the new file on / off, spec exactness, the served line, T1 with spec on 93.9 / 91.5).
- **`--spec-k auto` drafts 6 deep at ≤ 2 active slots and ≤ 64K of context when the file carries the reduced vocabulary** (4 before; the
  engine reports it through the new `qwn_draft_vocab_size`, 60 entry points). With the cheap draft step the deeper
  draft pays: 60 traced prompts 177.8 → 188.2 tok/s at N = 1 (K 4 → 6; 3.41 → 3.84 tokens per round), outputs
  identical; the served line's speed prompt 191 → 209 tok/s; at 128K the drafter's attention per step grows and K = 6 lost
  7 %, so 4 stays past 64K); 3 at ≤ 8 and 2 above stay (the verify wave's width rules there). Gate: `docs/evidence/cand26-all.log`.
- **A public yardstick next to ours**: exllamav3's `eval/model_diff.py` on wikitext-2 (100 × 2048 tokens) for the
  served and candidate weights vs the bf16 original — docs/QUALITY.md § 7 (`docs/evidence/wiki2/`). It also shows
  that the 16 GB model `t45v4` (−19 % KL on the code yardstick) reads +7 % on prose: its allocation is code-weighted.
- **The verify wave runs 33..64 columns in ONE pass over the weights** (`k_tq_exl3_wide_gemm<KB, 8, F32>`: 64-token
  EXL3 units with fp32 accumulators, the decode step's numerics). The verify used to cap its GEMM units at 32 tokens,
  so a wave of N (K + 1) > 32 columns ran as a 32-column pass plus a tail pass that re-read and re-decoded every Linear:
  at 12 slots with K = 2 (the `--spec-k auto` rule above 8 slots) every round paid it — 49.5 ms per round vs 31.3 at
  K = 1 on the frozen code corpus at 32K; at 8 slots K = 4 cost 48.5 vs 33.7 at K = 3. Now (same corpus, 12 slots):
  K 2 / 3 / 4 = 40.1 / 45.8 / 53.2 ms per round = **861 / 984 / 1022 tok/s aggregate (698 before)**; 8 slots K = 4 39.6 ms.
  Every column is bit-identical to plain decode's (`spec_paged_check.py` at 36 / 40 / 48 / 60 columns, ctx 300, the
  standard cells: 0 differing logits); the ≤ 32-column waves, decode and prefill are untouched (212 / 212 kernels
  SASS-identical, 7 new). The engine reports the one-pass width (`qwn_verify_onepass_cols`, 61 entry points) and
  **`--spec-k auto` drafts 3 deep above 8 active slots** (2 before; the table reads 6/3/3 with the reduced vocabulary)
  — with an older library the server keeps 2. `/health` reports `verify_onepass`. `tools/spec_paged_check.py` gained
  `--kmax` / `--k-fixed` (pinned wave widths) and `--bench` cells at 40 / 36 columns. Gate: `docs/evidence/cand27-all.log`.
- **`--spec-k auto` adapts the draft depth to the recent acceptance** (server only). Deeper drafts pay when the model's
  continuation is predictable (code) and cost when it is not (prose): at 4 streams @32K, K = 6 vs the table's 3 read
  708 vs 564 tok/s aggregate on the frozen CUDA corpus and 442 vs 480 on prose answers. The server now estimates the
  per-draft acceptance of the last ~32 rounds (`spec_p_hat`, a geometric estimate from the accepted counts) and drafts
  6 deep at 3–4 active slots when it is ≥ 0.82, 4 deep at 5–6 slots when ≥ 0.78, the table otherwise; the waves stay
  ≤ 30 columns. Through the server: code continuation at 4 streams 591 → 650 tok/s (+10 %), the design-note answers
  unchanged at 4 / 6 / 8 streams; outputs are the verify's whatever K is (A/B texts identical, T1 unchanged).
  `--spec-k table` keeps the fixed table; `/health` reports `spec_p_hat`. Gate: `docs/evidence/cand28-all.log`.
- **The prefill wave's norms run one block per row** (`k_tq_qwen_rmsnorm_b` and friends: 8 stripes per row each
  recomputed the row's full reduction and only split the writes — a leftover of the single-stream decoder — so the
  128-column wave spent 1.5 ms of its 48 on 128 launches of 11.8 µs). From 16 rows up one block per row: −0.75 ms per
  128-column wave at 32K (47.9 → 47.1 ms), −0.7 at 128K, bitwise (the stripe count cannot change the output; the wave
  logits equal the reference dumps); the decode step and the drafts keep the 8 stripes (one block per row costs +0.3
  ms per round at one stream). Host code only (219 / 219 kernels SASS-identical). Gate: `docs/evidence/cand29-all.log`.

## [v0.12] — 2026-10-21

Two fixes to speculative decoding on the served line, the engine behind a config struct with the served line as its
defaults, the retirement of the single-stream path, and a better 16 GB model. **A long-standing bug is fixed:** since
speculative decoding became the default (2026-10-16), a request whose prompt was shorter than 384 tokens and whose
answer ran past position 384 could turn into token 0 (`!!!!`) from there on — about half of such requests; long-context
use was never affected. Found by running T1 (EvalPlus) through the served line with speculative decoding on, which the
gate now does. The drafter also reads the right hidden state now (+0.17 tokens per round). Engine `40892ff`, library
`build-qwen/libforward_qwen.so` md5 `5f9cee32da7c`, the served line `--max-slots 12 --num-blocks 4500` with no flags and
no environment on the opt-5.75 version-2 `.tqf`; the 16 GB line serves `t45v4` (`tools/recipes/qwen3_8-27b-exl3-t45v4.yaml`).
CUDA 13.1, driver 610.57.04, RTX 5090 at 550 W.

### Numbers pinned at this tag

| | value | log |
|---|---|---|
| Pooled KV capacity | **576 000 tokens** (4500 × 128-token pages, 12 slots × up to 256K) at **29 898 MiB** at start (30 300 after a 128K prefill and the 8-stream runs) | `docs/evidence/rel12-table.log` |
| Needle-in-a-repository | **6/6** at 32K and 128K (3 depths × 3 decoys) | `rel12-table.log`; `cand23-all.log` gate 3 |
| Time to first token, cold | **11.6 s @32K · 60.1 s @128K** (v0.11: 11.6 / 59.7) | `rel12-table.log` |
| Decode per stream @32K, N = 1 / 2 / 4 / 8, spec on | **179 / 153 / 114 / 82 tok/s** — the mean of two readings of one prompt per stream (178.8 / 179.2, 159.4 / 147.2, 115.2 / 112.1, 82.4 / 82.3); 3.83 tokens per round at N = 1 (v0.11: 177 / 153 / 106 / 79 at 3.78) | `rel12-table.log` |
| Decode per stream @32K, spec off | 66 / 62 / 57 / 48 tok/s | `rel12-table.log` |
| Decode per stream @128K, N = 1 / 2, spec on (off) | **114 / 100 tok/s** (58 / 52) at 2.82 tokens per round (v0.11: 115 / 99) | `rel12-table.log` |
| Spec A/B | 10/10 identical texts, and the same texts as every gate since `cand12` | `rel12-table.log` |
| T1 EvalPlus **with speculative decoding on** | HumanEval/+ **93.9 / 91.5**, MBPP/+ **90.7 / 77.8** (= the spec-off reference; before the fix 53.7 / 53.0 with 75 of 164 answers degenerate) | `cand23-all.log` gate 3, `cand23-t1-probe-served-{spec,nospec}.log` |
| Quality vs the bf16 original | KL 0.0021–0.0027, top-1 99.0–99.1 % — the prefill-path rows re-measured on this library are bitwise the 2026-10-17 rows (4K 0.00235 / 98.97 %, 32K 0.00209 / 99.05 %) plus a new 32K seed-2 cell 0.00256 / 98.96 % | `cand22-all.log` gate 1, `e2e/qw-cand1211_*.json` |
| **The 16 GB card** (emulated, 2 slots, 400 blocks, `t45v4`) | 51K tokens at 15 604 → 15 922 MiB; needle 3/3 at 32K and 49K; TTFT 12.0 / 18.8 s; **198 / 182 tok/s @32K / 49K** (78 / 77 off); the engine end to end 0.00921 / 97.99 % (4K), 0.00750 / 98.32 % (32K), 0.00999 / 98.09 % (32K s2) vs the previous `t45-s425` 0.01154 / 97.83, 0.00925 / 98.15, 0.01281 / 97.82 on the same tokens; T1 HumanEval/+ 93.9 / 89.6 | `cand24-all.log`, `rel12-p16.log`, `e2e/qw-t45v4_*.json` |

The decode kernels are v0.10's bit for bit except the three segment-attention kernels of the fix, whose change only
removes the NaN write of blocks that have nothing to write (`cand23-all.log` gate 1: 199 / 211 kernels SASS-identical, the
paged-decode logits bitwise at N = 3 / 8 and at 32K). The spec-on cells moved with the accepted draft length again
(the drafter's input fix: 3.78 → 3.83 tokens per round @32K N = 1), a round still costing ~21 ms @32K and ~25 ms @128K.

### Changes

- **The 16 GB model is `t45v4`**: the same 4.25 bpw, a per-tensor allocation chosen under a KL-weighted, floor-free
  sensitivity table (the research repo's chain v4), converted with a 250-row code-heavy calibration set and a
  token-weighted Hessian (`tools/recipes/qwen3_8-27b-exl3-t45v4.yaml`, `tools/recipes/cal250.safetensors`,
  `tools/exl3_token_weights_patch.py`; the conversion block in docs/ARCHITECTURE.md). Engine end to end vs the previous
  `t45-s425` on the same tokens: ΔKL −0.00234 / −0.00175 / −0.00282 at 4K / 32K s1 / 32K s2 (−19 %), Δtop-1 +0.16 /
  +0.17 / +0.26 pt, every CI clear; the 16 GB line itself (2 slots, 400 blocks) 15 922 MiB after a full run, needle 3/3
  at 49K, 198 / 182 tok/s @32K / 49K, T1 HumanEval/+ 93.9 / 89.6 with speculative decoding on. docs/VARIANTS.md keeps
  the previous model's cells, re-measured on the same library and run. Gate: `docs/evidence/cand24-all.log`.
- **FIXED — speculative decoding could turn a short-prompt answer into `!!!!` from position 384 on (present since
  speculative decoding became the default, 2026-10-16).** A verify wave gives every column its own key-split count
  S_t; the first change (1 → 2) is at position 384, so a wave straddling 383 | 384 mixes S_t = 1 and S_t = 2 columns.
  The three segment-attention kernels (`k_tq_paged_attn_q4_gqa_seg`, `_f16`, `_seg_t4`) launch S_max blocks per
  segment, and a block with split ≥ 1 also ran the final-row path for an S_t = 1 column — with an empty tile range,
  so it wrote 0 × ∞ = NaN over split 0's correct output; the DeltaNet recurrence carried the NaN into the slot's
  state and every later token was token 0. Affected: any spec-on request whose prompt was shorter than 384 tokens
  and whose generation crossed position 384 with a multi-column wave on the boundary (about half of such requests
  at K = 4) — typical chat turns, not the long-context benchmarks (prompts of 384+ tokens never see an S_t = 1
  column; later S_t changes mix only S_t ≥ 2 rows, which take the partials + merge path). Found by running T1
  (EvalPlus) through the served line with speculative decoding ON: HumanEval 0.537 / 0.530 against 0.939 / 0.915
  with `--no-spec`, 75 of 164 answers degenerate. Fix: only split 0 writes final rows (`if (split > 0) return;` in
  the three kernels; 199 / 211 kernels SASS-identical, the 12 instances of the three differ; decode steps carry no
  S_t = 1 row inside an S > 1 grid, so the decode path is bitwise). The gate gains what would have caught it:
  `tools/spec_paged_check.py --ctx 300 / 100` (unaligned short contexts whose rounds cross 384; the old library
  fails them with NaN logits at the wave [383, 384, 385]) on every pool and attention mode, and T1 through the
  served line with speculative decoding on (must equal the spec-off reference). Gate: `docs/evidence/cand23-all.log`.
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
- **The last placeholder constants are gone**: `gc_use_ldsm`, `gc_use_p2p` (never read) and `gc_kv_q4_fp32s` (never
  set since the `TQ_KV_Q4_FP32S` switch went), with the dead fp32 branch of the `q4s_*` helpers. Gate
  (`docs/evidence/cand19-all.log`, `cand19-matrix.log`, `cand19-bank3.log`): 184 of the 211 kernels keep their SASS
  bit for bit, the 27 that change are exactly the readers of the moved or removed constants, and every output is
  bitwise unchanged — the paged path on the trellis, FP4, E4M3-V and FP6 pools, spec exactness, the test decoder,
  the served line.
- **The engine's defaults are the served line's.** `TQ_CTX` 262144 and `TQ_KV_Q4` / `TQ_KV_V4` / `TQ_KV_T4` / `TQ_PDL` /
  `TQ_EMBED_HOST` / `TQ_WIDE_ATTN_MMA` on (were 8192 and off; the server and the tools set them): the README's serve
  command needs no environment and its `[config]` line reads `every TQ_* switch at its default`. A script that relied
  on the earlier defaults sets them explicitly — `tools/paged_smoke.py` now pins the int4-K / E4M3-V pool, the scalar
  wave attention and an 8K context its single-stream comparison was written for.
- **A config struct in the C ABI.** `src/qwentin.h` declares `qwn_config` (one field per `TQ_*` switch, generated from
  the switch table) with `qwn_config_defaults`, `qwn_config_from_env`, `qwn_config_get` and `qwn_init_with(path, cfg)`;
  `qwn_init(path)` is `qwn_config_from_env` + `qwn_init_with`, and `tools/qwentin_abi.py` carries the struct for ctypes
  (`Config`, `CONFIG_FIELDS`). `tools/serve_batched.py` starts the engine through it instead of writing its arguments
  into `os.environ`: the arguments fill every switch the environment leaves unset or empty (an explicit environment
  value still wins; an empty one now counts as unset), and the startup line and `/health` report the engine's
  effective settings. 55 → 59 entry points.
- Gate (`docs/evidence/cand20-all.log`, `cand20-matrix.log`): host code only — all 211 kernels keep their SASS bit
  for bit; the library without `TQ_PDL` / `TQ_KV_V4` / `TQ_KV_T4` reproduces the previous one with them bitwise, and
  the served line runs with no `TQ_*` environment at all (needle, A/B texts and speed as before).
- **The drafter reads the trunk hidden after the final norm.** The paged MTP drafter used to be fed the pre-final-norm
  residual; the Qwen3.5 / 3.6 MTP head was trained on the normalized hidden (exllamav3's `qwen3_5_mtp`, vLLM and
  SGLang feed it that), so `mtp_fc_forward` now applies the model's final RMSNorm to every gathered trunk row
  before the head's own `pre_fc_norm_hidden`. Host code only (211 / 211 kernels SASS-identical) and the outputs
  are unchanged by construction — the verify decides every token; what changes is how often the drafts are
  accepted: **+0.17 tokens per round (3.24 → 3.41 on 38 traced prompts), 176 → 179 tok/s at N = 1 and 644 → 657
  aggregate at N = 8 @32K** on the served line, +7 % on this repository's own code. Found by the drafter
  retraining study, which otherwise came back negative: a head retrained on 100M on-policy tokens adds only
  +0.07 tokens per round over the stock head (63–81 % of the rejections are near-ties of the target that a
  one-layer head cannot resolve). Gate (`docs/evidence/cand21-all.log`): paged logits bitwise with the drafter's
  trunk hooks on, spec exactness 4K + 32K, the served line's needle 6/6 and A/B texts as before.

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
