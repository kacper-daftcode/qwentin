# The legacy single-stream FP6 path (the 2026-08 build)

`tools/serve_openai.py` with FP6 E2M3 block-scaled weights, an int4-K / E4M3-V KV cache, the MTP covering tree with
batched FP6 verify, the Reservoir Dogs spec-round watchdog and the residual-stream steering hook — the path the
2026-08 README was written for. It is still in the tree (`serve_openai.py` needs an E2M3 `.tqf`; it does not load EXL3
weights), its sections are kept here as they stood in the 2026-10-17 README, and **its numbers are the 2026-08 ones**
(CUDA 13.3, driver 595): the 2026-10 evidence pack covers the batched server only, so nothing on this page was
re-measured on engine `2cf671c` / `b2e7a3f`. Since 2026-10 the primary path is the batched server
(`tools/serve_batched.py`, the [README](../README.md)); the 2026-08 column of the 32 GB comparison table is kept in
[`NOTES.md`](NOTES.md) ("Why it's interesting"), and the previous stacks are one flag away on the batched server
(`--no-kv-t4 --no-spec`; [`FLAGS.md`](FLAGS.md) "Off-switches").

Qwen3.6-27B shares the text-tower layout of Qwen3.8-27B and runs identically on this path (the EXL3 path was
measured on Qwen3.8 only).

## Performance (2026-08 build, CUDA 13.3 + driver 595)

Qwen3.8-27B, **FP6 weights + Q4 KV**, single stream, MTP spec-decode (depth 6,
`TQ_MTP_VOCAB_CAP=32768 TQ_NGRAM_DRAFT=1`), cold prefill on the default wide+MMA path with 32-row
query tiles (TQ_WIDE_ATTN_QROWS=32). Measured on an **RTX 5090** (170 SM), 256k ship config
(`TQ_KV_Q4=1 TQ_EMBED_FP8=2`), on code-heavy repo contexts via `tools/bench_coding.py`. Decode
tok/s moves with accept-length at that text offset; ms/round is the hardware truth; "follow-up"
is the second turn on the same conversation (prefix cache hit, only the suffix prefills):

| Context | Decode (ms/round) | Decode (tok/s, cold / follow-up) | Cold prefill (tok/s) | Follow-up prefill |
|--------:|------------------:|---------------------------------:|---------------------:|------------------:|
|  ~short (4k) | 21.9 | **140.9** / **217.2** | 2534 (1.9 s) | 0.16 s (~350 tok suffix) |
|     8k-11k   | ~21.5-21.9 | ~128-157 | ~2170 (10.8k in 5.0 s) | ~0.13 s |
|    32k  | 22.3 | 142.3 / 211.5 | 2059 (18.7 s) | 0.20 s |
|    65k  | 23.1 | ~127-141 | 1866 (35.1 s) | ~0.27 s |
|   128k  | 25.1 | 122.7 / **113.6-140.1*** | 1399 (116 s) | 0.35 s |
|   243k  | ~26-28 | 111.8 / ~105.6 | 826 (294 s) | 0.87 s |

<sub>Provenance: the 2026-08 FP6 build as published in the 2026-08 README (CUDA 13.3, driver 595, RTX 5090 at stock power, `tools/bench_coding.py`); that build's commit, library md5 and gate logs are not in `docs/evidence/` (the 2026-10 pack covers the batched server only) — the row is historical, not re-measured on engine `2cf671c`.</sub>

\* 128k+ follow-up decode is 113.6 at depth 6 and **140.1 tok/s at `--depth 8`**; for
cold 128k keep depth 6.

Steady-state VRAM @256k ≈ **30.8 / 31.4 GiB** (the `TQ_EMBED_FP8=2` 6-bit embed table
is what makes 256k fit on this path — without it the 32 GB card OOMs past ~230k). The decode
column is nearly flat: a 200k-deep conversation decodes at ~83% of the short-context speed. The
long-context gains come from a producer/consumer group attention kernel (one 512-thread CTA: 8
warps score a whole kv group's K read ONCE per super-tile while 8 warps run the previous tile's
P·V from a double-buffered smem slab), fused Q4 scale/code loads, key-split prefill attention and
a context-gated standalone attention path — needle retrieval on this card: 4/4 @120k and
**4/4 @239k** (ship config). End-to-end on the server: a cold 10.8k-token first turn runs in
**~5.0 s**; a follow-up turn hits the prefix cache (e.g. 38528 of 38585 tokens reused) and
prefills only the new suffix in **0.20 s** (32k conversation).

<details>
<summary>RTX PRO 6000 Blackwell (188 SM, same GB202/SM120 class, same FP6 build)</summary>

| Context | Decode (ms/round) | Decode (tok/s) |
|--------:|------------------:|---------------:|
|    32k  | 22.7 | 118 |
|    64k  | 23.5 | 114 |
|   128k  | 25.5 | 115 |
|   200k  | 27.9 | 108 |
|   245k  | 28.0 | ~108 |

<sub>Provenance: the 2026-08 FP6 build on an RTX PRO 6000 Blackwell, as published in the 2026-08 README (CUDA 13.3, driver 595); no log in `docs/evidence/`, not re-measured on engine `2cf671c`.</sub>

Short-context rows and cold prefill are within a few percent of the 5090 (same
silicon class). These rows predate the one-wave grid sizing (measured at the old
default chunk count), so the bigger die has a little headroom left on top.
</details>

The 2026-06 batched-server rows of the previous README (FP6, 447 / 730 / 1159 aggregate tok/s
at N = 8 / 16 / 32, single-stream reference 70 tok/s) are superseded by the served-stack table
in [`NOTES.md`](NOTES.md); the FP6 tqf still runs on `serve_batched.py` with `--no-kv-t4 --no-spec` (the hand-off's
documented FP6 configuration: `--no-kv-v4 --num-blocks 2112` = 270K tokens with ≤ 8 slots, 29 GB)
for an A/B.

## Convert a model: FP6 E2M3 (the legacy single-stream format)

```bash
# FP6 (E2M3) block-scaled weights + MTP head — the production format of the single-stream path (~22.6 GB).
# Since 2026-10 this recipe IS the converter's default for a HF checkpoint: --block-scaled always
# --block-layout qmma-e2m3 --block-scale-policy pow2 and the MTP section on (the flags below are
# therefore optional and kept for explicitness). The pre-2026-10 defaults (`auto`, which on a BF16
# checkpoint silently fell back to plain global-scale FP8 without the MTP head) are behind --legacy-fp8,
# and the converter prints a WARNING whenever the output is not one of the two production formats.
# Optional: TQ_GPU_PACK=1 quantizes/packs on the GPU via the built
# libforward_qwen.so (much faster than the ~13 min numpy path).
# The SAME command converts the official Qwen3.8-27B-FP8 checkpoint:
# tensors stored as block-scaled FP8 (`*_scale_inv`) are dequantized first, so
# "always"/"pow2"/"qmma-e2m3" requantize from true float weights rather than
# from raw E4M3 codes (that includes the MTP projections — HF quantizes those
# too, and the recovered values are stored BF16 in the MTP section). Converting
# from the original bf16 instead of the FP8 checkpoint gives KL 0.0139 instead of 0.0169.
TQ_EMIT_MTP=1 python3 tools/convert_qwen_tqf.py /path/to/Qwen3.6-27B \
    -o /path/to/qwen3_6-27b-e2m3-mtp.tqf \
    --block-scaled always --block-layout qmma-e2m3 --block-scale-policy pow2
```

`.tqf` is qwentin's on-disk format (`docs/TQF_FORMAT.md`): quantized weights in the QMMA fragment
layout (or EXL3 trellis records), the non-quant tensors (embeddings, norms, conv1d) in bf16, and
the MTP draft head for spec-decode. A production FP6 file reports `verdict production FP6 ... PASS`
in `tools/inspect_tqf.py` (`block_scaled_e2m3: True`, `has_mtp_section: True`, header flags
`0x53d`); `flags: 0x1d` / `has_mtp_section: False` (and ~28 GB instead of ~22.6) is plain FP8
without the MTP head (`verdict production FP6: FAIL -- no block scales ...`) — reconvert without
`--legacy-fp8`. The format has no version field or checksum yet; the design for both and for a
loader that refuses mismatched files is the TODO at the end of `docs/TQF_FORMAT.md`.

## Serve (OpenAI API): FP6 + MTP spec-decode (`serve_openai.py`)

```bash
# FP6 + 4-bit KV, 256k context, MTP spec-decode (needs the MTP
# section in the .tqf — see "Convert a model"). The wide+MMA cold-prefill path
# is ON by default (2-3x faster first turns at any length; --no-wide-prefill or
# TQ_WIDE_PREFILL=0 reverts to the 16-token chunked baseline). TQ_EMBED_FP8=2
# (6-bit embed table) is required for the full 256k window on a 32 GB card.
# --model-dir is the original HF checkpoint dir (tokenizer + chat template);
# run from the repo root so the relative --lib path resolves.
CUDA_VISIBLE_DEVICES=0 TQ_CTX=262144 TQ_KV_Q4=1 TQ_EMBED_FP8=2 \
    python3 tools/serve_openai.py --port 8000 --no-thinking \
    --tqf /path/to/qwen3_6-27b-e2m3-mtp.tqf \
    --model-dir /path/to/Qwen3.6-27B \
    --lib build-qwen/libforward_qwen.so
```

Per-request stats are returned under `x_qwentin` (prefill seconds, accept-length, gen tok/s,
prefix-cache hit). `--no-thinking` defaults `enable_thinking=false` (recommended for agents — it
keeps the prefix cache valid across turns). Since 2026-10 this server also binds `127.0.0.1` by
default (it was `0.0.0.0`; pass `--host 0.0.0.0 --api-key KEY` to expose it) and takes the same
`--api-key` / `--max-body-bytes` / `--max-tokens-cap` / `--socket-timeout` guards as the batched
server ([`FLAGS.md`](FLAGS.md) "Exposure"). Note the cap applies to the Responses API's own 8192-token default
budget too — the 16384 default leaves the Responses API's 8192 default untouched; lower it only for public endpoints.

### Reservoir Dogs: spec-round watchdog + fallback drafter (default ON)

When speculative decoding goes sideways mid-request — accept-length collapse (drafter-vs-
target distribution drift) or a dead spec round — the Reservoir Dogs supervisory layer in
`serve_openai.py` keeps the request alive instead of letting it stall or hard-error:

- **Mr. Orange bleeds**: accept_len EMA below `--dogs-accept-min` (default 1.30) across
  `--dogs-min-rounds` (6) steps the drafter down a `(depth, k)` ladder (6/3 -> 4/2 -> 2/1).
- **Mr. Pink walks**: persistent bleeding or a hard round error switches the rest of the
  request to dense decode (still verified against the target model — lossless, just slower).
- **Mr. White watches**: a round exceeding `--dogs-hang-s` (10 s) triggers a non-blocking
  persistent-kernel scratch dump naming the stuck barrier; `--dogs-exit-on-hang` exits with
  code 3 for supervisor restarts.

Per-request visibility: `x_qwentin.dogs` reports the ladder, the rung the request finished
on, the running accept-EMA, and every event. `--no-dogs` disables. Downgrades are one-way
per request; the next request starts at the top rung again. (The batched server's speculative
path has no Dogs: its rounds are bitwise plain decode, and a pool-exhausted draft falls back to a
plain step by itself.)

### Residual-stream steering hook (affine map after layer L)

`serve_openai.py --residual-map maps.json` arms one affine map per hooked decoder layer. The
engine applies it in fp32 to the residual stream right after that layer, on-device, in every
forward path (wide prefill, per-node spec verify, dense decode; the graph-captured variants
included), so the steered tower serves at full spec-decode speed. Per residual row `x`:

```
x += scale * (shift + P (A - I)^T P^T (x - mu))        P: H x 2,  A - I: 2 x 2,  mu, shift: H   (H = 5120)
```

The map acts in a 2-dimensional subspace of the residual stream (rank-2 kernel, one 256-thread
block per row, ~25k FLOPs per token and layer — not measurable in tok/s). Steering maps from
any method that produces this form — a mean-difference direction with its projection, a fitted
affine transport map between two activation distributions, a learned 2-d steering subspace —
can be served this way; the engine does not know or care what the map is for. The maps file
(JSON) carries the layers and the scale, the flags override them:

```
{"format": "qwentin-residual-map-v1", "hidden_size": 5120, "layers": [37, 38], "scale": 0.47,
 "maps": {"37": {"P": [[p0, p1], "... H rows"], "A_minus_I": [[a00, a01], [a10, a11]],
                 "mu": ["... H"], "shift": ["... H"]},
          "38": {"...": "..."}}}
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--residual-map PATH` | off | arm the hook with this maps file (env `TQ_RESIDUAL_MAP`) |
| `--residual-map-layers L1,L2` | the file's `layers` (else every layer in it) | which hooked layers to arm |
| `--residual-map-scale S` | the file's `scale` | the scale; required when the file carries none |
| `--residual-map-decay END,LEN[,reset[,shape[,end_answer[,len_answer]]]]` | fixed scale | default schedule of non-thinking requests: the scale decays from the armed value to `END` over `LEN` generated tokens (`linear`, `pow`, `exp`, `invlog`), restarting after `</think>` with `reset`; the answer phase can get its own floor / length (env `TQ_RESIDUAL_MAP_DECAY`) |
| `--residual-map-sched-api` | off | accept a per-request `residual_map_decay` object (`start`, `end`, `len`, `reset_think`, `shape`, `p`, `end_answer`, `len_answer`) |

The 2026-08 spelling (`--bark-all-day --ot-maps`, `--ot-layers`, `--ot-scale`,
`--bark-decay-default`, `--bark-sched-api`, request field `bark_decay`, env `TQ_OT_MAPS` /
`TQ_BARK_DECAY`) is accepted for one more release as hidden aliases that print a one-line
deprecation notice; the old layer / scale defaults are gone — a maps file without `layers` /
`scale` (the 2026-08 file format: top-level `ot_maps`, keys `A_k_minus_I` / `mu_H` /
`mean_shift`, both still read) needs `--residual-map-scale`. Per-request stats carry
`x_qwentin.residual_map` (layers, scale, the schedule that ran). The Dogs keep watching the
drafter: a map that moves the tower away from the MTP drafter shows up in `x_qwentin.dogs`.
Before serving a new maps file, run the kernel-vs-numpy parity check (GPU; fp64 reference of the
formula above, tolerance 1e-3, measured ~1e-6):

```bash
python3 tools/residual_map_check.py --lib build-qwen/libforward_qwen.so --tqf /path/to/model.tqf --residual-map maps.json
```

The hook is single-stream only (`serve_openai.py`); the batched server serves the plain tower.
The engine entry points are `qwn_rmap_hook_add` / `qwn_rmap_hook_count` / `qwn_rmap_set_scale` /
`qwn_rmap_apply_debug` in `src/forward_qwen.cu` (the 2026-08 `qwn_ot_*` names stay as aliases for one release). Maps,
tuning tools and results for a specific steering method live outside this repository — the
Optimal-Transport refusal-removal maps of
[LLobotomy](https://github.com/kacper-daftcode/LLobotomy), for example, in
`kacper-daftcode/llobotomy-qwentin` (link to follow once published).

## Verification tools of the single-stream path

Legacy single-stream tools (FP6 tqf): `tools/mtp_spec_smoke.py` (end-to-end MTP spec-decode:
tok/s, accept-length, divergence vs greedy), `tools/bench_rounds.py` (decode-only round benchmark;
`--profile` brackets the timed rounds for `nsys profile -c cudaProfilerApi`) and
`tools/needle_check.py` (`TQ_MODEL_TQF` / `TQ_MODEL_DIR` / `TQ_LIB` from the environment), with
the invocations of the 2026-08 README:

```bash
CUDA_VISIBLE_DEVICES=0 python3 tools/mtp_spec_smoke.py --prompt-tokens 1024 --gen 128 \
    --tqf /path/to/qwen3_6-27b-e2m3-mtp.tqf --model-dir /path/to/Qwen3.6-27B --lib build-qwen/libforward_qwen.so
CUDA_VISIBLE_DEVICES=0 TQ_KV_Q4=1 TQ_CTX=262144 TQ_EMBED_FP8=2 python3 tools/bench_rounds.py --prompt-tokens 65536 --rounds 200 \
    --tqf /path/to/qwen3_6-27b-e2m3-mtp.tqf --model-dir /path/to/Qwen3.6-27B --lib build-qwen/libforward_qwen.so
CUDA_VISIBLE_DEVICES=0 TQ_CTX=16384 TQ_MODEL_TQF=/path/to/qwen3_6-27b-e2m3-mtp.tqf TQ_MODEL_DIR=/path/to/Qwen3.6-27B \
    TQ_LIB=build-qwen/libforward_qwen.so python3 tools/needle_check.py
```
