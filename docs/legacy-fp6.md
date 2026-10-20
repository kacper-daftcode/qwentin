# FP6 E2M3 models, and the retired single-stream path

The 2026-08 single-stream server `tools/serve_openai.py` — FP6 E2M3 weights, an int4-K / E4M3-V cache, the MTP covering
tree with its batched FP6 verify and persistent kernel, the Reservoir Dogs spec-round watchdog and the residual-stream
steering hook (`--residual-map`) — is not in the tree any more. **v0.11 is the last release with it**: its code, its
tools and this page as it described them are at the `v0.11` tag (`git checkout v0.11`; `docs/legacy-fp6.md` there has
the 2026-08 tables, the serving flags, Dogs and the hook's maps-file format). External code built on the hook pins v0.11.

What stays from that path:

- **FP6 E2M3 `.tqf` files load and serve on the batched server**: `tools/serve_batched.py ... --no-kv-t4 --no-spec`
  (plain decode — the paged MTP drafter needs an EXL3 MTP layer, `convert_qwen_tqf.py --mtp-exl3`; the documented
  FP6 configuration of 2026-10 is `--no-kv-v4 --num-blocks 2112`, 270K tokens with ≤ 8 slots, 29 GB). The served
  stack is the EXL3 line of the [README](../README.md).
- **The converter's FP6 recipe** (below) and `tools/inspect_tqf.py`'s "production FP6" verdict.
- **The single-stream decoder `qwn_decode`** as an independent reference: `serve_batched.py --selftest` and
  `tools/paged_smoke.py` compare the paged path's tokens with it. It runs FP6 / FP8 (QMMA) weights, not EXL3.

## Convert a model: FP6 E2M3

```bash
# FP6 (E2M3) block-scaled weights + MTP head (~22.6 GB). Since 2026-10 this recipe IS the converter's default for a HF
# checkpoint: --block-scaled always --block-layout qmma-e2m3 --block-scale-policy pow2 and the MTP section on (the flags
# below are therefore optional and kept for explicitness). The pre-2026-10 defaults (`auto`, which on a BF16 checkpoint
# silently fell back to plain global-scale FP8 without the MTP head) are behind --legacy-fp8, and the converter prints a
# WARNING whenever the output is not one of the two production formats.
# Optional: TQ_GPU_PACK=1 quantizes/packs on the GPU via the built libforward_qwen.so (much faster than the ~13 min
# numpy path). The SAME command converts the official Qwen3.8-27B-FP8 checkpoint: tensors stored as block-scaled FP8
# (`*_scale_inv`) are dequantized first, so "always"/"pow2"/"qmma-e2m3" requantize from true float weights rather than
# from raw E4M3 codes. Converting from the original bf16 instead of the FP8 checkpoint gives KL 0.0139 instead of 0.0169.
TQ_EMIT_MTP=1 python3 tools/convert_qwen_tqf.py /path/to/Qwen3.6-27B \
    -o /path/to/qwen3_6-27b-e2m3-mtp.tqf \
    --block-scaled always --block-layout qmma-e2m3 --block-scale-policy pow2
```

A production FP6 file reports `verdict production FP6 ... PASS` in `tools/inspect_tqf.py` (`block_scaled_e2m3: True`,
`has_mtp_section: True`, header flags `0x53d`); `flags: 0x1d` / `has_mtp_section: False` (and ~28 GB instead of ~22.6)
is plain FP8 without the MTP head (`verdict production FP6: FAIL -- no block scales ...`) — reconvert without
`--legacy-fp8`. The format itself, with the version-2 header extension: [`TQF_FORMAT.md`](TQF_FORMAT.md).
