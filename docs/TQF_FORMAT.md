# The `.tqf` file format (TQF1) — layout, flags, format version 2 (recorded size and hashes)

`.tqf` is qwentin's on-disk model format: one header, then the text tower's tensors in a fixed order, then (optionally)
the MTP head section. It is written by `tools/convert_qwen_tqf.py`, read by the engine's `parse_tqf` (in
`src/engine/loader.cuh`) and validated by `tools/inspect_tqf.py`, whose slot walk mirrors the writer and is the
reference for this document. Byte order is little-endian throughout.

## Header

```
offset  size  field
0       4     magic "TQF1"
4       4     u32 header_bytes            = 8 + 108 + num_hidden_layers (+ 80 in format version 2: the extension below)
8       108   struct "<16I3f8I":
                u32 flags                 (below)
                u32 model_family          1 = Qwen3.5-class text tower (Qwen3.5 / 3.6 / 3.8 share the layout)
                u32 non_quant_dtype       1 = fp16, 2 = bf16 (the storage dtype of embeddings, norms, conv, dt_bias)
                u32 hidden_size, intermediate_size, num_hidden_layers, vocab_size
                u32 num_attention_heads, num_key_value_heads, head_dim
                u32 linear_num_key_heads, linear_key_head_dim, linear_num_value_heads, linear_value_head_dim
                u32 linear_conv_kernel_dim, max_position_embeddings
                f32 rms_norm_eps, rope_theta, partial_rotary_factor
                u32 mrope_interleaved, mrope_section[3]
                u32 tie_word_embeddings, text_only, has_vision_config, has_mtp
116     L     u8 layer_types[num_hidden_layers]   1 = linear_attention (gated DeltaNet), 2 = full_attention
```

`has_vision_config` / `has_mtp` describe the *source* checkpoint (the vision tower is always dropped); whether an MTP
section is *present in the file* is flag `0x400`.

### Flags

| bit | name | meaning |
|---|---|---|
| `0x1` | `fp8_qmma_weights` | always set: the Linear slots hold quantized records (QMMA tiles or EXL3 records) |
| `0x2` | `tied_embeddings` | no `lm_head` slot; the embedding table is the head |
| `0x4` | `source_bf16_non_quant` | non-quant tensors are bf16 (else fp16) |
| `0x8` | `dropped_vision` | the source had a vision tower (not stored) |
| `0x10` | `dropped_mtp` | the source had an MTP head (stored only with `0x400`) |
| `0x20` | `block_scaled_qmma` | Linear records carry 128×128 block scales |
| `0x40` | `block_scaled_rowmajor` | block-scaled FP8 payload in row-major order (not a tensor-core layout) |
| `0x80` | `block_scaled_qmma_wordmajor` | block-scaled FP8 payload, word-major tiles (an experiment layout) |
| `0x100` | `block_scaled_e2m3` | **FP6**: 6-bit E2M3 codes in dense QMMA tiles (384 B per 16×32 tile), activations E4M3 |
| `0x200` | `sparse_24_e2m3` | an experimental 2:4-sparse draft record follows each linear-attention projection (sizes not derivable from the header) |
| `0x400` | `has_mtp_section` | the MTP head section follows the main payload |
| `0x800` | `exl3_weights` | **EXL3**: Linear slots are exllamav3 trellis records (variable size, see below) |
| `0x1000` | `mtp_exl3` | the MTP section's 8 projections are EXL3 records (else bf16 matrices) |

The two production formats and what `tools/inspect_tqf.py` checks for its PASS / FAIL lines:

- **production FP6** (the 2026-08 format; `tools/serve_batched.py --no-kv-t4 --no-spec` serves it): `0x1 | 0x20 | 0x100 | 0x400`, i.e. flags
  `0x53d` for a bf16 source with a dropped vision tower. Written by `tools/convert_qwen_tqf.py` with its defaults
  (`--block-scaled always --block-layout qmma-e2m3 --block-scale-policy pow2`, MTP on). 22.6 GB for the 27B.
- **EXL3 + MTP** (the served format, `tools/serve_batched.py --spec`): `0x800 | 0x400 | 0x1000`, i.e. `0x1c1d`.
  Written from an exllamav3 checkpoint with `--mtp-exl3`. 21.3 GB for opt-5.75, 16.7 GB for t45-s425.

Other combinations run (`0xc1d` = EXL3 with a bf16 MTP head, the `-mtp.tqf` of the single-stream MTP tools;
`0x1d` = plain global-scale FP8 without the drafter, what the pre-2026-10 converter defaults produced from a bf16
checkpoint) but are not what the README measured; the verdict lines say so.

## Main payload: tensor slots in file order

For the layout below `h` = hidden_size, `i` = intermediate_size, `v` = vocab_size, `hd` = head_dim, `A` / `KV` =
attention / KV heads, `Vh × Vd` = linear value heads × dim, `Kh × Kd` = linear key heads × dim, `conv_dim = 2·Kh·Kd + Vh·Vd`.

```
embed_tokens          non_quant  [v, h]
norm                  non_quant  [h]
lm_head               qm8        [v, h]            (absent when tied_embeddings)
per layer l:
  input_layernorm     non_quant  [h]
  post_attention_ln   non_quant  [h]
  mlp.gate_proj       qm8        [i, h]
  mlp.up_proj         qm8        [i, h]
  mlp.down_proj       qm8        [h, i]
  if layer_types[l] == 1 (gated DeltaNet):
    A_log             f32        [Vh]
    dt_bias           non_quant  [Vh]
    norm              f32        [Vd]
    conv1d            non_quant  [conv_dim, 1, conv_kernel]
    in_proj_a         qm8        [Vh, h]           (48 wide: EXL3 checkpoints keep it bf16 -> kbits = 0 record)
    in_proj_b         qm8        [Vh, h]
    in_proj_qkv       qm8        [conv_dim, h]
    in_proj_z         qm8        [Vh·Vd, h]
    out_proj          qm8        [h, Vh·Vd]
  if layer_types[l] == 2 (full attention):
    q_norm, k_norm    non_quant  [hd] each
    q_proj            qm8        [2·A·hd, h]        (q and the gate, fused as in the HF checkpoint)
    k_proj, v_proj    qm8        [KV·hd, h] each
    o_proj            qm8        [h, A·hd]
```

Slot encodings:

- `non_quant`: the elements as u16 (bf16 or fp16 per `non_quant_dtype`), row-major, no header.
- `f32`: the elements as f32.
- `qm8`, plain QMMA (no `0x20`): `f32 global_scale` + tiles of 16×32 E4M3 codes in the tensor-core fragment order
  (512 B per tile). Size `4 + (m/16)(k/32)·512`.
- `qm8`, block-scaled (`0x20`): `u32 scale_rows, u32 scale_cols` + `f32 scale[scale_rows·scale_cols]` (one per 128×128
  block; `pow2` policy = powers of two for the `mxf8f6f4.block_scale` MMA) + the tiles — 512 B per 16×32 tile of
  E4M3, **384 B of E2M3 with `0x100`** (6-bit codes packed densely, unpacked by `ldmatrix ...b6x16`). Size
  `8 + 4·⌈m/128⌉·⌈k/128⌉ + (m/16)(k/32)·(384 | 512)`.
- `qm8`, EXL3 (`0x800`): `struct "<IIII"`: `kbits, K (= in_features), M (= out_features), codebook` (0 = 3inst, 1 = mcg,
  2 = mul1), then
  - `kbits > 0`: `suh fp16[K]`, `svh fp16[M]` (exllamav3's input / output sign-Hadamard scales), `trellis int16[K/16][M/16][16·kbits]`
    exactly as exllamav3 stores it (the engine dequantizes 16×16 tiles on the fly); size `16 + 2K + 2M + (K/16)(M/16)·16·kbits·2`;
  - `kbits = 0`: the checkpoint left this Linear unquantized → `bf16[M][K]` row-major; size `16 + 2·M·K`.
  Because `kbits` varies per Linear (the per-tensor bitrate recipes in `tools/recipes/`), an EXL3 file's payload can only
  be sized by reading each record header — `tools/inspect_tqf.py` does exactly that.

## MTP head section (`0x400`)

Appended right after the main payload, in this order (`mtp_tensor_specs()` in the converter): `pre_fc_norm_embedding
[h]`, `pre_fc_norm_hidden [h]`, `fc [h, 2h]`, `layers.0.input_layernorm [h]`, `layers.0.post_attention_layernorm [h]`,
`q_norm [hd]`, `k_norm [hd]`, `q_proj [2·A·hd, h]`, `k_proj [KV·hd, h]`, `v_proj [KV·hd, h]`, `o_proj [h, A·hd]`,
`gate_proj [i, h]`, `up_proj [i, h]`, `down_proj [h, i]`, `norm [h]`. The norms are `non_quant`; the 8 projections are
`non_quant` bf16 matrices (0.85 GB for the 27B; the single-stream MTP tools and `--mtp-source`) or, with `0x1000`,
EXL3 records (0.25 GB on the device; the batched server's paged drafter, `--mtp-exl3`).

## Format version 2: recorded size, hashes, and a loader that refuses

Since `src/engine/loader.cuh` / `tools/convert_qwen_tqf.py` of 2026-10-18 the converter writes **format version 2**:
the header above, then an 80-byte **extension** right after the layer-type bytes (`header_bytes` = 116 + L + 80),
then the payload as before. The magic stays `TQF1`; a file without the extension is version 1 and loads as it always
did (the served `-mtp4.tqf` of v0.10 is version 1; v0.11 was measured on its version-2 re-conversion). An engine
older than this rule refuses a version-2 file
("Unexpected header_bytes") instead of misreading it.

```
extension (struct "<IIQ32s32s", 80 bytes)
0       4     u32 format_version            2
4       4     u32 extension_bytes           80 (later versions may append fields after these)
8       8     u64 payload_bytes             file size - header_bytes: the main payload + the MTP section
16      32    u8  payload_hash[32]          BLAKE2b-256 (digest_size 32, unkeyed) of the payload
48      32    u8  header_hash[32]           BLAKE2b-256 of the header [0, header_bytes) with bytes 16..80 of the
                                            extension (both hashes) zeroed
```

The converter hashes the payload while writing it and seals the header last, so an interrupted conversion leaves a
header with `payload_bytes` 0 and zero hashes, which the engine refuses. The checks, and who makes them:

| check | engine (`qwn_init`, `qwn_probe`) | `tools/inspect_tqf.py` |
|---|---|---|
| format version above 2 | refused, code **-40** ("format vN needs a newer engine") | FAIL |
| malformed extension (`extension_bytes` < 80 or not matching `header_bytes`) | refused, **-45** | FAIL (`ValueError`) |
| unknown flag bits | refused in a version-2 file, **-44**; a warning in version 1 | FAIL in version 2, a note in version 1 |
| header hash | always (a few hundred bytes), **-42** on a mismatch | always |
| `payload_bytes` vs the file size (a truncated or extended copy) | always (one `fseek`), **-41** | always |
| payload hash (a flipped bit anywhere) | with `TQ_VERIFY_TQF=1`: one extra read of the file, ~20 s for 21 GB at ~1 GB/s, **-43** | with `--verify` (~30 s) |

`qwn_init` prints one line per file (`TQF1 format v2: payload ... bytes as recorded, header hash OK, payload hash
...`); the servers stop on a negative code. `tests/test_inspect_tqf.py` (CPU) covers the extension, the hashes and each
defect on synthetic files; `tests/test_tqf_hash.py` checks the engine's BLAKE2b against `hashlib` and each refusal code
through `qwn_probe` (it needs a built library and is skipped without one). Re-converting a version-1 file gives the same
payload byte for byte (checked on the served opt-5.75 `-mtp4.tqf`: the version-2 file is the version-1 file plus the
80-byte extension, and the engine's paged-decode logits on both are bitwise equal).
