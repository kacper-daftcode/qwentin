#!/usr/bin/env python3
"""Inspect and validate a .tqf model file (qwentin's on-disk format, docs/TQF_FORMAT.md).

Prints the header (flags, model dimensions, layer types), walks the payload to check that the file holds exactly the
tensors the header announces, and prints two one-line verdicts:

  verdict production FP6 ...  PASS  = FP6 E2M3 block-scaled weights + the MTP head section (the 2026-08 format;
                                      tools/serve_batched.py serves it with plain decode; flags 0x1 | 0x20 | 0x100 | 0x400)
  verdict EXL3 + MTP ...      PASS  = EXL3 trellis weights + the EXL3 MTP layer (the served format, tools/serve_batched.py
                                      --spec; flags 0x800 | 0x400 | 0x1000)

A FAIL names what is missing (e.g. a plain-FP8 file written by an old `--block-scaled auto` default, an EXL3 file
converted without --mtp-exl3 that runs with speculative decoding OFF). A version-2 file (docs/TQF_FORMAT.md) also
carries its payload size, a payload hash and a header hash: the size and the header hash are always checked, the
payload hash with --verify (a full read). Exit code: 0 when the payload matches the header, 1 when it does not (a
truncated / foreign / corrupted file), 2 when --expect {fp6,exl3-mtp} names a verdict that failed.

  python3 tools/inspect_tqf.py model.tqf [--expect exl3-mtp] [--verify]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import struct
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

FIXED_HEADER_FMT = "<16I3f8I"
FIXED_HEADER_SIZE = struct.calcsize(FIXED_HEADER_FMT)
EXL3_RECORD_FMT = "<IIII"          # kbits, K (in), M (out), codebook

FLAG_FP8_QMMA_WEIGHTS = 1 << 0
FLAG_TIED_EMBEDDINGS = 1 << 1
FLAG_SOURCE_BF16_NON_QUANT = 1 << 2
FLAG_DROPPED_VISION = 1 << 3
FLAG_DROPPED_MTP = 1 << 4
FLAG_BLOCK_SCALED_QMMA = 1 << 5
FLAG_BLOCK_SCALED_ROWMAJOR = 1 << 6
FLAG_BLOCK_SCALED_QMMA_WORDMAJOR = 1 << 7
FLAG_BLOCK_SCALED_E2M3 = 1 << 8
FLAG_SPARSE_24_E2M3 = 1 << 9
FLAG_HAS_MTP = 1 << 10
FLAG_EXL3_WEIGHTS = 1 << 11
FLAG_MTP_EXL3 = 1 << 12
FLAG_NAMES = {
    FLAG_FP8_QMMA_WEIGHTS: "fp8_qmma_weights", FLAG_TIED_EMBEDDINGS: "tied_embeddings",
    FLAG_SOURCE_BF16_NON_QUANT: "source_bf16_non_quant", FLAG_DROPPED_VISION: "dropped_vision",
    FLAG_DROPPED_MTP: "dropped_mtp", FLAG_BLOCK_SCALED_QMMA: "block_scaled_qmma",
    FLAG_BLOCK_SCALED_ROWMAJOR: "block_scaled_rowmajor", FLAG_BLOCK_SCALED_QMMA_WORDMAJOR: "block_scaled_qmma_wordmajor",
    FLAG_BLOCK_SCALED_E2M3: "block_scaled_e2m3", FLAG_SPARSE_24_E2M3: "sparse_24_e2m3", FLAG_HAS_MTP: "has_mtp_section",
    FLAG_EXL3_WEIGHTS: "exl3_weights", FLAG_MTP_EXL3: "mtp_exl3",
}
KNOWN_FLAGS = (FLAG_MTP_EXL3 << 1) - 1
FORMAT_VERSION = 2                 # files with the header extension; a file without one is version 1
EXT_FMT = "<IIQ32s32s"             # format_version, extension_bytes, payload_bytes, payload_hash, header_hash
EXT_SIZE = struct.calcsize(EXT_FMT)
PRODUCTION_FP6_FLAGS = FLAG_FP8_QMMA_WEIGHTS | FLAG_BLOCK_SCALED_QMMA | FLAG_BLOCK_SCALED_E2M3 | FLAG_HAS_MTP
EXL3_MTP_FLAGS = FLAG_EXL3_WEIGHTS | FLAG_HAS_MTP | FLAG_MTP_EXL3


@dataclass(frozen=True)
class TqfHeader:
    header_bytes: int
    flags: int
    model_family: int
    non_quant_dtype: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    vocab_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    linear_num_key_heads: int
    linear_key_head_dim: int
    linear_num_value_heads: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    partial_rotary_factor: float
    mrope_interleaved: int
    mrope_section: tuple[int, int, int]
    tie_word_embeddings: int
    text_only: int
    has_vision_config: int
    has_mtp: int
    layer_types: list[int]
    format_version: int = 1
    payload_bytes: int | None = None       # version 2: the payload size the writer recorded
    payload_hash: bytes | None = None      # version 2: BLAKE2b-256 of the payload
    header_hash_ok: bool | None = None     # version 2: the stored header hash matches the header


@dataclass(frozen=True)
class Slot:
    """One tensor slot of the payload, in file order. kind: non_quant (u16 per element), f32, qm8 (a quantized Linear:
    a QMMA / block-scaled record, or an EXL3 record), with shape (rows, cols) for qm8."""
    name: str
    kind: str
    count: int                 # elements (non_quant / f32)
    shape: tuple[int, int] = (0, 0)


def header_hash(header: bytes, ext_start: int) -> bytes:
    """BLAKE2b-256 of a version-2 header with its two hash fields (extension bytes 16..80) zeroed."""
    h = bytearray(header)
    h[ext_start + 16:ext_start + 80] = bytes(64)
    return hashlib.blake2b(bytes(h), digest_size=32).digest()


def add_extension(header_v1: bytes) -> bytes:
    """A version-1 header (magic .. layer types) -> a version-2 header whose extension is not sealed yet."""
    return (header_v1[:4] + struct.pack("<I", len(header_v1) + EXT_SIZE) + header_v1[8:]
            + struct.pack(EXT_FMT, FORMAT_VERSION, EXT_SIZE, 0, bytes(32), bytes(32)))


def seal_header(header: bytes, payload_bytes: int, payload_hash: bytes) -> bytes:
    """Fill in the payload size, the payload hash and the header hash of a header made by add_extension."""
    ext = len(header) - EXT_SIZE
    h = bytearray(header)
    h[ext:] = struct.pack(EXT_FMT, FORMAT_VERSION, EXT_SIZE, payload_bytes, payload_hash, bytes(32))
    h[ext + 48:] = header_hash(bytes(h), ext)
    return bytes(h)


def payload_hash(f: BinaryIO, header_bytes: int) -> bytes:
    f.seek(header_bytes)
    h = hashlib.blake2b(digest_size=32)
    while chunk := f.read(1 << 24):
        h.update(chunk)
    return h.digest()


def read_exact(f: BinaryIO, nbytes: int) -> bytes:
    data = f.read(nbytes)
    if len(data) != nbytes:
        raise EOFError(f"expected {nbytes} bytes, got {len(data)}")
    return data


def read_header(f: BinaryIO) -> TqfHeader:
    magic = read_exact(f, 4)
    if magic != b"TQF1":
        raise ValueError(f"bad magic: {magic!r}")
    header_bytes = struct.unpack("<I", read_exact(f, 4))[0]
    if header_bytes < 8 + FIXED_HEADER_SIZE:
        raise ValueError(f"invalid header_bytes: {header_bytes}")

    rest = read_exact(f, header_bytes - 8)
    values = struct.unpack_from(FIXED_HEADER_FMT, rest, 0)
    (
        flags,
        model_family,
        non_quant_dtype,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        vocab_size,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        linear_num_key_heads,
        linear_key_head_dim,
        linear_num_value_heads,
        linear_value_head_dim,
        linear_conv_kernel_dim,
        max_position_embeddings,
        rms_norm_eps,
        rope_theta,
        partial_rotary_factor,
        mrope_interleaved,
        mrope0,
        mrope1,
        mrope2,
        tie_word_embeddings,
        text_only,
        has_vision_config,
        has_mtp,
    ) = values
    tail = rest[FIXED_HEADER_SIZE:]
    if len(tail) < num_hidden_layers:
        raise ValueError(f"layer type count mismatch: header has {num_hidden_layers}, bytes have {len(tail)}")
    layer_types = list(tail[:num_hidden_layers])
    ext = tail[num_hidden_layers:]
    version, stored_size, stored_hash, hh_ok = 1, None, None, None
    if ext:
        if len(ext) < EXT_SIZE:
            raise ValueError(f"layer type count mismatch: header has {num_hidden_layers}, bytes have {len(tail)}")
        version, ext_bytes, stored_size, stored_hash, stored_hh = struct.unpack_from(EXT_FMT, ext, 0)
        if version < 2 or ext_bytes != len(ext):
            raise ValueError(f"bad header extension: format_version {version}, extension_bytes {ext_bytes} "
                             f"({len(ext)} present)")
        ext_start = 8 + FIXED_HEADER_SIZE + num_hidden_layers
        hh_ok = header_hash(magic + struct.pack("<I", header_bytes) + rest, ext_start) == stored_hh
    return TqfHeader(
        header_bytes=header_bytes,
        flags=flags,
        model_family=model_family,
        non_quant_dtype=non_quant_dtype,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        vocab_size=vocab_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        linear_num_key_heads=linear_num_key_heads,
        linear_key_head_dim=linear_key_head_dim,
        linear_num_value_heads=linear_num_value_heads,
        linear_value_head_dim=linear_value_head_dim,
        linear_conv_kernel_dim=linear_conv_kernel_dim,
        max_position_embeddings=max_position_embeddings,
        rms_norm_eps=rms_norm_eps,
        rope_theta=rope_theta,
        partial_rotary_factor=partial_rotary_factor,
        mrope_interleaved=mrope_interleaved,
        mrope_section=(mrope0, mrope1, mrope2),
        tie_word_embeddings=tie_word_embeddings,
        text_only=text_only,
        has_vision_config=has_vision_config,
        has_mtp=has_mtp,
        layer_types=layer_types,
        format_version=version,
        payload_bytes=stored_size,
        payload_hash=stored_hash,
        header_hash_ok=hh_ok,
    )


def flag_names(flags: int) -> list[str]:
    names = [n for bit, n in sorted(FLAG_NAMES.items()) if flags & bit]
    unknown = flags & ~sum(FLAG_NAMES)
    if unknown:
        names.append(f"unknown:0x{unknown:x}")
    return names


# ------------------------------------------------------------------------------------------------ the slot sequence

def payload_slots(header: TqfHeader) -> Iterator[Slot]:
    """The main payload's tensors in file order -- mirrors tensor_specs() in tools/convert_qwen_tqf.py (and the
    engine's parse_tqf read order)."""
    h = header.hidden_size
    i = header.intermediate_size
    v = header.vocab_size
    heads = header.num_attention_heads
    kv_heads = header.num_key_value_heads
    hd = header.head_dim
    lin_key_dim = header.linear_num_key_heads * header.linear_key_head_dim
    lin_value_dim = header.linear_num_value_heads * header.linear_value_head_dim
    lin_conv_dim = lin_key_dim * 2 + lin_value_dim
    yield Slot("model.language_model.embed_tokens.weight", "non_quant", v * h)
    yield Slot("model.language_model.norm.weight", "non_quant", h)
    if not header.tie_word_embeddings:
        yield Slot("lm_head.weight", "qm8", 0, (v, h))
    for layer, layer_type in enumerate(header.layer_types):
        p = f"model.language_model.layers.{layer}"
        yield Slot(f"{p}.input_layernorm.weight", "non_quant", h)
        yield Slot(f"{p}.post_attention_layernorm.weight", "non_quant", h)
        yield Slot(f"{p}.mlp.gate_proj.weight", "qm8", 0, (i, h))
        yield Slot(f"{p}.mlp.up_proj.weight", "qm8", 0, (i, h))
        yield Slot(f"{p}.mlp.down_proj.weight", "qm8", 0, (h, i))
        if layer_type == 1:      # linear attention (gated DeltaNet)
            yield Slot(f"{p}.linear_attn.A_log", "f32", header.linear_num_value_heads)
            yield Slot(f"{p}.linear_attn.dt_bias", "non_quant", header.linear_num_value_heads)
            yield Slot(f"{p}.linear_attn.norm.weight", "f32", header.linear_value_head_dim)
            yield Slot(f"{p}.linear_attn.conv1d.weight", "non_quant", lin_conv_dim * header.linear_conv_kernel_dim)
            yield Slot(f"{p}.linear_attn.in_proj_a.weight", "qm8", 0, (header.linear_num_value_heads, h))
            yield Slot(f"{p}.linear_attn.in_proj_b.weight", "qm8", 0, (header.linear_num_value_heads, h))
            yield Slot(f"{p}.linear_attn.in_proj_qkv.weight", "qm8", 0, (lin_conv_dim, h))
            yield Slot(f"{p}.linear_attn.in_proj_z.weight", "qm8", 0, (lin_value_dim, h))
            yield Slot(f"{p}.linear_attn.out_proj.weight", "qm8", 0, (h, lin_value_dim))
        elif layer_type == 2:    # full attention
            yield Slot(f"{p}.self_attn.q_norm.weight", "non_quant", hd)
            yield Slot(f"{p}.self_attn.k_norm.weight", "non_quant", hd)
            yield Slot(f"{p}.self_attn.q_proj.weight", "qm8", 0, (heads * hd * 2, h))
            yield Slot(f"{p}.self_attn.k_proj.weight", "qm8", 0, (kv_heads * hd, h))
            yield Slot(f"{p}.self_attn.v_proj.weight", "qm8", 0, (kv_heads * hd, h))
            yield Slot(f"{p}.self_attn.o_proj.weight", "qm8", 0, (h, heads * hd))
        else:
            raise ValueError(f"unsupported layer type byte: {layer_type}")


def mtp_slots(header: TqfHeader) -> Iterator[Slot]:
    """The MTP head section (flag 0x400) -- mirrors mtp_tensor_specs() in tools/convert_qwen_tqf.py: one full-attention
    decoder layer + fc + the pre_fc / post norms. The 8 projections are bf16 matrices, or EXL3 records with flag 0x1000."""
    h, i, heads, kv_heads, hd = (header.hidden_size, header.intermediate_size, header.num_attention_heads,
                                 header.num_key_value_heads, header.head_dim)
    proj = "qm8" if header.flags & FLAG_MTP_EXL3 else "non_quant"
    yield Slot("mtp.pre_fc_norm_embedding.weight", "non_quant", h)
    yield Slot("mtp.pre_fc_norm_hidden.weight", "non_quant", h)
    yield Slot("mtp.fc.weight", proj, h * 2 * h, (h, 2 * h))
    yield Slot("mtp.layers.0.input_layernorm.weight", "non_quant", h)
    yield Slot("mtp.layers.0.post_attention_layernorm.weight", "non_quant", h)
    yield Slot("mtp.layers.0.self_attn.q_norm.weight", "non_quant", hd)
    yield Slot("mtp.layers.0.self_attn.k_norm.weight", "non_quant", hd)
    yield Slot("mtp.layers.0.self_attn.q_proj.weight", proj, heads * hd * 2 * h, (heads * hd * 2, h))
    yield Slot("mtp.layers.0.self_attn.k_proj.weight", proj, kv_heads * hd * h, (kv_heads * hd, h))
    yield Slot("mtp.layers.0.self_attn.v_proj.weight", proj, kv_heads * hd * h, (kv_heads * hd, h))
    yield Slot("mtp.layers.0.self_attn.o_proj.weight", proj, h * heads * hd, (h, heads * hd))
    yield Slot("mtp.layers.0.mlp.gate_proj.weight", proj, i * h, (i, h))
    yield Slot("mtp.layers.0.mlp.up_proj.weight", proj, i * h, (i, h))
    yield Slot("mtp.layers.0.mlp.down_proj.weight", proj, h * i, (h, i))
    yield Slot("mtp.norm.weight", "non_quant", h)


# ------------------------------------------------------------------------------------------------ sizes

def qmma_bytes(m: int, k: int, block_scaled: bool = False, e2m3: bool = False) -> int:
    """A QMMA Linear record: 4-byte global scale + tiles (512 B per 16x32 tile of E4M3, 384 B of E2M3), or with
    block scales: 8-byte header + one fp32 scale per 128x128 block + tiles."""
    if m % 16 or k % 32:
        raise ValueError(f"unaligned QMMA shape {m}x{k}")
    tiled = (m // 16) * (k // 32) * (384 if e2m3 else 512)
    if block_scaled:
        scale_rows = (m + 127) // 128
        scale_cols = (k + 127) // 128
        return 8 + scale_rows * scale_cols * 4 + tiled
    return 4 + tiled


def non_quant_bytes(count: int) -> int:
    return count * 2


def f32_bytes(count: int) -> int:
    return count * 4


def exl3_record_bytes(hdr: bytes, slot: Slot) -> int:
    """Size of one EXL3 record from its 16-byte header (write_exl3_record in tools/convert_qwen_tqf.py): kbits > 0 ->
    suh fp16[K] + svh fp16[M] + trellis int16[K/16][M/16][16*kbits]; kbits == 0 -> the Linear stayed bf16 [M][K]."""
    kbits, k, m, codebook = struct.unpack(EXL3_RECORD_FMT, hdr)
    if (m, k) != slot.shape:
        raise ValueError(f"{slot.name}: EXL3 record says [{m}, {k}], the header's layout says {slot.shape}")
    if kbits == 0:
        return 16 + m * k * 2
    if not 1 <= kbits <= 8 or k % 16 or m % 16 or codebook > 2:
        raise ValueError(f"{slot.name}: bad EXL3 record header kbits={kbits} K={k} M={m} codebook={codebook}")
    return 16 + 2 * k + 2 * m + (k // 16) * (m // 16) * 16 * kbits * 2


def walk_payload(header: TqfHeader, f: BinaryIO | None = None) -> tuple[int, Counter[str], dict]:
    """Expected payload bytes after the header. Static for QMMA files; for EXL3 files (flag 0x800) the record headers
    are read from `f` (positioned anywhere; it seeks), because each Linear's bitrate sets its size. Returns
    (bytes, kind counts, details: EXL3 bitrate histogram and codebooks when applicable)."""
    block_scaled = bool(header.flags & FLAG_BLOCK_SCALED_QMMA)
    e2m3 = bool(header.flags & FLAG_BLOCK_SCALED_E2M3)
    exl3 = bool(header.flags & FLAG_EXL3_WEIGHTS)
    mtp_exl3 = bool(header.flags & FLAG_MTP_EXL3)
    counts: Counter[str] = Counter()
    details: dict = {"exl3_kbits": Counter(), "exl3_codebooks": Counter(), "exl3_bf16_linears": []}
    offset = header.header_bytes
    slots = list(payload_slots(header))
    mtp_part = list(mtp_slots(header)) if header.flags & FLAG_HAS_MTP else []
    for section, seq in (("main", slots), ("mtp", mtp_part)):
        for slot in seq:
            if slot.kind == "non_quant":
                size = non_quant_bytes(slot.count)
            elif slot.kind == "f32":
                size = f32_bytes(slot.count)
            elif slot.kind == "qm8":
                record_is_exl3 = exl3 if section == "main" else mtp_exl3
                if record_is_exl3:
                    if f is None:
                        raise ValueError("an EXL3 file needs the open file to size its records")
                    f.seek(offset)
                    hdr = read_exact(f, 16)
                    size = exl3_record_bytes(hdr, slot)
                    kbits, _, _, codebook = struct.unpack(EXL3_RECORD_FMT, hdr)
                    if kbits:
                        details["exl3_kbits"][kbits] += 1
                        details["exl3_codebooks"][("3inst", "mcg", "mul1")[codebook]] += 1
                    else:
                        details["exl3_bf16_linears"].append(slot.name)
                else:
                    size = qmma_bytes(slot.shape[0], slot.shape[1], block_scaled, e2m3)
            else:
                raise ValueError(slot.kind)
            counts[slot.kind if section == "main" else f"mtp_{slot.kind}"] += 1
            offset += size
    return offset - header.header_bytes, counts, details


# ------------------------------------------------------------------------------------------------ verdicts

def verdict_production_fp6(flags: int, payload_ok: bool) -> tuple[bool, str]:
    """The legacy single-stream production format: FP6 E2M3 block-scaled QMMA weights + the MTP head section
    (`--block-scaled always --block-layout qmma-e2m3 --block-scale-policy pow2` with the MTP section on)."""
    if flags & FLAG_EXL3_WEIGHTS:
        return False, "EXL3 trellis weights, not FP6 (this is a batched-server file; see the EXL3 + MTP verdict)"
    missing = []
    if not flags & FLAG_BLOCK_SCALED_QMMA:
        missing.append("no block scales: plain global-scale FP8 (an old `--block-scaled auto` default on a bf16 checkpoint)")
    elif not flags & FLAG_BLOCK_SCALED_E2M3:
        missing.append("block-scaled FP8 (E4M3), not E2M3 (--block-layout qmma-e2m3)")
    if not flags & FLAG_HAS_MTP:
        missing.append("no MTP head section (TQ_EMIT_MTP was off): no speculative decoding")
    if missing:
        return False, "; ".join(missing)
    if not payload_ok:
        return False, "payload size does not match the header (truncated or foreign file)"
    return True, "FP6 E2M3 block-scaled weights + MTP head section"


def verdict_exl3_mtp(flags: int, payload_ok: bool) -> tuple[bool, str]:
    """The served format: EXL3 trellis Linears + the checkpoint's EXL3 MTP layer (tools/serve_batched.py --spec)."""
    if not flags & FLAG_EXL3_WEIGHTS:
        return False, "not an EXL3 file (QMMA / FP6 weights; serve_batched.py runs it without the EXL3 kernels' formats)"
    if not flags & FLAG_HAS_MTP:
        return False, ("EXL3 weights but no MTP section: serve_batched.py runs with speculative decoding OFF "
                       "(reconvert with --mtp-exl3)")
    if not flags & FLAG_MTP_EXL3:
        return False, ("EXL3 weights with a bf16 MTP head (an older `-mtp.tqf`; the engine reads past it): the batched "
                       "server's paged drafter needs the EXL3 MTP records (reconvert with --mtp-exl3) -> --spec OFF")
    if not payload_ok:
        return False, "payload size does not match the header (truncated or foreign file)"
    return True, "EXL3 trellis weights + EXL3 MTP layer: tools/serve_batched.py --spec reads it"


def inspect(path: Path, verbose: bool = True, verify: bool = False) -> dict:
    with path.open("rb") as f:
        header = read_header(f)
        v2 = header.format_version >= 2
        if v2 and not header.header_hash_ok:          # the dimensions cannot be trusted: no payload walk
            expected_payload_bytes, counts, walk_error = -1, Counter(), "the header hash does not match the header"
            details = {"exl3_kbits": Counter(), "exl3_codebooks": Counter(), "exl3_bf16_linears": []}
        else:
            expected_payload_bytes, counts, details = walk_payload(header, f)
            walk_error = None
        hash_ok = (payload_hash(f, header.header_bytes) == header.payload_hash) if (verify and v2) else None
    actual_payload_bytes = os.path.getsize(path) - header.header_bytes
    sparse = bool(header.flags & FLAG_SPARSE_24_E2M3)
    unknown_flags = header.flags & ~KNOWN_FLAGS
    integrity_ok = not v2 or (header.format_version <= FORMAT_VERSION and header.header_hash_ok
                              and header.payload_bytes == actual_payload_bytes and hash_ok is not False and not unknown_flags)
    payload_ok = integrity_ok and (expected_payload_bytes == actual_payload_bytes
                                   or (sparse and expected_payload_bytes < actual_payload_bytes))
    fp6_ok, fp6_why = verdict_production_fp6(header.flags, payload_ok)
    exl3_ok, exl3_why = verdict_exl3_mtp(header.flags, payload_ok)
    if verbose:
        print("== TQF1 header ==")
        print(f"path: {path}")
        print(f"size: {os.path.getsize(path)}")
        print(f"header_bytes: {header.header_bytes}")
        print(f"format_version: {header.format_version}" + ("" if v2 else " (no header extension: no recorded size, no hashes)"))
        print(f"flags: 0x{header.flags:08x} ({', '.join(flag_names(header.flags)) or 'none'})")
        for bit in sorted(FLAG_NAMES):
            if bit >= FLAG_BLOCK_SCALED_QMMA:
                print(f"{FLAG_NAMES[bit]}: {bool(header.flags & bit)}")
        print(f"model_family: {header.model_family}")
        print(f"non_quant_dtype: {header.non_quant_dtype}")
        print(f"hidden_size: {header.hidden_size}")
        print(f"intermediate_size: {header.intermediate_size}")
        print(f"num_hidden_layers: {header.num_hidden_layers}")
        print(f"vocab_size: {header.vocab_size}")
        print(f"attention heads/kv/head_dim: {header.num_attention_heads}/{header.num_key_value_heads}/{header.head_dim}")
        print(
            "linear heads/dims: "
            f"k={header.linear_num_key_heads}x{header.linear_key_head_dim}, "
            f"v={header.linear_num_value_heads}x{header.linear_value_head_dim}, "
            f"conv={header.linear_conv_kernel_dim}"
        )
        print(f"rms_norm_eps: {header.rms_norm_eps}")
        print(f"rope_theta: {header.rope_theta}")
        print(f"partial_rotary_factor: {header.partial_rotary_factor}")
        print(f"mrope_interleaved: {header.mrope_interleaved}")
        print(f"mrope_section: {header.mrope_section}")
        print(f"tie_word_embeddings: {header.tie_word_embeddings}")
        print(f"text_only: {header.text_only}")
        print(f"has_vision_config: {header.has_vision_config}")
        print(f"has_mtp: {header.has_mtp}")
        print(f"layer_types: {header.layer_types}")
        print(f"layer_type_counts: {dict(Counter(header.layer_types))}")
        print(f"tensor_kind_counts: {dict(counts)}")
        if header.flags & FLAG_EXL3_WEIGHTS:
            kb = details["exl3_kbits"]
            n = sum(kb.values())
            hist = ", ".join(f"K={k}: {c} ({100 * c / n:.1f} %)" for k, c in sorted(kb.items())) if n else "none"
            print(f"exl3_bitrates (trellis Linears incl. MTP): {hist}")
            print(f"exl3_codebooks: {dict(details['exl3_codebooks'])}")
            print(f"exl3_bf16_linears (kept unquantized by the checkpoint): {len(details['exl3_bf16_linears'])}")
        print()
        print("== Payload validation ==")
        if walk_error:
            print(f"payload walk skipped: {walk_error}")
        print(f"expected_payload_bytes: {expected_payload_bytes}")
        print(f"actual_payload_bytes: {actual_payload_bytes}")
        print(f"payload_match: {expected_payload_bytes == actual_payload_bytes}")
        if sparse:
            # Per-tensor sparse 2:4 records depend on the calibration set, which is not
            # encoded in the header, so the static walk cannot size them. Don't fail.
            print("note: sparse_24_e2m3 set; per-tensor sparse records not accounted -> "
                  "expected < actual is normal")
        print()
        print("== Integrity ==")
        if unknown_flags:
            print(f"unknown flag bits: 0x{unknown_flags:x}" + (" (refused in a version-2 file)" if v2 else " (warning)"))
        if v2:
            if header.format_version > FORMAT_VERSION:
                print(f"format v{header.format_version} is newer than this tool (v{FORMAT_VERSION})")
            print(f"payload_bytes (recorded): {header.payload_bytes} -- {'match' if header.payload_bytes == actual_payload_bytes else 'MISMATCH: truncated or extended file'}")
            print(f"header hash: {'OK' if header.header_hash_ok else 'MISMATCH'}")
            print(f"payload hash (BLAKE2b-256 {header.payload_hash.hex()[:16]}...): "
                  + ({True: 'OK', False: 'MISMATCH', None: 'not checked (--verify)'}[hash_ok]))
        else:
            print("version-1 file: only the slot walk above checks it (re-convert for a recorded size and hashes)")
        print()
        print("== Verdicts ==")
        print(f"verdict production FP6 (flags 0x1|0x20|0x100|0x400 + payload): {'PASS' if fp6_ok else 'FAIL'} -- {fp6_why}")
        print(f"verdict EXL3 + MTP (flags 0x800|0x400|0x1000 + payload):     {'PASS' if exl3_ok else 'FAIL'} -- {exl3_why}")
    return {"header": header, "expected_payload_bytes": expected_payload_bytes, "actual_payload_bytes": actual_payload_bytes,
            "payload_ok": payload_ok, "integrity_ok": integrity_ok, "payload_hash_ok": hash_ok,
            "counts": counts, "details": details,
            "production_fp6": fp6_ok, "exl3_mtp": exl3_ok}


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and validate a .tqf file (header, payload, format verdicts)")
    parser.add_argument("path", help="the .tqf file")
    parser.add_argument("--expect", choices=("fp6", "exl3-mtp"), default=None,
                        help="exit 2 unless this verdict is PASS (fp6 = production FP6 + MTP; exl3-mtp = EXL3 + EXL3 MTP)")
    parser.add_argument("--verify", action="store_true",
                        help="also hash the payload and compare it with the recorded BLAKE2b-256 (version-2 files; a full read)")
    args = parser.parse_args()
    r = inspect(Path(args.path), verify=args.verify)
    if not r["payload_ok"]:
        sys.exit(1)
    if args.expect == "fp6" and not r["production_fp6"]:
        sys.exit(2)
    if args.expect == "exl3-mtp" and not r["exl3_mtp"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
