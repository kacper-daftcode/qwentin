#!/usr/bin/env python3
"""Inspect and validate the experimental Qwen TQF1 binary format."""

from __future__ import annotations

import argparse
import os
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


DEFAULT_TQF = "/workspace/models/Qwen3.5-0.8B/qwen3_5-0_8b-text-fp8.tqf"
FIXED_HEADER_FMT = "<16I3f8I"
FIXED_HEADER_SIZE = struct.calcsize(FIXED_HEADER_FMT)
FLAG_BLOCK_SCALED_QMMA = 1 << 5
FLAG_BLOCK_SCALED_ROWMAJOR = 1 << 6
FLAG_BLOCK_SCALED_QMMA_WORDMAJOR = 1 << 7
FLAG_BLOCK_SCALED_E2M3 = 1 << 8
FLAG_SPARSE_24_E2M3 = 1 << 9
FLAG_HAS_MTP = 1 << 10


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

    values = struct.unpack(FIXED_HEADER_FMT, read_exact(f, FIXED_HEADER_SIZE))
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
    layer_type_bytes = read_exact(f, header_bytes - 8 - FIXED_HEADER_SIZE)
    layer_types = list(layer_type_bytes)
    if len(layer_types) != num_hidden_layers:
        raise ValueError(
            f"layer type count mismatch: header has {num_hidden_layers}, bytes have {len(layer_types)}"
        )
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
    )


def qmma_bytes(m: int, k: int, block_scaled: bool = False, e2m3: bool = False) -> int:
    if m % 16 or k % 32:
        raise ValueError(f"unaligned QMMA shape {m}x{k}")
    tiled = (m // 16) * (k // 32) * (384 if e2m3 else 512)
    if block_scaled:
        scale_rows = (m + 127) // 128
        scale_cols = (k + 127) // 128
        return 8 + scale_rows * scale_cols * 4 + tiled
    return 4 + tiled


def mtp_bytes(header: "TqfHeader", block_scaled: bool, e2m3: bool) -> tuple[int, Counter[str]]:
    """Bytes for the appended MTP head section (one full_attention decoder layer +
    fc + pre_fc/post norms). Mirrors mtp_tensor_specs in convert_qwen_tqf.py."""
    h = header.hidden_size
    i = header.intermediate_size
    heads = header.num_attention_heads
    kv_heads = header.num_key_value_heads
    hd = header.head_dim
    # All 8 MTP projections (fc + q/k/v/o + gate/up/down) are FULL-PRECISION BF16
    # (non_quant), like the norms. (block_scaled/e2m3 are unused for the MTP section.)
    total = 0
    counts: Counter[str] = Counter()
    total += non_quant_bytes(h) * 2  # pre_fc_norm_embedding, pre_fc_norm_hidden
    total += non_quant_bytes(h * 2 * h)  # fc [H, 2H]
    total += non_quant_bytes(h) * 2  # input_layernorm, post_attention_layernorm
    total += non_quant_bytes(hd) * 2  # q_norm, k_norm
    total += non_quant_bytes(heads * hd * 2 * h)  # q_proj
    total += non_quant_bytes(kv_heads * hd * h) * 2  # k_proj, v_proj
    total += non_quant_bytes(h * heads * hd)  # o_proj
    total += non_quant_bytes(i * h) * 2  # gate, up
    total += non_quant_bytes(h * i)  # down
    total += non_quant_bytes(h)  # mtp.norm
    counts.update(non_quant=15)
    return total, counts


def non_quant_bytes(count: int) -> int:
    return count * 2


def f32_bytes(count: int) -> int:
    return count * 4


def walk_payload(header: TqfHeader) -> tuple[int, Counter[str]]:
    h = header.hidden_size
    i = header.intermediate_size
    v = header.vocab_size
    heads = header.num_attention_heads
    kv_heads = header.num_key_value_heads
    hd = header.head_dim
    lin_k_heads = header.linear_num_key_heads
    lin_k_dim = header.linear_key_head_dim
    lin_v_heads = header.linear_num_value_heads
    lin_v_dim = header.linear_value_head_dim
    conv_kernel = header.linear_conv_kernel_dim
    lin_key_dim = lin_k_heads * lin_k_dim
    lin_value_dim = lin_v_heads * lin_v_dim
    lin_conv_dim = lin_key_dim * 2 + lin_value_dim
    block_scaled = (header.flags & FLAG_BLOCK_SCALED_QMMA) != 0
    e2m3 = (header.flags & FLAG_BLOCK_SCALED_E2M3) != 0

    total = non_quant_bytes(v * h) + non_quant_bytes(h)
    counts: Counter[str] = Counter(non_quant=2)
    if not header.tie_word_embeddings:
        total += qmma_bytes(v, h, block_scaled, e2m3)
        counts["qm8"] += 1

    for layer_type in header.layer_types:
        total += non_quant_bytes(h) * 2
        total += qmma_bytes(i, h, block_scaled, e2m3) * 2
        total += qmma_bytes(h, i, block_scaled, e2m3)
        counts.update(non_quant=2, qm8=3)
        if layer_type == 1:
            total += f32_bytes(lin_v_heads)
            total += non_quant_bytes(lin_v_heads)
            total += f32_bytes(lin_v_dim)
            total += non_quant_bytes(lin_conv_dim * conv_kernel)
            total += qmma_bytes(lin_v_heads, h, block_scaled, e2m3) * 2
            total += qmma_bytes(lin_conv_dim, h, block_scaled, e2m3)
            total += qmma_bytes(lin_value_dim, h, block_scaled, e2m3)
            total += qmma_bytes(h, lin_value_dim, block_scaled, e2m3)
            counts.update(f32=2, non_quant=2, qm8=5)
        elif layer_type == 2:
            total += non_quant_bytes(hd) * 2
            total += qmma_bytes(heads * hd * 2, h, block_scaled, e2m3)
            total += qmma_bytes(kv_heads * hd, h, block_scaled, e2m3) * 2
            total += qmma_bytes(h, heads * hd, block_scaled, e2m3)
            counts.update(non_quant=2, qm8=4)
        else:
            raise ValueError(f"unsupported layer type byte: {layer_type}")

    if header.flags & FLAG_HAS_MTP:
        mtp_total, mtp_counts = mtp_bytes(header, block_scaled, e2m3)
        total += mtp_total
        counts.update(dict(mtp_counts))
    return total, counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a TQF1 file")
    parser.add_argument("path", nargs="?", default=DEFAULT_TQF)
    args = parser.parse_args()

    path = Path(args.path)
    with path.open("rb") as f:
        header = read_header(f)
        expected_payload_bytes, counts = walk_payload(header)
        actual_payload_bytes = os.path.getsize(path) - header.header_bytes

    print("== TQF1 header ==")
    print(f"path: {path}")
    print(f"size: {os.path.getsize(path)}")
    print(f"header_bytes: {header.header_bytes}")
    print(f"flags: 0x{header.flags:08x}")
    print(f"block_scaled_qmma: {bool(header.flags & FLAG_BLOCK_SCALED_QMMA)}")
    print(f"block_scaled_rowmajor: {bool(header.flags & FLAG_BLOCK_SCALED_ROWMAJOR)}")
    print(f"block_scaled_qmma_wordmajor: {bool(header.flags & FLAG_BLOCK_SCALED_QMMA_WORDMAJOR)}")
    print(f"block_scaled_e2m3: {bool(header.flags & FLAG_BLOCK_SCALED_E2M3)}")
    print(f"sparse_24_e2m3: {bool(header.flags & FLAG_SPARSE_24_E2M3)}")
    print(f"has_mtp_section: {bool(header.flags & FLAG_HAS_MTP)}")
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
    print()

    print("== Payload validation ==")
    print(f"expected_payload_bytes: {expected_payload_bytes}")
    print(f"actual_payload_bytes: {actual_payload_bytes}")
    print(f"payload_match: {expected_payload_bytes == actual_payload_bytes}")
    if header.flags & FLAG_SPARSE_24_E2M3:
        # Per-tensor sparse 2:4 records depend on the calibration set, which is not
        # encoded in the header, so the static walk cannot size them. Don't fail.
        print("note: sparse_24_e2m3 set; per-tensor sparse records not accounted -> "
              "expected < actual is normal")
    elif expected_payload_bytes != actual_payload_bytes:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
