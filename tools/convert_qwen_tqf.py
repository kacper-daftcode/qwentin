#!/usr/bin/env python3
"""Convert a Qwen3.5/Qwen3.6 text tower to the experimental TQF1 format."""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.convert import (  # noqa: E402
    float_to_e4m3,
    float_to_e2m3,
    repack_to_qmma_layout_f8,
    repack_to_qmma_word_layout_f8,
    repack_to_qmma_layout_f6_dense,
)
from tools.sparse_pack import wanda_2of4_mask, pack_2of4_e2m3_codes  # noqa: E402


DEFAULT_MODEL = "/workspace/models/Qwen3.5-0.8B"
MODEL_FAMILY_QWEN3_5 = 1
DTYPE_FP16 = 1
DTYPE_BF16 = 2
LAYER_LINEAR_ATTENTION = 1
LAYER_FULL_ATTENTION = 2

FLAG_FP8_QMMA_WEIGHTS = 1 << 0
FLAG_TIED_EMBEDDINGS = 1 << 1
FLAG_SOURCE_BF16_NON_QUANT = 1 << 2
FLAG_DROPPED_VISION = 1 << 3
FLAG_DROPPED_MTP = 1 << 4
FLAG_BLOCK_SCALED_QMMA = 1 << 5
FLAG_BLOCK_SCALED_ROWMAJOR = 1 << 6
FLAG_BLOCK_SCALED_QMMA_WORDMAJOR = 1 << 7
FLAG_BLOCK_SCALED_E2M3 = 1 << 8  # weights E2M3 (6-bit dense), activations stay E4M3
FLAG_SPARSE_24_E2M3 = 1 << 9  # + tile-level WANDA 2:4 draft record after each linear_attn proj
FLAG_HAS_MTP = 1 << 10  # native MTP head section appended after the payload (env TQ_EMIT_MTP=1)

COMMON_SUFFIXES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)
FULL_ATTENTION_SUFFIXES = (
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
)
LINEAR_ATTENTION_SUFFIXES = (
    "linear_attn.A_log",
    "linear_attn.dt_bias",
    "linear_attn.norm.weight",
    "linear_attn.conv1d.weight",
    "linear_attn.in_proj_a.weight",
    "linear_attn.in_proj_b.weight",
    "linear_attn.in_proj_qkv.weight",
    "linear_attn.in_proj_z.weight",
    "linear_attn.out_proj.weight",
)


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    kind: str


class TensorStore:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.weight_map = self._load_weight_map()
        self.shards = sorted(set(self.weight_map.values()))
        self.stack = ExitStack()
        self.handles = {}

    def _load_weight_map(self) -> dict[str, str]:
        index_path = self.model_dir / "model.safetensors.index.json"
        if index_path.exists():
            with index_path.open("r", encoding="utf-8") as f:
                index = json.load(f)
            return dict(index["weight_map"])

        weight_map = {}
        for shard in sorted(self.model_dir.glob("*.safetensors")):
            with safe_open(shard, framework="pt", device="cpu") as f:
                for key in f.keys():
                    weight_map[key] = shard.name
        if not weight_map:
            raise FileNotFoundError(f"no safetensors weights found in {self.model_dir}")
        return weight_map

    def __enter__(self):
        for shard in self.shards:
            shard_path = self.model_dir / shard
            if not shard_path.exists():
                raise FileNotFoundError(f"missing safetensors shard: {shard_path}")
            self.handles[shard] = self.stack.enter_context(
                safe_open(shard_path, framework="pt", device="cpu")
            )
        return self

    def __exit__(self, exc_type, exc, tb):
        return self.stack.__exit__(exc_type, exc, tb)

    def names(self) -> set[str]:
        return set(self.weight_map)

    def get(self, name: str) -> torch.Tensor:
        try:
            shard = self.weight_map[name]
        except KeyError as exc:
            raise KeyError(f"missing tensor: {name}") from exc
        return self.handles[shard].get_tensor(name)


def as_text_config(config):
    return getattr(config, "text_config", config)


def u16_bytes(tensor: torch.Tensor, dtype_id: int) -> bytes:
    if dtype_id == DTYPE_BF16:
        arr = tensor.detach().cpu().to(torch.bfloat16).contiguous().view(torch.uint16).numpy()
    elif dtype_id == DTYPE_FP16:
        arr = tensor.detach().cpu().to(torch.float16).contiguous().view(torch.uint16).numpy()
    else:
        raise ValueError(f"unsupported non-quant dtype id: {dtype_id}")
    return arr.astype("<u2", copy=False).tobytes()


def f32_bytes(tensor: torch.Tensor) -> bytes:
    arr = tensor.detach().cpu().to(torch.float32).contiguous().numpy()
    return arr.astype("<f4", copy=False).tobytes()


def quantize_qmma_f8(weight: torch.Tensor) -> tuple[float, bytes]:
    w = weight.detach().cpu().float().contiguous().numpy()
    if w.ndim != 2:
        raise ValueError(f"expected 2D weight, got shape {w.shape}")
    m, k = w.shape
    if m % 16 != 0 or k % 32 != 0:
        raise ValueError(f"weight shape must be QMMA aligned, got {m}x{k}")

    absmax = float(np.max(np.abs(w)))
    scale = 448.0 / absmax if absmax > 0 else 1.0
    quantized = float_to_e4m3(w * scale)
    return scale, repack_to_qmma_layout_f8(quantized, m, k)


def e4m3_bytes_from_tensor(tensor: torch.Tensor) -> np.ndarray:
    """Return raw E4M3 bytes for a float8 tensor, with a float fallback."""
    t = tensor.detach().cpu().contiguous()
    if str(t.dtype).startswith("torch.float8"):
        return t.view(torch.uint8).numpy().astype(np.uint8, copy=False)
    return float_to_e4m3(t.float().numpy().astype(np.float32, copy=False))


def block_shape(m: int, k: int) -> tuple[int, int]:
    return (m + 127) // 128, (k + 127) // 128


def block_quantize_qmma_f8(weight: torch.Tensor, scale_policy: str = "float") -> tuple[np.ndarray, np.ndarray]:
    w = weight.detach().cpu().float().contiguous().numpy()
    if w.ndim != 2:
        raise ValueError(f"expected 2D weight, got shape {w.shape}")
    m, k = w.shape
    if m % 16 != 0 or k % 32 != 0:
        raise ValueError(f"weight shape must be QMMA aligned, got {m}x{k}")
    scale_rows, scale_cols = block_shape(m, k)
    scale_inv = np.empty((scale_rows, scale_cols), dtype=np.float32)
    quantized = np.empty((m, k), dtype=np.uint8)
    for br in range(scale_rows):
        r0 = br * 128
        r1 = min(r0 + 128, m)
        for bc in range(scale_cols):
            c0 = bc * 128
            c1 = min(c0 + 128, k)
            block = w[r0:r1, c0:c1]
            absmax = float(np.max(np.abs(block)))
            if absmax > 0:
                raw_scale_inv = absmax / 448.0
                if scale_policy == "pow2":
                    block_scale_inv = float(2.0 ** np.ceil(np.log2(raw_scale_inv)))
                elif scale_policy == "float":
                    block_scale_inv = raw_scale_inv
                else:
                    raise ValueError(f"unsupported scale policy: {scale_policy}")
            else:
                block_scale_inv = 1.0
            scale = 1.0 / block_scale_inv
            scale_inv[br, bc] = block_scale_inv
            quantized[r0:r1, c0:c1] = float_to_e4m3(block * scale)
    return scale_inv, quantized


_GPU_LIB = None


def _gpu_lib():
    """Load libforward_qwen.so for GPU-side BF16->E2M3 quant+pack (TQ_GPU_PACK=1)."""
    global _GPU_LIB
    if _GPU_LIB is None:
        import ctypes
        path = os.environ.get(
            "TQ_FORWARD_LIB",
            str(Path(__file__).resolve().parents[1] / "build-qwen" / "libforward_qwen.so"),
        )
        lib = ctypes.CDLL(path)
        lib.qwn_gpu_pack_e2m3_test.restype = ctypes.c_int
        lib.qwn_gpu_pack_e2m3_test.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
        ]
        _GPU_LIB = lib
    return _GPU_LIB


def gpu_pack_e2m3(tensor: torch.Tensor) -> tuple[np.ndarray, bytes]:
    """Quantize+pack a BF16 weight to dense E2M3 on the GPU (pow2 ue8m0 scale).

    Returns (scale_inv[sr,sc], dense payload bytes). Orders of magnitude faster
    than the numpy path because the GPU does the per-element quant and packing.
    """
    import ctypes
    m, k = tuple(tensor.shape)
    t = tensor.detach().cpu().contiguous().to(torch.bfloat16)
    bf16 = t.view(torch.uint16).numpy()
    mt, kt = m // 16, k // 32
    sr, sc = (m + 127) // 128, (k + 127) // 128
    payload = np.empty(mt * kt * 384, dtype=np.uint8)
    scale = np.empty(sr * sc, dtype=np.float32)
    lib = _gpu_lib()
    ret = lib.qwn_gpu_pack_e2m3_test(
        bf16.ctypes.data_as(ctypes.c_void_p), m, k,
        payload.ctypes.data_as(ctypes.c_void_p), scale.ctypes.data_as(ctypes.c_void_p),
    )
    if ret != 0:
        raise RuntimeError(f"qwn_gpu_pack_e2m3_test failed ({ret}) for {m}x{k}")
    return scale.reshape(sr, sc), payload.tobytes()


def block_quantize_qmma_e2m3(weight: torch.Tensor, scale_policy: str = "pow2") -> tuple[np.ndarray, np.ndarray]:
    """128x128 block-scaled quantization to E2M3 (FP6) codes.

    Same structure as the FP8 path but the per-block scale targets the E2M3
    range (max 7.5) instead of E4M3 (448). Returns (scale_inv, uint8 codes in
    [0,63]); the codes carry the same 3 mantissa bits as E4M3, so per-element
    precision matches FP8 while the block scale recovers the dynamic range.
    """
    w = weight.detach().cpu().float().contiguous().numpy()
    if w.ndim != 2:
        raise ValueError(f"expected 2D weight, got shape {w.shape}")
    m, k = w.shape
    if m % 16 != 0 or k % 32 != 0:
        raise ValueError(f"weight shape must be QMMA aligned, got {m}x{k}")
    scale_rows, scale_cols = block_shape(m, k)
    if scale_policy not in ("pow2", "float"):
        raise ValueError(f"unsupported scale policy: {scale_policy}")
    if m % 128 == 0 and k % 128 == 0:
        # Vectorised fast path: reshape into 128x128 blocks and quantise all at
        # once (the per-block Python loop is ~1M iterations on a 27B model).
        wb = w.reshape(scale_rows, 128, scale_cols, 128).transpose(0, 2, 1, 3)  # (sr,sc,128,128)
        absmax = np.abs(wb).reshape(scale_rows, scale_cols, -1).max(axis=2)     # (sr,sc)
        raw = absmax / 7.5
        if scale_policy == "pow2":
            block_scale_inv = np.where(absmax > 0, 2.0 ** np.ceil(np.log2(np.maximum(raw, 1e-30))), 1.0)
        else:
            block_scale_inv = np.where(absmax > 0, raw, 1.0)
        block_scale_inv = block_scale_inv.astype(np.float32)
        scaled = wb * (1.0 / block_scale_inv)[:, :, None, None]
        codes = float_to_e2m3(scaled).reshape(scale_rows, scale_cols, 128, 128)
        quantized = codes.transpose(0, 2, 1, 3).reshape(m, k).astype(np.uint8)
        return block_scale_inv, quantized

    scale_inv = np.empty((scale_rows, scale_cols), dtype=np.float32)
    quantized = np.empty((m, k), dtype=np.uint8)
    for br in range(scale_rows):
        r0, r1 = br * 128, min(br * 128 + 128, m)
        for bc in range(scale_cols):
            c0, c1 = bc * 128, min(bc * 128 + 128, k)
            block = w[r0:r1, c0:c1]
            absmax = float(np.max(np.abs(block)))
            if absmax > 0:
                raw_scale_inv = absmax / 7.5
                block_scale_inv = float(2.0 ** np.ceil(np.log2(raw_scale_inv))) if scale_policy == "pow2" else raw_scale_inv
            else:
                block_scale_inv = 1.0
            scale_inv[br, bc] = block_scale_inv
            quantized[r0:r1, c0:c1] = float_to_e2m3(block * (1.0 / block_scale_inv))
    return scale_inv, quantized


# ===== FP4 (E2M1) pre-quantization for linear_attention projections =====
# E2M1 magnitudes are {0, .5, 1, 1.5, 2, 3, 4, 6}. Every one of these is also an
# E2M3 grid point, so storing FP4-prequantized weights in the existing E2M3 (FP6)
# payload is bit-exact when the block scale is a power of two. That lets us measure
# true FP4 quality on the real engine without any kernel change; the 4-bit DRAM
# pack (the bandwidth win) is a separate step.
_E2M1_GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64)
_E2M1_THRESH = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float64)


def _quantize_e2m1_magnitude(a: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(_E2M1_THRESH, a, side="right")
    return _E2M1_GRID[idx]


def prequant_fp4_e2m1(weight: torch.Tensor, block: int = 16) -> torch.Tensor:
    """FP4 (E2M1) prequant with per-`block` (along K) pow2 block scales.

    Returns a dequantized float tensor; the per-element values land on the E2M1
    grid scaled by a power of two, so the downstream E2M3 pack stores them exactly.
    """
    w = weight.detach().cpu().float().contiguous().numpy()
    if w.ndim != 2:
        raise ValueError(f"prequant_fp4_e2m1 expects 2D weight, got {w.shape}")
    m, k = w.shape
    nb = (k + block - 1) // block
    pad = nb * block - k
    if pad:
        w = np.pad(w, ((0, 0), (0, pad)))
    wb = w.reshape(m, nb, block)
    absmax = np.abs(wb).max(axis=2, keepdims=True)
    raw = absmax / 6.0
    scale = np.where(absmax > 0, 2.0 ** np.ceil(np.log2(np.maximum(raw, 1e-30))), 1.0)
    sign = np.sign(wb)
    q = _quantize_e2m1_magnitude(np.abs(wb) / scale) * sign * scale
    return torch.from_numpy(q.reshape(m, nb * block)[:, :k].astype(np.float32))


def maybe_prequant_fp4(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """Apply FP4 prequant to linear_attention projection weights when enabled.

    Controlled by TQ_FP4_LINEAR (on/off), TQ_FP4_BLOCK (block size, default 16),
    and TQ_FP4_SKIP (comma-separated proj substrings to keep at full precision,
    default the small/sensitive in_proj_a,in_proj_b gates).
    """
    if not os.environ.get("TQ_FP4_LINEAR"):
        return tensor
    if ".linear_attn." not in name or not name.endswith(".weight") or tensor.ndim != 2:
        return tensor
    skip = os.environ.get("TQ_FP4_SKIP", "in_proj_a,in_proj_b")
    if any(s and s in name for s in skip.split(",")):
        return tensor
    block = int(os.environ.get("TQ_FP4_BLOCK", "16"))
    return prequant_fp4_e2m1(tensor, block)


# ===== Optimistic sparse-FP4 draft: tile-level WANDA 2:4 record =====
# When TQ_SPARSE_24=1 (qmma-e2m3 only), each linear_attn projection gets an extra
# record appended after its dense E2M3 record: the tile-level WANDA 2:4 draft
# (compressed-A in group_perm order + ordered-metadata). The kept values are the
# SAME E2M3 codes as the dense payload (a true 50% sub-read), so verify reuses the
# dense path. Single-copy [kept|dropped] reordering is a later 27B-footprint step.
_SPARSE_TARGET_SUFFIXES = (
    "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
)
_WANDA_ACT_CACHE: dict | None = None


def sparse_24_enabled() -> bool:
    return bool(os.environ.get("TQ_SPARSE_24"))


def _load_wanda_act() -> dict:
    global _WANDA_ACT_CACHE
    if _WANDA_ACT_CACHE is None:
        path = os.environ.get("TQ_WANDA_ACT")
        if not path:
            raise ValueError("TQ_SPARSE_24=1 requires TQ_WANDA_ACT=<calibration .npz>")
        z = np.load(path)
        out = {}
        for key in z.files:
            mm = re.search(r"layers\.(\d+)\.(.+)", key)
            if mm:
                out[(int(mm.group(1)), mm.group(2))] = z[key].astype(np.float32)
        _WANDA_ACT_CACHE = out
    return _WANDA_ACT_CACHE


def _sparse_key(name: str):
    if not name.endswith(".weight"):
        return None
    base = name[: -len(".weight")]
    if not any(base.endswith(suf) for suf in _SPARSE_TARGET_SUFFIXES):
        return None
    mm = re.search(r"layers\.(\d+)\.(.+)", base)
    return (int(mm.group(1)), mm.group(2)) if mm else None


def sparse_24_record_size(spec: "TensorSpec") -> int:
    m, k = spec.shape
    n_mt, kt64 = m // 16, k // 64
    return 8 + (n_mt * kt64 * 96) * 4 + (n_mt * kt64 * 32) * 4


def write_sparse_24_record(f, orig_tensor, spec, scale_policy, dry_run) -> int:
    """Append the tile-level WANDA 2:4 draft record for a calibrated projection.
    Only projections present in the calibration npz get a record (this restricts
    to the linear-attention layers' FP4 projections); others stay dense."""
    key = _sparse_key(spec.name)
    if key is None:
        return 0
    act = _load_wanda_act()
    if key not in act:
        return 0
    m, k = spec.shape
    if m % 16 != 0 or k % 64 != 0:
        raise ValueError(f"{spec.name}: sparse 2:4 needs M%16==0,K%64==0; got {m}x{k}")
    if act[key].shape[0] != k:
        raise ValueError(f"{spec.name}: act_norm len {act[key].shape[0]} != K {k}")
    size = sparse_24_record_size(spec)
    if dry_run:
        return size
    w_orig = orig_tensor.detach().cpu().float().numpy()
    t = maybe_prequant_fp4(orig_tensor, spec.name)
    policy = "float" if scale_policy == "float" else "pow2"
    _, codes = block_quantize_qmma_e2m3(t, policy)
    mask = wanda_2of4_mask(w_orig, act[key])
    a_u32, meta_u32 = pack_2of4_e2m3_codes(codes.astype(np.uint8), mask)
    f.write(struct.pack("<II", int(a_u32.size), int(meta_u32.size)))
    f.write(a_u32.astype("<u4", copy=False).tobytes())
    f.write(meta_u32.astype("<u4", copy=False).tobytes())
    print(f"    +sparse2:4 {spec.name:64s} a_u32={a_u32.size} meta_u32={meta_u32.size}")
    return size


# ===== Verify-path 2:4 mask (sparse-as-dense pre-gate) =====
# TQ_VMASK_ACT=<npz of per-matrix ||x||_2> + TQ_VMASK_INCLUDE=<file with one
# "layers.N.suffix" per line>: zero the WANDA-2:4-pruned elements (along K, top-2
# of |W|*||x|| per 4 consecutive inputs -- mma.sp decoder semantics) of the listed
# matrices BEFORE the standard dense E2M3 quantization. Output stays a regular
# dense TQF (no sparse records): the engine streams it unchanged, so an A/B vs
# the dense baseline measures PURE sparsification quality at identical speed.
_VMASK_CACHE: tuple | None = None


def _vmask_ctx() -> tuple[dict, set]:
    global _VMASK_CACHE
    if _VMASK_CACHE is None:
        act_path = os.environ.get("TQ_VMASK_ACT")
        inc_path = os.environ.get("TQ_VMASK_INCLUDE")
        if not act_path or not inc_path:
            _VMASK_CACHE = ({}, set())
        else:
            z = np.load(act_path)
            acts = {}
            for key in z.files:
                mm = re.search(r"layers\.(\d+)\.(.+)", key)
                if mm:
                    acts[(int(mm.group(1)), mm.group(2))] = z[key].astype(np.float32)
            inc = set()
            for line in open(inc_path):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                mm = re.search(r"layers\.(\d+)\.(.+?)(\.weight)?$", line)
                if mm:
                    inc.add((int(mm.group(1)), mm.group(2)))
            _VMASK_CACHE = (acts, inc)
    return _VMASK_CACHE


def maybe_verify_mask_24(tensor: torch.Tensor, name: str) -> torch.Tensor:
    acts, inc = _vmask_ctx()
    if not inc or tensor.ndim != 2 or not name.startswith("model."):
        return tensor
    mm = re.search(r"layers\.(\d+)\.(.+)\.weight$", name)
    if not mm:
        return tensor
    key = (int(mm.group(1)), mm.group(2))
    if key not in inc:
        return tensor
    if key not in acts:
        raise KeyError(f"TQ_VMASK_INCLUDE lists {name} but TQ_VMASK_ACT lacks it")
    w = tensor.detach().cpu().float().numpy()
    m, k = w.shape
    an = acts[key]
    if k % 4 != 0 or an.shape[0] != k:
        raise ValueError(f"{name}: bad K={k} vs act_norm {an.shape}")
    imp = (np.abs(w) * an[None, :]).reshape(m, k // 4, 4)
    order = np.argsort(-imp, axis=2)  # same selection as sparse_pack.wanda_2of4_mask
    keep = np.zeros(imp.shape, dtype=bool)
    np.put_along_axis(keep, order[:, :, :2], True, axis=2)
    pruned = float((imp * ~keep).sum() / max(float(imp.sum()), 1e-30))
    wm = (w.reshape(m, k // 4, 4) * keep).reshape(m, k)
    print(f"    vmask2:4 {name:64s} pruned_energy={pruned:.4f}")
    return torch.from_numpy(wm).to(tensor.dtype)


# ===== Precomputed-codes side-load (GPTQ/SparseGPT compensation) =====
# TQ_QCODES_DIR=<dir of layers.N.<suffix>.weight.npz {scale_inv, codes}> (from
# tools/ql_gptq_sweep.py): for matrices with a codes file, write the precomputed
# E2M3 codes + frozen scales VERBATIM (no re-quantization -- re-deriving scales
# from compensated weights would shift the grid and corrupt the swept codes).
# Matrices without a file (lm_head, MTP) take the standard path unchanged.
def maybe_load_qcodes(name: str, m: int, k: int):
    qdir = os.environ.get("TQ_QCODES_DIR")
    if not qdir or not name.startswith("model."):
        return None
    mm = re.search(r"layers\.(\d+)\.(.+)$", name)
    if not mm:
        return None
    path = Path(qdir) / f"layers.{mm.group(1)}.{mm.group(2)}.npz"
    if not path.exists():
        return None
    z = np.load(path)
    scale_inv, codes = z["scale_inv"], z["codes"]
    k32_shape = ((m + 127) // 128, (k + 31) // 32)   # MX-style per-k32 weight scales
    if codes.shape != (m, k) or scale_inv.shape not in (block_shape(m, k), k32_shape):
        raise ValueError(f"{name}: qcodes shape mismatch {codes.shape} vs {(m, k)}")
    return scale_inv, codes


def deblock_fp8_weight(weight: torch.Tensor, scale_inv: torch.Tensor, name: str) -> torch.Tensor:
    """Convert official block-scaled FP8 weights to float before TQF packing.

    Qwen3.6 FP8 checkpoints store 128x128 block scale inverses as
    `<name>_scale_inv`. TQF1 currently stores one global scale per tensor, so
    the converter first reconstructs a float matrix, then repacks it.
    """
    if weight.ndim != 2 or scale_inv.ndim != 2:
        raise ValueError(f"{name}: expected 2D FP8 weight and 2D scale_inv")
    m, k = tuple(weight.shape)
    sm, sk = tuple(scale_inv.shape)
    if sm != (m + 127) // 128 or sk != (k + 127) // 128:
        raise ValueError(f"{name}: scale_inv shape {tuple(scale_inv.shape)} incompatible with weight {m}x{k}")
    w = weight.detach().cpu().float().contiguous()
    scale = scale_inv.detach().cpu().float().contiguous()
    expanded = torch.repeat_interleave(torch.repeat_interleave(scale, 128, dim=0), 128, dim=1)
    return w * expanded[:m, :k]


def write_non_quant(f, tensor: torch.Tensor, spec: TensorSpec, dtype_id: int, dry_run: bool) -> int:
    validate_shape(tensor, spec)
    size = int(np.prod(spec.shape)) * 2
    if dry_run:
        return size
    f.write(u16_bytes(tensor, dtype_id))
    return size


def write_f32(f, tensor: torch.Tensor, spec: TensorSpec, dry_run: bool) -> int:
    validate_shape(tensor, spec)
    size = int(np.prod(spec.shape)) * 4
    if dry_run:
        return size
    f.write(f32_bytes(tensor))
    return size


def write_qmma_f8(f, tensor: torch.Tensor, spec: TensorSpec, dry_run: bool) -> int:
    validate_shape(tensor, spec)
    m, k = spec.shape
    payload_size = (m // 16) * (k // 32) * 512
    size = 4 + payload_size
    if dry_run:
        return size
    scale, tiled = quantize_qmma_f8(tensor)
    if len(tiled) != payload_size:
        raise ValueError(f"{spec.name}: expected {payload_size} tiled bytes, got {len(tiled)}")
    f.write(struct.pack("<f", scale))
    f.write(tiled)
    print(f"  {spec.name:72s} {spec.shape!s:18s} scale={scale:.6g}")
    return size


def write_qmma_f8_block_scaled(
    f,
    tensor: torch.Tensor,
    spec: TensorSpec,
    dry_run: bool,
    scale_inv: torch.Tensor | None = None,
    scale_policy: str = "source",
    block_layout: str = "rowmajor",
) -> int:
    validate_shape(tensor, spec)
    tensor = maybe_prequant_fp4(tensor, spec.name)
    m, k = spec.shape
    scale_rows, scale_cols = block_shape(m, k)
    is_e2m3 = block_layout == "qmma-e2m3"
    qcodes = maybe_load_qcodes(spec.name, m, k) if is_e2m3 else None
    if qcodes is not None:
        # qcodes may carry per-k32 weight scales (MX-style E2M1 rung); the TQF
        # record stores (sr, sc) explicitly and the engine keys indexing off it
        scale_rows, scale_cols = qcodes[0].shape
    payload_size = (m // 16) * (k // 32) * (384 if is_e2m3 else 512)
    scale_size = scale_rows * scale_cols * 4
    size = 8 + scale_size + payload_size
    if dry_run:
        return size

    if is_e2m3:
        # E2M3 weights must be (re)quantized from float against the FP6 range;
        # source FP8 scale_inv (448 target) does not apply.
        if qcodes is not None:
            scale_arr, quantized = qcodes
            scale_arr = scale_arr.astype("<f4", copy=False)
            payload = repack_to_qmma_layout_f6_dense(quantized, m, k)
            print(f"    +qcodes {spec.name}")
        elif os.environ.get("TQ_GPU_PACK"):
            scale_arr, payload = gpu_pack_e2m3(tensor)
            scale_arr = scale_arr.astype("<f4", copy=False)
        else:
            policy = "float" if scale_policy == "float" else "pow2"
            scale_arr, quantized = block_quantize_qmma_e2m3(tensor, policy)
            scale_arr = scale_arr.astype("<f4", copy=False)
            payload = repack_to_qmma_layout_f6_dense(quantized, m, k)
    else:
        if scale_inv is not None and scale_policy == "source":
            scale_shape = tuple(scale_inv.shape)
            if scale_shape != (scale_rows, scale_cols):
                raise ValueError(
                    f"{spec.name}: expected scale_inv {(scale_rows, scale_cols)}, got {scale_shape}"
                )
            quantized = e4m3_bytes_from_tensor(tensor)
            scale_arr = scale_inv.detach().cpu().float().contiguous().numpy().astype("<f4", copy=False)
        else:
            policy = "pow2" if scale_policy == "pow2" else "float"
            scale_arr, quantized = block_quantize_qmma_f8(tensor, policy)
            scale_arr = scale_arr.astype("<f4", copy=False)

        if block_layout == "rowmajor":
            payload = quantized.tobytes()
        elif block_layout == "qmma":
            payload = repack_to_qmma_layout_f8(quantized, m, k)
        elif block_layout == "qmma-word":
            payload = repack_to_qmma_word_layout_f8(quantized, m, k)
        else:
            raise ValueError(f"unsupported block layout: {block_layout}")

    if len(payload) != payload_size:
        raise ValueError(f"{spec.name}: expected {payload_size} FP8 bytes, got {len(payload)}")
    f.write(struct.pack("<II", scale_rows, scale_cols))
    f.write(scale_arr.tobytes())
    f.write(payload)
    print(
        f"  {spec.name:72s} {spec.shape!s:18s} "
        f"block_scales={scale_rows}x{scale_cols} layout={block_layout} policy={scale_policy}"
    )
    return size


def qmma_source_tensor(store: TensorStore, spec: TensorSpec) -> torch.Tensor:
    tensor = store.get(spec.name)
    scale_name = f"{spec.name}_scale_inv"
    if scale_name in store.names():
        return deblock_fp8_weight(tensor, store.get(scale_name), spec.name)
    return tensor


def qmma_scale_inv_tensor(store: TensorStore, spec: TensorSpec) -> torch.Tensor | None:
    scale_name = f"{spec.name}_scale_inv"
    return store.get(scale_name) if scale_name in store.names() else None


def validate_shape(tensor: torch.Tensor, spec: TensorSpec) -> None:
    shape = tuple(tensor.shape)
    if shape != spec.shape:
        raise ValueError(f"{spec.name}: expected {spec.shape}, got {shape}")


def tensor_specs(config, names: set[str]) -> list[TensorSpec]:
    text = as_text_config(config)
    hidden = int(text.hidden_size)
    intermediate = int(text.intermediate_size)
    layers = int(text.num_hidden_layers)
    vocab = int(text.vocab_size)
    heads = int(text.num_attention_heads)
    kv_heads = int(text.num_key_value_heads)
    head_dim = int(text.head_dim)
    linear_k_heads = int(text.linear_num_key_heads)
    linear_v_heads = int(text.linear_num_value_heads)
    linear_k_dim = int(text.linear_key_head_dim)
    linear_v_dim = int(text.linear_value_head_dim)
    conv_kernel = int(text.linear_conv_kernel_dim)
    layer_types = list(text.layer_types)

    specs = [
        TensorSpec("model.language_model.embed_tokens.weight", (vocab, hidden), "non_quant"),
        TensorSpec("model.language_model.norm.weight", (hidden,), "non_quant"),
    ]

    tied = bool(getattr(text, "tie_word_embeddings", getattr(config, "tie_word_embeddings", False)))
    lm_head_name = "lm_head.weight"
    if not tied or lm_head_name in names:
        specs.append(TensorSpec(lm_head_name, (vocab, hidden), "qm8"))

    for layer, layer_type in enumerate(layer_types):
        prefix = f"model.language_model.layers.{layer}"
        specs.extend(
            [
                TensorSpec(f"{prefix}.input_layernorm.weight", (hidden,), "non_quant"),
                TensorSpec(f"{prefix}.post_attention_layernorm.weight", (hidden,), "non_quant"),
                TensorSpec(f"{prefix}.mlp.gate_proj.weight", (intermediate, hidden), "qm8"),
                TensorSpec(f"{prefix}.mlp.up_proj.weight", (intermediate, hidden), "qm8"),
                TensorSpec(f"{prefix}.mlp.down_proj.weight", (hidden, intermediate), "qm8"),
            ]
        )
        if layer_type == "full_attention":
            specs.extend(
                [
                    TensorSpec(f"{prefix}.self_attn.q_norm.weight", (head_dim,), "non_quant"),
                    TensorSpec(f"{prefix}.self_attn.k_norm.weight", (head_dim,), "non_quant"),
                    TensorSpec(f"{prefix}.self_attn.q_proj.weight", (heads * head_dim * 2, hidden), "qm8"),
                    TensorSpec(f"{prefix}.self_attn.k_proj.weight", (kv_heads * head_dim, hidden), "qm8"),
                    TensorSpec(f"{prefix}.self_attn.v_proj.weight", (kv_heads * head_dim, hidden), "qm8"),
                    TensorSpec(f"{prefix}.self_attn.o_proj.weight", (hidden, heads * head_dim), "qm8"),
                ]
            )
        elif layer_type == "linear_attention":
            key_dim = linear_k_heads * linear_k_dim
            value_dim = linear_v_heads * linear_v_dim
            conv_dim = key_dim * 2 + value_dim
            specs.extend(
                [
                    TensorSpec(f"{prefix}.linear_attn.A_log", (linear_v_heads,), "f32"),
                    TensorSpec(f"{prefix}.linear_attn.dt_bias", (linear_v_heads,), "non_quant"),
                    TensorSpec(f"{prefix}.linear_attn.norm.weight", (linear_v_dim,), "f32"),
                    TensorSpec(
                        f"{prefix}.linear_attn.conv1d.weight",
                        (conv_dim, 1, conv_kernel),
                        "non_quant",
                    ),
                    TensorSpec(f"{prefix}.linear_attn.in_proj_a.weight", (linear_v_heads, hidden), "qm8"),
                    TensorSpec(f"{prefix}.linear_attn.in_proj_b.weight", (linear_v_heads, hidden), "qm8"),
                    TensorSpec(f"{prefix}.linear_attn.in_proj_qkv.weight", (conv_dim, hidden), "qm8"),
                    TensorSpec(f"{prefix}.linear_attn.in_proj_z.weight", (value_dim, hidden), "qm8"),
                    TensorSpec(f"{prefix}.linear_attn.out_proj.weight", (hidden, value_dim), "qm8"),
                ]
            )
        else:
            raise ValueError(f"unsupported layer type at layer {layer}: {layer_type!r}")

    return specs


def mtp_emit_enabled() -> bool:
    """Whether to append the native MTP head section (env-gated, default off).

    Default output stays byte-identical when this is off; the section and its
    header flag bit are only added when TQ_EMIT_MTP is set.
    """
    return bool(os.environ.get("TQ_EMIT_MTP"))


def mtp_tensor_specs(config) -> list[TensorSpec]:
    """Canonical MTP section tensor order. MUST match the engine's read order in
    forward_qwen.cu (parse_tqf MTP block). The MTP head is structurally one
    full_attention decoder layer (mtp.layers.0.*) plus the fc projection and the
    pre_fc/post norms, so its weights pack with the SAME quant/layout as the main
    model's full-attention layers."""
    text = as_text_config(config)
    hidden = int(text.hidden_size)
    intermediate = int(text.intermediate_size)
    heads = int(text.num_attention_heads)
    kv_heads = int(text.num_key_value_heads)
    head_dim = int(text.head_dim)
    p = "mtp."
    lp = "mtp.layers.0."
    # The MTP head is tiny (~0.42 GB FP8 / ~0.85 GB BF16); quantizing it costs accept
    # for negligible memory, so all 8 projection weights (fc + q/k/v/o + gate/up/down)
    # are stored FULL-PRECISION BF16 (non_quant), like the norms. Order MUST match the
    # engine's read order in forward_qwen.cu (parse_tqf MTP block).
    return [
        TensorSpec(f"{p}pre_fc_norm_embedding.weight", (hidden,), "non_quant"),
        TensorSpec(f"{p}pre_fc_norm_hidden.weight", (hidden,), "non_quant"),
        TensorSpec(f"{p}fc.weight", (hidden, 2 * hidden), "non_quant"),
        TensorSpec(f"{lp}input_layernorm.weight", (hidden,), "non_quant"),
        TensorSpec(f"{lp}post_attention_layernorm.weight", (hidden,), "non_quant"),
        TensorSpec(f"{lp}self_attn.q_norm.weight", (head_dim,), "non_quant"),
        TensorSpec(f"{lp}self_attn.k_norm.weight", (head_dim,), "non_quant"),
        TensorSpec(f"{lp}self_attn.q_proj.weight", (heads * head_dim * 2, hidden), "non_quant"),
        TensorSpec(f"{lp}self_attn.k_proj.weight", (kv_heads * head_dim, hidden), "non_quant"),
        TensorSpec(f"{lp}self_attn.v_proj.weight", (kv_heads * head_dim, hidden), "non_quant"),
        TensorSpec(f"{lp}self_attn.o_proj.weight", (hidden, heads * head_dim), "non_quant"),
        TensorSpec(f"{lp}mlp.gate_proj.weight", (intermediate, hidden), "non_quant"),
        TensorSpec(f"{lp}mlp.up_proj.weight", (intermediate, hidden), "non_quant"),
        TensorSpec(f"{lp}mlp.down_proj.weight", (hidden, intermediate), "non_quant"),
        TensorSpec(f"{p}norm.weight", (hidden,), "non_quant"),
    ]


def build_header(config, dtype_id: int, names: set[str], block_scaled: bool, block_layout: str,
                 emit_mtp: bool = False) -> bytes:
    text = as_text_config(config)
    rope = getattr(text, "rope_parameters", {}) or {}
    layer_types = list(text.layer_types)
    tied = bool(getattr(text, "tie_word_embeddings", getattr(config, "tie_word_embeddings", False)))
    has_vision = hasattr(config, "vision_config") and getattr(config, "vision_config") is not None
    has_mtp = int(getattr(text, "mtp_num_hidden_layers", 0) or 0) > 0 or any(
        name.startswith("mtp.") for name in names
    )

    flags = FLAG_FP8_QMMA_WEIGHTS
    if tied:
        flags |= FLAG_TIED_EMBEDDINGS
    if dtype_id == DTYPE_BF16:
        flags |= FLAG_SOURCE_BF16_NON_QUANT
    if has_vision:
        flags |= FLAG_DROPPED_VISION
    if has_mtp:
        flags |= FLAG_DROPPED_MTP
    if block_scaled:
        flags |= FLAG_BLOCK_SCALED_QMMA
        if block_layout == "rowmajor":
            flags |= FLAG_BLOCK_SCALED_ROWMAJOR
        elif block_layout == "qmma-word":
            flags |= FLAG_BLOCK_SCALED_QMMA_WORDMAJOR
        elif block_layout == "qmma-e2m3":
            flags |= FLAG_BLOCK_SCALED_E2M3
    if sparse_24_enabled() and block_layout == "qmma-e2m3":
        flags |= FLAG_SPARSE_24_E2M3
    if emit_mtp:
        flags |= FLAG_HAS_MTP

    encoded_layer_types = bytes(layer_type_id(layer_type) for layer_type in layer_types)
    fixed = struct.pack(
        "<16I3f8I",
        flags,
        MODEL_FAMILY_QWEN3_5,
        dtype_id,
        int(text.hidden_size),
        int(text.intermediate_size),
        int(text.num_hidden_layers),
        int(text.vocab_size),
        int(text.num_attention_heads),
        int(text.num_key_value_heads),
        int(text.head_dim),
        int(text.linear_num_key_heads),
        int(text.linear_key_head_dim),
        int(text.linear_num_value_heads),
        int(text.linear_value_head_dim),
        int(text.linear_conv_kernel_dim),
        int(text.max_position_embeddings),
        float(text.rms_norm_eps),
        float(rope.get("rope_theta", getattr(text, "rope_theta", 10000.0))),
        float(rope.get("partial_rotary_factor", getattr(text, "partial_rotary_factor", 1.0))),
        1 if rope.get("mrope_interleaved", False) else 0,
        int((rope.get("mrope_section") or [0, 0, 0])[0]),
        int((rope.get("mrope_section") or [0, 0, 0])[1]),
        int((rope.get("mrope_section") or [0, 0, 0])[2]),
        1 if tied else 0,
        1,  # text_only
        1 if has_vision else 0,
        1 if has_mtp else 0,
    )
    header_bytes = 4 + 4 + len(fixed) + len(encoded_layer_types)
    return b"TQF1" + struct.pack("<I", header_bytes) + fixed + encoded_layer_types


def layer_type_id(layer_type: str) -> int:
    if layer_type == "linear_attention":
        return LAYER_LINEAR_ATTENTION
    if layer_type == "full_attention":
        return LAYER_FULL_ATTENTION
    raise ValueError(f"unsupported layer type: {layer_type!r}")


def dtype_id_from_name(name: str) -> int:
    if name in ("bf16", "bfloat16"):
        return DTYPE_BF16
    if name in ("fp16", "float16"):
        return DTYPE_FP16
    raise ValueError(f"unsupported non-quant dtype: {name}")


def expected_names_for_text(config) -> set[str]:
    return {spec.name for spec in tensor_specs(config, set())}


def write_tqf(
    model_path: str,
    output_path: str,
    non_quant_dtype: str,
    dry_run: bool,
    block_scaled_mode: str,
    block_scale_policy: str,
    block_layout: str,
) -> dict[str, Any]:
    model_dir = Path(model_path)
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model_type = getattr(cfg, "model_type", None)
    if model_type != "qwen3_5":
        raise ValueError(f"convert_qwen_tqf.py expects model_type='qwen3_5', got {model_type!r}")

    dtype_id = dtype_id_from_name(non_quant_dtype)
    with TensorStore(model_dir) as store:
        specs = tensor_specs(cfg, store.names())
        missing = [spec.name for spec in specs if spec.name not in store.names()]
        if missing:
            raise KeyError("missing required tensor(s): " + ", ".join(missing))

        has_source_block_scales = any(f"{spec.name}_scale_inv" in store.names() for spec in specs if spec.kind == "qm8")
        if block_scaled_mode == "auto":
            block_scaled = has_source_block_scales
        elif block_scaled_mode == "always":
            block_scaled = True
        elif block_scaled_mode == "never":
            block_scaled = False
        else:
            raise ValueError(f"unsupported block_scaled_mode: {block_scaled_mode}")

        mtp_specs = mtp_tensor_specs(cfg)
        emit_mtp = mtp_emit_enabled() and all(s.name in store.names() for s in mtp_specs)
        if mtp_emit_enabled() and not emit_mtp:
            missing_mtp = [s.name for s in mtp_specs if s.name not in store.names()]
            print(f"  TQ_EMIT_MTP set but model lacks MTP tensors {missing_mtp}; MTP section NOT emitted")

        header = build_header(cfg, dtype_id, store.names(), block_scaled, block_layout, emit_mtp)
        bytes_written = len(header)
        kind_counts: dict[str, int] = {}
        output_file = None
        if not dry_run:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            output_file = open(output_path, "wb")

        try:
            if output_file is not None:
                output_file.write(header)
            for spec in specs:
                kind_counts[spec.kind] = kind_counts.get(spec.kind, 0) + 1
                tensor = store.get(spec.name)
                if spec.kind == "non_quant":
                    size = write_non_quant(output_file, tensor, spec, dtype_id, dry_run)
                elif spec.kind == "f32":
                    size = write_f32(output_file, tensor, spec, dry_run)
                elif spec.kind == "qm8":
                    if block_scaled:
                        scale_inv = None if dry_run else qmma_scale_inv_tensor(store, spec)
                        if not dry_run:
                            tensor = maybe_verify_mask_24(tensor, spec.name)
                        size = write_qmma_f8_block_scaled(
                            output_file, tensor, spec, dry_run, scale_inv, block_scale_policy, block_layout
                        )
                        if sparse_24_enabled() and block_layout == "qmma-e2m3":
                            size += write_sparse_24_record(
                                output_file, tensor, spec, block_scale_policy, dry_run
                            )
                    else:
                        qmma_tensor = tensor if dry_run else qmma_source_tensor(store, spec)
                        size = write_qmma_f8(output_file, qmma_tensor, spec, dry_run)
                else:
                    raise ValueError(f"{spec.name}: unsupported tensor kind {spec.kind!r}")
                bytes_written += size
                if dry_run:
                    print(f"  {spec.name:72s} {spec.shape!s:18s} {spec.kind}")

            if emit_mtp:
                print("  -- MTP head section (TQ_EMIT_MTP) --")
                for spec in mtp_specs:
                    kind_counts[spec.kind] = kind_counts.get(spec.kind, 0) + 1
                    tensor = store.get(spec.name)
                    if spec.kind == "non_quant":
                        size = write_non_quant(output_file, tensor, spec, dtype_id, dry_run)
                    elif spec.kind == "qm8":
                        if block_scaled:
                            scale_inv = None if dry_run else qmma_scale_inv_tensor(store, spec)
                            size = write_qmma_f8_block_scaled(
                                output_file, tensor, spec, dry_run, scale_inv, block_scale_policy, block_layout
                            )
                        else:
                            qmma_tensor = tensor if dry_run else qmma_source_tensor(store, spec)
                            size = write_qmma_f8(output_file, qmma_tensor, spec, dry_run)
                    else:
                        raise ValueError(f"{spec.name}: unsupported MTP tensor kind {spec.kind!r}")
                    bytes_written += size
                    if dry_run:
                        print(f"  {spec.name:72s} {spec.shape!s:18s} {spec.kind}")
        finally:
            if output_file is not None:
                output_file.close()

    return {
        "model": str(model_dir),
        "output": output_path,
        "dry_run": dry_run,
        "header_bytes": len(header),
        "total_bytes": bytes_written,
        "total_gb": bytes_written / 1e9,
        "tensor_count": len(specs),
        "kind_counts": kind_counts,
        "block_scaled": block_scaled,
        "block_scale_policy": block_scale_policy if block_scaled else None,
        "block_layout": block_layout if block_scaled else None,
        "emit_mtp": emit_mtp,
        "mtp_tensor_count": len(mtp_specs) if emit_mtp else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Qwen3.5/Qwen3.6 text tower to TQF1")
    parser.add_argument("model", nargs="?", default=DEFAULT_MODEL, help="Local Qwen HF model directory")
    parser.add_argument("-o", "--output", required=True, help="Output .tqf path")
    parser.add_argument(
        "--non-quant-dtype",
        default="bf16",
        choices=("bf16", "bfloat16", "fp16", "float16"),
        help="Storage dtype for embeddings, norms, dt_bias, and conv1d weights",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and estimate without writing output")
    parser.add_argument(
        "--block-scaled",
        default="auto",
        choices=("auto", "always", "never"),
        help="Store QMMA weights with 128x128 block scales. auto preserves source block-scaled FP8 checkpoints.",
    )
    parser.add_argument(
        "--block-scale-policy",
        default="source",
        choices=("source", "float", "pow2"),
        help=(
            "For block-scaled weights: source preserves checkpoint scale_inv when present; "
            "float recomputes exact absmax/448 scales; pow2 uses power-of-two scales for QMMA.SF/ue8m0."
        ),
    )
    parser.add_argument(
        "--block-layout",
        default="rowmajor",
        choices=("rowmajor", "qmma", "qmma-word", "qmma-e2m3"),
        help=(
            "FP8 payload layout for block-scaled weights. qmma is the tensor-core fragment layout; "
            "qmma-word stores each tile as word-major lanes for sector-efficiency experiments; "
            "qmma-e2m3 packs weights as dense 6-bit E2M3 (0.75x bytes) for the mixed E2M3xE4M3 path."
        ),
    )
    args = parser.parse_args()

    summary = write_tqf(
        args.model,
        args.output,
        args.non_quant_dtype,
        args.dry_run,
        args.block_scaled,
        args.block_scale_policy,
        args.block_layout,
    )
    print(json.dumps(summary, indent=2))
    if not args.dry_run:
        print(f"Wrote {args.output} ({summary['total_gb']:.2f} GB)")


if __name__ == "__main__":
    main()
