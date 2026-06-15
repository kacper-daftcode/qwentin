#!/usr/bin/env python3
"""Convert HuggingFace model to qwentin GGUF format.

Quantizes linear layer weights to FP8 (E4M3), FP6 (E2M3), or Sparse FP6
and repacks them into QMMA tensor core fragment layout for direct consumption
by the SM120 GEMV kernels.

Usage:
    python convert.py meta-llama/Llama-2-7b-hf --quant f6  -o llama-7b-f6.gguf
    python convert.py meta-llama/Llama-2-7b-hf --quant f8  -o llama-7b-f8.gguf
    python convert.py meta-llama/Llama-2-7b-hf --quant sf6 -o llama-7b-sf6.gguf
"""

import argparse
import math
import os
import struct
import sys

import numpy as np

# GGUF constants
GGUF_MAGIC = 0x46475547  # "GGUF"
GGUF_VERSION = 3
GGUF_TYPE_F32    = 0
GGUF_TYPE_F16    = 1
GGUF_TYPE_F8_E4M3  = 100
GGUF_TYPE_F6_E2M3  = 101
GGUF_TYPE_SF6_E2M3 = 102


# ---------------------------------------------------------------------------
# FP6 E2M3 conversion (NumPy vectorized)
# ---------------------------------------------------------------------------

_E2M3_ABS = None   # sorted positive representable magnitudes
_E2M3_CODE = None  # uint8 code for each magnitude
_E2M3_MID = None   # midpoints between consecutive magnitudes (round-to-nearest)


def _e2m3_tables():
    """Build the 32 positive E2M3 magnitudes and their codes once."""
    global _E2M3_ABS, _E2M3_CODE, _E2M3_MID
    if _E2M3_ABS is None:
        vals, codes = [], []
        for e in range(4):          # e=0 subnormal, e=1..3 normal (bias 1)
            for m in range(8):
                if e == 0:
                    v = m / 8.0
                    c = m
                else:
                    v = (1.0 + m / 8.0) * (2.0 ** (e - 1))
                    c = (e << 3) | m
                vals.append(v)
                codes.append(c)
        v = np.asarray(vals, dtype=np.float32)
        c = np.asarray(codes, dtype=np.uint8)
        order = np.argsort(v)
        _E2M3_ABS = v[order]
        _E2M3_CODE = c[order]
        _E2M3_MID = ((_E2M3_ABS[:-1] + _E2M3_ABS[1:]) * 0.5).astype(np.float32)
    return _E2M3_ABS, _E2M3_CODE, _E2M3_MID


def float_to_e2m3(x):
    """Convert float array to E2M3 (6-bit) values, returned as uint8.

    Round-to-nearest via a lookup table over the 32 representable magnitudes:
    no per-element log2/floor/pow, which is ~10x faster on large weights and is
    what makes on-load (FIFO) packing of a 27B model practical.
    """
    x = np.asarray(x, dtype=np.float32)
    sign = (x < 0).astype(np.uint8)
    ax = np.clip(np.abs(x), 0.0, 7.5)
    _, codes, mid = _e2m3_tables()
    idx = np.searchsorted(mid, ax)            # nearest representable magnitude
    result = codes[idx].astype(np.uint8)
    return (sign << 5) | result


def float_to_e4m3(x):
    """Convert float array to E4M3 (E4M3FN) codes, returned as uint8.

    Standard OCP/CUDA ``float8_e4m3fn``: 1 sign / 4 exponent (bias 7) / 3 mantissa,
    using the FULL normal range up to 448 = S.1111.110 (biased exponent up to 15),
    round-to-nearest-even on the mantissa, subnormals in the exponent-0 field, and
    saturation of out-of-range magnitudes to 448. The NaN encodings (S.1111.111)
    are never produced for finite inputs. Decodes exactly via the engine's
    tq_e4m3_to_float. (Previously the biased exponent was clipped to 14, capping the
    max representable at ~240 and clipping the largest weight in every 448-scaled
    block -> ~20% relL2 round-trip error on weights with outliers.)
    """
    x = np.asarray(x, dtype=np.float32)
    sign = (x < 0).astype(np.uint8)
    ax = np.abs(x).astype(np.float64)
    ax = np.where(np.isfinite(ax), ax, 448.0)   # inf/nan -> max finite

    result = np.zeros(ax.shape, dtype=np.uint8)
    MIN_NORMAL = 2.0 ** -6                       # smallest normal = 0.015625

    # Subnormal region (exponent field 0): step 2^-9, mantissa 0..7. A rounded
    # mantissa of 8 carries up to the smallest normal (code 0x08).
    sub = ax < MIN_NORMAL
    m_sub = np.rint(ax[sub] / (2.0 ** -9))       # round-half-to-even, 0..8
    result[sub] = np.clip(m_sub, 0, 8).astype(np.uint8)

    # Normal region: value = (1 + m/8) * 2^(b-7), biased exponent b in 1..15.
    norm = ~sub
    a = ax[norm]
    e = np.floor(np.log2(a)).astype(np.int64)    # unbiased exponent
    biased = e + 7
    m = np.rint(a / np.exp2(e.astype(np.float64)) * 8.0) - 8.0   # RNE mantissa, 0..8
    carry = m >= 8.0                             # mantissa overflow -> bump exponent
    m = np.where(carry, 0.0, m)
    biased = np.where(carry, biased + 1, biased)
    over = (biased > 15) | ((biased == 15) & (m > 6))           # saturate to 448
    biased = np.where(over, 15, biased)
    m = np.where(over, 6.0, m)
    biased = np.clip(biased, 1, 15).astype(np.uint8)
    m = np.clip(m, 0, 7).astype(np.uint8)
    result[norm] = (biased << 3) | m

    result |= sign << 7
    return result


# ---------------------------------------------------------------------------
# FP6 packing: 4 values → 3 bytes (24 bits)
# ---------------------------------------------------------------------------

def pack_f6(vals):
    """Pack array of 6-bit values into bytes. Length must be multiple of 4."""
    vals = vals.flatten()
    assert len(vals) % 4 == 0
    n_blocks = len(vals) // 4
    packed = np.zeros(n_blocks * 3, dtype=np.uint8)

    for b in range(n_blocks):
        v = vals[b*4 : b*4+4].astype(np.uint32)
        bits = v[0] | (v[1] << 6) | (v[2] << 12) | (v[3] << 18)
        packed[b*3]     = bits & 0xFF
        packed[b*3 + 1] = (bits >> 8) & 0xFF
        packed[b*3 + 2] = (bits >> 16) & 0xFF

    return packed


# ---------------------------------------------------------------------------
# QMMA A-fragment layout repack
# ---------------------------------------------------------------------------

def repack_to_qmma_layout_f6(weights, M, K):
    """Repack quantized FP6 weights [M, K] into QMMA tile layout (vectorized).

    Output: bytes of shape [M/16, K/32, 512].
    FP6: 8 values × 6 bits = 48 bits = 6 bytes per row, padded to 8 bytes.
    Per lane: 8 bytes row0 + 8 bytes row1 = 16 bytes.
    """
    Mt, Kt = M // 16, K // 32
    w = weights.reshape(Mt, 8, 2, Kt, 4, 8)       # [Mt, g, row01, Kt, s, 8vals]
    w = w.transpose(0, 3, 1, 4, 2, 5)             # [Mt, Kt, g, s, row01, 8vals]
    w = (w & 0x3F).astype(np.uint64)

    # Pack 8 × 6-bit values into 48-bit (6-byte) words
    packed = np.zeros((*w.shape[:-1], 1), dtype=np.uint64)  # [Mt,Kt,g,s,row01,1]
    for i in range(8):
        packed[..., 0] |= w[..., i] << np.uint64(i * 6)

    # Extract 6 bytes from the 48-bit packed value + 2 bytes padding = 8 bytes per row
    out = np.zeros((*w.shape[:-1], 8), dtype=np.uint8)      # [Mt,Kt,g,s,row01,8]
    for b in range(6):
        out[..., b] = ((packed[..., 0] >> np.uint64(b * 8)) & 0xFF).astype(np.uint8)

    # Reshape: [Mt, Kt, g, s, row01, 8bytes] → [Mt, Kt, 32lanes, 16bytes] → [Mt, Kt, 512]
    out = out.reshape(Mt, Kt, 8, 4, 2, 8)         # g, s, row01, 8bytes
    out = out.reshape(Mt, Kt, 32, 2, 8)           # 32lanes, 2rows, 8bytes
    out = out.reshape(Mt, Kt, 32, 16)             # 32lanes, 16bytes
    out = out.reshape(Mt, Kt, 512)
    return out.tobytes()


def repack_to_qmma_layout_f8(weights, M, K):
    """Repack quantized FP8 weights [M, K] into SM120 QMMA tile layout.

    SM120 (Blackwell) mma.sync.m16n8k32 A-fragment layout:
    Lane g*4+s (16 bytes):
      bytes  0-3:  A[row 2g,   k=s*8..s*8+3]
      bytes  4-7:  A[row 2g+1, k=s*8..s*8+3]
      bytes  8-11: A[row 2g,   k=s*8+4..s*8+7]
      bytes 12-15: A[row 2g+1, k=s*8+4..s*8+7]

    Key difference from Hopper/Ada: rows interleaved at 4-byte boundaries.
    Verified empirically: correlation 1.000000 vs PyTorch on 2048x2048.

    Output: bytes of shape [M/16, K/32, 512].
    """
    Mt, Kt = M // 16, K // 32
    w = weights.reshape(Mt, 16, Kt, 32).astype(np.uint8)
    # Split: 8 groups × 2 rows, 4 subs × (2 halves × 4 cols)
    w = w.reshape(Mt, 8, 2, Kt, 4, 2, 4)         # [Mt, g, row01, Kt, s, half, 4cols]
    # Interleave: row0_half0, row1_half0, row0_half1, row1_half1
    w = w.transpose(0, 3, 1, 4, 5, 2, 6)         # [Mt, Kt, g, s, half, row01, 4cols]
    # Flatten: lane = g*4+s, 16 bytes = half(2) × row01(2) × 4cols
    w = w.reshape(Mt, Kt, 32, 16)                 # [Mt, Kt, 32lanes, 16bytes]
    w = w.reshape(Mt, Kt, 512)
    return w.tobytes()


def repack_to_qmma_layout_f6_dense(weights, M, K):
    """Dense 6-bit QMMA A-layout for the mixed E2M3xE4M3 path.

    QMMA.SF ``kind::mxf8f6f4`` consumes the A operand byte-per-element (each
    byte holds one code in its low 6 bits; hardware-confirmed by the E3M4 probe
    "output = 32 x decode(byte)^2"). So we take the verified FP8 element order
    (``repack_to_qmma_layout_f8``), then bit-pack each lane's 16 codes (16 bytes,
    low 6 bits) into 12 bytes via a little-endian stream (code j at bit 6*j).
    The runtime reverses this to the byte-per-element fragment before the MMA.

    This stores 12 bytes/lane (0.75x of FP8) while keeping the exact same
    element->byte mapping the FP8 path uses, so the MMA sees an identical
    fragment with E2M3 codes substituted.

    Output: bytes of shape [M/16, K/32, 384].
    """
    Mt, Kt = M // 16, K // 32
    byte_per_elem = np.frombuffer(repack_to_qmma_layout_f8(weights, M, K), dtype=np.uint8)
    frag = byte_per_elem.reshape(-1, 16).astype(np.uint8)        # [n_frags, 16 codes]
    bits = ((frag[:, :, None] >> np.arange(6, dtype=np.uint8)) & 1).astype(np.uint8)
    bits = bits.reshape(frag.shape[0], 96)                       # code j -> bits [6j, 6j+6)
    packed = np.packbits(bits, axis=1, bitorder="little")        # [n_frags, 12]
    return np.ascontiguousarray(packed).reshape(Mt, Kt, 384).tobytes()


def repack_to_qmma_word_layout_f8(weights, M, K):
    """Repack FP8 weights so per-word warp loads are lane-contiguous.

    The QMMA register values are identical to ``repack_to_qmma_layout_f8``:
    each lane still receives four 32-bit words. Only the memory order inside a
    512-byte A tile changes from [lane][word] to [word][lane].
    """
    Mt, Kt = M // 16, K // 32
    lane_major = np.frombuffer(repack_to_qmma_layout_f8(weights, M, K), dtype=np.uint8)
    lane_major = lane_major.reshape(Mt, Kt, 32, 4, 4)
    word_major = lane_major.transpose(0, 1, 3, 2, 4).reshape(Mt, Kt, 512)
    return word_major.tobytes()


# ---------------------------------------------------------------------------
# 2:4 Structured sparsity
# ---------------------------------------------------------------------------

def sparsify_and_quantize(weight_fp32, M, K):
    """Apply 2:4 sparsity, quantize to FP6, return (data, metadata)."""
    weight = weight_fp32.reshape(M, K)
    K_groups = K // 4
    compressed = np.zeros((M, K // 2), dtype=np.uint8)
    metadata = np.zeros(M * K_groups // 2, dtype=np.uint8)

    for row in range(M):
        for g in range(K_groups):
            vals = weight[row, g*4 : g*4+4]
            mags = np.abs(vals)
            # Top-2 indices
            idx = np.argsort(mags)[::-1][:2]
            idx.sort()
            idx0, idx1 = int(idx[0]), int(idx[1])

            q0 = float_to_e2m3(np.array([vals[idx0]]))[0]
            q1 = float_to_e2m3(np.array([vals[idx1]]))[0]

            out_idx = row * (K // 2) + g * 2
            compressed[0, out_idx] = q0  # simplified flat layout
            compressed[0, out_idx + 1] = q1

            nibble = (idx0 & 3) | ((idx1 & 3) << 2)
            meta_idx = row * K_groups + g
            metadata[meta_idx // 2] |= nibble << ((meta_idx % 2) * 4)

    return compressed.flatten(), metadata


# ---------------------------------------------------------------------------
# GGUF writer
# ---------------------------------------------------------------------------

class GGUFWriter:
    """Minimal GGUF file writer for qwentin weight format."""

    def __init__(self, path):
        self.path = path
        self.tensors = []
        self.metadata = {}

    def add_metadata(self, key, value, vtype='string'):
        self.metadata[key] = (vtype, value)

    def add_tensor(self, name, data, tensor_type, shape):
        self.tensors.append({
            'name': name,
            'data': data if isinstance(data, bytes) else data.tobytes(),
            'type': tensor_type,
            'shape': shape,
        })

    def _write_string(self, f, s):
        encoded = s.encode('utf-8')
        f.write(struct.pack('<Q', len(encoded)))
        f.write(encoded)

    def _write_metadata_value(self, f, vtype, value):
        type_map = {
            'uint32': (4, '<I'), 'int32': (5, '<i'),
            'float32': (6, '<f'), 'uint64': (10, '<Q'),
        }
        if vtype == 'string':
            f.write(struct.pack('<I', 8))  # GGUF_TYPE_STRING
            self._write_string(f, value)
        elif vtype in type_map:
            tid, fmt = type_map[vtype]
            f.write(struct.pack('<I', tid))
            f.write(struct.pack(fmt, value))

    def write(self):
        with open(self.path, 'wb') as f:
            # Header
            f.write(struct.pack('<I', GGUF_MAGIC))
            f.write(struct.pack('<I', GGUF_VERSION))
            f.write(struct.pack('<Q', len(self.tensors)))
            f.write(struct.pack('<Q', len(self.metadata)))

            # Metadata KV pairs
            for key, (vtype, value) in self.metadata.items():
                self._write_string(f, key)
                self._write_metadata_value(f, vtype, value)

            # Tensor info headers
            data_offset = 0
            tensor_offsets = []
            for t in self.tensors:
                self._write_string(f, t['name'])
                n_dims = len(t['shape'])
                f.write(struct.pack('<I', n_dims))
                for dim in t['shape']:
                    f.write(struct.pack('<Q', dim))
                f.write(struct.pack('<I', t['type']))
                f.write(struct.pack('<Q', data_offset))
                tensor_offsets.append(data_offset)
                data_offset += len(t['data'])
                # Align to 32 bytes
                padding = (32 - (data_offset % 32)) % 32
                data_offset += padding

            # Alignment padding before tensor data
            pos = f.tell()
            alignment = (32 - (pos % 32)) % 32
            f.write(b'\x00' * alignment)

            # Tensor data
            for i, t in enumerate(self.tensors):
                f.write(t['data'])
                padding = (32 - (len(t['data']) % 32)) % 32
                f.write(b'\x00' * padding)

        print(f'Wrote {self.path} ({os.path.getsize(self.path) / 1e9:.2f} GB, '
              f'{len(self.tensors)} tensors)')


# ---------------------------------------------------------------------------
# Linear layer detection
# ---------------------------------------------------------------------------

LINEAR_SUFFIXES = [
    'q_proj.weight', 'k_proj.weight', 'v_proj.weight', 'o_proj.weight',
    'gate_proj.weight', 'up_proj.weight', 'down_proj.weight',
]


def is_linear_weight(name):
    return any(name.endswith(s) for s in LINEAR_SUFFIXES)


# ---------------------------------------------------------------------------
# Main conversion logic
# ---------------------------------------------------------------------------

def convert_model(model_path, output_path, quant):
    try:
        from transformers import AutoModelForCausalLM
        import torch
    except ImportError:
        print('ERROR: requires transformers and torch. '
              'Install with: pip install transformers torch')
        sys.exit(1)

    print(f'Loading model: {model_path}')
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, device_map='cpu')

    writer = GGUFWriter(output_path)
    writer.add_metadata('general.architecture', 'llama')
    writer.add_metadata('general.name', os.path.basename(model_path))
    writer.add_metadata('qwentin.quantization', quant)

    # Architecture metadata for the inference engine
    cfg = model.config
    writer.add_metadata('llama.hidden_size', cfg.hidden_size, 'uint32')
    writer.add_metadata('llama.intermediate_size', cfg.intermediate_size, 'uint32')
    writer.add_metadata('llama.num_hidden_layers', cfg.num_hidden_layers, 'uint32')
    writer.add_metadata('llama.num_attention_heads', cfg.num_attention_heads, 'uint32')
    writer.add_metadata('llama.num_key_value_heads', cfg.num_key_value_heads, 'uint32')
    writer.add_metadata('llama.vocab_size', cfg.vocab_size, 'uint32')
    writer.add_metadata('llama.max_position_embeddings', cfg.max_position_embeddings, 'uint32')
    writer.add_metadata('llama.rms_norm_eps', cfg.rms_norm_eps, 'float32')

    type_map = {'f8': GGUF_TYPE_F8_E4M3, 'f6': GGUF_TYPE_F6_E2M3,
                'sf6': GGUF_TYPE_SF6_E2M3}
    target_type = type_map[quant]

    n_quantized = 0
    n_kept = 0

    for name, param in model.named_parameters():
        weight = param.detach().float().numpy()
        shape = list(weight.shape)

        if is_linear_weight(name) and len(shape) == 2:
            M, K = shape

            # Ensure dimensions are tile-aligned
            M_aligned = ((M + 15) // 16) * 16
            K_aligned = ((K + 31) // 32) * 32

            if M_aligned != M or K_aligned != K:
                padded = np.zeros((M_aligned, K_aligned), dtype=np.float32)
                padded[:M, :K] = weight
                weight = padded
                shape = [M_aligned, K_aligned]

            M, K = shape

            if quant == 'f8':
                # Per-tensor absmax scaling for FP8 quality
                absmax = np.abs(weight).max()
                scale = 448.0 / absmax if absmax > 0 else 1.0
                weight_scaled = weight * scale
                quantized = float_to_e4m3(weight_scaled)
                # Store raw FP8 (row-major) for CPU inference compatibility
                # GPU kernel repacks on-the-fly or uses pre-packed variant
                writer.add_tensor(name, quantized.tobytes(), target_type, shape)
                # Store scale factor as separate F32 tensor
                scale_data = np.array([scale], dtype=np.float32)
                writer.add_tensor(name + '.scale', scale_data.tobytes(), GGUF_TYPE_F32, [1])
            elif quant == 'f6':
                quantized = float_to_e2m3(weight)
                tiled = repack_to_qmma_layout_f6(quantized, M, K)
                writer.add_tensor(name, tiled, target_type, shape)
            elif quant == 'sf6':
                data, meta = sparsify_and_quantize(weight, M, K)
                combined = np.concatenate([data, meta])
                writer.add_tensor(name, combined.tobytes(), target_type, shape)

            n_quantized += 1
            print(f'  {name:50s}  {M}x{K}  → {quant.upper()}')
        else:
            # Keep embeddings, norms, etc. as FP32
            writer.add_tensor(name, weight.tobytes(), GGUF_TYPE_F32, shape)
            n_kept += 1

    print(f'\nQuantized: {n_quantized} tensors to {quant.upper()}')
    print(f'Kept as FP32: {n_kept} tensors')

    writer.write()


def main():
    parser = argparse.ArgumentParser(
        description='Convert HuggingFace model to qwentin GGUF format')
    parser.add_argument('model', help='HuggingFace model ID or local path')
    parser.add_argument('--quant', choices=['f8', 'f6', 'sf6'], default='f6',
                        help='Quantization format (default: f6)')
    parser.add_argument('-o', '--output', required=True,
                        help='Output GGUF file path')
    args = parser.parse_args()

    convert_model(args.model, args.output, args.quant)


if __name__ == '__main__':
    main()
