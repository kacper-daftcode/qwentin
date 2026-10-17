"""CPU-only tests of tools/inspect_tqf.py on synthetic .tqf files: a tiny model layout (hidden 64, 2 layers) written
with the header struct the converter uses, a QMMA / FP6 payload of the right size, an EXL3 payload with per-record
headers, and the PASS / FAIL verdicts. No torch, no transformers."""

from __future__ import annotations

import io
import os
import struct
import sys

import pytest

TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
sys.path.insert(0, TOOLS)

import inspect_tqf as it  # noqa: E402

# a tiny Qwen3.5-class layout: hidden 64, intermediate 128, vocab 256, 2 layers (one DeltaNet, one full attention)
DIMS = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, vocab_size=256, num_attention_heads=2,
            num_key_value_heads=1, head_dim=32, linear_num_key_heads=2, linear_key_head_dim=16,
            linear_num_value_heads=16, linear_value_head_dim=16, linear_conv_kernel_dim=4, max_position_embeddings=4096)
LAYER_TYPES = bytes([1, 2])


def make_header(flags: int, tied: int = 0, has_mtp_src: int = 1) -> bytes:
    """Mirror build_header() in tools/convert_qwen_tqf.py."""
    fixed = struct.pack(
        it.FIXED_HEADER_FMT,
        flags, 1, 2,
        DIMS["hidden_size"], DIMS["intermediate_size"], DIMS["num_hidden_layers"], DIMS["vocab_size"],
        DIMS["num_attention_heads"], DIMS["num_key_value_heads"], DIMS["head_dim"],
        DIMS["linear_num_key_heads"], DIMS["linear_key_head_dim"], DIMS["linear_num_value_heads"],
        DIMS["linear_value_head_dim"], DIMS["linear_conv_kernel_dim"], DIMS["max_position_embeddings"],
        1e-6, 10000.0, 0.25,
        0, 0, 0, 0,
        tied, 1, 1, has_mtp_src,
    )
    header_bytes = 8 + len(fixed) + len(LAYER_TYPES)
    return b"TQF1" + struct.pack("<I", header_bytes) + fixed + LAYER_TYPES


def test_header_roundtrip():
    flags = it.PRODUCTION_FP6_FLAGS | it.FLAG_SOURCE_BF16_NON_QUANT | it.FLAG_DROPPED_VISION | it.FLAG_DROPPED_MTP
    h = it.read_header(io.BytesIO(make_header(flags)))
    assert h.flags == flags == 0x53d
    assert h.hidden_size == 64 and h.vocab_size == 256 and h.num_hidden_layers == 2
    assert h.layer_types == [1, 2]
    assert h.rms_norm_eps == pytest.approx(1e-6) and h.partial_rotary_factor == pytest.approx(0.25)
    assert h.tie_word_embeddings == 0 and h.has_mtp == 1
    assert "block_scaled_e2m3" in it.flag_names(h.flags) and "has_mtp_section" in it.flag_names(h.flags)
    assert it.flag_names(1 << 20) == ["unknown:0x100000"]


def test_header_rejects_garbage():
    with pytest.raises(ValueError):
        it.read_header(io.BytesIO(b"NOPE" + b"\0" * 200))
    bad = bytearray(make_header(0x1))
    bad[8 + 4 * 5 : 8 + 4 * 6] = struct.pack("<I", 7)      # num_hidden_layers = 7, but only 2 layer-type bytes
    with pytest.raises(ValueError):
        it.read_header(io.BytesIO(bytes(bad)))
    with pytest.raises(EOFError):
        it.read_header(io.BytesIO(make_header(0x1)[:40]))


def _write(tmp_path, name, header: bytes, payload: bytes):
    p = tmp_path / name
    p.write_bytes(header + payload)
    return p


def test_fp6_payload_and_verdicts(tmp_path, capsys):
    flags = 0x53d
    header = it.read_header(io.BytesIO(make_header(flags)))
    expected, counts, _ = it.walk_payload(header)
    # hand-count a few slots: embed 256x64 bf16, norm 64 bf16, lm_head block-scaled E2M3 256x64
    assert counts["non_quant"] == 2 + 2 * 2 + 2 + 2       # embed+norm, 2 norms per layer, dt_bias+conv1d, q_norm+k_norm
    assert counts["qm8"] == 1 + 2 * 3 + 5 + 4
    assert counts["f32"] == 2
    assert counts["mtp_non_quant"] == 15 and "mtp_qm8" not in counts
    assert it.qmma_bytes(256, 64, True, True) == 8 + 2 * 1 * 4 + (256 // 16) * (64 // 32) * 384
    p = _write(tmp_path, "fp6.tqf", make_header(flags), b"\0" * expected)
    r = it.inspect(p)
    out = capsys.readouterr().out
    assert r["payload_ok"] and r["production_fp6"] and not r["exl3_mtp"]
    assert "verdict production FP6 (flags 0x1|0x20|0x100|0x400 + payload): PASS" in out
    assert "verdict EXL3 + MTP (flags 0x800|0x400|0x1000 + payload):     FAIL" in out
    # truncated file -> payload mismatch -> both verdicts FAIL
    p2 = _write(tmp_path, "short.tqf", make_header(flags), b"\0" * (expected - 100))
    r2 = it.inspect(p2, verbose=False)
    assert not r2["payload_ok"] and not r2["production_fp6"]


def test_plain_fp8_without_mtp_fails_fp6_verdict(tmp_path):
    flags = 0x1d                       # fp8 qmma, bf16 non-quant, dropped vision + mtp: the old `auto` default on bf16
    header = it.read_header(io.BytesIO(make_header(flags)))
    expected, counts, _ = it.walk_payload(header)
    assert "mtp_non_quant" not in counts
    assert it.qmma_bytes(256, 64) == 4 + (256 // 16) * (64 // 32) * 512
    r = it.inspect(_write(tmp_path, "fp8.tqf", make_header(flags), b"\0" * expected), verbose=False)
    assert r["payload_ok"] and not r["production_fp6"]
    ok, why = it.verdict_production_fp6(flags, True)
    assert not ok and "no block scales" in why and "no MTP head section" in why
    ok, why = it.verdict_production_fp6(flags | it.FLAG_BLOCK_SCALED_QMMA | it.FLAG_HAS_MTP, True)
    assert not ok and "not E2M3" in why


def _exl3_record(kbits: int, k: int, m: int, codebook: int = 2) -> bytes:
    hdr = struct.pack(it.EXL3_RECORD_FMT, kbits, k, m, codebook)
    if kbits == 0:
        return hdr + b"\0" * (m * k * 2)
    return hdr + b"\0" * (2 * k + 2 * m + (k // 16) * (m // 16) * 16 * kbits * 2)


def _exl3_payload(header, kbits_main: int, mtp_exl3: bool):
    """Build a payload where every Linear is an EXL3 record (the 16-wide in_proj_a/b stay bf16 = kbits 0, like the
    real checkpoints' 48-wide ones), and the MTP projections are EXL3 records of 4 bits when mtp_exl3."""
    buf = bytearray()
    for section, slots in (("main", list(it.payload_slots(header))), ("mtp", list(it.mtp_slots(header)))):
        for s in slots:
            if s.kind == "non_quant":
                buf += b"\0" * (2 * s.count)
            elif s.kind == "f32":
                buf += b"\0" * (4 * s.count)
            else:
                m, k = s.shape
                if section == "mtp" and not mtp_exl3:
                    buf += b"\0" * (2 * m * k)
                else:
                    kb = 0 if "in_proj_a" in s.name or "in_proj_b" in s.name else (4 if section == "mtp" else kbits_main)
                    buf += _exl3_record(kb, k, m)
    return bytes(buf)


def test_exl3_payload_walk_and_verdicts(tmp_path, capsys):
    flags = 0x1c1d                     # the served -mtp4 files
    header = it.read_header(io.BytesIO(make_header(flags)))
    payload = _exl3_payload(header, kbits_main=5, mtp_exl3=True)
    p = _write(tmp_path, "exl3-mtp4.tqf", make_header(flags), payload)
    with pytest.raises(ValueError):
        it.walk_payload(header)        # an EXL3 file cannot be sized without the file
    r = it.inspect(p)
    out = capsys.readouterr().out
    assert r["payload_ok"] and r["exl3_mtp"] and not r["production_fp6"]
    assert r["details"]["exl3_kbits"][5] > 0 and r["details"]["exl3_kbits"][4] == 8
    assert len(r["details"]["exl3_bf16_linears"]) == 2 and r["details"]["exl3_codebooks"] == {"mul1": sum(r["details"]["exl3_kbits"].values())}
    assert "verdict EXL3 + MTP (flags 0x800|0x400|0x1000 + payload):     PASS" in out
    assert "exl3_bitrates" in out
    # a bf16 MTP head (-mtp.tqf): valid file, but not what --spec needs
    flags2 = 0xc1d
    header2 = it.read_header(io.BytesIO(make_header(flags2)))
    p2 = _write(tmp_path, "exl3-mtp.tqf", make_header(flags2), _exl3_payload(header2, 6, mtp_exl3=False))
    r2 = it.inspect(p2, verbose=False)
    assert r2["payload_ok"] and not r2["exl3_mtp"]
    assert "bf16 MTP head" in it.verdict_exl3_mtp(flags2, True)[1]
    assert "no MTP section" in it.verdict_exl3_mtp(it.FLAG_EXL3_WEIGHTS | 0x1d, True)[1]
    # a record header that contradicts the layout is an error, not a silent size
    bad = bytearray(payload)
    first_qm8 = 2 * (256 * 64 + 64)      # right after embed + norm comes lm_head's record
    bad[first_qm8 : first_qm8 + 16] = struct.pack(it.EXL3_RECORD_FMT, 5, 64, 999, 2)
    p3 = _write(tmp_path, "exl3-bad.tqf", make_header(flags), bytes(bad))
    with pytest.raises(ValueError):
        it.inspect(p3, verbose=False)


def test_cli_exit_codes(tmp_path):
    import subprocess

    flags = 0x53d
    header = it.read_header(io.BytesIO(make_header(flags)))
    expected, _, _ = it.walk_payload(header)
    p = _write(tmp_path, "fp6.tqf", make_header(flags), b"\0" * expected)
    script = os.path.join(TOOLS, "inspect_tqf.py")
    assert subprocess.run([sys.executable, script, str(p)], capture_output=True).returncode == 0
    assert subprocess.run([sys.executable, script, str(p), "--expect", "fp6"], capture_output=True).returncode == 0
    assert subprocess.run([sys.executable, script, str(p), "--expect", "exl3-mtp"], capture_output=True).returncode == 2
    p2 = _write(tmp_path, "short.tqf", make_header(flags), b"\0" * (expected - 1))
    assert subprocess.run([sys.executable, script, str(p2)], capture_output=True).returncode == 1
