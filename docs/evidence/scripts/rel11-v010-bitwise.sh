#!/usr/bin/env bash
# v0.11 release claim "the decode path is v0.10's bit for bit": the v0.11 library (build-qwen/, 158edd06 = main dd6d21e) with
# TQ_PATTN_GQA=0 against the v0.10 library (484bdb82, build-qwen/libforward_qwen.so.50b7e3d = b2e7a3f), paged_logit_ab
# N = 3 and 8 (prefill waves + 25 decode steps) on the v1 opt-5.75 file (v0.10 cannot read version 2), and the v0.11 library
# with its defaults against both (expected: different -- the prefill attention -- unless the prefill wave skips it). GPU 0.
# usage: nohup logs/rel11-v010-bitwise.sh > logs/rel11-v010-bitwise.log 2>&1 &        (~4 min)
set -u
Q=/mnt/nvme/qwen256k; PY=/root/work/venv/bin/python
NEW=$Q/qwentin/build-qwen/libforward_qwen.so
V010=$Q/qwentin/build-qwen/libforward_qwen.so.50b7e3d
T575=/mnt/nvme/models/qwen3_8-27b-exl3-opt5.75-mtp.tqf
GPU=0; LOCK=/tmp/exl3diff-gpu$GPU.lock
cd $Q; mkdir -p runs/rel11
fail() { echo "[rel11-v010] FAIL: $*   $(date -u +%T)"; exit 1; }
unset TQ_KV_T4_CODEBOOK TQ_PATTN_GQA TQ_GDN_PREFILL TQ_GRAPH TQ_GEMM_FORK TQ_GEMM_FORK_MAXN TQ_VERIFY_TQF
echo "[rel11-v010] start $(date -u +%T); v0.11 $(md5sum < $NEW | cut -c1-12), v0.10 $(md5sum < $V010 | cut -c1-12)"
export CUDA_VISIBLE_DEVICES=$GPU
B="TQ_PDL=1 TQ_KV_V4=1 TQ_KV_T4=1 TQ_NO_MTP=1"
P() { lib=$1; out=$2; n=$3; shift 3
  env $B "$@" flock $LOCK timeout 900 $PY -u $Q/qwentin/tools/paged_logit_ab.py $lib runs/rel11/$out.pt --n $n --tqf $T575 > logs/rel11-$out.log 2>&1 || fail "plab $out"; }
for N in 3 8; do P $V010 v010-n$N $N; P $NEW v011-gqa0-n$N $N TQ_PATTN_GQA=0; P $NEW v011-n$N $N; done
$PY - <<'PY' || fail "compare"
import torch, sys
Q = "/mnt/nvme/qwen256k/runs/rel11"; ok = True
def eq(a, b):
    A = torch.load(f"{Q}/{a}.pt"); B = torch.load(f"{Q}/{b}.pt")
    same = A.keys() == B.keys() and all(torch.equal(A[k], B[k]) for k in A)
    d = max((A[k].float() - B[k].float()).abs().max().item() for k in A) if A.keys() == B.keys() else float("nan")
    return same, d, len(A)
for N in (3, 8):
    s, d, n = eq(f"v011-gqa0-n{N}", f"v010-n{N}"); ok &= s
    print(f"[plab] N={N} v0.11 + TQ_PATTN_GQA=0 vs v0.10: bitwise {s} (max |d| {d:.3g}, {n} tensors)")
    s2, d2, _ = eq(f"v011-n{N}", f"v010-n{N}")
    print(f"[plab] N={N} v0.11 defaults vs v0.10: bitwise {s2} (max |d| {d2:.3g}) -- informational")
print("[rel11-v010]", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
PY
echo "[rel11-v010] done $(date -u +%T)"
