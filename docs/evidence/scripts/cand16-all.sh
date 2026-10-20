#!/usr/bin/env bash
# The served line moves to the version-2 .tqf (owner, 10-20: "przenieść do /mnt/nvme/models/, przełączyć serwowaną linię i
# przepuścić przez bramkę") + the v0.11 release measurements. Library = the served build-qwen/ (md5 158edd06 = main dd6d21e,
# landed 10-20 07:4x by cand15-all.log). File: /mnt/nvme/models/qwen3_8-27b-exl3-opt5.75-mtp4-v2.tqf (moved from scratch/;
# the v1 -mtp4.tqf stays in place for rollback). GPU 0, port 8146.
#  Gate 0  CPU: the v2 file passes inspect_tqf.py --verify --expect exl3-mtp; its payload = the v1 file's byte for byte.
#  Gate 1  plab N=3/8 with the served library: the v2 file (TQ_VERIFY_TQF=1: the engine hashes the payload) == the v1 file.
#  Gate 2  THE SERVED LINE on the v2 file, pure defaults: 576000 tokens, needle 32K + 128K 6/6, spec A/B 10/10 and its texts
#          == cand15's (same library, v1 file; cand12-15 read identical texts), then the release table: spec_speed 32K
#          N=1/2/4/8 spec on + off, 128K N=1/2 spec on (the commands of final-speed.sh).
# usage: nohup logs/cand16-all.sh > logs/cand16-all.log 2>&1 &        (~35 min)
set -u
Q=/mnt/nvme/qwen256k; PY=/root/work/venv/bin/python
LIB=$Q/qwentin/build-qwen/libforward_qwen.so
V1=/mnt/nvme/models/qwen3_8-27b-exl3-opt5.75-mtp4.tqf
V2=/mnt/nvme/models/qwen3_8-27b-exl3-opt5.75-mtp4-v2.tqf
MD=/mnt/nvme/models/Qwen3.8-27B-FP8
PORT=8146; GPU=0; LOCK=/tmp/exl3diff-gpu$GPU.lock
cd $Q; mkdir -p runs/cand16
fail() { echo "[cand16-all] FAIL: $*   $(date -u +%T)"; exit 1; }
unset TQ_KV_T4_CODEBOOK TQ_KV_T4 TQ_NO_MTP TQ_KV_V4 TQ_PDL TQ_PAGED_ATTN_ST TQ_EMBED_HOST TQ_EMBED_FP8 TQ_PATTN_GQA TQ_GDN_PREFILL \
      TQ_CTX TQ_KV_Q4 TQ_PAGED_ATTN_MMA TQ_GEMM_FORK_MAXN TQ_GEMM_FORK TQ_GRAPH TQ_VERIFY_TQF
echo "[cand16-all] start $(date -u +%T); library $(md5sum < $LIB | cut -c1-12) = main $(git -C $Q/qwentin log --oneline -1 | cut -c1-70)"
[ "$(md5sum < $LIB | cut -c1-8)" = 158edd06 ] || fail "build-qwen/ is not the served 158edd06"

echo "== gate 0: the v2 file   $(date -u +%T)"
$PY $Q/qwentin/tools/inspect_tqf.py $V2 --verify --expect exl3-mtp > logs/cand16-inspect-v2.log 2>&1 || { tail -n 12 logs/cand16-inspect-v2.log; fail "inspect_tqf --verify"; }
grep -E "^format_version|^payload_bytes|^header hash|^payload hash|^verdict EXL3" logs/cand16-inspect-v2.log | sed 's/^/   /'
read H1 H2 S1 S2 < <($PY - <<PY
import os, sys
sys.path.insert(0, "$Q/qwentin")
from tools.inspect_tqf import read_header
h = [read_header(open(p, "rb")).header_bytes for p in ("$V1", "$V2")]
print(h[0], h[1], os.path.getsize("$V1") - h[0], os.path.getsize("$V2") - h[1])
PY
)
echo "   header_bytes v1 $H1, v2 $H2; payload v1 $S1, v2 $S2 bytes"
[ "$S1" = "$S2" ] || fail "the payload sizes differ"
cmp -i $H1:$H2 $V1 $V2 || fail "the payloads differ"
echo "   payloads byte-identical (cmp -i $H1:$H2)"
echo "[gate0] PASS"

export CUDA_VISIBLE_DEVICES=$GPU
echo "== gate 1: bitwise, v2 file vs v1 file, served library   $(date -u +%T)"
B="TQ_PDL=1 TQ_KV_V4=1 TQ_KV_T4=1 TQ_NO_MTP=1"
P() { tqf=$1; out=$2; n=$3; shift 3
  env $B "$@" flock $LOCK timeout 900 $PY -u $Q/qwentin/tools/paged_logit_ab.py $LIB runs/cand16/$out.pt --n $n --tqf $tqf > logs/cand16-$out.log 2>&1 || fail "plab $out"; }
for N in 3 8; do P $V1 plab-v1-n$N $N; P $V2 plab-v2-n$N $N TQ_VERIFY_TQF=1; done
grep -h "^TQF1 format" logs/cand16-plab-v2-n3.log | sed 's/^/   /'
grep -q "payload hash OK (TQ_VERIFY_TQF)" logs/cand16-plab-v2-n3.log || fail "the engine did not verify the v2 payload"
$PY - <<'PY' || fail "gate 1"
import torch, sys
Q = "/mnt/nvme/qwen256k/runs/cand16"; ok = True
for N in (3, 8):
    A = torch.load(f"{Q}/plab-v2-n{N}.pt"); B = torch.load(f"{Q}/plab-v1-n{N}.pt")
    s = A.keys() == B.keys() and all(torch.equal(A[k], B[k]) for k in A)
    ok &= s; print(f"[plab] N={N} v2 file vs v1 file: bitwise {s} ({len(A)} tensors)")
print("[gate1]", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
PY

echo "== gate 2: THE SERVED LINE on the v2 file, pure defaults + the release table   $(date -u +%T)"
export TQ_CTX=262144 TQ_KV_Q4=1 TQ_EMBED_FP8=2 TQ_PAGED_ATTN_MMA=1 PROBE_PORT=$PORT
exec 9>$LOCK; flock 9
nohup $PY -u $Q/qwentin/tools/serve_batched.py --port $PORT --max-slots 12 --num-blocks 4500 --timeout 1800 --model-name qwen3.8-27b \
  --tqf $V2 --model-dir $MD --lib $LIB > logs/cand16-server.log 2>&1 &
SPID=""; for i in $(seq 1 120); do sleep 5; curl -s -m 2 http://127.0.0.1:$PORT/health > /dev/null 2>&1 && { SPID=$(pgrep -f "serve_batched.py --port $PORT --max-slots [0-9]" | head -1); break; }; done
[ -n "$SPID" ] || { tail -5 logs/cand16-server.log; fail "server did not come up"; }
echo "   server up after $((i * 5)) s, pid $SPID, $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -v p=$SPID -F', ' '$1==p {print $2}') MiB"
grep -E "^TQF1 format|pooled tokens|trellis-4|speculative decoding" logs/cand16-server.log | cut -c1-160 | sed 's/^/   /'
grep -q "576000 pooled tokens" logs/cand16-server.log || { kill $SPID; fail "the pool is not 576000 tokens"; }
timeout 3000 $PY eval/code_needle.py --base-url http://127.0.0.1:$PORT --model qwen3.8-27b --lengths 32768,131072 --depths 0.95,0.5,0.05 --decoys 3 \
  --tokenizer $MD --chat-kwargs '{"enable_thinking": false}' --out runs/cand16/needle.json > logs/cand16-needle.log 2>&1; nrc=$?
grep -E "\] (PASS|FAIL)" logs/cand16-needle.log | cut -c1-100
NP=$(grep -c "\] PASS" logs/cand16-needle.log); NF=$(grep -c "\] FAIL" logs/cand16-needle.log)
timeout 1800 $PY analysis/spec_server_ab.py --base http://127.0.0.1:$PORT --prompts 10 --max-tokens 300 --out runs/cand16/spec-ab.json > logs/cand16-spec-ab.log 2>&1; abrc=$?
grep -E "^\[ab\]" logs/cand16-spec-ab.log; AB=$(grep -c "identical outputs: 10/10" logs/cand16-spec-ab.log)
$PY - <<'PY'; ABT=$?
import json, sys
R = "/mnt/nvme/qwen256k/runs"
a = json.load(open(f"{R}/cand16/spec-ab.json")); b = json.load(open(f"{R}/cand15/spec-ab.json"))
same_p = a["prompts"] == b["prompts"]
same = {m: [x["text"] for x in a["res"][m]] == [x["text"] for x in b["res"][m]] for m in ("on", "off")}
print(f"[ab-vs-v1] prompts identical {same_p}; texts identical to cand15 (v1 file, same library): on {same['on']}, off {same['off']}")
sys.exit(0 if same_p and all(same.values()) else 1)
PY
timeout 3600 $PY analysis/spec_speed.py --base http://127.0.0.1:$PORT --ctx 32768 --n 1,2,4,8 --modes on,off --max-tokens 300 --out runs/cand16/spec-speed-32k.json > logs/cand16-spec-speed-32k.log 2>&1; s32=$?
timeout 3600 $PY analysis/spec_speed.py --base http://127.0.0.1:$PORT --ctx 131072 --n 1,2 --modes on --max-tokens 300 --out runs/cand16/spec-speed-128k.json > logs/cand16-spec-speed-128k.log 2>&1; s128=$?
grep -hE "\[speed\] ctx" logs/cand16-spec-speed-32k.log logs/cand16-spec-speed-128k.log
kill $SPID; for i in $(seq 1 30); do sleep 2; kill -0 $SPID 2>/dev/null || break; done
ERR=$(grep -ciE "xid|error|traceback" logs/cand16-server.log)
echo "   server log error lines: $ERR; needle PASS $NP FAIL $NF; A/B identical 10/10: $AB, texts = cand15: $((1 - ABT)); speed rc $s32 / $s128"
[ "$nrc" -eq 0 ] && [ "$NF" -eq 0 ] && [ "$NP" -ge 6 ] && [ "$abrc" -eq 0 ] && [ "$AB" -eq 1 ] && [ "$ABT" -eq 0 ] && [ "$s32" -eq 0 ] && [ "$s128" -eq 0 ] && [ "$ERR" -eq 0 ] \
  || fail "gate 2 (needle rc=$nrc PASS=$NP FAIL=$NF, ab rc=$abrc identical=$AB texts=$ABT, speed rc=$s32/$s128, err=$ERR)"
echo "[gate2] PASS"
exec 9>&-
echo "== ALL PASS   $(date -u +%T)  (the served line = the v2 file; the release commit uses this log's numbers)"
