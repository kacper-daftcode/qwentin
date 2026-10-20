#!/usr/bin/env bash
# cand-1205 = cand-1204 + 0727690 (.tqf format version 2: the converter seals an 80-byte header extension with the payload
# size and BLAKE2b-256 hashes; the loader checks them, refuses with -40..-45, TQ_VERIFY_TQF=1 hashes the payload;
# inspect_tqf.py --verify) + 0d647f9 (docs). Host code only -> SASS identity + bitwise. Candidate = the snapshot
# qwentin-hyg4/build-qwen-hyg4/libforward_qwen.so.cand1205 (a3731a98). GPU 0 under its lock, port 8144. Runs after cand13.
#  Gate 0  CPU: ruff + pytest, also with the candidate library (the engine's BLAKE2b vs hashlib, each refusal code).
#  Gate 1  per-kernel SASS identical to the cand-1204 snapshot; plab N=3/8 bitwise vs the served binary on the v1 file;
#          plab N=3 on the re-converted v2 file == on the v1 file.
#  Gate 2  spec exactness 4K.
#  Gate 3  THE SERVED LINE on the v1 -mtp4.tqf: needle 32K + 128K 6/6, spec A/B 10/10, speed N=1/8.
#  Gate 4  the server on the v2 file: startup reports "format v2 ... header hash OK", needle 32K x3; and qwn_probe with
#          TQ_VERIFY_TQF=1 on it: payload hash OK.
#  Landing by hand after ALL PASS (with cand12 + cand13): build-qwen/ := this candidate (old kept as .364aa9c),
#  main := cand-1205 (ff), patches 0102-.
# usage: nohup logs/cand14-all.sh > logs/cand14-all.log 2>&1 &        (~16 min)
set -u
Q=/mnt/nvme/qwen256k; PY=/root/work/venv/bin/python
W=$Q/qwentin-hyg4
LIB=$W/build-qwen-hyg4/libforward_qwen.so.cand1205
OLD=$Q/qwentin/build-qwen/libforward_qwen.so
PREV=$Q/qwentin-hyg3/build-qwen-hyg3/libforward_qwen.so.cand1204
T575=/mnt/nvme/models/qwen3_8-27b-exl3-opt5.75-mtp.tqf
T575M=/mnt/nvme/models/qwen3_8-27b-exl3-opt5.75-mtp4.tqf
TV2=/mnt/nvme/models/scratch/qwen3_8-27b-exl3-opt5.75-mtp4-v2.tqf
MD=/mnt/nvme/models/Qwen3.8-27B-FP8
PORT=8144; GPU=0; LOCK=/tmp/exl3diff-gpu$GPU.lock
FILT='^\[paged\]|Warning|warn|layer .*loaded|^TQ_|^TQF1|^KV prefix|^attention|^context|MTP head|transformers|mtp_layer'
cd $Q; mkdir -p runs/cand14
fail() { echo "[cand14-all] FAIL: $*   $(date -u +%T)"; echo "[cand14-all] nothing swapped; build-qwen/ and main stay at 364aa9c"; exit 1; }
unset TQ_KV_T4_CODEBOOK TQ_KV_T4 TQ_NO_MTP TQ_KV_V4 TQ_PDL TQ_PAGED_ATTN_ST TQ_EMBED_HOST TQ_EMBED_FP8 TQ_PATTN_GQA TQ_GDN_PREFILL \
      TQ_CTX TQ_KV_Q4 TQ_PAGED_ATTN_MMA TQ_GEMM_FORK_MAXN TQ_GEMM_FORK TQ_GRAPH TQ_VERIFY_TQF
echo "[cand14-all] start $(date -u +%T); cand $(md5sum $LIB | cut -c1-12) = $(git -C $W log --oneline -1 cand-1205 | cut -c1-70); served $(md5sum $OLD | cut -c1-12)"

echo "== gate 0: ruff + pytest (CPU, then with the candidate library)   $(date -u +%T)"
git -C $W diff --quiet cand-1205 || fail "the worktree differs from cand-1205"
(cd $W && /tmp/ci-venv/bin/ruff check tools/ tests/ && /tmp/ci-venv/bin/python -m pytest -q tests/ 2>&1 | tail -n 1) || fail "gate 0 (CPU)"
(cd $W && QWENTIN_LIB=$LIB /tmp/ci-venv/bin/python -m pytest -q tests/ 2>&1 | grep -E "passed|failed|error" | tail -n 1) | tee logs/cand14-pytest-lib.log
grep -q "failed\|error" logs/cand14-pytest-lib.log && fail "gate 0 (with the library)"
grep -q "43 passed" logs/cand14-pytest-lib.log || fail "gate 0: expected 43 passed with the library"
echo "[gate0] PASS"

export CUDA_VISIBLE_DEVICES=$GPU
echo "== gate 1: per-kernel SASS + bitwise   $(date -u +%T)"
$PY analysis/sass_diff.py $PREV $LIB > logs/cand14-sass.log; sassrc=$?; head -n 3 logs/cand14-sass.log; [ $sassrc -eq 0 ] || fail "SASS differs"
B="TQ_PDL=1 TQ_KV_V4=1 TQ_KV_T4=1 TQ_NO_MTP=1"
P() { lib=$1; out=$2; n=$3; tqf=$4
  env $B flock $LOCK timeout 900 $PY -u $W/tools/paged_logit_ab.py $lib runs/cand14/$out.pt --n $n --tqf $tqf > logs/cand14-$out.log 2>&1 || fail "plab $out"; }
for N in 3 8; do P $OLD plab-old-n$N $N $T575M; P $LIB plab-new-n$N $N $T575M; done
P $LIB plab-newv2-n3 3 $TV2
grep -q "TQF1 format v2: .*header hash OK" logs/cand14-plab-newv2-n3.log || fail "the v2 load did not report its header check"
$PY - <<'PY' || fail "gate 1"
import torch, sys
Q = "/mnt/nvme/qwen256k/runs/cand14"; ok = True
def cmp(a, b, label):
    global ok
    A = torch.load(f"{Q}/{a}.pt"); B = torch.load(f"{Q}/{b}.pt")
    s = A.keys() == B.keys() and all(torch.equal(A[k], B[k]) for k in A)
    ok &= s; print(f"[plab] {label}: bitwise {s} ({len(A)} tensors)")
for N in (3, 8): cmp(f"plab-new-n{N}", f"plab-old-n{N}", f"N={N} cand-1205 vs served (v1 file)")
cmp("plab-newv2-n3", "plab-new-n3", "N=3 the v2 file vs the v1 file (cand-1205)")
print("[gate1]", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
PY

echo "== gate 2: spec exactness 4K   $(date -u +%T)"
env TQ_KV_T4=1 flock $LOCK timeout 3600 $PY -u $W/tools/spec_paged_check.py --lib $LIB --tqf $T575 --model-dir $MD --ctx 4096 --n 1,3,8 --rounds 6 --no-pin 2>&1 | grep -Ev "$FILT" > logs/cand14-spec-4k.log
echo "   4k: $(grep -E '^\[spec-check\] (unpinned|ALL|[0-9]+ FAIL)' logs/cand14-spec-4k.log | tr '\n' ' ')"
grep -q "ALL PASS" logs/cand14-spec-4k.log || fail "spec_paged_check 4k"
echo "[gate2] PASS"

serve() { tag=$1; tqf=$2
  nohup $PY -u $W/tools/serve_batched.py --port $PORT --max-slots 12 --num-blocks 4500 --timeout 1800 --model-name qwen3.8-27b \
    --tqf $tqf --model-dir $MD --lib $LIB > logs/cand14-server-$tag.log 2>&1 &
  SPID=""; for i in $(seq 1 120); do sleep 5; curl -s -m 2 http://127.0.0.1:$PORT/health > /dev/null 2>&1 && { SPID=$(pgrep -f "serve_batched.py --port $PORT --max-slots [0-9]" | head -1); break; }; done
  [ -n "$SPID" ] || { tail -5 logs/cand14-server-$tag.log; fail "server ($tag) did not come up"; }
  echo "   server ($tag) up after $((i * 5)) s, pid $SPID, $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -v p=$SPID -F', ' '$1==p {print $2}') MiB"
  grep -oE "TQF1 format v[12].*|[0-9]+ pooled tokens" logs/cand14-server-$tag.log | cut -c1-110 | sed 's/^/   /'; }
stop() { kill $SPID; for i in $(seq 1 30); do sleep 2; kill -0 $SPID 2>/dev/null || break; done; }
needle() { tag=$1; lengths=$2
  timeout 3000 $PY eval/code_needle.py --base-url http://127.0.0.1:$PORT --model qwen3.8-27b --lengths $lengths --depths 0.95,0.5,0.05 --decoys 3 \
    --tokenizer $MD --chat-kwargs '{"enable_thinking": false}' --out runs/cand14/needle-$tag.json > logs/cand14-needle-$tag.log 2>&1; nrc=$?
  grep -E "\] (PASS|FAIL)" logs/cand14-needle-$tag.log | cut -c1-100
  NP=$(grep -c "\] PASS" logs/cand14-needle-$tag.log); NF=$(grep -c "\] FAIL" logs/cand14-needle-$tag.log); }

echo "== gate 3: THE SERVED LINE on the v1 file, pure defaults   $(date -u +%T)"
export TQ_CTX=262144 TQ_KV_Q4=1 TQ_EMBED_FP8=2 TQ_PAGED_ATTN_MMA=1 PROBE_PORT=$PORT
exec 9>$LOCK; flock 9
serve v1 $T575M
grep -q "576000 pooled tokens" logs/cand14-server-v1.log || { stop; fail "the pool is not 576000 tokens"; }
needle v1 32768,131072
timeout 1800 $PY analysis/spec_server_ab.py --base http://127.0.0.1:$PORT --prompts 10 --max-tokens 300 --out runs/cand14/spec-ab.json > logs/cand14-spec-ab.log 2>&1; abrc=$?
grep -E "identical outputs" logs/cand14-spec-ab.log; AB=$(grep -c "identical outputs: 10/10" logs/cand14-spec-ab.log)
timeout 2400 $PY analysis/spec_speed.py --base http://127.0.0.1:$PORT --ctx 32768 --n 1,8 --modes on --max-tokens 300 --out runs/cand14/spec-speed-32k.json > logs/cand14-spec-speed-32k.log 2>&1; src=$?
grep -E "\[speed\] ctx" logs/cand14-spec-speed-32k.log | cut -c1-110
stop
ERR=$(grep -ciE "xid|error|traceback" logs/cand14-server-v1.log)
[ "$nrc" -eq 0 ] && [ "$NF" -eq 0 ] && [ "$NP" -ge 6 ] && [ "$abrc" -eq 0 ] && [ "$AB" -eq 1 ] && [ "$src" -eq 0 ] && [ "$ERR" -eq 0 ] || fail "gate 3 (needle rc=$nrc PASS=$NP FAIL=$NF, ab rc=$abrc identical=$AB, speed rc=$src, err=$ERR)"
echo "[gate3] PASS"

echo "== gate 4: the server on the v2 file; qwn_probe with TQ_VERIFY_TQF=1   $(date -u +%T)"
serve v2 $TV2
grep -q "TQF1 format v2: .*header hash OK" logs/cand14-server-v2.log || { stop; fail "the server's load of the v2 file did not report its header check"; }
needle v2 32768
stop
ERR=$(grep -ciE "xid|error|traceback" logs/cand14-server-v2.log)
[ "$nrc" -eq 0 ] && [ "$NF" -eq 0 ] && [ "$NP" -ge 3 ] && [ "$ERR" -eq 0 ] || fail "gate 4 server (needle rc=$nrc PASS=$NP FAIL=$NF, err=$ERR)"
exec 9>&-
unset TQ_CTX TQ_KV_Q4 TQ_EMBED_FP8 TQ_PAGED_ATTN_MMA PROBE_PORT
env TQ_VERIFY_TQF=1 $PY -c "
import ctypes, sys; L = ctypes.CDLL('$LIB'); L.qwn_probe.argtypes = [ctypes.c_char_p]
sys.exit(L.qwn_probe(b'$TV2'))" > logs/cand14-probe-verify.log 2>&1; prc=$?
grep -E "TQF1 format v2" logs/cand14-probe-verify.log | cut -c1-120
[ $prc -eq 0 ] && grep -q "payload hash OK" logs/cand14-probe-verify.log || fail "gate 4 probe (rc=$prc)"
echo "[gate4] PASS"
echo "== ALL PASS   $(date -u +%T)"
echo "[cand14-all] land by hand: [ -f $OLD.364aa9c ] || cp -p $OLD $OLD.364aa9c; git -C $Q/qwentin merge --ff-only cand-1205; re-make build-qwen/,"
echo "[cand14-all]   check its SASS against $LIB, cp -p $LIB $OLD && touch $OLD; (cd $Q/qwentin && git format-patch -q --start-number 102 364aa9c..HEAD -o ../patches/)"
