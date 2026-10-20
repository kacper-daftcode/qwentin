#!/usr/bin/env bash
# cand-1207 = main 6a15ea3 (v0.11) + dd27894 (tools: the single-stream path's tools retired) + 32ab3c4 (engine: the
# single-stream path retired -- 138 -> 55 entry points, 398 -> 211 kernels; TQ_LINEAR_CORE_* and the test decoder's
# switches hard-coded) + 843cd59 (docs). Owner, 10-20: retire serve_openai.py, delete the residual-map hook and Dogs,
# keep qwn_decode as the test oracle, retire batched_decode. Candidate = qwentin-ret/build-qwen-ret/libforward_qwen.so
# .cand1207. GPU 0, ports 8150 / 8151.
#  Gate 0  CPU: ruff + pytest (and with the candidate library); the candidate exports exactly qwentin.h's entry points.
#  Gate 1  per-kernel SASS: 211 kernels left, 210 identical to the served library, the one that differs is
#          k_tq_full_attn_decode (the test decoder's attention, gate 3); plab N=3/8, the 32K variant and plab --mtp-on
#          (the paged drafter's trunk hooks in every step) bitwise on the served env.
#  Gate 2  spec exactness 4K + 32K.
#  Gate 3  the test decoder: analysis/oracle_ab.py old vs new bitwise (FP6 file: teacher-forced argmax, greedy, the paged
#          logits after its KV); tools/paged_smoke.py old vs new: the same token lines; serve_batched.py --selftest runs
#          through on both (8 clients x 64 tokens; its matches vary with timing).
#  Gate 4  FP6 on the batched path: plab N=3/8 with the FP6 file on the Q4 pool bitwise old vs new; the server on the FP6
#          file (--no-kv-t4 --no-kv-v4 --no-spec): needle 32K 3/3, the same answers with both libraries.
#  Gate 5  THE SERVED LINE (v2 file), pure defaults: 576000 tokens, needle 32K + 128K 6/6, spec A/B 10/10 and its texts
#          == cand16's, speed N=1/8.
#  NOT landed by this script.
# usage: nohup logs/cand17-all.sh > logs/cand17-all.log 2>&1 &        (~40 min)
set -u
Q=/mnt/nvme/qwen256k; PY=/root/work/venv/bin/python; CIV=/root/work/ci-venv/bin
W=$Q/qwentin-ret
LIB=$W/build-qwen-ret/libforward_qwen.so.cand1207
OLD=$Q/qwentin/build-qwen/libforward_qwen.so
T575=/mnt/nvme/models/qwen3_8-27b-exl3-opt5.75-mtp.tqf
V2=/mnt/nvme/models/qwen3_8-27b-exl3-opt5.75-mtp4-v2.tqf
FP6=/mnt/nvme/models/qwen3_8-27b-e2m3-mtp.tqf
MD=/mnt/nvme/models/Qwen3.8-27B-FP8
GPU=0; LOCK=/tmp/exl3diff-gpu$GPU.lock
FILT='^\[paged\]|Warning|warn|layer .*loaded|^TQ_|^TQF1|^KV prefix|^attention|^context|MTP head|transformers|mtp_layer'
cd $Q; mkdir -p runs/cand17
fail() { echo "[cand17-all] FAIL: $*   $(date -u +%T)"; exit 1; }
unset TQ_KV_T4_CODEBOOK TQ_KV_T4 TQ_NO_MTP TQ_KV_V4 TQ_PDL TQ_PAGED_ATTN_ST TQ_EMBED_HOST TQ_EMBED_FP8 TQ_PATTN_GQA TQ_GDN_PREFILL \
      TQ_CTX TQ_KV_Q4 TQ_PAGED_ATTN_MMA TQ_GEMM_FORK_MAXN TQ_GEMM_FORK TQ_GRAPH TQ_VERIFY_TQF
echo "[cand17-all] start $(date -u +%T); cand $(md5sum < $LIB | cut -c1-12) = $(git -C $W log --oneline -1 cand-1207 | cut -c1-60); served $(md5sum < $OLD | cut -c1-12)"

echo "== gate 0: ruff + pytest, the exported ABI   $(date -u +%T)"
git -C $W diff --quiet cand-1207 || fail "the worktree differs from cand-1207"
(cd $W && $CIV/ruff check tools/ tests/ && $CIV/python -m pytest -q tests/ 2>&1 | tail -n 1) || fail "gate 0 (CPU)"
(cd $W && QWENTIN_LIB=$LIB $CIV/python -m pytest -q tests/ 2>&1 | grep -E "passed|failed|error" | tail -n 1) | tee logs/cand17-pytest-lib.log
grep -q "25 passed" logs/cand17-pytest-lib.log && ! grep -q "failed\|error" logs/cand17-pytest-lib.log || fail "gate 0 (with the library)"
nm -D --defined-only $LIB | awk '$2 == "T" && $3 ~ /^qwn_/ {print $3}' | sort > runs/cand17/exported.txt
grep -oE "\bqwn_[a-z0-9_]+\(" $W/src/qwentin.h | tr -d '(' | sort -u > runs/cand17/header.txt
echo "   exported qwn_*: $(wc -l < runs/cand17/exported.txt), qwentin.h: $(wc -l < runs/cand17/header.txt)"
cmp -s runs/cand17/exported.txt runs/cand17/header.txt || { diff runs/cand17/exported.txt runs/cand17/header.txt | head; fail "the exports differ from qwentin.h"; }
echo "[gate0] PASS"

export CUDA_VISIBLE_DEVICES=$GPU
echo "== gate 1: per-kernel SASS + bitwise vs the served library   $(date -u +%T)"
$PY analysis/sass_diff.py $OLD $LIB > logs/cand17-sass.log; head -n 1 logs/cand17-sass.log
grep -q "^kernels: old 398, new 211, common 211; identical 210, different 1$" logs/cand17-sass.log || fail "SASS summary"
grep -q "DIFFERENT: _Z21k_tq_full_attn_decode" logs/cand17-sass.log || fail "the differing kernel is not k_tq_full_attn_decode"
grep -q "only in new" logs/cand17-sass.log && fail "a kernel only in the candidate"
echo "   removed kernels: $(grep -c 'only in old' logs/cand17-sass.log)"
B="TQ_PDL=1 TQ_KV_V4=1 TQ_KV_T4=1 TQ_NO_MTP=1"
P() { lib=$1; out=$2; shift 2
  env $B flock $LOCK timeout 900 $PY -u $W/tools/paged_logit_ab.py $lib runs/cand17/$out.pt "$@" > logs/cand17-$out.log 2>&1 || fail "plab $out"; }
for N in 3 8; do P $OLD plab-old-n$N --n $N --tqf $T575; P $LIB plab-new-n$N --n $N --tqf $T575; done
P $OLD plab-old-mtp --n 3 --tqf $V2 --mtp-on; P $LIB plab-new-mtp --n 3 --tqf $V2 --mtp-on
C() { lib=$1; out=$2
  env $B QWENTIN_TOOLS=$3 flock $LOCK timeout 1800 $PY -u analysis/paged_logit_ab_ctx.py $lib runs/cand17/$out.pt --n 2 --ctx 32768 --steps 16 --tqf $T575 > logs/cand17-$out.log 2>&1 || fail "ctx $out"; }
C $OLD ctx32k-old $Q/qwentin/tools; C $LIB ctx32k-new $W/tools
$PY - <<'PY' || fail "gate 1"
import torch, sys
Q = "/mnt/nvme/qwen256k/runs/cand17"; ok = True
def cmp(a, b, label):
    global ok
    A = torch.load(f"{Q}/{a}.pt"); B = torch.load(f"{Q}/{b}.pt")
    s = A.keys() == B.keys() and all(torch.equal(A[k], B[k]) for k in A)
    ok &= s; print(f"[plab] {label}: bitwise {s} ({len(A)} tensors)")
for N in (3, 8): cmp(f"plab-new-n{N}", f"plab-old-n{N}", f"N={N} cand-1207 vs served")
cmp("plab-new-mtp", "plab-old-mtp", "N=3 --mtp-on (v2 file), cand-1207 vs served")
cmp("ctx32k-new", "ctx32k-old", "32K x 2 slots + 16 steps, cand-1207 vs served")
print("[gate1]", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
PY

echo "== gate 2: spec exactness 4K + 32K   $(date -u +%T)"
R() { tag=$1; shift
  env TQ_KV_T4=1 flock $LOCK timeout 3600 $PY -u $W/tools/spec_paged_check.py --lib $LIB --tqf $T575 --model-dir $MD "$@" 2>&1 | grep -Ev "$FILT" > logs/cand17-spec-$tag.log
  echo "   $tag: $(grep -E '^\[spec-check\] (unpinned|ALL|[0-9]+ FAIL)' logs/cand17-spec-$tag.log | tr '\n' ' ')"
  grep -q "ALL PASS" logs/cand17-spec-$tag.log || fail "spec_paged_check $tag"; }
R 4k  --ctx 4096  --n 1,3,8 --rounds 6 --no-pin
R 32k --ctx 32768 --n 1,3,8 --rounds 6 --distinct 3 --no-pin
echo "[gate2] PASS"

echo "== gate 3: the test decoder (FP6 file), old vs new   $(date -u +%T)"
O() { lib=$1; out=$2
  flock $LOCK timeout 900 $PY -u analysis/oracle_ab.py $lib runs/cand17/$out.pt --tqf $FP6 > logs/cand17-$out.log 2>&1 || fail "oracle $out"; }
O $OLD oracle-old; O $LIB oracle-new
$PY analysis/oracle_ab.py --compare runs/cand17/oracle-new.pt runs/cand17/oracle-old.pt || fail "oracle_ab"
S() { lib=$1; tag=$2
  env TQ_CTX=8192 TQ_KV_Q4=1 flock $LOCK timeout 900 $PY -u $W/tools/paged_smoke.py --lib $lib --tqf $FP6 --model-dir $MD > logs/cand17-smoke-$tag.log 2>&1 || fail "paged_smoke $tag"
  grep -E "^  (slot|client) [0-9]+" logs/cand17-smoke-$tag.log > runs/cand17/smoke-$tag.txt; }
S $OLD old; S $LIB new
echo "   paged_smoke: $(grep -E 'argmax match|seed match' logs/cand17-smoke-new.log | tr '\n' ' ')"
cmp -s runs/cand17/smoke-old.txt runs/cand17/smoke-new.txt || { diff runs/cand17/smoke-old.txt runs/cand17/smoke-new.txt | head; fail "paged_smoke lines differ"; }
[ -s runs/cand17/smoke-new.txt ] || fail "paged_smoke printed no token lines"
T() { lib=$1; tag=$2
  env TQ_CTX=8192 TQ_KV_Q4=1 flock $LOCK timeout 1200 $PY -u $W/tools/serve_batched.py --selftest --lib $lib --tqf $FP6 --model-dir $MD \
      --max-slots 8 --num-blocks 512 --no-kv-t4 --no-kv-v4 --no-spec > logs/cand17-selftest-$tag.log 2>&1 || fail "selftest $tag"
  grep -E "^  client [0-9]+: gen=" logs/cand17-selftest-$tag.log > runs/cand17/selftest-$tag.txt; }
T $OLD old; T $LIB new
sed 's/^/   /' runs/cand17/selftest-new.txt | cut -c1-110
# --selftest staggers its clients by wall time and the FP6 path picks GEMV vs wide GEMM by column count, so its
# leading-match counts vary run to run (the served library twice: logs/cand17-selftest-old-{a,b}.log); it must run
# through with every client at gen=64 on both libraries. The bitwise evidence is oracle_ab, paged_smoke and gate 4.
for t in old new; do [ "$(grep -c "gen=64 " runs/cand17/selftest-$t.txt)" -eq 8 ] || fail "selftest $t: not 8 clients at gen=64"; done
echo "   paged_smoke: the same lines with both libraries; --selftest: 8 clients x 64 tokens with both"
echo "[gate3] PASS"

echo "== gate 4: FP6 on the batched path   $(date -u +%T)"
F() { lib=$1; out=$2; n=$3
  env TQ_PDL=1 TQ_NO_MTP=1 flock $LOCK timeout 900 $PY -u $W/tools/paged_logit_ab.py $lib runs/cand17/$out.pt --n $n --tqf $FP6 > logs/cand17-$out.log 2>&1 || fail "plab $out"; }
for N in 3 8; do F $OLD fp6-old-n$N $N; F $LIB fp6-new-n$N $N; done
$PY - <<'PY' || fail "gate 4 plab"
import torch, sys
Q = "/mnt/nvme/qwen256k/runs/cand17"; ok = True
for N in (3, 8):
    A = torch.load(f"{Q}/fp6-new-n{N}.pt"); B = torch.load(f"{Q}/fp6-old-n{N}.pt")
    s = A.keys() == B.keys() and all(torch.equal(A[k], B[k]) for k in A)
    ok &= s; print(f"[plab] FP6 file, Q4 pool, N={N}: cand-1207 vs served bitwise {s} ({len(A)} tensors)")
sys.exit(0 if ok else 1)
PY
FS() { lib=$1; tag=$2; port=$3
  exec 9>$LOCK; flock 9
  env TQ_CTX=262144 TQ_KV_Q4=1 TQ_EMBED_FP8=2 TQ_PAGED_ATTN_MMA=1 nohup $PY -u $W/tools/serve_batched.py --port $port --max-slots 8 --num-blocks 2112 \
      --timeout 1800 --model-name qwen3.8-27b --tqf $FP6 --model-dir $MD --lib $lib --no-kv-t4 --no-kv-v4 --no-spec > logs/cand17-fp6-server-$tag.log 2>&1 &
  local spid=""; for i in $(seq 1 120); do sleep 5; curl -s -m 2 http://127.0.0.1:$port/health > /dev/null 2>&1 && { spid=$(pgrep -f "serve_batched.py --port $port --max-slots [0-9]" | head -1); break; }; done
  [ -n "$spid" ] || { tail -5 logs/cand17-fp6-server-$tag.log; exec 9>&-; fail "FP6 server ($tag) did not come up"; }
  timeout 1800 $PY eval/code_needle.py --base-url http://127.0.0.1:$port --model qwen3.8-27b --lengths 32768 --depths 0.95,0.5,0.05 --decoys 3 \
    --tokenizer $MD --chat-kwargs '{"enable_thinking": false}' --out runs/cand17/fp6-needle-$tag.json > logs/cand17-fp6-needle-$tag.log 2>&1
  kill $spid; for i in $(seq 1 30); do sleep 2; kill -0 $spid 2>/dev/null || break; done
  exec 9>&-
  grep -E "\] (PASS|FAIL)" logs/cand17-fp6-needle-$tag.log | sed -E "s/ttft=[0-9.]+s e2e=[0-9.]+s //" > runs/cand17/fp6-needle-$tag.txt; }
FS $OLD old 8150; FS $LIB new 8151
grep -E "\] (PASS|FAIL)" logs/cand17-fp6-needle-new.log | cut -c1-100 | sed 's/^/   /'
[ "$(grep -c '\] PASS' runs/cand17/fp6-needle-new.txt)" -eq 3 ] || fail "FP6 needle"
cmp -s runs/cand17/fp6-needle-old.txt runs/cand17/fp6-needle-new.txt || { diff runs/cand17/fp6-needle-old.txt runs/cand17/fp6-needle-new.txt; fail "FP6 needle answers differ"; }
echo "   FP6 server: needle 3/3, the same answers as the served library"
echo "[gate4] PASS"

echo "== gate 5: THE SERVED LINE (v2 file), pure defaults   $(date -u +%T)"
PORT=8152
export TQ_CTX=262144 TQ_KV_Q4=1 TQ_EMBED_FP8=2 TQ_PAGED_ATTN_MMA=1 PROBE_PORT=$PORT
exec 9>$LOCK; flock 9
nohup $PY -u $W/tools/serve_batched.py --port $PORT --max-slots 12 --num-blocks 4500 --timeout 1800 --model-name qwen3.8-27b \
  --tqf $V2 --model-dir $MD --lib $LIB > logs/cand17-server.log 2>&1 &
SPID=""; for i in $(seq 1 120); do sleep 5; curl -s -m 2 http://127.0.0.1:$PORT/health > /dev/null 2>&1 && { SPID=$(pgrep -f "serve_batched.py --port $PORT --max-slots [0-9]" | head -1); break; }; done
[ -n "$SPID" ] || { tail -5 logs/cand17-server.log; fail "server did not come up"; }
echo "   server up after $((i * 5)) s, pid $SPID, $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -v p=$SPID -F', ' '$1==p {print $2}') MiB"
grep -oE "[0-9]+ pooled tokens" logs/cand17-server.log | sed 's/^/   /'
grep -q "576000 pooled tokens" logs/cand17-server.log || { kill $SPID; fail "the pool is not 576000 tokens"; }
timeout 3000 $PY eval/code_needle.py --base-url http://127.0.0.1:$PORT --model qwen3.8-27b --lengths 32768,131072 --depths 0.95,0.5,0.05 --decoys 3 \
  --tokenizer $MD --chat-kwargs '{"enable_thinking": false}' --out runs/cand17/needle.json > logs/cand17-needle.log 2>&1; nrc=$?
grep -E "\] (PASS|FAIL)" logs/cand17-needle.log | cut -c1-100
NP=$(grep -c "\] PASS" logs/cand17-needle.log); NF=$(grep -c "\] FAIL" logs/cand17-needle.log)
timeout 1800 $PY analysis/spec_server_ab.py --base http://127.0.0.1:$PORT --prompts 10 --max-tokens 300 --out runs/cand17/spec-ab.json > logs/cand17-spec-ab.log 2>&1; abrc=$?
grep -E "identical outputs" logs/cand17-spec-ab.log; AB=$(grep -c "identical outputs: 10/10" logs/cand17-spec-ab.log)
$PY - <<'PY'; ABT=$?
import json, sys
R = "/mnt/nvme/qwen256k/runs"
a = json.load(open(f"{R}/cand17/spec-ab.json")); b = json.load(open(f"{R}/cand16/spec-ab.json"))
same_p = a["prompts"] == b["prompts"]
same = {m: [x["text"] for x in a["res"][m]] == [x["text"] for x in b["res"][m]] for m in ("on", "off")}
print(f"[ab-vs-v011] prompts identical {same_p}; texts identical to cand16 (the v0.11 library, same file): on {same['on']}, off {same['off']}")
sys.exit(0 if same_p and all(same.values()) else 1)
PY
timeout 2400 $PY analysis/spec_speed.py --base http://127.0.0.1:$PORT --ctx 32768 --n 1,8 --modes on --max-tokens 300 --out runs/cand17/spec-speed-32k.json > logs/cand17-spec-speed-32k.log 2>&1; src=$?
grep -E "\[speed\] ctx" logs/cand17-spec-speed-32k.log | cut -c1-150
kill $SPID; for i in $(seq 1 30); do sleep 2; kill -0 $SPID 2>/dev/null || break; done
ERR=$(grep -ciE "xid|error|traceback" logs/cand17-server.log)
echo "   server log error lines: $ERR; needle PASS $NP FAIL $NF; A/B identical 10/10: $AB, texts = cand16: $((1 - ABT)); speed rc $src"
[ "$nrc" -eq 0 ] && [ "$NF" -eq 0 ] && [ "$NP" -ge 6 ] && [ "$abrc" -eq 0 ] && [ "$AB" -eq 1 ] && [ "$ABT" -eq 0 ] && [ "$src" -eq 0 ] && [ "$ERR" -eq 0 ] \
  || fail "gate 5 (needle rc=$nrc PASS=$NP FAIL=$NF, ab rc=$abrc identical=$AB texts=$ABT, speed rc=$src, err=$ERR)"
echo "[gate5] PASS"
exec 9>&-
echo "== ALL PASS   $(date -u +%T)  (landing: build-qwen/ := the candidate, main := cand-1207 (ff), patches 0111-0113)"
