#!/usr/bin/env bash
# cand-1222 = main 433ee7d + (1) the trellis page encoder's step loop without the per-step shuffle chain (k_t4_encode /
# k_t4_encode_dev: the 1024-tile launch 235 -> 178 us; codes bitwise on 8192 fixed rows, analysis/t4_encode_ab.py) and
# (2) the prefill wave on the verify wave's fused launch set (add + post norm, the final add + the next layer's input
# norm, one fragment build per shared input: 208 fewer launches per wave; host code). Both bitwise: the 4-wave logits
# at 32K / 128K = the base dumps (logs/s17-fuse-ab-encfuse.log: the wave 46.87 -> 45.87 ms at 32K, 71.85 -> 70.89 at
# 128K), plab N=3 = the served (logs/s17-enc-gpu3.log). Candidate qwentin-enc/build-enc/. GPU ${GPU:-2}, port 8221.
#  Gate 0  CPU tests (and with the library); exports = qwentin.h (61).
#  Gate 1  SASS: 217 / 219 identical, the 2 that differ are k_t4_encode and k_t4_encode_dev; plab N=3/8 + --mtp-on
#          BITWISE vs the served library.
#  Gate 2  spec exactness vs plain decode: 4K / 32K / 300 + the 40 / 28 / 18-column fixed-K waves.
#  Gate 3  THE SERVED LINE: needle 6/6 (TTFT: expect ~ -0.25 s at 32K, ~ -1 s at 128K), A/B 10/10 WITH the texts ==
#          cand21's, repro, speed N = 1 / 2 / 8 / 12 @32K + 128K (unchanged: decode-side kernels untouched), T1 == 93.9 / 91.5.
#  NOT landed by this script.
# usage: GPU=2 nohup logs/cand32-all.sh > logs/cand32-all.log 2>&1 < /dev/null &        (~40 min)
set -u
Q=/mnt/nvme/qwen256k; PY=/root/work/venv/bin/python; CIV=/root/work/ci-venv/bin
W=$Q/qwentin-enc
LIB=$W/build-enc/libforward_qwen.so
OLD=$Q/qwentin/build-qwen/libforward_qwen.so
T575=/mnt/nvme/models/qwen3_8-27b-exl3-opt5.75-mtp.tqf
DV=/mnt/nvme/models/qwen3_8-27b-exl3-opt5.75-mtp4-dv64k.tqf
MD=/mnt/nvme/models/Qwen3.8-27B-FP8
GPU=${GPU:-2}; LOCK=/tmp/exl3diff-gpu$GPU.lock; PORT=8221
FILT="Token indices|indexing errors|^\[config\]|^\[tqf\]|^\[spec\]|^\[graph\]|^\[kv\]|^\[embed\]|^\[mtp\]|^\[paged\]|^\[exl3\]|^$"
cd $Q; mkdir -p runs/cand32
fail() { echo "[cand32-all] FAIL: $*   $(date -u +%T)"; exit 1; }
unset $(env | grep -oE '^TQ_[A-Z0-9_]+')
echo "[cand32-all] start $(date -u +%T); cand $(md5sum < $LIB | cut -c1-12) = $(git -C $W log --oneline -1 | cut -c1-60); served $(md5sum < $OLD | cut -c1-12)"

echo "== gate 0: ruff + pytest, the exported ABI   $(date -u +%T)"
git -C $W diff --quiet HEAD || fail "the worktree has uncommitted changes"
(cd $W && $CIV/ruff check tools/ tests/ && $CIV/python -m pytest -q tests/ 2>&1 | tail -n 1) || fail "gate 0 (CPU)"
(cd $W && CUDA_VISIBLE_DEVICES=$GPU QWENTIN_LIB=$LIB flock $LOCK $CIV/python -m pytest -q tests/ 2>&1 | grep -E "passed|failed|error" | tail -n 1) | tee logs/cand32-pytest-lib.log
grep -q "84 passed" logs/cand32-pytest-lib.log && ! grep -q "failed\|error" logs/cand32-pytest-lib.log || fail "gate 0 (with the library)"
nm -D --defined-only $LIB | awk '$2 == "T" && $3 ~ /^qwn_/ {print $3}' | sort > runs/cand32/exported.txt
grep -oE "\bqwn_[a-z0-9_]+\(" $W/src/qwentin.h | tr -d '(' | sort -u > runs/cand32/header.txt
cmp -s runs/cand32/exported.txt runs/cand32/header.txt || fail "the exports differ from qwentin.h"
[ "$(wc -l < runs/cand32/exported.txt)" = 61 ] || fail "expected 61 exports"
echo "[gate0] PASS (61 exports)"

export CUDA_VISIBLE_DEVICES=$GPU
echo "== gate 1: SASS + bitwise   $(date -u +%T)"
$PY analysis/sass_diff.py $OLD $LIB > logs/cand32-sass.log; head -n 1 logs/cand32-sass.log
grep -q "^kernels: old 219, new 219, common 219; identical 217, different 2$" logs/cand32-sass.log || fail "SASS summary"
[ "$(grep -cE 'DIFFERENT: .*k_t4_encode' logs/cand32-sass.log)" = 2 ] || fail "a kernel outside the trellis encoder differs"
P() { lib=$1; out=$2; tools=$3; envs=$4; shift 4
  env $envs flock $LOCK timeout 900 $PY -u $tools/paged_logit_ab.py $lib runs/cand32/$out.pt "$@" > logs/cand32-$out.log 2>&1 || fail "plab $out"; }
for N in 3 8; do P $OLD plab-old-n$N $Q/qwentin/tools "TQ_NO_MTP=1" --n $N --tqf $T575; P $LIB plab-new-n$N $W/tools "TQ_NO_MTP=1" --n $N --tqf $T575; done
P $OLD plab-old-mtp $Q/qwentin/tools "NOENV=1" --n 3 --tqf $DV --mtp-on
P $LIB plab-new-mtp $W/tools "NOENV=1" --n 3 --tqf $DV --mtp-on
$PY - <<'PY' || fail "gate 1"
import torch, sys
Q = "/mnt/nvme/qwen256k/runs/cand32"; ok = True
for a, b, label in (("plab-new-n3", "plab-old-n3", "N=3"), ("plab-new-n8", "plab-old-n8", "N=8"), ("plab-new-mtp", "plab-old-mtp", "N=3 --mtp-on (dv64k file)")):
    A = torch.load(f"{Q}/{a}.pt"); B = torch.load(f"{Q}/{b}.pt")
    s = A.keys() == B.keys() and all(torch.equal(A[k], B[k]) for k in A); ok &= s
    md = max((A[k].float() - B[k].float()).abs().max().item() for k in A)
    print(f"[plab] {label} cand-1222 vs served: bitwise {s} (max |d| {md:.3e}, {len(A)} tensors)")
print("[gate1]", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
PY

echo "== gate 2: spec exactness -- the 33..64-column waves + the standard cells   $(date -u +%T)"
R() { tag=$1; shift
  flock $LOCK timeout 3600 $PY -u $W/tools/spec_paged_check.py --lib $LIB --tqf $T575 --model-dir $MD "$@" 2>&1 | grep -Ev "$FILT" > logs/cand32-spec-$tag.log
  echo "   $tag: $(grep -E '^\[spec-check\] (unpinned|ALL|[0-9]+ FAIL)' logs/cand32-spec-$tag.log | tr '\n' ' ' | cut -c1-200)"
  grep -q "ALL PASS" logs/cand32-spec-$tag.log || fail "spec_paged_check $tag"; }
R 4k-n8k4  --ctx 4096  --n 8  --kmax 4 --k-fixed --rounds 6 --no-pin
R 4k-n4k6  --ctx 4096  --n 4  --kmax 6 --k-fixed --rounds 6 --no-pin
R 4k-n2k8  --ctx 4096  --n 2  --kmax 8 --k-fixed --rounds 6 --no-pin
R 4k  --ctx 4096  --n 1,3,8 --rounds 6 --no-pin
R 32k --ctx 32768 --n 1,3,8 --rounds 6 --distinct 3 --no-pin
R 300 --ctx 300 --n 1,3 --rounds 40 --no-pin
echo "[gate2] PASS"

echo "== gate 3: THE SERVED LINE (dv64k file), K auto = 6/3/3, one-pass wave 64   $(date -u +%T)"
export PROBE_PORT=$PORT
exec 9>$LOCK; flock 9
serve() { lib=$1; log=$2
  nohup $PY -u $W/tools/serve_batched.py --port $PORT --max-slots 12 --num-blocks 4500 --timeout 1800 --model-name qwen3.8-27b \
    --tqf $DV --model-dir $MD --lib $lib > $log 2>&1 &
  SPID=""; for i in $(seq 1 120); do sleep 5; curl -s -m 2 http://127.0.0.1:$PORT/health > /dev/null 2>&1 && { SPID=$(pgrep -f "serve_batched.py --port $PORT --max-slots [0-9]" | head -1); break; }; done
  [ -n "$SPID" ] || { tail -5 $log; fail "server did not come up ($log)"; }
  MIB=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -v p=$SPID -F', ' '$1==p {print $2}')
  echo "   server up after $((i * 5)) s, pid $SPID, $MIB MiB; $(grep -oE '[0-9]+ pooled tokens' $log); $(grep -oE 'speculative decoding ON \([^)]*\)' $log | head -1)"; }
stop() { kill $SPID; for i in $(seq 1 30); do sleep 2; kill -0 $SPID 2>/dev/null || break; done; }
tpr() { grep -E "\[speed\] ctx" $1 | awk -v t=$2 '{for (i = 1; i <= NF; i++) { if ($i == "tokens" && $(i+1) == "in") tok = $(i-1); if ($i == "rounds") r = $(i+1) } printf "   %s %s %s: %.2f tokens per round (%s tokens / %s rounds)\n", t, $2, $3, tok / r, tok, r }'; }
serve $LIB logs/cand32-server.log
grep -q "K=auto = 6/3/3 by active slots; draft vocabulary 65536 ids; one-pass verify wave 64 columns" logs/cand32-server.log || { stop; fail "the startup line does not say 6/3/3 with a 64-column one-pass wave"; }
curl -s http://127.0.0.1:$PORT/health | grep -oE '"(draft_vocab|verify_onepass)": ?[0-9]+' | sed 's/^/   \/health /'
curl -s http://127.0.0.1:$PORT/health | grep -qE '"verify_onepass": ?64' || { stop; fail "/health verify_onepass"; }
$Q/venv-eval/bin/python analysis/t1_repro.py --base http://127.0.0.1:$PORT --out runs/cand32/repro.json > logs/cand32-repro.log 2>&1; rrc=$?
grep "garbage" logs/cand32-repro.log | tail -1 | sed 's/^/   /'; RG=$(grep -c "garbage 0 / 6" logs/cand32-repro.log)
timeout 3000 $PY eval/code_needle.py --base-url http://127.0.0.1:$PORT --model qwen3.8-27b --lengths 32768,131072 --depths 0.95,0.5,0.05 --decoys 3 \
  --tokenizer $MD --chat-kwargs '{"enable_thinking": false}' --out runs/cand32/needle.json > logs/cand32-needle.log 2>&1; nrc=$?
grep -E "\] (PASS|FAIL)" logs/cand32-needle.log | cut -c1-100
NP=$(grep -c "\] PASS" logs/cand32-needle.log); NF=$(grep -c "\] FAIL" logs/cand32-needle.log)
timeout 1800 $PY analysis/spec_server_ab.py --base http://127.0.0.1:$PORT --prompts 10 --max-tokens 300 --out runs/cand32/spec-ab.json > logs/cand32-spec-ab.log 2>&1; abrc=$?
grep -E "identical outputs|tokens/round" logs/cand32-spec-ab.log | cut -c1-160; AB=$(grep -c "identical outputs: 10/10" logs/cand32-spec-ab.log)
$PY - <<'PY'; ABT=$?
import json, sys
R = "/mnt/nvme/qwen256k/runs"
a = json.load(open(f"{R}/cand32/spec-ab.json")); b = json.load(open(f"{R}/cand21/spec-ab.json"))
same_p = a["prompts"] == b["prompts"]
n = {m: sum(x["text"] == y["text"] for x, y in zip(a["res"][m], b["res"][m])) for m in ("on", "off")}
print(f"[ab-vs-cand21] prompts identical {same_p}; texts identical to cand21's: on {n['on']}/10, off {n['off']}/10 (required: the library is bitwise)")
sys.exit(0 if same_p and n["on"] == 10 and n["off"] == 10 else 1)
PY
timeout 2400 $PY analysis/spec_speed.py --base http://127.0.0.1:$PORT --ctx 32768 --n 1,2,8 --modes on --max-tokens 300 --repeats 2 --out runs/cand32/spec-speed-32k.json > logs/cand32-spec-speed-32k.log 2>&1; src=$?
timeout 2400 $PY analysis/spec_speed.py --base http://127.0.0.1:$PORT --ctx 32768 --n 12 --modes on --max-tokens 300 --repeats 2 --out runs/cand32/spec-speed-32k-n12.json > logs/cand32-spec-speed-32k-n12.log 2>&1; src3=$?
timeout 2400 $PY analysis/spec_speed.py --base http://127.0.0.1:$PORT --ctx 131072 --n 1 --modes on --max-tokens 300 --out runs/cand32/spec-speed-128k.json > logs/cand32-spec-speed-128k.log 2>&1; src2=$?
grep -hE "\[speed\] ctx" logs/cand32-spec-speed-32k.log logs/cand32-spec-speed-32k-n12.log logs/cand32-spec-speed-128k.log | cut -c1-150
tpr logs/cand31-spec-speed-32k.log cand31; tpr logs/cand32-spec-speed-32k.log cand32
tpr logs/cand31-spec-speed-32k-n12.log cand31-n12; tpr logs/cand32-spec-speed-32k-n12.log cand32-n12
echo "   -- T1 HumanEval+ with spec on   $(date -u +%T)"
./eval/evalplus.sh http://127.0.0.1:$PORT qwen3.8-27b cand32 humaneval > logs/cand32-t1-humaneval.log 2>&1; echo "   humaneval exit $? $(date -u +%T): $(grep -E 'pass@1' logs/cand32-t1-humaneval.log | tr -s '\t ' ' ' | tr '\n' ' ')"
$PY - <<'PY'; t1rc=$?
import re, sys, json, glob
v = [float(x) for x in re.findall(r"pass@1:\s*([\d.]+)", open("/mnt/nvme/qwen256k/logs/cand32-t1-humaneval.log").read())]
raw = [json.loads(l) for l in open(glob.glob("/mnt/nvme/qwen256k/runs/evalplus-cand32/humaneval/*.raw.jsonl")[0])]
g = sum("!!!!!!!!" in r["solution"] for r in raw)
ok = v == [0.939, 0.915] and g == 0
print(f"   [t1] humaneval spec on, K auto 6/3/3: {v} (must equal 93.9 / 91.5: the library is bitwise); degenerate {g} / {len(raw)} -> {'ok' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PY
ERR=$(grep -ciE "xid|error|traceback" logs/cand32-server.log)
stop
echo "   server log error lines: $ERR; repro clean $RG; needle PASS $NP FAIL $NF; A/B 10/10: $AB, texts = cand21: $((1 - ABT)); speed rc $src/$src3/$src2; T1 rc $t1rc"
[ "$rrc" -eq 0 ] && [ "$RG" -eq 1 ] && [ "$nrc" -eq 0 ] && [ "$NF" -eq 0 ] && [ "$NP" -ge 6 ] && [ "$abrc" -eq 0 ] && [ "$AB" -eq 1 ] && [ "$ABT" -eq 0 ] && [ "$src" -eq 0 ] && [ "$src3" -eq 0 ] && [ "$src2" -eq 0 ] && [ "$t1rc" -eq 0 ] && [ "$ERR" -eq 0 ] \
  || fail "gate 3 (repro rc=$rrc clean=$RG, needle rc=$nrc PASS=$NP FAIL=$NF, ab rc=$abrc identical=$AB texts=$ABT, speed rc=$src/$src3/$src2, T1 rc=$t1rc, err=$ERR)"
echo "[gate3] PASS"
exec 9>&-
echo "== ALL PASS   $(date -u +%T)  (landing: build-qwen/ := the candidate, main := cand-1222 (ff), patches 0150-0152)"
