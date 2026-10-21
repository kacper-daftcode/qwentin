#!/usr/bin/env bash
# cand-1215 = main dec430c + beaa607: --spec-k auto = 6 at <= 2 active slots when the served file carries the drafter's reduced
# vocabulary (qwn_draft_vocab_size, 60 entry points), else 4 as before. Host code only. Candidate qwentin-k6/build-k6/.
# GPU 0, port 8212.
#  Gate 0  CPU tests (and with the library); exports = qwentin.h (60).
#  Gate 1  SASS 212 / 212 identical; plab N=3/8 + --mtp-on bitwise vs the served library (nothing in the engine changed
#          but a getter).
#  Gate 2  THE SERVED LINE (the dv64k file), no TQ_* env: the startup line says K=auto = 6/3/2 and draft vocabulary
#          65536; /health draft_vocab 65536; needle 6/6; A/B 10/10 with texts == cand21's (outputs are the verify's);
#          speed N = 1 / 2 / 8 @32K + N = 1 @128K (tokens per round vs cand25's K = 4: 3.85 / 18.46 at N = 1 / 8), the
#          6-prompt repro clean, T1 HumanEval+ with spec on == 93.9 / 91.5. Then the served v2 file (no section) on the
#          same server line: the startup line must say 4/3/2 and draft vocabulary 0 (the old behaviour kept).
#  NOT landed by this script.
# usage: nohup logs/cand26-all.sh > logs/cand26-all.log 2>&1 < /dev/null &        (~25 min)
set -u
Q=/mnt/nvme/qwen256k; PY=/root/work/venv/bin/python; CIV=/root/work/ci-venv/bin
W=$Q/qwentin-k6
LIB=$W/build-k6/libforward_qwen.so
OLD=$Q/qwentin/build-qwen/libforward_qwen.so
T575=/mnt/nvme/models/qwen3_8-27b-exl3-opt5.75-mtp.tqf
V2=/mnt/nvme/models/qwen3_8-27b-exl3-opt5.75-mtp4-v2.tqf
DV=/mnt/nvme/models/qwen3_8-27b-exl3-opt5.75-mtp4-dv64k.tqf
MD=/mnt/nvme/models/Qwen3.8-27B-FP8
GPU=0; LOCK=/tmp/exl3diff-gpu$GPU.lock; PORT=8212
cd $Q; mkdir -p runs/cand26
fail() { echo "[cand26-all] FAIL: $*   $(date -u +%T)"; exit 1; }
unset $(env | grep -oE '^TQ_[A-Z0-9_]+')
echo "[cand26-all] start $(date -u +%T); cand $(md5sum < $LIB | cut -c1-12) = $(git -C $W log --oneline -1 | cut -c1-60); served $(md5sum < $OLD | cut -c1-12)"

echo "== gate 0: ruff + pytest, the exported ABI   $(date -u +%T)"
git -C $W diff --quiet HEAD || fail "the worktree has uncommitted changes"
(cd $W && $CIV/ruff check tools/ tests/ && $CIV/python -m pytest -q tests/ 2>&1 | tail -n 1) || fail "gate 0 (CPU)"
(cd $W && QWENTIN_LIB=$LIB $CIV/python -m pytest -q tests/ 2>&1 | grep -E "passed|failed|error" | tail -n 1) | tee logs/cand26-pytest-lib.log
grep -q "84 passed" logs/cand26-pytest-lib.log && ! grep -q "failed\|error" logs/cand26-pytest-lib.log || fail "gate 0 (with the library)"
nm -D --defined-only $LIB | awk '$2 == "T" && $3 ~ /^qwn_/ {print $3}' | sort > runs/cand26/exported.txt
grep -oE "\bqwn_[a-z0-9_]+\(" $W/src/qwentin.h | tr -d '(' | sort -u > runs/cand26/header.txt
cmp -s runs/cand26/exported.txt runs/cand26/header.txt || fail "the exports differ from qwentin.h"
[ "$(wc -l < runs/cand26/exported.txt)" = 60 ] || fail "expected 60 exports"
echo "[gate0] PASS (60 exports)"

export CUDA_VISIBLE_DEVICES=$GPU
echo "== gate 1: SASS + bitwise   $(date -u +%T)"
$PY analysis/sass_diff.py $OLD $LIB > logs/cand26-sass.log; head -n 1 logs/cand26-sass.log
grep -q "^kernels: old 212, new 212, common 212; identical 212, different 0$" logs/cand26-sass.log || fail "SASS summary"
P() { lib=$1; out=$2; tools=$3; envs=$4; shift 4
  env $envs flock $LOCK timeout 900 $PY -u $tools/paged_logit_ab.py $lib runs/cand26/$out.pt "$@" > logs/cand26-$out.log 2>&1 || fail "plab $out"; }
for N in 3 8; do P $OLD plab-old-n$N $Q/qwentin/tools "TQ_NO_MTP=1" --n $N --tqf $T575; P $LIB plab-new-n$N $W/tools "TQ_NO_MTP=1" --n $N --tqf $T575; done
P $OLD plab-old-mtp $Q/qwentin/tools "NOENV=1" --n 3 --tqf $DV --mtp-on
P $LIB plab-new-mtp $W/tools "NOENV=1" --n 3 --tqf $DV --mtp-on
$PY - <<'PY' || fail "gate 1"
import torch, sys
Q = "/mnt/nvme/qwen256k/runs/cand26"; ok = True
for a, b, label in (("plab-new-n3", "plab-old-n3", "N=3"), ("plab-new-n8", "plab-old-n8", "N=8"), ("plab-new-mtp", "plab-old-mtp", "N=3 --mtp-on (dv64k file)")):
    A = torch.load(f"{Q}/{a}.pt"); B = torch.load(f"{Q}/{b}.pt")
    s = A.keys() == B.keys() and all(torch.equal(A[k], B[k]) for k in A); ok &= s
    print(f"[plab] {label} cand-1215 vs served: bitwise {s} ({len(A)} tensors)")
print("[gate1]", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
PY

echo "== gate 2: THE SERVED LINE (dv64k file), K auto = 6/3/2   $(date -u +%T)"
export PROBE_PORT=$PORT
exec 9>$LOCK; flock 9
serve() { tqf=$1; log=$2
  nohup $PY -u $W/tools/serve_batched.py --port $PORT --max-slots 12 --num-blocks 4500 --timeout 1800 --model-name qwen3.8-27b \
    --tqf $tqf --model-dir $MD --lib $LIB > $log 2>&1 &
  SPID=""; for i in $(seq 1 120); do sleep 5; curl -s -m 2 http://127.0.0.1:$PORT/health > /dev/null 2>&1 && { SPID=$(pgrep -f "serve_batched.py --port $PORT --max-slots [0-9]" | head -1); break; }; done
  [ -n "$SPID" ] || { tail -5 $log; fail "server did not come up ($log)"; }
  MIB=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -v p=$SPID -F', ' '$1==p {print $2}')
  echo "   server up after $((i * 5)) s, pid $SPID, $MIB MiB; $(grep -oE '[0-9]+ pooled tokens' $log); $(grep -oE 'speculative decoding ON \([^)]*\)' $log | head -1)"; }
stop() { kill $SPID; for i in $(seq 1 30); do sleep 2; kill -0 $SPID 2>/dev/null || break; done; }
tpr() { grep -E "\[speed\] ctx" $1 | awk -v t=$2 '{for (i = 1; i <= NF; i++) { if ($i == "tokens" && $(i+1) == "in") tok = $(i-1); if ($i == "rounds") r = $(i+1) } printf "   %s %s %s: %.2f tokens per round (%s tokens / %s rounds)\n", t, $2, $3, tok / r, tok, r }'; }
serve $DV logs/cand26-server.log
grep -q "K=auto = 6/3/2 by active slots; draft vocabulary 65536 ids" logs/cand26-server.log || { stop; fail "the startup line does not say 6/3/2 with the draft vocabulary"; }
curl -s http://127.0.0.1:$PORT/health | grep -oE '"draft_vocab": ?[0-9]+' | sed 's/^/   \/health /'
curl -s http://127.0.0.1:$PORT/health | grep -qE '"draft_vocab": ?65536' || { stop; fail "/health draft_vocab"; }
$Q/venv-eval/bin/python analysis/t1_repro.py --base http://127.0.0.1:$PORT --out runs/cand26/repro.json > logs/cand26-repro.log 2>&1; rrc=$?
grep "garbage" logs/cand26-repro.log | tail -1 | sed 's/^/   /'; RG=$(grep -c "garbage 0 / 6" logs/cand26-repro.log)
timeout 3000 $PY eval/code_needle.py --base-url http://127.0.0.1:$PORT --model qwen3.8-27b --lengths 32768,131072 --depths 0.95,0.5,0.05 --decoys 3 \
  --tokenizer $MD --chat-kwargs '{"enable_thinking": false}' --out runs/cand26/needle.json > logs/cand26-needle.log 2>&1; nrc=$?
grep -E "\] (PASS|FAIL)" logs/cand26-needle.log | cut -c1-100
NP=$(grep -c "\] PASS" logs/cand26-needle.log); NF=$(grep -c "\] FAIL" logs/cand26-needle.log)
timeout 1800 $PY analysis/spec_server_ab.py --base http://127.0.0.1:$PORT --prompts 10 --max-tokens 300 --out runs/cand26/spec-ab.json > logs/cand26-spec-ab.log 2>&1; abrc=$?
grep -E "identical outputs|tokens/round" logs/cand26-spec-ab.log | cut -c1-160; AB=$(grep -c "identical outputs: 10/10" logs/cand26-spec-ab.log)
$PY - <<'PY'; ABT=$?
import json, sys
R = "/mnt/nvme/qwen256k/runs"
a = json.load(open(f"{R}/cand26/spec-ab.json")); b = json.load(open(f"{R}/cand21/spec-ab.json"))
same_p = a["prompts"] == b["prompts"]
same = {m: [x["text"] for x in a["res"][m]] == [x["text"] for x in b["res"][m]] for m in ("on", "off")}
print(f"[ab-vs-cand21] prompts identical {same_p}; texts identical to cand21: on {same['on']}, off {same['off']}")
sys.exit(0 if same_p and all(same.values()) else 1)
PY
timeout 2400 $PY analysis/spec_speed.py --base http://127.0.0.1:$PORT --ctx 32768 --n 1,2,8 --modes on --max-tokens 300 --repeats 2 --out runs/cand26/spec-speed-32k.json > logs/cand26-spec-speed-32k.log 2>&1; src=$?
timeout 2400 $PY analysis/spec_speed.py --base http://127.0.0.1:$PORT --ctx 131072 --n 1 --modes on --max-tokens 300 --out runs/cand26/spec-speed-128k.json > logs/cand26-spec-speed-128k.log 2>&1; src2=$?
grep -hE "\[speed\] ctx" logs/cand26-spec-speed-32k.log logs/cand26-spec-speed-128k.log | cut -c1-150
tpr logs/cand25-spec-speed-32k.log cand25-K4; tpr logs/cand26-spec-speed-32k.log cand26-auto6
echo "   -- T1 HumanEval+ with spec on   $(date -u +%T)"
./eval/evalplus.sh http://127.0.0.1:$PORT qwen3.8-27b cand26-k6 humaneval > logs/cand26-t1-humaneval.log 2>&1; echo "   humaneval exit $? $(date -u +%T): $(grep -E 'pass@1' logs/cand26-t1-humaneval.log | tr -s '\t ' ' ' | tr '\n' ' ')"
$PY - <<'PY'; t1rc=$?
import re, sys, json, glob
v = [float(x) for x in re.findall(r"pass@1:\s*([\d.]+)", open("/mnt/nvme/qwen256k/logs/cand26-t1-humaneval.log").read())]
raw = [json.loads(l) for l in open(glob.glob("/mnt/nvme/qwen256k/runs/evalplus-cand26-k6/humaneval/*.raw.jsonl")[0])]
g = sum("!!!!!!!!" in r["solution"] for r in raw)
ok = v == [0.939, 0.915] and g == 0
print(f"   [t1] humaneval spec on, K auto 6: {v} (must equal 93.9 / 91.5); degenerate {g} / {len(raw)} -> {'ok' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PY
ERR=$(grep -ciE "xid|error|traceback" logs/cand26-server.log)
stop
echo "   server log error lines: $ERR; repro clean $RG; needle PASS $NP FAIL $NF; A/B 10/10: $AB, texts = cand21: $((1 - ABT)); speed rc $src/$src2; T1 rc $t1rc"
[ "$rrc" -eq 0 ] && [ "$RG" -eq 1 ] && [ "$nrc" -eq 0 ] && [ "$NF" -eq 0 ] && [ "$NP" -ge 6 ] && [ "$abrc" -eq 0 ] && [ "$AB" -eq 1 ] && [ "$ABT" -eq 0 ] && [ "$src" -eq 0 ] && [ "$src2" -eq 0 ] && [ "$t1rc" -eq 0 ] && [ "$ERR" -eq 0 ] \
  || fail "gate 2 (repro rc=$rrc clean=$RG, needle rc=$nrc PASS=$NP FAIL=$NF, ab rc=$abrc identical=$AB texts=$ABT, speed rc=$src/$src2, T1 rc=$t1rc, err=$ERR)"
echo "   -- the served v2 file (no draft section) on the same line: K auto must stay 4/3/2"
serve $V2 logs/cand26-server-v2.log
grep -q "K=auto = 4/3/2 by active slots; draft vocabulary 0 ids" logs/cand26-server-v2.log || { stop; fail "the v2 file's startup line is not 4/3/2"; }
timeout 2400 $PY analysis/spec_speed.py --base http://127.0.0.1:$PORT --ctx 32768 --n 1 --modes on --max-tokens 300 --out runs/cand26/spec-speed-32k-v2.json > logs/cand26-spec-speed-32k-v2.log 2>&1
tpr logs/cand26-spec-speed-32k-v2.log cand26-v2file-K4
stop
grep -qE "rounds 78 steps" logs/cand26-spec-speed-32k-v2.log || fail "the v2 file on the new server is not the stock K = 4 (78 rounds)"
echo "[gate2] PASS"
exec 9>&-
echo "== ALL PASS   $(date -u +%T)  (landing: build-qwen/ := the candidate, main := cand-1215 (ff), patches 0134-0135)"
