#!/usr/bin/env bash
# cand-1217 = main cea410d + the acceptance-adaptive --spec-k auto (server only): p_hat (a geometric per-draft acceptance
# estimate of the recent rounds) raises K to 6 at 3..4 active slots when >= 0.82 and to 4 at 5..6 slots when >= 0.78; the
# fixed table otherwise (--spec-k table = the table alone). Library unchanged (the served 9be3c8e7). GPU 0, port 8216.
#  Gate 0  CPU tests (and with the library); exports = qwentin.h (61) -- the library is the served one.
#  Gate 1  THE SERVED LINE (dv64k file, the served library), no TQ_* env: the startup line says 6/3/3 + the adaptive clause;
#          /health spec_p_hat; the 6-prompt repro clean; needle 6/6; A/B 10/10 with texts == cand21's; speed on the vLLM
#          prompts' design-note answers N = 1 / 2 / 8 @32K (tokens per round vs cand27's) + N = 12 + N = 1 @128K --
#          unchanged paths; then at N = 4 and N = 6 the design-note answers with k=table vs k=auto (must tie: p_hat below
#          the thresholds) and the code continuation with k=table vs k=auto (auto must win at N = 4); T1 HumanEval+ with
#          spec on == 93.9 / 91.5 (its concurrency exercises the adaptive K on code: outputs are the verify's).
#  NOT landed by this script.
# usage: nohup logs/cand28-all.sh > logs/cand28-all.log 2>&1 < /dev/null &        (~45 min)
set -u
Q=/mnt/nvme/qwen256k; PY=/root/work/venv/bin/python; CIV=/root/work/ci-venv/bin
W=$Q/qwentin-adk
LIB=$Q/qwentin/build-qwen/libforward_qwen.so
DV=/mnt/nvme/models/qwen3_8-27b-exl3-opt5.75-mtp4-dv64k.tqf
MD=/mnt/nvme/models/Qwen3.8-27B-FP8
GPU=0; LOCK=/tmp/exl3diff-gpu$GPU.lock; PORT=8216
cd $Q; mkdir -p runs/cand28
fail() { echo "[cand28-all] FAIL: $*   $(date -u +%T)"; exit 1; }
unset $(env | grep -oE '^TQ_[A-Z0-9_]+')
echo "[cand28-all] start $(date -u +%T); cand $(git -C $W log --oneline -1 | cut -c1-60); library (served) $(md5sum < $LIB | cut -c1-12)"

echo "== gate 0: ruff + pytest, the exported ABI   $(date -u +%T)"
git -C $W diff --quiet HEAD || fail "the worktree has uncommitted changes"
(cd $W && $CIV/ruff check tools/ tests/ && $CIV/python -m pytest -q tests/ 2>&1 | tail -n 1) || fail "gate 0 (CPU)"
(cd $W && CUDA_VISIBLE_DEVICES=$GPU QWENTIN_LIB=$LIB flock $LOCK $CIV/python -m pytest -q tests/ 2>&1 | grep -E "passed|failed|error" | tail -n 1) | tee logs/cand28-pytest-lib.log
grep -q "84 passed" logs/cand28-pytest-lib.log && ! grep -q "failed\|error" logs/cand28-pytest-lib.log || fail "gate 0 (with the library)"
nm -D --defined-only $LIB | awk '$2 == "T" && $3 ~ /^qwn_/ {print $3}' | sort > runs/cand28/exported.txt
grep -oE "\bqwn_[a-z0-9_]+\(" $W/src/qwentin.h | tr -d '(' | sort -u > runs/cand28/header.txt
cmp -s runs/cand28/exported.txt runs/cand28/header.txt || fail "the exports differ from qwentin.h"
echo "[gate0] PASS ($(wc -l < runs/cand28/exported.txt) exports)"

export CUDA_VISIBLE_DEVICES=$GPU PROBE_PORT=$PORT
echo "== gate 1: THE SERVED LINE, K auto = 6/3/3 + adaptive   $(date -u +%T)"
exec 9>$LOCK; flock 9
nohup $PY -u $W/tools/serve_batched.py --port $PORT --max-slots 12 --num-blocks 4500 --timeout 1800 --model-name qwen3.8-27b \
  --tqf $DV --model-dir $MD --lib $LIB > logs/cand28-server.log 2>&1 &
SPID=""; for i in $(seq 1 120); do sleep 5; curl -s -m 2 http://127.0.0.1:$PORT/health > /dev/null 2>&1 && { SPID=$(pgrep -f "serve_batched.py --port $PORT --max-slots [0-9]" | head -1); break; }; done
[ -n "$SPID" ] || { tail -5 logs/cand28-server.log; fail "server did not come up"; }
MIB=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -v p=$SPID -F', ' '$1==p {print $2}')
echo "   server up after $((i * 5)) s, pid $SPID, $MIB MiB; $(grep -oE '[0-9]+ pooled tokens' logs/cand28-server.log); $(grep -oE 'speculative decoding ON \([^)]*\)' logs/cand28-server.log | head -1)"
stop() { kill $SPID; for i in $(seq 1 30); do sleep 2; kill -0 $SPID 2>/dev/null || break; done; }
grep -q "K=auto = 6/3/3 by active slots; draft vocabulary 65536 ids; one-pass verify wave 64 columns; adaptive: 6 at <= 4 slots when p_hat >= 0.82 / 4 at <= 6 slots when p_hat >= 0.78" logs/cand28-server.log || { stop; fail "the startup line"; }
curl -s http://127.0.0.1:$PORT/health | grep -oE '"(draft_vocab|verify_onepass|spec_p_hat)": ?[0-9.]+' | sed 's/^/   \/health /'
curl -s http://127.0.0.1:$PORT/health | grep -qE '"spec_p_hat": ?[0-9.]+' || { stop; fail "/health spec_p_hat"; }
tpr() { grep -E "\[speed\] ctx" $1 | awk -v t=$2 '{for (i = 1; i <= NF; i++) { if ($i == "tokens" && $(i+1) == "in") tok = $(i-1); if ($i == "rounds") r = $(i+1) } printf "   %s %s %s: %.2f tokens per round (%s tokens / %s rounds)\n", t, $2, $3, tok / r, tok, r }'; }
agg() { grep -E "\[speed\] ctx" $1 | awk '{for (i = 1; i <= NF; i++) if ($i == "aggregate") { a = $(i+1); gsub(";", "", a); s += a; n++ } } END { if (n) printf "%.1f", s / n; else print "nan" }'; }
$Q/venv-eval/bin/python analysis/t1_repro.py --base http://127.0.0.1:$PORT --out runs/cand28/repro.json > logs/cand28-repro.log 2>&1; rrc=$?
grep "garbage" logs/cand28-repro.log | tail -1 | sed 's/^/   /'; RG=$(grep -c "garbage 0 / 6" logs/cand28-repro.log)
timeout 3000 $PY eval/code_needle.py --base-url http://127.0.0.1:$PORT --model qwen3.8-27b --lengths 32768,131072 --depths 0.95,0.5,0.05 --decoys 3 \
  --tokenizer $MD --chat-kwargs '{"enable_thinking": false}' --out runs/cand28/needle.json > logs/cand28-needle.log 2>&1; nrc=$?
grep -E "\] (PASS|FAIL)" logs/cand28-needle.log | cut -c1-100
NP=$(grep -c "\] PASS" logs/cand28-needle.log); NF=$(grep -c "\] FAIL" logs/cand28-needle.log)
timeout 1800 $PY analysis/spec_server_ab.py --base http://127.0.0.1:$PORT --prompts 10 --max-tokens 300 --out runs/cand28/spec-ab.json > logs/cand28-spec-ab.log 2>&1; abrc=$?
grep -E "identical outputs|tokens/round" logs/cand28-spec-ab.log | cut -c1-160; AB=$(grep -c "identical outputs: 10/10" logs/cand28-spec-ab.log)
$PY - <<'PY'; ABT=$?
import json, sys
R = "/mnt/nvme/qwen256k/runs"
a = json.load(open(f"{R}/cand28/spec-ab.json")); b = json.load(open(f"{R}/cand21/spec-ab.json"))
same_p = a["prompts"] == b["prompts"]
same = {m: [x["text"] for x in a["res"][m]] == [x["text"] for x in b["res"][m]] for m in ("on", "off")}
print(f"[ab-vs-cand21] prompts identical {same_p}; texts identical to cand21: on {same['on']}, off {same['off']}")
sys.exit(0 if same_p and all(same.values()) else 1)
PY
SP() { tag=$1; shift; timeout 2400 $PY analysis/spec_speed.py --base http://127.0.0.1:$PORT --ctx 32768 --modes on --max-tokens 300 --repeats 2 --out runs/cand28/$tag.json "$@" > logs/cand28-$tag.log 2>&1 || return 1
  grep -hE "\[speed\] ctx" logs/cand28-$tag.log | cut -c1-140; }
SP speed-32k --n 1,2,8; s1=$?
SP speed-32k-n12 --n 12; s2=$?
timeout 2400 $PY analysis/spec_speed.py --base http://127.0.0.1:$PORT --ctx 131072 --n 1 --modes on --max-tokens 300 --out runs/cand28/speed-128k.json > logs/cand28-speed-128k.log 2>&1; s3=$?
grep -hE "\[speed\] ctx" logs/cand28-speed-128k.log | cut -c1-140
tpr logs/cand27-spec-speed-32k.log cand27; tpr logs/cand28-speed-32k.log cand28
echo "   -- the adaptive K: design-note answers (table vs auto must tie) and code continuation (auto must win at N = 4)   $(date -u +%T)"
ok=1
for q in design continue; do for N in 4 6; do
  curl -s "http://127.0.0.1:$PORT/spec?k=table" > /dev/null; SP adk-$q-n$N-table --n $N --question $q > /dev/null || ok=0
  curl -s "http://127.0.0.1:$PORT/spec?k=auto" > /dev/null;  SP adk-$q-n$N-auto  --n $N --question $q > /dev/null || ok=0
  ph=$(curl -s http://127.0.0.1:$PORT/health | grep -oE '"spec_p_hat": ?[0-9.]+' | grep -oE '[0-9.]+$')
  echo "   $q N=$N: table $(agg logs/cand28-adk-$q-n$N-table.log) tok/s, auto $(agg logs/cand28-adk-$q-n$N-auto.log) tok/s (p_hat after $ph)"
done; done
$PY - <<'PY'; adk=$?
import re, sys
def agg(p):
    v = [float(m) for m in re.findall(r"aggregate ([\d.]+)", open(p).read())]; return sum(v) / len(v)
L = "/mnt/nvme/qwen256k/logs/cand28-adk-"
ok = True
for N in (4, 6):
    t, a = agg(f"{L}design-n{N}-table.log"), agg(f"{L}design-n{N}-auto.log")
    tie = abs(a - t) / t <= 0.05; ok &= tie
    print(f"   [adk] design N={N}: table {t:.1f} auto {a:.1f} ({(a / t - 1) * 100:+.1f} %) -> {'tie' if tie else 'NOT a tie'}")
t, a = agg(f"{L}continue-n4-table.log"), agg(f"{L}continue-n4-auto.log")
win = a >= 1.03 * t; ok &= win
print(f"   [adk] continue N=4: table {t:.1f} auto {a:.1f} ({(a / t - 1) * 100:+.1f} %) -> {'auto wins' if win else 'auto does NOT win'}")
t, a = agg(f"{L}continue-n6-table.log"), agg(f"{L}continue-n6-auto.log")
print(f"   [adk] continue N=6: table {t:.1f} auto {a:.1f} ({(a / t - 1) * 100:+.1f} %) (informational)")
ok &= a >= 0.97 * t
sys.exit(0 if ok else 1)
PY
echo "   -- T1 HumanEval+ with spec on   $(date -u +%T)"
./eval/evalplus.sh http://127.0.0.1:$PORT qwen3.8-27b cand28 humaneval > logs/cand28-t1-humaneval.log 2>&1; echo "   humaneval exit $? $(date -u +%T): $(grep -E 'pass@1' logs/cand28-t1-humaneval.log | tr -s '\t ' ' ' | tr '\n' ' ')"
$PY - <<'PY'; t1rc=$?
import re, sys, json, glob
v = [float(x) for x in re.findall(r"pass@1:\s*([\d.]+)", open("/mnt/nvme/qwen256k/logs/cand28-t1-humaneval.log").read())]
raw = [json.loads(l) for l in open(glob.glob("/mnt/nvme/qwen256k/runs/evalplus-cand28/humaneval/*.raw.jsonl")[0])]
g = sum("!!!!!!!!" in r["solution"] for r in raw)
ok = v == [0.939, 0.915] and g == 0
print(f"   [t1] humaneval spec on, K auto adaptive: {v} (must equal 93.9 / 91.5); degenerate {g} / {len(raw)} -> {'ok' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PY
PH=$(curl -s http://127.0.0.1:$PORT/health | grep -oE '"spec_p_hat": ?[0-9.]+'); echo "   /health after T1: $PH"
ERR=$(grep -ciE "xid|error|traceback" logs/cand28-server.log)
stop
echo "   server log error lines: $ERR; repro clean $RG; needle PASS $NP FAIL $NF; A/B 10/10: $AB, texts = cand21: $((1 - ABT)); speed rc $s1/$s2/$s3; adaptive cells ok $ok rc $adk; T1 rc $t1rc"
[ "$rrc" -eq 0 ] && [ "$RG" -eq 1 ] && [ "$nrc" -eq 0 ] && [ "$NF" -eq 0 ] && [ "$NP" -ge 6 ] && [ "$abrc" -eq 0 ] && [ "$AB" -eq 1 ] && [ "$ABT" -eq 0 ] && [ "$s1" -eq 0 ] && [ "$s2" -eq 0 ] && [ "$s3" -eq 0 ] && [ "$ok" -eq 1 ] && [ "$adk" -eq 0 ] && [ "$t1rc" -eq 0 ] && [ "$ERR" -eq 0 ] \
  || fail "gate 1 (repro rc=$rrc clean=$RG, needle rc=$nrc PASS=$NP FAIL=$NF, ab rc=$abrc identical=$AB texts=$ABT, speed rc=$s1/$s2/$s3, adaptive ok=$ok rc=$adk, T1 rc=$t1rc, err=$ERR)"
echo "[gate1] PASS"
exec 9>&-
echo "== ALL PASS   $(date -u +%T)  (landing: main := cand-1217 (ff), patches 0141-0142; the library stays 9be3c8e7)"
