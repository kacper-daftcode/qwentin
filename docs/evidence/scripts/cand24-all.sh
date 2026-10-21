#!/usr/bin/env bash
# cand24 = THE 16 GB MODEL := t45v4 (chain v4 milestone 4: the KL-weighted per-tensor allocation at 4.25 bpw, -17..-20 % KL
# vs the shipped t45-s425 on the simulated full stack, notes/session14-chain4.md; the owner's yes 10-21 10:15 UTC). The
# library is the landed cand-1212 (main 52b4974, md5 5f9cee32); the model file is the only change:
# /mnt/nvme/models/qwen3_8-27b-exl3-t45v4-mtp4.tqf (convert_qwen_tqf.py --mtp-exl3 of Qwen3.8-27B-exl3-t45v4). GPU 0, port 8204.
#  Gate 0  inspect_tqf --verify on the new file; flags = the shipped t45-s425 -mtp4's.
#  Gate 1  ENGINE e2e rows (prefill mode, the published rows' env): 4K / 32K s1 / 32K s2 for t45v4 AND the shipped t45-s425
#          -mtp4 on the same library (tags qw-t45v4_*, qw-t45s425_*): t45v4 better than t45-s425 on the same tokens in every
#          cell (both halves' CIs clear or across 0 on the good side); vs FP8 informational (the 16 GB tier is not FP8).
#  Gate 2  spec exactness on the new file: ctx 4096 (N = 1, 2; 6 rounds) and the unaligned ctx 300 (N = 1, 2; 40 rounds).
#  Gate 3  THE 16 GB LINE (docs/VARIANTS.md: 2 slots, --num-blocks 400, --park-ram-gb 16, no TQ_* environment) on t45v4:
#          process MiB at start / after park_probe 16K / after the needle at 49152 / after everything (<= 16 006 + 16 MiB),
#          needle 3/3 at 49152, spec_speed N = 1 on / off at 32K and 49152, T1 HumanEval+ WITH SPEC ON (sanity: 0 degenerate
#          answers, pass@1 within 3 pt of the 32 GB stack's 93.9 / 91.5 -- a 4.25 bpw model); then the same line on the
#          shipped t45-s425 -mtp4 (memory, needle, speed) for a same-library comparison.
#  NOT landed by this script.
# usage: nohup logs/cand24-all.sh > logs/cand24-all.log 2>&1 < /dev/null &        (~45 min)
set -u
Q=/mnt/nvme/qwen256k; PY=/root/work/venv/bin/python
LIB=$Q/qwentin/build-qwen/libforward_qwen.so
NEW=/mnt/nvme/models/qwen3_8-27b-exl3-t45v4-mtp4.tqf
OLD=/mnt/nvme/models/qwen3_8-27b-exl3-t45-s425-mtp4.tqf
MD=/mnt/nvme/models/Qwen3.8-27B-FP8
GPU=0; LOCK=/tmp/exl3diff-gpu$GPU.lock; PORT=8204
FILT='^\[paged\]|^\[config\]|Warning|warn|layer .*loaded|^TQ_|^TQF1|^KV prefix|^attention|^context|MTP head|transformers|mtp_layer'
cd $Q; mkdir -p runs/cand24
fail() { echo "[cand24-all] FAIL: $*   $(date -u +%T)"; exit 1; }
unset $(env | grep -oE '^TQ_[A-Z0-9_]+')
echo "[cand24-all] start $(date -u +%T); lib $(md5sum < $LIB | cut -c1-12) = $(git -C qwentin log --oneline -1 | cut -c1-60); new $NEW ($(stat -c %s $NEW) bytes), shipped $OLD ($(stat -c %s $OLD) bytes)"

echo "== gate 0: the file   $(date -u +%T)"
$PY qwentin/tools/inspect_tqf.py --verify --expect exl3-mtp $NEW > logs/cand24-inspect.log 2>&1 || { tail -5 logs/cand24-inspect.log; fail "inspect_tqf --verify"; }
grep -E "^flags|payload hash|verdict EXL3" logs/cand24-inspect.log | cut -c1-120 | sed 's/^/   /'
FN=$(grep -oE "^flags: 0x[0-9a-f]+" logs/cand24-inspect.log); FO=$($PY qwentin/tools/inspect_tqf.py $OLD 2>/dev/null | grep -oE "^flags: 0x[0-9a-f]+")
echo "   flags: new '$FN', shipped t45-s425 '$FO'"
[ -n "$FN" ] && [ "$FN" = "$FO" ] || fail "header flags differ from the shipped file's"
echo "[gate0] PASS"

export CUDA_VISIBLE_DEVICES=$GPU
echo "== gate 1: engine e2e rows (prefill mode), t45v4 and the shipped t45-s425 on the same library   $(date -u +%T)"
export TQ_KV_V4=1 TQ_PDL=1 TQ_KV_T4=1 TQ_EMBED_HOST=0 TQ_EMBED_FP8=1
E() { eval/qw_e2e_run.sh $GPU "$@"; }
E qw-t45s425_4k_prefill     --tqf $OLD --lib $LIB --seq 4096  --decode-from 4096
E qw-t45v4_4k_prefill       --tqf $NEW --lib $LIB --seq 4096  --decode-from 4096
E qw-t45s425_32k_prefill    --tqf $OLD --lib $LIB --seq 32768 --decode-from 32768
E qw-t45v4_32k_prefill      --tqf $NEW --lib $LIB --seq 32768 --decode-from 32768
E qw-t45s425_32k_s2_prefill --tqf $OLD --lib $LIB --seq 32768 --decode-from 32768 --seed 2
E qw-t45v4_32k_s2_prefill   --tqf $NEW --lib $LIB --seq 32768 --decode-from 32768 --seed 2
unset TQ_KV_V4 TQ_PDL TQ_KV_T4 TQ_EMBED_HOST TQ_EMBED_FP8
for t in qw-t45s425_4k_prefill qw-t45v4_4k_prefill qw-t45s425_32k_prefill qw-t45v4_32k_prefill qw-t45s425_32k_s2_prefill qw-t45v4_32k_s2_prefill; do
  [ -f runs/vsorig/e2e/$t.json ] || fail "e2e row $t missing (logs/e2e/$t.log)"; done
: > logs/cand24-frontier.txt
F() { $PY analysis/e2e_frontier.py "$@" 2>/dev/null; }
F --tags fp8_4k qw-t45s425_4k_prefill qw-t45v4_4k_prefill --vs fp8_4k | tail -4 | tee -a logs/cand24-frontier.txt
F --tags qw-t45s425_4k_prefill qw-t45v4_4k_prefill --vs qw-t45s425_4k_prefill | tail -2 | sed 's/^/vsold /' | tee -a logs/cand24-frontier.txt
for c in "32k fp8_32k" "32k_s2 fp8_32k_s2"; do set -- $c
  F --min-pos 4096 --tags $2 qw-t45s425_$1_prefill qw-t45v4_$1_prefill --vs $2 | tail -4 | tee -a logs/cand24-frontier.txt
  F --min-pos 4096 --tags qw-t45s425_$1_prefill qw-t45v4_$1_prefill --vs qw-t45s425_$1_prefill | tail -2 | sed 's/^/vsold /' | tee -a logs/cand24-frontier.txt
done
$PY - <<'PY' || fail "gate 1 (t45v4 must beat the shipped t45-s425 on the same tokens)"
import re, sys
ok = True; n = 0
for l in open("/mnt/nvme/qwen256k/logs/cand24-frontier.txt"):
    if not l.startswith("vsold qw-t45v4_"): continue
    l2 = l[6:]
    m = re.search(r"([+-]\d\.\d+) \[([+-]\d\.\d+),([+-]\d\.\d+)\]\s+[\d.]+\s+([+-]\d\.\d+) \[([+-]\d\.\d+),([+-]\d\.\d+)\]", l2)
    if not m: print("[gate1] cannot parse:", l2.strip()); ok = False; continue
    d, lo, hi, t, tlo, thi = map(float, m.groups()); n += 1
    kl_ok = hi < 0.0; top_ok = tlo > 0.0 or (tlo <= 0.0 <= thi)
    print(f"[gate1] {l2.split()[0]} vs the shipped t45-s425: dKL {d:+.5f} [{lo:+.5f},{hi:+.5f}] {'better' if kl_ok else 'NOT better'}; dtop-1 {t:+.2f} [{tlo:+.2f},{thi:+.2f}] {'ok' if top_ok else 'WORSE'}")
    ok &= kl_ok and top_ok
ok &= n == 3
print("[gate1]", "PASS" if ok else "FAIL", f"({n} cells)"); sys.exit(0 if ok else 1)
PY

echo "== gate 2: spec exactness on the new file (ctx 4096 and the unaligned ctx 300)   $(date -u +%T)"
R() { tag=$1; shift
  flock $LOCK timeout 3600 $PY -u qwentin/tools/spec_paged_check.py --lib $LIB --tqf $NEW --model-dir $MD "$@" 2>&1 | grep -Ev "$FILT" > logs/cand24-spec-$tag.log
  echo "   $tag: $(grep -E '^\[spec-check\] (unpinned|ALL|[0-9]+ FAIL)' logs/cand24-spec-$tag.log | tr '\n' ' ' | cut -c1-200)"
  grep -q "ALL PASS" logs/cand24-spec-$tag.log || fail "spec_paged_check $tag"; }
R 4k  --ctx 4096 --n 1,2 --rounds 6 --no-pin
R 300 --ctx 300  --n 1,2 --rounds 40 --no-pin
echo "[gate2] PASS"

echo "== gate 3: THE 16 GB LINE (2 slots, 400 blocks, --park-ram-gb 16), pure defaults   $(date -u +%T)"
export PROBE_PORT=$PORT
env | grep -q '^TQ_' && fail "a TQ_* variable is set"
exec 9>$LOCK; flock 9
B=400; MAXCTX=$((B * 128 - 2048))
start_server() { local tqf=$1 log=$2
  nohup $PY -u qwentin/tools/serve_batched.py --port $PORT --max-slots 2 --num-blocks $B --park-ram-gb 16 --timeout 3600 --model-name qwen3.8-27b \
    --tqf $tqf --model-dir $MD --lib $LIB > $log 2>&1 &
  SPID=""; for i in $(seq 1 60); do sleep 5; curl -s -m 2 http://127.0.0.1:$PORT/health > /dev/null 2>&1 && { SPID=$(pgrep -f "serve_batched.py --port $PORT --max-slots [0-9]" | head -1); return 0; }
    pgrep -f "serve_batched.py --port $PORT --max-slots [0-9]" > /dev/null || { tail -3 $log; return 1; }; done; return 1; }
stop_server() { kill $SPID; for i in $(seq 1 30); do sleep 2; kill -0 $SPID 2>/dev/null || break; done; }
gpu_mib() { nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -v p=$SPID -F', ' '$1==p {print $2}'; }
needle() { timeout 1800 $PY eval/code_needle.py --base-url http://127.0.0.1:$PORT --model qwen3.8-27b --lengths $1 --depths 0.95,0.5,0.05 --decoys 3 \
  --tokenizer $MD --chat-kwargs '{"enable_thinking": false}' --out runs/cand24/needle-$2.json > logs/cand24-needle-$2.log 2>&1; local rc=$?
  grep -E "\] (PASS|FAIL)" logs/cand24-needle-$2.log | cut -c1-100 | sed 's/^/   /'; echo "   needle $1 ($2): rc $rc, PASS $(grep -c '\] PASS' logs/cand24-needle-$2.log) FAIL $(grep -c '\] FAIL' logs/cand24-needle-$2.log)"; }
speed() { timeout 1800 $PY analysis/spec_speed.py --base http://127.0.0.1:$PORT --ctx $1 --n 1 --modes on,off --max-tokens 300 --out runs/cand24/speed-$2.json > logs/cand24-speed-$2.log 2>&1
  grep -F "[speed] ctx" logs/cand24-speed-$2.log | cut -c1-150 | sed 's/^/   /'; }
for which in new old; do
  [ $which = new ] && T=$NEW || T=$OLD
  echo "-- $which: $(basename $T)   $(date -u +%T)"
  start_server $T logs/cand24-server-$which.log || fail "$which server did not come up"
  M0=$(gpu_mib); echo "   server up, pid $SPID, $M0 MiB at start; $(grep -oE '[0-9]+ pooled tokens' logs/cand24-server-$which.log); $(grep -m1 '^\[config\]' logs/cand24-server-$which.log)"
  grep -q "$((B * 128)) pooled tokens" logs/cand24-server-$which.log || { stop_server; fail "$which pool"; }
  timeout 900 $PY eval/park_probe.py 16000 > logs/cand24-park-$which.log 2>&1; echo "   park_probe 16K exit $? ($(grep -c PASS logs/cand24-park-$which.log)/3 PASS); $(gpu_mib) MiB after it (ring allocated)"
  needle $MAXCTX $which-max; NP=$(grep -c '\] PASS' logs/cand24-needle-$which-max.log); NF=$(grep -c '\] FAIL' logs/cand24-needle-$which-max.log)
  M1=$(gpu_mib); echo "   $M1 MiB after the needle at $MAXCTX"
  speed 32768 $which-32k; speed $MAXCTX $which-max
  if [ $which = new ]; then
    echo "   T1 HumanEval+ with spec on through the 2-slot server   $(date -u +%T)"
    ./eval/evalplus.sh http://127.0.0.1:$PORT qwen3.8-27b cand24-t45v4 humaneval > logs/cand24-t1-humaneval.log 2>&1; echo "   humaneval exit $? $(date -u +%T): $(grep -E 'pass@1' logs/cand24-t1-humaneval.log | tr -s '\t ' ' ' | tr '\n' ' ')"
    $PY - <<'PY'; t1rc=$?
import re, sys, json, glob
v = [float(x) for x in re.findall(r"pass@1:\s*([\d.]+)", open("/mnt/nvme/qwen256k/logs/cand24-t1-humaneval.log").read())]
raw = [json.loads(l) for l in open(glob.glob("/mnt/nvme/qwen256k/runs/evalplus-cand24-t45v4/humaneval/*.raw.jsonl")[0])]
g = sum("!!!!!!!!" in r["solution"] for r in raw)
ok = len(v) == 2 and g == 0 and all(a >= b - 0.03 - 1e-9 for a, b in zip(v, [0.939, 0.915]))
print(f"   [t1] t45v4 spec on: {v} (the 32 GB stack 93.9 / 91.5); degenerate answers {g} / {len(raw)} -> {'ok' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PY
  fi
  M2=$(gpu_mib); echo "   $M2 MiB after everything (the VARIANTS budget: 16 006 MiB at 400 blocks)"
  ERR=$(grep -ciE "xid|error|traceback" logs/cand24-server-$which.log)
  stop_server
  echo "   $which: needle PASS $NP FAIL $NF; MiB $M0 / $M1 / $M2; server log error lines $ERR"
  if [ $which = new ]; then
    [ "$NF" -eq 0 ] && [ "$NP" -ge 3 ] && [ "$M2" -le 16022 ] && [ "$t1rc" -eq 0 ] && [ "$ERR" -eq 0 ] || fail "gate 3 (needle PASS=$NP FAIL=$NF, MiB $M2, T1 rc $t1rc, err $ERR)"
  fi
done
echo "[gate3] PASS"
exec 9>&-

echo "== ALL PASS   $(date -u +%T)  (landing: docs/VARIANTS.md 16 GB model := t45v4 (recipe + cal250 + the token-weights patch), README's smaller-cards line, tools/recipes, QUALITY.md, CHANGELOG, evidence)"
