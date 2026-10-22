#!/usr/bin/env bash
# README table numbers of THE SERVED LINE on main fbf26a6 (build-qwen/ md5 9be3c8e7; cand-1216 one-pass verify waves + cand-1217 adaptive
# K; session 17): capacity (pooled tokens, process MiB), needle 32K + 128K (TTFT), spec A/B 10/10, the speed table 32K N = 1 / 2 / 4 /
# 8 / 12 spec on and off and 128K N = 1 / 2 spec on and off, --repeats 2, the served dv64k file. No TQ_* environment. A clean re-pin of
# every cell on one library (the table mixed rel12 / cand26 / cand27 cells). GPU 0, port 8217, under its flock.
# usage: nohup logs/rel13-table.sh > logs/rel13-table.log 2>&1 < /dev/null &   (~30 min)
set -u
Q=/mnt/nvme/qwen256k; PY=/root/work/venv/bin/python
LIB=$Q/qwentin/build-qwen/libforward_qwen.so
V2=/mnt/nvme/models/qwen3_8-27b-exl3-opt5.75-mtp4-dv64k.tqf
MD=/mnt/nvme/models/Qwen3.8-27B-FP8
GPU=0; LOCK=/tmp/exl3diff-gpu$GPU.lock; PORT=8217
cd $Q; mkdir -p runs/rel13
unset $(env | grep -oE '^TQ_[A-Z0-9_]+')
export CUDA_VISIBLE_DEVICES=$GPU PROBE_PORT=$PORT
echo "[rel13-table] start $(date -u +%T); lib $(md5sum < $LIB | cut -c1-12) = $(git -C qwentin log --oneline -1 | cut -c1-60); file $(basename $V2)"
exec 9>$LOCK; flock 9
nohup $PY -u qwentin/tools/serve_batched.py --port $PORT --max-slots 12 --num-blocks 4500 --timeout 1800 --model-name qwen3.8-27b \
  --tqf $V2 --model-dir $MD --lib $LIB > logs/rel13-table-server.log 2>&1 &
SPID=""; for i in $(seq 1 120); do sleep 5; curl -s -m 2 http://127.0.0.1:$PORT/health > /dev/null 2>&1 && { SPID=$(pgrep -f "serve_batched.py --port $PORT --max-slots [0-9]" | head -1); break; }; done
[ -n "$SPID" ] || { tail -5 logs/rel13-table-server.log; echo "[rel13-table] FAIL: server"; exit 1; }
gpu_mib() { nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -v p=$SPID -F', ' '$1==p {print $2}'; }
echo "   server up after $((i * 5)) s, pid $SPID, $(gpu_mib) MiB; $(grep -oE '[0-9]+ pooled tokens' logs/rel13-table-server.log); $(grep -m1 '^\[config\]' logs/rel13-table-server.log)"
grep -E "^batched server:|pool blocks" logs/rel13-table-server.log | head -2 | cut -c1-200 | sed 's/^/   /'
timeout 3000 $PY eval/code_needle.py --base-url http://127.0.0.1:$PORT --model qwen3.8-27b --lengths 32768,131072 --depths 0.95,0.5,0.05 --decoys 3 \
  --tokenizer $MD --chat-kwargs '{"enable_thinking": false}' --out runs/rel13/needle.json > logs/rel13-needle.log 2>&1; echo "   needle rc $?"
grep -E "\] (PASS|FAIL)" logs/rel13-needle.log | cut -c1-100
echo "   $(gpu_mib) MiB after the needle"
timeout 1800 $PY analysis/spec_server_ab.py --base http://127.0.0.1:$PORT --prompts 10 --max-tokens 300 --out runs/rel13/spec-ab.json > logs/rel13-spec-ab.log 2>&1; echo "   A/B rc $?: $(grep -E 'identical outputs' logs/rel13-spec-ab.log)"
$PY - <<'PY'
import json
R = "/mnt/nvme/qwen256k/runs"
a = json.load(open(f"{R}/rel13/spec-ab.json")); b = json.load(open(f"{R}/cand21/spec-ab.json"))
print(f"   [ab-vs-cand21] texts identical to cand21's run (the same file): on {[x['text'] for x in a['res']['on']] == [x['text'] for x in b['res']['on']]}, off {[x['text'] for x in a['res']['off']] == [x['text'] for x in b['res']['off']]}")
PY
$PY analysis/spec_speed.py --base http://127.0.0.1:$PORT --ctx 32768 --n 1,2,4,8,12 --modes on,off --max-tokens 300 --repeats 2 --out runs/rel13/speed-32k.json > logs/rel13-speed-32k.log 2>&1; echo "   32k rc $?"
$PY analysis/spec_speed.py --base http://127.0.0.1:$PORT --ctx 131072 --n 1,2 --modes on,off --max-tokens 300 --repeats 2 --out runs/rel13/speed-128k.json > logs/rel13-speed-128k.log 2>&1; echo "   128k rc $?"
grep -hE "\[speed\] ctx" logs/rel13-speed-32k.log logs/rel13-speed-128k.log | cut -c1-160
echo "   $(gpu_mib) MiB after everything"
curl -s http://127.0.0.1:$PORT/health | cut -c1-400; echo
kill $SPID; for i in $(seq 1 30); do sleep 2; kill -0 $SPID 2>/dev/null || break; done
exec 9>&-
echo "[rel13-table] done $(date -u +%T); server log error lines: $(grep -ciE 'xid|error|traceback' logs/rel13-table-server.log)"
