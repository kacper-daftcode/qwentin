#!/usr/bin/env bash
# v0.12: the README's new "One 16 GB card" table -- the cells cand24 did not measure on the 16 GB line (t45v4, 2 slots,
# 400 blocks, --park-ram-gb 16, no TQ_* environment, the tagged library): needle at 32K (TTFT @32K) and the process MiB
# around it; the 49K needle / speed / T1 cells come from docs/evidence/cand24-all.log. GPU 1, port 8206.
# usage: nohup logs/rel12-p16.sh > logs/rel12-p16.log 2>&1 < /dev/null &   (~5 min)
set -u
Q=/mnt/nvme/qwen256k; PY=/root/work/venv/bin/python
LIB=$Q/qwentin/build-qwen/libforward_qwen.so
T=/mnt/nvme/models/qwen3_8-27b-exl3-t45v4-mtp4.tqf
MD=/mnt/nvme/models/Qwen3.8-27B-FP8
GPU=1; LOCK=/tmp/exl3diff-gpu$GPU.lock; PORT=8206
cd $Q; mkdir -p runs/rel12
unset $(env | grep -oE '^TQ_[A-Z0-9_]+')
export CUDA_VISIBLE_DEVICES=$GPU PROBE_PORT=$PORT
echo "[rel12-p16] start $(date -u +%T); lib $(md5sum < $LIB | cut -c1-12); file $(basename $T)"
exec 9>$LOCK; flock 9
nohup $PY -u qwentin/tools/serve_batched.py --port $PORT --max-slots 2 --num-blocks 400 --park-ram-gb 16 --timeout 3600 --model-name qwen3.8-27b \
  --tqf $T --model-dir $MD --lib $LIB > logs/rel12-p16-server.log 2>&1 &
SPID=""; for i in $(seq 1 60); do sleep 5; curl -s -m 2 http://127.0.0.1:$PORT/health > /dev/null 2>&1 && { SPID=$(pgrep -f "serve_batched.py --port $PORT --max-slots [0-9]" | head -1); break; }; done
[ -n "$SPID" ] || { tail -5 logs/rel12-p16-server.log; echo "[rel12-p16] FAIL: server"; exit 1; }
gpu_mib() { nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -v p=$SPID -F', ' '$1==p {print $2}'; }
echo "   server up, pid $SPID, $(gpu_mib) MiB; $(grep -oE '[0-9]+ pooled tokens' logs/rel12-p16-server.log); $(grep -m1 '^\[config\]' logs/rel12-p16-server.log)"
timeout 1800 $PY eval/code_needle.py --base-url http://127.0.0.1:$PORT --model qwen3.8-27b --lengths 32768 --depths 0.95,0.5,0.05 --decoys 3 \
  --tokenizer $MD --chat-kwargs '{"enable_thinking": false}' --out runs/rel12/p16-needle-32k.json > logs/rel12-p16-needle-32k.log 2>&1; echo "   needle 32K rc $?"
grep -E "\] (PASS|FAIL)" logs/rel12-p16-needle-32k.log | cut -c1-100
echo "   $(gpu_mib) MiB after the 32K needle"
$PY analysis/spec_speed.py --base http://127.0.0.1:$PORT --ctx 32768 --n 1,2 --modes on,off --max-tokens 300 --repeats 2 --out runs/rel12/p16-speed-32k.json > logs/rel12-p16-speed-32k.log 2>&1; echo "   32k speed rc $?"
grep -hE "\[speed\] ctx" logs/rel12-p16-speed-32k.log | cut -c1-160
echo "   $(gpu_mib) MiB after everything"
kill $SPID; for i in $(seq 1 30); do sleep 2; kill -0 $SPID 2>/dev/null || break; done
exec 9>&-
echo "[rel12-p16] done $(date -u +%T); server log error lines: $(grep -ciE 'xid|error|traceback' logs/rel12-p16-server.log)"
