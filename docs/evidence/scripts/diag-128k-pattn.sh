#!/usr/bin/env bash
# Why the 128K N=1 spec-on cell reads 115 tok/s in cand16 (v0.10: 133): same ms/round (26.7 vs ~27.5), fewer accepted
# tokens per round (2.85 vs 3.30). Hypothesis: TQ_PATTN_GQA=1 (default since 9c0a046, not bitwise) changes the greedy
# continuation of that one 128K prompt. Test: the same served line + spec_speed 128K N=1 with TQ_PATTN_GQA=0 (= the v0.10
# prefill kernel bit for bit), and 32K N=1 as the control. Library build-qwen/ (158edd06), v2 file. GPU 0, port 8147.
# usage: nohup logs/diag-128k-pattn.sh > logs/diag-128k-pattn.log 2>&1 &        (~6 min)
set -u
Q=/mnt/nvme/qwen256k; PY=/root/work/venv/bin/python
LIB=$Q/qwentin/build-qwen/libforward_qwen.so
V2=/mnt/nvme/models/qwen3_8-27b-exl3-opt5.75-mtp4-v2.tqf
MD=/mnt/nvme/models/Qwen3.8-27B-FP8
PORT=8147; GPU=0; LOCK=/tmp/exl3diff-gpu$GPU.lock
cd $Q; mkdir -p runs/diag-128k
echo "[diag-128k] start $(date -u +%T); library $(md5sum < $LIB | cut -c1-12)"
export CUDA_VISIBLE_DEVICES=$GPU TQ_CTX=262144 TQ_KV_Q4=1 TQ_EMBED_FP8=2 TQ_PAGED_ATTN_MMA=1 TQ_PATTN_GQA=0
exec 9>$LOCK; flock 9
nohup $PY -u $Q/qwentin/tools/serve_batched.py --port $PORT --max-slots 12 --num-blocks 4500 --timeout 1800 --model-name qwen3.8-27b \
  --tqf $V2 --model-dir $MD --lib $LIB > logs/diag-128k-server.log 2>&1 &
SPID=""; for i in $(seq 1 120); do sleep 5; curl -s -m 2 http://127.0.0.1:$PORT/health > /dev/null 2>&1 && { SPID=$(pgrep -f "serve_batched.py --port $PORT --max-slots [0-9]" | head -1); break; }; done
[ -n "$SPID" ] || { tail -5 logs/diag-128k-server.log; echo "[diag-128k] server did not come up"; exit 1; }
echo "   server up (TQ_PATTN_GQA=0), pid $SPID"
timeout 1800 $PY analysis/spec_speed.py --base http://127.0.0.1:$PORT --ctx 131072 --n 1 --modes on --max-tokens 300 --out runs/diag-128k/speed-128k-gqa0.json > logs/diag-128k-speed-128k.log 2>&1
timeout 1800 $PY analysis/spec_speed.py --base http://127.0.0.1:$PORT --ctx 32768 --n 1 --modes on --max-tokens 300 --out runs/diag-128k/speed-32k-gqa0.json > logs/diag-128k-speed-32k.log 2>&1
grep -hE "\[speed\] ctx" logs/diag-128k-speed-128k.log logs/diag-128k-speed-32k.log
kill $SPID; for i in $(seq 1 30); do sleep 2; kill -0 $SPID 2>/dev/null || break; done
exec 9>&-
echo "[diag-128k] done $(date -u +%T)"
