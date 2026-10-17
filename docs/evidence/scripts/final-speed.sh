#!/usr/bin/env bash
# Speed table of THE LANDED SERVED LINE (main 11f5879, build-qwen/ c9d5269c) for the decision list: pure defaults vs all opt-ins
# (TQ_T4_ATTN_F16=1 TQ_GRAPH=1 TQ_GEMM_FORK=1 + --dstate fp16sr --num-blocks 4500). spec_speed @32K N=1/2/4/8 (spec on + off) and
# @128K N=1/2 (spec on), 300 tokens per stream. GPU 2 under the lock, port 8136. ~20 min.
# usage: nohup logs/final-speed.sh > logs/final-speed.log 2>&1 &
set -u
Q=/mnt/nvme/qwen256k; PY=/root/work/venv/bin/python
LIB=$Q/qwentin/build-qwen/libforward_qwen.so
SERVE=$Q/qwentin/tools/serve_batched.py
T4M=/mnt/nvme/models/qwen3_8-27b-exl3-opt5.75-mtp4.tqf
CB=$Q/analysis/kv_trellis_cuda/codebooks/lloyd_kv12a.f32
PORT=8136; GPU=2
cd $Q; mkdir -p runs/final; exec 9>/tmp/exl3diff-gpu$GPU.lock; flock 9
export CUDA_VISIBLE_DEVICES=$GPU TQ_CTX=262144 TQ_KV_Q4=1 TQ_EMBED_FP8=2 TQ_PAGED_ATTN_MMA=1 TQ_KV_T4_CODEBOOK=$CB PROBE_PORT=$PORT
unset TQ_T4_ATTN_F16 TQ_GRAPH TQ_GEMM_FORK TQ_DSTATE_BF16
SPID=""
start_server() { local log=$1; shift
  nohup $PY -u $SERVE --port $PORT --max-slots 12 --timeout 1800 --model-name qwen3.8-27b --tqf $T4M --model-dir /mnt/nvme/models/Qwen3.8-27B-FP8 --lib $LIB "$@" > $log 2>&1 &
  SPID=""; for i in $(seq 1 120); do sleep 5; curl -s -m 2 http://127.0.0.1:$PORT/health > /dev/null 2>&1 && { SPID=$(pgrep -f "serve_batched.py --port $PORT --max-slots [0-9]" | head -1); break; }; done
  [ -n "$SPID" ] || { echo "   server did not come up"; tail -3 $log; return 1; }
  echo "   server up, pid $SPID, $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -v p=$SPID -F', ' '$1==p {print $2}') MiB"; }
stop_server() { kill $SPID 2>/dev/null; for i in $(seq 1 30); do sleep 2; kill -0 $SPID 2>/dev/null || break; done; SPID=""; }
measure() { local tag=$1
  timeout 3600 $PY analysis/spec_speed.py --base http://127.0.0.1:$PORT --ctx 32768 --n 1,2,4,8 --modes on,off --max-tokens 300 --out runs/final/speed-$tag-32k.json > logs/final-speed-$tag-32k.log 2>&1
  grep -E "\[speed\] ctx" logs/final-speed-$tag-32k.log | cut -c1-130
  timeout 3600 $PY analysis/spec_speed.py --base http://127.0.0.1:$PORT --ctx 131072 --n 1,2 --modes on --max-tokens 300 --out runs/final/speed-$tag-128k.json > logs/final-speed-$tag-128k.log 2>&1
  grep -E "\[speed\] ctx" logs/final-speed-$tag-128k.log | cut -c1-130; }
echo "[final-speed] start $(date -u +%T)  lib $(md5sum $LIB | cut -c1-12)"
echo "== (D) pure defaults, 4100 blocks   $(date -u +%T)"
start_server logs/final-speed-server-D.log --num-blocks 4100 && { measure D; stop_server; }
echo "== (R) TQ_T4_ATTN_F16=1 TQ_GRAPH=1 TQ_GEMM_FORK=1 + --dstate fp16sr --num-blocks 4500   $(date -u +%T)"
export TQ_T4_ATTN_F16=1 TQ_GRAPH=1 TQ_GEMM_FORK=1
start_server logs/final-speed-server-R.log --num-blocks 4500 --dstate fp16sr && { measure R; stop_server; }
unset TQ_T4_ATTN_F16 TQ_GRAPH TQ_GEMM_FORK
echo "   error lines: D $(grep -ciE 'xid|error|traceback' logs/final-speed-server-D.log), R $(grep -ciE 'xid|error|traceback' logs/final-speed-server-R.log)"
echo "[final-speed] done $(date -u +%T)"
