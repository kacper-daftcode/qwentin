#!/usr/bin/env bash
# Session 17: wave A/B/A of a candidate library vs the served one under ONE lock hold (the chained ws-try calls released
# the lock between runs and another line's TP4 window took the GPU in between: "init failed: -1"). 8 waves at 32K and
# 128K per library; the candidate's 4-wave logits dumps compared with the base dumps (bitwise / KL). GPU ${GPU:-2}.
# usage: GPU=2 nohup logs/s17-fuse-ab.sh TAG CAND_LIB > logs/s17-fuse-ab-TAG.log 2>&1 < /dev/null &
set -u
Q=/mnt/nvme/qwen256k; PY=/root/work/venv/bin/python
GPU=${GPU:-2}; TAG=$1; CAND=$2; shift 2
CENV="$*"                                   # ENV=VAL ... applied to the candidate runs only
SERVED=$Q/qwentin/build-qwen/libforward_qwen.so
cd $Q; unset $(env | grep -oE '^TQ_[A-Z0-9_]+'); export CUDA_VISIBLE_DEVICES=$GPU
exec 9>/tmp/exl3diff-gpu$GPU.lock; flock 9
u=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $GPU); while nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; do echo "[guard] GPU $GPU busy under our lock $(date -u +%T), waiting"; sleep 30; done
echo "[fuse-ab $TAG] start $(date -u +%T); cand $(md5sum < $CAND | cut -c1-12) env [$CENV], served $(md5sum < $SERVED | cut -c1-12)"
run() { lib=$1; name=$2; dump=$3; envs=${4:-}
  for ctx in 32768 131072; do k=$((ctx / 1024))k
    env $envs timeout 1800 $PY -u analysis/pattn_profile.py --lib $lib --ctx $ctx --waves 8 --tag $name-$k --load-blob runs/pattn/slot-$k.blob --out runs/ws/$name-$k.json 2>&1 | grep -E "waves at|Error|error|rc=" | sed 's/(host wall.*//; s/^/   /'
    if [ $dump = 1 ]; then
      env $envs timeout 1800 $PY -u analysis/pattn_profile.py --lib $lib --ctx $ctx --waves 4 --tag $name-$k-dump --load-blob runs/pattn/slot-$k.blob --dump-logits runs/ws/logits-$name-$k.pt 2>&1 | grep -E "Error|error|rc=" | sed 's/^/   /'
      $PY analysis/pattn_cmp.py runs/ws/logits-$name-$k.pt runs/ws/logits-base-$k.pt 2>&1 | sed "s/^/   vs base: /"
    fi
  done; }
run $SERVED $TAG-s1 0
run $CAND $TAG 1 "$CENV"
run $SERVED $TAG-s2 0
echo "[fuse-ab $TAG] done $(date -u +%T)"
