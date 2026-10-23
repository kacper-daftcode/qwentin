#!/usr/bin/env bash
# Session 17, the cheaper trellis encoder (cand-1222): codes bitwise vs the served library on 8192 fixed rows
# (analysis/t4_encode_ab.py), then the kernel time of the 1024-tile launch under nsys for each library. GPU ${GPU:-3}.
# usage: GPU=3 nohup logs/s17-enc-ab.sh TAG LIB [LIB2 ...] > logs/s17-enc-ab-TAG.log 2>&1 < /dev/null &
set -u
Q=/mnt/nvme/qwen256k; PY=/root/work/venv/bin/python
NSYS=/opt/nvidia/nsight-systems/2025.5.2/bin/nsys
GPU=${GPU:-3}; TAG=$1; shift
SERVED=$Q/qwentin/build-qwen/libforward_qwen.so
cd $Q; mkdir -p runs/enc; unset $(env | grep -oE '^TQ_[A-Z0-9_]+'); export CUDA_VISIBLE_DEVICES=$GPU
exec 9>/tmp/exl3diff-gpu$GPU.lock; flock 9
u=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $GPU); while nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; do echo "[guard] GPU $GPU busy under our lock $(date -u +%T), waiting"; sleep 30; done
echo "[enc-ab $TAG] start $(date -u +%T) on GPU $GPU"
[ -f runs/enc/ref.npz ] || $PY analysis/t4_encode_ab.py $SERVED runs/enc/ref.npz 2>&1 | grep -E "^\[t4-ab\]"
k=0
for LIB in "$@"; do
  k=$((k + 1)); name=$(basename $(dirname $LIB))
  echo "-- $name ($(md5sum < $LIB | cut -c1-12))"
  $PY analysis/t4_encode_ab.py $LIB runs/enc/$TAG-$k.npz 2>&1 | grep -E "^\[t4-ab\]"
  $PY analysis/t4_encode_ab.py --cmp runs/enc/$TAG-$k.npz runs/enc/ref.npz 2>&1 | grep -E "^\[t4-ab\]|first differing"
done
for LIB in $SERVED "$@"; do
  name=$(basename $(dirname $LIB)); rm -f runs/enc/nsys-$TAG-$name.nsys-rep runs/enc/nsys-$TAG-$name.sqlite
  $NSYS profile -t cuda -o runs/enc/nsys-$TAG-$name --force-overwrite true $PY analysis/t4_encode_ab.py $LIB runs/enc/tmp-$TAG.npz --batches 1 --reps 30 > /dev/null 2>&1
  $NSYS stats -q --report cuda_gpu_kern_sum --format csv runs/enc/nsys-$TAG-$name.nsys-rep 2>/dev/null | $PY -c "
import sys, csv
for r in csv.DictReader(sys.stdin):
    if 'k_t4_encode' in r['Name']:
        print(f\"   $name  {r['Name'][:40]:40s} {int(r['Instances']):4d} launches, avg {float(r['Avg (ns)'])/1e3:8.1f} us, min {float(r['Min (ns)'])/1e3:8.1f}, max {float(r['Max (ns)'])/1e3:8.1f}\")"
done
echo "[enc-ab $TAG] done $(date -u +%T)"
