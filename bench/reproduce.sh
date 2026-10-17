#!/usr/bin/env bash
# bench/reproduce.sh -- one script that reproduces the README's served-line numbers on one SM120 GPU and writes them,
# with the environment they were measured in, to bench/results-<date>-<commit>.json.
#
#   1. build          cmake + make of src/forward_qwen.cu into build-qwen-repro/ (skip with --no-build and LIB=...)
#   2. model check    tools/inspect_tqf.py on the .tqf (header flags, payload size, the EXL3 + MTP verdict)
#   3. serve          THE SERVED STACK line of the README (tools/serve_batched.py, 12 slots, 4500 blocks, 127.0.0.1:$PORT)
#   4. wait           for /health
#   5. needle         tools/code_needle.py   32K x depths 0.95 / 0.5 / 0.05, 3 decoys            -> 3/3, TTFT = cold prefill
#   6. spec A/B       tools/spec_server_ab.py 10 prompts, spec on vs off                          -> 10/10 identical
#   7. speed          tools/spec_speed.py    32K, N = 1 and 8, spec on and off                   -> per-stream tok/s
#   8. stop           the server (only the process this script started)
#   9. --quality      optionally the KL / top-1 yardstick (tools/qwentin_e2e.py + tools/e2e_frontier.py, docs/QUALITY.md)
#  10. collect        everything into one JSON with the environment block (commit, library md5, driver, nvcc, torch, ...)
#
# Paths are environment variables or options -- there are no box defaults:
#   TQF=/path/to/qwen3_8-27b-exl3-opt5.75-mtp4.tqf   required   the converted model (README "Convert a model")
#   MODEL_DIR=/path/to/Qwen3.8-27B                   required   HF dir: tokenizer + chat template (bf16 or FP8 dir)
#   LIB=/path/to/libforward_qwen.so                  optional   default: the library built in step 1 (build-qwen-repro/)
#   NVCC=/usr/local/cuda-13.1/bin/nvcc               optional   default: /usr/local/cuda-13.1/bin/nvcc if present, else `nvcc` on PATH
#   PYTHON=python3  GPU=0  PORT=8100  CORPUS=<this repo>  OUT_DIR=bench  MAX_SLOTS=12  NUM_BLOCKS=4500  MODEL_NAME=qwen3.8-27b
#   NOTE: the spec-decode speed depends on the prompt text (accepted draft length). The README's 167 / 81 tok/s were measured
#   on a checkout of github.com/vllm-project/vllm (CORPUS=<that checkout>); on this repository's own code the first run gave
#   129 / 72 (bench/results-20261017-145336-3e349a3.json). TTFT, needle and spec-off decode do not depend on the corpus.
#   --quality needs REF_DIR=/path/to/Qwen3.8-27B (the bf16 original) and E2E_DIR=/path/to/runs/e2e holding the reference
#   states and the FP8 rows (docs/QUALITY.md section 5 (a) and (c) `ref`); it scores the 4K and 32K prefill cells.
#
# Options: --dry-run (print every command, run nothing), --no-build, --no-check, --quality, --tag NAME, -h.
# Exit code 0 = every step ran; the verdicts (needle 3/3, A/B 10/10) are in the JSON, not in the exit code, so a slower
# card still produces a complete record. Times on a 5090: build ~70 s, server up ~15 s, needle ~1 min, A/B ~1.5 min,
# speed ~1 min; --quality adds ~5 min.
set -uo pipefail

HERE=$(cd "$(dirname "$0")/.." && pwd)
cd "$HERE"

DRY=0; BUILD=1; CHECK=1; QUALITY=0; TAG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --no-build) BUILD=0 ;;
    --no-check) CHECK=0 ;;
    --quality) QUALITY=1 ;;
    --tag) TAG=$2; shift ;;
    -h|--help) sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option $1 (see -h)"; exit 2 ;;
  esac
  shift
done

TQF=${TQF:-}; MODEL_DIR=${MODEL_DIR:-}; LIB=${LIB:-}
PYTHON=${PYTHON:-python3}; GPU=${GPU:-0}; PORT=${PORT:-8100}; CORPUS=${CORPUS:-$HERE}
OUT_DIR=${OUT_DIR:-$HERE/bench}; MAX_SLOTS=${MAX_SLOTS:-12}; NUM_BLOCKS=${NUM_BLOCKS:-4500}; MODEL_NAME=${MODEL_NAME:-qwen3.8-27b}
REF_DIR=${REF_DIR:-}; E2E_DIR=${E2E_DIR:-}
if [ -z "${NVCC:-}" ]; then
  if [ -x /usr/local/cuda-13.1/bin/nvcc ]; then NVCC=/usr/local/cuda-13.1/bin/nvcc; else NVCC=$(command -v nvcc || true); fi
fi
BUILD_DIR=$HERE/build-qwen-repro
[ -n "$LIB" ] || LIB=$BUILD_DIR/libforward_qwen.so

if [ $DRY -eq 0 ]; then
  [ -n "$TQF" ] || { echo "TQF=/path/to/model-mtp4.tqf is required (-h)"; exit 2; }
  [ -n "$MODEL_DIR" ] || { echo "MODEL_DIR=/path/to/HF-checkpoint is required (-h)"; exit 2; }
  [ -f "$TQF" ] || { echo "TQF $TQF not found"; exit 2; }
  [ -d "$MODEL_DIR" ] || { echo "MODEL_DIR $MODEL_DIR not found"; exit 2; }
  if [ $QUALITY -eq 1 ]; then
    [ -n "$REF_DIR" ] && [ -n "$E2E_DIR" ] || { echo "--quality needs REF_DIR= and E2E_DIR= (-h)"; exit 2; }
  fi
else
  TQF=${TQF:-/path/to/model-mtp4.tqf}; MODEL_DIR=${MODEL_DIR:-/path/to/Qwen3.8-27B}
  REF_DIR=${REF_DIR:-/path/to/Qwen3.8-27B}; E2E_DIR=${E2E_DIR:-/path/to/runs/e2e}
fi

COMMIT=$(git -C "$HERE" rev-parse --short HEAD 2>/dev/null || echo nogit)
DATE=$(date -u +%Y%m%d-%H%M%S)
TAG=${TAG:-$DATE-$COMMIT}
WORK=$OUT_DIR/work-$TAG
RESULT=$OUT_DIR/results-$TAG.json
BASE=http://127.0.0.1:$PORT
SERVED_ENV=(CUDA_VISIBLE_DEVICES="$GPU" TQ_CTX=262144 TQ_KV_Q4=1 TQ_EMBED_FP8=2 TQ_PAGED_ATTN_MMA=1)

exec 3>&1                    # fd 3 = the terminal, so dry-run lines bypass the per-step redirects and tees
show() { local suffix=$1; shift; { printf '+'; printf ' %q' "$@"; printf ' %s\n' "$suffix"; } >&3; }
run() {                      # run a command, or print it (--dry-run)
  if [ $DRY -eq 1 ]; then show "" "$@"; return 0; fi
  "$@"
}
log() { echo "[reproduce] $*" >&3; }
if [ $DRY -eq 1 ]; then WORK=$(mktemp -d /tmp/qwentin-reproduce-dry.XXXXXX); fi
mkdir -p "$WORK"

SPID=""
stop_server() {              # only the server this script started
  if [ -n "$SPID" ]; then
    log "stopping server pid $SPID"
    kill "$SPID" 2>/dev/null || true
    for _ in $(seq 1 30); do sleep 1; kill -0 "$SPID" 2>/dev/null || break; done
    kill -9 "$SPID" 2>/dev/null || true
    SPID=""
  fi
}
trap stop_server EXIT

# ---------------------------------------------------------------- 1. build
if [ $BUILD -eq 1 ]; then
  log "build: $NVCC -> $BUILD_DIR"
  run cmake -S "$HERE" -B "$BUILD_DIR" -DCMAKE_CUDA_COMPILER="$NVCC" > "$WORK/cmake.log" 2>&1 || { tail -20 "$WORK/cmake.log"; exit 1; }
  run make -C "$BUILD_DIR" -j"$(nproc)" > "$WORK/make.log" 2>&1 || { tail -20 "$WORK/make.log"; exit 1; }
  [ $DRY -eq 1 ] || [ -f "$LIB" ] || { echo "build produced no $LIB"; exit 1; }
else
  log "build skipped; LIB=$LIB"
  [ $DRY -eq 1 ] || [ -f "$LIB" ] || { echo "LIB $LIB not found"; exit 2; }
fi

# ---------------------------------------------------------------- 2. model check
if [ $CHECK -eq 1 ]; then
  log "model check: tools/inspect_tqf.py $TQF"
  run "$PYTHON" tools/inspect_tqf.py "$TQF" --expect exl3-mtp > "$WORK/inspect_tqf.log" 2>&1 || log "inspect_tqf exited non-zero: the file is not the served EXL3 + MTP format or is inconsistent (see $WORK/inspect_tqf.log)"
  [ $DRY -eq 1 ] || grep -E "^(flags|exl3_bitrates|payload_match|verdict)" "$WORK/inspect_tqf.log" || true
fi

# ---------------------------------------------------------------- 3. serve (THE SERVED STACK line of the README)
log "serve: $BASE, $MAX_SLOTS slots, $NUM_BLOCKS blocks"
SERVE_CMD=("$PYTHON" -u tools/serve_batched.py --host 127.0.0.1 --port "$PORT" --max-slots "$MAX_SLOTS" --num-blocks "$NUM_BLOCKS"
           --timeout 1800 --model-name "$MODEL_NAME" --tqf "$TQF" --model-dir "$MODEL_DIR" --lib "$LIB")
if [ $DRY -eq 1 ]; then
  show "> $WORK/server.log 2>&1 &" env "${SERVED_ENV[@]}" "${SERVE_CMD[@]}"
else
  env "${SERVED_ENV[@]}" "${SERVE_CMD[@]}" > "$WORK/server.log" 2>&1 &
  SPID=$!
fi

# ---------------------------------------------------------------- 4. wait for /health
log "waiting for $BASE/health"
if [ $DRY -eq 1 ]; then
  echo "+ until curl -sf $BASE/health; do sleep 5; done        # gives up after 20 min or when the server exits" >&3
else
  up=0
  for i in $(seq 1 240); do
    sleep 5
    if curl -sf -m 3 "$BASE/health" > /dev/null 2>&1; then up=1; break; fi
    if ! kill -0 "$SPID" 2>/dev/null; then echo "server exited; tail of $WORK/server.log:"; tail -20 "$WORK/server.log"; exit 1; fi
  done
  [ $up -eq 1 ] || { echo "server did not come up in 20 min; tail of $WORK/server.log:"; tail -20 "$WORK/server.log"; exit 1; }
  log "server up after $((i * 5)) s (pid $SPID)"
  grep -E "^batched server" "$WORK/server.log" | head -1 | cut -c1-300
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | awk -v p="$SPID" -F', ' '$1==p {print "   process " $2 " MiB after startup"}'
fi

TOOL_ARGS=(--tokenizer "$MODEL_DIR" --corpus "$CORPUS")

# ---------------------------------------------------------------- 5. needle 32K, 3 depths, 3 decoys
log "needle: 32K x 0.95 / 0.5 / 0.05, 3 decoys"
run "$PYTHON" tools/code_needle.py --base-url "$BASE" --model "$MODEL_NAME" --lengths 32768 --depths 0.95,0.5,0.05 --decoys 3 \
    --chat-kwargs '{"enable_thinking": false}' "${TOOL_ARGS[@]}" --out "$WORK/needle.json" 2>&1 | tee "$WORK/needle.log" | grep -E "\] (PASS|FAIL|ERR)|^wall|^\+" || true

# ---------------------------------------------------------------- 6. spec A/B, 10 prompts
log "spec A/B: 10 prompts x 300 tokens, spec on vs off"
run "$PYTHON" tools/spec_server_ab.py --base "$BASE" --model "$MODEL_NAME" --prompts 10 --max-tokens 300 "${TOOL_ARGS[@]}" \
    --out "$WORK/spec-ab.json" 2>&1 | tee "$WORK/spec-ab.log" | grep -E "^\[ab\]|^\+" || true

# ---------------------------------------------------------------- 7. speed 32K N = 1, 8
log "speed: 32K, N = 1 and 8, spec on and off"
run "$PYTHON" tools/spec_speed.py --base "$BASE" --model "$MODEL_NAME" --ctx 32768 --n 1,8 --modes on,off --max-tokens 300 "${TOOL_ARGS[@]}" \
    --out "$WORK/speed-32k.json" 2>&1 | tee "$WORK/speed-32k.log" | grep -E "^\[speed\] ctx|^\+" || true

if [ $DRY -eq 0 ]; then
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | awk -v p="$SPID" -F', ' '$1==p {print "   process " $2 " MiB after the runs"}' | tee "$WORK/process-mib.txt"
  grep -ciE "xid|error|traceback" "$WORK/server.log" > "$WORK/server-error-lines.txt" || true
fi

# ---------------------------------------------------------------- 8. stop the server
stop_server
trap - EXIT

# ---------------------------------------------------------------- 9. optional quality yardstick (docs/QUALITY.md)
if [ $QUALITY -eq 1 ]; then
  log "quality: tools/qwentin_e2e.py 4K + 32K prefill cells vs the FP8 rows in $E2E_DIR"
  QENV=(CUDA_VISIBLE_DEVICES="$GPU" TQ_KV_V4=1 TQ_KV_T4=1 TQ_PDL=1 TQ_EMBED_HOST=0 TQ_EMBED_FP8=1 TQ_KV_T4_CODEBOOK="$HERE/tools/codebooks/lloyd_kv12a.f32")
  QARGS=(--ref-dir "$REF_DIR" --corpus "$CORPUS" --out-dir "$E2E_DIR" --tqf "$TQF" --lib "$LIB")
  run env "${QENV[@]}" "$PYTHON" tools/qwentin_e2e.py score "${QARGS[@]}" --seq 4096 --decode-from 4096 --tag "repro-$TAG-4k-prefill" --force 2>&1 | tee "$WORK/e2e-4k.log" | grep -E "^   KL|^\+|wrote|rows .* done" || true
  run env "${QENV[@]}" "$PYTHON" tools/qwentin_e2e.py score "${QARGS[@]}" --seq 32768 --decode-from 32768 --tag "repro-$TAG-32k-prefill" --force 2>&1 | tee "$WORK/e2e-32k.log" | grep -E "^   KL|^\+|wrote|rows .* done" || true
  run "$PYTHON" tools/e2e_frontier.py --e2e-dir "$E2E_DIR" --tags "repro-$TAG-4k-prefill" "repro-$TAG-32k-prefill" 2>&1 | tee "$WORK/frontier-all.log"
  run "$PYTHON" tools/e2e_frontier.py --e2e-dir "$E2E_DIR" --min-pos 4096 --tags "repro-$TAG-32k-prefill" 2>&1 | tee "$WORK/frontier-4k.log"
fi

# ---------------------------------------------------------------- 10. collect
log "collect -> $RESULT"
if [ $DRY -eq 1 ]; then
  show "(environment block: git commit, md5 of $LIB, nvidia-smi driver + GPU name, $NVCC --version, torch / transformers / safetensors / numpy versions; + $WORK/*.json, the server startup line, process MiB, server error-line count) > $RESULT" "$PYTHON" - "<<PY"
  log "dry run complete; nothing was executed (scratch dir $WORK)"
  rm -rf "$WORK"
  exit 0
fi
WORK="$WORK" RESULT="$RESULT" LIB="$LIB" NVCC="$NVCC" TQF="$TQF" MODEL_DIR="$MODEL_DIR" CORPUS="$CORPUS" GPU="$GPU" \
MAX_SLOTS="$MAX_SLOTS" NUM_BLOCKS="$NUM_BLOCKS" QUALITY="$QUALITY" E2E_DIR="$E2E_DIR" TAG="$TAG" HERE="$HERE" "$PYTHON" - <<'PY'
import glob, hashlib, json, os, platform, subprocess, sys, time

def sh(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"unavailable ({type(e).__name__})"

def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def ver(mod):
    try:
        return __import__(mod).__version__
    except Exception as e:  # noqa: BLE001
        return f"unavailable ({type(e).__name__})"

def read(path, default=None):
    try:
        return open(path).read()
    except OSError:
        return default

W, lib, here = os.environ["WORK"], os.environ["LIB"], os.environ["HERE"]
env = {
    "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "tag": os.environ["TAG"],
    "commit": sh("git", "-C", here, "rev-parse", "HEAD"),
    "commit_dirty": bool(sh("git", "-C", here, "status", "--porcelain", "--", "src", "tools", "CMakeLists.txt")),
    "lib": lib, "lib_md5": md5(lib) if os.path.exists(lib) else None,
    "tqf": os.environ["TQF"], "tqf_bytes": os.path.getsize(os.environ["TQF"]) if os.path.exists(os.environ["TQF"]) else None,
    "model_dir": os.environ["MODEL_DIR"], "corpus": os.environ["CORPUS"],
    "gpu_index": os.environ["GPU"],
    "gpu": sh("nvidia-smi", "-i", os.environ["GPU"].split(",")[0], "--query-gpu=name,memory.total,power.limit", "--format=csv,noheader"),
    "driver": sh("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader").split("\n")[0],
    "nvcc": os.environ["NVCC"], "nvcc_version": sh(os.environ["NVCC"], "--version").split("\n")[-2:] if os.environ["NVCC"] else None,
    "python": platform.python_version(), "torch": ver("torch"), "transformers": ver("transformers"),
    "safetensors": ver("safetensors"), "numpy": ver("numpy"),
    "served_line": {"max_slots": int(os.environ["MAX_SLOTS"]), "num_blocks": int(os.environ["NUM_BLOCKS"]),
                    "env": {"TQ_CTX": "262144", "TQ_KV_Q4": "1", "TQ_EMBED_FP8": "2", "TQ_PAGED_ATTN_MMA": "1"},
                    "server_defaults": "tools/serve_batched.py at this commit (--kv-v4 --kv-t4 --spec --dstate fp16sr --decode-steps-per-wave 1 ...)"},
}
server_log = read(os.path.join(W, "server.log"), "")
startup = [l for l in server_log.splitlines() if l.startswith("batched server")]
result = {
    "environment": env,
    "inspect_tqf": read(os.path.join(W, "inspect_tqf.log")),
    "server_startup_line": startup[0] if startup else None,
    "process_mib_after_runs": (read(os.path.join(W, "process-mib.txt"), "") or "").strip(),
    "server_error_lines": int((read(os.path.join(W, "server-error-lines.txt"), "0") or "0").strip() or 0),
}
for name in ("needle", "spec-ab", "speed-32k"):
    p = os.path.join(W, f"{name}.json")
    try:
        result[name] = json.load(open(p))
    except (OSError, ValueError) as e:
        result[name] = {"error": f"{type(e).__name__}: {e}", "log_tail": (read(os.path.join(W, f'{name}.log'), '') or '')[-2000:]}
# one-line verdicts next to the raw data
try:
    r = result["needle"]["results"]
    result["summary"] = {"needle_pass": f"{sum(x['pass'] for x in r)}/{len(r)}",
                         "needle_ttft_s": [x["ttft_s"] for x in r]}
except Exception:  # noqa: BLE001
    result["summary"] = {"needle_pass": None}
try:
    ab = result["spec-ab"]["res"]
    on, off = ab.get("on", []), ab.get("off", [])
    result["summary"]["spec_ab_identical"] = f"{sum(a['text'] == b['text'] for a, b in zip(on, off))}/{len(on)}"
except Exception:  # noqa: BLE001
    result["summary"]["spec_ab_identical"] = None
try:
    result["summary"]["speed_32k_p50_tps"] = {f"N={x['N']} spec={x['mode']}": round(x["p50_tps"], 1) for x in result["speed-32k"]}
except Exception:  # noqa: BLE001
    result["summary"]["speed_32k_p50_tps"] = None
if os.environ["QUALITY"] == "1":
    q = {}
    for tag in (f"repro-{os.environ['TAG']}-4k-prefill", f"repro-{os.environ['TAG']}-32k-prefill"):
        p = os.path.join(os.environ["E2E_DIR"], tag + ".json")
        try:
            d = json.load(open(p))
            q[tag] = {k: d.get(k) for k in ("kl_mean", "kl16_mean", "top1_agree", "ppl_ratio", "tokens", "seq", "seed", "corpus")}
        except (OSError, ValueError) as e:
            q[tag] = {"error": str(e)}
    q["frontier_all_positions"] = read(os.path.join(W, "frontier-all.log"))
    q["frontier_min_pos_4096"] = read(os.path.join(W, "frontier-4k.log"))
    result["quality"] = q
os.makedirs(os.path.dirname(os.environ["RESULT"]), exist_ok=True)
json.dump(result, open(os.environ["RESULT"], "w"), indent=1)
print(json.dumps(result["summary"], indent=1))
print(f"wrote {os.environ['RESULT']} (work files in {W})")
PY
