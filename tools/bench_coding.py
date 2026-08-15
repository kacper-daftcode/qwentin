#!/usr/bin/env python3
"""Coding-workload benchmark for qwentin: cold prefill, decode@depth, multi-turn.

Builds repo-like contexts from qwentin's own sources (code!), sizes them in
tokens, and measures via the x_qwentin stats of serve_openai.py.

Scenarios per context size:
  cold     : unique doc (cache-busted) -> cold prefill + code-gen completion
  followup : second turn on same conversation -> prefix-hit suffix prefill + gen

Usage: python3 tools/bench_coding.py [port] [out.json]
"""
import json, sys, time, uuid, urllib.request
from pathlib import Path

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8077

# --- assemble source corpus (the engine's own code makes a realistic repo) ---
files = []
for pat in ("src/*.cu", "tools/*.py"):
    files += sorted(Path("/root/qwentin").glob(pat))
CORPUS = "\n\n".join(f"# FILE: {p}\n{p.read_text(errors='ignore')}" for p in files)
# approximate tokens via chars/3.5 for planning; exact count not needed
def sized_doc(target_chars, seed):
    spacer = f"\n# BENCH-MARKER {seed}\n"
    need = target_chars - len(spacer)
    reps = (need // len(CORPUS)) + 1
    return (CORPUS * reps)[:need] + spacer

def chat(messages, max_tokens=192):
    body = json.dumps({"messages": messages, "temperature": 0.0, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1200) as r:
        out = json.loads(r.read())
    wall = time.time() - t0
    return wall, out.get("x_qwentin") or {}, out["choices"][0]["message"]["content"]

CODE_TASK = ("Below is a repository. Write the CUDA kernel `k_tq_add_rmsnorm_bf16` "
             "(no launch bounds) that adds vector `a` into `resid`, applies RMSNorm "
             "with weight `w` over N=5120 and returns RMSE-scaled output. Code only.")
FOLLOWUP = "Now add a comment documenting the launch configuration. Code only."

def run(size_tag, chars):
    seed = uuid.uuid4().hex[:8]
    doc = sized_doc(chars, seed)
    m1 = [{"role": "system", "content": "You are a CUDA engineer. Repository:\n" + doc},
          {"role": "user", "content": CODE_TASK}]
    w, x, txt = chat(m1, max_tokens=256)
    n_pref, n_reuse = x.get("prefilled_tokens", 0), x.get("reused_tokens", 0)
    row1 = dict(scn="cold", ctx=size_tag, wall=round(w, 2),
                prefill_s=x.get("prefill_s"), prefilled=n_pref,
                prefill_tok_s=round(n_pref / x["prefill_s"]) if x.get("prefill_s") else None,
                gen_tok_s=round(x.get("gen_tok_s", 0), 1), rounds=x.get("rounds"),
                accept_len=x.get("accept_len"))
    print(size_tag, "cold   :", json.dumps(row1), flush=True)
    m2 = m1 + [{"role": "assistant", "content": txt}, {"role": "user", "content": FOLLOWUP}]
    w, x, txt2 = chat(m2, max_tokens=192)
    n_reuse = x.get("reused_tokens", 0)
    row2 = dict(scn="follow", ctx=size_tag, wall=round(w, 2),
                prefill_s=x.get("prefill_s"), reused=n_reuse, prefilled=x.get("prefilled_tokens"),
                gen_tok_s=round(x.get("gen_tok_s", 0), 1),
                accept_len=x.get("accept_len"))
    print(size_tag, "follow :", json.dumps(row2), flush=True)
    return row1, row2

if __name__ == "__main__":
    sizes = [("4k", 14_000), ("32k", 112_000), ("128k", 448_000)]
    results = {}
    for tag, chars in sizes:
        results[tag] = run(tag, chars)
    out = sys.argv[2] if len(sys.argv) > 2 else None
    if out:
        Path(out).write_text(json.dumps(results, indent=2))
        print("saved", out)
