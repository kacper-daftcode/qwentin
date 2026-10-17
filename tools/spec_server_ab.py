#!/usr/bin/env python3
"""Gate (5a) through the server: the same prompts with speculative decoding on and off (the server's run-time
switch /spec?on=1|0, or two servers via --base-on/--base-off), greedy, max_tokens tokens each; reports how many
outputs are identical and, for the others, the first divergence index + the texts around it, plus tok/s and the
server's tokens per round. Prompts: code snapshots from eval/common's corpus (continue / explain / refactor) and
a few plain coding tasks.
usage: spec_server_ab.py [--prompts 20] [--max-tokens 300] [--base http://127.0.0.1:8100] [--concurrency 1] [--out ...]
                         [--model qwen3.8-27b] [--tokenizer /path/to/hf-dir] [--corpus DIR]
This is the tool behind the README's "spec A/B 10/10 identical" gate (--prompts 10). Vendored from the qwen256k research
harness (analysis/spec_server_ab.py); needs a server started with serve_batched.py (the /health and /spec endpoints).
"""
import argparse, json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness_common import DEFAULT_CORPUS, Tok, load_corpus, build_context  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://127.0.0.1:8100")
ap.add_argument("--model", default="qwen3.8-27b", help="the server's --model-name")
ap.add_argument("--tokenizer", default=None, help="HF model dir (tokenizer.json) for exact prompt sizing; else an estimate")
ap.add_argument("--corpus", default=DEFAULT_CORPUS, help="code tree for the snapshot prompts")
ap.add_argument("--base-off", default=None, help="a second server for the OFF run (else the switch on --base)")
ap.add_argument("--prompts", type=int, default=20)
ap.add_argument("--max-tokens", type=int, default=300)
ap.add_argument("--concurrency", type=int, default=1)
ap.add_argument("--ctx", type=int, default=1500, help="code snapshot tokens per prompt")
ap.add_argument("--out", default=None)
ap.add_argument("--modes", default="on,off")
args = ap.parse_args()

TASKS = ["Continue this file: write the next function that would naturally follow, with a docstring.",
         "Explain what this code does in detail, then point out two possible bugs.",
         "Refactor the longest function in this snapshot for readability and show the result.",
         "Write unit tests (pytest) for the most important function in this snapshot."]
PLAIN = ["Implement an LRU cache class in Python with O(1) get/put and explain the design.",
         "Write a C function that parses an IPv4 address string into a uint32_t, with error handling.",
         "Explain the difference between a mutex and a semaphore with short code examples.",
         "Write a bash script that rotates log files older than 7 days and compresses them."]


def prompts():
    corpus = load_corpus(args.corpus); T = Tok(args.tokenizer)
    out = []
    for i in range(args.prompts):
        if i % 5 == 4:
            out.append(PLAIN[(i // 5) % len(PLAIN)])
        else:
            snap = build_context(corpus, args.ctx, seed=100 + i, tok=T)
            out.append(f"Repository snapshot:\n\n{snap}\n\n{TASKS[i % len(TASKS)]}")
    return out


def chat(base, prompt):
    body = {"model": args.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": args.max_tokens,
            "temperature": 0.0, "enable_thinking": False}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=3600).read())
    dt = time.time() - t0
    return r["choices"][0]["message"]["content"], r["usage"]["completion_tokens"], dt


def health(base):
    return json.loads(urllib.request.urlopen(base + "/health", timeout=60).read())


def switch(base, on):
    return json.loads(urllib.request.urlopen(f"{base}/spec?on={1 if on else 0}", timeout=60).read())


P = prompts()
res = {}
for mode in args.modes.split(","):
    base = args.base
    if args.base_off and mode == "off":
        base = args.base_off
    else:
        print(f"[ab] {mode}: {switch(base, mode == 'on')}", flush=True)
    h0 = health(base)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        outs = list(ex.map(lambda p: chat(base, p), P))
    wall = time.time() - t0
    h1 = health(base)
    toks = sum(o[1] for o in outs)
    rounds = h1["rounds"] - h0["rounds"]; st = h1["spec_tokens"] - h0["spec_tokens"]
    print(f"[ab] {mode}: {len(P)} prompts, {toks} tokens in {wall:.1f}s = {toks / wall:.1f} tok/s (concurrency {args.concurrency}); "
          f"per-request mean {sum(o[1] / o[2] for o in outs) / len(outs):.1f} tok/s; rounds {rounds}, "
          f"tokens/round {st / rounds if rounds else float('nan'):.2f}", flush=True)
    res[mode] = outs
if "on" in res and "off" in res:
    same = 0
    for i, (a, b) in enumerate(zip(res["on"], res["off"])):
        if a[0] == b[0]:
            same += 1
            continue
        j = next((k for k in range(min(len(a[0]), len(b[0]))) if a[0][k] != b[0][k]), min(len(a[0]), len(b[0])))
        print(f"  prompt {i}: diverges at char {j}/{len(b[0])}: on={a[0][max(0, j - 30):j + 40]!r} | off={b[0][max(0, j - 30):j + 40]!r}")
    print(f"[ab] identical outputs: {same}/{len(P)}", flush=True)
if args.out:
    json.dump({"prompts": P, "res": {m: [{"text": o[0], "tokens": o[1], "s": o[2]} for o in v] for m, v in res.items()}},
              open(args.out, "w"))
