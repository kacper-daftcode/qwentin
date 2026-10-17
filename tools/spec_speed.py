#!/usr/bin/env python3
"""Gate (5e): concurrent decode speed through the batched server at N streams and a context length, speculative
decoding on vs off (the server's /spec switch), with the decode phases overlapping:
  phase 1: each of the N sessions sends its ctx-token code prompt with max_tokens=1 (prefill, slot parked);
  phase 2: all N send the follow-up at once (prefix-cache hit + a short question) with --max-tokens tokens,
           streamed -> per-stream decode tok/s = (completion_tokens - 1) / (e2e - ttft); aggregate = sum.
Reports p50 / min per-stream tok/s, aggregate, and the server's tokens per round for the phase.
usage: spec_speed.py --ctx 32768 --n 1,2,4,8 [--modes on,off] [--max-tokens 300] [--base http://127.0.0.1:8100]
                     [--model qwen3.8-27b] [--tokenizer /path/to/hf-dir] [--corpus DIR]
This is the tool behind the README's per-stream tok/s table (served line: --ctx 32768 --n 1,2,4,8 and --ctx 131072
--n 1,2). Vendored from the qwen256k research harness (analysis/spec_speed.py); needs a server started with
serve_batched.py (the /health and /spec endpoints).
"""
import argparse, asyncio, json, os, sys, time, urllib.request

import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness_common import DEFAULT_CORPUS, Tok, load_corpus, build_context, chat_stream  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://127.0.0.1:8100")
ap.add_argument("--model", default="qwen3.8-27b", help="the server's --model-name")
ap.add_argument("--tokenizer", default=None, help="HF model dir (tokenizer.json) for exact prompt sizing; else an estimate")
ap.add_argument("--corpus", default=DEFAULT_CORPUS, help="code tree for the snapshot prompts")
ap.add_argument("--ctx", type=int, default=32768)
ap.add_argument("--n", default="1,2,4,8")
ap.add_argument("--modes", default="on,off")
ap.add_argument("--max-tokens", type=int, default=300)
ap.add_argument("--repeats", type=int, default=1)
ap.add_argument("--out", default=None)
args = ap.parse_args()
MODEL = args.model
QUESTION = ("Write a detailed design note (as a numbered list) on how you would refactor the modules in this snapshot "
            "for testability, with code snippets for the two most important changes.")


def health():
    return json.loads(urllib.request.urlopen(args.base + "/health", timeout=60).read())


def switch(on):
    return json.loads(urllib.request.urlopen(f"{args.base}/spec?on={1 if on else 0}", timeout=60).read())


async def run(N, snaps, mode):
    async with httpx.AsyncClient() as client:
        # phase 1: prefill each prompt (sequential: keeps the pool/parking simple), 1 token each
        t0 = time.time()
        for i in range(N):
            msgs = [{"role": "user", "content": f"Repository snapshot:\n\n{snaps[i]}\n\nSay OK."}]
            r = await chat_stream(client, args.base, MODEL, msgs, 1, chat_kwargs={"enable_thinking": False})
            assert r.ok, r.error
        t_pf = time.time() - t0
        h0 = health()
        # phase 2: everyone decodes together from the cached prefix
        async def one(i):
            msgs = [{"role": "user", "content": f"Repository snapshot:\n\n{snaps[i]}\n\nSay OK."},
                    {"role": "assistant", "content": "OK"},
                    {"role": "user", "content": QUESTION}]
            return await chat_stream(client, args.base, MODEL, msgs, args.max_tokens, chat_kwargs={"enable_thinking": False})
        t1 = time.time()
        rs = await asyncio.gather(*[one(i) for i in range(N)])
        wall = time.time() - t1
        h1 = health()
    for r in rs:
        assert r.ok, r.error
    tps = sorted(r.decode_tps for r in rs)
    toks = sum(r.completion_tokens or 0 for r in rs)
    rounds = h1["rounds"] - h0["rounds"]; st = h1["spec_tokens"] - h0["spec_tokens"]
    steps = h1["steps"] - h0["steps"]
    cached = [r.extra.get("x_qwentin", {}).get("prefix_cached_tokens") for r in rs]
    delta = {k: h1[k] - h0[k] for k in ("prefilled_tokens", "host_saves", "host_hits", "prefix_cache_hits", "spec_fallbacks")}
    print(f"   phase-2 server deltas: {delta}, free_blocks {h1['free_blocks']}, parked {h1['parked']}, host_parked {h1['host_parked']}", flush=True)
    return {"N": N, "mode": mode, "p50_tps": tps[len(tps) // 2], "min_tps": tps[0], "max_tps": tps[-1], "agg_tps": sum(tps),
            "tokens": toks, "wall_s": wall, "prefill_s": t_pf, "rounds": rounds, "steps": steps, "deltas": delta,
            "tokens_per_round": (st / rounds) if rounds else None, "ttft_p50": sorted(r.ttft for r in rs)[len(rs) // 2],
            "cached": cached, "completion_tokens": [r.completion_tokens for r in rs]}


def main():
    corpus = load_corpus(args.corpus); T = Tok(args.tokenizer)
    Ns = [int(x) for x in args.n.split(",")]
    snaps = [build_context(corpus, args.ctx - 120, seed=300 + i, tok=T) for i in range(max(Ns))]
    out = []
    for mode in args.modes.split(","):
        print(f"[speed] {mode}: {switch(mode == 'on')}", flush=True)
        for N in Ns:
            for rep in range(args.repeats):
                r = asyncio.run(run(N, snaps, mode))
                out.append(r)
                print(f"[speed] ctx={args.ctx} N={N} spec={mode}: per-stream tok/s p50 {r['p50_tps']:.1f} (min {r['min_tps']:.1f}, "
                      f"max {r['max_tps']:.1f}), aggregate {r['agg_tps']:.1f}; {r['tokens']} tokens in {r['wall_s']:.1f}s; "
                      f"rounds {r['rounds']} steps {r['steps']} tokens/round {r['tokens_per_round'] if r['tokens_per_round'] is None else round(r['tokens_per_round'], 2)}; "
                      f"ttft p50 {r['ttft_p50']:.2f}s; cached {r['cached']}", flush=True)
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)


main()
