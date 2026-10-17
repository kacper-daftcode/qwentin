#!/usr/bin/env python3
"""Long-context retrieval probe on real code ("needle in a repository").

Builds a repository snapshot of L tokens from --corpus, inserts one or more
synthetic functions returning unique identifiers at chosen depths (plus optional
look-alike decoys), and asks the model for the returned value. Exact match on
the identifier. Runs a lengths x depths matrix and prints pass/fail cells.

Example (the README's "needle 6/6 at 32K / 128K" = lengths 32768,131072 x depths 0.95,0.5,0.05 with 3 decoys):
  python3 tools/code_needle.py --base-url http://127.0.0.1:8100 --model qwen3.8-27b \
      --lengths 32768,131072 --depths 0.95,0.5,0.05 --decoys 3 \
      --tokenizer /path/to/Qwen3.8-27B --chat-kwargs '{"enable_thinking": false}' --out needle.json

The code corpus defaults to this repository (or $QWENTIN_CORPUS); see harness_common.py. Vendored from the qwen256k
research harness (eval/code_needle.py).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import string
import sys
import time

import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness_common import (  # noqa: E402
    AGENT_SYSTEM_PROMPT,
    DEFAULT_CORPUS,
    Tok,
    build_context,
    chat_stream,
    fmt_tokens,
    load_corpus,
)

WORDS = ["deploy", "session", "routing", "cache", "quota", "tenant", "shard", "ledger", "replay", "beacon"]


def make_needle(rng: random.Random):
    name = f"{rng.choice(WORDS)}_{rng.choice(WORDS)}_{''.join(rng.choices(string.hexdigits.lower(), k=4))}"
    value = f"{''.join(rng.choices(string.ascii_uppercase, k=2))}-{rng.randrange(100000, 999999)}-{''.join(rng.choices(string.ascii_uppercase, k=2))}"
    block = (
        f"\n\ndef get_{name}():\n"
        f'    """Return the {name.split("_")[0]} identifier used by the {name.split("_")[1]} subsystem."""\n'
        f'    return "{value}"\n\n'
    )
    return name, value, block


def make_decoy(rng: random.Random, name: str):
    # Same prefix words, different suffix and value: forces exact-name matching.
    stem = "_".join(name.split("_")[:2])
    dname = f"{stem}_{''.join(rng.choices(string.hexdigits.lower(), k=4))}"
    dvalue = f"{''.join(rng.choices(string.ascii_uppercase, k=2))}-{rng.randrange(100000, 999999)}-{''.join(rng.choices(string.ascii_uppercase, k=2))}"
    return f'\n\ndef get_{dname}():\n    return "{dvalue}"\n\n'


def insert_at(lines: list[str], depth: float, block: str) -> None:
    idx = min(len(lines) - 1, max(1, int(depth * len(lines))))
    lines.insert(idx, block)


async def probe(client, args, tok, corpus, length, depths, rep, chat_kwargs):
    rng = random.Random(args.seed * 100003 + length * 7 + int(depths[0] * 1000) + rep)
    haystack = build_context(corpus, length - 400, seed=args.seed * 31 + length + rep, tok=tok)
    lines = haystack.split("\n")
    needles = []
    for d in depths:
        name, value, block = make_needle(rng)
        needles.append((d, name, value))
        insert_at(lines, d, block)
    for _ in range(args.decoys):
        insert_at(lines, rng.random(), make_decoy(rng, needles[0][1]))
    text = "\n".join(lines)
    if len(needles) == 1:
        q = (
            f"In the repository snapshot above, what string does the function "
            f"get_{needles[0][1]}() return? Answer with the string only."
        )
    else:
        names = ", ".join(f"get_{n}()" for _, n, _ in needles)
        q = (
            f"In the repository snapshot above, what strings do the functions {names} return? "
            f"Answer as 'name=value' pairs, one per line, nothing else."
        )
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Repository snapshot:\n\n{text}\n\nQuestion: {q}"},
    ]
    res = await chat_stream(
        client, args.base_url, args.model, messages, max_tokens=args.max_tokens,
        temperature=0.0, chat_kwargs=chat_kwargs, timeout=args.timeout,
    )
    answer = res.text.strip()
    hits = [v in answer for _, _, v in needles]
    return {
        "length": length,
        "depths": depths,
        "rep": rep,
        "ok": res.ok,
        "status": res.status,
        "pass": all(hits) and res.ok,
        "hits": hits,
        "expected": [v for _, _, v in needles],
        "answer": answer[:200],
        "ttft_s": round(res.ttft, 3),
        "e2e_s": round(res.e2e, 3),
        "prompt_tokens": res.prompt_tokens,
        "error": res.error[:200],
    }


async def main_async(args):
    chat_kwargs = json.loads(args.chat_kwargs) if args.chat_kwargs else None
    tok = Tok(args.tokenizer)
    print(f"tokenizer: {tok.source}", flush=True)
    corpus = load_corpus(args.corpus)
    lengths = [int(x) for x in args.lengths.split(",")]
    depths = [float(x) for x in args.depths.split(",")]
    results = []
    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.perf_counter()

    async def one(client, L, ds, rep):
        async with sem:
            r = await probe(client, args, tok, corpus, L, ds, rep, chat_kwargs)
            results.append(r)
            mark = "PASS" if r["pass"] else ("ERR" if not r["ok"] else "FAIL")
            print(
                f"[{fmt_tokens(L):>6} depth={','.join(f'{d:.2f}' for d in ds)} rep={rep}] {mark} "
                f"prompt={r['prompt_tokens']} ttft={r['ttft_s']}s e2e={r['e2e_s']}s "
                f"ans={r['answer'][:40]!r} exp={r['expected']} {r['error'][:80]}",
                flush=True,
            )

    jobs = []
    async with httpx.AsyncClient(timeout=args.timeout) as client:
        for L in lengths:
            for rep in range(args.repeats):
                if args.needles == 1:
                    for d in depths:
                        jobs.append(one(client, L, [d], rep))
                else:
                    # Multi-needle: one prompt per repeat, needles spread over the depths.
                    jobs.append(one(client, L, depths[: args.needles], rep))
        await asyncio.gather(*jobs)
    wall = time.perf_counter() - t0

    # Matrix: rows = lengths, cols = depths (single-needle mode).
    print("\npass matrix (rows: context length, cols: needle depth)")
    hdr = " ".join(f"{d:>6.2f}" for d in depths) if args.needles == 1 else "  multi"
    print(f"{'ctx':>7} {hdr}  total")
    for L in lengths:
        rows = [r for r in results if r["length"] == L]
        cells = []
        if args.needles == 1:
            for d in depths:
                rr = [r for r in rows if r["depths"] == [d]]
                cells.append(f"{sum(r['pass'] for r in rr)}/{len(rr)}".rjust(6))
        else:
            cells.append(f"{sum(r['pass'] for r in rows)}/{len(rows)}".rjust(7))
        print(f"{fmt_tokens(L):>7} {' '.join(cells)}  {sum(r['pass'] for r in rows)}/{len(rows)}")
    print(f"wall {wall:.1f}s, {len(results)} probes")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"config": vars(args), "results": results}, f, indent=1)
        print(f"wrote {args.out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", required=True)
    p.add_argument("--lengths", default="32768,65536,131072,262144")
    p.add_argument("--depths", default="0.05,0.25,0.5,0.75,0.95")
    p.add_argument("--needles", type=int, default=1, help="needles per prompt (1 = matrix mode)")
    p.add_argument("--decoys", type=int, default=0, help="look-alike decoy functions per prompt")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=48)
    p.add_argument("--corpus", default=DEFAULT_CORPUS)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--chat-kwargs", default=None)
    p.add_argument("--timeout", type=float, default=3600.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
