#!/usr/bin/env python3
"""OpenAI-compatible API server for the qwentin FP6 spec-decode engine.

Endpoints: /v1/chat/completions, /v1/completions, /v1/models, /health,
/v1/responses (OpenAI Responses API subset: messages/function calls streaming;
added for the Codex CLI, which dropped wire_api=chat).
Features: SSE streaming, temperature (lossless tree speculative sampling,
TQ_TEMP kernel path; temp=0 = bit-exact greedy), per-request seed, stop
strings + EOS, max_tokens, usage accounting + speculative stats
(x_qwentin: accept_len, rounds, tok/s).

Single-stream engine: requests are serialized with a lock (queue waits).
top_p: only 1.0 supported (v1 sampler); other values are clamped with a
warning field. tools: OpenAI function-calling via the Qwen <tool_call>
convention (JSON or <function=> XML-ish), parsed and round-tripped.
n>1, logprobs: unsupported -> 400.

Reservoir Dogs -- the LLobotomy connection, in two halves:
1) Bark (opt-in): --bark-all-day ports LLobotomy's Optimal-Transport refusal
   intervention into the engine itself. The rank-2 OT map from a save_maps JSON
   is applied on-device right after the hooked decoder layers in every forward
   path (wide prefill, per-node spec verify, dense decode) -- the served tower
   is the lobotomized tower, at spec-decode speed. See README ("--bark-all-day")
   for scale tuning; kernel parity is gated by tools/ot_hook_check.py.
2) Watchdog + fallback-drafter ladder (default ON): every spec round is
   supervised. accept_len-EMA collapse walks the drafter down a (depth, k)
   ladder (6/3 -> 4/2 -> 2/1); a hard round error moves the rest of the request
   to dense qwn_decode; a round exceeding --dogs-hang-s triggers a
   persistent-kernel scratch dump. All rungs are lossless w.r.t. the target
   distribution -- the ladder only gives up speed.

Prefix/KV cache between requests (--prefix-cache, default ON): every prefill
drops a DeltaNet snapshot (slot 0) at the last 16-aligned chunk boundary
inside the prompt (the "anchor"); the positional KV/trunk state below the
anchor stays in place. A new request whose token ids match the previous
prompt through the anchor restores the snapshot and prefills ONLY from the
anchor on -- through the exact same 16-token chunking a fresh prefill would
use, so the cache path is BIT-IDENTICAL to full reset+prefill (gate-tested;
that is also why the anchor must be 16-aligned and why reuse stops at the
previous PROMPT rather than at generated tokens: generation writes its state
through different verify seams, which is equally-valid-forward but not
bit-equal). Anything else falls back to a full reset+prefill.
--prefix-cache-live (default OFF, eps-equivalent NOT token-identical):
additionally continue straight from the live committed state when the new
prompt extends prompt+generation, prefilling only the strict suffix.
x_qwentin reports prefix/reused_tokens/prefilled_tokens per request.

Run:  CUDA_VISIBLE_DEVICES=6 python3 tools/serve_openai.py --port 8000
Test: curl localhost:8000/v1/chat/completions -d '{"messages":[{"role":"user",
      "content":"Hej!"}],"temperature":0.8,"max_tokens":64,"stream":true}'
"""
from __future__ import annotations
import argparse, ctypes, json, os, sys, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mtp_spec_smoke import load_lib, Eng, prefill, ck  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--host", default="0.0.0.0")
ap.add_argument("--tqf", default=os.environ.get("TQ_MODEL_TQF",
                "/workspace/models/Qwen3.6-27B/qwen3_6-27b-e2m3-mtp.tqf"))
ap.add_argument("--model-dir", default="/workspace/models/Qwen3.6-27B")
ap.add_argument("--lib", default="/workspace/qwentin/build-qwen/libforward_qwen.so")
ap.add_argument("--model-name", default="qwentin-qwen3.6-27b-fp6")
ap.add_argument("--ctx", type=int, default=0, help="max prompt+gen tokens (0 = engine limit)")
ap.add_argument("--depth", type=int, default=6)
ap.add_argument("--k", type=int, default=3)
ap.add_argument("--tau", type=float, default=12.0)
ap.add_argument("--maxn", type=int, default=8)
ap.add_argument("--dogs", dest="dogs", action="store_true", default=True,
                help="Reservoir Dogs: spec-round watchdog + fallback-drafter ladder (default ON)")
ap.add_argument("--no-dogs", dest="dogs", action="store_false")
ap.add_argument("--dogs-accept-min", type=float, default=1.30,
                help="accept_len EMA below this (after --dogs-min-rounds at a rung) steps the ladder down")
ap.add_argument("--dogs-min-rounds", type=int, default=6,
                help="spec rounds observed at a rung before a downgrade is allowed")
ap.add_argument("--dogs-hang-s", type=float, default=10.0,
                help="log + persistent-kernel scratch dump when a single spec round exceeds this")
ap.add_argument("--dogs-exit-on-hang", action="store_true", default=False,
                help="os._exit(3) after a hung-round dump (pair with a supervisor that restarts)")
ap.add_argument("--bark-all-day", action="store_true", default=False,
                help="arm the LLobotomy OT intervention inside the engine: residual-stream"
                     " hooks from --ot-maps applied after --ot-layers in every forward"
                     " path (wide prefill / spec verify / dense decode) -- the served"
                     " tower is the lobotomized tower, at spec-decode speed")
ap.add_argument("--ot-maps", default=os.environ.get("TQ_OT_MAPS", ""),
                help="LLobotomy save_maps JSON (default $TQ_OT_MAPS)")
ap.add_argument("--ot-layers", default="37,38",
                help="comma-separated hooked decoder layers (auto-tuned for Qwen3.8-27B: 37,38)")
ap.add_argument("--ot-scale", type=float, default=0.47,
                help="OT intervention scale, from tools/bark_autotune.py on the FP6 tower"
                     " (2026-08-16): refusal leaks 5/6 @0.21, 2/6 @0.40-0.44, clean 0/6 from"
                     " 0.47 up; below that the bf16-HF lab value (0.21) does not transfer."
                     " Re-tune per tower/quantization: probe through the live API")
ap.add_argument("--prefix-cache", dest="prefix_cache", action="store_true", default=True,
                help="reuse engine state across requests sharing a token prefix (default ON)")
ap.add_argument("--no-prefix-cache", dest="prefix_cache", action="store_false")
ap.add_argument("--prefix-cache-min", type=int, default=256,
                help="min reusable prefix tokens (also gated at 25%% of the new prompt)")
ap.add_argument("--prefix-cache-live", action="store_true", default=False,
                help="also continue from the live post-generation state (float-eps "
                     "equivalent, NOT token-identical to a full prefill)")
ap.add_argument("--no-thinking", dest="no_thinking", action="store_true",
                default=(os.environ.get("TQ_NO_THINK", "") == "1"),
                help="default enable_thinking=false (reasoning off) unless a request "
                     "explicitly sets enable_thinking. Recommended for agent/tool use: "
                     "keeps the prefix cache valid across turns (reasoning otherwise "
                     "forces a full re-prefill each turn). Env: TQ_NO_THINK=1")
ap.add_argument("--wide-prefill", dest="wide_prefill", action="store_true", default=True,
                help="wide (N=128) tensor-core prefill path for cold prompts (default ON; "
                     "~2-3x faster first-turn prefill, works at Q4-KV/256k; validated "
                     "needle 4/4 @24k/@120k). Sets TQ_WIDE_PREFILL=1 + TQ_WIDE_ATTN_MMA=1 "
                     "unless those env vars are set explicitly (env always wins).")
ap.add_argument("--no-wide-prefill", dest="wide_prefill", action="store_false")
args = ap.parse_args()

# ----------------------------- Reservoir Dogs -----------------------------
# The crew that keeps the spec-decode heist on schedule when the tower came
# through LLobotomy (refusal direction cut) and its decode distribution can
# drift away from what the MTP drafter was calibrated on:
#
#   Mr. Blonde  shoots first  : full --depth/--k drafting (the ship config).
#   Mr. Orange  bleeds        : accept_len EMA collapses (draft/verify drift)
#                               -> ladder steps down (depth, k), one rung per
#                               --dogs-min-rounds window.
#   Mr. Pink    walks         : stubborn bleeding or a hard round error ->
#                               dense decode finishes the job ("professional").
#   Mr. White   watches       : a round exceeding --dogs-hang-s triggers a
#                               non-blocking persistent-kernel scratch dump, so
#                               the log says which barrier the grid died on.
#
# Every rung verifies against the same target model, so demotion is lossless
# w.r.t. output distribution -- the ladder only gives up speed. Downgrades are
# one-way per request (no flapping); the next request starts at rung 0 again.

def _dogs_ladder(d0, k0):
    rungs, seen = [], set()
    for d, k in ((d0, k0), (max(1, d0 - 2), max(1, k0 - 1)), (max(1, (d0 + 1) // 3), 1)):
        p = (max(1, d), max(1, k))
        if p not in seen:
            seen.add(p)
            rungs.append(p)
    rungs.append(None)                      # None = Mr. Pink (dense, no drafting)
    return rungs


def _dogs_ladder_labels(ladder):
    return ["dense" if r is None else f"{r[0]}/{r[1]}" for r in ladder]


class _Dogs:
    def __init__(self, d0, k0):
        self.ladder = _dogs_ladder(d0, k0)
        self.level = 0
        self.ema = None                     # tokens-per-round EMA at the current rung
        self.at_level = 0                   # rounds observed at the current rung
        self.events = []

    @property
    def params(self):
        return self.ladder[self.level]

    @property
    def label(self):
        return "dense" if self.params is None else f"{self.params[0]}/{self.params[1]}"

    def log(self, msg):
        print(f"[dogs] {msg}", file=sys.stderr, flush=True)
        self.events.append(msg)

    def demote(self, why):
        prev = self.label
        if self.level < len(self.ladder) - 1:
            self.level += 1
        self.ema, self.at_level = None, 0
        self.log(f"{why} -- ladder {prev} -> {self.label}")

    def note(self, accepted, pos):
        """Feed one healthy spec round (accepted = tokens gained, incl. bonus)."""
        self.at_level += 1
        self.ema = float(accepted) if self.ema is None else 0.75 * self.ema + 0.25 * accepted
        if (self.params is not None and self.at_level >= args.dogs_min_rounds
                and self.ema < args.dogs_accept_min):
            self.demote(f"Mr. Orange is bleeding: accept_len EMA {self.ema:.2f} "
                        f"< {args.dogs_accept_min:g} over {self.at_level} rounds "
                        f"(draft/verify drift at pos {pos})")


def _dogs_hang_watch(done, dogs, d, k, pos):
    """Fires only if the guarded qwn_spec_round call exceeds --dogs-hang-s.
    ctypes releases the GIL around the engine call, and qwn_spec_persist_dump
    uses its own non-blocking stream -- this is the watchdog the engine's dump
    hook was written for."""
    if done.wait(args.dogs_hang_s):
        return
    buf = (ctypes.c_int * 64)()             # TQ_PERSIST_SCRATCH_INTS
    rc = LIB.qwn_spec_persist_dump(buf)
    detail = (f"persist scratch counters {list(buf)}" if rc == 0
              else f"scratch dump unavailable (rc={rc})")
    dogs.log(f"Mr. White on the radio: spec round hung > {args.dogs_hang_s:g}s "
             f"(depth {d}/{k}, pos {pos}) -- {detail}")
    if args.dogs_exit_on_hang:
        dogs.log("Mr. White takes the crew down: os._exit(3), supervisor restarts clean")
        os._exit(3)


def _spec_round_guarded(cur_seed, cur_pos, d, k, chain_buf, st_buf, dogs):
    if dogs is None:
        return LIB.qwn_spec_round(int(cur_seed), int(cur_pos), d, k,
                                  ctypes.c_float(args.tau), args.maxn, chain_buf, st_buf)
    done = threading.Event()
    threading.Thread(target=_dogs_hang_watch,
                     args=(done, dogs, d, k, cur_pos), daemon=True).start()
    try:
        return LIB.qwn_spec_round(int(cur_seed), int(cur_pos), d, k,
                                  ctypes.c_float(args.tau), args.maxn, chain_buf, st_buf)
    finally:
        done.set()


# The engine lib and prefill() read these lazily; explicit env overrides the flag.
os.environ.setdefault("TQ_WIDE_PREFILL", "1" if args.wide_prefill else "0")
os.environ.setdefault("TQ_WIDE_ATTN_MMA", "1" if args.wide_prefill else "0")

print(f"[serve] loading {args.tqf} ...", flush=True)
TOK = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
LIB = load_lib(args.lib)
LIB.qwn_set_sampling.argtypes = [ctypes.c_float, ctypes.c_ulonglong]
LIB.qwn_set_sampling.restype = ctypes.c_int
ck(LIB.qwn_init(args.tqf.encode()), "init")
try:
    eng_max = int(LIB.qwn_max_seq())
except Exception:
    eng_max = 2048
if args.ctx <= 0 or args.ctx > eng_max - args.depth - 4:
    args.ctx = eng_max - args.depth - 4
ENG = Eng(LIB)
ENG_LOCK = threading.Lock()
EOS_IDS = set(int(t) for t in [TOK.eos_token_id] if t is not None)
for name in ("<|im_end|>", "<|endoftext|>"):
    try:
        t = TOK.convert_tokens_to_ids(name)
        if t is not None and t >= 0:
            EOS_IDS.add(int(t))
    except Exception:
        pass

# Prefix/KV cache between requests (single slot: the engine holds ONE sequence).
#   prompt : token ids of the request that produced the engine state. The
#            positional KV/trunk rows for prompt[0..anchor-1] are live in the
#            engine; DeltaNet snapshot slot 0 holds the state at the anchor.
#   anchor : 16-aligned chunk boundary inside that prompt where slot 0 was
#            taken. Restoring slot 0 + prefilling from `anchor` replays the
#            EXACT chunking of a fresh full prefill -> bit-identical state.
#   tokens : (--prefix-cache-live only) every token that went THROUGH the
#            engine and is committed in its live state (prompt + chain commits,
#            WITHOUT the pending bonus seed). This is the engine sequence, not
#            the (possibly stop-string/max_tokens-cut) emitted text.
#   pending/pending_greedy: bonus token predicted right after `tokens`, and
#            whether it was an argmax (temp==0 or straight prefill return).
#   high   : number of positions the engine committed (prompt + generation).
#            Reuse REQUIRES the new prompt to reach past it (P >= high): a
#            shrunken prompt under a longer old state was measured to flip
#            near-tie argmaxes from ~100 tokens in even with the rows above P
#            zeroed (drafter-side eps leak, mechanism not pinned down) -- those
#            requests take the full reset+prefill path instead.
PC = {"valid": False, "prompt": [], "anchor": 0, "high": 0,
      "tokens": [], "pending": -1, "pending_greedy": False}
LIB.qwn_kv_clear_rows.argtypes = [ctypes.c_int, ctypes.c_int]
LIB.qwn_kv_clear_rows.restype = ctypes.c_int
if args.prefix_cache_live and EOS_IDS:
    # keep spec rounds from committing tokens PAST an EOS (the truncated tail
    # was never emitted; this keeps `tokens` a clean continuation point)
    LIB.qwn_set_commit_stop.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    LIB.qwn_set_commit_stop.restype = ctypes.c_int
    _stop = sorted(EOS_IDS)
    ck(LIB.qwn_set_commit_stop((ctypes.c_int * len(_stop))(*_stop), len(_stop)),
       "set_commit_stop")
if args.dogs:
    LIB.qwn_spec_persist_dump.restype = ctypes.c_int
    LIB.qwn_spec_persist_dump.argtypes = [ctypes.POINTER(ctypes.c_int)]
BARK = None
if args.bark_all_day:
    if not args.ot_maps:
        sys.exit("--bark-all-day needs --ot-maps PATH (or TQ_OT_MAPS)")
    import numpy as _np
    _maps = json.load(open(args.ot_maps))["ot_maps"]
    LIB.qwn_ot_hook_add.restype = ctypes.c_int
    LIB.qwn_ot_hook_add.argtypes = ([ctypes.c_int, ctypes.c_float] +
                                    [ctypes.POINTER(ctypes.c_float)] * 4)
    _H = int(LIB.qwn_hidden_size())
    _armed = []
    for _ln in [int(x) for x in args.ot_layers.split(",") if x.strip()]:
        m = _maps[str(_ln)]
        P  = _np.ascontiguousarray(_np.asarray(m["P"], dtype=_np.float32))
        A  = _np.ascontiguousarray(_np.asarray(m["A_k_minus_I"], dtype=_np.float32)).reshape(4)
        mu = _np.ascontiguousarray(_np.asarray(m["mu_H"], dtype=_np.float32))
        ms = _np.ascontiguousarray(_np.asarray(m["mean_shift"], dtype=_np.float32))
        assert P.shape == (_H, 2) and mu.shape == (_H,) and ms.shape == (_H,), (P.shape, mu.shape)
        def _f32p(a):
            return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        ck(LIB.qwn_ot_hook_add(_ln, ctypes.c_float(args.ot_scale),
                               _f32p(P), _f32p(A), _f32p(mu), _f32p(ms)),
           f"ot_hook_add layer {_ln}")
        _armed.append(_ln)
    assert int(LIB.qwn_ot_hook_count()) == len(_armed)
    BARK = {"layers": _armed, "scale": args.ot_scale, "maps": os.path.basename(args.ot_maps)}
    print(f"[bark] OT hooks armed on layers {_armed}, scale={args.ot_scale:g} "
          f"({args.ot_maps}) -- the tower barks all day now", flush=True)
print(f"[serve] ready on :{args.port} (eos={sorted(EOS_IDS)}, "
      f"prefix_cache={'on' if args.prefix_cache else 'off'}"
      f"{'+live' if args.prefix_cache_live else ''}, "
      f"thinking={'off (default)' if args.no_thinking else 'template-default (on)'}, "
      f"wide_prefill={'on' if os.environ.get('TQ_WIDE_PREFILL', '0') not in ('', '0') else 'off'}"
      f"+mma={'on' if os.environ.get('TQ_WIDE_ATTN_MMA', '0') not in ('', '0') else 'off'}"
      f"{', bark=' + str(BARK['layers']) + '@' + format(BARK['scale'], 'g') if BARK else ''})", flush=True)
if args.dogs:
    print(f"[serve] dogs on: ladder {' -> '.join(_dogs_ladder_labels(_dogs_ladder(args.depth, args.k)))} "
          f"(accept_min={args.dogs_accept_min:g}, min_rounds={args.dogs_min_rounds}, "
          f"hang_s={args.dogs_hang_s:g}, exit_on_hang={'yes' if args.dogs_exit_on_hang else 'no'})",
          flush=True)


def _common_prefix(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _anchor_of(P):
    """Anchor boundary for a P-token prompt: the last 128-aligned chunk boundary
    at least 8 tokens inside it.

    Margin 8: the chat template's generation tail (e.g. the closed empty
    `<think>` block, 4 tokens) is NOT part of the next turn's prompt, so an
    anchor inside it can never match -- the old `last 16-aligned boundary`
    put the anchor past the shared prefix for P % 16 in {1,2,3} (~19% of
    turns = guaranteed full re-prefill).

    128-alignment: the wide prefill path chunks at 128; a replay from a
    128-aligned anchor reproduces the exact chunk seams of a fresh prefill
    (16-aligned-only anchors kept the 16-token path bit-identical but made
    wide-path replays eps-equivalent instead). 128 is a multiple of 16, so
    the baseline chunked path keeps its bit-identity guarantee too."""
    A = ((P - 8) // 128) * 128
    return A if A > 0 else 0


def _pc_store(prompt_ids, anchor, eng_extra, temp, rounds):
    """Record the engine state contents after a finished request. eng_extra =
    [first seed] + every FULL round chain tail (uncut by the EOS/max_tokens
    emission rules); its last element is the pending bonus seed."""
    seq = list(prompt_ids) + list(eng_extra)
    PC["prompt"] = list(prompt_ids)
    PC["anchor"] = anchor
    PC["tokens"] = seq[:-1]
    PC["pending"] = seq[-1]
    PC["pending_greedy"] = (temp == 0.0 or rounds == 0)
    PC["high"] = len(seq) - 1          # committed positions 0..high-1
    PC["valid"] = True


def generate(prompt_ids, max_new, temp, seed, on_tokens, no_cache=False, dbg=False):
    """Run spec rounds; call on_tokens(new_token_list) as chunks commit.
    Returns (gen_count, finish_reason, stats)."""
    P = len(prompt_ids)
    t0 = time.time()
    with ENG_LOCK:
        LIB.qwn_set_sampling(ctypes.c_float(temp), ctypes.c_ulonglong(seed))
        # ---- prefix-cache decision (token-level, against the LAST request) ----
        mode, reused = "full", 0
        if args.prefix_cache and not no_cache and PC["valid"]:
            need = max(args.prefix_cache_min, P // 4)
            if args.prefix_cache_live:
                ct = PC["tokens"]
                C = _common_prefix(prompt_ids, ct)
                if C == len(ct) and C == P and PC["pending_greedy"]:
                    mode, reused = "exact", C      # same sequence: state is ready
                elif C == len(ct) and C < P and C >= need:
                    mode, reused = "live", C       # committed state continues at C
            if mode == "full":
                A = PC["anchor"]
                if (0 < A < P and A >= need and P >= PC["high"]
                        and prompt_ids[:A] == PC["prompt"][:A]):
                    mode, reused = "anchor", A     # slot-0 restore, bit-identical
        PC["valid"] = False        # re-validated when this request finishes
        if mode != "full":
            # zero positional rows past this prompt: a previous request whose
            # state ran longer must not leak into rounds (reset-equivalence;
            # rows below P are rewritten by the continuation prefill)
            ck(LIB.qwn_kv_clear_rows(P, -1), "kv_clear_rows")
        A_new = _anchor_of(P)
        if mode == "exact":
            seed_tok = PC["pending"]   # bonus argmax; root hidden already live
            anchor_in_slot = PC["anchor"]
        else:
            if mode == "anchor":
                # DeltaNet back to the previous prompt's anchor; positional
                # KV/trunk rows below it are live, the rest gets re-prefilled
                # through the same 16-token chunk grid a full prefill uses
                ENG.restore(0)
            start = reused if mode != "full" else 0
            run_anchor = A_new if A_new > start else 0
            seed_tok = prefill(ENG, prompt_ids, P - 1, start=start, anchor=run_anchor)
            ENG.snapshot_root()
            # slot 0 holds the new anchor if this prefill crossed it; on a reuse
            # path that didn't cross it, the previous snapshot stays valid (the
            # new prompt extends the old one through the old anchor). After a
            # full reset, an uncrossed anchor means no usable snapshot.
            if run_anchor:
                anchor_in_slot = run_anchor
            else:
                anchor_in_slot = PC["anchor"] if mode != "full" else 0
        t_prefill = time.time() - t0
        chain_buf = (ctypes.c_int * (args.maxn + 2))()
        st_buf = (ctypes.c_int * 2)()
        cur_seed, cur_pos = seed_tok, P - 1
        out, rounds, finish = [], 0, "length"
        eng_extra = [seed_tok]     # engine-side continuation (incl. pending seed)
        # the prefill return IS the first generated token (greedy at v1 even for
        # temp>0 -- single-token asymmetry, documented); emit + EOS-check it
        if seed_tok in EOS_IDS:
            stats0 = {"prefill_s": round(t_prefill, 3), "gen_s": 0.0, "rounds": 0,
                      "accept_len": 0.0, "gen_tok_s": None,
                      "prefix": mode, "reused_tokens": reused,
                      "prefilled_tokens": P - reused}
            if dbg:
                stats0["gen_ids"] = []
                stats0["eng_tail_ids"] = list(eng_extra)
            _pc_store(prompt_ids, anchor_in_slot, eng_extra, temp, 0)
            return 0, "stop", stats0
        out.append(seed_tok)
        on_tokens([seed_tok])
        t1 = time.time()
        dogs = _Dogs(args.depth, args.k) if args.dogs else None
        dense = False          # Mr. Pink's shift: dense decode for the remaining tokens
        while len(out) < max_new:
            if dense:
                nt = LIB.qwn_decode(int(cur_seed), int(cur_pos))
                if nt < 0:
                    finish = "error"
                    break
                chunk = [nt]
                cur_seed, cur_pos = nt, cur_pos + 1
                d_now = 1
            else:
                d_eff, k_eff = dogs.params if dogs else (args.depth, args.k)
                cl = _spec_round_guarded(cur_seed, cur_pos, d_eff, k_eff,
                                         chain_buf, st_buf, dogs)
                if cl < 0:
                    if dogs is None:
                        finish = "error"
                        break
                    # state through cur_pos is untouched by a failed round
                    # (tree build/config errors abort before any commit)
                    dogs.log(f"Mr. Pink walks: spec round error {cl} at pos {cur_pos} "
                             f"after {len(out)} tokens; dense decode finishes the job")
                    dense = True
                    continue
                chunk = list(chain_buf[1:cl])
                rounds += 1
                cur_seed, cur_pos = st_buf[0], st_buf[1]
                d_now = d_eff
                if dogs:
                    dogs.note(len(chunk), cur_pos)
                    if dogs.params is None:
                        dense = True
            eng_extra.extend(chunk)
            cut = None
            for i, t in enumerate(chunk):
                if t in EOS_IDS:
                    cut = i
                    break
            if cut is not None:
                out.extend(chunk[:cut])
                on_tokens(chunk[:cut])
                finish = "stop"
                break
            room = max_new - len(out)
            emit = chunk[:room]
            out.extend(emit)
            on_tokens(emit)
            if cur_pos + d_now + 2 >= args.ctx:
                finish = "length"
                break
        dt = time.time() - t1
        if finish != "error":
            _pc_store(prompt_ids, anchor_in_slot, eng_extra, temp, rounds)
    stats = {"prefill_s": round(t_prefill, 3), "gen_s": round(dt, 3),
             "rounds": rounds, "accept_len": round(len(out) / max(1, rounds), 2),
             "gen_tok_s": round(len(out) / dt, 1) if dt > 0 else None,
             "prefix": mode, "reused_tokens": reused,
             "prefilled_tokens": P - reused}
    if dogs is not None:
        stats["dogs"] = {"ladder": _dogs_ladder_labels(dogs.ladder),
                         "final": "dense" if dense else dogs.label,
                         "accept_ema": round(dogs.ema, 2) if dogs.ema is not None else None,
                         "events": list(dogs.events)}
    if dbg:
        stats["gen_ids"] = list(out)
        stats["eng_tail_ids"] = list(eng_extra)
    return len(out), finish, stats


TOOL_OPEN, TOOL_CLOSE = "<tool_call>", "</tool_call>"


def split_think(full, primed):
    """Qwen3.6 reasoning split. The chat template opens `<think>\\n` in the PROMPT
    (when thinking is on), so the model emits `<reasoning></think><answer>` -- only
    the close tag is in the output. Returns (reasoning, answer). When thinking is
    off the template emits a closed empty block, primed is False, and everything
    is the answer. If the close never arrives (length cut mid-think) it is all
    reasoning. Raw (no strip) so streaming char counters stay monotonic."""
    if not primed:
        return "", full
    i = full.find("</think>")
    if i < 0:
        return full, ""
    return full[:i], full[i + len("</think>"):]

def _coerce_arg(val, typ):
    """Coerce an XML-ish parameter STRING to its JSON-schema type. The Qwen3.6
    template serializes every <parameter> value as text, so numbers/bools/arrays
    arrive as strings and fail strict client schemas (e.g. zod number). With the
    declared type we coerce exactly; without it we try a JSON literal and keep the
    string on failure (so paths like "client.py" stay strings)."""
    s = val.strip() if isinstance(val, str) else val
    if typ in ("integer", "number"):
        try:
            return int(s)
        except (ValueError, TypeError):
            try:
                return float(s)
            except (ValueError, TypeError):
                return val
    if typ == "boolean":
        if isinstance(s, str) and s.lower() in ("true", "false"):
            return s.lower() == "true"
        return val
    if typ in ("array", "object"):
        try:
            return json.loads(s)
        except Exception:
            return val
    if typ == "string":
        return val
    # type unknown: best-effort JSON literal (numbers/bools/null/array/object)
    try:
        j = json.loads(s)
        return j if isinstance(j, (int, float, bool, list, dict)) or j is None else val
    except Exception:
        return val


def parse_tool_calls(text, tools=None):
    """Qwen tool-call formats, both supported:
    (a) JSON:   <tool_call>{"name":..., "arguments": {...}}</tool_call>
    (b) XML-ish (Qwen3.6 template): <tool_call><function=NAME>
        <parameter=KEY>VALUE</parameter>...</function></tool_call>
    XML values are type-coerced against the tool's declared parameter schema.
    Returns (clean_text, tool_calls_list_or_None)."""
    import re as _re
    # name -> {param: json-schema-type} from the request's tools
    types = {}
    for t in (tools or []):
        fn = t.get("function", t) if isinstance(t, dict) else {}
        props = (((fn.get("parameters") or {}).get("properties")) or {})
        types[fn.get("name", "")] = {k: (v or {}).get("type") for k, v in props.items()}
    calls = []
    def _emit(name, args):
        calls.append({
            "id": "call_" + uuid.uuid4().hex[:16],
            "type": "function",
            "function": {"name": name,
                         "arguments": json.dumps(args, ensure_ascii=False)},
        })
    def _take(m):
        raw = m.group(1).strip()
        try:
            obj = json.loads(raw)
            _emit(obj.get("name", ""), obj.get("arguments", {}))
            return ""
        except Exception:
            pass
        fm = _re.search(r"<function=([^>\s]+)>(.*?)(?:</function>|$)", raw, _re.S)
        if fm:
            name = fm.group(1)
            ptypes = types.get(name, {})
            args = {}
            for pm in _re.finditer(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", fm.group(2), _re.S):
                key = pm.group(1)
                args[key] = _coerce_arg(pm.group(2), ptypes.get(key))
            _emit(name, args)
            return ""
        return m.group(0)   # unparseable: leave verbatim in content
    clean = _re.sub(_re.escape(TOOL_OPEN) + r"(.*?)" + _re.escape(TOOL_CLOSE), _take,
                    text, flags=_re.S)
    return clean.strip(), (calls or None)


# --------------------------- /v1/responses ---------------------------
# Minimal-but-correct OpenAI Responses API surface so the Codex CLI (which
# dropped wire_api=chat) runs against the local tower. Translates the item
# list to our chat messages, reuses the same generate() path (dogs, bark,
# prefix cache all active), and re-emits the result as Responses SSE events.

def _parse_jsonish(a):
    if isinstance(a, str):
        try:
            return json.loads(a)
        except Exception:
            return a
    return a if a is not None else {}


def responses_to_chat_inputs(req):
    """Responses request -> (messages, tools|None) in the internal chat shape."""
    msgs = []
    sys_parts = []                     # Qwen templates admit exactly ONE leading system
    instr = req.get("instructions")
    if instr:
        sys_parts.append(instr)
    inp = req.get("input", "")
    if isinstance(inp, str):
        if inp:
            msgs.append({"role": "user", "content": inp})
    else:
        for item in inp:
            if not isinstance(item, dict):
                continue
            t = item.get("type", "message")
            if t == "message":
                role = {"developer": "system"}.get(item.get("role"), item.get("role", "user"))
                if role == "system":   # dev/system items fold into the single header system
                    content = item.get("content", "")
                    if isinstance(content, list):
                        content = "\n".join(c.get("text", "") for c in content
                                            if isinstance(c, dict) and c.get("type") in
                                            ("input_text", "output_text"))
                    if content:
                        sys_parts.append(content)
                    continue
                content = item.get("content", "")
                if isinstance(content, list):
                    parts = []
                    for c in content:
                        if isinstance(c, str):
                            parts.append(c)
                        elif isinstance(c, dict) and c.get("type") in ("input_text", "output_text"):
                            parts.append(c.get("text", ""))
                    content = "\n".join(p for p in parts if p)
                msgs.append({"role": role, "content": content})
            elif t == "function_call":
                msgs.append({"role": "assistant", "content": None, "tool_calls": [{
                    "id": item.get("call_id") or ("call_" + uuid.uuid4().hex[:16]),
                    "type": "function",
                    "function": {"name": item.get("name", ""),
                                 "arguments": _parse_jsonish(item.get("arguments", {}))}}]})
            elif t == "function_call_output":
                msgs.append({"role": "tool",
                             "tool_call_id": item.get("call_id", ""),
                             "content": item.get("output", "")})
            # reasoning items add no signal for the model here: dropped
    if sys_parts:
        msgs.insert(0, {"role": "system", "content": "\n\n".join(sys_parts)})
    tools = []
    for tb in req.get("tools") or []:
        if isinstance(tb, dict) and tb.get("type") == "function" and "function" not in tb:
            tools.append({"type": "function", "function": {
                "name": tb.get("name", ""),
                "description": tb.get("description", ""),
                "parameters": tb.get("parameters", {})}})
        elif isinstance(tb, dict):
            tools.append(tb)
    return msgs, (tools or None)


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):
        print(f"[serve] {self.address_string()} {fmt % a}", flush=True)

    def end_headers(self):
        # LAN/browser clients: permissive CORS (this box serves no sensitive data)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _do_responses(self, req):
        """POST /v1/responses: Codex-CLI-grade subset (message/function_call items,
        plain-text streaming, tool calling). Internally the same engine path as
        /v1/chat/completions -- dogs bark identically."""
        msgs, tools = responses_to_chat_inputs(req)
        if req.get("n", 1) != 1 or req.get("logprobs"):
            return self._json(400, {"error": {"message": "n>1/logprobs unsupported"}})
        no_cache = bool(req.get("tl_no_cache", False))
        dbg = bool(req.get("tl_debug", False))
        # clients that omit temperature (Codex) get the engine's greedy path:
        # temp=1.0 sampling turns a barked tower erratic in long agentic prompts
        temp = float(req.get("temperature", 0.0))
        seed = int(req.get("seed", int.from_bytes(os.urandom(4), "little")))
        # 4096 default: a thinking turn burns reasoning tokens on the same budget,
        # and 1024 left the message empty after a long think (the "pusto" report)
        max_new = int(req.get("max_output_tokens") or req.get("max_tokens") or 4096)
        stops = req.get("stop") or []
        if isinstance(stops, str):
            stops = [stops]

        ckw = dict(req.get("chat_template_kwargs") or {})
        rs = req.get("reasoning") if isinstance(req.get("reasoning"), dict) else None
        if "enable_thinking" in req:
            ckw["enable_thinking"] = bool(req["enable_thinking"])
        elif "enable_thinking" in ckw:
            pass
        elif rs and rs.get("effort") is not None:
            eff = str(rs.get("effort")).strip().lower()
            ckw["enable_thinking"] = eff not in ("none", "minimal", "off", "0", "false", "")
        elif args.no_thinking:
            ckw["enable_thinking"] = False
        try:
            text = TOK.apply_chat_template(msgs, tools=tools, add_generation_prompt=True,
                                           tokenize=False, **ckw)
            ids = TOK(text, add_special_tokens=False).input_ids
        except Exception as e:
            print(f"[serve] responses template error: {e} -- roles: "
                  f"{[m.get('role') for m in msgs]} -- items: "
                  f"{[i.get('type') for i in (req.get('input') if isinstance(req.get('input'), list) else [])]}",
                  file=sys.stderr, flush=True)
            return self._json(400, {"error": {"message": f"chat template: {e}"}})
        thinking_primed = text.rstrip().endswith("<think>")
        if len(ids) < 2:
            ids = (TOK("\n", add_special_tokens=False).input_ids + list(ids))[-2:]
        if len(ids) + max_new > args.ctx:
            max_new = max(1, args.ctx - len(ids))

        rid = "resp_" + uuid.uuid4().hex[:24]
        created = int(time.time())
        stream = bool(req.get("stream", False))
        # reasoning streams as its own Responses item (the Codex app renders it in
        # the thinking bubble); only post-</think> text belongs to the message item
        THINK_END = "</think>"
        oi_msg = 1 if thinking_primed else 0
        rs_item_id = "rs_" + uuid.uuid4().hex[:20]
        text_parts, stop_hit = [], threading.Event()
        msg_item_id = "msg_" + uuid.uuid4().hex[:24]

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            seq = {"n": 0}

            def sse(obj):
                obj = dict(obj, sequence_number=seq["n"])   # app-server UI keys off it
                seq["n"] += 1
                data = f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode()
                self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                self.wfile.flush()

            sse({"type": "response.created", "response": {
                "id": rid, "object": "response", "created_at": created,
                "model": args.model_name, "status": "in_progress", "output": []}})
            if thinking_primed:
                sse({"type": "response.output_item.added", "output_index": 0,
                     "item": {"id": rs_item_id, "type": "reasoning", "summary": []}})
                sse({"type": "response.reasoning_summary_part.added", "output_index": 0,
                     "item_id": rs_item_id, "summary_index": 0,
                     "part": {"type": "summary_text", "text": ""}})
            sse({"type": "response.output_item.added", "output_index": oi_msg,
                 "item": {"id": msg_item_id, "type": "message", "status": "in_progress",
                          "role": "assistant", "content": []}})
            sse({"type": "response.content_part.added", "output_index": oi_msg,
                 "item_id": msg_item_id, "content_index": 0,
                 "part": {"type": "output_text", "text": ""}})

            dec_state = {"buf": [], "rs_sent": 0, "ct_sent": 0,
                         "think_open": thinking_primed, "think_done_rs": ""}

            def _hold_tag_tail(body):
                # chars at the end that prefix "</think>": hold until decided
                for m in range(len(THINK_END) - 1, 0, -1):
                    if body.endswith(THINK_END[:m]):
                        return body[:-m]
                return body

            def on_tokens(toks):
                if stop_hit.is_set():
                    return
                dec_state["buf"].extend(toks)
                txt = TOK.decode(dec_state["buf"])
                if txt.endswith("\ufffd"):
                    return
                if tools and (TOOL_OPEN in ("".join(text_parts) + txt) or dec_state.get("tool")):
                    dec_state["tool"] = True         # buffer tool block to the end
                    dec_state["buf"] = []
                    text_parts.append(txt)
                    return
                dec_state["buf"] = []
                text_parts.append(txt)
                full_ns = "".join(text_parts)
                for st in stops:
                    if st and st in full_ns:
                        stop_hit.set()
                        return
                if dec_state["think_open"]:
                    body = _hold_tag_tail(full_ns)
                    if THINK_END in body:
                        rs_all, after = body.split(THINK_END, 1)
                        d = rs_all[dec_state["rs_sent"]:]
                        if d:
                            sse({"type": "response.reasoning_summary_text.delta",
                                 "output_index": 0, "item_id": rs_item_id,
                                 "summary_index": 0, "delta": d})
                        dec_state["rs_sent"] = len(rs_all)
                        dec_state["think_open"] = False
                        dec_state["think_done_rs"] = rs_all
                        dec_state["content0"] = after.lstrip("\n")
                        dec_state["ct_sent"] = len(dec_state["content0"])
                        if dec_state["content0"]:
                            sse({"type": "response.output_text.delta", "output_index": oi_msg,
                                 "item_id": msg_item_id, "content_index": 0,
                                 "delta": dec_state["content0"]})
                    else:
                        d = body[dec_state["rs_sent"]:]
                        if d:
                            sse({"type": "response.reasoning_summary_text.delta",
                                 "output_index": 0, "item_id": rs_item_id,
                                 "summary_index": 0, "delta": d})
                        dec_state["rs_sent"] = len(body)
                    return
                cont = (full_ns.split(THINK_END, 1)[1].lstrip("\n")
                        if thinking_primed else full_ns)
                d = cont[dec_state["ct_sent"]:]
                if d:
                    sse({"type": "response.output_text.delta", "output_index": oi_msg,
                         "item_id": msg_item_id, "content_index": 0, "delta": d})
                dec_state["ct_sent"] = len(cont)
        else:
            def on_tokens(toks):
                text_parts.append(TOK.decode(toks))

        try:
            ngen, finish, stats = generate(list(ids), max_new, temp, seed, on_tokens,
                                           no_cache=no_cache, dbg=dbg)
        except BrokenPipeError:
            return
        full = "".join(text_parts)
        for st in stops:
            if st and st in full:
                full = full.split(st)[0]
        reasoning = ""
        if thinking_primed:
            reasoning, full = split_think(full, thinking_primed)
            reasoning, full = reasoning.strip("\n"), full.lstrip("\n")
        tool_calls = None
        if tools:
            full, tool_calls = parse_tool_calls(full, tools)
        full = full or None

        out_items = []
        if reasoning:
            out_items.append({"id": "rs_" + uuid.uuid4().hex[:20], "type": "reasoning",
                              "summary": [{"type": "summary_text", "text": reasoning}]})
        msg_out = {"id": msg_item_id, "type": "message", "status": "completed",
                   "role": "assistant",
                   "content": ([{"type": "output_text", "text": full}] if full else [])}
        out_items.append(msg_out)
        fc_items = []
        for tc in (tool_calls or []):
            fc_items.append({"id": "fc_" + uuid.uuid4().hex[:20], "type": "function_call",
                             "status": "completed", "call_id": tc["id"],
                             "name": tc["function"]["name"],
                             "arguments": tc["function"]["arguments"]})
        out_items.extend(fc_items)

        resp_obj = {"id": rid, "object": "response", "created_at": created,
                    "model": args.model_name, "status": "completed",
                    "output": out_items,
                    "usage": {"input_tokens": len(ids), "output_tokens": ngen,
                              "total_tokens": len(ids) + ngen},
                    "x_qwentin": stats}
        if stream:
            if thinking_primed:
                sse({"type": "response.reasoning_summary_text.done", "output_index": 0,
                     "item_id": rs_item_id, "summary_index": 0, "text": reasoning})
                sse({"type": "response.reasoning_summary_part.done", "output_index": 0,
                     "item_id": rs_item_id, "summary_index": 0,
                     "part": {"type": "summary_text", "text": reasoning}})
                sse({"type": "response.output_item.done", "output_index": 0,
                     "item": out_items[0]})
            sse({"type": "response.output_text.done", "output_index": oi_msg,
                 "item_id": msg_item_id, "content_index": 0, "text": full or ""})
            sse({"type": "response.content_part.done", "output_index": oi_msg,
                 "item_id": msg_item_id, "content_index": 0,
                 "part": {"type": "output_text", "text": full or ""}})
            sse({"type": "response.output_item.done", "output_index": oi_msg,
                 "item": msg_out})
            oi = oi_msg + 1
            for fc in fc_items:
                sse({"type": "response.output_item.added", "output_index": oi,
                     "item": {**fc, "arguments": ""}})
                sse({"type": "response.function_call_arguments.delta",
                     "output_index": oi, "item_id": fc["id"], "delta": fc["arguments"]})
                sse({"type": "response.function_call_arguments.done",
                     "output_index": oi, "item_id": fc["id"], "arguments": fc["arguments"]})
                sse({"type": "response.output_item.done", "output_index": oi, "item": fc})
                oi += 1
            sse({"type": "response.completed", "response": resp_obj})
            data = b"data: [DONE]\n\n"
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            return
        self._json(200, resp_obj)

    def do_GET(self):
        if self.path == "/health":
            return self._json(200, {"status": "ok"})
        if self.path == "/v1/models":
            return self._json(200, {"object": "list", "data": [
                {"id": args.model_name, "object": "model", "owned_by": "qwentin"}]})
        return self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._json(400, {"error": {"message": f"bad json: {e}"}})
        chat = self.path.rstrip("/").endswith("/chat/completions")
        comp = self.path.rstrip("/").endswith("/completions") and not chat
        if self.path.rstrip("/").endswith("/responses"):
            return self._do_responses(req)
        if not (chat or comp):
            return self._json(404, {"error": {"message": "not found"}})
        if req.get("n", 1) != 1 or req.get("logprobs"):
            return self._json(400, {"error": {"message": "n>1/logprobs unsupported"}})
        tools = req.get("tools") or None
        warn = None
        top_p = float(req.get("top_p", 1.0))
        if top_p != 1.0:
            warn = f"top_p={top_p} unsupported, clamped to 1.0"
        temp = float(req.get("temperature", 1.0 if chat else 1.0))
        seed = int(req.get("seed", int.from_bytes(os.urandom(4), "little")))
        max_new = int(req.get("max_tokens") or req.get("max_completion_tokens") or 256)
        stops = req.get("stop") or []
        if isinstance(stops, str):
            stops = [stops]
        # non-OpenAI test knobs: tl_no_cache forces the full reset+prefill path,
        # tl_debug echoes emitted/engine token ids in x_qwentin (gate tooling)
        no_cache = bool(req.get("tl_no_cache", False))
        dbg = bool(req.get("tl_debug", False))

        thinking_primed = False
        if chat:
            msgs = req.get("messages", [])
            # OpenAI wire format carries tool_call arguments as a JSON STRING;
            # the Qwen template iterates them as a mapping -- parse in place.
            for m in msgs:
                for tc in (m.get("tool_calls") or []):
                    fn = tc.get("function") or {}
                    if isinstance(fn.get("arguments"), str):
                        try:
                            fn["arguments"] = json.loads(fn["arguments"])
                        except Exception:
                            pass
            ckw = dict(req.get("chat_template_kwargs") or {})
            # Resolve thinking with explicit precedence (high -> low):
            #   1. top-level enable_thinking            (request is explicit)
            #   2. chat_template_kwargs.enable_thinking  (honored as-is)
            #   3. reasoning_effort                      (none/minimal/off -> off, else on;
            #                                            OpenAI-style knob, vLLM #43401 parity)
            #   4. server --no-thinking default          -> off
            #   5. otherwise leave to the chat template  (Qwen default = on)
            if "enable_thinking" in req:
                ckw["enable_thinking"] = bool(req["enable_thinking"])
            elif "enable_thinking" in ckw:
                pass
            elif req.get("reasoning_effort") is not None or req.get("reasoningEffort") is not None:
                # accept snake_case (OpenAI/vLLM convention) and camelCase
                # (Vercel AI SDK forwards providerOptions like opencode
                # --variant selections verbatim)
                eff = str(req.get("reasoning_effort", req.get("reasoningEffort"))).strip().lower()
                ckw["enable_thinking"] = eff not in ("none", "minimal", "off", "0", "false", "")
            elif args.no_thinking:
                ckw["enable_thinking"] = False
            try:
                text = TOK.apply_chat_template(msgs, tools=tools, add_generation_prompt=True,
                                               tokenize=False, **ckw)
                ids = TOK(text, add_special_tokens=False).input_ids
            except Exception as e:
                return self._json(400, {"error": {"message": f"chat template: {e}"}})
            # template opened an unclosed <think> -> model output carries reasoning
            # up to </think>; split it out of content into reasoning_content.
            thinking_primed = text.rstrip().endswith("<think>")
        else:
            ids = TOK(req.get("prompt", ""), add_special_tokens=False).input_ids
        if len(ids) < 2:
            ids = (TOK("\n", add_special_tokens=False).input_ids + list(ids))[-2:]
        if len(ids) + max_new > args.ctx:
            max_new = max(1, args.ctx - len(ids))
        rid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:24]
        created = int(time.time())
        stream = bool(req.get("stream", False))
        obj_t = "chat.completion" if chat else "text_completion"

        text_parts, stop_hit = [], threading.Event()

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            def sse(obj):
                data = f"data: {json.dumps(obj)}\n\n".encode()
                self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                self.wfile.flush()

            if chat:
                sse({"id": rid, "object": "chat.completion.chunk", "created": created,
                     "model": args.model_name, "choices": [{"index": 0, "delta": {"role": "assistant"}}]})

            dec_state = {"buf": []}

            def on_tokens(toks):
                if stop_hit.is_set():
                    return
                dec_state["buf"].extend(toks)
                txt = TOK.decode(dec_state["buf"])
                if txt.endswith("\ufffd"):   # incomplete utf-8: hold the tail
                    return
                # tool-call block: stop streaming content, buffer to the end
                # (v1: a detected opening tag mutes the stream; the parsed calls
                # go out in the final chunk as a tool_calls delta)
                if tools and (TOOL_OPEN in ("".join(text_parts) + txt) or dec_state.get("tool")):
                    dec_state["tool"] = True
                    dec_state["buf"] = []
                    text_parts.append(txt)
                    return
                dec_state["buf"] = []
                text_parts.append(txt)
                full = "".join(text_parts)
                for st in stops:
                    if st and st in full:
                        stop_hit.set()
                        return
                if chat:
                    # route deltas: reasoning_content before </think>, content after.
                    # recompute the split from the full text each chunk so a close
                    # tag straddling a chunk boundary still lands correctly.
                    r_full, c_full = split_think(full, thinking_primed)
                    r_new = r_full[dec_state.get("r_emit", 0):]
                    c_new = c_full[dec_state.get("c_emit", 0):]
                    dec_state["r_emit"] = len(r_full)
                    dec_state["c_emit"] = len(c_full)
                    delta = {}
                    if r_new:
                        delta["reasoning_content"] = r_new
                    if c_new:
                        delta["content"] = c_new
                    if not delta:
                        return
                    ch = {"index": 0, "delta": delta}
                else:
                    ch = {"index": 0, "text": txt}
                sse({"id": rid, "object": obj_t + (".chunk" if chat else ""),
                     "created": created, "model": args.model_name, "choices": [ch]})

            ngen, finish, stats = generate(list(ids), max_new, temp, seed, on_tokens,
                                           no_cache=no_cache, dbg=dbg)
            if stop_hit.is_set():
                finish = "stop"
            final_delta = {}
            if chat and tools and dec_state.get("tool"):
                _, tcs = parse_tool_calls("".join(text_parts), tools)
                if tcs:
                    for i, tc in enumerate(tcs):
                        tc["index"] = i
                    final_delta = {"tool_calls": tcs}
                    finish = "tool_calls"
            ch = ({"index": 0, "delta": final_delta, "finish_reason": finish} if chat
                  else {"index": 0, "text": "", "finish_reason": finish})
            sse({"id": rid, "object": obj_t + (".chunk" if chat else ""), "created": created,
                 "model": args.model_name, "choices": [ch],
                 "usage": {"prompt_tokens": len(ids), "completion_tokens": ngen,
                           "total_tokens": len(ids) + ngen},
                 "x_qwentin": stats})
            data = b"data: [DONE]\n\n"
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            return

        def on_tokens_ns(toks):
            text_parts.append(TOK.decode(toks))

        ngen, finish, stats = generate(list(ids), max_new, temp, seed, on_tokens_ns,
                                       no_cache=no_cache, dbg=dbg)
        text = "".join(text_parts)
        for st in stops:
            if st and st in text:
                text = text.split(st)[0]
                finish = "stop"
        reasoning = ""
        if chat:
            reasoning, text = split_think(text, thinking_primed)
            reasoning, text = reasoning.strip("\n"), text.lstrip("\n")
        tool_calls = None
        if chat and TOOL_OPEN in text:
            text, tool_calls = parse_tool_calls(text, tools)
        msg = {"role": "assistant", "content": text or None}
        if reasoning:
            msg["reasoning_content"] = reasoning
        if tool_calls:
            msg["tool_calls"] = tool_calls
            finish = "tool_calls"
        choice = ({"index": 0, "message": msg, "finish_reason": finish} if chat
                  else {"index": 0, "text": text, "finish_reason": finish})
        out = {"id": rid, "object": obj_t, "created": created, "model": args.model_name,
               "choices": [choice],
               "usage": {"prompt_tokens": len(ids), "completion_tokens": ngen,
                         "total_tokens": len(ids) + ngen},
               "x_qwentin": stats}
        if warn:
            out["x_qwentin"]["warning"] = warn
        self._json(200, out)


if __name__ == "__main__":
    ThreadingHTTPServer((args.host, args.port), H).serve_forever()
