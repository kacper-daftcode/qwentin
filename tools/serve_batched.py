#!/usr/bin/env python3
"""Roadmap E milestone 3.2 + 3.3: continuous-batching scheduler + multi-client OpenAI server.

A single engine thread owns the CUDA engine (paged Q4 KV pool). HTTP handler threads only
tokenize, submit a Request, and wait on its completion event. The engine thread runs a
continuous-batching loop:
  - ADMIT queued requests into free slots (their prompts prefill in paged waves, below), subject
    to admission control (free slot + enough free pool blocks).
  - DECODE one paged step over ALL active slots (qwn_paged_decode_step), push each slot's new
    token to its request, advance positions, and DETACH finished requests (EOS / max_tokens),
    returning their blocks to the pool.

Decodes are batched (the throughput win). Short prompts are batched into one ragged prefill wave;
a long prompt is prefilled in <=128-column waves that carry the active slots' decode steps as
1-token final segments (--mixed-prefill, default), so decoders keep producing one token per wave
while the long prompt prefills at ~full speed (--no-mixed-prefill alternates waves and decode
steps; --prefill-waves-per-step 0 restores the blocking whole-prompt prefill).

Modes:
  --selftest     : staggered join/leave correctness vs single-stream + aggregate tok/s (no HTTP)
  (default)      : run the OpenAI server on --port

Run (GPU7; the engine's defaults are the served line's, so the server needs no TQ_* environment):
  CUDA_VISIBLE_DEVICES=7 TQ_CTX=8192 python3 -u tools/serve_batched.py --selftest --tqf ... --model-dir ...
  CUDA_VISIBLE_DEVICES=7 python3 -u tools/serve_batched.py --port 8100 --tqf ... --model-dir ...

Defaults since 10-16 (owner, HANDOFF Decisions 16): --kv-v4 --kv-t4 (trellis-4 K+V, 16.5 KiB/token), --spec (the paged
MTP drafter; needs the --mtp-exl3 tqf, e.g. qwen3_8-27b-exl3-opt5.75-mtp4.tqf, else it warns and runs plain decode) and
the engine's TQ_PAGED_ATTN_ST=3 (a no-op under --kv-t4). Since 10-17 also --dstate fp16sr (DeltaNet state in fp16 with
deterministic stochastic rounding, 80 MiB/slot) and --decode-steps-per-wave 1, with the engine defaults TQ_T4_ATTN_F16=1
TQ_GRAPH=1 TQ_GEMM_FORK=1. The served line on one 32 GB RTX 5090: 12 slots, --num-blocks 4500 (576K pooled tokens;
~5000 = 640K with --no-spec). --no-kv-t4 / --no-spec / TQ_PAGED_ATTN_ST=1 revert to the 10-13 stack; TQ_T4_ATTN_F16=0
TQ_GRAPH=0 TQ_GEMM_FORK=0 --dstate fp32 --decode-steps-per-wave 0 --num-blocks 4100 restore the 10-16 line bit for bit.
"""
from __future__ import annotations
import argparse, ctypes, json, os, sys, threading, time, queue, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WAVE_MAX = 128   # prefill wave column cap = the fast activation quantizer's smem limit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwentin_abi  # noqa: E402
from serve_common import (BodyError, add_hardening_args, apply_socket_timeout, auth_ok, clamp_fields,  # noqa: E402
                          clamp_max_tokens, read_json_body, unauthorized_body)


def load_lib(path):
    L = ctypes.CDLL(path)
    L.qwn_init.argtypes = [ctypes.c_char_p]; L.qwn_init.restype = ctypes.c_int
    L.qwn_hidden_size.restype = ctypes.c_int
    L.qwn_reset_state.restype = ctypes.c_int
    L.qwn_decode.argtypes = [ctypes.c_int, ctypes.c_int]; L.qwn_decode.restype = ctypes.c_int
    L.qwn_paged_init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]; L.qwn_paged_init.restype = ctypes.c_int
    L.qwn_paged_free.restype = ctypes.c_int
    L.qwn_paged_reset_slot.argtypes = [ctypes.c_int]; L.qwn_paged_reset_slot.restype = ctypes.c_int
    L.qwn_paged_load_client.argtypes = [ctypes.c_int, ctypes.c_int]; L.qwn_paged_load_client.restype = ctypes.c_int
    L.qwn_paged_decode_step.argtypes = [ctypes.POINTER(ctypes.c_int)] * 3 + [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    L.qwn_paged_decode_step.restype = ctypes.c_int
    L.qwn_paged_stats.argtypes = [ctypes.POINTER(ctypes.c_int)] * 4; L.qwn_paged_stats.restype = ctypes.c_int
    L.qwn_paged_prefill_batch.argtypes = [ctypes.POINTER(ctypes.c_int)] * 7 + [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    L.qwn_paged_prefill_batch.restype = ctypes.c_int
    L.qwn_free.restype = None
    # session parking (slot KV blocks + DeltaNet state <-> host buffer); older builds lack these
    try:
        L.qwn_paged_slot_bytes.argtypes = [ctypes.c_int]; L.qwn_paged_slot_bytes.restype = ctypes.c_size_t
        L.qwn_paged_slot_save.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
        L.qwn_paged_slot_save.restype = ctypes.c_int
        L.qwn_paged_slot_restore.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
        L.qwn_paged_slot_restore.restype = ctypes.c_int
        L.has_parking = True
    except AttributeError:
        L.has_parking = False
    try:
        L.qwn_paged_token_bytes.restype = ctypes.c_size_t
        L.has_token_bytes = True
    except AttributeError:
        L.has_token_bytes = False
    # speculative decoding (task 10): verify / commit waves + the paged MTP drafter; older builds lack these
    try:
        L.qwn_paged_verify.argtypes = [ctypes.POINTER(ctypes.c_int)] * 6 + [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        L.qwn_paged_verify.restype = ctypes.c_int
        L.qwn_paged_commit.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        L.qwn_paged_commit.restype = ctypes.c_int
        L.qwn_paged_mtp_init.argtypes = [ctypes.c_int]; L.qwn_paged_mtp_init.restype = ctypes.c_int
        L.qwn_paged_mtp_draft.argtypes = [ctypes.POINTER(ctypes.c_int)] * 3 + [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        L.qwn_paged_mtp_draft.restype = ctypes.c_int
        L.has_spec = True
    except AttributeError:
        L.has_spec = False
    # the configuration struct (qwn_config): qwn_init_with takes the server's settings directly; older builds lack it
    try:
        L.qwn_config_from_env.argtypes = [ctypes.c_void_p]; L.qwn_config_from_env.restype = None
        L.qwn_config_get.argtypes = [ctypes.c_void_p]; L.qwn_config_get.restype = None
        L.qwn_init_with.argtypes = [ctypes.c_char_p, ctypes.c_void_p]; L.qwn_init_with.restype = ctypes.c_int
        L.has_config = True
    except AttributeError:
        L.has_config = False
    return L


def config_overrides(args):
    """The engine switches the server's arguments set, by TQ_* name (None = the engine's built-in codebook)."""
    return {"TQ_WIDE_ATTN_MMA": int(args.wide_attn_mma and args.page >= 128), "TQ_EMBED_HOST": int(args.embed_host),
            "TQ_NO_MTP": int(not args.mtp), "TQ_KV_V4": int(args.kv_v4), "TQ_KV_T4": int(args.kv_t4),
            "TQ_KV_T4_CODEBOOK": args.kv_t4_codebook or None, "TQ_PDL": int(args.pdl), "TQ_DSTATE_BF16": args.dstate,
            "TQ_GDN_PREFILL": int(args.gdn_prefill)}


def engine_config(L, overrides):
    """The qwn_config the server starts the engine with: the TQ_* environment as the engine reads it, then `overrides`
    for every switch the environment leaves unset or empty (explicit environment overrides the arguments)."""
    cfg = qwentin_abi.Config()
    L.qwn_config_from_env(ctypes.byref(cfg))
    for name, value in overrides.items():
        if not os.environ.get(name, ""):
            setattr(cfg, qwentin_abi.CONFIG_FIELDS[name], value.encode() if isinstance(value, str) else value)
    return cfg


def init_engine(L, tqf, overrides):
    """qwn_init with the server's settings: qwn_init_with(engine_config(...)), or, with a library older than it, the
    same precedence through the environment."""
    if getattr(L, "has_config", False):
        cfg = engine_config(L, overrides)
        kv_t4, kv_v4 = cfg.kv_t4, cfg.kv_v4
        cb = cfg.kv_t4_codebook.decode() if cfg.kv_t4_codebook else ""
    else:
        for name, value in overrides.items():
            if value is not None:
                os.environ.setdefault(name, str(value))
        kv_t4 = os.environ.get("TQ_KV_T4", "0") not in ("", "0")
        kv_v4 = os.environ.get("TQ_KV_V4", "0") not in ("", "0")
        cb = os.environ.get("TQ_KV_T4_CODEBOOK", "")
    if kv_t4 and not kv_v4:
        sys.exit("--kv-t4 needs --kv-v4 (the open page is staged as int4 K + V)")
    if kv_t4 and cb and not os.path.exists(cb):
        sys.exit(f"--kv-t4: codebook {cb} not found (--kv-t4-codebook / TQ_KV_T4_CODEBOOK; "
                 f"unset = the engine's built-in lloyd_kv12a)")
    if getattr(L, "has_config", False):
        ck(L.qwn_init_with(tqf.encode(), ctypes.byref(cfg)), "init")
    else:
        ck(L.qwn_init(tqf.encode()), "init")


def engine_settings(L):
    """The engine's effective kv_v4 / kv_t4 / park_ring (qwn_config_get; the environment with an older library)."""
    if getattr(L, "has_config", False):
        cfg = qwentin_abi.Config()
        L.qwn_config_get(ctypes.byref(cfg))
        return {"kv_v4": bool(cfg.kv_v4), "kv_t4": bool(cfg.kv_t4), "park_ring": int(cfg.park_ring)}
    return {"kv_v4": os.environ.get("TQ_KV_V4", "0") not in ("", "0"),
            "kv_t4": os.environ.get("TQ_KV_T4", "0") not in ("", "0"),
            "park_ring": int(os.environ.get("TQ_PARK_RING", "1") not in ("", "0"))}


TQ_FLAG_MTP_EXL3 = 1 << 12      # tqf header flag: the MTP projections are EXL3 records (convert_qwen_tqf.py --mtp-exl3)


def tqf_has_exl3_mtp(path):
    """True/False from the tqf header's flags word (offset 8); None if the header cannot be read."""
    try:
        with open(path, "rb") as f:
            h = f.read(12)
        if len(h) < 12 or h[:4] != b"TQF1":
            return None
        return bool(int.from_bytes(h[8:12], "little") & TQ_FLAG_MTP_EXL3)
    except OSError:
        return None


def attn_st_summary(L):
    """'decode attention TQ_PAGED_ATTN_ST=<n> (...)' for the startup line: the engine's effective value when the library
    exports qwn_paged_attn_st (>= the 10-16 defaults commit), else the environment with the engine's default rule."""
    try:
        L.qwn_paged_attn_st.restype = ctypes.c_int
        st = int(L.qwn_paged_attn_st())
    except AttributeError:
        e = os.environ.get("TQ_PAGED_ATTN_ST", "")
        st = int(e) if e.strip().lstrip("-").isdigit() else 1
    es = engine_settings(L)
    t4, v4 = es["kv_t4"], es["kv_v4"]
    what = ("a no-op under --kv-t4 today: the trellis pool runs its own GQA kernels" if t4
            else "fp16 int4 GQA decode/verify kernel" if (st >= 2 and v4)
            else "E4M3 V: base kernel (the fp16 path needs --kv-v4)" if st >= 2
            else "staged-V kernel, bitwise = e1ecb3a" if st == 1 else "base kernel, bitwise = e1ecb3a")
    return f"decode attention TQ_PAGED_ATTN_ST={st} ({what})"


def kv_summary(L, total_blocks, page):
    """'KV <format> <KiB>/token, <tokens> pooled tokens' for the startup line (needs qwn_paged_token_bytes)."""
    es = engine_settings(L)
    fmt = "int4 K + int4 V" if es["kv_v4"] else "int4 K + E4M3 V"
    if es["kv_t4"]:
        fmt = "trellis-4 K + V (12-bit states, every committed token; in-flight columns exact fp16)"
    if not L.has_token_bytes:
        return fmt
    tb = int(L.qwn_paged_token_bytes())
    s = f"{fmt} {tb / 1024:.2f} KiB/token, {total_blocks * page} pooled tokens"
    try:   # TQ_DSTATE_BF16 (dstate): the engine's effective recurrent-state format + the per-slot DeltaNet state bytes
        L.qwn_paged_dstate.restype = ctypes.c_int
        L.qwn_paged_state_bytes.restype = ctypes.c_size_t
        ds = int(L.qwn_paged_dstate())
        s += f"; DeltaNet state {('fp32', 'bf16', 'fp16', 'bf16+8', 'fp16sr')[ds]} {int(L.qwn_paged_state_bytes()) / 2**20:.0f} MiB/slot"
    except (AttributeError, IndexError):
        pass
    return s


def paged_prefill_slot(L, slot, ids, page, wave_max=128):
    """Prefill one paged slot to len(ids) tokens via qwn_paged_prefill_batch in
    <=wave_max-col chunks (the server's _prefill_long path; allocates blocks from the
    pool, no single-stream/spec scratch). Returns the seed (argmax of the final chunk).
    Raises on -4 (pool exhausted = this worker count does not fit)."""
    n = len(ids); pos = 0; seed = 0
    oseed = (ctypes.c_int * 1)()
    while pos < n:
        c = min(wave_max, n - pos)
        final = 1 if pos + c >= n else 0
        rc = L.qwn_paged_prefill_batch(
            _ci([int(x) for x in ids[pos:pos + c]]), _ci([slot] * c),
            _ci(list(range(pos, pos + c))), _ci([slot]), _ci([0]), _ci([c]),
            _ci([final]), 1, c, oseed)
        if rc != 0:
            raise RuntimeError(f"qwn_paged_prefill_batch rc={rc} at pos {pos} slot {slot}")
        seed = oseed[0]; pos += c
    return seed


def _ci(a):
    return (ctypes.c_int * len(a))(*a)


def ck(r, what):
    if isinstance(r, int) and r < 0:
        raise RuntimeError(f"{what} failed: {r}")
    return r


class Request:
    __slots__ = ("ids", "max_new", "eos", "out", "done", "slot", "pos", "next_tok",
                 "started", "t_admit", "t_first", "t_done", "n_prompt", "err",
                 "start", "tokens", "t_submit")

    def __init__(self, ids, max_new, eos):
        self.ids = ids; self.max_new = max_new; self.eos = set(eos)
        self.out = []; self.done = threading.Event(); self.slot = -1
        self.pos = 0; self.next_tok = 0; self.started = False
        self.t_admit = self.t_first = self.t_done = 0.0; self.n_prompt = len(ids); self.err = None
        self.start = 0              # first prompt position to prefill (> 0 = prefix-cache hit)
        self.tokens = queue.Queue() # streamed token ids; None = finished (see BatchedEngine._finish)
        self.t_submit = time.time()


SPEC_MAX_T = 64      # verify wave column cap (qwn_paged_verify)


class BatchedEngine:
    def __init__(self, lib, tqf, max_slots, num_blocks, page, prefill_waves_per_step=2,
                 mixed_prefill=True, prefix_cache=True, park_ram_bytes=0, decode_steps_per_wave=0,
                 spec=False, spec_k="auto", park_ram_ttl=0.0, cfg_overrides=None):
        self.L = lib
        init_engine(self.L, tqf, cfg_overrides or {})
        self.page = page; self.max_slots = max_slots
        ck(self.L.qwn_paged_init(max_slots, num_blocks, page), "paged_init")
        fb, tb, pg, mb = self._stats()
        self.num_blocks = tb; self.max_blocks_per_seq = mb
        # Speculative decoding (task 10): in decode-only phases every engine iteration runs a ROUND
        # instead of a step -- the paged MTP drafter guesses K tokens per slot (qwn_paged_mtp_draft),
        # one verify wave scores [next, d_1..d_K] per slot (qwn_paged_verify), the longest matching
        # prefix plus the bonus token is committed (qwn_paged_commit replays the DeltaNet state over
        # the accepted columns; the KV is already in the pool). Needs the MTP section loaded (--mtp)
        # from a --mtp-exl3 tqf; the drafter keeps its own KV rows on the pool blocks (+1.25 KiB/token)
        # and a per-slot last hidden, both parked/restored with the slot. Mixed prefill waves keep
        # the 1-token decode columns (the drafter's trunk is maintained there too). spec_on can be
        # toggled at run time (/spec?on=0|1) for A/B: off = plain steps with the drafter idle.
        self.spec = bool(spec)
        self.spec_on = self.spec
        self.spec_k = spec_k
        self.spec_off_reason = ""
        if self.spec:
            if not getattr(lib, "has_spec", False):
                raise RuntimeError("--spec needs an engine library with qwn_paged_verify/commit/mtp_* (build >= eae20ab)")
            rc = self.L.qwn_paged_mtp_init(1)
            if rc == -3:
                # --spec is the default since 10-16: a tqf without the EXL3 MTP section (an older ...-mtp.tqf) runs
                # plain decode instead of refusing to start (convert_qwen_tqf.py --mtp-exl3 makes the -mtp4.tqf).
                self.spec_off_reason = ("the tqf has no EXL3 MTP section (qwn_paged_mtp_init -> -3); use the --mtp-exl3 "
                                        "tqf (...-mtp4.tqf) for speculative decoding")
                print(f"[spec] WARNING: speculative decoding OFF: {self.spec_off_reason}", flush=True)
                self.spec = self.spec_on = False
            elif rc != 0:
                raise RuntimeError(f"qwn_paged_mtp_init failed: {rc} (needs --mtp and a --mtp-exl3 tqf, e.g. ...-mtp4.tqf)")
        self.rounds = 0; self.spec_tokens = 0; self.spec_cols = 0; self.spec_fallbacks = 0
        self.spec_draft_s = self.spec_verify_s = self.spec_commit_s = 0.0
        self.log_admits = os.environ.get("TQ_SERVE_LOG_ADMITS", "1") not in ("", "0")   # one line per admitted request
        self.free_slots = list(range(max_slots))
        self.active = {}            # slot -> Request
        self.q = []                 # pending Requests (FIFO)
        # Slot-level prefix cache: a finished request leaves its slot PARKED (KV blocks + DeltaNet
        # state intact) together with its token history = prompt + generated tokens. The engine has
        # consumed history[:-1] (the last generated token was emitted, never fed), so a new request
        # whose ids extend history[:-1] continues from position len(history)-1 with a normal
        # continuation prefill of just the new tokens (an agent's follow-up turn: assistant reply +
        # tool output + question). Parked slots are evicted LRU when a slot or blocks are needed.
        self.prefix_cache = prefix_cache
        self.parked = {}            # slot -> dict(ids, consumed, t, nblocks)
        self.cache_hits = 0; self.cache_tokens = 0
        # Second tier: a GPU-parked slot that has to be evicted is SAVED to host RAM (its KV blocks
        # + DeltaNet state, ~26 KiB/token + 155 MiB) instead of dropped; a later follow-up restores
        # it into a free slot (~1 GB per 40K tokens, sub-second) instead of re-prefilling for tens of
        # seconds. LRU within the RAM budget. Needs the qwn_paged_slot_save/restore ABI.
        self.park_ram_bytes = park_ram_bytes if getattr(lib, "has_parking", False) else 0
        self.host_parked = {}       # key -> dict(ids, consumed, buf, nbytes, t)
        self.host_bytes = 0; self.host_key = 0
        self.host_hits = 0; self.host_saves = 0; self.host_save_s = 0.0; self.host_restore_s = 0.0
        self.host_save_bytes = 0; self.host_restore_bytes = 0
        # RAM-tier policy: LRU within --park-ram-gb (above), plus an optional age limit (--park-ram-ttl s, 0 = off):
        # a parked session older than the TTL is dropped at the next admission (its blob freed) -- the follow-up
        # that never came should not pin RAM for hours. Counters per reason for /health.
        self.park_ram_ttl = float(park_ram_ttl or 0.0)
        self.host_evict_lru = 0; self.host_evict_ttl = 0
        # Long-prompt prefill in flight, interleaved with decode steps: at most one at a time,
        # advanced by <= prefill_waves_per_step waves (<= WAVE_MAX columns each) per engine
        # iteration so the active decoders keep stepping (0 = legacy: whole prompt at once).
        self.prefills = []          # in-flight long prefills: dict(req, slot, pos, need)
        # In-flight COLD prefills (start == 0) are capped: each holds a slot and a growing share of
        # the pool while it runs, and 8 simultaneous 32K cold starts thrash the parked tier (54 host
        # saves / 43 restores in one 4-turn run vs 15 / 12 with the cap). Continuations (start > 0,
        # the agent follow-up) always enter -- SRPT finishes them in a few waves.
        self.max_cold_prefills = 2
        self.pf_round = 0
        self.prefill_waves_per_step = prefill_waves_per_step
        self.mixed_prefill = mixed_prefill
        # extra pure decode steps after every mixed wave: with the GQA decode kernel a step costs
        # ~17-25 ms vs ~70-110 ms per prefill wave, so k=1 gives decoders 2 tokens per ~95 ms
        # (~21 tok/s) instead of 1 per wave (~11-14), at ~25-35% lower prefill speed (0 = off).
        self.decode_steps_per_wave = decode_steps_per_wave
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.running = True
        self.steps = 0; self.decoded_tokens = 0
        self.prefill_waves = 0; self.prefilled_tokens = 0
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _stats(self):
        fb, tb, pg, mb = (ctypes.c_int(), ctypes.c_int(), ctypes.c_int(), ctypes.c_int())
        ck(self.L.qwn_paged_stats(ctypes.byref(fb), ctypes.byref(tb), ctypes.byref(pg), ctypes.byref(mb)), "stats")
        return fb.value, tb.value, pg.value, mb.value

    def submit(self, ids, max_new, eos):
        req = Request(ids, max_new, eos)
        with self.cv:
            self.q.append(req)
            self.cv.notify()
        return req

    def cancel_queued(self, req):
        # drop a request that is still waiting for admission (client gave up); active ones run on
        with self.cv:
            if req in self.q:
                self.q.remove(req)
                return True
        return False

    def _free_blocks(self):
        return self._stats()[0]

    def _finish(self, req, err=None):
        # the ONE place a request completes: error (if any), stream sentinel, done event
        if err is not None:
            req.err = err
        req.t_done = time.time()
        req.tokens.put(None)
        req.done.set()

    def _activate(self, req, slot, seed):
        req.slot = slot; req.pos = req.n_prompt; req.next_tok = seed
        req.out.append(seed); req.started = True; req.t_admit = time.time(); req.t_first = req.t_admit
        req.tokens.put(seed)
        self.active[slot] = req
        if seed in req.eos or len(req.out) >= req.max_new:
            self._detach(slot)

    # ---- prefix cache: parked slots ----
    def _park(self, slot, req):
        ids = list(req.ids) + list(req.out)
        consumed = len(ids) - 1                     # positions [0, consumed) have KV; ids[-1] pending
        self.parked[slot] = {"ids": ids, "consumed": consumed, "t": time.time(),
                             "nblocks": (consumed + self.page - 1) // self.page}

    def _evict(self, slot):
        p = self.parked.pop(slot, None)
        if p is not None and self.park_ram_bytes > 0 and p["consumed"] >= self.page:
            self._park_to_host(slot, p)
        ck(self.L.qwn_paged_reset_slot(slot), "reset_slot")
        self.free_slots.append(slot)

    def _park_to_host(self, slot, p):
        upto = p["consumed"] - 1
        nbytes = int(self.L.qwn_paged_slot_bytes(upto))
        if nbytes <= 0 or nbytes > self.park_ram_bytes:
            return
        # make room (LRU) within the RAM budget
        while self.host_parked and self.host_bytes + nbytes > self.park_ram_bytes:
            k = min(self.host_parked, key=lambda k: self.host_parked[k]["t"])
            self.host_bytes -= self.host_parked.pop(k)["nbytes"]; self.host_evict_lru += 1
        buf = (ctypes.c_char * nbytes)()
        t0 = time.time()
        rc = self.L.qwn_paged_slot_save(slot, upto, buf, nbytes)
        dt = time.time() - t0
        if rc != 0:
            print(f"[park] save slot {slot} failed rc={rc}", flush=True)
            return
        self.host_key += 1
        self.host_parked[self.host_key] = {"ids": p["ids"], "consumed": p["consumed"], "buf": buf,
                                           "nbytes": nbytes, "t": p["t"]}
        self.host_bytes += nbytes; self.host_saves += 1; self.host_save_s += dt; self.host_save_bytes += nbytes
        print(f"[park] slot {slot} -> host: {p['consumed']} tokens, {nbytes / 2**20:.0f} MiB in {dt:.2f}s "
              f"({nbytes / 2**30 / max(dt, 1e-6):.1f} GB/s); host tier {len(self.host_parked)} sessions, "
              f"{self.host_bytes / 2**30:.1f} GiB", flush=True)

    def _restore_from_host(self, key, slot):
        """Restore a host-parked session into the (reset, empty) slot. Returns True on success."""
        h = self.host_parked.pop(key)
        self.host_bytes -= h["nbytes"]
        t0 = time.time()
        rc = self.L.qwn_paged_slot_restore(slot, h["consumed"] - 1, h["buf"], h["nbytes"])
        dt = time.time() - t0
        if rc != 0:
            print(f"[park] restore into slot {slot} failed rc={rc} ({h['consumed']} tokens)", flush=True)
            self.L.qwn_paged_reset_slot(slot)
            return False
        self.host_hits += 1; self.host_restore_s += dt; self.host_restore_bytes += h["nbytes"]
        print(f"[park] host -> slot {slot}: {h['consumed']} tokens, {h['nbytes'] / 2**20:.0f} MiB in {dt:.2f}s "
              f"({h['nbytes'] / 2**30 / max(dt, 1e-6):.1f} GB/s)", flush=True)
        return True

    def _sweep_host_ttl(self):
        if self.park_ram_ttl <= 0 or not self.host_parked:
            return
        now = time.time()
        for k in [k for k, h in self.host_parked.items() if now - h["t"] > self.park_ram_ttl]:
            self.host_bytes -= self.host_parked.pop(k)["nbytes"]; self.host_evict_ttl += 1

    def _evict_lru(self):
        if not self.parked:
            return False
        slot = min(self.parked, key=lambda s: self.parked[s]["t"])
        self._evict(slot)
        return True

    def _match_parked(self, ids):
        """Best parked session whose CONSUMED history is a prefix of ids:
        ("gpu", slot, consumed) | ("host", key, consumed) | None. GPU tier wins ties."""
        best = None
        for slot, p in self.parked.items():
            c = p["consumed"]
            if c <= 0 or len(ids) < c or c < self.page:
                continue                            # too short to be worth a continuation
            if ids[:c] == p["ids"][:c] and (best is None or c > best[2]):
                best = ("gpu", slot, c)
        for key, h in self.host_parked.items():
            c = h["consumed"]
            if len(ids) < c or c < self.page:
                continue
            if ids[:c] == h["ids"][:c] and (best is None or c > best[2]):
                best = ("host", key, c)
        return best

    def _prefill_long(self, req, slot):
        # prompt > WAVE_MAX: prefill alone in <=WAVE_MAX-column chunks (continuation; the slot
        # state + paged blocks carry across chunks). seed computed on the final chunk.
        # Legacy blocking form (prefill_waves_per_step == 0): the whole prompt in one go.
        self._prefill_start(req, slot)
        while self.prefills:
            self._prefill_step(1 << 30)

    def _prefill_start(self, req, slot):
        # positions [req.start, n_prompt) still to prefill (start > 0 = continuation of a parked slot)
        self.prefills.append({"req": req, "slot": slot, "pos": req.start,
                              "need": (req.n_prompt + self.page - 1) // self.page})

    def _prefill_reserved(self):
        # blocks the in-flight prefills still have to allocate (kept out of admission control so
        # a wave admitted mid-prefill cannot run their pool dry)
        r = 0
        for pf in self.prefills:
            got = (pf["pos"] + self.page - 1) // self.page
            r += max(0, pf["need"] - got)
        return r

    def _pick_prefill(self):
        """Which in-flight prefill gets the next wave: shortest remaining first (a 3K follow-up
        continuation must not wait behind a 128K cold prompt), with ageing -- every 4th wave goes
        to the oldest request so a long cold prefill cannot be starved by a stream of short ones."""
        self.pf_round += 1
        if self.pf_round % 4 == 0:
            return min(self.prefills, key=lambda pf: pf["req"].t_submit)
        return min(self.prefills, key=lambda pf: pf["req"].n_prompt - pf["pos"])

    def _prefill_fail(self, pf, err):
        if pf in self.prefills:
            self.prefills.remove(pf)
        try:
            self.L.qwn_paged_reset_slot(pf["slot"])
        except Exception:
            pass
        self.free_slots.append(pf["slot"])
        self._finish(pf["req"], err)

    def _prefill_one_wave(self, pf, extra=None):
        """One wave for prefill pf (<= WAVE_MAX - len(extra) columns) plus optional 1-token final
        decode segments `extra` = [(slot, token, pos), ...]. Returns the list of decode seeds (or
        None on failure, after failing pf). Activates pf's request when its last wave completes."""
        req, slot, n = pf["req"], pf["slot"], pf["req"].n_prompt
        nd = len(extra) if extra else 0
        pos = pf["pos"]
        c = min(WAVE_MAX - nd, n - pos)
        final = 1 if pos + c >= n else 0
        toks = list(req.ids[pos:pos + c]); cslot = [slot] * c; cpos = list(range(pos, pos + c))
        sslot = [slot]; soff = [0]; slen = [c]; sfin = [final]
        off = c
        for s, t, p in (extra or []):
            toks.append(t); cslot.append(s); cpos.append(p)
            sslot.append(s); soff.append(off); slen.append(1); sfin.append(1)
            off += 1
        K = 1 + nd
        oseed = (ctypes.c_int * K)()
        rc = self.L.qwn_paged_prefill_batch(_ci(toks), _ci(cslot), _ci(cpos), _ci(sslot),
                                            _ci(soff), _ci(slen), _ci(sfin), K, off, oseed)
        if rc != 0:
            self._prefill_fail(pf, f"prefill wave rc={rc} at pos {pos} ({nd} decode columns)")
            return None
        self.prefill_waves += 1; self.prefilled_tokens += c
        pf["pos"] = pos + c
        if final:
            self.prefills.remove(pf)
            self._activate(req, slot, oseed[0])
        return [oseed[1 + j] for j in range(nd)]

    def _prefill_step(self, max_waves):
        # advance the in-flight prefills by <= max_waves waves (SRPT order); no decode columns
        for _ in range(max_waves):
            if not self.prefills:
                return
            self._prefill_one_wave(self._pick_prefill())

    def _mixed_wave(self):
        """One ragged wave = the next chunk of an in-flight prefill + a 1-token FINAL segment per
        active slot (its decode step). qwn_paged_prefill_batch runs the projections/MLPs over all
        columns with one weight read, the MMA ragged attention per segment, and returns the argmax
        of every final segment -> the decoders get one token per wave (~70 ms @32K, ~110 ms @128K)
        while the prompt prefills at nearly full speed, instead of stalling for the whole prefill
        (17 s @32K, 115 s @128K). Decode-only steps keep qwn_paged_decode_step."""
        pf = self._pick_prefill()
        self._ensure_decode_blocks()
        slots = list(self.active.keys())
        nd = len(slots)
        if pf["req"].n_prompt - pf["pos"] < 1 or nd >= WAVE_MAX:
            # degenerate: too many decoders for one wave -> alternate
            self._prefill_one_wave(pf)
            if self.active:
                self._step()
            return
        extra = [(s, self.active[s].next_tok, self.active[s].pos) for s in slots]
        seeds = self._prefill_one_wave(pf, extra)
        if seeds is None:
            return                                  # the prefill failed; decoders untouched this round
        # decode columns: same bookkeeping as _step()
        self.steps += 1; self.decoded_tokens += nd
        finished = []
        for j, s in enumerate(slots):
            r = self.active[s]
            o = seeds[j]
            r.out.append(o); r.pos += 1; r.next_tok = o; r.tokens.put(o)
            if o in r.eos or len(r.out) >= r.max_new:
                finished.append(s)
        for s in finished:
            self._detach(s)

    def _prefill_wave(self, wave):
        # one ragged wave over several short (or short-continuation) requests; each segment covers
        # positions [req.start, n_prompt) of its slot
        toks = []; cslot = []; cpos = []; soff = []; slen = []; sslot = []; sfin = []
        off = 0
        for req, slot in wave:
            n = req.n_prompt - req.start
            soff.append(off); slen.append(n); sslot.append(slot); sfin.append(1)
            toks += req.ids[req.start:]; cslot += [slot] * n; cpos += list(range(req.start, req.n_prompt))
            off += n
        K = len(wave)
        oseed = (ctypes.c_int * K)()
        rc = self.L.qwn_paged_prefill_batch(_ci(toks), _ci(cslot), _ci(cpos), _ci(sslot),
                                            _ci(soff), _ci(slen), _ci(sfin), K, off, oseed)
        if rc != 0:
            for req, slot in wave:
                try:
                    self.L.qwn_paged_reset_slot(slot)
                except Exception:
                    pass
                self.free_slots.append(slot); self._finish(req, f"prefill wave rc={rc}")
            return
        self.prefill_waves += 1; self.prefilled_tokens += off
        for i, (req, slot) in enumerate(wave):
            self._activate(req, slot, oseed[i])

    def _admit(self):
        # Batch short queued requests (and short prefix-cache continuations) into one <=WAVE_MAX-
        # column ragged wave (1 weight read for all clients); a long (continuation) prefill runs
        # alone, one in flight at a time, advanced between decode steps by _loop.
        self._sweep_host_ttl()
        free_blk = self._free_blocks() - self._prefill_reserved() - self._decode_reserved()
        wave = []; cols = 0
        while self.q:
            req = self.q[0]
            if req.n_prompt < 1:
                self.q.pop(0); self._finish(req, "empty prompt"); continue
            need_total = (req.n_prompt + self.page - 1) // self.page
            if need_total > self.max_blocks_per_seq:
                self.q.pop(0); self._finish(req, "prompt exceeds context"); continue
            # prefix cache: a parked session (GPU slot or host buffer) this prompt extends
            m = self._match_parked(req.ids) if self.prefix_cache else None
            tier = m[0] if m else None
            hit_slot = m[1] if tier == "gpu" else -1
            start = m[2] if m else 0
            n_new = req.n_prompt - start
            is_long = n_new > WAVE_MAX
            n_cold = sum(1 for pf in self.prefills if pf["req"].start == 0)
            if is_long and (wave or (start == 0 and n_cold >= self.max_cold_prefills)):
                break                                   # flush the short wave first / cold cap reached
            if not is_long and cols + n_new > WAVE_MAX:
                break                                   # wave full -> process it, rest waits
            # blocks to add: a GPU hit already holds its history; a host hit needs them all back
            need = need_total - ((start + self.page - 1) // self.page if tier == "gpu" else 0)
            # blocks: evict LRU parked slots (never the one we are about to reuse) until it fits
            while need > free_blk:
                victims = [s for s in self.parked if s != hit_slot]
                if not victims:
                    break
                self._evict(min(victims, key=lambda s: self.parked[s]["t"]))
                free_blk = self._free_blocks() - self._prefill_reserved() - self._decode_reserved()
            if need > free_blk:
                break                                   # pool full even so -> wait (admission control)
            # slot
            if tier == "gpu":
                p = self.parked.pop(hit_slot); slot = hit_slot
                self.cache_hits += 1; self.cache_tokens += start
                pending = p["ids"][-1]
            else:
                if not self.free_slots and not self._evict_lru():
                    break                               # every slot is active -> wait
                slot = self.free_slots.pop()
                ck(self.L.qwn_paged_reset_slot(slot), "reset_slot")
                pending = None
                if tier == "host":
                    h = self.host_parked[m[1]]
                    if self._restore_from_host(m[1], slot):
                        self.cache_hits += 1; self.cache_tokens += start
                        pending = h["ids"][-1]
                    else:
                        start = 0; n_new = req.n_prompt; is_long = n_new > WAVE_MAX
                        if is_long and (wave or n_cold >= self.max_cold_prefills):
                            self.free_slots.append(slot); break
            self.q.pop(0)
            req.start = start
            free_blk -= need
            if self.log_admits:
                print(f"[admit] slot {slot} prompt {req.n_prompt} cached {start} new {n_new} max_new {req.max_new} "
                      f"tier {tier or 'none'} free_blk {free_blk} active {len(self.active)} prefilling {len(self.prefills)}", flush=True)
            if n_new == 0:
                # the prompt IS the consumed history: the parked slot's pending token is the seed
                self._activate(req, slot, pending)
                continue
            if is_long:
                if self.prefill_waves_per_step > 0:
                    self._prefill_start(req, slot)      # advanced by _loop between decode steps
                else:
                    self._prefill_long(req, slot)
                continue
            cols += n_new
            wave.append((req, slot))
        if wave:
            self._prefill_wave(wave)

    def _detach(self, slot):
        req = self.active.pop(slot)
        if self.prefix_cache and req.err is None:
            self._park(slot, req)                   # blocks + DeltaNet state stay for a follow-up
        else:
            ck(self.L.qwn_paged_reset_slot(slot), "reset_slot")
            self.free_slots.append(slot)
        self._finish(req)

    def _decode_reserved(self):
        # Every active decoder (and every prefill about to activate) may start a new page on any step or round
        # (one block each; a round's <= 64 columns never span two new pages). Admission keeps these out of the
        # pool budget so a decode step cannot find the pool empty (paged_step -62 failed EVERY active request).
        return len(self.active) + len(self.prefills)

    def _ensure_decode_blocks(self):
        # Before a step/round: free blocks >= active decoders, evicting GPU-parked sessions (LRU) if needed.
        need = len(self.active)
        while self._free_blocks() < need and self._evict_lru():
            pass
        return self._free_blocks() >= need

    def _step(self):
        slots = list(self.active.keys())
        n = len(slots)
        toks = (ctypes.c_int * n)(*[self.active[s].next_tok for s in slots])
        sid = (ctypes.c_int * n)(*slots)
        pos = (ctypes.c_int * n)(*[self.active[s].pos for s in slots])
        out = (ctypes.c_int * n)()
        ck(self.L.qwn_paged_decode_step(toks, sid, pos, n, out), "paged_step")
        self.steps += 1; self.decoded_tokens += n
        finished = []
        for j, s in enumerate(slots):
            req = self.active[s]
            o = out[j]
            req.out.append(o); req.pos += 1; req.next_tok = o; req.tokens.put(o)
            if o in req.eos or len(req.out) >= req.max_new:
                finished.append(s)
        for s in finished:
            self._detach(s)

    def _spec_k_for(self, n):
        """Draft depth for n active slots: auto = 4 at n <= 2, 3 at n <= 8, 2 above (measured 10-15: deeper
        drafts pay at few clients, the verify's per-column attention at many); capped by the wave width."""
        if self.spec_k == "auto":
            k = 4 if n <= 2 else (3 if n <= 8 else 2)
        else:
            k = int(self.spec_k)
        while k > 0 and n * (k + 1) > SPEC_MAX_T:
            k -= 1
        return k

    def _round(self):
        """One speculative round over all active slots (decode-only phase): draft K per slot, verify
        [next_tok, d_1..d_K] at positions pos..pos+K, commit the longest matching prefix + the bonus
        token. Per accepted token the usual bookkeeping (stream, EOS, max_tokens): a request that stops
        inside the accepted prefix commits exactly the columns that produced its emitted tokens, so the
        slot is left as plain decode would (the stop token itself is never fed). Falls back to a plain
        step when the drafter/verify cannot get pool blocks for the draft positions."""
        slots = list(self.active.keys())
        n = len(slots)
        K = self._spec_k_for(n)
        if K < 1:
            return self._step()
        L = self.L
        reqs = [self.active[s] for s in slots]
        t0 = time.time()
        drafts = (ctypes.c_int * (n * K))()
        rc = L.qwn_paged_mtp_draft(_ci(slots), _ci([r.pos - 1 for r in reqs]), _ci([r.next_tok for r in reqs]), n, K, drafts)
        if rc == -7:                                    # pool exhausted for the draft positions
            self.spec_fallbacks += 1
            return self._step()
        ck(rc, "mtp_draft")
        t1 = time.time()
        toks = []; cslot = []; cpos = []; soff = []
        for i, (s, r) in enumerate(zip(slots, reqs)):
            soff.append(len(toks))
            toks.append(r.next_tok); toks += [int(drafts[i * K + k]) for k in range(K)]
            cslot += [s] * (K + 1); cpos += [r.pos + j for j in range(K + 1)]
        T = len(toks)
        am = (ctypes.c_int * T)()
        rc = L.qwn_paged_verify(_ci(toks), _ci(cslot), _ci(cpos), _ci(slots), _ci(soff), _ci([K + 1] * n), n, T, am)
        if rc == -91:
            self.spec_fallbacks += 1
            return self._step()
        ck(rc, "paged_verify")
        t2 = time.time()
        n_acc = []; finished = []
        for i, (s, r) in enumerate(zip(slots, reqs)):
            off = soff[i]
            d = toks[off: off + K + 1]
            j = 0
            while j < K and am[off + j] == d[j + 1]:
                j += 1
            cand = d[1: j + 1] + [int(am[off + j])]  # j accepted drafts + the bonus token
            e = 0
            for t in cand:
                r.out.append(t); r.tokens.put(t); e += 1
                if t in r.eos or len(r.out) >= r.max_new:
                    finished.append(s)
                    break
            r.pos += e; r.next_tok = r.out[-1]
            n_acc.append(e)
        ck(L.qwn_paged_commit(_ci(list(range(n))), _ci(n_acc), n), "paged_commit")
        t3 = time.time()
        tot = sum(n_acc)
        self.rounds += 1; self.steps += 1; self.decoded_tokens += tot; self.spec_tokens += tot; self.spec_cols += T
        self.spec_draft_s += t1 - t0; self.spec_verify_s += t2 - t1; self.spec_commit_s += t3 - t2
        for s in finished:
            self._detach(s)

    def _loop(self):
        while True:
            with self.cv:
                while self.running and not self.q and not self.active and not self.prefills:
                    self.cv.wait(timeout=0.5)
                if not self.running and not self.active and not self.q and not self.prefills:
                    return
                try:
                    self._admit()
                except Exception as e:
                    print(f"[engine] admit error: {e}", flush=True)
                did_decode = False
                if self.prefills:
                    # mixed: the decoders ride along in the prefill wave (one token per wave);
                    # alternate: a few prefill waves, then one decode step for everyone else.
                    # With no active decoders: a bounded burst of waves, then back to _admit so a
                    # short follow-up arriving mid-prefill gets its SRPT turn within ~0.5 s.
                    try:
                        if self.active and self.mixed_prefill:
                            self._mixed_wave(); did_decode = True
                            for _ in range(self.decode_steps_per_wave):
                                if self.active:
                                    # with --spec a round (3-5 tokens per slot) instead of a step
                                    if self.spec and self.spec_on:
                                        self._round()
                                    else:
                                        self._step()
                        else:
                            self._prefill_step(self.prefill_waves_per_step if self.active else 8)
                    except Exception as e:
                        print(f"[engine] prefill error: {e}", flush=True)
                        for pf in list(self.prefills):
                            self._prefill_fail(pf, f"prefill failed: {e}")
                have = bool(self.active) and not did_decode
                if have and not self._ensure_decode_blocks():
                    print(f"[engine] pool exhausted for {len(self.active)} decoders (free {self._free_blocks()}); "
                          f"stepping anyway", flush=True)
            if have:
                try:
                    if self.spec and self.spec_on:
                        self._round()
                    else:
                        self._step()
                except Exception as e:                  # never let the engine thread die
                    print(f"[engine] step error: {e}", flush=True)
                    with self.cv:
                        for s in list(self.active.keys()):
                            r = self.active.pop(s)
                            try:
                                self.L.qwn_paged_reset_slot(s)
                            except Exception:
                                pass
                            self.free_slots.append(s)
                            self._finish(r, f"step failed: {e}")

    def shutdown(self):
        with self.cv:
            self.running = False
            self.cv.notify_all()
        self.thread.join(timeout=5)
        self.L.qwn_paged_free(); self.L.qwn_free()


# ----------------------------- self test (milestone 3.2) -----------------------------
def selftest(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    eos = [tok.eos_token_id] if tok.eos_token_id is not None else []
    prompts_txt = [
        "The history of cartography is the study of how maps",
        "In quantum mechanics, the wave function describes",
        "def fibonacci(n):\n    # return the nth Fibonacci number\n",
        "The recipe for a classic margherita pizza starts with",
        "Climate scientists measure global temperature using",
        "The French Revolution began in 1789 when",
        "To train a neural network you first need to",
        "The mitochondria is the powerhouse of",
    ][:args.clients]
    ids_list = [tok(t, add_special_tokens=False).input_ids for t in prompts_txt]
    MAXNEW = args.gen

    eng = BatchedEngine(load_lib(args.lib), args.tqf, args.max_slots, args.num_blocks, args.page,
                        prefill_waves_per_step=args.prefill_waves_per_step,
                        mixed_prefill=args.mixed_prefill, prefix_cache=args.prefix_cache,
                        park_ram_bytes=int(args.park_ram_gb * 2**30),
                        decode_steps_per_wave=args.decode_steps_per_wave, cfg_overrides=args.cfg_overrides)
    fb, tb, pg, mb = eng._stats()
    print(f"engine up: pool blocks={tb} page={pg} max_slots={eng.max_slots} free={fb}", flush=True)

    # single-stream reference (greedy) for each prompt
    refs = []
    for ids in ids_list:
        eng.L.qwn_reset_state()
        am = 0
        for t, tk in enumerate(ids):
            am = ck(eng.L.qwn_decode(int(tk), t), "ref")
        out = [am]; p = len(ids) - 1
        for _ in range(MAXNEW - 1):
            am = ck(eng.L.qwn_decode(am, p + 1), "ref"); p += 1
            out.append(am)
            if am in set(eos):
                break
        refs.append(out)

    # staggered submission: client i joins after i*stagger seconds
    reqs = []
    t0 = time.time()

    def submit_staggered():
        for i, ids in enumerate(ids_list):
            time.sleep(args.stagger)
            reqs.append((i, eng.submit(ids, MAXNEW, eos)))

    th = threading.Thread(target=submit_staggered, daemon=True); th.start()
    th.join()
    for _, r in reqs:
        r.done.wait(timeout=120)
    elapsed = time.time() - t0

    print("\n" + "=" * 72, flush=True)
    print(f"  3.2 CONTINUOUS-BATCHING SELFTEST: {len(ids_list)} clients, stagger={args.stagger}s, gen={MAXNEW}", flush=True)
    print("=" * 72, flush=True)
    total_tok = 0
    for i, r in reqs:
        ref = refs[i]
        n = min(len(ref), len(r.out))
        match = 0
        for a, b in zip(r.out[:n], ref[:n]):
            if a == b:
                match += 1
            else:
                break
        total_tok += len(r.out)
        cont = tok.decode(r.out[1:]) if len(r.out) > 1 else ""
        print(f"  client {i}: gen={len(r.out)} leading-match-vs-single={match}/{n}  "
              f"text={cont[:54]!r}", flush=True)
    agg = total_tok / elapsed
    print(f"\n  aggregate: {total_tok} tokens / {elapsed:.2f}s = {agg:.1f} tok/s across "
          f"{len(ids_list)} staggered clients", flush=True)
    print(f"  engine steps={eng.steps} batched-decode-tokens={eng.decoded_tokens}", flush=True)
    eng.shutdown()


# ----------------------------- tool-calling -----------------
TOOL_OPEN, TOOL_CLOSE = "<tool_call>", "</tool_call>"


def _coerce_arg(val, typ):
    """Coerce an XML-ish <parameter> string to its JSON-schema type (Qwen3.6 template
    serializes every value as text). With no declared type, best-effort JSON literal."""
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
    try:
        j = json.loads(s)
        return j if isinstance(j, (int, float, bool, list, dict)) or j is None else val
    except Exception:
        return val


def parse_tool_calls(text, tools=None):
    """Qwen tool-call formats: (a) JSON <tool_call>{"name":..,"arguments":{..}}</tool_call>
    (b) XML-ish <tool_call><function=NAME><parameter=KEY>VALUE</parameter>..</function></tool_call>.
    Returns (clean_text, tool_calls_list_or_None)."""
    import re as _re
    types = {}
    for t in (tools or []):
        fn = t.get("function", t) if isinstance(t, dict) else {}
        props = (((fn.get("parameters") or {}).get("properties")) or {})
        types[fn.get("name", "")] = {k: (v or {}).get("type") for k, v in props.items()}
    calls = []

    def _emit(name, args):
        calls.append({"id": "call_" + uuid.uuid4().hex[:16], "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})

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
                args[pm.group(1)] = _coerce_arg(pm.group(2), ptypes.get(pm.group(1)))
            _emit(name, args)
            return ""
        return m.group(0)

    clean = _re.sub(_re.escape(TOOL_OPEN) + r"(.*?)" + _re.escape(TOOL_CLOSE), _take, text, flags=_re.S)
    return clean.strip(), (calls or None)


# ----------------------------- OpenAI server (milestone 3.3) -----------------------------
def make_handler(eng, tok, args):
    # EOS: eos_token_id + im_end/endoftext so tool-call turns terminate cleanly
    eos = set(int(t) for t in [tok.eos_token_id] if t is not None)
    for _name in ("<|im_end|>", "<|endoftext|>"):
        try:
            _t = tok.convert_tokens_to_ids(_name)
            if _t is not None and _t >= 0:
                eos.add(int(_t))
        except Exception:
            pass
    eos = list(eos)

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _json(self, code, obj, extra_headers=()):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            for k, v in extra_headers:
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self):
            """--api-key: every path but /health needs `Authorization: Bearer <key>`; answers 401 itself."""
            if auth_ok(self.headers, args.api_key, self.path):
                return True
            self._json(401, unauthorized_body(), extra_headers=[("WWW-Authenticate", "Bearer")])
            return False

        def do_GET(self):
            if not self._authorized():
                return
            if self.path.startswith("/v1/models"):
                self._json(200, {"object": "list", "data": [{"id": args.model_name, "object": "model"}]})
            elif self.path.startswith("/health"):
                fb, tb, _, _ = eng._stats()
                self._json(200, {"status": "ok", "free_blocks": fb, "total_blocks": tb,
                                 "active": len(eng.active), "queued": len(eng.q),
                                 "parked": len(eng.parked), "prefilling": len(eng.prefills),
                                 "prefix_cache_hits": eng.cache_hits,
                                 "prefix_cache_tokens": eng.cache_tokens,
                                 "host_parked": len(eng.host_parked),
                                 "host_parked_gib": round(eng.host_bytes / 2**30, 2),
                                 "host_hits": eng.host_hits, "host_saves": eng.host_saves,
                                 "host_save_s": round(eng.host_save_s, 2),
                                 "host_restore_s": round(eng.host_restore_s, 2),
                                 "host_save_gbs": round(eng.host_save_bytes / max(eng.host_save_s, 1e-9) / 1e9, 1),
                                 "host_restore_gbs": round(eng.host_restore_bytes / max(eng.host_restore_s, 1e-9) / 1e9, 1),
                                 "host_evict_lru": eng.host_evict_lru, "host_evict_ttl": eng.host_evict_ttl,
                                 "park_ram_ttl_s": eng.park_ram_ttl,
                                 "park_ram_gb": round(eng.park_ram_bytes / 2**30, 1),
                                 "host_parked_oldest_s": round(time.time() - min(h["t"] for h in eng.host_parked.values()), 1) if eng.host_parked else 0,
                                 "park_ring": str(engine_settings(eng.L)["park_ring"]),
                                 "prefilled_tokens": eng.prefilled_tokens,
                                 "decoded_tokens": eng.decoded_tokens, "steps": eng.steps,
                                 "spec": eng.spec, "spec_on": eng.spec_on, "spec_k": eng.spec_k,
                                 "rounds": eng.rounds, "spec_tokens": eng.spec_tokens,
                                 "tokens_per_round": round(eng.spec_tokens / eng.rounds, 3) if eng.rounds else None,
                                 "spec_verify_cols": eng.spec_cols, "spec_fallbacks": eng.spec_fallbacks,
                                 "spec_ms_per_round": {"draft": round(1000 * eng.spec_draft_s / eng.rounds, 2),
                                                       "verify": round(1000 * eng.spec_verify_s / eng.rounds, 2),
                                                       "commit": round(1000 * eng.spec_commit_s / eng.rounds, 2)}
                                 if eng.rounds else None})
            elif self.path.startswith("/spec"):
                # run-time A/B switch: /spec?on=0 -> plain decode steps (the drafter idles), /spec?on=1 -> rounds
                if not eng.spec:
                    self._json(400, {"error": "server started without --spec"}); return
                q = self.path.split("?", 1)[1] if "?" in self.path else ""
                for kv in q.split("&"):
                    if kv.startswith("on="):
                        eng.spec_on = kv[3:] not in ("0", "false", "off")
                    elif kv.startswith("k="):
                        eng.spec_k = "auto" if kv[2:] == "auto" else str(int(kv[2:]))
                self._json(200, {"spec_on": eng.spec_on, "spec_k": eng.spec_k})
            else:
                self._json(404, {"error": "not found"})

        def _stream_tokens(self, req, cid, max_new, include_usage, clamped_from=None):
            # True token streaming: the engine thread pushes every generated id into req.tokens
            # (None = finished); text is re-decoded from the full id list and only the new suffix
            # is emitted, held back while it ends in an incomplete UTF-8 sequence.
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            base = {"id": cid, "object": "chat.completion.chunk", "model": args.model_name}

            def _sse(o):
                self.wfile.write(("data: " + json.dumps(o) + "\n\n").encode()); self.wfile.flush()

            _sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
            out_ids = []; emitted = 0; text = ""
            deadline = time.time() + args.timeout
            timed_out = False
            while True:
                try:
                    t = req.tokens.get(timeout=1.0)
                except queue.Empty:
                    if time.time() > deadline and not req.started and eng.cancel_queued(req):
                        timed_out = True; break       # never admitted -> give up (queue too long)
                    continue
                if t is None:
                    break
                out_ids.append(t)
                if t in req.eos:
                    continue
                text = tok.decode(out_ids, skip_special_tokens=True)
                if text.endswith("\ufffd"):
                    continue
                if len(text) > emitted:
                    _sse({**base, "choices": [{"index": 0, "delta": {"content": text[emitted:]},
                                               "finish_reason": None}]})
                    emitted = len(text)
            if timed_out:
                _sse({"error": {"message": "generation timeout (never admitted)", "type": "timeout"}})
            elif req.err:
                _sse({"error": {"message": req.err, "type": "engine"}})
            else:
                text = tok.decode(out_ids, skip_special_tokens=True)
                if len(text) > emitted:
                    _sse({**base, "choices": [{"index": 0, "delta": {"content": text[emitted:]},
                                               "finish_reason": None}]})
                gen = len(req.out)
                finish = "length" if gen >= max_new else "stop"
                _sse({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                      "x_qwentin": {"prefix_cached_tokens": req.start, "slot": req.slot,
                                    "server_ttft_s": round(req.t_first - req.t_submit, 3),
                                    **clamp_fields(clamped_from, args.max_tokens_cap)}})
                if include_usage:
                    _sse({**base, "choices": [],
                          "usage": {"prompt_tokens": req.n_prompt, "completion_tokens": gen,
                                    "total_tokens": req.n_prompt + gen,
                                    "prompt_tokens_details": {"cached_tokens": req.start}}})
            self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()

        def do_POST(self):
            if not self._authorized():
                return
            if not self.path.startswith("/v1/chat/completions") and not self.path.startswith("/v1/completions"):
                self._json(404, {"error": "not found"}); return
            try:
                body = read_json_body(self, args.max_body_bytes)      # 411 / 413 / 400 before the body is read
            except BodyError as e:
                self.close_connection = True                         # the unread body is not drained
                self._json(e.status, e.body); return
            max_new, clamped_from = clamp_max_tokens(body.get("max_tokens"), args.max_tokens_cap, 128)
            stream = bool(body.get("stream", False))
            is_chat = self.path.startswith("/v1/chat")
            tools = body.get("tools") or None
            if is_chat:
                msgs = body.get("messages", [])
                # OpenAI carries tool_call arguments as a JSON string; the Qwen
                # template iterates them as a mapping -> parse in place.
                for m in msgs:
                    for tc in (m.get("tool_calls") or []):
                        fn = tc.get("function") or {}
                        if isinstance(fn.get("arguments"), str):
                            try:
                                fn["arguments"] = json.loads(fn["arguments"])
                            except Exception:
                                pass
                think = bool(body.get("enable_thinking", False))
                try:
                    tmpl = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True,
                                                   tokenize=False, enable_thinking=think)
                except TypeError:
                    tmpl = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True,
                                                   tokenize=False)
                ids = tok(tmpl, add_special_tokens=False).input_ids
            else:
                ids = tok(body.get("prompt", ""), add_special_tokens=False).input_ids
            req = eng.submit(list(ids), max_new, eos)
            cid = f"chatcmpl-{int(time.time()*1000)}"
            if eng.log_admits:
                print(f"[req] {self.client_address[0]}:{self.client_address[1]} prompt {len(ids)} max_new {max_new} "
                      f"stream {int(stream)} queued {len(eng.q)} active {len(eng.active)}", flush=True)
            include_usage = bool((body.get("stream_options") or {}).get("include_usage", False))
            if stream and is_chat and not tools:
                self._stream_tokens(req, cid, max_new, include_usage, clamped_from); return
            if not req.done.wait(timeout=args.timeout):
                eng.cancel_queued(req)
                self._json(504, {"error": "generation timeout"}); return
            if getattr(req, "err", None):
                self._json(500, {"error": req.err}); return
            # include the full committed continuation (out[0] is the first generated token)
            text = tok.decode(req.out if len(req.out) else [], skip_special_tokens=True)
            gen = len(req.out)
            tool_calls = None
            if is_chat and tools and TOOL_OPEN in text:
                text, tool_calls = parse_tool_calls(text, tools)
            finish = "tool_calls" if tool_calls else ("length" if gen >= max_new else "stop")
            if stream:
                # framed OpenAI SSE after the fact (tool-call turns: the XML must be parsed whole):
                # role delta, then content OR tool_calls delta, finish, [DONE].
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                base = {"id": cid, "object": "chat.completion.chunk", "model": args.model_name}

                def _sse(o):
                    self.wfile.write(("data: " + json.dumps(o) + "\n\n").encode()); self.wfile.flush()

                _sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
                if tool_calls:
                    for i, tc in enumerate(tool_calls):
                        tc["index"] = i
                    _sse({**base, "choices": [{"index": 0, "delta": {"tool_calls": tool_calls}, "finish_reason": None}]})
                elif text:
                    _sse({**base, "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]})
                _sse({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
                self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
                return
            msg = {"role": "assistant", "content": text or None}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            resp = {
                "id": cid, "object": "chat.completion",
                "model": args.model_name,
                "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                "usage": {"prompt_tokens": req.n_prompt, "completion_tokens": gen,
                          "total_tokens": req.n_prompt + gen,
                          "prompt_tokens_details": {"cached_tokens": req.start}},
                "x_qwentin": {"prefix_cached_tokens": req.start, "slot": req.slot,
                              "server_ttft_s": round(req.t_first - req.t_submit, 3),
                              **clamp_fields(clamped_from, args.max_tokens_cap)},
            }
            self._json(200, resp)

    apply_socket_timeout(H, args.socket_timeout)
    return H


def serve(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    eng = BatchedEngine(load_lib(args.lib), args.tqf, args.max_slots, args.num_blocks, args.page,
                        prefill_waves_per_step=args.prefill_waves_per_step,
                        mixed_prefill=args.mixed_prefill, prefix_cache=args.prefix_cache,
                        park_ram_bytes=int(args.park_ram_gb * 2**30),
                        decode_steps_per_wave=args.decode_steps_per_wave,
                        spec=args.spec, spec_k=args.spec_k, park_ram_ttl=args.park_ram_ttl,
                        cfg_overrides=args.cfg_overrides)
    fb, tb, pg, mb = eng._stats()
    spec_line = (f"speculative decoding ON (MTP drafter, K={args.spec_k})" if eng.spec
                 else "speculative decoding OFF (" + (eng.spec_off_reason or getattr(args, "spec_off_reason", "") or "--no-spec") + ")")
    print(f"batched server: pool blocks={tb} page={pg} max_slots={eng.max_slots} free={fb}; "
          f"{kv_summary(eng.L, tb, pg)}; {spec_line}; {attn_st_summary(eng.L)}; "
          f"decode steps per mixed wave {args.decode_steps_per_wave}", flush=True)
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(eng, tok, args))
    print(f"listening on {args.host}:{args.port}  (model={args.model_name}; api key {'required on /v1/*' if args.api_key else 'none'}, "
          f"max body {args.max_body_bytes} B, max_tokens cap {args.max_tokens_cap}, socket timeout {args.socket_timeout:.0f} s)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        eng.shutdown()


# ----------------------------- steady-state throughput bench -----------------------------
def bench(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    L = load_lib(args.lib)
    init_engine(L, args.tqf, args.cfg_overrides)
    text = "The history of cartography is the study of maps and "
    base_ids = tok(text, add_special_tokens=False).input_ids
    # tile the base text up to bench_p so we can measure decode at long contexts (e.g. 64k)
    reps = (args.bench_p + len(base_ids) - 1) // max(1, len(base_ids))
    ids = (base_ids * reps)[: args.bench_p]
    Ns = [int(x) for x in args.bench_ns.split(",")]
    Nmax = max(Ns)
    ck(L.qwn_paged_init(max(args.max_slots, Nmax), args.num_blocks, args.page), "paged_init")
    # Prefill all Nmax slots DIRECTLY via the paged path (qwn_paged_prefill_batch,
    # <=128-col chunks) -- the real server prefill. No single-stream spec scratch, so the
    # KV pool can be sized to the actual worker count at long contexts (e.g. 64k). A slot
    # prefill that runs the pool dry returns -4 (= that worker count does not fit).
    seed = 0
    for s in range(Nmax):
        ck(L.qwn_paged_reset_slot(s), "reset")
        seed = paged_prefill_slot(L, s, ids, args.page)
    p = len(ids) - 1
    M = args.bench_iters
    fb = ctypes.c_int(); tb = ctypes.c_int(); pgv = ctypes.c_int(); mb = ctypes.c_int()
    L.qwn_paged_stats(ctypes.byref(fb), ctypes.byref(tb), ctypes.byref(pgv), ctypes.byref(mb))
    print("\n" + "=" * 72, flush=True)
    print(f"  3.x STEADY-STATE THROUGHPUT (paged decode) @ ctx={len(ids)} tok, "
          f"{Nmax} slots prefilled, KV blocks used={tb.value - fb.value}/{tb.value}", flush=True)
    print("=" * 72, flush=True)
    print(f"  {'N':>4} {'ms/step':>9} {'agg tok/s':>11} {'per-cli':>9}", flush=True)
    for N in Ns:
        toks = (ctypes.c_int * N)(*([seed] * N))
        sid = (ctypes.c_int * N)(*list(range(N)))
        pos = (ctypes.c_int * N)(*([p + 1] * N))
        out = (ctypes.c_int * N)()
        ck(L.qwn_paged_decode_step(toks, sid, pos, N, out), "warm")
        t0 = time.time()
        for _ in range(M):
            ck(L.qwn_paged_decode_step(toks, sid, pos, N, out), "step")
        t_step = (time.time() - t0) / M
        agg = N / t_step
        print(f"  {N:>4} {1000*t_step:>9.3f} {agg:>11.1f} {1.0/t_step:>9.1f}", flush=True)
    L.qwn_paged_free(); L.qwn_free()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tqf", required=True, help="the .tqf model file")
    ap.add_argument("--model-dir", required=True, help="the HF checkpoint dir (tokenizer and chat template only)")
    ap.add_argument("--lib", default=os.path.join(HERE, "build-qwen", "libforward_qwen.so"))
    ap.add_argument("--model-name", default="qwentin-qwen3.6-27b-batched")
    ap.add_argument("--page", type=int, default=128)
    ap.add_argument("--max-slots", type=int, default=12)
    ap.add_argument("--num-blocks", type=int, default=1024)
    add_hardening_args(ap)        # --host (127.0.0.1) --api-key --max-body-bytes --max-tokens-cap --socket-timeout
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--timeout", type=float, default=300.0,
                    help="admission wait in seconds when all slots are busy (504 / a timeout event after it); the "
                         "client socket's read/write timeout is --socket-timeout")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--bench-ns", default="1,2,4,8,16,32")
    ap.add_argument("--bench-p", type=int, default=48)
    ap.add_argument("--bench-iters", type=int, default=64)
    ap.add_argument("--clients", type=int, default=8)
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--stagger", type=float, default=0.15)
    ap.add_argument("--wide-attn-mma", dest="wide_attn_mma", action="store_true", default=True,
                    help="tensor-core ragged-prefill attention for the paged prefill waves "
                         "(launch_paged_attn_q4_mma_prefill; needs --page >= 128). Sets "
                         "TQ_WIDE_ATTN_MMA=1 unless the env already says otherwise. Without it "
                         "the waves fall back to the scalar DECODE kernel per query column: "
                         "~310 tok/s @32K and super-linear beyond, vs ~1900 tok/s @32K / "
                         "~1500 @64K with it (2026-09-21, RTX 5090). The decode path is unaffected.")
    ap.add_argument("--no-wide-attn-mma", dest="wide_attn_mma", action="store_false")
    ap.add_argument("--prefill-waves-per-step", type=int, default=2,
                    help="long-prompt prefill is advanced by this many <=128-column waves between "
                         "decode steps of the other slots (interleaving; one long prefill in flight "
                         "at a time). 0 = legacy: the whole prompt in one blocking pass. "
                         "One wave ~= 70 ms @32K, ~110 ms @128K on a 5090.")
    ap.add_argument("--mixed-prefill", dest="mixed_prefill", action="store_true", default=True,
                    help="while a long prefill is in flight, the active slots' decode steps ride in "
                         "the same ragged wave as 1-token final segments (one token per wave for "
                         "every decoder, prefill at ~full speed). --no-mixed-prefill alternates "
                         "whole prefill waves and decode steps instead.")
    ap.add_argument("--no-mixed-prefill", dest="mixed_prefill", action="store_false")
    ap.add_argument("--prefix-cache", dest="prefix_cache", action="store_true", default=True,
                    help="slot-level prefix cache: finished requests leave their slot parked (KV "
                         "blocks + DeltaNet state + token history); a follow-up whose prompt extends "
                         "the consumed history continues from there (only the new tokens are "
                         "prefilled). Parked slots are evicted LRU when slots/blocks are needed.")
    ap.add_argument("--no-prefix-cache", dest="prefix_cache", action="store_false")
    ap.add_argument("--decode-steps-per-wave", type=int, default=1,
                    help="extra pure decode steps after each mixed prefill wave (decoders get 1+k tokens "
                         "per wave; prefill slows by ~k x 25%%). 0 = one token per wave. With --spec each extra "
                         "step is a speculative round (3-5 tokens per slot for ~20-30 ms). DEFAULT 1 since 10-17 "
                         "(owner's decision (a): T5 mix 13 / 8 / 11 -> 43 / 28 / 43 tok/s per client under prefill load, "
                         "wall -6 %%; the cold 96K-class TTFT +16 s).")
    ap.add_argument("--park-ram-gb", type=float, default=64.0,
                    help="host-RAM budget for parked sessions evicted from the GPU (KV blocks + DeltaNet "
                         "state, ~26 KiB/token + 155 MiB each; LRU). 0 disables the host tier. Needs the "
                         "qwn_paged_slot_save/restore ABI in the engine library.")
    ap.add_argument("--park-ram-ttl", type=float, default=0.0,
                    help="drop host-parked sessions older than this many seconds (checked at every admission; "
                         "0 = LRU only). /health reports host_evict_lru / host_evict_ttl and the oldest entry's age.")
    ap.add_argument("--embed-host", dest="embed_host", action="store_true", default=True,
                    help="keep the bf16 token-embedding table in pinned host memory (TQ_EMBED_HOST=1): no device "
                         "copy (0.95 GB as E2M3, 1.27 GB as E4M3) and no quantization error; each looked-up row "
                         "(10 KB) is read over PCIe. Overrides TQ_EMBED_FP8.")
    ap.add_argument("--no-embed-host", dest="embed_host", action="store_false")
    ap.add_argument("--mtp", dest="mtp", action="store_true", default=False,
                    help="load the tqf's EXL3 MTP head (0.21 GB; the engine never loads a bf16 head). Only --spec uses "
                         "it, so by default TQ_NO_MTP=1 skips it.")
    ap.add_argument("--kv-v4", dest="kv_v4", action="store_true", default=True,
                    help="store V like K in the paged pool (TQ_KV_V4=1, default): FWHT-256 rotated int4, asymmetric "
                         "per 32 channels, 20.0 instead of 26.2 KiB per token (+31%% pooled tokens in the same pool "
                         "memory); required by --kv-t4 (the default since 10-16). The 10-13 served stack was EXL3 "
                         "opt-6.25 + int4 V (--no-kv-t4: 3300 blocks with 12 slots); opt-5.75 + FP4 V needs --no-kv-v4 "
                         "(E4M3 V) to stay at FP8 quality at long context -- with --kv-t4 opt-5.75 is FP8 quality.")
    ap.add_argument("--no-kv-v4", dest="kv_v4", action="store_false")
    ap.add_argument("--kv-t4", dest="kv_t4", action="store_true", default=True,
                    help="B2 (session 10): trellis-coded K and V for every committed token (TQ_KV_T4=1; 12-bit-state "
                         "bitshift trellis, 4 bits per value, fp16 RMS scale per vector, trained codebook "
                         "--kv-t4-codebook): 16.5 instead of 20.0 KiB per pooled token. Prefill and verify waves encode "
                         "their columns per layer, decode steps after the step (~3 ms per 128-column wave, ~0.2 ms per "
                         "step); the in-flight columns are attended exact from an fp16 staging ring (needs --kv-v4). "
                         "DEFAULT since 10-16 (owner, Decisions 16a: the served stack is EXL3 opt-5.75 + trellis-4, "
                         "FP8 quality end to end; 12 slots: --num-blocks 4600 = 589K tokens, 4100 = 525K with --spec); "
                         "--no-kv-t4 reverts to FP4 K+V (size with analysis/pool_size.py).")
    ap.add_argument("--no-kv-t4", dest="kv_t4", action="store_false")
    ap.add_argument("--kv-t4-codebook", default=None,
                    help="a file of 4096 raw fp32 codebook entries for --kv-t4 (TQ_KV_T4_CODEBOOK); default: the "
                         "engine's built-in Lloyd-trained 12-bit-state codebook lloyd_kv12a "
                         "(= tools/codebooks/lloyd_kv12a.f32)")
    ap.add_argument("--pdl", dest="pdl", action="store_true", default=True,
                    help="programmatic dependent launch in the decode step (TQ_PDL=1, default): each EXL3 GEMM and "
                         "its bfrag launch while the predecessor drains and wait on the device; bit-identical outputs, "
                         "-0.55..-0.61 ms per step at N = 1-8 @32K (5090 at 550 W, opt-6.25 + int4 V).")
    ap.add_argument("--no-pdl", dest="pdl", action="store_false")
    ap.add_argument("--spec", dest="spec", action="store_true", default=True,
                    help="speculative decoding with the paged MTP drafter (task 10): in decode-only phases each "
                         "engine iteration drafts K tokens per slot, verifies them in one wave and commits the "
                         "longest matching prefix + 1 (bitwise = plain decode at the same client count). Implies --mtp; "
                         "needs a tqf with EXL3 MTP projections (convert_qwen_tqf.py --mtp-exl3, e.g. ...-opt5.75-mtp4.tqf) "
                         "and ~11%% of the trellis pool for the drafter's FP4 KV rows (12 slots: --num-blocks 4100 instead "
                         "of 4600). DEFAULT since 10-16 (owner, Decisions 16b); a tqf without the EXL3 MTP section runs "
                         "with spec OFF after a warning; --no-spec turns it off. Measured through the server (5090, 550 W, "
                         "trellis-4 @32K): 2.7x per stream at N=1, 1.9x at N=4, 1.7x at N=8.")
    ap.add_argument("--no-spec", dest="spec", action="store_false")
    ap.add_argument("--spec-k", default="auto",
                    help="draft depth: auto = 4 at <= 2 active slots, 3 at <= 8, 2 above; or a fixed number (1..16).")
    ap.add_argument("--dstate-bf16", dest="dstate", action="store_const", const=1, default=4,
                    help="store the per-slot GatedDeltaNet recurrent state as bf16 instead of fp32 (TQ_DSTATE_BF16=1; "
                         "compute stays fp32): 75.5 instead of 151 MiB per slot, ~0.9 GB freed at 12 slots (~+440 "
                         "trellis blocks); parking blobs shrink the same way (versioned header). bf16 FAILED its "
                         "quality gate; the DEFAULT since 10-17 is fp16sr (owner's decision (c)); --dstate fp32 = the "
                         "10-16 served line bitwise. Quality gate: notes/session10-dstate.md.")
    ap.add_argument("--dstate-fp16", dest="dstate", action="store_const", const=2,
                    help="the same with fp16 storage (TQ_DSTATE_BF16=2; 11-bit significand, clamped to +-65504).")
    ap.add_argument("--gdn-prefill", dest="gdn_prefill", action="store_true", default=True,
                    help="session 11: the paged prefill waves run the gated-DeltaNet layers through the hoisted-prep + "
                         "register-resident chunk kernels (TQ_GDN_PREFILL=1; bit-identical to the 10-16 kernels, "
                         "DeltaNet 14.9 -> 2.8 ms per 128-column wave, 32K TTFT 16.0 -> 13.2 s; notes/session11-gdn.md). "
                         "DEFAULT ON since 10-17; --no-gdn-prefill = the 10-16 kernels.")
    ap.add_argument("--no-gdn-prefill", dest="gdn_prefill", action="store_false")
    ap.add_argument("--dstate", dest="dstate_name", default=None, choices=["fp32", "bf16", "fp16", "bf16+8", "fp16sr"],
                    help="the recurrent-state storage format by name (TQ_DSTATE_BF16 = 0 / 1 / 2 / 3 / 4): fp16sr (DEFAULT "
                         "since 10-17) = fp16 with deterministic stochastic rounding (e2e-neutral in prefill AND pure decode at "
                         "4K-64K, verify == decode bitwise, 75.5 MiB per slot -> --num-blocks 4500 fits the 4100 footprint); "
                         "bf16+8 = 24-bit (108 MiB per slot); fp32 = the 10-16 line.")
    args = ap.parse_args()
    if args.dstate_name is not None:
        args.dstate = ["fp32", "bf16", "fp16", "bf16+8", "fp16sr"].index(args.dstate_name)
    args.spec_off_reason = ""
    if args.spec and (args.bench or args.selftest):
        args.spec = False                  # neither mode runs speculative rounds (and --mtp would only load the head)
    if args.spec:
        has_mtp4 = tqf_has_exl3_mtp(args.tqf)
        if has_mtp4 is False:
            # Decided before qwn_init: TQ_NO_MTP below follows args.mtp.
            args.spec_off_reason = (f"{os.path.basename(args.tqf)} has no EXL3 MTP section (header flag 0x1000); "
                                    f"speculative decoding needs the --mtp-exl3 tqf (...-mtp4.tqf)")
            print(f"[spec] WARNING: speculative decoding OFF: {args.spec_off_reason} (--no-spec silences this)", flush=True)
            args.spec = False
    if args.spec:
        args.mtp = True
        if args.spec_k != "auto":
            k = int(args.spec_k)
            if not 1 <= k <= 16:
                raise SystemExit("--spec-k must be auto or 1..16")
    args.cfg_overrides = config_overrides(args)   # handed to qwn_init_with; explicit env overrides the flags
    if args.bench:
        bench(args)
    elif args.selftest:
        selftest(args)
    else:
        serve(args)


if __name__ == "__main__":
    main()
