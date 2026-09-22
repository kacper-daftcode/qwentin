#!/usr/bin/env python3
"""Roadmap E milestone 3.2 + 3.3: continuous-batching scheduler + multi-client OpenAI server.

A single engine thread owns the CUDA engine (paged Q4 KV pool). HTTP handler threads only
tokenize, submit a Request, and wait on its completion event. The engine thread runs a
continuous-batching loop:
  - ADMIT queued requests into free slots (prefill the prompt single-stream, snapshot into the
    slot's paged blocks via qwn_paged_load_client = the validated bring-up path), subject to
    admission control (free slot + enough free pool blocks).
  - DECODE one paged step over ALL active slots (qwn_paged_decode_step), push each slot's new
    token to its request, advance positions, and DETACH finished requests (EOS / max_tokens),
    returning their blocks to the pool.

Decodes are batched (the throughput win). Short prompts are batched into one ragged prefill wave;
a long prompt is prefilled in <=128-column waves that carry the active slots' decode steps as
1-token final segments (--mixed-prefill, default), so decoders keep producing one token per wave
while the long prompt prefills at ~full speed (--no-mixed-prefill alternates waves and decode
steps; --prefill-waves-per-step 0 restores the blocking whole-prompt prefill). Default-off
relative to the single-stream serve_openai.py; this is a SEPARATE server.

Modes:
  --selftest     : staggered join/leave correctness vs single-stream + aggregate tok/s (no HTTP)
  (default)      : run the OpenAI server on --port

Run (GPU7, prod Q4):
  CUDA_VISIBLE_DEVICES=7 TQ_CTX=8192 TQ_KV_Q4=1 python3 -u tools/serve_batched.py --selftest
  CUDA_VISIBLE_DEVICES=7 TQ_CTX=131072 TQ_KV_Q4=1 python3 -u tools/serve_batched.py --port 8100
"""
from __future__ import annotations
import argparse, ctypes, json, os, threading, time, queue, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WAVE_MAX = 128   # prefill wave column cap = the fast activation quantizer's smem limit


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
    L.qwn_prefill_chunk.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    L.qwn_prefill_chunk.restype = ctypes.c_int
    L.qwn_free.restype = ctypes.c_int
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
    return L


def fast_prefill(L, ids, chunk=16):
    """Single-stream prefill in <=16-token spec chunks (qwn_prefill_chunk) so long
    contexts (e.g. 64k) prefill in seconds instead of token-by-token minutes.
    Returns (final_argmax, last_pos). Timing-grade: the committed KV is what
    qwn_paged_load_client snapshots into each paged slot."""
    L.qwn_reset_state()
    n = len(ids); pos = 0; am = 0
    out = (ctypes.c_int * 1)()
    while pos < n:
        c = min(chunk, n - pos)
        buf = (ctypes.c_int * c)(*[int(x) for x in ids[pos:pos + c]])
        rc = L.qwn_prefill_chunk(buf, pos - 1, c, out)
        if rc != 0:
            raise RuntimeError(f"qwn_prefill_chunk rc={rc} at pos {pos}")
        am = out[0]; pos += c
    return am, n - 1


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


class BatchedEngine:
    def __init__(self, lib, tqf, max_slots, num_blocks, page, prefill_waves_per_step=2,
                 mixed_prefill=True, prefix_cache=True, park_ram_bytes=0, decode_steps_per_wave=0):
        self.L = lib
        ck(self.L.qwn_init(tqf.encode()), "init")
        self.page = page; self.max_slots = max_slots
        ck(self.L.qwn_paged_init(max_slots, num_blocks, page), "paged_init")
        fb, tb, pg, mb = self._stats()
        self.num_blocks = tb; self.max_blocks_per_seq = mb
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
            self.host_bytes -= self.host_parked.pop(k)["nbytes"]
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
        self.host_bytes += nbytes; self.host_saves += 1; self.host_save_s += dt
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
        self.host_hits += 1; self.host_restore_s += dt
        print(f"[park] host -> slot {slot}: {h['consumed']} tokens, {h['nbytes'] / 2**20:.0f} MiB in {dt:.2f}s "
              f"({h['nbytes'] / 2**30 / max(dt, 1e-6):.1f} GB/s)", flush=True)
        return True

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
        free_blk = self._free_blocks() - self._prefill_reserved()
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
                free_blk = self._free_blocks() - self._prefill_reserved()
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
                                    self._step()
                        else:
                            self._prefill_step(self.prefill_waves_per_step if self.active else 8)
                    except Exception as e:
                        print(f"[engine] prefill error: {e}", flush=True)
                        for pf in list(self.prefills):
                            self._prefill_fail(pf, f"prefill failed: {e}")
                have = bool(self.active) and not did_decode
            if have:
                try:
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
                        decode_steps_per_wave=args.decode_steps_per_wave)
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


# ----------------------------- tool-calling (ported from serve_openai.py) -----------------
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

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
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
                                 "prefilled_tokens": eng.prefilled_tokens,
                                 "decoded_tokens": eng.decoded_tokens, "steps": eng.steps})
            else:
                self._json(404, {"error": "not found"})

        def _stream_tokens(self, req, cid, max_new, include_usage):
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
                                    "server_ttft_s": round(req.t_first - req.t_submit, 3)}})
                if include_usage:
                    _sse({**base, "choices": [],
                          "usage": {"prompt_tokens": req.n_prompt, "completion_tokens": gen,
                                    "total_tokens": req.n_prompt + gen,
                                    "prompt_tokens_details": {"cached_tokens": req.start}}})
            self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()

        def do_POST(self):
            if not self.path.startswith("/v1/chat/completions") and not self.path.startswith("/v1/completions"):
                self._json(404, {"error": "not found"}); return
            n = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(n) or b"{}")
            max_new = int(body.get("max_tokens", 128))
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
            include_usage = bool((body.get("stream_options") or {}).get("include_usage", False))
            if stream and is_chat and not tools:
                self._stream_tokens(req, cid, max_new, include_usage); return
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
                              "server_ttft_s": round(req.t_first - req.t_submit, 3)},
            }
            self._json(200, resp)

    return H


def serve(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    eng = BatchedEngine(load_lib(args.lib), args.tqf, args.max_slots, args.num_blocks, args.page,
                        prefill_waves_per_step=args.prefill_waves_per_step,
                        mixed_prefill=args.mixed_prefill, prefix_cache=args.prefix_cache,
                        park_ram_bytes=int(args.park_ram_gb * 2**30),
                        decode_steps_per_wave=args.decode_steps_per_wave)
    fb, tb, pg, mb = eng._stats()
    print(f"batched server: pool blocks={tb} page={pg} max_slots={eng.max_slots} free={fb}", flush=True)
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(eng, tok, args))
    print(f"listening on {args.host}:{args.port}  (model={args.model_name})", flush=True)
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
    ck(L.qwn_init(args.tqf.encode()), "init")
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
    ap.add_argument("--tqf", default="/workspace/models/Qwen3.6-27B/qwen3_6-27b-e2m3-mtp.tqf")
    ap.add_argument("--model-dir", default="/workspace/models/Qwen3.6-27B")
    ap.add_argument("--lib", default=os.path.join(HERE, "build-qwen", "libforward_qwen.so"))
    ap.add_argument("--model-name", default="qwentin-qwen3.6-27b-batched")
    ap.add_argument("--page", type=int, default=128)
    ap.add_argument("--max-slots", type=int, default=12)
    ap.add_argument("--num-blocks", type=int, default=1024)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--timeout", type=float, default=300.0)
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
    ap.add_argument("--decode-steps-per-wave", type=int, default=0,
                    help="extra pure decode steps after each mixed prefill wave (decoders get 1+k tokens "
                         "per wave; prefill slows by ~k x 25%%). 0 = one token per wave.")
    ap.add_argument("--park-ram-gb", type=float, default=64.0,
                    help="host-RAM budget for parked sessions evicted from the GPU (KV blocks + DeltaNet "
                         "state, ~26 KiB/token + 155 MiB each; LRU). 0 disables the host tier. Needs the "
                         "qwn_paged_slot_save/restore ABI in the engine library.")
    args = ap.parse_args()
    # The engine lib reads this lazily on the first prefill wave; explicit env overrides the flag.
    os.environ.setdefault("TQ_WIDE_ATTN_MMA", "1" if (args.wide_attn_mma and args.page >= 128) else "0")
    if args.bench:
        bench(args)
    elif args.selftest:
        selftest(args)
    else:
        serve(args)


if __name__ == "__main__":
    main()
