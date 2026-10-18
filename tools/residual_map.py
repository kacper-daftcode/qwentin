#!/usr/bin/env python3
"""Residual-stream steering hook of the single-stream engine: the maps file, the server
flags (with the deprecated 2026-08 aliases), the decay-schedule parser, the numpy reference.

The engine (src/forward_qwen.cu) can apply one affine map per hooked decoder layer to the
fp32 residual stream right after that layer, in every forward path (wide prefill, per-node
spec verify, dense decode):

    x += scale * (shift + P (A - I)^T P^T (x - mu))

P is H x 2 (the kernel is rank-2: the map acts in a 2-dimensional subspace of the residual
stream), A - I is 2 x 2, mu and shift are H-vectors, H the hidden size (5120 on the 27B).
Any steering method that produces such a map -- a mean-difference direction with a
projection, a fitted affine transport map, a learned 2-d steering subspace -- can be served
this way. The engine entry points are qwn_rmap_hook_add, qwn_rmap_hook_count, qwn_rmap_set_scale and
qwn_rmap_apply_debug (the 2026-08 qwn_ot_* names stay as aliases for one release).

Maps file (JSON, `--residual-map PATH`):

    {
      "format": "qwentin-residual-map-v1",     # optional
      "hidden_size": 5120,                     # optional; checked against the engine when present
      "layers": [37, 38],                      # hooked layers (default: every key of "maps")
      "scale": 0.47,                           # required here or as --residual-map-scale
      "maps": {
        "37": {"P": [[p0, p1], ... H rows ...],
               "A_minus_I": [[a00, a01], [a10, a11]],
               "mu": [... H ...],
               "shift": [... H ...]},
        "38": {...}
      }
    }

A file written by the 2026-08 tooling (top-level "ot_maps", per-layer keys "A_k_minus_I",
"mu_H", "mean_shift", no "layers"/"scale") is read as well: pass --residual-map-layers and
--residual-map-scale with it, or wrap it once:

    python3 -c 'import json, sys; m = json.load(open(sys.argv[1])); json.dump(
        {"format": "qwentin-residual-map-v1", "layers": [37, 38], "scale": 0.47, "maps": m["ot_maps"]},
        open(sys.argv[2], "w"))' old.json maps.json

No numpy at import time: the loader returns plain lists so the file format and the flag
aliases are unit-testable on CPU (tests/test_residual_map.py); arm_residual_map() and
apply_reference() import numpy when called.
"""
from __future__ import annotations
import argparse, json, os, sys

ENV_MAP = "TQ_RESIDUAL_MAP"
ENV_DECAY = "TQ_RESIDUAL_MAP_DECAY"
LEGACY_ENV_MAP = "TQ_OT_MAPS"            # 2026-08: the path, armed only together with --bark-all-day
LEGACY_ENV_DECAY = "TQ_BARK_DECAY"       # 2026-08: the default schedule
FORMAT = "qwentin-residual-map-v1"

_KEYS = (("P", "P"), ("A_minus_I", "A_k_minus_I"), ("mu", "mu_H"), ("shift", "mean_shift"))


def add_residual_map_args(ap, env=None):
    """The server's flags. The 2026-08 spelling (--bark-all-day, --ot-maps, --ot-layers, --ot-scale,
    --bark-decay-default, --bark-sched-api) stays accepted for one release, hidden from --help;
    resolve_residual_map_args() maps it onto the new flags and returns the deprecation notices."""
    env = os.environ if env is None else env
    g = ap.add_argument_group("residual-stream steering hook (affine map after layer L; off unless --residual-map)")
    g.add_argument("--residual-map", default=env.get(ENV_MAP) or None, metavar="PATH",
                   help="arm the hook: a JSON of affine maps, one per hooked decoder layer "
                        "(x += scale * (shift + P (A - I)^T P^T (x - mu)), applied on-device in every "
                        "forward path). Format: tools/residual_map.py docstring. Env: TQ_RESIDUAL_MAP")
    g.add_argument("--residual-map-layers", default=None, metavar="L1,L2",
                   help="hooked decoder layers; overrides the file's \"layers\" (default: every layer in the file)")
    g.add_argument("--residual-map-scale", type=float, default=None, metavar="S",
                   help="scale of the map; overrides the file's \"scale\" (required when the file has none)")
    g.add_argument("--residual-map-decay", default=env.get(ENV_DECAY) or None,
                   metavar="END,LEN[,reset[,shape[,end_answer[,len_answer]]]]",
                   help="default schedule for every non-thinking request: the scale decays from the armed value "
                        "to END over LEN generated tokens (shape linear|pow|exp|invlog), `reset` restarts the decay "
                        "when </think> commits, END_ANSWER/LEN_ANSWER give the answer phase its own floor and length. "
                        "Empty = fixed scale. Env: TQ_RESIDUAL_MAP_DECAY")
    g.add_argument("--residual-map-sched-api", action="store_true", default=False,
                   help="accept per-request schedules in the residual_map_decay request field (keys: start, end, "
                        "len, reset_think, shape, p, end_answer, len_answer); requests without it run the fixed "
                        "scale or --residual-map-decay. Test knob")
    # deprecated aliases (one release), hidden
    ap.add_argument("--bark-all-day", action="store_true", default=False, help=argparse.SUPPRESS)
    ap.add_argument("--ot-maps", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--ot-layers", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--ot-scale", type=float, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--bark-decay-default", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--bark-sched-api", action="store_true", default=False, help=argparse.SUPPRESS)
    return ap


def resolve_residual_map_args(args, env=None):
    """Map the deprecated flags onto the new ones, in place; an explicitly given new flag wins over
    its alias. Returns the deprecation notices (one line per alias used, empty when none was).
    Raises ValueError when --bark-all-day is given without any maps path (the 2026-08 server exited
    there too)."""
    env = os.environ if env is None else env
    notes = []

    def note(old, new):
        notes.append(f"[serve] deprecated: {old} -> {new} (the alias goes away in the next release)")

    if args.bark_all_day:
        legacy = args.ot_maps or env.get(LEGACY_ENV_MAP) or None
        if args.residual_map is None:
            args.residual_map = legacy
        if args.residual_map is None:
            raise ValueError("--bark-all-day (deprecated) needs a maps path: use --residual-map PATH")
        note("--bark-all-day --ot-maps PATH", "--residual-map PATH")
    elif args.ot_maps:
        # without --bark-all-day the 2026-08 server armed nothing; keep that, say so
        note("--ot-maps (ignored without --bark-all-day, as before)", "--residual-map PATH")
    if args.ot_layers is not None:
        if args.residual_map_layers is None:
            args.residual_map_layers = args.ot_layers
        note("--ot-layers", "--residual-map-layers")
    if args.ot_scale is not None:
        if args.residual_map_scale is None:
            args.residual_map_scale = args.ot_scale
        note("--ot-scale", "--residual-map-scale")
    legacy_decay = args.bark_decay_default or env.get(LEGACY_ENV_DECAY) or None
    if legacy_decay:
        if args.residual_map_decay is None:
            args.residual_map_decay = legacy_decay
        note("--bark-decay-default / TQ_BARK_DECAY", "--residual-map-decay / TQ_RESIDUAL_MAP_DECAY")
    if args.bark_sched_api:
        args.residual_map_sched_api = True
        note("--bark-sched-api (request field bark_decay)", "--residual-map-sched-api (request field residual_map_decay)")
    return notes


def parse_layers(spec):
    """'37,38' -> [37, 38]; a list/tuple of ints passes through."""
    if isinstance(spec, (list, tuple)):
        return [int(x) for x in spec]
    return [int(x) for x in str(spec).split(",") if x.strip()]


def parse_decay(spec):
    """--residual-map-decay END,LEN[,reset[,shape[,end_answer[,len_answer]]]] -> the schedule dict the
    server runs (same keys as a per-request residual_map_decay object); None for an empty spec."""
    if not spec:
        return None
    p = [x.strip() for x in str(spec).split(",")]
    return {"end": float(p[0]),
            "len": int(p[1]) if len(p) > 1 and p[1] else 192,
            "reset_think": len(p) > 2 and p[2] in ("reset", "1", "true"),
            "shape": p[3] if len(p) > 3 and p[3] else "linear",
            "end_answer": float(p[4]) if len(p) > 4 and p[4] else None,
            "len_answer": int(p[5]) if len(p) > 5 and p[5] else 400}


_DEPRECATED_SEEN = set()


def request_decay(req):
    """The per-request schedule object: `residual_map_decay`, or the deprecated `bark_decay` (one
    notice per process). None unless it is a JSON object."""
    if "residual_map_decay" in req:
        v = req.get("residual_map_decay")
    elif "bark_decay" in req:
        v = req.get("bark_decay")
        if "bark_decay" not in _DEPRECATED_SEEN:
            _DEPRECATED_SEEN.add("bark_decay")
            print("[serve] deprecated: request field bark_decay -> residual_map_decay", file=sys.stderr, flush=True)
    else:
        v = None
    return v if isinstance(v, dict) else None


def load_residual_map(path, layers=None, scale=None, hidden_size=None):
    """Read a maps file and resolve the hooked layers and the scale (explicit arguments win over the
    file's fields). Returns {"path", "format", "hidden_size", "layers": [int], "scale": float,
    "maps": {layer: {"P": H x 2, "A_minus_I": 2 x 2, "mu": [H], "shift": [H]}}} as plain lists.
    Raises ValueError with a one-line reason on a malformed file."""
    with open(path) as f:
        doc = json.load(f)
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: expected a JSON object")
    raw = doc.get("maps", doc.get("ot_maps"))
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f'{path}: no "maps" object (nor the legacy "ot_maps")')
    have = sorted(int(k) for k in raw)
    if layers is not None and layers != "":
        want = parse_layers(layers)
    elif doc.get("layers") is not None:
        want = parse_layers(doc["layers"])
    else:
        want = have
    if not want:
        raise ValueError(f"{path}: no hooked layers")
    if scale is None:
        scale = doc.get("scale")
    if scale is None:
        raise ValueError(f'{path}: no "scale" field -- pass --residual-map-scale S (or add "scale" to the file)')
    scale = float(scale)
    H = doc.get("hidden_size")
    if H is not None:
        H = int(H)
        if hidden_size is not None and int(hidden_size) != H:
            raise ValueError(f"{path}: hidden_size {H} but the engine has {hidden_size}")
    elif hidden_size is not None:
        H = int(hidden_size)
    maps = {}
    for ln in want:
        m = raw.get(str(ln))
        if not isinstance(m, dict):
            raise ValueError(f"{path}: layer {ln} is not in the file (it has {have})")
        e = {}
        for key, legacy in _KEYS:
            v = m.get(key, m.get(legacy))
            if v is None:
                raise ValueError(f'{path}: layer {ln}: missing "{key}"')
            e[key] = v
        P = e["P"]
        if H is None:
            H = len(P)
        if len(P) != H or any(len(r) != 2 for r in P):
            raise ValueError(f"{path}: layer {ln}: P must be hidden_size x 2 = {H} x 2 (rank-2 hook), got "
                             f"{len(P)} x {len(P[0]) if P and hasattr(P[0], '__len__') else '?'}")
        A = e["A_minus_I"]
        flat = [x for r in A for x in r] if A and isinstance(A[0], (list, tuple)) else list(A)
        if len(flat) != 4:
            raise ValueError(f"{path}: layer {ln}: A_minus_I must be 2 x 2")
        e["A_minus_I"] = [[float(flat[0]), float(flat[1])], [float(flat[2]), float(flat[3])]]
        for key in ("mu", "shift"):
            if len(e[key]) != H:
                raise ValueError(f"{path}: layer {ln}: {key} must have hidden_size = {H} entries, got {len(e[key])}")
        maps[ln] = e
    return {"path": path, "format": doc.get("format"), "hidden_size": H, "layers": want, "scale": scale, "maps": maps}


def arm_residual_map(LIB, rm):
    """Install rm (from load_residual_map) through the engine's hook ABI; the scale armed is rm["scale"]
    (retunable per layer with qwn_rmap_set_scale). Returns the armed layers; raises RuntimeError on an
    engine refusal (rc < 0: not initialized, layer out of range, slot table full, layer already armed,
    device allocation)."""
    import ctypes
    import numpy as np
    LIB.qwn_rmap_hook_add.restype = ctypes.c_int
    LIB.qwn_rmap_hook_add.argtypes = [ctypes.c_int, ctypes.c_float] + [ctypes.POINTER(ctypes.c_float)] * 4

    def f32p(a):
        return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    armed = []
    for ln in rm["layers"]:
        m = rm["maps"][ln]
        P = np.ascontiguousarray(np.asarray(m["P"], dtype=np.float32))                 # H x 2 row-major
        A = np.ascontiguousarray(np.asarray(m["A_minus_I"], dtype=np.float32)).reshape(4)
        mu = np.ascontiguousarray(np.asarray(m["mu"], dtype=np.float32))
        sh = np.ascontiguousarray(np.asarray(m["shift"], dtype=np.float32))
        rc = LIB.qwn_rmap_hook_add(int(ln), ctypes.c_float(rm["scale"]), f32p(P), f32p(A), f32p(mu), f32p(sh))
        if rc != 0:
            raise RuntimeError(f"residual map: hook_add layer {ln} failed: {rc}")
        armed.append(int(ln))
    return armed


def apply_reference(x, m, scale):
    """fp64 numpy reference of the hook on rows x [n x H] for one layer's map m (as loaded):
    x + scale * (shift + ((x - mu) P (A - I)^T) P^T)."""
    import numpy as np
    P = np.asarray(m["P"], dtype=np.float64)
    A = np.asarray(m["A_minus_I"], dtype=np.float64)
    mu = np.asarray(m["mu"], dtype=np.float64)
    sh = np.asarray(m["shift"], dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    return x + float(scale) * (sh[None, :] + ((x - mu[None, :]) @ P @ A.T) @ P.T)
