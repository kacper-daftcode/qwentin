"""src/qwentin.h and tools/qwentin_abi.py match the engine source, and every hand-written ctypes declaration in tools/
(`L.qwn_x.argtypes = [...]`, `.restype = ...`) matches the header: same arity, same scalar types, pointers as pointers."""

from __future__ import annotations

import ast
import ctypes
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import gen_qwentin_h as g  # noqa: E402
import qwentin_abi  # noqa: E402

_NS = {"ctypes": ctypes, **{k: getattr(ctypes, k) for k in dir(ctypes) if k.startswith("c_") or k == "POINTER"}}


def _is_ptr(t) -> bool:
    return t in (ctypes.c_void_p, ctypes.c_char_p) or (isinstance(t, type) and issubclass(t, (ctypes._Pointer, ctypes.Array)))


def _declarations():
    for fn in sorted(os.listdir(os.path.join(ROOT, "tools"))):
        if not fn.endswith(".py") or fn == "qwentin_abi.py":
            continue
        path = os.path.join(ROOT, "tools", fn)
        with open(path) as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
                continue
            t = node.targets[0]
            if (isinstance(t, ast.Attribute) and t.attr in ("argtypes", "restype")
                    and isinstance(t.value, ast.Attribute) and t.value.attr.startswith("qwn_")):
                yield fn, node.lineno, t.value.attr, t.attr, eval(compile(ast.Expression(node.value), path, "eval"), dict(_NS))


def test_generated_files_are_current():
    d = g.definitions(ROOT)
    for rel, text in ((g.HEADER, g.render_header(d)), (g.ABI, g.render_abi(d))):
        with open(os.path.join(ROOT, rel)) as f:
            assert f.read() == text, f"{rel} is stale: run python3 tools/gen_qwentin_h.py"


def test_handwritten_ctypes_match_the_header():
    bad, n = [], 0
    for fn, line, name, kind, val in _declarations():
        n += 1
        if name not in qwentin_abi.SIGNATURES:
            bad.append(f"{fn}:{line} {name}: not an entry point of the engine")
            continue
        res, args = qwentin_abi.SIGNATURES[name]
        if kind == "restype":
            ok = val is None or val == res or (res is not None and _is_ptr(res) and _is_ptr(val))
        else:
            ok = len(val) == len(args) and all(a == b or (_is_ptr(a) and _is_ptr(b)) for a, b in zip(val, args))
        if not ok:
            bad.append(f"{fn}:{line} {name}.{kind} = {val!r}, header: {res if kind == 'restype' else args}")
    assert n > 100
    assert not bad, "\n".join(bad)
