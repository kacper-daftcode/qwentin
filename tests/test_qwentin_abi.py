"""src/qwentin.h and tools/qwentin_abi.py match the engine source, qwentin_abi.Config has qwn_config's layout (compiled
with g++; skipped without it), and every hand-written ctypes declaration in tools/ (`L.qwn_x.argtypes = [...]`,
`.restype = ...`) matches the header: same arity, same scalar types, pointers as pointers."""

from __future__ import annotations

import ast
import ctypes
import os
import shutil
import subprocess
import sys
import tempfile

import pytest

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
    d, cf = g.definitions(ROOT), g.config_fields(ROOT)
    for rel, text in ((g.HEADER, g.render_header(d, cf)), (g.ABI, g.render_abi(d, cf))):
        with open(os.path.join(ROOT, rel)) as f:
            assert f.read() == text, f"{rel} is stale: run python3 tools/gen_qwentin_h.py"


def test_config_struct_layout():
    gxx = shutil.which("g++")
    if not gxx:
        pytest.skip("no g++")
    fields = [f for f, _ in qwentin_abi.Config._fields_]
    probe = "\n".join(["#include <stdio.h>", "#include <stddef.h>", '#include "qwentin.h"', "int main(void) {",
                       '    printf("%zu\\n", sizeof(qwn_config));']
                      + [f'    printf("%zu\\n", offsetof(qwn_config, {f}));' for f in fields] + ["}"])
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "probe.cpp"), "w") as f:
            f.write(probe)
        subprocess.run([gxx, "-I", os.path.join(ROOT, "src"), "-o", os.path.join(d, "probe"), os.path.join(d, "probe.cpp")],
                       check=True)
        got = [int(x) for x in subprocess.run([os.path.join(d, "probe")], check=True, capture_output=True,
                                               text=True).stdout.split()]
    want = [ctypes.sizeof(qwentin_abi.Config)] + [getattr(qwentin_abi.Config, f).offset for f in fields]
    assert got == want, "tools/qwentin_abi.py's Config does not match qwn_config"
    assert sorted(qwentin_abi.CONFIG_FIELDS.values()) == sorted(fields)


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
    assert n > 50
    assert not bad, "\n".join(bad)
