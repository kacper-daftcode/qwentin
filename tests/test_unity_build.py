"""src/forward_qwen.cu is a unity build: it includes qwentin.h and every src/engine/*.cuh part exactly once, no code."""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import engine_text  # noqa: E402


def test_every_part_included_once():
    parts = sorted(f for f in os.listdir(os.path.join(ROOT, "src", "engine")) if f.endswith(".cuh"))
    included = [os.path.basename(p) for p in engine_text.engine_source_files(ROOT)]
    assert len(included) == len(set(included)), "a part is included twice"
    assert sorted(included) == parts


def test_unity_file_holds_only_includes():
    with open(os.path.join(ROOT, "src", "forward_qwen.cu")) as f:
        body = re.sub(r"/\*.*?\*/", "", f.read(), flags=re.S)
    code = [ln for ln in body.splitlines() if ln.strip() and not ln.startswith('#include "engine/') and ln != '#include "qwentin.h"']
    assert code == []


def test_engine_text_is_the_whole_source():
    assert len(engine_text.engine_source_text(ROOT)) > 1_000_000
