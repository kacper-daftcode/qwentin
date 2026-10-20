"""src/engine/config.cuh is the only place the engine reads its environment, docs/FLAGS.md lists exactly its switches,
and the parse rule holds (unset or empty = the default, atoi otherwise, range handling per the table) -- the last one
compiles config.cuh with the host C++ compiler (skipped without g++)."""

from __future__ import annotations

import os
import re
import shutil
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE = os.path.join(ROOT, "src", "engine")
CONFIG = os.path.join(ENGINE, "config.cuh")


def _read(*parts: str) -> str:
    with open(os.path.join(ROOT, *parts)) as f:
        return f.read()


def _switches() -> list[str]:
    return re.findall(r'^\s*\{"(TQ_[A-Z0-9_]+)",', _read("src", "engine", "config.cuh"), flags=re.M) + ["TQ_KV_T4_CODEBOOK"]


def test_getenv_only_in_config():
    for fn in sorted(os.listdir(ENGINE)):
        if fn.endswith(".cuh") and fn != "config.cuh":
            assert "getenv(" not in _read("src", "engine", fn), f"{fn} reads the environment; parse it in config.cuh"
    assert "getenv(" not in _read("src", "forward_qwen.cu")


def test_flags_md_lists_exactly_the_switches():
    doc = _read("docs", "FLAGS.md")
    section = doc.split("## Engine environment flags", 1)[1].split("\n## ", 1)[0]
    rows = re.findall(r"^\| `(TQ_[A-Z0-9_]+)", section, flags=re.M)
    assert len(rows) == len(set(rows)), "a switch has two rows"
    assert sorted(rows) == sorted(_switches())


HARNESS = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "config.cuh"
int main(int argc, char **argv) {
    // argv[1..]: NAME=VALUE (NAME= sets the empty string); prints every switch as NAME=value, then the [config] line
    for (int i = 1; i < argc; i++) { char *eq = strchr(argv[i], '='); *eq = 0; setenv(argv[i], eq + 1, 1); }
    tq_config_load();
    for (const tq_switch_t &s : k_tq_switches) printf("%s=%d\n", s.name, g_cfg.*s.field);
    printf("TQ_KV_T4_CODEBOOK=%s\n", g_cfg.kv_t4_codebook ? g_cfg.kv_t4_codebook : "<built-in>");
    tq_config_print(stdout, &g_cfg);
    return 0;
}
"""


@pytest.fixture(scope="module")
def parse(tmp_path_factory):
    gxx = shutil.which("g++")
    if not gxx:
        pytest.skip("no g++")
    max_seq = re.search(r"#define TQ_MAX_SEQ (\d+)", _read("src", "engine", "base.cuh")).group(1)
    d = tmp_path_factory.mktemp("config")
    (d / "main.cpp").write_text(HARNESS)
    exe = d / "main"
    subprocess.run([gxx, "-std=c++17", "-Wall", "-Werror", f"-DTQ_MAX_SEQ={max_seq}", "-I", ENGINE, "-o", str(exe),
                    str(d / "main.cpp")], check=True)

    def run(*assignments: str) -> tuple[dict[str, str], str]:
        env = {k: v for k, v in os.environ.items() if not k.startswith("TQ_")}
        out = subprocess.run([str(exe), *assignments], check=True, capture_output=True, text=True, env=env).stdout
        lines = out.splitlines()
        return dict(ln.split("=", 1) for ln in lines[:-1]), lines[-1]

    return run


DEFAULTS = {
    "TQ_CTX": "8192", "TQ_KV_Q4": "0", "TQ_KV_V4": "0", "TQ_KV_T4": "0", "TQ_DSTATE_BF16": "4", "TQ_EMBED_HOST": "0",
    "TQ_EMBED_FP8": "0", "TQ_NO_MTP": "0", "TQ_VERIFY_TQF": "0", "TQ_PDL": "0", "TQ_WIDE_ATTN_MMA": "0",
    "TQ_PAGED_ATTN_MMA": "1", "TQ_GDN_PREFILL": "1", "TQ_T4_ATTN_F16": "1", "TQ_PAGED_ATTN_ST": "3", "TQ_GRAPH": "1",
    "TQ_GEMM_FORK": "1", "TQ_GEMM_FORK_MAXN": "64", "TQ_EXL3_WIDE_NG16": "1", "TQ_EXL3_BF16_R4": "1", "TQ_PATTN_GQA": "1",
    "TQ_WAVE_MAX": "128", "TQ_WIDE_ATTN_QROWS": "32", "TQ_KV_T4_ASYNC": "1", "TQ_PAGED_TOPK_PAGES": "0",
    "TQ_PAGED_TOPK_MIN_CTX": "16384", "TQ_PARK_RING": "1", "TQ_QMMA_SF_K_SPLIT": "0", "TQ_PAGED_GQA_SPLIT": "-1",
    "TQ_PATTN_DEBUG": "0", "TQ_EXL3_DEBUG": "0", "TQ_GRAPH_DEBUG": "0", "TQ_PAGED_TOPK_DEBUG": "0",
    "TQ_KV_T4_CODEBOOK": "<built-in>",
}


def test_defaults(parse):
    got, line = parse()
    assert got == DEFAULTS
    assert line == "[config] every TQ_* switch at its default"
    assert parse(*(f"{k}=" for k in DEFAULTS))[0] == DEFAULTS, "an empty value is not the default"


@pytest.mark.parametrize("assignment, expected", [
    ("TQ_KV_T4=1", "1"), ("TQ_KV_T4=0", "0"), ("TQ_KV_T4=-1", "1"), ("TQ_KV_T4=yes", "0"), ("TQ_GRAPH_DEBUG=7", "1"),
    ("TQ_EXL3_DEBUG=0", "0"), ("TQ_GEMM_FORK=0", "0"),
    ("TQ_CTX=262144", "262144"), ("TQ_CTX=1", "2048"), ("TQ_CTX=999999", "262144"),
    ("TQ_WAVE_MAX=256", "256"), ("TQ_WAVE_MAX=0", "1"), ("TQ_WAVE_MAX=300", "256"),
    ("TQ_GRAPH=2", "2"), ("TQ_GRAPH=-1", "0"), ("TQ_GEMM_FORK_MAXN=1000000", "1000000"), ("TQ_GEMM_FORK_MAXN=-5", "0"),
    ("TQ_PAGED_ATTN_ST=1", "1"), ("TQ_PAGED_ATTN_ST=7", "3"), ("TQ_GDN_PREFILL=0", "0"), ("TQ_GDN_PREFILL=9", "0"),
    ("TQ_PDL=2", "2"), ("TQ_PDL=5", "0"), ("TQ_EMBED_FP8=2", "2"), ("TQ_EMBED_FP8=7", "1"),
    ("TQ_WIDE_ATTN_QROWS=16", "16"), ("TQ_WIDE_ATTN_QROWS=24", "32"),
    ("TQ_PAGED_TOPK_PAGES=256", "256"), ("TQ_PAGED_TOPK_PAGES=9000", "4096"), ("TQ_PAGED_TOPK_MIN_CTX=-1", "0"),
    ("TQ_QMMA_SF_K_SPLIT=4", "4"), ("TQ_QMMA_SF_K_SPLIT=0", "1"), ("TQ_QMMA_SF_K_SPLIT=12", "8"),
    ("TQ_PAGED_GQA_SPLIT=8", "8"), ("TQ_DSTATE_BF16=0", "0"), ("TQ_DSTATE_BF16=9", "9"), ("TQ_PATTN_GQA=0", "0"),
    ("TQ_KV_T4_CODEBOOK=/a/b.f32", "/a/b.f32"),
])
def test_rule(parse, assignment, expected):
    name = assignment.split("=", 1)[0]
    got, _ = parse(assignment)
    assert got[name] == expected
    assert {k: v for k, v in got.items() if k != name} == {k: v for k, v in DEFAULTS.items() if k != name}


def test_config_line_lists_the_non_defaults(parse):
    _, line = parse("TQ_CTX=262144", "TQ_KV_T4=1", "TQ_GRAPH=1", "TQ_KV_T4_CODEBOOK=/c.f32")
    assert line == "[config] TQ_CTX=262144 TQ_KV_T4=1 TQ_KV_T4_CODEBOOK=/c.f32 (the other 31 TQ_* switches at their defaults)"
