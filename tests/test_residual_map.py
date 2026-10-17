"""CPU-only tests of tools/residual_map.py: the server's --residual-map flags, the deprecated 2026-08 aliases
mapped onto them, the maps-file loader (new and legacy spelling) and the decay-schedule parser. No engine,
no numpy (the reference-formula test skips without numpy)."""

from __future__ import annotations

import argparse
import json
import os
import sys

import pytest

TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
sys.path.insert(0, TOOLS)

import residual_map as rm  # noqa: E402

H = 4
MAP = {"P": [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
       "A_minus_I": [[0.5, 0.0], [0.0, -0.5]],
       "mu": [0.0, 0.0, 0.0, 0.0],
       "shift": [0.1, 0.2, 0.3, 0.4]}
LEGACY_MAP = {"P": MAP["P"], "A_k_minus_I": MAP["A_minus_I"], "mu_H": MAP["mu"], "mean_shift": MAP["shift"]}


def _parse(argv, env=None):
    ap = argparse.ArgumentParser()
    rm.add_residual_map_args(ap, env=env if env is not None else {})
    a = ap.parse_args(argv)
    notes = rm.resolve_residual_map_args(a, env=env if env is not None else {})
    return a, notes


def _write(tmp_path, name, doc):
    p = os.path.join(str(tmp_path), name)
    with open(p, "w") as f:
        json.dump(doc, f)
    return p


# ----------------------------------------------------------------------------- flags


def test_new_flags_defaults_are_off():
    a, notes = _parse([])
    assert a.residual_map is None and a.residual_map_layers is None and a.residual_map_scale is None
    assert a.residual_map_decay is None and a.residual_map_sched_api is False
    assert notes == []


def test_new_flags_parse():
    a, notes = _parse(["--residual-map", "m.json", "--residual-map-layers", "3,4", "--residual-map-scale", "0.3",
                       "--residual-map-decay", "0.1,200,reset", "--residual-map-sched-api"])
    assert (a.residual_map, a.residual_map_layers, a.residual_map_scale) == ("m.json", "3,4", 0.3)
    assert a.residual_map_decay == "0.1,200,reset" and a.residual_map_sched_api is True
    assert notes == []


def test_new_flags_from_env():
    a, notes = _parse([], env={"TQ_RESIDUAL_MAP": "env.json", "TQ_RESIDUAL_MAP_DECAY": "0.2,100"})
    assert a.residual_map == "env.json" and a.residual_map_decay == "0.2,100"
    assert notes == []


def test_deprecated_aliases_map_onto_new_flags():
    a, notes = _parse(["--bark-all-day", "--ot-maps", "old.json", "--ot-layers", "37,38", "--ot-scale", "0.47",
                       "--bark-decay-default", "0.35,400,reset", "--bark-sched-api"])
    assert a.residual_map == "old.json"
    assert a.residual_map_layers == "37,38"
    assert a.residual_map_scale == 0.47
    assert a.residual_map_decay == "0.35,400,reset"
    assert a.residual_map_sched_api is True
    assert len(notes) == 5 and all(n.startswith("[serve] deprecated: ") and "\n" not in n for n in notes)
    assert any("--residual-map PATH" in n for n in notes)
    assert any("--residual-map-sched-api" in n for n in notes)


def test_deprecated_aliases_hidden_from_help():
    ap = argparse.ArgumentParser(prog="x")
    rm.add_residual_map_args(ap, env={})
    text = ap.format_help()
    for new in ("--residual-map PATH", "--residual-map-layers", "--residual-map-scale", "--residual-map-decay",
                "--residual-map-sched-api"):
        assert new in text
    for old in ("--bark-all-day", "--ot-maps", "--ot-layers", "--ot-scale", "--bark-decay-default", "--bark-sched-api"):
        assert old not in text


def test_new_flag_wins_over_alias():
    a, notes = _parse(["--residual-map", "new.json", "--bark-all-day", "--ot-maps", "old.json",
                       "--residual-map-scale", "0.3", "--ot-scale", "0.47"])
    assert a.residual_map == "new.json" and a.residual_map_scale == 0.3
    assert len(notes) == 2


def test_legacy_env_vars():
    # TQ_OT_MAPS armed nothing on its own in 2026-08 and still does not
    a, notes = _parse([], env={"TQ_OT_MAPS": "env.json"})
    assert a.residual_map is None and notes == []
    # ... but supplied the path to --bark-all-day
    a, notes = _parse(["--bark-all-day"], env={"TQ_OT_MAPS": "env.json"})
    assert a.residual_map == "env.json" and len(notes) == 1
    # TQ_BARK_DECAY was the default schedule
    a, notes = _parse([], env={"TQ_BARK_DECAY": "0.3,100"})
    assert a.residual_map_decay == "0.3,100" and len(notes) == 1
    # the new env var wins over the old one
    a, notes = _parse([], env={"TQ_BARK_DECAY": "0.3,100", "TQ_RESIDUAL_MAP_DECAY": "0.2,50"})
    assert a.residual_map_decay == "0.2,50"


def test_bark_all_day_without_maps_is_an_error():
    with pytest.raises(ValueError):
        _parse(["--bark-all-day"])


def test_ot_maps_without_bark_all_day_arms_nothing():
    a, notes = _parse(["--ot-maps", "old.json"])
    assert a.residual_map is None and len(notes) == 1 and "ignored" in notes[0]


def test_no_tuning_defaults_leak_from_aliases():
    # the 2026-08 defaults (--ot-layers 37,38 / --ot-scale 0.47) are gone: layers and scale come from the file
    a, _ = _parse(["--bark-all-day", "--ot-maps", "old.json"])
    assert a.residual_map_layers is None and a.residual_map_scale is None


# ----------------------------------------------------------------------------- the maps file


def test_load_new_format(tmp_path):
    p = _write(tmp_path, "maps.json", {"format": rm.FORMAT, "hidden_size": H, "layers": [3], "scale": 0.25,
                                       "maps": {"3": MAP, "4": MAP}})
    r = rm.load_residual_map(p)
    assert r["layers"] == [3] and r["scale"] == 0.25 and r["hidden_size"] == H and r["format"] == rm.FORMAT
    assert set(r["maps"]) == {3}
    assert r["maps"][3]["A_minus_I"] == [[0.5, 0.0], [0.0, -0.5]]
    assert r["maps"][3]["shift"] == MAP["shift"] and r["maps"][3]["mu"] == MAP["mu"]
    # flags override the file
    r = rm.load_residual_map(p, layers="3,4", scale=0.5)
    assert r["layers"] == [3, 4] and r["scale"] == 0.5
    # the engine's hidden size is checked against the file's
    assert rm.load_residual_map(p, hidden_size=H)["hidden_size"] == H
    with pytest.raises(ValueError):
        rm.load_residual_map(p, hidden_size=H + 1)


def test_load_legacy_format_needs_scale(tmp_path):
    p = _write(tmp_path, "old.json", {"ot_maps": {"3": LEGACY_MAP, "4": LEGACY_MAP}})
    with pytest.raises(ValueError) as e:
        rm.load_residual_map(p)
    assert "scale" in str(e.value)
    r = rm.load_residual_map(p, scale=0.47)
    assert r["layers"] == [3, 4] and r["scale"] == 0.47 and r["hidden_size"] == H     # layers default: all in the file
    assert r["maps"][4]["shift"] == MAP["shift"] and r["maps"][4]["A_minus_I"] == MAP["A_minus_I"]
    r = rm.load_residual_map(p, layers=[4], scale=0.47)
    assert r["layers"] == [4]


def test_load_rejects_bad_files(tmp_path):
    with pytest.raises(ValueError):
        rm.load_residual_map(_write(tmp_path, "a.json", {"scale": 1.0}))                       # no maps
    with pytest.raises(ValueError):
        rm.load_residual_map(_write(tmp_path, "b.json", {"scale": 1.0, "maps": {"3": MAP}}), layers="7")   # layer absent
    bad = dict(MAP, P=[[1.0, 0.0, 0.0]] * H)                                                     # P not H x 2
    with pytest.raises(ValueError):
        rm.load_residual_map(_write(tmp_path, "c.json", {"scale": 1.0, "maps": {"3": bad}}))
    bad = dict(MAP, mu=[0.0] * (H + 1))                                                          # mu wrong length
    with pytest.raises(ValueError):
        rm.load_residual_map(_write(tmp_path, "d.json", {"scale": 1.0, "maps": {"3": bad}}))
    bad = {k: v for k, v in MAP.items() if k != "shift"}                                         # missing key
    with pytest.raises(ValueError):
        rm.load_residual_map(_write(tmp_path, "e.json", {"scale": 1.0, "maps": {"3": bad}}))
    with pytest.raises(FileNotFoundError):
        rm.load_residual_map(os.path.join(str(tmp_path), "missing.json"))


def test_flat_a_matrix_is_accepted(tmp_path):
    p = _write(tmp_path, "f.json", {"scale": 1.0, "maps": {"3": dict(MAP, A_minus_I=[0.5, 0.0, 0.0, -0.5])}})
    assert rm.load_residual_map(p)["maps"][3]["A_minus_I"] == [[0.5, 0.0], [0.0, -0.5]]


# ----------------------------------------------------------------------------- the schedule


def test_parse_decay():
    assert rm.parse_decay("") is None and rm.parse_decay(None) is None
    assert rm.parse_decay("0.35") == {"end": 0.35, "len": 192, "reset_think": False, "shape": "linear",
                                      "end_answer": None, "len_answer": 400}
    assert rm.parse_decay("0.35,400,reset") == {"end": 0.35, "len": 400, "reset_think": True, "shape": "linear",
                                                "end_answer": None, "len_answer": 400}
    assert rm.parse_decay("0.3,100,0,pow,0.2,250") == {"end": 0.3, "len": 100, "reset_think": False, "shape": "pow",
                                                       "end_answer": 0.2, "len_answer": 250}
    with pytest.raises(ValueError):
        rm.parse_decay("x")


def test_request_decay_field_and_alias():
    assert rm.request_decay({}) is None
    assert rm.request_decay({"residual_map_decay": {"end": 0.1}}) == {"end": 0.1}
    assert rm.request_decay({"residual_map_decay": "no"}) is None
    assert rm.request_decay({"bark_decay": {"end": 0.2}}) == {"end": 0.2}             # deprecated spelling still read
    assert rm.request_decay({"residual_map_decay": {"end": 0.1}, "bark_decay": {"end": 0.2}}) == {"end": 0.1}


def test_parse_layers():
    assert rm.parse_layers("37,38") == [37, 38] and rm.parse_layers(" 3, ") == [3] and rm.parse_layers([1, "2"]) == [1, 2]


# ----------------------------------------------------------------------------- the formula


def test_apply_reference_matches_hand_computation():
    np = pytest.importorskip("numpy")
    x = np.array([[2.0, 4.0, 1.0, 1.0]])
    # P picks coordinates 0 and 1, A - I = diag(0.5, -0.5), mu = 0: delta = shift + (0.5 * 2, -0.5 * 4, 0, 0)
    out = rm.apply_reference(x, MAP, 2.0)
    assert np.allclose(out, [[2.0 + 2 * (0.1 + 1.0), 4.0 + 2 * (0.2 - 2.0), 1.0 + 2 * 0.3, 1.0 + 2 * 0.4]])
