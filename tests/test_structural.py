"""Structural vocabulary: reserved range, stability across builds."""

from __future__ import annotations

import copy
from pathlib import Path

from senselm.common import load_config
from senselm.inventory.ids import STRUCTURAL_BASE, Layout
from senselm.inventory.structural import build

CFG = load_config(Path(__file__).resolve().parents[1] / "configs" / "structural.yaml")


def test_layout_and_reserved_range() -> None:
    df = build(CFG, None)
    lay = Layout()
    assert df["token_id"].min() == STRUCTURAL_BASE  # <pad> is the first structural id
    assert all(lay.is_structural(t) for t in df["token_id"].to_list())
    byte_a = df.filter(df["name"] == "0x41").row(0, named=True)
    assert byte_a["index"] == 0x41 and byte_a["token"] == "<byte:0x41>"
    start = df.filter(df["name"] == "bytes_start").row(0, named=True)
    assert start["index"] == 256  # listed byte-category items follow the 256 byte values
    assert df.filter(df["category"] == "digit").height == 10


def test_append_only_across_spec_changes() -> None:
    v1 = build(CFG, None)
    cfg2 = copy.deepcopy(CFG)
    morph = next(c for c in cfg2["categories"] if c["name"] == "morph")
    morph["items"].append(["EVID", "evidential (new)"])
    morph["items"] = [i for i in morph["items"] if i[0] != "FUT"]  # removal -> tombstone
    v2 = build({**cfg2, "build_id": "v2"}, v1)
    ids1 = dict(zip(v1["token"], v1["token_id"], strict=True))
    ids2 = dict(zip(v2["token"], v2["token_id"], strict=True))
    assert all(ids2[t] == i for t, i in ids1.items())  # nothing renumbered
    assert v2.filter(v2["name"] == "FUT")["status"].to_list() == ["deprecated"]
    evid = v2.filter(v2["name"] == "EVID").row(0, named=True)
    assert evid["index"] == v1.filter(v1["category"] == "morph")["index"].max() + 1
