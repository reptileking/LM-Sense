"""End-to-end pipeline on the tiny fixtures, with a deterministic fake embedder."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from senselm.common import load_config
from senselm.ingest import combine, oewn, wiktionary
from senselm.inventory import align, build, normalise
from senselm.inventory.ids import Layout, check_append_only

REPO = Path(__file__).resolve().parents[1]
CONCEPTS = {"money": 0, "financial": 0, "river": 1, "slop": 1, "move fast": 2, "feet": 2}


def fake_embed(
    texts: list[str], cfg: dict[str, Any], embedder: Any = None
) -> tuple[np.ndarray, dict[str, Any]]:
    out = np.zeros((len(texts), 16), dtype=np.float32)
    for i, t in enumerate(texts):
        hits = [c for k, c in CONCEPTS.items() if k in t.lower()]
        if hits:
            out[i, hits[0]] = 1.0
        else:  # unrelated texts: pseudo-random direction in the remaining dims
            h = hashlib.blake2b(t.encode(), digest_size=8).digest()
            out[i, 3 + h[0] % 13] = 1.0
    return out, {"model": "fake", "computed": len(texts)}


@pytest.fixture()
def pipeline(
    tmp_path: Path, wikt_dump: Path, oewn_files: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    monkeypatch.setattr(align, "embed_texts", fake_embed)
    p = {
        k: str(tmp_path / f"{k}.parquet")
        for k in (
            "w",
            "o",
            "raw",
            "senses",
            "forms",
            "variants",
            "dedupe_map",
            "pairs",
            "inv",
            "reg",
            "lem",
        )
    }
    wiktionary.run(
        {
            "stage": "t",
            "input": str(wikt_dump),
            "output": p["w"],
            "lang_code": "en",
            "source_version": "w1",
        }
    )
    oewn.run(
        {
            "stage": "t",
            "input": str(oewn_files[0]),
            "base_input": str(oewn_files[1]),
            "output": p["o"],
            "source_version": "o1",
        }
    )
    combine.run({"stage": "t", "inputs": [p["w"], p["o"]], "output": p["raw"]})
    ncfg = load_config(REPO / "configs/normalise.yaml")
    normalise.run(
        {
            **ncfg,
            "input": p["raw"],
            "outputs": {k: p[k] for k in ("senses", "forms", "variants", "dedupe_map")},
        }
    )
    acfg = load_config(REPO / "configs/align.yaml")
    align.run({**acfg, "input": p["senses"], "output": p["pairs"]})
    icfg = load_config(REPO / "configs/inventory.yaml")
    icfg = {
        **icfg,
        "inputs": {"senses": p["senses"], "pairs": p["pairs"], "dedupe_map": p["dedupe_map"]},
        "outputs": {
            "inventory": p["inv"],
            "registry": p["reg"],
            "lemma_registry": p["lem"],
            "history_dir": str(tmp_path / "hist"),
        },
        "policy": {**icfg["policy"], "auto_merge_threshold": 0.9},
    }
    return {"paths": p, "icfg": icfg, "tmp": tmp_path}


def test_build_end_to_end(pipeline: dict[str, Any]) -> None:
    p, icfg = pipeline["paths"], pipeline["icfg"]
    stats = build.run(icfg)
    inv = pl.read_parquet(p["inv"])
    assert inv["token_id"].null_count() == 0 and inv["token_id"].n_unique() == inv.height
    lay = Layout(**icfg["ids"]["layout"])
    for tid, lid, si in inv.select("token_id", "lemma_id", "sense_index").iter_rows():
        assert lay.unpack(tid) == (lid, si)
    bank = inv.filter(pl.col("lemma") == "bank").sort("sense_index")
    # WordNet sense order: 'sloping land' (OEWN sense 1) gets index 0 and absorbed the river sense
    first = bank.row(0, named=True)
    assert first["gloss"].startswith("sloping land")
    assert {m["source"] for m in first["members"]} == {"oewn", "wiktionary"}
    assert first["licences"] == ["CC-BY-4.0", "CC-BY-SA-4.0+GFDL"]
    fin = bank.filter(pl.col("gloss").str.contains("financial")).row(0, named=True)
    assert any(m["source_id"] == "en-bank-en-noun-AAAA" for m in fin["members"])
    assert "NASA" in set(inv["lemma"])
    paris = inv.filter(pl.col("lemma") == "Paris").row(0, named=True)
    assert paris["upos"] == "PROPN" and "oewn:namenet" in paris["tags"]
    assert stats["ids"]["new_ids"] == inv.height


def test_rebuild_is_deterministic_and_append_only(pipeline: dict[str, Any]) -> None:
    p, icfg = pipeline["paths"], pipeline["icfg"]
    build.run(icfg)
    reg1 = pl.read_parquet(p["reg"])
    inv1 = pl.read_parquet(p["inv"])
    build.run(icfg)  # identical inputs -> identical registry
    reg2 = pl.read_parquet(p["reg"])
    assert reg1.equals(reg2)
    assert (
        pl.read_parquet(p["inv"])
        .select("token_id", "canonical_key")
        .equals(inv1.select("token_id", "canonical_key"))
    )
    # drop one source sense ("because") from the input: its id must become a tombstone
    senses = pl.read_parquet(p["senses"])
    senses.filter(pl.col("lemma") != "because").write_parquet(p["senses"])
    build.run(icfg)
    reg3 = pl.read_parquet(p["reg"])
    assert check_append_only(reg2, reg3) == []
    dead = reg3.filter(pl.col("lemma") == "because")
    assert dead["status"].to_list() == ["deprecated"] and dead["deprecated_in"].to_list() == [
        "v0.1"
    ]
    assert list(
        (pipeline["tmp"] / "hist").glob("id_registry.*.parquet")
    )  # previous version snapshotted


def test_rebuild_from_committed_ledger_after_data_loss(pipeline: dict[str, Any]) -> None:
    """Fresh machine: data/processed is gone, only the git-tracked ledger remains."""
    p, icfg = pipeline["paths"], pipeline["icfg"]
    ledger_dir = pipeline["tmp"] / "registry"
    cfg = {**icfg, "outputs": {**icfg["outputs"], "registry_commit_dir": str(ledger_dir)}}
    build.run(cfg)
    reg1 = pl.read_parquet(p["reg"])
    assert (ledger_dir / "id_registry.parquet").exists()
    assert "gloss" not in pl.read_parquet(ledger_dir / "id_registry.parquet").columns
    Path(p["reg"]).unlink()
    Path(p["lem"]).unlink()
    build.run(cfg)
    reg2 = pl.read_parquet(p["reg"])
    assert check_append_only(reg1, reg2) == []
    assert reg1.select("token_id", "canonical_key").equals(reg2.select("token_id", "canonical_key"))
