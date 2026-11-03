"""Wikidata items: line filter, multi-member inflation and the concept selection (D-032, D-038)."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import polars as pl
import pyarrow as pa
import pytest

from senselm.ingest import wikidata_items as wi


def _entity(
    qid: str,
    label: str | None,
    *,
    desc: str | None = "a thing",
    p31: list[str] | None = None,
    p279: list[str] | None = None,
    xrefs: dict[str, list[str]] | None = None,
    enwiki: str | None = "Page",
) -> dict:
    claims: dict = {}

    def item(q: str) -> dict:
        return {"mainsnak": {"datavalue": {"value": {"id": q}}}}

    def string(v: str) -> dict:
        return {"mainsnak": {"datavalue": {"value": v}}}

    if p31:
        claims["P31"] = [item(q) for q in p31]
    if p279:
        claims["P279"] = [item(q) for q in p279]
    for prop, vals in (xrefs or {}).items():
        claims[prop] = [string(v) for v in vals]
    return {  # key order and compact separators as in the dump (the line filter relies on them)
        "type": "item",
        "id": qid,
        "labels": {"en": {"language": "en", "value": label}} if label else {},
        "descriptions": {"en": {"value": desc}} if desc else {},
        "aliases": {},
        "claims": claims,
        "sitelinks": {"enwiki": {"title": enwiki}} if enwiki else {},
    }


def test_filter_and_inflate_multi_member() -> None:
    ents = [
        _entity("Q1", "red dwarf", p279=["Q2"]),
        _entity("Q3", None, p279=["Q2"]),  # no English label
        _entity("Q4", "Mus musculus", p31=["Q16521"], xrefs={"P685": ["10090"]}),  # domain xref
        _entity("Q5", "some person", p31=["Q5"]),  # neither class, xref nor linked
        _entity("Q6", "seal", p31=["Q9"]),  # linked from a lexeme sense
    ]
    lines = ["["] + [json.dumps(e, separators=(",", ":")) + "," for e in ents] + ["]"]
    raw = "\n".join(lines).encode()
    blob = gzip.compress(b"[\n") + gzip.compress(raw[2:])  # two gzip members, like the dump
    md5 = hashlib.md5()
    out: list[bytes] = []
    for batch in wi.inflate_lines(iter([blob[:50], blob[50:]]), [md5]):
        out += batch
    assert md5.hexdigest() == hashlib.md5(blob).hexdigest()
    wi._init_worker({"Q6"})
    recs, st = wi.filter_lines(out)
    assert st["items"] == 5
    assert {r["qid"] for r in recs} == {"Q1", "Q4", "Q6"}
    q4 = next(r for r in recs if r["qid"] == "Q4")
    assert json.loads(q4["xrefs"]) == {"P685": ["10090"]}


def test_select_concepts(tmp_path: Path) -> None:
    rows = [
        # qid, label, description, p31, p279, enwiki, linked
        ("Q1", "red dwarf", "type of star", [], ["Q2"], "Red dwarf", False),  # concept
        (
            "Q2",
            "Portuguese water dog",
            "dog breed",
            [],
            ["Q3"],
            "P.w.d.",
            False,
        ),  # capitalised class
        ("Q4", "Lausanne", "city in Switzerland", ["Q515"], [], "Lausanne", True),  # linked: kept
        (
            "Q5",
            "cyclopropane",
            "chemical compound",
            ["Q11173"],
            ["Q6"],
            "C",
            False,
        ),  # excluded type
        ("Q7", "obscure thing", "a class", [], ["Q2"], None, False),  # no Wikipedia article
        ("Q8", "no description", None, [], ["Q2"], "X", False),  # no gloss
    ]
    sup = pa.table(
        {
            "qid": [r[0] for r in rows],
            "label": [r[1] for r in rows],
            "description": [r[2] for r in rows],
            "aliases": [[] for _ in rows],
            "p31": [r[3] for r in rows],
            "p279": [r[4] for r in rows],
            "xrefs": ["{}" for _ in rows],
            "enwiki": [r[5] for r in rows],
            "n_sitelinks": [1 if r[5] else 0 for r in rows],
            "linked": [r[6] for r in rows],
        },
        schema=wi.SUPERSET_SCHEMA,
    )
    sup_path = tmp_path / "sup.parquet"
    pl.from_arrow(sup).write_parquet(sup_path)
    cfg = {
        "stage": "t",
        "version": "t",
        "superset_output": str(sup_path),
        "output": str(tmp_path / "items.parquet"),
        "select": {
            "require_enwiki": True,
            "require_lowercase_label": True,
            "min_sitelinks": 1,
            "exclude_p31": ["Q11173"],
        },
    }
    st = wi.select(cfg)
    got = pl.read_parquet(tmp_path / "items.parquet")
    assert sorted(got["source_id"].to_list()) == ["Q1", "Q4"]
    assert st["selected"] == 2 and st["linked"] == 1
    q1 = got.filter(pl.col("source_id") == "Q1").row(0, named=True)
    assert q1["lemma"] == "red dwarf" and q1["gloss"] == "type of star" and q1["pos"] == "item"


def test_resumable_stream_survives_a_crash(tmp_path: Path) -> None:
    """A checkpointed pass killed after its first checkpoint resumes in a new call and produces
    the same superset as an uninterrupted pass."""
    import gzip
    import json

    lines = []
    for i in range(1, 3001):
        item = {
            "type": "item",
            "id": f"Q{i}",
            "labels": {"en": {"language": "en", "value": f"thing {i}"}},
            "descriptions": {"en": {"language": "en", "value": f"a thing number {i}"}},
            "claims": {"P279": [{"mainsnak": {"datavalue": {"value": {"id": "Q35120"}}}}]}
            if i % 3 == 0
            else {},
            "sitelinks": {},
        }
        lines.append(json.dumps(item, separators=(",", ":")).encode() + b",\n")
    data = b"[\n" + b"".join(lines) + b"]\n"
    cut = data.index(b"\n", len(data) // 2) + 1
    dump = tmp_path / "dump.json.gz"
    dump.write_bytes(gzip.compress(data[:cut]) + gzip.compress(data[cut:]))  # two members
    pl.DataFrame({"qid": ["Q1"]}).write_parquet(tmp_path / "linked.parquet")

    def cfg(name: str) -> dict:
        return {
            "url": f"file://{dump}",
            "version": "test",
            "linked_items": str(tmp_path / "linked.parquet"),
            "superset_output": str(tmp_path / f"{name}.parquet"),
            "resumable": True,
            "checkpoint_gb": 20_000 / 1e9,  # every ~20 kB of compressed input
            "index_spacing_mb": 0.0625,
            "read_mb": 0.0078125,  # 8 KiB reads
            "segment_mb": 0.015625,
            "workers": 1,
        }

    base = wi.stream_resumable(cfg("base"))
    with pytest.raises(RuntimeError, match="simulated crash"):
        wi.stream_resumable(cfg("resumed"), crash_after=1)
    res = wi.stream_resumable(cfg("resumed"))
    assert res["resumed"] == 1 and base["resumed"] == 0
    a = pl.read_parquet(tmp_path / "base.parquet")
    b = pl.read_parquet(tmp_path / "resumed.parquet")
    assert a.height == 1001 and a["qid"].to_list() == b["qid"].to_list()  # Q1 linked + every 3rd
    # the whole-file md5 exists only where one process read the file contiguously from byte 0
    assert base["md5"] is not None and res["md5"] in (None, base["md5"])
    assert not (tmp_path / "resumed.parquet.work").exists()
