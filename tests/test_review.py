"""Owner review of first-pass "same" pairs: page data export and decision ingest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from senselm.inventory import review


def _setup(tmp_path: Path) -> dict[str, Any]:
    senses = pl.DataFrame(
        {
            "source_id": ["o1", "o2", "o3", "w1", "w2", "w3"],
            "gloss": [
                "a small rodent",
                "a hand-operated device",
                "timid person",
                "A rodent.",
                "An input device.",
                "A shy  person.",
            ],
            "examples": [
                [],
                ["move the mouse"],
                [],
                ["1890, A. Author, Book: a mouse ran", "the mouse squeaked"],
                [],
                ["x" * 80],
            ],
            "example_types": [[], ["example"], [], ["quotation", "example"], [], ["example"]],
            "tags": [[], [], [], ["countable", "raw:of animals"], [], ["figuratively"]],
            "topics": [[], [], [], ["sciences", "zoology"], ["computing"], []],
            "gloss_path": [[], [], [], [], [], ["A person."]],
            "relations": [
                '{"members": ["mouse", "Mus"]}',
                '{"members": ["mouse", "computer mouse"]}',
                "{}",
                None,
                None,
                None,
            ],
            "lexfile": ["noun.animal", "noun.artifact", "noun.person", None, None, None],
        }
    )
    senses_path = tmp_path / "senses.parquet"
    senses.write_parquet(senses_path)
    align = tmp_path / "align.yaml"
    align.write_text(f"input: {senses_path}\n", encoding="utf-8")
    job = pl.DataFrame(
        {
            "job_id": ["j1", "j2", "j3", "g1"],
            "kind": ["band", "band", "band", "gold"],
            "oewn_id": ["o1", "o2", "o3", "o1"],
            "wikt_id": ["w1", "w2", "w3", "w2"],
            "lemma": ["mouse"] * 4,
            "upos": ["NOUN"] * 4,
            "cosine": [0.60, 0.66, 0.57, 0.5],
            "gold_label": [None, None, None, "different"],
            "message": ["m"] * 4,
        }
    )
    job_path = tmp_path / "job.parquet"
    job.write_parquet(job_path)
    gold = tmp_path / "band_gold.csv"
    gold.write_text("job_id,label,ambiguous\nj2,same,0\nj3,related,1\n", encoding="utf-8")
    return {
        "verify_job_path": str(job_path),
        "align_config": str(align),
        "band_gold_path": str(gold),
        "band_strata": [0.55, 0.62, 0.69],
        "review": {
            "seed": "t",
            "block_size": 2,
            "example_chars": 40,
            "decisions_path": str(tmp_path / "human.parquet"),
            "report_path": str(tmp_path / "report.json"),
        },
    }


def test_export_blocks_order_and_context(tmp_path: Path) -> None:
    cfg = _setup(tmp_path)
    out = tmp_path / "pairs"
    idx = review.export(cfg, out)
    assert idx["total"] == 3 and idx["blocks"] == 2  # the gold row is not reviewed
    pairs = (
        json.loads((out / "p000.json").read_text())["pairs"]
        + json.loads((out / "p001.json").read_text())["pairs"]
    )
    ids = [p[0] for p in pairs]
    assert ids == sorted(["j1", "j2", "j3"], key=lambda j: review.order_key("t", j))
    rows = {p[0]: dict(zip(review.FIELDS, p, strict=True)) for p in pairs}
    assert rows["j1"]["b_example"] == "the mouse squeaked"  # usage example beats a quotation
    assert rows["j1"]["b_labels"] == "of animals · zoology"  # noise tags and broad topics dropped
    assert rows["j1"]["a_synonyms"] == "Mus" and rows["j1"]["a_class"] == "noun.animal"
    assert rows["j2"]["a_example"] == "move the mouse" and rows["j2"]["b_labels"] == "computing"
    assert rows["j3"]["b_parent"] == "A person." and rows["j3"]["b_gloss"] == "A shy person."
    assert len(rows["j3"]["b_example"]) == 40 and rows["j3"]["b_example"].endswith("…")


def test_ingest_dump_shapes_and_precision(tmp_path: Path) -> None:
    cfg = _setup(tmp_path)
    order = sorted(["j1", "j2", "j3"], key=lambda j: review.order_key("t", j))
    pos = {j: i for i, j in enumerate(order)}
    block = {j: f"b{pos[j] // 2:03d}" for j in order}
    answers = {"j1": "s", "j2": "r", "j3": "u"}
    docs: dict[str, dict[str, Any]] = {}
    for n, (j, letter) in enumerate(answers.items()):
        docs.setdefault(block[j], {})[j] = [letter, 1_000_000 + 3000 * n]
    docs.setdefault("b000", {})["junk"] = ["x", 1]  # unknown label: ignored
    as_list = [{"id": f"reviews/{k}", "data": v} for k, v in docs.items()]
    assert review.parse_dump(docs).height == review.parse_dump(as_list).height == 3

    dump = tmp_path / "dump.json"
    dump.write_text(json.dumps({"documents": as_list}), encoding="utf-8")
    res = review.ingest(cfg, dump)
    assert res["reviewed"] == 3 and res["contiguous_prefix"] == 3
    fp = res["first_pass_same"]
    assert fp["same"] == 1 and fp["unsure"] == 1 and fp["precision"] == 0.5
    assert res["median_seconds_per_pair"] == 3.0
    assert res["agreement_with_fresh_gold"] == {"pairs": 1, "agree": 0, "agree_same_vs_not": 0}
    saved = pl.read_parquet(cfg["review"]["decisions_path"])
    assert dict(zip(saved["job_id"], saved["decision"], strict=True)) == {
        "j1": "same",
        "j2": "related",
        "j3": "unsure",
    }
    assert review.wilson(0, 0) is None and review.wilson(5, 10) == (0.2366, 0.7634)
