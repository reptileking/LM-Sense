from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import polars as pl

from senselm.common import read_provenance
from senselm.ingest import oewn, wiktionary
from senselm.ingest.schema import REQUIRED_COLUMNS


def test_classify() -> None:
    assert wiktionary.classify({"glosses": ["x"]}) == "sense"
    assert (
        wiktionary.classify({"glosses": ["plural of x"], "form_of": [{"word": "x"}]}) == "form_of"
    )
    assert wiktionary.classify({"glosses": ["Alt of x"], "tags": ["alt-of"]}) == "alt_of"
    assert wiktionary.classify({"tags": ["no-gloss"]}) == "no_gloss"
    assert wiktionary.classify({"glosses": []}) == "no_gloss"


def test_most_specific_topic_prefers_rarest() -> None:
    freq = Counter({"sciences": 100, "mathematics": 20, "category-theory": 2})
    assert (
        wiktionary.most_specific_topic(["category-theory", "mathematics", "sciences"], freq)
        == "category-theory"
    )
    assert wiktionary.most_specific_topic([], freq) is None


def test_disambiguate_id_is_content_stable() -> None:
    seen = {"en-x-en-noun-A"}
    a = wiktionary.disambiguate_id("en-x-en-noun-A", ["parent", "child one"], seen)
    b = wiktionary.disambiguate_id("en-x-en-noun-A", ["parent", "child one"], {"en-x-en-noun-A"})
    assert a == b and a.startswith("en-x-en-noun-A#")
    seen.add(a)
    c = wiktionary.disambiguate_id("en-x-en-noun-A", ["parent", "child one"], seen)
    assert c != a and c.startswith(a)
    assert wiktionary.disambiguate_id("fresh", ["g"], seen) == "fresh"


def test_wiktionary_ingest(tmp_path: Path, wikt_dump: Path) -> None:
    out = tmp_path / "senses_raw_wiktionary.parquet"
    cfg = {
        "stage": "t",
        "input": str(wikt_dump),
        "output": str(out),
        "lang_code": "en",
        "source_version": "test-1",
        "batch_rows": 2,
        "drop_form_tags": ["table-tags"],
    }
    stats = wiktionary.run(cfg)
    df = pl.read_parquet(out)
    assert set(REQUIRED_COLUMNS) <= set(df.columns)
    assert stats["entries_other_lang"] == 1  # German entry skipped
    assert df.height == 10
    assert df["source_id"].n_unique() == df.height  # duplicate wiktextract id disambiguated
    bank1 = df.filter(pl.col("source_id") == "en-bank-en-noun-AAAA").row(0, named=True)
    assert bank1["domain"] == "business" or bank1["domain"] == "finance"
    assert bank1["examples"] == ["I went to the bank.", "A long quotation."]
    assert bank1["example_types"] == ["example", "quotation"]
    assert bank1["etymology_number"] == 1  # string "1" coerced
    assert {"form": "banks", "tags": ["plural"]} in bank1["forms"]
    assert all(f["form"] != "no-table-tags" for f in bank1["forms"])
    sub = df.filter(pl.col("source_id").str.starts_with("en-bank-en-noun-AAAA#")).row(0, named=True)
    assert sub["gloss"] == "A branch office of such an institution."
    assert sub["gloss_path"] == ["An institution where one can place and borrow money."]
    kinds = dict(zip(df["source_id"], df["kind"], strict=True))
    assert kinds["en-banks-en-noun-CCCC"] == "form_of"
    assert kinds["en-NASA-en-name-DDDD"] == "alt_of"
    assert kinds["en-because-en-conj-IIII"] == "no_gloss"
    nasa = df.filter(pl.col("lemma") == "NASA").row(0, named=True)
    assert nasa["targets"] == ["National Aeronautics and Space Administration"]
    assert json.loads(nasa["relations"])["alt_of"] == [
        "National Aeronautics and Space Administration"
    ]
    prov = read_provenance(out)
    assert prov and prov["source_versions"] == {"wiktionary": "test-1"}
    assert prov["inputs"][0]["sha256"]


def test_wiktionary_ingest_is_deterministic(tmp_path: Path, wikt_dump: Path) -> None:
    outs = []
    for i in range(2):
        out = tmp_path / f"o{i}.parquet"
        wiktionary.run(
            {
                "stage": "t",
                "input": str(wikt_dump),
                "output": str(out),
                "lang_code": "en",
                "source_version": "v",
                "batch_rows": 3,
            }
        )
        outs.append(pl.read_parquet(out))
    assert outs[0].equals(outs[1])


def test_oewn_ingest(tmp_path: Path, oewn_files: tuple[Path, Path]) -> None:
    plus, base = oewn_files
    out = tmp_path / "senses_raw_oewn.parquet"
    oewn.run(
        {
            "stage": "t",
            "input": str(plus),
            "base_input": str(base),
            "output": str(out),
            "source_version": "oewn-test",
            "batch_rows": 2,
        }
    )
    df = pl.read_parquet(out)
    assert df.height == 5
    bank = df.filter(pl.col("lemma") == "bank").sort("sense_rank")
    assert bank["gloss"].to_list()[0].startswith("sloping land")
    assert bank["sense_rank"].to_list() == [0, 1]
    first = bank.row(0, named=True)
    assert first["domain"] == "geography"
    assert first["lexfile"] == "noun.object"
    assert first["hypernyms"] == ["oewn-06128570-n"]
    assert first["examples"] == ["they pulled the canoe up on the bank"]
    assert json.loads(first["relations"])["hypernym_names"] == ["geography"]
    assert bank.row(1, named=True)["tags"] == ["usage:american_spelling"]
    paris = df.filter(pl.col("lemma") == "Paris").row(0, named=True)
    assert "oewn:namenet" in paris["tags"]  # absent from the base edition
    run_ = df.filter(pl.col("lemma") == "run").row(0, named=True)
    assert run_["forms"] == [{"form": "ran", "tags": []}]
