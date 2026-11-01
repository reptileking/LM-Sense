from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from senselm.common import load_config
from senselm.ingest import combine, oewn, wiktionary
from senselm.inventory import normalise
from senselm.inventory.normalise import normalise_lemma, to_upos, variant_target

CFG = load_config(Path(__file__).resolve().parents[1] / "configs" / "normalise.yaml")


@pytest.mark.parametrize(
    ("gloss", "target"),
    [
        ("Alternative spelling of color.", "color"),
        ("Dated spelling of jail.", "jail"),
        ("Obsolete form of bank", "bank"),
        ("Misspelling of separate.", "separate"),
        ("Alternative letter-case form of CAT: Abbreviation of computed axial tomography.", "CAT"),
        ("Non-Oxford British English standard spelling of organize.", "organize"),
        ("A form of government in which the people rule.", None),  # not a variant gloss
        ("Synonym of prostitute.", None),
        ("Abbreviation of catapult.", None),
        (None, None),
    ],
)
def test_variant_target(gloss: str | None, target: str | None) -> None:
    assert variant_target(gloss) == target


def test_normalise_lemma_keeps_case_and_maps_apostrophes() -> None:
    m = CFG["lemma_char_map"]
    assert normalise_lemma("Polish", m) == "Polish"
    assert normalise_lemma("polish", m) == "polish"
    assert normalise_lemma("o’clock", m) == "o'clock"
    assert normalise_lemma("  hot   dog ", m) == "hot dog"
    assert normalise_lemma("café", m) == "café"  # NFC


def test_to_upos() -> None:
    assert to_upos("wiktionary", "noun", "bank", [], "{}", CFG) == "NOUN"
    assert to_upos("wiktionary", "name", "Paris", [], "{}", CFG) == "PROPN"
    assert to_upos("wiktionary", "conj", "and", [], "{}", CFG) == "CCONJ"
    assert to_upos("wiktionary", "conj", "because", [], "{}", CFG) == "SCONJ"
    assert to_upos("wiktionary", "romanization", "x", [], "{}", CFG) is None
    assert to_upos("oewn", "s", "big", [], "{}", CFG) == "ADJ"
    assert to_upos("oewn", "n", "Paris", ["oewn:namenet"], "{}", CFG) == "PROPN"
    assert to_upos("oewn", "n", "Paris", [], '{"synset_relations": {"instance_hypernym": ["x"]}}', CFG) == "PROPN"
    assert to_upos("oewn", "n", "bank", [], "{}", CFG) == "NOUN"


def test_normalise_end_to_end(tmp_path: Path, wikt_dump: Path, oewn_files: tuple[Path, Path]) -> None:
    w = tmp_path / "w.parquet"
    o = tmp_path / "o.parquet"
    raw = tmp_path / "raw.parquet"
    wiktionary.run({"stage": "t", "input": str(wikt_dump), "output": str(w), "lang_code": "en",
                    "source_version": "v", "drop_form_tags": ["table-tags"]})
    oewn.run({"stage": "t", "input": str(oewn_files[0]), "output": str(o), "source_version": "v"})
    combine.run({"stage": "t", "inputs": [str(w), str(o)], "output": str(raw)})
    outs = {k: str(tmp_path / f"{k}.parquet") for k in ("senses", "forms", "variants", "dedupe_map")}
    cfg = {**CFG, "input": str(raw), "outputs": outs}
    stats = normalise.run(cfg)
    senses = pl.read_parquet(outs["senses"])
    forms = pl.read_parquet(outs["forms"])
    variants = pl.read_parquet(outs["variants"])
    lemmas = set(senses["lemma"])
    assert "NASA" in lemmas  # abbreviation-type alt-of stays a sense
    assert "colour" not in lemmas and "gaol" not in lemmas  # variants
    assert "banks" not in lemmas  # form-of
    assert stats["drop:no_gloss"] == 1
    assert senses.filter(pl.col("lemma") == "NASA")["is_abbrev"].to_list() == [True]
    v = dict(zip(variants["variant"], variants["target"], strict=True))
    assert v["colour"] == "color" and v["gaol"] == "jail" and v["banck"] == "bank"
    f = forms.select("lemma", "form", "origin").rows()
    assert ("bank", "banks", "wikt_entry") in f
    assert ("bank", "banks", "wikt_form_of") in f
    assert ("run", "ran", "oewn_form") in f
    upos = dict(zip(senses["lemma"], senses["upos"], strict=False))
    assert upos["and"] == "CCONJ" and upos["because"] == "SCONJ"
