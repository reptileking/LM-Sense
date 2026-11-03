"""Phase 3 codec: hand-written sequences decode to exact strings (spec 3.1); the baseline encoder
round-trips text through a tiny inventory (spec 3.2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest

from senselm.codec import morph
from senselm.codec.decoder import Decoder
from senselm.codec.ortho import Spacer, case_for, title_case, whitespace_tokens
from senselm.codec.tables import Lexicon, Structural, Variants
from senselm.inventory.ids import Layout

REGISTRY = Path(__file__).resolve().parents[1] / "registry" / "structural_registry.parquet"
LEMMAS: list[tuple[str, list[str]]] = [  # lemma, its senses' UPOS in sense-index order
    ("the", ["DET"]),
    ("scientist", ["NOUN"]),
    ("be", ["VERB"]),
    ("run", ["VERB", "NOUN"]),
    ("photosynthesis", ["NOUN"]),
    ("experiment", ["NOUN"]),
    ("in", ["ADP"]),
    ("New York", ["PROPN"]),
    ("mouse", ["NOUN", "NOUN"]),
    ("I", ["PRON"]),
    ("they", ["PRON"]),
    ("do", ["VERB"]),
    ("can", ["VERB", "NOUN"]),
    ("will", ["VERB"]),
    ("dog", ["NOUN"]),
    ("bone", ["NOUN"]),
    ("travel", ["VERB"]),
    ("have", ["VERB"]),
    ("it", ["PRON"]),
    ("this", ["DET"]),
    ("go", ["VERB"]),
    ("lab", ["NOUN"]),
    ("sure", ["ADJ"]),
    ("good", ["ADJ"]),
    ("physical therapy", ["NOUN"]),
    ("and", ["CCONJ"]),
    ("let", ["VERB"]),
    ("us", ["PRON"]),
    ("dictionary", ["NOUN"]),
    ("medium", ["NOUN"]),
]


@pytest.fixture(scope="module")
def tables(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    lay = Layout()
    rows = []
    for lid, (lemma, poses) in enumerate(LEMMAS, start=1):
        for si, upos in enumerate(poses):
            rows.append(
                {"token_id": lay.pack(lid, si), "lemma": lemma, "upos": upos, "sense_index": si}
            )
    inv = pl.DataFrame(rows, schema_overrides={"token_id": pl.UInt32, "sense_index": pl.UInt16})
    d = tmp_path_factory.mktemp("codec")
    inv.write_parquet(d / "inv.parquet")
    exc = pl.DataFrame(
        {
            "lemma": ["travel", "medium", "medium"],
            "upos": ["VERB", "NOUN", "NOUN"],
            "bundle": ["PST", "PL", "PL"],
            "variant": ["uk", "", "other"],
            "form": ["travelled", "media", "mediums"],
        }
    )
    var = pl.DataFrame(
        {
            "target": ["dictionary"],
            "upos": ["NOUN"],
            "kind": ["archaic"],
            "variant": ["dictionnary"],
        }
    )
    return {
        "structural": Structural.load(REGISTRY),
        "lexicon": Lexicon(d / "inv.parquet"),
        "paradigms": morph.Paradigms(exc),
        "variants": Variants(var),
    }


def seq(t: dict[str, Any], *names: str) -> list[int]:
    """Build ids from names: '<cat:name>' structural tokens, 'lemma/UPOS' sense tokens (MFS)."""
    out = []
    for n in names:
        if n.startswith("<"):
            out.append(t["structural"][n.strip("<>")])
        else:
            lemma, upos = n.rsplit("/", 1)
            tid = t["lexicon"].mfs(lemma, upos)
            assert tid is not None, n
            out.append(tid)
    return out


CASES: list[tuple[list[str], str]] = [
    (
        [
            "<case:cap>",
            "the/DET",
            "scientist/NOUN",
            "<morph:PL>",
            "be/VERB",
            "<morph:PST>",
            "<morph:PL>",
            "<morph:NEG>",
            "run/VERB",
            "<morph:V.PTCP>",
            "photosynthesis/NOUN",
            "experiment/NOUN",
            "<morph:PL>",
            "in/ADP",
            "New York/PROPN",
            "<punct:.>",
        ],
        "The scientists weren't running photosynthesis experiments in New York.",
    ),
    (["I/PRON", "can/VERB", "<morph:NEG>", "go/VERB", "<punct:!>"], "I can't go!"),
    (["they/PRON", "will/VERB", "<morph:NEG>", "go/VERB"], "they won't go"),
    (["they/PRON", "<morph:ACC>", "do/VERB", "<morph:3>", "<morph:NEG>"], "them doesn't"),
    (["it/PRON", "<space:clitic>", "be/VERB", "<morph:3>", "sure/ADJ"], "it's sure"),
    (["let/VERB", "<space:clitic>", "us/PRON", "go/VERB"], "let's go"),
    (["<orth:apos_curly>", "the/DET", "dog/NOUN", "<morph:GEN>", "bone/NOUN"], "the dog’s bone"),
    (
        ["the/DET", "dog/NOUN", "<morph:PL>", "<morph:GEN>", "bone/NOUN", "<morph:PL>"],
        "the dogs' bones",
    ),
    (["mouse/NOUN", "<morph:PL>", "<case:allcaps>", "good/ADJ", "<morph:SPRL>"], "mice BEST"),
    (
        ["<variant:uk>", "travel/VERB", "<morph:PST>", "travel/VERB", "<morph:PST>"],
        "travelled traveled",
    ),
    (
        ["medium/NOUN", "<morph:PL>", "<variant:other>", "medium/NOUN", "<morph:PL>"],
        "media mediums",
    ),
    (
        ["<variant:archaic>", "dictionary/NOUN", "<case:title>", "physical therapy/NOUN"],
        "dictionnary Physical Therapy",
    ),
    (
        [
            "<punct:(>",
            "<digit:3>",
            "<punct:.>",
            "<digit:1>",
            "<digit:4>",
            "<punct:)>",
            "<punct:,>",
            "<digit:1>",
            "<punct:,>",
            "<digit:0>",
            "<digit:0>",
            "<digit:0>",
        ],
        "(3.14), 1,000",
    ),
    (
        ['<punct:">', "go/VERB", '<punct:">', "and/CCONJ", '<punct:">', "go/VERB", '<punct:">'],
        '"go" and "go"',
    ),
    (
        [
            "go/VERB",
            "<space:newline>",
            "<space:indent>",
            "go/VERB",
            "<space:nospace>",
            "<punct:->",
            "go/VERB",
            "<space:paragraph>",
        ],
        "go\n    go-go\n\n",
    ),
    (
        [
            "<byte:bytes_start>",
            "<byte:0x4F>",
            "<byte:0x6B>",
            "<byte:bytes_end>",
            "<morph:GEN>",
            "dog/NOUN",
        ],
        "Ok's dog",
    ),
    (["<special:bos>", "<prag:hedge>", "go/VERB", "<special:eos>"], "go"),
]


@pytest.mark.parametrize(("names", "expected"), CASES)
def test_decoder_hand_written(tables: dict[str, Any], names: list[str], expected: str) -> None:
    dec = Decoder.from_tables(tables)
    assert dec.decode(seq(tables, *names)) == expected


def test_decoder_degrades_gracefully(tables: dict[str, Any]) -> None:
    dec = Decoder.from_tables(tables)
    s = tables["structural"]
    junk = [s["morph:PL"], 123456789, s["byte:bytes_start"], s["byte:0x41"]]
    assert dec.decode(junk) == "A"  # stray morph, unknown id, unterminated byte span


def test_auto_cap(tables: dict[str, Any]) -> None:
    dec = Decoder.from_tables(tables, auto_cap=True)
    ids = seq(
        tables,
        "the/DET",
        "dog/NOUN",
        "<punct:.>",
        "the/DET",
        "dog/NOUN",
        "<punct:.>",
        "<case:lower>",
        "go/VERB",
    )
    assert dec.decode(ids) == "The dog. The dog. go"


def test_ortho_helpers() -> None:
    assert whitespace_tokens("") == ["nospace"]
    assert whitespace_tokens("\n\n    x") is None
    assert whitespace_tokens("\n    \t") == ["newline", "indent", "tab"]
    assert title_case("one-legged man") == "One-Legged Man"
    assert case_for("NEW YORK", "New York", False) == "allcaps"
    assert case_for("The", "the", True) == ""
    assert case_for("the", "the", True) == "lower"
    sp = Spacer()
    seps = []
    for kind, text in [
        ("word", "x"),
        ("punct", "("),
        ("digit", "1"),
        ("punct", ","),
        ("digit", "2"),
        ("punct", ")"),
        ("word", "y"),
    ]:
        seps.append(sp.sep(kind, text))
        sp.push(kind, text)
    assert seps == ["", " ", "", "", "", "", " "]


def test_paradigm_rules() -> None:
    par = morph.Paradigms()
    assert par.realize("carry the can", "VERB", ("PST",)) == "carried the can"
    assert par.realize("Lake Superior", "PROPN", ("PL",)) == "Lake Superiors"
    assert par.realize("be", "VERB", ("PST", "PL")) == "were"
    assert par.realize("they", "PRON", ("GEN",)) == "their"
    assert morph.tags_to_bundle(["UK", "participle", "past"], "VERB") == (("PST", "V.PTCP"), "uk")
    assert morph.tags_to_bundle(["archaic", "past"], "VERB") is None


@pytest.fixture(scope="module")
def encoder(tables: dict[str, Any]) -> Any:
    spacy = pytest.importorskip("spacy")
    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    except OSError:
        pytest.skip("en_core_web_sm not installed")
    from senselm.codec.encoder import Encoder

    return Encoder.from_tables(tables, nlp)


TEXTS = [
    "The scientists weren't running photosynthesis experiments in New York.",
    "I can't go; they won't go, and it's sure.",
    "The dog’s bones and the dogs' bones: (3.14), 1,000 mice!",
    '  Okonkwo travelled.\n\n    Let\'s go—"go" ',
    "",
    "THE SCIENTISTS. Physical Therapy",
]


@pytest.mark.parametrize("text", TEXTS)
def test_encoder_round_trip(tables: dict[str, Any], encoder: Any, text: str) -> None:
    dec = Decoder.from_tables(tables)
    assert dec.decode(encoder.encode(text)) == text


def test_encoder_choices(tables: dict[str, Any], encoder: Any) -> None:
    s, lex = tables["structural"], tables["lexicon"]
    ids = encoder.encode("they weren't")
    assert ids == [
        lex.mfs("they", "PRON"),
        lex.mfs("be", "VERB"),
        s["morph:PST"],
        s["morph:PL"],
        s["morph:NEG"],
    ]
    ids = encoder.encode("Okonkwo")  # not in the inventory: one byte span
    assert ids[0] == s["byte:bytes_start"] and ids[-1] == s["byte:bytes_end"] and len(ids) == 9
    ids = encoder.encode("mice")
    assert ids == [lex.mfs("mouse", "NOUN"), s["morph:PL"]]
