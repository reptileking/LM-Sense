from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

WIKT_ENTRIES = [
    {
        "word": "bank", "pos": "noun", "lang": "English", "lang_code": "en", "etymology_number": "1",
        "forms": [{"form": "banks", "tags": ["plural"]}, {"form": "banck", "tags": ["alternative", "obsolete"]},
                  {"form": "no-table-tags", "tags": ["table-tags"]}],
        "senses": [
            {"id": "en-bank-en-noun-AAAA", "glosses": ["An institution where one can place and borrow money."],
             "topics": ["business", "finance"], "tags": ["countable"],
             "examples": [{"text": "I went to the bank.", "type": "example"},
                          {"text": "A long quotation.", "type": "quotation", "ref": "1900"}]},
            {"id": "en-bank-en-noun-AAAA", "glosses": ["An institution where one can place and borrow money.",
                                                        "A branch office of such an institution."]},
        ],
    },
    {
        "word": "bank", "pos": "noun", "lang": "English", "lang_code": "en", "etymology_number": 2,
        "senses": [{"id": "en-bank-en-noun-BBBB", "glosses": ["The edge of a river."],
                    "topics": ["geography", "natural-sciences", "sciences"]}],
    },
    {
        "word": "banks", "pos": "noun", "lang": "English", "lang_code": "en",
        "senses": [{"id": "en-banks-en-noun-CCCC", "glosses": ["plural of bank"], "tags": ["form-of", "plural"],
                    "form_of": [{"word": "bank"}]}],
    },
    {
        "word": "NASA", "pos": "name", "lang": "English", "lang_code": "en",
        "senses": [{"id": "en-NASA-en-name-DDDD", "glosses": ["Acronym of National Aeronautics and Space Administration."],
                    "tags": ["abbreviation", "acronym", "alt-of"],
                    "alt_of": [{"word": "National Aeronautics and Space Administration"}]}],
    },
    {
        "word": "colour", "pos": "noun", "lang": "English", "lang_code": "en",
        "senses": [{"id": "en-colour-en-noun-EEEE", "glosses": ["Alternative spelling of color."],
                    "tags": ["alt-of", "alternative"], "alt_of": [{"word": "color"}]}],
    },
    {
        "word": "gaol", "pos": "noun", "lang": "English", "lang_code": "en",
        "senses": [{"id": "en-gaol-en-noun-FFFF", "glosses": ["Dated spelling of jail."], "tags": ["UK"]}],
    },
    {
        "word": "and", "pos": "conj", "lang": "English", "lang_code": "en",
        "senses": [{"id": "en-and-en-conj-GGGG", "glosses": ["As well as."]}],
    },
    {
        "word": "because", "pos": "conj", "lang": "English", "lang_code": "en",
        "senses": [{"id": "en-because-en-conj-HHHH", "glosses": ["By or for the cause that."]},
                   {"id": "en-because-en-conj-IIII", "tags": ["no-gloss"]}],
    },
    {"word": "Hund", "pos": "noun", "lang": "German", "lang_code": "de",
     "senses": [{"id": "x", "glosses": ["dog"]}]},
]

OEWN_XML = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE LexicalResource SYSTEM "http://globalwordnet.github.io/schemas/WN-LMF-1.3.dtd">
<LexicalResource xmlns:dc="https://globalwordnet.github.io/schemas/dc/">
  <Lexicon id="oewn" label="Open English Wordnet" language="en" email="x@y" license="https://creativecommons.org/licenses/by/4.0" version="2025+" url="https://example.org">
    <LexicalEntry id="oewn-bank-n">
      <Lemma writtenForm="bank" partOfSpeech="n"/>
      <Sense id="oewn-bank__1.17.01.." synset="oewn-09236472-n"/>
      <Sense id="oewn-bank__1.14.00.." synset="oewn-08437235-n">
        <SenseRelation relType="exemplifies" target="oewn-american_spelling__1.10.01.."/>
      </Sense>
    </LexicalEntry>
    <LexicalEntry id="oewn-Paris-n">
      <Lemma writtenForm="Paris" partOfSpeech="n"/>
      <Sense id="oewn-paris__1.15.00.." synset="oewn-08945277-n"/>
    </LexicalEntry>
    <LexicalEntry id="oewn-geography-n">
      <Lemma writtenForm="geography" partOfSpeech="n"/>
      <Sense id="oewn-geography__1.09.00.." synset="oewn-06128570-n"/>
    </LexicalEntry>
    <LexicalEntry id="oewn-run-v">
      <Lemma writtenForm="run" partOfSpeech="v"/>
      <Form writtenForm="ran"/>
      <Sense id="oewn-run__2.38.00.." synset="oewn-01928838-v"/>
    </LexicalEntry>
    <Synset id="oewn-09236472-n" ili="i1" members="oewn-bank-n" partOfSpeech="n" lexfile="noun.object">
      <Definition>sloping land (especially the slope beside a body of water)</Definition>
      <SynsetRelation relType="hypernym" target="oewn-06128570-n"/>
      <SynsetRelation relType="domain_topic" target="oewn-06128570-n"/>
      <Example>they pulled the canoe up on the bank</Example>
    </Synset>
    <Synset id="oewn-08437235-n" ili="i2" members="oewn-bank-n" partOfSpeech="n" lexfile="noun.group">
      <Definition>a financial institution that accepts deposits</Definition>
    </Synset>
    <Synset id="oewn-08945277-n" ili="i3" members="oewn-Paris-n" partOfSpeech="n" lexfile="noun.location">
      <Definition>the capital and largest city of France</Definition>
      <SynsetRelation relType="instance_hypernym" target="oewn-06128570-n"/>
    </Synset>
    <Synset id="oewn-06128570-n" ili="i4" members="oewn-geography-n" partOfSpeech="n" lexfile="noun.cognition">
      <Definition>study of the earth's surface</Definition>
    </Synset>
    <Synset id="oewn-01928838-v" ili="i5" members="oewn-run-v" partOfSpeech="v" lexfile="verb.motion">
      <Definition>move fast by using one's feet</Definition>
    </Synset>
  </Lexicon>
</LexicalResource>
"""

OEWN_BASE_XML = OEWN_XML.replace(
    """      <Sense id="oewn-paris__1.15.00.." synset="oewn-08945277-n"/>\n""", ""
)


@pytest.fixture()
def wikt_dump(tmp_path: Path) -> Path:
    p = tmp_path / "raw" / "wikt.jsonl.gz"
    p.parent.mkdir(parents=True)
    with gzip.open(p, "wt", encoding="utf-8") as fh:
        for e in WIKT_ENTRIES:
            fh.write(json.dumps(e) + "\n")
    return p


@pytest.fixture()
def oewn_files(tmp_path: Path) -> tuple[Path, Path]:
    d = tmp_path / "raw_oewn"
    d.mkdir()
    plus, base = d / "plus.xml.gz", d / "base.xml.gz"
    with gzip.open(plus, "wt", encoding="utf-8") as fh:
        fh.write(OEWN_XML)
    with gzip.open(base, "wt", encoding="utf-8") as fh:
        fh.write(OEWN_BASE_XML)
    return plus, base
