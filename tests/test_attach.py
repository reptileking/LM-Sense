"""Stage-2 attachment rules (attach.py) on a synthetic inventory, with stage 1 and the embedder
replaced by fixtures: one-to-one per source, cross-references, the case rule and the
kingdom rule for taxa."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from senselm.inventory import attach

TOPICS = ["rodent", "device", "genus", "plant", "bird"]


def fake_embed(
    texts: list[str], cfg: dict[str, Any], embedder: Any = None
) -> tuple[np.ndarray, dict[str, Any]]:
    out = np.zeros((len(texts), len(TOPICS) + 1), dtype=np.float32)
    for i, t in enumerate(texts):
        for j, k in enumerate(TOPICS):
            out[i, j] = float(k in t)
        out[i, -1] = 0.05  # keeps unrelated texts at a small positive cosine
        out[i] /= np.linalg.norm(out[i])
    return out, {}


def row(
    source: str,
    sid: str,
    lemma: str,
    upos: str,
    gloss: str,
    *,
    rel: dict[str, Any] | None = None,
    lexfile: str | None = None,
) -> dict[str, Any]:
    return {
        "source": source,
        "source_id": sid,
        "lemma": lemma,
        "upos": upos,
        "gloss": gloss,
        "gloss_path": [],
        "examples": [],
        "example_types": [],
        "relations": json.dumps(rel or {}),
        "entry_key": sid,
        "line_no": 0,
        "sense_rank": 0,
        "lexfile": lexfile,
    }


def test_attach_rules(tmp_path: Path, monkeypatch: Any) -> None:
    senses = pl.DataFrame(
        [
            row("oewn", "o-mouse-1", "mouse", "NOUN", "small rodent", lexfile="noun.animal"),
            row(
                "oewn",
                "o-mouse-2",
                "mouse",
                "NOUN",
                "hand-held pointing device",
                lexfile="noun.artifact",
            ),
            row("wiktionary", "w-mouse-1", "mouse", "NOUN", "a small rodent (genus Mus)"),
            row(
                "wiktionary",
                "w-mus",
                "Mus",
                "PROPN",
                "the genus of rodent mice",
                rel={"wikidata": ["Q39275"]},
            ),
            row(
                "oewn", "o-ficus", "Ficus", "NOUN", "genus of plant: figs", lexfile="noun.animal"
            ),  # (sic) wrong file
            row("wikidata_lexemes", "L1-S1", "mouse", "NOUN", "rodent"),
            row(
                "wikidata_lexemes", "L1-S2", "mouse", "NOUN", "rodent, small"
            ),  # same cluster: one-to-one
            row("wikidata_lexemes", "L1-S3", "mouse", "NOUN", "pointing device"),
            row(
                "wikidata_items",
                "Q39275",
                "Mus",
                "PROPN",
                "genus of rodents",
                rel={"xrefs": {"P685": ["10088"]}},
            ),
            row(
                "ncbi_taxonomy",
                "taxon:10088",
                "Mus",
                "PROPN",
                "unrelated words",
                rel={"lineage": {"kingdom": "Metazoa"}},
            ),  # attaches through the item's P685, not cosine
            row(
                "ncbi_taxonomy",
                "taxon:3493",
                "Ficus",
                "PROPN",
                "genus of plant",
                rel={"lineage": {"kingdom": "Viridiplantae"}},
            ),  # kingdom rule: noun.animal cluster refused
            row(
                "ncbi_taxonomy", "taxon:1", "Mouse", "PROPN", "rodent genus"
            ),  # no case fallback for taxa
            row("mesh", "D051379", "mouse", "NOUN", "rodent of the genus Mus"),
            row(
                "wiktionary", "w-aspirin", "aspirin", "NOUN", "a drug", rel={"wikidata": ["Q18216"]}
            ),
            row(
                "chebi", "CHEBI:15365", "aspirin", "NOUN", "an acetate ester"
            ),  # cosine too low: bridge
        ],
        schema_overrides={
            "gloss_path": pl.List(pl.String),
            "examples": pl.List(pl.String),
            "example_types": pl.List(pl.String),
            "lexfile": pl.String,
            "line_no": pl.Int64,
            "sense_rank": pl.Int64,
        },
    )
    senses_path = tmp_path / "senses.parquet"
    senses.write_parquet(senses_path)
    pairs_path = tmp_path / "pairs.parquet"
    pl.DataFrame({"x": [1]}).write_parquet(pairs_path)
    stage1 = pl.DataFrame(
        {
            "source_id": ["o-mouse-1", "w-mouse-1", "o-mouse-2", "w-mus", "o-ficus", "w-aspirin"],
            "cluster": ["o-mouse-1", "o-mouse-1", "o-mouse-2", "w-mus", "o-ficus", "w-aspirin"],
        }
    )
    # items superset: Q18216 (aspirin) is a chemical compound, not selected as a sense, but its
    # ChEBI cross-reference still links the Wiktionary sense to the ChEBI entity
    bridge = tmp_path / "superset.parquet"
    pl.DataFrame({"qid": ["Q18216", "Q1"], "xrefs": ['{"P683": ["15365"]}', "{}"]}).write_parquet(
        bridge
    )
    monkeypatch.setattr(attach, "stage1", lambda cfg, s: stage1)
    monkeypatch.setattr(attach, "embed_texts", fake_embed)
    monkeypatch.setattr(attach, "Embedder", lambda cfg: None)
    cfg = {
        "senses": str(senses_path),
        "pairs": str(pairs_path),
        "output": str(tmp_path / "att.parquet"),
        "embedding": {"text": {}},
        "pos_classes": {"NOUN": "N", "PROPN": "N"},
        "source_order": ["wikidata_lexemes", "wikidata_items", "mesh", "chebi", "ncbi_taxonomy"],
        "bridge": str(bridge),
        "thresholds": {"default": 0.6},
        "no_case_fallback": ["ncbi_taxonomy"],
        "taxon_lexfiles": {
            "Metazoa": ["noun.animal"],
            "Viridiplantae": ["noun.plant"],
            "default": ["noun.animal", "noun.plant"],
        },
    }
    stats = attach.run(cfg)
    att = {
        r["source_id"]: r for r in pl.read_parquet(tmp_path / "att.parquet").iter_rows(named=True)
    }
    assert att["L1-S1"]["cluster"] == "o-mouse-1" and att["L1-S1"]["via"] == "cosine"
    assert att["L1-S2"]["via"] == "new"  # o-mouse-1 already took a sense of this source
    assert att["L1-S3"]["cluster"] == "o-mouse-2"
    assert (
        att["Q39275"]["cluster"] == "w-mus" and att["Q39275"]["via"] == "xref"
    )  # Wiktionary's Q-id
    assert (
        att["taxon:10088"]["cluster"] == "w-mus" and att["taxon:10088"]["via"] == "xref"
    )  # item P685
    assert att["taxon:3493"]["via"] == "new" and stats["ncbi_taxonomy:kingdom_rejected"] == 1
    assert att["taxon:1"]["via"] == "new"  # "Mouse" never falls back to "mouse"
    assert att["D051379"]["cluster"] == "o-mouse-1"  # one-to-one is per source
    assert att["CHEBI:15365"]["cluster"] == "w-aspirin" and att["CHEBI:15365"]["via"] == "xref"
