"""Ingest English Wikidata lexemes (CC0) into the unified raw sense schema.

``python -m senselm.ingest.wikidata_lexemes --config configs/ingest_wikidata_lexemes.yaml``

The lexemes dump is one JSON array with one entity per line; it is streamed line by line.
Only lexemes whose language is English (Q1860) are kept. One row per lexeme sense:

* ``lemma`` = the ``en`` lemma (else the first English variant such as ``en-gb``);
* ``pos`` = the lexical-category Q-id (mapped to UPOS in normalise; table in the config);
* ``gloss`` = the English gloss; senses without one are ``no_gloss``;
* ``examples`` = usage examples (P5831) whose "subject sense" qualifier (P6072) is this sense,
  or all of the lexeme's examples when it has a single sense;
* ``forms`` = the lexeme's forms with grammatical features mapped to the tag vocabulary the
  Wiktionary forms use (``plural``, ``past``, ``participle`` …; unknown features keep their
  Q-id) — this is CC0 inflection data for the decoder;
* ``relations`` = linked item (P5137 "item for this sense"), synonyms (P5973), antonyms
  (P5974), hyperonyms (P6593), field of usage (P9488), plus the lexeme id.

Lexemes without senses still yield one ``no_gloss`` row so their forms reach the forms table.
Also writes the set of items linked from English senses (``linked_items_output``), which
defines part of the Wikidata items filter (ingest/wikidata_items.py).
"""

from __future__ import annotations

import gzip
import logging
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import orjson
import polars as pl

from senselm.common import BatchWriter, Provenance, load_config, resolve, stage_args, write_table
from senselm.hardware import resolve_auto
from senselm.ingest.manifest import source_version
from senselm.ingest.schema import SENSE_SCHEMA, make_row

LOG = logging.getLogger("senselm.ingest.wikidata_lexemes")
SOURCE = "wikidata_lexemes"


def iter_entities(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Stream the dump's JSON array: one entity per line, trailing commas stripped."""
    with gzip.open(path, "rb") as fh:
        for line_no, line in enumerate(fh):
            line = line.strip()
            if line in (b"[", b"]", b""):
                continue
            yield line_no, orjson.loads(line.rstrip(b","))


def _value_ids(claims: dict[str, Any], prop: str) -> list[str]:
    out = []
    for st in claims.get(prop, []):
        v = (st.get("mainsnak", {}).get("datavalue") or {}).get("value")
        if isinstance(v, dict) and v.get("id"):
            out.append(v["id"])
    return out


def english(rep: dict[str, Any]) -> str | None:
    if "en" in rep:
        return rep["en"]["value"]
    for k in sorted(rep):
        if k.startswith("en"):
            return rep[k]["value"]
    return None


def usage_examples(lexeme: dict[str, Any]) -> list[tuple[str, list[str]]]:
    """(text, [sense ids it illustrates]) for every English usage example (P5831)."""
    out = []
    for st in lexeme.get("claims", {}).get("P5831", []):
        v = (st.get("mainsnak", {}).get("datavalue") or {}).get("value") or {}
        if v.get("language", "en").startswith("en") and v.get("text"):
            senses = [
                ((q.get("datavalue") or {}).get("value") or {}).get("id")
                for q in st.get("qualifiers", {}).get("P6072", [])
            ]
            out.append((v["text"], [s for s in senses if s]))
    return out


def lexeme_rows(
    line_no: int, lex: dict[str, Any], *, version: str, features: dict[str, list[str]]
) -> Iterator[dict[str, Any]]:
    lemma = english(lex.get("lemmas", {}))
    if not lemma:
        return
    cat = lex.get("lexicalCategory") or "unknown"
    forms = []
    for fm in lex.get("forms", []):
        rep = english(fm.get("representations", {}))
        if not rep:
            continue
        tags: list[str] = []
        for q in fm.get("grammaticalFeatures", []):
            tags.extend(features.get(q, [f"wd:{q}"]))
        forms.append({"form": rep, "tags": sorted(set(tags))})
    examples = usage_examples(lex)
    senses = lex.get("senses", [])
    base = {
        "source": SOURCE,
        "lemma": lemma,
        "pos": cat,
        "forms": forms,
        "source_version": version,
        "entry_key": lex["id"],
        "line_no": line_no,
    }
    if not senses:
        yield make_row(
            **base,
            source_id=lex["id"],
            kind="no_gloss",
            sense_rank=0,
            relations=orjson.dumps({"lexeme": lex["id"]}).decode(),
        )
        return
    for rank, sense in enumerate(senses):
        gloss = (sense.get("glosses", {}).get("en") or {}).get("value")
        claims = sense.get("claims", {})
        exs = [t for t, ids in examples if sense["id"] in ids or (len(senses) == 1 and not ids)]
        rel: dict[str, Any] = {"lexeme": lex["id"]}
        for key, prop in (
            ("item", "P5137"),
            ("synonyms", "P5973"),
            ("antonyms", "P5974"),
            ("hyperonyms", "P6593"),
            ("field_of_usage", "P9488"),
        ):
            ids = _value_ids(claims, prop)
            if ids:
                rel[key] = ids
        yield make_row(
            **base,
            source_id=sense["id"],
            gloss=gloss,
            examples=exs,
            example_types=["example"] * len(exs),
            relations=orjson.dumps(rel, option=orjson.OPT_SORT_KEYS).decode(),
            kind="sense" if gloss else "no_gloss",
            sense_rank=rank,
            targets=rel.get("item", []),
        )


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    src = resolve(cfg["input"])
    version = source_version(cfg, SOURCE)
    prov = Provenance(
        stage="ingest.wikidata_lexemes", config=cfg, source_versions={SOURCE: version}
    )
    prov.add_input(src)
    features: dict[str, list[str]] = cfg["grammatical_features"]
    language = cfg.get("language_qid", "Q1860")
    stats: Counter[str] = Counter()
    links: list[dict[str, str]] = []
    batch_rows = int(resolve_auto(cfg.get("batch_rows", "auto"), "parquet_batch_rows"))
    out = BatchWriter(cfg["output"], SENSE_SCHEMA, prov, batch_rows=batch_rows)
    try:
        for line_no, ent in iter_entities(src):
            stats["lexemes"] += 1
            if ent.get("language") != language:
                continue
            stats["english_lexemes"] += 1
            for row in lexeme_rows(line_no, ent, version=version, features=features):
                stats[f"kind:{row['kind']}"] += 1
                for q in row["targets"]:
                    links.append({"qid": q, "sense_id": row["source_id"], "lemma": row["lemma"]})
                out.write(row)
    except BaseException:
        out.abort()
        raise
    st = dict(sorted(stats.items()))
    out.close(stats=st)
    link_df = pl.DataFrame(
        links, schema={"qid": pl.String, "sense_id": pl.String, "lemma": pl.String}
    )
    write_table(
        link_df.sort("qid", "sense_id").to_arrow(),
        cfg["linked_items_output"],
        prov,
        stats={"links": link_df.height, "items": link_df["qid"].n_unique()},
    )
    LOG.info("done: %s; %d linked items", st, link_df["qid"].n_unique())
    return st


def main() -> None:
    args = stage_args("Ingest English Wikidata lexemes")
    run(load_config(args.config))


if __name__ == "__main__":
    main()
