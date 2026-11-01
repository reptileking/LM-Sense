"""Ingest Open English WordNet (WN-LMF XML) into the unified raw sense schema.

``python -m senselm.ingest.oewn --config configs/ingest_oewn.yaml``

One row per (lexical entry, sense) = per synset member. The XML is streamed with
``lxml.etree.iterparse`` (DTD loading, entity resolution and network access disabled) in two
passes:

1. synsets (definition, examples, relations, supersense / lexfile) + entry-id → lemma map
   (needed to name relation targets such as domain topics);
2. lexical entries → rows, joined to the synset info from pass 1.

The OEWN 2025 base edition is scanned once to flag senses that only exist in the 2025+
edition (proper nouns imported from Open English Namenet): those rows get the tag
``oewn:namenet`` so the subset can be excluded later.

Sense order inside a lexical entry is preserved in ``sense_rank`` (it follows Princeton
WordNet sense numbering, which was ordered by SemCor tag frequency).
"""

from __future__ import annotations

import gzip
import logging
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import orjson
from lxml import etree

from senselm.common import BatchWriter, Provenance, load_config, resolve, stage_args
from senselm.hardware import resolve_auto
from senselm.ingest.manifest import source_version
from senselm.ingest.schema import SENSE_SCHEMA, make_row

LOG = logging.getLogger("senselm.ingest.oewn")
SOURCE = "oewn"
DC_TYPE = "{https://globalwordnet.github.io/schemas/dc/}type"
DC_SOURCE = "{https://globalwordnet.github.io/schemas/dc/}source"


def iterparse(path: Path, tags: tuple[str, ...]) -> Iterator[etree._Element]:
    """Stream elements with the given tags, freeing memory as we go."""
    with gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb") as fh:
        ctx = etree.iterparse(
            fh,
            events=("end",),
            tag=tags,
            load_dtd=False,
            no_network=True,
            resolve_entities=False,
            huge_tree=True,
        )
        for _, elem in ctx:
            yield elem
            elem.clear(keep_tail=False)
            parent = elem.getparent()
            if parent is not None:
                while elem.getprevious() is not None:
                    del parent[0]


@dataclass
class SynsetInfo:
    pos: str
    lexfile: str | None
    ili: str | None
    members: list[str]
    definitions: list[str]
    examples: list[str]
    relations: dict[str, list[str]] = field(default_factory=dict)


def sense_label(sense_id: str) -> str:
    """'oewn-american_spelling__1.10.01..' -> 'american_spelling'."""
    s = sense_id.removeprefix("oewn-")
    return s.split("__", 1)[0]


def read_synsets(path: Path) -> tuple[dict[str, SynsetInfo], dict[str, str]]:
    synsets: dict[str, SynsetInfo] = {}
    entry_lemma: dict[str, str] = {}
    for elem in iterparse(path, ("Synset", "LexicalEntry")):
        if elem.tag == "LexicalEntry":
            lemma = elem.find("Lemma")
            if lemma is not None:
                entry_lemma[elem.get("id")] = lemma.get("writtenForm")
            continue
        rels: dict[str, list[str]] = defaultdict(list)
        for r in elem.iterfind("SynsetRelation"):
            rels[r.get("relType")].append(r.get("target"))
        synsets[elem.get("id")] = SynsetInfo(
            pos=elem.get("partOfSpeech"),
            lexfile=elem.get("lexfile"),
            ili=elem.get("ili") or None,
            members=(elem.get("members") or "").split(),
            definitions=[d.text.strip() for d in elem.iterfind("Definition") if d.text],
            examples=[e.text.strip() for e in elem.iterfind("Example") if e.text and e.text.strip()],
            relations=dict(rels),
        )
    return synsets, entry_lemma


def read_sense_ids(path: Path) -> set[str]:
    return {e.get("id") for e in iterparse(path, ("Sense",))}


def synset_name(sid: str, synsets: dict[str, SynsetInfo], entry_lemma: dict[str, str]) -> str | None:
    info = synsets.get(sid)
    if not info or not info.members:
        return None
    return entry_lemma.get(info.members[0])


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    src = resolve(cfg["input"])
    base = resolve(cfg["base_input"]) if cfg.get("base_input") else None
    version = source_version(cfg, SOURCE)
    prov = Provenance(stage="ingest.oewn", config=cfg, source_versions={SOURCE: version})
    prov.add_input(src)
    base_ids: set[str] | None = None
    if base is not None:
        prov.add_input(base)
        base_ids = read_sense_ids(base)
        LOG.info("base edition: %d sense ids", len(base_ids))

    synsets, entry_lemma = read_synsets(src)
    LOG.info("pass 1: %d synsets, %d entries", len(synsets), len(entry_lemma))

    stats: Counter[str] = Counter()
    batch_rows = int(resolve_auto(cfg.get("batch_rows", "auto"), "parquet_batch_rows"))
    out = BatchWriter(cfg["output"], SENSE_SCHEMA, prov, batch_rows=batch_rows)
    try:
        for line_no, entry in enumerate(iterparse(src, ("LexicalEntry",))):
            lemma_el = entry.find("Lemma")
            lemma = lemma_el.get("writtenForm")
            pos = lemma_el.get("partOfSpeech")
            forms = [{"form": f.get("writtenForm"), "tags": []} for f in entry.iterfind("Form")]
            for rank, sense in enumerate(entry.iterfind("Sense")):
                sid = sense.get("id")
                ss_id = sense.get("synset")
                info = synsets[ss_id]
                sense_rels: dict[str, list[str]] = defaultdict(list)
                for r in sense.iterfind("SenseRelation"):
                    rt = r.get("relType")
                    if rt == "other" and r.get(DC_TYPE):
                        rt = f"other:{r.get(DC_TYPE)}"
                    sense_rels[rt].append(r.get("target"))
                tags = sorted({f"usage:{sense_label(t)}" for t in sense_rels.get("exemplifies", [])})
                regions = [
                    synset_name(t, synsets, entry_lemma)
                    for t in info.relations.get("domain_region", [])
                ]
                tags += [f"region:{r}" for r in regions if r]
                if base_ids is not None and sid not in base_ids:
                    tags.append("oewn:namenet")
                    stats["namenet_senses"] += 1
                topics = [
                    n
                    for n in (
                        synset_name(t, synsets, entry_lemma)
                        for t in info.relations.get("domain_topic", [])
                    )
                    if n
                ]
                sense_examples = [
                    e.text.strip() for e in sense.iterfind("Example") if e.text and e.text.strip()
                ]
                examples = sense_examples + info.examples
                hypernyms = info.relations.get("hypernym", []) + info.relations.get(
                    "instance_hypernym", []
                )
                rel = {
                    "synset": ss_id,
                    "ili": info.ili,
                    "members": [entry_lemma.get(m, m) for m in info.members],
                    "synset_relations": info.relations,
                    "sense_relations": dict(sense_rels),
                    "hypernym_names": [
                        n for n in (synset_name(h, synsets, entry_lemma) for h in hypernyms) if n
                    ],
                }
                if sense.get("subcat"):
                    rel["subcat"] = sense.get("subcat").split()
                if sense.get("adjposition"):
                    rel["adjposition"] = sense.get("adjposition")
                stats["rows"] += 1
                if examples:
                    stats["rows_with_examples"] += 1
                out.write(
                    make_row(
                        source=SOURCE,
                        source_id=sid,
                        lemma=lemma,
                        pos=pos,
                        gloss="; ".join(info.definitions) or None,
                        examples=examples,
                        tags=tags,
                        domain=topics[0] if topics else None,
                        forms=forms,
                        relations=orjson.dumps(rel, option=orjson.OPT_SORT_KEYS).decode(),
                        source_version=version,
                        kind="sense",
                        example_types=["example"] * len(examples),
                        topics=topics,
                        entry_key=entry.get("id"),
                        line_no=line_no,
                        sense_rank=rank,
                        synset_id=ss_id,
                        lexfile=info.lexfile,
                        hypernyms=hypernyms,
                    )
                )
            stats["entries"] += 1
    except BaseException:
        out.abort()
        raise
    out_stats = dict(sorted(stats.items()))
    out.close(stats=out_stats)
    LOG.info("done: %s", out_stats)
    return out_stats


def main() -> None:
    args = stage_args("Ingest Open English WordNet (WN-LMF XML)")
    run(load_config(args.config))


if __name__ == "__main__":
    main()
