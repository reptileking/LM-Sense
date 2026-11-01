"""Ingest SNOMED CT (UK edition, RF2 Snapshot) — prepared ahead of the licensed files.

``python -m senselm.ingest.snomed --config configs/ingest_snomed.yaml``

Input: the TRUD release zip(s) or extracted folder(s) of the SNOMED CT UK Edition (RF2). The
reader globs the Snapshot files of every package it finds (International + UK clinical
extension), so it does not depend on folder names:

* ``sct2_Concept_Snapshot*``          id, effectiveTime, active, moduleId, definitionStatusId
* ``sct2_Description_Snapshot-en*``   …, conceptId, languageCode, typeId, term, caseSignificanceId
* ``sct2_TextDefinition_Snapshot-en*`` same columns (typeId = Definition)
* ``der2_cRefset_Language*Snapshot*`` …, refsetId, referencedComponentId, acceptabilityId
* ``sct2_Relationship_Snapshot*``     …, sourceId, destinationId, …, typeId (116680003 = is a)

One sense per active concept: lemma = the *preferred* synonym in the first language refset of
``language_refsets`` that has one (UK clinical → GB English by default); the initial capital is
dropped unless the description is case-sensitive. Gloss = the text definition, else
"<semantic tag>; a kind of <parents>". FSN, other synonyms (alignment-only) and parents go to
``relations``. Every row is tagged ``licence:restricted``: SNOMED CT content may not be
redistributed, so restricted rows are excluded from git-tracked artefacts (D-036).
"""

from __future__ import annotations

import csv
import io
import logging
import re
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import orjson

from senselm.common import BatchWriter, Provenance, load_config, resolve, stage_args
from senselm.ingest.schema import SENSE_SCHEMA, make_row

LOG = logging.getLogger("senselm.ingest.snomed")
SOURCE = "snomed_ct_uk"
FSN = "900000000000003001"
SYNONYM = "900000000000013009"
DEFINITION = "900000000000550004"
PREFERRED = "900000000000548007"
IS_A = "116680003"
CASE_SENSITIVE = "900000000000017005"
TAG_RE = re.compile(r"\(([^()]+)\)\s*$")

csv.field_size_limit(1 << 24)


def _members(inputs: list[Path]) -> Iterator[tuple[str, Any]]:
    """(name, opener) for every file in the given zips / directories."""
    for p in inputs:
        if p.is_dir():
            for f in sorted(p.rglob("*.txt")):
                yield f.name, (lambda f=f: f.open("rb"))
        elif p.suffix == ".zip":
            z = zipfile.ZipFile(p)
            for n in sorted(z.namelist()):
                if n.endswith(".txt"):
                    yield n.rsplit("/", 1)[-1], (lambda n=n, z=z: z.open(n))


def rows(inputs: list[Path], prefix: str) -> Iterator[dict[str, str]]:
    for name, opener in _members(inputs):
        if name.startswith(prefix) and "Snapshot" in name:
            with opener() as raw:
                yield from csv.DictReader(
                    io.TextIOWrapper(raw, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE
                )


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    inputs = [resolve(p) for p in cfg["inputs"]]
    missing = [str(p) for p in inputs if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"SNOMED CT release files not found: {missing}. Download the UK Edition (RF2) from "
            "NHS TRUD with your account and place the zip(s) there."
        )
    prov = Provenance(stage="ingest.snomed", config=cfg, source_versions={SOURCE: cfg["version"]})
    for p in inputs:
        if p.is_file():
            prov.add_input(p)
    active = {r["id"] for r in rows(inputs, "sct2_Concept_") if r["active"] == "1"}
    descs: dict[str, dict[str, str]] = {}
    by_concept: dict[str, list[str]] = defaultdict(list)
    for r in rows(inputs, "sct2_Description_"):
        if r["active"] == "1" and r["conceptId"] in active and r["languageCode"] == "en":
            descs[r["id"]] = r
            by_concept[r["conceptId"]].append(r["id"])
    definitions: dict[str, str] = {}
    for r in rows(inputs, "sct2_TextDefinition_"):
        if r["active"] == "1" and r["typeId"] == DEFINITION and r["conceptId"] in active:
            definitions.setdefault(r["conceptId"], r["term"])
    refsets = [str(x) for x in cfg["language_refsets"]]
    preferred: dict[str, dict[str, str]] = defaultdict(dict)  # refset -> concept -> desc id
    for r in rows(inputs, "der2_cRefset_Language"):
        if r["active"] == "1" and r["acceptabilityId"] == PREFERRED and r["refsetId"] in refsets:
            d = descs.get(r["referencedComponentId"])
            if d is not None and d["typeId"] == SYNONYM:
                preferred[r["refsetId"]][d["conceptId"]] = d["id"]
    parents: dict[str, list[str]] = defaultdict(list)
    for r in rows(inputs, "sct2_Relationship_"):
        if r["active"] == "1" and r["typeId"] == IS_A and r["sourceId"] in active:
            parents[r["sourceId"]].append(r["destinationId"])

    def preferred_term(cid: str) -> dict[str, str] | None:
        for rs in refsets:
            did = preferred[rs].get(cid)
            if did:
                return descs[did]
        return None

    stats: Counter[str] = Counter()
    out = BatchWriter(cfg["output"], SENSE_SCHEMA, prov, batch_rows=100_000)
    try:
        for i, cid in enumerate(sorted(active, key=int)):
            pt = preferred_term(cid)
            if pt is None:
                stats["drop:no_preferred_term"] += 1
                continue
            fsn = next((descs[d]["term"] for d in by_concept[cid] if descs[d]["typeId"] == FSN), "")
            m = TAG_RE.search(fsn)
            tag = m.group(1) if m else "concept"
            term = pt["term"]
            lemma = (
                term if pt["caseSignificanceId"] == CASE_SENSITIVE else term[:1].lower() + term[1:]
            )
            pnames = [p["term"] for p in (preferred_term(x) for x in parents[cid]) if p]
            gloss = definitions.get(cid) or (
                f"{tag}; a kind of {', '.join(pnames[:3])}" if pnames else tag
            )
            syns = sorted(
                {
                    descs[d]["term"]
                    for d in by_concept[cid]
                    if descs[d]["typeId"] == SYNONYM and d != pt["id"]
                }
            )
            rel = {"fsn": fsn, "semantic_tag": tag, "is_a": parents[cid], "synonyms": syns[:30]}
            out.write(
                make_row(
                    source=SOURCE,
                    source_id=f"sct:{cid}",
                    lemma=lemma,
                    pos=f"sct:{tag}",
                    gloss=gloss,
                    tags=[f"snomed:{tag}", "licence:restricted"],
                    domain=cfg["tag_domains"].get(tag, "medicine"),
                    relations=orjson.dumps(rel, option=orjson.OPT_SORT_KEYS).decode(),
                    source_version=cfg["version"],
                    kind="sense",
                    entry_key=f"sct:{cid}",
                    line_no=i,
                    sense_rank=0,
                )
            )
            stats["kept"] += 1
            stats[f"tag:{tag}"] += 1
    except BaseException:
        out.abort()
        raise
    st = dict(sorted(stats.items()))
    out.close(stats=st)
    LOG.info("done: %s", st)
    return st


def main() -> None:
    args = stage_args("Ingest SNOMED CT UK Edition (RF2 Snapshot)")
    run(load_config(args.config))


if __name__ == "__main__":
    main()
