"""Streaming OBO 1.2/1.4 reader and the GO / ChEBI ingests built on it.

``python -m senselm.ingest.obo --config configs/ingest_go.yaml``      (Gene Ontology)
``python -m senselm.ingest.obo --config configs/ingest_chebi.yaml``   (ChEBI)

One sense per non-obsolete ``[Term]`` passing the config filter (e.g. ChEBI 3-star only):
lemma = ``name``, gloss = ``def`` (quotes, citation brackets and inline HTML such as ChEBI's
``<em>``/``<sub>`` removed) or, when a term has no
definition, a gloss synthesised from its ``is_a`` parents ("a chemical entity that is a …").
Synonyms (EXACT / RELATED / NARROW / BROAD), ``is_a`` parents, cross-references and the
namespace / subsets go to ``relations`` and ``tags``. Synonyms are alignment-only (D-034).
"""

from __future__ import annotations

import gzip
import html
import logging
import re
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import orjson

from senselm.common import BatchWriter, Provenance, load_config, resolve, stage_args
from senselm.ingest.manifest import source_version
from senselm.ingest.schema import SENSE_SCHEMA, make_row

LOG = logging.getLogger("senselm.ingest.obo")
DEF_RE = re.compile(r'^"(.*)"\s*\[.*\]\s*$', re.DOTALL)
SYN_RE = re.compile(r'^"(.*)"\s+(EXACT|RELATED|NARROW|BROAD)\b')
ISA_RE = re.compile(r"^(\S+)(?:\s*!\s*(.*))?$")
HTML_TAG_RE = re.compile(r"</?(?:em|i|b|u|sub|sup|small|smallcaps|strong|span|br)\b[^>]*>", re.I)


def iter_stanzas(path: Path) -> Iterator[dict[str, list[str]]]:
    opener = gzip.open if path.suffix == ".gz" else open
    cur: dict[str, list[str]] | None = None
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("["):
                if cur is not None:
                    yield cur
                cur = {"_stanza": [line.strip()]}
                continue
            if cur is None or not line or line.startswith("!") or ":" not in line:
                continue
            key, val = line.split(":", 1)
            cur.setdefault(key.strip(), []).append(val.strip())
    if cur is not None:
        yield cur


def unescape(s: str) -> str:
    """OBO escapes, then inline HTML markup and entities, then whitespace."""
    s = s.replace('\\"', '"').replace("\\n", " ").replace("\\\\", "\\")
    if "<" in s or "&" in s:
        s = html.unescape(HTML_TAG_RE.sub("", s))
    return " ".join(s.split())


def definition(term: dict[str, list[str]]) -> str | None:
    for d in term.get("def", []):
        m = DEF_RE.match(d)
        txt = unescape(m.group(1) if m else d).strip()
        if txt:
            return txt
    return None


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    src = resolve(cfg["input"])
    source = cfg["source"]
    version = source_version(cfg, source)
    prov = Provenance(stage=f"ingest.{source}", config=cfg, source_versions={source: version})
    prov.add_input(src)
    require_subsets = set(cfg.get("require_any_subset", []))
    names: dict[str, str] = {}
    terms: list[dict[str, list[str]]] = []
    for st in iter_stanzas(src):  # pass 1: names (for is_a glosses), keep candidate terms
        if st["_stanza"][0] != "[Term]" or "id" not in st or "name" not in st:
            continue
        names[st["id"][0]] = st["name"][0]
        if st.get("is_obsolete", ["false"])[0] == "true":
            continue
        if require_subsets and not require_subsets.intersection(st.get("subset", [])):
            continue
        terms.append(st)
    stats: Counter[str] = Counter()
    out = BatchWriter(cfg["output"], SENSE_SCHEMA, prov, batch_rows=100_000)
    try:
        for i, t in enumerate(terms):
            tid, name = t["id"][0], unescape(t["name"][0])
            parents = []
            for isa in t.get("is_a", []):
                m = ISA_RE.match(isa)
                if m:
                    parents.append(m.group(1))
            gloss = definition(t)
            if gloss is None:
                pnames = [names[p] for p in parents if p in names]
                gloss = (
                    f"{cfg['undefined_gloss_prefix']} {' and '.join(pnames[:3])}"
                    if pnames
                    else None
                )
                stats["synthesised_gloss"] += gloss is not None
            if gloss is None:
                stats["drop:no_gloss"] += 1
                continue
            syn: dict[str, list[str]] = {}
            for s in t.get("synonym", []):
                m = SYN_RE.match(s)
                if m:
                    syn.setdefault(m.group(2).lower(), []).append(unescape(m.group(1)))
            ns = t.get("namespace", [])
            tags = [f"{cfg['tag_prefix']}:{x}" for x in ns + t.get("subset", [])]
            rel = {"is_a": parents, "synonyms": syn, "xrefs": t.get("xref", [])[:30]}
            out.write(
                make_row(
                    source=source,
                    source_id=tid,
                    lemma=name,
                    pos=cfg["pos_label"],
                    gloss=gloss,
                    tags=tags,
                    domain=cfg["domain"],
                    relations=orjson.dumps(rel, option=orjson.OPT_SORT_KEYS).decode(),
                    source_version=version,
                    kind="sense",
                    entry_key=tid,
                    line_no=i,
                    sense_rank=0,
                )
            )
            stats["kept"] += 1
            for x in ns:
                stats[f"namespace:{x}"] += 1
    except BaseException:
        out.abort()
        raise
    st = dict(sorted(stats.items()))
    out.close(stats=st)
    LOG.info("%s: %s", source, st)
    return st


def main() -> None:
    args = stage_args("Ingest an OBO ontology (GO, ChEBI)")
    run(load_config(args.config))


if __name__ == "__main__":
    main()
