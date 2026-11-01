"""Ingest NCBI Taxonomy (taxdump) as scientific-name senses.

``python -m senselm.ingest.ncbi_taxonomy --config configs/ingest_ncbi_taxonomy.yaml``

Filter (configurable; counts in the Phase-2 report):

* rank in ``ranks`` (default: domain … species, the formal Linnaean ranks);
* not inside a subtree whose name marks it as placeholder material ("environmental samples",
  "unclassified …", "uncultured …", "metagenomes");
* the scientific name has the form the rank requires: a single capitalised word for genus and
  above, a strict binomial for species (``Genus epithet``), or a virus-style name ending in
  virus / viroid / phage / satellite for viral species; names with ``sp.``, ``cf.``, ``aff.``,
  strain-like digit runs or annotations such as "(nom. inval.)" are excluded. This keeps
  vocabulary (*Mus musculus*, *Escherichia coli*) and drops ~1.5M specimen placeholders
  (*Brevundimonas sp. S2*).

One sense per kept taxon: lemma = scientific name, POS label ``taxon`` (→ PROPN), gloss
synthesised from rank, lineage and common names ("species in the genus Mus; family Muridae,
order Rodentia, class Mammalia; common name: house mouse"), tags ``rank:<rank>``,
domain ``taxonomy``. Common names and synonyms are kept in ``relations`` for alignment only
(docs/DECISIONS.md D-034): they never create senses.
"""

from __future__ import annotations

import io
import logging
import re
import zipfile
from collections import Counter, defaultdict
from typing import Any

import orjson

from senselm.common import BatchWriter, Provenance, load_config, resolve, stage_args
from senselm.ingest.manifest import source_version
from senselm.ingest.schema import SENSE_SCHEMA, make_row

LOG = logging.getLogger("senselm.ingest.ncbi_taxonomy")
SOURCE = "ncbi_taxonomy"

GENUS_UP_RE = re.compile(r"^[A-Z][a-z]+$")
BINOMIAL_RE = re.compile(r"^[A-Z][a-z]+ [a-z][a-z-]+$")
VIRUS_RE = re.compile(r"^[A-Za-z][A-Za-z' -]*(virus|viroid|phage|satellite)$")
BAD_RE = re.compile(
    r"\b(sp|spp|cf|aff|nr|gen|nom|inval)\.|\buncultured\b|\bunidentified\b|\benvironmental\b|"
    r"\bunclassified\b|\bmetagenome\b|\bstrain\b|\bisolate\b|\bclone\b|[0-9]{2,}|[()\"]",
    re.IGNORECASE,
)
PLACEHOLDER_SUBTREE_RE = re.compile(
    r"environmental samples|^unclassified|^uncultured|metagenomes|^other sequences|^synthetic",
    re.IGNORECASE,
)
LINEAGE_RANKS = ("genus", "family", "order", "class", "phylum", "kingdom", "domain")


def _rows(z: zipfile.ZipFile, name: str) -> Any:
    with z.open(name) as fh:
        for line in io.TextIOWrapper(fh, encoding="utf-8"):
            yield line.rstrip("\n").rstrip("|").rstrip("\t").split("\t|\t")


def name_ok(name: str, rank: str) -> bool:
    if BAD_RE.search(name):
        return False
    if rank == "species":
        return bool(BINOMIAL_RE.match(name) or VIRUS_RE.match(name))
    return bool(GENUS_UP_RE.match(name))


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    src = resolve(cfg["input"])
    version = source_version(cfg, SOURCE)
    prov = Provenance(stage="ingest.ncbi_taxonomy", config=cfg, source_versions={SOURCE: version})
    prov.add_input(src)
    ranks = set(cfg["ranks"])
    z = zipfile.ZipFile(src)
    parent: dict[str, str] = {}
    rank: dict[str, str] = {}
    for p in _rows(z, "nodes.dmp"):
        parent[p[0]], rank[p[0]] = p[1], p[2]
    sci: dict[str, str] = {}
    common: dict[str, list[str]] = defaultdict(list)
    synonyms: dict[str, list[str]] = defaultdict(list)
    for p in _rows(z, "names.dmp"):
        tid, name, cls = p[0], p[1], p[3]
        if cls == "scientific name":
            sci[tid] = name
        elif cls in ("genbank common name", "common name"):
            common[tid].append(name)
        elif cls in ("synonym", "equivalent name"):
            synonyms[tid].append(name)
    LOG.info("%d nodes, %d scientific names", len(parent), len(sci))

    placeholder: dict[str, bool] = {}

    def in_placeholder(tid: str) -> bool:
        path = []
        cur = tid
        while cur not in placeholder:
            path.append(cur)
            if PLACEHOLDER_SUBTREE_RE.search(sci.get(cur, "")):
                placeholder[cur] = True
                break
            nxt = parent.get(cur, cur)
            if nxt == cur:  # root
                placeholder[cur] = False
                break
            cur = nxt
        val = placeholder[cur]
        for t in path:
            placeholder[t] = val
        return val

    def lineage(tid: str) -> dict[str, str]:
        out: dict[str, str] = {}
        cur = parent.get(tid)
        while cur and cur != parent.get(cur):
            if rank.get(cur) in LINEAGE_RANKS and rank[cur] not in out:
                out[rank[cur]] = sci.get(cur, "")
            cur = parent.get(cur)
        return out

    stats: Counter[str] = Counter()
    out = BatchWriter(cfg["output"], SENSE_SCHEMA, prov, batch_rows=100_000)
    try:
        for i, tid in enumerate(sorted(sci, key=int)):
            r = rank.get(tid, "")
            if r not in ranks:
                continue
            stats[f"rank:{r}"] += 1
            name = sci[tid]
            if not name_ok(name, r):
                stats["drop:name"] += 1
                continue
            if in_placeholder(tid):
                stats["drop:placeholder_subtree"] += 1
                continue
            lin = lineage(tid)
            parts = []
            up = next((rk for rk in LINEAGE_RANKS if rk in lin), None)
            head = f"{r} in the {up} {lin[up]}" if up else r
            rest = [f"{rk} {lin[rk]}" for rk in LINEAGE_RANKS if rk in lin and rk != up][:4]
            parts.append(head + ("; " + ", ".join(rest) if rest else ""))
            if common.get(tid):
                parts.append("common name: " + ", ".join(sorted(set(common[tid]))[:3]))
            gloss = "; ".join(parts)
            rel = {
                "taxid": int(tid),
                "rank": r,
                "parent": int(parent[tid]),
                "lineage": lin,
                "common_names": sorted(set(common.get(tid, []))),
                "synonyms": sorted(set(synonyms.get(tid, [])))[:20],
            }
            out.write(
                make_row(
                    source=SOURCE,
                    source_id=f"taxon:{tid}",
                    lemma=name,
                    pos="taxon",
                    gloss=gloss,
                    tags=[f"rank:{r}"],
                    domain="taxonomy",
                    relations=orjson.dumps(rel, option=orjson.OPT_SORT_KEYS).decode(),
                    source_version=version,
                    kind="sense",
                    entry_key=f"taxon:{tid}",
                    line_no=i,
                    sense_rank=0,
                )
            )
            stats["kept"] += 1
            stats[f"kept:{r}"] += 1
    except BaseException:
        out.abort()
        raise
    st = dict(sorted(stats.items()))
    out.close(stats=st)
    LOG.info("done: %s", st)
    return st


def main() -> None:
    args = stage_args("Ingest NCBI Taxonomy scientific names")
    run(load_config(args.config))


if __name__ == "__main__":
    main()
