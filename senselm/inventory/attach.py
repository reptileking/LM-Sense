"""Stage 2 alignment: attach senses of the Phase-2 sources to the inventory.

``python -m senselm.inventory.attach --config configs/attach.yaml``            (attach)
``python -m senselm.inventory.attach --config configs/attach.yaml --sample``   (blind gold sample)

Stage 1 (unchanged from v0.1) clusters OEWN and Wiktionary senses under the fine policy; its
clusters keep their anchors, so every v0.1 token id survives. Stage 2 processes the remaining
sources in ``source_order``. For a sense *s* of source *S*:

* candidates = existing clusters with the same lemma and POS class (case-insensitive only when
  no exact-case cluster exists — the stage-1 rule);
* score(s, c) = max cosine between s's text and the texts of c's members (same embedding model
  and cache as stage 1);
* an edge is eligible if score ≥ ``thresholds[S]`` or a cross-reference links s to a member
  of c (Wiktionary sense → Wikidata item; lexeme sense → item via P5137; item → NCBI / MeSH /
  ChEBI / GO ids, also through the items superset for items that are not senses themselves:
  ``bridge``). Cross-reference edges rank above every cosine edge;
* greedy one-to-one per source: each sense attaches to at most one cluster and each cluster
  receives at most one sense of S (the fine policy's rule; S's own distinctions survive);
* senses that do not attach start new clusters, which later sources can attach to.

Output ``attachments.parquet``: (source, source_id, cluster, score, via) for every sense of the
stage-2 sources; the build stage unions it with the stage-1 clusters.
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import orjson
import polars as pl

from senselm.common import Provenance, dump_json, load_config, resolve, setup_logging, write_table
from senselm.inventory.align import sense_text
from senselm.inventory.candidates import DEFAULT_POS_CLASSES
from senselm.inventory.embed import Embedder, embed_texts
from senselm.inventory.policy import assign_clusters

LOG = logging.getLogger("senselm.inventory.attach")
STAGE1_SOURCES = ("oewn", "wiktionary")
COLS = [
    "source",
    "source_id",
    "lemma",
    "upos",
    "gloss",
    "gloss_path",
    "examples",
    "example_types",
    "relations",
    "entry_key",
    "line_no",
    "sense_rank",
    "lexfile",
]


def taxon_compatible(
    cfg: dict[str, Any], relations: str, member_lexfiles: list[str | None]
) -> bool:
    """Kingdom consistency for taxa: a WordNet member of the candidate cluster must have a
    lexicographer file compatible with the taxon's kingdom (an insect genus cannot be a
    noun.plant sense; no taxon is a noun.person / noun.location sense)."""
    rules = cfg.get("taxon_lexfiles")
    lexfiles = [x for x in member_lexfiles if x]
    if not rules or not lexfiles:
        return True
    lineage = orjson.loads(relations or "{}").get("lineage", {})
    allowed = set(rules.get(lineage.get("kingdom", ""), rules["default"]))
    return any(x in allowed for x in lexfiles)


def xref_keys(source: str, relations: str) -> list[str]:
    """Identity keys a sense carries (shared keys = same concept), e.g. 'wd:Q42', 'taxon:9606'."""
    try:
        rel = orjson.loads(relations or "{}")
    except orjson.JSONDecodeError:
        return []
    keys: list[str] = []
    if source == "wiktionary":
        keys += [f"wd:{q}" for q in rel.get("wikidata", []) if str(q).startswith("Q")]
    elif source == "wikidata_lexemes":
        keys += [f"wd:{q}" for q in rel.get("item", [])]
    elif source == "wikidata_items":
        x = rel.get("xrefs", {})
        keys += [f"taxon:{v}" for v in x.get("P685", [])]
        keys += [f"mesh:{v}" for v in x.get("P486", [])]
        keys += [
            f"chebi:{v}" if str(v).startswith("CHEBI:") else f"chebi:CHEBI:{v}"
            for v in x.get("P683", [])
        ]
        keys += [f"go:{v}" for v in x.get("P686", [])]
    return keys


def load_bridge(cfg: dict[str, Any], senses: pl.DataFrame) -> dict[str, list[str]]:
    """Wikidata Q-id -> domain identity keys, from the streamed items superset (``bridge``).

    Wiktionary and lexeme senses carry Q-ids; the superset knows which MeSH / ChEBI / GO / NCBI
    ids those items cross-reference, even for items that are not selected as senses
    themselves (taxa, chemical compounds). Only the Q-ids that some sense carries are loaded.
    """
    path = cfg.get("bridge")
    if not path or not resolve(path).exists():
        return {}
    qids: set[str] = set()
    for src, rel in (
        senses.filter(pl.col("source").is_in(["wiktionary", "wikidata_lexemes"]))
        .select("source", "relations")
        .iter_rows()
    ):
        qids.update(k[3:] for k in xref_keys(src, rel))
    sup = (
        pl.scan_parquet(resolve(path))
        .filter(pl.col("qid").is_in(sorted(qids)) & (pl.col("xrefs") != "{}"))
        .select("qid", "xrefs")
        .collect()
    )
    out: dict[str, list[str]] = {}
    for q, x in sup.iter_rows():
        keys = xref_keys("wikidata_items", orjson.dumps({"xrefs": orjson.loads(x)}).decode())
        if keys:
            out[f"wd:{q}"] = keys
    LOG.info(
        "bridge: %d of %d sense-linked items carry domain cross-references", len(out), len(qids)
    )
    return out


def identity_keys(source: str, relations: str, bridge: dict[str, list[str]]) -> list[str]:
    keys = xref_keys(source, relations)
    return keys + [b for k in keys for b in bridge.get(k, ())]


def own_key(source: str, source_id: str) -> str | None:
    return {
        "wikidata_items": f"wd:{source_id}",
        "ncbi_taxonomy": source_id,  # already "taxon:<id>"
        "mesh": f"mesh:{source_id}",
        "chebi": f"chebi:{source_id}",
        "gene_ontology": f"go:{source_id}",
    }.get(source)


def stage1(cfg: dict[str, Any], senses: pl.DataFrame) -> pl.DataFrame:
    pol = load_config(resolve(cfg["policy_config"]))["policy"]
    pairs = pl.read_parquet(resolve(cfg["pairs"]))
    s1 = senses.filter(pl.col("source").is_in(STAGE1_SOURCES))
    clusters, st = assign_clusters(
        s1.select(
            "source", "source_id", "entry_key", "gloss", "gloss_path", "line_no", "sense_rank"
        ),
        pairs,
        pol,
    )
    LOG.info("stage 1: %s", st)
    return clusters.select("source_id", "cluster")


def texts_for(senses: pl.DataFrame, ids: list[str], tcfg: dict[str, Any]) -> dict[str, str]:
    """Embedding text per sense: the stage-1 text for OEWN / Wiktionary, the gloss otherwise."""
    sub = senses.filter(pl.col("source_id").is_in(ids))
    return {
        r["source_id"]: sense_text(r, tcfg) if r["source"] in STAGE1_SOURCES else (r["gloss"] or "")
        for r in sub.iter_rows(named=True)
    }


def run(cfg: dict[str, Any], *, sample: bool = False) -> dict[str, Any]:
    if "embedding" not in cfg:  # single source of truth: the stage-1 alignment config
        cfg = {**cfg, "embedding": load_config(resolve(cfg["align_config"]))["embedding"]}
    prov = Provenance(stage="inventory.attach", config=cfg)
    for k in ("senses", "pairs"):
        prov.add_input(cfg[k])
    classes = cfg.get("pos_classes", DEFAULT_POS_CLASSES)
    senses = pl.read_parquet(resolve(cfg["senses"]), columns=COLS).with_columns(
        pl.col("upos").replace_strict(classes, default=pl.col("upos")).alias("pcls")
    )
    cl1 = stage1(cfg, senses)
    info = senses.select("source_id", "source", "lemma", "pcls", "relations")
    # cluster index: (lemma, pcls) of every stage-1 cluster, from its anchor-defining members
    members = cl1.join(info, on="source_id")
    cluster_of: dict[str, str] = dict(
        zip(cl1["source_id"].to_list(), cl1["cluster"].to_list(), strict=True)
    )
    cl_members: dict[str, list[str]] = defaultdict(list)
    for sid, c in zip(cl1["source_id"].to_list(), cl1["cluster"].to_list(), strict=True):
        cl_members[c].append(sid)
    cl_lemma = (
        members.sort(pl.col("source") != "wiktionary")  # Wiktionary lemma wins (canonical rule)
        .unique("cluster", keep="first", maintain_order=True)
        .select("cluster", "lemma", "pcls")
    )
    index: dict[tuple[str, str], list[str]] = defaultdict(list)
    for c, lem, pc in cl_lemma.iter_rows():
        index[(lem, pc)].append(c)
    lower_index: dict[tuple[str, str], list[str]] = defaultdict(list)
    for (lem, pc), cs in index.items():
        lower_index[(lem.lower(), pc)].extend(cs)
    # identity keys carried by members of each cluster (plus their Wikidata bridge)
    bridge = load_bridge(cfg, senses)
    key_cluster: dict[str, set[str]] = defaultdict(set)
    for sid, src, rel in members.select("source_id", "source", "relations").iter_rows():
        for k in identity_keys(src, rel, bridge):
            key_cluster[k].add(cluster_of[sid])
    del members, cl_lemma, info
    # WordNet lexicographer files (kingdom rule for taxa); only OEWN rows carry one
    lexfile_of = dict(
        senses.filter(pl.col("lexfile").is_not_null()).select("source_id", "lexfile").iter_rows()
    )

    tcfg = {**cfg["embedding"].get("text", {}), "include_example": False}
    emb = Embedder(cfg["embedding"])

    out: dict[str, list[Any]] = {k: [] for k in ("source", "source_id", "cluster", "score", "via")}
    stats: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(cfg.get("gold", {}).get("seed", 0)))
    for src in cfg["source_order"]:
        sub = senses.filter(pl.col("source") == src).select(
            "source_id", "lemma", "pcls", "relations"
        )
        if sub.height == 0:
            continue
        rel_of: dict[str, str] = dict(
            zip(sub["source_id"].to_list(), sub["relations"].to_list(), strict=True)
        )
        cand: list[tuple[str, str, bool]] = []  # (sid, cluster, case_fallback)
        fallback_ok = src not in set(cfg.get("no_case_fallback", []))
        for sid, lem, pc in sub.select("source_id", "lemma", "pcls").iter_rows():
            cs = index.get((lem, pc))
            if cs:
                cand += [(sid, c, False) for c in cs]
            elif fallback_ok:
                cand += [(sid, c, True) for c in lower_index.get((lem.lower(), pc), [])]
        if src == "ncbi_taxonomy":
            before = len(cand)
            cand = [
                x
                for x in cand
                if taxon_compatible(
                    cfg, rel_of[x[0]], [lexfile_of.get(m) for m in cl_members[x[1]]]
                )
            ]
            stats[f"{src}:kingdom_rejected"] = before - len(cand)
        stats[f"{src}:senses"] = sub.height
        stats[f"{src}:with_candidates"] = len({c[0] for c in cand})
        if sample:
            if src in set(cfg.get("restricted_sources", [])):  # the gold set is committed (D-036)
                continue
            by_s: dict[str, list[str]] = defaultdict(list)
            for sid, c, _ in cand:
                by_s[sid].append(c)
            keys = sorted(by_s)
            n = min(int(cfg["gold"]["per_source"]), len(keys))
            for sid in sorted(rng.choice(len(keys), size=n, replace=False).tolist()):
                s_id = keys[sid]
                samples.append({"source": src, "source_id": s_id, "candidates": by_s[s_id][:8]})
            continue
        # texts: the new senses and every member of every candidate cluster
        need = sorted({c[0] for c in cand} | {m for _, c, _ in cand for m in cl_members[c]})
        text_of = texts_for(senses, need, tcfg)
        mat, _ = embed_texts([text_of[x] for x in need], cfg["embedding"], embedder=emb)
        vec = {x: mat[i] for i, x in enumerate(need)}
        del text_of
        thr = float(cfg["thresholds"].get(src, cfg["thresholds"]["default"]))
        edges = []
        for sid, c, fb in cand:
            score = max(float(vec[sid] @ vec[m]) for m in cl_members[c])
            own = own_key(src, sid)
            keys = set(identity_keys(src, rel_of[sid], bridge)) | ({own} if own else set())
            via_x = any(c in key_cluster.get(k, ()) for k in keys)
            if via_x or score >= thr:
                edges.append(
                    (
                        1.0 + score if via_x else score,
                        sid,
                        c,
                        "xref" if via_x else "cosine",
                        fb,
                        score,
                    )
                )
        del vec, cand
        edges.sort(key=lambda e: (-e[0], e[1], e[2]))
        used_s: set[str] = set()
        used_c: set[str] = set()
        attached: dict[str, tuple[str, float, str]] = {}
        for _, sid, c, via, _fb, score in edges:
            if sid in used_s or c in used_c:
                continue
            used_s.add(sid)
            used_c.add(c)
            attached[sid] = (c, round(score, 4), via)
        for sid, lem, pc in sub.select("source_id", "lemma", "pcls").iter_rows():
            if sid in attached:
                c, score, via = attached[sid]
                stats[f"{src}:attached_{via}"] += 1
            else:
                c, score, via = sid, None, "new"
                index[(lem, pc)].append(c)
                lower_index[(lem.lower(), pc)].append(c)
                stats[f"{src}:new"] += 1
            cl_members[c].append(sid)
            cluster_of[sid] = c
            own = own_key(src, sid)
            for k in identity_keys(src, rel_of[sid], bridge) + ([own] if own else []):
                key_cluster[k].add(c)
            for k, v in (
                ("source", src),
                ("source_id", sid),
                ("cluster", c),
                ("score", score),
                ("via", via),
            ):
                out[k].append(v)
        LOG.info("%s: %s", src, {k: v for k, v in stats.items() if k.startswith(src)})
    if sample:
        dump_json({"samples": samples}, resolve(cfg["gold"]["sample_output"]))
        return dict(stats)
    df = pl.DataFrame(
        out,
        schema={
            "source": pl.String,
            "source_id": pl.String,
            "cluster": pl.String,
            "score": pl.Float64,
            "via": pl.String,
        },
    )
    write_table(df.to_arrow(), cfg["output"], prov, stats=dict(stats))
    return dict(stats)


def main() -> None:
    ap = argparse.ArgumentParser(description="Attach Phase-2 senses to the inventory")
    ap.add_argument("--config", required=True)
    ap.add_argument("--sample", action="store_true", help="write the blind gold sample instead")
    args = ap.parse_args()
    setup_logging()
    print(run(load_config(args.config), sample=args.sample))


if __name__ == "__main__":
    main()
