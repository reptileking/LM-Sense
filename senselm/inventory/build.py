"""Build the canonical sense inventory and assign permanent token IDs.

``python -m senselm.inventory.build --config configs/inventory.yaml``          (fine, IDs)
``python -m senselm.inventory.build --config configs/inventory_coarse.yaml``   (coarse, counts)

1. cluster source senses under the configured granularity policy (``policy.py``); for a
   multi-source build (v1.0+) the OEWN / Wiktionary clusters are unioned with the stage-2
   attachments of the other sources (``attach.py``, ``inputs.attachments``);
2. aggregate each cluster into one canonical sense that lists every contributing
   ``(source, source_id)`` (provenance, so licensed subsets can be excluded later);
3. order senses within each lemma by a frequency proxy (docs/DECISIONS.md D-021):
   WordNet-attested senses first in WordNet sense order (SemCor-frequency based), POS
   interleaved with the lemma's dominant POS first; then Wiktionary-only senses in
   Wiktionary order (top-level senses before their subsenses), those tagged
   obsolete/archaic/rare/dated/historical after the rest; then senses only Wikidata lexemes
   attest; then concept / domain senses (Wikidata items, MeSH, GO, ChEBI, NCBI Taxonomy,
   SNOMED CT) in ``source_order``;
4. assign token IDs append-only against the previous registry (``ids.py``) and refuse to
   write if the permanence invariant would be violated. The previous registry is
   snapshotted to ``registry_history/`` before it is replaced.
"""

from __future__ import annotations

import logging
from typing import Any

import polars as pl

from senselm.common import Provenance, dump_json, load_config, resolve, stage_args, write_table
from senselm.inventory.align import is_header
from senselm.inventory.ids import (
    Layout,
    assign_ids,
    check_append_only,
    from_ledger,
    ledger_paths,
    read_ledger,
    redact_ledger,
    redact_lemmas,
    rehydrate,
    restricted_key,
    to_ledger,
    write_ledger,
)
from senselm.inventory.policy import assign_clusters

LOG = logging.getLogger("senselm.inventory.build")

SENSE_COLS = [
    "source",
    "source_id",
    "lemma",
    "upos",
    "gloss",
    "gloss_path",
    "examples",
    "tags",
    "domain",
    "lexfile",
    "synset_id",
    "hypernyms",
    "entry_key",
    "line_no",
    "sense_rank",
    "is_abbrev",
    "source_version",
]
POS_ORDER = [
    "NOUN",
    "VERB",
    "ADJ",
    "ADV",
    "PROPN",
    "PRON",
    "DET",
    "ADP",
    "CCONJ",
    "SCONJ",
    "NUM",
    "PART",
    "INTJ",
    "X",
    "SYM",
    "PUNCT",
]


def wiktionary_ranks(wikt: pl.DataFrame) -> dict[str, float]:
    """Rank of each Wiktionary sense among its (lemma, UPOS) senses: top-level senses count
    0, 1, 2, ... in dump order; a subsense gets its parent's rank + 0.001·k."""
    out: dict[str, float] = {}
    rows = (
        wikt.sort("lemma", "upos", "line_no", "sense_rank")
        .select("source_id", "lemma", "upos", "entry_key", "gloss", "gloss_path")
        .rows()
    )
    cur_key: tuple[str, str] | None = None
    rank = -1
    sub = 0
    top_of: dict[tuple[str, str], int] = {}
    for sid, lemma, upos, entry, gloss, path in rows:
        if (lemma, upos) != cur_key:
            cur_key, rank, top_of = (lemma, upos), -1, {}
        full = [p for p in [*(path or []), gloss or ""] if p and not is_header(p)]
        top = full[0] if full else sid
        tkey = (entry, top)
        if tkey not in top_of:
            rank += 1
            top_of[tkey] = rank
            sub = 0
        if len(full) <= 1:
            out[sid] = float(top_of[tkey])
        else:
            sub += 1
            out[sid] = top_of[tkey] + 0.001 * sub
    return out


def canonical_senses(
    senses: pl.DataFrame, clusters: pl.DataFrame, cfg: dict[str, Any]
) -> pl.DataFrame:
    rare = set(cfg["ordering"]["rare_tags"])
    licences: dict[str, str] = cfg["licences"]
    wrank = wiktionary_ranks(senses.filter(pl.col("source") == "wiktionary"))
    order = {src: i for i, src in enumerate(cfg.get("source_order", ["oewn", "wiktionary"]))}
    gloss_order = {src: i for i, src in enumerate(cfg.get("gloss_order", list(order)))}
    s = senses.join(clusters.select("source_id", "cluster"), on="source_id", how="inner")
    s = s.with_columns(
        pl.when(pl.col("source") == "oewn")
        .then(pl.col("sense_rank").cast(pl.Float64))
        .when(pl.col("source") == "wiktionary")
        .then(pl.col("source_id").replace_strict(wrank, default=None, return_dtype=pl.Float64))
        .otherwise(pl.col("sense_rank").cast(pl.Float64))
        .alias("rank"),
        pl.col("source")
        .replace_strict(order, default=len(order), return_dtype=pl.Int32)
        .alias("src_order"),
        pl.col("source")
        .replace_strict(gloss_order, default=len(gloss_order), return_dtype=pl.Int32)
        .alias("gloss_rank"),
    ).sort("cluster", "src_order", "line_no", "sense_rank", "source_id")

    agg = s.group_by("cluster", maintain_order=True).agg(
        pl.struct("source", "source_id").alias("members"),
        pl.col("source").unique(maintain_order=True).alias("sources"),
        # lemma / POS: the Wiktionary member's when present (page titles carry the curated
        # capitalisation; Wiktionary distinguishes proper nouns), else WordNet's
        pl.col("lemma").filter(pl.col("source") == "wiktionary").first().alias("_w_lemma"),
        pl.col("lemma").first().alias("_any_lemma"),
        pl.col("upos").filter(pl.col("source") == "wiktionary").first().alias("_w_upos"),
        pl.col("upos").first().alias("_any_upos"),
        # gloss: WordNet's (concise, uniform style) when present, else the first Wiktionary one,
        # else the first in gloss_order (curated definitions before Wikidata's short descriptions)
        pl.col("gloss")
        .filter(pl.col("gloss_rank") == pl.col("gloss_rank").min())
        .first()
        .alias("gloss"),
        pl.struct("source", "gloss").alias("glosses"),
        pl.col("examples")
        .list.explode(keep_nulls=False, empty_as_null=False)
        .drop_nulls()
        .unique(maintain_order=True)
        .alias("examples"),
        pl.col("tags")
        .list.explode(keep_nulls=False, empty_as_null=False)
        .drop_nulls()
        .unique()
        .sort()
        .alias("tags"),
        pl.col("domain")
        .filter(pl.col("source") == "wiktionary")
        .drop_nulls()
        .first()
        .alias("_w_domain"),
        pl.col("domain").drop_nulls().first().alias("_any_domain"),
        pl.col("lexfile").drop_nulls().first().alias("lexfile"),
        pl.col("synset_id").drop_nulls().first().alias("synset_id"),
        pl.col("hypernyms")
        .list.explode(keep_nulls=False, empty_as_null=False)
        .drop_nulls()
        .unique(maintain_order=True)
        .alias("hypernyms"),
        pl.col("is_abbrev").any().alias("is_abbrev"),
        pl.col("rank").filter(pl.col("source") == "oewn").min().alias("oewn_rank"),
        pl.col("rank").filter(pl.col("source") == "wiktionary").min().alias("wikt_rank"),
        pl.col("line_no").filter(pl.col("source") == "wiktionary").min().alias("wikt_line"),
        pl.col("src_order").first().alias("primary_order"),
        pl.col("source").first().alias("primary_source"),
        pl.col("rank").first().alias("primary_rank"),
        pl.col("line_no").first().alias("primary_line"),
        pl.col("source_version").unique().sort().alias("source_versions"),
    )
    # exact duplicates removed in normalise (dedupe_map) are members too: they matter for ID
    # continuity if a later dump changes which duplicate comes first
    extra = (
        clusters.join(senses.select("source_id"), on="source_id", how="anti")
        .sort("source_id")
        .group_by("cluster")
        .agg(pl.struct("source", "source_id").alias("extra_members"))
    )
    agg = (
        agg.join(extra, on="cluster", how="left", maintain_order="left")
        .with_columns(
            pl.when(pl.col("extra_members").is_not_null())
            .then(pl.concat_list("members", "extra_members"))
            .otherwise(pl.col("members"))
            .alias("members")
        )
        .drop("extra_members")
    )
    agg = agg.with_columns(
        pl.coalesce("_w_lemma", "_any_lemma").alias("lemma"),
        pl.coalesce("_w_upos", "_any_upos").alias("upos"),
        pl.coalesce("_w_domain", "_any_domain").alias("domain"),
        pl.col("cluster").alias("canonical_key"),
        pl.col("members").list.len().alias("n_members"),
        pl.col("sources").list.contains("oewn").alias("has_oewn"),
        pl.col("sources").list.contains("wiktionary").alias("has_wiktionary"),
        pl.col("examples").list.len().gt(0).alias("has_example"),
        pl.col("sources").list.eval(pl.element().replace_strict(licences)).alias("licences"),
        pl.col("tags").list.set_intersection(sorted(rare)).list.len().gt(0).alias("rare"),
    ).drop("_w_lemma", "_any_lemma", "_w_upos", "_any_upos", "_w_domain", "_any_domain", "cluster")
    return agg


def order_senses(inv: pl.DataFrame) -> pl.DataFrame:
    """Sort canonical senses into sense-index order within each lemma (frequency proxy)."""
    pos_rank = {p: i for i, p in enumerate(POS_ORDER)}
    dom = (
        inv.group_by("lemma", "upos")
        .agg(pl.col("has_oewn").sum().alias("n_oewn"), pl.len().alias("n_all"))
        .with_columns(
            pl.col("upos").replace_strict(pos_rank, default=len(POS_ORDER)).alias("pos_fixed")
        )
        .sort(["lemma", "n_oewn", "n_all", "pos_fixed"], descending=[False, True, True, False])
        .with_columns(pl.int_range(pl.len()).over("lemma").alias("pos_dominance"))
        .select("lemma", "upos", "pos_dominance")
    )
    inv = inv.join(dom, on=["lemma", "upos"], how="left")
    # tiers (0-2 exactly as in v0.1): 0 WordNet, 1 Wiktionary, 2 rare Wiktionary (obsolete,
    # archaic, ...), 3 Wikidata-lexeme-only, 4 concept / domain sources. Within the lexicographic
    # tiers the source's sense order; within tier 4 the source order, then the source's order.
    lexical = pl.col("has_oewn") | pl.col("has_wiktionary")
    inv = inv.with_columns(
        pl.when(pl.col("has_oewn"))
        .then(0)
        .when(pl.col("has_wiktionary") & ~pl.col("rare"))
        .then(1)
        .when(pl.col("has_wiktionary"))
        .then(2)
        .when(pl.col("primary_source") == "wikidata_lexemes")
        .then(3)
        .otherwise(4)
        .alias("tier"),
        pl.when(pl.col("has_oewn"))
        .then(pl.col("oewn_rank"))
        .when(pl.col("has_wiktionary"))
        .then(pl.col("wikt_rank"))
        .otherwise(pl.col("primary_rank"))
        .fill_null(1e9)
        .alias("rank_key"),
        pl.when(lexical).then(0).otherwise(pl.col("primary_order")).alias("src_key"),
        pl.when(lexical)
        .then(pl.col("wikt_line"))
        .otherwise(pl.col("primary_line"))
        .alias("line_key"),
    )
    return inv.sort(
        ["lemma", "tier", "src_key", "rank_key", "pos_dominance", "line_key", "canonical_key"],
        nulls_last=True,
    ).with_columns(pl.int_range(pl.len()).over("lemma").alias("order_key"))


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    prov = Provenance(stage="inventory.build", config=cfg)
    ins, outs = cfg["inputs"], cfg["outputs"]
    for k in ("senses", "pairs", "dedupe_map"):
        prov.add_input(ins[k])
    senses = pl.read_parquet(resolve(ins["senses"]), columns=SENSE_COLS)
    # restricted-licence sources (SNOMED CT, D-036): their member ids start with one of
    # restricted_id_prefixes; the committed ledger only carries hashed keys and lemmas for them
    prefixes = list(cfg.get("restricted_id_prefixes", []))
    restricted = set(cfg.get("restricted_sources", [])) & set(senses["source"].unique().to_list())
    if restricted:
        ids_of = senses.filter(pl.col("source").is_in(sorted(restricted)))["source_id"].to_list()
        uncovered = [x for x in ids_of if not any(x.startswith(pre) for pre in prefixes)]
        if not prefixes or uncovered:
            raise RuntimeError(
                f"restricted sources {sorted(restricted)}: ids not covered by restricted_id_prefixes"
            )
    pairs = pl.read_parquet(resolve(ins["pairs"]))
    dedupe = pl.read_parquet(resolve(ins["dedupe_map"]))
    stage1 = ["oewn", "wiktionary"]
    clusters, pstats = assign_clusters(
        senses.filter(pl.col("source").is_in(stage1)).select(
            "source", "source_id", "entry_key", "gloss", "gloss_path", "line_no", "sense_rank"
        ),
        pairs,
        cfg["policy"],
        dedupe.filter(pl.col("source").is_in(stage1)),
    )
    LOG.info("policy %s: %s", cfg["policy"]["name"], pstats)
    if ins.get("attachments"):  # stage 2: the other sources (attach.py)
        prov.add_input(ins["attachments"])
        att = pl.read_parquet(resolve(ins["attachments"])).select("source", "source_id", "cluster")
        dups = (
            dedupe.filter(~pl.col("source").is_in(stage1))
            .join(att.select(pl.col("source_id").alias("kept_id"), "cluster"), on="kept_id")
            .select("source", pl.col("dropped_id").alias("source_id"), "cluster")
        )
        clusters = pl.concat([clusters.select("source", "source_id", "cluster"), att, dups])
        pstats["attached"] = int((att["cluster"] != att["source_id"]).sum())
        pstats["stage2_new_clusters"] = int((att["cluster"] == att["source_id"]).sum())
    missing = senses.join(clusters, on="source_id", how="anti")
    if missing.height:
        raise RuntimeError(
            f"{missing.height} senses have no cluster, e.g. {missing['source_id'][:5].to_list()}"
        )
    inv = order_senses(canonical_senses(senses, clusters, cfg))
    stats: dict[str, Any] = {
        "policy": cfg["policy"]["name"],
        **pstats,
        "canonical_senses": inv.height,
        "lemmas": inv["lemma"].n_unique(),
    }

    if cfg.get("assign_ids", True):
        layout = Layout(**cfg["ids"]["layout"])
        reg_path, lem_path = resolve(outs["registry"]), resolve(outs["lemma_registry"])
        prev = pl.read_parquet(reg_path) if reg_path.exists() else None
        prev_lem = pl.read_parquet(lem_path) if lem_path.exists() else None
        if prev is not None:
            prov.add_input(reg_path)
        elif outs.get("registry_commit_dir"):
            # data/ was lost (fresh machine): continue from the committed ledger (D-025)
            commit_dir = resolve(outs["registry_commit_dir"])
            ledger, lemmas = (
                read_ledger(commit_dir, "id_registry"),
                read_ledger(commit_dir, "lemma_registry"),
            )
            if ledger is not None and lemmas is not None:
                LOG.warning(
                    "no %s: continuing from the committed ledger in %s", reg_path, commit_dir
                )
                prev, prev_lem = from_ledger(ledger), lemmas
                for f in ledger_paths(commit_dir, "id_registry"):
                    prov.add_input(f)
        held = None
        assign_prev = prev
        if prev is not None and prev_lem is not None and prefixes:
            # restore redacted ledger rows of the restricted senses this build contains ...
            rk = inv.filter(restricted_key("canonical_key", prefixes))
            prev, prev_lem = rehydrate(
                prev, prev_lem, rk["canonical_key"].to_list(), rk["lemma"].to_list()
            )
            assign_prev = prev
            if not restricted:  # ... or hold them out untouched when it contains none
                mask = restricted_key("canonical_key", prefixes)
                held, assign_prev = prev.filter(mask), prev.filter(~mask)
        reg, lem, istats = assign_ids(
            inv.select(
                "canonical_key",
                "lemma",
                "upos",
                pl.col("members").list.eval(pl.element().struct.field("source_id")),
                "gloss",
                "order_key",
            ),
            assign_prev,
            prev_lem,
            cfg["build_id"],
            layout,
            reserved=held,
        )
        if held is not None and held.height:
            reg = pl.concat([reg, held.select(reg.columns).cast(reg.schema)]).sort("token_id")
            istats["held_restricted_ids"] = held.height
        if prev is not None:
            problems = check_append_only(prev, reg)
            if problems:
                raise RuntimeError(f"ID registry would violate permanence: {problems}")
        stats["ids"] = istats
        active = reg.filter(pl.col("status") == "active").select(
            "token_id", "lemma_id", "sense_index", "canonical_key"
        )
        inv = inv.join(active, on="canonical_key", how="left")
        if inv["token_id"].null_count():
            raise RuntimeError("some canonical senses did not receive a token id")
        inv = inv.sort("token_id")
        # snapshot the previous registry, then write the new one
        if prev is not None and not prev.equals(reg):
            hist = resolve(outs["history_dir"])
            hist.mkdir(parents=True, exist_ok=True)
            seq = len(list(hist.glob("id_registry.*.parquet")))  # name order = chronological order
            prev_build = prev["created_in"].max()
            prev.write_parquet(
                hist / f"id_registry.{seq:04d}.{prev_build}.parquet", compression="zstd"
            )
        reg_prov = Provenance(
            stage="inventory.build.registry", config=cfg, extra={"layout": layout.to_dict()}
        )
        reg_prov.source_versions = prov.source_versions
        write_table(reg.to_arrow(), reg_path, reg_prov, stats=istats)
        write_table(lem.to_arrow(), lem_path, reg_prov)
        if outs.get("registry_commit_dir"):  # durable, git-tracked ledger (D-025), sharded if large
            commit_dir = resolve(outs["registry_commit_dir"])
            write_ledger(
                redact_ledger(to_ledger(reg), prefixes),
                commit_dir,
                "id_registry",
                lambda t, path: write_table(t, path, reg_prov, stats=istats, compression_level=19),
            )
            write_ledger(
                redact_lemmas(lem, reg, prefixes),
                commit_dir,
                "lemma_registry",
                lambda t, path: write_table(t, path, reg_prov, compression_level=19),
            )
    inv = inv.drop(
        "tier",
        "rank_key",
        "pos_dominance",
        "wikt_line",
        "src_key",
        "line_key",
        "primary_order",
        "primary_rank",
        "primary_line",
    )
    prov.extra["layout"] = cfg.get("ids", {}).get("layout")
    write_table(inv.to_arrow(), outs["inventory"], prov, stats=stats)
    if outs.get("stats"):
        dump_json(stats, outs["stats"])
    LOG.info("build: %s", stats)
    return stats


def main() -> None:
    args = stage_args("Build canonical sense inventory + permanent IDs")
    run(load_config(args.config))


if __name__ == "__main__":
    main()

__all__ = ["canonical_senses", "order_senses", "run", "wiktionary_ranks"]
