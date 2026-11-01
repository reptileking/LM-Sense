"""Build the canonical sense inventory and assign permanent token IDs.

``python -m senselm.inventory.build --config configs/inventory.yaml``          (fine, IDs)
``python -m senselm.inventory.build --config configs/inventory_coarse.yaml``   (coarse, counts)

1. cluster source senses under the configured granularity policy (``policy.py``);
2. aggregate each cluster into one canonical sense that lists every contributing
   ``(source, source_id)`` (provenance, so licensed subsets can be excluded later);
3. order senses within each lemma by a frequency proxy (docs/DECISIONS.md D-021):
   WordNet-attested senses first in WordNet sense order (SemCor-frequency based), POS
   interleaved with the lemma's dominant POS first; then Wiktionary-only senses in
   Wiktionary order (top-level senses before their subsenses); senses tagged
   obsolete/archaic/rare/dated/historical last;
4. assign token IDs append-only against the previous registry (``ids.py``) and refuse to
   write if the permanence invariant would be violated. The previous registry is
   snapshotted to ``registry_history/`` before it is replaced.
"""

from __future__ import annotations

import logging
import shutil
from typing import Any

import polars as pl

from senselm.common import Provenance, dump_json, load_config, resolve, stage_args, write_table
from senselm.inventory.align import is_header
from senselm.inventory.ids import Layout, assign_ids, check_append_only
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
    s = senses.join(clusters.select("source_id", "cluster"), on="source_id", how="inner")
    s = s.with_columns(
        pl.when(pl.col("source") == "oewn")
        .then(pl.col("sense_rank").cast(pl.Float64))
        .otherwise(pl.col("source_id").replace_strict(wrank, default=None, return_dtype=pl.Float64))
        .alias("rank"),
        pl.when(pl.col("source") == "oewn").then(0).otherwise(1).alias("src_order"),
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
        # gloss: WordNet's (concise, uniform style) when present, else the first Wiktionary one
        pl.col("gloss").first().alias("gloss"),
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
        pl.col("source_version").unique().sort().alias("source_versions"),
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
    inv = inv.with_columns(
        pl.when(pl.col("has_oewn"))
        .then(0)
        .when(~pl.col("rare"))
        .then(1)
        .otherwise(2)
        .alias("tier"),
        pl.when(pl.col("has_oewn"))
        .then(pl.col("oewn_rank"))
        .otherwise(pl.col("wikt_rank"))
        .fill_null(1e9)
        .alias("rank_key"),
    )
    return inv.sort(
        ["lemma", "tier", "rank_key", "pos_dominance", "wikt_line", "canonical_key"],
        nulls_last=True,
    ).with_columns(pl.int_range(pl.len()).over("lemma").alias("order_key"))


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    prov = Provenance(stage="inventory.build", config=cfg)
    ins, outs = cfg["inputs"], cfg["outputs"]
    for k in ("senses", "pairs", "dedupe_map"):
        prov.add_input(ins[k])
    senses = pl.read_parquet(resolve(ins["senses"]), columns=SENSE_COLS)
    pairs = pl.read_parquet(resolve(ins["pairs"]))
    dedupe = pl.read_parquet(resolve(ins["dedupe_map"]))
    clusters, pstats = assign_clusters(
        senses.select(
            "source", "source_id", "entry_key", "gloss", "gloss_path", "line_no", "sense_rank"
        ),
        pairs,
        cfg["policy"],
        dedupe,
    )
    LOG.info("policy %s: %s", cfg["policy"]["name"], pstats)
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
        reg, lem, istats = assign_ids(
            inv.select(
                "canonical_key",
                "lemma",
                "upos",
                pl.col("members").list.eval(pl.element().struct.field("source_id")),
                "gloss",
                "order_key",
            ),
            prev,
            prev_lem,
            cfg["build_id"],
            layout,
        )
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
            shutil.copy2(reg_path, hist / f"id_registry.{seq:04d}.{prev_build}.parquet")
        reg_prov = Provenance(
            stage="inventory.build.registry", config=cfg, extra={"layout": layout.to_dict()}
        )
        reg_prov.source_versions = prov.source_versions
        write_table(reg.to_arrow(), reg_path, reg_prov, stats=istats)
        write_table(lem.to_arrow(), lem_path, reg_prov)
        if outs.get("registry_commit_dir"):  # durable, git-tracked copy (D-025)
            commit_dir = resolve(outs["registry_commit_dir"])
            commit_dir.mkdir(parents=True, exist_ok=True)
            for src in (reg_path, lem_path):
                shutil.copy2(src, commit_dir / src.name)
                shutil.copy2(
                    src.with_name(src.name + ".meta.json"), commit_dir / (src.name + ".meta.json")
                )
    inv = inv.drop("tier", "rank_key", "pos_dominance", "wikt_line")
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
