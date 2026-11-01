"""Statistics for the phase-1 report.

``python -m senselm.inventory.stats --config configs/stats.yaml``

Writes ``docs/phase1_stats.json`` (every number) and ``docs/phase1_stats.md`` (tables that
docs/phase1_report.md quotes).
"""

from __future__ import annotations

import json
import logging
from typing import Any

import polars as pl

from senselm.common import dump_json, load_config, resolve, stage_args

LOG = logging.getLogger("senselm.inventory.stats")
BUCKETS = [(1, 1), (2, 2), (3, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, 10_000)]


def _sidecar(path: str) -> dict[str, Any]:
    p = resolve(path)
    side = p.with_name(p.name + ".meta.json")
    return json.loads(side.read_text())["stats"] if side.exists() else {}


def per_lemma(inv: pl.DataFrame) -> dict[str, Any]:
    per = inv.group_by("lemma").agg(pl.len().alias("n"))
    total = per.height
    dist = []
    for lo, hi in BUCKETS:
        k = per.filter(pl.col("n").is_between(lo, hi)).height
        label = f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10_000 else f">{lo - 1}")
        dist.append({"senses": label, "lemmas": k, "pct": round(100 * k / total, 2)})
    top = (
        inv.group_by("lemma")
        .agg(
            pl.len().alias("n"),
            pl.col("upos").value_counts(sort=True).alias("by_pos"),
            pl.col("has_oewn").sum().alias("with_wordnet"),
        )
        .sort(["n", "lemma"], descending=[True, False])
        .head(50)
    )
    top_rows = [
        {
            "lemma": r["lemma"],
            "senses": r["n"],
            "with_wordnet": r["with_wordnet"],
            "by_pos": {d["upos"]: d["count"] for d in r["by_pos"]},
        }
        for r in top.iter_rows(named=True)
    ]
    return {
        "lemmas": total,
        "mean": round(per["n"].mean(), 3),
        "median": float(per["n"].median()),
        "p99": float(per["n"].quantile(0.99)),
        "max": int(per["n"].max()),
        "distribution": dist,
        "top50": top_rows,
    }


def compute(cfg: dict[str, Any]) -> dict[str, Any]:
    fine = pl.read_parquet(resolve(cfg["inventory"]))
    coarse = pl.read_parquet(
        resolve(cfg["inventory_coarse"]), columns=["lemma", "sources", "has_oewn", "has_wiktionary"]
    )
    norm = pl.read_parquet(resolve(cfg["senses_norm"]), columns=["source", "upos", "examples"])
    pairs = pl.read_parquet(
        resolve(cfg["pairs"]), columns=["oewn_id", "wikt_id", "cosine", "case_fallback"]
    )
    out: dict[str, Any] = {}

    out["raw_ingest"] = {
        "wiktionary": _sidecar(cfg["raw_wiktionary"]),
        "oewn": _sidecar(cfg["raw_oewn"]),
    }
    out["normalise"] = _sidecar(cfg["senses_norm"])
    out["input_senses"] = dict(norm.group_by("source").len().sort("source").iter_rows())
    out["forms_rows"] = pl.scan_parquet(resolve(cfg["forms"])).select(pl.len()).collect().item()
    out["variants_rows"] = (
        pl.scan_parquet(resolve(cfg["variants"])).select(pl.len()).collect().item()
    )

    both = fine.filter(pl.col("has_oewn") & pl.col("has_wiktionary")).height
    out["fine"] = {
        "canonical_senses": fine.height,
        "lemmas": fine["lemma"].n_unique(),
        "oewn_only": fine.filter(pl.col("has_oewn") & ~pl.col("has_wiktionary")).height,
        "wiktionary_only": fine.filter(~pl.col("has_oewn") & pl.col("has_wiktionary")).height,
        "both_sources": both,
        "with_oewn": fine.filter(pl.col("has_oewn")).height,
        "with_wiktionary": fine.filter(pl.col("has_wiktionary")).height,
        "with_example": fine.filter(pl.col("has_example")).height,
        "pct_with_example": round(100 * fine["has_example"].mean(), 2),
        "pct_with_example_by_composition": {
            "both": round(
                100
                * fine.filter(pl.col("has_oewn") & pl.col("has_wiktionary"))["has_example"].mean(),
                2,
            ),
            "oewn_only": round(
                100
                * fine.filter(pl.col("has_oewn") & ~pl.col("has_wiktionary"))["has_example"].mean(),
                2,
            ),
            "wiktionary_only": round(
                100
                * fine.filter(~pl.col("has_oewn") & pl.col("has_wiktionary"))["has_example"].mean(),
                2,
            ),
        },
        "abbreviation_senses": fine.filter(pl.col("is_abbrev")).height,
        "multi_member": fine.filter(pl.col("n_members") > 1).height,
    }
    out["coarse"] = {
        "canonical_senses": coarse.height,
        "lemmas": coarse["lemma"].n_unique(),
        "both_sources": coarse.filter(pl.col("has_oewn") & pl.col("has_wiktionary")).height,
    }
    out["no_merge_upper_bound"] = int(norm.height)
    out["overlap"] = {
        "candidate_pairs": pairs.height,
        "case_fallback_pairs": int(pairs["case_fallback"].sum()),
        "oewn_senses_with_candidates": pairs["oewn_id"].n_unique(),
        "wikt_senses_with_candidates": pairs["wikt_id"].n_unique(),
        "oewn_senses_merged": both,
        "pct_oewn_senses_merged": round(100 * both / out["input_senses"]["oewn"], 2),
        "pct_wikt_senses_merged": round(100 * both / out["input_senses"]["wiktionary"], 2),
    }
    out["per_lemma"] = per_lemma(fine)
    out["upos"] = dict(fine.group_by("upos").len().sort("len", descending=True).iter_rows())
    out["upos_by_composition"] = [
        {
            "upos": r["upos"],
            "both": r["both"],
            "oewn_only": r["oewn_only"],
            "wiktionary_only": r["wikt_only"],
        }
        for r in fine.group_by("upos")
        .agg(
            (pl.col("has_oewn") & pl.col("has_wiktionary")).sum().alias("both"),
            (pl.col("has_oewn") & ~pl.col("has_wiktionary")).sum().alias("oewn_only"),
            (~pl.col("has_oewn") & pl.col("has_wiktionary")).sum().alias("wikt_only"),
        )
        .sort("upos")
        .iter_rows(named=True)
    ]
    tags = (
        fine.select(pl.col("tags").explode())
        .drop_nulls()
        .group_by("tags")
        .len()
        .sort(["len", "tags"], descending=[True, False])
    )
    out["tags_top40"] = dict(tags.head(40).iter_rows())
    out["senses_with_any_tag"] = fine.filter(pl.col("tags").list.len() > 0).height
    reg = [
        "obsolete",
        "archaic",
        "dated",
        "rare",
        "historical",
        "slang",
        "informal",
        "colloquial",
        "vulgar",
        "derogatory",
        "offensive",
        "humorous",
        "nonstandard",
        "dialectal",
        "UK",
        "US",
        "Australia",
        "Canada",
        "Ireland",
        "India",
        "Scotland",
        "New-Zealand",
        "South-Africa",
    ]
    out["register_and_regional_tags"] = {
        t: fine.filter(pl.col("tags").list.contains(t)).height for t in reg
    }
    dom = (
        fine.drop_nulls("domain")
        .group_by("domain")
        .len()
        .sort(["len", "domain"], descending=[True, False])
    )
    out["senses_with_domain"] = fine.filter(pl.col("domain").is_not_null()).height
    out["domains_top30"] = dict(dom.head(30).iter_rows())
    lex = (
        fine.drop_nulls("lexfile")
        .group_by("lexfile")
        .len()
        .sort(["len", "lexfile"], descending=[True, False])
    )
    out["supersenses"] = dict(lex.iter_rows())
    lic = fine.with_columns(pl.col("licences").list.join("+").alias("lic")).group_by("lic").len()
    out["licence_subsets"] = dict(lic.sort("lic").iter_rows())
    out["namenet_senses"] = fine.filter(pl.col("tags").list.contains("oewn:namenet")).height

    if "token_id" in fine.columns:
        reg_df = pl.read_parquet(resolve(cfg["registry"]))
        lem_df = pl.read_parquet(resolve(cfg["lemma_registry"]))
        out["ids"] = {
            "active_token_ids": reg_df.filter(pl.col("status") == "active").height,
            "tombstones": reg_df.filter(pl.col("status") == "deprecated").height,
            "lemma_ids": lem_df.height,
            "max_lemma_id": int(lem_df["lemma_id"].max()),
            "max_sense_index": int(reg_df["sense_index"].max()),
            "lemma_with_max_sense_index": reg_df.sort("sense_index", descending=True)["lemma"][0],
            "lemma_id_space_used_pct": round(
                100 * int(lem_df["lemma_id"].max()) / ((1 << 22) - 4096), 3
            ),
            "sense_index_space_used_pct_at_max": round(
                100 * (int(reg_df["sense_index"].max()) + 1) / 1024, 2
            ),
            "registry_bytes": resolve(cfg["registry"]).stat().st_size,
            "inventory_bytes": resolve(cfg["inventory"]).stat().st_size,
        }
    for k in ("alignment_eval", "embedding_selection", "judge_estimate"):
        p = resolve(cfg[k])
        if p.exists():
            out[k] = json.loads(p.read_text())
    out["build_stats"] = {
        "fine": _sidecar(cfg["inventory"]),
        "coarse": _sidecar(cfg["inventory_coarse"]),
    }
    return out


def markdown(st: dict[str, Any]) -> str:
    f, c = st["fine"], st["coarse"]
    pl_ = st["per_lemma"]
    lines = [
        "## Senses per lemma (fine policy)",
        "",
        f"Lemmas: {pl_['lemmas']:,}; mean {pl_['mean']}, median {pl_['median']}, p99 {pl_['p99']}, max {pl_['max']}.",
        "",
        "| senses per lemma | lemmas | % |",
        "|---|---|---|",
    ]
    lines += [f"| {d['senses']} | {d['lemmas']:,} | {d['pct']} |" for d in pl_["distribution"]]
    lines += [
        "",
        "## Top 50 most polysemous lemmas (fine policy)",
        "",
        "| # | lemma | senses | WordNet-backed | by POS |",
        "|---|---|---|---|---|",
    ]
    for i, r in enumerate(pl_["top50"], 1):
        pos = ", ".join(f"{k} {v}" for k, v in r["by_pos"].items())
        lines.append(f"| {i} | {r['lemma']} | {r['senses']} | {r['with_wordnet']} | {pos} |")
    lines += [
        "",
        "## POS breakdown (fine)",
        "",
        "| UPOS | total | both sources | OEWN only | Wiktionary only |",
        "|---|---|---|---|---|",
    ]
    for r in sorted(
        st["upos_by_composition"],
        key=lambda r: -(r["both"] + r["oewn_only"] + r["wiktionary_only"]),
    ):
        tot = r["both"] + r["oewn_only"] + r["wiktionary_only"]
        lines.append(
            f"| {r['upos']} | {tot:,} | {r['both']:,} | {r['oewn_only']:,} | {r['wiktionary_only']:,} |"
        )
    lines += [
        "",
        f"## Tags (top 40; {st['senses_with_any_tag']:,} senses carry at least one tag)",
        "",
        "| tag | senses |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v:,} |" for k, v in st["tags_top40"].items()]
    lines += [
        "",
        f"## Domains (top 30; {st['senses_with_domain']:,} senses have a domain)",
        "",
        "| domain | senses |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v:,} |" for k, v in st["domains_top30"].items()]
    lines += [
        "",
        "## WordNet supersenses (lexicographer files) of WordNet-backed senses",
        "",
        "| lexfile | senses |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v:,} |" for k, v in st["supersenses"].items()]
    lines += [
        "",
        "## Licence subsets",
        "",
        "| licences of contributing sources | senses |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v:,} |" for k, v in st["licence_subsets"].items()]
    lines += [
        "",
        f"Fine: {f['canonical_senses']:,} senses; coarse: {c['canonical_senses']:,}; no merging at all: {st['no_merge_upper_bound']:,}.",
        "",
    ]
    return "\n".join(lines)


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    st = compute(cfg)
    dump_json(st, cfg["output_json"])
    resolve(cfg["output_md"]).write_text(markdown(st), encoding="utf-8")
    LOG.info("wrote %s and %s", cfg["output_json"], cfg["output_md"])
    return st


def main() -> None:
    args = stage_args("Phase-1 inventory statistics")
    run(load_config(args.config))


if __name__ == "__main__":
    main()
