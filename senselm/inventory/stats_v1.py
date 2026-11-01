"""Statistics for the Phase-2 report (inventory v1.0).

``python -m senselm.inventory.stats_v1 --config configs/stats_v1.yaml``

Writes ``docs/phase2_stats.json`` (every number) and ``docs/phase2_stats.md`` (the tables that
docs/phase2_report.md quotes): vocabulary size by source, domain, tier, POS and licence; the
structural vocabulary; ID-space utilisation; storage; and embedding-parameter estimates for a
naive dense table versus factorised / low-rank schemes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import polars as pl

from senselm.common import dump_json, load_config, resolve, stage_args
from senselm.inventory.ids import Layout, ledger_paths

LOG = logging.getLogger("senselm.inventory.stats_v1")
INV_COLS = [
    "token_id",
    "lemma",
    "lemma_id",
    "sense_index",
    "upos",
    "domain",
    "sources",
    "licences",
    "has_oewn",
    "has_wiktionary",
    "rare",
    "primary_source",
    "n_members",
]
BUCKETS = [(1, 1), (2, 2), (3, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, 1024)]


def _size(paths: list[Path]) -> int:
    total = 0
    for p in paths:
        if p.is_dir():
            total += sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        elif p.exists():
            total += p.stat().st_size
    return total


def _glob(pattern: str) -> list[Path]:
    root = resolve(".")
    return sorted(root.glob(pattern))


def tier_of(inv: pl.DataFrame) -> pl.Expr:
    return (
        pl.when(pl.col("has_oewn"))
        .then(pl.lit("0 WordNet"))
        .when(pl.col("has_wiktionary") & ~pl.col("rare"))
        .then(pl.lit("1 Wiktionary"))
        .when(pl.col("has_wiktionary"))
        .then(pl.lit("2 Wiktionary (rare/obsolete)"))
        .when(pl.col("primary_source") == "wikidata_lexemes")
        .then(pl.lit("3 Wikidata lexemes only"))
        .otherwise(pl.lit("4 concept / domain sources"))
    )


def vocabulary(cfg: dict[str, Any], inv: pl.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {"canonical_senses": inv.height, "lemmas": inv["lemma"].n_unique()}
    order = cfg["source_order"]
    exploded = inv.select("token_id", "sources").explode("sources")
    covered = dict(exploded.group_by("sources").len().iter_rows())
    primary = dict(inv.group_by("primary_source").len().iter_rows())
    att = pl.read_parquet(resolve(cfg["attachments"])) if cfg.get("attachments") else None
    via = (
        {(s, v): n for s, v, n in att.group_by("source", "via").len().iter_rows()}
        if att is not None
        else {}
    )
    senses = pl.read_parquet(resolve(cfg["senses_norm"]), columns=["source"])
    ingested = dict(senses.group_by("source").len().iter_rows())
    rows = []
    for src in order:
        if src not in ingested and src not in covered:
            continue
        rows.append(
            {
                "source": src,
                "senses_ingested": ingested.get(src, 0),
                "attached_xref": via.get((src, "xref"), 0),
                "attached_cosine": via.get((src, "cosine"), 0),
                "new_clusters": via.get((src, "new"), 0),
                "canonical_senses_with_source": covered.get(src, 0),
                "canonical_senses_first_from_source": primary.get(src, 0),
            }
        )
    out["by_source"] = rows
    out["by_tier"] = [
        {"tier": t, "senses": n}
        for t, n in inv.with_columns(tier_of(inv).alias("tier"))
        .group_by("tier")
        .len()
        .sort("tier")
        .iter_rows()
    ]
    dom = inv.with_columns(pl.col("domain").fill_null("(general)")).group_by("domain").len()
    out["by_domain"] = [
        {"domain": d, "senses": n} for d, n in dom.sort("len", descending=True).iter_rows()
    ]
    out["n_domains"] = int(inv["domain"].drop_nulls().n_unique())
    out["by_upos"] = [
        {"upos": u, "senses": n}
        for u, n in inv.group_by("upos").len().sort("len", descending=True).iter_rows()
    ]
    out["multi_source_senses"] = int((inv["sources"].list.len() > 1).sum())
    # licensing: what survives if the senses only some sources attest are dropped
    scen = []
    for name, drop in cfg.get("licence_scenarios", {}).items():
        keep = inv.filter(pl.col("sources").list.set_difference(drop).list.len() > 0).height
        scen.append(
            {
                "scenario": name,
                "exclude": drop,
                "senses_kept": keep,
                "pct": round(100 * keep / inv.height, 2),
            }
        )
    out["licence_scenarios"] = scen
    per = inv.group_by("lemma").len()
    dist = []
    for lo, hi in BUCKETS:
        k = per.filter(pl.col("len").is_between(lo, hi)).height
        dist.append(
            {
                "senses": str(lo) if lo == hi else f"{lo}-{hi}",
                "lemmas": k,
                "pct": round(100 * k / per.height, 2),
            }
        )
    out["senses_per_lemma"] = {
        "mean": round(float(per["len"].mean()), 3),
        "median": float(per["len"].median()),
        "max": int(per["len"].max()),
        "distribution": dist,
    }
    return out


def id_space(
    cfg: dict[str, Any],
    inv: pl.DataFrame,
    reg: pl.DataFrame,
    lem: pl.DataFrame,
    structural: pl.DataFrame,
) -> dict[str, Any]:
    lay = Layout(**cfg["layout"])
    lexical_lemma_slots = lay.max_lemma_id  # ids 1..max (0 never used)
    issued = reg.height
    created = dict(reg.group_by("created_in").len().iter_rows())
    status = dict(reg.group_by("status").len().iter_rows())
    max_idx = (
        reg.group_by("lemma_id")
        .agg(pl.col("sense_index").max().alias("m"))
        .sort("m", descending=True)
    )
    top = max_idx.head(10).join(lem.select("lemma_id", "lemma"), on="lemma_id", how="left")
    lexical_space = lexical_lemma_slots << lay.sense_bits
    return {
        "layout": lay.to_dict(),
        "lemma_ids_issued": lem.height,
        "lemma_ids_active": int((lem["status"] == "active").sum()),
        "lemma_slots_lexical": lexical_lemma_slots,
        "lemma_slot_utilisation_pct": round(100 * lem.height / lexical_lemma_slots, 3),
        "token_ids_issued": issued,
        "token_ids_by_status": status,
        "token_ids_by_build": created,
        "lexical_id_space": lexical_space,
        "token_id_utilisation_pct": round(100 * issued / lexical_space, 4),
        "max_sense_index": int(reg["sense_index"].max()),
        "sense_index_capacity": lay.max_sense_index + 1,
        "lemmas_over_half_capacity": int((max_idx["m"] >= (lay.max_sense_index + 1) // 2).sum()),
        "top_sense_index": [
            {"lemma": r["lemma"], "max_sense_index": r["m"]} for r in top.iter_rows(named=True)
        ],
        "structural": {
            "active": int((structural["status"] == "active").sum()),
            "slots_used": int(structural["slot"].n_unique()),
            "slots_reserved": lay.reserved_lemma_slots,
            "by_category": dict(structural.group_by("category").len().sort("category").iter_rows()),
            "first_id": f"0x{lay.structural_base:08X}",
        },
        "inventory_token_ids_active": inv.height,
    }


def storage(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, patterns in cfg["storage"].items():
        paths = [p for pat in patterns for p in _glob(pat)]
        rows.append({"artefact": name, "bytes": _size(paths), "files": len(paths)})
    return rows


def embedding_estimates(
    cfg: dict[str, Any], vocab: dict[str, Any], ids: dict[str, Any]
) -> dict[str, Any]:
    """Parameter counts of the input embedding (tied output) for a few model widths.

    * naive: one dense row per token (lexical + structural): V x d;
    * factorised: E_lemma (L x d) + E_sense_index (1024 x d) + E_domain (D x d) — senses of a
      lemma differ only through their index and domain (cheap, but too coarse on its own);
    * factorised + low-rank residual: the above plus a per-sense code (V x r) projected by
      (r x d): every sense keeps its own vector at rank r;
    * low-rank (ALBERT-style): V x r + r x d;
    * hybrid: dense rows for the K most frequent senses (frequency known after Phase 4 tagging)
      plus a low-rank tail: K x d + (V - K) x r + r x d.
    """
    e = cfg["embedding"]
    v = vocab["canonical_senses"] + ids["structural"]["active"]
    lemmas = vocab["lemmas"] + ids["structural"]["slots_used"]
    senses = ids["sense_index_capacity"]
    domains = vocab["n_domains"] + 1
    bpp = int(e.get("bytes_per_param", 2))
    rows = []
    for d in e["dims"]:
        naive = v * d
        fact = lemmas * d + senses * d + domains * d
        rows.append({"d": d, "scheme": "naive V x d", "params": naive})
        rows.append({"d": d, "scheme": "E_lemma + E_sense_index + E_domain", "params": fact})
        for r in e["ranks"]:
            if r < d:
                rows.append(
                    {
                        "d": d,
                        "scheme": f"E_lemma + E_domain + low-rank sense residual (r={r})",
                        "params": fact - senses * d + v * r + r * d,
                    }
                )
                rows.append(
                    {"d": d, "scheme": f"low-rank V x r + r x d (r={r})", "params": v * r + r * d}
                )
        hr = int(e.get("hybrid_rank", 64))
        for k in e.get("dense_top_k", []):
            if k < v:
                rows.append(
                    {
                        "d": d,
                        "scheme": f"dense top {k // 1000}k senses + low-rank tail (r={hr})",
                        "params": k * d + (v - k) * hr + hr * d,
                    }
                )
    for r in rows:
        r["params_M"] = round(r["params"] / 1e6, 1)
        r["memory_GB"] = round(r["params"] * bpp / 1e9, 2)
        naive = next(x["params"] for x in rows if x["d"] == r["d"] and x["scheme"] == "naive V x d")
        r["vs_naive"] = round(r["params"] / naive, 3)
    return {
        "vocab_rows": v,
        "lemma_rows": lemmas,
        "sense_index_rows": senses,
        "domain_rows": domains,
        "bytes_per_param": bpp,
        "table": rows,
    }


def compute(cfg: dict[str, Any]) -> dict[str, Any]:
    inv = pl.read_parquet(resolve(cfg["inventory"]), columns=INV_COLS)
    reg = pl.read_parquet(
        resolve(cfg["registry"]),
        columns=["token_id", "lemma_id", "sense_index", "status", "created_in"],
    )
    lem = pl.read_parquet(resolve(cfg["lemma_registry"]))
    structural = pl.read_parquet(resolve(cfg["structural_registry"]))
    vocab = vocabulary(cfg, inv)
    ids = id_space(cfg, inv, reg, lem, structural)
    ledger = [
        p
        for name in ("id_registry", "lemma_registry", "structural_registry")
        for p in ledger_paths(resolve(cfg["ledger_dir"]), name)
    ]
    return {
        "build": cfg["build_id"],
        "vocabulary": vocab,
        "id_space": ids,
        "storage": storage(cfg),
        "ledger_files": [{"file": p.name, "bytes": p.stat().st_size} for p in ledger],
        "embedding": embedding_estimates(cfg, vocab, ids),
        "total_vocabulary": vocab["canonical_senses"] + ids["structural"]["active"],
    }


def _fmt(n: float) -> str:
    return f"{n:,}" if isinstance(n, int) else f"{n:,.2f}"


def _mb(b: int) -> str:
    if b < 1e6:
        return f"{b / 1e3:,.0f} kB"
    return f"{b / 1e6:,.1f} MB" if b < 1e9 else f"{b / 1e9:,.2f} GB"


def markdown(st: dict[str, Any]) -> str:
    v, ids, emb = st["vocabulary"], st["id_space"], st["embedding"]
    out = [
        f"# Phase 2 statistics (inventory {st['build']})",
        "",
        "Generated by `python -m senselm.inventory.stats_v1 --config configs/stats_v1.yaml`.",
        "",
        f"Total vocabulary: **{st['total_vocabulary']:,}** tokens = {v['canonical_senses']:,} sense tokens "
        f"({v['lemmas']:,} lemmas) + {ids['structural']['active']:,} structural tokens.",
        "",
        "## By source",
        "",
        "| source | senses ingested | attached (xref) | attached (cosine) | new clusters | "
        "canonical senses with source | first from source |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in v["by_source"]:
        out.append(
            f"| {r['source']} | {_fmt(r['senses_ingested'])} | {_fmt(r['attached_xref'])} | "
            f"{_fmt(r['attached_cosine'])} | {_fmt(r['new_clusters'])} | "
            f"{_fmt(r['canonical_senses_with_source'])} | {_fmt(r['canonical_senses_first_from_source'])} |"
        )
    out += [
        "",
        "OEWN and Wiktionary are clustered by the stage-1 (v0.1) policy; the attachment columns apply "
        "to stage-2 sources. A canonical sense can have members from several sources "
        f"({v['multi_source_senses']:,} do).",
        "",
        "## By tier (sense-index order)",
        "",
        "| tier | senses |",
        "|---|---|",
    ]
    out += [f"| {r['tier']} | {_fmt(r['senses'])} |" for r in v["by_tier"]]
    out += ["", "## By domain", "", "| domain | senses |", "|---|---|"]
    out += [f"| {r['domain']} | {_fmt(r['senses'])} |" for r in v["by_domain"][:40]]
    if len(v["by_domain"]) > 40:
        rest = sum(r["senses"] for r in v["by_domain"][40:])
        out.append(f"| ({len(v['by_domain']) - 40} more domains) | {_fmt(rest)} |")
    out += ["", "## By UPOS", "", "| UPOS | senses |", "|---|---|"]
    out += [f"| {r['upos']} | {_fmt(r['senses'])} |" for r in v["by_upos"]]
    out += [
        "",
        "## Licence scenarios",
        "",
        "Senses that keep at least one member when the senses of the listed "
        "sources are removed (their lemma and gloss may then need to come from another member).",
        "",
        "| scenario | excluded sources | senses kept | % |",
        "|---|---|---|---|",
    ]
    out += [
        f"| {r['scenario']} | {', '.join(r['exclude'])} | {_fmt(r['senses_kept'])} | {r['pct']} |"
        for r in v["licence_scenarios"]
    ]
    spl = v["senses_per_lemma"]
    out += [
        "",
        "## Senses per lemma",
        "",
        f"Mean {spl['mean']}, median {spl['median']}, max {spl['max']}.",
        "",
        "| senses | lemmas | % |",
        "|---|---|---|",
    ]
    out += [f"| {r['senses']} | {_fmt(r['lemmas'])} | {r['pct']} |" for r in spl["distribution"]]
    s = ids["structural"]
    out += [
        "",
        "## ID space",
        "",
        "| quantity | value |",
        "|---|---|",
        f"| lemma ids issued / lexical lemma slots | {ids['lemma_ids_issued']:,} / "
        f"{ids['lemma_slots_lexical']:,} ({ids['lemma_slot_utilisation_pct']}%) |",
        f"| token ids issued / lexical id space | {ids['token_ids_issued']:,} / {ids['lexical_id_space']:,} "
        f"({ids['token_id_utilisation_pct']}%) |",
        f"| token ids by status | {', '.join(f'{k} {n:,}' for k, n in sorted(ids['token_ids_by_status'].items()))} |",
        f"| token ids by build | {', '.join(f'{k} {n:,}' for k, n in sorted(ids['token_ids_by_build'].items()))} |",
        f"| max sense index / capacity | {ids['max_sense_index']} / {ids['sense_index_capacity']} |",
        f"| lemmas using more than half the sense-index capacity | {ids['lemmas_over_half_capacity']} |",
        f"| structural tokens / reserved ids | {s['active']} / {s['slots_reserved'] * ids['sense_index_capacity']:,} "
        f"({s['slots_used']} of {s['slots_reserved']} reserved slots) |",
        "",
        "Most senses per lemma (max sense index): "
        + ", ".join(f"{r['lemma']} {r['max_sense_index']}" for r in ids["top_sense_index"])
        + ".",
        "",
        "## Storage",
        "",
        "| artefact | size | files |",
        "|---|---|---|",
    ]
    out += [f"| {r['artefact']} | {_mb(r['bytes'])} | {r['files']} |" for r in st["storage"]]
    out += [
        "",
        "Git-tracked ledger files: "
        + ", ".join(f"`{r['file']}` {_mb(r['bytes'])}" for r in st["ledger_files"])
        + ".",
        "",
        "## Embedding parameters",
        "",
        f"Rows: V = {emb['vocab_rows']:,} tokens, L = {emb['lemma_rows']:,} lemmas (+ structural slots), "
        f"{emb['sense_index_rows']} sense indices, {emb['domain_rows']} domains (incl. none); "
        f"memory at {emb['bytes_per_param']} bytes/parameter (bf16).",
        "",
        "| d | scheme | parameters | memory | vs naive |",
        "|---|---|---|---|---|",
    ]
    out += [
        f"| {r['d']} | {r['scheme']} | {r['params_M']:,} M | {r['memory_GB']} GB | {r['vs_naive']} |"
        for r in emb["table"]
    ]
    return "\n".join(out) + "\n"


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    st = compute(cfg)
    dump_json(st, resolve(cfg["output_json"]))
    resolve(cfg["output_md"]).write_text(markdown(st), encoding="utf-8")
    LOG.info("wrote %s", cfg["output_md"])
    return st


def main() -> None:
    args = stage_args("Phase-2 statistics")
    print(json.dumps(run(load_config(args.config))["vocabulary"]["by_source"], indent=1))


if __name__ == "__main__":
    main()
