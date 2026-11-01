"""Tune and evaluate the aligner (full granularity policy, matching included) on the gold set.

``python -m senselm.eval.alignment_eval --config configs/alignment_eval.yaml``

For each policy config (fine, coarse):

* sweep ``auto_merge_threshold`` on the *dev* half of the gold pairs and pick the value that
  maximises the policy's ``tune_metric`` (``f0.5`` for the fine policy: a false merge destroys a
  distinction irreversibly, a false split only duplicates a token; ``f1`` for coarse); ties go
  to the higher threshold, i.e. precision;
* report precision / recall / F1 on the *test* half at that threshold, and on all labelled
  pairs at the threshold actually configured in the policy file (the numbers quoted in the
  phase report), both as plain sample estimates and Horvitz-Thompson-weighted estimates for
  the population of all candidate pairs (see alignment_gold.py for the sampling design).

A gold pair counts as "merged" when both senses end up in the same canonical sense. Positive
labels: fine = ``same``; coarse = ``same`` or ``related``.

Only the alignment groups that contain gold pairs are re-clustered (policy decisions are
local to a lemma group), which makes the sweep fast.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from senselm.common import dump_json, load_config, resolve, setup_logging
from senselm.eval.alignment_gold import load_gold, score
from senselm.eval.embedding_selection import split_of
from senselm.inventory.policy import assign_clusters, pair_merged

LOG = logging.getLogger("senselm.eval.alignment_eval")
POSITIVE = {"fine": ("same",), "coarse": ("same", "related")}


def fbeta(p: float, r: float, beta: float) -> float:
    if np.isnan(p) or np.isnan(r) or p + r == 0:
        return float("nan")
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def gold_frame(cfg: dict[str, Any]) -> pl.DataFrame:
    gold = load_gold(resolve(cfg["gold"]["path"]))
    ids = gold["pair_id"].str.split("|")
    return gold.with_columns(
        ids.list.get(0).alias("oewn_id"),
        ids.list.get(1).alias("wikt_id"),
        pl.col("pair_id")
        .map_elements(
            lambda p: split_of(p, float(cfg["gold"]["dev_fraction"])), return_dtype=pl.String
        )
        .alias("split"),
    )


def local_problem(cfg: dict[str, Any], gold: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    pairs = pl.read_parquet(resolve(cfg["pairs"]))
    groups = (
        pairs.join(gold.select("oewn_id", "wikt_id"), on=["oewn_id", "wikt_id"])
        .select("group_key")
        .unique()
    )
    sub = pairs.join(groups, on="group_key")
    ids = set(sub["oewn_id"]) | set(sub["wikt_id"])
    senses = pl.read_parquet(
        resolve(cfg["senses"]),
        columns=[
            "source",
            "source_id",
            "entry_key",
            "gloss",
            "gloss_path",
            "line_no",
            "sense_rank",
        ],
    ).filter(pl.col("source_id").is_in(list(ids)))
    return senses, sub


def merged_for(
    gold: pl.DataFrame, senses: pl.DataFrame, pairs: pl.DataFrame, policy: dict[str, Any]
) -> np.ndarray:
    clusters, _ = assign_clusters(senses, pairs, policy)
    g = gold.select("oewn_id", "wikt_id")
    return pair_merged(g, clusters).to_numpy()


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    gold = gold_frame(cfg)
    senses, pairs = local_problem(cfg, gold)
    LOG.info(
        "gold: %d pairs; local problem: %d senses, %d pairs",
        gold.height,
        senses.height,
        pairs.height,
    )
    grid = np.round(np.arange(*cfg["sweep"]), 3)
    out: dict[str, Any] = {
        "n_gold": gold.height,
        "label_counts": dict(gold["label"].value_counts().iter_rows()),
        "ambiguous": int(gold["ambiguous"].sum()),
        "policies": {},
    }
    for pcfg_path in cfg["policies"]:
        pcfg = load_config(resolve(pcfg_path))
        policy, name = pcfg["policy"], pcfg["policy"]["name"]
        positive = POSITIVE[policy.get("gold_positive", name)]
        dev = gold.filter(pl.col("split") == "dev")
        sweep = []
        for t in grid:
            m = merged_for(dev, senses, pairs, {**policy, "auto_merge_threshold": float(t)})
            s = score(dev.with_columns(pl.Series("merged", m)), "merged", positive, weighted=False)
            sweep.append(
                {"threshold": float(t), **{k: s[k] for k in ("precision", "recall", "f1")}}
            )
        metric = policy.get("tune_metric", "f1")
        for r in sweep:
            r["f0.5"] = fbeta(r["precision"], r["recall"], 0.5)
        best = max(sweep, key=lambda r: (np.nan_to_num(r[metric]), r["threshold"]))
        best = {**best, "metric": metric}
        test = gold.filter(pl.col("split") == "test")
        m_test = merged_for(
            test, senses, pairs, {**policy, "auto_merge_threshold": best["threshold"]}
        )
        res: dict[str, Any] = {
            "configured_threshold": policy["auto_merge_threshold"],
            "dev_best": best,
            "test_at_dev_best": score(
                test.with_columns(pl.Series("merged", m_test)), "merged", positive, False
            ),
            "sweep_dev": sweep,
        }
        m_all = merged_for(gold, senses, pairs, policy)
        gm = gold.with_columns(pl.Series("merged", m_all))
        res["configured"] = {
            "all_sample": score(gm, "merged", positive, weighted=False),
            "all_weighted": score(gm, "merged", positive, weighted=True),
            "test_sample": score(gm.filter(pl.col("split") == "test"), "merged", positive, False),
            "unambiguous_sample": score(
                gm.filter(pl.col("ambiguous") == 0), "merged", positive, False
            ),
            "stratum_U": score(gm.filter(pl.col("stratum") == "U"), "merged", positive, False),
            "stratum_G": score(gm.filter(pl.col("stratum") == "G"), "merged", positive, False),
        }
        out["policies"][name] = res
        LOG.info(
            "%s: dev-best thr %.2f (dev F1 %.3f) -> test P %.3f R %.3f F1 %.3f | configured %.2f: all P %.3f R %.3f",
            name,
            best["threshold"],
            best["f1"],
            res["test_at_dev_best"]["precision"],
            res["test_at_dev_best"]["recall"],
            res["test_at_dev_best"]["f1"],
            policy["auto_merge_threshold"],
            res["configured"]["all_sample"]["precision"],
            res["configured"]["all_sample"]["recall"],
        )
    dump_json(out, resolve(cfg["output"]))
    resolve(cfg["output"]).with_suffix(".md").write_text(to_markdown(out), encoding="utf-8")
    return out


def _fmt(s: dict[str, Any]) -> str:
    return f"{s['precision']:.3f} / {s['recall']:.3f} / {s['f1']:.3f}"


def to_markdown(res: dict[str, Any]) -> str:
    lines = [
        f"Gold: {res['n_gold']} pairs {res['label_counts']}, {res['ambiguous']} flagged ambiguous.",
        "",
        "| policy | configured threshold | all gold P/R/F1 (sample) | all gold P/R/F1 (HT-weighted) | test half P/R/F1 | unambiguous P/R/F1 | stratum U | stratum G | dev-optimal threshold (metric) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in res["policies"].items():
        c = r["configured"]
        lines.append(
            f"| {name} | {r['configured_threshold']:.2f} | {_fmt(c['all_sample'])} | {_fmt(c['all_weighted'])} | "
            f"{_fmt(c['test_sample'])} | {_fmt(c['unambiguous_sample'])} | {_fmt(c['stratum_U'])} | "
            f"{_fmt(c['stratum_G'])} | {r['dev_best']['threshold']:.2f} ({r['dev_best']['metric']}) |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    setup_logging()
    res = run(load_config(args.config))
    print(to_markdown(res))


if __name__ == "__main__":
    main()
