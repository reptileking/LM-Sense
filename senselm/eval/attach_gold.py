"""Gold set for stage-2 attachment (blind): render the sample, then evaluate thresholds.

``python -m senselm.eval.attach_gold render --config configs/attach.yaml``
    eval/attach_gold_sample.json (from ``attach --sample``) -> eval/attach_gold.csv, one row per
    sampled new-source sense with its candidate senses (no scores). Label columns:
    ``match`` = the candidate number that is the same sense (``2|5`` when two candidate clusters
    are the same sense, e.g. a WordNet and a Wiktionary cluster that stage 1 kept apart; ``none``
    when no candidate is), ``related`` =
    candidate numbers that are broader/narrower/metonymic, ``ambiguous``, ``note``.

``python -m senselm.eval.attach_gold evaluate --config configs/attach.yaml``
    For each source, sweep the cosine threshold: a gold sense counts as attached to its
    best-scoring candidate if a cross-reference links them or the score reaches the threshold.
    Precision = attached to the labelled match; recall = labelled match exists and was attached.
    (Competition with other senses of the same source for a cluster is ignored: second order.)
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from typing import Any

import numpy as np
import polars as pl

from senselm.common import dump_json, load_config, resolve, setup_logging
from senselm.inventory.align import sense_text
from senselm.inventory.attach import (
    COLS,
    STAGE1_SOURCES,
    identity_keys,
    load_bridge,
    own_key,
    stage1,
    taxon_compatible,
)
from senselm.inventory.embed import embed_texts

LOG = logging.getLogger("senselm.eval.attach_gold")


def _cluster_members(cfg: dict[str, Any], senses: pl.DataFrame) -> dict[str, list[str]]:
    cl1 = stage1(cfg, senses)
    out: dict[str, list[str]] = defaultdict(list)
    for sid, c in cl1.iter_rows():
        out[c].append(sid)
    return out


def render(cfg: dict[str, Any]) -> None:
    sample = json.loads(resolve(cfg["gold"]["sample_output"]).read_text())["samples"]
    senses = pl.read_parquet(resolve(cfg["senses"]), columns=COLS)
    rows = {r["source_id"]: r for r in senses.iter_rows(named=True)}
    members = _cluster_members(cfg, senses)
    out = []
    for s in sample:
        r = rows[s["source_id"]]
        cands = []
        for c in s["candidates"]:
            ms = sorted(members[c], key=lambda m: rows[m]["source"] != "oewn")
            cands.append(
                {
                    "cluster": c,
                    "upos": rows[ms[0]]["upos"],
                    "glosses": [f"[{rows[m]['source'][:4]}] {rows[m]['gloss']}" for m in ms[:2]],
                }
            )
        out.append(
            {
                "source": s["source"],
                "source_id": s["source_id"],
                "lemma": r["lemma"],
                "upos": r["upos"],
                "gloss": r["gloss"],
                "candidates": json.dumps(cands, ensure_ascii=False),
                "match": None,
                "related": None,
                "ambiguous": None,
                "note": None,
            }
        )
    path = resolve(cfg["gold"]["path"])
    if path.exists() and pl.read_csv(path, infer_schema_length=0)["match"].drop_nulls().len():
        raise FileExistsError(f"{path} already has labels")
    pl.DataFrame(out, infer_schema_length=None).write_csv(path)
    LOG.info("wrote %s (%d senses)", path, len(out))


def evaluate(cfg: dict[str, Any]) -> dict[str, Any]:
    if "embedding" not in cfg:
        cfg = {**cfg, "embedding": load_config(resolve(cfg["align_config"]))["embedding"]}
    gold = pl.read_csv(resolve(cfg["gold"]["path"]), infer_schema_length=0).filter(
        pl.col("match").is_not_null()
    )
    senses = pl.read_parquet(resolve(cfg["senses"]), columns=COLS)
    absent = gold.filter(~pl.col("source_id").is_in(senses["source_id"].implode()))
    if absent.height:  # e.g. items a later version of the selection filter no longer keeps
        LOG.info(
            "%d gold senses are not in this build and are skipped: %s",
            absent.height,
            absent["source_id"].to_list(),
        )
        gold = gold.join(absent.select("source_id"), on="source_id", how="anti")
    members = _cluster_members(cfg, senses)
    tcfg = {**cfg["embedding"].get("text", {}), "include_example": False}
    cand_clusters = {x["cluster"] for c in gold["candidates"] for x in json.loads(c)}
    need = sorted(set(gold["source_id"]) | {m for c in cand_clusters for m in members[c]})
    # only the rows the evaluation reads (a dict per sense of millions would not fit in memory)
    rows = {
        r["source_id"]: r
        for r in senses.filter(pl.col("source_id").is_in(need)).iter_rows(named=True)
    }
    texts = [
        sense_text(rows[x], tcfg)
        if rows[x]["source"] in STAGE1_SOURCES
        else (rows[x]["gloss"] or "")
        for x in need
    ]
    mat, _ = embed_texts(texts, cfg["embedding"])
    vec = {x: mat[i] for i, x in enumerate(need)}
    bridge = load_bridge(cfg, senses)
    key_cluster: dict[str, set[str]] = defaultdict(set)  # only candidate clusters are looked up
    for c in cand_clusters:
        for m in members[c]:
            for k in identity_keys(rows[m]["source"], rows[m]["relations"], bridge):
                key_cluster[k].add(c)
    recs = []
    for g in gold.iter_rows(named=True):
        src, sid = g["source"], g["source_id"]
        cands = [x["cluster"] for x in json.loads(g["candidates"])]
        if src in set(cfg.get("no_case_fallback", [])):
            cands = [
                c if any(rows[m]["lemma"] == rows[sid]["lemma"] for m in members[c]) else None
                for c in cands
            ]
        if src == "ncbi_taxonomy":
            cands = [
                c
                if c
                and taxon_compatible(
                    cfg, rows[sid]["relations"], [rows[m]["lexfile"] for m in members[c]]
                )
                else None
                for c in cands
            ]
        if all(c is None for c in cands):
            recs.append(
                {
                    "source": src,
                    "match": g["match"],
                    "best_score": -1.0,
                    "best_xref": False,
                    "best_idx": "none",
                    "ambiguous": g["ambiguous"] == "1",
                }
            )
            continue
        own = own_key(src, sid)
        keys = set(identity_keys(src, rows[sid]["relations"], bridge)) | ({own} if own else set())
        scored = []
        for i, c in enumerate(cands):
            if c is None:
                continue
            score = max(float(vec[sid] @ vec[m]) for m in members[c])
            scored.append((score, any(c in key_cluster.get(k, ()) for k in keys), i + 1))
        top = max(scored, key=lambda t: (t[1], t[0]))
        recs.append(
            {
                "source": src,
                "match": g["match"],
                "best_score": top[0],
                "best_xref": top[1],
                "best_idx": str(top[2]),
                "ambiguous": g["ambiguous"] == "1",
            }
        )
    df = pl.DataFrame(recs)
    res: dict[str, Any] = {}
    grid = np.round(np.arange(0.50, 0.951, 0.01), 2)
    for src in [*sorted(df["source"].unique().to_list()), "ALL"]:
        d = df if src == "ALL" else df.filter(pl.col("source") == src)
        sweep = []
        for t in grid:
            att = (d["best_xref"] | (d["best_score"] >= t)).to_numpy()
            ok = [
                b in m.split("|")
                for b, m in zip(d["best_idx"].to_list(), d["match"].to_list(), strict=True)
            ]
            correct = att & np.array(ok, dtype=bool)
            has = (d["match"] != "none").to_numpy()
            tp, n_att, n_has = int(correct.sum()), int(att.sum()), int(has.sum())
            p = tp / n_att if n_att else float("nan")
            r = tp / n_has if n_has else float("nan")
            f05 = 1.25 * p * r / (0.25 * p + r) if n_att and n_has and p + r else float("nan")
            sweep.append(
                {"threshold": float(t), "precision": p, "recall": r, "f0.5": f05, "attached": n_att}
            )
        best = max(sweep, key=lambda r: (np.nan_to_num(r["f0.5"]), r["threshold"]))
        thr = cfg.get("thresholds", {})
        conf = float(thr.get(src, thr.get("default", 0.6))) if src != "ALL" else None
        res[src] = {
            "n": d.height,
            "with_match": int((d["match"] != "none").sum()),
            "xref_best": int(d["best_xref"].sum()),
            "best": best,
            # the threshold actually configured (a plateau value, not the noisy argmax)
            "configured": next(
                (r for r in sweep if conf is not None and abs(r["threshold"] - conf) < 1e-9), None
            ),
            "sweep": sweep,
        }
    # all sources at their own configured thresholds (micro-average)
    tp = n_att = n_has = 0
    for src, r in res.items():
        if src != "ALL" and r["configured"]:
            c = r["configured"]
            n_att += c["attached"]
            n_has += r["with_match"]
            tp += round(c["recall"] * r["with_match"]) if r["with_match"] else 0
    if n_att and n_has:
        p_, r_ = tp / n_att, tp / n_has
        res["ALL"]["configured"] = {
            "threshold": None,
            "precision": p_,
            "recall": r_,
            "attached": n_att,
            "f0.5": 1.25 * p_ * r_ / (0.25 * p_ + r_) if p_ + r_ else float("nan"),
        }
    dump_json(res, resolve(cfg["gold"]["eval_output"]))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage-2 attachment gold set")
    ap.add_argument("command", choices=["render", "evaluate"])
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    if args.command == "render":
        render(cfg)
    else:
        res = evaluate(cfg)
        for src, r in res.items():
            b, c = r["best"], r["configured"]
            at = ""
            if c:
                t = "per-source" if c["threshold"] is None else f"{c['threshold']:.2f}"
                at = f" | configured thr={t} P={c['precision']:.3f} R={c['recall']:.3f}"
            print(
                f"{src:18s} n={r['n']:3d} with_match={r['with_match']:3d} xref={r['xref_best']:3d} "
                f"best thr={b['threshold']:.2f} P={b['precision']:.3f} R={b['recall']:.3f} F0.5={b['f0.5']:.3f}{at}"
            )


if __name__ == "__main__":
    main()
