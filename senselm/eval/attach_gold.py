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
    rows = {r["source_id"]: r for r in senses.iter_rows(named=True)}
    members = _cluster_members(cfg, senses)
    tcfg = {**cfg["embedding"].get("text", {}), "include_example": False}
    need = sorted(
        {g for g in gold["source_id"]}
        | {m for c in gold["candidates"] for x in json.loads(c) for m in members[x["cluster"]]}
    )
    texts = [
        sense_text(rows[x], tcfg)
        if rows[x]["source"] in STAGE1_SOURCES
        else (rows[x]["gloss"] or "")
        for x in need
    ]
    mat, _ = embed_texts(texts, cfg["embedding"])
    vec = {x: mat[i] for i, x in enumerate(need)}
    bridge = load_bridge(cfg, senses)
    key_cluster: dict[str, set[str]] = defaultdict(set)
    for c, ms in members.items():
        for m in ms:
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
        keys = set(identity_keys(src, rows[sid]["relations"], bridge)) | (
            {own_key(src, sid)} if own_key(src, sid) else set()
        )
        scored = []
        for i, c in enumerate(cands):
            if c is None:
                continue
            score = max(float(vec[sid] @ vec[m]) for m in members[c])
            scored.append((score, any(c in key_cluster.get(k, ()) for k in keys), i + 1))
        best = max(scored, key=lambda t: (t[1], t[0]))
        recs.append(
            {
                "source": src,
                "match": g["match"],
                "best_score": best[0],
                "best_xref": best[1],
                "best_idx": str(best[2]),
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
        res[src] = {
            "n": d.height,
            "with_match": int((d["match"] != "none").sum()),
            "xref_best": int(d["best_xref"].sum()),
            "best": best,
            "sweep": sweep,
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
            b = r["best"]
            print(
                f"{src:18s} n={r['n']:3d} with_match={r['with_match']:3d} xref={r['xref_best']:3d} "
                f"best thr={b['threshold']:.2f} P={b['precision']:.3f} R={b['recall']:.3f} F0.5={b['f0.5']:.3f}"
            )


if __name__ == "__main__":
    main()
