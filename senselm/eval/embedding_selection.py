"""Choose the alignment embedding model and text template on the gold pairs.

``python -m senselm.eval.embedding_selection --config configs/embedding_selection.yaml``

For every variant (model × pooling × dtype × text template) it embeds the senses of the
labelled gold pairs, then reports on a deterministic dev/test split (by pair-id hash):

* ROC-AUC of the cosine for ``same`` vs rest (fine) and ``same|related`` vs ``different``
  (coarse) on all labelled pairs;
* the F1-optimal threshold on *dev* and precision / recall / F1 at that threshold on *test*;
* CPU throughput (texts/s) on a fixed random sample of glosses, measured per model.

Pairwise thresholding ignores the one-to-one matching step of the real aligner (which can
only remove false positives), so these numbers are a lower bound on aligner precision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from senselm.common import dump_json, load_config, resolve, setup_logging
from senselm.eval.alignment_gold import load_gold
from senselm.inventory.align import SENSE_COLUMNS, sense_text
from senselm.inventory.embed import Embedder, embed_texts

LOG = logging.getLogger("senselm.eval.embedding_selection")


def split_of(pair_id: str, dev_fraction: float) -> str:
    h = int.from_bytes(hashlib.blake2b(pair_id.encode(), digest_size=8).digest(), "big")
    return "dev" if (h % 10_000) / 10_000 < dev_fraction else "test"


def auc(scores: np.ndarray, y: np.ndarray) -> float:
    pos, neg = scores[y], scores[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]), kind="stable"), kind="stable") + 1.0
    # average ranks for ties
    allv = np.concatenate([pos, neg])
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    u = ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def prf_at(scores: np.ndarray, y: np.ndarray, thr: float) -> dict[str, float]:
    yhat = scores >= thr
    tp = float((yhat & y).sum())
    fp = float((yhat & ~y).sum())
    fn = float((~yhat & y).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {
        "precision": round(p, 3),
        "recall": round(r, 3),
        "f1": round(f, 3),
        "threshold": round(thr, 3),
    }


def best_threshold(scores: np.ndarray, y: np.ndarray) -> float:
    cands = np.unique(np.round(scores, 3))
    best = max(cands, key=lambda t: (prf_at(scores, y, float(t))["f1"], -t))
    return float(best)


def evaluate_scores(gold: pl.DataFrame, cos: np.ndarray) -> dict[str, Any]:
    res: dict[str, Any] = {}
    split = gold["split"].to_numpy()
    for policy, positive in (("fine", ("same",)), ("coarse", ("same", "related"))):
        y = gold["label"].is_in(list(positive)).to_numpy()
        dev, test = split == "dev", split == "test"
        thr = best_threshold(cos[dev], y[dev])
        res[policy] = {
            "auc_all": round(auc(cos, y), 4),
            "dev": prf_at(cos[dev], y[dev], thr),
            "test": prf_at(cos[test], y[test], thr),
        }
    return res


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    gold = load_gold(resolve(cfg["gold"]["path"]))
    gold = gold.with_columns(
        pl.col("pair_id")
        .map_elements(
            lambda p: split_of(p, float(cfg["gold"]["dev_fraction"])), return_dtype=pl.String
        )
        .alias("split")
    )
    ids = gold["pair_id"].str.split("|")
    gold = gold.with_columns(ids.list.get(0).alias("oewn_id"), ids.list.get(1).alias("wikt_id"))
    need = set(gold["oewn_id"]) | set(gold["wikt_id"])
    senses = pl.read_parquet(resolve(cfg["input"]), columns=SENSE_COLUMNS).filter(
        pl.col("source_id").is_in(list(need))
    )
    rows = {r["source_id"]: r for r in senses.iter_rows(named=True)}
    bench_texts = [
        t
        for t in pl.read_parquet(resolve(cfg["input"]), columns=["gloss"])
        .sample(512, seed=0)["gloss"]
        .to_list()
        if t
    ]
    results: list[dict[str, Any]] = []
    speed_cache: dict[tuple[str, str], float] = {}
    for var in cfg["variants"]:
        ecfg = {**cfg["embedding_defaults"], **var.get("embedding", {})}
        tcfg = {**cfg["text_defaults"], **var.get("text", {})}
        emb = Embedder(ecfg)
        key = (ecfg["model"], ecfg.get("dtype", "float32"))
        if key not in speed_cache:
            emb.encode(bench_texts[:32])
            t0 = time.time()
            emb.encode(bench_texts)
            speed_cache[key] = round(len(bench_texts) / (time.time() - t0), 1)
        o_txt = [sense_text(rows[i], tcfg) for i in gold["oewn_id"]]
        w_txt = [sense_text(rows[i], tcfg) for i in gold["wikt_id"]]
        mat, _ = embed_texts(o_txt + w_txt, ecfg, embedder=emb)
        n = gold.height
        cos = np.einsum("ij,ij->i", mat[:n], mat[n:])
        rec = {
            "name": var["name"],
            "model": ecfg["model"],
            "dtype": ecfg.get("dtype", "float32"),
            "text": tcfg,
            "texts_per_s": speed_cache[key],
            **evaluate_scores(gold, cos),
        }
        LOG.info(
            "%s: fine AUC %.3f test F1 %.3f | coarse AUC %.3f test F1 %.3f | %.1f texts/s",
            var["name"],
            rec["fine"]["auc_all"],
            rec["fine"]["test"]["f1"],
            rec["coarse"]["auc_all"],
            rec["coarse"]["test"]["f1"],
            rec["texts_per_s"],
        )
        results.append(rec)
        del emb
    out = {
        "n_gold": gold.height,
        "dev": int((gold["split"] == "dev").sum()),
        "test": int((gold["split"] == "test").sum()),
        "variants": results,
    }
    dump_json(out, resolve(cfg["output"]))
    return out


def to_markdown(res: dict[str, Any]) -> str:
    lines = [
        f"Gold pairs: {res['n_gold']} (dev {res['dev']} / test {res['test']}). Thresholds chosen on dev.",
        "",
        "| variant | model | dtype | texts/s (4 CPU) | fine AUC | fine test P/R/F1 @thr | coarse AUC | coarse test P/R/F1 @thr |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for v in res["variants"]:
        f, c = v["fine"], v["coarse"]
        lines.append(
            f"| {v['name']} | {v['model'].split('/')[-1]} | {v['dtype']} | {v['texts_per_s']} | {f['auc_all']:.3f} | "
            f"{f['test']['precision']:.2f}/{f['test']['recall']:.2f}/{f['test']['f1']:.2f} @{f['dev']['threshold']:.3f} | "
            f"{c['auc_all']:.3f} | {c['test']['precision']:.2f}/{c['test']['recall']:.2f}/{c['test']['f1']:.2f} @{c['dev']['threshold']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    res = run(cfg)
    md = to_markdown(res)
    resolve(cfg["output"]).with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)
    print(
        json.dumps(
            {
                "best_by_fine_test_f1": max(res["variants"], key=lambda v: v["fine"]["test"]["f1"])[
                    "name"
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
