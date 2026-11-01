"""Gold set for the cross-source aligner: sampling (blind) and evaluation.

``python -m senselm.eval.alignment_gold sample --config configs/align.yaml``
    Draws candidate pairs into ``eval/alignment_gold.csv`` with empty label columns. The file
    contains no aligner scores, so labelling is blind. Refuses to overwrite a labelled file.

Evaluation (precision / recall of the full aligner, threshold tuning on the dev half) lives in
``senselm.eval.alignment_eval``.

Sampling design (documented in docs/DECISIONS.md D-014). Uniform sampling over pairs yields
almost only negatives (a 40×60-sense verb contributes 2,400 pairs and ≤40 matches), which
makes recall unmeasurable with 300 labels. So the sample has two strata:

* ``U``: ``n_uniform`` pairs uniformly over all candidate pairs;
* ``G``: ``n_group`` pairs drawn by picking a (lemma, POS-class) group uniformly, then a pair
  uniformly within it (over-represents small groups, where matches are common).

Each row stores its inclusion probability π ≈ n_U/N + n_G/(G·n_g), so the evaluation can
report both plain sample P/R and Horvitz-Thompson-weighted estimates for the population of
all candidate pairs.

Labels (see docs/GRANULARITY.md): ``same`` (paraphrases / same meaning), ``related``
(broader/narrower or regular extension: merge only under the coarse policy), ``different``.
``ambiguous`` = 1 when the labeller was unsure.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from senselm.common import dump_json, load_config, resolve, setup_logging
from senselm.inventory.candidates import candidate_pairs

LOG = logging.getLogger("senselm.eval.alignment_gold")

LABEL_COLUMNS = ("label", "ambiguous", "note")
LABELS = ("same", "related", "different")


def _first(xs: list[str] | None, limit: int = 300) -> str:
    if not xs:
        return ""
    x = xs[0]
    return x if len(x) <= limit else x[: limit - 1] + "…"


def sample(cfg: dict[str, Any], *, force: bool = False) -> Path:
    gcfg = cfg["gold"]
    out = resolve(gcfg["path"])
    if out.exists() and not force:
        existing = pl.read_csv(out, infer_schema_length=0)
        if existing["label"].drop_nulls().len() > 0:
            raise FileExistsError(f"{out} already has labels; refusing to overwrite (use --force)")
    senses = pl.read_parquet(
        resolve(cfg["input"]),
        columns=[
            "source",
            "source_id",
            "lemma",
            "upos",
            "gloss",
            "gloss_path",
            "examples",
            "tags",
            "lexfile",
            "domain",
        ],
    )
    pairs = candidate_pairs(senses, cfg).with_row_index("i")
    n_pairs = pairs.height
    sizes = pairs.group_by("group_key", maintain_order=True).agg(
        pl.len().alias("n_g"), pl.col("i").min().alias("start")
    )
    n_groups = sizes.height
    rng = np.random.default_rng(int(gcfg["seed"]))
    n_u, n_g = int(gcfg["n_uniform"]), int(gcfg["n_group"])
    chosen: dict[int, str] = {int(i): "U" for i in rng.choice(n_pairs, size=n_u, replace=False)}
    starts = sizes["start"].to_numpy()
    lens = sizes["n_g"].to_numpy()
    while sum(1 for s in chosen.values() if s == "G") < n_g:
        g = int(rng.integers(n_groups))
        j = int(starts[g] + rng.integers(lens[g]))
        if j not in chosen:
            chosen[j] = "G"
    idx = sorted(chosen)
    samp = pairs.filter(pl.col("i").is_in(idx)).with_columns(
        pl.col("i").replace_strict(chosen, return_dtype=pl.String).alias("stratum")
    )
    samp = samp.join(sizes.select("group_key", "n_g"), on="group_key")
    samp = samp.with_columns((n_u / n_pairs + n_g / (n_groups * pl.col("n_g"))).alias("pi"))
    o = senses.filter(pl.col("source") == "oewn").select(
        pl.col("source_id").alias("oewn_id"),
        pl.col("upos").alias("oewn_upos"),
        pl.col("gloss").alias("oewn_gloss"),
        pl.col("examples").alias("oewn_examples"),
        "lexfile",
    )
    w = senses.filter(pl.col("source") == "wiktionary").select(
        pl.col("source_id").alias("wikt_id"),
        pl.col("upos").alias("wikt_upos"),
        pl.col("gloss").alias("wikt_gloss"),
        pl.col("gloss_path").alias("wikt_parents"),
        pl.col("examples").alias("wikt_examples"),
        pl.col("tags").alias("wikt_tags"),
        pl.col("domain").alias("wikt_domain"),
    )
    samp = samp.join(o, on="oewn_id").join(w, on="wikt_id")
    rows = []
    for r in samp.sort("i").iter_rows(named=True):
        rows.append(
            {
                "pair_id": f"{r['oewn_id']}|{r['wikt_id']}",
                "stratum": r["stratum"],
                "pi": round(r["pi"], 10),
                "group_pairs": r["n_g"],
                "lemma": r["lemma"],
                "wikt_lemma": r["wikt_lemma"],
                "oewn_upos": r["oewn_upos"],
                "wikt_upos": r["wikt_upos"],
                "oewn_gloss": r["oewn_gloss"],
                "oewn_example": _first(r["oewn_examples"]),
                "lexfile": r["lexfile"],
                "wikt_gloss": r["wikt_gloss"],
                "wikt_parents": " > ".join(r["wikt_parents"] or []),
                "wikt_example": _first(r["wikt_examples"]),
                "wikt_tags": ",".join(r["wikt_tags"] or []),
                "wikt_domain": r["wikt_domain"] or "",
                "label": None,
                "ambiguous": None,
                "note": None,
            }
        )
    df = pl.DataFrame(rows, infer_schema_length=None)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(out)
    meta = {
        "n_candidate_pairs": n_pairs,
        "n_groups": n_groups,
        "n_uniform": n_u,
        "n_group": n_g,
        "seed": int(gcfg["seed"]),
        "input": str(cfg["input"]),
    }
    dump_json(meta, out.with_suffix(".sampling.json"))
    LOG.info("wrote %s (%d pairs; N=%d pairs in %d groups)", out, df.height, n_pairs, n_groups)
    return out


# --------------------------------------------------------------------------- evaluation


def prf(tp: float, fp: float, fn: float) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if p + r and not (np.isnan(p) or np.isnan(r)) else float("nan")
    return {"precision": p, "recall": r, "f1": f}


def score(
    gold: pl.DataFrame, predicted_col: str, positive: tuple[str, ...], weighted: bool
) -> dict[str, Any]:
    y = gold["label"].is_in(list(positive)).to_numpy()
    yhat = gold[predicted_col].to_numpy().astype(bool)
    w = (1.0 / gold["pi"].to_numpy()) if weighted else np.ones(gold.height)
    tp = float((w * (y & yhat)).sum())
    fp = float((w * (~y & yhat)).sum())
    fn = float((w * (y & ~yhat)).sum())
    out = prf(tp, fp, fn)
    out.update({"tp": tp, "fp": fp, "fn": fn, "n": gold.height})
    return out


def load_gold(path: Path) -> pl.DataFrame:
    gold = pl.read_csv(path, infer_schema_length=0).filter(pl.col("label").is_not_null())
    bad = set(gold["label"].unique()) - set(LABELS)
    if bad:
        raise ValueError(f"unknown labels in {path}: {bad}")
    return gold.with_columns(
        pl.col("pi").cast(pl.Float64),
        pl.col("ambiguous").cast(pl.Int8, strict=False).fill_null(0),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Alignment gold set: blind sampling")
    ap.add_argument("command", choices=["sample"])
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--force", action="store_true", help="overwrite an existing labelled file")
    args = ap.parse_args()
    setup_logging()
    sample(load_config(args.config), force=args.force)


if __name__ == "__main__":
    main()
