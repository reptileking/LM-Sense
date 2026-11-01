"""Score cross-source candidate pairs with gloss embeddings.

``python -m senselm.inventory.align --config configs/align.yaml``

Output ``data/interim/align_pairs.parquet``: one row per candidate pair (same lemma + POS
class, see ``candidates.py``) with the cosine similarity of the two senses' texts and, when
the LLM judge has been run, its verdict. Merge decisions are *not* made here: they depend on
the granularity policy (``senselm.inventory.policy``) and are applied by the build stage.

Sense text = gloss (+ first usage example), optionally with lemma / Wiktionary parent gloss,
as configured in ``embedding.text``.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
import polars as pl

from senselm.common import Provenance, load_config, resolve, stage_args, write_table
from senselm.inventory.candidates import candidate_pairs
from senselm.inventory.embed import embed_texts

LOG = logging.getLogger("senselm.inventory.align")

HEADER_RE = re.compile(r"^(?:terms|senses) relating to\b|:\s*$", re.IGNORECASE)
SENSE_COLUMNS = [
    "source",
    "source_id",
    "lemma",
    "upos",
    "gloss",
    "gloss_path",
    "examples",
    "example_types",
]


def is_header(gloss: str) -> bool:
    """Wiktionary grouping lines such as 'Terms relating to animals.' or 'A placename:'."""
    return bool(HEADER_RE.search(gloss.strip()))


def pick_example(
    examples: list[str] | None, types: list[str] | None, prefer: list[str]
) -> str | None:
    if not examples:
        return None
    types = types or ["example"] * len(examples)
    for want in prefer:
        for ex, t in zip(examples, types, strict=False):
            if t == want:
                return ex
    return examples[0]


def sense_text(row: dict[str, Any], tcfg: dict[str, Any]) -> str:
    gloss = (row.get("gloss") or "").strip()
    parts: list[str] = []
    if tcfg.get("include_lemma"):
        parts.append(f"{row['lemma']}: ")
    if tcfg.get("include_parent") and row.get("gloss_path"):
        parents = [p for p in row["gloss_path"] if not is_header(p)]
        if parents:
            parts.append(parents[-1].rstrip(" .") + " — ")
    parts.append(gloss)
    if tcfg.get("include_example", True):
        ex = pick_example(
            row.get("examples"),
            row.get("example_types"),
            tcfg.get("prefer_example_types", ["example"]),
        )
        if ex:
            ex = " ".join(ex.split())
            limit = int(tcfg.get("max_example_chars", 200))
            if len(ex) > limit:
                ex = ex[: limit - 1].rstrip() + "…"
            parts.append(f" Example: {ex}")
    return "".join(parts)


def pair_cosines(
    pairs: pl.DataFrame, vec_index: dict[str, int], mat: np.ndarray, chunk: int = 200_000
) -> np.ndarray:
    oi = np.array([vec_index[x] for x in pairs["oewn_id"].to_list()], dtype=np.int64)
    wi = np.array([vec_index[x] for x in pairs["wikt_id"].to_list()], dtype=np.int64)
    out = np.empty(len(oi), dtype=np.float32)
    for s in range(0, len(oi), chunk):
        a, b = mat[oi[s : s + chunk]], mat[wi[s : s + chunk]]
        out[s : s + chunk] = np.einsum("ij,ij->i", a, b)
    return out


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    prov = Provenance(stage="inventory.align", config=cfg)
    prov.add_input(cfg["input"])
    senses = pl.read_parquet(resolve(cfg["input"]), columns=SENSE_COLUMNS)
    pairs = candidate_pairs(senses, cfg)
    LOG.info("candidate pairs: %d in %d groups", pairs.height, pairs["group_key"].n_unique())

    involved = set(pairs["oewn_id"].to_list()) | set(pairs["wikt_id"].to_list())
    sub = senses.filter(pl.col("source_id").is_in(list(involved))).sort("source_id")
    tcfg = cfg["embedding"].get("text", {})
    ids = sub["source_id"].to_list()
    texts = [sense_text(r, tcfg) for r in sub.iter_rows(named=True)]
    mat, emb_meta = embed_texts(texts, cfg["embedding"])
    vec_index = {sid: i for i, sid in enumerate(ids)}
    cos = pair_cosines(pairs, vec_index, mat)
    pairs = pairs.with_columns(pl.Series("cosine", np.round(cos, 4), dtype=pl.Float32))

    # Optional LLM-judge verdicts (cached by pair); absent when the judge is disabled.
    jcfg = cfg.get("llm_judge", {})
    verdicts = None
    jpath = resolve(jcfg.get("cache_path", "data/interim/judge_cache.parquet"))
    if jcfg.get("enabled") and jpath.exists():
        verdicts = pl.read_parquet(jpath).select("oewn_id", "wikt_id", "verdict")
    if verdicts is not None:
        pairs = pairs.join(verdicts, on=["oewn_id", "wikt_id"], how="left")
    else:
        pairs = pairs.with_columns(pl.lit(None, dtype=pl.String).alias("verdict"))

    band = cfg.get("judge_band", {"low": 0.70, "high": 0.80})
    in_band = pairs.filter(
        (pl.col("cosine") > band["low"]) & (pl.col("cosine") < band["high"])
    ).height
    stats = {
        "pairs": pairs.height,
        "groups": pairs["group_key"].n_unique(),
        "case_fallback_pairs": int(pairs["case_fallback"].sum()),
        "senses_embedded": len(ids),
        "judge_band_pairs": in_band,
        "judge_verdicts": int(pairs["verdict"].is_not_null().sum()),
        "embedding": emb_meta,
        "cosine_quantiles": {
            str(q): float(pairs["cosine"].quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
        },
    }
    prov.extra["embedding"] = emb_meta
    table = pairs.sort("group_key", "oewn_id", "wikt_id").to_arrow()
    write_table(table, cfg["output"], prov, stats=stats)
    LOG.info("align: %s", {k: v for k, v in stats.items() if k != "cosine_quantiles"})
    return stats


def main() -> None:
    args = stage_args("Score cross-source candidate pairs with gloss embeddings")
    run(load_config(args.config))


if __name__ == "__main__":
    main()

__all__ = ["is_header", "pick_example", "run", "sense_text"]
