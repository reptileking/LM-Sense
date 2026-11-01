"""Candidate pairs for cross-source alignment.

A candidate pair is (OEWN sense, Wiktionary sense) with the same lemma and a compatible POS
class (``pos_classes`` in config; by default NOUN and PROPN are one class because OEWN has
no proper-noun category of its own).

Case: lemma identity is case-sensitive (Polish ≠ polish). Only when an OEWN (lemma, class)
group has *no* exact-case Wiktionary counterpart do we fall back to a case-insensitive
match (``case_insensitive_fallback``) - e.g. OEWN ``christly`` vs Wiktionary ``Christly``.
Such pairs are flagged ``case_fallback`` and counted in the report.
"""

from __future__ import annotations

from typing import Any

import polars as pl

DEFAULT_POS_CLASSES = {"NOUN": "N", "PROPN": "N", "VERB": "V", "ADJ": "A", "ADV": "R"}


def with_pos_class(df: pl.DataFrame | pl.LazyFrame, classes: dict[str, str]) -> Any:
    return df.with_columns(pl.col("upos").replace_strict(classes, default=pl.col("upos")).alias("pcls"))


def candidate_pairs(senses: pl.DataFrame, cfg: dict[str, Any]) -> pl.DataFrame:
    """Return one row per candidate pair.

    Columns: group_key, lemma, pcls, oewn_id, wikt_id, case_fallback.
    ``senses`` needs: source, source_id, lemma, upos.
    """
    classes = cfg.get("pos_classes", DEFAULT_POS_CLASSES)
    s = with_pos_class(senses.select("source", "source_id", "lemma", "upos"), classes)
    o = s.filter(pl.col("source") == "oewn").select(
        pl.col("source_id").alias("oewn_id"), "lemma", "pcls"
    )
    w = s.filter(pl.col("source") == "wiktionary").select(
        pl.col("source_id").alias("wikt_id"), pl.col("lemma").alias("wikt_lemma"), "pcls"
    )
    exact = o.join(w.with_columns(pl.col("wikt_lemma").alias("lemma")), on=["lemma", "pcls"], how="inner")
    exact = exact.with_columns(pl.lit(False).alias("case_fallback"))
    parts = [exact]
    if cfg.get("case_insensitive_fallback", True):
        matched = exact.select("lemma", "pcls").unique()
        o_un = o.join(matched, on=["lemma", "pcls"], how="anti").with_columns(
            pl.col("lemma").str.to_lowercase().alias("lk")
        )
        w_lk = w.with_columns(pl.col("wikt_lemma").str.to_lowercase().alias("lk"))
        fb = o_un.join(w_lk, on=["lk", "pcls"], how="inner").drop("lk")
        parts.append(fb.with_columns(pl.lit(True).alias("case_fallback")))
    pairs = pl.concat(parts, how="diagonal_relaxed")
    # Group = OEWN lemma + class (one alignment problem per group).
    pairs = pairs.with_columns(
        (pl.col("lemma") + pl.lit("\x1f") + pl.col("pcls")).alias("group_key")
    ).select("group_key", "lemma", "pcls", "oewn_id", "wikt_id", "wikt_lemma", "case_fallback")
    return pairs.sort("group_key", "oewn_id", "wikt_id")
