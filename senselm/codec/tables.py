"""Token tables for the codec: structural ids by name, sense ids <-> (lemma, UPOS)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from senselm.common import resolve

LOG = logging.getLogger("senselm.codec")

# Characters that make a lemma span several spaCy tokens (multi-word / hyphenated lemmas).
MWE_CHARS = set(" -/'’&+")


@dataclass
class Structural:
    """Structural tokens: ``ids["morph:PST"]`` and ``info[token_id] = (category, name)``."""

    ids: dict[str, int]
    info: dict[int, tuple[str, str]]
    punct: dict[str, int]
    byte: list[int]

    @classmethod
    def load(cls, path: str | Path) -> Structural:
        df = pl.read_parquet(resolve(path)).filter(pl.col("status") == "active")
        ids, info, punct, byte = {}, {}, {}, [0] * 256
        for tid, cat, idx, name in df.select("token_id", "category", "index", "name").iter_rows():
            ids[f"{cat}:{name}"] = tid
            info[tid] = (cat, name)
            if cat == "punct":
                punct[name] = tid
            if cat == "byte" and idx < 256:
                byte[idx] = tid
        return cls(ids, info, punct, byte)

    def __getitem__(self, key: str) -> int:
        return self.ids[key]


class Lexicon:
    """Sense tokens of an inventory build.

    Decoding: ``lookup(token_id) -> (lemma, upos)`` by binary search over sorted ids.
    Encoding: ``mfs(lemma, upos)`` = the lemma's first sense with that UPOS (sense indices are
    ordered by estimated frequency, spec 1.5), plus case-insensitive and multi-word indexes.
    """

    def __init__(self, inventory: str | Path, *, for_encoding: bool = True) -> None:
        df = pl.read_parquet(
            resolve(inventory), columns=["token_id", "lemma", "upos", "sense_index"]
        ).sort("token_id")
        self._ids = df["token_id"].cast(pl.Int64).to_numpy()  # int64: searchsorted on a python
        # int against a uint32 array would convert the whole array on every call
        self._lemma = df["lemma"].to_list()
        self._upos = df["upos"].to_list()
        self.n_tokens = df.height
        self._mfs: dict[str, int] = {}
        self._poses: dict[str, list[str]] = {}
        self._casefold: dict[str, list[str]] = {}
        self.mwe: set[str] = set()
        if not for_encoding:
            return
        first = (
            df.sort("sense_index", "token_id")
            .group_by("lemma", "upos", maintain_order=True)
            .agg(pl.col("token_id").first(), pl.col("sense_index").first())
        )
        keys = (first["lemma"] + "\t" + first["upos"]).to_list()
        self._mfs = dict(zip(keys, first["token_id"].to_list(), strict=True))
        for lemma, upos in first.sort("sense_index").select("lemma", "upos").iter_rows():
            self._poses.setdefault(lemma, []).append(upos)
        lemmas = first["lemma"].unique()
        low = lemmas.str.to_lowercase()
        for lemma, lo in zip(lemmas.to_list(), low.to_list(), strict=True):
            if lemma != lo:
                self._casefold.setdefault(lo, []).append(lemma)
            if any(c in MWE_CHARS for c in lo):
                self.mwe.add(lo)
        for v in self._casefold.values():
            v.sort()
        LOG.info(
            "lexicon: %d sense tokens, %d (lemma, upos) pairs, %d multi-word lemmas",
            self.n_tokens,
            len(self._mfs),
            len(self.mwe),
        )

    def lookup(self, token_id: int) -> tuple[str, str] | None:
        i = int(np.searchsorted(self._ids, np.int64(token_id)))
        if i < len(self._ids) and int(self._ids[i]) == token_id:
            return self._lemma[i], self._upos[i]
        return None

    def mfs(self, lemma: str, upos: str) -> int | None:
        return self._mfs.get(f"{lemma}\t{upos}")

    def poses(self, lemma: str) -> list[str]:
        return self._poses.get(lemma, [])

    def spellings(self, lemma: str) -> list[str]:
        """The lemma itself (if listed) and its case-variant spellings in the inventory."""
        out = [lemma] if lemma in self._poses else []
        lo = lemma.lower()
        if lo != lemma and lo in self._poses:
            out.append(lo)
        out += [s for s in self._casefold.get(lo, []) if s not in out]
        return out


class Variants:
    """Orthographic variants (``<variant:kind>``): (target lemma, UPOS, kind) <-> spelling."""

    def __init__(self, df: pl.DataFrame | None = None) -> None:
        self.spelling: dict[tuple[str, str, str], str] = {}
        self.target: dict[str, list[tuple[str, str, str]]] = {}
        if df is None:
            return
        for target, upos, kind, variant in df.select(
            "target", "upos", "kind", "variant"
        ).iter_rows():
            self.spelling[(target, upos, kind)] = variant
            self.target.setdefault(variant.lower(), []).append((target, upos, kind))

    @classmethod
    def load(cls, path: str | Path | None) -> Variants:
        p = resolve(path) if path else None
        return cls(pl.read_parquet(p) if p is not None and p.exists() else None)


def load_tables(cfg: dict[str, Any], *, for_encoding: bool = True) -> dict[str, Any]:
    from senselm.codec.morph import Paradigms

    exc = resolve(cfg["paradigm_exceptions"])
    return {
        "structural": Structural.load(cfg["structural_registry"]),
        "lexicon": Lexicon(cfg["inventory"], for_encoding=for_encoding),
        "paradigms": Paradigms(pl.read_parquet(exc) if exc.exists() else None),
        "variants": Variants.load(cfg.get("variant_table")),
    }
