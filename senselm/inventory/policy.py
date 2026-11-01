"""Granularity policy: turn scored candidate pairs into canonical sense clusters.

The policy (``policy`` block of ``configs/inventory*.yaml``, rules in docs/GRANULARITY.md):

* ``subsenses``: ``keep`` every Wiktionary subsense as its own unit, or ``collapse`` subsenses
  into their top-level (non-header) parent sense.
* eligibility of a cross-source edge (OEWN sense, Wiktionary unit): cosine ≥
  ``auto_merge_threshold``, or ``separate_threshold`` < cosine < auto threshold *and* an LLM
  judge verdict in ``judge_accept`` (when the judge has run; otherwise kept separate).
* ``matching``:
    - ``one_to_one`` (fine): greedy by descending cosine; each sense joins at most one sense
      of the other source, so both sources' distinctions survive;
    - ``many_to_one`` (coarse): each Wiktionary unit joins its best eligible WordNet sense;
      WordNet senses absorb any number of Wiktionary units but never each other;
    - ``components`` (coarsest): every eligible edge merges (transitive closure).

Output: one row per source sense with ``cluster`` = the cluster's anchor source_id (the
WordNet sense if the cluster has one - WordNet sense keys are the most stable ids we have -
else the cluster's first Wiktionary sense in dump order).
"""

from __future__ import annotations

from typing import Any

import polars as pl

from senselm.inventory.align import is_header


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
            return x
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # deterministic: smaller string becomes root
            if rb < ra:
                ra, rb = rb, ra
            self.parent[rb] = ra


def wiktionary_units(wikt: pl.DataFrame, mode: str) -> dict[str, str]:
    """Map each Wiktionary source_id to its unit id (itself unless subsenses collapse).

    ``wikt`` needs: source_id, entry_key, gloss, gloss_path, line_no, sense_rank.
    """
    if mode == "keep":
        return {sid: sid for sid in wikt["source_id"].to_list()}
    if mode != "collapse":
        raise ValueError(f"unknown subsenses mode {mode!r}")
    unit_of: dict[str, str] = {}
    first_in_unit: dict[tuple[str, str], str] = {}
    exact_parent: dict[tuple[str, str], str] = {}
    rows = (
        wikt.sort("line_no", "sense_rank")
        .select("source_id", "entry_key", "gloss", "gloss_path")
        .rows()
    )
    keys: list[tuple[str, tuple[str, str]]] = []
    for sid, entry, gloss, path in rows:
        full = [p for p in [*(path or []), gloss or ""] if p and not is_header(p)]
        top = full[0] if full else (gloss or sid)
        key = (entry, top)
        keys.append((sid, key))
        first_in_unit.setdefault(key, sid)
        if len(full) == 1:
            exact_parent.setdefault(key, sid)
    for sid, key in keys:
        unit_of[sid] = exact_parent.get(key, first_in_unit[key])
    return unit_of


def select_edges(edges: pl.DataFrame, matching: str) -> list[tuple[str, str]]:
    """edges: group_key, oewn_id, unit, score (already filtered to eligible)."""
    chosen: list[tuple[str, str]] = []
    if matching == "components":
        return list(zip(edges["oewn_id"].to_list(), edges["unit"].to_list(), strict=True))
    if matching == "many_to_one":
        best = edges.sort(["unit", "score", "oewn_id"], descending=[False, True, False]).unique(
            "unit", keep="first", maintain_order=True
        )
        return list(zip(best["oewn_id"].to_list(), best["unit"].to_list(), strict=True))
    if matching != "one_to_one":
        raise ValueError(f"unknown matching {matching!r}")
    # Greedy by descending score. "Used" is global, not per group: a Wiktionary sense can sit
    # in two groups via the case-insensitive fallback and must still match at most once.
    srt = edges.sort(["score", "oewn_id", "unit"], descending=[True, False, False])
    used_o: set[str] = set()
    used_u: set[str] = set()
    for o, u in srt.select("oewn_id", "unit").iter_rows():
        if o in used_o or u in used_u:
            continue
        used_o.add(o)
        used_u.add(u)
        chosen.append((o, u))
    return chosen


def assign_clusters(
    senses: pl.DataFrame,
    pairs: pl.DataFrame,
    policy: dict[str, Any],
    dedupe_map: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Return (source, source_id, cluster) for every sense, plus policy statistics.

    ``senses`` needs: source, source_id, entry_key, gloss, gloss_path, line_no, sense_rank.
    ``pairs`` needs: group_key, oewn_id, wikt_id, cosine, verdict.
    """
    wikt = senses.filter(pl.col("source") == "wiktionary")
    unit_of = wiktionary_units(wikt, policy.get("subsenses", "keep"))
    hi = float(policy["auto_merge_threshold"])
    lo = float(policy["separate_threshold"])
    accept = list(policy.get("judge_accept", []))
    p = pairs.with_columns(
        pl.col("wikt_id").replace_strict(unit_of, return_dtype=pl.String).alias("unit")
    )
    eligible = (pl.col("cosine") >= hi) | (
        (pl.col("cosine") > lo) & pl.col("verdict").is_in(accept).fill_null(False)
    )
    p = p.with_columns(eligible.alias("eligible"))
    # unit-level edge score = best member pair
    edges = (
        p.filter(pl.col("eligible"))
        .group_by("group_key", "oewn_id", "unit")
        .agg(pl.col("cosine").max().alias("score"))
    )
    chosen = select_edges(edges, policy["matching"])

    uf = UnionFind()
    for sid in senses["source_id"].to_list():
        uf.find(sid)
    for sid, u in unit_of.items():
        uf.union(sid, u)
    for o, u in chosen:
        uf.union(o, u)

    # anchors: WordNet member (lowest sense_rank, then id) else first Wiktionary sense in dump order
    order = senses.select(
        "source_id",
        pl.when(pl.col("source") == "oewn").then(0).otherwise(1).alias("src_order"),
        "line_no",
        "sense_rank",
    )
    roots = {sid: uf.find(sid) for sid in senses["source_id"].to_list()}
    order = order.with_columns(
        pl.col("source_id").replace_strict(roots, return_dtype=pl.String).alias("root")
    )
    anchor = (
        order.sort(["root", "src_order", "line_no", "sense_rank", "source_id"])
        .unique("root", keep="first", maintain_order=True)
        .select("root", pl.col("source_id").alias("cluster"))
    )
    out = (
        senses.select("source", "source_id")
        .with_columns(
            pl.col("source_id").replace_strict(roots, return_dtype=pl.String).alias("root")
        )
        .join(anchor, on="root")
        .drop("root")
    )

    if dedupe_map is not None and dedupe_map.height:
        cl = dict(zip(out["source_id"].to_list(), out["cluster"].to_list(), strict=True))
        extra = dedupe_map.filter(pl.col("kept_id").is_in(list(cl))).select(
            "source",
            pl.col("dropped_id").alias("source_id"),
            pl.col("kept_id").replace_strict(cl, return_dtype=pl.String).alias("cluster"),
        )
        out = pl.concat([out, extra])

    n_multi_oewn = (
        out.join(senses.select("source_id", "source"), on="source_id", how="left", suffix="_s")
        .filter(pl.col("source") == "oewn")
        .group_by("cluster")
        .len()
        .filter(pl.col("len") > 1)
        .height
    )
    stats = {
        "eligible_pairs": int(p["eligible"].sum()),
        "band_pairs_without_verdict": int(
            p.filter(
                (pl.col("cosine") > lo) & (pl.col("cosine") < hi) & pl.col("verdict").is_null()
            ).height
        ),
        "wiktionary_units": len(set(unit_of.values())),
        "merge_edges": len(chosen),
        "clusters": out["cluster"].n_unique(),
        "clusters_with_multiple_oewn": n_multi_oewn,
    }
    return out.sort("cluster", "source", "source_id"), stats


def pair_merged(pairs: pl.DataFrame, clusters: pl.DataFrame) -> pl.Series:
    """Boolean per pair: do the two senses end up in the same canonical sense?"""
    cl = dict(zip(clusters["source_id"].to_list(), clusters["cluster"].to_list(), strict=True))
    a = pairs["oewn_id"].replace_strict(cl, return_dtype=pl.String)
    b = pairs["wikt_id"].replace_strict(cl, return_dtype=pl.String)
    return a == b
