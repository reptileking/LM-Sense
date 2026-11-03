"""Codec tables: paradigm exceptions and orthographic variants.

``python -m senselm.codec.build --config configs/codec.yaml``

* ``paradigm_exceptions``: for every (lemma, UPOS, bundle[, uk/us]) that Wiktionary or Wikidata
  lexemes give a standard form for, the canonical form is the rule-based inflector's form when
  the dictionaries list it, else the first listed form; only the forms the inflector would get
  wrong are stored (mice, went, sisters-in-law, travelled as the UK variant), so the decoder needs
  the rules plus a small table.
* ``variant_table``: spelling variants that are not inventory lemmas themselves (dixnary,
  encyclopædia), one per (target lemma, UPOS, kind), for ``<variant:kind>`` tokens.
"""

from __future__ import annotations

import logging
from typing import Any

import polars as pl

from senselm.codec.morph import bundle_key, parse_bundle, rule_inflect, tags_to_bundle
from senselm.common import Provenance, load_config, resolve, stage_args, write_table

LOG = logging.getLogger("senselm.codec.build")

ORIGIN_RANK = {"wikt_entry": 0, "wikidata_lexemes_form": 1, "wikt_form_of": 2, "oewn_form": 3}
INFLECTING = ["NOUN", "PROPN", "VERB", "ADJ", "ADV"]
VARIANT_KINDS: list[tuple[str, set[str]]] = [
    ("uk", {"UK", "British", "Commonwealth", "Australia", "New-Zealand", "Ireland", "India"}),
    ("us", {"US", "American", "Canada"}),
    ("abbrev", {"abbreviation", "initialism", "acronym", "clipping", "contraction"}),
    (
        "nonstandard",
        {
            "misspelling",
            "nonstandard",
            "pronunciation-spelling",
            "eye-dialect",
            "dialectal",
            "informal",
            "slang",
            "colloquial",
            "Internet",
        },
    ),
    ("archaic", {"obsolete", "archaic", "dated", "historical"}),
]


def variant_kind(tags: list[str] | None) -> str:
    t = set(tags or [])
    for kind, keys in VARIANT_KINDS:
        if t & keys:
            return kind
    return "other"


def build_paradigms(cfg: dict[str, Any]) -> tuple[pl.DataFrame, dict[str, Any]]:
    pairs = pl.scan_parquet(resolve(cfg["inventory"])).select("lemma", "upos").unique().collect()
    forms = (
        pl.read_parquet(resolve(cfg["forms"]), columns=["lemma", "upos", "form", "tags", "origin"])
        .with_row_index("row")
        .filter(pl.col("upos").is_in(INFLECTING))
        .join(pairs, on=["lemma", "upos"], how="semi")
        # periphrastic forms (more beautiful) never match one spaCy token
        .filter(~pl.col("form").str.contains(" ") | pl.col("lemma").str.contains(" "))
    )
    rows = []
    for row, lemma, upos, form, tags, origin in forms.select(
        "row", "lemma", "upos", "form", "tags", "origin"
    ).iter_rows():
        m = tags_to_bundle(tags or [], upos)
        if m is None or not form:
            continue
        rows.append((lemma, upos, bundle_key(m[0]), m[1], form, ORIGIN_RANK.get(origin, 9), row))
    df = pl.DataFrame(
        rows,
        schema={
            "lemma": pl.String,
            "upos": pl.String,
            "bundle": pl.String,
            "variant": pl.String,
            "form": pl.String,
            "rank": pl.Int32,
            "row": pl.UInt32,
        },
        orient="row",
    ).sort("lemma", "upos", "bundle", "variant", "rank", "row")
    cands = df.group_by("lemma", "upos", "bundle", "variant", maintain_order=True).agg(
        pl.col("form").unique(maintain_order=True)
    )
    out = []
    default: dict[tuple[str, str, str], str] = {}
    n_rule = n_first = 0
    for lemma, upos, b, var, forms_ in cands.sort("variant").iter_rows():  # "" before uk / us
        rule = rule_inflect(lemma, upos, parse_bundle(b))
        if var == "":
            chosen = rule if rule in forms_ else forms_[0]
            n_rule += chosen == rule
            n_first += chosen != rule
            default[(lemma, upos, b)] = chosen
            if chosen != rule:
                out.append((lemma, upos, b, "", chosen))
            alt = [f for f in forms_ if f != chosen]
            if alt:  # mediums beside media, indices beside indexes: <variant:other>
                out.append((lemma, upos, b, "other", alt[0]))
        else:
            base = default.get((lemma, upos, b), rule)
            chosen = forms_[0]
            if chosen != base:
                out.append((lemma, upos, b, var, chosen))
    exc = pl.DataFrame(
        out,
        schema={
            "lemma": pl.String,
            "upos": pl.String,
            "bundle": pl.String,
            "variant": pl.String,
            "form": pl.String,
        },
        orient="row",
    )
    stats = {
        "form_rows_used": df.height,
        "slots": cands.height,
        "rule_agrees": n_rule,
        "exceptions": exc.filter(pl.col("variant") == "").height,
        "uk_us_variants": exc.filter(pl.col("variant").is_in(["uk", "us"])).height,
        "other_forms": exc.filter(pl.col("variant") == "other").height,
    }
    return exc, stats


def build_variants(cfg: dict[str, Any]) -> tuple[pl.DataFrame, dict[str, Any]]:
    lemmas = pl.scan_parquet(resolve(cfg["inventory"])).select("lemma", "upos").unique().collect()
    v = (
        pl.read_parquet(resolve(cfg["variants"]), columns=["variant", "target", "upos", "tags"])
        .with_row_index("row")
        .join(lemmas.select(pl.col("lemma").alias("variant")).unique(), on="variant", how="anti")
        .join(lemmas.rename({"lemma": "target"}), on=["target", "upos"], how="semi")
        .filter(pl.col("variant") != pl.col("target"))
    )
    v = v.with_columns(pl.Series("kind", [variant_kind(t) for t in v["tags"].to_list()]))
    table = v.sort("row").unique(["target", "upos", "kind"], keep="first", maintain_order=True)
    table = table.select("target", "upos", "kind", "variant").sort("target", "upos", "kind")
    return table, {"variant_rows": v.height, "variant_slots": table.height}


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    prov = Provenance(stage="codec.build", config=cfg)
    for k in ("inventory", "forms", "variants"):
        prov.add_input(cfg[k])
    exc, st1 = build_paradigms(cfg)
    write_table(exc.to_arrow(), cfg["paradigm_exceptions"], prov, stats=st1)
    var, st2 = build_variants(cfg)
    write_table(var.to_arrow(), cfg["variant_table"], prov, stats=st2)
    stats = {**st1, **st2}
    LOG.info("codec tables: %s", stats)
    return stats


def main() -> None:
    args = stage_args(__doc__.splitlines()[0])
    run(load_config(args.config))


if __name__ == "__main__":
    main()
