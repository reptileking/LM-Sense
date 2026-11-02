"""Owner review of the band pairs the first-pass judge called "same" (D-030, option 3).

``python -m senselm.inventory.review <command> --config configs/judge_local.yaml``

* ``export``  write the pairs, in a fixed random order, as JSON blocks for the review page
  (``index.json`` plus ``p000.json`` … one file per block). The order is
  ``sha256(seed:job_id)``, so any reviewed prefix is a uniform random sample of the set and the
  running share of "same" answers is an unbiased estimate for all of it.
* ``ingest``  read the decisions back (a JSON dump of the page's ``reviews`` collection, one
  document per block mapping ``job_id -> [label letter, epoch ms]``), write
  ``judge/review/human_<name>.parquet`` and an evaluation of the first-pass judge against them.

The page source is ``tools/review_page/sense-pair-review.html`` (published as a claude.ai artifact
with the export as its ``pairs/`` files). It shows each pair with more context than the judge had (WordNet synonyms and lexicographer
file, Wiktionary labels and topics), so the human answer is the reference, not a replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
from pathlib import Path
from typing import Any

import polars as pl

from senselm.common import dump_json, load_config, resolve, setup_logging

LOG = logging.getLogger("senselm.inventory.review")

LETTERS = {"s": "same", "r": "related", "d": "different", "u": "unsure"}
FIELDS = [
    "job_id",
    "lemma",
    "upos",
    "a_gloss",
    "a_example",
    "a_synonyms",
    "a_class",
    "b_gloss",
    "b_parent",
    "b_example",
    "b_labels",
]
# Wiktionary tags that say nothing about meaning; topic ancestors that only repeat a narrower one.
_NOISE_TAGS = {"countable", "uncountable", "not-comparable", "comparable", "usually", "plural"}
_BROAD_TOPICS = {
    "sciences",
    "natural-sciences",
    "physical-sciences",
    "human-sciences",
    "social-sciences",
    "applied-sciences",
    "lifestyle",
    "nature",
    "communications",
    "tools",
}
_WS = re.compile(r"\s+")


def order_key(seed: str, job_id: str) -> str:
    return hashlib.sha256(f"{seed}:{job_id}".encode()).hexdigest()


def _clip(text: str | None, limit: int) -> str:
    if not text:
        return ""
    t = _WS.sub(" ", text).strip()
    return t if len(t) <= limit else t[: limit - 1].rstrip() + "…"


def _pick_example(examples: list[str] | None, types: list[str] | None, limit: int) -> str:
    """Prefer a short usage example over a dated quotation; fall back to the first quotation."""
    if not examples:
        return ""
    types = types or []
    for ex, ty in zip(examples, types, strict=False):
        if ty == "example":
            return _clip(ex, limit)
    return _clip(examples[0], limit)


def _members(relations: str | None, lemma: str) -> str:
    if not relations:
        return ""
    try:
        members = json.loads(relations).get("members") or []
    except (ValueError, AttributeError):
        return ""
    low = lemma.casefold()
    return ", ".join(m for m in members if m.casefold() != low)


def _labels(tags: list[str] | None, topics: list[str] | None, limit: int = 4) -> str:
    out: list[str] = []
    for t in list(tags or []) + list(topics or []):
        t = t.removeprefix("raw:").strip()  # wiktextract keeps unparsed qualifiers as raw:<text>
        if not t or t in _NOISE_TAGS or t in _BROAD_TOPICS or t in out:
            continue
        out.append(t)
    return " · ".join(out[:limit])


def review_rows(cfg: dict[str, Any]) -> pl.DataFrame:
    """The band pairs the first pass called "same", with the context the page shows, in order."""
    rcfg = cfg["review"]
    job = pl.read_parquet(resolve(cfg["verify_job_path"])).filter(pl.col("kind") == "band")
    acfg = load_config(resolve(cfg["align_config"]))
    ids = pl.concat([job["oewn_id"], job["wikt_id"]]).unique()
    senses = (
        pl.scan_parquet(resolve(acfg["input"]))
        .filter(pl.col("source_id").is_in(ids.implode()))
        .select(
            "source_id",
            "gloss",
            "examples",
            "example_types",
            "tags",
            "topics",
            "gloss_path",
            "relations",
            "lexfile",
        )
        .collect()
    )
    by_id = {r["source_id"]: r for r in senses.iter_rows(named=True)}
    limit = int(rcfg.get("example_chars", 240))
    rows = []
    for r in job.iter_rows(named=True):
        o, w = by_id.get(r["oewn_id"]), by_id.get(r["wikt_id"])
        if o is None or w is None:
            raise SystemExit(f"sense text missing for job {r['job_id']}: rebuild senses_norm first")
        path = w["gloss_path"] or []
        rows.append(
            {
                "job_id": r["job_id"],
                "lemma": r["lemma"],
                "upos": r["upos"],
                "a_gloss": _clip(o["gloss"], 2000),
                "a_example": _pick_example(o["examples"], o["example_types"], limit),
                "a_synonyms": _members(o["relations"], r["lemma"]),
                "a_class": o["lexfile"] or "",
                "b_gloss": _clip(w["gloss"], 2000),
                "b_parent": _clip(path[-1], 600) if path else "",
                "b_example": _pick_example(w["examples"], w["example_types"], limit),
                "b_labels": _labels(w["tags"], w["topics"]),
                "cosine": r["cosine"],
                "order": order_key(rcfg["seed"], r["job_id"]),
            }
        )
    return pl.DataFrame(rows).sort("order").with_row_index("position")


def export(cfg: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    rcfg = cfg["review"]
    size = int(rcfg["block_size"])
    df = review_rows(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("p*.json"):
        old.unlink()
    blocks = math.ceil(df.height / size)
    for b in range(blocks):
        part = df.slice(b * size, size).select(FIELDS)
        body = {"block": b, "pairs": [list(row) for row in part.iter_rows()]}
        (out_dir / f"p{b:03d}.json").write_text(
            json.dumps(body, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
        )
    src = resolve(cfg["verify_job_path"])
    index = {
        "format": 1,
        "total": df.height,
        "block_size": size,
        "blocks": blocks,
        "fields": FIELDS,
        "order": f"sha256('{rcfg['seed']}:' + job_id), ascending",
        "labels": LETTERS,
        "source": {
            "path": str(cfg["verify_job_path"]),
            "rows": "kind = band (first-pass 'same' from the 32B judge)",
            "sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        },
    }
    dump_json(index, out_dir / "index.json")
    size_mb = sum(p.stat().st_size for p in out_dir.glob("*.json")) / 1e6
    LOG.info("wrote %d pairs in %d blocks to %s (%.1f MB)", df.height, blocks, out_dir, size_mb)
    return index


def parse_dump(dump: Any) -> pl.DataFrame:
    """Decisions from the page's ``reviews`` collection.

    Accepts ``{doc_id: body}``, ``[{"id": doc_id, "data": body}, …]`` or
    ``{"documents": [...]}``; each body maps ``job_id -> [letter, epoch_ms]``.
    """
    if isinstance(dump, dict) and "documents" in dump:
        dump = dump["documents"]
    if isinstance(dump, list):
        docs = {d.get("id") or d.get("doc_id"): d.get("data") or {} for d in dump}
    else:
        docs = dump
    rows = []
    for doc_id, body in docs.items():
        m = re.fullmatch(r"b(\d{3})", str(doc_id).rsplit("/", 1)[-1])
        if not m:
            continue
        for jid, val in (body or {}).items():
            if not isinstance(val, list) or len(val) < 2 or val[0] not in LETTERS:
                continue
            rows.append(
                {
                    "job_id": jid,
                    "block": int(m.group(1)),
                    "decision": LETTERS[val[0]],
                    "decided_at_ms": int(val[1]),
                }
            )
    schema = {
        "job_id": pl.String,
        "block": pl.Int32,
        "decision": pl.String,
        "decided_at_ms": pl.Int64,
    }
    return pl.DataFrame(rows, schema=schema)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 4), round(c + h, 4))


def _precision(df: pl.DataFrame) -> dict[str, Any]:
    judged = df.filter(pl.col("decision") != "unsure")
    k = judged.filter(pl.col("decision") == "same").height
    return {
        "reviewed": df.height,
        "unsure": df.height - judged.height,
        "same": k,
        "precision": round(k / judged.height, 4) if judged.height else None,
        "ci95": wilson(k, judged.height),
    }


def ingest(cfg: dict[str, Any], dump_path: Path) -> dict[str, Any]:
    rcfg = cfg["review"]
    dec = parse_dump(json.loads(dump_path.read_text(encoding="utf-8")))
    rows = review_rows(cfg).select("job_id", "position", "lemma", "upos", "cosine")
    unknown = dec.join(rows, on="job_id", how="anti").height
    df = rows.join(dec.drop("block"), on="job_id", how="inner").sort("position")
    out = resolve(rcfg["decisions_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    total = rows.height
    prefix = df.height
    while prefix and df["position"][prefix - 1] != prefix - 1:  # reviewed out of order
        prefix -= 1
    strata = cfg["band_strata"]
    by_stratum = []
    for lo, hi in zip(strata[:-1], strata[1:], strict=True):
        part = df.filter((pl.col("cosine") >= lo) & (pl.col("cosine") < hi))
        by_stratum.append({"cosine": [lo, hi], **_precision(part)})
    res: dict[str, Any] = {
        "pairs": total,
        "reviewed": df.height,
        "unknown_job_ids": unknown,
        "contiguous_prefix": prefix,
        "decisions": dict(df.group_by("decision").len().iter_rows()),
        "first_pass_same": _precision(df),
        "by_cosine": by_stratum,
    }
    times = df["decided_at_ms"].sort().diff().drop_nulls()
    active = times.filter((times >= 300) & (times < 120_000))  # as the page: no bursts, no breaks
    if active.len():
        res["median_seconds_per_pair"] = round(float(active.median()) / 1000, 2)
    gold_path = resolve(cfg["band_gold_path"])
    if gold_path.exists():
        gold = pl.read_csv(gold_path).select("job_id", pl.col("label").alias("gold"), "ambiguous")
        both = df.join(gold, on="job_id").filter(pl.col("decision") != "unsure")
        res["agreement_with_fresh_gold"] = {
            "pairs": both.height,
            "agree": both.filter(pl.col("decision") == pl.col("gold")).height,
            "agree_same_vs_not": both.filter(
                (pl.col("decision") == "same") == (pl.col("gold") == "same")
            ).height,
        }
    dump_json(res, resolve(rcfg["report_path"]))
    LOG.info("ingested %d decisions (%d unknown job ids) -> %s", df.height, unknown, out)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Owner review of first-pass 'same' band pairs")
    ap.add_argument("command", choices=["export", "ingest"])
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None, help="export: directory for the JSON blocks")
    ap.add_argument("--dump", type=Path, default=None, help="ingest: JSON dump of the decisions")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    if args.command == "export":
        out = args.out or resolve(cfg["review"]["export_dir"])
        idx = export(cfg, out)
        print(f"{idx['total']} pairs in {idx['blocks']} blocks -> {out}")
    else:
        if not args.dump:
            raise SystemExit("set --dump")
        res = ingest(cfg, args.dump)
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
