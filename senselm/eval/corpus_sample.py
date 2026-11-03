"""General-text sample for codec evaluation (coverage, round-trip, sequence length).

``python -m senselm.eval.corpus_sample --config configs/corpus_sample.yaml``

Reads the first ``rows`` documents of one pinned FineWeb-Edu shard over HTTP range requests (only
the row groups needed are fetched; the 2 GB shard is never stored) and writes them with a fixed
split: the first ``dev_docs`` documents are for developing codec rules, the rest are held out
for the Phase 3 metrics.
"""

from __future__ import annotations

import logging
from typing import Any

import polars as pl

from senselm.common import (
    Provenance,
    dump_json,
    load_config,
    resolve,
    setup_logging,
    stage_args,
    write_table,
)

LOG = logging.getLogger("senselm.eval.corpus_sample")


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    rows, dev = int(cfg["rows"]), int(cfg["dev_docs"])
    df = (
        pl.scan_parquet(cfg["url"])
        .select("id", "url", "text", "language", "score")
        .head(rows)
        .collect()
        .with_row_index("doc")
        .with_columns(
            split=pl.when(pl.col("doc") < dev).then(pl.lit("dev")).otherwise(pl.lit("eval")),
            words=pl.col("text").str.count_matches(r"\S+"),
            chars=pl.col("text").str.len_chars(),
        )
    )
    stats = {
        "docs": df.height,
        **{
            f"{s}_{k}": int(df.filter(pl.col("split") == s)[k].sum())
            for s in ("dev", "eval")
            for k in ("words", "chars")
        },
    }
    prov = Provenance(
        stage="eval.corpus_sample", config=cfg, source_versions={"fineweb_edu": cfg["version"]}
    )
    write_table(df.to_arrow(), cfg["output"], prov, stats=stats)
    dump_json(stats, resolve(cfg["output"]).with_suffix(".stats.json"))
    LOG.info("sample: %s", stats)
    return stats


def main() -> None:
    args = stage_args(__doc__.splitlines()[0])
    setup_logging()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
