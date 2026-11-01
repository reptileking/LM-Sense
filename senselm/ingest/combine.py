"""Union the per-source raw sense files into ``data/interim/senses_raw.parquet``.

``python -m senselm.ingest.combine --config configs/ingest.yaml``

Streams record batches (never loads a whole source) and checks that ``source_id`` is unique
within each source (the unified key is ``(source, source_id)``).
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

import pyarrow.parquet as pq

from senselm.common import BatchWriter, Provenance, load_config, resolve, stage_args
from senselm.ingest.schema import SENSE_SCHEMA

LOG = logging.getLogger("senselm.ingest.combine")


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    prov = Provenance(stage="ingest.combine", config=cfg)
    for path in cfg["inputs"]:
        prov.add_input(path)
    stats: Counter[str] = Counter()
    out = BatchWriter(cfg["output"], SENSE_SCHEMA, prov)
    try:
        for path in cfg["inputs"]:
            seen: set[str] = set()
            pf = pq.ParquetFile(resolve(path))
            for batch in pf.iter_batches(batch_size=int(cfg.get("batch_rows", 100_000))):
                ids = batch.column("source_id").to_pylist()
                for source, sid in zip(batch.column("source").to_pylist(), ids, strict=True):
                    if sid in seen:
                        raise ValueError(f"duplicate source_id {sid!r} in {path}")
                    seen.add(sid)
                    stats[f"rows:{source}"] += 1
                out.write_batch(batch)
    except BaseException:
        out.abort()
        raise
    out_stats = dict(sorted(stats.items()))
    out.close(stats=out_stats)
    LOG.info("combined: %s", out_stats)
    return out_stats


def main() -> None:
    args = stage_args("Combine per-source raw sense files")
    run(load_config(args.config))


if __name__ == "__main__":
    main()
