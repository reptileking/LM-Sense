"""Shared utilities for every pipeline stage.

Every stage is a CLI (``python -m senselm.<stage> --config configs/<stage>.yaml``) that:

* loads a YAML config (and records it verbatim plus its hash),
* checksums its inputs,
* writes Parquet outputs whose key-value metadata carries a provenance record
  (config, input checksums, source versions, code revision), plus a sidecar
  ``<output>.meta.json`` with the same record and a timestamp.

The Parquet metadata deliberately contains no timestamps so that a re-run with the
same config and inputs produces identical data and identical embedded metadata.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import hashlib
import json
import logging
import platform
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
META_KEY = b"senselm"
LOG = logging.getLogger("senselm")

_TRACKED_PACKAGES = (
    "polars",
    "duckdb",
    "pyarrow",
    "orjson",
    "lxml",
    "numpy",
    "scipy",
    "torch",
    "transformers",
    "tokenizers",
)


# --------------------------------------------------------------------------- logging / CLI


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


def stage_args(description: str, argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Standard CLI: ``--config`` (required) and ``--log-level``."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, type=Path, help="YAML config for this stage")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    setup_logging(args.log_level)
    return args


# --------------------------------------------------------------------------- config


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"config {path} must be a mapping")
    cfg.setdefault("_config_path", str(path))
    return cfg


def config_hash(cfg: Mapping[str, Any]) -> str:
    clean = {k: v for k, v in cfg.items() if not str(k).startswith("_")}
    blob = json.dumps(clean, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def resolve(path: str | Path) -> Path:
    """Resolve a config path relative to the repository root."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


# --------------------------------------------------------------------------- checksums / versions


def sha256_file(path: str | Path, chunk_size: int = 1 << 23) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def git_revision() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit, dirty = "unknown", True
    return {"git_commit": commit, "dirty": dirty}


def package_versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for name in _TRACKED_PACKAGES:
        with contextlib.suppress(importlib_metadata.PackageNotFoundError):
            out[name] = importlib_metadata.version(name)
    return out


# --------------------------------------------------------------------------- provenance


@dataclass
class Provenance:
    """Provenance record embedded in every output."""

    stage: str
    config: Mapping[str, Any]
    inputs: list[dict[str, Any]] = field(default_factory=list)
    source_versions: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def add_input(self, path: str | Path, *, checksum: bool = True) -> dict[str, Any]:
        p = resolve(path)
        rec: dict[str, Any] = {"path": _rel(p), "bytes": p.stat().st_size}
        if checksum:
            rec["sha256"] = sha256_file(p)
            # Propagate upstream source versions recorded in Parquet inputs.
            if p.suffix == ".parquet":
                upstream = read_provenance(p)
                if upstream:
                    rec["upstream_config_sha256"] = upstream.get("config_sha256")
                    for k, v in (upstream.get("source_versions") or {}).items():
                        self.source_versions.setdefault(k, v)
        self.inputs.append(rec)
        return rec

    def record(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "config_sha256": config_hash(self.config),
            "config": {k: v for k, v in self.config.items() if not str(k).startswith("_")},
            "config_path": self.config.get("_config_path"),
            "inputs": self.inputs,
            "source_versions": self.source_versions,
            "code": git_revision(),
            "packages": package_versions(),
            **({"extra": self.extra} if self.extra else {}),
        }

    def parquet_metadata(self) -> dict[bytes, bytes]:
        return {META_KEY: json.dumps(self.record(), sort_keys=True, default=str).encode("utf-8")}

    def write_sidecar(self, out_path: str | Path, stats: Mapping[str, Any] | None = None) -> Path:
        out_path = resolve(out_path)
        side = out_path.with_name(out_path.name + ".meta.json")
        rec = self.record()
        rec["created_at"] = _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")
        if out_path.exists():
            rec["output"] = {"path": _rel(out_path), "bytes": out_path.stat().st_size}
            if out_path.is_file():
                rec["output"]["sha256"] = sha256_file(out_path)
        if stats:
            rec["stats"] = dict(stats)
        side.write_text(
            json.dumps(rec, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
        )
        return side


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def read_provenance(path: str | Path) -> dict[str, Any] | None:
    try:
        md = pq.read_schema(resolve(path)).metadata or {}
    except (OSError, pa.ArrowInvalid):
        return None
    raw = md.get(META_KEY)
    return json.loads(raw) if raw else None


# --------------------------------------------------------------------------- Parquet I/O


def write_table(
    table: pa.Table,
    path: str | Path,
    prov: Provenance,
    *,
    row_group_size: int = 128_000,
    stats: Mapping[str, Any] | None = None,
) -> Path:
    """Write a whole table with provenance metadata (+ sidecar)."""
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    md = dict(table.schema.metadata or {})
    md.update(prov.parquet_metadata())
    table = table.replace_schema_metadata(md)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp, row_group_size=row_group_size, compression="zstd")
    tmp.replace(path)
    prov.write_sidecar(path, stats=stats)
    LOG.info("wrote %s (%d rows, %.1f MB)", _rel(path), table.num_rows, path.stat().st_size / 1e6)
    return path


class BatchWriter:
    """Streaming Parquet writer: buffer rows (dicts) and flush them in batches."""

    def __init__(
        self,
        path: str | Path,
        schema: pa.Schema,
        prov: Provenance,
        *,
        batch_rows: int = 50_000,
    ) -> None:
        self.path = resolve(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.tmp = self.path.with_name(self.path.name + ".tmp")
        md = dict(schema.metadata or {})
        md.update(prov.parquet_metadata())
        self.schema = schema.with_metadata(md)
        self.prov = prov
        self.batch_rows = batch_rows
        self._buf: list[dict[str, Any]] = []
        self._writer = pq.ParquetWriter(self.tmp, self.schema, compression="zstd")
        self.rows = 0

    def write(self, row: dict[str, Any]) -> None:
        self._buf.append(row)
        if len(self._buf) >= self.batch_rows:
            self.flush()

    def write_many(self, rows: Iterable[dict[str, Any]]) -> None:
        for r in rows:
            self.write(r)

    def write_batch(self, batch: pa.RecordBatch | pa.Table) -> None:
        """Write an Arrow batch/table directly (must match the schema, metadata aside)."""
        self.flush()
        if isinstance(batch, pa.RecordBatch):
            batch = pa.Table.from_batches([batch])
        table = batch.select(self.schema.names).cast(self.schema)
        self._writer.write_table(table)
        self.rows += table.num_rows

    def flush(self) -> None:
        if not self._buf:
            return
        batch = pa.RecordBatch.from_pylist(self._buf, schema=self.schema)
        self._writer.write_batch(batch)
        self.rows += len(self._buf)
        self._buf.clear()

    def close(self, stats: Mapping[str, Any] | None = None) -> Path:
        self.flush()
        self._writer.close()
        self.tmp.replace(self.path)
        self.prov.write_sidecar(self.path, stats=stats)
        LOG.info(
            "wrote %s (%d rows, %.1f MB)",
            _rel(self.path),
            self.rows,
            self.path.stat().st_size / 1e6,
        )
        return self.path

    def abort(self) -> None:
        """Discard the partial output (no half-written files are ever left behind)."""
        self._buf.clear()
        self._writer.close()
        self.tmp.unlink(missing_ok=True)

    def __enter__(self) -> BatchWriter:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


def dump_json(obj: Any, path: str | Path) -> Path:
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path
