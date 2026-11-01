"""Sentence embeddings for gloss alignment, with a resumable content-addressed cache.

* Hugging Face encoder (``AutoModel``), explicit pooling (``cls`` for BGE, ``mean`` for
  E5/GTE), L2-normalised output, texts sorted by length to minimise padding.
* ``dtype: bfloat16`` runs under CPU autocast (AMX on Sapphire Rapids); ``float32`` is exact.
* The cache stores float16 vectors keyed by a hash of the exact input text, one directory per
  (model, revision, pooling, prefix, max_length, dtype). Re-runs reuse cached vectors, which
  makes the alignment deterministic even though CPU matmul results can differ in the last
  bits between batch compositions.

Benchmark: ``python -m senselm.inventory.embed --config configs/align.yaml --benchmark``
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from senselm.common import load_config, resolve, setup_logging
from senselm.hardware import resolve_auto

LOG = logging.getLogger("senselm.inventory.embed")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def text_key(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


class Embedder:
    def __init__(self, cfg: dict[str, Any]) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.cfg = cfg
        self.model_name: str = cfg["model"]
        self.revision: str | None = cfg.get("revision") or None
        self.pooling: str = cfg.get("pooling", "cls")
        self.normalize: bool = bool(cfg.get("normalize", True))
        self.prefix: str = cfg.get("prefix", "") or ""
        self.max_length = int(cfg.get("max_length", 128))
        self.batch_size = int(resolve_auto(cfg.get("batch_size", "auto"), "embed_batch_size"))
        self.device = str(resolve_auto(cfg.get("device", "auto"), "device"))
        self.dtype = cfg.get("dtype", "float32")
        threads = int(resolve_auto(cfg.get("threads", "auto"), "torch_threads"))
        torch.set_num_threads(threads)
        self.tok = AutoTokenizer.from_pretrained(self.model_name, revision=self.revision)
        self.model = AutoModel.from_pretrained(self.model_name, revision=self.revision)
        self.model.eval().to(self.device)
        self.commit = self._resolve_commit()
        self.dim = int(self.model.config.hidden_size)
        LOG.info(
            "embedder %s@%s pooling=%s dtype=%s device=%s threads=%d batch=%d",
            self.model_name,
            str(self.commit)[:10],
            self.pooling,
            self.dtype,
            self.device,
            threads,
            self.batch_size,
        )

    def _resolve_commit(self) -> str:
        """The exact HF commit loaded (transformers 5 no longer exposes config._commit_hash)."""
        commit = getattr(self.model.config, "_commit_hash", None)
        if commit:
            return str(commit)
        try:
            from huggingface_hub import snapshot_download

            path = snapshot_download(
                self.model_name,
                revision=self.revision,
                local_files_only=True,
                allow_patterns=["config.json"],
            )
            return Path(path).name
        except Exception:  # offline cache layouts differ; fall back to the requested revision
            return self.revision or "unknown"

    def cache_tag(self) -> str:
        raw = f"{self.model_name}@{self.commit}|{self.pooling}|{self.prefix}|{self.max_length}|{self.dtype}"
        slug = re.sub(r"[^A-Za-z0-9]+", "-", self.model_name).strip("-")
        return f"{slug}-{hashlib.blake2b(raw.encode(), digest_size=6).hexdigest()}"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        if not texts:
            return out
        order = np.argsort([-len(t) for t in texts], kind="stable")
        autocast = (
            torch.autocast(device_type=self.device, dtype=torch.bfloat16)
            if self.dtype == "bfloat16"
            else _nullcontext()
        )
        with torch.inference_mode(), autocast:
            for s in range(0, len(texts), self.batch_size):
                idx = order[s : s + self.batch_size]
                batch = [self.prefix + texts[i] for i in idx]
                enc = self.tok(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                hidden = self.model(**enc).last_hidden_state.float()
                if self.pooling == "cls":
                    vec = hidden[:, 0]
                elif self.pooling == "mean":
                    mask = enc["attention_mask"].unsqueeze(-1).float()
                    vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                else:
                    raise ValueError(f"unknown pooling {self.pooling}")
                if self.normalize:
                    vec = torch.nn.functional.normalize(vec, p=2, dim=-1)
                out[idx] = vec.cpu().numpy()
        return out


class _nullcontext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *a: Any) -> None:
        return None


class EmbeddingCache:
    """Append-only Parquet shards of (key, vector float16)."""

    def __init__(self, root: Path, tag: str, dim: int) -> None:
        self.dir = root / tag
        self.dir.mkdir(parents=True, exist_ok=True)
        self.dim = dim

    def load(self) -> tuple[dict[str, int], np.ndarray]:
        keys: list[str] = []
        mats: list[np.ndarray] = []
        for f in sorted(self.dir.glob("part-*.parquet")):
            t = pq.read_table(f)
            keys.extend(t.column("key").to_pylist())
            flat = t.column("vec").combine_chunks().flatten().to_numpy(zero_copy_only=False)
            mats.append(flat.reshape(-1, self.dim).astype(np.float32))
        mat = np.concatenate(mats) if mats else np.zeros((0, self.dim), np.float32)
        return {k: i for i, k in enumerate(keys)}, mat

    def add(self, keys: list[str], vecs: np.ndarray) -> Path:
        n = len(list(self.dir.glob("part-*.parquet")))
        path = self.dir / f"part-{n:05d}.parquet"
        arr = pa.FixedSizeListArray.from_arrays(pa.array(vecs.astype(np.float16).ravel()), self.dim)
        tmp = path.with_suffix(".tmp")
        pq.write_table(pa.table({"key": keys, "vec": arr}), tmp, compression="zstd")
        tmp.replace(path)
        return path


def embed_texts(
    texts: Sequence[str], cfg: dict[str, Any], embedder: Embedder | None = None
) -> tuple[np.ndarray, dict[str, Any]]:
    """Embed ``texts`` (any order, duplicates allowed) using and filling the cache."""
    emb = embedder or Embedder(cfg)
    cache = EmbeddingCache(
        resolve(cfg.get("cache_dir", "data/interim/embeddings")), emb.cache_tag(), emb.dim
    )
    index, mat = cache.load()
    keys = [text_key(t) for t in texts]
    missing: dict[str, str] = {}
    for k, t in zip(keys, texts, strict=True):
        if k not in index and k not in missing:
            missing[k] = t
    LOG.info(
        "embedding: %d texts, %d unique, %d cached, %d to compute",
        len(texts),
        len(set(keys)),
        len(set(keys)) - len(missing),
        len(missing),
    )
    chunk = int(cfg.get("write_every", 20_000))
    todo = sorted(missing.items(), key=lambda kv: kv[0])  # deterministic shard contents
    t0 = time.time()
    for s in range(0, len(todo), chunk):
        part = todo[s : s + chunk]
        vecs = emb.encode([t for _, t in part])
        cache.add([k for k, _ in part], vecs)
        done = s + len(part)
        rate = done / max(time.time() - t0, 1e-6)
        LOG.info(
            "  %d/%d embedded (%.1f texts/s, ETA %.0f min)",
            done,
            len(todo),
            rate,
            (len(todo) - done) / rate / 60,
        )
    if todo:
        index, mat = cache.load()
    out = mat[[index[k] for k in keys]]
    meta = {
        "model": emb.model_name,
        "revision": emb.commit,
        "pooling": emb.pooling,
        "dtype": emb.dtype,
        "max_length": emb.max_length,
        "cache": str(cache.dir),
        "computed": len(todo),
    }
    return out, meta


def benchmark(
    base_cfg: dict[str, Any], models: list[dict[str, Any]], texts: list[str]
) -> list[dict[str, Any]]:
    rows = []
    for m in models:
        cfg = {**base_cfg, **m}
        emb = Embedder(cfg)
        emb.encode(texts[:32])  # warm-up
        t0 = time.time()
        emb.encode(texts)
        dt = time.time() - t0
        rows.append(
            {
                "model": cfg["model"],
                "dtype": cfg.get("dtype", "float32"),
                "texts_per_s": round(len(texts) / dt, 1),
                "n": len(texts),
            }
        )
        LOG.info("benchmark %s", rows[-1])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Embedding utilities")
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--n", type=int, default=512)
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    if args.benchmark:
        import polars as pl

        s = pl.read_parquet(resolve(cfg["input"]), columns=["gloss"]).sample(args.n, seed=0)
        texts = [t for t in s["gloss"].to_list() if t]
        print(benchmark(cfg["embedding"], cfg.get("benchmark_models", [cfg["embedding"]]), texts))


if __name__ == "__main__":
    main()
