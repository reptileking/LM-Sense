"""Wikidata items: one streaming pass over the full JSON dump, then a config-driven selection.

``python -m senselm.ingest.wikidata_items stream --config configs/ingest_wikidata_items.yaml``
``python -m senselm.ingest.wikidata_items select --config configs/ingest_wikidata_items.yaml``

**stream.** The full dump (~157 GB gzip, ~1.6 TB JSON) does not fit on disk and is never
stored. It is fetched as 64 MB byte ranges over two parallel connections (Wikimedia caps each
connection at ~5 MB/s), reassembled in order, hashed on the fly (sha256 + md5, the latter
checked against Wikimedia's published sum) and inflated in the main process (gzip multi-member
aware). Every range's length is verified and a failed range is simply re-requested, so a
dropped connection never truncates or restarts the pass. Line filtering runs in worker
processes (in a thread it starved the network reader of the GIL); cheap byte tests run before
any JSON parsing. Items that pass go to a **superset** Parquet:

    English label AND ( linked from an English lexeme sense (P5137)
                        OR has "subclass of" (P279)
                        OR carries a cross-reference to a Phase-2 domain source:
                           NCBI taxon P685, GO P686, ChEBI P683, MeSH P486 / P6694,
                           SNOMED CT P5806, WordNet 3.1 synset P8814 )

**select.** Applies the "common nouns / concepts" filter from the config to the superset and
writes ``senses_raw_wikidata_items.parquet`` (one sense per item; gloss = English description).
The filter can be changed without re-streaming.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import threading
import time
import urllib.error
import urllib.request
import zlib
from collections import Counter
from typing import Any

import orjson
import polars as pl
import pyarrow as pa

from senselm.common import BatchWriter, Provenance, load_config, resolve, setup_logging, write_table
from senselm.ingest.download import DEFAULT_USER_AGENT
from senselm.ingest.schema import SENSE_SCHEMA, make_row

LOG = logging.getLogger("senselm.ingest.wikidata_items")
SOURCE = "wikidata_items"
XREF_PROPS = ("P685", "P686", "P683", "P486", "P6694", "P5806", "P8814")
ID_RE = re.compile(rb'^\{"type":"item","id":"(Q\d+)"')
EN_LABEL = b'"en":{"language":"en","value":'

SUPERSET_SCHEMA = pa.schema(
    [
        ("qid", pa.string()),
        ("label", pa.string()),
        ("description", pa.string()),
        ("aliases", pa.list_(pa.string())),
        ("p31", pa.list_(pa.string())),
        ("p279", pa.list_(pa.string())),
        ("xrefs", pa.string()),  # JSON {prop: [values]}
        ("enwiki", pa.string()),
        ("n_sitelinks", pa.int32()),
        ("linked", pa.bool_()),
    ]
)


# --------------------------------------------------------------------------- streaming


class SegmentedDownload:
    """Download ``url`` as fixed-size byte ranges over ``connections`` parallel connections and
    yield the segments **in order**. Each segment is one HTTP Range request whose length is
    verified, so a dropped or cleanly-closed connection just means that segment is retried;
    nothing is ever silently truncated and the stream never restarts."""

    def __init__(
        self,
        url: str,
        *,
        segment: int = 64 << 20,
        connections: int = 2,
        window: int = 6,
        timeout: int = 120,
        retries: int = 20,
        max_bytes: int | None = None,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        self.url, self.seg, self.conns, self.window = url, segment, connections, window
        self.timeout, self.retries, self.ua = timeout, retries, user_agent
        head = urllib.request.Request(url, method="HEAD", headers={"User-Agent": user_agent})
        with urllib.request.urlopen(head, timeout=timeout) as r:
            self.total = int(r.headers["Content-Length"])
        self.limit = min(self.total, max_bytes) if max_bytes else self.total
        self.n = (self.limit + self.seg - 1) // self.seg
        self.retried = 0
        self._store: dict[int, bytes] = {}
        self._next = 0
        self._cv = threading.Condition()
        self._error: BaseException | None = None

    def _fetch(self, idx: int) -> bytes:
        start = idx * self.seg
        end = min(self.limit, start + self.seg) - 1
        want = end - start + 1
        for attempt in range(self.retries + 1):
            try:
                req = urllib.request.Request(
                    self.url, headers={"User-Agent": self.ua, "Range": f"bytes={start}-{end}"}
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    if r.status != 206:
                        raise RuntimeError(f"expected 206 for a range request, got {r.status}")
                    data = r.read()
                if len(data) != want:
                    raise ConnectionError(f"segment {idx}: got {len(data)} of {want} bytes")
                return data
            except Exception as e:  # noqa: BLE001 — any transport failure (incl. http.client.IncompleteRead) is retried
                if attempt == self.retries:
                    raise
                self.retried += 1
                LOG.warning("segment %d attempt %d failed (%s); retrying", idx, attempt + 1, e)
                time.sleep(min(60, 2 ** (attempt + 1)))
        raise AssertionError("unreachable")

    def _worker(self, k: int) -> None:
        try:
            for idx in range(k, self.n, self.conns):
                with self._cv:
                    while idx >= self._next + self.window and self._error is None:
                        self._cv.wait()
                    if self._error is not None:
                        return
                data = self._fetch(idx)
                with self._cv:
                    self._store[idx] = data
                    self._cv.notify_all()
        except BaseException as e:  # surface in the consumer
            with self._cv:
                self._error = e
                self._cv.notify_all()

    def segments(self) -> Any:
        for k in range(self.conns):
            threading.Thread(target=self._worker, args=(k,), daemon=True).start()
        while self._next < self.n:
            with self._cv:
                while self._next not in self._store and self._error is None:
                    self._cv.wait()
                if self._error is not None:
                    raise self._error
                data = self._store.pop(self._next)
                self._next += 1
                self._cv.notify_all()
            yield data


def inflate_lines(segments: Any, hashes: list[Any], piece: int = 1 << 22) -> Any:
    """Hash raw bytes, inflate (multi-member gzip) and yield batches of complete lines."""
    d = zlib.decompressobj(wbits=31)
    rest = b""
    for seg in segments:
        for h in hashes:
            h.update(seg)
        for i in range(0, len(seg), piece):
            chunk = seg[i : i + piece]
            out = []
            while chunk:
                out.append(d.decompress(chunk))
                if d.eof:  # next gzip member
                    chunk = d.unused_data
                    d = zlib.decompressobj(wbits=31)
                else:
                    chunk = b""
            data = rest + b"".join(out)
            lines = data.split(b"\n")
            rest = lines.pop()
            if lines:
                yield lines
    tail = rest + d.flush()
    if tail.strip():
        yield tail.split(b"\n")


def _ids(claims: dict[str, Any], prop: str) -> list[str]:
    out = []
    for st in claims.get(prop, []):
        v = (st.get("mainsnak", {}).get("datavalue") or {}).get("value")
        if isinstance(v, dict) and v.get("id"):
            out.append(v["id"])
        elif isinstance(v, str):
            out.append(v)
    return out


def parse_item(line: bytes, linked: set[str]) -> dict[str, Any] | None:
    d = orjson.loads(line.rstrip(b","))
    label = (d.get("labels", {}).get("en") or {}).get("value")
    if not label:
        return None
    claims = d.get("claims", {})
    p279 = _ids(claims, "P279")
    xrefs = {p: _ids(claims, p) for p in XREF_PROPS if p in claims}
    is_linked = d["id"] in linked
    if not (is_linked or p279 or xrefs):
        return None
    sl = d.get("sitelinks", {})
    return {
        "qid": d["id"],
        "label": label,
        "description": (d.get("descriptions", {}).get("en") or {}).get("value"),
        "aliases": [a["value"] for a in d.get("aliases", {}).get("en", [])],
        "p31": _ids(claims, "P31"),
        "p279": p279,
        "xrefs": orjson.dumps(xrefs, option=orjson.OPT_SORT_KEYS).decode(),
        "enwiki": (sl.get("enwiki") or {}).get("title"),
        "n_sitelinks": len(sl),
        "linked": is_linked,
    }


_LINKED: set[str] = set()
_XREF_MARKERS = [f'"{p}"'.encode() for p in XREF_PROPS]


def _init_worker(linked: set[str]) -> None:
    global _LINKED
    _LINKED = linked


def filter_lines(lines: list[bytes]) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Worker: cheap byte tests, then JSON parsing of the few candidates."""
    stats: Counter[str] = Counter()
    keep = []
    for line in lines:
        m = ID_RE.match(line)
        if not m:
            continue
        stats["items"] += 1
        if EN_LABEL not in line:
            continue
        qid = m.group(1).decode()
        if not (qid in _LINKED or b'"P279"' in line or any(x in line for x in _XREF_MARKERS)):
            continue
        stats["parsed"] += 1
        rec = parse_item(line, _LINKED)
        if rec is not None:
            keep.append(rec)
    return keep, stats


def stream(cfg: dict[str, Any]) -> dict[str, Any]:
    import multiprocessing as mp
    from collections import deque

    linked = set(pl.read_parquet(resolve(cfg["linked_items"]))["qid"].to_list())
    prov = Provenance(
        stage="ingest.wikidata_items.stream", config=cfg, source_versions={SOURCE: cfg["version"]}
    )
    prov.add_input(cfg["linked_items"])
    dl = SegmentedDownload(
        cfg["url"],
        connections=int(cfg.get("connections", 2)),
        segment=int(cfg.get("segment_mb", 64)) << 20,
        max_bytes=cfg.get("max_compressed_bytes"),
        user_agent=cfg.get("user_agent", DEFAULT_USER_AGENT),
    )
    sha, md5 = hashlib.sha256(), hashlib.md5()
    stats: Counter[str] = Counter()
    out = BatchWriter(cfg["superset_output"], SUPERSET_SCHEMA, prov, batch_rows=50_000)
    workers = int(cfg.get("workers", 2))
    pool = mp.get_context("fork").Pool(workers, initializer=_init_worker, initargs=(linked,))
    inflight: deque[Any] = deque()
    t0 = last = time.time()
    done_bytes = 0

    def collect(res: Any) -> None:
        recs, st = res.get()
        stats.update(st)
        for rec in recs:
            stats["kept"] += 1
            stats["kept_linked"] += rec["linked"]
            stats["kept_p279"] += bool(rec["p279"])
            stats["kept_xref"] += rec["xrefs"] != "{}"
            out.write(rec)

    try:
        for lines in inflate_lines(dl.segments(), [sha, md5]):
            done_bytes += sum(len(x) for x in lines)
            inflight.append(pool.apply_async(filter_lines, (lines,)))
            while len(inflight) > workers * 3:  # backpressure
                collect(inflight.popleft())
            now = time.time()
            if now - last > 300:
                last = now
                frac = dl._next / max(dl.n, 1)
                LOG.info(
                    "%.1f%% (%d/%d segments, %.1f MB/s compressed, %d retried), %.0f GB inflated, "
                    "%d items, %d kept, ETA %.1f h",
                    100 * frac,
                    dl._next,
                    dl.n,
                    dl._next * dl.seg / (now - t0) / 1e6,
                    dl.retried,
                    done_bytes / 1e9,
                    stats["items"],
                    stats["kept"],
                    (now - t0) / max(frac, 1e-9) * (1 - frac) / 3600,
                )
        while inflight:
            collect(inflight.popleft())
    except BaseException:
        pool.terminate()
        out.abort()
        raise
    pool.close()
    pool.join()
    complete = dl.limit == dl.total
    st = {
        **dict(stats),
        "compressed_bytes": dl.limit,
        "decompressed_bytes": done_bytes,
        "sha256": sha.hexdigest(),
        "md5": md5.hexdigest(),
        "segments_retried": dl.retried,
        "seconds": round(time.time() - t0),
        "complete": complete,
    }
    if complete and cfg.get("md5") and st["md5"] != cfg["md5"]:
        out.abort()
        raise RuntimeError(f"md5 mismatch for the streamed dump: {st['md5']} != {cfg['md5']}")
    prov.extra["stream"] = st
    out.close(stats=st)
    LOG.info("stream done: %s", st)
    return st


# --------------------------------------------------------------------------- selection


def select(cfg: dict[str, Any]) -> dict[str, Any]:
    """Superset -> senses under the configured filter (no re-streaming needed)."""
    scfg = cfg["select"]
    sup = pl.scan_parquet(
        resolve(cfg["superset_output"])
    )  # lazy: the superset has millions of rows
    excluded = set(scfg.get("exclude_p31", []))
    has_excluded = pl.col("p31").list.eval(pl.element().is_in(list(excluded))).list.any()
    concept = (
        (pl.col("p279").list.len() > 0)
        & pl.col("description").is_not_null()
        & (pl.col("n_sitelinks") >= int(scfg.get("min_sitelinks", 1)))
        & ~has_excluded
    )
    if scfg.get("require_enwiki", True):
        concept = concept & pl.col("enwiki").is_not_null()
    if scfg.get("require_lowercase_label", False):
        # Wikidata labels common nouns in lower case; capitalised classes are mostly proper names
        # (languages, products, recurring events, breeds); items linked from lexemes are kept anyway
        concept = concept & ~pl.col("label").str.contains(r"^\p{Lu}")
    keep = pl.col("linked") | concept
    sel = (
        sup.with_columns(concept.alias("concept"))
        .filter(keep & pl.col("description").is_not_null())
        .collect()
    )
    n_superset = sup.select(pl.len()).collect().item()
    prov = Provenance(
        stage="ingest.wikidata_items.select", config=cfg, source_versions={SOURCE: cfg["version"]}
    )
    prov.add_input(cfg["superset_output"])
    rows = []
    for i, r in enumerate(sel.sort("qid").iter_rows(named=True)):
        rel = {
            "p31": r["p31"],
            "p279": r["p279"],
            "xrefs": orjson.loads(r["xrefs"]),
            "aliases": r["aliases"],
            "enwiki": r["enwiki"],
        }
        rows.append(
            make_row(
                source=SOURCE,
                source_id=r["qid"],
                lemma=r["label"],
                pos="item",
                gloss=r["description"],
                tags=(["wd:linked-from-lexeme"] if r["linked"] else [])
                + (["wd:concept"] if r["concept"] else []),
                relations=orjson.dumps(rel, option=orjson.OPT_SORT_KEYS).decode(),
                source_version=cfg["version"],
                kind="sense",
                entry_key=r["qid"],
                line_no=i,
                sense_rank=0,
            )
        )
    table = pa.Table.from_pylist(rows, schema=SENSE_SCHEMA)
    st = {
        "superset": n_superset,
        "selected": sel.height,
        "linked": int(sel["linked"].sum()),
        "concept": int(sel["concept"].sum()),
    }
    write_table(table, cfg["output"], prov, stats=st)
    LOG.info("select: %s", st)
    return st


def main() -> None:
    ap = argparse.ArgumentParser(description="Wikidata items: stream the dump / select concepts")
    ap.add_argument("command", choices=["stream", "select"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    setup_logging(args.log_level)
    cfg = load_config(args.config)
    (stream if args.command == "stream" else select)(cfg)


if __name__ == "__main__":
    main()
