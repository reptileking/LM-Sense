"""A seekable, resumable view of a remote file, for streaming huge gzip dumps (D-050).

``SeekableRemote`` is a read-only file object over HTTP range requests: sequential reads are served
from fixed-size segments fetched ahead over several connections (each segment's length verified
and retried on any transport error, as in D-032), and ``seek`` costs nothing until the next read.
Wrapped in ``indexed_gzip.IndexedGzipFile`` it gives a gzip stream whose seek points (a 32 KiB
window every ``spacing`` bytes of output) can be exported, so a pass over a 157 GB dump can stop
at any checkpoint and resume there in a new process instead of starting again.
"""

from __future__ import annotations

import hashlib
import io
import logging
import threading
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol

LOG = logging.getLogger("senselm.ingest.remote_gzip")


class RangeSource(Protocol):
    total: int

    def fetch(self, start: int, end: int) -> bytes:  # inclusive end
        ...


class HttpRange:
    def __init__(self, url: str, *, user_agent: str, timeout: int = 120, retries: int = 20) -> None:
        self.url, self.ua, self.timeout, self.retries = url, user_agent, timeout, retries
        head = urllib.request.Request(url, method="HEAD", headers={"User-Agent": user_agent})
        with urllib.request.urlopen(head, timeout=timeout) as r:
            self.total = int(r.headers["Content-Length"])
        self.retried = 0

    def fetch(self, start: int, end: int) -> bytes:
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
                    raise ConnectionError(f"bytes {start}-{end}: got {len(data)} of {want}")
                return data
            except Exception as e:  # noqa: BLE001 - every transport failure is retried
                if attempt == self.retries:
                    raise
                self.retried += 1
                LOG.warning(
                    "range %d-%d attempt %d failed (%s); retrying", start, end, attempt + 1, e
                )
                time.sleep(min(60, 2 ** (attempt + 1)))
        raise AssertionError("unreachable")


class FileRange:
    """Local stand-in for tests."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.total = self.path.stat().st_size
        self.retried = 0

    def fetch(self, start: int, end: int) -> bytes:
        with self.path.open("rb") as f:
            f.seek(start)
            return f.read(end - start + 1)


class _Prefetch:
    """Segments from ``start`` in order, fetched ahead by ``connections`` threads."""

    def __init__(
        self, src: RangeSource, start: int, segment: int, connections: int, window: int
    ) -> None:
        self.src, self.start, self.seg = src, start, segment
        self.n = max(0, (src.total - start + segment - 1) // segment)
        self.conns, self.window = connections, window
        self.store: dict[int, bytes] = {}
        self.next = 0
        self.cv = threading.Condition()
        self.error: BaseException | None = None
        self.stopped = False
        for k in range(connections):
            threading.Thread(target=self._worker, args=(k,), daemon=True).start()

    def _worker(self, k: int) -> None:
        try:
            for idx in range(k, self.n, self.conns):
                with self.cv:
                    while idx >= self.next + self.window and not self.stopped:
                        self.cv.wait()
                    if self.stopped:
                        return
                a = self.start + idx * self.seg
                data = self.src.fetch(a, min(self.src.total, a + self.seg) - 1)
                with self.cv:
                    self.store[idx] = data
                    self.cv.notify_all()
        except BaseException as e:  # surface in the consumer
            with self.cv:
                self.error = e
                self.cv.notify_all()

    def __iter__(self) -> Iterator[bytes]:
        while self.next < self.n:
            with self.cv:
                while self.next not in self.store and self.error is None:
                    self.cv.wait()
                if self.error is not None:
                    raise self.error
                data = self.store.pop(self.next)
                self.next += 1
                self.cv.notify_all()
            yield data

    def stop(self) -> None:
        with self.cv:
            self.stopped = True
            self.cv.notify_all()


class SeekableRemote(io.RawIOBase):
    """Read-only, seekable file over a :class:`RangeSource`; hashes the bytes it serves while they
    are contiguous from offset 0 (``hashes_valid`` turns False after any jump)."""

    def __init__(
        self, src: RangeSource, *, segment: int = 64 << 20, connections: int = 2, window: int = 6
    ) -> None:
        super().__init__()
        self.src, self.seg, self.conns, self.window = src, segment, connections, window
        self.pos = 0
        self._buf = b""
        self._buf_start = 0
        self._it: Iterator[bytes] | None = None
        self._pf: _Prefetch | None = None
        self.sha256, self.md5 = hashlib.sha256(), hashlib.md5()
        self._hashed_to = 0
        self.hashes_valid = True

    # io.RawIOBase protocol
    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            new = offset
        elif whence == io.SEEK_CUR:
            new = self.pos + offset
        else:
            new = self.src.total + offset
        self.pos = max(0, min(new, self.src.total))
        return self.pos

    def _restart(self) -> None:
        if self._pf is not None:
            self._pf.stop()
        self._pf = _Prefetch(self.src, self.pos, self.seg, self.conns, self.window)
        self._it = iter(self._pf)
        self._buf, self._buf_start = b"", self.pos

    def readinto(self, b: Any) -> int:
        want = len(b)
        if want == 0 or self.pos >= self.src.total:
            return 0
        end = self._buf_start + len(self._buf)
        if not (self._buf_start <= self.pos <= end) or (self.pos == end and self._it is None):
            self._restart()
        while self._buf_start + len(self._buf) < min(self.pos + want, self.src.total):
            assert self._it is not None
            try:
                chunk = next(self._it)
            except StopIteration:
                break
            drop = self.pos - self._buf_start  # keep memory bounded: forget what was served
            if drop > 0:
                self._buf, self._buf_start = self._buf[drop:], self.pos
            self._buf += chunk
        i = self.pos - self._buf_start
        data = self._buf[i : i + want]
        n = len(data)
        b[:n] = data
        if self.hashes_valid:
            if self.pos <= self._hashed_to < self.pos + n:  # hash only bytes not seen before
                fresh = data[self._hashed_to - self.pos :]
                self.sha256.update(fresh)
                self.md5.update(fresh)
                self._hashed_to += len(fresh)
            elif self.pos > self._hashed_to:  # a jump past unhashed bytes
                self.hashes_valid = False
        self.pos += n
        return n

    def close(self) -> None:
        if self._pf is not None:
            self._pf.stop()
        super().close()


def open_gzip(
    src: RangeSource, *, spacing: int, segment: int, connections: int
) -> tuple[Any, SeekableRemote]:
    """An ``indexed_gzip`` reader over ``src`` (plus the raw remote file, for progress/hashes)."""
    import indexed_gzip

    if (4 * spacing) % (1 << 32) == 0:
        # indexed_gzip 1.10 computes 4 * spacing in 32 bits; a multiple of 1 GiB wraps to 0 and
        # the process dies with SIGFPE on the first read (found 2026-11-03)
        spacing -= 1 << 20
    raw = SeekableRemote(src, segment=segment, connections=connections)
    gz = indexed_gzip.IndexedGzipFile(
        fileobj=raw, spacing=spacing, readbuf_size=32 << 20, buffer_size=128 << 20
    )
    return gz, raw
