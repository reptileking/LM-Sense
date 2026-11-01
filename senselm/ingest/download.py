"""Download raw source dumps listed in ``configs/download.yaml``.

``python -m senselm.ingest.download --config configs/download.yaml [--only NAME ...]``

* Streams each URL to ``<path>.part`` (resumable via HTTP Range) and renames on success.
* Verifies ``sha256`` when pinned in the config; a mismatch is a hard error (the upstream
  file changed: update the pin deliberately, never silently).
* Records url, size, sha256, Last-Modified and ETag in ``data/raw/MANIFEST.json``.
* Never tries alternative hosts: if a URL is blocked it stops and prints the exact URL and
  target path so the file can be fetched manually (project ground rule).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from senselm.common import dump_json, load_config, resolve, setup_logging, sha256_file

LOG = logging.getLogger("senselm.download")
CHUNK = 1 << 20
# Wikimedia and others reject anonymous clients (e.g. Python's default User-Agent) with 403;
# their policies ask for a descriptive agent with a contact URL. No personal data.
DEFAULT_USER_AGENT = "SenseLM-research/0.1 (+https://github.com/reptileking/LM-Sense)"


class BlockedDownload(RuntimeError):
    pass


def _head(url: str, timeout: int, user_agent: str = DEFAULT_USER_AGENT) -> dict[str, str]:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return {k.lower(): v for k, v in resp.headers.items()}


def fetch(
    url: str, dest: Path, *, timeout: int = 60, user_agent: str = DEFAULT_USER_AGENT
) -> dict[str, Any]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    try:
        headers = _head(url, timeout, user_agent)
    except urllib.error.HTTPError as e:  # 403/407 from the egress proxy or server
        raise BlockedDownload(f"HTTP {e.code} for {url}") from e
    except urllib.error.URLError as e:
        raise BlockedDownload(f"cannot reach {url}: {e.reason}") from e
    total = int(headers.get("content-length", 0)) or None
    start = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    if start:
        req.add_header("Range", f"bytes={start}-")
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        raise BlockedDownload(f"HTTP {e.code} for {url}") from e
    with resp, part.open("ab" if start and resp.status == 206 else "wb") as out:
        done = start if resp.status == 206 else 0
        last_report = 0
        while True:
            block = resp.read(CHUNK)
            if not block:
                break
            out.write(block)
            done += len(block)
            if total and done - last_report > 200 * CHUNK:
                LOG.info("  %s: %.0f / %.0f MB", dest.name, done / 1e6, total / 1e6)
                last_report = done
    if total and part.stat().st_size != total:
        raise RuntimeError(f"incomplete download {url}: {part.stat().st_size} != {total} bytes")
    part.replace(dest)
    return {
        "last_modified": headers.get("last-modified"),
        "etag": headers.get("etag"),
        "content_length": total,
    }


def run(cfg: dict[str, Any], only: list[str] | None = None) -> dict[str, Any]:
    manifest_path = resolve(cfg.get("manifest", "data/raw/MANIFEST.json"))
    manifest: dict[str, Any] = (
        json.loads(manifest_path.read_text()) if manifest_path.exists() else {"sources": {}}
    )
    blocked: list[tuple[str, str, str]] = []
    for src in cfg["sources"]:
        name = src["name"]
        if only and name not in only:
            continue
        dest = resolve(src["path"])
        pinned = src.get("sha256")
        if dest.exists() and (not pinned or sha256_file(dest) == pinned):
            LOG.info("%s: present (%s)", name, dest)
            meta = manifest["sources"].get(name, {})
            if not meta.get("last_modified"):
                try:  # record upstream identity for files fetched outside this stage
                    h = _head(
                        src["url"],
                        int(cfg.get("timeout_s", 60)),
                        cfg.get("user_agent", DEFAULT_USER_AGENT),
                    )
                    if int(h.get("content-length", -1)) == dest.stat().st_size:
                        meta = {
                            "last_modified": h.get("last-modified"),
                            "etag": h.get("etag"),
                            "content_length": int(h["content-length"]),
                        }
                except (urllib.error.URLError, ValueError, KeyError):
                    pass
        else:
            LOG.info("%s: downloading %s", name, src["url"])
            try:
                meta = fetch(
                    src["url"],
                    dest,
                    timeout=int(cfg.get("timeout_s", 60)),
                    user_agent=cfg.get("user_agent", DEFAULT_USER_AGENT),
                )
            except BlockedDownload as e:
                blocked.append((name, src["url"], str(dest)))
                LOG.error("%s: BLOCKED (%s)", name, e)
                continue
        digest = sha256_file(dest)
        if src.get("md5"):  # publishers such as Wikimedia only provide md5/sha1 sums
            import hashlib as _h

            md5 = _h.md5()
            with dest.open("rb") as fh:
                for block in iter(lambda: fh.read(CHUNK), b""):
                    md5.update(block)
            if md5.hexdigest() != src["md5"]:
                raise RuntimeError(f"{name}: md5 mismatch ({md5.hexdigest()} != {src['md5']})")
        if pinned and digest != pinned:
            raise RuntimeError(
                f"{name}: sha256 mismatch ({digest} != pinned {pinned}). The upstream file has "
                "changed; update the pin in the config deliberately (and bump the version)."
            )
        manifest["sources"][name] = {
            **{k: v for k, v in meta.items() if k in ("last_modified", "etag", "content_length")},
            "url": src["url"],
            "path": src["path"],
            "version": src.get("version"),
            "licence": src.get("licence"),
            "bytes": dest.stat().st_size,
            "sha256": digest,
        }
        LOG.info("%s: ok sha256=%s", name, digest[:16])
    dump_json(manifest, manifest_path)
    if blocked:
        msg = "\n".join(f"  - {n}: fetch {u}\n      and save it to {p}" for n, u, p in blocked)
        LOG.error("Some downloads are blocked. Fetch these manually, then re-run:\n%s", msg)
        manifest["blocked"] = blocked
    return manifest


def sha256_stream(path: Path) -> str:  # pragma: no cover - convenience for notebooks
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these source names")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    setup_logging(args.log_level)
    manifest = run(load_config(args.config), args.only)
    if manifest.get("blocked"):
        sys.exit(2)


if __name__ == "__main__":
    main()
