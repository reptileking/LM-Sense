"""Access to ``data/raw/MANIFEST.json`` written by the download stage."""

from __future__ import annotations

import json
from typing import Any

from senselm.common import resolve, sha256_file


def load_manifest(cfg: dict[str, Any]) -> dict[str, Any]:
    path = resolve(cfg.get("manifest", "data/raw/MANIFEST.json"))
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing: run `python -m senselm.ingest.download --config configs/download.yaml`"
        )
    return json.loads(path.read_text())


def source_version(cfg: dict[str, Any], name: str, *, verify: bool = False) -> str:
    """Version string for source ``name`` (config override > manifest), optionally verified."""
    if cfg.get("source_version"):
        return str(cfg["source_version"])
    entry = load_manifest(cfg)["sources"][cfg.get("source_name", name)]
    if verify and sha256_file(resolve(entry["path"])) != entry["sha256"]:
        raise RuntimeError(f"{entry['path']} does not match the manifest checksum")
    return f"{entry['version']} sha256:{entry['sha256'][:12]}"
