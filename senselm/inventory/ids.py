"""Hierarchical, permanent token IDs and the append-only ID registry.

Layout (fixed for the lifetime of the registry, recorded in its metadata):

    token_id = (lemma_id << SENSE_BITS) | sense_index          (uint32)

* ``lemma_id`` - ``LEMMA_BITS`` = 22 bits (4,194,304 lemma slots); lemma_id 0 is never
  assigned, so token id 0 is never a lexical token.
* ``sense_index`` - ``SENSE_BITS`` = 10 bits (1,024 senses per lemma).
* The top ``RESERVED_LEMMA_SLOTS`` = 4,096 lemma slots are reserved for structural tokens
  (Phase 2): token ids >= ``STRUCTURAL_BASE`` = 0xFFC00000 (4,194,304 ids).

Rules:

* IDs are append-only and permanent. A lexical token id, once issued, always denotes the same
  canonical sense. When a sense disappears or is merged, its id becomes a tombstone
  (``status = deprecated``, optional ``replaced_by``); it is never re-issued.
* Identity across builds: a canonical sense keeps its previous id if its anchor
  (``canonical_key``) is unchanged, else if it shares source members with a previous sense of
  the same lemma (best overlap wins; ties go to the old anchor's continuation).
* New senses get the next unused ``sense_index`` of their lemma (indices of tombstones are
  never reused), in the frequency order supplied by the build.

:func:`check_append_only` is the invariant used by the build (refuses to write a violating
registry) and by the tests.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

LEMMA_BITS = 22
SENSE_BITS = 10
RESERVED_LEMMA_SLOTS = 4096
MAX_LEMMA_ID = (1 << LEMMA_BITS) - RESERVED_LEMMA_SLOTS - 1  # last lexical lemma id
STRUCTURAL_BASE = ((1 << LEMMA_BITS) - RESERVED_LEMMA_SLOTS) << SENSE_BITS
MAX_SENSE_INDEX = (1 << SENSE_BITS) - 1


@dataclass(frozen=True)
class Layout:
    lemma_bits: int = LEMMA_BITS
    sense_bits: int = SENSE_BITS
    reserved_lemma_slots: int = RESERVED_LEMMA_SLOTS

    @property
    def max_lemma_id(self) -> int:
        return (1 << self.lemma_bits) - self.reserved_lemma_slots - 1

    @property
    def max_sense_index(self) -> int:
        return (1 << self.sense_bits) - 1

    @property
    def structural_base(self) -> int:
        return ((1 << self.lemma_bits) - self.reserved_lemma_slots) << self.sense_bits

    def pack(self, lemma_id: int, sense_index: int) -> int:
        if not 1 <= lemma_id <= self.max_lemma_id:
            raise ValueError(f"lemma_id {lemma_id} outside lexical range 1..{self.max_lemma_id}")
        if not 0 <= sense_index <= self.max_sense_index:
            raise ValueError(f"sense_index {sense_index} outside 0..{self.max_sense_index}")
        return (lemma_id << self.sense_bits) | sense_index

    def unpack(self, token_id: int) -> tuple[int, int]:
        return token_id >> self.sense_bits, token_id & self.max_sense_index

    def is_structural(self, token_id: int) -> bool:
        return token_id >= self.structural_base

    def to_dict(self) -> dict[str, int]:
        return {
            "lemma_bits": self.lemma_bits,
            "sense_bits": self.sense_bits,
            "reserved_lemma_slots": self.reserved_lemma_slots,
            "structural_base": self.structural_base,
        }


REGISTRY_SCHEMA = {
    "token_id": pl.UInt32,
    "lemma_id": pl.UInt32,
    "sense_index": pl.UInt16,
    "lemma": pl.String,
    "upos": pl.String,
    "canonical_key": pl.String,
    "members": pl.List(pl.String),
    "gloss": pl.String,
    "status": pl.String,  # active | deprecated
    "created_in": pl.String,
    "deprecated_in": pl.String,
    "replaced_by": pl.UInt32,
}
LEMMA_SCHEMA = {
    "lemma_id": pl.UInt32,
    "lemma": pl.String,
    "status": pl.String,
    "created_in": pl.String,
}


def empty_registry() -> pl.DataFrame:
    return pl.DataFrame(schema=REGISTRY_SCHEMA)


def empty_lemmas() -> pl.DataFrame:
    return pl.DataFrame(schema=LEMMA_SCHEMA)


def assign_ids(
    senses: pl.DataFrame,
    prev: pl.DataFrame | None,
    prev_lemmas: pl.DataFrame | None,
    build: str,
    layout: Layout | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """Assign token ids to the canonical senses of a build.

    ``senses``: one row per canonical sense, columns ``canonical_key, lemma, upos, members
    (list[str]), gloss, order_key`` - rows must already be sorted in the desired frequency
    order within each lemma (``order_key`` ascending is used as a tiebreak-safe check).

    Returns (registry, lemma_registry, stats). The registry contains every id ever issued.
    """
    layout = layout or Layout()
    prev = prev if prev is not None else empty_registry()
    prev_lemmas = prev_lemmas if prev_lemmas is not None else empty_lemmas()
    stats: dict[str, Any] = defaultdict(int)

    # ---- lemma ids (append-only)
    lemma_ids: dict[str, int] = dict(
        zip(prev_lemmas["lemma"].to_list(), prev_lemmas["lemma_id"].to_list(), strict=True)
    )
    next_lemma = (max(lemma_ids.values()) + 1) if lemma_ids else 1
    lemma_rows = prev_lemmas.to_dicts()
    seen_now: set[str] = set()
    # first-build lemma order: more senses first, then code-point order (deterministic)
    counts = senses.group_by("lemma").len().sort(["len", "lemma"], descending=[True, False])
    for lemma in counts["lemma"].to_list():
        seen_now.add(lemma)
        if lemma not in lemma_ids:
            if next_lemma > layout.max_lemma_id:
                raise OverflowError("lemma id space exhausted")
            lemma_ids[lemma] = next_lemma
            lemma_rows.append(
                {"lemma_id": next_lemma, "lemma": lemma, "status": "active", "created_in": build}
            )
            next_lemma += 1
            stats["new_lemmas"] += 1
    for r in lemma_rows:
        if r["lemma"] not in seen_now and r["status"] == "active":
            r["status"] = "deprecated"
            stats["deprecated_lemmas"] += 1
        elif r["lemma"] in seen_now and r["status"] == "deprecated":
            r["status"] = "active"  # a lemma can come back; its id never changed

    # ---- previous sense ids
    prev_active = prev.filter(pl.col("status") == "active")
    by_key = dict(
        zip(prev_active["canonical_key"].to_list(), prev_active["token_id"].to_list(), strict=True)
    )
    member_to_tid: dict[str, int] = {}
    for tid, members in zip(
        prev_active["token_id"].to_list(), prev_active["members"].to_list(), strict=True
    ):
        for m in members or []:
            member_to_tid[m] = tid
    prev_info = {r["token_id"]: r for r in prev.to_dicts()}
    used_index: dict[int, int] = defaultdict(lambda: -1)  # lemma_id -> max sense_index ever used
    for tid in prev["token_id"].to_list():
        lid, si = layout.unpack(tid)
        used_index[lid] = max(used_index[lid], si)

    rows = senses.to_dicts()
    claim: dict[int, tuple[int, int]] = {}  # tid -> (row idx, strength)
    proposals: list[list[tuple[int, int]]] = [[] for _ in rows]  # row -> [(tid, strength)]
    for i, r in enumerate(rows):
        lid = lemma_ids[r["lemma"]]
        cand: dict[int, int] = {}
        if r["canonical_key"] in by_key:
            cand[by_key[r["canonical_key"]]] = 1_000_000  # anchor match dominates
        for m in r["members"] or []:
            tid = member_to_tid.get(m)
            if tid is not None:
                cand[tid] = cand.get(tid, 0) + 1
        for tid, strength in cand.items():
            if layout.unpack(tid)[0] == lid:  # an id can only continue within its lemma
                proposals[i].append((tid, strength))
        proposals[i].sort(key=lambda x: (-x[1], x[0]))
        for tid, strength in proposals[i]:
            best = claim.get(tid)
            # strongest claimant wins an old id; ties go to the earlier (more frequent) row
            if best is None or strength > best[1] or (strength == best[1] and i < best[0]):
                claim[tid] = (i, strength)
    # A row that won several old ids keeps its strongest claim (anchor > member overlap,
    # then lowest id); the other ids it won are deprecated with replaced_by -> the kept id.
    won: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for tid, (i, strength) in claim.items():
        won[i].append((strength, tid))
    tid_of_row: dict[int, int] = {
        i: min(lst, key=lambda st: (-st[0], st[1]))[1] for i, lst in won.items()
    }
    kept = set(tid_of_row.values())

    out: list[dict[str, Any]] = []
    for i, r in enumerate(rows):
        lid = lemma_ids[r["lemma"]]
        if i in tid_of_row:
            tid = tid_of_row[i]
            old = prev_info[tid]
            out.append(
                {
                    **old,
                    "upos": r["upos"],
                    "canonical_key": old["canonical_key"],
                    "members": sorted(set(r["members"] or [])),
                    "gloss": r["gloss"],
                    "status": "active",
                    "deprecated_in": None,
                    "replaced_by": None,
                }
            )
            stats["kept_ids"] += 1
        else:
            si = used_index[lid] + 1
            if si > layout.max_sense_index:
                raise OverflowError(
                    f"lemma {r['lemma']!r} exceeds {layout.max_sense_index + 1} senses"
                )
            used_index[lid] = si
            tid = layout.pack(lid, si)
            out.append(
                {
                    "token_id": tid,
                    "lemma_id": lid,
                    "sense_index": si,
                    "lemma": r["lemma"],
                    "upos": r["upos"],
                    "canonical_key": r["canonical_key"],
                    "members": sorted(set(r["members"] or [])),
                    "gloss": r["gloss"],
                    "status": "active",
                    "created_in": build,
                    "deprecated_in": None,
                    "replaced_by": None,
                }
            )
            stats["new_ids"] += 1
    # tombstones: previously issued ids not continued
    winner_of_claim = {tid: tid_of_row.get(i) for tid, (i, _) in claim.items()}
    for tid, old in sorted(prev_info.items()):
        if tid in kept:
            continue
        rec = dict(old)
        if rec["status"] == "active":
            rec["status"] = "deprecated"
            rec["deprecated_in"] = build
            rec["replaced_by"] = winner_of_claim.get(tid)  # merged into another id, if any
            stats["deprecated_ids"] += 1
        out.append(rec)
    reg = pl.DataFrame(out, schema=REGISTRY_SCHEMA).sort("token_id")
    lem = pl.DataFrame(lemma_rows, schema=LEMMA_SCHEMA).sort("lemma_id")
    stats["max_sense_index"] = int(reg["sense_index"].max() or 0)
    stats["max_lemma_id"] = int(lem["lemma_id"].max() or 0)
    return reg, lem, dict(stats)


def check_append_only(old: pl.DataFrame, new: pl.DataFrame) -> list[str]:
    """Return violations of the permanence rules (empty list = OK).

    * every old token_id is still present;
    * its lemma, lemma_id, sense_index, canonical_key and created_in never change;
    * a deprecated id never becomes active again;
    * token ids are unique.
    """
    problems: list[str] = []
    if new["token_id"].n_unique() != new.height:
        problems.append("duplicate token_id in new registry")
    cols = ["token_id", "lemma_id", "sense_index", "lemma", "canonical_key", "created_in", "status"]
    j = old.select(cols).join(new.select(cols), on="token_id", how="left", suffix="_new")
    missing = j.filter(pl.col("lemma_id_new").is_null())
    if missing.height:
        problems.append(
            f"{missing.height} token ids disappeared, e.g. {missing['token_id'][:5].to_list()}"
        )
    for c in ("lemma_id", "sense_index", "lemma", "canonical_key", "created_in"):
        changed = j.filter(pl.col(f"{c}_new").is_not_null() & (pl.col(c) != pl.col(f"{c}_new")))
        if changed.height:
            problems.append(
                f"{changed.height} token ids changed {c}, e.g. {changed['token_id'][:5].to_list()}"
            )
    revived = j.filter((pl.col("status") == "deprecated") & (pl.col("status_new") == "active"))
    if revived.height:
        problems.append(
            f"{revived.height} deprecated ids were revived, e.g. {revived['token_id'][:5].to_list()}"
        )
    return problems


def to_ledger(reg: pl.DataFrame) -> pl.DataFrame:
    """Slim, git-committable form of the registry: no glosses, and ``members`` replaced by
    ``other_members`` (members minus the anchor; empty for most senses)."""
    return reg.with_columns(
        pl.col("members")
        .list.set_difference(pl.concat_list("canonical_key"))
        .list.sort()
        .alias("other_members")
    ).drop("gloss", "members")


def from_ledger(ledger: pl.DataFrame) -> pl.DataFrame:
    """Inverse of :func:`to_ledger` (glosses come back as null)."""
    return ledger.with_columns(
        pl.concat_list("canonical_key", "other_members").list.unique().list.sort().alias("members"),
        pl.lit(None, dtype=pl.String).alias("gloss"),
    ).select(list(REGISTRY_SCHEMA))


# --------------------------------------------------------------------------- committed ledger files
# GitHub refuses files over 100 MB (and warns above 50 MB), so a large ledger table is written as
# shards <name>-00000.parquet, <name>-00001.parquet, ... (contiguous token_id ranges).
LEDGER_SHARD_BYTES = 45 << 20


def ledger_paths(commit_dir: Path, name: str) -> list[Path]:
    """The files holding ledger table ``name`` (a single file or its shards, in order)."""
    shards = sorted(commit_dir.glob(f"{name}-[0-9][0-9][0-9][0-9][0-9].parquet"))
    single = commit_dir / f"{name}.parquet"
    return shards or ([single] if single.exists() else [])


def read_ledger(commit_dir: Path, name: str) -> pl.DataFrame | None:
    paths = ledger_paths(commit_dir, name)
    return pl.concat([pl.read_parquet(p) for p in paths], how="vertical") if paths else None


def write_ledger(
    df: pl.DataFrame,
    commit_dir: Path,
    name: str,
    writer: Any,
    *,
    max_bytes: int = LEDGER_SHARD_BYTES,
) -> list[Path]:
    """Write ledger table ``name`` with ``writer(table, path)`` as one file, or as equal shards
    when the single file would exceed ``max_bytes``. Stale files of the other layout are removed."""
    commit_dir.mkdir(parents=True, exist_ok=True)
    old = ledger_paths(commit_dir, name)
    single = commit_dir / f"{name}.parquet"
    writer(df.to_arrow(), single)
    size = single.stat().st_size
    if size <= max_bytes:
        out = [single]
    else:
        n = -(-size // max_bytes) + 1  # ceil, plus slack: shards compress a little worse
        step = -(-df.height // n)
        out = []
        for i in range(n):
            path = commit_dir / f"{name}-{i:05d}.parquet"
            writer(df.slice(i * step, step).to_arrow(), path)
            out.append(path)
        for f in (single, single.with_name(single.name + ".meta.json")):
            f.unlink(missing_ok=True)
    for f in old:
        if f not in out:
            f.unlink(missing_ok=True)
            f.with_name(f.name + ".meta.json").unlink(missing_ok=True)
    return out
