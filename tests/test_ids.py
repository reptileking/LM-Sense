"""ID layout and registry permanence.

``test_ids_never_change_between_builds`` is the project's guard that existing IDs never
change: it simulates a sequence of builds with additions, removals, merges, splits and
reorderings and checks every registry against its predecessor. Two tests apply the same
check to the real registry: ``test_committed_registry_is_append_only`` compares the git-tracked
``registry/id_registry.parquet`` in the working tree with the version committed at HEAD (so a
rebuild that changes any existing id fails before it can be committed, on any machine), and
``test_registry_history`` walks the local snapshots under data/processed.
"""

from __future__ import annotations

import io
import subprocess
from pathlib import Path

import polars as pl
import pytest

from senselm.inventory.ids import (
    MAX_SENSE_INDEX,
    STRUCTURAL_BASE,
    Layout,
    assign_ids,
    check_append_only,
)

REPO = Path(__file__).resolve().parents[1]


def senses(rows: list[tuple[str, str, list[str]]]) -> pl.DataFrame:
    """rows: (canonical_key, lemma, members) in frequency order."""
    return pl.DataFrame(
        {
            "canonical_key": [r[0] for r in rows],
            "lemma": [r[1] for r in rows],
            "upos": ["NOUN"] * len(rows),
            "members": [r[2] for r in rows],
            "gloss": [f"gloss of {r[0]}" for r in rows],
            "order_key": list(range(len(rows))),
        }
    )


def test_layout_pack_unpack() -> None:
    lay = Layout()
    assert lay.pack(1, 0) == 1 << 10
    assert lay.unpack(lay.pack(12345, 678)) == (12345, 678)
    assert lay.structural_base == STRUCTURAL_BASE == 0xFFC00000
    assert not lay.is_structural(lay.pack(lay.max_lemma_id, MAX_SENSE_INDEX))
    assert lay.pack(lay.max_lemma_id, MAX_SENSE_INDEX) < STRUCTURAL_BASE
    assert lay.is_structural(STRUCTURAL_BASE)
    with pytest.raises(ValueError):
        lay.pack(0, 0)  # lemma id 0 is never lexical
    with pytest.raises(ValueError):
        lay.pack(lay.max_lemma_id + 1, 0)  # reserved structural slots
    with pytest.raises(ValueError):
        lay.pack(5, MAX_SENSE_INDEX + 1)


def test_first_build_orders_sense_indices() -> None:
    reg, lem, _ = assign_ids(
        senses(
            [
                ("o:bank1", "bank", ["o:bank1", "w:b1"]),
                ("o:bank2", "bank", ["o:bank2"]),
                ("w:run1", "run", ["w:run1"]),
            ]
        ),
        None,
        None,
        "b1",
    )
    bank = reg.filter(pl.col("lemma") == "bank").sort("sense_index")
    assert bank["canonical_key"].to_list() == ["o:bank1", "o:bank2"]
    assert bank["sense_index"].to_list() == [0, 1]
    assert set(lem["lemma"]) == {"bank", "run"}
    assert lem.filter(pl.col("lemma") == "bank")["lemma_id"].item() == 1  # more senses -> lower id
    assert (reg["status"] == "active").all()


def test_ids_never_change_between_builds() -> None:
    b1 = senses(
        [
            ("o:a1", "a", ["o:a1", "w:a1"]),
            ("o:a2", "a", ["o:a2"]),
            ("w:a3", "a", ["w:a3"]),
            ("o:b1", "b", ["o:b1"]),
            ("w:c1", "c", ["w:c1"]),
        ]
    )
    r1, l1, _ = assign_ids(b1, None, None, "b1")
    ids1 = dict(zip(r1["canonical_key"], r1["token_id"], strict=True))

    # build 2: reorder, add a sense, remove w:c1 (lemma c vanishes), merge o:a2 + w:a3,
    # and the anchor of a1 survives with an extra member
    b2 = senses(
        [
            ("w:a4", "a", ["w:a4"]),
            ("o:a1", "a", ["o:a1", "w:a1", "w:a1b"]),
            ("o:a2", "a", ["o:a2", "w:a3"]),
            ("o:b1", "b", ["o:b1"]),
            ("o:b2", "b", ["o:b2"]),
        ]
    )
    r2, l2, st2 = assign_ids(b2, r1, l1, "b2")
    assert check_append_only(r1, r2) == []
    ids2 = dict(
        zip(
            r2.filter(pl.col("status") == "active")["canonical_key"],
            r2.filter(pl.col("status") == "active")["token_id"],
            strict=True,
        )
    )
    assert ids2["o:a1"] == ids1["o:a1"]
    assert ids2["o:a2"] == ids1["o:a2"]
    assert ids2["o:b1"] == ids1["o:b1"]
    dep = r2.filter(pl.col("status") == "deprecated")
    assert set(dep["canonical_key"]) == {"w:a3", "w:c1"}
    assert (
        dep.filter(pl.col("canonical_key") == "w:a3")["replaced_by"].item() == ids1["o:a2"]
    )  # merged
    # new sense of lemma a takes the next index after every index ever used (no reuse)
    a4 = r2.filter(pl.col("canonical_key") == "w:a4").row(0, named=True)
    assert a4["sense_index"] == 3
    assert l2.filter(pl.col("lemma") == "c")["status"].item() == "deprecated"
    assert st2["deprecated_ids"] == 2

    # build 3: split o:a1 into two senses; the anchor keeps the id, the split-off gets a new one
    b3 = senses(
        [
            ("o:a1", "a", ["o:a1"]),
            ("w:a1", "a", ["w:a1", "w:a1b"]),
            ("o:a2", "a", ["o:a2", "w:a3"]),
            ("w:a4", "a", ["w:a4"]),
            ("o:b1", "b", ["o:b1"]),
            ("o:b2", "b", ["o:b2"]),
            ("w:c1", "c", ["w:c1"]),  # returns: lemma id reused, but the sense gets a NEW token id
        ]
    )
    r3, l3, _ = assign_ids(b3, r2, l2, "b3")
    assert check_append_only(r2, r3) == []
    assert check_append_only(r1, r3) == []
    act3 = r3.filter(pl.col("status") == "active")
    assert act3.filter(pl.col("canonical_key") == "o:a1")["token_id"].item() == ids1["o:a1"]
    c1 = act3.filter(pl.col("lemma") == "c").row(0, named=True)
    assert c1["token_id"] != ids1["w:c1"]
    assert c1["lemma_id"] == l1.filter(pl.col("lemma") == "c")["lemma_id"].item()
    assert r3["token_id"].n_unique() == r3.height


def test_check_append_only_detects_changes() -> None:
    r1, _, _ = assign_ids(senses([("x", "a", ["x"]), ("y", "a", ["y"])]), None, None, "b1")
    tampered = r1.with_columns(
        pl.when(pl.col("canonical_key") == "x")
        .then(pl.lit("z"))
        .otherwise(pl.col("canonical_key"))
        .alias("canonical_key")
    )
    assert any("canonical_key" in p for p in check_append_only(r1, tampered))
    dropped = r1.filter(pl.col("canonical_key") != "y")
    assert any("disappeared" in p for p in check_append_only(r1, dropped))
    revived = r1.with_columns(pl.lit("deprecated").alias("status"))
    assert any("revived" in p for p in check_append_only(revived, r1))


@pytest.mark.data
def test_registry_history() -> None:
    """Real registry: every snapshot must be an append-only extension of the previous one."""
    hist = sorted((REPO / "data/processed/registry_history").glob("id_registry.*.parquet"))
    current = REPO / "data/processed/id_registry.parquet"
    chain = [*hist, current] if current.exists() else hist
    if len(chain) < 2:
        pytest.skip("fewer than two registry versions available")
    for older, newer in zip(chain, chain[1:], strict=False):
        problems = check_append_only(pl.read_parquet(older), pl.read_parquet(newer))
        assert problems == [], f"{older.name} -> {newer.name}: {problems}"


def test_committed_registry_is_append_only() -> None:
    """Working-tree registry/ must extend the registry committed at HEAD (cross-machine guard)."""
    rel = "registry/id_registry.parquet"
    current = REPO / rel
    try:
        blob = subprocess.run(
            ["git", "show", f"HEAD:{rel}"], cwd=REPO, capture_output=True, check=True
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("no registry committed at HEAD yet")
    if not current.exists():
        pytest.fail(f"{rel} is committed at HEAD but missing from the working tree")
    problems = check_append_only(pl.read_parquet(io.BytesIO(blob)), pl.read_parquet(current))
    assert problems == [], problems
