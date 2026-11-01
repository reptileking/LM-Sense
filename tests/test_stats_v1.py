"""Phase-2 statistics on a tiny synthetic build (vocabulary, ID space, embedding estimates)."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from senselm.inventory import stats_v1
from senselm.inventory.ids import Layout


def test_stats_v1_small(tmp_path: Path) -> None:
    lay = Layout()
    inv = pl.DataFrame(
        {
            "token_id": [lay.pack(1, 0), lay.pack(1, 1), lay.pack(2, 0), lay.pack(3, 0)],
            "lemma": ["mouse", "mouse", "Mus", "aspirin"],
            "lemma_id": [1, 1, 2, 3],
            "sense_index": [0, 1, 0, 0],
            "upos": ["NOUN", "NOUN", "PROPN", "NOUN"],
            "domain": [None, "computing", "taxonomy", "pharmacology"],
            "sources": [
                ["oewn", "wiktionary", "mesh"],
                ["wiktionary"],
                ["ncbi_taxonomy"],
                ["oewn", "chebi"],
            ],
            "licences": [
                ["CC-BY-4.0"],
                ["CC-BY-SA-4.0+GFDL"],
                ["public-domain-NCBI"],
                ["CC-BY-4.0"],
            ],
            "has_oewn": [True, False, False, True],
            "has_wiktionary": [True, True, False, False],
            "rare": [False, False, False, False],
            "primary_source": ["oewn", "wiktionary", "ncbi_taxonomy", "oewn"],
            "n_members": [3, 1, 1, 2],
        },
        schema_overrides={"token_id": pl.UInt32, "lemma_id": pl.UInt32, "sense_index": pl.UInt16},
    )
    reg = inv.select("token_id", "lemma_id", "sense_index").with_columns(
        status=pl.lit("active"), created_in=pl.Series(["v0.1", "v0.1", "v1.0", "v1.0"])
    )
    lem = pl.DataFrame(
        {"lemma_id": [1, 2, 3], "lemma": ["mouse", "Mus", "aspirin"], "status": ["active"] * 3}
    )
    structural = pl.DataFrame(
        {"category": ["special", "byte"], "slot": [0, 1], "status": ["active"] * 2}
    )
    att = pl.DataFrame(
        {
            "source": ["mesh", "ncbi_taxonomy", "chebi"],
            "source_id": ["D1", "taxon:1", "C1"],
            "via": ["cosine", "new", "xref"],
        }
    )
    senses = pl.DataFrame(
        {"source": ["oewn", "oewn", "wiktionary", "wiktionary", "mesh", "ncbi_taxonomy", "chebi"]}
    )
    paths = {}
    for name, df in [
        ("inv", inv),
        ("reg", reg),
        ("lem", lem),
        ("st", structural),
        ("att", att),
        ("sen", senses),
    ]:
        paths[name] = tmp_path / f"{name}.parquet"
        df.write_parquet(paths[name])
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    reg.write_parquet(ledger / "id_registry.parquet")
    cfg = {
        "build_id": "t",
        "inventory": str(paths["inv"]),
        "registry": str(paths["reg"]),
        "lemma_registry": str(paths["lem"]),
        "structural_registry": str(paths["st"]),
        "attachments": str(paths["att"]),
        "senses_norm": str(paths["sen"]),
        "ledger_dir": str(ledger),
        "layout": {"lemma_bits": 22, "sense_bits": 10, "reserved_lemma_slots": 4096},
        "source_order": ["oewn", "wiktionary", "mesh", "chebi", "ncbi_taxonomy"],
        "licence_scenarios": {"no share-alike": ["wiktionary"]},
        "storage": {"ledger": [str(ledger)]},
        "embedding": {
            "dims": [8],
            "ranks": [2],
            "dense_top_k": [2],
            "hybrid_rank": 2,
            "bytes_per_param": 2,
        },
    }
    st = stats_v1.compute(cfg)
    v = st["vocabulary"]
    assert v["canonical_senses"] == 4 and v["lemmas"] == 3 and st["total_vocabulary"] == 6
    by_src = {r["source"]: r for r in v["by_source"]}
    assert by_src["mesh"]["attached_cosine"] == 1 and by_src["chebi"]["attached_xref"] == 1
    assert (
        by_src["ncbi_taxonomy"]["new_clusters"] == 1
        and by_src["wiktionary"]["canonical_senses_with_source"] == 2
    )
    tiers = {r["tier"]: r["senses"] for r in v["by_tier"]}
    assert tiers == {"0 WordNet": 2, "1 Wiktionary": 1, "4 concept / domain sources": 1}
    assert v["licence_scenarios"][0]["senses_kept"] == 3  # the Wiktionary-only sense goes
    ids = st["id_space"]
    assert ids["lemma_ids_issued"] == 3 and ids["max_sense_index"] == 1
    assert ids["token_ids_by_build"] == {"v0.1": 2, "v1.0": 2}
    emb = {r["scheme"]: r["params"] for r in st["embedding"]["table"]}
    v_rows, lemma_rows = 4 + 2, 3 + 2  # + structural tokens / slots
    assert emb["naive V x d"] == v_rows * 8
    assert (
        emb["E_lemma + E_sense_index + E_domain"] == (lemma_rows + 1024 + 4) * 8
    )  # 3 domains + none
    assert emb["low-rank V x r + r x d (r=2)"] == v_rows * 2 + 2 * 8
    md = stats_v1.markdown(st)
    assert "Total vocabulary: **6**" in md and "## Embedding parameters" in md
