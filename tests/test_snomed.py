"""SNOMED CT RF2 ingest on a synthetic fixture (real files need a TRUD licence)."""

from __future__ import annotations

import zipfile
from pathlib import Path

import polars as pl

from senselm.ingest import snomed

H_CONCEPT = "id\teffectiveTime\tactive\tmoduleId\tdefinitionStatusId\n"
H_DESC = "id\teffectiveTime\tactive\tmoduleId\tconceptId\tlanguageCode\ttypeId\tterm\tcaseSignificanceId\n"
H_LANG = "id\teffectiveTime\tactive\tmoduleId\trefsetId\treferencedComponentId\tacceptabilityId\n"
H_REL = "id\teffectiveTime\tactive\tmoduleId\tsourceId\tdestinationId\trelationshipGroup\ttypeId\tcharacteristicTypeId\tmodifierId\n"
CI = "900000000000448009"  # case insensitive
CS = snomed.CASE_SENSITIVE


def make_release(tmp: Path) -> Path:
    files = {
        "sct2_Concept_Snapshot_INT_20260101.txt": H_CONCEPT
        + "22298006\t20020131\t1\t0\t0\n414545008\t20060731\t1\t0\t0\n99\t20020131\t0\t0\t0\n",
        "sct2_Description_Snapshot-en_INT_20260101.txt": H_DESC
        + f"1\t2002\t1\t0\t22298006\ten\t{snomed.FSN}\tMyocardial infarction (disorder)\t{CI}\n"
        + f"2\t2002\t1\t0\t22298006\ten\t{snomed.SYNONYM}\tMyocardial infarction\t{CI}\n"
        + f"3\t2002\t1\t0\t22298006\ten\t{snomed.SYNONYM}\tHeart attack\t{CI}\n"
        + f"4\t2006\t1\t0\t414545008\ten\t{snomed.FSN}\tIschaemic heart disease (disorder)\t{CI}\n"
        + f"5\t2006\t1\t0\t414545008\ten\t{snomed.SYNONYM}\tIschaemic heart disease\t{CI}\n"
        + f"6\t2006\t1\t0\t414545008\ten\t{snomed.SYNONYM}\tIHD - Ischaemic heart disease\t{CS}\n",
        "der2_cRefset_LanguageSnapshot-en_GB_20260101.txt": H_LANG
        + f"a\t2002\t1\t0\t900000000000508004\t2\t{snomed.PREFERRED}\n"
        + "b\t2002\t1\t0\t900000000000508004\t3\t900000000000549004\n"
        + f"c\t2006\t1\t0\t900000000000508004\t5\t{snomed.PREFERRED}\n",
        "sct2_Relationship_Snapshot_INT_20260101.txt": H_REL
        + f"r1\t2002\t1\t0\t22298006\t414545008\t0\t{snomed.IS_A}\t0\t0\n",
    }
    zp = tmp / "uk_edition.zip"
    with zipfile.ZipFile(zp, "w") as z:
        for name, text in files.items():
            z.writestr(f"SnomedCT_UK/Snapshot/Terminology/{name}", text)
    return zp


def test_snomed_ingest(tmp_path: Path) -> None:
    zp = make_release(tmp_path)
    out = tmp_path / "snomed.parquet"
    st = snomed.run(
        {
            "stage": "t",
            "inputs": [str(zp)],
            "version": "test",
            "output": str(out),
            "language_refsets": ["999001261000000100", "900000000000508004"],
            "tag_domains": {},
        }
    )
    assert st["kept"] == 2
    d = pl.read_parquet(out).sort("source_id")
    mi = d.filter(pl.col("source_id") == "sct:22298006").row(0, named=True)
    assert mi["lemma"] == "myocardial infarction"  # initial capital dropped (case insensitive)
    assert mi["gloss"] == "disorder; a kind of Ischaemic heart disease"
    assert "licence:restricted" in mi["tags"] and "snomed:disorder" in mi["tags"]
    assert '"Heart attack"' in mi["relations"]  # synonyms kept for alignment only


def test_missing_release_is_a_clear_error(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(FileNotFoundError, match="TRUD"):
        snomed.run(
            {
                "stage": "t",
                "inputs": [str(tmp_path / "nope")],
                "version": "t",
                "output": str(tmp_path / "o.parquet"),
                "language_refsets": [],
                "tag_domains": {},
            }
        )
