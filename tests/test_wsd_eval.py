"""WSD evaluation plumbing: sense-key mapping, framework parsing and scoring on a toy corpus."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from senselm.eval import wsd_eval
from senselm.inventory.ids import Layout


def test_oewn_id() -> None:
    assert wsd_eval.oewn_id("bank%1:17:01::") == "oewn-bank__1.17.01.."
    assert wsd_eval.oewn_id("10th%5:00:00:ordinal:00") == "oewn-10th__5.00.00.ordinal.00"
    assert wsd_eval.oewn_id("ex-boyfriend%1:18:00::") == "oewn-ex--boyfriend__1.18.00.."
    assert wsd_eval.oewn_id("o'clock%4:02:00::") == "oewn-o-ap-clock__4.02.00.."
    assert wsd_eval.oewn_id("new_york%1:15:00::") == "oewn-new_york__1.15.00.."


def test_score_toy(tmp_path: Path) -> None:
    lay = Layout()
    inv = pl.DataFrame(
        {
            "token_id": [lay.pack(1, 0), lay.pack(1, 1), lay.pack(1, 2)],
            "lemma": ["bank", "bank", "bank"],
            "upos": ["NOUN", "NOUN", "NOUN"],
            "sense_index": [0, 1, 2],
            "oewn_rank": [None, 1.0, 0.0],  # sense 0 is Wiktionary-only; WordNet's first is 2
            "members": [
                [{"source": "wiktionary", "source_id": "w1"}],
                [{"source": "oewn", "source_id": "oewn-bank__1.14.00.."}],
                [{"source": "oewn", "source_id": "oewn-bank__1.17.01.."}],
            ],
        },
        schema_overrides={"token_id": pl.UInt32, "sense_index": pl.UInt16},
    )
    inv.write_parquet(tmp_path / "inv.parquet")
    xml = tmp_path / "toy.data.xml"
    xml.write_text(
        '<corpus><text id="d0"><sentence id="d0.s0"><wf lemma="the" pos="DET">The</wf>'
        '<instance id="d0.s0.t0" lemma="bank" pos="NOUN">bank</instance></sentence>'
        '<sentence id="d0.s1"><instance id="d0.s1.t0" lemma="bank" pos="NOUN">bank</instance>'
        "</sentence></text></corpus>",
        encoding="utf-8",
    )
    key = tmp_path / "toy.gold.key.txt"
    key.write_text("d0.s0.t0 bank%1:17:01::\nd0.s1.t0 bank%1:99:00::\n", encoding="utf-8")
    sents, gold = wsd_eval.read_dataset(xml, key)
    assert len(sents) == 2 and gold["d0.s0.t0"] == ["bank%1:17:01::"]
    res = wsd_eval.score(sents, gold, wsd_eval.SenseMaps(tmp_path / "inv.parquet"))
    assert res["instances"] == 2 and res["gold_mapped"] == 1  # the second key is not in OEWN
    assert res["acc_mfs"] == 0.0  # inventory MFS is the Wiktionary-only sense
    assert res["acc_wn_mfs"] == 0.5 and res["acc_wn_mfs_on_mapped"] == 1.0
