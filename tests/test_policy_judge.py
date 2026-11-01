from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest

from senselm.inventory import judge
from senselm.inventory.align import is_header, sense_text
from senselm.inventory.policy import assign_clusters, pair_merged, wiktionary_units

# --------------------------------------------------------------------------- policy

SENSES = pl.DataFrame(
    {
        "source": ["oewn", "oewn", "wiktionary", "wiktionary", "wiktionary", "wiktionary"],
        "source_id": ["o1", "o2", "w0", "w1", "w2", "w3"],
        "entry_key": ["e-o", "e-o", "E", "E", "E", "E"],
        "gloss": [
            "sloping land",
            "financial institution",
            "Unconstrained.",
            "Not imprisoned.",
            "Generous.",
            "An institution for money.",
        ],
        "gloss_path": [[], [], [], ["Unconstrained."], ["Unconstrained."], []],
        "line_no": [0, 0, 5, 5, 5, 5],
        "sense_rank": [0, 1, 0, 1, 2, 3],
    }
)
PAIRS = pl.DataFrame(
    {
        "group_key": ["g"] * 8,
        "oewn_id": ["o1", "o1", "o1", "o1", "o2", "o2", "o2", "o2"],
        "wikt_id": ["w0", "w1", "w2", "w3", "w0", "w1", "w2", "w3"],
        "cosine": [0.90, 0.85, 0.40, 0.30, 0.20, 0.30, 0.71, 0.95],
        "verdict": [None] * 8,
    },
    schema_overrides={"verdict": pl.String},
)


def policy(**kw: Any) -> dict[str, Any]:
    base = {
        "name": "fine",
        "subsenses": "keep",
        "auto_merge_threshold": 0.8,
        "separate_threshold": 0.6,
        "judge_accept": ["same"],
        "matching": "one_to_one",
    }
    return {**base, **kw}


def clusters_of(cl: pl.DataFrame) -> dict[str, str]:
    return dict(zip(cl["source_id"], cl["cluster"], strict=True))


def test_one_to_one_keeps_finer_distinctions() -> None:
    cl, st = assign_clusters(SENSES, PAIRS, policy())
    c = clusters_of(cl)
    assert c["w0"] == c["o1"] == "o1"  # best edge 0.90 wins; anchor is the WordNet sense
    assert c["w1"] != c["o1"]  # o1 already matched: w1 (0.85) stays a separate sense
    assert c["w3"] == c["o2"] == "o2"
    assert c["w2"] == "w2"
    assert st["merge_edges"] == 2 and st["clusters"] == 4


def test_lineage_absorption_merges_paraphrasing_parent_and_child_only() -> None:
    # w1 and w2 are sibling subsenses of w0. o1 matches the parent w0 best.
    def with_o1(scores: list[float]) -> pl.DataFrame:
        return PAIRS.with_columns(
            pl.when(pl.col("oewn_id") == "o1")
            .then(pl.Series(scores * 2))
            .otherwise(pl.col("cosine"))
            .alias("cosine")
        )

    one_child = with_o1([0.90, 0.85, 0.40, 0.30])  # only w1 also paraphrases o1
    assert clusters_of(assign_clusters(SENSES, one_child, policy())[0])["w1"] == "w1"
    c = clusters_of(assign_clusters(SENSES, one_child, policy(absorb_lineage=True))[0])
    assert c["w0"] == c["w1"] == c["o1"]  # unique paraphrasing child absorbed

    two_children = with_o1([0.90, 0.85, 0.82, 0.30])  # both siblings eligible
    c = clusters_of(assign_clusters(SENSES, two_children, policy(absorb_lineage=True))[0])
    assert c["w0"] == c["o1"] and c["w1"] == "w1" and c["w2"] == "w2"  # siblings never merged

    child_first = with_o1([0.80, 0.95, 0.40, 0.30])  # o1 matches the subsense w1 best
    c = clusters_of(assign_clusters(SENSES, child_first, policy(absorb_lineage=True))[0])
    assert c["w1"] == c["w0"] == c["o1"]  # eligible ancestor absorbed


def test_judge_band_requires_verdict() -> None:
    p = PAIRS.with_columns(
        pl.when(pl.col("wikt_id") == "w2")
        .then(pl.lit("same"))
        .otherwise(pl.col("verdict"))
        .alias("verdict")
    )
    c = clusters_of(assign_clusters(SENSES, p, policy(matching="many_to_one"))[0])
    assert c["w2"] == "o2"  # 0.71 is inside the band and the judge said "same"
    c = clusters_of(assign_clusters(SENSES, PAIRS, policy(matching="many_to_one"))[0])
    assert c["w2"] == "w2"  # no verdict -> kept separate


def test_subsense_collapse_and_many_to_one() -> None:
    units = wiktionary_units(SENSES.filter(pl.col("source") == "wiktionary"), "collapse")
    assert units["w1"] == units["w2"] == "w0"  # subsenses fold into the emitted parent
    cl, _ = assign_clusters(SENSES, PAIRS, policy(subsenses="collapse", matching="many_to_one"))
    c = clusters_of(cl)
    assert c["w0"] == c["w1"] == c["w2"] == c["o1"]
    merged = pair_merged(PAIRS, cl)
    assert merged.to_list() == [True, True, True, False, False, False, False, True]


def test_components_is_coarsest() -> None:
    p = PAIRS.with_columns(
        pl.when(pl.col("wikt_id") == "w0").then(0.9).otherwise(pl.col("cosine")).alias("cosine")
    )
    c = clusters_of(assign_clusters(SENSES, p, policy(matching="components"))[0])
    assert c["o1"] == c["o2"]  # w0 links both WordNet senses transitively


def test_header_detection_and_sense_text() -> None:
    assert is_header("Terms relating to animals.")
    assert is_header("A number of places in the United States:")
    assert not is_header("A mammal of the family Felidae.")
    row = {
        "lemma": "free",
        "gloss": "Complimentary.",
        "gloss_path": ["Terms relating to x.", "Obtainable without payment."],
        "examples": ["a long quote", "free drinks"],
        "example_types": ["quotation", "example"],
    }
    assert (
        sense_text(row, {"include_parent": True, "include_example": False})
        == "Obtainable without payment — Complimentary."
    )
    assert sense_text(row, {"include_example": True}).endswith("Example: free drinks")


# --------------------------------------------------------------------------- judge

JCFG = {
    "model": "claude-opus-5-5",
    "max_tokens": 2048,
    "expected_output_tokens": 400,
    "use_batches": True,
    "batch_discount": 0.5,
    "max_spend_gbp": 50,
    "usd_per_gbp": 1.2,
    "chars_per_token": 3.0,
    "prices_usd_per_mtok": {"claude-opus-5-5": {"input": 4.0, "output": 20.0}},
}


def test_estimate_heuristic_and_batch_pricing() -> None:
    msgs = ["x" * 300] * 1000
    est = judge.estimate(msgs, JCFG)
    in_tok = int(1000 * (len(judge.SYSTEM_PROMPT) + 300) / 3.0)
    assert est.input_tokens == in_tok
    assert est.worst_case_usd == pytest.approx(in_tok * 2e-6 + 1000 * 2048 * 10e-6, abs=0.01)
    assert est.expected_usd < est.worst_case_usd
    assert est.worst_case_gbp == pytest.approx(est.worst_case_usd / 1.2, abs=0.01)
    assert "heuristic" in est.token_method


def test_estimate_uses_count_tokens_when_available() -> None:
    client = SimpleNamespace(
        messages=SimpleNamespace(count_tokens=lambda **kw: SimpleNamespace(input_tokens=250))
    )
    est = judge.estimate(["a"] * 10, JCFG, client)
    assert est.input_tokens == 2500 and "count_tokens" in est.token_method


def test_user_message_and_custom_id() -> None:
    msg = judge.user_message(
        {
            "lemma": "bank",
            "upos": "NOUN",
            "o_gloss": "sloping land",
            "w_gloss": "Edge of a river.",
            "w_parent": None,
            "o_examples": ["the canoe on the bank"],
            "w_examples": [],
        }
    )
    assert (
        "Word: bank (NOUN)" in msg
        and "A (WordNet): sloping land" in msg
        and 'Example: "the canoe' in msg
    )
    cid = judge.custom_id("oewn-bank__1.17.01..", "en-bank-en-noun-x#y")
    assert len(cid) == 32 and cid.isalnum()
