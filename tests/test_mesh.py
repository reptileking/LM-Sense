"""MeSH dictionary-form lemma rules (D-035)."""

from __future__ import annotations

import pytest

from senselm.ingest.mesh import choose_lemma

TAXA = frozenset(
    {
        "Populus",
        "Escherichia",
        "Pseudomonas",
        "Borrelia",
        "Streptococcus",
        "Ophiocordyceps",
        "Neisseria",
        "Aegle",
        "Gorilla",
        "Mus",
        "Bacillus",
    }
)


@pytest.mark.parametrize(
    ("term", "organism", "scope", "lemma"),
    [
        ("Self-Assessment", False, "", "self-assessment"),
        ("Lymph Nodes", False, "", "lymph node"),
        ("Mice", True, "The common name for the genus Mus.", "mouse"),
        ("T-Lymphocytes", False, "", "T-lymphocyte"),
        ("Vitamin B 12", False, "", "vitamin B 12"),
        ("Diabetes Mellitus", False, "", "diabetes mellitus"),
        ("Signs and Symptoms", False, "", "signs and symptoms"),
        ("Insulin, Regular, Pork", False, "", "pork regular insulin"),
        ("Rats, Inbred BN", True, "An inbred strain of rat", "inbred BN rat"),
        ("Escherichia coli Infections", False, "", "Escherichia coli infection"),
        ("Populus", True, "A plant genus of the family SALICACEAE.", "Populus"),
        ("Chrysanthemum", True, "A plant genus of the family ASTERACEAE.", "Chrysanthemum"),
        ("Xenarthra", True, "A superorder of New World mammals", "Xenarthra"),
        ("Streptococcus pyogenes", True, "", "Streptococcus pyogenes"),
        ("Pseudomonas Phages", True, "Viruses whose host is Pseudomonas.", "Pseudomonas phage"),
        ("Coliphages", True, "Viruses whose host is Escherichia coli.", "coliphage"),
        (
            "Neisseria meningitidis, Serogroup W-135",
            True,
            "Strains of",
            "Neisseria meningitidis serogroup W-135",
        ),
        (
            "Sideroblastic anemia with spinocerebellar ataxia",
            False,
            "",
            "sideroblastic anemia with spinocerebellar ataxia",
        ),
        ("heart-hand syndrome, slovenian type", False, "", "heart-hand syndrome, slovenian type"),
        (
            "Phosphotransferases (Nitrogenous Group Acceptor)",
            False,
            "",
            "phosphotransferases (nitrogenous group acceptor)",
        ),
        ("National Cancer Institute (U.S.)", False, "", "national cancer institute (U.S.)"),
    ],
)
def test_dictionary_form(term: str, organism: bool, scope: str, lemma: str) -> None:
    assert (
        choose_lemma(
            [(term, "NON", False)], proper=False, organism=organism, scope=scope, taxa=TAXA
        )[0]
        == lemma
    )


def test_preferred_non_inverted_entry_term_and_qualifier() -> None:
    terms = [("Plants, Medicinal", "NON", False), ("Medicinal Plants", "NON", False)]
    assert choose_lemma(terms, proper=False)[:2] == ("medicinal plant", ["Plants, Medicinal"])
    assert choose_lemma(
        [("Aegle, moths", "NON", False)], proper=False, organism=True, taxa=TAXA
    ) == ("Aegle", [], "moths")
    assert (
        choose_lemma([("Alzheimer Disease", "EPO", False)], proper=False)[0] == "Alzheimer disease"
    )
    assert choose_lemma([("Rio de Janeiro", "NON", False)], proper=True)[0] == "Rio de Janeiro"
    scr = {"proper": False, "taxa": TAXA, "sentence_case": True}
    assert (
        choose_lemma([("Kocher-Debre-Semelaigne syndrome", "NON", False)], **scr)[0]
        == "Kocher-Debre-Semelaigne syndrome"
    )
    assert (
        choose_lemma([("Maf protein, Bacillus subtilis", "NON", False)], **scr)[0]
        == "Bacillus subtilis maf protein"
    )
    assert (
        choose_lemma([("Liberomyces macrosporus", "NON", False)], organism=True, **scr)[0]
        == "Liberomyces macrosporus"
    )
    assert choose_lemma([("Tomato yellow leaf curl Thailand virus", "NON", False)], **scr)[0] == (
        "tomato yellow leaf curl Thailand virus"
    )
    assert choose_lemma([("Diastasis Recti And Weakness Of The Linea Alba", "NON", False)], **scr)[
        0
    ] == ("diastasis recti and weakness of the linea alba")
