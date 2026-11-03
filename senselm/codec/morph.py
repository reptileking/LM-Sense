"""Inflection for the codec: feature bundles, paradigms, contractions and genitives.

A word is written ``[case] [variant] SENSE [morph…]``. The morph tokens after a sense token form
a *bundle*; the decoder turns (lemma, UPOS, bundle) into a surface form and the encoder searches
the same function for the bundle that reproduces the word it sees, so the two can never drift
apart.

Bundles are minimal, not full UniMorph analyses (D-048): only the features needed to pick the
spelling are emitted, and the present tense is implied where person / number are marked.

    noun        ()  ("PL",)
    verb        ()  ("3",)  ("PST",)  ("PST","V.PTCP")  ("V.PTCP",)     -ing = ("V.PTCP",)
    be          + ("1",) am, ("PL",) are, ("PST","PL") were

Singular is unmarked (SG is never emitted): ("3",) is the third person singular present.
    adj / adv   ()  ("CMPR",)  ("SPRL",)
    pronoun     ()  ("ACC",)  ("GEN",)  ("PSS",)

On top of a bundle a word may carry ``NEG`` (contracted *n't*) and ``GEN`` (genitive *'s*), and a
word preceded by ``<space:clitic>`` is realised in its contracted form (*'s 're 'm 've 'll 'd*).

Surface forms come from, in order: the closed-class tables below, the exception table built from
Wiktionary / Wikidata-lexeme forms (``senselm.codec.build``: only forms the rule-based
inflector would get wrong are stored), and ``lemminflect`` (pinned in ``uv.lock``) applied to the
head word of the lemma (first word of a verb, last word of a noun).
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache

import polars as pl
from lemminflect import getInflection

Bundle = tuple[str, ...]

NOUN_BUNDLES: list[Bundle] = [(), ("PL",)]
VERB_BUNDLES: list[Bundle] = [(), ("3",), ("PST",), ("PST", "V.PTCP"), ("V.PTCP",)]
BE_BUNDLES: list[Bundle] = VERB_BUNDLES + [("1",), ("PL",), ("PST", "PL")]
DEGREE_BUNDLES: list[Bundle] = [(), ("CMPR",), ("SPRL",)]
PRON_BUNDLES: list[Bundle] = [(), ("ACC",), ("GEN",), ("PSS",)]

BUNDLES_BY_POS: dict[str, list[Bundle]] = {
    "NOUN": NOUN_BUNDLES,
    "PROPN": NOUN_BUNDLES,
    "VERB": VERB_BUNDLES,
    "AUX": VERB_BUNDLES,
    "ADJ": DEGREE_BUNDLES,
    "ADV": DEGREE_BUNDLES,
    "PRON": PRON_BUNDLES,
    "DET": PRON_BUNDLES,
}

PENN: dict[tuple[str, Bundle], str] = {
    ("NOUN", ("PL",)): "NNS",
    ("PROPN", ("PL",)): "NNS",
    ("VERB", ("3",)): "VBZ",
    ("VERB", ("PST",)): "VBD",
    ("VERB", ("PST", "V.PTCP")): "VBN",
    ("VERB", ("V.PTCP",)): "VBG",
    ("ADJ", ("CMPR",)): "JJR",
    ("ADJ", ("SPRL",)): "JJS",
    ("ADV", ("CMPR",)): "RBR",
    ("ADV", ("SPRL",)): "RBS",
}

# Closed-class paradigms (lemma, UPOS) -> bundle -> form. Small, stable and hand-checked; they
# override everything else.
_VERB_IRREG: dict[str, dict[Bundle, str]] = {
    "be": {
        (): "be",
        ("1",): "am",
        ("3",): "is",
        ("PL",): "are",
        ("PST",): "was",
        ("PST", "PL"): "were",
        ("PST", "V.PTCP"): "been",
        ("V.PTCP",): "being",
    },
    "have": {
        (): "have",
        ("3",): "has",
        ("PST",): "had",
        ("PST", "V.PTCP"): "had",
        ("V.PTCP",): "having",
    },
    "do": {
        (): "do",
        ("3",): "does",
        ("PST",): "did",
        ("PST", "V.PTCP"): "done",
        ("V.PTCP",): "doing",
    },
}
_PRONOUNS: dict[str, dict[Bundle, str]] = {
    "I": {(): "I", ("ACC",): "me", ("GEN",): "my", ("PSS",): "mine"},
    "you": {(): "you", ("GEN",): "your", ("PSS",): "yours"},
    "he": {(): "he", ("ACC",): "him", ("GEN",): "his", ("PSS",): "his"},
    "she": {(): "she", ("ACC",): "her", ("GEN",): "her", ("PSS",): "hers"},
    "it": {(): "it", ("GEN",): "its"},
    "we": {(): "we", ("ACC",): "us", ("GEN",): "our", ("PSS",): "ours"},
    "they": {(): "they", ("ACC",): "them", ("GEN",): "their", ("PSS",): "theirs"},
    "who": {(): "who", ("ACC",): "whom", ("GEN",): "whose"},
}
CLOSED: dict[tuple[str, str], dict[Bundle, str]] = {}
for _lemma, _forms in _VERB_IRREG.items():
    CLOSED[(_lemma, "VERB")] = _forms
for _lemma, _forms in _PRONOUNS.items():
    CLOSED[(_lemma, "PRON")] = _forms
for _pos in ("DET", "PRON"):
    CLOSED[("this", _pos)] = {(): "this", ("PL",): "these"}
    CLOSED[("that", _pos)] = {(): "that", ("PL",): "those"}
# Lemmas spaCy leaves as the surface form although the inventory lists them under another lemma.
LEMMA_OF: dict[str, str] = {"these": "this", "those": "that"}

# Contracted forms after <space:clitic>, without the apostrophe.
CLITIC: dict[tuple[str, Bundle], str] = {
    ("be", ("3",)): "s",
    ("be", ("1",)): "m",
    ("be", ("PL",)): "re",
    ("have", ()): "ve",
    ("have", ("3",)): "s",
    ("have", ("PST",)): "d",
    ("will", ()): "ll",
    ("would", ()): "d",
    ("shall", ()): "ll",
    ("we", ("ACC",)): "s",  # let's
    ("us", ()): "s",
}
# Stems whose n't contraction is irregular (can't, won't, shan't).
NEG_STEM: dict[str, str] = {"can": "ca", "will": "wo", "shall": "sha"}

# Wiktionary form tags -> bundle; forms carrying any other tag are not used for the paradigm.
_TAG_BUNDLES: list[tuple[frozenset[str], Bundle, tuple[str, ...]]] = [
    (frozenset({"plural"}), ("PL",), ("NOUN", "PROPN")),
    (frozenset({"past"}), ("PST",), ("VERB",)),
    (frozenset({"participle", "past"}), ("PST", "V.PTCP"), ("VERB",)),
    (frozenset({"participle", "present"}), ("V.PTCP",), ("VERB",)),
    (frozenset({"gerund", "participle", "present"}), ("V.PTCP",), ("VERB",)),
    (frozenset({"present", "singular", "third-person"}), ("3",), ("VERB",)),
    (frozenset({"indicative", "present", "singular", "third-person"}), ("3",), ("VERB",)),
    (frozenset({"comparative"}), ("CMPR",), ("ADJ", "ADV")),
    (frozenset({"superlative"}), ("SPRL",), ("ADJ", "ADV")),
]
VARIANT_TAGS = {"UK": "uk", "US": "us"}


def bundle_key(b: Bundle) -> str:
    return "|".join(b)


def parse_bundle(s: str) -> Bundle:
    return tuple(s.split("|")) if s else ()


def tags_to_bundle(tags: Iterable[str], upos: str) -> tuple[Bundle, str] | None:
    """Map a Wiktionary form's tags to (bundle, variant kind '' / 'uk' / 'us'), or None."""
    tags = set(tags)
    variant = ""
    for t, kind in VARIANT_TAGS.items():
        if t in tags:
            tags.discard(t)
            variant = kind
    for need, bundle, poses in _TAG_BUNDLES:
        if tags == need and upos in poses:
            return bundle, variant
    return None


def _case_like(template: str, word: str) -> str:
    if template[:1].isupper() and template[1:].islower():
        return word[:1].upper() + word[1:]
    if len(template) > 1 and template.isupper():
        return word.upper()
    return word


def rule_inflect(lemma: str, upos: str, bundle: Bundle) -> str | None:
    """lemminflect on the head word (first word of a verb, last word of anything else)."""
    pos = "VERB" if upos == "AUX" else upos
    tag = PENN.get((pos, bundle))
    if tag is None:
        return None
    parts = lemma.split(" ")
    head_i = 0 if pos == "VERB" else len(parts) - 1
    head = parts[head_i]
    if not head or not any(c.isalpha() for c in head):
        return None
    low = head.lower() if (head[:1].isupper() and head[1:].islower()) or head.isupper() else head
    forms = getInflection(low, tag, inflect_oov=True)
    if not forms:
        return None
    parts[head_i] = _case_like(head, forms[0]) if low != head else forms[0]
    return " ".join(parts)


class Paradigms:
    """Surface forms for (lemma, UPOS, bundle[, variant]); see the module docstring."""

    def __init__(self, exceptions: pl.DataFrame | None = None) -> None:
        self.table: dict[str, dict[tuple[str, str], str]] = {}
        if exceptions is not None and exceptions.height:
            for lemma, upos, b, var, form in exceptions.select(
                "lemma", "upos", "bundle", "variant", "form"
            ).iter_rows():
                self.table.setdefault(f"{lemma}\t{upos}", {})[(b, var)] = form
        self.realize = lru_cache(maxsize=1 << 20)(self._realize)

    def _realize(self, lemma: str, upos: str, bundle: Bundle, variant: str = "") -> str | None:
        pos = "VERB" if upos == "AUX" else upos
        closed = CLOSED.get((lemma, pos))
        if closed is not None and not variant:
            return closed.get(bundle)
        if not bundle:
            return lemma
        row = self.table.get(f"{lemma}\t{pos}")
        if row is not None:
            form = row.get((bundle_key(bundle), variant))
            if form is not None:
                return form
            if variant:
                return None
        elif variant:
            return None
        return rule_inflect(lemma, pos, bundle)

    def bundles(self, lemma: str, upos: str) -> list[Bundle]:
        pos = "VERB" if upos == "AUX" else upos
        if pos == "VERB" and lemma == "be":
            return BE_BUNDLES
        if (lemma, pos) in CLOSED:
            return list(CLOSED[(lemma, pos)])
        return BUNDLES_BY_POS.get(pos, [()])


def realize_clitic(lemma: str, bundle: Bundle, apos: str) -> str | None:
    form = CLITIC.get((lemma, bundle))
    return None if form is None else apos + form


def realize_neg(base: str, lemma: str, bundle: Bundle, apos: str) -> str:
    """Contracted negation on an (uncased) surface: do -> don't, can -> can't, will -> won't."""
    stem = NEG_STEM.get(lemma) if not bundle else None
    return (stem if stem is not None else base) + "n" + apos + "t"


def realize_gen(surface: str, bundle: Bundle, apos: str) -> str:
    if "PL" in bundle and surface.endswith("s"):
        return surface + apos
    return surface + apos + "s"
