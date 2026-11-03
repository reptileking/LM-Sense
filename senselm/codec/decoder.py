"""Deterministic decoder: token ids -> English text (spec 3.1).

Grammar of a sequence (docs/STRUCTURAL_TOKENS.md):

    word   = [<case:*>] [<variant:*>] SENSE [<morph:*>…]        (morph incl. NEG and GEN)
    clitic = <space:clitic> word                                 ('s 're 'm 've 'll 'd)
    bytes  = <byte:bytes_start> <byte:0x..>… <byte:bytes_end> [<morph:GEN>]
    unit   = word | clitic | bytes | <digit:*> | <punct:*>
    between units: <space:*>… (exact separator) or nothing (default, senselm.codec.ortho)
    anywhere: <orth:apos_*> (stateful apostrophe style), <special:*> and <prag:*> (no surface)

Malformed input never raises: an unknown id, a morph token with no word or an unrealisable
bundle degrades to the lemma, so a model's raw output can always be read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from senselm.codec import morph
from senselm.codec.ortho import SPACE_STR, Spacer, apply_case, auto_capitalise
from senselm.codec.tables import Lexicon, Structural, Variants

APOS = {"apos_curly": "’", "apos_straight": "'"}


@dataclass
class _Word:
    lemma: str | None = None
    upos: str = ""
    text: str | None = None  # byte span
    feats: list[str] = field(default_factory=list)
    case: str | None = None
    variant: str = ""
    clitic: bool = False


class Decoder:
    def __init__(
        self,
        structural: Structural,
        lexicon: Lexicon,
        paradigms: morph.Paradigms,
        variants: Variants | None = None,
        *,
        auto_cap: bool = False,
    ) -> None:
        self.s, self.lex, self.par = structural, lexicon, paradigms
        self.var = variants or Variants()
        self.auto_cap = auto_cap

    @classmethod
    def from_tables(cls, tables: dict[str, Any], *, auto_cap: bool = False) -> Decoder:
        return cls(
            tables["structural"],
            tables["lexicon"],
            tables["paradigms"],
            tables.get("variants"),
            auto_cap=auto_cap,
        )

    # ------------------------------------------------------------------ word realisation
    def surface(self, w: _Word, apos: str) -> str:
        neg, gen = "NEG" in w.feats, "GEN" in w.feats
        bundle = tuple(f for f in w.feats if f not in ("NEG", "GEN"))
        if w.text is not None:  # byte span
            return morph.realize_gen(w.text, (), apos) if gen else w.text
        assert w.lemma is not None
        spelling, inflection_variant = w.lemma, ""
        if w.variant:  # a listed spelling variant of the lemma, else a UK / US inflected form
            sp = self.var.spelling.get((w.lemma, w.upos, w.variant))
            if sp is not None:
                spelling = sp
            else:
                inflection_variant = w.variant
        if w.clitic:
            form = morph.realize_clitic(w.lemma, bundle, apos)
            if form is not None:
                return apply_case(form, w.case)
        base = self.par.realize(spelling, w.upos, bundle, inflection_variant)
        if base is None and inflection_variant:
            base = self.par.realize(spelling, w.upos, bundle, "")
        if base is None:
            base = spelling
        out = morph.realize_neg(base, w.lemma, bundle, apos) if neg else base
        if gen:
            out = morph.realize_gen(out, bundle, apos)
        return apply_case(out, w.case)

    # ------------------------------------------------------------------ sequence
    def decode(self, ids: list[int]) -> str:
        out: list[str] = []
        spacer = Spacer(auto_cap=self.auto_cap)
        explicit: list[str] | None = None
        apos = "'"
        case: str | None = None
        variant = ""
        clitic = False
        word: _Word | None = None
        buf: bytearray | None = None

        def emit(kind: str, text: str, *, is_clitic: bool = False, cased: bool = True) -> None:
            nonlocal explicit
            if is_clitic:
                out.append(text)
                explicit = None
                return
            sep = "".join(explicit) if explicit is not None else spacer.sep(kind, text)
            if kind == "word" and self.auto_cap and not cased and spacer.sentence_start(sep):
                text = auto_capitalise(text)
            out.append(sep + text)
            spacer.push(kind, text)
            explicit = None

        def flush() -> None:
            nonlocal word
            if word is not None:
                text = self.surface(word, apos)
                fixed = word.case is not None or word.text is not None  # byte spans are literal
                emit("word", text, is_clitic=word.clitic, cased=fixed)
                word = None

        for tid in ids:
            info = self.s.info.get(int(tid))
            if info is None:  # sense token
                flush()
                hit = self.lex.lookup(int(tid))
                if hit is None:
                    case, variant, clitic = None, "", False
                    continue
                word = _Word(lemma=hit[0], upos=hit[1], case=case, variant=variant, clitic=clitic)
                case, variant, clitic = None, "", False
                continue
            cat, name = info
            if buf is not None:
                if cat == "byte" and name != "bytes_start":
                    if name == "bytes_end":
                        word = _Word(text=buf.decode("utf-8", errors="replace"))
                        buf = None
                    else:
                        buf.append(int(name, 16))
                    continue
                word = _Word(text=buf.decode("utf-8", errors="replace"))  # unterminated span
                buf = None
            if cat == "morph":
                if word is not None:
                    word.feats.append(name)
                continue
            flush()
            if cat == "case":
                case = name
            elif cat == "variant":
                variant = name
            elif cat == "space":
                if name == "clitic":
                    clitic = True
                else:
                    explicit = (explicit or []) + [SPACE_STR.get(name, "")]
            elif cat == "punct":
                emit("punct", name)
            elif cat == "digit":
                emit("digit", name)
            elif cat == "byte":
                if name == "bytes_start":
                    buf = bytearray()
            elif cat == "orth":
                apos = APOS.get(name, apos)
            # special, prag: no surface
        if buf is not None:
            word = _Word(text=buf.decode("utf-8", errors="replace"))
        flush()
        if explicit is not None:
            out.append("".join(explicit))
        return "".join(out)
