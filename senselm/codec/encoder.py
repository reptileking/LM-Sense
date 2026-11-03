"""Baseline encoder: English text -> token ids (spec 3.2). No machine learning beyond spaCy.

For each spaCy token (or the longest multi-word lemma starting at it) the encoder tries candidate
(lemma, UPOS) pairs -- spaCy's lemma and POS first, then the surface form and other parts of
speech -- takes the **most frequent sense** (lowest sense index) of the first pair that works and
searches that lemma's paradigm for the bundle reproducing the word, spaCy's morphology choosing
between bundles that spell the same. A word nothing reproduces is written as a byte span.
Contractions become NEG / ``<space:clitic>``, possessive *'s* becomes GEN.

Every word is checked against the decoder's own realisation functions and every gap against the
decoder's default spacing, so ``decode(encode(text)) == text`` by construction; byte fallback is
the price of anything the decoder cannot spell.
"""

from __future__ import annotations

import re
from collections import Counter, deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from lemminflect import getLemma

from senselm.codec import morph
from senselm.codec.morph import Bundle
from senselm.codec.ortho import Spacer, case_for, whitespace_tokens
from senselm.codec.tables import Lexicon, Structural, Variants

POS_TRY: dict[str, list[str]] = {
    "NOUN": ["NOUN", "PROPN"],
    "PROPN": ["PROPN", "NOUN"],
    "VERB": ["VERB"],
    "AUX": ["VERB"],
    "ADJ": ["ADJ"],
    "ADV": ["ADV", "ADJ"],
    "PRON": ["PRON", "DET"],
    "DET": ["DET", "PRON"],
    "ADP": ["ADP", "ADV", "SCONJ"],
    "SCONJ": ["SCONJ", "ADP", "ADV"],
    "CCONJ": ["CCONJ", "SCONJ"],
    "PART": ["PART", "ADV", "ADP"],
    "NUM": ["NUM", "NOUN", "ADJ"],
    "INTJ": ["INTJ"],
    "SYM": ["SYM", "NOUN", "X"],
    "X": ["X", "NOUN"],
}
RULE_LEMMA_POS = {
    "NOUN": "NOUN",
    "PROPN": "NOUN",
    "VERB": "VERB",
    "AUX": "VERB",
    "ADJ": "ADJ",
    "ADV": "ADV",
}
NEG_TEXT = {"n't", "n’t"}
CLITIC_LEMMA = {"'s": "be", "'re": "be", "'m": "be", "'ve": "have", "'ll": "will", "'d": "would"}
CLITIC_ALT = {"'s": ["be", "have"], "'d": ["would", "have"], "'ll": ["will", "shall"]}
NUMBER = re.compile(r"\d+(?:[.,:/]\d+)*")  # 3.14, 1,000, 10:30, 1/5
NUMBER_SUFFIX = re.compile(r"(\d+(?:[.,:/]\d+)*)([^\W\d_]+)")
CURLY = "’"


def _norm_apos(s: str) -> str:
    return s.replace(CURLY, "'")


def _attaches(t: Any) -> bool:
    """A token the encoder attaches to the previous word (n't, 's, 're, ...)."""
    low = _norm_apos(t.text.lower())
    return low == "n't" or t.tag_ == "POS" or low in CLITIC_LEMMA


@dataclass
class Item:
    kind: str  # word | clitic | number | punct | bytes
    start: int
    end: int
    text: str
    toks: list[Any] = field(default_factory=list)
    neg: str = ""
    gen: str = ""
    head: int = 0  # index into toks of the token whose morphology decides the bundle
    forms: list[str] = field(default_factory=list)  # lemma forms to try first (multi-word)


@dataclass
class Trace:
    """What the encoder did, for the Phase 3 metrics."""

    items: Counter[str] = field(default_factory=Counter)
    tokens: Counter[str] = field(default_factory=Counter)
    words: int = 0  # spaCy word tokens (not punctuation, numbers or whitespace)
    words_sense: int = 0
    words_bytes: int = 0
    oov: Counter[str] = field(default_factory=Counter)

    def merge(self, other: Trace) -> None:
        self.items.update(other.items)
        self.tokens.update(other.tokens)
        self.words += other.words
        self.words_sense += other.words_sense
        self.words_bytes += other.words_bytes
        self.oov.update(other.oov)


class Encoder:
    def __init__(
        self,
        structural: Structural,
        lexicon: Lexicon,
        paradigms: morph.Paradigms,
        variants: Variants | None,
        nlp: Any,
        *,
        auto_cap: bool = False,
        max_mwe_tokens: int = 8,
        keep_oov: int = 0,
    ) -> None:
        self.s, self.lex, self.par = structural, lexicon, paradigms
        self.var = variants or Variants()
        self.nlp = nlp
        self.auto_cap = auto_cap
        self.max_mwe = max_mwe_tokens
        self.keep_oov = keep_oov
        s = structural
        self.BS, self.BE = s["byte:bytes_start"], s["byte:bytes_end"]
        self.NEG, self.GEN = s["morph:NEG"], s["morph:GEN"]
        self.CLITIC = s["space:clitic"]
        self.APOS = {"'": s["orth:apos_straight"], CURLY: s["orth:apos_curly"]}

    @classmethod
    def from_tables(cls, tables: dict[str, Any], nlp: Any, **kw: Any) -> Encoder:
        return cls(
            tables["structural"],
            tables["lexicon"],
            tables["paradigms"],
            tables["variants"],
            nlp,
            **kw,
        )

    # ------------------------------------------------------------------ segmentation
    def items(self, doc: Any) -> list[Item]:
        return self._segment([t for t in doc if not t.is_space], doc.text, allow_mwe=True)

    def _segment(self, toks: list[Any], text: str, *, allow_mwe: bool) -> list[Item]:
        out: list[Item] = []
        i = 0
        while i < len(toks):
            t = toks[i]
            start, end = t.idx, t.idx + len(t.text)
            glued = bool(out) and out[-1].end == start
            low = _norm_apos(t.text.lower())
            prev = out[-1] if out else None
            if glued and prev is not None and prev.kind in ("word", "clitic"):
                if low == "n't" and prev.kind == "word" and not prev.neg and not prev.gen:
                    prev.neg, prev.end = t.text, end
                    i += 1
                    continue
                if t.tag_ == "POS" and low in ("'s", "'") and prev.kind == "word" and not prev.gen:
                    prev.gen, prev.end = t.text, end
                    i += 1
                    continue
                if low in CLITIC_LEMMA and t.pos_ in ("AUX", "VERB", "PRON", "PART"):
                    out.append(Item("clitic", start, end, t.text, [t]))
                    i += 1
                    continue
            if NUMBER.fullmatch(t.text):
                out.append(Item("number", start, end, t.text, [t]))
                i += 1
                continue
            if len(t.text) > 1 and t.text.endswith("-") and t.text[:-1].replace("-", "").isalpha():
                out.append(Item("word", start, end - 1, t.text[:-1], [t]))  # pressure- and
                out.append(Item("punct", end - 1, end, "-", [t]))
                i += 1
                continue
            if not any(c.isalnum() for c in t.text):
                out.append(Item("punct", start, end, t.text, [t]))
                i += 1
                continue
            if allow_mwe:
                n, form, head = self._mwe_match(toks, i, text)
                if n > 1:
                    last = toks[i + n - 1]
                    stop = last.idx + len(last.text)
                    item = Item("word", start, stop, text[start:stop], toks[i : i + n], head=head)
                    item.forms = [form] if form.lower() != item.text.lower() else []
                    out.append(item)
                    i += n
                    continue
            out.append(Item("word", start, end, t.text, [t]))
            i += 1
        return out

    def _mwe_match(self, toks: list[Any], i: int, text: str) -> tuple[int, str, int]:
        """Longest multi-word lemma at token i: (length, lemma form to look up, head index)."""
        best: tuple[int, str, int] = (1, "", 0)
        if not toks[i].text[:1].isalnum():
            return best
        first = toks[i]
        start = first.idx
        verbal = first.pos_ in ("VERB", "AUX")
        for n in range(2, min(self.max_mwe, len(toks) - i) + 1):
            last = toks[i + n - 1]
            if _attaches(last):  # weren't, it's, dog's: contraction lemmas would hide be + NEG
                break
            span = text[start : last.idx + len(last.text)]
            if "\n" in span:
                break
            if span.lower() in self.lex.mwe:
                best = (n, span, 0 if verbal else n - 1)
                continue
            if first.lemma_ and first.lemma_ != first.text:  # carried the can -> carry the can
                form = first.lemma_ + span[len(first.text) :]
                if form.lower() in self.lex.mwe:
                    best = (n, form, 0)
                    continue
            if last.lemma_ and last.lemma_ != last.text:  # attorneys general, Great Lakes
                form = span[: len(span) - len(last.text)] + last.lemma_
                if form.lower() in self.lex.mwe:
                    best = (n, form, n - 1)
        return best

    # ------------------------------------------------------------------ words
    def _preferred(self, t: Any, upos: str) -> Bundle:
        m = t.morph
        if upos in ("NOUN", "PROPN"):
            return ("PL",) if "Plur" in m.get("Number") else ()
        if upos in ("VERB", "AUX"):
            vf, tense = m.get("VerbForm"), m.get("Tense")
            if "Part" in vf:
                return ("PST", "V.PTCP") if "Past" in tense else ("V.PTCP",)
            if "Ger" in vf:
                return ("V.PTCP",)
            if "Past" in tense:
                return ("PST",)
            if "Pres" in tense and "3" in m.get("Person") and "Sing" in m.get("Number"):
                return ("3",)
            return ()
        if upos in ("ADJ", "ADV"):
            deg = m.get("Degree")
            return ("CMPR",) if "Cmp" in deg else ("SPRL",) if "Sup" in deg else ()
        if upos in ("PRON", "DET"):
            return (
                ("ACC",) if "Acc" in m.get("Case") else ("GEN",) if "Yes" in m.get("Poss") else ()
            )
        return ()

    def _candidates(self, item: Item) -> list[tuple[str, str]]:
        """(lemma, UPOS) pairs to try, best first: spaCy's lemma and POS, then the surface form,
        lemminflect's lemma (spaCy leaves title-case plurals such as "Systems" unlemmatised),
        and finally any part of speech the lemma has."""
        h = item.toks[item.head]
        poses = POS_TRY.get(h.pos_, [h.pos_])
        if len(item.toks) > 1:
            forms = item.forms + [item.text]
        else:
            low = item.text.lower()
            forms = [h.lemma_, morph.LEMMA_OF.get(low, ""), item.text, low]
            rule_pos = RULE_LEMMA_POS.get(h.pos_)
            if rule_pos:
                forms += list(getLemma(low, upos=rule_pos))
        spellings: list[str] = []
        for f in forms:
            if not f:
                continue
            for sp in self.lex.spellings(f):
                if sp not in spellings:
                    spellings.append(sp)
        out: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for sp in spellings:
            for upos in poses:
                if (sp, upos) not in seen and self.lex.mfs(sp, upos) is not None:
                    seen.add((sp, upos))
                    out.append((sp, upos))
        for sp in spellings:
            for upos in self.lex.poses(sp):
                if (sp, upos) not in seen:
                    seen.add((sp, upos))
                    out.append((sp, upos))
        return out

    def _bundles(self, lemma: str, upos: str, prefer: Bundle) -> list[Bundle]:
        allb = self.par.bundles(lemma, upos)
        rest = sorted((b for b in allb if b != prefer), key=len)
        return ([prefer] if prefer in allb else []) + rest

    def _morph_ids(self, bundle: Bundle) -> list[int]:
        return [self.s[f"morph:{f}"] for f in bundle]

    def encode_word(self, item: Item, auto: bool) -> tuple[list[int], str] | None:
        apos = CURLY if CURLY in item.neg + item.gen else "'"
        target = item.text + item.neg + item.gen
        h = item.toks[item.head]
        for lemma, upos in self._candidates(item):
            tid = self.lex.mfs(lemma, upos)
            if tid is None:
                continue
            for bundle in self._bundles(lemma, upos, self._preferred(h, upos)):
                base = self.par.realize(lemma, upos, bundle, "")
                if base is None:
                    continue
                full = morph.realize_neg(base, lemma, bundle, apos) if item.neg else base
                if item.gen:
                    full = morph.realize_gen(full, bundle, apos)
                case = case_for(target, full, auto)
                if case is None:
                    continue
                ids = ([self.s[f"case:{case}"]] if case else []) + [tid] + self._morph_ids(bundle)
                ids += ([self.NEG] if item.neg else []) + ([self.GEN] if item.gen else [])
                return ids, ("mwe" if len(item.toks) > 1 else "sense")
            hit = self._inflection_variant(item, auto, apos, target, lemma, upos, tid)
            if hit is not None:
                return hit
        return self._encode_variant(item, auto, apos, target)

    def _inflection_variant(
        self, item: Item, auto: bool, apos: str, target: str, lemma: str, upos: str, tid: int
    ) -> tuple[list[int], str] | None:
        """travelled = <variant:uk> travel PST (the decoder reads a variant token as a spelling
        variant when the lemma has one of that kind, so those lemmas are skipped here)."""
        for kind in ("uk", "us", "other"):
            if (lemma, upos, kind) in self.var.spelling:
                continue
            for bundle in self.par.bundles(lemma, upos):
                if not bundle:
                    continue
                base = self.par.realize(lemma, upos, bundle, kind)
                if base is None:
                    continue
                full = morph.realize_neg(base, lemma, bundle, apos) if item.neg else base
                if item.gen:
                    full = morph.realize_gen(full, bundle, apos)
                case = case_for(target, full, auto)
                if case is None:
                    continue
                ids = ([self.s[f"case:{case}"]] if case else []) + [self.s[f"variant:{kind}"], tid]
                ids += self._morph_ids(bundle)
                ids += ([self.NEG] if item.neg else []) + ([self.GEN] if item.gen else [])
                return ids, "variant"
        return None

    def _encode_variant(
        self, item: Item, auto: bool, apos: str, target: str
    ) -> tuple[list[int], str] | None:
        if len(item.toks) != 1 or item.neg:
            return None
        h = item.toks[0]
        for key in {h.lemma_.lower(), item.text.lower()}:
            for lemma, upos, kind in self.var.target.get(key, []):
                tid = self.lex.mfs(lemma, upos)
                if tid is None:
                    continue
                spelling = self.var.spelling[(lemma, upos, kind)]
                for bundle in self._bundles(spelling, upos, self._preferred(h, upos)):
                    base = self.par.realize(spelling, upos, bundle, "")
                    if base is None:
                        continue
                    full = morph.realize_gen(base, bundle, apos) if item.gen else base
                    case = case_for(target, full, auto)
                    if case is None:
                        continue
                    ids = ([self.s[f"case:{case}"]] if case else []) + [
                        self.s[f"variant:{kind}"],
                        tid,
                    ]
                    ids += self._morph_ids(bundle) + ([self.GEN] if item.gen else [])
                    return ids, "variant"
        return None

    def encode_clitic(self, item: Item) -> list[int] | None:
        t = item.toks[0]
        text = item.text
        apos = CURLY if CURLY in text else "'"
        key = _norm_apos(text.lower())
        lemmas: list[tuple[str, str]] = []
        lem = t.lemma_
        if lem and not lem.startswith(("'", CURLY)):
            lemmas.append((lem, "PRON" if t.pos_ == "PRON" else "VERB"))
        for alt in CLITIC_ALT.get(key, [CLITIC_LEMMA[key]]):
            if (alt, "VERB") not in lemmas:
                lemmas.append((alt, "VERB"))
        for lemma, upos in lemmas:
            tid = self.lex.mfs(lemma, upos)
            if tid is None:
                continue
            for (cl, bundle), form in morph.CLITIC.items():
                if cl != lemma:
                    continue
                case = case_for(text, apos + form, False)
                if case is None:
                    continue
                return (
                    [self.CLITIC]
                    + ([self.s[f"case:{case}"]] if case else [])
                    + [tid]
                    + self._morph_ids(bundle)
                )
        return None

    def byte_span(self, text: str) -> list[int]:
        return [self.BS] + [self.s.byte[b] for b in text.encode("utf-8")] + [self.BE]

    # ------------------------------------------------------------------ document
    def encode_doc(self, doc: Any, trace: Trace | None = None) -> list[int]:
        text = doc.text
        out: list[int] = []
        spacer = Spacer(auto_cap=self.auto_cap)
        apos_state = "'"
        pos = 0  # end of the text already covered
        tr = trace if trace is not None else Trace()

        def put(ids: list[int], kind: str) -> None:
            out.extend(ids)
            tr.tokens[kind] += len(ids)

        def gap_to(start: int, kind: str, unit_text: str) -> str:
            """Emit space tokens for text[pos:start] if it differs from the default; return it."""
            gap = text[pos:start]
            default = spacer.sep(kind, unit_text)
            if gap == default:
                return gap
            names = whitespace_tokens(gap)
            if names is None:  # unusual whitespace: literal byte span between explicit no-spaces
                put([self.s["space:nospace"]], "space")
                put(self.byte_span(gap), "bytes_ws")
                spacer.push("word", gap)
                tr.items["odd_whitespace"] += 1
                if spacer.sep(kind, unit_text) != "":
                    put([self.s["space:nospace"]], "space")
                return ""
            put([self.s[f"space:{n}"] for n in names], "space")
            return gap

        def unit(start: int, end: int, kind: str, unit_text: str, ids: list[int], cat: str) -> None:
            nonlocal pos
            gap_to(start, kind, unit_text)
            put(ids, cat)
            spacer.push(kind, unit_text)
            pos = end

        queue = deque(self.items(doc))
        while queue:
            it = queue.popleft()
            if it.kind == "clitic":
                tr.words += 1
                ids = self.encode_clitic(it)
                if ids is not None:
                    need = CURLY if CURLY in it.text else "'"
                    if need != apos_state:
                        put([self.APOS[need]], "orth")
                        apos_state = need
                    put(ids, "sense")
                    tr.items["clitic"] += 1
                    tr.words_sense += 1
                    pos = it.end
                else:
                    put([self.s["space:nospace"]], "space")
                    put(self.byte_span(it.text), "bytes")
                    spacer.push("word", it.text)
                    tr.items["clitic_bytes"] += 1
                    tr.words_bytes += 1
                    pos = it.end
                continue
            if it.kind == "number":
                for k, ch in enumerate(it.text):
                    if ch.isdigit():
                        unit(
                            it.start + k,
                            it.start + k + 1,
                            "digit",
                            ch,
                            [self.s[f"digit:{ch}"]],
                            "digit",
                        )
                    else:
                        unit(
                            it.start + k, it.start + k + 1, "punct", ch, [self.s.punct[ch]], "punct"
                        )
                tr.items["number"] += 1
                continue
            if it.kind == "punct":
                for k, ch in enumerate(it.text):
                    tid = self.s.punct.get(ch)
                    if tid is not None:
                        unit(it.start + k, it.start + k + 1, "punct", ch, [tid], "punct")
                    else:
                        unit(
                            it.start + k,
                            it.start + k + 1,
                            "word",
                            ch,
                            self.byte_span(ch),
                            "bytes_char",
                        )
                tr.items["punct"] += 1
                continue
            # word
            gap = text[pos : it.start]
            # the decoder auto-capitalises from the separator it actually uses (= the gap, unless
            # the gap needs a literal byte span, which makes the previous unit a word)
            auto = (
                self.auto_cap and whitespace_tokens(gap) is not None and spacer.sentence_start(gap)
            )
            res = self.encode_word(it, auto)
            if res is None and len(it.toks) > 1:  # multi-word lemma not spellable: single words
                parts = self._segment(it.toks, text, allow_mwe=False)
                if it.neg or it.gen:
                    last = parts[-1]
                    if last.kind != "word":
                        last.kind = "word"
                    last.neg, last.gen, last.end = it.neg, it.gen, it.end
                queue.extendleft(reversed(parts))
                tr.items["mwe_split"] += 1
                continue
            nwords = sum(1 for t in it.toks if any(c.isalnum() for c in t.text))
            tr.words += nwords
            if res is not None:
                ids, cat = res
                if it.neg or it.gen:
                    need = CURLY if CURLY in it.neg + it.gen else "'"
                    if need != apos_state:
                        put([self.APOS[need]], "orth")
                        apos_state = need
                unit(it.start, it.end, "word", it.text, ids, "sense")
                tr.items[cat] += 1
                tr.words_sense += nwords
                continue
            m = NUMBER_SUFFIX.fullmatch(it.text) if len(it.toks) == 1 else None
            if m is not None and not it.neg and not it.gen:  # 1990s, 2nd: digits + glued suffix
                num, suf = m.group(1), m.group(2)
                for k, ch in enumerate(num):
                    kind = "digit" if ch.isdigit() else "punct"
                    tid = self.s[f"digit:{ch}"] if ch.isdigit() else self.s.punct[ch]
                    unit(it.start + k, it.start + k + 1, kind, ch, [tid], kind)
                unit(it.start + len(num), it.end, "word", suf, self.byte_span(suf), "bytes")
                tr.items["number_suffix"] += 1
                tr.words_bytes += nwords
                continue
            literal = it.text + it.neg + it.gen
            unit(it.start, it.end, "word", literal, self.byte_span(literal), "bytes")
            reason = "bytes_oov" if not self._candidates(it) else "bytes_form"
            tr.items[reason] += 1
            tr.words_bytes += nwords
            if self.keep_oov:
                tr.oov[literal] += 1
        tail = text[pos:]
        if tail:
            names = whitespace_tokens(tail)
            if names is None:
                put([self.s["space:nospace"]], "space")
                put(self.byte_span(tail), "bytes_ws")
            else:
                put([self.s[f"space:{n}"] for n in names], "space")
        return out

    def encode(self, text: str, trace: Trace | None = None) -> list[int]:
        return self.encode_doc(self.nlp(text), trace)

    def encode_many(
        self, texts: Iterable[str], *, batch_size: int = 64, n_process: int = 1
    ) -> Iterator[tuple[list[int], Trace]]:
        for doc in self.nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
            tr = Trace()
            yield self.encode_doc(doc, tr), tr
