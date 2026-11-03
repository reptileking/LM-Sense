"""Orthography shared by the encoder and the decoder: default spacing, quotes, case.

The decoder puts a *default separator* before every unit (word, byte span, digit, punctuation
mark) unless ``<space:*>`` tokens stand between the two units, in which case the separator is
exactly the concatenation of those tokens' strings. The encoder runs the same :class:`Spacer`
and emits space tokens only where the text differs from the default, so the rules below decide
how many space tokens real text needs, never whether it round-trips.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SPACE_STR: dict[str, str] = {
    "nospace": "",
    "space": " ",
    "newline": "\n",
    "paragraph": "\n\n",
    "tab": "\t",
    "indent": "    ",
}

OPENERS = set("([{“‘«¿¡")
CLOSERS = set(".,;:!?)]}”’»%…°™®‰")
GLUE = set("-/")  # no space on either side by default
AFTER_GLUE = set("$£€¥#")  # no space after (currency, hash) -- the next unit sticks
STRAIGHT_QUOTES = {'"', "'"}
SENTENCE_END = set(".!?")


@dataclass
class Spacer:
    """Default separators. Feed every unit through :meth:`sep` then :meth:`push`."""

    auto_cap: bool = False
    prev: tuple[str, str] | None = None
    prev2: tuple[str, str] | None = None
    stick: bool = False  # the previous unit glues the next one to it
    quote_open: dict[str, bool] = field(default_factory=lambda: {'"': False, "'": False})

    def sep(self, kind: str, text: str) -> str:
        if self.prev is None or self.stick:
            return ""
        pk, pt = self.prev
        if kind == "digit":
            if pk == "digit":
                return ""
            if (
                pk == "punct"
                and pt in ".,:/"
                and self.prev2 is not None
                and self.prev2[0] == "digit"
            ):
                return ""
        if kind == "punct":
            if text in STRAIGHT_QUOTES:
                return "" if self.quote_open[text] else " "
            if text in CLOSERS or text in GLUE:
                return ""
        if pk == "punct" and pt in GLUE:
            return ""
        return " "

    def push(self, kind: str, text: str) -> None:
        stick = False
        if kind == "punct":
            if text in STRAIGHT_QUOTES:
                opening = not self.quote_open[text]
                self.quote_open[text] = opening
                stick = opening
            else:
                stick = text in OPENERS or text in AFTER_GLUE
        self.prev2, self.prev, self.stick = self.prev, (kind, text), stick

    def sentence_start(self, separator: str) -> bool:
        """Auto-capitalisation position (only used when ``auto_cap`` is on)."""
        if self.prev is None or "\n" in separator:
            return True
        pk, pt = self.prev
        if pk != "punct":
            return False
        if pt in SENTENCE_END:
            return True
        return (
            pt in CLOSERS | STRAIGHT_QUOTES
            and self.prev2 is not None
            and self.prev2[0] == "punct"
            and self.prev2[1] in SENTENCE_END
        )


def apply_case(surface: str, case: str | None) -> str:
    if not case or not surface:
        return surface
    if case == "cap":
        return surface[:1].upper() + surface[1:]
    if case == "allcaps":
        return surface.upper()
    if case == "lower":
        return surface.lower()
    if case == "title":
        return title_case(surface)
    return surface


def title_case(surface: str) -> str:
    """Upper-case the first letter of every space- or hyphen-separated part, nothing else."""
    out, up = [], True
    for ch in surface:
        out.append(ch.upper() if up else ch)
        up = ch in " -"
    return "".join(out)


def auto_capitalise(surface: str) -> str:
    return surface[:1].upper() + surface[1:] if surface[:1].islower() else surface


def case_for(text: str, generated: str, auto: bool) -> str | None:
    """The case token ('' = none) turning ``generated`` into ``text``, or None if impossible.

    ``auto`` says the decoder would auto-capitalise this word (sentence start, ``auto_cap`` on).
    """
    default = auto_capitalise(generated) if auto else generated
    if text == default:
        return ""
    if text == generated[:1].upper() + generated[1:] and not auto:
        return "cap"
    if text == generated.lower():
        return "lower"
    if text == generated.upper() and text != generated:
        return "allcaps"
    if text == title_case(generated) and text != generated:
        return "title"
    return None


def whitespace_tokens(gap: str) -> list[str] | None:
    """Space-token names spelling ``gap`` exactly, or None if it holds other characters."""
    if gap == "":
        return ["nospace"]
    out: list[str] = []
    i, line_start = 0, False
    while i < len(gap):
        if gap.startswith("\n\n", i):
            out.append("paragraph")
            i += 2
            line_start = True
        elif gap[i] == "\n":
            out.append("newline")
            i += 1
            line_start = True
        elif gap[i] == "\t":
            out.append("tab")
            i += 1
        elif line_start and gap.startswith("    ", i):
            out.append("indent")
            i += 4
        elif gap[i] == " ":
            out.append("space")
            i += 1
            line_start = False
        else:
            return None
    return out
