"""The judge prompt: system prompt, output schema and per-pair message.

Kept free of heavy imports (no NumPy, Torch or alignment code) so that the local-LLM judge
(``judge_local``) runs on the judging machine with only polars, pyarrow and pyyaml installed.
"""

from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = """You are a lexicographer aligning two English dictionaries.
You will see one word with a part of speech and two definitions of it: definition A from \
WordNet and definition B from Wiktionary (B may show its parent sense for context).
Decide how the two definitions relate:
- "same": they describe the same meaning (paraphrases, or the same sense defined with \
different detail or wording; a speaker would never need to tell them apart).
- "related": one subsumes the other (broader/narrower, a genus and a member of it, a \
domain-restricted version of the same meaning) or a regular metonymy (an animal and its meat, \
an instrument and the method it implements). These would be one sense in a coarse dictionary.
- "different": distinct meanings or referents, including figurative extensions that \
dictionaries list as separate senses.
Answer with the JSON object only."""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["same", "related", "different"]},
        "confidence": {"type": "number"},
    },
    "required": ["label", "confidence"],
    "additionalProperties": False,
}


def user_message(row: dict[str, Any]) -> str:
    def ex(xs: list[str] | None) -> str:
        return f'\n  Example: "{xs[0][:200]}"' if xs else ""

    parent = f"\n  (parent sense: {row['w_parent']})" if row.get("w_parent") else ""
    return (
        f"Word: {row['lemma']} ({row['upos']})\n"
        f"A (WordNet): {row['o_gloss']}{ex(row.get('o_examples'))}\n"
        f"B (Wiktionary): {row['w_gloss']}{parent}{ex(row.get('w_examples'))}"
    )
