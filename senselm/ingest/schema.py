"""Unified raw sense schema (``data/interim/senses_raw.parquet``).

Required columns (project spec): ``source, source_id, lemma, pos, gloss, examples[list],
tags[list], domain, forms[list], relations[json], source_version``.

Extra columns keep structure that later stages need (and that would otherwise have to be
re-derived from the multi-GB dumps): sense order, etymology number, subsense gloss path,
synset / supersense for WordNet, the raw record kind (sense / form-of / alt-of / no-gloss),
and form-of / alt-of targets.

``pos`` here is the *source* label; it is mapped to UPOS by ``senselm.inventory.normalise``.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

FORM_TYPE = pa.struct([("form", pa.string()), ("tags", pa.list_(pa.string()))])

SENSE_SCHEMA = pa.schema(
    [
        # --- required unified columns
        ("source", pa.string()),
        ("source_id", pa.string()),
        ("lemma", pa.string()),
        ("pos", pa.string()),
        ("gloss", pa.string()),
        ("examples", pa.list_(pa.string())),
        ("tags", pa.list_(pa.string())),
        ("domain", pa.string()),
        ("forms", pa.list_(FORM_TYPE)),
        ("relations", pa.string()),  # JSON object
        ("source_version", pa.string()),
        # --- extras
        ("kind", pa.string()),  # sense | form_of | alt_of | no_gloss
        ("gloss_path", pa.list_(pa.string())),  # ancestor glosses, root first (Wiktionary)
        ("example_types", pa.list_(pa.string())),  # parallel to examples: example | quotation
        ("topics", pa.list_(pa.string())),
        ("targets", pa.list_(pa.string())),  # form_of / alt_of target lemmas
        ("etymology_number", pa.int16()),
        ("entry_key", pa.string()),  # groups senses of one dictionary entry
        ("line_no", pa.int64()),  # ordinal of the entry in the dump (stable for a given dump)
        ("sense_rank", pa.int32()),  # 0-based order of the sense inside its entry
        ("synset_id", pa.string()),  # WordNet only
        ("lexfile", pa.string()),  # WordNet supersense, e.g. noun.animal
        ("hypernyms", pa.list_(pa.string())),  # WordNet synset ids (hypernym + instance_hypernym)
    ]
)

REQUIRED_COLUMNS = (
    "source",
    "source_id",
    "lemma",
    "pos",
    "gloss",
    "examples",
    "tags",
    "domain",
    "forms",
    "relations",
    "source_version",
)

_DEFAULTS: dict[str, Any] = {
    "examples": [],
    "tags": [],
    "forms": [],
    "gloss_path": [],
    "example_types": [],
    "topics": [],
    "targets": [],
    "hypernyms": [],
    "relations": "{}",
}


def make_row(**values: Any) -> dict[str, Any]:
    """Build a row with every schema column present (missing → default / null)."""
    unknown = set(values) - set(SENSE_SCHEMA.names)
    if unknown:
        raise KeyError(f"unknown columns: {sorted(unknown)}")
    return {name: values.get(name, _DEFAULTS.get(name)) for name in SENSE_SCHEMA.names}
