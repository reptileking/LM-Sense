"""Normalise raw senses: Unicode, lemma casing, UPOS, and sense / form / variant routing.

``python -m senselm.inventory.normalise --config configs/normalise.yaml``

Input: ``data/interim/senses_raw.parquet``. Outputs:

* ``senses_norm.parquet``  - the senses that enter alignment (one row per source sense)
* ``forms.parquet``        - inflection table for the decoder: (lemma, upos, form, tags)
  from Wiktionary headword/inflection forms, Wiktionary form-of senses ("plural of X")
  and OEWN irregular forms
* ``variants.parquet``     - orthographic variants (alternative / obsolete spellings,
  misspellings, letter-case variants): (variant, target, upos, tags)
* ``dedupe_map.parquet``   - exact duplicate senses dropped → the kept source_id

Routing rules (all configurable, see ``configs/normalise.yaml`` and docs/DECISIONS.md):

* ``no_gloss`` senses are dropped.
* ``form_of`` senses are not senses: they become forms-table rows.
* ``alt_of`` senses are variants, *except* abbreviation-type ones (initialism, acronym,
  abbreviation, clipping, ellipsis, contraction) which are distinct lexical items and stay
  senses (``NASA`` must exist as a lemma).
* untagged senses whose whole gloss is "<qualifier> spelling/form of X" are variants too.
* Lemma case is preserved (``Polish`` ≠ ``polish``); lemmas are NFC-normalised, whitespace
  is collapsed and typographic apostrophes / hyphens are mapped to ASCII.
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from typing import Any

import orjson
import pyarrow as pa
import pyarrow.parquet as pq

from senselm.common import BatchWriter, Provenance, load_config, resolve, stage_args
from senselm.ingest.schema import FORM_TYPE, SENSE_SCHEMA

LOG = logging.getLogger("senselm.inventory.normalise")

NORM_SCHEMA = (
    SENSE_SCHEMA.append(pa.field("upos", pa.string()))
    .append(pa.field("lemma_src", pa.string()))
    .append(pa.field("is_abbrev", pa.bool_()))
)

FORMS_SCHEMA = pa.schema(
    [
        ("lemma", pa.string()),
        ("upos", pa.string()),
        ("form", pa.string()),
        ("tags", pa.list_(pa.string())),
        ("origin", pa.string()),  # wikt_entry | wikt_form_of | oewn_form
        ("source", pa.string()),
        ("source_id", pa.string()),
    ]
)

VARIANTS_SCHEMA = pa.schema(
    [
        ("variant", pa.string()),
        ("target", pa.string()),
        ("upos", pa.string()),
        ("tags", pa.list_(pa.string())),
        ("gloss", pa.string()),
        ("origin", pa.string()),  # wikt_alt_of | wikt_untagged | wikt_entry_form
        ("source", pa.string()),
        ("source_id", pa.string()),
    ]
)

DEDUPE_SCHEMA = pa.schema(
    [("source", pa.string()), ("dropped_id", pa.string()), ("kept_id", pa.string())]
)

BOOKKEEPING_TAGS = {"form-of", "alt-of", "no-gloss"}

# "<qualifiers> spelling/form of TARGET" with an optional ": gloss of target" tail.
_VARIANT_QUALIFIERS = (
    r"alternative|obsolete|archaic|dated|rare|uncommon|nonstandard|non-standard|informal|"
    r"colloquial|dialectal|eye[- ]dialect|pronunciation|deliberate|nonce|historical|superseded|"
    r"former|earlier|later|less\s+common|more\s+common|common|standard|early\s+modern|"
    r"non-oxford|british|english|american|us|uk|canadian|australian|scottish|irish|"
    r"letter-case|honorific|chiefly|mainly|especially|now|also|and"
)
VARIANT_RE = re.compile(
    rf"^\s*(?:(?:{_VARIANT_QUALIFIERS})[\s,]+)*"
    r"(?:spelling|form|misspelling|typography|capitali[sz]ation|letter-case\s+form)"
    r"\s+of\s+(?P<target>[^;:()]{1,80}?)\s*(?:\.\s*)?(?::|$)",
    re.IGNORECASE,
)
_MISSPELLING_RE = re.compile(
    r"^\s*(?:common\s+)?misspelling\s+of\s+(?P<target>[^;:()]{1,80}?)\s*\.?\s*$", re.I
)


def nfc(s: str | None) -> str | None:
    return unicodedata.normalize("NFC", s) if s is not None else None


def normalise_lemma(s: str, char_map: dict[str, str]) -> str:
    s = unicodedata.normalize("NFC", s)
    if char_map:
        s = s.translate(str.maketrans(char_map))
    return " ".join(s.split())


def variant_target(gloss: str | None) -> str | None:
    """Return X if the gloss is just '<qualifiers> spelling/form of X', else None."""
    if not gloss:
        return None
    m = VARIANT_RE.match(gloss) or _MISSPELLING_RE.match(gloss)
    if not m:
        return None
    target = m.group("target").strip().rstrip(".").strip()
    return target if 0 < len(target.split()) <= 6 else None


def to_upos(
    source: str, pos: str, lemma: str, tags: Iterable[str], relations: str, cfg: dict[str, Any]
) -> str | None:
    table = cfg["upos_map"][source]
    upos = table.get(pos)
    if upos is None:  # prefix rules ("sct:" -> …) and a per-source default ("*")
        upos = next((v for k, v in table.items() if k.endswith(":") and pos.startswith(k)), None)
        upos = upos or table.get("*")
    if upos is None:
        return None
    if upos == "NOUN|PROPN":  # case convention of Wikidata labels / MeSH organism names
        upos = "PROPN" if lemma[:1].isupper() else "NOUN"
    if upos == "CONJ":  # coordinating vs subordinating by a closed list
        upos = "CCONJ" if lemma.lower() in set(cfg["coordinating_conjunctions"]) else "SCONJ"
    if source == "oewn" and upos == "NOUN":
        rules = set(cfg.get("oewn_propn_when", []))
        if ("namenet" in rules and "oewn:namenet" in tags) or (
            "instance" in rules and '"instance_hypernym"' in relations
        ):
            upos = "PROPN"
    return upos


def dedupe_key(
    source: str, lemma: str, upos: str, gloss: str, gloss_path: list[str], tags: list[str]
) -> bytes:
    g = re.sub(r"[\s.;:]+$", "", gloss.casefold()).strip()
    blob = "\x1f".join([source, lemma, upos, g, "\x1e".join(gloss_path), ",".join(sorted(tags))])
    return hashlib.blake2b(blob.encode("utf-8"), digest_size=16).digest()


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    prov = Provenance(stage="inventory.normalise", config=cfg)
    prov.add_input(cfg["input"])
    char_map: dict[str, str] = cfg.get("lemma_char_map", {})
    abbrev_tags = set(cfg["alt_of"]["abbreviation_tags"])
    entry_form_variant_tags = set(cfg["entry_forms"]["variant_tags"])
    entry_form_skip_tags = set(cfg["entry_forms"]["skip_tags"])
    detect_untagged = bool(cfg.get("detect_untagged_variants", True))
    dedupe_mode = cfg.get("dedupe", "exact")

    outs = cfg["outputs"]
    w_sense = BatchWriter(outs["senses"], NORM_SCHEMA, prov)
    w_forms = BatchWriter(outs["forms"], FORMS_SCHEMA, prov)
    w_var = BatchWriter(outs["variants"], VARIANTS_SCHEMA, prov)
    w_dup = BatchWriter(outs["dedupe_map"], DEDUPE_SCHEMA, prov)
    writers = (w_sense, w_forms, w_var, w_dup)

    stats: Counter[str] = Counter()
    seen_entries: set[str] = set()
    seen_keys: dict[bytes, str] = {}
    pf = pq.ParquetFile(resolve(cfg["input"]))
    try:
        for batch in pf.iter_batches(batch_size=int(cfg.get("batch_rows", 50_000))):
            for row in batch.to_pylist():
                src = row["source"]
                stats[f"in:{src}"] += 1
                lemma_src = row["lemma"]
                lemma = normalise_lemma(lemma_src, char_map)
                tags = [t for t in row["tags"] if t not in BOOKKEEPING_TAGS]
                upos = to_upos(src, row["pos"], lemma, row["tags"], row["relations"], cfg)
                if upos is None:
                    stats[f"drop_pos:{src}:{row['pos']}"] += 1
                    continue
                gloss = nfc(row["gloss"])
                entry_key = row["entry_key"]

                # entry-level forms (once per entry)
                if entry_key not in seen_entries:
                    seen_entries.add(entry_key)
                    for f in row["forms"]:
                        form = normalise_lemma(f["form"], char_map)
                        ftags = list(f["tags"])
                        if not form or entry_form_skip_tags.intersection(ftags):
                            continue
                        if entry_form_variant_tags.intersection(ftags):
                            w_var.write(
                                {
                                    "variant": form,
                                    "target": lemma,
                                    "upos": upos,
                                    "tags": ftags,
                                    "gloss": None,
                                    "origin": "wikt_entry_form",
                                    "source": src,
                                    "source_id": entry_key,
                                }
                            )
                            stats["variants:entry_form"] += 1
                        elif form != lemma:
                            w_forms.write(
                                {
                                    "lemma": lemma,
                                    "upos": upos,
                                    "form": form,
                                    "tags": ftags,
                                    "origin": {"oewn": "oewn_form", "wiktionary": "wikt_entry"}.get(
                                        src, f"{src}_form"
                                    ),
                                    "source": src,
                                    "source_id": entry_key,
                                }
                            )
                            stats[f"forms:{src}_entry"] += 1

                kind = row["kind"]
                if kind == "no_gloss":
                    stats["drop:no_gloss"] += 1
                    continue
                if kind == "form_of":
                    targets = row["targets"] or []
                    for t in targets:
                        w_forms.write(
                            {
                                "lemma": normalise_lemma(t, char_map),
                                "upos": upos,
                                "form": lemma,
                                "tags": tags,
                                "origin": "wikt_form_of",
                                "source": src,
                                "source_id": row["source_id"],
                            }
                        )
                    stats["route:form_of"] += 1
                    continue
                vt = variant_target(gloss)
                is_abbrev = bool(abbrev_tags.intersection(tags))
                if kind == "alt_of":
                    if vt is not None or not is_abbrev:
                        target = vt or (row["targets"][0] if row["targets"] else None)
                        w_var.write(
                            {
                                "variant": lemma,
                                "target": normalise_lemma(target, char_map) if target else None,
                                "upos": upos,
                                "tags": tags,
                                "gloss": gloss,
                                "origin": "wikt_alt_of",
                                "source": src,
                                "source_id": row["source_id"],
                            }
                        )
                        stats["route:variant_alt_of"] += 1
                        continue
                    stats["route:abbrev_sense"] += 1
                elif detect_untagged and src == "wiktionary" and vt is not None:
                    w_var.write(
                        {
                            "variant": lemma,
                            "target": normalise_lemma(vt, char_map),
                            "upos": upos,
                            "tags": tags,
                            "gloss": gloss,
                            "origin": "wikt_untagged",
                            "source": src,
                            "source_id": row["source_id"],
                        }
                    )
                    stats["route:variant_untagged"] += 1
                    continue

                gloss_path = [nfc(g) for g in row["gloss_path"]]
                if dedupe_mode == "exact":
                    key = dedupe_key(src, lemma, upos, gloss or "", gloss_path, tags)
                    kept = seen_keys.get(key)
                    if kept is not None:
                        w_dup.write(
                            {"source": src, "dropped_id": row["source_id"], "kept_id": kept}
                        )
                        stats[f"drop:exact_duplicate:{src}"] += 1
                        continue
                    seen_keys[key] = row["source_id"]

                row.update(
                    lemma=lemma,
                    lemma_src=lemma_src if lemma_src != lemma else None,
                    gloss=gloss,
                    gloss_path=gloss_path,
                    examples=[nfc(e) for e in row["examples"]],
                    tags=tags,
                    forms=[{"form": nfc(f["form"]), "tags": f["tags"]} for f in row["forms"]],
                    upos=upos,
                    is_abbrev=is_abbrev,
                )
                w_sense.write(row)
                stats[f"out:{src}"] += 1
                stats[f"out:{src}:{upos}"] += 1
    except BaseException:
        for w in writers:
            w.abort()
        raise
    out_stats = dict(sorted(stats.items()))
    for w in writers:
        w.close(stats=out_stats)
    LOG.info("normalise: %s", orjson.dumps(out_stats, option=orjson.OPT_INDENT_2).decode())
    return out_stats


def main() -> None:
    args = stage_args("Normalise raw senses (Unicode, casing, UPOS, form-of/variant routing)")
    run(load_config(args.config))


if __name__ == "__main__":
    main()

__all__ = ["FORM_TYPE", "VARIANT_RE", "normalise_lemma", "run", "to_upos", "variant_target"]
