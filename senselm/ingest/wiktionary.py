"""Ingest the kaikki.org wiktextract English dump into the unified raw sense schema.

``python -m senselm.ingest.wiktionary --config configs/ingest_wiktionary.yaml``

Streams the gzipped JSONL line by line (never materialising the dump) and writes
``data/interim/senses_raw_wiktionary.parquet`` in batches. Every Wiktionary sense becomes
one row, including form-of / alt-of / gloss-less senses; they are classified in ``kind`` and
the policy decision about what counts as a sense is made later in
``senselm.inventory.normalise`` (keeps ingest a faithful extraction).

Two passes over the dump:
1. count sense topics (to choose the most specific topic as ``domain``: wiktextract emits
   topics alphabetised together with their auto-added ancestors such as ``sciences``);
2. extract rows.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import logging
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import orjson

from senselm.common import (
    BatchWriter,
    Provenance,
    load_config,
    resolve,
    stage_args,
)
from senselm.hardware import resolve_auto
from senselm.ingest.manifest import source_version
from senselm.ingest.schema import SENSE_SCHEMA, make_row

LOG = logging.getLogger("senselm.ingest.wiktionary")

SOURCE = "wiktionary"

# Sense-level relation fields copied (as word lists) into the relations JSON.
_REL_FIELDS = (
    "synonyms",
    "antonyms",
    "hypernyms",
    "hyponyms",
    "holonyms",
    "meronyms",
    "troponyms",
    "coordinate_terms",
    "related",
    "derived",
)


def iter_entries(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield (line_no, entry) from a (gzipped) JSONL dump, streaming."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as raw:
        fh = io.BufferedReader(raw, buffer_size=1 << 22) if path.suffix == ".gz" else raw
        for line_no, line in enumerate(fh):
            if line.strip():
                yield line_no, orjson.loads(line)


def count_topics(path: Path, lang_code: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for _, entry in iter_entries(path):
        if entry.get("lang_code") != lang_code:
            continue
        for sense in entry.get("senses") or ():
            counts.update(sense.get("topics") or ())
    return counts


def most_specific_topic(topics: list[str], freq: Counter[str]) -> str | None:
    """The least frequent topic in the list (ancestors such as `sciences` are frequent)."""
    if not topics:
        return None
    return min(topics, key=lambda t: (freq.get(t, 0), t))


def classify(sense: dict[str, Any]) -> str:
    tags = sense.get("tags") or ()
    if not sense.get("glosses") or "no-gloss" in tags:
        return "no_gloss"
    if "form_of" in sense or "form-of" in tags:
        return "form_of"
    if "alt_of" in sense or "alt-of" in tags:
        return "alt_of"
    return "sense"


def _words(items: Any) -> list[str]:
    out: list[str] = []
    for it in items or ():
        w = it.get("word") if isinstance(it, dict) else it
        if isinstance(w, str) and w:
            out.append(w)
    return out


ID_DELIM = "#"  # never occurs in wiktextract ids (their alphabet includes '~', '_', ':')


def disambiguate_id(sid: str, glosses: list[str], seen: set[str]) -> str:
    """Make a colliding wiktextract sense id unique, stably.

    Collisions mostly come from subsenses inheriting their parent's ``senseid``-derived id.
    The suffix is a hash of the gloss path, so it survives reordering of subsenses; only
    identical gloss paths fall back to an ordinal.
    """
    if sid not in seen:
        return sid
    digest = hashlib.blake2b("\x1e".join(glosses).encode("utf-8"), digest_size=5).hexdigest()
    cand = f"{sid}{ID_DELIM}{digest}"
    k = 2
    while cand in seen:
        cand = f"{sid}{ID_DELIM}{digest}{ID_DELIM}{k}"
        k += 1
    return cand


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _strs(items: Any) -> list[str]:
    return [str(x) for x in items or () if x is not None and str(x) != ""]


def entry_forms(entry: dict[str, Any], drop_tags: set[str]) -> list[dict[str, Any]]:
    forms = []
    for f in entry.get("forms") or ():
        form = f.get("form")
        tags = _strs(f.get("tags"))
        if not isinstance(form, str) or form in ("", "-", "—") or drop_tags.intersection(tags):
            continue
        forms.append({"form": form, "tags": tags})
    return forms


def sense_rows(
    line_no: int,
    entry: dict[str, Any],
    *,
    version: str,
    topic_freq: Counter[str],
    seen_ids: set[str],
    drop_form_tags: set[str],
    max_examples: int,
) -> Iterator[dict[str, Any]]:
    word = entry["word"]
    pos = entry.get("pos") or "unknown"
    etym = _int_or_none(entry.get("etymology_number"))
    entry_key = f"{word}|{pos}|{etym or 0}|L{line_no}"
    forms = entry_forms(entry, drop_form_tags)
    for rank, sense in enumerate(entry.get("senses") or ()):
        glosses = [g for g in (sense.get("glosses") or ()) if isinstance(g, str)]
        sid = disambiguate_id(
            sense.get("id") or f"en-{word}-en-{pos}-L{line_no}-S{rank}", glosses, seen_ids
        )
        seen_ids.add(sid)

        examples, ex_types = [], []
        for ex in sense.get("examples") or ():
            text = ex.get("text")
            if isinstance(text, str) and text.strip():
                examples.append(text)
                ex_types.append(str(ex.get("type") or "example"))
        if max_examples and len(examples) > max_examples:
            examples, ex_types = examples[:max_examples], ex_types[:max_examples]

        tags = _strs(sense.get("tags"))
        tags += [f"raw:{t}" for t in _strs(sense.get("raw_tags"))]
        topics = _strs(sense.get("topics"))
        targets = _words(sense.get("form_of")) + _words(sense.get("alt_of"))

        rel: dict[str, Any] = {}
        for k in _REL_FIELDS:
            ws = _words(sense.get(k))
            if ws:
                rel[k] = ws
        if sense.get("form_of"):
            rel["form_of"] = _words(sense["form_of"])
        if sense.get("alt_of"):
            rel["alt_of"] = _words(sense["alt_of"])
        for k in ("wikidata", "senseid", "wikipedia"):
            if sense.get(k):
                rel[k] = _strs(sense[k]) if isinstance(sense[k], list) else [str(sense[k])]
        if sense.get("qualifier"):
            rel["qualifier"] = sense["qualifier"]

        yield make_row(
            source=SOURCE,
            source_id=sid,
            lemma=word,
            pos=pos,
            gloss=glosses[-1] if glosses else None,
            examples=examples,
            tags=tags,
            domain=most_specific_topic(topics, topic_freq),
            forms=forms,
            relations=orjson.dumps(rel, option=orjson.OPT_SORT_KEYS).decode(),
            source_version=version,
            kind=classify(sense),
            gloss_path=glosses[:-1],
            example_types=ex_types,
            topics=topics,
            targets=targets,
            etymology_number=etym,
            entry_key=entry_key,
            line_no=line_no,
            sense_rank=rank,
        )


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    src = resolve(cfg["input"])
    lang_code = cfg.get("lang_code", "en")
    version = source_version(cfg, SOURCE)
    prov = Provenance(stage="ingest.wiktionary", config=cfg, source_versions={SOURCE: version})
    prov.add_input(src)

    LOG.info("pass 1/2: counting topics in %s", src)
    topic_freq = count_topics(src, lang_code)
    LOG.info("  %d distinct topics", len(topic_freq))

    LOG.info("pass 2/2: extracting senses")
    stats: Counter[str] = Counter()
    seen_ids: set[str] = set()
    batch_rows = int(resolve_auto(cfg.get("batch_rows", "auto"), "parquet_batch_rows"))
    drop_form_tags = set(cfg.get("drop_form_tags", []))
    max_examples = int(cfg.get("max_examples", 0))
    out = BatchWriter(cfg["output"], SENSE_SCHEMA, prov, batch_rows=batch_rows)
    try:
        for line_no, entry in iter_entries(src):
            if entry.get("lang_code") != lang_code:
                stats["entries_other_lang"] += 1
                continue
            stats["entries"] += 1
            for row in sense_rows(
                line_no,
                entry,
                version=version,
                topic_freq=topic_freq,
                seen_ids=seen_ids,
                drop_form_tags=drop_form_tags,
                max_examples=max_examples,
            ):
                stats["rows"] += 1
                stats[f"kind:{row['kind']}"] += 1
                if ID_DELIM in row["source_id"]:
                    stats["disambiguated_ids"] += 1
                out.write(row)
            if stats["entries"] % 250_000 == 0:
                LOG.info("  %d entries, %d rows", stats["entries"], stats["rows"])
    except BaseException:
        out.abort()
        raise
    out_stats = dict(sorted(stats.items()))
    out.close(stats=out_stats)
    LOG.info("done: %s", out_stats)
    return out_stats


def main() -> None:
    args = stage_args("Ingest kaikki.org wiktextract English JSONL")
    run(load_config(args.config))


if __name__ == "__main__":
    main()
