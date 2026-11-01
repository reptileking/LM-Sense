# Licences and obligations

Every source used by SenseLM, its exact version, its licence, and what that obliges us to do.
Every sense record carries its provenance (`members`: list of `(source, source_id)`, plus
`sources` and `licences` columns in `inventory_v0_1.parquet`), so any licensed subset can be
excluded by filtering, e.g. the CC BY-only subset is `sources == ["oewn"]`.

**Never used, in any form (hard constraint):** the Oxford English Dictionary (OED) and
BabelNet. No code path downloads or reads them; no derived resource (e.g. BabelNet-based
WSD datasets or sense mappings) is used either.

## Data sources (Phase 1)

| Source | Version used | Licence | Where it is used |
|---|---|---|---|
| English Wiktionary via kaikki.org (wiktextract) | kaikki.org English extraction of 2026-10-25 from the enwiktionary dump of 2026-10-01 (wiktextract f9fe161 + 6f96ddf); `kaikki.org-dictionary-English.jsonl.gz`, sha256 `3ff7bef2…41392dc` | CC BY-SA 4.0 and GFDL (Wiktionary text is dual-licensed; kaikki.org data is distributed under the same terms) | glosses, examples, tags, topics, forms, variants, abbreviation senses |
| Open English WordNet 2025+ | `english-wordnet-2025-plus.xml.gz` (release 2025-12-31), sha256 `31f4af16…5b586bc`; the base 2025 edition (sha256 `9ca6d1dc…1674ad3fe`) is read only to flag Namenet senses | CC BY 4.0 (declared on the `Lexicon` element); derived from Princeton WordNet 3.x | glosses, examples, synsets, supersenses, relations, irregular forms |
| Open English Namenet (inside OEWN 2025+) | as above | CC BY 4.0 (same lexicon licence) | proper-noun senses, tagged `oewn:namenet` |

Full checksums, URLs and HTTP metadata: `data/raw/MANIFEST.json` (written by
`python -m senselm.ingest.download`) and the provenance block embedded in every Parquet output.

### Obligations

**Wiktionary (CC BY-SA 4.0 / GFDL)**
- *Attribution*: credit "Wiktionary contributors" with a link to the source entries (each
  Wiktionary-derived record keeps its wiktextract sense id, from which the page is
  recoverable: `https://en.wiktionary.org/wiki/<lemma>`), credit kaikki.org / wiktextract
  (Ylonen, *Wiktextract: Wiktionary as Machine-Readable Structured Data*, LREC 2022), and
  link the licence.
- *Indicate changes*: our records are normalised and merged (see docs/DECISIONS.md); say so.
- *Share-alike*: any distributed dataset that contains Wiktionary-derived text (glosses,
  examples, the alignment gold set, the inventory when Wiktionary members are included) must
  be released under CC BY-SA 4.0 (or a compatible licence). The inventory and the gold set are
  therefore CC BY-SA 4.0 as a whole; the `sources == ["oewn"]` subset is CC BY 4.0.
- *Open question for later phases*: whether share-alike reaches **model weights** trained on
  SA data is legally unsettled. Phase 4/5 artefacts should keep the option of training on
  the CC BY-only subset; see the Gate 1 open decisions.

**Open English WordNet (CC BY 4.0)**
- *Attribution*: cite McCrae et al., *English WordNet 2019 – An Open-Source WordNet for
  English* (GWC 2019) and the release used (OEWN 2025+), link the licence, indicate changes.
- OEWN derives from Princeton WordNet. As a conservative measure, redistributions should also
  carry the Princeton notice: "WordNet 3.0 Copyright 2006 by Princeton University. All rights
  reserved." together with its permission/disclaimer text (the WordNet licence is permissive
  and requires the notice to be included in copies).

## Tools and models (not redistributed)

| Tool / model | Version | Licence | Use |
|---|---|---|---|
| BAAI/bge-base-en-v1.5 | HF commit `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a` | MIT | gloss embeddings for alignment (outputs are similarity scores only) |
| BAAI/bge-small-en-v1.5, BAAI/bge-large-en-v1.5 | HF `main` at 2026-11-01 | MIT | model selection only |
| intfloat/e5-large-v2 | HF `main` at 2026-11-01 | MIT | model selection only |
| sentence-transformers/all-mpnet-base-v2 | HF `main` at 2026-11-01 | Apache-2.0 | model selection only |
| Anthropic API (LLM judge, optional) | not used in Phase 1 (no API key; dry-run only) | commercial terms | would label uncertain pairs; outputs are short verdicts |
| Python packages | pinned in `uv.lock` | permissive (MIT/BSD/Apache) | pipeline code |

## Project artefacts

| Artefact | Licence |
|---|---|
| Code in this repository | to be chosen by the owner (pyproject currently says MIT for code) |
| `eval/alignment_gold.csv` (contains glosses from both sources) | CC BY-SA 4.0 |
| `data/processed/inventory_v0_1.parquet`, `id_registry.parquet` | CC BY-SA 4.0 as a whole; OEWN-only rows CC BY 4.0 |

## Phase 2 sources (to be added with their ingest modules)

Wikidata lexemes and items (CC0), NCBI Taxonomy (public domain, NCBI), Gene Ontology (CC BY
4.0), ChEBI (CC BY 4.0), MeSH (NLM terms and conditions; free, attribution and no
modification of the vocabulary's meaning), SNOMED CT UK edition (TRUD licence: affiliate /
UK national licence, **not redistributable**; would need its own excluded subset).
