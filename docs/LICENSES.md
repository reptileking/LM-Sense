# Licences and obligations

Every source used by SenseLM, its exact version, its licence, and what that obliges us to do.
Every sense record carries its provenance (`members`: list of `(source, source_id)`, plus
`sources` and `licences` columns in `inventory_v0_1.parquet` / `inventory_v1_0.parquet`), so any
licensed subset can be excluded by filtering, e.g. the CC BY-only subset of v0.1 is
`sources == ["oewn"]`.

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

## Data sources (Phase 2)

| Source | Version used | Licence | Where it is used |
|---|---|---|---|
| Wikidata lexemes | `wikidata-20261028-lexemes.json.gz` (md5 `b7513e4d…9ffed6`, sha256 pinned in `configs/download.yaml`) | CC0 1.0 | English lexeme senses (glosses, examples), forms with grammatical features, sense → item links |
| Wikidata items | `wikidata-20261026-all.json.gz`, streamed once and never stored (md5 `356990468877421dc061694ad61c487f`, checked after the pass; D-032) — **not in v1.0** (pass lost, D-045) | CC0 1.0 | concept senses (label = lemma, English description = gloss), cross-references to the domain sources |
| NCBI Taxonomy | taxdump monthly archive `taxdmp_2026-11-01.zip` | Public domain (U.S. Government work; NCBI places no restrictions on use or redistribution; citation requested) | scientific-name senses (genus and above, species); names also decide MeSH genus casing |
| Gene Ontology | `go.obo`, release 2026-07-26 | CC BY 4.0 | GO term senses (molecular function, biological process, cellular component) |
| ChEBI | release 255, `chebi.obo.gz` (3-star entities only) | CC BY 4.0 | chemical entity senses |
| MeSH | 2026 descriptors `desc2026.xml` and supplementary concept records `supp2026.xml` (files of 2026-08-12) | NLM Terms and Conditions for MeSH (free of charge; attribution; modifications must be indicated; no implied NLM endorsement) | descriptor and SCR senses (chemicals, rare diseases, organisms) |
| SNOMED CT UK Edition | **not yet used** (needs the owner's NHS TRUD download; D-036) | SNOMED CT Affiliate Licence via the UK national licence — **use only, no redistribution** | clinical concept senses, tagged `licence:restricted` |

Exact URLs, sizes and checksums: `data/raw/MANIFEST.json` and `configs/download.yaml`.

### Obligations (Phase 2)

**Wikidata (CC0 1.0)** — none. Attribution ("Wikidata, CC0") is given as a courtesy.

**NCBI Taxonomy (public domain)** — none legally; cite Schoch et al., *NCBI Taxonomy: a
comprehensive update on curation, resources and tools*, Database 2020, and the dump date.

**Gene Ontology (CC BY 4.0)** — *attribution*: cite the GO Consortium (Ashburner et al., Nat.
Genet. 2000; The Gene Ontology Consortium, *The Gene Ontology knowledgebase in 2023*, Genetics
2023) and the release date, link the licence, indicate changes (terms become senses; glosses are
GO definitions; undefined terms get synthesised glosses).

**ChEBI (CC BY 4.0)** — *attribution*: cite Hastings et al., *ChEBI in 2016*, Nucleic Acids Res.
2016, and release 255, link the licence, indicate changes (HTML markup removed from definitions;
synthesised glosses for undefined entities).

**MeSH (NLM terms)** — include the NLM acknowledgement with every redistribution, in substance:
"This product uses publicly available data from the U.S. National Library of Medicine (NLM),
National Institutes of Health, Department of Health and Human Services; NLM is not responsible
for the product and does not endorse or recommend this or any other product"; state the MeSH
version; say that the data were modified (headings turned into dictionary-form lemmas, D-035;
SCR glosses synthesised) and do not present the result as MeSH. The owner should re-read the
current NLM terms before any public release.

**SNOMED CT UK (affiliate licence, restricted)** — SNOMED content may be used under the UK
licence but must not be redistributed: no SNOMED-derived text in git, in released inventories or
in released models trained on it without checking the licence terms. Enforced by D-036: rows are
tagged `licence:restricted`, the build refuses restricted sources unless the redacting ledger
writer is enabled, and the `no restricted` subset is a simple filter on `sources`.

**The v1.0 inventory as a whole** stays CC BY-SA 4.0 (it contains Wiktionary text). Subsets by
filtering `sources` / `licences`: senses attested only by CC0 / public-domain sources (Wikidata,
NCBI Taxonomy) carry no obligations; adding OEWN, GO, ChEBI needs attribution (CC BY 4.0);
adding MeSH needs the NLM statement; Wiktionary brings share-alike. Counts per subset:
docs/phase2_stats.md ("Licence scenarios").

### Tools added in Phase 2

| Tool | Licence | Use |
|---|---|---|
| lemminflect 0.2.3 | MIT | singularising MeSH headings (dictionary lookups) |
| lxml | BSD | streaming MeSH XML |
