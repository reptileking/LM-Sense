# SenseLM

Research system for **sense-level tokenization**: every distinct meaning of every English word
(plus domain terms) is one token. A translator converts English to and from this sense-token
language; a core language model is trained and runs entirely on sense tokens.

Status: **Phase 2 — inventory v1.0 (2,557,970 sense tokens) + 393 structural tokens**, at Gate 2
(Wiktionary, Open English WordNet, Wikidata lexemes, MeSH, Gene Ontology, ChEBI, NCBI Taxonomy;
Wikidata items and SNOMED CT UK prepared for a later build).
See [`docs/phase2_report.md`](docs/phase2_report.md) and, for v0.1,
[`docs/phase1_report.md`](docs/phase1_report.md).

## Setup

```bash
uv sync                      # Python 3.12, CPU PyTorch (see docs/DECISIONS.md D-001)
uv run pytest                # unit + end-to-end tests on tiny fixtures
uv run python -m senselm.hardware --out docs/HARDWARE.md
```

## Phase 1 pipeline

Every stage is `python -m senselm.<stage> --config configs/<stage>.yaml`, deterministic given its
config and inputs, and records input checksums + source versions in its outputs' metadata.

```bash
uv run python -m senselm.ingest.download   --config configs/download.yaml          # raw dumps (sha256-pinned)
uv run python -m senselm.ingest.wiktionary --config configs/ingest_wiktionary.yaml # ~4 min, streaming
uv run python -m senselm.ingest.oewn       --config configs/ingest_oewn.yaml       # ~30 s
uv run python -m senselm.ingest.combine    --config configs/ingest.yaml            # -> data/interim/senses_raw.parquet
uv run python -m senselm.inventory.normalise --config configs/normalise.yaml       # UPOS, forms, variants
uv run python -m senselm.inventory.align   --config configs/align.yaml             # gloss embeddings, ~20 min on 4 CPUs
uv run python -m senselm.inventory.judge   --config configs/align.yaml             # LLM-judge cost estimate (dry run)
uv run python -m senselm.eval.alignment_eval --config configs/alignment_eval.yaml  # P/R vs eval/alignment_gold.csv
uv run python -m senselm.inventory.build   --config configs/inventory.yaml         # fine policy + permanent IDs
uv run python -m senselm.inventory.build   --config configs/inventory_coarse.yaml  # coarse policy (counts)
uv run python -m senselm.inventory.stats   --config configs/stats.yaml             # report numbers
```

## Phase 2 pipeline

```bash
uv run python -m senselm.ingest.download          --config configs/download.yaml             # + Phase-2 files
uv run python -m senselm.ingest.wikidata_lexemes  --config configs/ingest_wikidata_lexemes.yaml
uv run python -m senselm.ingest.wikidata_items stream --config configs/ingest_wikidata_items.yaml  # ~4.5 h, never stored
uv run python -m senselm.ingest.wikidata_items select --config configs/ingest_wikidata_items.yaml
uv run python -m senselm.ingest.ncbi_taxonomy     --config configs/ingest_ncbi_taxonomy.yaml
uv run python -m senselm.ingest.obo               --config configs/ingest_go.yaml
uv run python -m senselm.ingest.obo               --config configs/ingest_chebi.yaml
uv run python -m senselm.ingest.mesh              --config configs/ingest_mesh.yaml
uv run python -m senselm.ingest.snomed            --config configs/ingest_snomed.yaml        # needs TRUD files (D-036)
uv run python -m senselm.ingest.combine           --config configs/ingest_v1.yaml
uv run python -m senselm.inventory.normalise      --config configs/normalise_v1.yaml
uv run python -m senselm.inventory.attach         --config configs/attach.yaml               # stage-2 alignment
uv run python -m senselm.inventory.build          --config configs/inventory_v1.yaml         # v1.0 + permanent IDs
uv run python -m senselm.inventory.structural     --config configs/structural.yaml           # structural tokens
uv run python -m senselm.inventory.stats_v1       --config configs/stats_v1.yaml             # report numbers
uv run python -m senselm.eval.attach_gold evaluate --config configs/attach.yaml              # stage-2 P/R
```

Uncertain OEWN–Wiktionary pairs are judged by a locally hosted LLM (D-030); on the judging machine:

```bash
uv run python -m senselm.inventory.judge_local probe --config configs/judge_local.yaml --model <model>
uv run python -m senselm.inventory.judge_local pilot --config configs/judge_local.yaml --model <model>
uv run python -m senselm.inventory.judge_local run   --config configs/judge_local.yaml --model <model>
```

## Phase 3 pipeline (codec)

```bash
uv run python -m senselm.inventory.structural --config configs/structural.yaml  # + codec-v1 tokens
uv run python -m senselm.codec.build          --config configs/codec.yaml       # paradigm / variant tables
uv run python -m senselm.codec --config configs/codec.yaml roundtrip "It's the dogs' bones."
uv run python -m senselm.ingest.download      --config configs/download.yaml --only gpt2_tokenizer qwen25_tokenizer wsd_framework
uv run python -m senselm.eval.corpus_sample   --config configs/corpus_sample.yaml   # FineWeb-Edu sample
uv run python -m senselm.eval.codec_eval      --config configs/codec_eval.yaml --split eval
uv run python -m senselm.eval.wsd_eval        --config configs/wsd_eval.yaml        # needs the WSD framework
```

One-off experiments (also reproducible):

```bash
uv run python -m senselm.eval.alignment_gold sample --config configs/align.yaml          # blind gold sample
uv run python -m senselm.eval.embedding_selection --config configs/embedding_selection.yaml
```

## Layout

```
senselm/  ingest/      one module per source (+ download, combine)
          inventory/   normalise, candidates, align, embed, judge(_local), policy, attach, ids, build,
                       structural, stats(_v1)
          codec/       (Phase 3) encode text→IDs, decode IDs→text
          tagging/     (Phase 4) LLM-assisted corpus tagging + synthetic data
          models/      (Phase 5) translator + core model
          eval/        alignment gold set, embedding selection, metrics
data/     raw/ interim/ processed/   (git-ignored)
configs/  one YAML per stage
docs/     DECISIONS.md, LICENSES.md, GRANULARITY.md, phase reports
eval/     hand-labelled gold data and evaluation outputs
registry/ committed ledger of the permanent ID registry (+ structural tokens)
judge/    local-LLM judge job file and committed verdicts
tests/
```

## Licences

Data products inherit their sources' licences (Wiktionary: CC BY-SA 4.0 / GFDL; OEWN, GO, ChEBI:
CC BY 4.0; Wikidata: CC0; NCBI Taxonomy: public domain; MeSH: NLM terms; SNOMED CT: restricted,
never committed).
Every sense records its provenance so licensed subsets can be excluded. The OED and BabelNet are
never used. Details: [`docs/LICENSES.md`](docs/LICENSES.md).
