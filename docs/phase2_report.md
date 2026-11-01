# Phase 2 report — Sense inventory v1.0 (domain extension) and structural tokens

**Gate 2: waiting for approval.** Branch `claude/senselm-research-system-psq9xl`.
Supporting detail: [`phase2_stats.md`](phase2_stats.md) (all tables), [`phase2_stats.json`](phase2_stats.json)
(every number), [`STRUCTURAL_TOKENS.md`](STRUCTURAL_TOKENS.md), [`DECISIONS.md`](DECISIONS.md)
(D-030 – D-045 are new), [`LICENSES.md`](LICENSES.md),
[`../eval/attach_gold_eval.json`](../eval/attach_gold_eval.json) (stage-2 gold set).

## 1. What was built

### 1.1 Sources (spec 2.1)

| Source | Version | Licence | Ingest | Senses ingested | What is kept |
|---|---|---|---|---|---|
| Wikidata lexemes | dump 2026-10-28 | CC0 | `ingest.wikidata_lexemes` | 59,747 | every English lexeme sense with a gloss; forms with grammatical features feed the forms table |
| Wikidata items | JSON dump 2026-10-26 (156.7 GB, streamed, never stored) | CC0 | `ingest.wikidata_items stream` + `select` | **not in v1.0** (D-045) | ready: items linked from English lexeme senses + *concepts*: classes (P279) with a lower-case English label (common nouns), an English description and a Wikipedia article, minus people, Wikimedia pages and the types the domain sources own (D-038). The full pass was lost at 74% when the cloud container was recycled; a trial on the first 100 MB selects 987 of 2,435 superset items |
| NCBI Taxonomy | taxdump 2026-11-01 | public domain | `ingest.ncbi_taxonomy` | 957,340 | formal ranks domain … species with well-formed names; placeholders and specimen labels dropped (D-039) |
| Gene Ontology | release 2026-07-26 | CC BY 4.0 | `ingest.obo` | 38,092 | non-obsolete terms |
| ChEBI | release 255 | CC BY 4.0 | `ingest.obo` | 62,119 | 3-star entities; HTML stripped from definitions (D-040) |
| MeSH | 2026 descriptors + supplementary records | NLM terms | `ingest.mesh` | 351,984 | descriptors (30,960) + chemical (249,246), rare-disease (6,542) and organism (65,236) records; dictionary-form lemmas (D-035) |
| SNOMED CT UK | — | restricted | `ingest.snomed` | needs TRUD files | module + redaction path built and tested on fixtures (D-036) |

Only preferred names create senses; synonyms, entry terms and common names are kept for
alignment (D-034). Every sense keeps its source id, so any licensed subset can be removed by
filtering `sources` / `licences`.

### 1.2 Alignment of domain senses (spec 2.1, last bullet)

Stage 1 (OEWN ↔ Wiktionary) is the v0.1 fine policy, recomputed unchanged, so every v0.1 token
keeps its id. Stage 2 (`inventory.attach`, D-037) attaches each new source in turn to the
clusters that already exist: same lemma and POS class, score = best cosine to the cluster's
members (the stage-1 embedding model and cache), cross-references (Wikidata Q-ids of Wiktionary
and lexeme senses; with the items, also P685 / P486 / P683 / P686 through the items superset) beat
any cosine, greedy one-to-one per source; a sense that does not attach becomes a new cluster. Examples from the build: MeSH
*Mice* ("the common name for the genus Mus") and the lexeme sense "any small rodent of the genus
Mus" join Wiktionary's genus-*Mus* subsense of "mouse", not the broader "any small rodent" sense;
"aspirin" and "malaria" each merge MeSH, OEWN, Wiktionary and a Wikidata lexeme; *Escherichia
coli* merges MeSH, NCBI Taxonomy and OEWN; the NCBI genus *Mus* joins OEWN's "type genus of the
Muridae". A taxon can never join a cluster whose WordNet lexicographer file contradicts its
kingdom.

Per-source thresholds come from a blind gold set of 240 new-source senses (40 per source,
labelled before scores were looked at; `eval/attach_gold.csv`):

| Source | gold senses | with a true match | threshold | precision | recall |
|---|---|---|---|---|---|
| Wikidata lexemes | 40 | 36 | 0.60 | 0.78 | 0.78 |
| Wikidata items (trial superset) | 34 | 34 | 0.58 | 0.97 | 0.97 |
| MeSH | 40 | 40 | 0.60 | 0.97 | 0.93 |
| Gene Ontology | 40 | 37 | 0.65 | 0.95 | 1.00 |
| ChEBI | 40 | 36 | 0.65 | 0.89 | 0.94 |
| NCBI Taxonomy | 40 | 27 | 0.53 | 1.00 | 0.74 |
| **all v1.0 sources, micro-averaged** | 200 | 176 | per source | **0.91** | **0.89** |

Precision = attached to the labelled sense; recall = a true match exists and was attached. The
items row comes from the trial on the first 100 MB of the dump (6 of its 40 gold items are
language names that the final lower-case-label filter no longer selects). Lexemes are the weak
spot: their glosses are often vague ("chemical compound", "type of paint") and pick the wrong
one of two close senses; the errors are wrong candidates, not low scores, so no threshold fixes
them (a job for the LLM judge later).

### 1.3 Structural vocabulary (spec 2.2)

393 tokens in the reserved range `0xFFC00000`– (one reserved lemma slot of 1,024 ids per
category, so each can grow without renumbering; append-only registry
`registry/structural_registry.parquet`; full list in [`STRUCTURAL_TOKENS.md`](STRUCTURAL_TOKENS.md)):

| category | tokens | content |
|---|---|---|
| special | 12 | pad, bos, eos, unk, mask, sep, document start/end, system/user/assistant turn, turn end |
| byte | 258 | 256 byte values (byte = sense index) + `<bytes_start>` / `<bytes_end>` |
| morph | 32 | UniMorph features (tense, aspect, mood, voice, person, number, degree, finiteness, case, gender, polarity, possessive, definiteness), emitted after the sense token |
| case | 4 | capitalised, all caps, forced lower case, mixed (→ byte fallback) |
| space | 7 | no-space, space, newline, paragraph, tab, indent, clitic |
| punct | 56 | one token per punctuation character / symbol |
| digit | 10 | numbers are digit sequences |
| prag | 8 | formal, informal, emphasis start/end, quotation start/end, irony, hedge |
| variant | 6 | British / American / archaic / abbreviated / nonstandard / other spelling of the next sense's lemma |

### 1.4 Commands

| Step | Command | Wall time (4 vCPU) | Output |
|---|---|---|---|
| download (Phase-2 files) | `python -m senselm.ingest.download --config configs/download.yaml` | a few minutes | `data/raw/{wikidata,ncbi_taxonomy,go,chebi,mesh}/` (2.5 GB in all) |
| Wikidata lexemes | `python -m senselm.ingest.wikidata_lexemes --config configs/ingest_wikidata_lexemes.yaml` | 71 s | `senses_raw_wikidata_lexemes.parquet`, `wikidata_lexeme_items.parquet` |
| Wikidata items, pass | `python -m senselm.ingest.wikidata_items stream --config configs/ingest_wikidata_items.yaml` | ~4.5 h, network-bound at 10 MB/s (the v1.0 attempt was lost at 74%, D-045) | `wikidata_items_superset.parquet` |
| Wikidata items, filter | `python -m senselm.ingest.wikidata_items select --config configs/ingest_wikidata_items.yaml` | seconds (trial) | `senses_raw_wikidata_items.parquet` |
| NCBI Taxonomy | `python -m senselm.ingest.ncbi_taxonomy --config configs/ingest_ncbi_taxonomy.yaml` | 71 s | `senses_raw_ncbi_taxonomy.parquet` |
| Gene Ontology | `python -m senselm.ingest.obo --config configs/ingest_go.yaml` | 3 s | `senses_raw_gene_ontology.parquet` |
| ChEBI | `python -m senselm.ingest.obo --config configs/ingest_chebi.yaml` | 10 s | `senses_raw_chebi.parquet` |
| MeSH | `python -m senselm.ingest.mesh --config configs/ingest_mesh.yaml` | 80 s | `senses_raw_mesh.parquet` |
| SNOMED CT UK | `python -m senselm.ingest.snomed --config configs/ingest_snomed.yaml` | — (needs TRUD files) | `senses_raw_snomed.parquet` |
| combine | `python -m senselm.ingest.combine --config configs/ingest_v1.yaml` | 45 s | `senses_raw_v1.parquet` |
| normalise | `python -m senselm.inventory.normalise --config configs/normalise_v1.yaml` | 2.5 min | `senses_norm_v1`, `forms_v1`, `variants_v1`, `dedupe_map_v1` |
| stage-2 alignment | `python -m senselm.inventory.attach --config configs/attach.yaml` | 4 min with the embedding cache warm (embedding the ~140k new glosses the first time took ~50 min on this CPU) | `attachments_v1.parquet` |
| build v1.0 + ids | `python -m senselm.inventory.build --config configs/inventory_v1.yaml` | 3.6 min | `inventory_v1_0.parquet`, `id_registry.parquet`, `lemma_registry.parquet`, `registry/` ledger |
| structural tokens | `python -m senselm.inventory.structural --config configs/structural.yaml` | < 1 s | `structural_registry.parquet`, `docs/STRUCTURAL_TOKENS.md` |
| statistics | `python -m senselm.inventory.stats_v1 --config configs/stats_v1.yaml` | 3 s | `docs/phase2_stats.{json,md}` |
| stage-2 gold evaluation | `python -m senselm.eval.attach_gold evaluate --config configs/attach.yaml` | 35 s | `eval/attach_gold_eval.json` |

Tests: `uv run pytest` — 77 tests, all pass. New in Phase 2: stage-2 attachment rules (cosine,
cross-reference, Wikidata bridge, one-to-one per source, case and kingdom rules), MeSH lemma
rules (22 cases), Wikidata line filter and concept selection, structural registry (layout, append-only), SNOMED RF2 reader on fixtures,
restricted-source redaction / rehydration / hold-out, sharded ledger, config inheritance, and
the committed ledger read from HEAD in either layout.

## 2. Key numbers

### Vocabulary

| | v0.1 | **v1.0** |
|---|---|---|
| sense tokens (canonical senses, active token ids) | 1,236,563 | **2,557,970** |
| lemmas | 837,967 | **2,147,145** |
| structural tokens | — | **393** |
| **total vocabulary** | 1,236,563 | **2,558,363** |
| v0.1 ids continued / deprecated | — | 1,236,563 / **0** |
| senses with members from ≥ 2 sources | 101,541 | 211,739 |
| senses per lemma: mean / max | 1.48 / 178 | 1.19 / 178 ("draw") |
| lemmas with exactly one sense | — | 92.3% |

### By source

| Source | senses (after normalisation) | joined an existing sense | new senses | canonical senses citing it |
|---|---|---|---|---|
| OEWN | 212,659 | (stage 1) | — | 212,659 |
| Wiktionary | 1,126,537 | (stage 1) | — | 1,125,445 |
| Wikidata lexemes | 59,747 | 51,111 (2,695 by Q-id) | 8,636 | 59,747 |
| MeSH | 351,984 | 20,459 | 331,525 | 351,984 |
| Gene Ontology | 38,092 | 643 | 37,449 | 38,092 |
| ChEBI | 62,119 | 10,420 | 51,699 | 62,119 |
| NCBI Taxonomy | 957,339 | 65,241 (658 refused by the kingdom rule) | 892,098 | 957,339 |
| Wikidata items | — (D-045) | | | |

### By domain (925 domains; top 10)

| domain | senses | | domain | senses |
|---|---|---|---|---|
| (general) | 991,053 | | organic-chemistry | 15,755 |
| taxonomy | 963,912 | | pharmacology | 11,980 |
| chemistry | 302,829 | | biology | 11,210 |
| molecular-biology | 37,563 | | anatomy | 8,749 |
| medicine | 22,310 | | zoology | 7,423 |

Sense-index tiers (order of new ids within a lemma): WordNet 212,659 · Wiktionary 941,179 ·
rare/obsolete Wiktionary 82,725 · lexeme-only 8,636 · concept/domain 1,312,771. By POS: PROPN
1,211,613 (taxa are proper names), NOUN 983,337, ADJ 204,487, VERB 104,001, ADV 33,634.

### ID space (22 + 10-bit layout)

| | value |
|---|---|
| lemma ids issued / lexical lemma slots | 2,147,145 / 4,190,207 = **51.2%** (v0.1: 20.0%) |
| token ids issued / lexical id space | 2,557,970 / 4,290,771,968 = 0.06% |
| max sense index / capacity | 177 / 1,024 ("draw"); no lemma above half the capacity |
| structural tokens / reserved ids | 393 / 4,194,304 (9 of 4,096 reserved slots in use) |

The binding constraint is lemma slots, not token ids (decision 4).

### Storage

| artefact | size |
|---|---|
| raw downloads (`data/raw`; the 157 GB Wikidata dump is streamed, never stored) | 2.48 GB |
| interim tables (`data/interim`) + embedding cache | 2.01 GB + 0.73 GB |
| inventory v1.0 (`data/processed/inventory_v1_0.parquet`, all glosses, examples, members) | 254 MB |
| full id registry, local (with glosses and member lists) | 159 MB |
| **committed ledger** (`registry/`: 3 id shards of 14–17 MB + 19.7 MB lemma ledger + structural) | **68 MB** |

### Embedding parameters (spec 2.3)

V = 2,558,363 tokens, L = 2,147,154 lemma rows (incl. 9 structural slots), 1,024 sense indices,
926 domain rows; memory at bf16. d = 768 is the ~125M-parameter core model planned for Phase 5.

| scheme | d = 768 | d = 1024 |
|---|---|---|
| (a) naive table V × d | 1,965 M (3.9 GB) | 2,620 M (5.2 GB) |
| (b) factorised E_lemma + E_sense[sense_index] + E_domain (spec 5.4) | 1,650 M (3.3 GB), 0.84× | 2,201 M (4.4 GB), 0.84× |
| (b′) low-rank V × r + r × d, r = 64 | 164 M (0.33 GB), 0.08× | 164 M (0.33 GB), 0.06× |
| (b′) low-rank, r = 128 | 328 M (0.66 GB), 0.17× | 328 M (0.66 GB), 0.13× |
| hybrid: dense rows for the 100k most frequent senses + rank-64 tail | 234 M (0.47 GB), 0.12× | 260 M (0.52 GB), 0.10× |

**The factorised scheme of spec 5.4 saves only 16%**: 92% of lemmas have exactly one sense
(taxa, chemicals, multi-word terms), so E_lemma is nearly as large as a per-token table. Even
the naive table at d = 768 would be ~16× the 125M-parameter core. What shrinks it is low rank or
a dense head with a low-rank tail; I recommend Phase 5 compare those two (5.4 asks for the
factorised-vs-low-rank comparison anyway) and initialise the low-rank codes from the sense
glosses' embeddings, so that senses never seen in training still start from a meaningful vector.
The same applies to the output side: the "predict the lemma first" head of 5.4 is a softmax over
2.1M lemmas and needs an adaptive or low-rank output layer too. Full table:
[`phase2_stats.md`](phase2_stats.md).

### Licence subsets

Every sense lists its members' sources and licences. Dropping Wiktionary (share-alike) keeps
1,569,663 senses (61%); dropping Wiktionary and OEWN keeps 1,398,997 (55%); CC0 / public-domain
sources alone (Wikidata, NCBI Taxonomy) keep 1,017,040 (40%). No restricted (SNOMED) content is
in v1.0.

## 3. Problems hit and how they were resolved

1. **The Wikidata dump cannot be stored** (156.7 GB gzip, ~1.6 TB JSON, ≈20 GB free disk). It is
   streamed once over two polite range connections into a filtered superset (D-032). A first
   streaming design reached 1.4 MB/s because a filter thread starved the network reader of the
   GIL; moving filtering into worker processes gave 9.9 MB/s (the per-connection cap).
2. **The first full pass died at 12%** on an `http.client.IncompleteRead` that the retry clause
   did not cover; every transport error is now retried with back-off. The pass cannot resume
   (one deflate stream; nothing to checkpoint), so the crash cost ~30 minutes plus a restart.
3. **HTTP 403** from Wikimedia and geneontology.org for Python's default User-Agent: the
   downloader now sends a descriptive UA with the repository URL only (D-031).
4. **Wikidata API rate limit (429)** when looking up labels of grammatical-feature Q-ids: the
   meanings were read off the data instead (D-033).
5. **NCBI taxa attached to the wrong senses** in the first gold evaluation (precision 0.84):
   genus names matched common nouns case-insensitively ("Mus" ↔ "mus") and insect genera
   matched plant senses. Fixed with no case fallback for taxa and a kingdom ↔ WordNet
   lexicographer-file rule (precision 1.00, D-037).
6. **MeSH lemmas** were Title Case, plural and inverted, and the first dictionary-form rules
   garbled Latin (lemminflect's out-of-vocabulary rules: "Xenarthra" → "xenarthrum") and
   lower-cased genera. The rules now use lemminflect's dictionary, singularise only regular
   English plurals, keep scientific names verbatim and detect genera with the NCBI name list
   (D-035, 22 test cases); ~103k MeSH lemmas changed relative to the first version.
7. **ChEBI definitions carried HTML** (`<em>`, `<sub>`, `<small>D</small>`; 28,736 of them): stripped
   in names, definitions and synonyms (D-040).
8. **Out of memory**: the first stage-2 implementation kept one Python dict per sense and loaded
   the whole embedding cache as float32; it was OOM-killed at 12.7 GB (16 GB machine, the stream
   running alongside). Rewritten to keep columns in Polars and load only the cached vectors a
   query needs: peak 5.9 GB. The id assignment had the same problem (12.3 GB) and now peaks at
   5.6 GB, with an identical registry.
9. **GitHub's 100 MB file limit** would eventually block the ID ledger: ledger tables above
   45 MB are now sharded (D-043).
10. **SNOMED CT must never reach git**: hashed ledger entries, rehydration on rebuild and hold-out
    when absent were implemented and tested before the data exists (D-036).
11. **The cloud container was recycled while the session was idle (~16:25 UTC)**, killing the
    Wikidata pass at 74% after 3 h 15 min. It cannot resume, so v1.0 is built without Wikidata
    items rather than holding the gate for another ~4.5 h pass (D-045). Everything for items is
    ready (ingest, concept filter, threshold); see decision 2 in section 4.
12. **The local LLM judge could not run here**: Remote Control from a cloud session cannot use a
    Windows folder, so the judge now runs directly on your PC (only polars, pyarrow and pyyaml
    needed; it previously pulled in NumPy through the alignment code). Its pilots did not pass
    the precision bar (decision 1), so the 56,565 uncertain pairs stay unmerged, the fine
    policy's conservative default.

## 4. Open decisions that need you

1. **Local LLM judge (56,565 uncertain OEWN–Wiktionary pairs).** It runs directly on your PC
   (Remote Control cannot use Windows folders). Two pilots on the 300 gold pairs, both with a 4k
   context window (Ollama's 32k window pushed the 14B 65% onto the CPU):

   | model | 3-way accuracy | in-band precision / recall | speed (full run) |
   |---|---|---|---|
   | qwen2.5 7B instruct | 0.82 | **0.82** / 0.56 (9 of 11 merges right) | 1.3 req/s (≈ 12 h) |
   | qwen2.5 14B instruct | 0.75 | **0.64** / 0.56 (9 of 14 merges right) | 0.26 req/s (≈ 60 h) |

   Neither reaches the ~0.9 bar (the aligner's own precision), so no verdicts are used and the
   uncertain pairs stay unmerged, as in v1.0 (D-030). *Recommendation:* close this for now; a
   missed merge only leaves a near-duplicate token. Revisit if Phase 3/4 evaluation shows those
   near-duplicates hurting, with a stronger model on hardware that runs it at ≥ ~1 req/s, judged
   by the same pilot.
2. **Wikidata items (not in v1.0, D-045).** The ingest, filter and threshold are ready; only the
   ~4.5 h pass over the dump is missing, and it dies if this cloud session sits idle.
   *Recommendation:* run it in the background at the start of Phase 3, while the session is busy
   with codec work anyway, and add the items as v1.1 (append-only; v1.0 ids unaffected). Items
   matter for Phase 4 tagging coverage, not for the Phase 3 codec.
3. **SNOMED CT UK.** The ingest and the redaction path are ready; I need the RF2 release zip(s)
   from NHS TRUD in `data/raw/snomed/`. Because SNOMED cannot be redistributed, any model trained
   on SNOMED-tagged text inherits the restriction. *Recommendation:* add it as a local,
   restricted extension (ids issued, ledger redacted) but keep SNOMED senses out of training data
   for any model you might release; or skip it until there is a clinical use case.
4. **Lemma-id budget.** The approved 22 + 10 layout has 4,190,207 lexical lemma slots; v1.0 uses
   51% of them (2,147,145; v0.1 used 20%), because every taxon, chemical and multi-word term is
   its own lemma (92% of lemmas have exactly one sense). SNOMED would add ~0.3M, the Wikidata
   items a few hundred thousand. *Recommendation:* keep the layout (changing it later would break id permanence) and treat lemma slots as the scarce
   resource: no bulk entity imports (people, places, works stay byte-spelled), and new domain
   sources only with a use case.
5. **Long-tail domain senses.** NCBI taxa (957k) and MeSH chemical records (249k) are most of the
   growth and are rare in general text. *Recommendation:* keep their ids (cheap, permanent) and
   decide per training run which senses get their own embedding row (e.g. those seen in the
   tagged corpus); a low-rank or hybrid embedding (section 2) keeps unseen senses cheap.
6. **Named entities stay out of v1.0** (Wikidata items are filtered to concepts). *Recommendation:*
   confirm; revisit after Phase 3 measures how much real text falls back to bytes.
7. **Ledger growth in git.** The committed ledger is 68 MB (three id shards + the lemma ledger)
   and each build commits a new version. *Recommendation:* fine for now; move older versions to Git LFS or release assets if the
   repository passes ~1 GB.
8. **MeSH terms.** NLM's conditions (acknowledgement, indicate modifications, no implied
   endorsement) are recorded in `LICENSES.md`. *Recommendation:* re-read the current NLM terms
   before any public release of the inventory.

## 5. Estimated time and cost for Phase 3 (codec)

| Work item | What | Estimate |
|---|---|---|
| 3.1 deterministic decoder | sense id → lemma → surface form from morph tokens + forms tables (Wiktionary, Wikidata lexemes (CC0), OEWN irregular forms; lemminflect fallback), case / spacing / punctuation / digit / variant / byte-span tokens; round-trip tests on hand-written sequences | ~0.5 day |
| 3.2 baseline encoder | spaCy tokenise + lemmatise + POS + morphology → UniMorph tokens; longest-match multi-word lemmas (many domain lemmas are multi-word); most-frequent sense per lemma + POS (sense index order); byte fallback | ~1 day |
| 3.3 metrics | surface round-trip (exact match, character edit distance) on a general-text sample; sense accuracy on SemCor and the Raganato et al. unified WSD framework (WordNet sense keys → OEWN → token ids via the Phase-1 provenance); coverage (inventory hits vs byte fallback) | ~1 day |
| 3.4 report | `docs/phase3_report.md` | ~0.25 day |

* **Compute:** CPU only. spaCy's small/medium English pipelines parse ~5–10k words/s on this
  machine, so a 1M-word evaluation sample takes minutes; the transformer pipeline is not needed
  for the baseline.
* **Cost:** £0 — no paid API is needed in Phase 3.
* **Downloads I will need** (I will stop and give you exact URLs if any is blocked): a spaCy English
  model (MIT), SemCor and the Raganato WSD evaluation framework (WordNet-keyed; its licence will be
  checked and recorded before use — no BabelNet-keyed files are used), and a general-text sample
  for coverage and round-trip (e.g. a few million words of FineWeb-Edu, ODC-By).
* **Wall-clock:** about 2–3 working sessions of agent time, dominated by the encoder's
  multi-word matching and the WordNet-key mapping. If you approve decision 2, the ~4.5 h
  Wikidata items pass runs in the background during that work and v1.1 adds the items.
