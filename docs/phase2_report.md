# Phase 2 report — Sense inventory v1.0 (domain extension) and structural tokens

**Gate 2: waiting for approval.** Branch `claude/senselm-research-system-psq9xl`.
Supporting detail: [`phase2_stats.md`](phase2_stats.md) (all tables), [`phase2_stats.json`](phase2_stats.json)
(every number), [`STRUCTURAL_TOKENS.md`](STRUCTURAL_TOKENS.md), [`DECISIONS.md`](DECISIONS.md)
(D-030 – D-044 are new), [`LICENSES.md`](LICENSES.md),
[`../eval/attach_gold_eval.json`](../eval/attach_gold_eval.json) (stage-2 gold set).

## 1. What was built

### 1.1 Sources (spec 2.1)

| Source | Version | Licence | Ingest | Senses ingested | What is kept |
|---|---|---|---|---|---|
| Wikidata lexemes | dump 2026-10-28 | CC0 | `ingest.wikidata_lexemes` | TBD_LEX | every English lexeme sense with a gloss; forms with grammatical features feed the forms table |
| Wikidata items | JSON dump 2026-10-26 (156.7 GB, streamed, never stored) | CC0 | `ingest.wikidata_items stream` + `select` | TBD_ITEMS | items linked from English lexeme senses + *concepts*: classes (P279) with an English description and Wikipedia article, minus people, works, Wikimedia pages and the types the domain sources own (D-038) |
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
members (the stage-1 embedding model and cache), cross-references (Wikidata Q-ids, P685 / P486 /
P683 / P686, also bridged through the items superset) beat any cosine, greedy one-to-one per
source; a sense that does not attach becomes a new cluster. "mouse" the MeSH descriptor
(*Mice*), the Wikidata item and the lexeme sense join the existing animal sense; *Mus*
(NCBI genus) joins the Wiktionary genus sense through Wikidata; a taxon can never join a cluster
whose WordNet lexicographer file contradicts its kingdom.

Per-source thresholds come from a blind gold set of 240 new-source senses (40 per source,
labelled before scores were looked at; `eval/attach_gold.csv`):

TBD_GOLD

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
TBD_COMMANDS

Tests: `uv run pytest` — 74 tests (73 pass, the local registry-history test skips until a second local registry version exists). New in Phase 2: stage-2 attachment rules (cosine,
cross-reference, Wikidata bridge, one-to-one per source, case and kingdom rules), MeSH lemma
rules (27 cases), structural registry (layout, append-only), SNOMED RF2 reader on fixtures,
restricted-source redaction / rehydration / hold-out, sharded ledger, config inheritance, and
the committed ledger read from HEAD in either layout.

## 2. Key numbers

TBD_NUMBERS

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
   (D-035, 29 test cases); ~103k MeSH lemmas changed relative to the first version.
7. **ChEBI definitions carried HTML** (`<em>`, `<sub>`, `<small>D</small>`; 28,736 of them): stripped
   in names, definitions and synonyms (D-040).
8. **Out of memory**: the first stage-2 implementation kept one Python dict per sense and loaded
   the whole embedding cache as float32; it was OOM-killed at 12.7 GB (16 GB machine, the stream
   running alongside). Rewritten to keep columns in Polars and load only the cached vectors a
   query needs: peak TBD_ATTACH_RSS.
9. **GitHub's 100 MB file limit** would eventually block the ID ledger: ledger tables above
   45 MB are now sharded (D-043).
10. **SNOMED CT must never reach git**: hashed ledger entries, rehydration on rebuild and hold-out
    when absent were implemented and tested before the data exists (D-036).
11. **The local LLM judge could not run**: no computer of yours was connected to this session
    (Remote Control), so the 56,565 uncertain OEWN–Wiktionary pairs are still unjudged; the fine
    policy's default for them is "keep separate" (no merge), which is the conservative choice.

## 4. Open decisions that need you

1. **Run the local LLM judge (uncertain OEWN–Wiktionary pairs).** Nothing of yours was connected
   to this session, so the band is unjudged. To run it here, start a session on your machine
   (`claude remote-control` in a clone of this repository, or the Claude desktop app's Remote
   Control) with an OpenAI-compatible server running (Ollama, vLLM, llama.cpp, LM Studio), then
   tell me the model name: I run `probe`, the 300-pair `pilot`, and only if the pilot beats the
   cosine baseline the 56,565-pair `run` (resumable), commit the verdicts and rebuild. A merge
   found later deprecates one of the two ids (`replaced_by`), so ids stay valid.
   *Recommendation:* do it before Phase 4 tagging, while no corpus refers to the ids yet; it does
   not block Phase 3. A 14B–32B instruction model (e.g. a Qwen2.5 or Llama-3.x instruct) on a GPU
   should finish in well under a day; accept it only on pilot accuracy.
2. **SNOMED CT UK.** The ingest and the redaction path are ready; I need the RF2 release zip(s)
   from NHS TRUD in `data/raw/snomed/`. Because SNOMED cannot be redistributed, any model trained
   on SNOMED-tagged text inherits the restriction. *Recommendation:* add it as a local,
   restricted extension (ids issued, ledger redacted) but keep SNOMED senses out of training data
   for any model you might release; or skip it until there is a clinical use case.
3. **Lemma-id budget.** The approved 22 + 10 layout has 4,190,207 lexical lemma slots; v1.0 uses
   TBD_LEMMA_PCT of them, because every taxon, chemical and multi-word term is its own lemma
   (TBD_ONE_SENSE of lemmas have a single sense). SNOMED would add ~0.3M. *Recommendation:* keep
   the layout (changing it later would break id permanence) and treat lemma slots as the scarce
   resource: no bulk entity imports (people, places, works stay byte-spelled), and new domain
   sources only with a use case.
4. **Long-tail domain senses.** NCBI taxa (957k) and MeSH chemical records (249k) are most of the
   growth and are rare in general text. *Recommendation:* keep their ids (cheap, permanent) and
   decide per training run which senses get their own embedding row (e.g. those seen in the
   tagged corpus); the factorised embedding below makes unseen senses cheap anyway.
5. **Named entities stay out of v1.0** (Wikidata items are filtered to concepts). *Recommendation:*
   confirm; revisit after Phase 3 measures how much real text falls back to bytes.
6. **Ledger growth in git.** The committed ledger is TBD_LEDGER_MB per version and grows with each
   build. *Recommendation:* fine for now; move older versions to Git LFS or release assets if the
   repository passes ~1 GB.
7. **MeSH terms.** NLM's conditions (acknowledgement, indicate modifications, no implied
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
  multi-word matching and the WordNet-key mapping.
