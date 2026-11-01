# Phase 1 report — Sense inventory v0.1 (Wiktionary + Open English WordNet)

**Gate 1: approved 2026-11-01** — 22 + 10-bit ID layout and the fine policy approved; the
uncertain band is to be judged by a locally hosted LLM on the owner's server (not a paid API).
Branch `claude/senselm-research-system-psq9xl`.
Supporting detail: [`phase1_stats.md`](phase1_stats.md) (all tables), [`phase1_stats.json`](phase1_stats.json)
(every number), [`DECISIONS.md`](DECISIONS.md), [`GRANULARITY.md`](GRANULARITY.md),
[`LICENSES.md`](LICENSES.md), [`../eval/alignment_eval.md`](../eval/alignment_eval.md),
[`../eval/embedding_selection.md`](../eval/embedding_selection.md).

## 0. Hardware (detected at start)

| Resource | Detected | Consequence |
|---|---|---|
| CPU | 4 vCPU Intel Xeon (Sapphire Rapids), AVX-512, **AMX-BF16** | bf16 CPU inference is 2.6–3.4× faster than fp32 at identical quality |
| RAM | 16 GB, no swap | all stages stream or batch; the alignment stage peaked at ~2.8 GB RSS |
| Disk | ~30 GB writable (24 GB free now) | fine for Phase 1–2 except the full Wikidata dump (157 GB) |
| GPU | **none** | embedding model chosen for CPU throughput; CPU-only PyTorch wheels |

`python -m senselm.hardware` regenerates [`HARDWARE.md`](HARDWARE.md); `auto` config values are sized from it.

## 1. What was built

A reproducible pipeline from raw dumps to a sense inventory with permanent token IDs. Every
stage is `python -m senselm.<stage> --config configs/<stage>.yaml`, deterministic given config
and inputs (downloads are sha256-pinned; embeddings are cached content-addressed), and embeds a
provenance record (config + hash, input sha256s, source versions, git commit, package versions)
in its Parquet metadata.

| Step | Command | Wall time (4 vCPU) | Output |
|---|---|---|---|
| download | `python -m senselm.ingest.download --config configs/download.yaml` | ~1 min | `data/raw/{wiktionary,oewn}/`, `MANIFEST.json` |
| Wiktionary ingest | `python -m senselm.ingest.wiktionary --config configs/ingest_wiktionary.yaml` | 4 min (2 streaming passes) | `senses_raw_wiktionary.parquet` |
| OEWN ingest | `python -m senselm.ingest.oewn --config configs/ingest_oewn.yaml` | 30 s | `senses_raw_oewn.parquet` |
| combine | `python -m senselm.ingest.combine --config configs/ingest.yaml` | 15 s | `data/interim/senses_raw.parquet` |
| normalise | `python -m senselm.inventory.normalise --config configs/normalise.yaml` | 1.5 min | `senses_norm`, `forms`, `variants`, `dedupe_map` |
| align | `python -m senselm.inventory.align --config configs/align.yaml` | 19 min | `align_pairs.parquet` (656,079 scored pairs) |
| judge (dry run) | `python -m senselm.inventory.judge --config configs/align.yaml` | 11 s | cost estimate (no API calls) |
| evaluate | `python -m senselm.eval.alignment_eval --config configs/alignment_eval.yaml` | 5 s | `eval/alignment_eval.{json,md}` |
| build (fine + IDs) | `python -m senselm.inventory.build --config configs/inventory.yaml` | 2 min | `data/processed/inventory_v0_1.parquet`, `id_registry.parquet`, `lemma_registry.parquet`, `registry/` ledger |
| build (coarse) | `python -m senselm.inventory.build --config configs/inventory_coarse.yaml` | 1 min | `inventory_v0_1_coarse.parquet` (counts only) |
| stats | `python -m senselm.inventory.stats --config configs/stats.yaml` | 5 s | `docs/phase1_stats.{json,md}` |

End to end ≈ 30 minutes on this machine. One-off experiments are also commands:
`python -m senselm.eval.alignment_gold sample …` (blind gold sample) and
`python -m senselm.eval.embedding_selection --config configs/embedding_selection.yaml`.
Tests: `uv run pytest` — 39 tests (ingest, normalisation, policy, judge cost/cap logic, ID
layout, registry permanence across simulated builds, end-to-end build on fixtures, rebuild from
the committed ledger after data loss, and the committed registry vs HEAD). The local
registry-history test skips until a second registry version exists.

**Sources.** Wiktionary: kaikki.org English extraction of 2026-10-25 (enwiktionary dump
2026-10-01, wiktextract f9fe161 + 6f96ddf). Open English WordNet **2025+** (2025-12-31; the
2025 lexical edition plus curated proper nouns from Open English Namenet, which are tagged
`oewn:namenet`). Licences and obligations: [`LICENSES.md`](LICENSES.md). No OED or BabelNet data
is used anywhere.

**Inventory record** (`inventory_v0_1.parquet`, one row per canonical sense): `token_id`,
`lemma_id`, `sense_index`, `lemma`, `upos`, `gloss` (+ all member `glosses`), `examples`, `tags`,
`domain`, `lexfile` (WordNet supersense), `synset_id`, `hypernyms`, `members` (every contributing
`(source, source_id)`), `sources`, `licences`, `is_abbrev`, … Example:

| token_id | lemma_id | idx | POS | sources | gloss | members |
|---|---|---|---|---|---|---|
| 0x0002B000 | 172 | 0 | NOUN | oewn+wiktionary | sloping land (especially the slope beside a body of water) | `oewn-bank__1.17.01..`, `en-bank-en-noun-en:slope` |
| 0x0002B001 | 172 | 1 | VERB | oewn+wiktionary | tip laterally | `oewn-bank__2.38.00..`, `en-bank-en-verb-6BSwyQvb` |
| 0x0002B002 | 172 | 2 | NOUN | oewn+wiktionary | a financial institution that accepts deposits … | `oewn-bank__1.14.00..`, `en-bank-en-noun-en:Q22687` |
| 0x0002B004 | 172 | 4 | NOUN | oewn | a long ridge or pile | `oewn-bank__1.17.00..` |

Case-distinct lemmas stay separate: *Polish* is lemma 42983 (0x029F9C00 = "the Slavic language
of Poland"), *polish* is lemma 8367 (0x0082BC00 = "make (a surface) shine").

## 2. Key numbers

### 2.1 Headline

| | |
|---|---|
| **Canonical senses (fine policy, v0.1, permanent IDs)** | **1,236,563** |
| Lemmas | 837,967 |
| Senses with both sources / OEWN only / Wiktionary only | 101,541 / 111,118 / 1,023,904 |
| Senses with ≥ 1 example sentence | 387,306 (**31.3%**) |
| Size under the coarse policy | 1,140,514 (−7.8%) |
| Upper bound (no cross-source merging) | 1,339,196 |
| Aligner precision / recall, fine (test half; all 300 gold pairs) | **0.927 / 0.773**; 0.895 / 0.791 |
| Aligner precision / recall, coarse (test half; all 300) | **0.923 / 0.900**; 0.894 / 0.894 |
| Inflection-table rows (`forms.parquet`) / orthographic variants (`variants.parquet`) | 1,364,798 / 276,957 |
| Max senses on one lemma | 178 (*draw*) |

### 2.2 Per-source counts and overlap

| | Wiktionary | OEWN 2025+ |
|---|---|---|
| Dictionary entries | 1,499,161 | 161,875 |
| Raw sense records ingested | 1,795,048 | 212,659 |
| … form-of ("plural of X") → `forms.parquet`, not senses | 539,341 | – |
| … orthographic variants (alt-of 126,917 + untagged "spelling of" 819) → `variants.parquet` | 127,736 | – |
| … abbreviation-type alt-of kept as senses (NASA, ad, 'cause) | 45,346 | – |
| … gloss-less, dropped | 1,335 | 0 |
| … exact duplicates folded into the kept sense | 99 | 0 |
| **Senses entering alignment** | **1,126,537** | **212,659** (27,530 Namenet) |
| … with ≥ 1 cross-source candidate (same lemma + POS class) | 253,914 | 148,078 |
| … **merged with a sense of the other source** (fine) | **102,641 (9.1%)** | **101,541 (47.7%; 68.6% of those with candidates)** |

656,079 candidate pairs in 99,478 lemma groups (4,672 via the case-insensitive fallback, e.g.
OEWN *christly* ↔ Wiktionary *Christly*). 64,581 OEWN senses (mostly Namenet proper nouns and
taxonomic names) have no Wiktionary counterpart at all; Wiktionary's coverage is ~5× larger.

### 2.3 Granularity: fine vs coarse

| Policy | Rule (docs/GRANULARITY.md) | Canonical senses |
|---|---|---|
| no merging | every source sense is a token | 1,339,196 |
| **fine (default, issues IDs)** | merge paraphrases only, one-to-one (+ lineage rule F5); subsenses kept | **1,236,563** |
| coarse (counts only) | subsenses folded into parents; Wiktionary senses join their best WordNet sense if *same or related* | 1,140,514 |

The difference is modest because most of the inventory is Wiktionary-only (no partner to merge
with); within the 99,478 shared lemma groups the coarse policy merges 180,076 Wiktionary units
into WordNet senses vs 102,633 under fine.

### 2.4 Alignment precision / recall

Gold set: 300 candidate pairs hand-labelled blind to the scores ([`eval/alignment_gold.csv`](../eval/alignment_gold.csv)):
129 `same`, 31 `related`, 140 `different`, 34 flagged ambiguous; two strata (150 uniform over
pairs, 150 uniform over lemma groups) with inclusion probabilities recorded. A pair counts as
merged if both senses land in the same canonical sense. Fine positives = `same`; coarse
positives = `same` + `related`. Thresholds were tuned on the dev half only.

| Policy (threshold) | test half P / R / F1 | all 300 (sample) | all 300, population-weighted (HT) | unambiguous only | stratum U (uniform pairs) |
|---|---|---|---|---|---|
| fine (cos ≥ 0.69; tuned for F0.5) | **0.927 / 0.773 / 0.843** | 0.895 / 0.791 / 0.840 | 0.861 / 0.795 / 0.827 | 0.941 / 0.814 / 0.873 | 0.812 / 0.743 / 0.776 |
| coarse (cos ≥ 0.59; tuned for F1) | **0.923 / 0.900 / 0.911** | 0.894 / 0.894 / 0.894 | 0.855 / 0.844 / 0.849 | 0.897 / 0.931 / 0.914 | 0.872 / 0.872 / 0.872 |

Error profile (fine, all 300): of 12 false merges only **3 are genuinely different meanings**
(*doughy*, *prejudice*, *underexpose*); the other 9 are `related` pairs (domain-restricted or
broader/narrower) that the coarse policy would merge anyway. Of 27 missed merges, 16 have
cosine 0.55–0.69 (the LLM-judge band), 9 are still blocked by one-to-one matching (another
near-duplicate sense of the same lemma took the match) and 2 score below 0.55.

Model and text were selected on the gold pairs ([`embedding_selection.md`](../eval/embedding_selection.md)):

| Variant (CPU, 4 threads) | texts/s | fine AUC | coarse AUC |
|---|---|---|---|
| bge-small / gloss + example | 306 | 0.873 | 0.904 |
| bge-base / gloss + example | 94 | 0.902 | 0.936 |
| bge-base / lemma + gloss + example | 94 | 0.856 | 0.898 |
| bge-base / gloss | 94 | 0.920 | 0.935 |
| **bge-base / parent gloss + gloss, bf16 (chosen)** | **247** | **0.924** | **0.935** |
| bge-large / gloss, bf16 | 88 | 0.928 | 0.937 |
| e5-large-v2 / gloss | 25 | 0.924 | 0.928 |
| all-mpnet-base-v2 / gloss | 92 | 0.918 | 0.918 |

The large models are within noise (SE ≈ 0.015 at n = 300) at 3–10× the CPU cost.

### 2.5 Senses per lemma

Mean 1.48, median 1, p99 9, max 178.

| senses per lemma | 1 | 2 | 3–5 | 6–10 | 11–20 | 21–50 | 51–100 | >100 |
|---|---|---|---|---|---|---|---|---|
| lemmas | 679,307 | 93,616 | 47,205 | 11,744 | 4,466 | 1,480 | 136 | 13 |
| % | 81.07 | 11.17 | 5.63 | 1.40 | 0.53 | 0.18 | 0.02 | 0.00 |

Top 50 most polysemous lemmas (senses; WordNet-backed in brackets): draw 178 [45], **Ivanivka 157
[0]**, break 155 [75], run 135 [57], cut 123 [70], **Midway 119 [1]**, set 116 [44], take 109 [44],
**Mykolaivka 108 [0]**, AA 104 [2], **Hopewell 103 [0]**, drop 103 [32], **Riverside 101 [1]**, go 100
[35], stick 96 [25], point 95 [40], flat 92 [24], head 92 [42], close 90 [37], line 90 [36], down 89
[25], open 89 [35], make 88 [51], **Pavlivka 86 [0]**, pass 86 [42], strike 85 [29], **Hamilton 83 [6]**,
light 80 [47], turn 80 [38], **Santa Cruz 79 [2]**, roll 79 [33], call 77 [41], lead 77 [31], catch 76
[39], play 76 [52], mark 75 [28], A. 74 [0], hold 74 [45], touch 74 [27], AAA 73 [1], double 73 [21],
round 73 [25], **Antonivka 72 [0]**, clear 72 [45], hack 72 [16], on 72 [5], snap 72 [26], flutter 71
[9], up 71 [14], back 70 [28]. (Per-POS breakdown for each: [`phase1_stats.md`](phase1_stats.md).)
Bold entries are place names with one sense per referent (157 villages called Ivanivka) — see
open decision 5.

### 2.6 POS breakdown

| UPOS | senses | both sources | OEWN only | Wiktionary only |
|---|---|---|---|---|
| NOUN | 622,902 | 61,547 | 61,726 | 499,629 |
| PROPN | 253,548 | 4,721 | 23,738 | 225,089 |
| ADJ | 203,658 | 17,772 | 12,407 | 173,479 |
| VERB | 102,979 | 13,597 | 11,550 | 77,832 |
| ADV | 33,474 | 3,904 | 1,697 | 27,873 |
| X (phrases, proverbs, affixes, contractions) | 11,559 | – | – | 11,559 |
| INTJ / ADP / PRON / SYM / NUM / SCONJ / DET / PART / PUNCT / CCONJ | 4,165 / 1,253 / 795 / 794 / 411 / 383 / 353 / 153 / 72 / 64 | – | – | all |

### 2.7 Tags and domains

617,936 senses (50%) carry at least one tag; 238,245 (19%) have a domain.

| Register / time | senses | | Region | senses |
|---|---|---|---|---|
| obsolete | 28,533 | | US | 16,564 |
| slang | 28,165 | | UK | 14,293 |
| informal | 18,870 | | Australia | 4,236 |
| historical | 17,009 | | Canada | 3,524 |
| rare | 16,730 | | Scotland | 3,304 |
| archaic | 15,155 | | Ireland | 2,630 |
| dated | 10,067 | | India | 2,160 |
| derogatory | 9,099 | | New Zealand | 1,789 |
| colloquial | 6,982 | | South Africa | 1,007 |
| vulgar / humorous / dialectal / offensive / nonstandard | 4,864 / 4,410 / 4,659 / 2,081 / 1,912 | | | |

Top domains: organic chemistry 15.8k, medicine 9.2k, anatomy 7.6k, zoology 7.4k, biochemistry
7.3k, computing 7.0k, chemistry 6.4k, law 6.4k, biology 5.9k, botany 5.9k, music 5.7k,
mineralogy 5.5k, mathematics 5.3k, physics 5.2k. Top WordNet supersenses of WordNet-backed
senses: adj.all 25.4k, noun.person 22.2k, noun.artifact 19.7k, noun.plant 18.8k, noun.animal
14.9k, noun.act 11.3k, noun.communication 10.4k. Full tables: [`phase1_stats.md`](phase1_stats.md).

### 2.8 Example sentences

387,306 of 1,236,563 senses (**31.3%**) have at least one example or quotation: 65.3% of
senses attested in both sources, 23.6% of OEWN-only senses, 28.8% of Wiktionary-only senses.
This is the main input gap for Phase 4 (rare-sense synthetic data).

### 2.9 IDs and registry

| | |
|---|---|
| Layout | `token_id = lemma_id << 10 | sense_index` (uint32), 22 + 10 bits — see decision 1 |
| Lemma ids used | 837,967 of 4,190,207 lexical slots (20.0%) — already 80% of a 20-bit field |
| Max sense index | 177 (*draw*) of 1,023 (17%) |
| Structural range (Phase 2) | `0xFFC00000`–`0xFFFFFFFF` (4,194,304 ids); lemma id 0 never used |
| Active ids / tombstones | 1,236,563 / 0 |
| Registry | `data/processed/id_registry.parquet` 79 MB (with glosses + member lists); git-tracked ledger `registry/` 25.4 MB + 7.1 MB lemma registry |
| Sense order within a lemma | WordNet-attested senses by WordNet sense number (POS interleaved, dominant POS first), then Wiktionary-only senses in Wiktionary order, rare/obsolete last |

Permanence is enforced three ways: the build refuses to write a registry that changes any
existing id; `test_committed_registry_is_append_only` diffs the working-tree ledger against
the one committed at HEAD; and a simulated multi-build test covers additions, removals,
merges (tombstone + `replaced_by`), splits and a returning sense (new id, never a revived one).

### 2.10 LLM judge (middle band)

The API is not enabled here (no project key), so middle-band pairs are kept separate. Band for
the fine policy: 0.55 < cosine < 0.69 → 244,284 pairs, of which **56,292** can actually change a
decision (both senses still unmatched; top-1 candidate of one of them). Dry-run estimate with the
Batch API (−50%), heuristic token counts (3 chars/token, conservative):

| Model | expected | worst case (every request hits max_tokens) |
|---|---|---|
| claude-opus-5-5 (configured default) | £223 | £996 |
| claude-sonnet-5-5 | £112 | £501 |
| claude-haiku-5-5 | £5.60 | £25 |

The hard cap (£50/run, worst-case accounting per 2,000-request chunk) would stop an Opus run
after ~4,000 requests. A pilot on the 300 gold pairs costs ≈ £1.20 with Opus and measures the
judge's accuracy directly — see decision 4.

### 2.11 Storage

`data/raw` 533 MB, `data/interim` 1.2 GB (incl. 0.5 GB embedding cache), `data/processed`
413 MB (inventory 180 MB, coarse inventory 163 MB, registry 79 MB, lemma registry 10 MB).

## 3. Problems hit and how they were resolved

1. **OEWN GitHub release not reachable.** The `globalwordnet/english-wordnet` repository is
   outside this session's GitHub scope and the request to add it was declined. The `wn`
   package's own index (v1.1.1) points `oewn:2025+` at `en-word.net/static/…`, the project's
   official download site, so I used that URL: byte-identical to the `/downloads/` copy, sha256
   pinned. If you want the GitHub asset verified as well: download the 2025 release asset
   `english-wordnet-2025-plus.xml.gz` from https://github.com/globalwordnet/english-wordnet/releases
   to `data/raw/oewn/`; the download stage will check it against the pinned hash.
2. **wiktextract quirks.** `etymology_number` is sometimes a string (coerced). 890 sense ids
   collide (subsenses inheriting a parent's `senseid`); my first suffix scheme used `~`, which
   turned out to be part of wiktextract's id alphabet — replaced by `#` + a hash of the gloss
   path (stable under reordering). Topic lists include auto-added ancestors (`sciences`), so
   `domain` = the least frequent topic.
3. **Dropping all alt-of senses would have deleted real words.** 45,346 abbreviation-type
   senses (initialisms, acronyms, clippings: *NASA*) are kept; only spelling variants go to the
   variants table. 819 untagged variants ("Dated spelling of jail.") are caught by a pattern.
4. **CPU-only embeddings.** bge-large / e5-large would take 1.3–4.5 h for 345k texts. bf16
   autocast (AMX) gave identical gold-set scores at 2.6× speed; bge-base with parent-gloss text
   matched the large models within noise. The full run took 19 minutes.
5. **Recall is unmeasurable with uniformly sampled pairs** (almost all negatives). Solved with a
   two-stratum sample plus Horvitz–Thompson weights, so both sample and population estimates
   are reported.
6. **Pairwise thresholds are the wrong target.** One-to-one matching removes most false
   positives, so under the full policy the F1-optimal threshold drops from 0.676 (pairwise) to
   0.59. Thresholds are therefore tuned under the full policy; for the ID-issuing fine policy I
   optimised F0.5 instead (→ 0.69) because false merges are irreversible while false splits only
   leave near-duplicate tokens.
7. **One-to-one matching blocked parent/subsense paraphrases** (11 of 29 missed merges). Added
   rule F5 (lineage absorption); a unit test caught my first version merging sibling subsenses,
   fixed before use. Recall +1.6 points, precision unchanged; 9 of the remaining 27 misses are
   still one-to-one blocks between non-lineal senses (left alone by design).
8. **Library drift.** Polars 2.x removed `Expr.flatten`; transformers 5 no longer exposes the
   model commit hash (now read from the local HF snapshot). Both fixed; versions locked in
   `uv.lock`.
9. **Registry too large for git** (79 MB). A lossless-for-IDs ledger (no glosses, only
   non-anchor members) is committed instead (32 MB); the build continues from it if `data/` is
   lost (tested).
10. **ID space.** 837,967 lemmas after Phase 1 alone made the specified ~20-bit lemma field
    untenable for Phase 2 (decision 1).

## 4. Open decisions that need you

1. **ID layout: 22-bit lemma + 10-bit sense (implemented) vs the specified ≈20 + 10.**
   Phase 1 already uses 837,967 lemma ids (80% of 20 bits); NCBI Taxonomy alone adds ~2.6M names
   in Phase 2. Because IDs are permanent, widening later means renumbering everything.
   *Recommendation: approve 22 + 10.* (Sense index: max used 177, so 10 bits has ample headroom;
   8 bits would already be tight.)
2. **Fine policy as the ID-issuing policy (incl. rule F5).** A fine inventory can always be
   mapped to the coarse one; a coarse inventory cannot be split later without new ids.
   *Recommendation: approve fine for IDs; keep coarse as a derived mapping.*
3. **Gold labels were produced by me (Claude) acting as annotator, not by a human.**
   *Recommendation: a human spot-check of the 34 ambiguous pairs + ~50 random ones (~1 hour);
   if disagreement exceeds ~10%, I relabel and re-tune (5 minutes of compute).*
4. **LLM judge for the uncertain band.** Needs an API key in the environment. *Recommendation:*
   (a) pilot on the 300 gold pairs with Opus 5.5 and Haiku 5.5 (≈ £1.30 total) to measure judge
   accuracy against the gold labels; (b) if Haiku 5.5 is at least as accurate as the cosine
   aligner on the band, judge all 56,292 decision-relevant pairs with it (≈ £6, under the cap);
   Opus 5.5 for everything (≈ £223) would need a raised cap.
5. **Proper names.** 253,548 PROPN senses, including ~56k surname senses ("An English
   surname…"), ~12k given names and ~106k place senses — one token per referent (157 villages
   named Ivanivka). That is questionable for a language model: the referent is rarely
   recoverable from text. *Recommendation: keep their IDs (provenance, no cost), but in Phase 3
   route surname/given-name/place mentions through a name class (a generic name token per
   lemma + spelling) and exclude per-referent place senses from tagging targets. Decide before
   Phase 4.*
6. **Registry storage.** The ledger is in git (~32 MB per version; v0.1 is the first). *Recommendation:
   fine for v0.1; before Phase 2 (≈2–3× larger) move it to a release asset or a dataset repo,
   keeping a checksum + the HEAD-diff test in git.*
7. **Archive the raw dumps.** kaikki.org serves only the latest weekly extraction; the pinned
   2026-10-25 file (534 MB) will disappear upstream. I cannot upload outside this environment.
   *Recommendation: copy `data/raw/` to storage you control.*
8. **Share-alike.** 91% of senses (1,125,445) include Wiktionary content (CC BY-SA). Provenance
   allows a CC BY-only subset (111,118 OEWN-only senses). *Recommendation: get advice before
   releasing model weights trained on SA-derived data; nothing to decide for Phase 2.*
9. **Phase 2 — Wikidata items.** The full JSON dump is 157 GB compressed (~1.6 TB raw) vs 24 GB
   free disk. Options: (a) stream-parse straight from dumps.wikimedia.org without storing it
   (6–10 h; reproducible via dump date + sha256 computed while streaming); (b) provision ≥ 2 TB;
   (c) lexemes dump (657 MB) + SPARQL for linked items (not versioned). *Recommendation: (a).*
10. **Phase 2 — SNOMED CT UK** needs your TRUD account. I will build the RF2 ingest against the
    format spec; when ready, put the UK edition release zip under `data/raw/snomed/`.

## 5. Estimate for Phase 2 (domain extension to 2M+, structural tokens)

| Item | Estimate |
|---|---|
| Engineering (agent time) | 1.5–2.5 days: ingests for Wikidata lexemes, Wikidata items, NCBI Taxonomy, GO, ChEBI, MeSH; SNOMED RF2 module (untested until files arrive); domain-term alignment; structural vocabulary + `STRUCTURAL_TOKENS.md`; report |
| Downloads (all reachable, checked 2026-11-01) | Wikidata lexemes 657 MB; NCBI taxdump 80 MB; GO 37 MB; ChEBI OBO 271 MB; MeSH 2026 descriptors 313 MB + supplementary 786 MB; Wikidata items: 157 GB streamed (option 9a) |
| Compute | Wikidata streaming parse 6–10 h; other parsers < 30 min; embedding domain terms that collide with existing lemmas (~0.2–0.5M texts) 15–30 min |
| Disk | ~10–15 GB of interim outputs (fits) |
| Money | £0 (no API use) unless the judge is enabled: ≈ £5–15 with Haiku 5.5 for Phase 1 + 2 bands |
| Expected size | 2–3.5M senses; the taxonomy filter (e.g. species and above, excluding environmental/unclassified samples) is the main lever and will be proposed in the Phase 2 report |
