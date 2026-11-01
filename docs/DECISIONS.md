# Decision log

Every non-obvious engineering decision, with the reason and where it is configured.
IDs are stable; superseded decisions are marked, never deleted.

## Environment and infrastructure

**D-001 Toolchain.** Python 3.12 (`.python-version`; the package supports ≥ 3.11), `uv` with a
committed `uv.lock`. PyTorch comes from the CPU wheel index (`[tool.uv.sources]` in
`pyproject.toml`) because the machine has no GPU and the CUDA wheels cost ~2.5 GB of disk; on a
GPU machine delete that source mapping.

**D-002 Hardware-driven sizing.** `python -m senselm.hardware` reports CPU / RAM / disk / GPU
(docs/HARDWARE.md). Config values set to `auto` (batch sizes, torch threads, Parquet batch rows,
device) are resolved from it at run time. Phase-1 machine: 4 vCPU Xeon (Sapphire Rapids, AVX-512,
AMX-BF16), 16 GB RAM, ~30 GB free disk, no GPU.

**D-003 Provenance.** Every stage writes Parquet with a `senselm` key-value metadata record
(full config, config hash, input sha256s, upstream source versions, git commit, package
versions) and a sidecar `<file>.meta.json` that adds a timestamp and output stats. Timestamps are
kept out of the Parquet so identical inputs give identical files.

**D-004 Downloads are pinned.** `configs/download.yaml` pins each file's sha256; a changed
upstream file fails the run instead of silently changing the build. kaikki.org only serves the
latest weekly extraction, so `data/raw/` must be archived to reproduce v0.1 bit-for-bit later
(see Gate-1 open decisions).

## Sources

**D-005 Wiktionary file.** The per-language English file
`kaikki.org/dictionary/English/kaikki.org-dictionary-English.jsonl.gz` (English-language entries
of English Wiktionary), not the all-language raw wiktextract dump. Verified on kaikki.org on
2026-11-01: extraction of 2026-10-25 from the enwiktionary dump of 2026-10-01.

**D-006 OEWN edition and download route.** OEWN **2025+** (2025 lexical edition + curated proper
nouns from Open English Namenet; "coverage similar to previous OEWN releases"). Proper nouns are
real senses that Wiktionary also has (`name` POS), so excluding them would leave 27,530
Wiktionary-matchable senses without a WordNet counterpart. Namenet senses are tagged
`oewn:namenet` (computed by diffing against the base 2025 edition) so they can be excluded.
The file is fetched from `https://en-word.net/static/…`, which is exactly the URL the `wn`
package (v1.1.1) indexes for `oewn:2025+`; the GitHub release asset is not reachable from this
session (repository not in the session's GitHub scope), and the bytes are the release's.

**D-007 OEWN parsing.** The WN-LMF XML is parsed directly with `lxml.iterparse` (streaming, DTD
loading / entity resolution / network disabled) instead of loading it into `wn`'s SQLite
database: one fewer moving part, deterministic, and it exposes everything we need (sense order,
`Form`s, relation `dc:type`s, usage labels via `exemplifies`). Two passes: synsets first, then
entries; only the extracted synset fields stay in memory (~120k synsets).

## Ingest and normalisation

**D-008 Faithful raw ingest.** `senses_raw.parquet` contains *every* Wiktionary sense, including
form-of, alt-of and gloss-less ones, classified in a `kind` column. All policy decisions (what is
a sense) happen in `normalise`, so they can be changed without re-reading the 3.4 GB dump.

**D-009 Wiktionary sense ids.** `source_id` = wiktextract's sense `id`. 890 ids collide (mostly
subsenses inheriting their parent's `senseid`-derived id). Colliding ids get
`#<blake2b(gloss path)>` appended — content-based, so the suffix survives reordering of
subsenses; `#` never occurs in wiktextract ids (whose alphabet includes `~`, `_`, `:`).

**D-010 Domain.** wiktextract emits topics alphabetised together with auto-added ancestors
(`category-theory` comes with `mathematics`, `sciences` …). `domain` = the topic in the list
that is least frequent dump-wide (the most specific); the full list is kept in `topics`.
OEWN `domain` = name of the `domain_topic` target synset; the supersense is kept in `lexfile`,
usage labels (`exemplifies` → `usage:british_spelling` …) and `domain_region` in `tags`.

**D-011 Lemma normalisation and case.** NFC; whitespace collapsed; typographic apostrophes and
hyphens mapped to ASCII (`configs/normalise.yaml: lemma_char_map`). Case is preserved and lemma
identity is case-sensitive (`Polish` ≠ `polish`). Dictionary headwords carry no sentence-initial
capitals, so no case folding is needed.

**D-012 UPOS mapping** (`configs/normalise.yaml: upos_map`). Non-obvious rows: Wiktionary
`prep_phrase` → ADV ("in vain", "at hand" are adverbials); `phrase`, `proverb`, `contraction`
and affixes → X (original label kept in `pos`); `conj` → CCONJ for a closed list of coordinators,
else SCONJ; `character`/`symbol` → SYM. OEWN nouns → PROPN when the sense is from Namenet or its
synset has an `instance_hypernym` (OEWN has no proper-noun category of its own).

**D-013 What is a sense.**
- `form_of` senses ("plural of X") are not senses; they become rows of `forms.parquet`
  (`form → lemma` with feature tags) for the Phase-3 decoder, together with headword-line
  inflections and OEWN irregular `Form`s.
- `alt_of` senses split two ways. Orthographic variants (alternative/obsolete/archaic spelling,
  misspelling, letter-case form, pronunciation spelling …) go to `variants.parquet`.
  Abbreviation-type alt-of senses (tags initialism, acronym, abbreviation, clipping, ellipsis,
  contraction) **stay senses**: "NASA", "ad", "'cause" are lexical items a tokenizer must emit.
- Untagged senses whose whole gloss is "<qualifiers> spelling/form of X" ("Dated spelling of
  jail.", "UK standard spelling of journaling.") are also variants (819 found).
- Gloss-less senses are dropped (1,335).

**D-014 Exact duplicates.** Senses of one source with identical lemma, UPOS, gloss path, gloss and
tags are data artefacts (99 in Wiktionary): the first is kept, the rest are recorded in
`dedupe_map.parquet` and listed as members of the kept sense's canonical sense.

## Alignment

**D-015 Candidate pairs.** Same lemma and same POS class, where NOUN and PROPN form one class
(OEWN has no PROPN). Case-insensitive matching is used only for OEWN (lemma, class) groups with
*no* exact-case Wiktionary counterpart (OEWN `christly` ↔ Wiktionary `Christly`); such pairs are
flagged `case_fallback`. Result: 656,079 pairs in 99,478 groups.

**D-016 Embedding model and text.** Chosen on the hand-labelled pairs
(`eval/embedding_selection.md`): **BAAI/bge-base-en-v1.5**, CLS pooling, L2-normalised, run under
**bf16 CPU autocast**; sense text = Wiktionary parent gloss (for subsenses, headers skipped) +
gloss; no usage example, no lemma. Evidence: gloss-only beat gloss+example (fine AUC 0.920 vs
0.902) and lemma prefixes hurt (0.856); bf16 matched fp32 exactly at 2.6× the speed (AMX);
bge-large (0.928), e5-large-v2 (0.924) and mpnet (0.918) were within noise of bge-base
(0.924 with parent) at 3–10× the cost. Caveat: model and template were chosen on all 300 gold
pairs, so the reported P/R is slightly optimistic; thresholds were tuned on the dev half only.

**D-017 Determinism of embeddings.** Vectors are cached as float16 in content-addressed Parquet
shards (`data/interim/embeddings/<model>-<hash>/`), keyed by the exact text and namespaced by
model, revision, pooling, prefix, max length and dtype; the model revision is pinned
(`a5beb1e…`). Re-runs reuse the cache, so cosine scores (rounded to 4 d.p.) and therefore all
merge decisions are reproducible even though CPU matmul results can differ in the last bits
between batch compositions.

**D-018 Gold set.** 300 candidate pairs in two strata: 150 uniform over pairs (unbiased, but few
positives) and 150 uniform over lemma groups then over pairs within the group (many more
positives, so recall is measurable). Each row stores its inclusion probability so
Horvitz–Thompson-weighted population estimates can be reported next to plain sample P/R. The
file contains no aligner scores (blind labelling). Labels: `same` / `related` / `different` +
`ambiguous`, defined in docs/GRANULARITY.md. **The labels were produced by the AI agent (Claude)
acting as annotator, not by a human**; a human spot-check is recommended (Gate-1 decision).
Dev/test halves are assigned by hashing the pair id.

**D-019 LLM judge.** Implemented (`senselm.inventory.judge`, Message Batches, JSON-schema output,
dry-run estimate, hard £ cap checked against the worst case before each chunk) but **disabled**:
no project API key is configured. Middle-band pairs are kept separate and counted; the dry-run
estimate is in the Phase-1 report. Default model `claude-opus-5-5` (configurable; a cheaper
model is the owner's call). Batch requests cannot use server-side refusal fallbacks; a refusal
yields no verdict (pair stays separate).

## Granularity, IDs, ordering

**D-020 Policies.** See docs/GRANULARITY.md. Fine (default, issues IDs): subsenses kept,
merge only paraphrases, greedy one-to-one matching. Coarse (counts only): subsenses collapsed,
same-or-related merged, many-to-one onto WordNet senses. A third mode, `components`, exists for
experiments but is not used (transitive chaining collapses polysemous verbs).

**D-021 Canonical sense fields.** Lemma and UPOS come from the Wiktionary member when there is
one (page titles carry curated capitalisation; Wiktionary marks proper nouns), else from OEWN.
The primary gloss is OEWN's when present (short, uniform style), else Wiktionary's; all member
glosses are kept in `glosses`. Examples and tags are unions. The `canonical_key` (registry
identity) is the anchor member: the OEWN sense id when present (WordNet sense keys are the most
stable identifiers available), else the cluster's first Wiktionary sense in dump order.

**D-022 Token-ID layout: 22 + 10 bits, not 20 + 10.** `token_id = lemma_id << 10 | sense_index`
in a uint32. 20 lemma bits (1,048,576 lemmas) would not hold Phase 2: NCBI Taxonomy alone has
~2.6M scientific names, plus MeSH supplementary concepts, ChEBI, GO and Wikidata items — each
distinct term is a lemma. Because IDs are permanent, the layout cannot be widened later without
renumbering everything, so it is fixed now at 22 lemma bits (4,194,304 slots) and 10 sense bits
(1,024 senses per lemma). The top 4,096 lemma slots are reserved for structural tokens:
`token_id ≥ 0xFFC00000` (4,194,304 ids). Lemma id 0 is never assigned (token id 0 is never
lexical). **Open decision for Gate 1** (alternative: 20 + 10 with a separate namespace for domain
terms).

**D-023 Sense order within a lemma** (frequency proxy; OEWN 2025 ships no tag counts):
1. senses with a WordNet member, by WordNet sense number (Princeton numbering follows SemCor
   frequency), POS interleaved with the lemma's dominant POS first (the POS with more
   WordNet senses);
2. Wiktionary-only senses in Wiktionary order (top-level senses in entry order, each subsense
   right after its parent);
3. senses tagged obsolete / archaic / rare / dated / historical / nonce-word / uncommon last.
Ties break on dump order then id, so the order is deterministic.

**D-024 Registry identity across builds.** A canonical sense keeps its previous token id if its
`canonical_key` matches; otherwise the previous id of the same lemma with which it shares the
most members (a split keeps the id on the anchor's side; a merge keeps the strongest claim and
tombstones the other id with `replaced_by`). New senses take the next sense index never used by
the lemma; tombstoned ids and indices are never re-issued or revived (a returning sense gets a
new id). The build refuses to write a registry that fails `check_append_only`; the previous
registry is copied to `registry_history/` first. Lemma ids follow the same rules.

**D-025 Registry durability.** `data/` is git-ignored (multi-GB), but a permanent-ID registry
must survive this ephemeral container. The full registry (79 MB, with glosses and member lists)
stays in `data/processed/`; every ID-issuing build also writes a slim **ledger** to the
git-tracked `registry/` directory (25.5 MB + 7.1 MB lemma registry, zstd level 19): no glosses,
and `members` replaced by `other_members` (members minus the anchor, empty for 92% of senses), which
loses nothing needed for ID continuity (`ids.to_ledger` / `ids.from_ledger` round-trip). When
`data/processed/` is missing, the build continues from the ledger.
`tests/test_ids.py::test_committed_registry_is_append_only` compares the working-tree ledger
with the one committed at HEAD, so a rebuild that changes any existing ID fails before it can be
committed, on any machine. Long-term storage (git history grows ~30 MB per registry version) is
a Gate-1 decision.

**D-026 Threshold selection.** `auto_merge_threshold` is chosen on the dev half of the gold set
under the *full* policy (matching included), maximising `tune_metric`: F0.5 for the fine policy
(a false merge destroys a distinction irreversibly; a false split only leaves a near-duplicate
token), F1 for coarse. Result: fine 0.69, coarse 0.59 (`eval/alignment_eval.md`). With one-to-one
matching the optimum is far below the pairwise one (0.676 for F1), because matching removes most
false positives. `separate_threshold` (bottom of the judge band) = 0.55: below it only 2 of 83 gold
pairs are `same`.

**D-027 Judge workload reduction.** Only band pairs whose verdict can change the fine policy's
outcome are sent to the judge: both senses still unmatched after auto-merge, and the pair is the
top-1 band candidate of one of its senses (`llm_judge.relevant_only`, `top_k`). This reduces the
band from 244,284 pairs to 56,580 requests.

**D-028 Lineage absorption (fine policy).** Error analysis of the gold set showed 11 of 29 missed
merges were blocked by one-to-one matching: a Wiktionary parent sense took the WordNet match and
its paraphrasing subsense was left as a near-duplicate token. Rule F5 (docs/GRANULARITY.md) lets
a matched WordNet sense also absorb eligible ancestors and a *unique* eligible descendant;
several eligible children are never absorbed, so siblings never merge (a first version absorbed
all descendants and a unit test caught it merging siblings). Effect on the gold set: recall
0.775 → 0.791 (HT-weighted 0.774 → 0.795), precision 0.893 → 0.895; 1,092 extra merges in the
full inventory.

**D-029 Fresh v0.1 registry.** The ID registry was regenerated from scratch by the final
Phase-1 build (development builds before the first commit were never released). From the first
commit of `registry/` onwards, IDs are permanent and protected by the append-only checks.
