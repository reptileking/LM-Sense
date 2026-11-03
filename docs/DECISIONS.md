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

**D-019 LLM judge.** *Gate 1 (2026-11-01): the owner chose a locally hosted LLM on their own
server instead of a paid API (see D-030); the Anthropic Batch path below remains available but
is not used.* Implemented (`senselm.inventory.judge`, Message Batches, JSON-schema output,
dry-run estimate, hard £ cap checked against the worst case before each chunk) but **disabled**:
no project API key is configured. Middle-band pairs are kept separate and counted; the dry-run
estimate is in the Phase-1 report. Default model `claude-opus-5-5` (configurable; a cheaper
model is the owner's call). Batch requests cannot use server-side refusal fallbacks; a refusal
yields no verdict (pair stays separate).

## Granularity, IDs, ordering

**D-020 Policies.** *Approved at Gate 1 (2026-11-01): the fine policy issues the IDs.* See docs/GRANULARITY.md. Fine (default, issues IDs): subsenses kept,
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
lexical). *Approved at Gate 1 (2026-11-01).*

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

## Phase 2

**D-030 Local LLM judge.** *Gate 1 (2026-11-01): the owner chose a locally hosted LLM on their own
computer/server for the ~56k uncertain-band pairs instead of a paid API.*
`senselm/inventory/judge_local.py` talks to any OpenAI-compatible `/v1/chat/completions` server
(Ollama, vLLM, llama.cpp server, LM Studio) using only the standard library, so the judging
machine needs nothing beyond a clone of the repository and `uv`. The job file
(`judge/job_v0_1.parquet`: 300 gold pairs + 56,265 decision-relevant band pairs, D-027) is
committed; `probe` checks the endpoint and detects structured output (json_schema → json_object →
plain text with tolerant parsing, including `<think>` blocks); `pilot` judges the 300 gold pairs
first and reports accuracy, so a model is accepted only if it beats the cosine band baseline;
`run` writes resumable shards under `judge/verdicts/<model>/` (temperature 0, seed 0, fixed
prompt version recorded with every verdict); `import` builds `data/interim/judge_cache.parquet`,
which `align` joins when `judge_verdicts.use: true`. Verdicts are committed, so the judging can run
on the owner's machine while everything else runs here. The owner's machine is Windows, which
Remote Control from a cloud session cannot use, so the judge runs there directly with
`uv run --no-project --with polars --with pyarrow --with pyyaml` (the judge was made NumPy-free
for this). Pilot 1 (2026-11-01): `qwen2.5-7b-instruct` with a 4k context (the 14B with Ollama's
32k context spilled 65% onto the CPU): 3-way accuracy 0.817, same-vs-rest P 0.933 / R 0.76 overall
but **in the band P 0.818 / R 0.562** (9 of 11 merges right, n = 78 band pairs), 1.3 req/s
(≈ 12 h for the band). Pilot 2: `qwen2.5-14b-instruct`, 4k context (still partly on the CPU):
3-way accuracy 0.747, overall P 0.895 / R 0.791, **in the band P 0.643 / R 0.562** (9 of 14 merges
right), 0.26 req/s (≈ 60 h for the band). Neither passes the ~0.9 bar, so **no judge verdicts are
used**: the band stays "keep separate", the fine policy's conservative default (a missed merge
leaves a near-duplicate token; a wrong merge would destroy a distinction). Both models confuse
"same" with "related" most often. Revisit only with a stronger model on hardware that runs it at
≥ ~1 req/s (the large local MoE models run mostly on the CPU here), judged by the same pilot.
Pilot 3: `qwen2.5-32b-instruct`, 4k context, fully on the GPU: 3-way accuracy 0.85, overall
P 0.911 / R 0.868, **in the band P 0.737 / R 0.875** (14 of 19 merges right; its errors are mostly
"related" called "same"), 1.29 req/s (≈ 12 h for the band). Too eager to use raw, but close
enough that an acceptance rule (minimum confidence, or agreement with a second model) might
reach the bar. Such a rule cannot be tuned and tested on the same 78 pilot pairs, so a **fresh
blind validation set** was labelled: `eval/judge_band_gold.csv`, 200 pairs drawn from the band
job rows themselves (50 per cosine stratum 0.55/0.585/0.62/0.655/0.69, gold pairs excluded;
57 same, 49 related, 94 different, 38 flagged ambiguous). The full 32B run judges those pairs as
part of the band, so `judge_local evaluate --model … [--agree-with …]` measures every rule on
them (raw and re-weighted to the band's stratum sizes, with an estimate of the merges it would
make), and `judge_local import --min-confidence … [--agree-with …]` imports only accepted "same"
verdicts (the rest become "uncertain", which the fine policy does not merge). Verdicts are
imported only if a rule reaches ~0.9 precision on the fresh set.
**Outcome (2026-11-02).** The full 32B run judged all 56,265 band pairs (17,874 "same", ~12 h).
On the fresh set its "same" verdicts are 0.79 precise (38 of 48) at 0.67 recall; its confidences
are coarse (0.85 / 0.9 / 0.95) and no threshold helps (0.79 at ≥ 0.95, recall 0.46); precision does
not rise with cosine (0.80 / 0.89 / 0.92 / 0.67 by stratum, the top stratum worst); requiring the
7B to agree gives exactly the 7B's set on the pilot band pairs (0.82). All 10 false merges are
"related" pairs (broader / narrower), none "different", and 7 of them are pairs flagged ambiguous
when labelling (on the 162 unambiguous pairs: 0.92). No rule meets the bar, so **no verdicts are
imported** and v1.0's ids stay as built. The verdicts stay in the repository
(`judge/verdicts/qwen2.5-32b-judge/`, evaluation in `eval/judge_band_eval_qwen2.5-32b-judge.json`):
they can serve as a *soft* near-duplicate table (pairs of token ids judged "same", ~79% precise)
for evaluation and for sharing embeddings later, without merging any ids. Cheap structural checks
over all 17,751 band "same" verdicts do not separate the errors: keeping only mutual best cosine
matches gives 0.79, dropping senses judged "same" with two partners 0.83, both 0.80 (on the 48
labelled verdicts); the errors are broader / narrower judgements that only a semantic check
(a model or a person) can catch. A focused second pass (`--task verify`: the 32B asked only
whether a first-pass "same" is exactly the same meaning or a near miss — broader / narrower,
object / contents, literal / figurative ...; invented examples) was tested on the 171 labelled
first-pass "same" verdicts: it confirmed 163 and, on the 67 band pairs, moved precision from
0.776 (52 / 67) to 0.790 (49 / 62) while dropping 3 true matches, i.e. it barely discriminates
(0.82 req/s). Not pursued; the owner reviews the first-pass "same" pairs by hand instead
(2026-11-02). `senselm.inventory.review export` writes the 17,751 pairs in a fixed random order
(`sha256(seed:job_id)`), so any reviewed prefix is a uniform sample and the running share of
"same" answers estimates the 32B's precision with a Wilson interval. The review page (a private
claude.ai artifact, https://claude.ai/artifact/54CBaU4K9oq4aZznxKANe5; source in
`tools/review_page/`) shows each pair with more context than the judge had (WordNet synonyms and
lexicographer file, Wiktionary labels and parent sense) and stores answers per block of 500
(`reviews/bNNN`: `job_id -> [s|r|d|u, epoch ms]`). `review ingest` reads a dump of that collection
into `judge/review/human_v0_1.parquet` and `eval/judge_human_review.json`. No id changes until the
owner approves what to do with the confirmed pairs (merge, or soft links only).

**D-031 HTTP User-Agent.** Wikimedia and geneontology.org return 403 to Python's default
User-Agent. The downloader sends `SenseLM-research/0.1 (+https://github.com/reptileking/LM-Sense)`
— the project name and repository only, no personal data — as Wikimedia's policy requests.

**D-032 Streaming the Wikidata JSON dump.** The full dump (`wikidata-20261026-all.json.gz`,
156.7 GB gzip, ~1.6 TB JSON) cannot be stored (≈20 GB free) and is streamed once:
* 64 MB HTTP range requests over **two** parallel connections (Wikimedia's per-IP courtesy limit;
  each connection is capped at ~5 MB/s), reassembled in order; every range's length is verified
  and any transport error is retried with back-off. The first full attempt died at 12% on an
  `http.client.IncompleteRead`, which the retry clause did not catch; it now retries every
  exception;
* sha256 and md5 are computed on the fly; the md5 is checked against Wikimedia's published sum
  after the pass, and gzip's own CRC-32 checks the inflated content;
* inflation in the main process, line filtering in two worker processes: a filter *thread* starved
  the network reader through the GIL (1.4 MB/s → 9.9 MB/s after the change);
* the pass cannot resume after a crash: the dump is one deflate stream (plus a 22-byte "[" member)
  whose decompressor state cannot be checkpointed, and storing the compressed file is impossible.
  A crash costs a restart (~4.3 h).
The pass writes a *superset* (D-038); the concept filter runs afterwards on it.

**D-033 Wikidata lexemes.** English lexemes (Q1860), one row per sense (gloss = English gloss;
lexemes without senses yield one `no_gloss` row so their forms still reach the forms table).
The Q-ids of lexical categories and grammatical features were mapped by reading the data (e.g. the
feature on "ran" of lexeme "run" is Q1392475 = past tense) because the Wikidata API rate-limited
label lookups (HTTP 429); the full tables are in `configs/normalise.yaml` and
`senselm/ingest/wikidata_lexemes.py`.

**D-034 Synonyms never create senses.** In the domain sources only the preferred name of a
concept becomes a sense (MeSH preferred term, GO/ChEBI `name`, NCBI scientific name, SNOMED
preferred synonym, Wikidata English label). Entry terms, synonyms, common names and aliases go to
`relations` and are used for alignment only. Otherwise every synonym would be a separate token
for the same meaning, which is exactly what the sense inventory exists to avoid.

**D-035 MeSH dictionary forms.** MeSH headings are Title Case, often plural and sometimes
inverted ("Insulin, Regular, Pork"); a token's lemma must be the dictionary form a text would
contain. Rules (module docstring of `senselm/ingest/mesh.py`, tests in `tests/test_mesh.py`):
prefer a non-inverted entry term, else un-invert short inversions; keep the case of acronyms,
parts with digits or internal capitals, single capital letters and genus names; lower-case the
rest, **per hyphen part** ("Self-Assessment" → "self-assessment"); singularise the head noun with
lemminflect's dictionary, applying its OOV rules only to regular English plurals of Title Case
words (its OOV rules turn Latin into nonsense: "Xenarthra" → "xenarthrum", "Sophora" →
"sophorum"). Organism records keep scientific names verbatim ("Escherichia coli", "Populus");
genus detection uses the single-word scientific names of the pinned NCBI taxdump (165,860 names),
because no rule on the word itself separates "Ophiocordyceps" from "Coliphages". Supplementary
records are mostly sentence case, where only the first capital is uninformative. A first version
lower-cased genera and garbled 7,700 descriptor lemmas; the dev review caught it before release.

**D-036 SNOMED CT is restricted.** The UK edition (NHS TRUD, SNOMED CT affiliate licence via
the UK national licence) may be used but not redistributed. `senselm/ingest/snomed.py` reads the
RF2 Snapshot files straight from the release zip(s) and tags every row `licence:restricted`.
SNOMED senses get permanent token ids like every other sense, but no SNOMED text may reach git:
* in the committed ledger a token anchored on a SNOMED member (`sct:` id) carries a hashed key and
  lemma (`restricted:` + sha256 prefix), SNOMED member ids are removed from other tokens' member
  lists, and lemmas used only by such tokens are hashed in the lemma ledger
  (`ids.redact_ledger` / `ids.redact_lemmas`);
* a rebuild that contains SNOMED again (e.g. on a fresh machine with only the ledger) recomputes
  the hashes and restores the real values before assigning ids (`ids.rehydrate`), so ids
  continue; a build *without* SNOMED holds those ids out of the assignment untouched (neither
  continued nor deprecated) and keeps their sense indices reserved;
* `inventory.build` refuses a restricted source whose ids are not covered by
  `restricted_id_prefixes`; the stage-2 gold sampler never samples restricted sources (the gold
  set is committed);
* `tests/test_build.py::test_restricted_source_never_reaches_the_ledger` checks all three paths.
The hashes are unsalted on purpose (a salt kept outside git would make the fresh-machine rebuild
impossible); they keep SNOMED text out of the repository, not secret from SNOMED licensees.
Files needed from the owner: the SNOMED CT UK Edition RF2 release zip(s) from TRUD (the package
with the Snapshot files of the International release and the UK clinical extension) in
`data/raw/snomed/`; then uncomment SNOMED in `configs/ingest_v1.yaml`.

**D-037 Stage-2 alignment (attach).** Stage 1 (OEWN ↔ Wiktionary, the v0.1 fine policy) is
recomputed unchanged, so every v0.1 cluster anchor survives. The other sources are then attached
one at a time in `source_order` (lexemes, items, MeSH, GO, ChEBI, NCBI Taxonomy, SNOMED):
candidates share the lemma (exact case; case-insensitive only when no exact-case cluster exists,
never for taxa) and the POS class (NOUN ≡ PROPN); score = max cosine to the cluster's members
(same embedding model and cache as stage 1); cross-references (Wiktionary/lexeme Q-ids, item
P685/P486/P683/P686, also bridged through the items superset for items that are not senses) rank
above every cosine edge; greedy one-to-one per source; a sense that does not attach starts a new
cluster that later sources can join. Thresholds per source from the plateaus of the precision/
recall curves of a blind 240-sense gold set (`eval/attach_gold.csv`, 40 per source, labelled
before any score was looked at; a `match` may name two candidates when a WordNet and a
Wiktionary cluster that stage 1 kept apart are both the same sense): lexemes 0.60, items 0.58,
MeSH 0.60, GO 0.65, ChEBI 0.65, NCBI 0.53 — micro-averaged P 0.92 / R 0.90 at these values
(`eval/attach_gold_eval.json`). Two rules came from that gold set's errors:
taxa never use the case-insensitive fallback (a genus "Mus" is not the noun "mus"), and a taxon
may only join a cluster whose WordNet members have a lexicographer file compatible with its
kingdom (Metazoa → noun.animal, plants and fungi → noun.plant) — NCBI precision 0.84 → 1.00.
Memory: a first implementation kept a Python dict per sense (12.7 GB, OOM-killed); it now keeps
columns in Polars and loads only the cached vectors a query needs.

**D-038 Wikidata items: the concept filter.** Superset (written by the stream, 2 steps so the
filter can change without re-streaming): English label AND (linked from an English lexeme sense
via P5137 OR has "subclass of" P279 OR carries a domain cross-reference: P685 NCBI, P486/P6694
MeSH, P683 ChEBI, P686 GO, P5806 SNOMED, P8814 WordNet 3.1). Senses: items linked from lexeme
senses, plus *concepts*: P279 (a class), a **lower-case English label** (Wikidata's convention
for common nouns), an English description (the gloss), an English Wikipedia article
(notability) and no excluded type (scholarly article, human, Wikimedia category / disambiguation
/ template, and taxon / gene / protein / chemical compound, whose dedicated sources are NCBI, GO,
ChEBI and MeSH). The lower-case rule was added after reviewing a trial on the first 100 MB of the
dump: 559 of 1,548 selected items were capitalised classes that are really names (languages,
rocket and aircraft models, championships, dog breeds, a 2011 film). Named entities (people,
places, works, organisations) are out of scope for v1.0: their surface names are not senses of
English words, they would consume lemma ids (see the Gate-2 decisions) and the byte fallback
spells them; those linked from English lexemes are kept.

**D-039 NCBI Taxonomy filter.** Formal ranks domain … species only; names must have the form the
rank requires (one capitalised word above species; a strict binomial or a virus-style name for
species); placeholder subtrees (environmental samples, unclassified, uncultured, metagenomes) and
names with `sp.`, `cf.`, strain codes or nomenclatural annotations are dropped. This keeps
957,340 of ~2.7M taxa: vocabulary (*Mus musculus*) rather than specimen labels
(*Brevundimonas sp. S2*). Glosses are synthesised from rank, lineage and common names.

**D-040 GO and ChEBI.** One sense per non-obsolete term; ChEBI only 3-star (manually curated)
entities. Terms without a definition get a gloss synthesised from their `is_a` parents
(10,859 ChEBI terms). ChEBI definitions contain inline HTML (`<em>`, `<sub>`, `<small>D</small>`
in 28,736 of them): tags are stripped and entities unescaped in names, definitions and synonyms.

**D-041 Structural vocabulary.** 393 tokens in the reserved range (docs/STRUCTURAL_TOKENS.md),
one reserved lemma slot (1,024 ids) per category so every category can grow without renumbering:
specials, 256 byte tokens + `<bytes_start>`/`<bytes_end>` (byte value = sense index, so decoding a
byte token is a mask), 32 UniMorph features emitted after the sense token they inflect, case (4),
whitespace (7, incl. a clitic marker), punctuation (56, one per character), digits (10; numbers
are digit sequences), minimal pragmatics (8: register, emphasis, quotation, irony, hedging) and
orthographic variant selectors (6: British/American/archaic/…). The structural registry is
append-only like the lexical one (`registry/structural_registry.parquet`).

**D-042 Inventory v1.0 build.** Clusters = stage-1 clusters ∪ stage-2 attachments (exact
duplicates removed in normalise follow their kept sense). A cluster's lemma and POS come from its
Wiktionary member when present, else from the first member in `source_order`; gloss from the
first member (WordNet's when present). New senses get sense indices in this order: WordNet,
Wiktionary, rare Wiktionary (exactly the v0.1 tiers), then Wikidata-lexeme-only senses, then
concept / domain senses in source order — general vocabulary before technical senses of the same
lemma. Every canonical sense lists its members and their licences (`licences`), so any licensed
subset can be removed by filtering.

**D-043 Ledger sharding.** GitHub rejects files over 100 MB and warns above 50 MB. A ledger table
larger than 45 MB is written as equal shards (`registry/id_registry-00000.parquet`, …, contiguous
token-id ranges; `ids.write_ledger` / `ids.read_ledger`); the cross-machine append-only test reads
either layout from HEAD and from the working tree.

**D-044 Config inheritance.** A config may say `extends: <path>`; mappings merge recursively and
everything else is replaced (`common.load_config`). The v1.0 configs (`normalise_v1.yaml`,
`inventory_v1.yaml`) extend the v0.1 ones, so the rules cannot drift apart; provenance records the
merged config.

**D-045 Inventory v1.0 ships without Wikidata items.** The full-dump pass (D-032) had reached
74% (88.6M items scanned, 3.68M superset rows, 3 h 15 min) when the cloud container running it
was recycled while the session was idle (~16:25 UTC); the process died with it, and the pass
cannot resume (one deflate stream; nothing on disk to continue from). Rather than hold Gate 2 for
another ~4.5 h pass, v1.0 is built from the other sources; the items ingest, its concept filter
(D-038) and its stage-2 threshold (tuned on a trial of the first 100 MB of the dump) are ready.
Adding items later is safe for ids: a later build attaches items to the existing clusters and
issues new ids for the rest, append-only; domain senses that would have joined an item cluster
keep their v1.0 ids through member overlap. Lesson for long jobs in this environment: background
work only survives while a session turn is active, so a multi-hour job must either keep the
session busy or be resumable (for this dump: the bz2 multistream variant, whose streams can be
restarted independently, or a gzip index with seek points).

## Gate 2 and Phase 3

**D-046 Gate 2 approved (2026-11-03) with the recommendations of `docs/phase2_report.md`
section 4.** (1) No ids are merged from the local-LLM judge; its first-pass "same" verdicts are
kept as a soft near-duplicate table, and the owner's manual review (D-030) continues alongside;
confirmed pairs are merged only with the owner's approval. (2) Wikidata items are streamed again
at the start of Phase 3 and added as v1.1 (append-only). (3) SNOMED CT UK waits for the owner's
TRUD files; when they arrive it becomes a local, restricted extension whose senses stay out of
training data for any released model. (4) The 22 + 10 id layout stays; lemma slots are the scarce
resource (no bulk entity imports). (5) Long-tail domain senses keep their ids; each training run
decides which senses get an embedding row. (6) Named entities stay out; Phase 3 measures how much
real text falls back to bytes. (7) The ledger stays in git until the repository nears ~1 GB.
(8) The current NLM terms are re-read before any public release that includes MeSH.
