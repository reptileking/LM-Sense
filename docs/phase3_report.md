# Phase 3 report — codec (deterministic decoder, baseline encoder)

Gate 3. Everything below is reproducible from the commands in section 1; numbers come from
`eval/codec_eval.json` (held-out text) unless stated otherwise. Decisions are in
`docs/DECISIONS.md` (D-046 to D-050).

**In one paragraph.** English now converts to sense tokens and back. The decoder is
deterministic, and the baseline encoder (spaCy, most frequent sense) is lossless by construction:
all 4,500 held-out FineWeb-Edu documents (3.5M words) come back character for character. 98.4% of
words get a sense token; the rest are spelled in bytes, mostly names, rare words, numbers with
letters and URLs. Sequences are **1.18× as long as GPT-2's** (1.58 tokens per word against 1.35),
because grammar (13% of tokens) and byte spelling (10.5%) cost tokens that a BPE vocabulary folds
into words. Sense accuracy on SemCor / Senseval could not be measured yet: the server hosting the
standard test sets is unreachable from this environment (decision 1).

## 1. What was built

### 1.1 Decoder (spec 3.1)

`senselm/codec/decoder.py`. Reads a token sequence and writes text:

* **Words**: `[case] [variant] SENSE [morph…]`. The sense token gives the lemma; the morph tokens
  after it pick the spelling from the paradigm (`morph.py`): closed-class tables (be, have, do,
  personal pronouns, this / that), then 41,910 Wiktionary / Wikidata-lexeme forms that the
  rule-based inflector gets wrong (it agrees on 91.9% of the 557,423 listed forms), then
  `lemminflect` on the head word (`carry the can` + PST → *carried the can*). UK/US forms
  (*travelled*) and second standard forms (*mediums* beside *media*) via `<variant:…>`.
* **Contractions**: `NEG` gives *n't* (*can't*, *won't*, *shan't* irregular); `<space:clitic>`
  gives *'s 're 'm 've 'll 'd* from be / have / will / would / us. **Genitive** `GEN` gives *'s*
  (a bare apostrophe after a plural in -s). The apostrophe is straight or typographic according
  to a stateful `<orth:apos_curly|apos_straight>` switch.
* **Orthography**: case (`cap`, `allcaps`, `lower`, `title`; sentence starts capitalised by
  default), default spacing (one space between words, none before closing or after opening marks,
  none around `-` and `/`, none inside numbers, straight quotes alternating open / close) and
  `<space:…>` tokens wherever the text differs; digits one token each; byte spans for anything
  else. Malformed input degrades to lemmas instead of raising.
* **Tests**: 17 hand-written sequences decode to exact expected strings (`tests/test_codec.py`),
  plus spacing, case and paradigm unit tests.

### 1.2 Baseline encoder (spec 3.2)

`senselm/codec/encoder.py`. spaCy `en_core_web_sm` 3.8 (tagger, lemmatiser, morphology); for each
word the longest multi-word lemma (up to 8 tokens) is tried first, then (lemma, POS) candidates
in order — spaCy's lemma and POS, other POS, the surface form, lemminflect's lemma — and the first
candidate whose paradigm reproduces the word wins, with its **most frequent sense** (lowest sense
index). spaCy's morphology chooses between bundles that spell the same word (*had walked* gets the
participle). Every word and every gap is checked against the decoder's own functions, so the
output always decodes to the input; whatever cannot be spelled becomes a byte span.

Three structural tokens were appended (build `codec-v1`; no existing id changed):
`<case:title>`, `<orth:apos_curly>`, `<orth:apos_straight>` (396 structural tokens in all).

Example (gloss of the chosen sense on the right):

```
He sat on the bank of the river and watched the bats.
he#2/PRON sit#0/VERB <morph:PST> on#7/ADP the#2/DET bank#0/NOUN of#0/ADP the#2/DET
river#0/NOUN and#2/CCONJ watch#0/VERB <morph:PST> the#2/DET bat#0/NOUN <morph:PL> <punct:.>
    bank#0/NOUN   sloping land (especially the slope beside a body of water)
    bat#0/NOUN    nocturnal mouselike mammal with forelimbs modified to form membranous wings ...
    of#0/ADP      Since, from (a given time, earlier state etc.).          <- most-frequent-sense limit

It's the dogs' bones; they've travelled 3.14 km.
it#0/PRON <space:clitic> be#0/VERB <morph:3> the#2/DET dog#0/NOUN <morph:PL> <morph:GEN>
bone#0/NOUN <morph:PL> <punct:;> they#0/PRON <space:clitic> have#0/VERB <variant:uk> travel#0/VERB
<morph:PST> <digit:3> <punct:.> <digit:1> <digit:4> km#0/NOUN <punct:.>
```

### 1.3 Metrics (spec 3.3) and inputs

* `senselm/eval/codec_eval.py`: surface round-trip (exact match, Levenshtein distance), coverage
  (sense token vs byte span, with the kinds of byte-spelled words) and sequence length against
  the GPT-2 and Qwen2.5 BPE tokenizers on the same text.
* `senselm/eval/wsd_eval.py`: sense accuracy on SemCor and the Raganato framework — WordNet keys
  mapped to inventory tokens through the OEWN members (`bank%1:17:01::` → `oewn-bank__1.17.01..`)
  — for three predictors: the inventory's most frequent sense, WordNet's first sense, and the
  encoder in context. Tested on a toy corpus; waiting for the data (decision 1).
* General text: the first 5,000 documents of a pinned FineWeb-Edu shard, read by HTTP range
  requests; documents 0–499 (355k words) for developing rules, 500–4,999 (3.5M words) held out.

### 1.4 Also in this phase

* **Wikidata items pass** (Gate 2, decision 2): restarted at the start of the phase; the pass can
  now checkpoint and resume (D-050), tested by killing a pass after its first checkpoint.
  *Status at the time of writing: running (about 30%); see section 2.4.*
* Gate 2 approval recorded (D-046).

### 1.5 Commands

| Step | Command | Time |
|---|---|---|
| structural tokens (+ codec-v1) | `uv run python -m senselm.inventory.structural --config configs/structural.yaml` | 1 s |
| codec tables | `uv run python -m senselm.codec.build --config configs/codec.yaml` | 45 s |
| try it | `uv run python -m senselm.codec --config configs/codec.yaml roundtrip "It's the dogs' bones."` | 15 s (table load) |
| reference tokenizers, WSD data | `uv run python -m senselm.ingest.download --config configs/download.yaml --only gpt2_tokenizer qwen25_tokenizer wsd_framework` | seconds |
| corpus sample | `uv run python -m senselm.eval.corpus_sample --config configs/corpus_sample.yaml` | 2 s |
| codec metrics | `uv run python -m senselm.eval.codec_eval --config configs/codec_eval.yaml --split eval` | 9 min |
| sense accuracy | `uv run python -m senselm.eval.wsd_eval --config configs/wsd_eval.yaml` | needs the WSD zip |
| Wikidata items (resumable) | `uv run python -m senselm.ingest.wikidata_items stream --config configs/ingest_wikidata_items.yaml` | ~4.5–5.5 h |
| tests | `uv run pytest` | 112 tests, ~2 min |

## 2. Key numbers

### 2.1 Held-out text (4,500 documents, 3,505,433 words, 21.7M characters)

| Metric | Value |
|---|---|
| Surface round-trip, exact documents | **4,500 / 4,500 (100%)**, 0 characters of edit distance |
| Words written with a sense token | **98.39%** (3,399,009 of 3,454,555 spaCy words) |
| Words spelled in bytes | 1.61% (55,546) — 10.5% of all tokens |
| Multi-word senses used | 229,918 (*New York*, *of course*, *give up*); 11,730 matches re-split word by word |
| Sense-codec tokens per word | **1.581** (3.91 characters per token) |
| GPT-2 / Qwen2.5 tokens per word | 1.345 / 1.352 |
| Length ratio, sense codec : GPT-2 / Qwen2.5 | **1.175 / 1.169** |
| Speed (one process: spaCy + encode + decode + both BPE tokenizers) | 6,400 words/s; decoding alone ~71,000 words/s |

Where the tokens go:

| Token kind | Tokens | Share |
|---|---|---|
| sense tokens | 3,139,312 | 56.6% |
| morphology (`PL`, `PST`, `3`, `V.PTCP`, `NEG`, `GEN`, …) | 738,745 | 13.3% |
| byte spans (incl. start / end markers) | 583,581 | 10.5% |
| punctuation | 574,873 | 10.4% |
| digits | 205,731 | 3.7% |
| spacing corrections | 157,558 | 2.8% |
| case | 136,429 | 2.5% |
| variant / apostrophe style | 5,411 | 0.1% |

What falls back to bytes (held-out, by byte-span tokens):

| Kind of word | Occurrences | Token cost | Share of byte tokens |
|---|---|---|---|
| capitalised (mostly names: *Loliondo*, *Seiwert*, *Onegin*) | 21,650 | 206,027 | 35% |
| lower case (rare words, typos, foreign words) | 11,397 | 123,549 | 21% |
| containing digits (*MiG-21*, *T-64*) | 7,949 | 75,748 | 13% |
| URLs, e-mail addresses, paths | 2,892 | 70,866 | 12% |
| mixed case (*MySQL*, *PLoS*, units such as *MPa*) | 3,575 | 42,153 | 7% |
| non-ASCII letters (*Zoöl*) | 3,223 | 34,133 | 6% |
| all-caps acronyms (*NECTA*, *MACV*) | 3,785 | 25,637 | 4% |

On the development split, spelling these words with GPT-2's subword pieces instead of single
bytes would cut 47% of the byte tokens, 4.7% of all tokens (decision 3).

### 2.2 Choices measured on the development split

| Choice | Effect |
|---|---|
| sentence starts capitalised by default (`auto_cap`) | −2.3% tokens, round-trip unchanged |
| multi-word lemmas: verb + preposition (*sat on*) and determiner-final spans (*on the*) not matched | +1% tokens, removes wrong senses such as *sit on* "to suppress" in "sat on the bank" |
| contractions decomposed (*it's* = it + clitic be `3`) rather than the inventory's contraction lemmas | keeps *is* vs *has* apart |

### 2.3 Codec tables

| Table | Rows |
|---|---|
| sense tokens / (lemma, POS) pairs / multi-word lemmas | 2,557,970 / 2,200,027 / 1,386,248 |
| paradigm exceptions (forms the inflector gets wrong) | 41,910 |
| UK / US inflection variants; alternative standard forms | 1,621; 11,333 |
| spelling variants usable with `<variant:…>` | 108,589 |
| structural tokens | 396 (+3 in codec-v1) |

### 2.4 Sense accuracy and Wikidata items

* **Sense accuracy (SemCor, Senseval-2/3, SemEval-2007): not measured yet** — the data is
  unreachable (decision 1). The code is ready and runs in minutes once the file is in place.
* **Wikidata items**: pass in progress (D-050); v1.1 follows when it completes.

## 3. Problems hit and how they were resolved

1. **The WSD test data cannot be downloaded.** `lcl.uniroma1.it` times out through the
   environment's proxy (2026-11-03, several attempts). Per the network rule you were given the
   URL and target path; the evaluation code was written and tested on a toy corpus instead.
2. **Contraction lemmas swallowed contractions.** Wiktionary lists *weren't*, *it's*, *they've*
   as lemmas, so the multi-word matcher turned them into single opaque tokens. Matching now stops
   at contraction pieces, which are encoded as NEG / clitics.
3. **Failed multi-word matches fell back to bytes** ("the Gulf" vs the lemma *The Gulf*, title-case
   headings such as "Physical Therapy"). A match that cannot be spelled is now re-encoded word by
   word, and `<case:title>` was added for headings.
4. **Words spaCy does not lemmatise**: *these / those* (lemma = itself) and title-case plurals
   such as "Systems" (tagged as proper nouns). Fixed with a demonstrative table and lemminflect's
   lemma as an extra candidate.
5. **Typographic apostrophes** (36% of documents use ’) would have byte-spelled every *don’t*
   and *dog’s*; a stateful apostrophe switch handles them. spaCy also mis-tags a curly *’s* after
   names (323 byte spans on held-out text); it is now attached by the kind of host word.
6. **Phrasal-verb over-matching**: "sat on the bank" became *sit on* "to suppress". Verb +
   preposition is no longer matched as a multi-word sense (verb + particle, *gave up*, still is).
7. **A 70× slower decoder** from a numpy dtype mismatch in the id lookup (a `uint32` array
   searched with a Python int is converted on every call); fixed with `int64` ids (1,000 →
   71,000 words/s).
8. **A mislabelled structural token**: regenerating the structural registry stamped a new token
   with the old build id; rebuilt from the committed registry so the new tokens carry `codec-v1`.
   No id changed (the append-only test passes).
9. **The Wikidata pass still could not resume.** Now it can (D-050): seek points via
   `indexed_gzip` over HTTP range requests and checkpoints every 8 GB. That library crashes
   (SIGFPE) when the seek-point spacing is a multiple of 1 GiB; the code avoids such values. The
   pass started this session predates the change, so it is left to finish while the session works.

## 4. Open decisions that need you

1. **WSD test data (blocks the sense-accuracy metric and Phase 4's tagger validation).** Either
   allow `lcl.uniroma1.it` in the environment's network settings (environment menu → Edit →
   Network access → Allowed domains), or download
   http://lcl.uniroma1.it/wsdeval/data/WSD_Evaluation_Framework.zip and place it in
   `data/raw/wsd/`. *Recommendation:* allow the host; I then report MFS / WordNet-first-sense /
   encoder accuracy before starting Phase 4.
2. **SemEval-2013 and SemEval-2015** (part of the same framework; their WordNet keys were mapped
   from BabelNet annotations). *Recommendation:* keep them out under "no BabelNet in any form"
   (D-049) and report SemCor, Senseval-2, Senseval-3 and SemEval-2007.
3. **Names and other unknown words** (your Gate 2 decision 6 asked to revisit this with numbers):
   1.6% of words, 10.5% of tokens; names alone are 3.7% of all tokens. Options: (a) keep byte
   spelling (simple, already lossless); (b) add a small subword vocabulary for unknown words — a
   new structural category of a few thousand BPE pieces trained on the byte-spelled spans — which
   would save about 5% of all tokens; (c) add names to the inventory (no: the lemma budget,
   decision 4 of Gate 2). *Recommendation:* (b), done at the start of Phase 4 before any corpus is
   tagged, because the tagged corpus stores codec output.
4. **Sequence length** (1.18× GPT-2). Grammar tokens are 13% of all tokens; fusing frequent
   feature pairs into single tokens would save roughly 2–3% but departs from the spec's atomic
   UniMorph features. *Recommendation:* keep atomic features; Phase 5 compares models at matched
   FLOPs and in bits per character, so the length only costs compute, not fairness.
5. **Phase 4 tagger and budget.** The spec lets the tagger be an Anthropic model (Message Batches)
   or a local model. *Recommendation:* validate Claude Haiku 5.5, Claude Sonnet 5.5 and your local
   Qwen2.5-32B on the WSD gold first (under £10 in API cost), then tag the first 10M words with the
   best accuracy per pound that fits the £50 cap — on current prices that is Haiku 5.5 (~£40) or
   the local model (free, ~1–2 weeks of your GPU). API runs need an Anthropic API key in the
   environment's secrets; nothing is spent without a dry-run estimate and the cap.

## 5. Estimated time and cost for Phase 4 (training data)

Sizing from the development split: 56% of words are content words, 87% of those have more than
one sense; offering the top 8 senses plus "none of these" averages 7.2 candidates and ~150 tokens
of glosses and examples per ambiguous word. For 10M words that is ~4.9M decisions, ~0.75B input
tokens and ~50M output tokens (one request per sentence; the instructions cached).

| Item | Estimate |
|---|---|
| Tagger (candidates, structured JSON output, Message Batches and local OpenAI-compatible paths, caching, dry-run cost estimate, hard cap) | ~1.5 sessions |
| Validation on the WSD gold (needs decision 1) | ~0.5 session; API cost < £10 per model |
| 10M-word corpus, tagged, as aligned (text, sense ids) Parquet shards | ~0.5 session + run time |
| Rare-sense synthetic data (K sentences for senses with < K corpus examples) | ~1 session |
| Report (accuracy vs cost per million tokens per model, Zipf plot, coverage) | ~0.5 session |
| **Tagging 10M words** — Claude Haiku 5.5 / Sonnet 5.5 / Opus 5.5 (batch prices) | **~£40 / ~£820 / ~£2,000** |
| Tagging 10M words — local Qwen2.5-32B | £0; ~1–2 weeks of continuous GPU time |
| Synthetic data for ~1.1M general-vocabulary senses with no corpus example, K = 3 — Haiku 5.5 / Sonnet 5.5 | ~£30 / ~£640 |

Prices: $0.10 / $0.50, $2 / $10 and $4 / $20 per million input / output tokens for Haiku 5.5,
Sonnet 5.5 and Opus 5.5, halved for Message Batches, at $1.20 per £ (as in `configs/align.yaml`).
These are planning figures; the tagger's dry run computes the exact cost before anything is spent.
Anything above the £50 cap needs your approval (spec 4.2: scale only after approval).
