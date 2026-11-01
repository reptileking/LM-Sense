# Granularity policy: when are two senses one?

This document is the rule. The code that applies it is `senselm/inventory/policy.py`; the
parameters live in `configs/inventory.yaml` (fine, default) and `configs/inventory_coarse.yaml`
(coarse). Changing a policy and re-running `python -m senselm.inventory.build` regenerates the
counts (the coarse build issues no IDs; see "IDs and policy changes" below).

## Units

* A **source sense** is one OEWN sense (a lemma's membership in one synset) or one Wiktionary
  sense line (including subsenses). Wiktionary *header* lines ("Terms relating to animals.",
  "A number of places in the United States:") are not senses.
* A **canonical sense** is a set of source senses judged to be one meaning. It gets one token.
* Only senses with the **same lemma** (case-sensitive; case-insensitive fallback only when the
  other source lacks the lemma entirely) and the **same POS class** (NOUN and PROPN are one
  class) can ever be the same sense. Different lemmas are different tokens even when they
  are synonyms (`car` ≠ `automobile`); synset membership is kept as data (`synset_id`).

## Judgement scale (also the gold-annotation guideline)

| label | meaning | examples from `eval/alignment_gold.csv` |
|---|---|---|
| **same** | Both definitions denote the same meaning/referent: paraphrases, or the same sense defined with different detail, emphasis or wording. A speaker never needs to tell them apart. | *Brummagem* "cheap and showy" / "Cheap and showy; meretricious"; *Frunze* "the capital of Kyrgyzstan (known as Frunze 1926–1991)" / "Bishkek … (historical)"; *coalman* "a man who delivers coal" / "Someone who sells coal" |
| **related** | One subsumes the other: broader ↔ narrower, genus ↔ member, a domain-restricted version of the same meaning, or a regular metonymy (animal ↔ its meat, instrument ↔ method, discipline ↔ its subject matter), or a causative ↔ inchoative verb pair. One sense in a coarse dictionary, two in a fine one. | *Syringa* "genus of … lilacs" / "any of the flowering plants of the genus"; *guinea hen* "flesh of a guinea fowl" / "the female of the guinea fowl"; *member of parliament* "member of the British House of Commons" / "representative … to a house of a parliament (AU, CA, NZ, UK)" |
| **different** | Distinct meanings or referents, including figurative extensions that dictionaries list as separate senses. | *eclipse* "one celestial body obscures another" / "obscurity, decline"; *cabin* "compartment of an aircraft" / "room on a boat"; *Flint* (city in Michigan) / (town in Wales) |

`ambiguous = 1` marks pairs where the annotator hesitated between two labels.

## Fine policy (default; issues the permanent IDs)

> Keep the finer source's distinctions unless the two glosses are paraphrases.

* **F1 — within a source, never merge.** Each source's lexicographers drew their distinctions
  deliberately. Only exact duplicates (same lemma, POS, gloss path, gloss and tags) are merged
  — they are data artefacts (normalise stage, 99 cases).
* **F2 — subsenses are senses.** Every Wiktionary subsense is its own unit.
* **F3 — across sources, merge paraphrases only.** An OEWN sense and a Wiktionary sense merge
  iff they are *same*: operationally, cosine similarity of their texts (bge-base-en-v1.5,
  parent gloss + gloss) ≥ `auto_merge_threshold`, or — when the LLM judge has run — the cosine
  is in the band (`separate_threshold`, `auto_merge_threshold`) and the judge said `same`.
* **F4 — one-to-one.** A sense merges with at most one sense of the other source (greedy,
  highest similarity first). Where one source is finer, its extra senses survive as separate
  canonical senses instead of being swallowed by the coarser source's sense.

## Coarse policy (counts only)

> Keep the coarser source's distinctions where the sources overlap.

* **C1 — subsenses fold into their top-level sense** (header lines skipped).
* **C2 — merge same *or related*:** cosine ≥ the (lower) coarse threshold, or a judge verdict
  of `same` / `related` in the band.
* **C3 — many-to-one onto WordNet.** Each Wiktionary unit joins its best eligible WordNet sense;
  WordNet senses absorb any number of Wiktionary units but never merge with each other. In
  lemmas both sources cover, WordNet's distinctions are the floor of granularity.

A third matching mode, `components` (transitive closure of all eligible edges), exists for
experiments. It is not recommended: chains through polysemous verbs collapse many unrelated
senses into one.

## Parameters

| knob | fine | coarse | effect |
|---|---|---|---|
| `subsenses` | `keep` | `collapse` | Wiktionary subsense handling |
| `auto_merge_threshold` | 0.68 | 0.60 | cosine at/above which a cross-source pair merges |
| `separate_threshold` | 0.60 | 0.52 | cosine at/below which a pair never merges; between the two, only an LLM verdict merges |
| `judge_accept` | `[same]` | `[same, related]` | judge verdicts that allow a merge in the band |
| `matching` | `one_to_one` | `many_to_one` | see F4 / C3 |

The thresholds are tuned on the dev half of the gold set (`eval/alignment_eval.md`) and the
precision/recall at the configured values are reported in docs/phase1_report.md. A **finer**
inventory than "fine" is obtained by raising `auto_merge_threshold` (at 1.0 nothing merges:
the upper bound is the number of source senses); a **coarser** one by lowering the coarse
threshold or switching to `components`.

## IDs and policy changes

Only the fine build writes the ID registry. Because token IDs are permanent, switching the
production policy later is possible but not free: senses that merge are tombstoned
(`replaced_by` the surviving id), senses that split keep the id on the anchor's side and the
split-off part gets a new id. A fine inventory can always be *mapped* onto a coarse one (the
coarse build gives the cluster of every fine sense); a coarse inventory cannot be split back
without issuing new ids. This asymmetry is why the fine policy issues the IDs.

## Known limitations

* Cosine similarity is a proxy for "paraphrase". Its errors are measured on the gold set; the
  LLM judge is the intended fix for the uncertain band.
* The one-to-one constraint can keep near-duplicates apart, e.g. when Wiktionary splits one
  meaning into transitive and intransitive senses that WordNet keeps as one: only one of them
  merges with the WordNet sense.
* Within-Wiktionary near-duplicates (not exact) are never merged under the fine policy.
