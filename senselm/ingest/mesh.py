"""Ingest MeSH (NLM XML): descriptors and supplementary concept records (SCRs).

``python -m senselm.ingest.mesh --config configs/ingest_mesh.yaml``

Lemma (a dictionary form, docs/DECISIONS.md D-035). MeSH headings are Title Case, often plural
("Arteries", "Mice") and sometimes inverted ("Insulin, Regular, Pork"). For each record:

1. candidates = the preferred term, then the non-permuted entry terms of the preferred concept;
   the first candidate without an inversion comma (", ") wins; if none exists the preferred term
   is un-inverted ("Rats, Inbred BN" → "Inbred BN Rats");
2. organisms (tree B descriptors, organism SCRs): a scientific name ("Populus", "Escherichia
   coli", "Streptomyces") is kept verbatim — it is one when the term has the form
   ``Genus [epithet [epithet]]`` and either the scope note names a rank ("A plant genus of …")
   or the first word is a single-word NCBI Taxonomy scientific name (``taxon_names``; without
   it, a capitalised word missing from lemminflect's English dictionary). An organism SCR's
   lower-case disambiguator ("Aegle, moths") moves from the lemma to the gloss;
3. case, per hyphen-separated part: acronyms, parts with digits or internal capitals and single
   capital letters keep their case ("T-Lymphocytes", "Vitamin B 12", "mRNA"); so does a genus —
   a non-English NCBI taxon name followed by a lower-case content word, i.e. its species
   epithet ("Escherichia coli Infections") — and, in organism records, any non-English NCBI
   taxon name ("Pseudomonas Phages"); proper names (geographic descriptors, lexical tag NAM /
   TRD) keep it entirely; eponyms (EPO) keep the first word capitalised, and in sentence-case
   SCR names so do all-capitalised hyphenated non-English compounds ("Kocher-Debre-Semelaigne
   syndrome"), non-initial capitalised words ("… leaf curl Thailand virus") and hyphenated
   words with digits ("Val-Asp-8-Arg-vasopressin"); bracketed spans and words with slashes
   (strain designations) are kept verbatim; every other part is lower-cased
   ("Self-Assessment" → "self-assessment");
4. number: the head noun (last word, or the word before "of"; none in coordinations such as
   "Signs and Symptoms") is singularised. Known words use lemminflect's dictionary ("Mice" →
   "mouse"); a Title Case word that rule 3 lower-cased may also lose a regular English plural
   ending ("Lymphocytes" → "lymphocyte"), never a Latin one (-us, -is, -a, -ae, -ales …), and
   words that were lower-case already (species epithets, sentence-case SCRs) only change via
   the dictionary — so "Xenarthra", "Diabetes Mellitus" and "Streptococcus pyogenes" stay.

All other entry terms are kept in ``relations`` for alignment only (D-034).

Glosses: descriptor scope notes; SCRs have no definitions, so their gloss is synthesised from
their class and the descriptors they are mapped to ("chemical related to Insulin, Regular"),
plus the record note when it is descriptive. Domains come from the top-level MeSH tree
(C → medicine, D → pharmacology-chemistry, B → biology …).
"""

from __future__ import annotations

import logging
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import orjson
from lemminflect import getAllLemmas, getLemma
from lxml import etree

from senselm.common import BatchWriter, Provenance, load_config, resolve, stage_args
from senselm.ingest.manifest import source_version
from senselm.ingest.ncbi_taxonomy import _rows
from senselm.ingest.schema import SENSE_SCHEMA, make_row

LOG = logging.getLogger("senselm.ingest.mesh")
SOURCE = "mesh"
KEEP_CASE_RE = re.compile(r"[0-9]|^[A-Z]{2,}|.[A-Z]")  # applied to each hyphen-separated part
LATIN_S_ENDINGS = ("us", "is", "ss", "as", "os", "ys", "ans", "ens", "ics", "ales", "oides", "ces")
FUNCTION_WORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "as",
        "at",
        "by",
        "de",
        "for",
        "from",
        "in",
        "into",
        "of",
        "on",
        "or",
        "per",
        "than",
        "the",
        "to",
        "via",
        "vs",
        "with",
        "without",
    ]
)
GENUS_WORD_RE = re.compile(r"^[A-Z][a-z]+$")
SCI_NAME_RE = re.compile(r"^[A-Z][a-z]+(?: (?:[a-z][a-z-]+|subsp\.|var\.|f\.|pv\.|bv\.)){0,4}$")
TAXON_NAME_CLASSES = frozenset({"scientific name", "synonym", "equivalent name", "genbank synonym"})
RANK_NOTE_RE = re.compile(
    r"^(?:(?:an?|the|one|only)\s+)?(?:(?!of\b|in\b)[\w,-]+\s+){0,3}?(?:genus|species|subspecies|subgenus|"
    r"strain|serotype|serovar|variety|tribe|(?:sub|super)?family|(?:sub|infra|super)?order|"
    r"(?:sub|infra|super)?class|(?:sub)?phylum|division|kingdom)\b",
    re.I,
)
UNINFORMATIVE_NOTE_RE = re.compile(
    r"^(structure given|RN given|structure in first source|do not confuse|for the parent)", re.I
)
SCR_CLASSES = {
    "1": "chemical",
    "2": "protocol",
    "3": "rare disease",
    "4": "organism",
    "5": "population group",
    "6": "anatomy",
}


def iterparse(path: Any, tag: str) -> Any:
    for _, el in etree.iterparse(
        str(path),
        events=("end",),
        tag=tag,
        load_dtd=False,
        no_network=True,
        resolve_entities=False,
        huge_tree=True,
    ):
        yield el
        el.clear(keep_tail=False)
        parent = el.getparent()
        if parent is not None:
            while el.getprevious() is not None:
                del parent[0]


def known_word(w: str) -> bool:
    """In lemminflect's dictionary under any POS (its OOV rules garble Latin: sophora → sophorum)."""
    return bool(getAllLemmas(w.lower()))


def singular(w: str, *, oov_rules: bool = True) -> str:
    lem = getLemma(w, upos="NOUN", lemmatize_oov=False)
    if lem:
        return lem[0]
    if oov_rules and w.endswith("s") and not w.endswith(LATIN_S_ENDINGS):
        lem = getLemma(w, upos="NOUN")
        return lem[0] if lem else w
    return w


def load_taxa(path: Path) -> frozenset[str]:
    """Single-word NCBI scientific names and their synonyms (genera and higher taxa):
    'Pseudomonas', 'Ophiocordyceps'."""
    with zipfile.ZipFile(path) as z:
        return frozenset(
            r[1]
            for r in _rows(z, "names.dmp")
            if r[3] in TAXON_NAME_CLASSES and GENUS_WORD_RE.match(r[1])
        )


def latin_word(w: str, taxa: frozenset[str] | None = None) -> bool:
    """A capitalised non-English word naming a taxon: an NCBI genus or higher taxon when ``taxa``
    is given, else any such word without a regular English plural ending."""
    if not (w[:1].isupper() and w[1:].islower() and w.isalpha()) or known_word(w):
        return False
    if taxa is not None:
        return w in taxa
    return not w.endswith("s") or w.endswith(LATIN_S_ENDINGS)


def scientific_name(term: str, scope: str, taxa: frozenset[str] | None = None) -> bool:
    """For organism records: 'Genus epithet [subsp. x]' (Title Case never lower-cases a content
    word, so the shape suffices), or a single word that a rank note or the taxon list confirms."""
    if not SCI_NAME_RE.match(term):
        return False
    if " " in term:
        return True
    return bool(RANK_NOTE_RE.match(scope)) or (
        term in taxa if taxa is not None else latin_word(term)
    )


def title_cased(term: str) -> bool:
    """More capitalised than lower-case content words after the first ("Diastasis Recti And
    Weakness Of The Linea Alba"); function words do not count."""
    caps = low = 0
    for w in term.replace(",", " ").split()[1:]:
        if not w.isalpha() or w.lower() in FUNCTION_WORDS or len(w) < 3:
            continue
        caps += w[0].isupper()
        low += w.islower()
    return caps > low


def split_qualifier(term: str) -> tuple[str, str | None]:
    """Organism SCRs disambiguate homonymous genera with a lower-case suffix: 'Aegle, moths'."""
    parts = term.split(", ")
    if (
        len(parts) == 2
        and SCI_NAME_RE.match(parts[0])
        and parts[1].islower()
        and len(parts[1].split()) <= 3
    ):
        return parts[0], parts[1]
    return term, None


def uninvert(term: str, taxa: frozenset[str] | None = None) -> str:
    """'Rats, Inbred BN' → 'Inbred BN Rats', 'Insulin, Regular, Pork' → 'Pork Regular Insulin'; a
    leading scientific name stays first ('Neisseria meningitidis, Serogroup W-135'). Inversions
    with long, prepositional or 'type' qualifiers are left alone ('…, with anosmia')."""
    parts = term.split(", ")
    if len(parts) == 2:
        ok = 1 <= len(parts[1].split()) <= 2
    elif len(parts) == 3:
        ok = all(len(x.split()) == 1 for x in parts[1:])
    else:
        ok = False
    if not ok or not all(parts) or any(x[0].isdigit() for x in parts):
        return term
    if any(
        x.split()[0].lower() in FUNCTION_WORDS or "type" in x.lower().split() for x in parts[1:]
    ):
        return term
    first = parts[0].split(" ")[0]
    sci = SCI_NAME_RE.match(parts[0]) and " " in parts[0] and (taxa is None or first in taxa)
    return " ".join(parts if sci else reversed(parts))


def dictionary_form(
    term: str,
    *,
    proper: bool,
    eponym: bool,
    organism: bool = False,
    taxa: frozenset[str] | None = None,
    sentence_case: bool = False,
) -> str:
    if proper:
        return term
    orig = term.split(" ")
    title_case = sum(bool(re.match(r"[A-Z][a-z]", w)) for w in orig) >= 2
    words, lowered = [], []
    depth = 0
    for i, w in enumerate(orig):
        inside = depth > 0
        depth = max(0, depth + w.count("(") - w.count(")"))
        if sentence_case and (
            inside or "(" in w or "/" in w
        ):  # strain designations "(A/Hong Kong/45/2019 (H3N2))": verbatim
            words.append(w)
            lowered.append(False)
            continue
        # a genus followed by its species epithet keeps its capital ("Escherichia coli Infections"):
        # with the NCBI list, a capitalised non-English taxon name before a lower-case content word;
        # without it, the same shape inside a Title Case term (where only epithets are lower-case)
        nxt = orig[i + 1] if i + 1 < len(orig) else ""
        if (
            taxa is not None
        ):  # an English genus name ("Bacillus") needs a non-English epithet after it
            genus = (
                w in taxa
                and nxt.isalpha()
                and nxt.islower()
                and nxt not in FUNCTION_WORDS
                and not (known_word(w) and known_word(nxt))
            )
        else:
            genus = (
                title_case
                and w[:1].isupper()
                and nxt.isalpha()
                and nxt.islower()
                and nxt not in FUNCTION_WORDS
            )
        hy = w.split("-")
        # sentence-case terms (SCRs) capitalise hyphenated eponyms throughout ("Kocher-Debre-Semelaigne")
        eponym_compound = (
            sentence_case
            and len(hy) > 1
            and all(len(x) >= 3 and x.isalpha() and x[0].isupper() for x in hy)
            and not known_word(hy[0])
        )
        # in sentence-case SCR names only the first word's capital is uninformative: later
        # capitalised words are proper nouns ("Tomato yellow leaf curl Thailand virus"), and a
        # compound with digits is chemical nomenclature ("Val-Asp-8-Arg-vasopressin")
        sentence_keep = sentence_case and (
            (i > 0 and "-" not in w and w[:1].isupper())
            or ("-" in w and any(c.isdigit() for c in w))
        )
        if eponym_compound or sentence_keep:
            words.append(w)
            lowered.append(False)
            continue
        parts = []
        for p in hy:
            core = p.strip("()[],")  # "(Nitrogenous" is judged as "Nitrogenous", "(U.S.)" as "U.S."
            keep = (
                not core
                or KEEP_CASE_RE.search(core)
                or (len(core) == 1 and core.isupper())
                or genus
                or (eponym and i == 0)
                or (organism and latin_word(core, taxa))
            )
            parts.append(p if keep else p.lower())
        words.append("-".join(parts))
        lowered.append(parts[-1] != w.split("-")[-1] and w.split("-")[-1][:1].isupper())
    idx = words.index("of") - 1 if "of" in words[1:] else len(words) - 1
    head = words[idx].split("-")
    if (
        head[-1].isalpha() and head[-1].islower() and not {"and", "or"} & set(words)
    ):  # "Signs and Symptoms"
        # a Title Case word we lower-cased may use the OOV plural rules; a word that was already
        # lower-case (sentence-case SCRs, species epithets) only the dictionary
        head[-1] = singular(head[-1], oov_rules=lowered[idx])
        words[idx] = "-".join(head)
    return " ".join(words)


def choose_lemma(
    terms: list[tuple[str, str, bool]],
    *,
    proper: bool,
    organism: bool = False,
    scope: str = "",
    taxa: frozenset[str] | None = None,
    sentence_case: bool = False,
) -> tuple[str, list[str], str | None]:
    """terms: (string, lexical_tag, is_permuted) with the preferred term first.
    Returns (lemma, other terms, organism qualifier or None)."""
    pick = next((t for t in terms if ", " not in t[0] and not t[2]), terms[0])
    others = sorted({t[0] for t in terms if t[0] != pick[0]})
    text, qualifier = split_qualifier(pick[0]) if organism else (pick[0], None)
    if organism and scientific_name(text, scope, taxa):
        return text, others, qualifier
    if sentence_case and title_cased(
        text
    ):  # some SCR names (Orphanet-style) are Title Case after all
        sentence_case = False
    if (
        sentence_case
    ):  # the initial capital carries no information; drop it before un-inversion moves it
        w0, _, rest = text.partition(" ")
        if re.fullmatch(r"[A-Z][a-z]+,?", w0) and not latin_word(w0.rstrip(","), taxa):
            text = f"{w0.lower()} {rest}" if rest else w0.lower()
    if not proper and ", " in text:
        text = uninvert(text, taxa)
    lemma = dictionary_form(
        text,
        proper=proper or pick[1] in ("NAM", "TRD"),
        eponym=pick[1] == "EPO",
        organism=organism,
        taxa=taxa,
        sentence_case=sentence_case,
    )
    return lemma, others, qualifier


def concept_terms(record: Any) -> list[tuple[str, str, bool]]:
    concept = record.find("ConceptList/Concept[@PreferredConceptYN='Y']")
    if concept is None:
        return []
    out: list[tuple[str, str, bool]] = []
    for t in concept.findall("TermList/Term"):
        s = t.findtext("String")
        if not s:
            continue
        item = (s, t.get("LexicalTag") or "NON", t.get("IsPermutedTermYN") == "Y")
        if t.get("ConceptPreferredTermYN") == "Y":
            out.insert(0, item)
        else:
            out.append(item)
    return out


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    version = source_version(cfg, SOURCE)
    prov = Provenance(stage="ingest.mesh", config=cfg, source_versions={SOURCE: version})
    prov.add_input(cfg["descriptors"])
    tree_domains: dict[str, str] = cfg["tree_domains"]
    stats: Counter[str] = Counter()
    out = BatchWriter(cfg["output"], SENSE_SCHEMA, prov, batch_rows=100_000)
    keep_desc_classes = {str(c) for c in cfg["descriptor_classes"]}
    keep_scr_classes = {str(c) for c in cfg["scr_classes"]}
    taxa = None
    if cfg.get("taxon_names"):  # NCBI scientific names: genus detection for case (D-035)
        prov.add_input(cfg["taxon_names"])
        taxa = load_taxa(resolve(cfg["taxon_names"]))
        LOG.info("%d single-word NCBI taxon names", len(taxa))
    line = 0
    try:
        for rec in iterparse(resolve(cfg["descriptors"]), "DescriptorRecord"):
            cls = rec.get("DescriptorClass") or "1"
            if cls not in keep_desc_classes:
                continue
            ui = rec.findtext("DescriptorUI")
            name = rec.findtext("DescriptorName/String")
            terms = concept_terms(rec) or [(name, "NON", False)]
            proper = cls == "4"  # geographicals
            scope = (
                rec.findtext("ConceptList/Concept[@PreferredConceptYN='Y']/ScopeNote") or ""
            ).strip()
            trees = [t.text for t in rec.findall("TreeNumberList/TreeNumber") if t.text]
            organism = any(t.startswith("B") for t in trees)
            lemma, others, _ = choose_lemma(
                terms, proper=proper, organism=organism, scope=scope, taxa=taxa
            )
            if not scope:
                stats["drop:no_scope_note"] += 1
                continue
            letters = sorted({t[0] for t in trees})
            domain = tree_domains.get(letters[0], "medicine") if letters else "medicine"
            rel = {"heading": name, "entry_terms": others, "tree_numbers": trees}
            out.write(
                make_row(
                    source=SOURCE,
                    source_id=ui,
                    lemma=lemma,
                    pos="geographic" if proper else "descriptor",
                    gloss=" ".join(scope.split()),
                    tags=["mesh:descriptor"] + [f"mesh:tree:{x}" for x in letters],
                    domain=domain,
                    relations=orjson.dumps(rel, option=orjson.OPT_SORT_KEYS).decode(),
                    source_version=version,
                    kind="sense",
                    entry_key=ui,
                    line_no=line,
                    sense_rank=0,
                )
            )
            line += 1
            stats["descriptors"] += 1
            stats[f"descriptor_class:{cls}"] += 1
            stats["lemma_changed"] += lemma != name
        if cfg.get("supplementary"):
            prov.add_input(cfg["supplementary"])
            for rec in iterparse(resolve(cfg["supplementary"]), "SupplementalRecord"):
                cls = rec.get("SCRClass") or "1"
                if cls not in keep_scr_classes:
                    continue
                ui = rec.findtext("SupplementalRecordUI")
                name = rec.findtext("SupplementalRecordName/String")
                terms = concept_terms(rec) or [(name, "NON", False)]
                heads = [
                    h.findtext("DescriptorReferredTo/DescriptorName/String")
                    for h in rec.findall("HeadingMappedToList/HeadingMappedTo")
                ]
                heads = [h.lstrip("*") for h in heads if h]
                note = " ".join((rec.findtext("Note") or "").split())
                lemma, others, qual = choose_lemma(
                    terms,
                    proper=False,
                    organism=cls == "4",
                    scope=note,
                    taxa=taxa,
                    sentence_case=True,
                )
                kind_label = SCR_CLASSES.get(cls, "concept")
                label = f"{kind_label} ({qual})" if qual else kind_label
                gloss = f"{label} related to {', '.join(heads[:3])}" if heads else label
                if note and not UNINFORMATIVE_NOTE_RE.match(note):
                    gloss += f"; {note}"
                rel = {"heading": name, "entry_terms": others[:30], "mapped_to": heads}
                out.write(
                    make_row(
                        source=SOURCE,
                        source_id=ui,
                        lemma=lemma,
                        pos=f"scr:{kind_label}",
                        gloss=gloss,
                        tags=[f"mesh:scr:{kind_label}"],
                        domain=cfg["scr_domains"].get(kind_label, "medicine"),
                        relations=orjson.dumps(rel, option=orjson.OPT_SORT_KEYS).decode(),
                        source_version=version,
                        kind="sense",
                        entry_key=ui,
                        line_no=line,
                        sense_rank=0,
                    )
                )
                line += 1
                stats[f"scr:{kind_label}"] += 1
    except BaseException:
        out.abort()
        raise
    st = dict(sorted(stats.items()))
    out.close(stats=st)
    LOG.info("done: %s", st)
    return st


def main() -> None:
    args = stage_args("Ingest MeSH descriptors and supplementary concept records")
    run(load_config(args.config))


if __name__ == "__main__":
    main()
