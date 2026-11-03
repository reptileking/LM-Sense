"""Sense accuracy of the baseline encoder (spec 3.3) on SemCor and the Raganato et al. (2017)
unified WSD evaluation framework.

``python -m senselm.eval.wsd_eval --config configs/wsd_eval.yaml``

Gold WordNet 3.0 sense keys map to inventory token ids through the Phase-1 provenance: a key
``bank%1:17:01::`` is the OEWN sense ``oewn-bank__1.17.01..``, which is a member of exactly one
canonical sense (token). Keys OEWN 2025 no longer has are counted as errors and reported as
"unmapped". Three predictions per instance:

* ``mfs``: the inventory's most frequent sense for the gold lemma + POS (what the encoder does
  when its lemma and POS are right);
* ``wn_mfs``: WordNet's first sense for the lemma + POS (the textbook MFS baseline);
* ``encoder``: the sense token the baseline encoder emits for the word in context (its own spaCy
  lemma, POS and multi-word segmentation, run on the sentence detokenised with spaces).

SemEval-2013 and SemEval-2015 (and the framework's "ALL", which contains them) are skipped by
default: their WordNet keys were mapped from BabelNet annotations (D-049).
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import polars as pl

from senselm.common import Provenance, dump_json, load_config, resolve, stage_args

LOG = logging.getLogger("senselm.eval.wsd_eval")
POS_MAP = {
    "NOUN": "NOUN",
    "VERB": "VERB",
    "ADJ": "ADJ",
    "ADV": "ADV",
    "n": "NOUN",
    "v": "VERB",
    "a": "ADJ",
    "s": "ADJ",
    "r": "ADV",
}


def _escape(lemma: str) -> str:
    """OEWN's id escaping of lemma characters (english-wordnet scripts)."""
    out = []
    for c in lemma:
        if c.isascii() and (c.isalnum() or c == "."):
            out.append(c)
        elif c in (" ", "_"):
            out.append("_")
        else:
            out.append(
                {
                    "(": "-lb-",
                    ")": "-rb-",
                    "'": "-ap-",
                    "/": "-sl-",
                    ":": "-cl-",
                    ",": "-cm-",
                    "!": "-ex-",
                    "+": "-pl-",
                    "-": "--",
                }.get(c, f"-{ord(c):04x}-")
            )
    return "".join(out)


def oewn_id(sense_key: str) -> str:
    """``bank%1:17:01::`` -> ``oewn-bank__1.17.01..``"""
    lemma, rest = sense_key.split("%", 1)
    parts = rest.split(":")
    if len(parts) >= 4 and parts[3]:
        parts[3] = _escape(parts[3])
    return f"oewn-{_escape(lemma)}__{'.'.join(parts)}"


def read_dataset(
    xml_path: Path, key_path: Path
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Sentences as token lists with instance ids, and the gold keys."""
    sentences = []
    for _, el in ET.iterparse(str(xml_path), events=("end",)):
        if el.tag != "sentence":
            continue
        toks = []
        for w in el:
            if w.tag not in ("wf", "instance"):
                continue
            toks.append(
                {
                    "text": w.text or "",
                    "lemma": w.get("lemma", ""),
                    "pos": w.get("pos", ""),
                    "id": w.get("id") if w.tag == "instance" else None,
                }
            )
        sentences.append({"id": el.get("id"), "tokens": toks})
        el.clear()
    gold: dict[str, list[str]] = {}
    for line in key_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2:
            gold[parts[0]] = parts[1:]
    return sentences, gold


class SenseMaps:
    """OEWN sense id -> token id, and the two MFS predictors."""

    def __init__(self, inventory: str | Path) -> None:
        inv = pl.scan_parquet(resolve(inventory)).select(
            "token_id", "lemma", "upos", "sense_index", "oewn_rank", "members"
        )
        m = (
            inv.explode("members")
            .with_columns(
                pl.col("members").struct.field("source").alias("src"),
                pl.col("members").struct.field("source_id").alias("sid"),
            )
            .filter(pl.col("src") == "oewn")
            .select("sid", "token_id", "lemma", "upos", "oewn_rank")
            .collect()
        )
        self.token_of = dict(zip(m["sid"].to_list(), m["token_id"].to_list(), strict=True))
        wn = (
            m.sort("oewn_rank", "token_id")
            .group_by(pl.col("lemma").str.to_lowercase().alias("lo"), "upos", maintain_order=True)
            .agg(pl.col("token_id").first())
        )
        self.wn_first = {(lo, u): t for lo, u, t in wn.iter_rows()}
        mfs = (
            inv.select("token_id", "lemma", "upos", "sense_index")
            .collect()
            .sort("sense_index", "token_id")
            .group_by(pl.col("lemma").str.to_lowercase().alias("lo"), "upos", maintain_order=True)
            .agg(pl.col("token_id").first())
        )
        self.inv_first = {(lo, u): t for lo, u, t in mfs.iter_rows()}


def evaluate(cfg: dict[str, Any], *, use_encoder: bool = True) -> dict[str, Any]:
    root = resolve(cfg["framework_dir"])
    maps = SenseMaps(cfg["inventory"])
    codec = None
    if use_encoder:
        from senselm.codec.api import Codec

        codec = Codec(load_config(resolve(cfg["codec_config"])))
    results: dict[str, Any] = {}
    for name, rel in cfg["datasets"].items():
        xml_path, key_path = root / f"{rel}.data.xml", root / f"{rel}.gold.key.txt"
        if not xml_path.exists():
            results[name] = {"missing": str(xml_path)}
            continue
        sentences, gold = read_dataset(xml_path, key_path)
        res = score(sentences, gold, maps, codec, limit=cfg.get("max_sentences"))
        results[name] = res
        LOG.info("%s: %s", name, {k: v for k, v in res.items() if k != "by_pos"})
    return results


def score(
    sentences: list[dict[str, Any]],
    gold: dict[str, list[str]],
    maps: SenseMaps,
    codec: Any = None,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    n = mapped = 0
    hits = {"mfs": 0, "wn_mfs": 0, "encoder": 0}
    by_pos: dict[str, dict[str, int]] = {}
    for s in sentences[:limit] if limit else sentences:
        pred_enc: dict[str, int] = {}
        if codec is not None:
            pred_enc = encoder_predictions(s["tokens"], codec)
        for tok in s["tokens"]:
            iid = tok["id"]
            if iid is None or iid not in gold:
                continue
            n += 1
            golds = {maps.token_of.get(oewn_id(k)) for k in gold[iid]} - {None}
            mapped += bool(golds)
            upos = POS_MAP.get(tok["pos"], tok["pos"])
            lo = tok["lemma"].replace("_", " ").lower()
            p = by_pos.setdefault(upos, {"n": 0, "mfs": 0, "wn_mfs": 0, "encoder": 0})
            p["n"] += 1
            for kind, pred in (
                ("mfs", maps.inv_first.get((lo, upos))),
                ("wn_mfs", maps.wn_first.get((lo, upos))),
                ("encoder", pred_enc.get(iid)),
            ):
                if pred is not None and pred in golds:
                    hits[kind] += 1
                    p[kind] += 1
    out: dict[str, Any] = {
        "instances": n,
        "gold_mapped": mapped,
        "gold_mapped_rate": round(mapped / max(n, 1), 4),
    }
    for kind, h in hits.items():
        if kind == "encoder" and codec is None:
            continue
        out[f"acc_{kind}"] = round(h / max(n, 1), 4)
        out[f"acc_{kind}_on_mapped"] = round(h / max(mapped, 1), 4)
    out["by_pos"] = {
        k: {
            "n": v["n"],
            **{f"acc_{m}": round(v[m] / max(v["n"], 1), 4) for m in ("mfs", "wn_mfs", "encoder")},
        }
        for k, v in sorted(by_pos.items())
    }
    return out


def encoder_predictions(tokens: list[dict[str, Any]], codec: Any) -> dict[str, int]:
    """Instance id -> the sense token the encoder emits for the word covering it."""
    text, spans = "", []
    for t in tokens:
        if text:
            text += " "
        spans.append((len(text), len(text) + len(t["text"]), t["id"]))
        text += t["text"]
    enc = codec.encoder
    doc = enc.nlp(text)
    out: dict[str, int] = {}
    structural = codec.tables["structural"].info
    for it in enc.items(doc):
        if it.kind != "word":
            continue
        res = enc.encode_word(it, False)
        if res is None and len(it.toks) > 1:
            parts = enc._segment(it.toks, text, allow_mwe=False)
            pairs = [(p, enc.encode_word(p, False)) for p in parts if p.kind == "word"]
        else:
            pairs = [(it, res)]
        for item, r in pairs:
            if r is None:
                continue
            sense = next((i for i in r[0] if i not in structural), None)
            if sense is None:
                continue
            for a, b, iid in spans:
                if iid is not None and a < item.end and b > item.start:
                    out.setdefault(iid, sense)
    return out


def main() -> None:
    args = stage_args(__doc__.splitlines()[0])
    cfg = load_config(args.config)
    res = evaluate(cfg)
    prov = Provenance(stage="eval.wsd_eval", config=cfg)
    res["provenance"] = {
        "config_sha256": prov.record()["config_sha256"],
        "code": prov.record()["code"],
    }
    dump_json(res, resolve(cfg["output"]))
    for k, v in res.items():
        if k != "provenance":
            print(k, {a: b for a, b in v.items() if a != "by_pos"})


if __name__ == "__main__":
    main()
