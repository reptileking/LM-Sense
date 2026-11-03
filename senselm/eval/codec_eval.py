"""Codec metrics on the general-text sample (spec 3.3): round-trip, coverage, sequence length.

``python -m senselm.eval.codec_eval --config configs/codec_eval.yaml [--split dev|eval] [--limit N]``

* **Surface round-trip**: text -> ids -> text, exact match per document and Levenshtein distance
  (characters) where it differs.
* **Coverage**: of the word tokens spaCy finds, the share written with a sense token vs a byte
  span, and why bytes were needed.
* **Length**: tokens per word and characters per token for the sense codec and, on the same
  text, the GPT-2 and Qwen2.5 BPE tokenizers (only to compare sequence lengths, D-047).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any

import polars as pl
from rapidfuzz.distance import Levenshtein
from tokenizers import Tokenizer

from senselm.codec.api import Codec
from senselm.codec.encoder import Trace
from senselm.common import Provenance, dump_json, load_config, resolve, setup_logging

LOG = logging.getLogger("senselm.eval.codec_eval")


def run(cfg: dict[str, Any], split: str, limit: int | None = None) -> dict[str, Any]:
    ccfg = load_config(resolve(cfg["codec_config"]))
    if "conventions" in cfg:
        ccfg["conventions"] = {**ccfg.get("conventions", {}), **cfg["conventions"]}
    t0 = time.time()
    codec = Codec(ccfg)
    assert codec.encoder is not None
    enc = codec.encoder
    enc.keep_oov = 1
    load_s = time.time() - t0
    docs = pl.read_parquet(resolve(cfg["corpus"])).filter(pl.col("split") == split)
    if limit:
        docs = docs.head(limit)
    texts = docs["text"].to_list()
    bpe = {name: Tokenizer.from_file(str(resolve(path))) for name, path in cfg["bpe"].items()}

    total = Trace()
    exact = 0
    mismatch: list[dict[str, Any]] = []
    edit_chars = 0
    n_ids = 0
    bpe_tokens: Counter[str] = Counter()
    t1 = time.time()
    for k, (ids, tr) in enumerate(
        enc.encode_many(texts, batch_size=int(cfg.get("batch_size", 32)), n_process=1)
    ):
        text = texts[k]
        back = codec.decode(ids)
        n_ids += len(ids)
        total.merge(tr)
        if back == text:
            exact += 1
        else:
            d = Levenshtein.distance(back, text)
            edit_chars += d
            if len(mismatch) < 20:
                i = next(
                    (j for j, (a, b) in enumerate(zip(back, text, strict=False)) if a != b),
                    min(len(back), len(text)),
                )
                mismatch.append(
                    {
                        "doc": k,
                        "edit": d,
                        "orig": text[max(0, i - 40) : i + 40],
                        "back": back[max(0, i - 40) : i + 40],
                    }
                )
        for name, tok in bpe.items():
            bpe_tokens[name] += len(tok.encode(text, add_special_tokens=False).ids)
        if (k + 1) % 200 == 0:
            LOG.info(
                "%d/%d docs, %.0f words/s",
                k + 1,
                len(texts),
                docs["words"].head(k + 1).sum() / (time.time() - t1),
            )
    enc_s = time.time() - t1
    chars = int(docs["chars"].sum())
    ws_words = int(docs["words"].sum())
    oov = total.oov.most_common(int(cfg.get("top_oov", 60)))
    res: dict[str, Any] = {
        "split": split,
        "docs": len(texts),
        "chars": chars,
        "whitespace_words": ws_words,
        "spacy_words": total.words,
        "conventions": ccfg.get("conventions", {}),
        "roundtrip": {
            "exact_docs": exact,
            "exact_rate": round(exact / max(len(texts), 1), 6),
            "edit_chars": edit_chars,
            "edit_per_1k_chars": round(1000 * edit_chars / max(chars, 1), 4),
            "examples": mismatch,
        },
        "coverage": {
            "words_sense": total.words_sense,
            "words_bytes": total.words_bytes,
            "sense_rate": round(total.words_sense / max(total.words, 1), 4),
            "items": dict(total.items.most_common()),
            "top_byte_words": oov,
        },
        "length": {
            "sense_codec_tokens": n_ids,
            "tokens_by_kind": dict(total.tokens.most_common()),
            "tokens_per_word": round(n_ids / max(ws_words, 1), 4),
            "chars_per_token": round(chars / max(n_ids, 1), 3),
            **{f"{name}_tokens": n for name, n in bpe_tokens.items()},
            **{
                f"{name}_tokens_per_word": round(n / max(ws_words, 1), 4)
                for name, n in bpe_tokens.items()
            },
            **{f"ratio_vs_{name}": round(n_ids / max(n, 1), 4) for name, n in bpe_tokens.items()},
        },
        "seconds": {
            "load": round(load_s, 1),
            "encode_decode": round(enc_s, 1),
            "words_per_s": round(ws_words / max(enc_s, 1e-9)),
        },
    }
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--split", default="eval", choices=["dev", "eval"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--auto-cap", action="store_true", help="override: sentence-start capitals by default"
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    if args.auto_cap:
        cfg["conventions"] = {**cfg.get("conventions", {}), "auto_cap": True}
    res = run(cfg, args.split, args.limit)
    out = (
        args.out
        or resolve(cfg["output_dir"])
        / f"codec_{args.split}{'_autocap' if args.auto_cap else ''}.json"
    )
    prov = Provenance(stage="eval.codec_eval", config=cfg)
    prov.add_input(cfg["corpus"])
    res["provenance"] = {
        "config_sha256": prov.record()["config_sha256"],
        "code": prov.record()["code"],
    }
    dump_json(res, out)
    print(json.dumps({k: v for k, v in res.items() if k not in ("provenance",)}, indent=1)[:4000])


if __name__ == "__main__":
    main()
