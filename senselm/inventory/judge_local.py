"""Judge uncertain alignment pairs with a locally hosted LLM (OpenAI-compatible server).

The judging machine (the owner's server) needs only this repository: no pipeline data.

``python -m senselm.inventory.judge_local <command> --config configs/judge_local.yaml``

* ``export``  (pipeline environment) write the judge job ``judge/job_<name>.parquet``: one row
  per request with its full text — the hand-labelled gold pairs (``kind = gold``, used by the
  pilot) and the decision-relevant band pairs (``kind = band``, see judge.decision_relevant).
* ``probe``   (judging machine) list the server's models, send one request, detect which
  structured-output mode works (json_schema → json_object → plain text), report latency.
* ``pilot``   judge the gold pairs; print agreement with the hand labels and the in-band
  precision of "same" verdicts. Verdicts are stored like any other.
* ``run``     judge the band pairs. Resumable: verdicts are appended as shards under
  ``judge/verdicts/<model>/``; restarting skips finished requests.
* ``import``  (pipeline environment) collect the shards into ``data/interim/judge_cache.parquet``,
  which ``align`` joins when ``judge_verdicts.use`` is true.

Works with Ollama (``http://localhost:11434/v1``), vLLM (``:8000/v1``), llama.cpp server
(``:8080/v1``), LM Studio (``:1234/v1``) and anything else implementing
``POST /v1/chat/completions``. Requests use temperature 0 and a fixed seed. Reasoning models
are supported (``<think>…</think>`` blocks and ``reasoning_content`` are ignored when parsing).
No money is spent; there is nothing to cap except time (``max_requests``, ``max_hours``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import polars as pl

from senselm.common import dump_json, load_config, resolve, setup_logging
from senselm.inventory.judge import OUTPUT_SCHEMA, SYSTEM_PROMPT, user_message

LOG = logging.getLogger("senselm.inventory.judge_local")
LABELS = ("same", "related", "different")

# Few-shot examples written for this prompt (none is taken from the gold set).
FEW_SHOT = """
Examples:
Word: mouse (NOUN)
A (WordNet): a hand-operated electronic device that controls the coordinates of a cursor on your computer screen
B (Wiktionary): An input device moved over a flat surface to move a pointer on a display.
-> {"label": "same", "confidence": 0.95}
Word: mouse (NOUN)
A (WordNet): any of numerous small rodents typically resembling diminutive rats
B (Wiktionary): A small rodent of the genus Mus.
-> {"label": "related", "confidence": 0.8}
Word: lamb (NOUN)
A (WordNet): young sheep
B (Wiktionary): The flesh of a lamb used as food.
-> {"label": "related", "confidence": 0.85}
Word: crane (NOUN)
A (WordNet): large long-necked wading bird of marshes and plains
B (Wiktionary): A machine used for lifting and moving heavy objects.
-> {"label": "different", "confidence": 0.98}
Word: sharp (ADJ)
A (WordNet): having or made by a thin edge or sharp point; suitable for cutting or piercing
B (Wiktionary): Intelligent; quick-witted.
-> {"label": "different", "confidence": 0.9}
"""
LOCAL_SYSTEM_PROMPT = (
    SYSTEM_PROMPT.replace("Answer with the JSON object only.", "")
    + FEW_SHOT
    + '\nAnswer with one JSON object: {"label": "same" | "related" | "different", '
    '"confidence": a number between 0 and 1}. No other text.'
)
PROMPT_VERSION = hashlib.blake2b(LOCAL_SYSTEM_PROMPT.encode(), digest_size=6).hexdigest()

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)
LABEL_RE = re.compile(r"\b(same|related|different)\b", re.IGNORECASE)


def job_id(oewn_id: str, wikt_id: str) -> str:
    return hashlib.blake2b(f"{oewn_id}|{wikt_id}".encode(), digest_size=10).hexdigest()


def parse_verdict(text: str) -> tuple[str | None, float | None]:
    """Extract (label, confidence) from a model reply; tolerant of chatter and think blocks."""
    text = THINK_RE.sub("", text or "").strip()
    for m in reversed(JSON_RE.findall(text)):  # the answer is usually the last object
        try:
            obj = json.loads(m)
        except json.JSONDecodeError:
            continue
        label = str(obj.get("label", "")).strip().lower()
        if label in LABELS:
            try:
                conf = float(obj.get("confidence"))
            except (TypeError, ValueError):
                conf = None
            return label, conf
    m = LABEL_RE.findall(text)  # last resort: a bare label word
    return (m[-1].lower(), None) if m else (None, None)


# --------------------------------------------------------------------------- HTTP client


class LocalLLM:
    """Minimal OpenAI-compatible chat client (stdlib only)."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.base = cfg["base_url"].rstrip("/")
        self.model: str | None = cfg.get("model")
        self.timeout = float(cfg.get("timeout_s", 120))
        self.mode = cfg.get("structured_output", "auto")
        key = os.environ.get(cfg.get("api_key_env", "LOCAL_LLM_API_KEY"), "")
        self.headers = {"Content-Type": "application/json"}
        if key:
            self.headers["Authorization"] = f"Bearer {key}"
        host = urlparse(self.base).hostname or ""
        local = (
            host in ("localhost", "::1")
            or host.startswith(("127.", "10.", "192.168.", "172."))
            or host.endswith(".local")
        )
        # never send local traffic through an outbound proxy
        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({})
            if local or cfg.get("no_proxy")
            else urllib.request.ProxyHandler()
        )
        self.max_tokens = int(cfg.get("max_tokens", 256))
        self.seed = cfg.get("seed", 0)
        self.retries = int(cfg.get("retries", 3))

    def _call(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            self.base + path, data=data, headers=self.headers, method=method
        )
        with self.opener.open(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def models(self) -> list[str]:
        return [m.get("id") for m in self._call("GET", "/models").get("data", [])]

    def response_format(self, mode: str) -> dict[str, Any] | None:
        if mode == "json_schema":
            return {
                "type": "json_schema",
                "json_schema": {"name": "verdict", "schema": OUTPUT_SCHEMA, "strict": True},
            }
        if mode == "json_object":
            return {"type": "json_object"}
        return None

    def chat(self, user: str, mode: str | None = None) -> dict[str, Any]:
        mode = mode or self.mode
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": LOCAL_SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "top_p": 1,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if self.seed is not None:
            body["seed"] = self.seed
        rf = self.response_format(mode)
        if rf:
            body["response_format"] = rf
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            t0 = time.time()
            try:
                out = self._call("POST", "/chat/completions", body)
                msg = out["choices"][0]["message"]
                return {
                    "text": msg.get("content") or "",
                    "finish_reason": out["choices"][0].get("finish_reason"),
                    "latency_s": round(time.time() - t0, 3),
                    "prompt_tokens": (out.get("usage") or {}).get("prompt_tokens"),
                    "completion_tokens": (out.get("usage") or {}).get("completion_tokens"),
                    "served_model": out.get("model"),
                }
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
                last = e
                time.sleep(min(30, 2**attempt))
        raise RuntimeError(f"request failed after {self.retries + 1} attempts: {last}")

    def detect_mode(self, sample_user: str) -> str:
        """First structured-output mode the server accepts and answers correctly."""
        for mode in ("json_schema", "json_object", "none"):
            try:
                r = self.chat(sample_user, mode)
            except RuntimeError as e:
                LOG.info("structured output %s not usable: %s", mode, e)
                continue
            if parse_verdict(r["text"])[0] is not None:
                return mode
        raise RuntimeError("the server answered no request with a parseable verdict")


# --------------------------------------------------------------------------- export / import


def export(cfg: dict[str, Any]) -> Path:
    from senselm.eval.alignment_gold import load_gold
    from senselm.inventory.judge import band_requests

    acfg = load_config(resolve(cfg["align_config"]))
    band = band_requests(acfg)
    band_rows = [
        {
            "job_id": job_id(r["oewn_id"], r["wikt_id"]),
            "kind": "band",
            "oewn_id": r["oewn_id"],
            "wikt_id": r["wikt_id"],
            "lemma": r["lemma"],
            "upos": r["upos"],
            "cosine": float(r["cosine"]),
            "gold_label": None,
            "message": user_message(r),
        }
        for r in band.iter_rows(named=True)
    ]
    gold = load_gold(resolve(cfg["gold_path"]))
    ids = gold["pair_id"].str.split("|")
    gold = gold.with_columns(ids.list.get(0).alias("oewn_id"), ids.list.get(1).alias("wikt_id"))
    pairs = pl.read_parquet(resolve(acfg["output"]), columns=["oewn_id", "wikt_id", "cosine"])
    gold = gold.join(pairs, on=["oewn_id", "wikt_id"], how="left")
    senses = pl.read_parquet(
        resolve(acfg["input"]), columns=["source_id", "upos", "gloss", "gloss_path", "examples"]
    )
    o = senses.select(
        pl.col("source_id").alias("oewn_id"),
        pl.col("gloss").alias("o_gloss"),
        pl.col("examples").alias("o_examples"),
    )
    w = senses.select(
        pl.col("source_id").alias("wikt_id"),
        "upos",
        pl.col("gloss").alias("w_gloss"),
        pl.col("gloss_path").list.last().alias("w_parent"),
        pl.col("examples").alias("w_examples"),
    )
    g = gold.join(o, on="oewn_id").join(w, on="wikt_id")
    gold_rows = [
        {
            "job_id": job_id(r["oewn_id"], r["wikt_id"]),
            "kind": "gold",
            "oewn_id": r["oewn_id"],
            "wikt_id": r["wikt_id"],
            "lemma": r["lemma"],
            "upos": r["upos"],
            "cosine": float(r["cosine"]),
            "gold_label": r["label"],
            "message": user_message(r),
        }
        for r in g.iter_rows(named=True)
    ]
    job = pl.DataFrame(gold_rows + band_rows).unique("job_id", keep="first", maintain_order=True)
    out = resolve(cfg["job_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    job.write_parquet(out, compression="zstd", compression_level=19)
    meta = {
        "rows": job.height,
        "gold": len(gold_rows),
        "band": len(band_rows),
        "band_def": acfg["judge_band"],
        "prompt_version": PROMPT_VERSION,
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    }
    dump_json(meta, out.with_suffix(".json"))
    LOG.info("wrote %s: %s", out, meta)
    return out


def model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("_")


def verdict_dir(cfg: dict[str, Any], model: str) -> Path:
    return resolve(cfg["verdicts_dir"]) / model_slug(model)


def load_verdicts(directory: Path) -> pl.DataFrame:
    parts = sorted(directory.glob("part-*.parquet"))
    if not parts:
        return pl.DataFrame(schema={"job_id": pl.String})
    return pl.concat([pl.read_parquet(p) for p in parts], how="diagonal_relaxed").unique(
        "job_id", keep="last"
    )


def import_verdicts(cfg: dict[str, Any]) -> Path:
    """Collect verdict shards of the configured model into the align stage's verdict cache."""
    vdir = verdict_dir(cfg, cfg["model"])
    v = load_verdicts(vdir)
    job = pl.read_parquet(resolve(cfg["job_path"]), columns=["job_id", "oewn_id", "wikt_id"])
    v = (
        v.join(job, on="job_id", how="inner")
        .select("oewn_id", "wikt_id", "verdict", "confidence", "model", "prompt_version")
        .drop_nulls("verdict")
    )
    out = resolve(cfg["cache_path"])
    v.write_parquet(out)
    LOG.info("imported %d verdicts from %s -> %s", v.height, vdir, out)
    return out


# --------------------------------------------------------------------------- judging


def judge_rows(
    cfg: dict[str, Any], rows: list[dict[str, Any]], client: LocalLLM, mode: str
) -> dict[str, Any]:
    if not client.model:
        raise ValueError("no model configured")
    vdir = verdict_dir(cfg, client.model)
    vdir.mkdir(parents=True, exist_ok=True)
    done = set(load_verdicts(vdir)["job_id"].to_list())
    todo = [r for r in rows if r["job_id"] not in done]
    if cfg.get("max_requests"):
        todo = todo[: int(cfg["max_requests"])]
    LOG.info(
        "%d requests (%d already judged), concurrency %d, mode %s",
        len(todo),
        len(done),
        int(cfg.get("concurrency", 4)),
        mode,
    )
    deadline = time.time() + float(cfg.get("max_hours") or 1e9) * 3600
    buf: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    t0 = time.time()

    def flush() -> None:
        if buf:
            n = len(list(vdir.glob("part-*.parquet")))
            pl.DataFrame(buf).write_parquet(vdir / f"part-{n:05d}.parquet")
            buf.clear()

    def one(r: dict[str, Any]) -> dict[str, Any]:
        res = client.chat(r["message"], mode)
        label, conf = parse_verdict(res["text"])
        return {
            "job_id": r["job_id"],
            "verdict": label,
            "confidence": conf,
            "raw": res["text"][:500],
            "finish_reason": res["finish_reason"],
            "latency_s": res["latency_s"],
            "prompt_tokens": res["prompt_tokens"],
            "completion_tokens": res["completion_tokens"],
            "model": client.model,
            "served_model": res["served_model"],
            "prompt_version": PROMPT_VERSION,
            "structured_output": mode,
        }

    with ThreadPoolExecutor(max_workers=int(cfg.get("concurrency", 4))) as pool:
        futures = {pool.submit(one, r): r for r in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                rec = fut.result()
            except RuntimeError as e:
                stats["failed"] += 1
                LOG.warning("request failed: %s", e)
                continue
            buf.append(rec)
            stats[rec["verdict"] or "unparsed"] += 1
            if len(buf) >= int(cfg.get("flush_every", 500)):
                flush()
            if i % 500 == 0 or i == len(todo):
                rate = i / max(time.time() - t0, 1e-6)
                LOG.info(
                    "  %d/%d judged (%.2f req/s, ETA %.1f h) %s",
                    i,
                    len(todo),
                    rate,
                    (len(todo) - i) / rate / 3600,
                    dict(stats),
                )
            if time.time() > deadline:
                LOG.warning("max_hours reached; stopping (resume later)")
                for f in futures:
                    f.cancel()
                break
    flush()
    elapsed = time.time() - t0
    return {
        "judged": sum(v for k, v in stats.items() if k != "failed"),
        "stats": dict(stats),
        "seconds": round(elapsed, 1),
        "req_per_s": round(sum(stats.values()) / max(elapsed, 1e-6), 3),
    }


def pilot_report(cfg: dict[str, Any], model: str) -> dict[str, Any]:
    """Agreement of the judge with the hand labels on the gold pairs."""
    job = pl.read_parquet(resolve(cfg["job_path"])).filter(pl.col("kind") == "gold")
    v = load_verdicts(verdict_dir(cfg, model)).select("job_id", "verdict")
    j = job.join(v, on="job_id", how="inner")
    lo, hi = cfg["band"]["low"], cfg["band"]["high"]
    res: dict[str, Any] = {
        "model": model,
        "n": j.height,
        "unparsed": int(j["verdict"].is_null().sum()),
    }
    jj = j.drop_nulls("verdict")
    res["accuracy_3way"] = (
        round(float((jj["verdict"] == jj["gold_label"]).mean()), 3) if jj.height else None
    )
    res["confusion"] = {
        f"{g}->{p}": n
        for g, p, n in jj.group_by("gold_label", "verdict")
        .len()
        .sort("gold_label", "verdict")
        .iter_rows()
    }

    def prf(df: pl.DataFrame) -> dict[str, Any]:
        y = (df["gold_label"] == "same").to_numpy()
        p = (df["verdict"] == "same").to_numpy()
        tp, fp, fn = int((y & p).sum()), int((~y & p).sum()), int((y & ~p).sum())
        return {
            "n": df.height,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(tp / (tp + fp), 3) if tp + fp else None,
            "recall": round(tp / (tp + fn), 3) if tp + fn else None,
        }

    res["same_vs_rest_all"] = prf(jj)
    band = jj.filter((pl.col("cosine") > lo) & (pl.col("cosine") < hi))
    res["same_vs_rest_in_band"] = prf(band)  # the judge's real job
    res["note"] = (
        "In the band the cosine aligner merges nothing (recall 0 there); judge merges are "
        "worth having when their precision is at least the aligner's (~0.9)."
    )
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Local-LLM judge for uncertain alignment pairs")
    ap.add_argument("command", choices=["export", "probe", "pilot", "run", "import"])
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--model", default=None, help="override the configured model name")
    ap.add_argument(
        "--base-url",
        default=None,
        help="override the server URL, e.g. http://localhost:8000/v1 (vLLM), :8080/v1 (llama.cpp), "
        ":1234/v1 (LM Studio); the default config points at Ollama (:11434/v1)",
    )
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    if args.model:
        cfg["model"] = args.model
    if args.base_url:
        cfg["base_url"] = args.base_url
    if args.command == "export":
        export(cfg)
        return
    if args.command == "import":
        import_verdicts(cfg)
        return
    client = LocalLLM(cfg)
    if args.command == "probe":
        models = client.models()
        print(json.dumps({"server": client.base, "models": models}, indent=2))
        if not client.model:
            raise SystemExit("set `model` in the config (or --model) to one of the models above")
        first = (
            pl.read_parquet(resolve(cfg["job_path"]))
            .filter(pl.col("kind") == "gold")
            .head(1)
            .row(0, named=True)
        )
        mode = client.detect_mode(first["message"]) if client.mode == "auto" else client.mode
        r = client.chat(first["message"], mode)
        print(
            json.dumps(
                {
                    "model": client.model,
                    "structured_output": mode,
                    "reply": r["text"][:300],
                    "verdict": parse_verdict(r["text"]),
                    "latency_s": r["latency_s"],
                },
                indent=2,
            )
        )
        return
    if not client.model:
        raise SystemExit("set `model` in the config (or --model)")
    model: str = client.model
    job = pl.read_parquet(resolve(cfg["job_path"]))
    kind = "gold" if args.command == "pilot" else "band"
    rows = job.filter(pl.col("kind") == kind).to_dicts()
    mode = client.mode
    if mode == "auto":
        mode = client.detect_mode(rows[0]["message"])
    summary = judge_rows(cfg, rows, client, mode)
    print(json.dumps(summary, indent=2))
    if args.command == "pilot":
        rep = pilot_report(cfg, model)
        out = resolve(cfg["pilot_dir"]) / f"judge_pilot_{model_slug(model)}.json"
        dump_json({**rep, "throughput": summary, "prompt_version": PROMPT_VERSION}, out)
        print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
