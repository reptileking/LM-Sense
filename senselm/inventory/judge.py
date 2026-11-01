"""LLM judge for the uncertain cosine band of the aligner (disabled by default).

``python -m senselm.inventory.judge --config configs/align.yaml``             (estimate only)
``python -m senselm.inventory.judge --config configs/align.yaml --execute``   (spend money)

Spending money requires all three: ``llm_judge.enabled: true``, ``llm_judge.dry_run: false``
and ``--execute``. Every run first prints a cost estimate.

* Pairs judged: candidate pairs with ``judge_band.low < cosine < judge_band.high`` that are
  not already in the verdict cache (``llm_judge.cache_path``); with ``relevant_only`` (default)
  only those whose verdict can change the fine policy's outcome (both senses not yet merged,
  pair among the top-k candidates of one of its senses).
* Requests go through the Message Batches API (50% price) in chunks of
  ``chunk_requests``. Before each chunk the **hard cap** ``max_spend_gbp`` is checked
  against: actual spend so far + the chunk's *worst case* (every request using
  ``max_tokens`` output tokens). Nothing is submitted that could break the cap.
* Output is constrained with ``output_config.format`` (JSON schema): ``{"label":
  "same|related|different", "confidence": 0..1}``. A ``refusal`` stop reason or an errored
  request yields no verdict (the pair stays separate). Server-side refusal fallbacks are not
  available on the Batches API, so none are requested.
* Input tokens are measured with ``messages.count_tokens`` on a sample when credentials are
  available; otherwise a conservative heuristic (``chars_per_token`` = 3.0) is used and the
  estimate says so.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import polars as pl

from senselm.common import dump_json, load_config, resolve, setup_logging
from senselm.inventory.judge_prompt import OUTPUT_SCHEMA, SYSTEM_PROMPT, user_message
from senselm.inventory.policy import assign_clusters

LOG = logging.getLogger("senselm.inventory.judge")


def custom_id(oewn_id: str, wikt_id: str) -> str:
    return hashlib.blake2b(f"{oewn_id}|{wikt_id}".encode(), digest_size=16).hexdigest()


@dataclass
class Estimate:
    model: str
    n_requests: int
    input_tokens: int
    expected_output_tokens: int
    max_output_tokens: int
    expected_usd: float
    worst_case_usd: float
    expected_gbp: float
    worst_case_gbp: float
    cap_gbp: float
    batch_pricing: bool
    token_method: str


def prices(jcfg: dict[str, Any]) -> tuple[float, float]:
    """USD per token (input, output), batch discount applied when batching."""
    p = jcfg["prices_usd_per_mtok"][jcfg["model"]]
    f = float(jcfg.get("batch_discount", 0.5)) if jcfg.get("use_batches", True) else 1.0
    return p["input"] * f / 1e6, p["output"] * f / 1e6


def estimate(messages: list[str], jcfg: dict[str, Any], client: Any | None = None) -> Estimate:
    pin, pout = prices(jcfg)
    n = len(messages)
    if client is not None and n:
        sample = messages[: min(50, n)]
        counts = [
            client.messages.count_tokens(
                model=jcfg["model"], system=SYSTEM_PROMPT, messages=[{"role": "user", "content": m}]
            ).input_tokens
            for m in sample
        ]
        input_tokens = int(sum(counts) / len(counts) * n)
        method = f"count_tokens on {len(sample)} requests, extrapolated"
    else:
        cpt = float(jcfg.get("chars_per_token", 3.0))
        input_tokens = int(sum(len(SYSTEM_PROMPT) + len(m) for m in messages) / cpt)
        method = f"heuristic: {cpt} characters per token (conservative; no API credentials)"
    exp_out = int(jcfg.get("expected_output_tokens", 400)) * n
    max_out = int(jcfg["max_tokens"]) * n
    exp_usd = input_tokens * pin + exp_out * pout
    worst_usd = input_tokens * pin + max_out * pout
    fx = float(jcfg["usd_per_gbp"])
    return Estimate(
        jcfg["model"],
        n,
        input_tokens,
        exp_out,
        max_out,
        round(exp_usd, 2),
        round(worst_usd, 2),
        round(exp_usd / fx, 2),
        round(worst_usd / fx, 2),
        float(jcfg["max_spend_gbp"]),
        bool(jcfg.get("use_batches", True)),
        method,
    )


def decision_relevant(pairs: pl.DataFrame, cfg: dict[str, Any]) -> pl.DataFrame:
    """Band pairs whose verdict can change the fine policy's outcome.

    Under one-to-one matching a verdict is ignored when either sense is already merged by an
    auto-merge edge, and only an endpoint's best remaining candidates can win the greedy match.
    Keeps band pairs with both endpoints unmatched that rank within ``top_k`` (by cosine) for
    their OEWN sense or for their Wiktionary sense.
    """
    jcfg = cfg["llm_judge"]
    policy = load_config(resolve(jcfg["policy_config"]))["policy"]
    senses = pl.read_parquet(
        resolve(cfg["input"]),
        columns=[
            "source",
            "source_id",
            "entry_key",
            "gloss",
            "gloss_path",
            "line_no",
            "sense_rank",
        ],
    )
    clusters, _ = assign_clusters(senses, pairs, {**policy, "judge_accept": []})
    src = clusters.join(senses.select("source_id", "source"), on="source_id", how="left")
    mixed = (
        src.group_by("cluster").agg(pl.col("source").n_unique().alias("n")).filter(pl.col("n") > 1)
    )
    matched = set(src.join(mixed, on="cluster")["source_id"].to_list())
    band = cfg["judge_band"]
    k = int(jcfg.get("top_k", 1))
    sel = pairs.filter(
        (pl.col("cosine") > band["low"])
        & (pl.col("cosine") < band["high"])
        & ~pl.col("oewn_id").is_in(list(matched))
        & ~pl.col("wikt_id").is_in(list(matched))
    )
    sel = sel.with_columns(
        pl.col("cosine").rank("ordinal", descending=True).over("oewn_id").alias("r_o"),
        pl.col("cosine").rank("ordinal", descending=True).over("wikt_id").alias("r_w"),
    )
    return sel.filter((pl.col("r_o") <= k) | (pl.col("r_w") <= k)).drop("r_o", "r_w")


def band_requests(cfg: dict[str, Any]) -> pl.DataFrame:
    """Pairs to judge (minus cached verdicts), with the text of both senses."""
    band = cfg["judge_band"]
    pairs = pl.read_parquet(resolve(cfg["output"]))
    if cfg["llm_judge"].get("relevant_only", True):
        sel = decision_relevant(pairs.sort("group_key", "oewn_id", "wikt_id"), cfg)
    else:
        sel = pairs.filter((pl.col("cosine") > band["low"]) & (pl.col("cosine") < band["high"]))
    cache = resolve(cfg["llm_judge"]["cache_path"])
    if cache.exists():
        sel = sel.join(
            pl.read_parquet(cache).select("oewn_id", "wikt_id"),
            on=["oewn_id", "wikt_id"],
            how="anti",
        )
    s = pl.read_parquet(
        resolve(cfg["input"]), columns=["source_id", "upos", "gloss", "gloss_path", "examples"]
    )
    o = s.select(
        pl.col("source_id").alias("oewn_id"),
        pl.col("gloss").alias("o_gloss"),
        pl.col("examples").alias("o_examples"),
    )
    w = s.select(
        pl.col("source_id").alias("wikt_id"),
        "upos",
        pl.col("gloss").alias("w_gloss"),
        pl.col("gloss_path").list.last().alias("w_parent"),
        pl.col("examples").alias("w_examples"),
    )
    return sel.join(o, on="oewn_id").join(w, on="wikt_id").sort("oewn_id", "wikt_id")


def _client() -> Any | None:
    try:
        import anthropic
    except ImportError:
        return None
    try:
        client = anthropic.Anthropic()
        client.models.retrieve("claude-opus-5-5")  # cheap credential check
        return client
    except Exception as e:  # no credentials / no network: estimate heuristically
        LOG.info(
            "no usable Anthropic credentials (%s); using heuristic token counts", type(e).__name__
        )
        return None


def judge_chunk(
    client: Any, rows: list[dict[str, Any]], jcfg: dict[str, Any]
) -> tuple[list[dict[str, Any]], float]:
    """Submit one batch, wait, parse. Returns (verdict rows, actual USD cost)."""
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    pin, pout = prices(jcfg)
    reqs = [
        Request(
            custom_id=custom_id(r["oewn_id"], r["wikt_id"]),
            params=MessageCreateParamsNonStreaming(
                model=jcfg["model"],
                max_tokens=int(jcfg["max_tokens"]),
                system=[
                    {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}
                ],
                messages=[{"role": "user", "content": user_message(r)}],
                output_config={
                    "effort": jcfg.get("effort", "low"),
                    "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA},
                },
            ),
        )
        for r in rows
    ]
    batch = client.messages.batches.create(requests=reqs)
    LOG.info("submitted batch %s (%d requests)", batch.id, len(reqs))
    while True:
        batch = client.messages.batches.retrieve(batch.id)
        if batch.processing_status == "ended":
            break
        time.sleep(int(jcfg.get("poll_seconds", 60)))
    by_id = {custom_id(r["oewn_id"], r["wikt_id"]): r for r in rows}
    out, cost = [], 0.0
    for res in client.messages.batches.results(batch.id):
        r = by_id[res.custom_id]
        verdict, conf = None, None
        if res.result.type == "succeeded":
            msg = res.result.message
            u = msg.usage
            cost += (
                u.input_tokens * pin
                + u.output_tokens * pout
                + (u.cache_creation_input_tokens or 0) * pin * 1.25
                + (u.cache_read_input_tokens or 0) * pin * 0.1
            )
            if msg.stop_reason not in ("refusal", "max_tokens"):
                text = next((b.text for b in msg.content if b.type == "text"), "")
                try:
                    data = json.loads(text)
                    verdict, conf = data["label"], float(data["confidence"])
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    pass
        out.append(
            {
                "oewn_id": r["oewn_id"],
                "wikt_id": r["wikt_id"],
                "verdict": verdict,
                "confidence": conf,
                "model": jcfg["model"],
                "batch_id": batch.id,
                "result": res.result.type,
            }
        )
    return out, cost


def run(cfg: dict[str, Any], execute: bool = False, client: Any | None = None) -> dict[str, Any]:
    jcfg = cfg["llm_judge"]
    req = band_requests(cfg)
    msgs = [user_message(r) for r in req.iter_rows(named=True)]
    live = (
        client
        if client is not None
        else (_client() if (execute or jcfg.get("count_tokens", True)) else None)
    )
    est = estimate(msgs, jcfg, live)
    report: dict[str, Any] = {"estimate": asdict(est), "band": cfg["judge_band"], "executed": False}
    LOG.info("judge estimate: %s", report["estimate"])
    if not (execute and jcfg.get("enabled") and not jcfg.get("dry_run", True)):
        LOG.info("dry run: nothing submitted (needs enabled: true, dry_run: false and --execute)")
        return report
    if live is None:
        raise RuntimeError("no Anthropic credentials available")
    fx, cap = float(jcfg["usd_per_gbp"]), float(jcfg["max_spend_gbp"])
    pin, pout = prices(jcfg)
    spent_usd, rows_out = 0.0, []
    rows = req.to_dicts()
    chunk = int(jcfg.get("chunk_requests", 2000))
    cache_path = resolve(jcfg["cache_path"])
    for s in range(0, len(rows), chunk):
        part = rows[s : s + chunk]
        worst = (
            sum(len(SYSTEM_PROMPT) + len(m) for m in msgs[s : s + chunk]) / 3.0 * pin
            + len(part) * int(jcfg["max_tokens"]) * pout
        )
        if (spent_usd + worst) / fx > cap:
            LOG.warning(
                "stopping: next chunk could exceed the £%.2f cap (spent £%.2f, chunk worst case £%.2f); %d pairs left",
                cap,
                spent_usd / fx,
                worst / fx,
                len(rows) - s,
            )
            report["stopped_at_cap"] = True
            break
        verdicts, cost = judge_chunk(live, part, jcfg)
        spent_usd += cost
        rows_out.extend(verdicts)
        old = pl.read_parquet(cache_path) if cache_path.exists() else None
        new = pl.DataFrame(rows_out)
        (pl.concat([old, new], how="diagonal_relaxed") if old is not None else new).unique(
            ["oewn_id", "wikt_id"], keep="last"
        ).write_parquet(cache_path)
        LOG.info("chunk done: actual £%.2f so far", spent_usd / fx)
    report.update(executed=True, spent_gbp=round(spent_usd / fx, 2), judged=len(rows_out))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM judge for the uncertain alignment band")
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--execute", action="store_true", help="actually submit requests (costs money)")
    ap.add_argument("--report", type=Path, default=Path("data/interim/judge_estimate.json"))
    args = ap.parse_args()
    setup_logging()
    report = run(load_config(args.config), execute=args.execute)
    dump_json(report, resolve(args.report))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
