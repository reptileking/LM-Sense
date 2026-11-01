"""Local-LLM judge client against a throwaway OpenAI-compatible test server."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from senselm.inventory import judge_local as jl


class FakeServer(BaseHTTPRequestHandler):
    reject_json_schema = False
    think = False
    calls: list[dict[str, Any]] = []

    def log_message(self, *a: Any) -> None:  # quiet
        pass

    def _send(self, code: int, obj: dict[str, Any]) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        self._send(200, {"data": [{"id": "fake-7b"}]})

    def do_POST(self) -> None:  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeServer.calls.append(body)
        rf = (body.get("response_format") or {}).get("type")
        if rf == "json_schema" and FakeServer.reject_json_schema:
            self._send(400, {"error": "json_schema not supported"})
            return
        user = body["messages"][-1]["content"]
        label = "same" if "SAME" in user else "different"
        text = json.dumps({"label": label, "confidence": 0.9})
        if FakeServer.think:
            text = '<think>weighing {"label": "related"} options</think>\n' + text
        self._send(
            200,
            {
                "model": body["model"],
                "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 12},
            },
        )


@pytest.fixture()
def server() -> Iterator[str]:
    FakeServer.calls = []
    FakeServer.reject_json_schema = False
    FakeServer.think = False
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeServer)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    httpd.shutdown()


def make_cfg(tmp_path: Path, base_url: str) -> dict[str, Any]:
    job = pl.DataFrame(
        {
            "job_id": [f"j{i}" for i in range(6)],
            "kind": ["gold", "gold", "gold", "band", "band", "band"],
            "oewn_id": [f"o{i}" for i in range(6)],
            "wikt_id": [f"w{i}" for i in range(6)],
            "lemma": ["x"] * 6,
            "upos": ["NOUN"] * 6,
            "cosine": [0.60, 0.62, 0.90, 0.6, 0.6, 0.6],
            "gold_label": ["same", "different", "same", None, None, None],
            "message": ["SAME a", "b", "c", "SAME d", "e", "SAME f"],
        }
    )
    jp = tmp_path / "job.parquet"
    job.write_parquet(jp)
    return {
        "base_url": base_url,
        "model": "fake-7b",
        "job_path": str(jp),
        "verdicts_dir": str(tmp_path / "v"),
        "cache_path": str(tmp_path / "cache.parquet"),
        "pilot_dir": str(tmp_path),
        "concurrency": 3,
        "flush_every": 2,
        "retries": 0,
        "timeout_s": 10,
        "structured_output": "auto",
        "band": {"low": 0.55, "high": 0.69},
    }


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"label": "same", "confidence": 0.8}', ("same", 0.8)),
        ('Sure! {"label": "Related", "confidence": "0.6"}', ("related", 0.6)),
        (
            '<think>maybe {"label": "same"}</think>{"label": "different", "confidence": 1}',
            ("different", 1.0),
        ),
        ("The answer is: different.", ("different", None)),
        ("no idea", (None, None)),
    ],
)
def test_parse_verdict(text: str, expected: tuple[str | None, float | None]) -> None:
    assert jl.parse_verdict(text) == expected


def test_probe_detects_structured_output_fallback(tmp_path: Path, server: str) -> None:
    client = jl.LocalLLM(make_cfg(tmp_path, server))
    assert client.models() == ["fake-7b"]
    assert client.detect_mode("SAME x") == "json_schema"
    FakeServer.reject_json_schema = True
    assert client.detect_mode("SAME x") == "json_object"
    body = FakeServer.calls[-1]
    assert (
        body["temperature"] == 0 and body["seed"] == 0 and body["messages"][0]["role"] == "system"
    )


def test_run_is_resumable_and_pilot_scores(tmp_path: Path, server: str) -> None:
    cfg = make_cfg(tmp_path, server)
    FakeServer.think = True  # reasoning-model style replies must still parse
    client = jl.LocalLLM(cfg)
    job = pl.read_parquet(cfg["job_path"])
    gold_rows = job.filter(pl.col("kind") == "gold").to_dicts()
    s1 = jl.judge_rows(cfg, gold_rows, client, "json_schema")
    assert s1["judged"] == 3
    n_calls = len(FakeServer.calls)
    s2 = jl.judge_rows(cfg, gold_rows, client, "json_schema")  # nothing left to do
    assert s2["judged"] == 0 and len(FakeServer.calls) == n_calls
    rep = jl.pilot_report(cfg, "fake-7b")
    assert rep["n"] == 3 and rep["unparsed"] == 0
    # j0: gold same -> same (tp, in band); j1: gold different -> different; j2: gold same -> different (fn)
    assert rep["same_vs_rest_all"] == {
        "n": 3,
        "tp": 1,
        "fp": 0,
        "fn": 1,
        "precision": 1.0,
        "recall": 0.5,
    }
    assert rep["same_vs_rest_in_band"]["n"] == 2
    band_rows = job.filter(pl.col("kind") == "band").to_dicts()
    jl.judge_rows(cfg, band_rows, client, "json_schema")
    out = jl.import_verdicts(cfg)
    cache = pl.read_parquet(out)
    assert cache.height == 6
    assert dict(zip(cache["wikt_id"], cache["verdict"], strict=True))["w3"] == "same"
    assert set(cache.columns) >= {"oewn_id", "wikt_id", "verdict", "model", "prompt_version"}
