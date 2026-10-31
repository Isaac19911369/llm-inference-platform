"""Load tester for OpenAI-compatible chat endpoints (NIM, vLLM, TGI, Azure OpenAI, ...).

For every request it measures, from the client side:

* **TTFT**: time to first token, which covers queueing and prefill.
* **ITL**: mean inter-token latency, which is the decode speed each user sees.
* **End-to-end latency** and output tokens.

A run keeps ``concurrency`` requests in flight (closed loop) until
``num_requests`` complete. A sweep over concurrencies gives the latency vs
throughput curve used to pick an SLO-compliant operating point and the
autoscaling target.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass

import httpx
import numpy as np

_rng = np.random.default_rng(0)


@dataclass
class RequestResult:
    ok: bool
    ttft_s: float
    e2e_s: float
    output_tokens: int
    itl_s: float
    error: str | None = None


@dataclass
class RunSummary:
    concurrency: int
    requests: int
    errors: int
    duration_s: float
    request_throughput: float
    output_tokens_per_s: float
    ttft_p50_ms: float
    ttft_p95_ms: float
    itl_p50_ms: float
    itl_p95_ms: float
    e2e_p50_s: float
    e2e_p95_s: float

    def as_dict(self) -> dict:
        return asdict(self)


async def _one(
    client: httpx.AsyncClient, url: str, model: str, prompt: str, max_tokens: int, headers: dict
) -> RequestResult:
    body = {
        "model": model,
        "stream": True,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "stream_options": {"include_usage": True},
    }
    start = time.perf_counter()
    first = None
    stamps: list[float] = []
    usage_tokens = None
    try:
        async with client.stream("POST", url, json=body, headers=headers) as resp:
            if resp.status_code != 200:
                await resp.aread()
                return RequestResult(False, 0, 0, 0, 0, f"HTTP {resp.status_code}")
            async for line in resp.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                data = json.loads(line[6:])
                if data.get("usage"):
                    usage_tokens = data["usage"].get("completion_tokens")
                choices = data.get("choices") or []
                if choices and choices[0].get("delta", {}).get("content"):
                    now = time.perf_counter()
                    first = first or now
                    stamps.append(now)
    except httpx.HTTPError as exc:
        return RequestResult(False, 0, 0, 0, 0, type(exc).__name__)
    end = time.perf_counter()
    if first is None:
        return RequestResult(False, 0, end - start, 0, 0, "no tokens")
    itl = float(np.mean(np.diff(stamps))) if len(stamps) > 1 else 0.0
    return RequestResult(True, first - start, end - start, usage_tokens or len(stamps), itl)


async def run(
    base_url: str,
    concurrency: int,
    num_requests: int,
    model: str = "simulated-llm",
    prompt_tokens: int = 512,
    max_tokens: int = 128,
    api_key: str | None = None,
    timeout_s: float = 300.0,
    length_jitter: float = 0.5,
) -> RunSummary:
    """Closed-loop run. Output lengths vary by ``±length_jitter`` around ``max_tokens``
    so requests do not finish in lock-step (real traffic never does)."""
    url = base_url.rstrip("/") + "/chat/completions"
    prompt = "word " * prompt_tokens  # roughly one token per word for most tokenizers
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    results: list[RequestResult] = []
    remaining = num_requests
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)

    async with httpx.AsyncClient(timeout=timeout_s, limits=limits) as client:

        async def worker() -> None:
            nonlocal remaining
            while remaining > 0:
                remaining -= 1
                n = int(max_tokens * _rng.uniform(1 - length_jitter, 1 + length_jitter))
                results.append(await _one(client, url, model, prompt, max(1, n), headers))

        start = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(concurrency)))
        duration = time.perf_counter() - start

    ok = [r for r in results if r.ok]
    if not ok:
        raise RuntimeError(f"all {len(results)} requests failed: {results[0].error}")

    def pct(values: list[float], q: float) -> float:
        return float(np.percentile(values, q))

    ttft = [r.ttft_s * 1000 for r in ok]
    itl = [r.itl_s * 1000 for r in ok]
    e2e = [r.e2e_s for r in ok]
    return RunSummary(
        concurrency=concurrency,
        requests=len(results),
        errors=len(results) - len(ok),
        duration_s=round(duration, 2),
        request_throughput=round(len(ok) / duration, 2),
        output_tokens_per_s=round(sum(r.output_tokens for r in ok) / duration, 1),
        ttft_p50_ms=round(pct(ttft, 50), 1),
        ttft_p95_ms=round(pct(ttft, 95), 1),
        itl_p50_ms=round(pct(itl, 50), 2),
        itl_p95_ms=round(pct(itl, 95), 2),
        e2e_p50_s=round(pct(e2e, 50), 2),
        e2e_p95_s=round(pct(e2e, 95), 2),
    )


def max_concurrency_within_slo(
    rows: list[RunSummary], ttft_p95_ms: float, itl_p95_ms: float
) -> RunSummary | None:
    """Highest-throughput point of a sweep that still meets both latency SLOs."""
    ok = [
        r
        for r in rows
        if r.ttft_p95_ms <= ttft_p95_ms and r.itl_p95_ms <= itl_p95_ms and r.errors == 0
    ]
    return max(ok, key=lambda r: r.output_tokens_per_s) if ok else None
