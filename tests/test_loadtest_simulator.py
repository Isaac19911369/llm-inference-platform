import asyncio
import socket
import threading
import time

import httpx
import pytest
import uvicorn

from llmplatform.loadtest import RunSummary, max_concurrency_within_slo, run
from llmplatform.simulator import PerfModel, create_app


@pytest.fixture(scope="module")
def server():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    perf = PerfModel(decode_ms=2.0, decode_ms_per_seq=0.2, prefill_ms_per_token=0.005, max_batch=8)
    config = uvicorn.Config(create_app(perf), host="127.0.0.1", port=port, log_level="error")
    srv = uvicorn.Server(config)
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    for _ in range(100):
        try:
            httpx.get(f"http://127.0.0.1:{port}/health", timeout=0.2)
            break
        except httpx.HTTPError:
            time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    t.join(timeout=5)


def test_non_streaming_completion(server):
    r = httpx.post(
        f"{server}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 5},
    )
    body = r.json()
    assert body["usage"]["completion_tokens"] == 5
    assert body["choices"][0]["message"]["content"].count("tok") == 5


def test_load_test_measures_queueing(server):
    low = asyncio.run(
        run(
            f"{server}/v1",
            concurrency=1,
            num_requests=4,
            prompt_tokens=50,
            max_tokens=20,
            length_jitter=0,
        )
    )
    high = asyncio.run(
        run(
            f"{server}/v1",
            concurrency=24,
            num_requests=48,
            prompt_tokens=50,
            max_tokens=20,
            length_jitter=0,
        )
    )
    assert low.errors == high.errors == 0
    assert low.requests == 4 and high.requests == 48
    # batching raises throughput ...
    assert high.output_tokens_per_s > 2 * low.output_tokens_per_s
    # ... but with only 8 batch slots, 24 concurrent users must queue
    assert high.ttft_p95_ms > 3 * low.ttft_p95_ms


def test_metrics_endpoint(server):
    text = httpx.get(f"{server}/metrics").text
    assert "vllm:num_requests_waiting" in text
    assert "vllm:generation_tokens_total" in text


def _row(c, ttft, itl, tps):
    return RunSummary(c, 10, 0, 1, 1, tps, ttft, ttft, itl, itl, 1, 1)


def test_slo_selection():
    rows = [_row(1, 50, 10, 100), _row(8, 300, 20, 600), _row(32, 2000, 45, 900)]
    assert max_concurrency_within_slo(rows, 1000, 50).concurrency == 8
    assert max_concurrency_within_slo(rows, 10, 5) is None
