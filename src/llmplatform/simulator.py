"""An OpenAI-compatible LLM server *simulator* with continuous batching.

It reproduces the queueing behaviour of real engines (vLLM, NIM/TensorRT-LLM)
without a GPU, so the load tester, dashboards and autoscaling rules can be
developed and tested in CI:

* requests wait in a queue until a batch slot (KV-cache capacity) is free;
* every engine step decodes one token for every running sequence;
* step time grows with batch size, and a new sequence's prompt is prefilled in
  the step it joins, which delays everyone else's next token (the classic
  TTFT / inter-token-latency trade-off).

It also exposes Prometheus metrics with the same names vLLM uses.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel


@dataclass
class PerfModel:
    decode_ms: float = 12.0  # step time with a single sequence
    decode_ms_per_seq: float = 0.35  # extra step time per additional running sequence
    prefill_ms_per_token: float = 0.08  # prompt processing cost, paid in the joining step
    max_batch: int = 64  # running sequences (KV-cache slots)
    time_scale: float = 1.0  # <1 to run faster than real time in tests

    def step_seconds(self, running: int, prefill_tokens: int) -> float:
        ms = self.decode_ms + self.decode_ms_per_seq * max(running - 1, 0)
        ms += self.prefill_ms_per_token * prefill_tokens
        return ms / 1000 * self.time_scale


@dataclass
class _Seq:
    prompt_tokens: int
    max_tokens: int
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    generated: int = 0
    prefilled: bool = False


class Engine:
    def __init__(self, perf: PerfModel):
        self.perf = perf
        self.waiting: list[_Seq] = []
        self.running: list[_Seq] = []
        self.wake = asyncio.Event()
        self.counters = {
            "prompt_tokens_total": 0,
            "generation_tokens_total": 0,
            "request_success_total": 0,
        }
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.get_running_loop().create_task(self._loop())

    async def submit(self, seq: _Seq) -> None:
        self.waiting.append(seq)
        self.counters["prompt_tokens_total"] += seq.prompt_tokens
        self.wake.set()

    async def _loop(self) -> None:
        while True:
            if not self.waiting and not self.running:
                self.wake.clear()
                await self.wake.wait()
            # admit waiting sequences into free batch slots (FCFS)
            while self.waiting and len(self.running) < self.perf.max_batch:
                self.running.append(self.waiting.pop(0))
            prefill = sum(s.prompt_tokens for s in self.running if not s.prefilled)
            await asyncio.sleep(self.perf.step_seconds(len(self.running), prefill))
            still = []
            for s in self.running:
                s.prefilled = True
                s.generated += 1
                self.counters["generation_tokens_total"] += 1
                done = s.generated >= s.max_tokens
                s.queue.put_nowait(done)
                if done:
                    self.counters["request_success_total"] += 1
                else:
                    still.append(s)
            self.running = still

    def metrics(self) -> str:
        lines = [
            f"vllm:num_requests_running {len(self.running)}",
            f"vllm:num_requests_waiting {len(self.waiting)}",
            f"vllm:gpu_cache_usage_perc {len(self.running) / self.perf.max_batch:.4f}",
        ]
        lines += [f"vllm:{k} {v}" for k, v in self.counters.items()]
        return "\n".join(lines) + "\n"


class ChatRequest(BaseModel):
    model: str = "simulated-llm"
    messages: list[dict]
    max_tokens: int = 128
    stream: bool = False


def _prompt_tokens(messages: list[dict]) -> int:
    # ~4 characters per token is the usual rule of thumb for English text
    return max(1, sum(len(str(m.get("content", ""))) for m in messages) // 4)


def create_app(perf: PerfModel | None = None) -> FastAPI:
    engine = Engine(perf or PerfModel())
    app = FastAPI(title="LLM simulator")
    app.state.engine = engine

    @app.get("/v1/models")
    async def models() -> dict:
        return {"object": "list", "data": [{"id": "simulated-llm", "object": "model"}]}

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics() -> str:
        return engine.metrics()

    @app.post("/v1/chat/completions")
    async def chat(req: ChatRequest):
        engine.start()
        seq = _Seq(_prompt_tokens(req.messages), max(1, req.max_tokens))
        await engine.submit(seq)
        rid, created = f"chatcmpl-{uuid.uuid4().hex[:12]}", int(time.time())

        def chunk(delta: dict, finish: str | None = None, usage: dict | None = None) -> str:
            body = {
                "id": rid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req.model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            if usage:
                body["usage"] = usage
            return f"data: {json.dumps(body)}\n\n"

        usage = {
            "prompt_tokens": seq.prompt_tokens,
            "completion_tokens": seq.max_tokens,
            "total_tokens": seq.prompt_tokens + seq.max_tokens,
        }
        if not req.stream:
            while not await seq.queue.get():
                pass
            return {
                "id": rid,
                "object": "chat.completion",
                "created": created,
                "model": req.model,
                "usage": usage,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "length",
                        "message": {"role": "assistant", "content": "tok " * seq.max_tokens},
                    }
                ],
            }

        async def stream():
            yield chunk({"role": "assistant"})
            while True:
                done = await seq.queue.get()
                yield chunk({"content": "tok "})
                if done:
                    break
            yield chunk({}, "length", usage)
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app
