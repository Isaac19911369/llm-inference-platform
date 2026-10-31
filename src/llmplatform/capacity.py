"""GPU capacity planning for LLM serving.

First-order model used for sizing before a load test:

* **Weights**: ``params x bytes_per_param`` (per tensor-parallel shard).
* **KV cache per token**: ``2 (K and V) x layers x kv_heads x head_dim x kv_bytes``.
  Grouped-query attention (few KV heads) is what makes long contexts affordable.
* **Concurrency**: the memory left after weights and runtime overhead, divided by
  the KV cache of one sequence at the planned context length.
* **Decode speed**: decoding is memory-bandwidth bound. Each step streams all
  weights once, so a single sequence decodes at most ``bandwidth / weight_bytes``
  tokens/s. Batching amortises the weight reads across sequences until compute or
  KV-cache reads dominate. ``efficiency`` covers what real kernels achieve.

These are planning numbers to choose GPU type, tensor parallelism and replica
count. They are always validated with ``llmplat bench`` against the real server.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

GIB = 1024**3


@dataclass(frozen=True)
class ModelSpec:
    name: str
    params_b: float
    layers: int
    kv_heads: int
    head_dim: int
    max_context: int


@dataclass(frozen=True)
class GPUSpec:
    name: str
    memory_gib: float
    bandwidth_gbs: float  # HBM/GDDR bandwidth, GB/s
    fp8: bool


MODELS = {
    m.name: m
    for m in [
        ModelSpec("llama-3.2-3b", 3.21, 28, 8, 128, 131072),
        ModelSpec("llama-3.1-8b", 8.03, 32, 8, 128, 131072),
        ModelSpec("mistral-7b", 7.25, 32, 8, 128, 32768),
        ModelSpec("llama-3.1-70b", 70.6, 80, 8, 128, 131072),
    ]
}

GPUS = {
    g.name: g
    for g in [
        GPUSpec("L4", 24, 300, True),
        GPUSpec("L40S", 48, 864, True),
        GPUSpec("A100-80GB", 80, 2039, False),
        GPUSpec("H100-SXM", 80, 3350, True),
        GPUSpec("H200", 141, 4800, True),
    ]
}

BYTES = {"fp16": 2.0, "bf16": 2.0, "fp8": 1.0, "int4": 0.5}


@dataclass
class Plan:
    model: str
    gpu: str
    tensor_parallel: int
    weight_dtype: str
    kv_dtype: str
    context_tokens: int
    weights_gib_per_gpu: float
    kv_kib_per_token: float
    kv_gib_per_sequence: float
    free_kv_gib: float
    max_concurrent_sequences: int
    single_stream_tokens_s: float
    fits: bool
    note: str = ""

    def as_row(self) -> dict:
        return asdict(self)


def kv_bytes_per_token(m: ModelSpec, kv_dtype: str = "fp16") -> float:
    return 2 * m.layers * m.kv_heads * m.head_dim * BYTES[kv_dtype]


def plan(
    model: str,
    gpu: str,
    tensor_parallel: int = 1,
    weight_dtype: str = "bf16",
    kv_dtype: str = "fp16",
    context_tokens: int = 8192,
    gpu_memory_utilization: float = 0.90,
    runtime_overhead_gib: float = 2.0,
    efficiency: float = 0.6,
) -> Plan:
    m, g = MODELS[model], GPUS[gpu]
    note = ""
    if weight_dtype == "fp8" and not g.fp8:
        note = f"{g.name} has no FP8 tensor cores; FP8 weights run via slower kernels"
    weights = m.params_b * 1e9 * BYTES[weight_dtype] / tensor_parallel
    usable = g.memory_gib * GIB * gpu_memory_utilization - runtime_overhead_gib * GIB
    free_kv = (usable - weights) * tensor_parallel  # KV cache is sharded across TP ranks
    kv_tok = kv_bytes_per_token(m, kv_dtype)
    ctx = min(context_tokens, m.max_context)
    kv_seq = kv_tok * ctx
    max_seq = max(0, math.floor(free_kv / kv_seq)) if free_kv > 0 else 0
    # bandwidth-bound decode: every step reads each weight shard once (TP shards in parallel)
    tps = efficiency * (g.bandwidth_gbs * 1e9) / weights
    return Plan(
        model=m.name,
        gpu=g.name,
        tensor_parallel=tensor_parallel,
        weight_dtype=weight_dtype,
        kv_dtype=kv_dtype,
        context_tokens=ctx,
        weights_gib_per_gpu=round(weights / GIB, 2),
        kv_kib_per_token=round(kv_tok / 1024, 1),
        kv_gib_per_sequence=round(kv_seq / GIB, 3),
        free_kv_gib=round(max(free_kv, 0) / GIB, 2),
        max_concurrent_sequences=max_seq,
        single_stream_tokens_s=round(tps, 1),
        fits=max_seq >= 1,
        note=note,
    )


def replicas_needed(
    p: Plan,
    requests_per_s: float,
    output_tokens: int,
    target_tokens_s_per_user: float = 20.0,
    batch_efficiency: float = 0.5,
) -> int:
    """Replicas so that every active request still streams at ``target_tokens_s_per_user``.

    Active requests per replica are limited both by KV-cache capacity and by the
    aggregate decode rate: batching ``B`` sequences gives roughly
    ``single_stream x (1 + batch_efficiency x (B - 1))`` tokens/s in the
    bandwidth-bound regime, shared by ``B`` users.
    """
    if not p.fits:
        raise ValueError("model does not fit; increase tensor parallelism or quantise")
    if p.single_stream_tokens_s < target_tokens_s_per_user:
        raise ValueError("even a single stream is slower than the per-user target")
    # Little's law: concurrent requests = arrival rate x time in system
    seconds_per_request = output_tokens / target_tokens_s_per_user
    concurrent = requests_per_s * seconds_per_request
    best = 1
    for b in range(1, p.max_concurrent_sequences + 1):
        aggregate = p.single_stream_tokens_s * (1 + batch_efficiency * (b - 1))
        if aggregate / b >= target_tokens_s_per_user:
            best = b
        else:
            break
    return max(1, math.ceil(concurrent / best))
