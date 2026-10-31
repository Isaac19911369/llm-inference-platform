# ADR 0003: Autoscale on queue depth, not GPU utilisation

**Status:** accepted

## Context
GPU utilisation is a poor scaling signal for LLM serving:

- A continuously batching engine shows high utilisation at modest load, so the signal saturates long before latency degrades.
- Utilisation says nothing about whether users are waiting.

The [load-test sweep](../../README.md#latency-vs-throughput) shows the real failure mode. Once the batch is full, extra requests queue, and time-to-first-token rises steeply while throughput stays flat.

## Decision
Scale with KEDA on `num_requests_waiting`, the engine's own queue, scraped by Prometheus. Scale up immediately, one pod per minute. Scale down slowly, after a 15-minute stabilisation window and then one pod per 5 minutes.

## Consequences
- Scaling follows user-visible latency.
- New replicas take minutes to become ready, because of the image pull, model load and engine warm-up. A shared model cache (ADR 0004) and over-provisioning `minReplicas` for known peaks mitigate this. Scaling is not a substitute for capacity planning.
- The threshold is set from load tests: the queue depth at which TTFT p95 crosses the SLO.
