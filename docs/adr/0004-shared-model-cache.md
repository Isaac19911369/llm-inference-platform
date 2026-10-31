# ADR 0004: Shared persistent model cache

**Status:** accepted

## Context
Model weights are 15 GB (8B in BF16) to more than 140 GB (70B). Downloading them on every pod start makes scale-out slow, hits NGC and Hugging Face rate limits, and fails in air-gapped or on-premise environments.

## Decision
Weights live on a `ReadWriteMany` PVC mounted at `/cache`, which is `NIM_CACHE_PATH` or `HF_HOME`. The PVC is kept when the chart is uninstalled. For air-gapped clusters, the cache is pre-populated once from a mirror.

## Consequences
- Cold start falls from a download plus load to just the load.
- It needs an RWX storage class, such as Azure Files, NFS or HPE GreenLake for File. The chart refuses to render more than one replica with a non-RWX cache.
- Cache contents must be versioned with the model profile, so that pods don't mix weights from different model versions.
