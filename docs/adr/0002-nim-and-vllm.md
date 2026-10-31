# ADR 0002: Support NVIDIA NIM and vLLM behind one chart

**Status:** accepted

## Context
NVIDIA NIM packages optimised engines (TensorRT-LLM or vLLM) with model profiles tuned per GPU, enterprise support and validated containers, under an NVIDIA AI Enterprise licence. Open-source vLLM is free, gets new models on day one and is easy to customise, but tuning and support are your own responsibility.

## Decision
A single Helm chart with `engine: nim | vllm`. Shared concerns are identical for both engines: the GPU scheduling, model cache, probes, autoscaling, monitoring and network policy. Only the container spec differs.

## Guidance
| Choose | When |
|---|---|
| **NIM** | Production with support requirements; NVIDIA-validated models; best out-of-the-box latency on supported GPUs via TensorRT-LLM profiles |
| **vLLM** | Models or fine-tunes that NIM doesn't package; experimentation; no NVIDIA AI Enterprise licence |

## Consequences
Metric names differ (`vllm:` prefix in vLLM), so dashboards and alerts are parameterised by engine. Benchmarks must be repeated per engine and GPU with `llmplat sweep`.
