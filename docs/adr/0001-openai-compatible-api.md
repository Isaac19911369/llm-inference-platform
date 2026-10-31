# ADR 0001: Expose every model through the OpenAI-compatible API

**Status:** accepted

## Context
Application teams (assistants, agents, batch jobs) should not depend on which engine or vendor serves a model. NVIDIA NIM, vLLM, TGI, Ollama and Azure OpenAI all implement the OpenAI Chat Completions and Embeddings APIs.

## Decision
The platform contract is the OpenAI-compatible HTTP API (`/v1/chat/completions`, `/v1/embeddings`, `/v1/models`) plus the `model` name. Engine-specific features are only used behind that contract.

## Consequences
- Clients use the standard `openai` SDK. Moving between self-hosted NIM, vLLM and Azure OpenAI changes only a base URL and a model name. The [enterprise-rag-assistant](https://github.com/Isaac19911369/enterprise-rag-assistant) runs against any of them unchanged.
- The load tester and simulator in this repository work against every engine.
- Features outside the contract, such as engine-specific sampling or LoRA routing, need an explicit decision when someone wants them.
