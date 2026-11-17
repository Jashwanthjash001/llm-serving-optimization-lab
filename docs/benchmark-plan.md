# Benchmark Plan

> Proposal for review. No inference runs until the audit is confirmed and a config is approved.

## The core question

**Latency and throughput are not the same thing.**

- **Latency** — how long *one* request takes (TTFT, inter-token latency, total time).
- **Throughput** — how much *total* work the server completes per second across all users.

A server can give one user 40 tok/s yet, under 10 concurrent users, deliver far higher *aggregate*
throughput while each user waits longer. The whole project is about measuring and explaining that
curve — the trade-off between serving many users efficiently (batching) and keeping any single
user fast.

## Metrics (per request + per run)

| Metric | Level | Source |
| ------ | ----- | ------ |
| TTFT (time to first token) | per request | client (streamed) |
| Inter-token latency | per request | client (streamed) |
| Tokens/sec (decode) | per request | client |
| Aggregate throughput (tok/s) | per run | derived + vLLM `/metrics` |
| Concurrency (running/waiting) | per run | vLLM `/metrics` |
| GPU memory used | per run | `nvidia-smi` |
| GPU utilization % | per run | `nvidia-smi` |

## Proposed first configuration (baseline)

| Setting | Value |
| ------- | ----- |
| Model | `Llama-3.1-8B-Instruct`, **4-bit (AWQ)** — fits one 16 GB card with KV headroom |
| Server | vLLM, OpenAI-compatible API, 1 × RTX 4060 Ti |
| `max_model_len` | 4096 (start conservative) |
| Prompt / generation | 512 prompt tokens / 128 generated (matches the Project 1 benchmark shape) |
| Concurrency | **1** (establish the single-user baseline first) |

From that baseline, the sweeps:

1. **Concurrency sweep** — 1 → 2 → 4 → 8 → 16 … until latency degrades or the queue grows. Plot
   per-user latency **and** aggregate throughput on the same axis → the headline trade-off curve.
2. **Batching** — static vs vLLM continuous batching at matched concurrency.
3. **KV cache / context length** — vary `max_model_len` and batch size; watch VRAM and where
   requests start getting **preempted/queued**.
4. **GPU utilization tuning** — `gpu_memory_utilization`, `max_num_seqs`, replica count.
5. **(Optional) tensor-parallel across both GPUs** — document the PCIe (no-NVLink) penalty.

## Acceptance criteria for the project

- Every run is reproducible from its manifest (config → identical numbers within noise).
- The dashboard shows the latency-vs-throughput curve and the VRAM/utilization story.
- A written report explains *why* the curves look the way they do (batching, KV cache, queueing).

## What we will NOT claim

- No comparison to datacenter GPUs as if equivalent — these are consumer 16 GB cards.
- No single "tokens/sec" number without its concurrency and config context.
