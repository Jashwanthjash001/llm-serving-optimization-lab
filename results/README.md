# Benchmark Results

## Stage 1 — single-user baseline (concurrency = 1)  ✅ 2026-11-17

Measured with [`scripts/bench_client.py`](../scripts/bench_client.py) from the MacBook (Env B)
against the **live vLLM endpoint on the Ubuntu server** (Env A, `<gpu-host>:8102`).
Streamed `/v1/chat/completions`, 128 generated tokens, `temperature=0`.

**Endpoint under test:** `Qwen/Qwen2.5-14B-Instruct-AWQ` (vLLM, 1 × RTX 4060 Ti 16 GB, AWQ 4-bit)

| Sample | TTFT (s) | Decode (tok/s) | Aggregate (tok/s) | Wall (s) |
| ------ | -------- | -------------- | ----------------- | -------- |
| 1 (cold) | 0.134 | 30.9 | 30.1 | 4.25 |
| 2 | 0.054 | 30.9 | 30.8 | 4.16 |
| 3 | 0.058 | 31.0 | 30.8 | 4.15 |

Raw: [`baseline_c1_qwen14b.json`](baseline_c1_qwen14b.json)

**Read:**
- **TTFT ~55 ms warm** (first request ~134 ms includes connection + any scheduling warm-up).
- **Decode ~31 tok/s**, essentially identical across runs → a stable, reproducible baseline.
- ~31 tok/s for a **14B 4-bit (AWQ)** model on a consumer **RTX 4060 Ti** is in the expected band
  (memory-bandwidth bound at batch size 1).

## Stage 1b — single-user profile (concurrency = 1, zero load risk)  ✅ 2026-11-17

The server runs **two Qwen2.5-14B-AWQ replicas, one per GPU** (`:8101` = GPU-a, `:8102` = GPU-b);
a GLM-4-9B endpoint seen earlier was stopped, so the 9B-vs-14B comparison is deferred. All runs
below are concurrency=1 (one request at a time) — safe on a shared endpoint.

**Replica parity** (128 tok/req) — the two GPU replicas are equivalent:

| Replica | TTFT warm (s) | Decode (tok/s) |
| ------- | ------------- | -------------- |
| `:8101` (GPU-a) | 0.055 | 30.7 |
| `:8102` (GPU-b) | 0.056 | 31.0 |

**Generation-length scaling** (`:8102`, concurrency=1):

| max_tokens | TTFT (s) | Decode (tok/s) |
| ---------- | -------- | -------------- |
| 64  | 0.066 | 31.0 |
| 128 | 0.064 | 31.0 |
| 256 | 0.065 | 31.0 |

**Read:**
- **Decode rate is constant (~31 tok/s)** regardless of how many tokens are generated — single-token
  decode is memory-bandwidth bound, so length doesn't change the *rate*, only total time.
- **TTFT is flat (~65 ms)** for a fixed prompt — it's the prefill cost, independent of output length.
- The two replicas are interchangeable → a clean **data-parallel** topology (one replica per GPU)
  for the eventual concurrency/throughput scaling study.

## Stage 3 — concurrency sweep (the latency-vs-throughput curve)  ✅ 2026-11-17

Run during a quiet window with the server owner's authorization (GPUs idle at 0–1% beforehand).
Bounded, bursty load: each level is one simultaneous batch of `concurrency` requests, 128 tokens
each — ~1 minute of load total. Endpoint: `Qwen/Qwen2.5-14B-Instruct-AWQ` on `:8102` (1 × RTX 4060 Ti).
Raw: [`sweep_8102_qwen14b.json`](sweep_8102_qwen14b.json).

| Concurrency | TTFT p50 (s) | TTFT p95 (s) | Decode/user (tok/s) | **Aggregate (tok/s)** | Wall (s) |
| ----------- | ------------ | ------------ | ------------------- | --------------------- | -------- |
| 1  | 0.086 | 0.086 | 31.0 | 30.6  | 4.18 |
| 2  | 0.064 | 0.048 | 30.8 | 60.9  | 4.21 |
| 4  | 0.086 | 0.086 | 30.5 | 120.5 | 4.25 |
| 8  | 0.125 | 0.126 | 30.3 | **237.5** | 4.31 |
| 16 | **2.229** | **4.398** | 30.3 | 238.3 | 8.59 |

### The headline result

- **Linear throughput scaling 1 → 8:** aggregate throughput rises almost perfectly linearly
  (30.6 → 237.5 tok/s, ~8×) while per-user latency stays low (TTFT ~0.1 s). vLLM **continuous
  batching** packs concurrent requests onto the GPU at near-zero marginal cost.
- **Saturation at 16:** aggregate throughput **plateaus** (237.5 → 238.3 tok/s — no gain) while
  TTFT **explodes ~18×** (0.125 → 2.23 s p50, 4.40 s p95) and wall time doubles. The extra
  requests **queue**; they don't add work the GPU can do.
- **Per-request decode rate is ~30 tok/s throughout** — degradation under overload shows up as
  **queueing latency (TTFT)**, not slower decoding. That's the signature of continuous batching:
  admitted requests decode at full speed; the rest wait.
- **The knee at concurrency 8 is explained by the server config:** vLLM is launched with
  **`--max-num-seqs=8`** (see [environment audit](../docs/environment-audit.md)). The benchmark
  recovered the configured max batch size purely from external latency/throughput measurements.

### Operating recommendation

For this model on this card, **~8 concurrent requests is the sweet spot**: maximum aggregate
throughput (~238 tok/s) with still-low latency. Beyond it you trade an 18× latency penalty for
**zero** extra throughput — the point at which you should add a **second replica** (the other GPU)
rather than push a single engine harder.

### Still deferred

- **Sustained-load sweep** (steady arrival rate over minutes, not a single burst) — to see
  steady-state queue depth and p99 under continuous traffic.
- **Two-replica data-parallel scaling** (`:8101` + `:8102` together) — does aggregate throughput
  roughly double to ~475 tok/s across both GPUs?
- **`max_num_seqs` / `gpu_memory_utilization` tuning** — move the knee by changing the batch limit.

### GPU utilization (confirming the ceiling)

Sampled on the server during a short `concurrency=16` saturating burst
([`gpu_util_c16.txt`](gpu_util_c16.txt)):

| GPU | Serves | Peak util | VRAM | Role in this test |
| --- | ------ | --------- | ---- | ----------------- |
| **GPU1** | `:8102` (model under test) | **100 %** | ~15.2 GB (flat) | the saturated engine — the throughput ceiling |
| GPU0 | `:8101` (other replica) | ~23 % (mostly idle) | ~15.3 GB (flat) | independent engine, not exercised here |

This confirms the client-side read: the throughput plateau at concurrency 8→16 is **GPU1 hitting
100 %**. GPU0 sat idle throughout — the two replicas are **independent single-GPU engines**, so the
second GPU is pure headroom for **data-parallel scaling** (route across `:8101` + `:8102` to roughly
double aggregate throughput). VRAM stayed flat (KV cache fits inside the pre-allocated
`gpu_memory_utilization=0.90`), so the limit here is **compute**, not memory.
