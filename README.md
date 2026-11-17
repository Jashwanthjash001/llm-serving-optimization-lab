# LLM Serving & Optimization Lab

A reproducible benchmarking environment that measures and **explains** how a local LLM server
behaves under real multi-user load — latency, throughput, concurrency, KV-cache/VRAM usage, and
the trade-offs between them.

Third in a portfolio that moves from the model outward:
1. [`edge-ai-client`](https://github.com/Jashwanthjash001/edge-ai-client) — quantization & edge inference
2. [`local-document-assistant`](https://github.com/Jashwanthjash001/local-document-assistant) — RAG & application engineering
3. **`llm-serving-optimization-lab`** (this repo) — inference **infrastructure & performance engineering**

> **The core idea:** latency (one request's speed) and throughput (total work/sec) are *not*
> the same thing. A server can be fast for one user and inefficient under load — or vice versa.
> This lab measures that curve experimentally and explains it.

## Two-machine setup

| Machine | Role |
| ------- | ---- |
| **Ubuntu GPU server** — 2 × RTX 4060 Ti (16 GB each) | runs the inference server (vLLM) under measurement |
| **MacBook Pro M1 Pro** | runs the load generator, metrics collection, and dashboard — **no GPU load on the Mac** |

## Status — concurrency curve measured (2026-11-17)

| Stage | Topic | Status |
| ----- | ----- | ------ |
| 0 | Environment audit | ✅ both machines audited ([env](docs/environment-audit.md)) |
| 1 | Baseline serving | ✅ single-user baseline + profile ([results](results/README.md)) |
| 2 | Latency & throughput client | ✅ `scripts/bench_client.py` (streaming TTFT/decode/aggregate) |
| 3 | Concurrency | ✅ sweep 1→16 — **linear to 8, saturates at 16** ([results](results/README.md)) |
| 4 | Batching (static vs continuous) | ⬜ |
| 5 | KV cache & context length | ⬜ |
| 6 | GPU utilization tuning | ⬜ |
| 7 | Observability dashboard | ✅ `dashboard/app.py` (Streamlit, reads `results/`) |
| 8 | Final benchmark report | ⬜ |

> **Headline finding:** on 1 × RTX 4060 Ti, Qwen2.5-14B-AWQ scales linearly to **~238 tok/s at
> concurrency 8** (8× single-user), then saturates — at 16, throughput is flat but TTFT jumps ~18×
> as requests queue. The knee matches the server's `--max-num-seqs=8`, and GPU1 hits 100% there.

> **Note:** the GPU server is a **shared server already serving live traffic**. Benchmarks are run only as bounded,
> authorized bursts during quiet windows — see the results doc.

## Docs

- [`docs/environment-audit.md`](docs/environment-audit.md) — hardware/software inventory (+ limitations)
- [`docs/architecture.md`](docs/architecture.md) — proposed serving/benchmark/dashboard design
- [`docs/benchmark-plan.md`](docs/benchmark-plan.md) — metrics, first config, and the sweep plan

## Run it yourself

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# single-user baseline (safe on a shared endpoint — one request at a time)
.venv/bin/python scripts/bench_client.py \
  --url http://<gpu-host>:8102/v1 --model Qwen/Qwen2.5-14B-Instruct-AWQ \
  --concurrency 1,1,1 --max-tokens 128

# dashboard (reads results/)
.venv/bin/streamlit run dashboard/app.py
```

Bump `--concurrency` to `1,2,4,8,16` to reproduce the sweep — **only during an authorized quiet
window**, since the endpoint is shared and already serving traffic.

## Next steps

- **Two-replica data-parallel scaling** — route across `:8101` + `:8102` (both GPUs) and see if
  aggregate throughput ~doubles past the single-engine 238 tok/s ceiling.
- **Sustained-load** sweep (steady arrival rate over minutes) for steady-state queue depth / p99.
- **`max_num_seqs` / `gpu_memory_utilization` tuning** to move the knee.
- **Final written report** tying the curves back to batching / KV-cache / queueing.
