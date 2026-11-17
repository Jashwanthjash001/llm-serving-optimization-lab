# Stage 0 — Environment Audit

Inventory of the two machines this project spans, taken **before** installing or running
anything. Inspection and documentation only.

---

## Environment B — MacBook Pro (dev / client / dashboard)  ✅ audited

| Item | Value |
| ---- | ----- |
| macOS | 15.7.4 (build 24G517) |
| Chip | Apple M1 Pro |
| Logical CPUs | 8 |
| Unified memory | 16 GB |
| Free disk | ~99 GB available (460 GB volume) |
| Python | 3.14.6 (Anaconda, `/opt/anaconda3`) |
| Existing venv | `local-document-assistant/.venv` — has **httpx, pandas, streamlit, torch, sentence-transformers** (reusable for the benchmark client + dashboard) |
| llama.cpp | vendored build **b12119** (Metal) in `edge-ai-client/vendor/` |
| Docker | 29.4.1 (present) |
| Reusable models | `Llama-3.1-8B-Instruct` **Q4_K_M** (4.6 GB) + **Q8_0** (8.0 GB) GGUF |
| vLLM | not present (expected — vLLM is Linux/NVIDIA only) |

**Role:** runs the benchmark client, concurrent load generator, results storage, and the
Streamlit dashboard. **Does not run GPU inference** — this keeps heavy load off the Mac
(which ran hot during Project 1).

---

## Environment A — Ubuntu GPU server (inference)  ✅ audited (2026-11-17, via SSH)

> Audited by running [`scripts/audit_gpu_server.sh`](../scripts/audit_gpu_server.sh) over SSH
> (read-only; no installs, no inference). Host: `<user>@<gpu-host>` (`<gpu-host>`).

| Item | Confirmed value |
| ---- | --------------- |
| OS / kernel | Ubuntu 24.04.3 LTS |
| CPU | AMD Ryzen 9 5950X (16C / 32T) |
| System RAM | 62 GB |
| GPUs | 2 × NVIDIA RTX 4060 Ti **16 GB** (16380 MiB each) ✅ |
| GPU arch | compute cap **8.9** (Ada, FP8-capable) ✅ |
| GPU topology | **PHB (PCIe host bridge) — no NVLink** ✅ (confirms the TP-penalty assumption) |
| NVIDIA driver | 580.178.04 |
| CUDA toolkit | 12.0 (V12.0.140) |
| Docker | 27.1.1 + NVIDIA Container Toolkit 1.18.1 |
| vLLM | runs as **Docker** (`vllm/vllm-openai`); not a pip install |
| HF model cache | 8.5 GB — `lightonai/LightOnOCR-3-4B` |
| **Disk space** | **457 GB, 99% used, only 4.7 GB free** ⚠️ |

**Role:** runs the inference server (vLLM) under measurement.

### ⚠️ Audit findings that change the plan

The server is **shared and already serving live workloads**, not a free bench machine:

- **3 vLLM Docker containers** already running (all `Up` and `healthy`):
  | Port | Model (served-name) |
  | ---- | ------------------- |
  | 8101 | chat LLM |
  | 8102 | `Qwen/Qwen2.5-14B-Instruct-AWQ` |
  | 8103 | `BAAI/bge-m3` (embeddings) |
  | 8104 | GLM-4-9B-0414-FP8 |
- **Both GPUs are ~92% full** (≈15 GB/16 GB each) — **no VRAM headroom** to load a fresh baseline model.
- **Disk is 99% full (4.7 GB free)** — **cannot download** a new baseline model (AWQ 8B ≈ 5.6 GB).

**Implication:** the original Stage-1 plan (download our *own* clean Llama-3.1-8B-AWQ baseline and
sweep it on an idle card) is **blocked** by disk + VRAM, and the only servable models are the
ones already in use — which must not be load-tested without the owner's permission and an off-hours
window. The viable near-term path is a **light, read-only latency characterization** of a live
endpoint at concurrency = 1 (minimal load), deferring any concurrency sweep until a window +
disk/VRAM are freed.

---

## Hardware limitations & compatibility concerns

1. **16 GB per GPU is the binding constraint.** Llama 3.1 8B in FP16 is ~16 GB of weights
   alone — too tight to leave room for the KV cache and batching on a single card. Practical
   options: run a **4-bit quantized** 8B (AWQ/GPTQ, ~5–6 GB → lots of KV headroom), use **FP8**
   (Ada supports it), or split across both GPUs (see next point).
2. **No NVLink between the two RTX 4060 Ti.** Cross-GPU tensor parallelism would communicate
   over **PCIe**, which is much slower than NVLink and can bottleneck. Implication: for an 8B
   model, **single-GPU is cleaner**; the *second* GPU is better used as a **second replica**
   (data parallelism) to study concurrency/throughput scaling, rather than tensor-parallel.
3. **Consumer GPUs** (not datacenter) — no MIG, limited aggregate memory bandwidth. Great for
   *learning* serving dynamics (batching, KV cache, concurrency); not meant to match A100-class
   numbers.
4. **vLLM requires Linux + NVIDIA + CUDA** — so it lives only on Environment A. The Mac can't
   run it (hence the split: Mac = client/dashboard, Ubuntu = server).
5. **Network between the two machines** becomes part of measured latency — the benchmark client
   on the Mac talks to the server over the LAN. We'll record client-side vs server-side timings
   so network overhead is visible, not hidden.

## Reuse summary

- **Reuse:** the `Llama-3.1-8B-Instruct` model family (consistent across all three projects),
  the Mac's Python env for the client/dashboard, and the Streamlit dashboard pattern from Project 2.
- **New on the server:** vLLM (serving), GPU metrics collection (nvidia-smi / vLLM `/metrics`).
