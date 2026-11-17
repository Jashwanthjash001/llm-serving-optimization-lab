#!/usr/bin/env python3
"""
bench_client.py — streaming latency/throughput benchmark for an OpenAI-compatible
(vLLM) endpoint. Measures the latency-vs-throughput trade-off across a concurrency sweep.

READ-ONLY on the server: it only sends normal /v1/chat/completions requests (no admin,
no config change). Keep `--max-tokens` and the sweep modest when the endpoint is shared.

Per-request metrics (from the stream):
  * TTFT        — time to first token (s)
  * decode tok/s — completion_tokens / (total - TTFT)
  * total       — wall time for the whole response
Per-level metrics:
  * aggregate throughput — sum(completion_tokens) / wall time of the concurrent batch
  * p50/p95 TTFT, mean decode tok/s

Usage:
  python bench_client.py --url http://<gpu-host>:8102/v1 \
      --model Qwen/Qwen2.5-14B-Instruct-AWQ --concurrency 1,2,4,8 --max-tokens 128
"""
from __future__ import annotations
import argparse, asyncio, json, statistics, time
import httpx

PROMPT = (
    "You are a helpful assistant. Explain, in a single clear paragraph, how the TCP "
    "three-way handshake establishes a connection between a client and a server, and "
    "why each of the three steps is necessary. Be precise and technical."
)


async def one_request(client: httpx.AsyncClient, url: str, model: str,
                      prompt: str, max_tokens: int) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    ttft = None
    completion_tokens = 0
    seen_chunks = 0
    err = None
    try:
        async with client.stream("POST", f"{url}/chat/completions", json=payload) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode("utf-8", "ignore")[:200]
                return {"ok": False, "err": f"HTTP {resp.status_code}: {body}"}
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                # usage arrives on a final chunk (choices may be empty there)
                if chunk.get("usage"):
                    completion_tokens = chunk["usage"].get("completion_tokens", completion_tokens)
                choices = chunk.get("choices") or []
                if choices:
                    delta = choices[0].get("delta", {}).get("content")
                    if delta:
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        seen_chunks += 1
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    total = time.perf_counter() - t0
    if err:
        return {"ok": False, "err": err}
    ntok = completion_tokens or seen_chunks  # fall back to chunk count if usage absent
    decode_tps = (ntok - 1) / (total - ttft) if (ttft and ntok > 1 and total > ttft) else 0.0
    return {"ok": True, "ttft": ttft or 0.0, "total": total,
            "ntok": ntok, "decode_tps": decode_tps}


async def run_level(url: str, model: str, prompt: str, max_tokens: int,
                    concurrency: int, timeout: float) -> dict:
    limits = httpx.Limits(max_connections=concurrency + 4, max_keepalive_connections=concurrency + 4)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        t0 = time.perf_counter()
        results = await asyncio.gather(
            *[one_request(client, url, model, prompt, max_tokens) for _ in range(concurrency)]
        )
        wall = time.perf_counter() - t0
    ok = [r for r in results if r.get("ok")]
    bad = [r for r in results if not r.get("ok")]
    if not ok:
        return {"concurrency": concurrency, "wall": wall, "ok": 0,
                "errors": [b["err"] for b in bad][:3]}
    ttfts = sorted(r["ttft"] for r in ok)
    total_out = sum(r["ntok"] for r in ok)
    return {
        "concurrency": concurrency,
        "ok": len(ok),
        "failed": len(bad),
        "wall": wall,
        "ttft_p50": statistics.median(ttfts),
        "ttft_p95": ttfts[int(len(ttfts) * 0.95) - 1] if len(ttfts) > 1 else ttfts[0],
        "decode_tps_mean": statistics.mean(r["decode_tps"] for r in ok),
        "tokens_per_req": statistics.mean(r["ntok"] for r in ok),
        "aggregate_tps": total_out / wall if wall > 0 else 0.0,
        "errors": [b["err"] for b in bad][:3],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="OpenAI-compatible base, e.g. http://host:8102/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--concurrency", default="1,2,4,8", help="comma list, e.g. 1,2,4,8")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--json-out", default=None, help="write raw results JSON here")
    a = ap.parse_args()

    levels = [int(x) for x in a.concurrency.split(",") if x.strip()]
    print(f"Endpoint : {a.url}")
    print(f"Model    : {a.model}")
    print(f"Gen      : {a.max_tokens} tokens/req, temperature=0, streamed")
    print(f"Sweep    : concurrency {levels}\n")
    hdr = f"{'conc':>4} {'ok':>3} {'ttft_p50':>9} {'ttft_p95':>9} {'decode_tps':>11} {'agg_tps':>9} {'wall_s':>7}"
    print(hdr)
    print("-" * len(hdr))

    runs = []
    for c in levels:
        r = asyncio.run(run_level(a.url, a.model, PROMPT, a.max_tokens, c, a.timeout))
        runs.append(r)
        if not r.get("ok"):
            print(f"{c:>4}   0  {'--':>9} {'--':>9} {'--':>11} {'--':>9} {r['wall']:>7.2f}  ERR: {r.get('errors')}")
            continue
        print(f"{r['concurrency']:>4} {r['ok']:>3} {r['ttft_p50']:>9.3f} {r['ttft_p95']:>9.3f} "
              f"{r['decode_tps_mean']:>11.1f} {r['aggregate_tps']:>9.1f} {r['wall']:>7.2f}"
              + (f"  (failed {r['failed']})" if r.get("failed") else ""))

    if a.json_out:
        with open(a.json_out, "w") as fh:
            json.dump(runs, fh, indent=2)
        print(f"\nRaw results -> {a.json_out}")


if __name__ == "__main__":
    main()
