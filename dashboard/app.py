"""
LLM Serving & Optimization Lab — benchmark dashboard.

A read-only Streamlit app that visualizes the benchmark results produced by
`scripts/bench_client.py`. It loads the JSON/text files in `results/` and renders:

  * KPI tiles (single-user speed, peak throughput, the saturation "knee", worst-case latency)
  * the headline latency-vs-throughput curve (dual axis)
  * per-user decode rate vs concurrency (shows decode stays flat; latency is where overload shows)
  * the raw data table
  * GPU utilization captured during a saturating burst

Run:
    ~/agent-engineering-bootcamp/local-document-assistant/.venv/bin/streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

# --- Where the data lives --------------------------------------------------
# This file is dashboard/app.py, so the project root is its parent's parent,
# and results/ sits next to dashboard/. Computed relative to __file__ so the
# app works no matter which directory Streamlit is launched from.
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SWEEP_FILE = RESULTS / "sweep_8102_qwen14b.json"
BASELINE_FILE = RESULTS / "baseline_c1_qwen14b.json"
GPU_FILE = RESULTS / "gpu_util_c16.txt"

# Static context about the run (what the numbers describe).
MODEL = "Qwen/Qwen2.5-14B-Instruct-AWQ"
HARDWARE = "1 × RTX 4060 Ti (16 GB, Ada / AWQ 4-bit)"
OWNER = "Jashwanth Gowda B R"


# --- Data loading ----------------------------------------------------------
@st.cache_data  # Streamlit re-runs the whole script on every interaction;
                # caching means the files are parsed once, not on every click.
def load_sweep(path: Path) -> pd.DataFrame:
    """Load a bench_client sweep JSON (a list of per-concurrency-level dicts)
    into a tidy DataFrame, sorted by concurrency."""
    if not path.exists():
        return pd.DataFrame()
    rows = json.loads(path.read_text())
    df = pd.DataFrame(rows)
    # keep only levels that actually returned results
    if "ok" in df.columns:
        df = df[df["ok"] > 0]
    return df.sort_values("concurrency").reset_index(drop=True)


@st.cache_data
def load_gpu_peaks(path: Path) -> pd.DataFrame:
    """Parse the GPU sampler text (lines like '0, 100 %, 15193 MiB;1, 23 %, ...')
    into peak utilization and memory per GPU index."""
    if not path.exists():
        return pd.DataFrame()
    samples: dict[int, list[tuple[int, int]]] = {}
    pat = re.compile(r"(\d+),\s*(\d+)\s*%,\s*(\d+)\s*MiB")
    for line in path.read_text().splitlines():
        for gpu, util, mem in pat.findall(line):
            samples.setdefault(int(gpu), []).append((int(util), int(mem)))
    out = []
    for gpu, vals in sorted(samples.items()):
        utils = [u for u, _ in vals]
        mems = [m for _, m in vals]
        out.append({
            "gpu": f"GPU{gpu}",
            "peak_util_pct": max(utils),
            "active_samples": sum(u > 5 for u in utils),
            "peak_mem_mib": max(mems),
        })
    return pd.DataFrame(out)


def find_knee(df: pd.DataFrame) -> int:
    """The 'knee' = the smallest concurrency that already reaches 95% of the
    peak aggregate throughput. Past it, adding users buys ~no extra throughput."""
    if df.empty:
        return 0
    peak = df["aggregate_tps"].max()
    hit = df[df["aggregate_tps"] >= 0.95 * peak]
    return int(hit["concurrency"].min())


# --- Page setup ------------------------------------------------------------
st.set_page_config(page_title="LLM Serving Lab", page_icon="⚡", layout="wide")

sweep = load_sweep(SWEEP_FILE)
baseline = load_sweep(BASELINE_FILE)
gpu = load_gpu_peaks(GPU_FILE)

st.title("⚡ LLM Serving & Optimization Lab")
st.caption(f"{OWNER} · Latency-vs-throughput benchmark · **{MODEL}** on {HARDWARE}")

if sweep.empty:
    st.error(f"No sweep data found at `{SWEEP_FILE}`. Run scripts/bench_client.py first.")
    st.stop()

# --- KPI tiles -------------------------------------------------------------
peak_tps = sweep["aggregate_tps"].max()
knee = find_knee(sweep)
single = sweep[sweep["concurrency"] == 1]
single_decode = float(single["decode_tps_mean"].iloc[0]) if not single.empty else float("nan")
worst_ttft = sweep["ttft_p95"].max()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Single-user decode", f"{single_decode:.0f} tok/s")
c2.metric("Peak throughput", f"{peak_tps:.0f} tok/s", f"{peak_tps / single_decode:.1f}× single-user")
c3.metric("Saturation knee", f"{knee} concurrent", help="Smallest concurrency reaching 95% of peak throughput")
c4.metric("Worst-case TTFT", f"{worst_ttft:.2f} s", help="p95 time-to-first-token at the highest concurrency")

st.divider()

# --- Headline chart: throughput + latency vs concurrency (dual axis) -------
st.subheader("The trade-off curve")
st.caption("Aggregate throughput (bars, left) climbs with concurrency — until it saturates. "
           "Per-user latency (lines, right) is flat until the server queues, then spikes.")

base = alt.Chart(sweep).encode(
    x=alt.X("concurrency:O", title="Concurrent requests")
)
bars = base.mark_bar(color="#4c8bf5", opacity=0.85).encode(
    y=alt.Y("aggregate_tps:Q", title="Aggregate throughput (tok/s)"),
    tooltip=["concurrency", alt.Tooltip("aggregate_tps:Q", format=".0f")],
)
ttft50 = base.mark_line(color="#ff5c5c", point=True, strokeWidth=3).encode(
    y=alt.Y("ttft_p50:Q", title="TTFT (s)"),
    tooltip=["concurrency", alt.Tooltip("ttft_p50:Q", format=".3f")],
)
ttft95 = base.mark_line(color="#ffa94d", point=True, strokeDash=[4, 3]).encode(
    y="ttft_p95:Q",
    tooltip=["concurrency", alt.Tooltip("ttft_p95:Q", format=".3f")],
)
chart = alt.layer(bars, ttft50 + ttft95).resolve_scale(y="independent").properties(height=380)
st.altair_chart(chart, width="stretch")
st.caption("🔴 solid = median TTFT   🟠 dashed = p95 TTFT   🔵 bars = aggregate throughput")

# --- Secondary chart: decode rate stays flat -------------------------------
col_a, col_b = st.columns([3, 2])
with col_a:
    st.subheader("Per-user decode rate")
    st.caption("Individual decoding speed barely changes — overload shows up as *queueing* "
               "(the TTFT spike above), not slower token generation.")
    decode_chart = alt.Chart(sweep).mark_line(color="#51cf66", point=True, strokeWidth=3).encode(
        x=alt.X("concurrency:O", title="Concurrent requests"),
        y=alt.Y("decode_tps_mean:Q", title="Decode (tok/s/user)",
                scale=alt.Scale(zero=False)),
        tooltip=["concurrency", alt.Tooltip("decode_tps_mean:Q", format=".1f")],
    ).properties(height=300)
    st.altair_chart(decode_chart, width="stretch")

with col_b:
    st.subheader("GPU utilization")
    if gpu.empty:
        st.info("No GPU trace found.")
    else:
        st.caption("Captured during a concurrency=16 burst.")
        for _, r in gpu.iterrows():
            st.metric(r["gpu"], f"{r['peak_util_pct']}% peak",
                      f"{r['peak_mem_mib'] / 1024:.1f} GB VRAM")
        st.caption("One GPU saturates (the engine under test); the other replica stays idle "
                   "→ headroom for data-parallel scaling.")

st.divider()

# --- Raw data table --------------------------------------------------------
st.subheader("Raw measurements")
show = sweep[["concurrency", "ttft_p50", "ttft_p95", "decode_tps_mean",
              "aggregate_tps", "wall"]].copy()
show.columns = ["Concurrency", "TTFT p50 (s)", "TTFT p95 (s)",
                "Decode (tok/s)", "Aggregate (tok/s)", "Wall (s)"]
st.dataframe(show.style.format({
    "TTFT p50 (s)": "{:.3f}", "TTFT p95 (s)": "{:.3f}",
    "Decode (tok/s)": "{:.1f}", "Aggregate (tok/s)": "{:.1f}", "Wall (s)": "{:.2f}",
}), width="stretch", hide_index=True)

# --- Interpretation --------------------------------------------------------
with st.expander("📖 How to read this"):
    st.markdown(f"""
- **Throughput scales linearly up to the knee ({knee} users)** — vLLM's *continuous batching*
  packs concurrent requests onto the GPU, so total tok/s rises ~{peak_tps / single_decode:.0f}×
  while each user stays fast.
- **Past the knee it saturates** — aggregate throughput flattens (the GPU is at 100%), but
  latency explodes because new requests **wait in a queue**. Here p95 TTFT reaches
  **{worst_ttft:.1f} s**.
- **Decode rate stays ~flat** — admitted requests decode at full speed; overload is a
  *queueing* problem, not a *decoding* one. That's the fingerprint of continuous batching.
- **Operating point:** run near **{knee} concurrent requests** for max throughput at low latency;
  beyond it, add a **second replica** (the idle GPU) rather than pushing one engine harder.
""")
st.caption("Benchmarks were run only as bounded, authorized bursts — the server is shared "
           "and already serving live traffic.")
