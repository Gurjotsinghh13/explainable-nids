"""Streamlit analyst dashboard for the Explainable NIDS.

Replays raw CICIDS2017 flows to the FastAPI backend and shows a live
classification feed, the attack-type distribution, per-alert SHAP
explanations and timing metrics.

Run from the project root (with the API running):
    python -m streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API = os.environ.get("NIDS_API", "http://127.0.0.1:8000")
REPLAY = Path(__file__).resolve().parents[1] / "data" / "processed" / "replay_sample.csv"

st.set_page_config(page_title="Explainable NIDS", layout="wide")


@st.cache_data
def load_replay():
    return pd.read_csv(REPLAY)


def api_get(path):
    try:
        return requests.get(f"{API}{path}", timeout=5).json()
    except Exception:
        return None


def api_post(path, payload):
    try:
        return requests.post(f"{API}{path}", json=payload, timeout=10).json()
    except Exception as e:
        return {"error": str(e)}


ss = st.session_state
ss.setdefault("cursor", 0)
ss.setdefault("results", [])
ss.setdefault("running", False)

# ---- sidebar ----
st.sidebar.title("Controls")
health = api_get("/health")
if health and health.get("status") == "ok":
    st.sidebar.success(f"API online · {health['n_features']} features · "
                       f"{len(health['classes'])} classes")
else:
    st.sidebar.error("API offline. Start it with:\n`python -m uvicorn api.main:app`")

batch = st.sidebar.slider("Flows per step", 1, 25, 5)
c1, c2 = st.sidebar.columns(2)
if c1.button("▶ Start"):
    ss.running = True
if c2.button("⏸ Stop"):
    ss.running = False
if st.sidebar.button("↺ Reset feed"):
    ss.cursor, ss.results, ss.running = 0, [], False

try:
    replay = load_replay()
except Exception:
    st.error("replay_sample.csv not found. Run `python -m scripts.make_replay_sample` first.")
    st.stop()

feature_cols = [c for c in replay.columns if c != "Label"]
st.sidebar.caption(f"Replay position: {ss.cursor} / {len(replay)}")


def send_batch(n):
    for _ in range(n):
        if ss.cursor >= len(replay):
            ss.running = False
            break
        row = replay.iloc[ss.cursor]
        payload = {"features": {c: float(row[c]) for c in feature_cols},
                   "true_label": str(row["Label"])}
        r = api_post("/predict", payload)
        if "error" not in r:
            ss.results.insert(0, {"id": r["alert_id"], "true": str(row["Label"]),
                                  "predicted": r["predicted_label"],
                                  "confidence": r["confidence"],
                                  "latency_ms": r["inference_ms"],
                                  "features": payload["features"]})
        ss.cursor += 1


# ---- metrics ----
st.title("Explainable NIDS — Analyst Dashboard")
stats = api_get("/stats") or {}
res = ss.results
n = len(res)
acc = sum(r["true"] == r["predicted"] for r in res) / n if n else 0.0
lat = sum(r["latency_ms"] for r in res) / n if n else 0.0
m1, m2, m3, m4 = st.columns(4)
m1.metric("Flows processed", n)
m2.metric("Attacks flagged", sum(r["predicted"] != "BENIGN" for r in res))
m3.metric("Live accuracy", f"{acc * 100:.1f}%")
m4.metric("Avg inference", f"{lat:.1f} ms")

left, right = st.columns([3, 2])
with left:
    st.subheader("Live classification feed")
    if res:
        feed = pd.DataFrame([{k: v for k, v in r.items() if k != "features"} for r in res[:20]])
        st.dataframe(feed.style.apply(
            lambda row: ["" if row["predicted"] == "BENIGN" else "background-color:#4a1520"] * len(row),
            axis=1), use_container_width=True, height=430)
    else:
        st.info("Press ▶ Start to begin replaying flows.")
with right:
    st.subheader("Attack-type distribution")
    dist = stats.get("distribution", {})
    if dist:
        dd = pd.DataFrame({"class": list(dist), "count": list(dist.values())}).sort_values("count")
        fig = px.bar(dd, x="count", y="class", orientation="h", height=430)
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No alerts logged yet.")

# ---- per-alert SHAP ----
st.subheader("Why was this flagged? (per-alert SHAP)")
if res:
    labels = [f"#{r['id']} — {r['predicted']} (true: {r['true']})" for r in res[:20]]
    pick = st.selectbox("Select an alert", range(len(labels)), format_func=lambda i: labels[i])
    if st.button("Explain selected alert"):
        ex = api_post("/explain", {"features": res[pick]["features"]})
        if "error" in ex:
            st.error(f"Explain failed: {ex['error']}")
        else:
            st.caption(f"Base learner attributes this flow toward **{ex['predicted_label']}**")
            ef = pd.DataFrame(ex["top_features"])
            ef = ef.reindex(ef["shap_value"].abs().sort_values().index)
            fig = px.bar(ef, x="shap_value", y="feature", orientation="h",
                         color="shap_value", color_continuous_scale="RdBu_r", height=380)
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Explanations appear once flows have been classified.")

if ss.running:
    send_batch(batch)
    time.sleep(0.4)
    st.rerun()
