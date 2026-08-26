"""Streamlit analyst dashboard for the Explainable NIDS.

Replays raw CICIDS2017 flows to the FastAPI backend and shows, in real time:
  - a live classification feed,
  - the attack-type distribution,
  - per-alert SHAP explanations (why a flow was flagged),
  - inference timing metrics.

Run (from the project root, with the API already running):
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
import time

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API = os.environ.get("NIDS_API", "http://127.0.0.1:8000")
REPLAY = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "replay_sample.csv")

st.set_page_config(page_title="Explainable NIDS", layout="wide")


# --------------------------------------------------------------------------- #
# Data + state
# --------------------------------------------------------------------------- #
@st.cache_data
def load_replay():
    df = pd.read_csv(REPLAY)
    return df


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
ss.setdefault("results", [])       # list of dicts for the live feed
ss.setdefault("running", False)


# --------------------------------------------------------------------------- #
# Sidebar controls
# --------------------------------------------------------------------------- #
st.sidebar.title("Controls")
health = api_get("/health")
if health and health.get("status") == "ok":
    st.sidebar.success(f"API online · {health['n_features']} features · {len(health['classes'])} classes")
else:
    st.sidebar.error("API offline — start it with:\n`uvicorn api.main:app`")

batch = st.sidebar.slider("Flows per step", 1, 25, 5)
c1, c2 = st.sidebar.columns(2)
if c1.button("▶ Start"):
    ss.running = True
if c2.button("⏸ Stop"):
    ss.running = False
if st.sidebar.button("↺ Reset feed"):
    ss.cursor = 0
    ss.results = []
    ss.running = False

try:
    replay = load_replay()
except Exception:
    st.error("replay_sample.csv not found. Run the replay-sample cell first (Part A).")
    st.stop()

feature_cols = [c for c in replay.columns if c != "Label"]
st.sidebar.caption(f"Replay position: {ss.cursor} / {len(replay)}")


# --------------------------------------------------------------------------- #
# Send a batch of flows to the API
# --------------------------------------------------------------------------- #
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
            ss.results.insert(0, {
                "id": r["alert_id"],
                "true": str(row["Label"]),
                "predicted": r["predicted_label"],
                "confidence": r["confidence"],
                "latency_ms": r["inference_ms"],
                "features": payload["features"],
            })
        ss.cursor += 1


# --------------------------------------------------------------------------- #
# Header + metrics
# --------------------------------------------------------------------------- #
st.title("Explainable NIDS — Analyst Dashboard")

stats = api_get("/stats") or {}
res = ss.results
n_proc = len(res)
acc = (sum(1 for x in res if x["true"] == x["predicted"]) / n_proc) if n_proc else 0.0
avg_lat = (sum(x["latency_ms"] for x in res) / n_proc) if n_proc else 0.0
n_attacks = sum(1 for x in res if x["predicted"] != "BENIGN")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Flows processed", n_proc)
m2.metric("Attacks flagged", n_attacks)
m3.metric("Live accuracy", f"{acc*100:.1f}%")
m4.metric("Avg inference", f"{avg_lat:.1f} ms")

left, right = st.columns([3, 2])

# --------------------------------------------------------------------------- #
# Live feed
# --------------------------------------------------------------------------- #
with left:
    st.subheader("Live classification feed")
    if res:
        feed = pd.DataFrame([{k: v for k, v in r.items() if k != "features"} for r in res[:20]])
        def hl(row):
            color = "" if row["predicted"] == "BENIGN" else "background-color: #4a1520"
            return [color] * len(row)
        st.dataframe(feed.style.apply(hl, axis=1), use_container_width=True, height=430)
    else:
        st.info("Press ▶ Start (or send a batch) to begin replaying flows.")

# --------------------------------------------------------------------------- #
# Attack distribution
# --------------------------------------------------------------------------- #
with right:
    st.subheader("Attack-type distribution")
    dist = (stats or {}).get("distribution", {})
    if dist:
        dd = pd.DataFrame({"class": list(dist.keys()), "count": list(dist.values())})
        dd = dd.sort_values("count", ascending=True)
        fig = px.bar(dd, x="count", y="class", orientation="h", height=430)
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No alerts logged yet.")

# --------------------------------------------------------------------------- #
# Per-alert SHAP explanation
# --------------------------------------------------------------------------- #
st.subheader("Why was this flagged?  (per-alert SHAP)")
if res:
    labels = [f"#{r['id']} — {r['predicted']} (true: {r['true']})" for r in res[:20]]
    pick = st.selectbox("Select an alert to explain", range(len(labels)),
                        format_func=lambda i: labels[i])
    if st.button("Explain selected alert"):
        chosen = res[pick]
        ex = api_post("/explain", {"features": chosen["features"]})
        if "error" in ex:
            st.error(f"Explain failed: {ex['error']}")
        else:
            st.caption(f"Base learner attributes this toward **{ex['predicted_label']}**")
            ef = pd.DataFrame(ex["top_features"])
            ef["abs"] = ef["shap_value"].abs()
            ef = ef.sort_values("abs")
            fig = px.bar(ef, x="shap_value", y="feature", orientation="h",
                         color="shap_value", color_continuous_scale="RdBu_r",
                         height=380)
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Explanations appear once flows have been classified.")

# --------------------------------------------------------------------------- #
# Auto-play loop
# --------------------------------------------------------------------------- #
if ss.running:
    send_batch(batch)
    time.sleep(0.4)
    st.rerun()