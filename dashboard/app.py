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
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API = os.environ.get("NIDS_API", "http://127.0.0.1:8000")
REPLAY = Path(__file__).resolve().parents[1] / "data" / "processed" / "replay_sample.csv"

# Consistent class palette across the dashboard
CLASS_COLORS = {
    "BENIGN": "#2ecc71",
    "Bot": "#e74c3c",
    "DDoS": "#e67e22",
    "DoS GoldenEye": "#d35400",
    "DoS Hulk": "#c0392b",
    "DoS Slowhttptest": "#f39c12",
    "DoS slowloris": "#e67e22",
    "FTP-Patator": "#9b59b6",
    "PortScan": "#3498db",
    "SSH-Patator": "#8e44ad",
    "Web Attack": "#1abc9c",
}

# Domain knowledge descriptions for top features
FEATURE_DESCRIPTIONS = {
    "Init_Win_bytes_forward": "Initial TCP window size sent by the client; attack tools often use fixed, minimal or unusual values.",
    "Init_Win_bytes_backward": "Initial TCP window size sent by the server in response to the client.",
    "Flow IAT Min": "Smallest gap between packets in the flow; automated attacks show mechanical, near-zero intervals.",
    "Flow IAT Mean": "Average time interval between consecutive packets in the bidirectional flow.",
    "Flow IAT Max": "Maximum delay between packets; slow-rate attacks stretch this to keep connections alive.",
    "Flow IAT Std": "Variability in packet intervals; scripted attacks are rigid, while benign traffic varies.",
    "Fwd IAT Min": "Smallest gap between client-to-server packets.",
    "Fwd IAT Mean": "Average interval between outbound packets from the client.",
    "Fwd IAT Total": "Total time elapsed between the first and last forward packets.",
    "Bwd IAT Total": "Total time taken for server responses in the reverse direction.",
    "Flow Duration": "Total lifetime of the network connection in microseconds.",
    "Flow Bytes/s": "Flow throughput rate; high during volumetric floods, minimal during stealth probes.",
    "Flow Packets/s": "Packet rate per second; spikes dramatically during SYN and UDP floods.",
    "Total Fwd Packets": "Number of packets sent forward from the client to server.",
    "Total Backward Packets": "Number of packets returned from the server to client.",
    "Fwd Packet Length Max": "Maximum payload size sent in a single forward packet.",
    "Fwd Packet Length Mean": "Average size in bytes of forward packets.",
    "Bwd Packet Length Max": "Maximum payload size returned by the server.",
    "Bwd Packet Length Mean": "Average size of packets from server to client; zero or tiny values mean the server sent no real response.",
    "Bwd Packet Length Std": "Standard deviation of server response packet sizes.",
    "Average Packet Size": "Overall average packet size across both transmission directions.",
    "min_seg_size_forward": "Minimum segment size observed in forward packets, set by client TCP stack.",
    "Active Min": "Shortest period the connection stayed active before falling idle.",
    "Active Mean": "Average duration the connection actively transmitted data.",
    "Idle Min": "Shortest dormant period where no packets were observed.",
    "Idle Mean": "Average period of inactivity in the flow.",
    "PSH Flag Count": "Number of packets with Push flag set to bypass buffers.",
    "ACK Flag Count": "Number of acknowledgement packets confirming received data.",
    "URG Flag Count": "Number of packets with Urgent pointer set.",
}

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
        t0 = time.perf_counter()
        res = requests.post(f"{API}{path}", json=payload, timeout=10).json()
        rtt_ms = (time.perf_counter() - t0) * 1000.0
        return res, rtt_ms
    except Exception as e:
        return {"error": str(e)}, 0.0


# ---- session state ----
ss = st.session_state
ss.setdefault("cursor", 0)
ss.setdefault("alerts", {})           # alert_id -> alert dict
ss.setdefault("alert_ids", [])        # list of alert_ids (most recent first)
ss.setdefault("running", False)
ss.setdefault("current_explanation", None)  # dict: {"alert_id": id, "data": ex_json}
ss.setdefault("total_rtt_ms", 0.0)
ss.setdefault("rtt_count", 0)

# ---- header ----
st.title("Explainable AI-Based Network Intrusion Detection (NIDS)")
st.caption("Stacking (RF + XGBoost → LR) · within-fold SMOTE · SHAP · CICIDS2017")

# ---- sidebar controls ----
st.sidebar.title("Controls")
health = api_get("/health")
if health and health.get("status") == "ok":
    st.sidebar.markdown(
        '**API Status:** <span style="color:#2ecc71;font-weight:bold;">● Online</span> '
        f"({health['n_features']} features · {len(health['classes'])} classes)",
        unsafe_allow_html=True,
    )
else:
    st.sidebar.markdown(
        '**API Status:** <span style="color:#e74c3c;font-weight:bold;">● Offline</span>',
        unsafe_allow_html=True,
    )
    st.sidebar.error("Start API with:\n`python -m uvicorn api.main:app`")

batch = st.sidebar.slider("Flows per step", 1, 25, 5)
c1, c2 = st.sidebar.columns(2)
if c1.button("▶ Start", use_container_width=True):
    ss.running = True
if c2.button("⏸ Pause", use_container_width=True):
    ss.running = False
if st.sidebar.button("↺ Reset feed", use_container_width=True):
    ss.cursor = 0
    ss.alerts = {}
    ss.alert_ids = []
    ss.running = False
    ss.current_explanation = None
    ss.total_rtt_ms = 0.0
    ss.rtt_count = 0
    st.rerun()

try:
    replay = load_replay()
except Exception:
    st.error("replay_sample.csv not found. Run `python -m scripts.make_replay_sample` first.")
    st.stop()

feature_cols = [c for c in replay.columns if c != "Label"]
progress_val = min(max(ss.cursor / len(replay), 0.0), 1.0)
st.sidebar.progress(progress_val)
st.sidebar.caption(f"Replay position: {ss.cursor:,} / {len(replay):,} flows ({progress_val * 100:.1f}%)")

# Download alerts button in sidebar
st.sidebar.markdown("---")
api_alerts = api_get("/alerts?limit=1000") or []
if api_alerts:
    df_export = pd.DataFrame(api_alerts)
    csv_bytes = df_export.to_csv(index=False).encode("utf-8")
    st.sidebar.download_button(
        label="📥 Download alerts as CSV",
        data=csv_bytes,
        file_name="nids_alerts_log.csv",
        mime="text/csv",
        use_container_width=True,
    )


def send_batch(n):
    for _ in range(n):
        if ss.cursor >= len(replay):
            ss.running = False
            break
        row = replay.iloc[ss.cursor]
        payload = {
            "features": {c: float(row[c]) for c in feature_cols},
            "true_label": str(row["Label"]),
        }
        r, rtt_ms = api_post("/predict", payload)
        if "error" not in r and "alert_id" in r:
            aid = r["alert_id"]
            ss.alerts[aid] = {
                "id": aid,
                "time": datetime.now().strftime("%H:%M:%S"),
                "predicted": r["predicted_label"],
                "true": str(row["Label"]),
                "confidence": float(r["confidence"]),
                "latency_ms": float(r["inference_ms"]),
                "round_trip_ms": rtt_ms,
                "top_classes": r.get("top_classes", []),
                "features": payload["features"],
            }
            if aid not in ss.alert_ids:
                ss.alert_ids.insert(0, aid)
            ss.total_rtt_ms += rtt_ms
            ss.rtt_count += 1
        ss.cursor += 1


# ---- KPI metrics row ----
stats = api_get("/stats") or {}
n_proc = len(ss.alert_ids)
n_attacks = sum(1 for aid in ss.alert_ids if ss.alerts[aid]["predicted"] != "BENIGN")
acc = (sum(1 for aid in ss.alert_ids if ss.alerts[aid]["true"] == ss.alerts[aid]["predicted"]) / n_proc) if n_proc else 0.0
avg_infer = (sum(ss.alerts[aid]["latency_ms"] for aid in ss.alert_ids) / n_proc) if n_proc else 0.0
avg_rtt = (ss.total_rtt_ms / ss.rtt_count) if ss.rtt_count else 0.0

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Flows processed", f"{n_proc:,}")
m2.metric("Attacks flagged", f"{n_attacks:,}")
m3.metric("Live accuracy", f"{acc * 100:.1f}%")
m4.metric("Avg inference", f"{avg_infer:.1f} ms")
m5.metric("Avg round trip", f"{avg_rtt:.1f} ms")

st.markdown("---")

left, right = st.columns([3, 2])

# ---- Live feed table ----
with left:
    st.subheader("Live classification feed")
    if ss.alert_ids:
        feed_rows = []
        for aid in ss.alert_ids[:25]:
            al = ss.alerts[aid]
            feed_rows.append({
                "Alert ID": f"#{al['id']}",
                "Time": al["time"],
                "Predicted": al["predicted"],
                "True Label": al["true"],
                "Confidence": f"{al['confidence'] * 100:.1f}%",
                "Latency ms": f"{al['latency_ms']:.1f}",
                "Status": "✓" if al["predicted"] == al["true"] else "✗",
                "_misclassified": al["predicted"] != al["true"],
                "_attack": al["predicted"] != "BENIGN",
            })
        feed_df = pd.DataFrame(feed_rows)

        def style_feed_row(row):
            if row["_misclassified"]:
                return ["background-color: rgba(243, 156, 18, 0.35); font-weight: 500;"] * len(row)
            elif row["_attack"]:
                return ["background-color: rgba(235, 104, 52, 0.22);"] * len(row)
            else:
                return [""] * len(row)

        display_cols = ["Alert ID", "Time", "Predicted", "True Label", "Confidence", "Latency ms", "Status"]
        styled_table = feed_df.style.apply(style_feed_row, axis=1)
        st.dataframe(styled_table, column_order=display_cols, use_container_width=True, height=430)
    else:
        st.info("Press ▶ Start to begin replaying flows.")

# ---- Attack-type distribution ----
with right:
    st.subheader("Attack-type distribution")
    dist = stats.get("distribution", {})
    if dist:
        dd = pd.DataFrame({"class": list(dist.keys()), "count": list(dist.values())})
        dd = dd.sort_values("count", ascending=False)
        fig_dist = px.bar(
            dd,
            x="class",
            y="count",
            color="class",
            color_discrete_map=CLASS_COLORS,
            height=430,
        )
        fig_dist.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="",
            yaxis_title="Count",
            showlegend=False,
        )
        st.plotly_chart(fig_dist, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No alerts logged yet.")

st.markdown("---")

# ---- Per-alert SHAP explanation ----
st.subheader("Why was this flagged? (per-alert SHAP)")

if ss.alert_ids:
    options = ss.alert_ids[:50]

    def format_alert_option(aid):
        al = ss.alerts[aid]
        mark = "✓" if al["predicted"] == al["true"] else "✗"
        return f"#{aid} — {al['predicted']} (true: {al['true']}) [{mark}]"

    selected_alert_id = st.selectbox(
        "Select an alert to inspect",
        options=options,
        format_func=format_alert_option,
        key="selected_alert_id",
    )

    col_btn, _ = st.columns([1, 4])
    if col_btn.button("Explain selected alert", use_container_width=True):
        ss.running = False  # automatically pause replay while explanation is open
        chosen = ss.alerts[selected_alert_id]
        ex_res, _ = api_post("/explain", {"features": chosen["features"]})
        if "error" in ex_res:
            st.error(f"Explain failed: {ex_res['error']}")
        else:
            ss.current_explanation = {
                "alert_id": selected_alert_id,
                "data": ex_res,
            }

    # Render current explanation if available
    curr_ex = ss.current_explanation
    if curr_ex and curr_ex["alert_id"] in ss.alerts:
        alert_info = ss.alerts[curr_ex["alert_id"]]
        ex_data = curr_ex["data"]

        # Consistency check
        if ex_data.get("predicted_label") != alert_info["predicted"]:
            st.warning(
                f"⚠ The label returned by /explain ({ex_data.get('predicted_label')}) does not match "
                f"the label stored for this alert ({alert_info['predicted']}). "
                "The explanation may not correspond to this flow."
            )

        # Explanation card
        is_corr = alert_info["predicted"] == alert_info["true"]
        mark_symbol = "✓" if is_corr else "✗"
        conf_pct = alert_info["confidence"] * 100

        with st.container():
            st.markdown(
                f"### Verdict: Flagged as **{alert_info['predicted']}** ({conf_pct:.1f}% confidence). "
                f"True label: **{alert_info['true']}**. &nbsp;`{mark_symbol}`"
            )

            # Top-3 class probabilities
            top_classes = alert_info.get("top_classes", [])
            if top_classes:
                st.markdown("**Top Class Probabilities:**")
                prob_cols = st.columns(min(len(top_classes), 3))
                for idx_c, tc in enumerate(top_classes[:3]):
                    with prob_cols[idx_c]:
                        st.caption(f"**{tc['class']}**")
                        prob_val = min(max(float(tc["probability"]), 0.0), 1.0)
                        st.progress(prob_val, text=f"{prob_val * 100:.1f}%")

            # Plain-English summary from top 3 SHAP features using raw values
            def format_raw_val(val):
                if abs(val - round(val)) < 1e-5:
                    return f"{int(round(val))}"
                return f"{val:.2f}"

            summary_clauses = []
            for item in ex_data.get("top_features", [])[:3]:
                fname = item["feature"]
                raw_v = alert_info["features"].get(fname, 0.0)
                v_str = format_raw_val(raw_v)
                direction = "pushes towards" if item["shap_value"] > 0 else "pushes against"
                summary_clauses.append(f"**{fname}** = `{v_str}` ({direction})")

            if summary_clauses:
                st.markdown(f"**Main reasons:** {', '.join(summary_clauses)}.")

            # Domain knowledge descriptions for top features
            st.markdown("**Feature Descriptions:**")
            for item in ex_data.get("top_features", [])[:4]:
                fname = item["feature"]
                desc = FEATURE_DESCRIPTIONS.get(
                    fname,
                    "Extracted network flow measurement from the CICIDS2017 dataset.",
                )
                st.markdown(f"- **{fname}**: {desc}")

            st.caption("Explanation computed with SHAP TreeExplainer on the XGBoost base learner of the stacking model.")

        # Chart: horizontal bar chart with top 8 features sorted by |SHAP|
        top8 = ex_data.get("top_features", [])[:8][::-1]  # ascending order so largest magnitude appears at top
        y_labels = []
        x_values = []
        color_categories = []
        text_labels = []

        for item in top8:
            fname = item["feature"]
            raw_v = alert_info["features"].get(fname, 0.0)
            v_str = format_raw_val(raw_v)
            y_labels.append(f"{fname} = {v_str}")
            sv = float(item["shap_value"])
            x_values.append(sv)
            color_categories.append("Pushes Toward" if sv >= 0 else "Pushes Against")
            text_labels.append(f"{sv:+.2f}")

        fig_shap = px.bar(
            x=x_values,
            y=y_labels,
            orientation="h",
            color=color_categories,
            color_discrete_map={
                "Pushes Toward": "#eb6834",   # Orange
                "Pushes Against": "#2a78d6",  # Blue
            },
            text=text_labels,
            height=420,
        )
        fig_shap.update_layout(
            margin=dict(l=10, r=20, t=20, b=10),
            xaxis_title="Contribution to the predicted class (SHAP value)",
            yaxis_title="",
            legend_title_text="",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig_shap.update_traces(textposition="outside")
        st.plotly_chart(fig_shap, use_container_width=True, config={"displayModeBar": False})

else:
    st.info("Explanations appear once flows have been classified.")

# ---- auto-play loop ----
if ss.running:
    send_batch(batch)
    time.sleep(0.4)
    st.rerun()
