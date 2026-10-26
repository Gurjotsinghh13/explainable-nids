"""Streamlit analyst dashboard for the Explainable NIDS.

Multi-page layout:
  1. Live Monitor: real-time streaming feed, KPIs, filters, severity badges, and distribution.
  2. Investigate Alert: ID-based alert inspection, verdict card, plain-English summary,
     complete 64-feature dictionary, dual-color SHAP waterfall, and JSON export.
  3. Batch Analysis: bulk flow classification via CSV upload or sample dataset.
  4. Model Performance: empirical metrics, confusion matrix, and model comparison.
  5. About: project architecture, team information, dataset facts, and limitations.

Run from the project root (with the API running):
    python -m streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# --------------------------------------------------------------------------- #
# Paths & Configuration
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPLAY_FILE = PROJECT_ROOT / "data" / "processed" / "replay_sample.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
CONFUSION_MATRIX_IMG = FIGURES_DIR / "confusion_matrix.png"
MODEL_COMPARISON_CSV = REPORTS_DIR / "model_comparison.csv"
STACKING_PER_CLASS_CSV = REPORTS_DIR / "stacking_per_class.csv"

API_URL = os.environ.get("NIDS_API", "http://127.0.0.1:8000")

# Fixed class colors across all pages
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

# Threat severity mapping
SEVERITY_MAP = {
    "DDoS": "🔴 High",
    "DoS Hulk": "🔴 High",
    "DoS GoldenEye": "🔴 High",
    "DoS slowloris": "🔴 High",
    "DoS Slowhttptest": "🔴 High",
    "Bot": "🔴 High",
    "Web Attack": "🔴 High",
    "FTP-Patator": "🟡 Medium",
    "SSH-Patator": "🟡 Medium",
    "PortScan": "🔵 Low",
    "BENIGN": "⚪ None",
}

# Complete, precise 64-feature domain description dictionary
FEATURE_DESCRIPTIONS = {
    # Packet lengths & payload metrics
    "Total Length of Fwd Packets": "Total byte volume transmitted in the forward (client-to-server) direction.",
    "Total Length of Bwd Packets": "Total byte volume returned in the backward (server-to-client) direction.",
    "Fwd Packet Length Max": "Largest individual packet payload sent in the forward direction.",
    "Fwd Packet Length Min": "Smallest packet payload size sent in the forward direction.",
    "Fwd Packet Length Mean": "Average payload size of packets sent from client to server.",
    "Fwd Packet Length Std": "Standard deviation of packet payload sizes in the forward direction.",
    "Bwd Packet Length Max": "Largest individual packet payload returned by the server.",
    "Bwd Packet Length Min": "Smallest packet payload size returned in the backward direction.",
    "Bwd Packet Length Mean": "Average payload size returned by the server; near-zero values often indicate unanswered probes.",
    "Bwd Packet Length Std": "Standard deviation of server response packet sizes.",
    "Min Packet Length": "Minimum packet size observed across all packets in the bidirectional flow.",
    "Max Packet Length": "Maximum packet size observed across all packets in the bidirectional flow.",
    "Packet Length Mean": "Mean size in bytes across all packets in both directions.",
    "Packet Length Std": "Standard deviation of packet lengths across the entire connection.",
    "Packet Length Variance": "Statistical variance of packet lengths across the session.",
    "Average Packet Size": "Overall average packet size across both forward and backward transmissions.",
    "Avg Fwd Segment Size": "Average size of transport segments observed in the forward direction.",
    "Avg Bwd Segment Size": "Average size of transport segments returned in the backward direction.",
    "Subflow Fwd Bytes": "Total byte count recorded in the forward subflow.",
    "Subflow Bwd Bytes": "Total byte count recorded in the backward subflow.",

    # Rates and connection lifetime
    "Flow Duration": "Total lifetime of the network connection in microseconds.",
    "Total Fwd Packets": "Total number of packets transmitted forward from client to server.",
    "Total Backward Packets": "Total number of packets returned from server to client.",
    "Flow Bytes/s": "Flow throughput rate in bytes per second; spikes dramatically during volumetric floods.",
    "Flow Packets/s": "Packet rate per second; elevated during SYN floods, UDP floods, and rapid port scans.",
    "Fwd Packets/s": "Number of packets transmitted forward per second.",
    "Bwd Packets/s": "Number of packets returned per second from the destination host.",
    "Down/Up Ratio": "Ratio of downloaded (backward) packets to uploaded (forward) packets.",
    "act_data_pkt_fwd": "Count of forward packets carrying at least 1 byte of application payload data.",

    # Inter-Arrival Times (IAT)
    "Flow IAT Mean": "Average elapsed time between consecutive packets in the bidirectional flow.",
    "Flow IAT Std": "Standard deviation of inter-packet intervals; scripted attacks exhibit rigid, uniform timing.",
    "Flow IAT Max": "Maximum delay recorded between two consecutive packets in the flow.",
    "Flow IAT Min": "Minimum inter-arrival time between packets; automated flood tools show mechanical, near-zero gaps.",
    "Fwd IAT Total": "Total elapsed time between the first and last packets sent forward.",
    "Fwd IAT Mean": "Average delay between consecutive outbound packets sent by the client.",
    "Fwd IAT Std": "Standard deviation of delays between consecutive forward packets.",
    "Fwd IAT Max": "Longest pause observed between two consecutive forward packets.",
    "Fwd IAT Min": "Shortest time gap between two consecutive forward packets.",
    "Bwd IAT Total": "Total elapsed time between the first and last server response packets.",
    "Bwd IAT Mean": "Average delay between consecutive reverse packets returned by the server.",
    "Bwd IAT Std": "Standard deviation of delays between consecutive backward packets.",
    "Bwd IAT Max": "Longest pause observed between two consecutive backward packets.",
    "Bwd IAT Min": "Shortest time gap between two consecutive backward packets.",

    # Protocol flags & headers
    "Fwd PSH Flags": "Number of times the Push flag was set in forward packets to bypass input buffers.",
    "Fwd URG Flags": "Number of forward packets carrying the Urgent pointer flag.",
    "Fwd Header Length": "Total byte length of TCP/IP headers across forward packets.",
    "Bwd Header Length": "Total byte length of TCP/IP headers across backward packets.",
    "FIN Flag Count": "Count of packets carrying the FIN flag, signaling graceful connection teardown.",
    "RST Flag Count": "Count of packets with the RST flag, indicating abnormal connection reset or port rejection.",
    "PSH Flag Count": "Total Push flags asserted to force immediate socket buffer flush to the application.",
    "ACK Flag Count": "Total Acknowledgement packets; critical for handshakes and flow control.",
    "URG Flag Count": "Total packets carrying the Urgent flag.",
    "ECE Flag Count": "Count of ECN-Echo flags, indicating explicit network congestion notification.",

    # Window sizes & segment metrics
    "Init_Win_bytes_forward": "Initial TCP window size advertised by client; attack tools often use fixed or unusual values.",
    "Init_Win_bytes_backward": "Initial TCP window size advertised in the server's SYN-ACK response.",
    "min_seg_size_forward": "Minimum TCP segment size observed in forward packets, set by client TCP stack.",

    # Active & Idle periods
    "Active Mean": "Mean duration the connection remained actively transmitting packets before idling.",
    "Active Std": "Standard deviation of active transmission durations.",
    "Active Max": "Longest continuous duration the flow remained actively transferring data.",
    "Active Min": "Shortest duration the connection spent actively transferring data before an idle pause.",
    "Idle Mean": "Average duration the connection remained dormant with zero packet activity.",
    "Idle Std": "Standard deviation of idle dormancy intervals.",
    "Idle Max": "Longest idle period where no packets were observed on the wire.",
    "Idle Min": "Shortest idle duration between transmission bursts; slow-rate attacks stretch this to hold sockets open.",
}

st.set_page_config(
    page_title="Explainable NIDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Helpers & API Client
# --------------------------------------------------------------------------- #
@st.cache_data
def load_replay_data():
    if REPLAY_FILE.exists():
        return pd.read_csv(REPLAY_FILE)
    return pd.DataFrame()


def api_get(path: str):
    try:
        return requests.get(f"{API_URL}{path}", timeout=5).json()
    except Exception:
        return None


def api_post(path: str, payload: dict):
    try:
        t0 = time.perf_counter()
        res = requests.post(f"{API_URL}{path}", json=payload, timeout=10).json()
        rtt_ms = (time.perf_counter() - t0) * 1000.0
        return res, rtt_ms
    except Exception as e:
        return {"error": str(e)}, 0.0


def format_raw_val(val: float | int) -> str:
    try:
        if abs(val - round(val)) < 1e-5:
            return f"{int(round(val))}"
        return f"{val:.2f}"
    except Exception:
        return str(val)


# --------------------------------------------------------------------------- #
# Session State Initialization
# --------------------------------------------------------------------------- #
ss = st.session_state
ss.setdefault("cursor", 0)
ss.setdefault("alerts", {})               # alert_id -> record dict
ss.setdefault("alert_ids", [])            # list of alert_ids (newest first)
ss.setdefault("running", False)
ss.setdefault("current_explanation", None)
ss.setdefault("total_rtt_ms", 0.0)
ss.setdefault("rtt_count", 0)
ss.setdefault("batch_results", None)

replay = load_replay_data()
feature_cols = [c for c in replay.columns if c != "Label"] if not replay.empty else []


# --------------------------------------------------------------------------- #
# Sidebar Navigation & Status
# --------------------------------------------------------------------------- #
st.sidebar.markdown("## 🛡️ NIDS Framework")

health = api_get("/health")
api_online = health is not None and health.get("status") == "ok"
if api_online:
    st.sidebar.markdown(
        '<span style="color:#2ecc71;font-weight:600;">● API Online</span> '
        f"<small>({health.get('n_features', 64)} feats · {len(health.get('classes', []))} classes)</small>",
        unsafe_allow_html=True,
    )
else:
    st.sidebar.markdown(
        '<span style="color:#e74c3c;font-weight:600;">● API Offline</span> '
        "<small>(Start with <code>python -m uvicorn api.main:app</code>)</small>",
        unsafe_allow_html=True,
    )

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["📡 Live Monitor", "🔍 Investigate Alert", "📁 Batch Analysis", "📊 Model Performance", "ℹ️ About"],
    index=0,
)

st.sidebar.markdown("---")


# =========================================================================== #
# PAGE 1: LIVE MONITOR
# =========================================================================== #
if page == "📡 Live Monitor":
    st.title("Explainable AI-Based Network Intrusion Detection (NIDS)")
    st.caption("Stacking (RF + XGBoost → LR) · within-fold SMOTE · SHAP · CICIDS2017")

    # Replay controls in sidebar
    st.sidebar.markdown("### Replay Controls")
    batch_size = st.sidebar.slider("Flows per step", 1, 25, 5)

    c1, c2 = st.sidebar.columns(2)
    if c1.button("▶ Start", use_container_width=True):
        ss.running = True
    if c2.button("⏸ Pause", use_container_width=True):
        ss.running = False
    if st.sidebar.button("↺ Reset Feed", use_container_width=True):
        ss.cursor = 0
        ss.alerts = {}
        ss.alert_ids = []
        ss.running = False
        ss.current_explanation = None
        ss.total_rtt_ms = 0.0
        ss.rtt_count = 0
        st.rerun()

    progress_val = min(max(ss.cursor / len(replay), 0.0), 1.0) if len(replay) else 0.0
    st.sidebar.progress(progress_val)
    st.sidebar.caption(f"Replay Position: {ss.cursor:,} / {len(replay):,} flows ({progress_val * 100:.1f}%)")

    # Download alerts button in sidebar
    api_alerts = api_get("/alerts?limit=1000") or []
    if api_alerts:
        df_export = pd.DataFrame(api_alerts)
        st.sidebar.download_button(
            label="📥 Download Alerts (CSV)",
            data=df_export.to_csv(index=False).encode("utf-8"),
            file_name="nids_alerts_log.csv",
            mime="text/csv",
            use_container_width=True,
        )

    def send_batch(n):
        if not api_online:
            ss.running = False
            return
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
                    "severity": SEVERITY_MAP.get(r["predicted_label"], "⚪ None"),
                    "top_classes": r.get("top_classes", []),
                    "features": payload["features"],
                }
                if aid not in ss.alert_ids:
                    ss.alert_ids.insert(0, aid)
                ss.total_rtt_ms += rtt_ms
                ss.rtt_count += 1
            ss.cursor += 1

    # KPI Metrics Row
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

    # Cumulative line chart under KPIs
    if ss.alert_ids:
        # Build chronological history
        ordered_alerts = [ss.alerts[aid] for aid in reversed(ss.alert_ids)]
        timeline_df = pd.DataFrame(ordered_alerts)
        timeline_df["flow_idx"] = np.arange(1, len(timeline_df) + 1)
        timeline_df["is_attack"] = (timeline_df["predicted"] != "BENIGN").astype(int)
        timeline_df["cumulative_attacks"] = timeline_df["is_attack"].cumsum()

        fig_line = px.line(
            timeline_df,
            x="flow_idx",
            y="cumulative_attacks",
            title="Cumulative Detected Attacks Over Time",
            color_discrete_sequence=["#eb6834"],
            height=220,
        )
        fig_line.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="Flow Sequence",
            yaxis_title="Total Attacks",
        )
        st.plotly_chart(fig_line, use_container_width=True, config={"displayModeBar": False})

    st.markdown("---")

    left, right = st.columns([3, 2])

    with left:
        st.subheader("Live classification feed")
        st.caption("Demo replay: balanced sample of 1,650 real CICIDS2017 test flows (150 per class). Real traffic is about 80% benign.")

        # Filter controls
        f_col1, f_col2 = st.columns([1, 2])
        only_attacks = f_col1.toggle("Show only attacks", value=False)
        all_cls = list(CLASS_COLORS.keys())
        class_filter = f_col2.multiselect("Filter by predicted class", options=all_cls, default=[])

        if ss.alert_ids:
            feed_rows = []
            for aid in ss.alert_ids:
                al = ss.alerts[aid]
                if only_attacks and al["predicted"] == "BENIGN":
                    continue
                if class_filter and al["predicted"] not in class_filter:
                    continue
                feed_rows.append({
                    "Alert ID": f"#{al['id']}",
                    "Time": al["time"],
                    "Severity": al["severity"],
                    "Predicted": al["predicted"],
                    "True Label": al["true"],
                    "Confidence": f"{al['confidence'] * 100:.1f}%",
                    "Latency ms": f"{al['latency_ms']:.1f}",
                    "Status": "✓" if al["predicted"] == al["true"] else "✗",
                    "_misclassified": al["predicted"] != al["true"],
                    "_attack": al["predicted"] != "BENIGN",
                })
                if len(feed_rows) >= 30:
                    break

            if feed_rows:
                feed_df = pd.DataFrame(feed_rows)

                def style_feed_row(row):
                    if row["_misclassified"]:
                        return ["background-color: rgba(243, 156, 18, 0.35); font-weight: 500;"] * len(row)
                    elif row["_attack"]:
                        return ["background-color: rgba(235, 104, 52, 0.22);"] * len(row)
                    else:
                        return [""] * len(row)

                display_cols = ["Alert ID", "Time", "Severity", "Predicted", "True Label", "Confidence", "Latency ms", "Status"]
                styled_table = feed_df.style.apply(style_feed_row, axis=1)
                st.dataframe(styled_table, column_order=display_cols, hide_index=True, use_container_width=True, height=410)
            else:
                st.info("No alerts match the selected filters.")

            st.caption("⚪ **Neutral**: Correct Benign &nbsp;|&nbsp; 🔴 **Light Red**: Correct Attack &nbsp;|&nbsp; 🟡 **Amber**: Misclassified Alert")
        else:
            st.info("Press ▶ Start to begin replaying flows.")

    with right:
        st.subheader("Attack-type distribution")
        dist = stats.get("distribution", {})
        if dist:
            dd = pd.DataFrame({"class": list(dist.keys()), "count": list(dist.values())})
            # Horizontal bars sorted descending so labels don't rotate
            dd = dd.sort_values("count", ascending=True)
            fig_dist = px.bar(
                dd,
                x="count",
                y="class",
                orientation="h",
                color="class",
                color_discrete_map=CLASS_COLORS,
                height=460,
            )
            fig_dist.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Alert Count",
                yaxis_title="",
                showlegend=False,
            )
            st.plotly_chart(fig_dist, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("No alerts logged in the database yet.")

    # Auto-play loop
    if ss.running:
        send_batch(batch_size)
        time.sleep(0.4)
        st.rerun()


# =========================================================================== #
# PAGE 2: INVESTIGATE ALERT
# =========================================================================== #
elif page == "🔍 Investigate Alert":
    st.title("🔍 Investigate Alert & SHAP Explainability")
    st.caption("Inspect individual network flows with local SHAP feature attribution on the XGBoost base learner")

    if not api_online:
        st.warning("⚠️ API is offline. Start the backend with `python -m uvicorn api.main:app` to generate explanations.")

    if ss.alert_ids:
        options = ss.alert_ids[:60]

        def format_alert_option(aid):
            al = ss.alerts[aid]
            mark = "✓" if al["predicted"] == al["true"] else "✗"
            return f"#{aid} — {al['predicted']} (true: {al['true']}) [{mark}]"

        sel_col, btn_col = st.columns([3, 1])
        selected_alert_id = sel_col.selectbox(
            "Select an alert to inspect",
            options=options,
            format_func=format_alert_option,
            key="selected_alert_id_investigate",
        )

        if btn_col.button("Explain selected alert", use_container_width=True):
            ss.running = False  # automatically pause replay while examining
            chosen = ss.alerts[selected_alert_id]
            ex_res, _ = api_post("/explain", {"features": chosen["features"]})
            if "error" in ex_res:
                st.error(f"Explain failed: {ex_res['error']}")
            else:
                ss.current_explanation = {
                    "alert_id": selected_alert_id,
                    "data": ex_res,
                }

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

            # Verdict line
            is_corr = alert_info["predicted"] == alert_info["true"]
            mark_symbol = "✓" if is_corr else "✗"
            conf_pct = alert_info["confidence"] * 100

            st.markdown("---")
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

            # Analyst Summary Box (Template-based, no LLM)
            top_items = ex_data.get("top_features", [])
            pos_reasons = []
            neg_reasons = []
            for item in top_items:
                fname = item["feature"]
                raw_v = alert_info["features"].get(fname, 0.0)
                v_str = format_raw_val(raw_v)
                if item["shap_value"] > 0:
                    pos_reasons.append(f"{fname} = {v_str}")
                else:
                    neg_reasons.append(f"{fname} = {v_str}")

            context_signatures = {
                "DDoS": "massive volumetric flooding with elevated packet rates and abnormal payload lengths",
                "DoS Hulk": "high-rate HTTP GET floods designed to exhaust web server worker threads",
                "DoS GoldenEye": "persistent keep-alive connection exhaustion with characteristic TCP window sizes",
                "DoS slowloris": "incomplete slow-rate HTTP header transmission designed to tie up active sockets",
                "DoS Slowhttptest": "slow HTTP POST body transmissions maintaining prolonged connection lifetimes",
                "Bot": "automated command-and-control beaconing with uniform packet inter-arrival times",
                "PortScan": "rapid host reconnaissance probes scanning multiple port addresses",
                "FTP-Patator": "repetitive automated credential brute-force authentication requests",
                "SSH-Patator": "iterative SSH encrypted handshake connection attempts",
                "Web Attack": "injection, cross-site scripting (XSS), or brute-force web exploits",
                "BENIGN": "normal bidirectional TCP/IP flow dynamics and protocol handshakes",
            }
            pred_lbl = alert_info["predicted"]
            ctx = context_signatures.get(pred_lbl, "characteristic intrusion behaviors")

            pos_text = f"was driven primarily by {', '.join(pos_reasons[:2])}" if pos_reasons else "exhibited characteristic attack signatures"
            neg_text = f", while {neg_reasons[0]} slightly opposed this classification" if neg_reasons else ""

            summary_text = (
                f"**Analyst Summary:** This network flow was classified as **{pred_lbl}** ({conf_pct:.1f}% confidence). "
                f"The decision {pos_text}{neg_text}. "
                f"This pattern strongly aligns with {ctx} documented in the CICIDS2017 benchmark dataset."
            )
            st.info(summary_text)

            # Chart: Horizontal bar chart with top 8 features sorted by |SHAP|
            top8 = top_items[:8][::-1]
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
                color_categories.append(f"Pushes toward {pred_lbl}" if sv >= 0 else "Pushes against")
                text_labels.append(f"{sv:+.2f}")

            color_map = {
                f"Pushes toward {pred_lbl}": "#eb6834",
                "Pushes against": "#2a78d6",
            }

            fig_shap = px.bar(
                x=x_values,
                y=y_labels,
                orientation="h",
                color=color_categories,
                color_discrete_map=color_map,
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

            # Feature descriptions for top 4 features
            st.markdown("#### Feature Descriptions")
            for item in top_items[:5]:
                fname = item["feature"]
                desc = FEATURE_DESCRIPTIONS.get(fname, "Network flow measurement from the CICIDS2017 dataset.")
                st.markdown(f"- **{fname}**: {desc}")

            st.caption("Explanation computed with SHAP TreeExplainer on the XGBoost base learner of the stacking model.")

            # Download JSON button
            export_payload = {
                "alert_id": alert_info["id"],
                "timestamp": alert_info["time"],
                "predicted_label": alert_info["predicted"],
                "true_label": alert_info["true"],
                "confidence": alert_info["confidence"],
                "inference_ms": alert_info["latency_ms"],
                "top_classes": alert_info.get("top_classes", []),
                "top_features": ex_data.get("top_features", []),
                "raw_features": alert_info["features"],
            }
            st.download_button(
                label="📥 Download Explanation (JSON)",
                data=json.dumps(export_payload, indent=2),
                file_name=f"alert_{alert_info['id']}_explanation.json",
                mime="application/json",
            )
    else:
        st.info("No alerts recorded yet. Go to **📡 Live Monitor** and start the replay to capture alerts.")


# =========================================================================== #
# PAGE 3: BATCH ANALYSIS
# =========================================================================== #
elif page == "📁 Batch Analysis":
    st.title("📁 Batch Traffic Analysis")
    st.caption("Upload or load offline flow datasets for batch intrusion classification (Max 2,000 flows)")

    if not api_online:
        st.warning("⚠️ API is offline. Start the backend with `python -m uvicorn api.main:app` before running batch analysis.")

    col_u, col_s = st.columns([2, 1])
    uploaded_file = col_u.file_uploader("Upload CSV file (same columns as CICIDS2017)", type=["csv"])
    use_sample = col_s.button("Use Sample File (200 flows)", use_container_width=True)

    batch_df = None
    if use_sample:
        if not replay.empty:
            batch_df = replay.sample(min(200, len(replay)), random_state=42).copy()
            st.success("Loaded 200 random flows from `replay_sample.csv`.")
    elif uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Uploaded `{uploaded_file.name}` with {len(batch_df):,} rows.")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

    if batch_df is not None:
        if len(batch_df) > 2000:
            st.warning("Batch truncated to first 2,000 rows for real-time demonstration stability.")
            batch_df = batch_df.iloc[:2000]

        has_label = "Label" in batch_df.columns
        feats = [c for c in batch_df.columns if c != "Label"]

        if st.button("🚀 Run Batch Classification", type="primary"):
            if not api_online:
                st.error("API is offline. Cannot process batch.")
            else:
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                results = []

                total_rows = len(batch_df)
                t_start = time.perf_counter()

                for i, (_, row) in enumerate(batch_df.iterrows()):
                    payload = {
                        "features": {c: float(row[c]) for c in feats if c in row and pd.notna(row[c])},
                        "true_label": str(row["Label"]) if has_label else None,
                    }
                    r, rtt = api_post("/predict", payload)
                    if "error" not in r:
                        results.append({
                            "Index": i + 1,
                            "Predicted": r["predicted_label"],
                            "Confidence": r["confidence"],
                            "Inference ms": r["inference_ms"],
                            "True Label": str(row["Label"]) if has_label else "N/A",
                            "Correct": (r["predicted_label"] == str(row["Label"])) if has_label else None,
                        })
                    progress_bar.progress((i + 1) / total_rows)
                    if (i + 1) % 25 == 0 or (i + 1) == total_rows:
                        status_text.text(f"Processed {i + 1:,} of {total_rows:,} flows...")

                elapsed = time.perf_counter() - t_start
                status_text.success(f"Batch completed in {elapsed:.2f} seconds ({elapsed / total_rows * 1000:.1f} ms/flow).")
                ss.batch_results = pd.DataFrame(results)

    if ss.batch_results is not None:
        b_res = ss.batch_results
        st.markdown("---")
        st.subheader("Batch Results Summary")

        tot = len(b_res)
        attacks_found = sum(b_res["Predicted"] != "BENIGN")
        mean_conf = b_res["Confidence"].mean() * 100

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Flows", f"{tot:,}")
        k2.metric("Attacks Found", f"{attacks_found:,}")
        k3.metric("Attack Ratio", f"{(attacks_found / tot) * 100:.1f}%")
        k4.metric("Mean Confidence", f"{mean_conf:.1f}%")

        if "Correct" in b_res and b_res["Correct"].notna().any():
            acc_val = b_res["Correct"].mean() * 100
            st.metric("Batch Accuracy", f"{acc_val:.1f}%")

        # Class counts bar chart
        st.markdown("#### Predicted Threat Distribution")
        cnts = b_res["Predicted"].value_counts().reset_index()
        cnts.columns = ["class", "count"]
        fig_batch = px.bar(
            cnts,
            x="class",
            y="count",
            color="class",
            color_discrete_map=CLASS_COLORS,
            height=340,
        )
        fig_batch.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig_batch, use_container_width=True, config={"displayModeBar": False})

        # Confusion Matrix if Label is present
        if "True Label" in b_res and (b_res["True Label"] != "N/A").any():
            st.markdown("#### Batch Confusion Matrix")
            labels_present = sorted(list(set(b_res["True Label"]).union(set(b_res["Predicted"]))))
            cm_df = pd.crosstab(
                b_res["True Label"],
                b_res["Predicted"],
                rownames=["Actual"],
                colnames=["Predicted"],
                dropna=False,
            ).reindex(index=labels_present, columns=labels_present, fill_value=0)
            st.dataframe(cm_df, use_container_width=True)

        # Download results as CSV
        csv_data = b_res.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Batch Predictions (CSV)",
            data=csv_data,
            file_name="batch_nids_predictions.csv",
            mime="text/csv",
        )


# =========================================================================== #
# PAGE 4: MODEL PERFORMANCE
# =========================================================================== #
elif page == "📊 Model Performance":
    st.title("📊 Model Performance & Benchmark Evaluation")
    st.caption("Empirical evidence from the CICIDS2017 held-out test set (565,566 flows) and 5-fold cross-validation")

    # Key Results Card
    st.markdown("### 🏆 Key Verified Results")
    kr1, kr2, kr3, kr4 = st.columns(4)
    kr1.metric("Overall Accuracy", "99.9%")
    kr2.metric("Macro-F1 Score", "0.976")
    kr3.metric("ROC-AUC (OVR)", "0.9998")
    kr4.metric("Macro FPR", "1.8 × 10⁻⁴")

    kr5, kr6, kr7, kr8 = st.columns(4)
    kr5.metric("Bot Recall Recovery", "0.499 ➔ 0.844", delta="+69% vs RF")
    kr6.metric("5-Fold CV Macro-F1", "0.975 ± 0.003", help="Cross-validation stability test")
    kr7.metric("Inference Latency", "32.5 ms / flow")
    kr8.metric("SHAP Explanation", "7.0 ms / flow")

    st.markdown("---")

    # Section 1: Baseline vs Stacking Comparison
    st.subheader("1. Baseline vs. Stacking Ensemble Comparison")
    if MODEL_COMPARISON_CSV.exists():
        comp_df = pd.read_csv(MODEL_COMPARISON_CSV)
        st.dataframe(comp_df, hide_index=True, use_container_width=True)

        # Grouped bar chart comparing models
        fig_comp = go.Figure()
        metrics_to_plot = [c for c in comp_df.columns if c != "model"]
        for m in ["accuracy", "macro_F1", "Bot_recall"]:
            if m in comp_df.columns:
                fig_comp.add_trace(go.Bar(
                    name=m.replace("_", " ").title(),
                    x=comp_df["model"],
                    y=comp_df[m],
                    text=[f"{v:.4f}" for v in comp_df[m]],
                    textposition="auto",
                ))
        fig_comp.update_layout(
            barmode="group",
            height=380,
            margin=dict(l=10, r=10, t=30, b=10),
            yaxis=dict(range=[0.4, 1.05]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_comp, use_container_width=True, config={"displayModeBar": False})
    else:
        st.warning("`reports/model_comparison.csv` not found.")

    st.markdown("---")

    # Section 2: Per-Class Evaluation Table
    st.subheader("2. Per-Class Stacking Performance")
    if STACKING_PER_CLASS_CSV.exists():
        per_class_df = pd.read_csv(STACKING_PER_CLASS_CSV)
        if "Unnamed: 0" in per_class_df.columns:
            per_class_df = per_class_df.rename(columns={"Unnamed: 0": "Class"})

        st.caption("Evaluation on held-out 20% test set (565,566 real flows). Minority classes highlighted for recovery analysis.")

        def highlight_weak(row):
            # Highlight classes with low support or bot recall
            if row["Class"] in ["Bot", "Web Attack"]:
                return ["background-color: rgba(235, 104, 52, 0.25)"] * len(row)
            return [""] * len(row)

        st.dataframe(per_class_df.style.apply(highlight_weak, axis=1), hide_index=True, use_container_width=True)
    else:
        st.warning("`reports/stacking_per_class.csv` not found.")

    st.markdown("---")

    # Section 3: Confusion Matrix & Global SHAP Figures
    st.subheader("3. Confusion Matrix")
    if CONFUSION_MATRIX_IMG.exists():
        st.image(str(CONFUSION_MATRIX_IMG), caption="Normalized Confusion Matrix on 565,566 Held-Out Test Flows", use_column_width=True)
    else:
        st.info("Confusion matrix image not found at `reports/figures/confusion_matrix.png`.")

    global_shap_img = FIGURES_DIR / "shap_summary.png"
    if global_shap_img.exists():
        st.subheader("4. Global SHAP Summary")
        st.image(str(global_shap_img), caption="Global SHAP Summary Plot on Held-Out Test Flows", use_column_width=True)


# =========================================================================== #
# PAGE 5: ABOUT
# =========================================================================== #
elif page == "ℹ️ About":
    st.title("ℹ️ About the Project")
    st.subheader("Explainable AI-Based Network Intrusion Detection Framework Using Ensemble Machine Learning for Near Real-Time Threat Detection")
    st.caption("B.Tech Capstone Project (2026-27) · MPSTME Shirpur, SVKM's NMIMS")

    col_team, col_guide = st.columns(2)
    with col_team:
        st.markdown("### 👥 Student Team")
        st.markdown(
            """
            - **Gurjot Singh** (Roll No. B205)
            - **Lehar Nimodiya** (Roll No. B233)
            - **Lokesh Ingale** (Roll No. B250)
            """
        )
    with col_guide:
        st.markdown("### 🎓 Project Guide")
        st.markdown(
            """
            - **Dr. Naziya Hussain**  
              *Department of Computer Engineering, MPSTME Shirpur*
            """
        )

    st.markdown("---")

    st.subheader("Architecture & Pipeline")
    st.markdown(
        """
        ```
        [ Raw CICIDS2017 PCAP CSVs ]
                    │
                    ▼
        [ Leakage-Free Preprocessing ]
          • Drops 2,867 inf/NaN rows, constants & duplicate features
          • Eliminates socket identifiers (Destination Port leakage removed)
          • Result: 64 network features across 11 unified classes
                    │
                    ▼
        [ Stratified 80/20 Hold-Out Split ] (2,262,263 Train / 565,566 Test)
                    │
                    ▼
        [ 5-Fold Stratified Stacking Classifier ]
          ├── Base Learner 1: Random Forest (100 trees, depth 30)
          │     └── Within-Fold SMOTE (Floor = 50k, runs only on training folds)
          ├── Base Learner 2: XGBoost (100 estimators, depth 8, CUDA hist)
          │     └── Within-Fold SMOTE (Floor = 50k, runs only on training folds)
          └── Meta-Learner: Logistic Regression (predict_proba)
                    │
                    ▼
        [ Explainability & Deployment Layer ]
          ├── SHAP TreeExplainer on XGBoost Base Learner (CPU inference: 7.0 ms)
          ├── FastAPI Inference Engine (SQLite alert database)
          └── Streamlit Analyst Dashboard (real-time stream replay & batch audit)
        ```
        """
    )

    st.markdown("---")

    col_facts, col_limits = st.columns(2)
    with col_facts:
        st.markdown("### 📋 Dataset Facts (CICIDS2017)")
        st.markdown(
            """
            - **Source:** Canadian Institute for Cybersecurity (CICIDS2017 MachineLearningCVE)
            - **Total Valid Flows:** 2,827,829 flows (Heartbleed & Infiltration excluded due to <40 instances)
            - **Features:** 64 scaled continuous & discrete flow features
            - **Classes (11):** BENIGN, Bot, DDoS, DoS GoldenEye, DoS Hulk, DoS Slowhttptest, DoS slowloris, FTP-Patator, PortScan, SSH-Patator, Web Attack
            - **Split:** Stratified 80% train (2,262,263) / 20% test (565,566)
            """
        )

    with col_limits:
        st.markdown("### ⚠️ Recognized Limitations")
        st.markdown(
            """
            1. **Benchmark Evaluation:** The model is trained and validated on the CICIDS2017 dataset; cross-dataset evaluation (e.g. on CSE-CIC-IDS2018 or UNSW-NB15) is planned for future research.
            2. **Replay vs. Kernel Sniffing:** The demonstration operates via serialized replay of real captured test flows rather than real-time kernel packet capture (e.g. Scapy or PF_RING).
            3. **Base Learner Explanation:** TreeExplainer computes exact Shapley values on the XGBoost base learner rather than the Logistic Regression meta-learner directly.
            """
        )
