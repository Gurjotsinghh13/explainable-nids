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

# Restrained, professional cybersecurity color palette
CLASS_COLORS = {
    "BENIGN": "#10b981",          # Emerald green (normal traffic)
    "Bot": "#dc2626",             # Red (critical automation)
    "DDoS": "#ea580c",            # Orange (volumetric flood)
    "DoS GoldenEye": "#c2410c",   # Deep orange
    "DoS Hulk": "#b91c1c",        # Crimson
    "DoS Slowhttptest": "#d97706",# Amber
    "DoS slowloris": "#e11d48",   # Rose
    "FTP-Patator": "#7c3aed",     # Violet
    "PortScan": "#2563eb",        # Slate blue
    "SSH-Patator": "#9333ea",     # Purple
    "Web Attack": "#0d9488",      # Teal
}

# Threat severity mapping
SEVERITY_MAP = {
    "DDoS": "High",
    "DoS Hulk": "High",
    "DoS GoldenEye": "High",
    "DoS slowloris": "High",
    "DoS Slowhttptest": "High",
    "Bot": "High",
    "Web Attack": "High",
    "FTP-Patator": "Medium",
    "SSH-Patator": "Medium",
    "PortScan": "Low",
    "BENIGN": "None",
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
# Professional Light Enterprise CSS
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
    /* Clean font stack & layout constraints */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 1380px;
    }
    
    /* Enterprise header */
    .soc-header {
        margin-bottom: 1.25rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid #e2e8f0;
    }
    .soc-header h1 {
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        color: #0f172a !important;
        letter-spacing: -0.02em;
        margin: 0 0 0.2rem 0 !important;
        line-height: 1.25 !important;
    }
    .soc-header p {
        font-size: 0.85rem !important;
        color: #64748b !important;
        margin: 0 !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    [data-testid="stSidebar"] h2 {
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        color: #0f172a !important;
        margin-top: 0.5rem !important;
    }
    [data-testid="stSidebar"] h3 {
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        color: #475569 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
    
    /* Metric Cards */
    [data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 6px !important;
        padding: 10px 14px !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03) !important;
    }
    [data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
    }
    [data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }
    
    /* Buttons */
    button[kind="primary"] {
        background-color: #1e40af !important;
        border-color: #1e40af !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border-radius: 5px !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
    }
    button[kind="secondary"] {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        color: #334155 !important;
        font-weight: 500 !important;
        border-radius: 5px !important;
    }
    button[kind="secondary"]:hover {
        background-color: #f8fafc !important;
        border-color: #94a3b8 !important;
        color: #0f172a !important;
    }
    
    /* Alerts & Notifications */
    [data-testid="stAlert"] {
        border-radius: 6px !important;
        border: 1px solid #e2e8f0 !important;
        font-size: 0.875rem !important;
    }
    
    /* Tables & Dataframes */
    [data-testid="stDataFrame"] {
        border: 1px solid #e2e8f0 !important;
        border-radius: 6px !important;
        overflow: hidden !important;
    }
    
    /* Headings */
    h2, h3, h4 {
        color: #0f172a !important;
        font-weight: 600 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
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


def apply_plot_theme(fig, height: int | None = None):
    """Enforces a clean light-mode enterprise aesthetic on Plotly charts."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", size=12, color="#334155"),
        title_font=dict(size=13, color="#0f172a", family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"),
        legend=dict(
            font=dict(size=11, color="#475569"),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="#e2e8f0",
            borderwidth=1,
        ),
        xaxis=dict(
            gridcolor="#f1f5f9",
            linecolor="#cbd5e1",
            zerolinecolor="#e2e8f0",
            tickfont=dict(size=11, color="#64748b"),
            title_font=dict(size=12, color="#334155"),
        ),
        yaxis=dict(
            gridcolor="#f1f5f9",
            linecolor="#cbd5e1",
            zerolinecolor="#e2e8f0",
            tickfont=dict(size=11, color="#64748b"),
            title_font=dict(size=12, color="#334155"),
        ),
    )
    if height:
        fig.update_layout(height=height)
    return fig


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
ss.setdefault("class_counts", {cls: 0 for cls in CLASS_COLORS.keys()})

# Synchronize class_counts with alert_ids if alerts exist in session
if sum(ss.class_counts.values()) != len(ss.alert_ids):
    ss.class_counts = {cls: 0 for cls in CLASS_COLORS.keys()}
    for aid in ss.alert_ids:
        lbl = ss.alerts.get(aid, {}).get("predicted")
        if lbl in ss.class_counts:
            ss.class_counts[lbl] += 1

replay = load_replay_data()
feature_cols = [c for c in replay.columns if c != "Label"] if not replay.empty else []


# --------------------------------------------------------------------------- #
# Sidebar Navigation & System Status
# --------------------------------------------------------------------------- #
st.sidebar.markdown("## 🛡️ NIDS Framework")

health = api_get("/health")
api_online = health is not None and health.get("status") == "ok"
if api_online:
    st.sidebar.markdown(
        '<div style="font-size:12px; font-weight:600; color:#16a34a; margin-bottom:8px;">'
        '● System Online '
        f'<span style="color:#64748b; font-weight:400;">({health.get("n_features", 64)} feats · {len(health.get("classes", []))} classes)</span>'
        '</div>',
        unsafe_allow_html=True,
    )
else:
    st.sidebar.markdown(
        '<div style="font-size:12px; font-weight:600; color:#dc2626; margin-bottom:8px;">'
        '● System Offline '
        '<span style="color:#64748b; font-weight:400;">(Start backend on :8000)</span>'
        '</div>',
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
    st.markdown(
        """
        <div class="soc-header">
            <h1>Network Intrusion Detection System</h1>
            <p>Real-time flow telemetry, anomaly classification & ML decision provenance · CICIDS2017</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
        ss.class_counts = {cls: 0 for cls in CLASS_COLORS.keys()}
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
                pred_label = r["predicted_label"]
                ss.alerts[aid] = {
                    "id": aid,
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "predicted": pred_label,
                    "true": str(row["Label"]),
                    "confidence": float(r["confidence"]),
                    "latency_ms": float(r["inference_ms"]),
                    "round_trip_ms": rtt_ms,
                    "severity": SEVERITY_MAP.get(pred_label, "None"),
                    "top_classes": r.get("top_classes", []),
                    "features": payload["features"],
                }
                if aid not in ss.alert_ids:
                    ss.alert_ids.insert(0, aid)
                ss.class_counts[pred_label] = ss.class_counts.get(pred_label, 0) + 1
                ss.total_rtt_ms += rtt_ms
                ss.rtt_count += 1
            ss.cursor += 1

    # KPI Metrics Row
    n_proc = len(ss.alert_ids)
    n_attacks = sum(1 for aid in ss.alert_ids if ss.alerts[aid]["predicted"] != "BENIGN")
    acc = (sum(1 for aid in ss.alert_ids if ss.alerts[aid]["true"] == ss.alerts[aid]["predicted"]) / n_proc) if n_proc else 0.0
    avg_infer = (sum(ss.alerts[aid]["latency_ms"] for aid in ss.alert_ids) / n_proc) if n_proc else 0.0
    avg_rtt = (ss.total_rtt_ms / ss.rtt_count) if ss.rtt_count else 0.0

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Flows Processed", f"{n_proc:,}")
    m2.metric("Attacks Flagged", f"{n_attacks:,}")
    m3.metric("Live Accuracy", f"{acc * 100:.1f}%")
    m4.metric("Avg Inference", f"{avg_infer:.1f} ms")
    m5.metric("Avg Round Trip", f"{avg_rtt:.1f} ms")

    # Cumulative line chart under KPIs
    if ss.alert_ids:
        ordered_alerts = [ss.alerts[aid] for aid in reversed(ss.alert_ids)]
        timeline_df = pd.DataFrame(ordered_alerts)
        timeline_df["flow_idx"] = np.arange(1, len(timeline_df) + 1)
        timeline_df["is_attack"] = (timeline_df["predicted"] != "BENIGN").astype(int)
        timeline_df["cumulative_attacks"] = timeline_df["is_attack"].cumsum()

        fig_line = px.line(
            timeline_df,
            x="flow_idx",
            y="cumulative_attacks",
            title="Cumulative Detected Threats Over Flow Stream",
            color_discrete_sequence=["#dc2626"],
            height=200,
        )
        apply_plot_theme(fig_line, height=200)
        fig_line.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="Flow Sequence",
            yaxis_title="Total Attacks",
        )
        st.plotly_chart(fig_line, use_container_width=True, config={"displayModeBar": False})

    st.markdown("---")

    left, right = st.columns([3, 2])

    with left:
        st.subheader("Live Classification Feed")
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
                        return ["background-color: #fffbeb; color: #92400e; font-weight: 500;"] * len(row)
                    elif row["_attack"]:
                        return ["background-color: #fef2f2; color: #991b1b; font-weight: 500;"] * len(row)
                    else:
                        return ["background-color: #ffffff; color: #1e293b;"] * len(row)

                display_cols = ["Alert ID", "Time", "Severity", "Predicted", "True Label", "Confidence", "Latency ms", "Status"]
                styled_table = feed_df.style.apply(style_feed_row, axis=1)
                st.dataframe(styled_table, column_order=display_cols, hide_index=True, use_container_width=True, height=410)
            else:
                st.info("No alerts match the selected filters.")

            st.markdown(
                """
                <div style="display:flex; flex-wrap:wrap; gap:16px; font-size:12px; color:#64748b; margin-top:8px;">
                    <span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:#ffffff;border:1px solid #cbd5e1;margin-right:6px;vertical-align:middle;"></span><strong>Neutral:</strong> Correct Benign</span>
                    <span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:#fef2f2;border:1px solid #fecaca;margin-right:6px;vertical-align:middle;"></span><strong style="color:#991b1b;">Light Red:</strong> Correct Attack</span>
                    <span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:#fffbeb;border:1px solid #fde68a;margin-right:6px;vertical-align:middle;"></span><strong style="color:#92400e;">Amber:</strong> Misclassified Flow</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("Press ▶ Start in the sidebar to begin streaming flows.")

    with right:
        st.subheader("Attack-Type Distribution")
        session_dist = {cls: count for cls, count in ss.class_counts.items() if count > 0}
        if session_dist:
            dd = pd.DataFrame({"class": list(session_dist.keys()), "count": list(session_dist.values())})
            dd = dd.sort_values("count", ascending=True)
            fig_dist = px.bar(
                dd,
                x="count",
                y="class",
                orientation="h",
                color="class",
                color_discrete_map=CLASS_COLORS,
                height=450,
            )
            apply_plot_theme(fig_dist, height=450)
            fig_dist.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Alert Count",
                yaxis_title="",
                showlegend=False,
            )
            fig_dist.update_xaxes(tickformat="d")
            st.plotly_chart(fig_dist, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("No alerts logged in the current session yet. Press ▶ Start in the sidebar to stream flows.")

    # Auto-play loop
    if ss.running:
        send_batch(batch_size)
        time.sleep(0.4)
        st.rerun()


# =========================================================================== #
# PAGE 2: INVESTIGATE ALERT
# =========================================================================== #
elif page == "🔍 Investigate Alert":
    st.markdown(
        """
        <div class="soc-header">
            <h1>Investigate Alert & SHAP Explainability</h1>
            <p>Feature attribution and ML decision provenance via SHAP TreeExplainer on the XGBoost base learner</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

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

        if btn_col.button("Explain Selected Alert", use_container_width=True, type="primary"):
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

            is_corr = alert_info["predicted"] == alert_info["true"]
            mark_symbol = "✓ Correct" if is_corr else "✗ Misclassified"
            conf_pct = alert_info["confidence"] * 100
            is_attack = alert_info["predicted"] != "BENIGN"

            status_bg = "#fef2f2" if is_attack else "#f0fdf4"
            status_text_color = "#991b1b" if is_attack else "#166534"
            status_border = "#fecaca" if is_attack else "#bbf7d0"
            pred_color = CLASS_COLORS.get(alert_info["predicted"], "#0f172a")

            verdict_card_html = f"""
            <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:6px; padding:16px 20px; margin: 12px 0 16px 0; box-shadow:0 1px 2px rgba(0,0,0,0.03);">
                <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:12px;">
                    <div>
                        <div style="font-size:11px; text-transform:uppercase; font-weight:600; letter-spacing:0.05em; color:#64748b;">Alert Classification Verdict</div>
                        <div style="font-size:1.25rem; font-weight:700; color:#0f172a; margin-top:4px;">
                            Flagged as <span style="color:{pred_color};">{alert_info['predicted']}</span>
                            <span style="font-size:0.95rem; font-weight:500; color:#475569;">({conf_pct:.1f}% confidence)</span>
                        </div>
                        <div style="font-size:12px; color:#64748b; margin-top:4px;">
                            Ground Truth: <strong style="color:#1e293b;">{alert_info['true']}</strong> &nbsp;·&nbsp; 
                            Threat Severity: <strong style="color:#1e293b;">{alert_info['severity']}</strong>
                        </div>
                    </div>
                    <div style="text-align:right;">
                        <span style="display:inline-block; padding:4px 10px; border-radius:4px; font-size:12px; font-weight:600; background:{status_bg}; color:{status_text_color}; border:1px solid {status_border};">
                            {mark_symbol}
                        </span>
                        <div style="font-size:11px; color:#64748b; margin-top:6px;">Inference Latency: {alert_info['latency_ms']:.1f} ms</div>
                    </div>
                </div>
            </div>
            """
            st.markdown(verdict_card_html, unsafe_allow_html=True)

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

            # Analyst Summary Box (Template-based, deterministic)
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
                f"This network flow was classified as <strong>{pred_lbl}</strong> ({conf_pct:.1f}% confidence). "
                f"The model's decision {pos_text}{neg_text}. "
                f"This pattern aligns with {ctx} documented in the CICIDS2017 benchmark dataset."
            )
            analyst_html = f"""
            <div style="background:#f8fafc; border:1px solid #e2e8f0; border-left:4px solid #2563eb; border-radius:4px; padding:12px 16px; margin:14px 0;">
                <div style="font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; color:#1e40af; margin-bottom:4px;">Analyst Decision Summary</div>
                <div style="font-size:13px; color:#1e293b; line-height:1.5;">{summary_text}</div>
            </div>
            """
            st.markdown(analyst_html, unsafe_allow_html=True)

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
                f"Pushes toward {pred_lbl}": "#dc2626",
                "Pushes against": "#2563eb",
            }

            fig_shap = px.bar(
                x=x_values,
                y=y_labels,
                orientation="h",
                color=color_categories,
                color_discrete_map=color_map,
                text=text_labels,
                height=400,
            )
            apply_plot_theme(fig_shap, height=400)
            fig_shap.update_layout(
                margin=dict(l=10, r=20, t=20, b=10),
                xaxis_title="Contribution to predicted class (SHAP value)",
                yaxis_title="",
                legend_title_text="",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            fig_shap.update_traces(textposition="outside")
            st.plotly_chart(fig_shap, use_container_width=True, config={"displayModeBar": False})

            # Feature descriptions table
            st.markdown("#### Top Influential Feature Descriptions")
            desc_rows = []
            for item in top_items[:5]:
                fname = item["feature"]
                raw_v = alert_info["features"].get(fname, 0.0)
                desc = FEATURE_DESCRIPTIONS.get(fname, "Network flow measurement from the CICIDS2017 dataset.")
                desc_rows.append({
                    "Feature": fname,
                    "Observed Value": format_raw_val(raw_v),
                    "SHAP Impact": f"{float(item['shap_value']):+.4f}",
                    "Domain Interpretation": desc,
                })
            st.dataframe(pd.DataFrame(desc_rows), hide_index=True, use_container_width=True)

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
    st.markdown(
        """
        <div class="soc-header">
            <h1>Batch Flow Analysis</h1>
            <p>Bulk traffic auditing, offline log analysis and performance verification (Max 2,000 flows)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
            st.warning("Batch truncated to first 2,000 rows for demonstration stability.")
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
            height=320,
        )
        apply_plot_theme(fig_batch, height=320)
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
    st.markdown(
        """
        <div class="soc-header">
            <h1>Model Performance & Evaluation</h1>
            <p>Empirical evidence from the CICIDS2017 held-out test set (565,566 flows) and 5-fold cross-validation</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Key Results Card
    st.markdown("### Key Verified Results")
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
        metric_palette = {"accuracy": "#2563eb", "macro_F1": "#0d9488", "Bot_recall": "#d97706"}
        for m in ["accuracy", "macro_F1", "Bot_recall"]:
            if m in comp_df.columns:
                fig_comp.add_trace(go.Bar(
                    name=m.replace("_", " ").title(),
                    x=comp_df["model"],
                    y=comp_df[m],
                    text=[f"{v:.4f}" for v in comp_df[m]],
                    textposition="auto",
                    marker_color=metric_palette.get(m, "#2563eb"),
                ))
        apply_plot_theme(fig_comp, height=360)
        fig_comp.update_layout(
            barmode="group",
            height=360,
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
            if row["Class"] in ["Bot", "Web Attack"]:
                return ["background-color: #fef2f2; color: #991b1b; font-weight: 500;"] * len(row)
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
    st.markdown(
        """
        <div class="soc-header">
            <h1>About the Project</h1>
            <p>Explainable AI-Based Network Intrusion Detection Framework Using Ensemble Machine Learning for Near Real-Time Threat Detection</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_team, col_guide = st.columns(2)
    with col_team:
        st.markdown(
            """
            <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:6px; padding:16px; box-shadow:0 1px 2px rgba(0,0,0,0.03);">
                <div style="font-size:11px; text-transform:uppercase; font-weight:700; letter-spacing:0.05em; color:#64748b; margin-bottom:8px;">Project Authors</div>
                <div style="font-size:14px; color:#1e293b; line-height:1.7;">
                    • <strong>Gurjot Singh</strong> (Roll No. B205)<br>
                    • <strong>Lehar Nimodiya</strong> (Roll No. B233)<br>
                    • <strong>Lokesh Ingale</strong> (Roll No. B250)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_guide:
        st.markdown(
            """
            <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:6px; padding:16px; box-shadow:0 1px 2px rgba(0,0,0,0.03);">
                <div style="font-size:11px; text-transform:uppercase; font-weight:700; letter-spacing:0.05em; color:#64748b; margin-bottom:8px;">Faculty Supervision</div>
                <div style="font-size:14px; color:#1e293b; line-height:1.7;">
                    • <strong>Dr. Naziya Hussain</strong><br>
                    <span style="color:#64748b; font-size:13px;">Department of Computer Engineering, MPSTME Shirpur<br>SVKM's Narsee Monjee Institute of Management Studies (NMIMS)</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
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
        st.markdown(
            """
            <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:6px; padding:16px; box-shadow:0 1px 2px rgba(0,0,0,0.03);">
                <div style="font-size:11px; text-transform:uppercase; font-weight:700; letter-spacing:0.05em; color:#64748b; margin-bottom:8px;">Dataset Benchmark Facts</div>
                <div style="font-size:13px; color:#1e293b; line-height:1.6;">
                    • <strong>Source:</strong> Canadian Institute for Cybersecurity (CICIDS2017 MachineLearningCVE)<br>
                    • <strong>Total Valid Flows:</strong> 2,827,829 flows (Heartbleed & Infiltration excluded due to &lt;40 instances)<br>
                    • <strong>Features:</strong> 64 scaled continuous & discrete flow features<br>
                    • <strong>Classes (11):</strong> BENIGN, Bot, DDoS, DoS GoldenEye, DoS Hulk, DoS Slowhttptest, DoS slowloris, FTP-Patator, PortScan, SSH-Patator, Web Attack<br>
                    • <strong>Split:</strong> Stratified 80% train (2,262,263) / 20% test (565,566)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_limits:
        st.markdown(
            """
            <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:6px; padding:16px; box-shadow:0 1px 2px rgba(0,0,0,0.03);">
                <div style="font-size:11px; text-transform:uppercase; font-weight:700; letter-spacing:0.05em; color:#64748b; margin-bottom:8px;">Recognized Scope & Limitations</div>
                <div style="font-size:13px; color:#1e293b; line-height:1.6;">
                    1. <strong>Benchmark Evaluation:</strong> Trained and validated on CICIDS2017; cross-dataset evaluation (e.g. on CSE-CIC-IDS2018 or UNSW-NB15) is planned for future work.<br>
                    2. <strong>Replay vs. Kernel Sniffing:</strong> Operates via serialized replay of real captured test flows rather than real-time kernel packet capture.<br>
                    3. <strong>Base Learner Explanation:</strong> TreeExplainer computes exact Shapley values on the XGBoost base learner rather than the Logistic Regression meta-learner directly.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
