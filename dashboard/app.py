"""
dashboard/app.py

Streamlit interactive dashboard for GPU benchmark insights.
Run with: streamlit run dashboard/app.py
"""
import json
import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─── Config ────────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data"
SUMMARY_PATH = DATA_DIR / "summary.json"
DB_PATH = DATA_DIR / "benchmarks.db"

st.set_page_config(
    page_title="GPU PerfLab Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Space+Grotesk:wght@300;400;600;700&display=swap');
    
    .main { background: #0a0a0f; }
    .stApp { background: #0a0a0f; color: #e0e0e0; font-family: 'Space Grotesk', sans-serif; }
    
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #76b900;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
    }
    .gpu-title {
        font-family: 'JetBrains Mono', monospace;
        color: #76b900;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
    }
    .metric-label {
        color: #888;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .header-title {
        font-family: 'JetBrains Mono', monospace;
        color: #76b900;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .regression-badge {
        background: #ff4444;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 700;
    }
    div[data-testid="stSidebarContent"] {
        background: #0d0d1a;
        border-right: 1px solid #1a1a3e;
    }
</style>
""", unsafe_allow_html=True)


# ─── Data Loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_summary() -> pd.DataFrame:
    if not SUMMARY_PATH.exists():
        return pd.DataFrame()
    data = json.loads(SUMMARY_PATH.read_text())
    return pd.DataFrame(data)


@st.cache_data(ttl=60)
def load_time_series() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM benchmarks ORDER BY run_date", conn)
    conn.close()
    df["run_date"] = pd.to_datetime(df["run_date"])
    return df


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="header-title">⚡ PerfLab</div>', unsafe_allow_html=True)
    st.markdown("**GPU Intelligence Dashboard**")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["📊 Overview", "📈 Time Series", "⚔️ GPU Compare", "🤖 AI Agent", "🔴 Regressions"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    tier_filter = st.selectbox("Filter by Tier", ["All", "consumer", "datacenter"])
    st.markdown("---")
    st.markdown('<div class="metric-label">Data Pipeline</div>', unsafe_allow_html=True)
    if st.button("🔄 Re-run ETL Pipeline"):
        with st.spinner("Running ETL..."):
            import subprocess
            subprocess.run(["python", "-m", "pipeline.etl"], cwd=str(DATA_DIR.parent))
        st.cache_data.clear()
        st.success("Pipeline complete!")


# ─── Load data ────────────────────────────────────────────────────────────────

summary_df = load_summary()
ts_df = load_time_series()

if summary_df.empty:
    st.error("⚠️ No data found. Run `python -m pipeline.etl` to seed the database.")
    st.stop()

if tier_filter != "All":
    summary_df = summary_df[summary_df["tier"] == tier_filter]


# ─── Page: Overview ───────────────────────────────────────────────────────────

if page == "📊 Overview":
    st.markdown('<div class="header-title">GPU Benchmark Overview</div>', unsafe_allow_html=True)
    st.markdown("Real-time performance analytics across GPU models and workloads.")
    st.markdown("---")

    # Top KPI row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("GPUs Tracked", len(summary_df))
    with col2:
        best_llm = summary_df.loc[summary_df["llm_inference_tokens_per_sec_avg"].idxmax(), "gpu_name"] \
            if "llm_inference_tokens_per_sec_avg" in summary_df.columns else "N/A"
        st.metric("Best LLM GPU", best_llm.split()[-1] if best_llm != "N/A" else "N/A")
    with col3:
        regressions = summary_df["regression_count"].sum() if "regression_count" in summary_df.columns else 0
        st.metric("Regressions Detected", int(regressions), delta=None)
    with col4:
        total_runs = len(ts_df) if not ts_df.empty else 0
        st.metric("Total Benchmark Runs", total_runs)

    st.markdown("---")

    # LLM Performance bar chart
    if "llm_inference_tokens_per_sec_avg" in summary_df.columns:
        llm_df = summary_df.dropna(subset=["llm_inference_tokens_per_sec_avg"]).sort_values(
            "llm_inference_tokens_per_sec_avg", ascending=True
        )
        fig = go.Figure(go.Bar(
            x=llm_df["llm_inference_tokens_per_sec_avg"],
            y=llm_df["gpu_name"],
            orientation="h",
            marker=dict(
                color=llm_df["llm_inference_tokens_per_sec_avg"],
                colorscale=[[0, "#1a3a00"], [0.5, "#4a7a00"], [1, "#76b900"]],
                showscale=False,
            ),
            text=llm_df["llm_inference_tokens_per_sec_avg"].apply(lambda x: f"{x:,.0f} tok/s"),
            textposition="outside",
        ))
        fig.update_layout(
            title="LLM Inference Performance (tokens/sec)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0e0", family="Space Grotesk"),
            xaxis=dict(gridcolor="#1a1a3e", title="tokens/sec"),
            yaxis=dict(gridcolor="#1a1a3e"),
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    col_left, col_right = st.columns(2)

    with col_left:
        # Scatter: perf vs price
        if all(c in summary_df.columns for c in ["msrp_usd", "llm_inference_tokens_per_sec_avg"]):
            scatter_df = summary_df.dropna(subset=["msrp_usd", "llm_inference_tokens_per_sec_avg"])
            fig2 = px.scatter(
                scatter_df,
                x="msrp_usd", y="llm_inference_tokens_per_sec_avg",
                color="tier",
                size="vram_gb",
                hover_name="gpu_name",
                color_discrete_map={"consumer": "#76b900", "datacenter": "#00b4d8"},
                labels={"msrp_usd": "Price (USD)", "llm_inference_tokens_per_sec_avg": "LLM Perf (tok/s)"},
                title="Price vs Performance",
                size_max=30,
            )
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,10,20,0.8)",
                font=dict(color="#e0e0e0", family="Space Grotesk"),
                xaxis=dict(gridcolor="#1a1a3e"),
                yaxis=dict(gridcolor="#1a1a3e"),
            )
            st.plotly_chart(fig2, use_container_width=True)

    with col_right:
        # Radar chart — top consumer GPUs
        consumer = summary_df[summary_df["tier"] == "consumer"].head(4)
        if not consumer.empty:
            metrics_radar = [
                "llm_inference_tokens_per_sec_avg",
                "cnn_training_images_per_sec_avg",
                "memory_bandwidth_gbps_avg",
            ]
            available = [m for m in metrics_radar if m in consumer.columns]
            if available:
                fig3 = go.Figure()
                for _, row in consumer.iterrows():
                    vals = [row.get(m, 0) or 0 for m in available]
                    # Normalize to 0-1
                    maxes = [consumer[m].max() for m in available]
                    norm = [v/mx if mx else 0 for v, mx in zip(vals, maxes)]
                    norm.append(norm[0])
                    labels = ["LLM", "CNN", "Memory Bandwidth"][:len(available)] + ["LLM"]
                    fig3.add_trace(go.Scatterpolar(
                        r=norm, theta=labels, fill="toself",
                        name=row["gpu_name"].replace("NVIDIA ", "").replace("AMD ", ""),
                        opacity=0.7,
                    ))
                fig3.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, gridcolor="#1a1a3e"),
                        bgcolor="rgba(10,10,20,0.8)",
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e0e0e0"),
                    title="Consumer GPU Capability Radar",
                    showlegend=True,
                )
                st.plotly_chart(fig3, use_container_width=True)


# ─── Page: Time Series ────────────────────────────────────────────────────────

elif page == "📈 Time Series":
    st.markdown('<div class="header-title">Performance Over Time</div>', unsafe_allow_html=True)
    st.markdown("Track benchmark trends and detect drift across driver versions.")
    st.markdown("---")

    if ts_df.empty:
        st.warning("No time series data available.")
    else:
        metric_choice = st.selectbox("Metric", [
            "llm_inference_tokens_per_sec",
            "cnn_training_images_per_sec",
            "memory_bandwidth_gbps",
            "render_fps_4k",
        ])
        gpu_options = ts_df["gpu_name"].unique().tolist()
        selected_gpus = st.multiselect("GPUs", gpu_options, default=gpu_options[:4])

        if selected_gpus:
            filtered = ts_df[ts_df["gpu_name"].isin(selected_gpus)].dropna(subset=[metric_choice])
            fig = px.line(
                filtered, x="run_date", y=metric_choice,
                color="gpu_name",
                markers=True,
                title=f"{metric_choice.replace('_', ' ').title()} Over Time",
            )
            # Highlight regressions
            regressions = filtered[filtered["regression_flag"] == 1]
            if not regressions.empty:
                fig.add_trace(go.Scatter(
                    x=regressions["run_date"], y=regressions[metric_choice],
                    mode="markers", marker=dict(symbol="x", size=12, color="red"),
                    name="⚠️ Regression",
                ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,10,20,0.8)",
                font=dict(color="#e0e0e0", family="Space Grotesk"),
                xaxis=dict(gridcolor="#1a1a3e"),
                yaxis=dict(gridcolor="#1a1a3e"),
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)


# ─── Page: GPU Compare ────────────────────────────────────────────────────────

elif page == "⚔️ GPU Compare":
    st.markdown('<div class="header-title">GPU Head-to-Head</div>', unsafe_allow_html=True)
    st.markdown("Compare any two GPUs across all benchmark dimensions.")
    st.markdown("---")

    gpu_list = summary_df["gpu_name"].tolist()
    col_a, col_b = st.columns(2)
    with col_a:
        gpu_a = st.selectbox("GPU A", gpu_list, index=0)
    with col_b:
        gpu_b = st.selectbox("GPU B", gpu_list, index=min(2, len(gpu_list)-1))

    if gpu_a and gpu_b and gpu_a != gpu_b:
        row_a = summary_df[summary_df["gpu_name"] == gpu_a].iloc[0]
        row_b = summary_df[summary_df["gpu_name"] == gpu_b].iloc[0]

        compare_metrics = [
            ("llm_inference_tokens_per_sec_avg", "LLM Inference", "tok/s"),
            ("cnn_training_images_per_sec_avg", "CNN Training", "img/s"),
            ("memory_bandwidth_gbps_avg", "Memory Bandwidth", "GB/s"),
            ("llm_perf_per_watt_avg", "Perf / Watt", "tok/s/W"),
            ("llm_perf_per_dollar_avg", "Price-Performance", "tok/s per $1k"),
        ]

        st.markdown(f"### {gpu_a}  vs  {gpu_b}")
        for metric, label, unit in compare_metrics:
            if metric not in summary_df.columns:
                continue
            val_a = row_a.get(metric)
            val_b = row_b.get(metric)
            if val_a is None and val_b is None:
                continue
            c1, c2, c3 = st.columns([2, 1, 2])
            with c1:
                winner_a = "🏆 " if (val_a or 0) > (val_b or 0) else ""
                st.metric(
                    f"{winner_a}{gpu_a.replace('NVIDIA ', '').replace('AMD ', '')}",
                    f"{val_a:,.1f} {unit}" if val_a else "N/A"
                )
            with c2:
                st.markdown(f"<div style='text-align:center; padding-top:30px; color:#76b900; font-weight:700'>{label}</div>", unsafe_allow_html=True)
            with c3:
                winner_b = "🏆 " if (val_b or 0) > (val_a or 0) else ""
                st.metric(
                    f"{winner_b}{gpu_b.replace('NVIDIA ', '').replace('AMD ', '')}",
                    f"{val_b:,.1f} {unit}" if val_b else "N/A"
                )

        # Specs comparison
        st.markdown("---")
        st.markdown("### Specifications")
        specs_df = pd.DataFrame([
            {"Spec": "VRAM (GB)", gpu_a: str(row_a.get("vram_gb", "N/A")), gpu_b: str(row_b.get("vram_gb", "N/A"))},
            {"Spec": "TDP (W)", gpu_a: str(row_a.get("tdp_w", "N/A")), gpu_b: str(row_b.get("tdp_w", "N/A"))},
            {"Spec": "MSRP (USD)", gpu_a: f"${row_a.get('msrp_usd', 0):,}", gpu_b: f"${row_b.get('msrp_usd', 0):,}"},
            {"Spec": "Tier", gpu_a: str(row_a.get("tier", "N/A")), gpu_b: str(row_b.get("tier", "N/A"))},
        ])
        st.dataframe(specs_df, use_container_width=True, hide_index=True)


# ─── Page: AI Agent ───────────────────────────────────────────────────────────

elif page == "🤖 AI Agent":
    st.markdown('<div class="header-title">PerfBot AI Agent</div>', unsafe_allow_html=True)
    st.markdown("Ask natural language questions about GPU benchmark data.")
    st.markdown("---")

    # Example questions
    st.markdown("**Example questions:**")
    examples = [
        "Which GPU has the best LLM inference performance under $2000?",
        "Compare RTX 4090 vs RX 7900 XTX for AI workloads",
        "Are there any performance regressions I should know about?",
        "Generate a full benchmark report focused on efficiency",
        "What's the best datacenter GPU for training transformers?",
    ]
    cols = st.columns(len(examples))
    for col, ex in zip(cols, examples):
        with col:
            if st.button(ex, use_container_width=True, key=ex):
                st.session_state.setdefault("chat_history", [])
                st.session_state["pending_query"] = ex

    st.markdown("---")

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])

    # Input
    pending = st.session_state.pop("pending_query", None)
    user_input = st.chat_input("Ask PerfBot anything about GPU performance...") or pending

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("PerfBot is analyzing..."):
                try:
                    import httpx
                    resp = httpx.post(
                        "http://localhost:8000/agent/chat",
                        json={"message": user_input, "history": st.session_state.chat_history[:-1]},
                        timeout=60,
                    )
                    if resp.status_code == 200:
                        reply = resp.json()["response"]
                    else:
                        reply = f"API error: {resp.status_code}"
                except Exception:
                    # Fallback: call agent directly
                    try:
                        from agent.graph import run_agent
                        prior = st.session_state.chat_history[:-1]
                        reply = run_agent(user_input, history=prior)
                    except Exception as e2:
                        reply = f"⚠️ Agent unavailable: {e2}\n\nMake sure `ANTHROPIC_API_KEY` is set and the ETL pipeline has been run."
                st.markdown(reply)
                st.session_state.chat_history.append({"role": "assistant", "content": reply})

    if st.sidebar.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()


# ─── Page: Regressions ───────────────────────────────────────────────────────

elif page == "🔴 Regressions":
    st.markdown('<div class="header-title">Regression Monitor</div>', unsafe_allow_html=True)
    st.markdown("Automated detection of performance drops vs rolling baseline.")
    st.markdown("---")

    if ts_df.empty:
        st.warning("No data available.")
    else:
        regressions = ts_df[ts_df["regression_flag"] == 1].sort_values("run_date", ascending=False)
        st.metric("Total Regression Events", len(regressions))

        if regressions.empty:
            st.success("✅ No regressions detected across all GPUs.")
        else:
            st.markdown(f"**{len(regressions)} regression events** detected across benchmark history.")
            display_cols = ["gpu_name", "run_date", "driver_version",
                            "llm_inference_tokens_per_sec", "memory_bandwidth_gbps"]
            display_cols = [c for c in display_cols if c in regressions.columns]
            st.dataframe(
                regressions[display_cols].head(50),
                use_container_width=True,
                hide_index=True,
            )

            # Regressions by GPU
            reg_by_gpu = regressions["gpu_name"].value_counts().reset_index()
            reg_by_gpu.columns = ["gpu_name", "count"]
            fig = px.bar(reg_by_gpu, x="gpu_name", y="count",
                         title="Regression Count by GPU",
                         color="count",
                         color_continuous_scale=["#76b900", "#ff4444"])
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,10,20,0.8)",
                font=dict(color="#e0e0e0"), xaxis=dict(gridcolor="#1a1a3e"),
                yaxis=dict(gridcolor="#1a1a3e"),
            )
            st.plotly_chart(fig, use_container_width=True)
