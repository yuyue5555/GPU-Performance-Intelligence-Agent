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

# ─── Display theme (NVIDIA green on dark) ─────────────────────────────────────

PLOTLY_TITLE_FONT = dict(color="#f0f3f6", size=16, family="Space Grotesk")
PLOTLY_AXIS_TITLE_FONT = dict(color="#e8ecf2", size=13, family="Space Grotesk")

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#12121c",
    font=dict(color="#f0f3f6", family="Space Grotesk", size=13),
    title=dict(font=PLOTLY_TITLE_FONT, x=0.02, xanchor="left"),
    legend=dict(font=dict(color="#e0e6ef", size=12)),
    margin=dict(l=48, r=24, t=64, b=48),
    xaxis=dict(
        gridcolor="#2a2a3e",
        linecolor="#3a3a52",
        tickfont=dict(color="#d0d6e0", size=12),
        title_font=PLOTLY_AXIS_TITLE_FONT,
    ),
    yaxis=dict(
        gridcolor="#2a2a3e",
        linecolor="#3a3a52",
        tickfont=dict(color="#d0d6e0", size=12),
        title_font=PLOTLY_AXIS_TITLE_FONT,
    ),
    colorway=["#76b900", "#8ed100", "#00b4d8", "#a8d65a", "#5a8f00", "#4cc9f0"],
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --nv-green: #76b900;
        --nv-green-bright: #8ed100;
        --nv-green-dim: #5a8f00;
        --bg-base: #0a0a0f;
        --bg-surface: #12121c;
        --bg-elevated: #1a1a2e;
        --bg-sidebar: #0d0d16;
        --border-subtle: #2a2a3e;
        --border-accent: #3d5a14;
        --text-primary: #f0f3f6;
        --text-secondary: #c5cdd8;
        --text-muted: #9aa6b5;
        --danger: #ff5c5c;
        --radius: 10px;
        --space-xs: 0.25rem;
        --space-sm: 0.5rem;
        --space-md: 1rem;
        --space-lg: 1.5rem;
        --font-base: 15px;
        --font-sm: 0.8125rem;
        --font-md: 1rem;
        --font-lg: 1.25rem;
        --font-xl: 1.625rem;
        --header-offset: 2.75rem;
    }

    .stApp, .main {
        background: var(--bg-base);
        color: var(--text-primary);
        font-family: 'Space Grotesk', sans-serif;
        font-size: var(--font-base);
        line-height: 1.55;
    }

    /* Typography */
    .header-title {
        font-family: 'JetBrains Mono', monospace;
        color: var(--nv-green-bright);
        font-size: var(--font-xl);
        font-weight: 700;
        letter-spacing: -0.02em;
        line-height: 1.25;
        margin-bottom: var(--space-xs);
    }
    .sidebar-brand {
        font-family: 'JetBrains Mono', monospace;
        color: var(--nv-green-bright);
        font-size: 1.375rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        line-height: 1.35;
        margin-bottom: 0.35rem;
    }
    .page-subtitle {
        color: var(--text-secondary);
        font-size: var(--font-md);
        font-weight: 400;
        margin-top: 0;
        margin-bottom: var(--space-md);
        line-height: 1.5;
    }
    .section-label {
        color: var(--text-muted);
        font-size: var(--font-sm);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }
    .compare-label {
        text-align: center;
        padding-top: 1.75rem;
        color: var(--nv-green-bright);
        font-weight: 600;
        font-size: var(--font-sm);
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    /* Streamlit markdown & headings */
    [data-testid="stAppViewContainer"] .stMarkdown p,
    [data-testid="stAppViewContainer"] .stMarkdown li,
    [data-testid="stAppViewContainer"] .stMarkdown span {
        color: var(--text-primary);
        font-size: var(--font-md);
    }
    [data-testid="stAppViewContainer"] .stMarkdown strong {
        color: var(--text-primary);
    }
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] span {
        color: var(--text-primary) !important;
    }
    [data-testid="stAppViewContainer"] h1,
    [data-testid="stAppViewContainer"] h2,
    [data-testid="stAppViewContainer"] h3 {
        color: var(--text-primary);
        font-weight: 600;
    }
    [data-testid="stAppViewContainer"] h3 {
        font-size: 1.125rem;
        margin-top: var(--space-lg);
        margin-bottom: var(--space-sm);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--bg-sidebar);
    }
    div[data-testid="stSidebarContent"] {
        background: var(--bg-sidebar);
        border-right: 1px solid var(--border-subtle);
        padding-top: calc(var(--header-offset) + 1.25rem);
        padding-left: 0.75rem;
        padding-right: 0.75rem;
        padding-bottom: 1.25rem;
    }
    [data-testid="stSidebar"] .section-label {
        font-size: 0.875rem;
        letter-spacing: 0.07em;
        margin-top: 0.15rem;
    }
    [data-testid="stSidebar"] .page-subtitle {
        font-size: 1.0625rem !important;
        line-height: 1.55 !important;
        margin-bottom: 0.75rem !important;
    }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown label {
        color: var(--text-secondary) !important;
        font-size: 0.9375rem !important;
        line-height: 1.55;
    }
    [data-testid="stSidebar"] .stMarkdown strong {
        color: var(--text-primary) !important;
        font-size: 0.9375rem !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: var(--border-subtle);
        margin: 1.15rem 0;
    }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div {
        row-gap: 0.45rem;
        gap: 0.45rem;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] label,
    [data-testid="stSidebar"] [data-testid="stRadio"] label,
    [data-testid="stSidebar"] [data-testid="stMarkdown"] label {
        font-size: 0.9375rem !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        font-size: 0.875rem !important;
        padding: 0.55rem 0.95rem !important;
        min-height: 2.65rem !important;
    }

    /* Metrics — card-like modules with balanced scale */
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, var(--bg-elevated) 0%, var(--bg-surface) 100%);
        border: 1px solid var(--border-accent);
        border-radius: var(--radius);
        padding: var(--space-md) var(--space-lg);
        min-height: 5.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
    }
    [data-testid="stMetricLabel"] {
        color: var(--text-muted) !important;
        font-size: var(--font-sm) !important;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-size: 1.5rem !important;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1.2;
    }
    [data-testid="stMetricDelta"] {
        font-size: var(--font-sm) !important;
    }

    /* Dividers */
    [data-testid="stAppViewContainer"] hr {
        border-color: var(--border-subtle);
        margin: var(--space-lg) 0;
    }

    /* Form controls */
    [data-testid="stSelectbox"] label,
    [data-testid="stMultiSelect"] label,
    [data-testid="stRadio"] label,
    [data-testid="stTextInput"] label {
        color: var(--text-secondary) !important;
        font-size: var(--font-sm) !important;
        font-weight: 500;
    }
    [data-testid="stMultiSelect"] [data-baseweb="select"] {
        background: var(--bg-surface);
        border-color: var(--border-subtle);
    }
    [data-testid="stMultiSelect"] div[role="listbox"] span {
        color: var(--text-primary);
        font-size: var(--font-sm);
    }

    /* Single select: light surface + dark text (sidebar Filter by Tier, main Metric, etc.) */
    [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"],
    section.main [data-testid="stSelectbox"] [data-baseweb="select"],
    main[data-testid="stMain"] [data-testid="stSelectbox"] [data-baseweb="select"] {
        background-color: #f3f4f7 !important;
        border: 1px solid #b8c0cc !important;
        border-radius: 8px !important;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        color: #121418 !important;
        font-size: 0.875rem !important;
    }
    section.main [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    main[data-testid="stMain"] [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        color: #121418 !important;
        font-size: var(--font-sm) !important;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] svg,
    section.main [data-testid="stSelectbox"] [data-baseweb="select"] svg,
    main[data-testid="stMain"] [data-testid="stSelectbox"] [data-baseweb="select"] svg {
        fill: #1a1d21 !important;
    }

    /* Buttons */
    .stButton > button {
        background: var(--bg-elevated);
        color: var(--text-primary);
        border: 1px solid var(--border-accent);
        border-radius: 8px;
        font-size: var(--font-sm);
        font-weight: 500;
        padding: 0.5rem 0.85rem;
        line-height: 1.35;
        white-space: normal;
        height: auto;
        min-height: 2.5rem;
        transition: border-color 0.15s, background 0.15s;
    }
    .stButton > button:hover {
        border-color: var(--nv-green);
        background: #1e2a12;
        color: var(--text-primary);
    }
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {
        background: var(--nv-green-dim);
        border-color: var(--nv-green);
        color: #fff;
    }

    /* Radio navigation */
    [data-testid="stSidebar"] [data-testid="stRadio"] label p {
        font-size: 0.9375rem !important;
        line-height: 1.45 !important;
        color: var(--text-secondary) !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        padding: 0.35rem 0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"] p {
        color: var(--nv-green-bright) !important;
        font-weight: 600;
    }

    /* Dataframes */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius);
        overflow: hidden;
    }
    [data-testid="stDataFrame"] div[data-testid="stDataFrameResizable"] {
        font-size: var(--font-sm);
    }

    /* Alerts */
    [data-testid="stAlert"] {
        border-radius: var(--radius);
        font-size: var(--font-sm);
    }

    /* Chat — bubble layout */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 0.4rem 0 !important;
        margin-bottom: 0.65rem;
        gap: 0.65rem;
        align-items: flex-start;
    }
    [data-testid="stChatMessageAvatar"] {
        background: var(--bg-elevated) !important;
        border: 1px solid var(--border-accent);
        font-size: 1.1rem;
        min-width: 2.25rem;
        min-height: 2.25rem;
    }
    [data-testid="stChatMessageContent"] {
        border-radius: 14px;
        padding: 0.9rem 1.15rem !important;
        border: 2px solid var(--nv-green) !important;
        background: #fafbfc !important;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.14);
        flex: 1;
    }
    [data-testid="stChatMessageContent"]:has(.chat-sender-user) {
        background: #ffffff !important;
        border-color: #76b900 !important;
    }
    [data-testid="stChatMessageContent"]:has(.chat-sender-assistant) {
        background: #fafbfc !important;
        border-color: #5a8f00 !important;
    }
    .chat-sender {
        display: block;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        margin: 0 0 0.55rem 0;
        line-height: 1.2;
    }
    .chat-sender-user {
        color: #2563eb !important;
    }
    .chat-sender-assistant {
        color: var(--nv-green-dim) !important;
    }
    [data-testid="stChatMessage"] .stMarkdown,
    [data-testid="stChatMessage"] .stMarkdown p,
    [data-testid="stChatMessage"] .stMarkdown li,
    [data-testid="stChatMessage"] .stMarkdown ul,
    [data-testid="stChatMessage"] .stMarkdown ol,
    [data-testid="stChatMessage"] .stMarkdown span,
    [data-testid="stChatMessage"] .stMarkdown h1,
    [data-testid="stChatMessage"] .stMarkdown h2,
    [data-testid="stChatMessage"] .stMarkdown h3,
    [data-testid="stChatMessage"] .stMarkdown h4,
    [data-testid="stChatMessage"] .stMarkdown td,
    [data-testid="stChatMessage"] .stMarkdown th {
        color: #1a1d21 !important;
        font-size: var(--font-md);
        line-height: 1.65;
    }
    [data-testid="stChatMessage"] .stMarkdown strong,
    [data-testid="stChatMessage"] .stMarkdown b {
        color: #0d0f12 !important;
        font-weight: 600;
    }
    [data-testid="stChatMessage"] .stMarkdown em {
        color: #3d5a00 !important;
    }
    [data-testid="stChatMessage"] .stMarkdown a {
        color: #4a7a00 !important;
        text-decoration: underline;
        text-underline-offset: 2px;
    }
    [data-testid="stChatMessage"] .stMarkdown code {
        background: rgba(118, 185, 0, 0.18) !important;
        color: #2d4500 !important;
        padding: 0.12em 0.4em;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.88em;
    }
    [data-testid="stChatMessage"] .stMarkdown pre {
        background: #f0f2f5 !important;
        border: 1px solid #c8d4a8;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
    }
    [data-testid="stChatMessage"] .stMarkdown pre code {
        background: transparent !important;
        color: #1a1d21 !important;
    }
    [data-testid="stChatMessage"] .stMarkdown blockquote {
        border-left: 3px solid var(--nv-green);
        color: #2a2f36 !important;
        background: rgba(118, 185, 0, 0.12);
        padding: 0.45rem 0.9rem;
        margin: 0.45rem 0;
    }
    [data-testid="stChatMessage"] .stMarkdown hr {
        border-color: var(--border-subtle);
        margin: 0.75rem 0;
    }
    [data-testid="stChatMessage"] .stMarkdown table {
        border-color: var(--border-subtle);
    }

    /* Bottom dock — same background as main app */
    [data-testid="stBottomBlockContainer"] {
        background-color: var(--bg-base) !important;
        padding: 1rem 2rem 1.15rem !important;
        border-top: 1px solid var(--border-subtle);
        box-sizing: border-box;
    }

    /* Chat input — green border, solid white inside (AI Agent only; sole st.chat_input) */
    [data-testid="stChatInput"] {
        border-top: none;
        padding-top: 0;
    }
    [data-testid="stChatInput"] > div {
        background: #ffffff !important;
        border: 2px solid var(--nv-green) !important;
        border-radius: 12px;
    }
    [data-testid="stChatInput"] > div > div {
        background: #ffffff !important;
    }
    [data-testid="stChatInput"] textarea {
        background: #ffffff !important;
        color: #1a1d21 !important;
        font-size: var(--font-md);
        caret-color: var(--nv-green-bright);
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: var(--text-muted) !important;
        opacity: 1 !important;
        background: none !important;
        -webkit-text-fill-color: var(--text-muted);
    }

    /* Plotly chart container spacing */
    [data-testid="stPlotlyChart"] {
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius);
        padding: var(--space-sm);
        background: var(--bg-surface);
        margin-bottom: var(--space-md);
    }

    /* Hide Streamlit chrome — keep header visible so sidebar toggle remains usable */
    #MainMenu, footer {
        visibility: hidden;
    }
    header[data-testid="stHeader"] {
        background-color: var(--bg-base) !important;
        border-bottom: 1px solid var(--border-subtle);
        z-index: 1002;
    }
    /* Fixed header overlaps scroll area — reserve top space */
    section.main .block-container,
    [data-testid="stMain"] .block-container {
        padding-top: calc(var(--header-offset) + 1.25rem) !important;
    }
    .block-container {
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    .regression-badge {
        background: var(--danger);
        color: #fff;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: var(--font-sm);
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


def _apply_plotly_theme(fig, height=None, title=None, **extra):
    """Apply consistent dark NVIDIA theme to a Plotly figure."""
    layout = {**PLOTLY_LAYOUT, **extra}
    if height is not None:
        layout["height"] = height

    # Normalize title (Plotly ignores top-level title_font)
    title_text = title
    if title_text is None:
        existing = layout.get("title")
        if isinstance(existing, str):
            title_text = existing
            layout.pop("title", None)
        elif isinstance(existing, dict):
            title_text = existing.get("text")
    if title_text is None and fig.layout.title and fig.layout.title.text:
        title_text = fig.layout.title.text
    if title_text:
        layout["title"] = dict(
            text=title_text,
            font=PLOTLY_TITLE_FONT,
            x=0.02,
            xanchor="left",
        )

    fig.update_layout(**layout)
    # Re-apply axis title colors (px figures may reset them)
    fig.update_xaxes(title_font=PLOTLY_AXIS_TITLE_FONT, tickfont=dict(color="#d0d6e0", size=12))
    fig.update_yaxes(title_font=PLOTLY_AXIS_TITLE_FONT, tickfont=dict(color="#d0d6e0", size=12))
    return fig


CHAT_AVATARS = {"user": "👤", "assistant": "🤖"}
CHAT_LABELS = {"user": "You", "assistant": "PerfBot"}


def _render_chat_message(role: str, content: str) -> None:
    """Render a chat bubble with role-specific styling and readable markdown."""
    label = CHAT_LABELS.get(role, role.title())
    sender_cls = "chat-sender-user" if role == "user" else "chat-sender-assistant"
    with st.chat_message(role, avatar=CHAT_AVATARS.get(role, "💬")):
        st.markdown(
            f'<p class="chat-sender {sender_cls}">{label}</p>',
            unsafe_allow_html=True,
        )
        st.markdown(content)


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
    st.markdown(
        '<div class="sidebar-brand">⚡ GPU Performance Intelligence Agent</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<p class="page-subtitle" style="margin-bottom:0.5rem">GPU Intelligence Dashboard</p>', unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["📊 Overview", "📈 Time Series", "⚔️ GPU Compare", "🤖 AI Agent", "🔴 Regressions"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    tier_filter = st.selectbox("Filter by Tier", ["All", "consumer", "datacenter"])
    st.markdown("---")
    st.markdown('<div class="section-label">Data Pipeline</div>', unsafe_allow_html=True)
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
    st.markdown('<p class="page-subtitle">Real-time performance analytics across GPU models and workloads.</p>', unsafe_allow_html=True)
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
        fig.update_traces(textfont=dict(color="#f0f3f6", size=12))
        _apply_plotly_theme(fig, height=420, title="LLM Inference Performance (tokens/sec)")
        fig.update_layout(xaxis_title="tokens/sec")
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
            _apply_plotly_theme(fig2, height=380, title="Price vs Performance")
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
                _apply_plotly_theme(
                    fig3,
                    height=380,
                    title="Consumer GPU Capability Radar",
                    showlegend=True,
                    polar=dict(
                        bgcolor="#12121c",
                        radialaxis=dict(
                            visible=True,
                            gridcolor="#2a2a3e",
                            tickfont=dict(color="#c5cdd8", size=11),
                        ),
                        angularaxis=dict(tickfont=dict(color="#c5cdd8", size=12)),
                    ),
                )
                st.plotly_chart(fig3, use_container_width=True)


# ─── Page: Time Series ────────────────────────────────────────────────────────

elif page == "📈 Time Series":
    st.markdown('<div class="header-title">Performance Over Time</div>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Track benchmark trends and detect drift across driver versions.</p>', unsafe_allow_html=True)
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
                    mode="markers", marker=dict(symbol="x", size=12, color="#ff5c5c"),
                    name="⚠️ Regression",
                ))
            _apply_plotly_theme(
                fig,
                height=480,
                title=f"{metric_choice.replace('_', ' ').title()} Over Time",
            )
            st.plotly_chart(fig, use_container_width=True)


# ─── Page: GPU Compare ────────────────────────────────────────────────────────

elif page == "⚔️ GPU Compare":
    st.markdown('<div class="header-title">GPU Head-to-Head</div>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Compare any two GPUs across all benchmark dimensions.</p>', unsafe_allow_html=True)
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
                st.markdown(f'<div class="compare-label">{label}</div>', unsafe_allow_html=True)
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
    st.markdown('<p class="page-subtitle">Ask natural language questions about GPU benchmark data.</p>', unsafe_allow_html=True)
    st.markdown("---")

    # Example questions — two rows for readable button proportions
    st.markdown('<p class="section-label">Example questions</p>', unsafe_allow_html=True)
    examples = [
        "Which GPU has the best LLM inference performance under $2000?",
        "Compare RTX 4090 vs RX 7900 XTX for AI workloads",
        "Are there any performance regressions I should know about?",
        "Generate a full benchmark report focused on efficiency",
        "What's the best datacenter GPU for training transformers?",
    ]
    row1_cols = st.columns(3)
    row2_cols = st.columns(2)
    for col, ex in zip(row1_cols + row2_cols, examples):
        with col:
            if st.button(ex, use_container_width=True, key=ex):
                st.session_state.setdefault("chat_history", [])
                st.session_state["pending_query"] = ex

    st.markdown("---")

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        _render_chat_message(msg["role"], msg["content"])

    # Input
    pending = st.session_state.pop("pending_query", None)
    user_input = st.chat_input("Ask PerfBot anything about GPU performance...") or pending

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        _render_chat_message("user", user_input)

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
                try:
                    from agent.graph import run_agent
                    prior = st.session_state.chat_history[:-1]
                    reply = run_agent(user_input, history=prior)
                except Exception as e2:
                    reply = (
                        f"⚠️ Agent unavailable: {e2}\n\n"
                        "Make sure `ANTHROPIC_API_KEY` is set and the ETL pipeline has been run."
                    )
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        _render_chat_message("assistant", reply)

    if st.sidebar.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()


# ─── Page: Regressions ───────────────────────────────────────────────────────

elif page == "🔴 Regressions":
    st.markdown('<div class="header-title">Regression Monitor</div>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Automated detection of performance drops vs rolling baseline.</p>', unsafe_allow_html=True)
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
            _apply_plotly_theme(fig, height=380, title="Regression Count by GPU")
            st.plotly_chart(fig, use_container_width=True)
