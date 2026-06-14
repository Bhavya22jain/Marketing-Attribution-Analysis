"""
Marketing Attribution Analytics Dashboard
Run: streamlit run streamlit_dashboard_v2.py
Requirements: streamlit>=1.28, pandas, numpy, plotly, scikit-learn, reportlab
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import LabelEncoder
import io
import os

# ── ReportLab PDF ──────────────────────────────────────────────────────────────
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer
from reportlab.lib.units import cm

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MKTG·ATTR — Attribution Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── Hide Streamlit chrome ── */
#MainMenu, header, footer, .stToolbar, .stDecoration { display: none !important; }
.stAppDeployButton { display: none !important; }

/* ── Root vars ── */
:root {
  --bg-content: #fafafa;
  --bg-card: #ffffff;
  --text-primary: #111111;
  --text-secondary: #6b6b6b;
  --border: rgba(17,17,17,0.10);
  --lime: #c8f135;
  --danger: #dc2626;
  --success: #16a34a;
  --warning: #ca8a04;
  --sidebar-bg: #0f0f0f;
  --sidebar-text: #f0f0f0;
  --c1: #2563eb;
  --c2: #16a34a;
  --c3: #9333ea;
  --c4: #ea580c;
  --c5: #ca8a04;
  --c6: #0891b2;
}

/* ── App background ── */
.stApp { background: var(--bg-content); }
.main .block-container { padding: 0 2rem 2rem 2rem; max-width: 100%; }

/* ── Sidebar dark theme ── */
section[data-testid="stSidebar"] {
  background: var(--sidebar-bg) !important;
  width: 260px !important;
  min-width: 260px !important;
}
section[data-testid="stSidebar"] > div:first-child {
  background: var(--sidebar-bg) !important;
  padding: 1.5rem 1rem;
}
section[data-testid="stSidebar"] * { color: var(--sidebar-text) !important; }
section[data-testid="stSidebar"] .stSelectbox > div,
section[data-testid="stSidebar"] .stMultiSelect > div,
section[data-testid="stSidebar"] .stSlider > div { color: var(--sidebar-text) !important; }

/* Sidebar inputs dark */
section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] > div,
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {
  background: #1a1a1a !important;
  border-color: #333 !important;
  color: var(--sidebar-text) !important;
}
section[data-testid="stSidebar"] .stDateInput input {
  background: #1a1a1a !important;
  border-color: #333 !important;
  color: var(--sidebar-text) !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
  background: #1a1a1a !important;
  border: 1px dashed #444 !important;
  border-radius: 4px;
}
section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] {
  background: #333 !important;
}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] hr {
  border-color: #333 !important;
}

/* ── Radio as nav ── */
section[data-testid="stSidebar"] .stRadio label {
  display: block;
  padding: 0.55rem 0.75rem;
  margin: 2px 0;
  border-radius: 4px;
  cursor: pointer;
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 500;
  font-size: 0.85rem;
  transition: background 0.15s;
}
section[data-testid="stSidebar"] .stRadio label:hover { background: #1a1a1a; }
section[data-testid="stSidebar"] .stRadio [aria-checked="true"] + label,
section[data-testid="stSidebar"] .stRadio input:checked ~ label {
  background: var(--lime) !important;
  color: #0f0f0f !important;
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] { gap: 0; }

/* ── Card ── */
.mk-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 0;
  padding: 1.25rem 1.5rem;
  margin-bottom: 1rem;
}
.mk-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.08); transition: box-shadow 0.2s; }

/* ── KPI chip (header) ── */
.kpi-chip {
  background: var(--bg-card);
  border: 1px solid var(--border);
  padding: 0.5rem 1rem;
  text-align: center;
  min-width: 110px;
}
.kpi-chip .chip-label {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 0.65rem;
  font-weight: 500;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.08em;
}
.kpi-chip .chip-value {
  font-family: 'Space Mono', monospace;
  font-size: 1rem;
  font-weight: 700;
  color: var(--text-primary);
  margin-top: 2px;
}

/* ── Page header ── */
.page-header {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  padding: 1.5rem 0 1rem 0;
  border-bottom: 1px solid var(--border);
  margin-bottom: 1.5rem;
  flex-wrap: wrap;
  gap: 1rem;
}
.page-title {
  font-family: 'Space Mono', monospace;
  font-size: 2.5rem;
  font-weight: 700;
  color: var(--text-primary);
  letter-spacing: -0.02em;
  line-height: 1;
}
.chip-row { display: flex; gap: 0.5rem; flex-wrap: wrap; }

/* ── Section label ── */
.section-label {
  font-family: 'Space Mono', monospace;
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.15em;
  color: var(--text-secondary);
  text-transform: uppercase;
  margin-bottom: 0.75rem;
  margin-top: 1.5rem;
}

/* ── Metric cards ── */
.mk-metric {
  background: var(--bg-card);
  border: 1px solid var(--border);
  padding: 1.25rem 1.25rem 1rem 1.25rem;
}
.mk-metric .metric-num {
  font-family: 'Space Mono', monospace;
  font-size: 1.75rem;
  font-weight: 700;
  color: var(--text-primary);
}
.mk-metric .metric-label {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 0.8rem;
  color: var(--text-secondary);
  font-weight: 500;
  margin-top: 4px;
}
.mk-metric .delta-badge {
  display: inline-block;
  font-family: 'Space Mono', monospace;
  font-size: 0.7rem;
  padding: 2px 6px;
  margin-top: 6px;
  font-weight: 700;
}
.delta-up { background: #dcfce7; color: var(--success); }
.delta-down { background: #fee2e2; color: var(--danger); }
.delta-neutral { background: #f3f4f6; color: var(--text-secondary); }

/* ── Attribution buttons ── */
.attr-btn-row { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 1.5rem; }

/* ── Funnel bar ── */
.funnel-stage {
  margin-bottom: 0.6rem;
}
.funnel-label {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 0.8rem;
  font-weight: 500;
  color: var(--text-secondary);
  margin-bottom: 4px;
}
.funnel-bar-outer {
  background: #f0f0f0;
  height: 36px;
  width: 100%;
  position: relative;
}
.funnel-bar-inner {
  height: 100%;
  display: flex;
  align-items: center;
  padding-left: 10px;
  font-family: 'Space Mono', monospace;
  font-size: 0.8rem;
  font-weight: 700;
  color: white;
}
.funnel-dropoff {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 0.7rem;
  color: var(--danger);
  margin-top: 2px;
}

/* ── Path chips ── */
.path-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.6rem 0.75rem;
  margin-bottom: 4px;
}
.path-row:nth-child(even) { background: #f5f5f5; }
.path-chips { display: flex; align-items: center; gap: 0.3rem; flex-wrap: wrap; }
.ch-chip {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 0.72rem;
  font-weight: 600;
  padding: 3px 8px;
  border-radius: 2px;
  color: white;
}
.path-count {
  font-family: 'Space Mono', monospace;
  font-size: 0.78rem;
  font-weight: 700;
  color: var(--text-secondary);
  white-space: nowrap;
}

/* ── Insight items ── */
.insight-item {
  display: flex;
  gap: 1rem;
  padding: 1rem 0;
  border-bottom: 1px solid var(--border);
  align-items: flex-start;
}
.insight-num {
  font-family: 'Space Mono', monospace;
  font-size: 1.1rem;
  font-weight: 700;
  color: var(--lime);
  background: var(--text-primary);
  padding: 4px 10px;
  min-width: 42px;
  text-align: center;
}
.insight-text {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 0.9rem;
  color: var(--text-primary);
  line-height: 1.5;
}
.insight-bold { font-weight: 700; }

/* ── Budget cards ── */
.budget-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  padding: 1rem 1.25rem;
}
.budget-badge {
  display: inline-block;
  font-family: 'Space Mono', monospace;
  font-size: 0.65rem;
  font-weight: 700;
  padding: 3px 8px;
  letter-spacing: 0.1em;
  margin-bottom: 6px;
}
.badge-scale { background: #dcfce7; color: #15803d; }
.badge-maintain { background: #dbeafe; color: #1d4ed8; }
.badge-audit { background: #fee2e2; color: #b91c1c; }

/* ── Attribution model guide cards ── */
.model-guide-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  padding: 1rem 1.25rem;
}
.model-guide-card.active-model {
  background: var(--text-primary);
  border-color: var(--text-primary);
}
.model-guide-card.active-model * { color: var(--sidebar-text) !important; }
.model-name {
  font-family: 'Space Mono', monospace;
  font-size: 0.85rem;
  font-weight: 700;
  margin-bottom: 4px;
}
.model-use {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 0.78rem;
  color: var(--text-secondary);
}

/* ── Sidebar logo ── */
.sidebar-logo {
  font-family: 'Space Mono', monospace;
  font-size: 1.4rem;
  font-weight: 700;
  color: #c8f135 !important;
  letter-spacing: 0.05em;
  margin-bottom: 2px;
}
.sidebar-tagline {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 0.72rem;
  color: #888 !important;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  margin-bottom: 1rem;
}
.sidebar-section-label {
  font-family: 'Space Mono', monospace;
  font-size: 0.65rem;
  font-weight: 700;
  letter-spacing: 0.18em;
  color: #666 !important;
  text-transform: uppercase;
  margin: 1rem 0 0.4rem 0;
}
.sidebar-stats {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 0.72rem;
  color: #666 !important;
  line-height: 1.7;
}

/* ── Prediction output ── */
.pred-result-high {
  font-family: 'Space Mono', monospace;
  font-size: 1.6rem;
  font-weight: 700;
  color: #c8f135;
  background: #0f0f0f;
  padding: 1rem 1.5rem;
  display: inline-block;
  margin-top: 0.5rem;
}
.pred-result-std {
  font-family: 'Space Mono', monospace;
  font-size: 1.6rem;
  font-weight: 700;
  color: #6b6b6b;
  background: #f3f4f6;
  padding: 1rem 1.5rem;
  display: inline-block;
  margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
CHART_COLORS = ["#2563eb", "#16a34a", "#9333ea", "#ea580c", "#ca8a04", "#0891b2"]
CHANNEL_COLORS = {
    "Search": "#2563eb",
    "Email": "#16a34a",
    "Influencer": "#9333ea",
    "Social": "#ea580c",
    "Display": "#ca8a04",
}
FONT_MONO = "Space Mono"
FONT_BODY = "Space Grotesk"

# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY THEME HELPER
# ══════════════════════════════════════════════════════════════════════════════
def plotly_mono_theme():
    return dict(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family=FONT_BODY, color="#111111"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="left",
            x=0,
            font=dict(family=FONT_BODY, size=12),
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        hoverlabel=dict(font_family=FONT_BODY),
    )


def apply_theme(fig, title="", xaxis=None, yaxis=None, **extra):
    theme = plotly_mono_theme()
    layout_kwargs = {**theme, **extra}
    if title:
        layout_kwargs["title"] = dict(text=title, font=dict(family=FONT_MONO, size=14, color="#111111"))
    if xaxis:
        layout_kwargs["xaxis"] = xaxis
    if yaxis:
        layout_kwargs["yaxis"] = yaxis
    fig.update_layout(**layout_kwargs)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
SAMPLE_CSV = "marketing_campaign_performance_10000.csv"


@st.cache_data
def generate_synthetic_data(n=10000):
    np.random.seed(42)
    channels = ["Search", "Email", "Influencer", "Social", "Display"]
    start_dates = pd.date_range("2022-01-01", "2024-12-01", periods=n)
    durations = np.random.randint(7, 90, n)
    end_dates = start_dates + pd.to_timedelta(durations, unit="D")
    ch = np.random.choice(channels, n)
    impressions = np.random.randint(1000, 500000, n)
    ctr_base = {"Search": 0.05, "Email": 0.03, "Influencer": 0.02, "Social": 0.025, "Display": 0.01}
    clicks = (impressions * np.array([ctr_base[c] for c in ch]) * np.random.uniform(0.7, 1.3, n)).astype(int).clip(1)
    cvr_base = {"Search": 0.08, "Email": 0.12, "Influencer": 0.05, "Social": 0.06, "Display": 0.03}
    leads = (clicks * np.array([cvr_base[c] for c in ch]) * np.random.uniform(0.7, 1.3, n)).astype(int).clip(0)
    conversions = (leads * np.random.uniform(0.3, 0.7, n)).astype(int).clip(0)
    cost = np.random.uniform(500, 50000, n)
    revenue = cost * np.random.uniform(0.5, 6, n)
    roi = (revenue - cost) / cost * 100
    df = pd.DataFrame({
        "CampaignID": [f"C{str(i).zfill(5)}" for i in range(1, n + 1)],
        "StartDate": start_dates,
        "EndDate": end_dates,
        "Channel": ch,
        "Impressions": impressions,
        "Clicks": clicks,
        "Leads": leads,
        "Conversions": conversions,
        "Cost_USD": cost.round(2),
        "Revenue_USD": revenue.round(2),
        "ROI": roi.round(2),
    })
    return df


@st.cache_data
def load_and_preprocess(file_obj=None):
    if file_obj is not None:
        df = pd.read_csv(file_obj)
    elif os.path.exists(SAMPLE_CSV):
        df = pd.read_csv(SAMPLE_CSV)
    else:
        df = generate_synthetic_data()

    # Normalize channel casing
    if "Channel" in df.columns:
        df["Channel"] = df["Channel"].str.title()

    # Parse dates
    for col in ["StartDate", "EndDate"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Drop exact duplicates
    df = df.drop_duplicates()

    # Fill numeric nulls with median
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # Engineer features
    df["CTR"] = (df["Clicks"] / df["Impressions"].replace(0, np.nan) * 100).round(4)
    df["CVR"] = (df["Conversions"] / df["Clicks"].replace(0, np.nan) * 100).round(4)
    df["CPC"] = (df["Cost_USD"] / df["Clicks"].replace(0, np.nan)).round(4)
    df["CPL"] = (df["Cost_USD"] / df["Leads"].replace(0, np.nan)).round(4)
    df["CPA"] = (df["Cost_USD"] / df["Conversions"].replace(0, np.nan)).round(4)
    df["ROAS"] = (df["Revenue_USD"] / df["Cost_USD"].replace(0, np.nan)).round(4)
    df["Profit"] = (df["Revenue_USD"] - df["Cost_USD"]).round(2)
    df["Duration_Days"] = (df["EndDate"] - df["StartDate"]).dt.days.clip(lower=0)
    df["Month"] = df["StartDate"].dt.to_period("M").astype(str)

    # Fill NaN from engineering
    for col in ["CTR", "CVR", "CPC", "CPL", "CPA", "ROAS"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    df = df.sort_values("StartDate").reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# JOURNEY SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def simulate_journeys(n_journeys, channels):
    np.random.seed(7)
    journeys = []
    for _ in range(n_journeys):
        length = np.random.choice([1, 2, 3, 4, 5, 6], p=[0.25, 0.25, 0.2, 0.15, 0.1, 0.05])
        path = list(np.random.choice(channels, length))
        converted = np.random.random() < 0.35
        journeys.append({"path": path, "converted": converted})
    return journeys


# ══════════════════════════════════════════════════════════════════════════════
# ATTRIBUTION MODELS
# ══════════════════════════════════════════════════════════════════════════════
def compute_attribution(journeys, model, channels):
    credit = {c: 0.0 for c in channels}
    converted = [j for j in journeys if j["converted"]]
    for j in converted:
        path = j["path"]
        n = len(path)
        if model == "First Touch":
            credit[path[0]] += 1.0
        elif model == "Last Touch":
            credit[path[-1]] += 1.0
        elif model == "Linear":
            for c in path:
                credit[c] += 1.0 / n
        elif model == "Time Decay":
            weights = np.array([2 ** i for i in range(n)], dtype=float)
            weights /= weights.sum()
            for c, w in zip(path, weights):
                credit[c] += w
        elif model == "Position Based":
            if n == 1:
                credit[path[0]] += 1.0
            elif n == 2:
                credit[path[0]] += 0.4
                credit[path[-1]] += 0.4
            else:
                credit[path[0]] += 0.4
                credit[path[-1]] += 0.4
                mid_w = 0.2 / (n - 2)
                for c in path[1:-1]:
                    credit[c] += mid_w
    total = sum(credit.values()) or 1
    return {c: v / total * 100 for c, v in credit.items()}


def fmt_num(val, prefix="", suffix="", decimals=0):
    if abs(val) >= 1_000_000:
        return f"{prefix}{val/1_000_000:.1f}M{suffix}"
    elif abs(val) >= 1_000:
        return f"{prefix}{val/1_000:.1f}K{suffix}"
    return f"{prefix}{val:,.{decimals}f}{suffix}"


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sidebar-logo">MKTG·ATTR</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">Attribution Analytics Platform</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="sidebar-section-label">Dataset</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"], label_visibility="collapsed")

    raw_df = load_and_preprocess(uploaded_file)

    st.markdown('<div class="sidebar-section-label">Filters</div>', unsafe_allow_html=True)
    all_channels = sorted(raw_df["Channel"].unique().tolist())
    selected_channels = st.multiselect(
        "Channels", all_channels, default=all_channels, label_visibility="collapsed"
    )
    if not selected_channels:
        selected_channels = all_channels

    min_date = raw_df["StartDate"].min().date()
    max_date = raw_df["StartDate"].max().date()
    date_range = st.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        label_visibility="collapsed",
    )
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        d_start, d_end = date_range
    else:
        d_start, d_end = min_date, max_date

    n_journeys = st.slider("Simulated Journeys", 1000, 10000, 5000, 500)
    k_clusters = st.selectbox("ML Clusters (k)", [2, 3, 4, 5, 6], index=2)

    st.markdown("---")
    st.markdown('<div class="sidebar-section-label">Navigation</div>', unsafe_allow_html=True)
    page = st.radio(
        "Page",
        ["Overview", "Attribution", "Channels", "ML Segmentation", "Predictive ML", "Insights"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        f"""<div class="sidebar-stats">
        📊 {len(raw_df):,} campaigns<br>
        📅 {min_date} → {max_date}<br>
        📢 {', '.join(all_channels)}
        </div>""",
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# FILTER DATA
# ══════════════════════════════════════════════════════════════════════════════
df = raw_df[
    (raw_df["Channel"].isin(selected_channels))
    & (raw_df["StartDate"].dt.date >= d_start)
    & (raw_df["StartDate"].dt.date <= d_end)
].copy()

if df.empty:
    st.warning("No data matches the current filters. Please adjust.")
    st.stop()

journeys = simulate_journeys(n_journeys, selected_channels)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL KPIs
# ══════════════════════════════════════════════════════════════════════════════
total_revenue = df["Revenue_USD"].sum()
total_cost = df["Cost_USD"].sum()
total_conversions = df["Conversions"].sum()
avg_roas = df["ROAS"].mean()
n_campaigns = len(df)
journey_converted = sum(1 for j in journeys if j["converted"])
journey_conv_rate = journey_converted / len(journeys) * 100 if journeys else 0

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL HEADER
# ══════════════════════════════════════════════════════════════════════════════
header_chips = [
    ("Revenue", f"${total_revenue/1e6:.2f}M"),
    ("Cost", f"${total_cost/1e6:.2f}M"),
    ("Conversions", f"{total_conversions:,}"),
    ("ROAS", f"{avg_roas:.2f}×"),
    ("Campaigns", f"{n_campaigns:,}"),
    ("Conv Rate", f"{journey_conv_rate:.1f}%"),
]

chips_html = '<div class="chip-row">' + "".join(
    f'<div class="kpi-chip"><div class="chip-label">{lbl}</div><div class="chip-value">{val}</div></div>'
    for lbl, val in header_chips
) + "</div>"

st.markdown(
    f"""<div class="page-header">
    <div class="page-title">{page.upper()}</div>
    {chips_html}
    </div>""",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":

    # ── KPI Strip ──
    st.markdown('<div class="section-label">Key Performance Indicators</div>', unsafe_allow_html=True)
    avg_rev = df["Revenue_USD"].mean()
    avg_cost = df["Cost_USD"].mean()
    avg_profit = df["Profit"].mean()
    avg_conv = df["Conversions"].mean()

    metrics = [
        ("Total Revenue", fmt_num(total_revenue, "$"), df["Revenue_USD"].sum() / avg_rev - 1),
        ("Total Cost", fmt_num(total_cost, "$"), df["Cost_USD"].sum() / avg_cost - 1),
        ("Total Profit", fmt_num(df["Profit"].sum(), "$"), df["Profit"].sum() / (avg_profit * n_campaigns) - 1),
        ("Avg ROAS", f"{avg_roas:.2f}×", avg_roas / 2.0 - 1),
        ("Total Conversions", f"{total_conversions:,}", total_conversions / (avg_conv * n_campaigns) - 1),
        ("Journey Conv Rate", f"{journey_conv_rate:.1f}%", journey_conv_rate / 35 - 1),
    ]

    cols = st.columns(6)
    for col, (label, value, delta) in zip(cols, metrics):
        badge_cls = "delta-up" if delta > 0 else "delta-down"
        delta_sym = "▲" if delta > 0 else "▼"
        col.markdown(
            f"""<div class="mk-metric">
            <div class="metric-num">{value}</div>
            <div class="metric-label">{label}</div>
            <span class="delta-badge {badge_cls}">{delta_sym} {abs(delta)*100:.1f}%</span>
            </div>""",
            unsafe_allow_html=True,
        )

    # ── Revenue by Channel ──
    st.markdown('<div class="section-label">Revenue by Channel</div>', unsafe_allow_html=True)
    ch_rev = df.groupby("Channel")["Revenue_USD"].sum().sort_values()
    ch_rev_pct = ch_rev / ch_rev.sum() * 100
    bar_colors = [CHANNEL_COLORS.get(c, CHART_COLORS[i % 6]) for i, c in enumerate(ch_rev.index)]

    fig_bar = go.Figure(go.Bar(
        x=ch_rev.values,
        y=ch_rev.index,
        orientation="h",
        marker_color=bar_colors,
        text=[f"${v/1e6:.2f}M  ({p:.1f}%)" for v, p in zip(ch_rev.values, ch_rev_pct[ch_rev.index])],
        textposition="outside",
    ))
    apply_theme(fig_bar, title="Revenue by Channel",
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(showgrid=False))
    fig_bar.update_layout(height=300, margin=dict(l=20, r=150, t=50, b=20))
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Donut + Line ──
    st.markdown('<div class="section-label">Revenue Share & Monthly Trend</div>', unsafe_allow_html=True)
    col_l, col_r = st.columns(2)

    with col_l:
        ch_rev_all = df.groupby("Channel")["Revenue_USD"].sum()
        donut_colors = [CHANNEL_COLORS.get(c, "#ccc") for c in ch_rev_all.index]
        fig_donut = go.Figure(go.Pie(
            labels=ch_rev_all.index,
            values=ch_rev_all.values,
            hole=0.55,
            marker_colors=donut_colors,
            textfont=dict(family=FONT_BODY, size=12),
        ))
        apply_theme(fig_donut, title="Revenue Share by Channel")
        fig_donut.update_layout(height=350)
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_r:
        monthly = df.groupby("Month").agg(Revenue=("Revenue_USD", "sum"), Cost=("Cost_USD", "sum")).reset_index()
        monthly = monthly.sort_values("Month")
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=monthly["Month"], y=monthly["Revenue"],
            name="Revenue", line=dict(color="#2563eb", width=2.5),
            fill="tozeroy", fillcolor="rgba(37,99,235,0.08)",
        ))
        fig_line.add_trace(go.Scatter(
            x=monthly["Month"], y=monthly["Cost"],
            name="Cost", line=dict(color="#ea580c", width=2, dash="dash"),
            fill="tozeroy", fillcolor="rgba(234,88,12,0.04)",
        ))
        apply_theme(fig_line, title="Monthly Revenue vs Cost",
                    xaxis=dict(showgrid=False, tickangle=-30, tickfont=dict(size=10)),
                    yaxis=dict(showgrid=True, gridcolor="#f0f0f0", tickprefix="$"))
        fig_line.update_layout(height=350)
        st.plotly_chart(fig_line, use_container_width=True)

    # ── Conversion Funnel ──
    st.markdown('<div class="section-label">Conversion Funnel</div>', unsafe_allow_html=True)
    stages = [
        ("Impressions", int(df["Impressions"].sum())),
        ("Clicks", int(df["Clicks"].sum())),
        ("Leads", int(df["Leads"].sum())),
        ("Conversions", int(df["Conversions"].sum())),
    ]
    funnel_colors = ["#2563eb", "#4f86f7", "#7aa5f9", "#afc7fb"]

    max_val = stages[0][1]
    funnel_html_parts = ['<div style="width:100%">']
    for i, (label, val) in enumerate(stages):
        pct_width = max(val / max_val * 100, 2)  # min 2% so bar is always visible
        color = funnel_colors[i]

        # Drop-off line rendered ABOVE the bar (except first stage)
        if i > 0:
            prev_val = stages[i - 1][1]
            drop = (1 - val / prev_val) * 100 if prev_val else 0
            funnel_html_parts.append(
                f'<div style="font-family:\'Space Grotesk\',sans-serif;font-size:0.72rem;'
                f'color:#dc2626;margin:4px 0 4px 0;">▼ {drop:.1f}% drop-off from {stages[i-1][0]}</div>'
            )

        # Label
        funnel_html_parts.append(
            f'<div style="font-family:\'Space Grotesk\',sans-serif;font-size:0.82rem;'
            f'font-weight:500;color:#6b6b6b;margin-bottom:4px;">{label} — {val:,}</div>'
        )

        # Bar
        funnel_html_parts.append(
            f'<div style="background:#f0f0f0;height:36px;width:100%;margin-bottom:2px;">'
            f'<div style="height:100%;width:{pct_width:.2f}%;background:{color};'
            f'display:flex;align-items:center;padding-left:10px;'
            f'font-family:\'Space Mono\',monospace;font-size:0.8rem;font-weight:700;color:white;">'
            f'{val:,}'
            f'</div>'
            f'</div>'
        )

    funnel_html_parts.append('</div>')
    st.markdown("".join(funnel_html_parts), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ATTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Attribution":
    MODELS = ["First Touch", "Last Touch", "Linear", "Time Decay", "Position Based"]

    if "attr_model" not in st.session_state:
        st.session_state["attr_model"] = "Linear"

    # ── Model Selector ──
    st.markdown('<div class="section-label">Attribution Model</div>', unsafe_allow_html=True)
    btn_cols = st.columns(5)
    for col, m in zip(btn_cols, MODELS):
        is_active = st.session_state["attr_model"] == m
        style = "background:#c8f135;color:#0f0f0f;border:none;font-weight:700;" if is_active else "background:#fff;border:1px solid #ddd;"
        if col.button(
            m,
            key=f"attrBtn_{m}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state["attr_model"] = m
            st.rerun()

    active_model = st.session_state["attr_model"]

    # ── Active model results ──
    credits = compute_attribution(journeys, active_model, selected_channels)
    ch_rev_total = df.groupby("Channel")["Revenue_USD"].sum().to_dict()
    total_rev_all = sum(ch_rev_total.values()) or 1

    st.markdown(f'<div class="section-label">Active Model: {active_model}</div>', unsafe_allow_html=True)

    for ch, pct in sorted(credits.items(), key=lambda x: -x[1]):
        rev_attributed = ch_rev_total.get(ch, 0) * pct / 100
        bar_w = pct
        ch_col = CHANNEL_COLORS.get(ch, "#2563eb")
        st.markdown(
            f"""<div style="margin-bottom:0.6rem;">
            <div style="font-family:'Space Grotesk',sans-serif;font-size:0.82rem;font-weight:600;color:#111;margin-bottom:4px;">
              {ch} <span style="color:#6b6b6b;font-weight:400">— {pct:.1f}%  ·  ${rev_attributed/1e3:.1f}K attributed</span>
            </div>
            <div style="background:#f0f0f0;height:28px;width:100%;">
              <div style="height:100%;width:{bar_w}%;background:{ch_col};"></div>
            </div></div>""",
            unsafe_allow_html=True,
        )

    # ── All Models Comparison ──
    st.markdown('<div class="section-label">All Models Comparison</div>', unsafe_allow_html=True)
    all_credits = {m: compute_attribution(journeys, m, selected_channels) for m in MODELS}
    channels_sorted = selected_channels

    fig_grouped = go.Figure()
    for i, m in enumerate(MODELS):
        fig_grouped.add_trace(go.Bar(
            name=m,
            x=channels_sorted,
            y=[all_credits[m].get(c, 0) for c in channels_sorted],
            marker_color=CHART_COLORS[i % 6],
        ))
    apply_theme(fig_grouped, title="Attribution Credit by Model & Channel",
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="#f0f0f0", ticksuffix="%"))
    fig_grouped.update_layout(barmode="group", height=380)
    st.plotly_chart(fig_grouped, use_container_width=True)

    # ── Model Credit Matrix ──
    st.markdown('<div class="section-label">Model Credit Matrix</div>', unsafe_allow_html=True)
    matrix_data = {m: {c: round(all_credits[m].get(c, 0), 2) for c in channels_sorted} for m in MODELS}
    matrix_df = pd.DataFrame(matrix_data).T  # rows = models, cols = channels

    def highlight_matrix(df_in):
        styled = pd.DataFrame("", index=df_in.index, columns=df_in.columns)
        for col in df_in.columns:
            max_v = df_in[col].max()
            min_v = df_in[col].min()
            styled.loc[df_in[col] == max_v, col] = "background-color:#c8f135;color:#111;font-weight:700"
            styled.loc[df_in[col] == min_v, col] = "background-color:#fee2e2;color:#b91c1c"
        return styled

    st.dataframe(
        matrix_df.style.apply(highlight_matrix, axis=None).format("{:.2f}%"),
        use_container_width=True,
    )

    # ── Top 10 Conversion Paths ──
    st.markdown('<div class="section-label">Top 10 Conversion Paths</div>', unsafe_allow_html=True)
    from collections import Counter
    path_counts = Counter(tuple(j["path"]) for j in journeys if j["converted"])
    top_paths = path_counts.most_common(10)

    for i, (path, cnt) in enumerate(top_paths):
        chips = " → ".join(
            f'<span class="ch-chip" style="background:{CHANNEL_COLORS.get(c,"#666")}">{c}</span>'
            for c in path
        )
        bg = "#f5f5f5" if i % 2 == 0 else "#ffffff"
        st.markdown(
            f"""<div class="path-row" style="background:{bg}">
            <div class="path-chips">{chips}</div>
            <div class="path-count">{cnt} journeys</div>
            </div>""",
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — CHANNELS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Channels":

    ch_stats = df.groupby("Channel").agg(
        Revenue=("Revenue_USD", "sum"),
        Cost=("Cost_USD", "sum"),
        Profit=("Profit", "sum"),
        Conversions=("Conversions", "sum"),
        Impressions=("Impressions", "sum"),
        Clicks=("Clicks", "sum"),
        Campaigns=("CampaignID", "count"),
        ROAS=("ROAS", "mean"),
        ROI=("ROI", "mean"),
        CPA=("CPA", "mean"),
        CTR=("CTR", "mean"),
        CVR=("CVR", "mean"),
    ).reset_index()

    # ── Channel KPI cards ──
    st.markdown('<div class="section-label">Channel Overview</div>', unsafe_allow_html=True)
    cols = st.columns(len(ch_stats))
    for col, (_, row) in zip(cols, ch_stats.iterrows()):
        ch_color = CHANNEL_COLORS.get(row["Channel"], "#2563eb")
        col.markdown(
            f"""<div style="background:{ch_color};padding:1rem;color:white;font-family:'Space Grotesk',sans-serif;">
            <div style="font-family:'Space Mono',monospace;font-size:1rem;font-weight:700;margin-bottom:8px;">{row['Channel']}</div>
            <div style="font-size:0.8rem;opacity:0.85;">Revenue: ${row['Revenue']/1e6:.2f}M</div>
            <div style="font-size:0.8rem;opacity:0.85;">ROAS: {row['ROAS']:.2f}×</div>
            <div style="font-size:0.8rem;opacity:0.85;">CPA: ${row['CPA']:.0f}</div>
            <div style="font-size:0.8rem;opacity:0.85;">Conv: {int(row['Conversions']):,}</div>
            <div style="font-size:0.8rem;opacity:0.85;">CTR: {row['CTR']:.2f}%</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # ── ROAS + CPA charts ──
    st.markdown('<div class="section-label">ROAS & CPA by Channel</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig_roas = go.Figure(go.Bar(
            x=ch_stats["Channel"], y=ch_stats["ROAS"],
            marker_color=[CHANNEL_COLORS.get(c, "#ccc") for c in ch_stats["Channel"]],
            text=ch_stats["ROAS"].round(2), textposition="outside",
        ))
        apply_theme(fig_roas, title="Avg ROAS by Channel",
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor="#f0f0f0"))
        fig_roas.update_layout(height=320)
        st.plotly_chart(fig_roas, use_container_width=True)
    with c2:
        fig_cpa = go.Figure(go.Bar(
            x=ch_stats["Channel"], y=ch_stats["CPA"],
            marker_color=[CHANNEL_COLORS.get(c, "#ccc") for c in ch_stats["Channel"]],
            text=ch_stats["CPA"].round(0), textposition="outside",
        ))
        apply_theme(fig_cpa, title="Avg CPA by Channel (lower = better)",
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor="#f0f0f0", tickprefix="$"))
        fig_cpa.update_layout(height=320)
        st.plotly_chart(fig_cpa, use_container_width=True)

    # ── CTR + CVR ──
    st.markdown('<div class="section-label">Click-Through & Conversion Rates</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        fig_ctr = go.Figure(go.Bar(
            x=ch_stats["Channel"], y=ch_stats["CTR"],
            marker_color=[CHANNEL_COLORS.get(c, "#ccc") for c in ch_stats["Channel"]],
            text=ch_stats["CTR"].round(2), textposition="outside",
        ))
        apply_theme(fig_ctr, title="Avg CTR (%) by Channel",
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor="#f0f0f0", ticksuffix="%"))
        fig_ctr.update_layout(height=320)
        st.plotly_chart(fig_ctr, use_container_width=True)
    with c4:
        fig_cvr = go.Figure(go.Bar(
            x=ch_stats["Channel"], y=ch_stats["CVR"],
            marker_color=[CHANNEL_COLORS.get(c, "#ccc") for c in ch_stats["Channel"]],
            text=ch_stats["CVR"].round(2), textposition="outside",
        ))
        apply_theme(fig_cvr, title="Avg CVR (%) by Channel",
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor="#f0f0f0", ticksuffix="%"))
        fig_cvr.update_layout(height=320)
        st.plotly_chart(fig_cvr, use_container_width=True)

    # ── Revenue vs Cost Scatter ──
    st.markdown('<div class="section-label">Revenue vs Cost (Bubble = Conversions)</div>', unsafe_allow_html=True)
    fig_scatter = go.Figure()
    max_conv = ch_stats["Conversions"].max()
    for _, row in ch_stats.iterrows():
        fig_scatter.add_trace(go.Scatter(
            x=[row["Cost"]], y=[row["Revenue"]],
            mode="markers+text",
            marker=dict(
                size=20 + row["Conversions"] / max_conv * 50,
                color=CHANNEL_COLORS.get(row["Channel"], "#2563eb"),
                opacity=0.8,
                line=dict(width=2, color="white"),
            ),
            text=[row["Channel"]], textposition="top center",
            name=row["Channel"],
        ))

    max_val = max(ch_stats["Cost"].max(), ch_stats["Revenue"].max())
    fig_scatter.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines",
        line=dict(color="#dc2626", dash="dash", width=1.5),
        name="Break-even",
        showlegend=True,
    ))
    apply_theme(fig_scatter, title="Revenue vs Cost by Channel",
                xaxis=dict(tickprefix="$", showgrid=True, gridcolor="#f0f0f0", title="Cost"),
                yaxis=dict(tickprefix="$", showgrid=True, gridcolor="#f0f0f0", title="Revenue"))
    fig_scatter.update_layout(height=420)
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Channel Performance Table ──
    st.markdown('<div class="section-label">Full Channel Performance Table</div>', unsafe_allow_html=True)
    display_df = ch_stats.copy()
    display_df["Revenue"] = display_df["Revenue"].apply(lambda x: f"${x/1e6:.2f}M")
    display_df["Cost"] = display_df["Cost"].apply(lambda x: f"${x/1e6:.2f}M")
    display_df["Profit"] = display_df["Profit"].apply(lambda x: f"${x/1e6:.2f}M")
    display_df["ROAS"] = display_df["ROAS"].apply(lambda x: f"{x:.2f}×")
    display_df["ROI"] = display_df["ROI"].apply(lambda x: f"{x:.1f}%")
    display_df["CPA"] = display_df["CPA"].apply(lambda x: f"${x:.0f}")
    display_df["CTR"] = display_df["CTR"].apply(lambda x: f"{x:.2f}%")
    display_df["CVR"] = display_df["CVR"].apply(lambda x: f"{x:.2f}%")
    display_df["Conversions"] = display_df["Conversions"].astype(int)
    st.dataframe(display_df[["Channel", "Revenue", "Cost", "Profit", "ROAS", "ROI", "CPA", "CTR", "CVR", "Conversions", "Campaigns"]], use_container_width=True)

    csv_dl = ch_stats.to_csv(index=False).encode()
    st.download_button("⬇ Download Channel CSV", csv_dl, "channel_performance.csv", "text/csv")

    # ── Touchpoint Distribution ──
    st.markdown('<div class="section-label">Touchpoint Distribution</div>', unsafe_allow_html=True)
    tp_counts = [len(j["path"]) for j in journeys]
    tp_series = pd.Series(tp_counts).value_counts().sort_index()
    fig_tp = go.Figure(go.Bar(
        x=[f"{i} touchpoint{'s' if i>1 else ''}" for i in tp_series.index],
        y=tp_series.values,
        marker_color="#2563eb",
        text=tp_series.values, textposition="outside",
    ))
    apply_theme(fig_tp, title="Journey Touchpoint Distribution",
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="#f0f0f0"))
    fig_tp.update_layout(height=320)
    st.plotly_chart(fig_tp, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ML SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ML Segmentation":

    @st.cache_data
    def run_kmeans(data, k):
        features = ["CTR", "CVR", "CPC", "ROAS", "ROI", "Cost_USD", "Revenue_USD"]
        X = data[features].fillna(0).replace([np.inf, -np.inf], 0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias = []
        for ki in range(2, 9):
            km_i = KMeans(n_clusters=ki, random_state=42, n_init=10)
            km_i.fit(X_scaled)
            inertias.append(km_i.inertia_)
        return labels, inertias, X_scaled

    labels, inertias, X_scaled = run_kmeans(df, k_clusters)
    df["Cluster"] = labels

    CLUSTER_NAMES = ["High Performers", "Cost Efficient", "Steady Growth", "Underperformers", "Emerging", "Niche"]

    # ── Cluster Overview Cards ──
    st.markdown('<div class="section-label">Cluster Overview</div>', unsafe_allow_html=True)
    cluster_stats = df.groupby("Cluster").agg(
        Count=("CampaignID", "count"),
        ROI=("ROI", "mean"),
        ROAS=("ROAS", "mean"),
        CVR=("CVR", "mean"),
        CPA=("CPA", "mean"),
    ).reset_index()

    cols = st.columns(k_clusters)
    for col, (_, row) in zip(cols, cluster_stats.iterrows()):
        cluster_idx = int(row["Cluster"])
        name = CLUSTER_NAMES[cluster_idx % len(CLUSTER_NAMES)]
        col.markdown(
            f"""<div class="mk-card" style="border-top:3px solid {CHART_COLORS[cluster_idx % 6]}">
            <div style="font-family:'Space Mono',monospace;font-size:0.85rem;font-weight:700;margin-bottom:6px;color:{CHART_COLORS[cluster_idx%6]}">{name}</div>
            <div style="font-family:'Space Grotesk',sans-serif;font-size:0.78rem;color:#6b6b6b;">
              {int(row['Count'])} campaigns<br>
              ROI: {row['ROI']:.1f}% | ROAS: {row['ROAS']:.2f}×<br>
              CVR: {row['CVR']:.2f}% | CPA: ${row['CPA']:.0f}
            </div>
            </div>""",
            unsafe_allow_html=True,
        )

    # ── Elbow Method ──
    st.markdown('<div class="section-label">Elbow Method (Inertia vs k)</div>', unsafe_allow_html=True)
    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(
        x=list(range(2, 9)), y=inertias,
        mode="lines+markers",
        line=dict(color="#2563eb", width=2.5),
        marker=dict(size=8, color="#2563eb"),
        name="Inertia",
    ))
    fig_elbow.add_vline(x=k_clusters, line_dash="dash", line_color="#c8f135",
                        annotation_text=f"k={k_clusters}", annotation_position="top right")
    apply_theme(fig_elbow, title="KMeans Elbow Method",
                xaxis=dict(title="k (clusters)", showgrid=False, dtick=1),
                yaxis=dict(title="Inertia", showgrid=True, gridcolor="#f0f0f0"))
    fig_elbow.update_layout(height=320)
    st.plotly_chart(fig_elbow, use_container_width=True)

    # ── ROI vs ROAS Scatter ──
    st.markdown('<div class="section-label">ROI vs ROAS by Cluster</div>', unsafe_allow_html=True)
    fig_cls = go.Figure()
    for ci in range(k_clusters):
        seg = df[df["Cluster"] == ci]
        name = CLUSTER_NAMES[ci % len(CLUSTER_NAMES)]
        fig_cls.add_trace(go.Scatter(
            x=seg["ROI"], y=seg["ROAS"],
            mode="markers",
            name=name,
            marker=dict(size=6, color=CHART_COLORS[ci % 6], opacity=0.6),
        ))
    apply_theme(fig_cls, title="ROI vs ROAS — Cluster Segments",
                xaxis=dict(title="ROI (%)", showgrid=True, gridcolor="#f0f0f0"),
                yaxis=dict(title="ROAS", showgrid=True, gridcolor="#f0f0f0"))
    fig_cls.update_layout(height=400)
    st.plotly_chart(fig_cls, use_container_width=True)

    # ── Feature Importance (Logistic Regression) ──
    st.markdown('<div class="section-label">Feature Importance — Data-Driven Attribution</div>', unsafe_allow_html=True)
    st.caption("Which metrics best distinguish marketing channels")

    @st.cache_data
    def compute_lr_importance(data):
        features = ["CTR", "CVR", "CPC", "ROAS", "ROI", "Impressions", "Clicks"]
        X = data[features].fillna(0).replace([np.inf, -np.inf], 0)
        le = LabelEncoder()
        y = le.fit_transform(data["Channel"])
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)
        lr = LogisticRegression(max_iter=500, random_state=42)
        lr.fit(X_sc, y)
        importances = np.abs(lr.coef_).mean(axis=0)
        return dict(zip(features, importances))

    fi = compute_lr_importance(df)
    fi_sorted = sorted(fi.items(), key=lambda x: x[1])
    bar_colors_fi = ["#c8f135" if i == len(fi_sorted) - 1 else "#2563eb" for i in range(len(fi_sorted))]
    fig_fi = go.Figure(go.Bar(
        x=[v for _, v in fi_sorted],
        y=[k for k, _ in fi_sorted],
        orientation="h",
        marker_color=bar_colors_fi,
        text=[f"{v:.3f}" for _, v in fi_sorted],
        textposition="outside",
    ))
    apply_theme(fig_fi, title="Feature Importance (Logistic Regression)",
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(showgrid=False))
    fig_fi.update_layout(height=340, margin=dict(l=20, r=100, t=50, b=20))
    st.plotly_chart(fig_fi, use_container_width=True)

    # ── Cluster Detail Table ──
    st.markdown('<div class="section-label">Cluster Detail Table</div>', unsafe_allow_html=True)
    cluster_detail = df.groupby("Cluster").agg(
        Count=("CampaignID", "count"),
        Avg_ROI=("ROI", "mean"),
        Avg_ROAS=("ROAS", "mean"),
        Avg_CVR=("CVR", "mean"),
        Avg_CPA=("CPA", "mean"),
        Avg_Revenue=("Revenue_USD", "mean"),
    ).reset_index()
    cluster_detail["Label"] = cluster_detail["Cluster"].apply(lambda x: CLUSTER_NAMES[x % len(CLUSTER_NAMES)])
    cluster_detail = cluster_detail[["Cluster", "Label", "Count", "Avg_ROI", "Avg_ROAS", "Avg_CVR", "Avg_CPA", "Avg_Revenue"]]
    st.dataframe(cluster_detail.round(2), use_container_width=True)
    st.download_button("⬇ Download Cluster CSV", cluster_detail.to_csv(index=False).encode(), "cluster_detail.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — PREDICTIVE ML
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Predictive ML":

    FEATURES = ["CTR", "CVR", "CPC", "ROAS", "Cost_USD", "Impressions", "Clicks", "Duration_Days"]

    @st.cache_data
    def train_rf(data):
        data = data.copy()
        data["Target"] = (data["ROAS"] > 2.0).astype(int)
        X = data[FEATURES].fillna(0).replace([np.inf, -np.inf], 0)
        y = data["Target"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        y_proba = rf.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm_vals = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        fi_vals = dict(zip(FEATURES, rf.feature_importances_))
        return rf, acc, prec, rec, f1, cm_vals, fpr, tpr, roc_auc, fi_vals

    rf_model, acc, prec, rec, f1, cm_vals, fpr, tpr, roc_auc, fi_vals = train_rf(df)

    # ── Training Summary ──
    st.markdown('<div class="section-label">Model Training Summary (Random Forest · 80/20 Split)</div>', unsafe_allow_html=True)

    m_cols = st.columns(4)
    for col, (lbl, val) in zip(m_cols, [("Accuracy", acc), ("Precision", prec), ("Recall", rec), ("F1 Score", f1)]):
        col.markdown(
            f"""<div class="mk-metric">
            <div class="metric-num">{val:.3f}</div>
            <div class="metric-label">{lbl}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # Confusion matrix
    cm_fig = go.Figure(go.Heatmap(
        z=cm_vals,
        x=["Pred: Standard", "Pred: High Perf"],
        y=["Actual: Standard", "Actual: High Perf"],
        colorscale=[[0, "#ffffff"], [1, "#2563eb"]],
        text=cm_vals, texttemplate="%{text}",
        showscale=False,
    ))
    apply_theme(cm_fig, title="Confusion Matrix",
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False, autorange="reversed"))
    cm_fig.update_layout(height=300)
    st.plotly_chart(cm_fig, use_container_width=True)

    # ── Feature Importance ──
    st.markdown('<div class="section-label">Feature Importance (Random Forest)</div>', unsafe_allow_html=True)
    fi_sorted = sorted(fi_vals.items(), key=lambda x: x[1])
    max_feat = max(fi_vals, key=fi_vals.get)
    fi_colors = ["#c8f135" if k == max_feat else "#2563eb" for k, _ in fi_sorted]

    fig_rfi = go.Figure(go.Bar(
        x=[v for _, v in fi_sorted],
        y=[k for k, _ in fi_sorted],
        orientation="h",
        marker_color=fi_colors,
        text=[f"{v:.3f}" for _, v in fi_sorted],
        textposition="outside",
    ))
    apply_theme(fig_rfi, title="RF Feature Importances",
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(showgrid=False))
    fig_rfi.update_layout(height=340, margin=dict(l=20, r=100, t=50, b=20))
    st.plotly_chart(fig_rfi, use_container_width=True)

    # ── ROC Curve ──
    st.markdown('<div class="section-label">ROC Curve</div>', unsafe_allow_html=True)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={roc_auc:.3f})", line=dict(color="#2563eb", width=2.5)))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(color="#dc2626", dash="dash", width=1.5)))
    apply_theme(fig_roc, title=f"ROC Curve — AUC = {roc_auc:.3f}",
                xaxis=dict(title="False Positive Rate", showgrid=True, gridcolor="#f0f0f0"),
                yaxis=dict(title="True Positive Rate", showgrid=True, gridcolor="#f0f0f0"))
    fig_roc.update_layout(height=380)
    st.plotly_chart(fig_roc, use_container_width=True)

    # ── Live Prediction ──
    st.markdown('<div class="section-label">Live Prediction Tool</div>', unsafe_allow_html=True)
    p_cols = st.columns(4)
    in_ctr = p_cols[0].number_input("CTR (%)", 0.0, 100.0, 3.5, step=0.1)
    in_cvr = p_cols[1].number_input("CVR (%)", 0.0, 100.0, 8.0, step=0.1)
    in_cpc = p_cols[2].number_input("CPC ($)", 0.0, 1000.0, 1.2, step=0.01)
    in_roas = p_cols[3].number_input("ROAS", 0.0, 20.0, 2.5, step=0.1)

    p_cols2 = st.columns(4)
    in_cost = p_cols2[0].number_input("Cost ($)", 0.0, 1000000.0, 5000.0, step=100.0)
    in_imp = p_cols2[1].number_input("Impressions", 0, 10000000, 50000, step=1000)
    in_clicks = p_cols2[2].number_input("Clicks", 0, 1000000, 1500, step=100)
    in_dur = p_cols2[3].number_input("Duration (days)", 0, 365, 30, step=1)

    if st.button("PREDICT PERFORMANCE", type="primary", use_container_width=True):
        X_pred = pd.DataFrame([[in_ctr, in_cvr, in_cpc, in_roas, in_cost, in_imp, in_clicks, in_dur]],
                               columns=FEATURES)
        pred_label = rf_model.predict(X_pred)[0]
        pred_proba = rf_model.predict_proba(X_pred)[0][1]

        if pred_label == 1:
            st.markdown('<div class="pred-result-high">HIGH PERFORMER ✓</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="pred-result-std">STANDARD CAMPAIGN</div>', unsafe_allow_html=True)

        fig_prob = go.Figure(go.Bar(
            x=[pred_proba * 100, (1 - pred_proba) * 100],
            y=["High Performer", "Standard"],
            orientation="h",
            marker_color=["#c8f135", "#e5e7eb"],
            text=[f"{pred_proba*100:.1f}%", f"{(1-pred_proba)*100:.1f}%"],
            textposition="inside",
        ))
        apply_theme(fig_prob, title="Prediction Probability",
                    xaxis=dict(showgrid=False, showticklabels=False, range=[0, 100]),
                    yaxis=dict(showgrid=False))
        fig_prob.update_layout(height=160, margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig_prob, use_container_width=True)

    # ── Campaign Scoring Table ──
    st.markdown('<div class="section-label">Top 20 Predicted High Performers</div>', unsafe_allow_html=True)

    @st.cache_data
    def score_campaigns(data, _model):
        X = data[FEATURES].fillna(0).replace([np.inf, -np.inf], 0)
        scores = _model.predict_proba(X)[:, 1]
        result = data[["CampaignID", "Channel", "ROAS"]].copy()
        result["Predicted Score"] = scores.round(4)
        result["Label"] = result["Predicted Score"].apply(lambda x: "High Performer" if x >= 0.5 else "Standard")
        return result.sort_values("Predicted Score", ascending=False).head(20)

    scored = score_campaigns(df, rf_model)
    st.dataframe(scored, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Insights":

    ch_stats_ins = df.groupby("Channel").agg(
        Revenue=("Revenue_USD", "sum"),
        ROAS=("ROAS", "mean"),
        Conversions=("Conversions", "sum"),
        ROI=("ROI", "mean"),
        Cost=("Cost_USD", "sum"),
        CPA=("CPA", "mean"),
    )

    best_roas_ch = ch_stats_ins["ROAS"].idxmax()
    best_roas_val = ch_stats_ins["ROAS"].max()
    best_conv_ch = ch_stats_ins["Conversions"].idxmax()
    best_conv_val = int(ch_stats_ins["Conversions"].max())
    worst_roi_ch = ch_stats_ins["ROI"].idxmin()
    worst_roi_val = ch_stats_ins["ROI"].min()
    best_rev_ch = ch_stats_ins["Revenue"].idxmax()
    best_rev_val = ch_stats_ins["Revenue"].max()
    lowest_cpa_ch = ch_stats_ins["CPA"].idxmin()
    lowest_cpa_val = ch_stats_ins["CPA"].min()
    highest_cost_ch = ch_stats_ins["Cost"].idxmax()

    insights = [
        (best_roas_ch, f"delivers the highest average ROAS of <b>{best_roas_val:.2f}×</b> — the most efficient channel for revenue generation per dollar spent."),
        (best_conv_ch, f"leads in total conversions with <b>{best_conv_val:,}</b> — indicating strong bottom-funnel performance."),
        (worst_roi_ch, f"shows the lowest ROI at <b>{worst_roi_val:.1f}%</b> — this channel should be reviewed for budget reallocation."),
        (best_rev_ch, f"generates the highest total revenue at <b>${best_rev_val/1e6:.2f}M</b> — a high-priority channel worth scaling."),
        (lowest_cpa_ch, f"has the lowest Cost per Acquisition at <b>${lowest_cpa_val:.0f}</b> — making it the most cost-effective for customer acquisition."),
        (highest_cost_ch, f"consumes the highest budget — ensure spend efficiency is continuously monitored against ROI benchmarks."),
    ]

    # ── Key Findings ──
    st.markdown('<div class="section-label">Key Findings</div>', unsafe_allow_html=True)
    for i, (channel, text) in enumerate(insights, 1):
        num_str = f"{i:02d}"
        st.markdown(
            f"""<div class="insight-item">
            <div class="insight-num">{num_str}</div>
            <div class="insight-text"><span class="insight-bold">{channel}</span> {text}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # ── Budget Recommendations ──
    st.markdown('<div class="section-label">Budget Recommendations</div>', unsafe_allow_html=True)

    def budget_action(ch, stats):
        roas = stats.loc[ch, "ROAS"]
        roi = stats.loc[ch, "ROI"]
        if roas > 3.0 and roi > 50:
            return "Scale", "badge-scale", "High ROAS and ROI indicate strong returns. Increase budget allocation by 20–30%."
        elif roas > 1.5:
            return "Maintain", "badge-maintain", "Solid performance. Maintain current spend while optimizing creative and targeting."
        else:
            return "Audit", "badge-audit", "Below-average ROAS. Review targeting, creative, and audience before continuing investment."

    b_cols = st.columns(len(ch_stats_ins))
    for col, (ch, _) in zip(b_cols, ch_stats_ins.iterrows()):
        action, badge_cls, reason = budget_action(ch, ch_stats_ins)
        col.markdown(
            f"""<div class="budget-card">
            <div class="budget-badge {badge_cls}">{action}</div>
            <div style="font-family:'Space Mono',monospace;font-size:0.9rem;font-weight:700;margin-bottom:6px;">{ch}</div>
            <div style="font-family:'Space Grotesk',sans-serif;font-size:0.76rem;color:#6b6b6b;line-height:1.4">{reason}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # ── Attribution Model Guide ──
    st.markdown('<div class="section-label">Attribution Model Guide</div>', unsafe_allow_html=True)
    model_guides = [
        ("First Touch", "Best for: Brand awareness campaigns where first interaction drives discovery."),
        ("Last Touch", "Best for: Performance campaigns focused on closing — credits final action before conversion."),
        ("Linear", "Best for: Multi-touch journeys where all touchpoints contribute equally. RECOMMENDED."),
        ("Time Decay", "Best for: Short sales cycles where recent touchpoints are more influential."),
        ("Position Based", "Best for: Balanced weighting of first and last touch with mid-journey context."),
    ]
    g_cols = st.columns(5)
    for col, (mname, use_case) in zip(g_cols, model_guides):
        is_rec = mname == "Linear"
        card_cls = "model-guide-card active-model" if is_rec else "model-guide-card"
        text_color = "#f0f0f0" if is_rec else "#6b6b6b"
        name_color = "#c8f135" if is_rec else "#111"
        col.markdown(
            f"""<div class="{card_cls}">
            <div class="model-name" style="color:{name_color}">{mname}</div>
            <div class="model-use" style="color:{text_color}">{use_case}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # ── Export ──
    st.markdown('<div class="section-label">Export</div>', unsafe_allow_html=True)

    ch_summary_export = df.groupby("Channel").agg(
        Revenue=("Revenue_USD", "sum"),
        Cost=("Cost_USD", "sum"),
        Profit=("Profit", "sum"),
        ROAS=("ROAS", "mean"),
        ROI=("ROI", "mean"),
        CPA=("CPA", "mean"),
        CTR=("CTR", "mean"),
        CVR=("CVR", "mean"),
        Campaigns=("CampaignID", "count"),
    ).reset_index()

    all_credits_export = {m: compute_attribution(journeys, m, selected_channels) for m in
                          ["First Touch", "Last Touch", "Linear", "Time Decay", "Position Based"]}
    attr_rows = [{"Model": m, **credits_d} for m, credits_d in all_credits_export.items()]
    attr_df_export = pd.DataFrame(attr_rows)

    e_cols = st.columns(3)
    with e_cols[0]:
        st.download_button(
            "⬇ DOWNLOAD CHANNEL SUMMARY CSV",
            ch_summary_export.to_csv(index=False).encode(),
            "channel_summary.csv",
            "text/csv",
            use_container_width=True,
        )
    with e_cols[1]:
        st.download_button(
            "⬇ DOWNLOAD ATTRIBUTION RESULTS CSV",
            attr_df_export.to_csv(index=False).encode(),
            "attribution_results.csv",
            "text/csv",
            use_container_width=True,
        )
    with e_cols[2]:
        if st.button("📄 GENERATE PDF REPORT", use_container_width=True):
            buf = io.BytesIO()
            doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm,
                                    leftMargin=2*cm, rightMargin=2*cm)
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle("Title", parent=styles["Title"], fontSize=22, spaceAfter=12)
            h2_style = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, spaceBefore=14, spaceAfter=6)
            normal_style = ParagraphStyle("Normal", parent=styles["Normal"], fontSize=9, leading=13)

            story = []
            story.append(Paragraph("MKTG·ATTR — Marketing Attribution Report", title_style))
            story.append(Spacer(1, 0.3*cm))
            story.append(Paragraph(f"Generated from {n_campaigns:,} campaigns · {d_start} to {d_end}", normal_style))
            story.append(Spacer(1, 0.5*cm))

            story.append(Paragraph("Key Performance Indicators", h2_style))
            kpi_data = [
                ["Metric", "Value"],
                ["Total Revenue", f"${total_revenue:,.0f}"],
                ["Total Cost", f"${total_cost:,.0f}"],
                ["Total Profit", f"${df['Profit'].sum():,.0f}"],
                ["Avg ROAS", f"{avg_roas:.2f}×"],
                ["Total Conversions", f"{total_conversions:,}"],
                ["Journey Conv Rate", f"{journey_conv_rate:.1f}%"],
            ]
            kpi_table = Table(kpi_data, colWidths=[8*cm, 7*cm])
            kpi_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111111")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#c8f135")),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
                ("ALIGN", (1, 0), (1, -1), "RIGHT"),
            ]))
            story.append(kpi_table)
            story.append(Spacer(1, 0.5*cm))

            story.append(Paragraph("Channel Summary", h2_style))
            ch_cols_pdf = ["Channel", "Revenue", "Cost", "Profit", "ROAS", "CPA", "CTR", "CVR"]
            ch_rows_pdf = [ch_cols_pdf]
            for _, row in ch_summary_export.iterrows():
                ch_rows_pdf.append([
                    row["Channel"],
                    f"${row['Revenue']:,.0f}",
                    f"${row['Cost']:,.0f}",
                    f"${row['Profit']:,.0f}",
                    f"{row['ROAS']:.2f}×",
                    f"${row['CPA']:.0f}",
                    f"{row['CTR']:.2f}%",
                    f"{row['CVR']:.2f}%",
                ])
            ch_table = Table(ch_rows_pdf, colWidths=[2.5*cm, 2.2*cm, 2.2*cm, 2.2*cm, 1.8*cm, 1.8*cm, 1.8*cm, 1.8*cm])
            ch_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111111")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
            ]))
            story.append(ch_table)
            story.append(Spacer(1, 0.5*cm))

            story.append(Paragraph("Attribution Results (Linear Model)", h2_style))
            linear_creds = all_credits_export.get("Linear", {})
            attr_rows_pdf = [["Channel", "Credit (%)"]] + [[c, f"{v:.1f}%"] for c, v in linear_creds.items()]
            attr_table = Table(attr_rows_pdf, colWidths=[8*cm, 7*cm])
            attr_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111111")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
                ("ALIGN", (1, 0), (1, -1), "RIGHT"),
            ]))
            story.append(attr_table)
            story.append(Spacer(1, 0.5*cm))

            story.append(Paragraph("Key Insights", h2_style))
            for i, (ch, text_ins) in enumerate(insights, 1):
                clean_text = text_ins.replace("<b>", "").replace("</b>", "")
                story.append(Paragraph(f"{i:02d}. {ch} — {clean_text}", normal_style))
                story.append(Spacer(1, 0.15*cm))

            doc.build(story)
            buf.seek(0)
            st.download_button(
                "⬇ Download PDF Report",
                buf.getvalue(),
                "mktg_attr_report.pdf",
                "application/pdf",
                use_container_width=True,
            )
