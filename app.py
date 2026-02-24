import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# --- CONFIGURATION
st.set_page_config(page_title="NVDA Earnings Intelligence", layout="wide", initial_sidebar_state="expanded")

if 'app_theme' not in st.session_state:
    st.session_state.app_theme = 'Light'

IS_DARK = st.session_state.app_theme == 'Dark'

# --- ACCENT COLORS (same for both themes)
C_GREEN = "#76b900"   # NVDA green
C_BLUE  = "#00a8e0"   # accent blue
C_RED   = "#d94f4f"
C_GOLD  = "#d4a017"   # darker gold — readable on white

# --- THEME-DEPENDENT COLORS
if IS_DARK:
    C_BG          = "#0e1117"
    C_CARD        = "#1a1f2e"
    C_TEXT        = "#e8eaf0"
    C_MUTED       = "#9aa5b8"
    C_GRID        = "#2a3040"
    C_BORDER      = "#2a3040"
    C_CARD_ALT    = "#242938"
    C_LINE_INDIV  = "rgba(180,200,230,0.50)"
    PLOTLY_TPL    = "plotly_dark"
    NAV_BG        = "#141828"
    NAV_BORDER    = "#2a3040"
    NAV_TEXT      = "#9aa5b8"
    INSIGHT_BG    = "#1e2333"
    INSIGHT_BDR   = "#d4a017"
    GUIDE_BG      = "#1a2230"
    GUIDE_BDR     = "#2a7a3a"
    GUIDE_TEXT    = "#c8d4e8"
    METRIC_BG     = "#1e2333"
    METRIC_TEXT   = "#8899aa"
    H2_BORDER     = "#2a3040"
    HR_COLOR      = "#2a3040"
    CARD_SHADOW   = "0 2px 10px rgba(0,0,0,0.5)"
else:
    C_BG          = "#f4f6fa"
    C_CARD        = "#ffffff"
    C_TEXT        = "#1a202c"
    C_MUTED       = "#4a5568"
    C_GRID        = "#d8e0ee"
    C_BORDER      = "#dde4f0"
    C_CARD_ALT    = "#eef1f8"
    C_LINE_INDIV  = "rgba(50,80,160,0.35)"
    PLOTLY_TPL    = "plotly_white"
    NAV_BG        = "#e4e9f4"
    NAV_BORDER    = "#c8d0e4"
    NAV_TEXT      = "#374151"
    INSIGHT_BG    = "#fffdf0"
    INSIGHT_BDR   = "#c8900e"
    GUIDE_BG      = "#f0f7ff"
    GUIDE_BDR     = "#2d7d3a"
    GUIDE_TEXT    = "#1a202c"
    METRIC_BG     = "#f0f3fa"
    METRIC_TEXT   = "#4a5568"
    H2_BORDER     = "#d0d8ec"
    HR_COLOR      = "#d0d8ec"
    CARD_SHADOW   = "0 2px 8px rgba(0,0,0,0.08)"

PLOTLY_THEME = dict(
    template=PLOTLY_TPL,
    paper_bgcolor=C_CARD,
    plot_bgcolor=C_CARD,
    font=dict(family="Inter, sans-serif", color=C_TEXT, size=12),
    margin=dict(l=40, r=20, t=50, b=40),
)

# We use a unique class .nvda-nav-radio to scope nav styles
# away from the sidebar radio, avoiding cross-contamination.
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    /* ── Page base ── */
    .stApp, .stApp > header, section.main > div {{
        background-color: {C_BG} !important;
    }}
    .block-container {{
        padding: 1.5rem 2rem !important;
        max-width: 1600px !important;
        background-color: {C_BG} !important;
    }}
    /* Default text */
    .stApp, .stApp p, .stApp span, .stApp div, .stApp li {{
        color: {C_TEXT};
        font-family: 'Inter', sans-serif;
    }}
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span {{
        color: {C_TEXT} !important;
    }}

    /* ── Navigation tab bar (main content area only) ── */
    .nvda-nav-wrap div[data-testid="stRadio"] > div {{
        flex-direction: row !important;
        flex-wrap: wrap !important;
        gap: 3px !important;
        background: {NAV_BG} !important;
        border-radius: 10px !important;
        padding: 5px !important;
        border: 1px solid {NAV_BORDER} !important;
    }}
    .nvda-nav-wrap div[data-testid="stRadio"] > div > label {{
        border-radius: 7px !important;
        padding: 8px 18px !important;
        color: {NAV_TEXT} !important;
        font-weight: 500 !important;
        background: transparent !important;
        cursor: pointer !important;
        transition: all 0.15s ease !important;
        font-size: 13px !important;
    }}
    .nvda-nav-wrap div[data-testid="stRadio"] > div > label:hover {{
        background: rgba(118,185,0,0.18) !important;
        color: {C_TEXT} !important;
    }}
    .nvda-nav-wrap div[data-testid="stRadio"] > div > label[data-checked="true"] {{
        background: {C_GREEN} !important;
        color: #000000 !important;
        font-weight: 700 !important;
        box-shadow: 0 2px 6px rgba(118,185,0,0.4) !important;
    }}
    .nvda-nav-wrap div[data-testid="stRadio"] > div > label > div:first-child {{
        display: none !important;
    }}
    /* Hide the nav label text */
    .nvda-nav-wrap div[data-testid="stRadio"] > label {{
        display: none !important;
    }}

    /* ── KPI Metric Cards ── */
    div[data-testid="metric-container"] {{
        background: {C_CARD} !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        border: none !important;
        border-left: 4px solid {C_GREEN} !important;
        box-shadow: {CARD_SHADOW} !important;
    }}
    div[data-testid="metric-container"] label,
    div[data-testid="metric-container"] [data-testid="stMetricLabel"] p {{
        color: {C_MUTED} !important;
        font-size: 11px !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }}
    div[data-testid="metric-container"] [data-testid="stMetricValue"],
    div[data-testid="metric-container"] [data-testid="stMetricValue"] p {{
        color: {C_TEXT} !important;
        font-size: 26px !important;
        font-weight: 700 !important;
    }}

    /* ── Typography ── */
    h1 {{ color: {C_TEXT} !important; font-size: 30px !important; font-weight: 700 !important; margin-bottom: 8px !important; }}
    h2 {{ color: {C_TEXT} !important; font-size: 18px !important; font-weight: 600 !important;
          border-bottom: 1px solid {H2_BORDER} !important; padding-bottom: 6px !important; margin-top: 18px !important; }}
    h3 {{ color: {C_MUTED} !important; font-size: 13px !important; font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: 1px !important; }}

    /* ── Custom info boxes ── */
    .insight-box {{
        background: {INSIGHT_BG} !important;
        border-left: 4px solid {INSIGHT_BDR} !important;
        border-radius: 8px !important;
        padding: 14px 18px !important;
        margin: 10px 0 !important;
        font-size: 13px !important;
        line-height: 1.7 !important;
        color: {C_TEXT} !important;
        box-shadow: {CARD_SHADOW} !important;
    }}
    .insight-box b, .insight-box strong {{ color: {C_TEXT} !important; }}

    .guide-box {{
        background: {GUIDE_BG} !important;
        border: 1px solid {GUIDE_BDR} !important;
        border-left: 4px solid {GUIDE_BDR} !important;
        border-radius: 8px !important;
        padding: 14px 18px !important;
        margin: 10px 0 !important;
        font-size: 13px !important;
        line-height: 1.7 !important;
        color: {GUIDE_TEXT} !important;
    }}
    .guide-box b, .guide-box strong {{ color: {C_TEXT} !important; }}
    .guide-box span {{ color: inherit !important; }}

    .metric-explain {{
        background: {METRIC_BG} !important;
        border-radius: 6px !important;
        padding: 10px 14px !important;
        margin: 8px 0 !important;
        font-size: 12px !important;
        color: {METRIC_TEXT} !important;
        border: 1px solid {C_BORDER} !important;
    }}
    .metric-explain b, .metric-explain strong {{ color: {C_TEXT} !important; }}

    /* ── DataFrames / tables ── */
    .stDataFrame {{ border-radius: 10px !important; overflow: hidden !important; border: 1px solid {C_BORDER} !important; }}
    .stDataFrame table th {{ background: {C_CARD_ALT} !important; color: {C_TEXT} !important; font-weight: 600 !important; }}
    .stDataFrame table td {{ color: {C_TEXT} !important; background: {C_CARD} !important; }}

    /* ── Streamlit Arrow / glide-data-grid (iframe-based dataframe) dark mode ── */
    .stDataFrame > div {{ background: {C_CARD} !important; }}
    .stDataFrame [data-testid="stDataFrameResizable"] {{ background: {C_CARD} !important; }}
    /* Header row */
    .stDataFrame .dvn-scroller .gdg-header-row,
    .stDataFrame [role="columnheader"],
    .stDataFrame [class*="header"] {{ 
        background-color: {C_CARD_ALT} !important; 
        color: {C_TEXT} !important; 
    }}
    /* Data cells */
    .stDataFrame [role="gridcell"],
    .stDataFrame [role="row"],
    .stDataFrame [class*="row"] {{ 
        background-color: {C_CARD} !important; 
        color: {C_TEXT} !important; 
    }}
    /* Canvas-based grid — inject background via wrapper */
    .stDataFrame canvas {{ filter: {'invert(1) hue-rotate(180deg)' if IS_DARK else 'none'} !important; }}
    /* Full iframe override */
    .stDataFrame iframe {{ 
        background: {C_CARD} !important; 
        color-scheme: {'dark' if IS_DARK else 'light'} !important;
    }}

    /* ── Misc ── */
    hr {{ border-color: {HR_COLOR} !important; margin: 24px 0 !important; }}

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {{
        background: {C_CARD} !important;
        border-right: 1px solid {C_BORDER} !important;
    }}
    section[data-testid="stSidebar"] * {{ color: {C_TEXT} !important; }}
    section[data-testid="stSidebar"] label {{ color: {C_MUTED} !important; }}

    /* ── Widgets ── */
    .stSelectbox label, .stSlider label, .stNumberInput label,
    .stMultiSelect label {{ color: {C_TEXT} !important; font-weight: 500 !important; }}
    div[data-baseweb="select"] > div {{
        background: {C_CARD} !important;
        border-color: {C_BORDER} !important;
        color: {C_TEXT} !important;
    }}
    div[data-baseweb="select"] span {{ color: {C_TEXT} !important; }}
    input[type="number"] {{
        background: {C_CARD} !important;
        color: {C_TEXT} !important;
        border-color: {C_BORDER} !important;
    }}
</style>
""", unsafe_allow_html=True)

# --- FILE PATHS
RET_FILE  = "01_case_study_returns.csv"
LOAD_FILE = "02_case_study_factor_loadings.csv"
EARN_FILE = "03_case_study_earnings_dates.csv"

KNOWN_EVENTS = {
    "2022-11-16": ("Miss", "Crypto/Gaming collapse. Channel inventory glut."),
    "2023-02-22": ("Beat", "Data center recovery. Beat on margins."),
    "2023-05-24": ("Mega Beat", "The AI Inflection Quarter. Guidance +50%. Shocked consensus."),
    "2023-08-23": ("Beat", "AI Data Center demand confirmed. Blackwell roadmap unveiled."),
    "2023-11-21": ("Beat", "H100 supply ramp. Record data center revenue."),
    "2024-02-21": ("Mega Beat", "Most important stock on Earth.' Blowout beat on AI demand."),
    "2024-05-22": ("Beat", "Blackwell transition. Beat but supply concerns emerged."),
    "2024-08-28": ("Beat/Concern", "Blackwell delays flagged. Beat but guidance disappointed."),
    "2024-11-20": ("Beat", "Blackwell ramp confirmed. Data center record."),
    "2025-01-27": ("DeepSeek Shock", "DeepSeek R1 release triggered AI capex fears. NVDA -17% intraday - largest single-day market cap loss in history at the time."),
    "2025-02-26": ("Beat", "Strong beat. DeepSeek fears faded post-earnings. Guidance raised."),
    "2025-05-28": ("Beat", "Continued AI infrastructure spend. Export controls headwind."),
    "2025-08-27": ("Flat", " Data center Revenue ($41.1B) missed Street expectations ($41.3B)"),
}

_DF_PROPS = {}  # Never override per-cell — would kill background_gradient colors
_DF_TABLE_STYLES = [
    {'selector': 'th', 'props': [
        ('background-color', C_CARD_ALT),
        ('color', C_TEXT),
        ('border', f'1px solid {C_BORDER}'),
        ('font-weight', '600'),
    ]},
    {'selector': 'td', 'props': [
        ('color', C_TEXT),
        ('border', f'1px solid {C_BORDER}'),
    ]},
    {'selector': 'table', 'props': [
        ('background-color', C_CARD),
    ]},
    {'selector': '', 'props': [
        ('background-color', C_CARD),
    ]},
] if IS_DARK else []

# --- DATA LOADING
@st.cache_data
def load_data():
    df_ret = pd.read_csv(RET_FILE, skiprows=2)
    df_ret.columns = df_ret.columns.str.strip()
    df_ret['Date'] = pd.to_datetime(df_ret['Date'], errors='coerce')
    df_ret = df_ret.dropna(subset=['Date']).set_index('Date').sort_index()

    df_load = pd.read_csv(LOAD_FILE, skiprows=2)
    df_load.columns = df_load.columns.str.strip()
    df_load['Date'] = pd.to_datetime(df_load['Date'], errors='coerce')
    df_load = df_load.dropna(subset=['Date']).set_index('Date').sort_index()

    df_earn = pd.read_csv(EARN_FILE, header=None, skiprows=1)
    df_earn.columns = ['Date']
    df_earn['Date'] = pd.to_datetime(df_earn['Date'], errors='coerce')
    earnings_dates = df_earn['Date'].dropna().sort_values().tolist()

    target = 'NVDA'
    factor_cols = [c for c in df_ret.columns if c != target]
    if df_ret[target].abs().quantile(0.75) > 0.5:
        df_ret[target] /= 100.0
        for f in factor_cols:
            if df_ret[f].abs().quantile(0.75) > 0.5:
                df_ret[f] /= 100.0

    df = df_ret.join(df_load, how='inner', rsuffix='_L')
    load_cols = [c for c in df_load.columns]
    factor_cols = [c for c in df_ret.columns if c != target]

    contrib = pd.DataFrame(index=df.index)
    for f in factor_cols:
        lc = f + '_L'
        if lc in df.columns:
            contrib[f] = df[f] * df[lc]

    df['Factor_Return'] = contrib.sum(axis=1)
    df['Total_Return'] = df[target]
    df['Idio_Return'] = df['Total_Return'] - df['Factor_Return']

    mkt_cols   = [c for c in ['Market', 'Beta', 'Excess Beta'] if c in contrib.columns]
    semi_cols  = [c for c in ['Semiconductors'] if c in contrib.columns]
    style_cols = [c for c in ['Value','Growth','Momentum','Quality','Size','Mid Cap',
                               'Volatility','Liquidity','Leverage','Dividend Yield','Earnings Yield']
                  if c in contrib.columns]
    df['Grp_Market'] = contrib[mkt_cols].sum(axis=1) if mkt_cols else 0.0
    df['Grp_Semi']   = contrib[semi_cols].sum(axis=1) if semi_cols else 0.0
    df['Grp_Style']  = contrib[style_cols].sum(axis=1) if style_cols else 0.0

    return df, contrib, earnings_dates, factor_cols, df_load

@st.cache_data
def build_event_windows(df, earnings_dates, window=40):
    chunks = []
    for i, edate in enumerate(earnings_dates):
        future = df.index[df.index > edate]
        if len(future) == 0:
            continue
        reaction_day = future[0]
        try:
            loc = df.index.get_loc(reaction_day)
        except KeyError:
            continue
        start = max(0, loc - window)
        end   = min(len(df) - 1, loc + window)
        chunk = df.iloc[start:end+1].copy()
        rel_start = - (loc - start)
        rel_end   = end - loc
        chunk['Rel_Time']      = np.arange(rel_start, rel_end + 1)
        chunk['Event_ID']      = f"Q{i+1} ({edate.strftime('%b %Y')})"
        chunk['Earnings_Date'] = edate
        chunk['Quarter_Num']   = i + 1
        chunks.append(chunk)
    return pd.concat(chunks) if chunks else pd.DataFrame()

@st.cache_data
def event_stats(event_df):
    return event_df.groupby('Rel_Time')[['Total_Return','Factor_Return','Idio_Return']].agg(
        ['mean','median','std','min','max','count'])

# --- ANALYTICS HELPERS
def sharpe(returns, ann=252):
    r = np.array(returns)
    if r.std() == 0:
        return 0.0
    return (r.mean() / r.std()) * np.sqrt(ann)

def max_drawdown(equity_curve):
    peak = np.maximum.accumulate(equity_curve)
    dd   = (equity_curve - peak) / peak
    return dd.min()

def kelly(win_rate, avg_win, avg_loss):
    if avg_loss == 0:
        return 0.0
    return win_rate - (1 - win_rate) / (abs(avg_win) / abs(avg_loss))

# --- PLOT HELPERS
def apply_theme(fig, height=420, title=""):
    fig.update_layout(
        height=height,
        title=dict(text=title, font=dict(size=14, color=C_TEXT)),
        **PLOTLY_THEME
    )
    fig.update_xaxes(
        gridcolor=C_GRID, zeroline=False,
        tickfont=dict(color=C_TEXT),
        title_font=dict(color=C_TEXT),
        linecolor=C_GRID
    )
    fig.update_yaxes(
        gridcolor=C_GRID, zeroline=False,
        tickfont=dict(color=C_TEXT),
        title_font=dict(color=C_TEXT),
        linecolor=C_GRID
    )
    # Ensure legend text is visible
    if fig.layout.legend:
        fig.update_layout(legend=dict(font=dict(color=C_TEXT)))
    # Update text font on traces
    for trace in fig.data:
        if hasattr(trace, 'textfont') and trace.textfont is not None:
            trace.textfont.color = C_TEXT
        if hasattr(trace, 'text') and trace.text is not None:
            if not hasattr(trace, 'textfont') or trace.textfont is not None:
                trace.update(textfont=dict(color=C_TEXT))
    return fig

def plot_cumulative_paths(event_df, metric='Idio_Return', title=""):
    pivot = event_df.pivot(index='Rel_Time', columns='Event_ID', values=metric)
    cum   = pivot.fillna(0).cumsum() * 100
    mean  = cum.mean(axis=1)
    med   = cum.median(axis=1)
    se    = cum.std(axis=1) / np.sqrt(cum.shape[1])
    upper = mean + 1.96 * se
    lower = mean - 1.96 * se
    cmax  = cum.max(axis=1)
    cmin  = cum.min(axis=1)

    range_fill  = "rgba(100,120,160,0.25)" if IS_DARK else "rgba(60,80,140,0.10)"
    range_line  = "rgba(150,170,210,0.4)"  if IS_DARK else "rgba(80,100,160,0.3)"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(cmax.index) + list(cmin.index[::-1]),
        y=list(cmax) + list(cmin[::-1]),
        fill='toself', fillcolor=range_fill,
        line=dict(color=range_line, width=1),
        name='Full Range', showlegend=True, hoverinfo='skip'))
    fig.add_trace(go.Scatter(
        x=list(upper.index) + list(lower.index[::-1]),
        y=list(upper) + list(lower[::-1]),
        fill='toself', fillcolor='rgba(118,185,0,0.25)',
        line=dict(color='rgba(118,185,0,0.5)', width=1),
        name='95% CI', showlegend=True, hoverinfo='skip'))
    for col in cum.columns:
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum[col], mode='lines',
            line=dict(color=C_LINE_INDIV, width=1.2),
            name=col, showlegend=False,
            hovertemplate=f"{col}: %{{y:.2f}}%<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=med.index, y=med, mode='lines',
        line=dict(color=C_BLUE, width=2.5, dash='dash'), name='Median'))
    fig.add_trace(go.Scatter(
        x=mean.index, y=mean, mode='lines',
        line=dict(color=C_GREEN, width=3.5), name='Mean'))
    fig.add_vline(x=0, line_dash="dash", line_color=C_GOLD,
        annotation_text="Earnings", annotation_font_color=C_TEXT,
        annotation_font_size=11)
    fig.add_hline(y=0, line_color=C_MUTED, line_width=1.2)
    fig.update_layout(
        xaxis_title="Days Relative to Earnings",
        yaxis_title="Cumulative Return (%)",
        legend=dict(orientation="h", y=1.08, font=dict(color=C_TEXT)))
    return apply_theme(fig, 500, title)

def plot_waterfall(labels, values, title=""):
    measure = ['relative'] * (len(values) - 1) + ['total']
    connector_color = 'rgba(255,255,255,0.20)' if IS_DARK else 'rgba(0,0,0,0.15)'
    fig = go.Figure(go.Waterfall(
        orientation='v', measure=measure, x=labels, y=values,
        text=[f"{v:+.2f}%" for v in values], textposition='outside',
        textfont=dict(color=C_TEXT, size=11),
        connector=dict(line=dict(color=connector_color)),
        increasing=dict(marker_color=C_GREEN),
        decreasing=dict(marker_color=C_RED),
        totals=dict(marker_color=C_BLUE)
    ))
    return apply_theme(fig, 440, title)

def plot_vol_term_structure(event_df):
    windows = [(-40,-21),(-20,-11),(-10,-6),(-5,-1),(0,0),(1,5),(6,10),(11,20),(21,40)]
    labels  = ['T-40:-21','T-20:-11','T-10:-6','T-5:-1','T=0','T+1:+5','T+6:+10','T+11:+20','T+21:+40']
    vols, colors = [], []
    for s, e in windows:
        sub = event_df[(event_df['Rel_Time'] >= s) & (event_df['Rel_Time'] <= e)]
        v = sub['Total_Return'].std() * np.sqrt(252) * 100 if not sub.empty else 0
        vols.append(v)
        colors.append(C_RED if s == 0 else (C_GOLD if abs(s) <= 5 else C_BLUE))
    fig = go.Figure(go.Bar(
        x=labels, y=vols, marker_color=colors,
        text=[f"{v:.1f}%" for v in vols], textposition='outside',
        textfont=dict(color=C_TEXT, size=10)))
    fig.update_layout(xaxis_title="Event Window", yaxis_title="Annualized Vol (%)")
    return apply_theme(fig, 380, "Volatility Term Structure Around Earnings")

def plot_regime_comparison(event_df):
    ai_start = pd.Timestamp("2023-05-24")
    pre  = event_df[event_df['Earnings_Date'] <  ai_start]
    post = event_df[event_df['Earnings_Date'] >= ai_start]

    def t0_stats(sub):
        t = sub[sub['Rel_Time'] == 0]['Total_Return'] * 100
        return dict(mean=t.mean(), std=t.std(), win=(t > 0).mean() * 100,
                    idio=sub[sub['Rel_Time']==0]['Idio_Return'].abs().mean()*100)

    s_pre  = t0_stats(pre)
    s_post = t0_stats(post)
    cats   = ['Avg Move (%)', 'Volatility (%)', 'Win Rate (%)', 'Avg |Alpha| (%)']
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Pre-AI Era (2022-May 2023)', x=cats,
        y=[s_pre['mean'], s_pre['std'], s_pre['win'], s_pre['idio']],
        marker_color=C_BLUE,
        text=[f"{v:.1f}" for v in [s_pre['mean'],s_pre['std'],s_pre['win'],s_pre['idio']]],
        textposition='outside', textfont=dict(color=C_TEXT)))
    fig.add_trace(go.Bar(
        name='AI Era (May 2023-Present)', x=cats,
        y=[s_post['mean'], s_post['std'], s_post['win'], s_post['idio']],
        marker_color=C_GREEN,
        text=[f"{v:.1f}" for v in [s_post['mean'],s_post['std'],s_post['win'],s_post['idio']]],
        textposition='outside', textfont=dict(color=C_TEXT)))
    fig.update_layout(barmode='group', legend=dict(orientation='h', y=1.1, font=dict(color=C_TEXT)))
    return apply_theme(fig, 380, "Regime Comparison: Pre-AI vs AI Era")

def plot_factor_loading_evolution(df_load):
    key_factors = [c for c in ['Beta', 'Momentum', 'Growth', 'Semiconductors', 'Volatility'] if c in df_load.columns]
    palette = [C_GREEN, C_BLUE, C_GOLD, C_RED, '#c084fc']
    fig = go.Figure()
    for i, f in enumerate(key_factors):
        fig.add_trace(go.Scatter(
            x=df_load.index, y=df_load[f], mode='lines',
            name=f, line=dict(color=palette[i % len(palette)], width=2)))
    fig.add_hline(y=0, line_color=C_MUTED, line_width=1)
    fig.update_layout(
        xaxis_title="Date", yaxis_title="Factor Loading",
        legend=dict(orientation='h', y=1.1, font=dict(color=C_TEXT)))
    return apply_theme(fig, 420, "Factor Loading Evolution (2022-2025)")

def plot_rolling_alpha_quality(df, event_df, window=63):
    idio = df['Idio_Return'].dropna()
    roll_sharpe = idio.rolling(window).apply(lambda x: sharpe(x, ann=252), raw=True)
    fill_color = 'rgba(118,185,0,0.10)' if IS_DARK else 'rgba(118,185,0,0.12)'
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=roll_sharpe.index, y=roll_sharpe, mode='lines',
        line=dict(color=C_GREEN, width=2),
        name=f'Rolling {window}d Alpha Sharpe',
        fill='tozeroy', fillcolor=fill_color))
    fig.add_hline(y=0, line_color=C_MUTED, line_width=1.5)
    for edate in event_df['Earnings_Date'].unique():
        fig.add_vline(x=edate, line_dash='dot', line_color=C_GOLD, line_width=1)
    fig.update_layout(xaxis_title='Date', yaxis_title=f'Rolling Sharpe ({window}d)')
    return apply_theme(fig, 380, f'Rolling Idiosyncratic Alpha Quality ({window}-Day Window)')

def plot_scatter_pre_vs_react(event_df, pre_days=5):
    data = []
    for evt in event_df['Event_ID'].unique():
        sub = event_df[event_df['Event_ID'] == evt]
        pre   = (1 + sub[(sub['Rel_Time'] >= -pre_days) & (sub['Rel_Time'] <= -1)]['Total_Return']).prod() - 1
        react = sub[sub['Rel_Time'] == 0]['Total_Return'].values
        if len(react) == 0:
            continue
        data.append({'Event': evt, 'Pre': pre * 100, 'React': react[0] * 100})
    df_s = pd.DataFrame(data)
    if df_s.empty:
        return go.Figure()
    corr = df_s['Pre'].corr(df_s['React'])
    fig = px.scatter(df_s, x='Pre', y='React', text='Event',
        trendline='ols', color_discrete_sequence=[C_GREEN])
    fig.update_traces(
        textposition='top center', marker=dict(size=11),
        textfont=dict(color=C_TEXT, size=10))
    fig.add_hline(y=0, line_dash='dot', line_color=C_MUTED, line_width=1.5)
    fig.add_vline(x=0, line_dash='dot', line_color=C_MUTED, line_width=1.5)
    fig.update_layout(
        xaxis_title=f"Pre-Earnings Drift ({pre_days}d, %)",
        yaxis_title="Earnings Day Reaction (%)")
    return apply_theme(fig, 420, f"Information Leakage: Pre-Drift vs Reaction (r={corr:.2f})")

def main():
    # ── Theme toggle button (top-right, fixed position) ──
    st.markdown(f"""
    <style>
    div[data-testid="stButton"].theme-toggle-btn > button {{
        position: fixed !important;
        top: 14px !important;
        right: 20px !important;
        z-index: 9999 !important;
        background: {C_CARD} !important;
        color: {C_TEXT} !important;
        border: 1.5px solid {C_BORDER} !important;
        border-radius: 20px !important;
        padding: 4px 14px !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        cursor: pointer !important;
        box-shadow: {CARD_SHADOW} !important;
        transition: all 0.2s ease !important;
    }}
    div[data-testid="stButton"].theme-toggle-btn > button:hover {{
        background: {C_GREEN} !important;
        color: #000 !important;
        border-color: {C_GREEN} !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    col_head, col_btn = st.columns([10, 1])
    with col_btn:
        btn_label = "☀️ " if IS_DARK else "🌙 "
        st.markdown('<div data-testid="stButton" class="theme-toggle-btn">', unsafe_allow_html=True)
        if st.button(btn_label, key="theme_toggle"):
            st.session_state.app_theme = 'Light' if IS_DARK else 'Dark'
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style='display:flex; align-items:center; gap:16px; margin-bottom:20px; padding: 10px 0;'>
      <div style='background:#76b900; border-radius:8px; padding:8px 16px; 
           font-weight:800; font-size:22px; color:#000; letter-spacing:1px;'>NVDA</div>
      <div style='flex:1;'>
        <div style='font-size:28px; font-weight:700; color:{C_TEXT}; line-height:1.2;'>
          NVDA Earnings Intelligence Dashboard
        </div>
        <div style='font-size:13px; color:{C_MUTED}; margin-top:4px; line-height:1.4;'>
          NVIDIA Corporation · Quantitative Event Study Analysis · Made by Sunil Bhatia
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class='guide-box'>
    This dashboard analyzes NVDA's stock behavior around earnings announcements using factor models and event study methodology.
    Each tab provides different analytical perspectives. Use the navigation below to explore. 
    How NVDA reacts to earnings, what drives those moves (company-specific alpha vs market factors), 
    and how the AI era changed the game. All insights are backed by statistical analysis of actual trading data.
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading data..."):
        df, contrib, earnings_dates, factor_cols, df_load = load_data()
        event_df = build_event_windows(df, earnings_dates, window=40)
        if event_df.empty:
            st.error("Could not build event windows.")
            return
        stats_df = event_stats(event_df)
    
    t0 = event_df[event_df['Rel_Time'] == 0].copy()
    t0_ret = t0['Total_Return'] * 100
    t0_idio = t0['Idio_Return'] * 100
    t0_factor = t0['Factor_Return'] * 100

    ai_start = pd.Timestamp("2023-05-24")
    
    t0_pre_ai = t0[t0['Earnings_Date'] < ai_start]['Total_Return'] * 100
    t0_ai_era = t0[t0['Earnings_Date'] >= ai_start]['Total_Return'] * 100

    # --- TAB NAVIGATION WORKAROUND
    tab_names = [
        "Executive Brief",
        "Alpha Decomposition",
        "Regime Analysis",
        "Event Study",
        "Factor Evolution",
        "Trade Analytics",
        "Statistical Proof"
    ]
    st.markdown('<div class="nvda-nav-wrap">', unsafe_allow_html=True)
    active_tab = st.radio("Navigation", tab_names, horizontal=True, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    #
    # TAB 0 - EXECUTIVE BRIEF
    #
    if active_tab == "Executive Brief":
        st.markdown("## Key Findings Summary")
        
        st.markdown(f"""
        <div class='guide-box'>
        High-level metrics that summarize NVDA's earnings behavior. 
        The six KPIs below capture average moves, volatility, win rates, and the dominance of company-specific (alpha) factors.
        <b>Look for:</b> Pre-AI and AI Era moves, and the high "Alpha Dominance" percentage 
        showing that NVDA's earnings reactions are driven by company news, not market movements.
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Avg Earnings Move", f"{t0_ret.mean():.1f}%", f"σ = {t0_ret.std():.1f}%")
        c2.metric("Median Move", f"{t0_ret.median():.1f}%", f"Range: {t0_ret.min():.1f}% / {t0_ret.max():.1f}%")
        c3.metric("Win Rate (T=0)", f"{(t0_ret > 0).mean():.0%}", f"{int((t0_ret > 0).sum())}W / {int((t0_ret <= 0).sum())}L")
        c4.metric("Alpha Dominance", f"{(t0_idio.abs() > t0_factor.abs()).mean():.0%}", "Idio > Factor freq.")
        c5.metric("AI Era Avg Move", f"{t0_ai_era.mean():.1f}%")
        c6.metric("Earnings Day Vol", f"{t0_ret.std() * np.sqrt(252):.0f}%", "Annualized")

        st.markdown(f"""
        <div class='metric-explain'>
        
        <b>Avg Earnings Move:</b> The mean return on earnings day (T=0) across all 12 events. σ shows standard deviation.<br>
        <b>Median Move:</b> The middle value - less affected by outliers. Range shows best and worst single-day reactions.<br>
        <b>Win Rate:</b> Percentage of earnings days that closed positive. W/L shows wins vs losses (calculated from all events in dataset).<br>
        <b>Alpha Dominance:</b> How often idiosyncratic (company-specific) factors outweigh systematic (market/sector) factors.<br>
        <b>AI Era Avg Move:</b> Average earnings-day return since May 2023 (the AI inflection point).<br>
        <b>Earnings Day Vol:</b> Volatility on earnings day, annualized for comparison to typical annual vol.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        col_l, col_r = st.columns([3, 2])
        with col_l:
            st.markdown("## Earnings Day Returns - All Events")
            st.markdown(f"""
            <div class='guide-box'>
            Each bar is one earnings announcement. Green = positive return, Red = negative.
            The gold dashed line shows the average. <b>Notice:</b> The AI Era quarters (starting Q3 May 2023) show much larger moves.
            The DeepSeek shock (Q10 Jan 2025) is the extreme outlier at -17%.
            </div>
            """, unsafe_allow_html=True)
            
            colors_bar = [C_GREEN if v > 0 else C_RED for v in t0_ret.values]
            fig_bar = go.Figure(go.Bar(
                x=t0['Event_ID'].values, y=t0_ret.values,
                marker_color=colors_bar,
                text=[f"{v:+.1f}%" for v in t0_ret.values],
                textposition='auto',
                textfont=dict(color='white', size=11, family='Inter, sans-serif'),
                insidetextanchor='middle',
            ))
            fig_bar.add_hline(y=t0_ret.mean(), line_dash='dash',
                line_color=C_GOLD, annotation_text=f"Mean {t0_ret.mean():.1f}%",
                annotation_font_color=C_GOLD)
            fig_bar.update_layout(xaxis_title="", yaxis_title="Return (%)",
                                  showlegend=False)
            st.plotly_chart(apply_theme(fig_bar, 380), use_container_width=True)
        
        with col_r:
            st.markdown("## Alpha vs Beta Split (T=0)")
            st.markdown(f"""
            <div class='guide-box'>
            "Idiosyncratic Alpha" is the part of the return unique to NVDA (company news, earnings surprise).
            "Factor Beta" is the part explained by market/sector movements.  Alpha dominates - NVDA moves 
            are primarily about company-specific news, not just riding the market.
            </div>
            """, unsafe_allow_html=True)
            
            fig_ab = go.Figure()
            fig_ab.add_trace(go.Bar(
                x=['Idiosyncratic Alpha', 'Factor Beta'],
                y=[t0_idio.abs().mean(), t0_factor.abs().mean()],
                marker_color=[C_GREEN, C_BLUE],
                text=[f"{t0_idio.abs().mean():.2f}%", f"{t0_factor.abs().mean():.2f}%"],
                textposition='inside',
                textfont=dict(color='white', size=14, family='Inter, sans-serif'),
                insidetextanchor='middle',
            ))
            fig_ab.update_layout(showlegend=False, yaxis_title="Avg |Return| (%)")
            st.plotly_chart(apply_theme(fig_ab, 280), use_container_width=True)
            
            st.markdown(f"""
            <div class='insight-box'>
            Idiosyncratic alpha drives <b>{(t0_idio.abs() > t0_factor.abs()).mean():.0%}</b> 
            of earnings moves, confirming NVDA's reaction is primarily company-specific, not macro-driven. 
            The AI era (post May-2023) amplified average moves from 
            <b>{t0_pre_ai.mean():.1f}%</b> to <b>{t0_ai_era.mean():.1f}%</b>.
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("## Event Scorecard")
        
        st.markdown(f"""
        <div class='guide-box'>
        This table shows every earnings event with its outcome type (Beat/Miss/Shock), 
        the total return on announcement day (T=0), and how that return splits into Alpha (company-specific) and Beta (market-driven).
        <b>Pre-5d Drift:</b> Movement in the 5 days BEFORE earnings (potential information leakage).
        <b>Post-5d Drift:</b> Movement in the 5 days AFTER earnings (continuation or reversal).
      
        </div>
        """, unsafe_allow_html=True)
        
        best_evt = t0.loc[t0['Total_Return'].idxmax(), 'Event_ID']
        worst_evt = t0.loc[t0['Total_Return'].idxmin(), 'Event_ID']
        best_ret = t0['Total_Return'].max() * 100
        worst_ret = t0['Total_Return'].min() * 100
        st.markdown(f"""
        <div class='insight-box'>
        Across {len(t0)} earnings events, NVDA delivered a positive reaction
        <b>{int((t0_ret > 0).sum())}</b> out of {len(t0)} times ({(t0_ret > 0).mean():.0%} win rate).
        The strongest single-day reaction was <b>{best_evt}</b> at <b>+{best_ret:.1f}%</b>.
        The worst was <b>{worst_evt}</b> at <b>{worst_ret:.1f}%</b> - driven by the DeepSeek shock (Jan 2025),
        the largest single-day market cap loss in US equity history at the time.
        Pre-5d drift is consistently positive ahead of beats, suggesting informed positioning before announcements.
        </div>
        """, unsafe_allow_html=True)

        rows = []
        for evt in t0['Event_ID'].unique():
            row_t0 = t0[t0['Event_ID'] == evt].iloc[0]
            edate_str = row_t0['Earnings_Date'].strftime('%Y-%m-%d')
            known = KNOWN_EVENTS.get(edate_str, ("-", "-"))
            pre5  = event_df[(event_df['Event_ID'] == evt) & 
                             (event_df['Rel_Time'] >= -5) & (event_df['Rel_Time'] <= -1)]['Total_Return'].sum() * 100
            post5 = event_df[(event_df['Event_ID'] == evt) &
                             (event_df['Rel_Time'] >= 1) & (event_df['Rel_Time'] <= 5)]['Total_Return'].sum() * 100
            rows.append({
                'Event': evt,
                'Date': edate_str,
                'Type': known[0],
                'T=0 Total': row_t0['Total_Return'],
                'T=0 Alpha': row_t0['Idio_Return'],
                'T=0 Beta': row_t0['Factor_Return'],
                'Pre-5d Drift': pre5 / 100,
                'Post-5d Drift': post5 / 100,
                'Context': known[1]
            })
        scorecard = pd.DataFrame(rows)
        st.dataframe(
            scorecard.style
            .format({'T=0 Total': '{:.2%}', 'T=0 Alpha': '{:.2%}',
                     'T=0 Beta': '{:.2%}', 'Pre-5d Drift': '{:.2%}', 'Post-5d Drift': '{:.2%}'})
            .background_gradient(subset=['T=0 Total'], cmap='RdYlGn', vmin=-0.15, vmax=0.15)
            .background_gradient(subset=['T=0 Alpha'], cmap='RdYlGn', vmin=-0.15, vmax=0.15)
            .set_table_styles(_DF_TABLE_STYLES),
            use_container_width=True, height=420
        )

    
    # TAB 1 - ALPHA DECOMPOSITION
    
    elif active_tab == "Alpha Decomposition":
        st.markdown("## Return Attribution Breakdown")

        st.markdown(f"""
        <div class='guide-box'>
        Return attribution breaks down NVDA's total return into its component sources

        1. <b>Select Event:</b> Choose a specific earnings quarter or "Average" to see the typical pattern<br>
        2. <b>Set Window:</b> Choose the time period (e.g., T=0 for just earnings day, or T-5:+5 for the full 11-day window)<br>
        3. <b>Read the Waterfall:</b> Green bars add to return, red bars subtract. The final bar shows the total.<br>
        4. <b>Interpret Alpha Purity:</b> Higher % means the move was more company-specific, less driven by market factors.
        </div>
        """, unsafe_allow_html=True)

        q_options = ["Average"] + list(event_df['Event_ID'].unique())
        col_sel, col_win = st.columns([2, 3])
        q_sel = col_sel.selectbox("Select Event", q_options)
        w_range = col_win.slider("Attribution Window (Days Relative to Earnings)", -10, 10, (0, 0))

        if q_sel == "Average":
            mask = (event_df['Rel_Time'] >= w_range[0]) & (event_df['Rel_Time'] <= w_range[1])
            sub  = event_df[mask]
            ev_tot  = (1 + sub.groupby('Event_ID')['Total_Return'].sum()).mean() - 1
            ev_idio = (1 + sub.groupby('Event_ID')['Idio_Return'].sum()).mean() - 1
            c_sub   = contrib.loc[sub.index].copy()
            c_sub['Event_ID'] = sub['Event_ID']
            avg_cont = c_sub.groupby('Event_ID').sum().mean()
        else:
            mask = (event_df['Event_ID'] == q_sel) & \
                   (event_df['Rel_Time'] >= w_range[0]) & (event_df['Rel_Time'] <= w_range[1])
            sub  = event_df[mask]
            ev_tot  = (1 + sub['Total_Return']).prod() - 1 if not sub.empty else 0
            ev_idio = (1 + sub['Idio_Return']).prod() - 1 if not sub.empty else 0
            avg_cont = contrib.loc[sub.index].sum() if not sub.empty else pd.Series(dtype=float)

        top3 = avg_cont.abs().nlargest(3).index.tolist() if not avg_cont.empty else []
        other = ev_tot - ev_idio - (avg_cont[top3].sum() if top3 else 0)
        labels = ["Idio Alpha"] + top3 + ["Other Factors", "Total"]
        values = [ev_idio * 100] + \
                 ([avg_cont[f] * 100 for f in top3] if top3 else []) + \
                 [other * 100, ev_tot * 100]
        
        title_wf = f"Attribution: {q_sel} | T{w_range[0]:+d} to T{w_range[1]:+d}"
        st.plotly_chart(plot_waterfall(labels, values, title_wf), use_container_width=True)
        
        st.markdown(f"""
        <div class='guide-box'>
        <b>Start Point:</b> Imagine starting from zero on the left<br>
        <b>Green Bars (Increasing):</b> These add value - they move the cumulative total UP. Idio Alpha is typically green and large.<br>
        <b>Red Bars (Decreasing):</b> These subtract value - they move the cumulative total DOWN. Sometimes factors like "Volatility" can be negative drags.<br>
        <b>Blue "Total" Bar:</b> The final result - where you end up after all components<br>       
        <b>What "Idio Alpha" Usually Being Largest Means:</b> NVDA's earnings reactions are driven by company-specific news (guidance, product updates, demand commentary) rather than just riding the broader market or sector trends.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        col_dist, col_metrics = st.columns([3, 2])
        
        with col_dist:
            st.markdown("## Return Distribution Analysis")
            
            st.markdown(f"""
            <div class='guide-box'>
            <b>What This Distribution Shows:</b><br>
            <b>Gray Histogram:</b> Shows the frequency distribution of ALL daily returns over the 2022-2025 period. Most days cluster near zero (normal trading).<br>
            <b>Colored Vertical Lines at Bottom:</b> Each line represents one earnings day. Green = positive return, Red = negative return.<br>
            <b>Key Observation:</b> Earnings days cluster in the EXTREME TAILS - they are outlier events far beyond normal daily volatility.<br>
            Earnings are the dominant source of extreme moves in NVDA. Normal days do not produce these tail events.
            </div>
            """, unsafe_allow_html=True)
            
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=df['Total_Return'] * 100, nbinsx=80, histnorm='probability density',
                name='All Days', marker_color='rgba(136,146,164,0.4)', showlegend=True))
            fig_hist.add_trace(go.Scatter(
                x=t0_ret.values, y=np.zeros(len(t0_ret)),
                mode='markers', name='Earnings Days',
                marker=dict(color=[C_GREEN if v > 0 else C_RED for v in t0_ret],
                            size=14, symbol='line-ns-open', line=dict(width=3))))
            fig_hist.update_layout(xaxis_title="Daily Return (%)", yaxis_title="Density",
                                   legend=dict(orientation='h', y=1.1))
            st.plotly_chart(apply_theme(fig_hist, 320), use_container_width=True)
        
        with col_metrics:
            purity = ev_idio / ev_tot if ev_tot != 0 else 0
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Return", f"{ev_tot:.2%}")
            c2.metric("Alpha Return", f"{ev_idio:.2%}")
            c3.metric("Alpha Purity", f"{purity:.0%}")
            
            st.markdown(f"""
            <div class='metric-explain'>
            <b>Metrics Explained:</b><br>
            <b>Total Return:</b> The full move in the selected window (all sources combined)<br>
            <b>Alpha Return:</b> Only the idiosyncratic (company-specific) component<br>
            <b>Alpha Purity:</b> What % of total is pure alpha vs factor beta<br>
            <i>Higher Alpha Purity (closer to 100%) = more stock-picking opportunity, less market/sector dependency</i>
            </div>
            """, unsafe_allow_html=True)
            
            tail_pct = (df['Total_Return'] * 100).quantile(0.99)
            earn_above_tail = (t0_ret > tail_pct).sum()
            st.markdown(f"""
            <div class='insight-box'>
            <b>Distribution Insight:</b> NVDA's daily returns have fat tails - earnings days cluster in the extreme tails far beyond normal trading. 
            <b>{earn_above_tail} of {len(t0_ret)} earnings events</b> exceeded the 99th percentile of all daily returns (<b>{tail_pct:.1f}%</b>), 
            confirming earnings are the primary source of extreme outcomes.
            </div>
            """, unsafe_allow_html=True)
            
        alpha_purity_avg = (t0_idio.abs() / (t0_idio.abs() + t0_factor.abs())).mean()
        st.markdown(f"""
        <div class='insight-box'>
        <b>Attribution Summary:</b> On average, idiosyncratic alpha accounts for <b>{alpha_purity_avg:.0%}</b> of the absolute move on earnings day, 
        with factor beta explaining the remainder. This confirms NVDA's earnings reactions are overwhelmingly company-specific - 
        stock selection matters more than market timing for NVDA earnings trades.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("## Per-Quarter Attribution Matrix")
        
        st.markdown(f"""
        <div class='guide-box'>
        <b>Understanding the Attribution Matrix:</b><br>        
        
        <b>Event:</b> The quarter identifier (e.g., Q3 May 2023 = The AI Inflection Quarter)<br>
        <b>Window:</b> Which time period around earnings (Pre = before, Reaction = T=0, Post = after, Full = combined)<br>
        <b>Total:</b> The complete return including all sources<br>
        <b>Alpha:</b> Only the company-specific (idiosyncratic) component<br>
        <b>Beta:</b> Only the factor-driven (market/sector/style) component<br>        
        1. <b>Look for Pattern Shifts:</b> Did alpha accumulate Pre-earnings (information leakage) or Post-earnings (drift)?<br>
        2. <b>Compare Total vs Alpha:</b> When they are similar = high alpha purity. When they diverge = factors matter more.<br>
        3. <b>Identify Anomalies:</b> Quarters where Beta is unusually large = market/sector-driven moves coinciding with earnings.
        </div>
        """, unsafe_allow_html=True)

        all_edates = sorted(event_df['Earnings_Date'].unique())
        date_labels = [pd.Timestamp(d).strftime('%b %Y') for d in all_edates]
        qm_col1, qm_col2 = st.columns(2)
        qm_start_idx = qm_col1.selectbox("From Quarter", range(len(date_labels)),
            format_func=lambda i: date_labels[i], index=0, key='qm_start')
        qm_end_idx = qm_col2.selectbox("To Quarter", range(len(date_labels)),
            format_func=lambda i: date_labels[i], index=len(date_labels)-1, key='qm_end')
        if qm_end_idx < qm_start_idx:
            qm_end_idx = qm_start_idx
        qm_date_start = pd.Timestamp(all_edates[qm_start_idx])
        qm_date_end   = pd.Timestamp(all_edates[qm_end_idx])
        qm_events = event_df[(event_df['Earnings_Date'] >= qm_date_start) & 
                             (event_df['Earnings_Date'] <= qm_date_end)]

        q_rows = []
        for evt in qm_events['Event_ID'].unique():
            for win_name, (ws, we) in [('Pre (T-5:-1)',(-5,-1)),('Reaction (T=0)',(0,0)),
                                       ('Post (T+1:+5)',(1,5)),('Window (T-5:+5)',(-5,5))]:
                sub_q = qm_events[(qm_events['Event_ID']==evt) & 
                                  (qm_events['Rel_Time']>=ws) & (qm_events['Rel_Time']<=we)]
                if sub_q.empty:
                    continue
                q_rows.append({
                    'Event': evt, 'Window': win_name,
                    'Total': (1+sub_q['Total_Return']).prod()-1,
                    'Alpha': (1+sub_q['Idio_Return']).prod()-1,
                    'Beta':  (1+sub_q['Factor_Return']).prod()-1,
                })
        q_matrix = pd.DataFrame(q_rows)
        st.dataframe(
            q_matrix.style.format({'Total':'{:.2%}','Alpha':'{:.2%}','Beta':'{:.2%}'})
            .background_gradient(subset=['Total','Alpha'], cmap='RdYlGn', vmin=-0.12, vmax=0.12)
            .set_table_styles(_DF_TABLE_STYLES),
            use_container_width=True, height=380
        )

        st.markdown("---")
        st.markdown("## Cumulative Returns: Total vs Alpha vs Beta")
        
        st.markdown(f"""
        <div class='guide-box'>
        <b>Reading the Cumulative Decomposition Chart:</b><br>
        
        <b>Gold Line (Total Return):</b> What you would have earned buying and holding NVDA from Sept 2022 to present<br>
        <b>Green Line (Idiosyncratic Alpha):</b> What you would have earned from ONLY the company-specific component (if you could isolate it)<br>
        <b>Blue Line (Factor Beta):</b> What you would have earned from ONLY the systematic factor exposures<br>
        <b>Vertical Gold Dashes:</b> Mark each earnings announcement<br>        
        <b>Key Pattern to Notice:</b><br>
        <b>Step Changes at Earnings Lines:</b> The green alpha line jumps significantly at each gold vertical line - this proves that earnings are the key alpha generation events<br>
        <b>Between Earnings:</b> The lines drift more smoothly - normal factor exposures and gradual alpha accumulation<br>
        <b>Divergence Magnitude:</b> When green line >> blue line = alpha dominates. When they are similar = factors matter as much as stock selection.
        </div>
        """, unsafe_allow_html=True)
        
        cum_all = (1 + df[['Total_Return','Factor_Return','Idio_Return']]).cumprod() - 1
        fig_cum = go.Figure()
        for col, color, name in [('Total_Return', C_GOLD, 'Total Return'),
                                 ('Idio_Return', C_GREEN, 'Idiosyncratic Alpha'),
                                 ('Factor_Return', C_BLUE, 'Factor Beta')]:
            fig_cum.add_trace(go.Scatter(x=cum_all.index, y=cum_all[col]*100,
                mode='lines', name=name, line=dict(color=color, width=2)))
        for edate in earnings_dates:
            fig_cum.add_vline(x=edate, line_dash='dot', line_color=C_GOLD, line_width=1)
        fig_cum.update_layout(xaxis_title="Date", yaxis_title="Cumulative Return (%)",
                              legend=dict(orientation='h', y=1.08))
        st.plotly_chart(apply_theme(fig_cum, 420, "Full-Period Cumulative Return Decomposition"),
                        use_container_width=True)
        cum_total = cum_all['Total_Return'].iloc[-1] * 100
        cum_idio  = cum_all['Idio_Return'].iloc[-1] * 100
        cum_beta  = cum_all['Factor_Return'].iloc[-1] * 100
        st.markdown(f"""
        <div class='insight-box'>
        <b>Full-Period Decomposition:</b> Over the full 2022-2025 period, NVDA's total cumulative return was <b>{cum_total:+.0f}%</b>. 
        Of this, idiosyncratic alpha contributed <b>{cum_idio:+.0f}%</b> and factor beta contributed <b>{cum_beta:+.0f}%</b>.
        {'Idiosyncratic alpha is the dominant driver - NVDA is not just a market/sector ride, it is a genuine stock-picker opportunity. The step-function nature of the green line at earnings dates confirms that these catalysts are the primary alpha generation moments.' if abs(cum_idio) > abs(cum_beta) else 'Factor beta dominates - NVDA returns are largely explained by market and sector exposure rather than company-specific alpha.'}
        </div>
        """, unsafe_allow_html=True)

    
    # TAB 2 - REGIME ANALYSIS
    
    elif active_tab == "Regime Analysis":
        st.markdown("## Pre-AI vs AI Era Regime Shift")
        
        st.markdown(f"""
        <div class='guide-box'>
           
        <b>Pre-AI Era (Nov 2022 - Feb 2023):</b><br>
        Gaming/Crypto collapse aftermath: Nvidia earlier business largely focussed on gaming graphics<br>
        Channel inventory glut<br>
        Traditional cyclical semiconductor behavior<br>
        Smaller, more volatile moves<br><br>        
        <b>AI Era (May 2023 - Present):</b><br>
        AI data center demand explosion<br>
        NVDA as critical AI infrastructure<br>
        Larger magnitude moves<br>
        Sustained post-earnings drift (new phenomenon)<br>
        Higher win rate and lower dispersion<br>        
                </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class='insight-box'>
        <b>Regime Inflection Point:</b> The May 24, 2023 earnings call ("The Guidance Quarter") marked a structural shift in NVDA's dynamics. 
        Management raised guidance by ~50%, shocking consensus and signaling unprecedented AI demand. This transformed NVDA from a cyclical 
        semiconductor into a critical AI infrastructure play with sustained pricing power and visibility.
        </div>
        """, unsafe_allow_html=True)

        all_events = list(event_df['Event_ID'].unique())
        regime_options = ["All Events", "Pre-AI Era (before May 2023)", "AI Era (May 2023+)"]
        fc1, fc2 = st.columns([2, 3])
        regime_filter = fc1.selectbox("Regime Filter", regime_options)
        quarter_filter = fc2.multiselect("Drill into specific quarter(s)", all_events,
            placeholder="Leave empty = use regime filter above")

        if quarter_filter:
            filtered_events = event_df[event_df['Event_ID'].isin(quarter_filter)]
            filter_label = ", ".join(quarter_filter)
        elif regime_filter == "Pre-AI Era (before May 2023)":
            filtered_events = event_df[event_df['Earnings_Date'] < ai_start]
            filter_label = "Pre-AI Era"
        elif regime_filter == "AI Era (May 2023+)":
            filtered_events = event_df[event_df['Earnings_Date'] >= ai_start]
            filter_label = "AI Era"
        else:
            filtered_events = event_df
            filter_label = "All Events"

        ai_start_t0 = t0[t0['Earnings_Date'] >= ai_start]
        pre_ai_t0   = t0[t0['Earnings_Date'] <  ai_start]
        ai_win  = (ai_start_t0['Total_Return'] > 0).mean() if len(ai_start_t0) > 0 else 0
        pre_win = (pre_ai_t0['Total_Return'] > 0).mean() if len(pre_ai_t0) > 0 else 0
        
                


        st.markdown("---")
        st.markdown("## Cumulative Alpha: Pre-AI vs AI Era Overlay")
        
        st.markdown(f"""
        <div class='guide-box'>
        <b>Visualizing the Regime Shift:</b> This chart overlays all earnings events. 
        <span style='color:#00a8e0; font-weight:bold;'>Blue lines</span> represent the Pre-AI era, while 
        <span style='color:#76b900; font-weight:bold;'>Green lines</span> represent the AI era. 
        Notice how the AI era paths cluster tighter and consistently drift higher after T=0.
        </div>
        """, unsafe_allow_html=True)
        
        # Build the combined chart
        pivot = event_df.pivot(index='Rel_Time', columns='Event_ID', values='Idio_Return')
        cum = pivot.fillna(0).cumsum() * 100 
        
        pre_cols = [c for c in cum.columns if pd.to_datetime(c.split('(')[1][:-1], format='%b %Y') < ai_start]
        ai_cols = [c for c in cum.columns if c not in pre_cols]
        
        fig_comb = go.Figure()
        
        # Plot individual event lines
        for col in pre_cols:
            fig_comb.add_trace(go.Scatter(x=cum.index, y=cum[col], mode='lines',
                line=dict(color='rgba(0,168,224,0.3)', width=1.5), name=col, showlegend=False, hoverinfo='skip'))
        for col in ai_cols:
            fig_comb.add_trace(go.Scatter(x=cum.index, y=cum[col], mode='lines',
                line=dict(color='rgba(118,185,0,0.3)', width=1.5), name=col, showlegend=False, hoverinfo='skip'))
                
        # Plot means
        if pre_cols:
            fig_comb.add_trace(go.Scatter(x=cum.index, y=cum[pre_cols].mean(axis=1), mode='lines',
                line=dict(color=C_BLUE, width=3.5, dash='dot'), name='Pre-AI Mean Path'))
        if ai_cols:
            fig_comb.add_trace(go.Scatter(x=cum.index, y=cum[ai_cols].mean(axis=1), mode='lines',
                line=dict(color=C_GREEN, width=3.5), name='AI Era Mean Path'))
                
        fig_comb.add_vline(x=0, line_dash="dash", line_color=C_GOLD, annotation_text="Earnings Day")
        fig_comb.add_hline(y=0, line_color=C_MUTED, line_width=1.2)
        fig_comb.update_layout(xaxis_title="Days Relative to Earnings", yaxis_title="Cumulative Alpha (%)",
                               legend=dict(orientation="h", y=1.08))
                               
        st.plotly_chart(apply_theme(fig_comb, 500, "Overlay: Idiosyncratic Return Paths"), use_container_width=True)

        st.markdown("---")
        st.markdown(f"## Filtered Event Study - {filter_label}")
        
        st.markdown(f"""
        <div class='guide-box'>
        
        Select the Toggle the metric selector below to view Idiosyncratic (alpha), Total, or Factor (beta) return paths
                </div>
        """, unsafe_allow_html=True)
        
        if not filtered_events.empty:
            metric_f = st.radio("Metric", ['Idio_Return','Total_Return','Factor_Return'],
                horizontal=True, key='regime_metric',
                format_func=lambda x: x.replace('_Return',''))
            st.plotly_chart(plot_cumulative_paths(filtered_events, metric_f,
                f"Cumulative {metric_f.replace('_Return','')} - {filter_label}"), 
                use_container_width=True)
        else:
            st.info("No events match the selected filter.")

        st.markdown("---")
        st.markdown("## Rolling Alpha Quality")
        
        st.markdown(f"""
        <div class='guide-box'>

        <b>Sharpe Ratio:</b> Risk-adjusted return = (Mean Return) / (Standard Deviation of Returns). Higher = better return per unit of risk taken.<br>
        <b>Rolling Window:</b> Instead of calculating one Sharpe for the whole period, we calculate it over sliding windows (e.g., 63 trading days = 3 months).<br>
        <b>This Chart Shows:</b> How the quality (Sharpe) of idiosyncratic alpha evolved over time.<br>
        <b>Vertical Gold Lines:</b> Mark earnings events.<br>        
        
        <b>Positive Sharpe:</b> Alpha was profitable and consistent in that window. Good risk-reward.<br>
        <b>Negative Sharpe:</b> Alpha was either unprofitable or too volatile relative to returns. Bad risk-reward.<br>
        <b>Spikes Around Earnings:</b> Alpha quality jumps at earnings and often stays elevated in the AI era.<br>
        <b>Sustained Elevation:</b> AI era shows longer periods of positive Sharpe - alpha is more persistent and tradeable.
        </div>
        """, unsafe_allow_html=True)
        
        roll_w = st.slider("Rolling Window (trading days)", 10, 126, 63, 5,
            key='roll_sharpe_slider')
        st.plotly_chart(plot_rolling_alpha_quality(df, event_df, roll_w), use_container_width=True)
        idio_sharpe_full = sharpe(df['Idio_Return'].dropna().values)
        st.markdown(f"""
        <div class='insight-box'>
        <b>Alpha Quality Over Time:</b> The full-period idiosyncratic Sharpe ratio is <b>{idio_sharpe_full:.2f}</b>. 
        Periods of sustained positive rolling Sharpe correspond directly to AI-era quarters where NVDA's company-specific alpha 
        was both profitable AND consistent - the holy grail for active managers. Pre-AI era shows more volatility and lower average Sharpe.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("## Information Leakage Analysis")
        
        st.markdown(f"""
        <div class='guide-box'>
            

        <b>X-axis (Pre-Earnings Drift):</b> The cumulative return in the N days BEFORE earnings (default: 5 days, T-5 to T-1)<br>
        <b>Y-axis (Earnings Day Reaction):</b> The return on earnings day itself (T=0)<br>
        <b>Each Point:</b> One earnings quarter, labeled with the quarter ID<br>
        <b>Trendline:</b> Shows the relationship (correlation) between pre-drift and reaction<br>
        <b>r = Correlation Coefficient:</b> Shown in the title. Ranges from -1 to +1.<br>        
        <b>Four Quadrants:</b><br>
        <b>Top-Right (++ Quadrant):</b> Positive pre-drift, positive reaction = <b>INFORMED BUYING</b>. Stock ran up BEFORE earnings, 
          then beat expectations. Classic leakage pattern. Institutions were positioning ahead of good news.<br>
        <b>Top-Left (-+ Quadrant):</b> Negative pre-drift, positive reaction = <b>NEGATIVE LEAKAGE REVERSED</b>. Stock sold off before, 
          but earnings beat expectations. Either leakage was wrong, or the selloff created entry opportunity.<br>
        <b>Bottom-Right (+- Quadrant):</b> Positive pre-drift, negative reaction = <b>BUY THE RUMOR, SELL THE NEWS</b>. 
          Stock ran up before, then disappointed. Either leaked info was wrong, or guidance disappointed despite a beat.<br>
        <b>Bottom-Left (-- Quadrant):</b> Negative pre-drift, negative reaction = <b>INFORMED SELLING</b>. Stock sold off before 
          and then missed. Negative leakage confirmed.<br>        
        <b>What Correlation Means:</b><br>
        <b>Positive Correlation (r > 0):</b> Pre-drift and reaction move together. High leakage - pre-move contains signal about earnings.<br>
        <b>Negative Correlation (r < 0):</b> Pre-drift and reaction move opposite. Contrarian setup - pre-move is noise or false signal.<br>
        <b>r close to 0:</b> No relationship - pre-drift does not predict reaction.<br>
        <b>r > +0.5:</b> Strong positive relationship - significant leakage/informed positioning.<br>
        <b>r < -0.5:</b> Strong negative relationship - contrarian pattern.
        </div>
        """, unsafe_allow_html=True)
        
        leak_days = st.slider("Pre-earnings drift window (days)", 1, 20, 5, key='leak_slider')
        st.plotly_chart(plot_scatter_pre_vs_react(filtered_events, leak_days), use_container_width=True)
        
        # Fixed: Changed from event_df to filtered_events to match the regime filter selection
        # Previously calculated avg_pre5 from all events even when user selected a regime filter
        pre5_all = []
        for evt in filtered_events['Event_ID'].unique():
            sub_l = filtered_events[(filtered_events['Event_ID']==evt) & (filtered_events['Rel_Time'].between(-5,-1))]
            if not sub_l.empty:
                pre5_all.append((1+sub_l['Total_Return']).prod()-1)
        avg_pre5 = np.mean(pre5_all) * 100 if pre5_all else 0
        
        data_for_corr = []
        for evt in filtered_events['Event_ID'].unique():
            sub = filtered_events[filtered_events['Event_ID'] == evt]
            pre   = (1 + sub[(sub['Rel_Time'] >= -leak_days) & (sub['Rel_Time'] <= -1)]['Total_Return']).prod() - 1
            react = sub[sub['Rel_Time'] == 0]['Total_Return'].values
            if len(react) > 0:
                data_for_corr.append({'Pre': pre * 100, 'React': react[0] * 100})
        
        if len(data_for_corr) > 1:
            df_corr = pd.DataFrame(data_for_corr)
            actual_corr = df_corr['Pre'].corr(df_corr['React'])
            
            st.markdown(f"""
            <div class='insight-box'>
            <b>Leakage Statistics:</b><br>
            <b>Average {leak_days}-day pre-drift:</b> <b>{avg_pre5:.1f}%</b> (positive = systematic buying before announcements)<br>
            <b>Correlation coefficient:</b> <b>{actual_corr:.2f}</b><br>
            <b>Interpretation:</b> {
                'Strong positive correlation - significant information leakage. Pre-earnings drift is highly predictive of the reaction. Informed participants are positioning ahead of news.' if actual_corr > 0.5
                else 'Moderate positive correlation - some information leakage present. Pre-drift contains signal but not overwhelmingly predictive.' if actual_corr > 0.2
                else 'Weak positive correlation - limited leakage. Pre-drift has mild predictive value.' if actual_corr > 0
                else 'Negative correlation - contrarian pattern. Pre-drift moves opposite to reaction. Could indicate false signals or profit-taking setups.' if actual_corr < -0.2
                else 'Near-zero correlation - pre-drift does not predict reaction. Minimal information leakage.'
            }<br>
            <b>Dominant Quadrant:</b> Most points cluster in {
                'Top-Right (informed buying pattern)' if df_corr['Pre'].median() > 0 and df_corr['React'].median() > 0
                else 'Bottom-Left (informed selling pattern)' if df_corr['Pre'].median() < 0 and df_corr['React'].median() < 0
                else 'mixed quadrants (no dominant pattern)'
            }.
            </div>
            """, unsafe_allow_html=True)

    
    # TAB 3 - EVENT STUDY
    
    elif active_tab == "Event Study":
        st.markdown("## Event Study: Cumulative Return Paths")
        
        st.markdown(f"""
        <div class='guide-box'>      
        <b>Methodology:</b> We align all earnings events to a common timeline where T=0 = earnings announcement day, 
        then average the paths to reveal systematic patterns that repeat across quarters.<br>        
        <b>Pre-Run Patterns:</b> Do stocks systematically move before announcements? (information leakage or sentiment build)<br>
        <b>Reaction Magnitude:</b> How large is the typical T=0 jump?<br>
        <b>Post-Drift:</b> Do moves continue after the announcement or reverse?<br>
        <b>Consistency:</b> Tight bands = reliable patterns, wide bands = high uncertainty<br>        
        <b>Metric Selector:</b> Choose Idio (pure alpha), Total (everything), or Factor (beta) to see which component drives the pattern<br>
        <b>Window Slider:</b> Adjust how many days before/after earnings to include (wider = more context, narrower = focus on event)
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class='insight-box'>
        <b>Event Study Key Finding:</b> The mean cumulative idiosyncratic return path shows a clear <b>pre-earnings drift</b> 
        in the 5 days before announcement (T-5 to T-1), followed by a sharp discontinuity at T=0 (the earnings reaction), 
        then continued upward drift post-earnings in AI-era events (T+1 to T+5). This three-phase pattern (pre-run, pop, drift) 
        is systematic and tradeable.
        </div>
        """, unsafe_allow_html=True)
        
        es_col1, es_col2 = st.columns([2, 1])
        metric_sel = es_col1.radio("Metric", ['Idio_Return','Total_Return','Factor_Return'],
                                   horizontal=True, format_func=lambda x: x.replace('_Return',''))
        es_window = es_col2.slider("Window (+-days)", 1, 40, 40, key='es_window')
        es_df = event_df[event_df['Rel_Time'].between(-es_window, es_window)]
        st.plotly_chart(plot_cumulative_paths(es_df, metric_sel,
            f"Cumulative {metric_sel.replace('_Return','')} Return Around Earnings (+-{es_window} Days)"),
            use_container_width=True)

        st.markdown("---")
        col_v, col_s = st.columns(2)
        with col_v:
            st.markdown(f"""
            <div class='guide-box'>
                       
            Annualized volatility (standard deviation of returns) across different windows around earnings.<br>            
            <b>Red Bar at T=0:</b> Volatility spike on earnings day itself - this is what options traders price as the "expected move"<br>
            <b>Gold Bars (T-5:-1 and T+1:+5):</b> Elevated volatility in the 5 days before/after earnings<br>
            <b>Blue Bars (Further Windows):</b> Normal volatility far from earnings<br>            
            <br>
            <b>Options Pricing:</b> ATM straddles price in the T=0 volatility spike. Knowing the typical realized move vs implied move creates trading opportunities.<br>
            <b>Position Sizing:</b> Higher vol windows require smaller positions to maintain constant risk.<br>
            <b>Risk Management:</b> Volatility does not just spike at T=0 - it is elevated T-5 to T+5, so risk persists before and after the announcement.
            </div>
            """, unsafe_allow_html=True)
            
            st.plotly_chart(plot_vol_term_structure(event_df), use_container_width=True)
            t0_vol = event_df[event_df['Rel_Time']==0]['Total_Return'].std()*np.sqrt(252)*100
            st.markdown(f"""
            <div class='insight-box'>
            <b>Volatility Statistics:</b> Annualized volatility spikes to <b>{t0_vol:.0f}%</b> on earnings day (T=0) - 
            this is what the options market implied move is attempting to price. Realized moves often exceed or fall short of 
            this expectation, creating straddle P&L. Volatility remains elevated in adjacent T-5:-1 and T+1:+5 windows at 
            ~70-80% annualized, showing uncertainty persists before and after the announcement.
            </div>
            """, unsafe_allow_html=True)
        
        with col_s:
            st.markdown("## Shock Persistence (T=0, T+1, T+2)")
            
            st.markdown(f"""
            <div class='guide-box'>
            
            
            <b>Red Bars (T=0):</b> Return on earnings day itself<br>
            <b>Blue Bars (T+1):</b> Return the next trading day<br>
            <b>Gold Bars (T+2):</b> Return two days after earnings<br>            
            
            <b>Same Sign Continuation (e.g., +5%, +2%, +1%):</b> The initial shock continues/persists. Market is still processing and accumulating.<br>
            <b>Reversal (e.g., +5%, -2%, -1%):</b> Profit-taking or mean reversion after the initial pop.<br>
            <b>Amplification (e.g., +5%, +7%, +3%):</b> Momentum builds after earnings - often seen in AI era mega beats.<br>            
            <b>AI Era Pattern:</b> Notice that AI-era events (Q3 onwards) often show follow-through on T+1 and T+2 when T=0 is positive - 
            this is the post-earnings drift phenomenon. Institutions continue buying as they process the demand commentary.
            </div>
            """, unsafe_allow_html=True)
            
            t0s = event_df[event_df['Rel_Time']==0].set_index('Event_ID')['Total_Return']*100
            t1s = event_df[event_df['Rel_Time']==1].set_index('Event_ID')['Total_Return']*100
            t2s = event_df[event_df['Rel_Time']==2].set_index('Event_ID')['Total_Return']*100
            fig_shock = go.Figure()
            fig_shock.add_trace(go.Bar(x=t0s.index, y=t0s, name='T=0', marker_color=C_RED))
            fig_shock.add_trace(go.Bar(x=t1s.index, y=t1s, name='T+1', marker_color=C_BLUE))
            fig_shock.add_trace(go.Bar(x=t2s.index, y=t2s, name='T+2', marker_color=C_GOLD))
            fig_shock.update_layout(barmode='group', xaxis_title="", yaxis_title="Return (%)")
            st.plotly_chart(apply_theme(fig_shock, 380, ""), use_container_width=True)

    
    # TAB 4 - FACTOR EVOLUTION
    
    elif active_tab == "Factor Evolution":
        st.markdown("## Factor Risk Profile Evolution")
        
        st.markdown(f"""
        <div class='guide-box'>
          
        
         <b>Beta Compression:</b> Beta loading dropped from ~2.3x (highly volatile vs market) to ~1.4x (more stable)<br>
         <b>Growth Surge:</b> Growth loading jumped post-May 2023 as NVDA became the AI growth story<br>
         <b>Volatility Normalization:</b> Volatility loading shifted from deeply negative to near-zero (less vol-sensitive)
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class='insight-box'>
        <b>Factor Evolution Story:</b> NVDA's factor profile fundamentally shifted in 2023-2024. 
        Beta compressed from ~2.3x to ~1.4x, Growth loading surged, and Volatility sensitivity normalized. 
        This reflects NVDA's transformation from a volatile cyclical semiconductor (2022: crypto/gaming bust, inventory issues) 
        to a stable AI infrastructure play (from 2023: predictable data center demand, pricing power, multi-year visibility).
        </div>
        """, unsafe_allow_html=True)

        st.plotly_chart(plot_factor_loading_evolution(df_load), use_container_width=True)

        col_pie, col_radar = st.columns(2)
        with col_pie:
            st.markdown("## Risk Decomposition (Full Period)")
            
            st.markdown(f"""
            <div class='guide-box'>

             <b>Market Drivers (Blue):</b> Systematic market risk - Beta, general market moves
             <b>Semiconductors (Green):</b> Sector-specific risk - semiconductor industry movements
             <b>Style Factors (Gold):</b> Growth/Value/Momentum/Quality tilts
             <b>Idiosyncratic (Red):</b> Company-specific risk - unique to NVDA<br>
             <b>High Idiosyncratic % (>40%):</b> Stock is driven by company news, not market/sector. Stock-picker's opportunity.<br>
             <b>High Market % (>40%):</b> Stock is a "market beta play" - moves with indexes. Less alpha opportunity.<br>
             <b>High Sector % (>30%):</b> Industry-driven - use sector ETFs instead of individual stock.<br>
            
    
            </div>
            """, unsafe_allow_html=True)
            
            vols = {
                'Market Drivers': df['Grp_Market'].var(),
                'Semiconductors': df['Grp_Semi'].var(),
                'Style Factors':  df['Grp_Style'].var(),
                'Idiosyncratic':  df['Idio_Return'].var(),
            }
            fig_pie = go.Figure(go.Pie(
                labels=list(vols.keys()), values=list(vols.values()),
                marker_colors=[C_BLUE, C_GREEN, C_GOLD, C_RED],
                hole=0.45, textinfo='label+percent'))
            fig_pie.update_layout(showlegend=False)
            st.plotly_chart(apply_theme(fig_pie, 380, "Variance Decomposition"), use_container_width=True)
            total_var = sum(vols.values())
            idio_share = vols['Idiosyncratic'] / total_var if total_var > 0 else 0
            st.markdown(f"""
            <div class='insight-box'>
            <b>Risk Decomposition Result:</b> Idiosyncratic variance accounts for <b>{idio_share:.0%}</b> of NVDA's total return variance 
            over the full 2022-2025 period. This unusually high idiosyncratic share (typical stocks are 20-30%) confirms NVDA is a 
            <b>stock-picker's stock</b> where returns are driven by company-specific events (earnings, product launches, guidance changes) 
            far more than by market or sector movements. Active management and fundamental research add value.
            </div>
            """, unsafe_allow_html=True)

        with col_radar:
            st.markdown("## Average Factor Fingerprint at Earnings")
            
            st.markdown(f"""
            <div class='guide-box'>
             <b>Each Spoke:</b> A different style factor (Value, Growth, Momentum, etc.)<br>
             <b>Distance from Center:</b> Magnitude of the loading. Further out = higher exposure.<br>
             <b>Direction Does Not Matter Here:</b> Only magnitude - we show absolute values or typical values.<br><br>
             <b>Strong Growth:</b> NVDA is deeply exposed to growth factor - it is a growth stock<br>
             <b>Strong Momentum:</b> Tends to have positive momentum exposure (trending stocks)<br>
             <b>Strong Quality:</b> High profitability and quality metrics<br>
             <b>Low Value:</b> Not a value play - trades at premium multiples<br>
             <b>Moderate Size/MidCap:</b> Despite being mega-cap, has some mid-cap characteristics in behavior
            </div>
            """, unsafe_allow_html=True)
            
            style_cols_r = [c for c in ['Value','Growth','Momentum','Quality','Size',
                                        'Volatility','Liquidity','Leverage'] if c in df_load.columns]
            avg_loads = []
            for edate in earnings_dates:
                future = df_load.index[df_load.index >= edate]
                if len(future) > 0:
                    row = df_load.loc[future[0]]
                    avg_loads.append(row[style_cols_r].values)
            if avg_loads:
                avg_vec = np.mean(avg_loads, axis=0)
                fig_radar = go.Figure(go.Scatterpolar(
                    r=avg_vec, theta=style_cols_r, fill='toself',
                    line=dict(color=C_GREEN, width=2), fillcolor='rgba(118,185,0,0.2)'))
                fig_radar.update_layout(polar=dict(
                    radialaxis=dict(visible=True, tickfont=dict(color=C_MUTED), gridcolor=C_GRID),
                    angularaxis=dict(tickfont=dict(color=C_TEXT), gridcolor=C_GRID),
                    bgcolor=C_CARD))
                st.plotly_chart(apply_theme(fig_radar, 380, ""), use_container_width=True)

        st.markdown("---")
        st.markdown("## Factor Contribution Heatmap (By Quarter)")
        
        st.markdown(f"""
        <div class='guide-box'>
         <b>Identify Driver Quarters:</b> Which factors dominated specific quarters? (e.g., "May 2023 was all about Growth factor")<br>
         <b>Spot Regime Shifts:</b> When did factor importance change? (e.g., Volatility contribution flipped from negative to neutral)<br>
         <b>Correlation Hunting:</b> Do certain factors always move together? Or offset each other?<br>
         <b>Attribution Forensics:</b> "Why did Q10 Jan 2025 tank -17%?" -> Look at which factors were most negative.
        </div>
        """, unsafe_allow_html=True)
        
        hm_rows = []
        for evt in event_df['Event_ID'].unique():
            sub_hm = event_df[(event_df['Event_ID']==evt) & (event_df['Rel_Time']==0)]
            if sub_hm.empty:
                continue
            row_dict = {'Event': evt}
            for f in factor_cols[:10]:
                row_dict[f] = contrib.loc[sub_hm.index, f].sum() * 100 if f in contrib.columns else 0
            hm_rows.append(row_dict)
        if hm_rows:
            hm_df = pd.DataFrame(hm_rows).set_index('Event')
            
            fig_hm_q = px.imshow(hm_df.T, color_continuous_scale='RdYlGn',
                aspect='auto', origin='lower',
                color_continuous_midpoint=0)
            
            for i, factor in enumerate(hm_df.T.index):
                for j, event in enumerate(hm_df.T.columns):
                    val = hm_df.T.iloc[i, j]
                    text_color = '#000000' if abs(val) < 3 else '#ffffff'
                    fig_hm_q.add_annotation(
                        x=event, y=factor,
                        text=f'{val:.2f}',
                        showarrow=False,
                        font=dict(color=text_color, size=10)
                    )
            
            fig_hm_q.update_xaxes(side='top')
            fig_hm_q.update_layout(coloraxis_showscale=True)
            st.plotly_chart(apply_theme(fig_hm_q, 420, "Factor Contributions at T=0 (%)"),
                use_container_width=True)

    
    # TAB 5 - TRADE ANALYTICS
    
    elif active_tab == "Trade Analytics":
        st.markdown("## Trade Analytics: Backtesting Event-Driven Strategies")
        
        st.markdown(f"""
        <div class='guide-box'>
       
         <b>Win Rate:</b> % of trades that were profitable (closed positive). 70%+ is exceptional.<br>
         <b>Avg Win:</b> Average return on winning trades. Bigger = more upside when right.<br>
         <b>Avg Loss:</b> Average return on losing trades (shown as negative). Smaller absolute value = controlled downside.<br>
         <b>Risk/Reward Ratio:</b> |Avg Win| / |Avg Loss|. >2.0 = great asymmetry (wins are 2x larger than losses).<br>
         <b>Kelly %:</b> Optimal position size according to Kelly Criterion. Assumes reinvestment. >10% = strong edge.<br>
         <b>Sharpe Ratio:</b> Risk-adjusted return. >1.0 = good, >2.0 = excellent. Measures return per unit of volatility.<br>
         <b>N:</b> Number of trades (sample size). <5 = small sample, take with caution.    
        <b>Goal:</b> Identify which timing window (Pre-Run, Reaction Day, Post-Drift, etc.) offers the best risk-adjusted returns.
        </div>
        """, unsafe_allow_html=True)

        ta_options = ["All Events", "Pre-AI Era", "AI Era"]
        ta_filter = st.selectbox("Filter Events", ta_options, key='ta_filter')
        if ta_filter == "Pre-AI Era":
            ta_event_df = event_df[event_df['Earnings_Date'] < ai_start]
        elif ta_filter == "AI Era":
            ta_event_df = event_df[event_df['Earnings_Date'] >= ai_start]
        else:
            ta_event_df = event_df

        st.markdown("## Pre-Defined Strategy Performance")
        
        

        strategies = {
            'Pre-Run Play (T-5:-1)': (-5, -1),
            'Earnings Day Only (T=0)': (0, 0),
            'Overnight Hold (T=0:+1)': (0, 1),
            'Post-Drift Ride (T+1:+5)': (1, 5),
            'Full Window (T-5:+5)': (-5, 5),
        }
        wr_rows = []
        for name, (ent, ext) in strategies.items():
            sub = ta_event_df[(ta_event_df['Rel_Time'] >= ent) & (ta_event_df['Rel_Time'] <= ext)]
            grouped = sub.groupby('Event_ID')['Total_Return'].apply(lambda x: (1+x).prod()-1)
            wins = grouped[grouped > 0]
            losses = grouped[grouped < 0]
            wr    = len(wins) / len(grouped) if len(grouped) > 0 else 0
            aw    = wins.mean()    if len(wins)    > 0 else 0
            al    = losses.mean()  if len(losses)  > 0 else 0
            rr    = abs(aw / al)   if al != 0 else 0
            k     = kelly(wr, aw, al)
            sh    = sharpe(grouped.values)
            wr_rows.append({
                'Strategy': name,
                'Win Rate': wr, 'Avg Win': aw, 'Avg Loss': al,
                'Risk/Reward': rr, 'Kelly %': k, 'Sharpe': sh,
                'N': len(grouped)
            })
        wr_df = pd.DataFrame(wr_rows)
        st.dataframe(
            wr_df.style.format({
                'Win Rate': '{:.0%}', 'Avg Win': '{:.2%}', 'Avg Loss': '{:.2%}',
                'Risk/Reward': '{:.2f}', 'Kelly %': '{:.1%}', 'Sharpe': '{:.2f}'
            }).background_gradient(subset=['Win Rate', 'Sharpe'], cmap='RdYlGn')
            .set_table_styles(_DF_TABLE_STYLES),
            use_container_width=True
        )
        
        st.markdown("---")
        st.markdown("## Strategy Entry/Exit Heatmap")
        
        st.markdown(f"""
        <div class='guide-box'>
                
        <b>Structure:</b><br>
         <b>Rows (Y-axis):</b> Entry days relative to earnings (when to buy). Negative = days before earnings.<br>
         <b>Columns (X-axis):</b> Exit days relative to earnings (when to sell). Positive = days after earnings.                 
        <b><br>Color Interpretation:</b><br>
         <b>Green = Good:</b> For most metrics (Avg Return, Win Rate, Sharpe). Greener = better performance.<br>
         <b>Red = Bad:</b> For most metrics. Means poor performance for that timing combination.<br>
         <b>Exception - Worst Trade %:</b> Uses inverted scale (green = less bad).<br>
        

        1. <b>Set Window Range:</b> Choose how far before/after earnings to explore (e.g., -5 to +5 for typical event window)<br>
        2. <b>Select Metric:</b> Choose what to optimize for (return, win rate, risk-adjusted, worst-case)<br>
        3. <b>Find Hot Spots:</b> Look for clusters of green cells - these are the best timing combinations<br>
        4. <b>Avoid Cold Spots:</b> Red clusters = timing windows to avoid<br>
        5. <b>Compare Diagonals:</b> Cells along same diagonal = same holding period (e.g., all 3-day holds). Compare across diagonals to see if longer holds help or hurt.<br>
        
        </div>
        """, unsafe_allow_html=True)
        
        hm_col1, hm_col2, hm_col3 = st.columns(3)
        hm_min = hm_col1.number_input("Min Day (Entry)", -20, 0, -5, key="hm_min")
        hm_max = hm_col2.number_input("Max Day (Exit)", 0, 20, 5, key="hm_max")
        hm_metric = hm_col3.selectbox("Metric", ["Average Return (%)", "Win Rate (%)", "Worst Trade (%)", "Sharpe Ratio"], key="hm_metric")

        days = list(range(hm_min, hm_max + 1))
        hm_data = pd.DataFrame(index=days, columns=days, dtype=float)

        for ent in days:
            for ext in days:
                if ext <= ent:
                    hm_data.loc[ent, ext] = np.nan
                else:
                    rets = []
                    for q in ta_event_df['Event_ID'].unique():
                        sub = ta_event_df[(ta_event_df['Event_ID']==q) & 
                                          (ta_event_df['Rel_Time']>=ent) & 
                                          (ta_event_df['Rel_Time']<=ext)]
                        if not sub.empty:
                            ret = (1 + sub['Total_Return']).prod() - 1
                            rets.append(ret)
                    
                    if rets:
                        rets = np.array(rets)
                        if hm_metric == "Average Return (%)": val = rets.mean() * 100
                        elif hm_metric == "Win Rate (%)": val = (rets > 0).mean() * 100
                        elif hm_metric == "Worst Trade (%)": val = rets.min() * 100
                        elif hm_metric == "Sharpe Ratio": val = sharpe(rets)
                        hm_data.loc[ent, ext] = val

        fig_hm = px.imshow(
            hm_data, 
            aspect="auto",
            labels=dict(x="Exit Day (T)", y="Entry Day (T)", color=hm_metric),
            color_continuous_scale="RdYlGn" if hm_metric != "Worst Trade (%)" else "RdYlGn_r",
            origin="upper",
            color_continuous_midpoint=0 if hm_metric in ["Average Return (%)", "Sharpe Ratio"] else None
        )
        
        for i, ent_day in enumerate(hm_data.index):
            for j, ext_day in enumerate(hm_data.columns):
                val = hm_data.iloc[i, j]
                if not pd.isna(val):
                    if hm_metric == "Average Return (%)":
                        text_color = '#000000' if abs(val) < 5 else '#ffffff'
                    elif hm_metric == "Win Rate (%)":
                        text_color = '#000000' if abs(val - 50) < 15 else '#ffffff'
                    elif hm_metric == "Worst Trade (%)":
                        text_color = '#000000' if val > -10 else '#ffffff'
                    else:
                        text_color = '#000000' if abs(val) < 1 else '#ffffff'
                    
                    fig_hm.add_annotation(
                        x=ext_day, y=ent_day,
                        text=f'{val:.2f}',
                        showarrow=False,
                        font=dict(color=text_color, size=9)
                    )
        
        fig_hm.update_xaxes(side="top", tickmode='linear', dtick=1)
        fig_hm.update_yaxes(tickmode='linear', dtick=1)
        fig_hm = apply_theme(fig_hm, 500, f"Strategy Matrix: {hm_metric}")
        st.plotly_chart(fig_hm, use_container_width=True)

        st.markdown("---")
        st.markdown("## Equity Curve Simulator")
        
        st.markdown(f"""
        <div class='guide-box'>
        
        <b>Purpose:</b> Visualize what would have happened if you traded a specific strategy on every earnings event.<br>
        

        1. <b>Set Parameters:</b> Choose entry day (e.g., T-1), exit day (e.g., T+1), and starting capital<br>
        2. <b>Simulate:</b> We apply that strategy to each earnings quarter sequentially<br>
        3. <b>Compound:</b> Gains/losses compound - your position size grows or shrinks with equity<br>
        4. <b>Plot:</b> The equity curve shows your portfolio value over time<br>
        
         <b>Total Return:</b> Final equity / starting capital - 1. Your overall gain/loss %.<br>
         <b>Strategy Sharpe:</b> Risk-adjusted return quality. Accounts for volatility of the strategy.<br>
         <b>Max Drawdown:</b> Worst peak-to-trough loss. -30% = you lost 30% from the highest point. Critical for risk management.<br>
         <b>Win Rate:</b> % of profitable trades. Comes with W/L count.<br>
        

         <b>Green Line/Markers:</b> Equity curve path. Green markers = winning trade that quarter, Red markers = losing trade.<br>
         <b>Fill:</b> Area under curve visualizes total capital growth<br>
         <b>Dashed Horizontal Line:</b> Starting capital reference<br>
         <b>X-axis Labels:</b> Quarter identifiers (Q1, Q2, ...)<br>
        
         <b>Upward Sloping:</b> Consistent growth over time<br>
         <b>Smooth:</b> Not too jagged - predictable strategy<br>
         <b>Shallow Drawdowns:</b> Losses are controlled<br>
         <b>Recovery:</b> Bounces back quickly after losses<br>
        
        <b>Red Flags:</b><br>
         <b>Downward Sloping:</b> Strategy loses money over time<br>
         <b>Extreme Volatility:</b> Huge swings up and down - unsustainable<br>
         <b>Deep Drawdowns:</b> >-40% means one bad stretch wipes out months of gains<br>
         <b>Late Period Collapse:</b> If curve peaks early then bleeds = regime shift or overfitting
        </div>
        """, unsafe_allow_html=True)
        
        col_s1, col_s2, col_s3 = st.columns(3)
        ent = col_s1.number_input("Entry Day (T)", -10, 10, -1)
        ext = col_s2.number_input("Exit Day (T)", -10, 10, 1)
        cap = col_s3.number_input("Starting Capital ($)", 1000, 1_000_000, 100_000, step=10_000)

        trades, eq, curve, qlabels = [], cap, [cap], ["Start"]
        for q in ta_event_df['Event_ID'].unique():
            sub_q = ta_event_df[(ta_event_df['Event_ID']==q) & 
                                (ta_event_df['Rel_Time']>=ent) & (ta_event_df['Rel_Time']<=ext)]
            if sub_q.empty:
                continue
            ret = (1 + sub_q['Total_Return']).prod() - 1
            eq *= (1 + ret)
            trades.append({'Quarter': q, 'Return': ret, 'Equity': eq,
                           'PnL': eq - curve[-1]})
            curve.append(eq)
            qlabels.append(q.split(' ')[0])
        
        if trades:
            res = pd.DataFrame(trades)
            total_ret = eq / cap - 1
            sh_strat  = sharpe(res['Return'].values)
            mdd       = max_drawdown(np.array(curve))
            win_n     = (res['Return'] > 0).sum()
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Return", f"{total_ret:.1%}")
            m2.metric("Strategy Sharpe", f"{sh_strat:.2f}")
            m3.metric("Max Drawdown", f"{mdd:.1%}")
            m4.metric("Win Rate", f"{win_n/len(res):.0%}", f"{win_n}W/{len(res)-win_n}L")
            
            st.markdown(f"""
            <div class='metric-explain'>
            <b>Understanding Your Results:</b><br>
             <b>Total Return {total_ret:.1%}:</b> Your ${cap:,.0f} {'grew' if total_ret > 0 else 'shrank'} to ${eq:,.0f} over {len(res)} trades.<br>
             <b>Sharpe {sh_strat:.2f}:</b> {'Excellent risk-adjusted returns (>2.0)' if sh_strat > 2 else 'Good risk-adjusted returns (>1.0)' if sh_strat > 1 else 'Moderate risk-adjusted returns (>0.5)' if sh_strat > 0.5 else 'Poor risk-adjusted returns or negative' if sh_strat > 0 else 'Strategy lost money'}.<br>
             <b>Max Drawdown {mdd:.1%}:</b> Your worst loss from peak was {abs(mdd):.1%}. {'Very controlled' if mdd > -0.15 else 'Moderate' if mdd > -0.30 else 'Significant - risk management needed'}.<br>
             <b>Win Rate {win_n/len(res):.0%}:</b> You won {win_n} out of {len(res)} trades. {'Exceptional' if win_n/len(res) > 0.7 else 'Good' if win_n/len(res) > 0.6 else 'Fair' if win_n/len(res) > 0.5 else 'Below 50% - need higher winners or better asymmetry'}.
            </div>
            """, unsafe_allow_html=True)

            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(x=qlabels, y=curve, mode='lines+markers',
                line=dict(color=C_GREEN, width=2.5),
                marker=dict(size=8, color=[C_GREEN if i == 0 else
                    (C_GREEN if curve[i] >= curve[i-1] else C_RED) for i in range(len(curve))]),
                fill='tozeroy', fillcolor='rgba(118,185,0,0.08)', name='Equity'))
            fig_eq.add_hline(y=cap, line_dash='dash', line_color=C_MUTED,
                annotation_text="Starting Capital")
            fig_eq.update_layout(xaxis_title="Quarter", yaxis_title="Portfolio Value ($)")
            st.plotly_chart(apply_theme(fig_eq, 380, f"Equity Curve: Entry T{ent:+d} -> Exit T{ext:+d}"),
                use_container_width=True)
        else:
            st.info("No trades executed with the selected parameters. Try widening the entry/exit day range.")

    
    # TAB 6 - STATISTICAL PROOF
    
    elif active_tab == "Statistical Proof":
        st.markdown("## Statistical Significance Tests")
        
        st.markdown(f"""
        <div class='guide-box'>
        
        
        Are the patterns we observe (pre-drift, earnings reaction, post-drift) real and repeatable, 
        or could they just be random noise?<br>
        
        <b>The Statistical Test We Use:</b> One-sample t-test<br>
         <b>Null Hypothesis (H0):</b> Mean return in this window = 0 (no systematic pattern, just randomness)<br>
         <b>Alternative Hypothesis (H1):</b> Mean return in this window does not equal 0 (there IS a real pattern)<br>
         <b>Test Statistic (t-stat):</b> Measures how many standard deviations away from zero the mean is. Larger absolute value = stronger evidence.<br>
         <b>P-value:</b> Probability that we would observe this data if the null hypothesis (no pattern) were true. Lower = stronger evidence.<br>

         <b>p < 0.01:</b> Extremely strong evidence. Less than 1% chance this is random. "Highly significant."<br>
         <b>p < 0.05:</b> Strong evidence. Less than 5% chance this is random. "Statistically significant." Standard threshold.<br>
         <b>0.05 <= p < 0.10:</b> Weak evidence. "Marginal significance." Suggests a pattern but not definitive.<br>
         <b>p >= 0.10:</b> Insufficient evidence. Could be random. "Not significant."<br>        

         <b>|t| > 3.0:</b> Very strong signal (equivalent to p < 0.01 typically)<br>
         <b>|t| > 2.0:</b> Strong signal (equivalent to p < 0.05 typically)<br>
         <b>|t| > 1.5:</b> Moderate signal (marginal significance)<br>
         <b>|t| < 1.5:</b> Weak signal (likely random)<br>        
        <b>Why This Matters:</b> Without statistical testing, we cannot distinguish luck from skill, noise from signal. 
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class='insight-box'>
        <b>Methodology:</b> We perform one-sample t-tests against H0: mean return = 0 for multiple windows around earnings 
        (Pre-Run, Reaction, Overnight, Post-Drift, Full Window) and for multiple components (Total, Alpha, Beta). 
        P-values below 0.05 indicate statistically significant non-zero returns - proof of a systematic, tradeable pattern. 
        The table below shows results for all window/component combinations. <b>Lower p-values (darker green) = stronger evidence.</b>
        </div>
        """, unsafe_allow_html=True)

        test_rows = []
        for ws, we, label in [(-5,-1,'Pre-Run T-5:-1'),(0,0,'Reaction T=0'),
                              (0,1,'Overnight T=0:+1'),(1,5,'Post-Drift T+1:+5'),
                              (-5,5,'Full Window T-5:+5')]:
            sub = event_df[(event_df['Rel_Time']>=ws) & (event_df['Rel_Time']<=we)]
            for col, cname in [('Total_Return','Total'),('Idio_Return','Alpha'),('Factor_Return','Beta')]:
                per_event = sub.groupby('Event_ID')[col].apply(lambda x: (1+x).prod()-1)
                n = len(per_event)
                if n < 2:
                    continue
                t_stat, p_val = stats.ttest_1samp(per_event, 0)
                test_rows.append({
                    'Window': label, 'Component': cname,
                    'N Events': n, 'Mean': per_event.mean(),
                    'Std Dev': per_event.std(), 't-stat': t_stat, 'p-value': p_val,
                    'Significant': 'Yes' if p_val < 0.05 else ('Marginal' if p_val < 0.10 else 'No')
                })
        test_df = pd.DataFrame(test_rows)
        
        st.markdown(f"""
        <div class='guide-box'>

         <b>Window:</b> Time period around earnings being tested (e.g., Pre-Run T-5:-1 = 5 days before to 1 day before)<br>
         <b>Component:</b> Which return source - Total (all sources), Alpha (idiosyncratic), Beta (factor-driven)<br>
         <b>N Events:</b> Sample size - number of earnings events in the test. Larger N = more reliable results.<br>
         <b>Mean:</b> Average return across all events in that window/component. This is what we are testing if it is significantly different from zero.<br>
         <b>Std Dev:</b> Standard deviation of returns. Higher = more variable outcomes.<br>
         <b>t-stat:</b> Test statistic. Larger absolute value = stronger evidence. Sign indicates direction (positive = gains, negative = losses).<br>
         <b>p-value:</b> THE KEY METRIC. Probability the pattern is random. Lower = stronger proof it is real.<br>
         <b>Significant:</b> Quick summary - Yes (p<0.05), Marginal (0.05<=p<0.10), No (p>=0.10)<br><br>        
  
         <b>Low p-values (< 0.05) in Reaction T=0:</b> Proves earnings moves are real and significant<br>
         <b>Low p-values in Pre-Run T-5:</b> Proves information leakage is systematic<br>
         <b>Low p-values in Alpha component:</b> Proves company-specific factors drive the patterns<br>
         <b>High p-values in Beta component:</b> Suggests market factors DO NOT explain the patterns (it is idiosyncratic)
        </div>
        """, unsafe_allow_html=True)
        
        st.dataframe(
            test_df.style.format({'Mean':'{:.3%}','Std Dev':'{:.3%}','t-stat':'{:.2f}','p-value':'{:.4f}'})
            .background_gradient(subset=['p-value'], cmap='RdYlGn_r', vmin=0, vmax=0.15)
            .set_table_styles(_DF_TABLE_STYLES),
            use_container_width=True, height=500
        )
        if not test_df.empty:
            sig_count = (test_df['p-value'] < 0.05).sum()
            total_tests = len(test_df)
            very_sig = (test_df['p-value'] < 0.01).sum()
            
            sig_results = test_df[test_df['p-value'] < 0.05]
            sig_windows = sig_results['Window'].unique()
            sig_alpha = (sig_results['Component'] == 'Alpha').sum()
            
            
if __name__ == "__main__":
    main()