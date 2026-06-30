"""
styles.py
Shared CSS injection for the earthy/organic visual identity
used across all pages of the Plant Disease Classifier app.
"""

import streamlit as st


CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --forest: #2D3A2E;
    --parchment: #F5F1E8;
    --parchment-deep: #EDE6D6;
    --terracotta: #8B5E3C;
    --terracotta-light: #A87653;
    --sage: #6B8E4E;
    --sage-light: #8AAE6B;
    --wheat: #C9A876;
    --disease-red: #A33D2C;
    --healthy-green: #4A7A3C;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ---- Global background texture ---- */
.stApp {
    background-color: var(--parchment);
    background-image:
        radial-gradient(circle at 1px 1px, rgba(139,94,60,0.06) 1px, transparent 0);
    background-size: 24px 24px;
}

/* ---- Headings use Fraunces ---- */
h1, h2, h3 {
    font-family: 'Fraunces', serif !important;
    color: var(--forest) !important;
    letter-spacing: -0.01em;
}

h1 {
    font-weight: 600 !important;
    font-size: 2.6rem !important;
    border-bottom: 3px solid var(--wheat);
    padding-bottom: 0.4rem;
    margin-bottom: 1.2rem !important;
}

h2 {
    font-weight: 600 !important;
    font-size: 1.6rem !important;
}

h3 {
    font-weight: 500 !important;
    color: var(--terracotta) !important;
}

/* ---- Eyebrow label style ---- */
.field-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--sage);
    border: 1px solid var(--sage);
    display: inline-block;
    padding: 2px 10px;
    border-radius: 2px;
    margin-bottom: 0.6rem;
    background: rgba(107,142,78,0.06);
}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
    background-color: var(--forest);
    border-right: 1px solid var(--wheat);
}

section[data-testid="stSidebar"] * {
    color: var(--parchment) !important;
}

section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stCheckbox label {
    color: var(--parchment) !important;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

section[data-testid="stSidebar"] hr {
    border-color: rgba(245,241,232,0.2);   
}

/* ---- Sidebar selectbox box ---- */
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: var(--parchment) !important;
    border: 1px solid var(--wheat) !important;
    border-radius: 4px !important;
}

/* Force all text inside the selectbox value area to be dark */
section[data-testid="stSidebar"] [data-baseweb="select"] * {
    color: var(--forest) !important;
    -webkit-text-fill-color: var(--forest) !important;
}

/* ---- Dropdown options popup (rendered outside sidebar, at body level) ---- */
[data-baseweb="popover"] {
    background-color: var(--parchment) !important;
}

[data-baseweb="popover"] li {
    color: var(--forest) !important;
    background-color: var(--parchment) !important;
}

[data-baseweb="popover"] li:hover {
    background-color: var(--parchment-deep) !important;
}

/* ---- Buttons ---- */
.stButton > button, .stDownloadButton > button {
    background-color: var(--terracotta) !important;
    color: var(--parchment) !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    letter-spacing: 0.02em;
    padding: 0.5rem 1.4rem !important;
    transition: background-color 0.2s ease;
}

.stButton > button:hover, .stDownloadButton > button:hover {
    background-color: var(--terracotta-light) !important;
}

/* ---- File uploader ---- */
[data-testid="stFileUploader"] section {
    background-color: var(--parchment-deep);
    border: 2px dashed var(--wheat) !important;
    border-radius: 4px;
}

/* ---- Cards (custom container) ---- */
.field-card {
    background-color: #FFFEFA;
    border: 1px solid var(--wheat);
    border-left: 4px solid var(--sage);
    border-radius: 3px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}

.field-card.disease {
    border-left-color: var(--disease-red);
}

.field-card.healthy {
    border-left-color: var(--healthy-green);
}

/* ---- Stat block ---- */
.stat-block {
    text-align: left;
    padding: 0.8rem 0;
}

.stat-number {
    font-family: 'Fraunces', serif;
    font-size: 2.4rem;
    font-weight: 600;
    color: var(--terracotta);
    line-height: 1;
}

.stat-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--forest);
    opacity: 0.65;
    margin-top: 0.3rem;
}

/* ---- Divider ---- */
.field-divider {
    border: none;
    border-top: 1px dashed var(--wheat);
    margin: 1.6rem 0;
}

/* ---- Metric override ---- */
[data-testid="stMetric"] {
    background-color: #FFFEFA;
    border: 1px solid var(--wheat);
    border-radius: 3px;
    padding: 0.8rem 1rem;
}

[data-testid="stMetricLabel"] {
    font-family: 'JetBrains Mono', monospace !important;
    text-transform: uppercase;
    font-size: 0.7rem !important;
    letter-spacing: 0.06em;
}

[data-testid="stMetricValue"] {
    font-family: 'Fraunces', serif !important;
    color: var(--terracotta) !important;
}

/* ---- Progress bar ---- */
.stProgress > div > div {
    background-color: var(--sage) !important;
}

/* ---- Table styling ---- */
[data-testid="stTable"], .stDataFrame {
    border: 1px solid var(--wheat);
}

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.stTabs [aria-selected="true"] {
    color: var(--terracotta) !important;
}
</style>
"""


def inject_css() -> None:
    """Call once at the top of every page to apply the shared visual identity."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def field_label(text: str) -> None:
    """Small monospace eyebrow label, e.g. 'PHASE 03' or 'DATASET'."""
    st.markdown(f'<span class="field-label">{text}</span>', unsafe_allow_html=True)


def stat_block(number: str, label: str) -> str:
    """Returns HTML for a single stat (use inside st.markdown with unsafe_allow_html)."""
    return f"""
    <div class="stat-block">
        <div class="stat-number">{number}</div>
        <div class="stat-label">{label}</div>
    </div>
    """


def field_divider() -> None:
    st.markdown('<hr class="field-divider">', unsafe_allow_html=True)