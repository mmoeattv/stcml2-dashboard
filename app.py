# =============================================================================
# S-TCML2 — Dual-Resolution Thermal Comfort + Load Surrogate Dashboard
# Built from design_handoff_thermal_dashboard (Dashboard.dc.html + README.md)
# Backend: gui_package/predict.py (pre-trained XGBoost models, do not retrain)
# Run:  streamlit run app.py   (from inside gui_package/)
# Deps: streamlit plotly numpy xgboost
# =============================================================================

import itertools
import json
import math
import os
from datetime import date

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from predict import predict

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "climate_temp.json")) as _f:
    CLIMATE = json.load(_f)  # real West Cairo weather-file series: {"monthly": [12], "hourly": [8760]}

# -----------------------------------------------------------------------------
# CONSTANTS  (locked input domains — do not widen, see gui_handoff.md §3)
# -----------------------------------------------------------------------------
WALLS = [2, 4, 6, 9]
DEPTHS = [2, 4, 6, 9]
WWRS = [0.15, 0.30, 0.60, 0.90]
ORIENTATION_LIST = ["North", "East", "South", "West"]
ORIENTATION_DEG = {"North": 0, "East": 90, "South": 180, "West": 270}
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Validated outdoor-temp ranges. Hourly = middle 80% of the 8760-value climate
# series (matches gui_handoff.md's "~11.8-30.9C" exactly). Monthly = actual
# min/max of the 12 monthly-mean values used in training (computed from the
# raw north/south/east/west.xlsx climate column, which is identical across
# orientations - there are only 12 distinct "monthly mean" values in the
# training set, so min/max is the honest bound, not a percentile).
VALID_RANGE = {"hourly": (11.8, 30.9), "monthly": (12.9, 27.4)}

ACCURACY = {
    "hourly":  {"PMV": "0.862",  "Cooling": "0.824",  "Heating": "0.875",  "PPD": "0.752"},
    "monthly": {"PMV": "0.9998", "Cooling": "0.9948", "Heating": "0.9996", "PPD": "0.9990"},
}

PAPER_URL = "https://www.mdpi.com/2071-1050/18/7/3381"
FEEDBACK_URL = ("https://forms.office.com/Pages/ResponsePage.aspx?id="
                "R78MZ3FzakWFcN8uuKSnuwjDcFCkSUpOgNR3aIEY0WRUM01LNEEyWDFMOEFZWEJNRUwySzlXWDkwMC4u")

# v4 "Clean Professional White" palette (premium SaaS engineering look).
# Light theme values are exact per the design handoff; dark is a derived
# equivalent using the same accent hues on dark surfaces (spec only covers light).
THEMES = {
    "light": dict(bg="#F7F9FC", s1="#FFFFFF", s2="#FBFCFE", s3="#EEF4FF",
                  border="#E3E8F2", border_hover="#D4DCEC",
                  text="#172033", text_dim="#64748B", text_muted="#94A3B8",
                  blue="#1F5EFF", blue_hover="#255BEB", blue_light="#EEF4FF",
                  green="#22C55E", gold="#F5A623", orange="#F97316",
                  purple="#7C3AED", red="#DC2626"),
    "dark": dict(bg="#0B1120", s1="#141B2E", s2="#1B2438", s3="#232D45",
                 border="#2A3550", border_hover="#3A4770",
                 text="#F1F5FA", text_dim="#94A3C0", text_muted="#64748F",
                 blue="#3B82F6", blue_hover="#5B93F7", blue_light="#1E2A4A",
                 green="#22C55E", gold="#F5A623", orange="#F97316",
                 purple="#8B5CF6", red="#EF4444"),
}

# Card accent colors per target (fixed identity color, per design spec)
TARGET_COLOR = {"PMV": "purple", "PPD": "orange", "Cooling": "blue", "Heating": "red"}


def icon(name, size=16, color="currentColor", stroke_width=1.8):
    """Minimal Heroicons-style outline SVG icons — no emoji per design spec."""
    paths = {
        "mail": '<path d="M3 5h18v14H3z"/><path d="m3 6 9 7 9-7"/>',
        "moon": '<path d="M20 14.5A8.5 8.5 0 1 1 9.5 4a7 7 0 0 0 10.5 10.5z"/>',
        "sun": ('<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4'
                'M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>'),
        "warning": '<path d="M12 3 2 20h20L12 3z"/><path d="M12 10v4M12 17h.01"/>',
        "info": '<circle cx="12" cy="12" r="9"/><path d="M12 8h.01M11 12h1v5h1"/>',
        "trending-up": '<path d="M3 17l6-6 4 4 8-8"/><path d="M15 7h6v6"/>',
        "lightbulb": ('<path d="M9 18h6M10 22h4M12 2a6 6 0 0 0-4 10.5c.7.7 1 1.3 1 2.5h6'
                       'c0-1.2.3-1.8 1-2.5A6 6 0 0 0 12 2z"/>'),
        "sliders": ('<path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h13M21 18h0"/>'
                     '<circle cx="16" cy="6" r="2"/><circle cx="8" cy="12" r="2"/><circle cx="17" cy="18" r="2"/>'),
        "chart-bar": '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
        "cube": '<path d="M12 2 3 7v10l9 5 9-5V7z"/><path d="M3 7l9 5 9-5M12 12v10"/>',
        "building": '<path d="M4 21h16M5 21V9l7-5 7 5v12M9 21v-6h6v6"/>',
        "lock": '<rect x="5" y="11" width="14" height="9" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/>',
        "search": '<circle cx="11" cy="11" r="7"/><path d="m21 21-4.3-4.3"/>',
        "star": ('<path d="M12 2l3 7h7l-5.6 4.3L18.5 21 12 16.8 5.5 21l2.1-7.7L2 9h7z" '
                  'fill="currentColor" stroke="none"/>'),
    }
    p = paths.get(name, "")
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" '
            f'stroke-linejoin="round" style="display:inline-block;vertical-align:middle;flex-shrink:0;">{p}</svg>')

st.set_page_config(page_title="S-TCML2 · Thermal Comfort", page_icon="🌡️",
                    layout="wide", initial_sidebar_state="collapsed")

# -----------------------------------------------------------------------------
# SESSION STATE
# -----------------------------------------------------------------------------
_DEFAULTS = dict(theme="dark", mode="hourly", wall_raw=6.0, depth_raw=4.0,
                  wwr_raw=0.30, orientation="South", outdoor_temp=34.0,
                  month=7, day=15, hour=12,
                  lock_wall=True, lock_depth=True, lock_orientation=False, lock_wwr=False,
                  alt_results=None)
for k, v in _DEFAULTS.items():
    st.session_state.setdefault(k, v)


def nearest(v, options):
    return min(options, key=lambda x: abs(x - v))


def day_of_year(month, day):
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    day = min(day, days_in_month[month - 1])
    return date(2025, month, day).timetuple().tm_yday


def apply_preset(name):
    if name == "summer":
        st.session_state.update(month=7, day=15, hour=12, outdoor_temp=34.0)
    else:
        st.session_state.update(month=1, day=15, hour=2, outdoor_temp=2.0)


# -----------------------------------------------------------------------------
# CSS  (design tokens from design_handoff_thermal_dashboard/README.md)
# -----------------------------------------------------------------------------
C = THEMES[st.session_state.theme]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

#MainMenu, footer, header {{ display:none !important; }}
[data-testid="stSidebar"], [data-testid="collapsedControl"] {{ display:none !important; }}

html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewBlockContainer"],
.main, .block-container {{
    background-color:{C['bg']} !important; color:{C['text']} !important;
    font-family:'IBM Plex Sans',sans-serif !important;
}}
.block-container {{ padding:0.4rem 1rem 0.4rem !important; max-width:100% !important; }}

/* Compact vertical rhythm so the dashboard fits one viewport, still breathable */
[data-testid="stVerticalBlock"] {{ gap: 0.3rem !important; }}
[data-testid="stElementContainer"], .element-container {{ margin-bottom: 0 !important; }}
[data-testid="stSlider"] {{ padding-top: 0 !important; padding-bottom: 0.15rem !important; }}
[data-testid="stSlider"] > div {{ padding-bottom: 0 !important; }}

p, span, label, div, h1, h2, h3, h4, li,
[data-testid="stMarkdownContainer"] *, [data-baseweb="select"] *, [data-baseweb="slider"] * {{
    color:{C['text']} !important;
}}
[data-testid="stCaptionContainer"], caption {{ color:{C['text_dim']} !important; }}

[data-baseweb="select"] > div {{ background:{C['s1']} !important; border:1px solid {C['border']} !important;
    border-radius:10px !important; }}
[data-baseweb="popover"] * {{ background:{C['s1']} !important; color:{C['text']} !important; }}

/* Sliders: thin track, round blue handle */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {{
    background:{C['blue']} !important; box-shadow:0 1px 4px rgba(31,94,255,.35) !important; }}
[data-testid="stSlider"] [data-baseweb="slider"] > div > div {{ background:{C['blue_light']} !important; }}
[data-testid="stSlider"] [data-baseweb="slider"] > div > div > div {{ background:{C['blue']} !important; }}

/* Monthly/Hourly segmented control (built on st.radio) — selected=solid blue, inactive=white+blue border */
div[data-testid="stRadio"] > div {{ flex-direction:row !important; gap:6px; background:{C['s1']};
    border:1px solid {C['border']}; border-radius:12px; padding:4px; width:fit-content; }}
div[data-testid="stRadio"] label {{ margin:0 !important; padding:8px 18px !important; border-radius:9px !important;
    font:600 13px 'IBM Plex Sans',sans-serif !important; border:1.5px solid transparent;
    transition:all 200ms ease !important; cursor:pointer; }}
div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {{ display:none !important; }}
div[data-testid="stRadio"] label[data-baseweb="radio"] {{ width:auto !important; flex-shrink:0 !important; }}
div[data-testid="stRadio"] label[data-baseweb="radio"] div {{ width:auto !important; flex-shrink:0 !important; }}
div[data-testid="stRadio"] label[data-baseweb="radio"] p {{ white-space:nowrap !important; width:auto !important; }}
div[data-testid="stRadio"] label:has(input:checked) {{ background:{C['blue']} !important; border-color:{C['blue']} !important; }}
div[data-testid="stRadio"] label:has(input:checked) p {{ color:#ffffff !important; }}
div[data-testid="stRadio"] label:not(:has(input:checked)) {{ border-color:{C['blue']} !important; }}
div[data-testid="stRadio"] label:not(:has(input:checked)) p {{ color:{C['blue']} !important; }}

/* Buttons */
.stButton > button {{ background:{C['s1']} !important; color:{C['text']} !important;
    border:1px solid {C['border']} !important; border-radius:12px !important; height:38px !important;
    font-family:'IBM Plex Sans',sans-serif !important; font-weight:500 !important; font-size:0.82rem !important;
    transition:all 200ms ease !important; box-shadow:0 1px 3px rgba(15,23,42,.04) !important; }}
.stButton > button:hover {{ transform:translateY(-1px); box-shadow:0 4px 10px rgba(15,23,42,.08) !important;
    border-color:{C['border_hover']} !important; }}
.stButton > button:focus {{ box-shadow:0 0 0 3px {C['blue']}33 !important; }}
.stButton > button[kind="primary"] {{ background:{C['blue']} !important; color:#fff !important;
    border-color:{C['blue']} !important; }}
.stButton > button[kind="primary"]:hover {{ background:{C['blue_hover']} !important; }}

[data-testid="stLinkButton"] a {{ background:{C['s1']} !important; border:1px solid {C['border']} !important;
    border-radius:12px !important; height:38px !important; display:flex !important; align-items:center !important;
    justify-content:center !important; gap:6px; transition:all 200ms ease !important;
    box-shadow:0 1px 3px rgba(15,23,42,.04) !important; }}
[data-testid="stLinkButton"] a:hover {{ transform:translateY(-1px); box-shadow:0 4px 10px rgba(15,23,42,.08) !important; }}
[data-testid="stLinkButton"] p {{ color:{C['text']} !important; font:500 13px 'IBM Plex Sans',sans-serif !important; }}

.card-title {{ font:700 13px 'Syne',sans-serif; letter-spacing:.1px;
    color:{C['text']}; margin-bottom:6px; display:flex; align-items:center; gap:7px; }}
.card-title svg {{ color:{C['blue']}; }}

.hdr-title {{ font:700 24px 'Syne',sans-serif; letter-spacing:.1px; white-space:nowrap; color:{C['text']}; }}
.hdr-byline {{ font:500 13px 'IBM Plex Sans',sans-serif; color:{C['blue']}; white-space:nowrap; }}
.hdr-sub {{ font:400 12px 'IBM Plex Sans',sans-serif; color:{C['text_dim']}; }}
.hdr-sub a {{ color:{C['blue']}; text-decoration:none; font-weight:500; }}
.hdr-sub a:hover {{ text-decoration:underline; }}

/* Banners — icon + title + divider + description, one visual family for all four */
.banner {{ border-radius:14px; padding:10px 16px; font-size:12.5px; line-height:1.35; margin-bottom:8px;
    background:{C['s1']}; border:1px solid {C['border']}; box-shadow:0 2px 8px rgba(15,23,42,.04);
    display:flex; align-items:center; gap:10px; }}
.banner svg {{ flex-shrink:0; }}
.banner .b-title {{ font:700 11.5px 'IBM Plex Mono',monospace; letter-spacing:.04em; text-transform:uppercase;
    white-space:nowrap; }}
.banner .b-div {{ width:1px; align-self:stretch; background:{C['border']}; flex-shrink:0; }}
.banner .b-desc {{ color:{C['text_dim']}; }}
.banner-warning svg {{ color:{C['orange']}; }} .banner-warning .b-title {{ color:{C['orange']}; }}
.banner-peak svg {{ color:{C['gold']}; }} .banner-peak .b-title {{ color:{C['gold']}; }}
.banner-accuracy svg {{ color:{C['blue']}; }} .banner-accuracy .b-title {{ color:{C['blue']}; }}
.banner-improve svg {{ color:{C['green']}; }} .banner-improve .b-title {{ color:{C['green']}; }}
.banner-improve .b-desc .line {{ display:block; }}

.result-card {{ background:{C['s1']}; border:1px solid {C['border']}; border-radius:16px;
    padding:12px 14px; display:flex; flex-direction:column; gap:6px; min-height:210px;
    box-shadow:0 2px 8px rgba(15,23,42,.04); }}
.result-icon-row {{ display:flex; align-items:center; gap:7px; }}
.result-label {{ font:600 11px 'IBM Plex Mono',monospace; color:{C['text_dim']}; text-transform:uppercase; letter-spacing:.05em; }}
.result-value {{ font:700 28px 'Syne',sans-serif; color:{C['text']}; }}
.result-unit {{ font:500 12px 'IBM Plex Sans',sans-serif; color:{C['text_dim']}; }}
.result-badge {{ font:500 11.5px 'IBM Plex Sans',sans-serif; padding:2px 8px; border-radius:20px; }}
.result-interval {{ font:400 10.5px 'IBM Plex Sans',sans-serif; color:{C['text_dim']}; }}
.result-explainer {{ font:400 10px 'IBM Plex Sans',sans-serif; color:{C['text_muted']}; }}

.ivl-track {{ position:relative; height:5px; border-radius:3px; background:{C['s3']}; margin:2px 0; }}
.ivl-fill {{ position:absolute; top:0; bottom:0; border-radius:3px; opacity:0.55; }}
.ivl-point {{ position:absolute; top:-3px; width:3px; height:11px; border-radius:2px; }}

.input-label-row {{ display:flex; justify-content:space-between; align-items:center;
    font:500 12px 'IBM Plex Sans',sans-serif; color:{C['text_dim']}; margin-top:10px; }}
.value-chip {{ background:{C['s1']}; border:1px solid {C['border']}; border-radius:8px; padding:2px 10px;
    font:600 12.5px 'IBM Plex Mono',monospace; color:{C['text']}; }}
.input-caption {{ font:400 10.5px 'IBM Plex Sans',sans-serif; color:{C['text_muted']}; margin-top:-6px; }}
/* This caption is followed by the month/day widgets row (hourly mode) or month slider
   (monthly mode) rather than another plain label-row, and the default spacing left it
   hidden behind that row's own box — pull it up a touch more and add clearance below. */
.temp-range-caption {{ margin-top:-8px !important; margin-bottom:14px !important; }}
.hoy-caption {{ font:500 10.5px 'IBM Plex Mono',monospace; color:{C['text_dim']}; }}

/* Real Streamlit bordered containers (st.container(border=True)) — this is the actual
   card wrapper now; the old raw-HTML "<div class='panel'>...</div>" pattern never worked
   because each st.markdown() call renders into its own isolated DOM node and cannot wrap
   sibling widgets, which is what caused the broken/overlapping card visuals.
   Streamlit gives every st.container(border=True) a plain stVerticalBlock with an opaque,
   build-specific Emotion class — there is no stable "this one has a border" selector, so
   instead this targets structurally: the stVerticalBlock whose first child renders our own
   .card-title marker, which is only ever the first thing inside one of our 5 real panels,
   never inside a plain nested st.columns() sub-block. */
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"]:first-child .card-title) {{
    background:{C['s1']} !important; border:1px solid {C['border']} !important; border-radius:18px !important;
    padding:14px 16px !important; box-shadow:0 4px 12px rgba(15,23,42,.05) !important; height:100% !important; }}

/* Streamlit already stretches each top-level stColumn to match its row's tallest sibling
   (align-items:stretch) — confirmed via getBoundingClientRect. But the panel's own wrapper
   (stLayoutWrapper, a column-flex child) doesn't grow to fill that stretched column; it just
   sits at its natural content height, leaving empty space below shorter panels (Results, 3D
   Preview, Design Alternatives all fell short of Inputs' height). Give that wrapper flex-grow
   so it fills the column, which then lets the panel's existing height:100% rule (above)
   resolve correctly and the panel border/background extends the rest of the way down. */
[data-testid="stLayoutWrapper"]:has(> [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:first-child .card-title) {{
    flex:1 1 auto !important; min-height:0 !important; }}

/* Prediction Results panel: the visual gap between title/row1/row2 is set entirely by this
   outer stVerticalBlock's flex `gap` (confirmed empirically — per-card margin-top does NOT
   create extra space here since the cards can overflow past their stretched row's own box).
   Override just this panel's gap to push the card rows down, without touching the global
   0.3rem compact rhythm used everywhere else. */
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"]:first-child .card-title.title-results) {{
    gap: 16px !important; justify-content: center !important; }}
/* 3D Room Preview panel is now stretched to match the Inputs column's height (see the
   flex-grow rule above); center the chart+caption block within that extra vertical room
   instead of leaving it pinned to the top with empty space below. */
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"]:first-child .card-title.title-3d) {{
    justify-content: center !important; }}
/* Streamlit's internal markdown-row wrapper around each result card renders ~16px shorter
   than the card's actual content height (an internal sizing quirk, not affected by height/
   align-items overrides on that wrapper — verified directly), so the card visually overflows
   past its row and eats into the gap before the next row. Compensate with extra margin on the
   second row only, so both row-to-row gaps end up visually even. */
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"]:first-child .card-title.title-results)
    > [data-testid="stLayoutWrapper"] + [data-testid="stLayoutWrapper"] {{ margin-top: 18px !important; }}

/* Checkboxes as rounded outlined containers, per spec.
   Selector is deliberately doubled ([data-testid="x"][data-testid="x"]) to raise specificity
   above Streamlit's own Emotion-generated classes, which are equal-specificity and can win
   plain single-attribute selectors on source-order alone even with !important. */
[data-testid="stCheckbox"][data-testid="stCheckbox"] {{ border:1.5px solid {C['border']} !important;
    border-radius:10px !important; height:38px !important;
    display:flex !important; align-items:center !important; padding:0 10px !important; transition:all 200ms ease !important;
    background:{C['s1']} !important; box-sizing:border-box !important; }}
[data-testid="stCheckbox"][data-testid="stCheckbox"]:has(input:checked) {{
    border-color:{C['blue']} !important; background:{C['blue_light']} !important; }}
[data-testid="stCheckbox"][data-testid="stCheckbox"] label p {{ font:500 13px 'IBM Plex Sans',sans-serif !important; }}
[data-testid="stCheckbox"][data-testid="stCheckbox"] span[data-baseweb="checkbox"] {{ accent-color:{C['blue']} !important; }}

.alt-table {{ width:100%; border-collapse:collapse; font:13.5px 'IBM Plex Mono',monospace; margin-top:6px; }}
.alt-table th, .alt-table td {{ padding:8px 10px; text-align:center; border-bottom:1px solid {C['border']}; }}
.alt-table th {{ color:{C['text_muted']}; text-transform:uppercase; font-size:10.5px; letter-spacing:.06em;
    font-weight:600; border-bottom:1.5px solid {C['border']}; }}
.alt-table tr.baseline td {{ color:{C['text_dim']}; }}
.alt-table tr.best td:first-child {{ color:{C['text']}; font-weight:700; }}
.alt-table tr:last-child td {{ border-bottom:none; }}
.alt-note {{ font:400 12.5px 'IBM Plex Sans',sans-serif; color:{C['text_muted']}; margin-top:6px; line-height:1.5; }}

.badge-circle {{ width:26px; height:26px; border-radius:50%; display:flex; align-items:center; justify-content:center;
    font:700 12.5px 'Syne',sans-serif; color:#fff; margin:0 auto 4px; }}

/* Design Alternatives panel: the shared 0.3rem compact gap made this column feel cramped
   (checkboxes, button, table, thumbnails all packed tight) — give it noticeably more
   breathing room between each element, independent of the compact rhythm used elsewhere. */
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"]:first-child .card-title.title-alt) {{
    gap: 14px !important; }}
</style>
""", unsafe_allow_html=True)


def band(pmv):
    if pmv > 1.5:
        return "hot", C["red"]
    elif pmv > 0.5:
        return "slightly warm", C["orange"]
    elif pmv > -0.5:
        return "comfortable", C["green"]
    elif pmv > -1.5:
        return "slightly cool", C["blue"]
    return "cold", C["blue"]


def pct(v, mn, mx):
    return max(0.0, min(100.0, (v - mn) / (mx - mn) * 100))


def interval_bar_html(lo, hi, point, mn, mx, color):
    fill_l = pct(lo, mn, mx)
    fill_w = max(1.0, pct(hi, mn, mx) - fill_l)
    pt_l = pct(point, mn, mx)
    return (f'<div class="ivl-track">'
            f'<div class="ivl-fill" style="left:{fill_l}%;width:{fill_w}%;background:{color};"></div>'
            f'<div class="ivl-point" style="left:{pt_l}%;background:{color};"></div>'
            f'</div>')


def make_3d_room(wall_width, room_depth, wwr, orientation_label, colors, pmv_val=None):
    """Adapted from example_stcml_gui.py's make_3d_room, recolored per theme."""
    W, D, H = float(wall_width), float(room_depth), 3.0
    traces = []

    def mesh_face(xs, ys, zs, color, opacity):
        return go.Mesh3d(x=xs, y=ys, z=zs, i=[0, 0], j=[1, 2], k=[2, 3],
                          color=color, opacity=opacity, flatshading=True,
                          showscale=False, hoverinfo="skip")

    traces.append(mesh_face([0, W, W, 0], [0, 0, D, D], [0, 0, 0, 0], colors["s3"], 0.92))
    traces.append(mesh_face([0, W, W, 0], [0, 0, D, D], [H, H, H, H], colors["s2"], 0.45))
    traces.append(mesh_face([0, W, W, 0], [D, D, D, D], [0, 0, H, H], colors["s3"], 0.65))
    traces.append(mesh_face([0, 0, 0, 0], [0, D, D, 0], [0, 0, H, H], colors["s2"], 0.60))
    traces.append(mesh_face([W, W, W, W], [0, D, D, 0], [0, 0, H, H], colors["s2"], 0.60))

    gz_bot, gz_h = H * 0.10, H * 0.72
    gz_top = gz_bot + gz_h
    gw = W * wwr
    gx0, gx1 = (W - gw) / 2, (W - gw) / 2 + gw

    traces.append(mesh_face([0, W, W, 0], [0, 0, 0, 0], [0, 0, gz_bot, gz_bot], colors["s3"], 0.78))
    traces.append(mesh_face([0, W, W, 0], [0, 0, 0, 0], [gz_top, gz_top, H, H], colors["s3"], 0.78))
    if gx0 > 0.01:
        traces.append(mesh_face([0, gx0, gx0, 0], [0, 0, 0, 0], [gz_bot, gz_bot, gz_top, gz_top], colors["s3"], 0.78))
    if gx1 < W - 0.01:
        traces.append(mesh_face([gx1, W, W, gx1], [0, 0, 0, 0], [gz_bot, gz_bot, gz_top, gz_top], colors["s3"], 0.78))

    glass_col = colors["blue"] if (pmv_val is None or pmv_val <= 0.5) else colors["red"]
    traces.append(go.Mesh3d(x=[gx0, gx1, gx1, gx0], y=[0, 0, 0, 0], z=[gz_bot, gz_bot, gz_top, gz_top],
                             i=[0, 0], j=[1, 2], k=[2, 3], color=glass_col, opacity=0.50,
                             flatshading=True, showscale=False, hoverinfo="skip"))

    fw = dict(color=colors["blue"], width=1.8)
    for xl, yl, zl in [
        ([gx0, gx1], [0, 0], [gz_bot, gz_bot]), ([gx0, gx1], [0, 0], [gz_top, gz_top]),
        ([gx0, gx0], [0, 0], [gz_bot, gz_top]), ([gx1, gx1], [0, 0], [gz_bot, gz_top]),
        ([(gx0 + gx1) / 2] * 2, [0, 0], [gz_bot, gz_top]), ([gx0, gx1], [0, 0], [(gz_bot + gz_top) / 2] * 2),
    ]:
        traces.append(go.Scatter3d(x=xl, y=yl, z=zl, mode="lines", line=fw,
                                    hoverinfo="skip", showlegend=False))

    ori_deg = ORIENTATION_DEG.get(orientation_label, 180)
    ang = math.radians(ori_deg)
    sx, sy = W / 2 + 1.4 * math.sin(ang), D / 2 + 1.4 * math.cos(ang)
    ax, ay = W / 2 + 0.5 * math.sin(ang), D / 2 + 0.5 * math.cos(ang)
    traces.append(go.Scatter3d(x=[sx, ax], y=[sy, ay], z=[H * 0.65, H * 0.55],
                                mode="lines+markers+text", line=dict(color=colors["gold"], width=3),
                                marker=dict(size=[10, 3], color=colors["gold"]), text=["☀", ""],
                                textfont=dict(color=colors["gold"], size=16), textposition="top center",
                                hoverinfo="skip", showlegend=False))

    for xi in np.linspace(0, W, 5):
        traces.append(go.Scatter3d(x=[xi, xi], y=[0, D], z=[0, 0], mode="lines",
                                    line=dict(color=colors["border"], width=1), hoverinfo="skip", showlegend=False))
    for yi in np.linspace(0, D, 5):
        traces.append(go.Scatter3d(x=[0, W], y=[yi, yi], z=[0, 0], mode="lines",
                                    line=dict(color=colors["border"], width=1), hoverinfo="skip", showlegend=False))

    for xv, yv, zv, txt in [
        (W / 2, -0.3, 0.05, f"W = {W:.1f} m"), (-0.3, D / 2, 0.05, f"D = {D:.1f} m"),
        (-0.3, 0, H / 2, "H = 3.0 m"), ((gx0 + gx1) / 2, -0.15, gz_top + 0.12, f"WWR {wwr:.0%}"),
    ]:
        traces.append(go.Scatter3d(x=[xv], y=[yv], z=[zv], mode="text", text=[txt],
                                    textfont=dict(color=colors["text_dim"], size=13),
                                    hoverinfo="skip", showlegend=False))

    fig = go.Figure(data=traces)
    fig.update_layout(
        height=340, margin=dict(t=0, b=0, l=0, r=0), paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(bgcolor="rgba(0,0,0,0)",
                   xaxis=dict(visible=False, range=[-0.6, W + 0.6]),
                   yaxis=dict(visible=False, range=[-0.6, D + 0.6]),
                   zaxis=dict(visible=False, range=[0, H + 0.4]),
                   camera=dict(eye=dict(x=1.55, y=-1.55, z=1.05)),
                   aspectmode="manual", aspectratio=dict(x=W / 3.0, y=D / 3.0, z=H / 3.0)),
        showlegend=False,
    )
    return fig


def recommendations(mode, base_kwargs, pmv_val, wwr_snapped, orientation_label):
    """Live recommendations computed from real predict() deltas (not the design
    prototype's placeholder formula)."""
    recs = []
    if pmv_val > 0.5:
        idx = WWRS.index(wwr_snapped)
        if idx > 0:
            lower = WWRS[idx - 1]
            alt = predict(**{**base_kwargs, "wwr": lower})
            recs.append(f"Drop WWR {wwr_snapped:.0%}→{lower:.0%} → PMV moves to "
                        f"{alt['PMV']['value']:+.2f}, cooling to {alt['Cooling']['value']:.1f} kWh")
        if orientation_label in ("South", "West"):
            alt2 = predict(**{**base_kwargs, "orientation": 0})
            recs.append(f"Switch orientation {orientation_label}→North → PPD moves to "
                        f"{alt2['PPD']['value']:.0f}%")
    elif pmv_val < -0.5:
        idx = WWRS.index(wwr_snapped)
        if idx < len(WWRS) - 1:
            higher = WWRS[idx + 1]
            alt = predict(**{**base_kwargs, "wwr": higher})
            recs.append(f"Raise WWR {wwr_snapped:.0%}→{higher:.0%} → PMV moves to "
                        f"{alt['PMV']['value']:+.2f}")
        else:
            recs.append("PMV is on the cool side — a warmer-facing orientation would move it toward neutral")
    else:
        recs.append("Current configuration is within the comfortable band — no changes recommended")
    return recs[:2]


def search_alternatives(base_kwargs, locks, current_wall, current_depth, current_orientation_deg, current_wwr):
    """Exhaustive search over the free (unlocked) architectural variables, holding
    climate/time (month/hour/outdoor_temp) and locked variables fixed. Ranks every
    combination by an equally-weighted trade-off between normalized PPD (comfort)
    and normalized total (cooling+heating) load — a compromise pick, not a hard
    PMV +-0.5 filter, since that filter can legitimately eliminate every option
    under peak conditions and leave nothing to compare. Also reports whether any
    searched combination would have met the strict +-0.5 band, for an honest note.

    Cooling-load point predictions carry meaningfully more relative error than PMV/PPD
    (~4% MAPE vs ~1.6%, verified against the training simulation data), which is enough
    to flip the ranking between two genuinely close candidates even though the model's
    aggregate accuracy is high. Rather than present a falsely confident #1 in that case,
    candidates whose 90% conformal total-load intervals overlap are grouped as tied and
    ordered within the tie by PPD, the more reliable signal, instead of by raw score.
    """
    domains = {
        "exterior_wall_width": [current_wall] if locks["wall"] else WALLS,
        "room_depth": [current_depth] if locks["depth"] else DEPTHS,
        "orientation": [current_orientation_deg] if locks["orientation"] else list(ORIENTATION_DEG.values()),
        "wwr": [current_wwr] if locks["wwr"] else WWRS,
    }
    keys = list(domains.keys())
    candidates = []
    for combo in itertools.product(*(domains[k] for k in keys)):
        kwargs = {**base_kwargs, **dict(zip(keys, combo))}
        r = predict(**kwargs)
        c_lo, c_hi = r["Cooling"]["lower"], r["Cooling"]["upper"]
        h_lo, h_hi = r["Heating"]["lower"], r["Heating"]["upper"]
        candidates.append({
            "wall": kwargs["exterior_wall_width"], "depth": kwargs["room_depth"],
            "orientation_deg": kwargs["orientation"], "wwr": kwargs["wwr"],
            "pmv": r["PMV"]["value"], "ppd": r["PPD"]["value"],
            "cooling": r["Cooling"]["value"], "heating": r["Heating"]["value"],
            "load_lo": (c_lo + h_lo) if (c_lo is not None and h_lo is not None) else None,
            "load_hi": (c_hi + h_hi) if (c_hi is not None and h_hi is not None) else None,
        })

    ppds = [c["ppd"] for c in candidates]
    loads = [c["cooling"] + c["heating"] for c in candidates]
    ppd_lo, ppd_hi = min(ppds), max(ppds)
    load_lo, load_hi = min(loads), max(loads)
    for c in candidates:
        n_ppd = 0.0 if ppd_hi == ppd_lo else (c["ppd"] - ppd_lo) / (ppd_hi - ppd_lo)
        n_load = 0.0 if load_hi == load_lo else ((c["cooling"] + c["heating"]) - load_lo) / (load_hi - load_lo)
        c["score"] = n_ppd + n_load

    candidates.sort(key=lambda c: c["score"])
    any_comfortable = any(-0.5 <= c["pmv"] <= 0.5 for c in candidates)

    top3 = candidates[:3]

    def overlaps(a, b):
        if a["load_lo"] is None or b["load_lo"] is None:
            return False
        return a["load_lo"] <= b["load_hi"] and b["load_lo"] <= a["load_hi"]

    clusters, current = [], [0]
    for i in range(1, len(top3)):
        if overlaps(top3[current[-1]], top3[i]):
            current.append(i)
        else:
            clusters.append(current)
            current = [i]
    clusters.append(current)

    ordered, rank = [], 1
    for cluster in clusters:
        members = sorted((top3[i] for i in cluster), key=lambda c: c["ppd"]) if len(cluster) > 1 else [top3[cluster[0]]]
        for m in members:
            m["rank"], m["tied"] = rank, len(members) > 1
            ordered.append(m)
        rank += 1

    has_ties = any(c["tied"] for c in ordered)
    return ordered, any_comfortable, has_ties


def compute_trend(mode, wall, depth, orientation_deg, wwr, month, day):
    """PMV/PPD across the year (monthly mode) or across the selected day's 24
    hours (hourly mode), using the real bundled climate series for outdoor
    temp at each step — not the single value on the outdoor-temp slider."""
    pmv_vals, ppd_vals = [], []
    if mode == "monthly":
        x_labels = MONTH_NAMES
        for m in range(1, 13):
            temp = CLIMATE["monthly"][m - 1]
            r = predict(mode="monthly", exterior_wall_width=wall, room_depth=depth,
                        orientation=orientation_deg, wwr=wwr, outdoor_temp=temp, month=m)
            pmv_vals.append(r["PMV"]["value"])
            ppd_vals.append(r["PPD"]["value"])
        current_idx = month - 1
    else:
        x_labels = [f"{h:02d}:00" for h in range(24)]
        base_doy = day_of_year(month, day)
        for h in range(24):
            hoy = (base_doy - 1) * 24 + h
            temp = CLIMATE["hourly"][hoy]
            r = predict(mode="hourly", exterior_wall_width=wall, room_depth=depth,
                        orientation=orientation_deg, wwr=wwr, outdoor_temp=temp, hour_of_year=hoy)
            pmv_vals.append(r["PMV"]["value"])
            ppd_vals.append(r["PPD"]["value"])
        current_idx = st.session_state.hour
    return x_labels, pmv_vals, ppd_vals, current_idx


def make_trend_chart(x_labels, pmv_vals, ppd_vals, current_idx, colors):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_hrect(y0=-0.5, y1=0.5, fillcolor=colors["green"], opacity=0.10, line_width=0, secondary_y=False)

    fig.add_trace(go.Scatter(x=x_labels, y=pmv_vals, mode="lines+markers", name="PMV",
                              line=dict(color=colors["blue"], width=2.5), marker=dict(size=5)),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=x_labels, y=ppd_vals, mode="lines+markers", name="PPD (%)",
                              line=dict(color=colors["orange"], width=2.5, dash="dot"), marker=dict(size=5)),
                  secondary_y=True)

    fig.add_vline(x=x_labels[current_idx], line=dict(color=colors["gold"], width=1.5, dash="dash"))

    fig.update_layout(
        height=170, margin=dict(t=10, b=5, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=colors["text_dim"], family="IBM Plex Mono", size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(color=colors["text"])),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, tickfont=dict(color=colors["text_dim"]))
    fig.update_yaxes(title_text="PMV", secondary_y=False, showgrid=True, gridcolor=colors["border"],
                      zeroline=False, tickfont=dict(color=colors["blue"]), title_font=dict(color=colors["blue"]))
    fig.update_yaxes(title_text="PPD (%)", secondary_y=True, showgrid=False,
                      tickfont=dict(color=colors["orange"]), title_font=dict(color=colors["orange"]))
    return fig


# =============================================================================
# HEADER
# =============================================================================
hdr_l, hdr_r = st.columns([2.3, 2.0], gap="small")
with hdr_l:
    st.markdown(f"""
    <div style="display:flex;align-items:baseline;gap:8px;">
      <span class="hdr-title">S-TCML2: A Dual-Resolution Machine Learning Surrogate</span>
      <span class="hdr-byline">by Mayar Moeat</span>
    </div>
    <div class="hdr-sub">This tool is part of PhD research &middot;
      <a href="{PAPER_URL}" target="_blank">preview the published paper &#9656;</a>
    </div>
    """, unsafe_allow_html=True)

with hdr_r:
    b1, b2, b3 = st.columns([1.4, 1.7, 0.7], gap="small")
    with b1:
        st.link_button("Contact / feedback", FEEDBACK_URL, use_container_width=True)
    with b2:
        mode_label = st.radio("mode", ["Monthly", "Hourly"],
                               index=0 if st.session_state.mode == "monthly" else 1,
                               horizontal=True, label_visibility="collapsed", key="mode_radio")
        st.session_state.mode = mode_label.lower()
    with b3:
        if st.button("Light" if st.session_state.theme == "dark" else "Dark",
                     key="theme_btn", use_container_width=True):
            st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
            st.rerun()

mode = st.session_state.mode
is_hourly = mode == "hourly"

# =============================================================================
# INPUTS (compute snapped values + hour_of_year up front, needed by banners too)
# =============================================================================
wall_snapped = nearest(st.session_state.wall_raw, WALLS)
depth_snapped = nearest(st.session_state.depth_raw, DEPTHS)
wwr_snapped = nearest(st.session_state.wwr_raw, WWRS)
orientation_label = st.session_state.orientation
orientation_deg = ORIENTATION_DEG[orientation_label]
outdoor_temp = st.session_state.outdoor_temp

if is_hourly:
    hoy = (day_of_year(st.session_state.month, st.session_state.day) - 1) * 24 + st.session_state.hour
    base_kwargs = dict(mode="hourly", exterior_wall_width=wall_snapped, room_depth=depth_snapped,
                        orientation=orientation_deg, wwr=wwr_snapped, outdoor_temp=outdoor_temp,
                        hour_of_year=hoy)
else:
    base_kwargs = dict(mode="monthly", exterior_wall_width=wall_snapped, room_depth=depth_snapped,
                        orientation=orientation_deg, wwr=wwr_snapped, outdoor_temp=outdoor_temp,
                        month=st.session_state.month)

result = predict(**base_kwargs)
pmv_val = result["PMV"]["value"]

outer_left, col_alt = st.columns([70, 30], gap="medium")

with outer_left:
    # =============================================================================
    # BANNERS
    # =============================================================================
    lo_range, hi_range = VALID_RANGE[mode]
    if not (lo_range <= outdoor_temp <= hi_range):
        st.markdown(f"""<div class="banner banner-warning">{icon('warning', 18)}
          <span class="b-title">Outdoor temperature out of validated range</span>
          <span class="b-div"></span>
          <span class="b-desc">Selected outdoor temperature ({outdoor_temp:.1f}°C) is outside the validated range
          [{lo_range}°C – {hi_range}°C]. Predictions may be less reliable.</span></div>""", unsafe_allow_html=True)

    if not is_hourly:
        st.markdown(f"""<div class="banner banner-peak">{icon('info', 18)}
          <span class="b-title">Peak-hiding note (monthly mode)</span>
          <span class="b-div"></span>
          <span class="b-desc">Monthly mode hides short-term peaks. For hourly peaks and detailed
          behavior, switch to Hourly mode.</span></div>""", unsafe_allow_html=True)

    acc = ACCURACY[mode]
    st.markdown(f"""<div class="banner banner-accuracy">{icon('trending-up', 18)}
      <span class="b-title">Model accuracy ({mode})</span>
      <span class="b-div"></span>
      <span class="b-desc">PMV R²={acc['PMV']} &middot; Cooling R²={acc['Cooling']}
      &middot; Heating R²={acc['Heating']} &middot; PPD R²={acc['PPD']}</span></div>""", unsafe_allow_html=True)

    recs = recommendations(mode, base_kwargs, pmv_val, wwr_snapped, orientation_label)
    rec_lines = "".join(f'<span class="line">{r}</span>' for r in recs)
    st.markdown(f"""<div class="banner banner-improve">{icon('lightbulb', 18)}
      <span class="b-title">How to improve</span>
      <span class="b-div"></span>
      <span class="b-desc">{rec_lines}</span></div>""", unsafe_allow_html=True)

    # =============================================================================
    # 4-COLUMN BODY
    # =============================================================================
    col_in, col_res, col_3d = st.columns([24, 20, 26], gap="medium")

    # ---- INPUTS -----------------------------------------------------------------
    with col_in:
        with st.container(border=True):
            st.markdown(f'<div class="card-title">{icon("sliders", 16)} Inputs</div>', unsafe_allow_html=True)

            st.markdown(f'<div class="input-label-row"><span>Wall width</span>'
                        f'<span class="value-chip">{wall_snapped:.2f} m</span></div>', unsafe_allow_html=True)
            st.session_state.wall_raw = st.slider("wall_width", 2.0, 9.0, st.session_state.wall_raw, 0.1,
                                                   label_visibility="collapsed")
            st.markdown(f'<div class="input-caption">Snaps to nearest: {wall_snapped:.2f} m</div>', unsafe_allow_html=True)

            st.markdown(f'<div class="input-label-row"><span>Room depth</span>'
                        f'<span class="value-chip">{depth_snapped:.2f} m</span></div>', unsafe_allow_html=True)
            st.session_state.depth_raw = st.slider("room_depth", 2.0, 9.0, st.session_state.depth_raw, 0.1,
                                                    label_visibility="collapsed")
            st.markdown(f'<div class="input-caption">Snaps to nearest: {depth_snapped:.2f} m</div>', unsafe_allow_html=True)

            st.markdown(f'<div class="input-label-row"><span>Orientation</span>'
                        f'<span class="value-chip">{orientation_label}</span></div>', unsafe_allow_html=True)
            st.session_state.orientation = st.select_slider("orientation", ORIENTATION_LIST,
                                                              value=st.session_state.orientation,
                                                              label_visibility="collapsed")
            st.markdown('<div class="input-caption">Compass direction of the glazing</div>', unsafe_allow_html=True)

            st.markdown(f'<div class="input-label-row"><span>WWR</span>'
                        f'<span class="value-chip">{wwr_snapped:.0%}</span></div>', unsafe_allow_html=True)
            st.session_state.wwr_raw = st.slider("wwr", 0.15, 0.90, st.session_state.wwr_raw, 0.01,
                                                  label_visibility="collapsed")
            st.markdown(f'<div class="input-caption">Snaps to nearest: {wwr_snapped:.0%}</div>', unsafe_allow_html=True)

            st.markdown('<hr style="border-color:%s;margin:4px 0;">' % C["border"], unsafe_allow_html=True)

            temp_label = "Outdoor air temperature (Monthly mean)" if not is_hourly else "Outdoor air temperature (Hourly)"
            st.markdown(f'<div class="input-label-row"><span>{temp_label}</span>'
                        f'<span class="value-chip">{outdoor_temp:.1f} °C</span></div>', unsafe_allow_html=True)
            st.session_state.outdoor_temp = st.slider("outdoor_temp", -10.0, 45.0, st.session_state.outdoor_temp, 0.5,
                                                       label_visibility="collapsed")
            st.markdown(f'<div class="input-caption temp-range-caption">Validated range: {lo_range:.1f} °C – '
                        f'{hi_range:.1f} °C</div>', unsafe_allow_html=True)

            if is_hourly:
                mc, dc = st.columns(2)
                with mc:
                    m_label = st.selectbox("month", MONTH_NAMES, index=st.session_state.month - 1,
                                            label_visibility="collapsed")
                    st.session_state.month = MONTH_NAMES.index(m_label) + 1
                with dc:
                    st.session_state.day = st.slider("day", 1, 31, st.session_state.day, label_visibility="collapsed")

                st.markdown(f'<div class="input-label-row"><span>Hour of day</span>'
                             f'<span class="value-chip">{st.session_state.hour:02d}:00</span></div>', unsafe_allow_html=True)
                st.session_state.hour = st.slider("hour", 0, 23, st.session_state.hour, label_visibility="collapsed")

                pc1, pc2, pc3 = st.columns([1.1, 1, 1])
                with pc1:
                    st.markdown(f'<div class="hoy-caption">hr {hoy}/8759</div>', unsafe_allow_html=True)
                with pc2:
                    if st.button("Noon", key="preset_summer", use_container_width=True):
                        apply_preset("summer"); st.rerun()
                with pc3:
                    if st.button("Night", key="preset_winter", use_container_width=True):
                        apply_preset("winter"); st.rerun()
            else:
                st.markdown(f'<div class="input-label-row"><span>Month</span>'
                             f'<span class="value-chip">{MONTH_NAMES[st.session_state.month - 1]}</span></div>',
                             unsafe_allow_html=True)
                st.session_state.month = st.select_slider("month", list(range(1, 13)), value=st.session_state.month,
                                                            format_func=lambda m: MONTH_NAMES[m - 1],
                                                            label_visibility="collapsed")


    # ---- RESULTS ------------------------------------------------------------------
    with col_res:
        with st.container(border=True):
            st.markdown(f'<div class="card-title title-results">{icon("chart-bar", 16)} Prediction Results</div>', unsafe_allow_html=True)

            pmv_lbl, pmv_col = band(pmv_val)
            ppd_val = result["PPD"]["value"]
            cool = result["Cooling"]
            heat = result["Heating"]

            pmv_accent, ppd_accent, cool_accent, heat_accent = C["purple"], C["orange"], C["blue"], C["red"]

            r1c1, r1c2 = st.columns(2)
            with r1c1:
                pmv_lo, pmv_hi = result["PMV"]["lower"], result["PMV"]["upper"]
                ivl = (interval_bar_html(pmv_lo, pmv_hi, pmv_val, -3, 3, pmv_accent)
                       if pmv_lo is not None else '<div class="result-interval">interval unavailable</div>')
                interval_txt = f"90% PI: [{pmv_lo:.2f}, {pmv_hi:.2f}]" if pmv_lo is not None else ""
                st.markdown(f"""<div class="result-card" style="border-top:4px solid {pmv_accent};">
                  <div class="result-icon-row"><span class="icon-circle" style="background:{pmv_accent};"></span>
                    <span class="result-label">PMV</span></div>
                  <div style="display:flex;align-items:baseline;gap:8px;">
                    <span class="result-value">{pmv_val:+.2f}</span></div>
                  <div><span class="result-badge" style="background:{pmv_col}1a;color:{pmv_col};">{pmv_lbl}</span></div>
                  {ivl}<div class="result-interval">{interval_txt}</div>
                  <div class="result-explainer">Thermal sensation on 7-point scale (-3 cold to +3 hot).</div>
                </div>""", unsafe_allow_html=True)

            with r1c2:
                st.markdown(f"""<div class="result-card" style="border-top:4px solid {ppd_accent};">
                  <div class="result-icon-row"><span class="icon-diamond" style="background:{ppd_accent};"></span>
                    <span class="result-label">PPD</span></div>
                  <div><span class="result-value">{ppd_val:.1f}</span><span class="result-unit"> %</span></div>
                  <div class="result-interval">no interval (ISO 7730)</div>
                  <div class="result-explainer">Predicted Percentage of Dissatisfied (%) per ISO 7730.</div>
                </div>""", unsafe_allow_html=True)

            r2c1, r2c2 = st.columns(2)
            with r2c1:
                c_lo, c_hi, c_val = cool["lower"], cool["upper"], cool["value"]
                mx = max(c_hi * 1.3, 5) if c_hi is not None else max(c_val * 1.3, 5)
                ivl = (interval_bar_html(c_lo, c_hi, c_val, 0, mx, cool_accent)
                       if c_lo is not None else '<div class="result-interval">interval unavailable</div>')
                interval_txt = f"90% PI: [{c_lo:.1f}, {c_hi:.1f}]" if c_lo is not None else ""
                st.markdown(f"""<div class="result-card" style="border-top:4px solid {cool_accent};">
                  <div class="result-icon-row"><span class="icon-tri-down" style="border-top-color:{cool_accent};"></span>
                    <span class="result-label">Cooling Load</span></div>
                  <div><span class="result-value">{c_val:.1f}</span><span class="result-unit"> kWh</span></div>
                  {ivl}<div class="result-interval">{interval_txt}</div>
                  <div class="result-explainer">Estimated monthly cooling energy required.</div>
                </div>""", unsafe_allow_html=True)

            with r2c2:
                h_lo, h_hi, h_val = heat["lower"], heat["upper"], heat["value"]
                mx = max(h_hi * 1.3, 5) if h_hi is not None else max(h_val * 1.3, 5)
                ivl = (interval_bar_html(h_lo, h_hi, h_val, 0, mx, heat_accent)
                       if h_lo is not None else '<div class="result-interval">interval unavailable</div>')
                interval_txt = f"90% PI: [{h_lo:.1f}, {h_hi:.1f}]" if h_lo is not None else ""
                st.markdown(f"""<div class="result-card" style="border-top:4px solid {heat_accent};">
                  <div class="result-icon-row"><span class="icon-tri-up" style="border-bottom-color:{heat_accent};"></span>
                    <span class="result-label">Heating Load</span></div>
                  <div><span class="result-value">{h_val:.1f}</span><span class="result-unit"> kWh</span></div>
                  {ivl}<div class="result-interval">{interval_txt}</div>
                  <div class="result-explainer">Estimated monthly heating energy required.</div>
                </div>""", unsafe_allow_html=True)


    # ---- 3D ROOM PREVIEW ----------------------------------------------------------
    with col_3d:
        with st.container(border=True):
            st.markdown(f'<div class="card-title title-3d">{icon("cube", 16)} 3D Room Preview <span style="color:{C["text_muted"]};'
                        f'font-weight:400;">&middot; drag to orbit</span></div>', unsafe_allow_html=True)
            fig = make_3d_room(wall_snapped, depth_snapped, wwr_snapped, orientation_label, C, pmv_val)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown(f'<div style="text-align:center;font:13px \'IBM Plex Mono\',monospace;color:{C["text_dim"]};">'
                        f'{wall_snapped}m &times; {depth_snapped}m &middot; WWR {wwr_snapped:.0%} &middot; '
                        f'{orientation_label}</div>', unsafe_allow_html=True)

# ---- DESIGN ALTERNATIVES EXPLORER (right rail, full column height) ------------
with col_alt:
    with st.container(border=True):
        st.markdown(f'<div class="card-title title-alt">{icon("building", 16)} Design Alternatives</div>', unsafe_allow_html=True)
        st.markdown('<div class="alt-note">Lock at least 2 variables; the tool searches the rest for the top 3 '
                    'best trade-offs between acceptable PMV and minimal cooling+heating load.</div>', unsafe_allow_html=True)

        lc1, lc2 = st.columns(2)
        with lc1:
            st.session_state.lock_wall = st.checkbox(f"Wall ({wall_snapped} m)",
                                                      value=st.session_state.lock_wall, key="lock_wall_cb")
            st.session_state.lock_orientation = st.checkbox(f"Orient. ({orientation_label})",
                                                             value=st.session_state.lock_orientation, key="lock_ori_cb")
        with lc2:
            st.session_state.lock_depth = st.checkbox(f"Depth ({depth_snapped} m)",
                                                       value=st.session_state.lock_depth, key="lock_depth_cb")
            st.session_state.lock_wwr = st.checkbox(f"WWR ({wwr_snapped:.0%})",
                                                     value=st.session_state.lock_wwr, key="lock_wwr_cb")

        n_locked = sum([st.session_state.lock_wall, st.session_state.lock_depth,
                        st.session_state.lock_orientation, st.session_state.lock_wwr])
        n_free = 4 - n_locked

        if n_locked < 2:
            st.markdown(f'<div class="alt-note" style="color:{C["orange"]};display:flex;align-items:center;gap:5px;">'
                        f'{icon("warning", 13, C["orange"])} Lock at least two variables to generate alternatives.'
                        f'</div>', unsafe_allow_html=True)
        else:
            search_size = 1
            for locked, opts in [(st.session_state.lock_wall, WALLS), (st.session_state.lock_depth, DEPTHS),
                                  (st.session_state.lock_orientation, ORIENTATION_LIST), (st.session_state.lock_wwr, WWRS)]:
                if not locked:
                    search_size *= len(opts)
            if st.button(f"Suggest best 3 ({search_size} combos)", key="search_alts_btn",
                         use_container_width=True, type="primary"):
                locks = {"wall": st.session_state.lock_wall, "depth": st.session_state.lock_depth,
                          "orientation": st.session_state.lock_orientation, "wwr": st.session_state.lock_wwr}
                st.session_state.alt_results = search_alternatives(
                    base_kwargs, locks, wall_snapped, depth_snapped, orientation_deg, wwr_snapped)

        if st.session_state.alt_results is not None:
            alts, any_comfortable, has_ties = st.session_state.alt_results
            deg_to_label = {v: k for k, v in ORIENTATION_DEG.items()}
            rows = ""
            base_row = (f'<tr class="baseline"><td>Now (baseline)</td><td>{orientation_label}</td>'
                        f'<td>{wwr_snapped:.0%}</td><td>{pmv_val:+.2f}</td><td>{ppd_val:.0f}%</td></tr>')
            for a in alts:
                cls = "best" if a["rank"] == 1 else ""
                star = f' <span style="color:{C["gold"]};">&#9733;</span>' if a["rank"] == 1 else ""
                tie_mark = " &approx;" if a["tied"] else ""
                label = f"{a['rank']}{tie_mark}{star}"
                rows += (f'<tr class="{cls}"><td>{label}</td><td>{deg_to_label[a["orientation_deg"]]}</td>'
                         f'<td>{a["wwr"]:.0%}</td><td>{a["pmv"]:+.2f}</td><td>{a["ppd"]:.0f}%</td></tr>')
            st.markdown(f"""<div style="overflow-x:auto;"><table class="alt-table">
              <tr><th>Alt</th><th>Ori</th><th>WWR</th><th>PMV</th><th>PPD (%)</th></tr>
              {base_row}{rows}
            </table></div>
            <div class="alt-note"><span style="color:{C['gold']};">&#9733;</span> best trade-off (acceptable PMV
            with minimal total load)</div>""", unsafe_allow_html=True)
            if has_ties:
                st.markdown(f'<div class="alt-note" style="color:{C["text_muted"]};display:flex;align-items:flex-start;gap:5px;">'
                            f'<span style="margin-top:1px;">{icon("info", 13, C["text_muted"])}</span>'
                            f'<span>&approx; marks alternatives whose predicted cooling+heating load is within '
                            f'the model\'s own uncertainty range — treat tied ranks as equally good, not strictly '
                            f'ordered.</span></div>', unsafe_allow_html=True)

            badge_colors = [C["blue"], C["green"], C["purple"]]
            alt_cols = st.columns(len(alts))
            for i, (col, a) in enumerate(zip(alt_cols, alts)):
                with col:
                    st.markdown(f'<div class="badge-circle" style="background:{badge_colors[i % 3]};">'
                                f'{chr(65+i)}</div>', unsafe_allow_html=True)
                    fig_alt = make_3d_room(a["wall"], a["depth"], a["wwr"], deg_to_label[a["orientation_deg"]],
                                            C, a["pmv"])
                    fig_alt.update_layout(height=130, margin=dict(t=0, b=0, l=0, r=0))
                    st.plotly_chart(fig_alt, use_container_width=True, config={"displayModeBar": False},
                                    key=f"alt3d_{i}")

            if not any_comfortable:
                st.markdown(f'<div class="alt-note" style="color:{C["orange"]};display:flex;align-items:flex-start;gap:5px;">'
                            f'<span style="margin-top:1px;">{icon("warning", 13, C["orange"])}</span>'
                            f'<span>No alternative achieves strict comfort (|PMV| &le; 0.5) under current '
                            f'conditions — these are the least-uncomfortable, lowest-load options available, '
                            f'not a fully comfortable design.</span></div>', unsafe_allow_html=True)


# =============================================================================
# PREDICTION TREND — PMV & PPD across the year (monthly) or the day (hourly)
# =============================================================================
with st.container(border=True):
    trend_title = ("Prediction Trend · PMV & PPD across the year (July mean per month, real climate data)"
                   if not is_hourly else
                   f"Prediction Trend · PMV & PPD across {MONTH_NAMES[st.session_state.month-1]} "
                   f"{st.session_state.day} (real hourly climate data)")
    st.markdown(f'<div class="card-title">{icon("trending-up", 16)} {trend_title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="alt-note">Computed for the current geometry ({wall_snapped}m &times; {depth_snapped}m, '
                f'{orientation_label}, WWR {wwr_snapped:.0%}) using the real West Cairo weather-file temperature at '
                f'each {"month" if not is_hourly else "hour"} — not the outdoor-temp slider, which only sets the '
                f'single-point prediction above. The dashed gold line marks your current selection; the green band '
                f'is the &plusmn;0.5 PMV comfort zone.</div>', unsafe_allow_html=True)

    trend_x, trend_pmv, trend_ppd, trend_idx = compute_trend(
        mode, wall_snapped, depth_snapped, orientation_deg, wwr_snapped,
        st.session_state.month, st.session_state.day if is_hourly else 15)
    trend_fig = make_trend_chart(trend_x, trend_pmv, trend_ppd, trend_idx, C)
    st.plotly_chart(trend_fig, use_container_width=True, config={"displayModeBar": False})
