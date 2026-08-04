# =============================================================================
# S-TCML2 — Dual-Resolution Thermal Comfort + Load Surrogate Dashboard
# Backend: predict.py (pre-trained XGBoost models, do not retrain)
# Run:  streamlit run app.py   (from the repo root)
# Deps: streamlit plotly numpy xgboost
# =============================================================================

import math
from datetime import date

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from predict import predict

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

THEMES = {
    "dark": dict(bg="#0d1117", s1="#161b26", s2="#1c2333", s3="#222a3a",
                 border="#2d3650", text="#ffffff", text_dim="#a0aabf",
                 gold="#f0a500", blue="#3b82f6", green="#22c55e",
                 red="#ef4444", orange="#f97316"),
    "light": dict(bg="#f7f8fa", s1="#ffffff", s2="#f0f2f6", s3="#e6e9f0",
                  border="#d7dbe6", text="#141a26", text_dim="#5a6480",
                  gold="#d89000", blue="#2563eb", green="#16a34a",
                  red="#dc2626", orange="#ea580c"),
}

st.set_page_config(page_title="S-TCML2 · Thermal Comfort", page_icon="🌡️",
                    layout="wide", initial_sidebar_state="collapsed")

# -----------------------------------------------------------------------------
# SESSION STATE
# -----------------------------------------------------------------------------
_DEFAULTS = dict(theme="dark", mode="hourly", wall_raw=6.0, depth_raw=4.0,
                  wwr_raw=0.30, orientation="South", outdoor_temp=34.0,
                  month=7, day=15, hour=12)
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
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

#MainMenu, footer, header {{ display:none !important; }}
[data-testid="stSidebar"], [data-testid="collapsedControl"] {{ display:none !important; }}

html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewBlockContainer"],
.main, .block-container {{
    background-color:{C['bg']} !important; color:{C['text']} !important;
    font-family:'IBM Plex Sans',sans-serif !important;
}}
.block-container {{ padding:0.5rem 0.9rem 0.6rem !important; max-width:100% !important; }}

p, span, label, div, h1, h2, h3, h4, li,
[data-testid="stMarkdownContainer"] *, [data-baseweb="select"] *, [data-baseweb="slider"] * {{
    color:{C['text']} !important;
}}
[data-testid="stCaptionContainer"], caption {{ color:{C['text_dim']} !important; }}

[data-baseweb="select"] > div {{ background:{C['s2']} !important; border-color:{C['border']} !important; }}
[data-baseweb="popover"] * {{ background:{C['s2']} !important; color:{C['text']} !important; }}
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {{ background:{C['blue']} !important; }}

/* Monthly/Hourly segmented control (built on st.radio) */
div[data-testid="stRadio"] > div {{ flex-direction:row !important; gap:4px; background:{C['s2']};
    border:1px solid {C['border']}; border-radius:8px; padding:3px; width:fit-content; }}
div[data-testid="stRadio"] label {{ margin:0 !important; padding:5px 14px !important; border-radius:6px !important;
    font:600 12.5px 'IBM Plex Sans',sans-serif !important; }}
div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {{ display:none !important; }}
div[data-testid="stRadio"] label[data-baseweb="radio"] {{ width:auto !important; flex-shrink:0 !important; }}
div[data-testid="stRadio"] label[data-baseweb="radio"] div {{ width:auto !important; flex-shrink:0 !important; }}
div[data-testid="stRadio"] label[data-baseweb="radio"] p {{ white-space:nowrap !important; width:auto !important; }}
div[data-testid="stRadio"] label:has(input:checked) {{ background:{C['blue']} !important; }}
div[data-testid="stRadio"] label:has(input:checked) p {{ color:#ffffff !important; }}
div[data-testid="stRadio"] label:not(:has(input:checked)) p {{ color:{C['text_dim']} !important; }}

.stButton > button {{ background:{C['s2']} !important; color:{C['text']} !important;
    border:1px solid {C['border']} !important; border-radius:7px !important;
    font-family:'IBM Plex Sans',sans-serif !important; font-weight:500 !important; font-size:0.78rem !important; }}

[data-testid="stLinkButton"] a {{ background:{C['s2']} !important; border:1px solid {C['border']} !important;
    border-radius:7px !important; }}
[data-testid="stLinkButton"] a:hover {{ background:{C['s3']} !important; }}
[data-testid="stLinkButton"] p {{ color:{C['text']} !important; font:500 12.5px 'IBM Plex Sans',sans-serif !important; }}

.card-title {{ font:600 9px 'IBM Plex Mono',monospace; text-transform:uppercase; letter-spacing:.6px;
    color:{C['text_dim']}; margin-bottom:6px; }}

.hdr-title {{ font:800 16px 'Syne',sans-serif; letter-spacing:.2px; white-space:nowrap; }}
.hdr-byline {{ font:400 12px 'IBM Plex Sans',sans-serif; color:{C['text_dim']}; white-space:nowrap; }}
.hdr-sub {{ font:400 11.5px 'IBM Plex Sans',sans-serif; color:{C['text_dim']}; }}
.hdr-sub a {{ color:{C['blue']}; text-decoration:underline; }}

.banner {{ border-radius:7px; padding:6px 12px; font-size:12px; line-height:1.35; margin-bottom:6px; }}
.banner-warning {{ border:1px solid {C['orange']}; background:{C['orange']}22; }}
.banner-peak {{ border:1px solid {C['gold']}; background:{C['gold']}1f; color:{C['text_dim']}; }}
.banner-accuracy {{ border:1px solid {C['blue']}; background:{C['blue']}1a;
    font:500 12.5px 'IBM Plex Mono',monospace; }}
.banner-improve {{ border:1px solid {C['green']}; background:{C['green']}1a; }}
.banner-improve .hdr {{ font:600 12.5px 'IBM Plex Mono',monospace; color:{C['green']}; margin-bottom:2px; }}
.banner-improve .line {{ font:400 13px 'IBM Plex Sans',sans-serif; }}

.result-card {{ background:{C['s2']}; border:1px solid {C['border']}; border-radius:8px;
    padding:10px 14px; display:flex; flex-direction:column; gap:5px; height:100%; }}
.result-icon-row {{ display:flex; align-items:center; gap:6px; }}
.result-label {{ font:600 12px 'IBM Plex Mono',monospace; color:{C['text_dim']}; text-transform:uppercase; }}
.result-value {{ font:700 24px 'Syne',sans-serif; }}
.result-unit {{ font:500 12px 'IBM Plex Sans',sans-serif; color:{C['text_dim']}; }}
.result-badge {{ font:400 12px 'IBM Plex Sans',sans-serif; }}
.result-interval {{ font:400 10px 'IBM Plex Sans',sans-serif; color:{C['text_dim']}; }}
.result-explainer {{ font:400 9px 'IBM Plex Sans',sans-serif; color:{C['text_dim']}; }}

.ivl-track {{ position:relative; height:4px; border-radius:3px; background:{C['s3']}; margin:2px 0; }}
.ivl-fill {{ position:absolute; top:0; bottom:0; border-radius:3px; opacity:0.6; }}
.ivl-point {{ position:absolute; top:-3px; width:2px; height:11px; }}

.icon-circle {{ width:10px; height:10px; border-radius:50%; display:inline-block; }}
.icon-diamond {{ width:9px; height:9px; transform:rotate(45deg); display:inline-block; background:{C['gold']}; }}
.icon-tri-down {{ width:0; height:0; border-left:5px solid transparent; border-right:5px solid transparent;
    border-top:8px solid {C['green']}; display:inline-block; }}
.icon-tri-up {{ width:0; height:0; border-left:5px solid transparent; border-right:5px solid transparent;
    border-bottom:8px solid {C['orange']}; display:inline-block; }}

.input-label-row {{ display:flex; justify-content:space-between; font:10.5px 'IBM Plex Mono',monospace;
    color:{C['text_dim']}; }}
.input-label-row b {{ color:{C['text']}; font-weight:600; }}
.input-caption {{ font:8px 'IBM Plex Sans',sans-serif; color:{C['text_dim']}; }}
.hoy-caption {{ font:10px 'IBM Plex Mono',monospace; color:{C['text_dim']}; }}

.panel {{ background:{C['s1']}; border:1px solid {C['border']}; border-radius:10px; padding:10px 12px; }}
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
                                textfont=dict(color=colors["gold"], size=12), textposition="top center",
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
                                    textfont=dict(color=colors["text_dim"], size=9),
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
    b1, b2, b3 = st.columns([1.3, 1.7, 0.6], gap="small")
    with b1:
        st.link_button("✉ Contact / feedback", FEEDBACK_URL, use_container_width=True)
    with b2:
        mode_label = st.radio("mode", ["Monthly", "Hourly"],
                               index=0 if st.session_state.mode == "monthly" else 1,
                               horizontal=True, label_visibility="collapsed", key="mode_radio")
        st.session_state.mode = mode_label.lower()
    with b3:
        if st.button("☾" if st.session_state.theme == "dark" else "☀", key="theme_btn"):
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

# =============================================================================
# BANNERS
# =============================================================================
lo_range, hi_range = VALID_RANGE[mode]
if not (lo_range <= outdoor_temp <= hi_range):
    st.markdown(f"""<div class="banner banner-warning">
      ⚠ <strong>Outdoor temp {outdoor_temp:.1f}°C is outside the validated range
      ({lo_range}–{hi_range}°C)</strong> — predictions in this zone showed near-zero or
      negative R² in validation. Treat this result as unreliable.</div>""", unsafe_allow_html=True)

if not is_hourly:
    st.markdown("""<div class="banner banner-peak">
      ⓘ Monthly resolution underestimates true hourly peak load by ~80–88% on average
      — use Hourly mode for equipment sizing.</div>""", unsafe_allow_html=True)

acc = ACCURACY[mode]
st.markdown(f"""<div class="banner banner-accuracy">
  MODEL ACCURACY ({mode.upper()}) — PMV R²={acc['PMV']} &middot; Cooling R²={acc['Cooling']}
  &middot; Heating R²={acc['Heating']} &middot; PPD R²={acc['PPD']}</div>""", unsafe_allow_html=True)

recs = recommendations(mode, base_kwargs, pmv_val, wwr_snapped, orientation_label)
rec_lines = "".join(f'<div class="line">• {r}</div>' for r in recs)
st.markdown(f"""<div class="banner banner-improve">
  <div class="hdr">↳ HOW TO IMPROVE — based on current inputs</div>{rec_lines}</div>""",
            unsafe_allow_html=True)

# =============================================================================
# 3-COLUMN BODY
# =============================================================================
col_in, col_res, col_3d = st.columns([1.0, 2.3, 1.15], gap="small")

# ---- INPUTS -----------------------------------------------------------------
with col_in:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Inputs</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="input-label-row"><span>Wall width</span><b>{wall_snapped} m</b></div>',
                unsafe_allow_html=True)
    st.session_state.wall_raw = st.slider("wall_width", 2.0, 9.0, st.session_state.wall_raw, 0.1,
                                           label_visibility="collapsed")
    st.markdown(f'<div class="input-caption">Snaps to nearest: {wall_snapped} m</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="input-label-row"><span>Room depth</span><b>{depth_snapped} m</b></div>',
                unsafe_allow_html=True)
    st.session_state.depth_raw = st.slider("room_depth", 2.0, 9.0, st.session_state.depth_raw, 0.1,
                                            label_visibility="collapsed")
    st.markdown(f'<div class="input-caption">Snaps to nearest: {depth_snapped} m</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="input-label-row"><span>Orientation</span><b>{orientation_label}</b></div>',
                unsafe_allow_html=True)
    st.session_state.orientation = st.select_slider("orientation", ORIENTATION_LIST,
                                                      value=st.session_state.orientation,
                                                      label_visibility="collapsed")
    st.markdown('<div class="input-caption">Compass direction of the glazing</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="input-label-row"><span>WWR</span><b>{wwr_snapped:.0%}</b></div>',
                unsafe_allow_html=True)
    st.session_state.wwr_raw = st.slider("wwr", 0.15, 0.90, st.session_state.wwr_raw, 0.01,
                                          label_visibility="collapsed")
    st.markdown(f'<div class="input-caption">Snaps to nearest: {wwr_snapped:.0%}</div>', unsafe_allow_html=True)

    st.markdown('<hr style="border-color:%s;margin:8px 0;">' % C["border"], unsafe_allow_html=True)

    temp_label = "Outdoor temp at this hour" if is_hourly else "Monthly mean outdoor temp"
    st.markdown(f'<div class="input-label-row"><span>{temp_label}</span><b>{outdoor_temp:.1f}°C</b></div>',
                unsafe_allow_html=True)
    st.session_state.outdoor_temp = st.slider("outdoor_temp", -10.0, 45.0, st.session_state.outdoor_temp, 0.5,
                                               label_visibility="collapsed")
    st.markdown('<div class="input-caption">Ambient air temperature for this prediction</div>',
                unsafe_allow_html=True)

    if is_hourly:
        mc, dc = st.columns(2)
        with mc:
            m_label = st.selectbox("month", MONTH_NAMES, index=st.session_state.month - 1,
                                    label_visibility="collapsed")
            st.session_state.month = MONTH_NAMES.index(m_label) + 1
        with dc:
            st.session_state.day = st.slider("day", 1, 31, st.session_state.day, label_visibility="collapsed")

        st.markdown(f'<div class="input-label-row"><span>Hour of day</span>'
                     f'<b>{st.session_state.hour:02d}:00</b></div>', unsafe_allow_html=True)
        st.session_state.hour = st.slider("hour", 0, 23, st.session_state.hour, label_visibility="collapsed")

        pc1, pc2, pc3 = st.columns([1.1, 1, 1])
        with pc1:
            st.markdown(f'<div class="hoy-caption">→ hr {hoy}/8759</div>', unsafe_allow_html=True)
        with pc2:
            if st.button("☀ Noon", key="preset_summer", use_container_width=True):
                apply_preset("summer"); st.rerun()
        with pc3:
            if st.button("☾ Night", key="preset_winter", use_container_width=True):
                apply_preset("winter"); st.rerun()
    else:
        st.markdown(f'<div class="input-label-row"><span>Month</span>'
                     f'<b>{MONTH_NAMES[st.session_state.month - 1]}</b></div>', unsafe_allow_html=True)
        st.session_state.month = st.select_slider("month", list(range(1, 13)), value=st.session_state.month,
                                                    format_func=lambda m: MONTH_NAMES[m - 1],
                                                    label_visibility="collapsed")

    st.markdown('</div>', unsafe_allow_html=True)

# ---- RESULTS ------------------------------------------------------------------
with col_res:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Prediction Results</div>', unsafe_allow_html=True)

    pmv_lbl, pmv_col = band(pmv_val)
    ppd_val = result["PPD"]["value"]
    cool = result["Cooling"]
    heat = result["Heating"]

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        pmv_lo, pmv_hi = result["PMV"]["lower"], result["PMV"]["upper"]
        ivl = (interval_bar_html(pmv_lo, pmv_hi, pmv_val, -3, 3, pmv_col)
               if pmv_lo is not None else '<div class="result-interval">interval unavailable</div>')
        interval_txt = f"90% [{pmv_lo:.2f}, {pmv_hi:.2f}]" if pmv_lo is not None else ""
        st.markdown(f"""<div class="result-card" style="border-top:4px solid {pmv_col};">
          <div class="result-icon-row"><span class="icon-circle" style="background:{pmv_col};"></span>
            <span class="result-label">PMV</span></div>
          <div style="display:flex;align-items:baseline;gap:8px;">
            <span class="result-value">{pmv_val:+.2f}</span>
            <span class="result-badge" style="color:{pmv_col};">{pmv_lbl}</span></div>
          {ivl}<div class="result-interval">{interval_txt}</div>
          <div class="result-explainer">Thermal sensation, −3 cold to +3 hot</div>
        </div>""", unsafe_allow_html=True)

    with r1c2:
        st.markdown(f"""<div class="result-card" style="border-top:4px solid {C['gold']};">
          <div class="result-icon-row"><span class="icon-diamond"></span>
            <span class="result-label">PPD</span></div>
          <span class="result-value">{ppd_val:.0f}%</span>
          <div class="result-interval">no interval (ISO 7730)</div>
          <div class="result-explainer">% of occupants dissatisfied</div>
        </div>""", unsafe_allow_html=True)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        c_lo, c_hi, c_val = cool["lower"], cool["upper"], cool["value"]
        mx = max(c_hi * 1.3, 5) if c_hi is not None else max(c_val * 1.3, 5)
        ivl = (interval_bar_html(c_lo, c_hi, c_val, 0, mx, C["green"])
               if c_lo is not None else '<div class="result-interval">interval unavailable</div>')
        interval_txt = f"90% [{c_lo:.1f}, {c_hi:.1f}]" if c_lo is not None else ""
        st.markdown(f"""<div class="result-card" style="border-top:4px solid {C['green']};">
          <div class="result-icon-row"><span class="icon-tri-down"></span>
            <span class="result-label">Cooling</span></div>
          <div><span class="result-value">{c_val:.1f}</span><span class="result-unit"> kWh</span></div>
          {ivl}<div class="result-interval">{interval_txt}</div>
          <div class="result-explainer">Energy to cool the space</div>
        </div>""", unsafe_allow_html=True)

    with r2c2:
        h_lo, h_hi, h_val = heat["lower"], heat["upper"], heat["value"]
        mx = max(h_hi * 1.3, 5) if h_hi is not None else max(h_val * 1.3, 5)
        ivl = (interval_bar_html(h_lo, h_hi, h_val, 0, mx, C["orange"])
               if h_lo is not None else '<div class="result-interval">interval unavailable</div>')
        interval_txt = f"90% [{h_lo:.1f}, {h_hi:.1f}]" if h_lo is not None else ""
        st.markdown(f"""<div class="result-card" style="border-top:4px solid {C['orange']};">
          <div class="result-icon-row"><span class="icon-tri-up"></span>
            <span class="result-label">Heating</span></div>
          <div><span class="result-value">{h_val:.1f}</span><span class="result-unit"> kWh</span></div>
          {ivl}<div class="result-interval">{interval_txt}</div>
          <div class="result-explainer">Energy to heat the space</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ---- 3D ROOM PREVIEW ----------------------------------------------------------
with col_3d:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">3D Room Preview · drag to orbit</div>', unsafe_allow_html=True)
    fig = make_3d_room(wall_snapped, depth_snapped, wwr_snapped, orientation_label, C, pmv_val)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown(f'<div style="text-align:center;font:11px \'IBM Plex Mono\',monospace;color:{C["text_dim"]};">'
                f'{wall_snapped}m &times; {depth_snapped}m &middot; WWR {wwr_snapped:.0%} &middot; '
                f'{orientation_label}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
