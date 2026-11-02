# ruff: noqa: E501 - long inline HTML/CSS strings
"""Look and feel for the dashboard: palette, page CSS and Plotly styling. Owner: P4.

Dark command-centre theme: near-black canvas, thin-bordered panels, lime for
live and successful states, red for hazards, orange for warnings, muted blue
for secondary information. Kept in Python so the repo stays Python-first.
"""

import plotly.graph_objects as go
import streamlit as st

BG = "#171717"
SURFACE = "#202020"
PANEL = "#1d1d1d"
RAISED = "#262626"
BORDER = "rgba(255,255,255,0.07)"
BORDER_STRONG = "rgba(255,255,255,0.12)"
TEXT = "#ececec"
MUTED = "#8b8b8b"
FAINT = "#5c5c5c"
LIME = "#c5f23a"
LIME_DIM = "rgba(197,242,58,0.14)"
RED = "#ff5a52"
ORANGE = "#ffa53d"
BLUE = "#7d9bd1"
PURPLE = "#b886ff"
GREY = "#5a5a5a"

POLICY_COLOURS = {"broadcast": BLUE, "circular": ORANGE, "directional": LIME}
OUTCOME_COLOURS = {
    "warned, needed it": LIME,
    "addressed, not delivered": ORANGE,
    "missed": RED,
    "warned, didn't need it": BLUE,
    "not warned": GREY,
}
CONDITION_COLOURS = {"clean": LIME, "noisy": ORANGE, "misleading": BLUE}

FONT = "Inter, 'SF Pro Display', 'Segoe UI', system-ui, sans-serif"

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root {{ color-scheme: dark; }}
html, body, .stApp, [data-testid="stAppViewContainer"] {{
  background: {BG} !important; color: {TEXT}; font-family: {FONT};
}}
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stSidebarCollapseButton"], [data-testid="stSidebarHeader"],
[data-testid="stMainMenu"], footer {{ display: none !important; }}
.block-container {{ padding: 18px 28px 28px 28px !important; max-width: 100% !important; }}
p, span, label, div {{ font-family: {FONT}; }}
h1, h2, h3, h4 {{ font-family: {FONT}; color: {TEXT}; letter-spacing: -0.01em; }}
[data-testid="stVerticalBlock"] {{ gap: 0.75rem; }}

/* --- slim icon rail --- */
section[data-testid="stSidebar"] {{
  width: 74px !important; min-width: 74px !important; max-width: 74px !important;
  background: {BG} !important; border-right: none !important;
}}
section[data-testid="stSidebar"] > div {{ padding: 0 !important; }}
[data-testid="stSidebarUserContent"] {{ padding: 16px 10px !important; }}
[data-testid="stSidebarResizer"] {{ display: none !important; }}
.st-key-rail {{
  background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 22px;
  padding: 14px 0 14px 0; align-items: center; min-height: calc(100vh - 36px);
}}
.st-key-rail [data-testid="stVerticalBlock"] {{ align-items: center; gap: 10px; }}
.st-key-rail .stButton button {{
  width: 40px; height: 40px; min-height: 40px; border-radius: 50%; padding: 0;
  background: transparent; border: 1px solid {BORDER}; color: {MUTED};
}}
.st-key-rail .stButton button p {{ display: none; }}
.st-key-rail .stButton button:hover {{ color: {TEXT}; border-color: {BORDER_STRONG}; }}
.st-key-rail .stButton button[kind="primary"] {{
  background: {TEXT}; color: {BG}; border-color: {TEXT};
}}
.v2v-logo {{
  width: 34px; height: 34px; border-radius: 10px; margin: 2px auto 14px auto;
  background: {LIME}; display: grid; place-items: center;
  color: {BG}; font-weight: 800; font-size: 14px; letter-spacing: -0.04em;
  box-shadow: 0 0 22px rgba(197,242,58,0.35);
}}

/* --- widgets --- */
.stButton button {{
  background: {SURFACE}; color: {TEXT}; border: 1px solid {BORDER}; border-radius: 999px;
  font-size: 12.5px; padding: 4px 14px; min-height: 34px;
}}
.stButton button:hover {{ border-color: {BORDER_STRONG}; color: {TEXT}; }}
.stButton button[kind="primary"] {{ background: {LIME}; color: {BG}; border-color: {LIME}; }}
[data-baseweb="select"] > div {{
  background: {SURFACE} !important; border: 1px solid {BORDER} !important;
  border-radius: 999px !important; min-height: 36px; font-size: 13px;
}}
[data-baseweb="popover"] ul, [data-baseweb="menu"] {{ background: {RAISED} !important; }}
[data-baseweb="tag"] {{ background: {LIME_DIM} !important; color: {LIME} !important; }}
[data-testid="stWidgetLabel"] p {{
  color: {MUTED} !important; font-size: 11px !important; text-transform: uppercase;
  letter-spacing: 0.06em;
}}
/* radios as segmented pills */
[data-testid="stRadio"] > div[role="radiogroup"] {{
  background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 999px;
  padding: 3px; gap: 2px; flex-wrap: wrap; width: fit-content;
}}
[data-testid="stRadioOption"] {{
  margin: 0 !important; padding: 5px 13px; border-radius: 999px; cursor: pointer;
  font-size: 12.5px; color: {MUTED};
}}
[data-testid="stRadioOption"] > div > div:first-child {{ display: none !important; }}
[data-testid="stRadioOption"][data-selected="true"] {{ background: {RAISED}; }}
[data-testid="stRadioOption"][data-selected="true"] p {{ color: {TEXT} !important; font-weight: 500; }}
[data-testid="stRadioOption"] p {{ font-size: 12.5px; color: {MUTED}; }}
[data-testid="stRadioOption"]:hover p {{ color: {TEXT}; }}
/* toggles */
[data-testid="stCheckbox"] label p {{ font-size: 12px; color: {TEXT}; }}
[data-testid="stCheckbox"] label > div:nth-of-type(1) {{ background: {RAISED} !important; }}
[data-testid="stCheckbox"] label[data-selected="true"] > div:nth-of-type(1) {{
  background: {LIME} !important;
}}
[data-testid="stCheckbox"] label > div:nth-of-type(1) > div {{ background: {TEXT} !important; }}
[data-testid="stCheckbox"] label[data-selected="true"] > div:nth-of-type(1) > div {{
  background: {BG} !important;
}}
[data-testid="stPopover"] button {{ border-radius: 999px; }}
[data-testid="stPopoverBody"] {{
  background: {RAISED} !important; border: 1px solid {BORDER_STRONG}; border-radius: 16px;
}}
[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 14px; }}
[data-testid="stExpander"] details {{
  background: {PANEL}; border: 1px solid {BORDER}; border-radius: 16px;
}}
[data-testid="stAlert"] {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 14px; }}

/* --- cards --- */
.st-key-card, [class*="st-key-card_"] {{
  background: {PANEL}; border: 1px solid {BORDER}; border-radius: 20px; padding: 16px 18px;
  box-shadow: 0 8px 30px rgba(0,0,0,0.25);
}}
[class*="st-key-card_"] [data-testid="stVerticalBlock"] {{ gap: 0.5rem; }}
.st-key-stage {{
  position: relative; background: {PANEL}; border: 1px solid {BORDER}; border-radius: 24px;
  padding: 0; overflow: hidden; box-shadow: 0 18px 50px rgba(0,0,0,0.35);
}}
.st-key-stage > div {{ margin: 0; }}
.st-key-float_left, .st-key-float_right {{
  position: absolute; top: 16px; z-index: 20;
  background: rgba(26,26,26,0.93); backdrop-filter: blur(14px);
  border: 1px solid {BORDER_STRONG}; border-radius: 20px; padding: 14px 16px;
  box-shadow: 0 16px 40px rgba(0,0,0,0.45);
}}
.st-key-float_left {{ left: 16px; width: 336px; }}
.st-key-float_right {{ right: 16px; width: 292px; }}
.st-key-float_left [data-testid="stVerticalBlock"],
.st-key-float_right [data-testid="stVerticalBlock"] {{ gap: 0.55rem; }}

.v2v-chip {{
  display: inline-flex; align-items: center; gap: 6px; padding: 6px 12px; margin: 0 6px 6px 0;
  background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 999px;
  font-size: 12px; color: {MUTED}; white-space: nowrap;
}}
.v2v-chip b {{ color: {TEXT}; font-weight: 600; }}
.v2v-live {{
  display: inline-flex; align-items: center; gap: 8px; padding: 6px 12px;
  background: {LIME_DIM}; border: 1px solid rgba(197,242,58,0.25); border-radius: 999px;
  color: {LIME}; font-size: 12px; font-weight: 600; white-space: nowrap;
}}
.v2v-dot {{
  width: 7px; height: 7px; border-radius: 50%; background: {LIME};
  box-shadow: 0 0 0 0 rgba(197,242,58,0.6); animation: v2vpulse 1.8s infinite;
}}
@keyframes v2vpulse {{
  0% {{ box-shadow: 0 0 0 0 rgba(197,242,58,0.55); }}
  70% {{ box-shadow: 0 0 0 9px rgba(197,242,58,0); }}
  100% {{ box-shadow: 0 0 0 0 rgba(197,242,58,0); }}
}}
.v2v-brand {{ display: flex; align-items: center; gap: 10px; padding-top: 4px; }}
.v2v-brand .name {{ font-weight: 700; font-size: 15px; letter-spacing: -0.01em; color: {TEXT}; }}
.v2v-brand .sub {{ font-size: 11px; color: {MUTED}; }}
.v2v-title {{ font-size: 26px; font-weight: 600; letter-spacing: -0.02em; margin: 6px 0 0 0; }}
.v2v-subtitle {{ font-size: 12px; color: {MUTED}; margin: 2px 0 6px 0; }}
.v2v-label {{
  font-size: 10.5px; color: {MUTED}; text-transform: uppercase; letter-spacing: 0.07em;
}}
.v2v-h {{ font-size: 13px; font-weight: 600; color: {TEXT}; margin: 0; }}
.v2v-big {{ font-size: 26px; font-weight: 600; letter-spacing: -0.02em; color: {TEXT}; }}
.v2v-mid {{ font-size: 17px; font-weight: 600; color: {TEXT}; }}
.v2v-muted {{ color: {MUTED}; font-size: 12px; }}
.v2v-row {{ display: flex; justify-content: space-between; align-items: center; gap: 8px; }}
.v2v-badge {{
  display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 10.5px;
  font-weight: 600; letter-spacing: 0.02em;
}}
.v2v-bar {{ height: 6px; border-radius: 999px; background: {RAISED}; overflow: hidden; }}
.v2v-bar > div {{ height: 100%; border-radius: 999px; }}
.v2v-alert {{
  display: flex; gap: 8px; align-items: flex-start; padding: 9px 12px; border-radius: 12px;
  font-size: 12px; line-height: 1.4;
}}
.v2v-kpi {{
  background: {PANEL}; border: 1px solid {BORDER}; border-radius: 18px; padding: 14px 16px;
  height: 100%;
}}
.v2v-kpi .v {{ font-size: 24px; font-weight: 600; letter-spacing: -0.02em; margin-top: 4px; }}
.v2v-kpi .d {{ font-size: 11.5px; color: {MUTED}; margin-top: 2px; }}
.v2v-step {{
  background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 14px; padding: 10px 12px;
  height: 100%; font-size: 12px; line-height: 1.45; color: {TEXT};
}}
.v2v-step .n {{
  display: inline-grid; place-items: center; width: 18px; height: 18px; border-radius: 50%;
  font-size: 10px; font-weight: 700; margin-right: 6px;
}}
.v2v-legend {{
  position: absolute; left: 16px; bottom: 14px; z-index: 15; display: flex; flex-wrap: wrap;
  gap: 4px 12px; max-width: 64%; padding: 8px 12px; border-radius: 14px;
  background: rgba(26,26,26,0.9); border: 1px solid {BORDER}; font-size: 11.5px; color: {MUTED};
}}
.v2v-legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
.v2v-legend i {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; }}
.v2v-foot {{
  display: flex; justify-content: space-between; font-size: 11px; color: {FAINT};
  padding: 2px 6px;
}}
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int = 280, legend: bool = True) -> go.Figure:
    """Dark, minimal-grid styling shared by every chart."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": FONT, "color": MUTED, "size": 11.5},
        margin={"l": 8, "r": 8, "t": 40 if legend else 16, "b": 8},
        height=height,
        showlegend=legend,
        legend={
            "orientation": "h",
            "y": 1.02,
            "yanchor": "bottom",
            "x": 0,
            "title": {"text": ""},
            "font": {"color": TEXT, "size": 11},
            "bgcolor": "rgba(0,0,0,0)",
        },
        hoverlabel={"bgcolor": RAISED, "bordercolor": BORDER_STRONG, "font": {"color": TEXT}},
        bargap=0.35,
        bargroupgap=0.12,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=BORDER, tickcolor=BORDER)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)", zeroline=False, linecolor=BORDER)
    return fig


def badge(text: str, colour: str) -> str:
    return (
        f'<span class="v2v-badge" style="color:{colour};background:{colour}22;'
        f'border:1px solid {colour}44">{text}</span>'
    )


def bar(share: float | None, colour: str = LIME) -> str:
    width = 0.0 if share is None else max(0.0, min(1.0, share)) * 100
    return f'<div class="v2v-bar"><div style="width:{width:.1f}%;background:{colour}"></div></div>'


def kpi(label: str, value: str, detail: str = "", colour: str = TEXT) -> str:
    return (
        f'<div class="v2v-kpi"><div class="v2v-label">{label}</div>'
        f'<div class="v" style="color:{colour}">{value}</div><div class="d">{detail}</div></div>'
    )


def chip(label: str, value: str, icon: str = "") -> str:
    return f'<span class="v2v-chip">{icon}{label} <b>{value}</b></span>'


def pct(x: float | None) -> str:
    return "—" if x is None else f"{x:.0%}"


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)
