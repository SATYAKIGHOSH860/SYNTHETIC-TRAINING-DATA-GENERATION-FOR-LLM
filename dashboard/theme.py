"""
Visual theme for the dashboard. Presentation only - no pipeline logic here.

The design follows a reference set built around: an off-white ground, very
large uppercase display headings in a geometric sans, soft pastel gradient
orbs floating behind the content, pill-shaped buttons with a blue-to-mint
gradient, and statistics laid out in rows separated by thin vertical rules.

Two things about Streamlit 1.63 shaped how this is written, both discovered by
inspecting the live DOM rather than assuming:

1. The `data-baseweb` hooks older guides rely on are gone. Tabs are
   `div[role="tablist"]` holding `[data-testid="stTab"]`; radio options are
   `label[data-testid="stRadioOption"]` carrying `data-selected`.
2. Streamlit injects its own emotion stylesheet AFTER this one, so at equal
   specificity it wins. Every component rule here is therefore prefixed with
   `.stApp` and marked important - without that, roughly half the rules are
   silently overridden and only some of the design lands.
"""

from __future__ import annotations

import streamlit as st

# --- Design tokens ---------------------------------------------------------
INK = "#0A0A0B"          # display headings
BODY = "#55555F"         # body copy
MUTED = "#8A8A95"        # captions, labels
LINE = "#E8E8ED"         # hairlines and borders
SURFACE = "#FFFFFF"      # cards
CANVAS = "#FBFBFC"       # page ground
PANEL = "#F4F4F7"        # recessed sections
BLUE = "#4F6EF7"         # accent
BLUE_DEEP = "#3A56D4"
MINT = "#A8D5BA"         # gradient tail
GRADIENT = f"linear-gradient(95deg, {BLUE} 0%, #7C8FF5 45%, {MINT} 100%)"

FONT_URL = (
    "https://fonts.googleapis.com/css2?"
    "family=Outfit:wght@300;400;500;600;700;800&display=swap"
)

CSS = f"""
<style>
@import url('{FONT_URL}');

/* ---------- foundations ---------- */
html, body, [class*="st-"], .stMarkdown, button, input, textarea, select {{
    font-family: 'Outfit', 'Segoe UI', system-ui, -apple-system, sans-serif;
}}
/* Streamlit draws its chevrons and icons as Material Symbols LIGATURES: the
   element's text is literally "arrow_right" and the icon font turns it into a
   glyph. The rule above was overriding that font, so the ligature could not
   resolve and the raw word printed on top of the summary text. Icons must
   keep their own font. */
.stApp [data-testid="stIconMaterial"],
.stApp span[data-testid="stIconMaterial"],
.stApp .material-symbols-rounded, .stApp .material-icons {{
    font-family: 'Material Symbols Rounded', 'Material Icons' !important;
    font-feature-settings: 'liga' !important;
    -webkit-font-feature-settings: 'liga' !important;
    letter-spacing: normal !important; text-transform: none !important;
}}
.stApp {{ background: {CANVAS}; color: {BODY}; }}

/* Soft gradient orbs. Fixed and non-interactive so they never catch clicks. */
.stApp::before, .stApp::after {{
    content: ""; position: fixed; border-radius: 50%;
    pointer-events: none; z-index: 0; filter: blur(70px);
}}
.stApp::before {{
    width: 620px; height: 620px; top: -220px; right: -180px;
    background: radial-gradient(circle at 30% 30%,
                rgba(79,110,247,.16), rgba(168,213,186,.10) 55%, transparent 72%);
}}
.stApp::after {{
    width: 540px; height: 540px; bottom: -200px; left: -200px;
    background: radial-gradient(circle at 60% 40%,
                rgba(168,213,186,.16), rgba(79,110,247,.08) 55%, transparent 72%);
}}
.stApp [data-testid="stMainBlockContainer"] {{
    position: relative; z-index: 1;
    padding-top: 2.2rem !important; padding-bottom: 5rem !important;
    max-width: 1280px;
}}
.stApp header[data-testid="stHeader"] {{ background: transparent !important; }}
.stApp button[data-testid="stBaseButton-header"] {{
    background: transparent !important; border: none !important; box-shadow: none !important;
}}
#MainMenu, footer {{ visibility: hidden; }}

/* ---------- display type ---------- */
.stApp h1, .stApp h2, .stApp h3 {{
    font-family: 'Outfit', sans-serif !important; color: {INK} !important;
    letter-spacing: -0.02em !important; text-transform: uppercase;
    font-weight: 700 !important;
}}
.stApp h1 {{ font-size: 2.6rem !important; line-height: 1.06 !important; font-weight: 800 !important; }}
.stApp h2 {{ font-size: 1.5rem !important; line-height: 1.14 !important; }}
.stApp h3 {{ font-size: 1.08rem !important; letter-spacing: -0.01em !important; }}
.stApp h4 {{
    font-family: 'Outfit', sans-serif !important; color: {INK} !important;
    font-weight: 600 !important; font-size: .97rem !important;
    text-transform: none !important; margin-bottom: .5rem !important;
}}
.stApp p, .stApp li {{ color: {BODY}; font-size: .93rem; line-height: 1.62; }}
.stApp [data-testid="stCaptionContainer"] p {{
    color: {MUTED} !important; font-size: .79rem !important; line-height: 1.55 !important;
}}

/* ---------- hero ---------- */
.ast-hero {{ padding: .2rem 0 1.2rem; }}
.ast-brand {{
    display: flex; align-items: center; gap: .62rem; font-weight: 700;
    letter-spacing: .16em; text-transform: uppercase; font-size: .75rem;
    color: {INK}; margin-bottom: 1.5rem;
}}
.ast-brand .dot {{
    width: 22px; height: 22px; border-radius: 7px; background: {GRADIENT};
    box-shadow: 0 4px 12px rgba(79,110,247,.30);
}}
.ast-display {{
    font-size: clamp(2rem, 4.3vw, 3.2rem); font-weight: 800; line-height: 1.04;
    letter-spacing: -0.03em; color: {INK}; text-transform: uppercase;
    margin: 0 0 1rem 0; max-width: 20ch;
}}
.ast-display .soft {{
    background: {GRADIENT}; -webkit-background-clip: text;
    background-clip: text; color: transparent;
}}
.ast-lede {{ font-size: 1rem; color: {BODY}; max-width: 62ch; margin-bottom: 1.4rem; }}

.ast-stats {{
    display: flex; flex-wrap: wrap; margin: 1.1rem 0 .3rem;
    border-top: 1px solid {LINE}; border-bottom: 1px solid {LINE}; padding: 1rem 0;
}}
.ast-stat {{ flex: 1 1 150px; padding: 0 1.4rem; border-left: 1px solid {LINE}; }}
.ast-stat:first-child {{ border-left: 0; padding-left: 0; }}
.ast-stat .v {{ font-size: 1.2rem; font-weight: 700; color: {INK}; letter-spacing: -0.01em; }}
.ast-stat .k {{
    font-size: .69rem; text-transform: uppercase; letter-spacing: .13em;
    color: {MUTED}; margin-top: .22rem;
}}

.ast-section {{ text-align: center; margin: 2.2rem 0 1.4rem; }}
.ast-section .t {{
    font-size: 1.45rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: -0.02em; color: {INK};
}}
.ast-section .s {{ font-size: .88rem; color: {MUTED}; max-width: 70ch; margin: .5rem auto 0; }}

/* ---------- tabs: pill rail ---------- */
.stApp [data-testid="stTabs"] div[role="tablist"] {{
    gap: .3rem !important; background: {SURFACE} !important; padding: .36rem !important;
    border: 1px solid {LINE} !important; border-radius: 999px !important;
    box-shadow: 0 2px 14px rgba(16,18,40,.05) !important;
    display: inline-flex !important; flex-wrap: wrap !important;
}}
.stApp [data-testid="stTabs"] div[role="tablist"]::before,
.stApp [data-testid="stTabs"] div[role="tablist"]::after {{ display: none !important; }}
.stApp [data-testid="stTab"] {{
    height: auto !important; padding: .48rem 1.02rem !important;
    border-radius: 999px !important; background: transparent !important;
    border: none !important; white-space: nowrap !important;
    transition: background .18s ease !important;
}}
.stApp [data-testid="stTab"] p {{
    font-size: .83rem !important; font-weight: 500 !important;
    color: {MUTED} !important; margin: 0 !important;
}}
.stApp [data-testid="stTab"]:hover {{ background: {PANEL} !important; }}
.stApp [data-testid="stTab"]:hover p {{ color: {INK} !important; }}
.stApp [data-testid="stTab"][aria-selected="true"] {{
    background: {GRADIENT} !important;
    box-shadow: 0 4px 14px rgba(79,110,247,.30) !important;
}}
.stApp [data-testid="stTab"][aria-selected="true"] p {{
    color: #FFFFFF !important; font-weight: 600 !important;
}}
/* the library draws its own underline indicator - the pill replaces it */
.stApp [data-testid="stTab"] .react-aria-SelectionIndicator {{ display: none !important; }}
.stApp [data-testid="stTabPanel"] {{ padding-top: 1.4rem !important; }}

/* ---------- buttons: gradient pills ---------- */
.stApp button[data-testid^="stBaseButton"]:not([data-testid="stBaseButton-header"]):not([data-testid^="stBaseButton-element"]) {{
    border-radius: 999px !important;
    font-weight: 600 !important; font-size: .77rem !important;
    letter-spacing: .07em !important; text-transform: uppercase !important;
    padding: .6rem 1.45rem !important;
    border: 1px solid {LINE} !important;
    background: {SURFACE} !important; color: {INK} !important;
    box-shadow: 0 1px 3px rgba(16,18,40,.05) !important;
    transition: transform .16s ease, box-shadow .16s ease !important;
}}
.stApp button[data-testid^="stBaseButton"]:not([data-testid="stBaseButton-header"]) p {{
    font-size: .77rem !important; font-weight: 600 !important;
    letter-spacing: .07em !important; text-transform: uppercase !important;
    margin: 0 !important;
}}
.stApp button[data-testid^="stBaseButton"]:not([data-testid="stBaseButton-header"]):not([data-testid^="stBaseButton-element"]):hover {{
    transform: translateY(-1px);
    border-color: {BLUE} !important;
    box-shadow: 0 6px 18px rgba(79,110,247,.16) !important;
}}
/* The generic button rule above carries two :not() clauses, which makes it
   MORE specific than a plain [data-testid="...primary"] selector - so the
   white surface beat the gradient and, with white label text, produced an
   invisible button. These selectors mirror that specificity and add
   [kind] on top, so primary always wins. */
.stApp button[kind="primary"][data-testid="stBaseButton-primary"]:not([data-testid="stBaseButton-header"]):not([data-testid^="stBaseButton-element"]),
.stApp button[kind="primaryFormSubmit"][data-testid="stBaseButton-primaryFormSubmit"]:not([data-testid="stBaseButton-header"]) {{
    background: {GRADIENT} !important; border: none !important;
    box-shadow: 0 6px 18px rgba(79,110,247,.30) !important;
}}
.stApp button[kind="primary"][data-testid="stBaseButton-primary"] p,
.stApp button[kind="primaryFormSubmit"] p {{ color: #FFFFFF !important; }}
.stApp button[kind="primary"][data-testid="stBaseButton-primary"]:not([data-testid="stBaseButton-header"]):not([data-testid^="stBaseButton-element"]):hover {{
    border-color: transparent !important;
    box-shadow: 0 10px 26px rgba(79,110,247,.38) !important;
}}
.stApp button[data-testid^="stBaseButton"]:disabled {{
    opacity: .42 !important; transform: none !important; box-shadow: none !important;
}}

/* ---------- metrics as cards ---------- */
.stApp [data-testid="stMetric"] {{
    background: {SURFACE} !important; border: 1px solid {LINE} !important;
    border-radius: 16px !important; padding: 1rem 1.1rem !important;
    box-shadow: 0 2px 10px rgba(16,18,40,.04) !important;
    transition: transform .16s ease, box-shadow .16s ease !important;
}}
.stApp [data-testid="stMetric"]:hover {{
    transform: translateY(-2px); box-shadow: 0 10px 24px rgba(16,18,40,.08) !important;
}}
.stApp [data-testid="stMetricLabel"] p {{
    color: {MUTED} !important; font-size: .67rem !important; font-weight: 500 !important;
    text-transform: uppercase !important; letter-spacing: .13em !important;
}}
.stApp [data-testid="stMetricValue"] {{
    color: {INK} !important; font-weight: 700 !important;
    font-size: 1.58rem !important; letter-spacing: -0.02em !important;
}}
.stApp [data-testid="stMetricDelta"] {{ font-size: .73rem !important; }}

/* ---------- radio ----------
   Two shapes, told apart by data-orientation. A horizontal group reads as a
   segmented pill rail. A vertical one cannot: its options carry captions and
   stack, so a 999px radius on a tall box draws an ellipse round the list. */
.stApp [data-testid="stRadioGroup"][data-orientation="horizontal"] {{
    gap: .32rem !important; background: {SURFACE} !important; padding: .32rem !important;
    border: 1px solid {LINE} !important; border-radius: 999px !important;
    display: inline-flex !important; flex-wrap: wrap !important; align-items: center !important;
}}
.stApp [data-testid="stRadioGroup"][data-orientation="horizontal"] [data-testid="stRadioOption"] {{
    border-radius: 999px !important; padding: .4rem .95rem !important;
    margin: 0 !important; cursor: pointer !important;
    transition: background .18s ease !important;
}}
.stApp [data-testid="stRadioGroup"][data-orientation="horizontal"] [data-testid="stRadioOption"] p {{
    font-size: .81rem !important; color: {MUTED} !important; margin: 0 !important;
}}
.stApp [data-testid="stRadioGroup"][data-orientation="horizontal"] [data-testid="stRadioOption"]:hover {{
    background: {PANEL} !important;
}}
.stApp [data-testid="stRadioGroup"][data-orientation="horizontal"] [data-testid="stRadioOption"][data-selected="true"] {{
    background: {GRADIENT} !important;
    box-shadow: 0 3px 10px rgba(79,110,247,.26) !important;
}}
.stApp [data-testid="stRadioGroup"][data-orientation="horizontal"] [data-testid="stRadioOption"][data-selected="true"] p {{
    color: #FFFFFF !important; font-weight: 600 !important;
}}
/* The marker is drawn as a pseudo-element on the label; the pill already
   shows the state, so suppress it - but only on the pill rail. A vertical
   list keeps its marker, because a tinted row alone is a weak signal next to
   three lines of caption. */
.stApp [data-testid="stRadioGroup"][data-orientation="horizontal"] [data-testid="stRadioOption"]::before,
.stApp [data-testid="stRadioGroup"][data-orientation="horizontal"] [data-testid="stRadioOption"] svg {{
    display: none !important;
}}

.stApp [data-testid="stRadioGroup"][data-orientation="vertical"] {{
    display: flex !important; flex-direction: column !important;
    gap: .25rem !important; background: transparent !important;
    border: 0 !important; padding: 0 !important; border-radius: 0 !important;
}}
.stApp [data-testid="stRadioGroup"][data-orientation="vertical"] [data-testid="stRadioOption"] {{
    border-radius: 10px !important; padding: .35rem .5rem !important;
    margin: 0 !important; cursor: pointer !important;
    align-items: flex-start !important;
    transition: background .18s ease !important;
}}
.stApp [data-testid="stRadioGroup"][data-orientation="vertical"] [data-testid="stRadioOption"]:hover {{
    background: {PANEL} !important;
}}
.stApp [data-testid="stRadioGroup"][data-orientation="vertical"] [data-testid="stRadioOption"][data-selected="true"] {{
    background: {PANEL} !important;
}}
.stApp [data-testid="stRadioGroup"][data-orientation="vertical"] [data-testid="stRadioOption"][data-selected="true"] p {{
    color: {INK} !important; font-weight: 600 !important;
}}

/* ---------- sliders ---------- */
.stApp [data-testid="stSlider"] [role="slider"] {{
    background: {SURFACE} !important; border: 2px solid {BLUE} !important;
    box-shadow: 0 2px 8px rgba(79,110,247,.3) !important;
}}
.stApp [data-testid="stSliderTickBar"] {{ background: transparent !important; }}
.stApp [data-testid="stSliderThumbValue"] {{ color: {INK} !important; font-weight: 600 !important; }}

/* ---------- text inputs ---------- */
.stApp [data-testid="stTextInputRootElement"] {{
    border-radius: 999px !important; border: 1px solid {LINE} !important;
    background: {SURFACE} !important;
}}
.stApp [data-testid="stTextInputRootElement"]:focus-within {{
    border-color: {BLUE} !important; box-shadow: 0 0 0 3px rgba(79,110,247,.12) !important;
}}
.stApp [data-testid="stTextInputField"] {{
    background: transparent !important; color: {INK} !important; font-size: .88rem !important;
}}
.stApp [data-testid="stMultiSelect"] div[data-baseweb="select"] > div,
.stApp [data-testid="stMultiSelect"] > div > div {{
    border-radius: 14px !important; border-color: {LINE} !important;
    background: {SURFACE} !important;
}}

/* ---------- uploader ---------- */
.stApp [data-testid="stFileUploaderDropzone"] {{
    background: {SURFACE} !important; border: 1.5px dashed {LINE} !important;
    border-radius: 20px !important; padding: 1.9rem 1.4rem !important;
    transition: border-color .18s ease, background .18s ease !important;
}}
.stApp [data-testid="stFileUploaderDropzone"]:hover {{ border-color: {BLUE} !important; }}

/* ---------- expanders as cards ---------- */
.stApp [data-testid="stExpander"] {{
    border: 1px solid {LINE} !important; border-radius: 16px !important;
    background: {SURFACE} !important; margin-bottom: .5rem !important;
    box-shadow: 0 1px 4px rgba(16,18,40,.03) !important; overflow: hidden !important;
}}
.stApp [data-testid="stExpander"] summary {{
    padding: .74rem 1rem !important; font-size: .87rem !important; color: {INK} !important;
}}
.stApp [data-testid="stExpander"] summary:hover {{ background: {PANEL} !important; }}

/* ---------- progress ---------- */
.stApp [data-testid="stProgress"] > div > div > div {{
    background: {PANEL} !important; border-radius: 999px !important; height: 9px !important;
}}
.stApp [data-testid="stProgress"] > div > div > div > div {{
    background: {GRADIENT} !important; border-radius: 999px !important;
}}

/* ---------- alerts ---------- */
.stApp [data-testid="stAlert"] {{
    border-radius: 14px !important; border: 1px solid {LINE} !important;
    box-shadow: 0 1px 4px rgba(16,18,40,.03) !important;
}}
.stApp [data-testid="stAlert"] p {{ font-size: .87rem !important; }}
/* No left accent bar - it read as a stray vertical line before every
   sentence. Alerts are plain tinted cards. */
.stApp [data-testid="stAlert"] > div,
.stApp [data-testid="stAlertContainer"] {{
    border-left: none !important; box-shadow: none !important;
}}

/* ---------- charts and tables ---------- */
.stApp [data-testid="stPlotlyChart"] {{
    background: {SURFACE} !important; border: 1px solid {LINE} !important;
    border-radius: 18px !important; padding: .6rem !important;
    box-shadow: 0 2px 10px rgba(16,18,40,.04) !important;
}}
.stApp [data-testid="stDataFrame"] {{
    border: 1px solid {LINE} !important; border-radius: 16px !important; overflow: hidden !important;
}}
.stApp hr {{ border-color: {LINE} !important; opacity: .85; }}
.stApp code {{
    background: {PANEL} !important; color: {BLUE_DEEP} !important;
    border-radius: 6px !important; padding: .1rem .36rem !important;
}}
.stApp [data-testid="stCode"] {{ border-radius: 14px !important; border: 1px solid {LINE} !important; }}

/* ---------- sidebar ---------- */
.stApp [data-testid="stSidebar"] {{
    background: {SURFACE} !important; border-right: 1px solid {LINE} !important;
}}
.stApp [data-testid="stSidebar"] [data-testid="stMainBlockContainer"],
.stApp [data-testid="stSidebar"] > div {{ padding-top: 1.6rem; }}
.ast-kv {{
    display: flex; justify-content: space-between; align-items: baseline;
    gap: .8rem; padding: .36rem 0; border-bottom: 1px solid {LINE};
    font-size: .8rem;
}}
.ast-kv:last-child {{ border-bottom: 0; }}
.ast-kv .k {{ color: {MUTED}; }}
.ast-kv .v {{ color: {INK}; font-weight: 600; text-align: right; }}

/* ---------- small helpers ---------- */
.ast-note {{
    border: 1px solid {LINE}; background: {SURFACE};
    border-radius: 12px; padding: .76rem 1rem; font-size: .84rem; color: {BODY};
}}
.ast-passage {{
    border: 1px solid {LINE}; background: {PANEL};
    border-radius: 10px; padding: .7rem .85rem; margin: .35rem 0 .2rem;
    font-size: .8rem; line-height: 1.55; color: {BODY};
    max-height: 190px; overflow-y: auto; white-space: pre-wrap;
}}
.ast-step {{
    display: inline-flex; align-items: center; justify-content: center;
    width: 26px; height: 26px; border-radius: 50%; background: {GRADIENT};
    color: #fff; font-size: .75rem; font-weight: 700; margin-right: .55rem;
    box-shadow: 0 3px 9px rgba(79,110,247,.28);
}}
.ast-head {{
    display: flex; align-items: center; margin: 1.5rem 0 .3rem;
    font-size: 1.02rem; font-weight: 600; color: {INK};
}}

@media (max-width: 860px) {{
    .ast-display {{ font-size: 2rem; }}
    .ast-stat {{ flex-basis: 50%; border-left: 0; padding: .5rem 0; }}
}}
</style>
"""


def apply() -> None:
    """Inject the stylesheet. Call once, immediately after set_page_config."""
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title_html: str, lede: str, stats: list[tuple[str, str]] | None = None) -> None:
    """
    Brand mark, display headline, supporting line and an optional stat row.

    `title_html` may contain <span class="soft"> to tint part of the headline
    with the gradient, as the reference does on its closing words.
    """
    blocks = [
        '<div class="ast-hero">',
        '<div class="ast-brand"><span class="dot"></span>'
        "SYNTHETIC TRAINING DATA STUDIO</div>",
        f'<div class="ast-display">{title_html}</div>',
        f'<div class="ast-lede">{lede}</div>',
    ]
    if stats:
        cells = "".join(
            f'<div class="ast-stat"><div class="v">{v}</div>'
            f'<div class="k">{k}</div></div>'
            for v, k in stats
        )
        blocks.append(f'<div class="ast-stats">{cells}</div>')
    blocks.append("</div>")
    st.markdown("".join(blocks), unsafe_allow_html=True)


def section(title: str, subtitle: str = "") -> None:
    """Centred uppercase section heading, as used between major blocks."""
    sub = f'<div class="s">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="ast-section"><div class="t">{title}</div>{sub}</div>',
        unsafe_allow_html=True,
    )


def step(number: int, title: str) -> None:
    """Numbered step heading for the upload flow."""
    st.markdown(
        f'<div class="ast-head"><span class="ast-step">{number}</span>{title}</div>',
        unsafe_allow_html=True,
    )


def note(text: str) -> None:
    """Quiet bordered note - lighter than st.info for incidental guidance."""
    st.markdown(f'<div class="ast-note">{text}</div>', unsafe_allow_html=True)


def plotly(fig, height: int = 320):
    """Apply the palette to a Plotly figure so charts match the page."""
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Outfit, sans-serif", size=12, color=BODY),
        xaxis=dict(gridcolor=LINE, zerolinecolor=LINE),
        yaxis=dict(gridcolor=LINE, zerolinecolor=LINE),
        legend=dict(font=dict(size=11)),
    )
    return fig
