"""Taylor's University | EV Charging Analytics (Streamlit dashboard).

Run with:  streamlit run app.py
Optional files next to this script:
    cleaned_ev_dataset.csv / ev_campus_dataset.csv   data (required)
    ml_model.py                                      predict(), explain() (optional)
    assets/taylors_logo.png                          sidebar + banner logo
    assets/hero_exec.jpg, hero_analytics.jpg, hero_ml.jpg   banner photos (optional)
"""
import base64
import inspect
import io
import math
import re
import textwrap
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

try:
    import ml_model
except ImportError:
    ml_model = None

BASE_DIR = Path(__file__).resolve().parent
ASSETS = BASE_DIR / "assets"


def _stretch(fn):
    """Full-width kwarg that works on old (use_container_width) and new (width="stretch") Streamlit."""
    try:
        return {"width": "stretch"} if "width" in inspect.signature(fn).parameters else {"use_container_width": True}
    except (TypeError, ValueError):
        return {"use_container_width": True}


STRETCH = {name: _stretch(getattr(st, name)) for name in ("button", "popover", "download_button", "plotly_chart") if hasattr(st, name)}
try:
    MATERIAL_ICONS = tuple(int(x) for x in re.findall(r"\d+", str(st.__version__))[:2]) >= (1, 40)
except Exception:
    MATERIAL_ICONS = False


def icon(name, fallback):
    """Material icon in labels on recent Streamlit, plain emoji on older versions."""
    return f":material/{name}:" if MATERIAL_ICONS else fallback

# ============================================================
# THEME + LANGUAGE (kept in session state, chosen in the sidebar)
# ============================================================
st.session_state.setdefault("dark_mode", True)
st.session_state.setdefault("lang", "en")
MODE, LANG = ("dark" if st.session_state["dark_mode"] else "light"), st.session_state["lang"]

LANGS = {"en": "English", "ms": "Bahasa Melayu", "zh": "中文"}
_LANG_INDEX = {"en": 0, "ms": 1, "zh": 2}
LANG_CODES = {"en": "EN", "ms": "BM", "zh": "中文"}


def tr(en, ms, zh):
    """Return the text for the active language."""
    return (en, ms, zh)[_LANG_INDEX.get(LANG, 0)]


THEMES = {
    "dark": dict(BG="#0A0F1F", SIDE="#0D1326", PANEL="#111A33", PANEL2="#0E1630", LINE="#1E2B4D", GRID="#1A2547",
                 TEXT="#EAF0FF", MUTED="#9AA8C7", DIM="#6E7C9E", INPUT="#243055", HOVER="#16203D",
                 BLUE="#4F7CFF", GREEN="#3DD68C", ORANGE="#F59E42", PURPLE="#A66BFF", RED="#F0566A", AMBER="#F2B544",
                 SCHEME="dark"),
    "light": dict(BG="#F3F5F9", SIDE="#FFFFFF", PANEL="#FFFFFF", PANEL2="#F1F4F9", LINE="#DCE2EE", GRID="#E8ECF4",
                  TEXT="#0F172A", MUTED="#51607A", DIM="#8592A8", INPUT="#C7D0E2", HOVER="#E8EDF7",
                  BLUE="#2F5FE8", GREEN="#16A267", ORANGE="#E07B16", PURPLE="#8B4FE0", RED="#D6334A", AMBER="#C98A0A",
                  SCHEME="light"),
}
PALETTE_KEYS = ("BG", "SIDE", "PANEL", "PANEL2", "LINE", "GRID", "TEXT", "MUTED", "DIM", "INPUT", "HOVER",
                "BLUE", "GREEN", "ORANGE", "PURPLE", "RED", "AMBER", "SCHEME")
BG, SIDE, PANEL, PANEL2, LINE, GRID, TEXT, MUTED, DIM, INPUT, HOVER, BLUE, GREEN, ORANGE, PURPLE, RED, AMBER, SCHEME = (
    THEMES[MODE][k] for k in PALETTE_KEYS)
FONT = "Inter, 'Segoe UI', sans-serif"
SERIES = [BLUE, GREEN, ORANGE, PURPLE, "#38BDF8", RED]

try:  # best effort: align Streamlit's native widgets with the chosen theme
    for _k, _v in {"theme.base": MODE, "theme.primaryColor": BLUE, "theme.backgroundColor": BG,
                   "theme.secondaryBackgroundColor": PANEL, "theme.textColor": TEXT}.items():
        st._config.set_option(_k, _v)
except Exception:
    pass

st.set_page_config(
    page_title="Taylor's University | EV Charging Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

pio.templates["ev"] = go.layout.Template(
    layout=dict(
        font=dict(family=FONT, color=MUTED, size=12),
        colorway=SERIES,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(size=12, color=MUTED), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(font_family=FONT, bgcolor="#1B2750", font_color="#FFFFFF", bordercolor=LINE),
    )
)
pio.templates.default = "ev"

# ============================================================
# DATA
# ============================================================


@st.cache_data(show_spinner=False)
def load_data():
    for f in (BASE_DIR / "cleaned_ev_dataset.csv", BASE_DIR / "ev_campus_dataset.csv"):
        if f.exists():
            try:
                return pd.read_csv(f)
            except Exception:
                continue
    return None


_raw = load_data()
if _raw is None:
    st.error(tr("Dataset not found. Place 'cleaned_ev_dataset.csv' or 'ev_campus_dataset.csv' in the project folder.",
                "Set data tidak dijumpai. Letakkan 'cleaned_ev_dataset.csv' atau 'ev_campus_dataset.csv' dalam folder projek.",
                "未找到数据集。请将 'cleaned_ev_dataset.csv' 或 'ev_campus_dataset.csv' 放入项目文件夹。"))
    st.stop()

df = _raw.copy()
df.columns = [str(c).strip() for c in df.columns]


def first_col(keys, exclude_words=(), exclude_cols=(), object_only=False):
    """First column whose name contains one of `keys` (checked in order)."""
    for key in keys:
        for c in df.columns:
            low = c.lower()
            if c in exclude_cols or key not in low or any(w in low for w in exclude_words):
                continue
            if object_only and not (pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])):
                continue
            return c
    return None


def numeric_column(name, keys, default):
    if name not in df.columns:
        c = first_col(keys)
        df[name] = pd.to_numeric(df[c], errors="coerce") if c else default
    df[name] = pd.to_numeric(df[name], errors="coerce")


numeric_column("Energy_kWh", ["energy", "kwh"], 0.0)
numeric_column("Duration_Mins", ["duration", "mins", "minutes"], 0.0)

_rev = first_col(["revenue", "cost", "price", "fee", "amount_rm", "_rm"])
df["Revenue_Val"] = pd.to_numeric(df[_rev], errors="coerce").fillna(0.0) if _rev else df["Energy_kWh"].fillna(0) * 0.60

_peak = "Peak_Hour" if "Peak_Hour" in df.columns else first_col(["peak"])
if _peak:
    df["Peak_Hour_Label"] = (df[_peak].astype(str).str.strip().str.lower()
                             .isin(["1", "1.0", "true", "peak", "yes"]).map({True: "Peak", False: "Non-peak"}))
else:
    df["Peak_Hour_Label"] = "Standard"

_temp = first_col(["temp"])
df["Battery_Temp"] = pd.to_numeric(df[_temp], errors="coerce") if _temp else 30.0
HAS_TEMP = _temp is not None
USER_COL = first_col(["user"])
OVERSTAY_COL = first_col(["overstay"])
POWER_COL = first_col(["power"], exclude_words=["kwh", "energy"])
SOC_COL = first_col(["soc", "state_of_charge"])

_grid = "Grid_Status" if "Grid_Status" in df.columns else first_col(["grid"])
df["Grid_Status_Label"] = df[_grid].astype(str).str.strip() if _grid else "Normal"

# Date / time detection
_start = first_col(["start", "timestamp", "datetime", "date", "time"],
                   exclude_words=["duration", "min", "hour", "end"], object_only=True)
_end = first_col(["end", "stop"], exclude_words=["spend", "trend"], object_only=True)
df["Start_Dt"] = pd.NaT
HAS_DT = False
if _start:
    _parsed = pd.to_datetime(df[_start], errors="coerce")
    if _parsed.notna().mean() > 0.5:
        df["Start_Dt"], HAS_DT = _parsed, True
if HAS_DT:
    df["End_Dt"] = pd.to_datetime(df[_end], errors="coerce") if _end else pd.NaT
    if _end is None or df["End_Dt"].isna().all():
        df["End_Dt"] = df["Start_Dt"] + pd.to_timedelta(df["Duration_Mins"].fillna(0), unit="m")
    DATE_MIN, DATE_MAX = df["Start_Dt"].min().date(), df["Start_Dt"].max().date()

df["Hour_of_Day"] = np.nan
_hour = "Hour" if "Hour" in df.columns else first_col(["hour"], exclude_cols=[_peak or "", "Peak_Hour_Label", "Hour_of_Day"])
if HAS_DT:
    df["Hour_of_Day"] = df["Start_Dt"].dt.hour
elif _hour:
    df["Hour_of_Day"] = pd.to_numeric(df[_hour], errors="coerce")

if "Charger_Type" not in df.columns and first_col(["charger"]):
    df["Charger_Type"] = df[first_col(["charger"])]
HAS_CHARGER = "Charger_Type" in df.columns
SESSION_ID_COL = next((c for c in df.columns if "session" in c.lower() and "id" in c.lower()), None)

# ============================================================
# SMALL HELPERS
# ============================================================


def render(markup):
    """Render HTML without blank lines / indentation that would break Markdown."""
    markup = textwrap.dedent(markup)
    st.markdown("\n".join(line.strip() for line in markup.splitlines() if line.strip()), unsafe_allow_html=True)


def options(col):
    if col not in df.columns:
        return ["All"]
    return ["All"] + sorted(df[col].dropna().astype(str).str.strip().unique().tolist())


def cols(spec, **kw):
    """st.columns with vertically centred content when the Streamlit version supports it."""
    try:
        return st.columns(spec, vertical_alignment="center", **kw)
    except TypeError:
        return st.columns(spec, **kw)


def b64_file(path):
    return base64.b64encode(Path(path).read_bytes()).decode()


def find_asset(names):
    for name in names:
        p = ASSETS / name
        if p.exists():
            return p
    return None


def find_logo():
    p = find_asset(["taylors_logo.png", "taylors_logo.png.png", "taylors_logo.jpg", "taylors_logo.jpeg"])
    if p or not ASSETS.exists():
        return p
    return next((f for f in sorted(ASSETS.iterdir())
                 if "logo" in f.name.lower() and f.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")), None)


def pk(label):
    """Display text for peak labels (data values stay in English)."""
    return {"Peak": tr("Peak", "Puncak", "高峰"), "Non-peak": tr("Non-peak", "Bukan puncak", "非高峰"),
            "Standard": tr("Standard", "Standard", "标准")}.get(label, label)


def all_label(value):
    return tr("All", "Semua", "全部") if value == "All" else value


# ============================================================
# GENERATED BANNER ILLUSTRATIONS (fallback when no photo is in assets/)
# ============================================================
def _scene_campus():
    """Campus at dusk (Executive dashboard)."""
    w, h = 1400, 340
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" preserveAspectRatio="xMidYMid slice">',
        '<defs><linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0" stop-color="#070c1c"/><stop offset=".55" stop-color="#14264d"/><stop offset="1" stop-color="#2c4a7c"/>'
        '</linearGradient><linearGradient id="gl" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0" stop-color="#9fd0ff" stop-opacity=".55"/><stop offset="1" stop-color="#9fd0ff" stop-opacity="0"/>'
        '</linearGradient></defs>',
        f'<rect width="{w}" height="{h}" fill="url(#sky)"/>',
        '<rect x="760" y="30" width="520" height="200" fill="#0f1c38"/>',
        '<rect x="1040" y="70" width="300" height="160" fill="#13244a"/>',
    ]
    rng = np.random.default_rng(7)
    for bx, by, bw, bh in [(780, 46, 480, 170), (1060, 86, 260, 130)]:
        for x in range(bx, bx + bw, 26):
            for y in range(by, by + bh, 24):
                if rng.random() > 0.38:
                    parts.append(f'<rect x="{x}" y="{y}" width="14" height="12" fill="#ffd98a" opacity="{rng.uniform(.35, .9):.2f}"/>')
    parts += [
        '<rect x="700" y="214" width="640" height="12" fill="#cfd9ee"/>',
        '<rect x="700" y="226" width="640" height="26" fill="url(#gl)"/>',
    ]
    for px_ in (740, 960, 1180, 1300):
        parts.append(f'<rect x="{px_}" y="226" width="8" height="70" fill="#9fb0cf"/>')
    parts.append(f'<rect x="0" y="292" width="{w}" height="48" fill="#080d1b"/>')
    for cx in (820, 1010, 1120):
        parts.append(f'<rect x="{cx}" y="248" width="14" height="44" rx="3" fill="#1f2c4d"/>'
                     f'<rect x="{cx + 3}" y="254" width="8" height="10" fill="#3DD68C"/>')
    parts.append('<path d="M870 292 q4-34 40-40 l70-4 q34 0 52 24 l16 20 z" fill="#e8edf7"/>'
                 '<path d="M926 254 l60-3 q22 2 36 18 l-100 4z" fill="#2a3a5e"/>'
                 '<circle cx="920" cy="294" r="13" fill="#0a1020"/><circle cx="1012" cy="294" r="13" fill="#0a1020"/>')
    for tx in (600, 660, 1350):
        parts.append(f'<circle cx="{tx}" cy="250" r="34" fill="#0c2a2a"/><rect x="{tx - 4}" y="270" width="8" height="30" fill="#0a1a1a"/>')
    parts.append("</svg>")
    return base64.b64encode("".join(parts).encode()).decode()

def _scene_analytics():
    """Glowing bar + line chart and donut (Visual Analytics)."""
    w, h = 1400, 340
    rng = np.random.default_rng(3)
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" preserveAspectRatio="xMidYMid slice">',
         '<defs><linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#070c1c"/><stop offset="1" stop-color="#112250"/></linearGradient>'
         '<linearGradient id="bar" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#4F7CFF"/><stop offset="1" stop-color="#4F7CFF" stop-opacity=".12"/></linearGradient>'
         '<linearGradient id="bar2" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#3DD68C"/><stop offset="1" stop-color="#3DD68C" stop-opacity=".1"/></linearGradient></defs>',
         f'<rect width="{w}" height="{h}" fill="url(#bg)"/>']
    for y in range(50, 320, 45):
        p.append(f'<line x1="640" y1="{y}" x2="{w}" y2="{y}" stroke="#2a3d78" stroke-opacity=".45"/>')
    n = 14
    hs = np.linspace(70, 190, n) + rng.uniform(-28, 28, n)
    xs = [700 + i * 46 for i in range(n)]
    for i, (x, hh) in enumerate(zip(xs, hs)):
        p.append(f'<rect x="{x}" y="{300 - hh:.0f}" width="26" height="{hh:.0f}" rx="5" fill="url(#{"bar" if i % 3 else "bar2"})"/>')
    ly = [300 - hh - 36 - rng.uniform(0, 24) for hh in hs]
    pts = " ".join(f"{x + 13},{y:.0f}" for x, y in zip(xs, ly))
    p.append(f'<polyline points="{pts}" fill="none" stroke="#3DD68C" stroke-width="3" stroke-linejoin="round" opacity=".95"/>')
    for x, y in zip(xs, ly):
        p.append(f'<circle cx="{x + 13}" cy="{y:.0f}" r="5" fill="#0A0F1F" stroke="#3DD68C" stroke-width="2.5"/>')
    pts2 = " ".join(f"{x + 13},{300 - hh * 0.55 - 10:.0f}" for x, hh in zip(xs, hs))
    p.append(f'<polyline points="{pts2}" fill="none" stroke="#F59E42" stroke-width="2" stroke-dasharray="7 6" opacity=".8"/>')
    p.append('<circle cx="1290" cy="84" r="44" fill="none" stroke="#1d2c55" stroke-width="14"/>'
             '<circle cx="1290" cy="84" r="44" fill="none" stroke="#4F7CFF" stroke-width="14" stroke-dasharray="150 277" transform="rotate(-90 1290 84)"/>'
             '<circle cx="1290" cy="84" r="44" fill="none" stroke="#A66BFF" stroke-width="14" stroke-dasharray="70 277" stroke-dashoffset="-155" transform="rotate(-90 1290 84)"/>')
    for _ in range(40):
        p.append(f'<circle cx="{rng.uniform(560, 1400):.0f}" cy="{rng.uniform(10, 330):.0f}" r="{rng.uniform(1, 2.6):.1f}" fill="#8FAAFF" opacity="{rng.uniform(.15, .5):.2f}"/>')
    p.append("</svg>")
    return base64.b64encode("".join(p).encode()).decode()


def _scene_ml():
    """AI x EV: electric car on a charging bay with a neural network and AI chip above it."""
    w, h = 1400, 340
    rng = np.random.default_rng(21)
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" preserveAspectRatio="xMidYMid slice">',
         '<defs>'
         '<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#050816"/><stop offset=".55" stop-color="#150f3d"/><stop offset="1" stop-color="#0a2a55"/></linearGradient>'
         '<linearGradient id="body" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#3a52a8"/><stop offset="1" stop-color="#0f1a42"/></linearGradient>'
         '<radialGradient id="ug"><stop offset="0" stop-color="#38BDF8" stop-opacity=".75"/><stop offset="1" stop-color="#38BDF8" stop-opacity="0"/></radialGradient>'
         '<radialGradient id="gp"><stop offset="0" stop-color="#A66BFF" stop-opacity=".55"/><stop offset="1" stop-color="#A66BFF" stop-opacity="0"/></radialGradient>'
         '<linearGradient id="beam" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#ffffff" stop-opacity=".22"/><stop offset="1" stop-color="#ffffff" stop-opacity="0"/></linearGradient>'
         '</defs>',
         f'<rect width="{w}" height="{h}" fill="url(#bg)"/>',
         '<g transform="translate(-100,-24)">']
    # perspective floor
    for i in range(8):
        y = 262 + i * 12
        p.append(f'<line x1="560" y1="{y}" x2="1400" y2="{y}" stroke="#4F7CFF" stroke-opacity="{0.34 - i * 0.035:.2f}"/>')
    for k in range(-8, 14):
        p.append(f'<line x1="{1000 + k * 70}" y1="262" x2="{1000 + k * 170}" y2="360" stroke="#4F7CFF" stroke-opacity=".16"/>')
    p.append('<ellipse cx="1008" cy="300" rx="260" ry="22" fill="url(#ug)"/>')
    p.append('<circle cx="1230" cy="112" r="150" fill="url(#gp)" opacity=".6"/>')

    # neural network cloud
    nodes = [(rng.uniform(830, 1180), rng.uniform(82, 150)) for _ in range(16)]
    for i, (x1, y1) in enumerate(nodes):
        d = sorted(((x1 - x2) ** 2 + (y1 - y2) ** 2, j) for j, (x2, y2) in enumerate(nodes) if j != i)[:2]
        for _, j in d:
            x2, y2 = nodes[j]
            p.append(f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" stroke="#8F7BFF" stroke-opacity=".38"/>')
    cols_ = ["#4F7CFF", "#A66BFF", "#38BDF8", "#3DD68C"]
    for i, (x, y) in enumerate(nodes):
        c = cols_[i % 4]
        p.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="11" fill="{c}" opacity=".16"/><circle cx="{x:.0f}" cy="{y:.0f}" r="4.5" fill="{c}"/>')
    # data streams from the network down to the car
    for x1, x2 in ((960, 1000), (1040, 1050), (1110, 1100)):
        p.append(f'<line x1="{x1}" y1="150" x2="{x2}" y2="186" stroke="#38BDF8" stroke-width="1.6" stroke-dasharray="3 6" opacity=".8"/>')

    # AI chip
    cx, cy = 1250, 112
    for k in range(4):
        o = 52 + k * 18 - 36
        p.append(f'<line x1="{cx - 52}" y1="{cy - 27 + k * 18}" x2="{cx - 38}" y2="{cy - 27 + k * 18}" stroke="#A66BFF" stroke-width="2"/>'
                 f'<line x1="{cx + 38}" y1="{cy - 27 + k * 18}" x2="{cx + 52}" y2="{cy - 27 + k * 18}" stroke="#A66BFF" stroke-width="2"/>'
                 f'<line x1="{cx - 27 + k * 18}" y1="{cy - 52}" x2="{cx - 27 + k * 18}" y2="{cy - 38}" stroke="#A66BFF" stroke-width="2"/>'
                 f'<line x1="{cx - 27 + k * 18}" y1="{cy + 38}" x2="{cx - 27 + k * 18}" y2="{cy + 52}" stroke="#A66BFF" stroke-width="2"/>')
    p.append(f'<rect x="{cx - 38}" y="{cy - 38}" width="76" height="76" rx="14" fill="#120f35" stroke="#A66BFF" stroke-width="2.5"/>'
             f'<rect x="{cx - 28}" y="{cy - 28}" width="56" height="56" rx="9" fill="none" stroke="#8F7BFF" stroke-opacity=".5"/>'
             f'<text x="{cx}" y="{cy + 12}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="32" font-weight="800" fill="#ffffff">AI</text>')
    p.append(f'<line x1="{cx - 52}" y1="{cy + 9}" x2="1180" y2="{cy + 9}" stroke="#A66BFF" stroke-opacity=".5" stroke-dasharray="4 5"/>')

    # lightning hexagon
    hx, hy = 930, 112
    hexp = " ".join(f"{hx + 36 * np.cos(np.pi / 3 * i + np.pi / 6):.0f},{hy + 36 * np.sin(np.pi / 3 * i + np.pi / 6):.0f}" for i in range(6))
    p.append(f'<polygon points="{hexp}" fill="rgba(245,158,66,0.12)" stroke="#F59E42" stroke-width="2.5"/>'
             f'<polygon points="{hx + 5},{hy - 22} {hx - 14},{hy + 5} {hx - 2},{hy + 5} {hx - 6},{hy + 24} {hx + 14},{hy - 5} {hx + 1},{hy - 5}" fill="#F59E42"/>')

    # charging post + cable
    p.append('<rect x="722" y="172" width="44" height="104" rx="10" fill="#101a42" stroke="#3DD68C" stroke-width="2"/>'
             '<rect x="731" y="183" width="26" height="24" rx="4" fill="#0b3b2d"/>'
             '<polygon points="746,186 738,198 744,198 741,205 751,193 745,193" fill="#3DD68C"/>'
             '<rect x="716" y="276" width="56" height="8" rx="3" fill="#1b2a5a"/>'
             '<path d="M766 236 C 780 268, 796 238, 808 247" fill="none" stroke="#3DD68C" stroke-width="4" stroke-linecap="round"/>'
             '<circle cx="808" cy="247" r="6" fill="#3DD68C" opacity=".4"/><circle cx="808" cy="247" r="3" fill="#3DD68C"/>')

    # electric car
    p.append('<polygon points="1216,246 1400,205 1400,290" fill="url(#beam)"/>')
    p.append('<path d="M800 270 L800 252 Q806 240 835 236 L888 224 Q925 190 985 184 L1075 184 Q1125 190 1150 222 L1192 232 Q1214 238 1216 258 L1216 270 Z" '
             'fill="url(#body)" stroke="#6EA8FF" stroke-width="2"/>'
             '<path d="M905 222 Q935 197 985 192 L1030 192 L1030 224 Z" fill="#0b1530" stroke="#5aa0ff" stroke-opacity=".7"/>'
             '<path d="M1042 192 L1078 192 Q1112 198 1134 224 L1042 224 Z" fill="#0b1530" stroke="#5aa0ff" stroke-opacity=".7"/>'
             '<line x1="1036" y1="192" x2="1036" y2="270" stroke="#6EA8FF" stroke-opacity=".45"/>'
             '<rect x="801" y="252" width="8" height="6" rx="2" fill="#F0566A"/>'
             '<ellipse cx="1212" cy="248" rx="6" ry="4" fill="#ffffff"/>'
             '<path d="M880 244 L1196 244" stroke="#38BDF8" stroke-opacity=".7" stroke-width="2"/>')
    for wx in (872, 1146):
        p.append(f'<circle cx="{wx}" cy="270" r="27" fill="#060a1a" stroke="#6EA8FF" stroke-width="3"/>'
                 f'<circle cx="{wx}" cy="270" r="12" fill="#1a2750" stroke="#38BDF8" stroke-opacity=".8"/>')
    # floating data dots
    for _ in range(46):
        p.append(f'<circle cx="{rng.uniform(600, 1400):.0f}" cy="{rng.uniform(40, 330):.0f}" r="{rng.uniform(1, 2.4):.1f}" fill="#C9B5FF" opacity="{rng.uniform(.12, .5):.2f}"/>')
    p.append('</g></svg>')
    return base64.b64encode("".join(p).encode()).decode()


@st.cache_data(show_spinner=False)
def scene_uri(kind):
    builder = {"exec": _scene_campus, "analytics": _scene_analytics, "ml": _scene_ml}[kind]
    return f"data:image/svg+xml;base64,{builder()}"


@st.cache_data(show_spinner=False)
def file_uri(path, mtime, max_width=1600):
    """Data URI of an image, downscaled so large photos stay light."""
    try:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        if img.width > max_width:
            img = img.resize((max_width, round(img.height * max_width / img.width)), Image.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, "JPEG", quality=80, optimize=True)
        return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
    except Exception:
        ext = Path(path).suffix.lstrip(".").lower()
        return f"data:image/{'jpeg' if ext == 'jpg' else ext};base64,{b64_file(path)}"


HERO_KEYWORDS = {"exec": (), "analytics": ("analytics",), "ml": ("hub", "smart_hub", "ev_hub")}
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def hero_image(kind):
    """(uri, background-position, is_photo). Uses your photo from assets/ when found, else a generated illustration.

    Looks for hero_<page>.jpg/png first, then any image whose name contains a keyword
    (for example 'EV HUB - TAYLORS.png' for the ML page, 'taylors_banner...' for the first page).
    """
    names = [f"hero_{kind}{ext}" for ext in IMAGE_EXTS]
    if kind == "exec":
        names += [f"campus_hero{ext}" for ext in IMAGE_EXTS]
    photo = find_asset(names)
    if photo is None and ASSETS.exists():
        words = HERO_KEYWORDS[kind]
        photo = next((f for f in sorted(ASSETS.iterdir()) if f.suffix.lower() in IMAGE_EXTS and "logo" not in f.name.lower()
                      and any(w in re.sub(r"[\s\-]+", "_", f.name.lower()) for w in words)), None)
    if photo:
        return file_uri(str(photo), photo.stat().st_mtime), "center 45%", True
    return scene_uri(kind), "center right", False


# ============================================================
# CHART HELPERS
# ============================================================
def _plain(values):
    """Plain Python lists so Plotly never mis-reads pandas/numpy dtypes."""
    ser = pd.Series(list(values)) if isinstance(values, (list, tuple)) else pd.Series(values)
    if pd.api.types.is_datetime64_any_dtype(ser):
        return ser.dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
    if pd.api.types.is_bool_dtype(ser):
        return ser.astype(str).tolist()
    if pd.api.types.is_numeric_dtype(ser):
        return ser.astype(float).tolist()
    return ser.astype(str).tolist()


def show(fig, height=340):
    """Draw a themed Plotly figure that fits its card (no toolbar, no scrollbars)."""
    for trace in fig.data:
        for attr in ("x", "y"):
            val = getattr(trace, attr, None)
            if val is not None and getattr(trace, "type", "") != "heatmap":
                try:
                    setattr(trace, attr, _plain(val))
                except Exception:
                    pass
    axis = dict(gridcolor=GRID, linecolor=LINE, zeroline=False, showgrid=True, ticks="outside", ticklen=4,
                tickcolor=LINE, tickfont=dict(size=12, color=MUTED), showticklabels=True, automargin=True,
                title_standoff=10, title_font=dict(size=12, color=MUTED),
                exponentformat="none", showexponent="none", separatethousands=True)
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    fig.update_layout(height=height, template="ev", autosize=True, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family=FONT, color=MUTED, size=12))
    if fig.layout.margin.l is None:
        fig.update_layout(margin=dict(l=64, r=26, t=24, b=64))
    st.plotly_chart(fig, **STRETCH["plotly_chart"], theme=None,
                    config={"displayModeBar": False, "responsive": True, "displaylogo": False})


def panel_title(icon, title, sub=""):
    sub_html = f'<div class="psub">{escape(sub)}</div>' if sub else ""
    render(f'<div class="phead"><div class="ptitle"><span class="pico">{icon}</span>{escape(title)}</div>{sub_html}</div>')


def html_table(headers, rows, wrap_class="tbl-wrap"):
    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(cells) + "</tr>" for cells in rows)
    render(f'<div class="{wrap_class}"><table class="tbl"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>')


# ============================================================
# KPI + BANNER COMPONENTS
# ============================================================
def sparkline(values, color, uid, w=130, h=46):
    v = np.array([x for x in values if x is not None and not np.isnan(x)], dtype=float)
    if len(v) < 2:
        return ""
    span = (v.max() - v.min()) or 1.0
    xs = np.linspace(2, w - 2, len(v))
    ys = h - 6 - (v - v.min()) / span * (h - 14)
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    area = f"M{xs[0]:.1f},{h} " + " ".join(f"L{x:.1f},{y:.1f}" for x, y in zip(xs, ys)) + f" L{xs[-1]:.1f},{h} Z"
    return (f'<svg class="spark" viewBox="0 0 {w} {h}" preserveAspectRatio="none"><defs>'
            f'<linearGradient id="sg{uid}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{color}" stop-opacity=".35"/>'
            f'<stop offset="1" stop-color="{color}" stop-opacity="0"/></linearGradient></defs>'
            f'<path d="{area}" fill="url(#sg{uid})"/><polyline points="{pts}" fill="none" stroke="{color}" '
            f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/></svg>')


def series_and_delta(data, fn, buckets=12):
    """`fn` evaluated on 12 slices (sparkline) and the % change between the two halves of the period."""
    if data.empty:
        return [], None
    data = data.sort_values("Start_Dt") if HAS_DT else data
    parts = np.array_split(np.arange(len(data)), min(buckets, len(data)))
    series = [fn(data.iloc[p]) if len(p) else np.nan for p in parts]
    if len(data) < 2:
        return series, None
    first, second = np.array_split(np.arange(len(data)), 2)
    a, b = fn(data.iloc[first]), fn(data.iloc[second])
    delta = None if (a == 0 or np.isnan(a) or np.isnan(b)) else (b - a) / abs(a) * 100
    return series, delta


def kpi(col, tone, icon, label, value, unit, series, delta, uid):
    accent = {"green": GREEN, "blue": BLUE, "purple": PURPLE, "orange": ORANGE}[tone]
    if delta is None:
        change = f'<div class="kd flat">{tr("n/a", "t/a", "无")}</div>'
    else:
        change = f'<div class="kd {"up" if delta >= 0 else "down"}">{"▲" if delta >= 0 else "▼"} {abs(delta):.1f}%</div>'
    with col:
        render(f"""
        <div class="kpi {tone}">
            <div class="ktop"><div class="kico">{icon}</div>
            <div><div class="klabel">{label}</div><div class="kvalue">{value}<span>{unit}</span></div></div></div>
            <div class="kbot"><div>{change}<div class="kvs">{tr("vs previous half", "berbanding separuh sebelumnya", "对比前半段")}</div></div>
            {sparkline(series, accent, uid)}</div>
        </div>
        """)


def hero(kind, eyebrow, title, desc):
    logo = find_logo()
    if logo:
        ext = "jpeg" if logo.suffix.lower() in (".jpg", ".jpeg") else logo.suffix.lstrip(".").lower()
        mark = f'<img src="data:image/{ext};base64,{b64_file(logo)}" alt="logo"/>'
    else:
        mark = '<div class="mono">T</div>'
    uri, position, is_photo = hero_image(kind)
    shade = ("rgba(8,13,28,.90) 0%, rgba(8,13,28,.62) 38%, rgba(8,13,28,.04) 72%" if is_photo
             else "rgba(8,13,28,.94) 0%, rgba(8,13,28,.72) 44%, rgba(8,13,28,.08) 100%")
    style = (f"background-image: linear-gradient(90deg, {shade}), url('{uri}'); "
             f"background-size: cover; background-position: {position};")
    render(f"""
    <div class="hero{' photo' if is_photo else ''}" style="{style}">
        <div class="hero-body">
            <div class="hero-eyebrow">{escape(eyebrow)}</div>
            <div class="hero-title">{escape(title)}</div>
            <div class="hero-desc">{escape(desc)}</div>
        </div>
        <div class="hero-logo">{mark}<div><b>Taylor's University</b><small>{tr("Smart Campus", "Kampus Pintar", "智慧校园")}</small></div></div>
    </div>
    """)


# ============================================================
# ML ENGINE GUIDE (inline, open the first time the page is visited)
# ============================================================
def ml_guide():
    first_visit = not st.session_state.get("ml_guide_seen", False)
    st.session_state["ml_guide_seen"] = True
    steps = [
        (tr("Pick a scenario", "Pilih senario", "选择场景"),
         tr("Click one of the 8 ready-made scenarios, load a random real session from your data, or reset to defaults. The blue button shows what is active.",
            "Klik salah satu daripada 8 senario sedia ada, muat sesi sebenar rawak daripada data anda, atau set semula kepada lalai. Butang biru menunjukkan senario aktif.",
            "点击 8 个现成场景之一，随机载入真实会话，或恢复默认。蓝色按钮表示当前场景。")),
        (tr("Adjust the inputs", "Laraskan input", "调整输入"),
         tr("Change duration, energy, power, battery temperature, overstay or grid status on the left. The result updates instantly, so there is no Run button.",
            "Ubah tempoh, tenaga, kuasa, suhu bateri, tinggal lebih masa atau status grid di sebelah kiri. Keputusan dikemas kini serta-merta, jadi tiada butang Jalankan.",
            "在左侧修改时长、能量、功率、电池温度、超时占位或电网状态。结果即时更新，无需点击运行。")),
        (tr("Read the result", "Baca keputusan", "解读结果"),
         tr("The card shows the prediction and the gauge a 0 to 100 risk score. Below it you see average power, utilisation, cost, penalty and CO₂.",
            "Kad menunjukkan ramalan dan tolok menunjukkan skor risiko 0 hingga 100. Di bawahnya anda boleh lihat purata kuasa, penggunaan, kos, penalti dan CO₂.",
            "卡片显示预测结果，仪表盘显示 0–100 风险评分。下方可查看平均功率、利用率、费用、罚款和 CO₂。")),
        (tr("Act on it", "Ambil tindakan", "采取行动"),
         tr("The bar chart shows which factor drives the risk. The action cards tell the operator what to do, most urgent first. A red banner means stop charging.",
            "Carta bar menunjukkan faktor yang mendorong risiko. Kad tindakan memberitahu operator apa yang perlu dilakukan, paling mendesak dahulu. Sepanduk merah bermaksud henti pengecasan.",
            "柱状图显示哪个因素推高风险。行动卡片按紧急程度告诉运维该怎么做。红色横幅表示应停止充电。")),
    ]
    terms = [
        (tr("Overstay", "Tinggal lebih masa", "超时占位"),
         tr("Minutes a car stays plugged in after charging has finished.", "Minit kereta kekal bersambung selepas pengecasan selesai.", "充电结束后车辆仍占用充电桩的分钟数。")),
        (tr("Utilisation", "Penggunaan", "利用率"),
         tr("Energy delivered compared with what the rated power could deliver in that time.", "Tenaga dibekalkan berbanding apa yang boleh dibekalkan oleh kuasa terkadar dalam tempoh itu.", "实际输出能量与额定功率在该时段内可输出能量之比。")),
        (tr("SoC (state of charge)", "SoC (keadaan cas)", "SoC（电量状态）"),
         tr("Battery level when charging starts.", "Paras bateri apabila pengecasan bermula.", "开始充电时的电池电量。")),
        (tr("Rule-based", "Berasaskan peraturan", "规则模型"),
         tr("The score comes from transparent rules. A trained model is used automatically if ml_model.py provides one.",
            "Skor datang daripada peraturan yang telus. Model terlatih digunakan secara automatik jika ml_model.py menyediakannya.",
            "评分来自透明规则。若 ml_model.py 提供训练模型，将自动使用。")),
    ]
    cards = "".join(f'<div class="gcard"><span class="gnum">{i}</span><div class="gtitle">{escape(t)}</div><div class="gtext">{escape(x)}</div></div>'
                    for i, (t, x) in enumerate(steps, 1))
    glossary = "".join(f'<div class="gterm"><b>{escape(t)}</b><span>{escape(x)}</span></div>' for t, x in terms)
    intro = tr("Predict how healthy a charging session is, see what the operator should do about it and what it costs. Nothing here changes real chargers, so experiment freely.",
               "Ramal kesihatan sesi pengecasan, lihat apa yang operator patut lakukan dan berapa kosnya. Tiada apa di sini mengubah pengecas sebenar, jadi bebas mencuba.",
               "预测充电会话的健康状况，查看运维人员应采取的措施及相应成本。这里的操作不会影响真实充电桩，可放心尝试。")
    legend = (f'<span class="chip good">0 – 25 · {tr("Healthy", "Sihat", "健康")}</span>'
              f'<span class="chip warn">25 – 55 · {tr("Monitor", "Pantau", "关注")}</span>'
              f'<span class="chip bad">55 – 100 · {tr("Action needed", "Tindakan diperlukan", "需要处理")}</span>')
    with st.expander("📘 " + tr("How to use the ML Engine", "Cara menggunakan Enjin ML", "如何使用机器学习引擎"), expanded=first_visit):
        render(f"""
        <div class="gintro">{intro}</div>
        <div class="ggrid">{cards}</div>
        <div class="glegend"><b>{tr("Risk score", "Skor risiko", "风险评分")}</b>{legend}</div>
        <div class="gterms">{glossary}</div>
        """)


# ============================================================
# STYLES (all colours come from the active theme)
# ============================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp { font-family: __FONT__; color-scheme: __SCHEME__; }
.stApp { background: __BG__; color: __TEXT__; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1500px; }
h1, h2, h3, h4 { color: __TEXT__ !important; }
.main p, .main label, .main li, .main span, .main summary { color: __TEXT__; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] { background: __SIDE__; border-right: 1px solid __LINE__; }
[data-testid="stSidebar"] * { color: __TEXT__; }
[data-testid="stSidebar"] h3 { font-size: 15px; font-weight: 700; margin: 0 0 8px 0; }
[data-testid="stSidebar"] label p { color: __MUTED__ !important; font-size: 12.5px; font-weight: 600; }
[data-testid="stSidebar"] div[data-baseweb="select"] > div, [data-testid="stSidebar"] div[data-baseweb="input"] {
    background: __PANEL2__ !important; border: 1px solid __INPUT__ !important; border-radius: 8px; }
[data-testid="stSidebar"] div[data-baseweb="select"] *, [data-testid="stSidebar"] input { color: __TEXT__ !important; }
[data-testid="stSidebar"] .stButton > button, [data-testid="stPopover"] > div > button {
    background: __PANEL2__; color: __TEXT__; border: 1px solid __INPUT__; border-radius: 10px; font-weight: 600;
    width: 100%; min-height: 42px; padding: .3rem .6rem; font-size: 13px; }
[data-testid="stSidebar"] .stButton > button:hover, [data-testid="stPopover"] > div > button:hover { background: __HOVER__; border-color: __BLUE__; }
[data-testid="stSidebar"] .stButton > button[kind="primary"], [data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] {
    background: __BLUE__; border-color: __BLUE__; }
[data-testid="stSidebar"] .stButton > button[kind="primary"] *, [data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] * { color: #fff !important; }
[class*="st-key-nav_"] button, [class*="st-key-nav_"] button > div { justify-content: flex-start; text-align: left; }
[class*="st-key-nav_"] button p { text-align: left; width: 100%; }
[data-testid="stSidebar"] [data-testid="stToggle"] { background: __PANEL2__; border: 1px solid __INPUT__; border-radius: 10px; min-height: 42px;
    padding: 0 10px; display: flex; align-items: center; justify-content: center; }
[data-testid="stSidebar"] [data-testid="stToggle"] p { color: __TEXT__ !important; font-size: 13px; font-weight: 600; }
.mono-logo { width: 64px; height: 64px; border-radius: 12px; background: #B3122D; color: #fff !important; display: grid; place-items: center; font-weight: 800; font-size: 30px; }
.uni-name { font-weight: 800; font-size: 14px; letter-spacing: .04em; margin-top: 12px; color: __TEXT__ !important; }
.uni-sub { color: __MUTED__ !important; font-size: 12px; margin-top: 2px; }
.rule { height: 1px; background: __LINE__; margin: 18px 0; }
.green-card { border: 1px solid __LINE__; border-radius: 12px; padding: 14px 16px; background: __PANEL2__; display: flex; gap: 12px; align-items: center; }
.green-card .leaf { font-size: 26px; }
.green-card b { font-size: 14px; color: __TEXT__ !important; }
.green-card small { color: __MUTED__ !important; font-size: 11.5px; display: block; margin-top: 2px; }

/* ---------- Banner ---------- */
.hero { position: relative; border-radius: 18px; overflow: hidden; border: 1px solid __LINE__; padding: 40px 44px; margin-bottom: 14px;
    min-height: 210px; display: flex; align-items: center; background-color: #0A1020; }
.hero-body { position: relative; z-index: 1; max-width: calc(100% - 290px); }
.hero-eyebrow { color: #9DB0DD !important; font-size: 12px; font-weight: 700; letter-spacing: .18em; }
.hero-title { color: #fff !important; font-size: 40px; font-weight: 800; letter-spacing: -0.02em; margin-top: 10px; line-height: 1.1; }
.hero-desc { color: #D3DCF2 !important; font-size: 15.5px; margin-top: 10px; max-width: 620px; }
.hero-logo { position: absolute; top: 50%; transform: translateY(-50%); right: 26px; z-index: 2; display: flex; align-items: center; gap: 12px;
    background: rgba(12,18,38,.78); border: 1px solid #2A3866; border-radius: 12px; padding: 10px 16px; backdrop-filter: blur(6px); }
.hero-logo img { height: 36px; border-radius: 6px; background: #fff; padding: 2px; }
.hero-logo .mono { width: 36px; height: 36px; border-radius: 8px; background: #B3122D; color: #fff; display: grid; place-items: center; font-weight: 800; }
.hero-logo b { color: #fff !important; font-size: 14px; display: block; }
.hero-logo small { color: #9DB0DD !important; font-size: 12px; }
.hero.photo { aspect-ratio: 3 / 1; min-height: 260px; }
.hero.photo .hero-body { max-width: 44%; }
.hero.photo .hero-title, .hero.photo .hero-desc { text-shadow: 0 2px 12px rgba(0,0,0,.55); }
@media (max-width: 1150px) { .hero.photo .hero-body { max-width: 100%; } .hero-logo { display: none; } .hero-body { max-width: 100%; } .hero-title { font-size: 32px; } }

/* ---------- KPI cards ---------- */
.kpi { border-radius: 14px; padding: 18px 20px 14px 20px; border: 1px solid var(--bd); min-height: 150px;
    background: linear-gradient(135deg, var(--tint) 0%, __PANEL__ 62%); }
.kpi.green  { --bd: rgba(61,214,140,.38); --tint: rgba(61,214,140,.20); --ic: #2FBF77; }
.kpi.blue   { --bd: rgba(79,124,255,.40); --tint: rgba(79,124,255,.20); --ic: #4F7CFF; }
.kpi.purple { --bd: rgba(166,107,255,.40); --tint: rgba(166,107,255,.20); --ic: #9B5CF6; }
.kpi.orange { --bd: rgba(245,158,66,.40); --tint: rgba(245,158,66,.20); --ic: #F08A24; }
.ktop { display: flex; gap: 14px; align-items: center; }
.kico { width: 46px; height: 46px; border-radius: 11px; background: var(--ic); display: grid; place-items: center; font-size: 22px; flex: none; }
.klabel { color: __MUTED__ !important; font-size: 13.5px; font-weight: 500; }
.kvalue, .kvalue span { color: __TEXT__ !important; }
.kvalue { font-size: 27px; font-weight: 800; letter-spacing: -0.02em; margin-top: 2px; white-space: nowrap; }
.kvalue span { font-size: 17px; font-weight: 600; margin-left: 6px; }
.kbot { display: flex; justify-content: space-between; align-items: flex-end; margin-top: 14px; }
.kd { font-weight: 700; font-size: 13.5px; } .kd.up { color: __GREEN__ !important; } .kd.down { color: __RED__ !important; } .kd.flat { color: __MUTED__ !important; }
.kvs { color: __MUTED__ !important; font-size: 12.5px; margin-top: 2px; }
.spark { width: 130px; height: 46px; }

/* ---------- Panels, charts, headings ---------- */
.phead { min-height: 52px; display: flex; flex-direction: column; justify-content: center; margin: 4px 0 0 4px; }
.ptitle { font-weight: 700; font-size: 16px; color: __TEXT__ !important; line-height: 1.25; }
.pico { margin-right: 10px; }
.psub { color: __DIM__ !important; font-size: 12.5px; margin: 2px 0 0 0; }
[data-testid="stPlotlyChart"] { background: __PANEL__; border: 1px solid __LINE__; border-radius: 14px; padding: 0; margin-top: 8px; overflow: hidden; box-sizing: border-box; width: 100%; }
[data-testid="stPlotlyChart"], [data-testid="stPlotlyChart"] * { scrollbar-width: none !important; -ms-overflow-style: none; box-sizing: border-box; }
[data-testid="stPlotlyChart"] *::-webkit-scrollbar { display: none !important; width: 0; height: 0; }
[data-testid="stPlotlyChart"] > div, [data-testid="stPlotlyChart"] > div > div { width: 100% !important; max-width: 100%; overflow: hidden !important; }
.js-plotly-plot, .plotly, .plot-container, .svg-container { overflow: hidden !important; background: transparent !important; max-width: 100% !important; }
.section { font-size: 19px; font-weight: 800; color: __TEXT__ !important; margin: 26px 0 4px 2px; }
.section-sub { font-size: 13px; color: __MUTED__ !important; margin: 0 0 10px 2px; }
.step { display: inline-grid; place-items: center; width: 28px; height: 28px; border-radius: 50%; background: __BLUE__; color: #fff !important; font-size: 14px; margin-right: 12px; font-weight: 800; }
.grp { color: __MUTED__ !important; font-weight: 700; font-size: 12.5px; letter-spacing: .06em; text-transform: uppercase; margin: 4px 0 8px 0; }

/* ---------- Table ---------- */
.tbl-wrap { background: __PANEL__; border: 1px solid __LINE__; border-radius: 14px; overflow-x: auto; margin-top: 8px; }
table.tbl { width: 100%; border-collapse: collapse; font-size: 13.5px; }
.tbl th { background: __PANEL2__; color: __MUTED__; text-align: left; font-weight: 600; padding: 13px 16px; white-space: nowrap; border-bottom: 1px solid __LINE__; }
.tbl td { color: __TEXT__; padding: 13px 16px; border-bottom: 1px solid __LINE__; white-space: nowrap; }
.tbl tr:last-child td { border-bottom: none; }
.tbl tr:hover td { background: __HOVER__; }
.tbl td.id { font-weight: 700; }
.dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 8px; }
.badge { display: inline-block; padding: 4px 14px; border-radius: 8px; font-size: 12.5px; font-weight: 600; }
.badge.peak { background: rgba(245,158,66,.18); color: __ORANGE__; border: 1px solid rgba(245,158,66,.40); }
.badge.off  { background: rgba(79,124,255,.16); color: __BLUE__; border: 1px solid rgba(79,124,255,.40); }
.badge.std  { background: rgba(154,168,199,.16); color: __MUTED__; border: 1px solid rgba(154,168,199,.35); }
.tbl-foot { color: __MUTED__ !important; font-size: 13px; }
.pager { color: __TEXT__ !important; text-align: center; font-weight: 600; font-size: 13px; }

/* ---------- Widgets ---------- */
[data-testid="stMetric"] { min-height: 104px; background: __PANEL__; border: 1px solid __LINE__; border-radius: 12px; padding: 14px 18px; }
[data-testid="stMetricLabel"] p { color: __MUTED__ !important; font-weight: 600; font-size: 13px; }
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * { color: __TEXT__ !important; font-weight: 800; }
[data-testid="stMetricValue"] { font-size: 24px; }
[data-testid="stExpander"] { background: __PANEL__; border: 1px solid __LINE__; border-radius: 12px; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * { color: __TEXT__ !important; }
[data-testid="stVerticalBlockBorderWrapper"] { border-color: __LINE__; }
.main div[data-baseweb="select"] > div, .main div[data-baseweb="input"], .main .stTextInput input, .main .stNumberInput input {
    background: __PANEL2__ !important; border: 1px solid __INPUT__ !important; border-radius: 8px; color: __TEXT__ !important; min-height: 46px; }
.main .stNumberInput button { background: __PANEL2__; color: __TEXT__; border-color: __INPUT__; }
.main div[data-baseweb="select"] * { color: __TEXT__ !important; }
div[data-baseweb="popover"] ul, div[data-baseweb="popover"] li { background: __PANEL__ !important; color: __TEXT__ !important; }
div[data-baseweb="popover"] li:hover { background: __HOVER__ !important; }
[data-testid="stPopoverBody"], div[role="dialog"] { background: __PANEL__ !important; border: 1px solid __LINE__; border-radius: 14px; }
[data-testid="stPopoverBody"] *, div[role="dialog"] * { color: __TEXT__; }
.main [data-testid="stSlider"] *, .main [role="radiogroup"] * { color: __TEXT__; }
.main .stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
    background: __BLUE__; color: #fff; border: none; border-radius: 9px; font-weight: 700; padding: .55rem 1.2rem;
    min-height: 46px; white-space: normal; line-height: 1.25; }
.main .stButton > button *, .stDownloadButton > button * { color: inherit; }
.main .stButton > button:hover, .stDownloadButton > button:hover { filter: brightness(1.08); color: #fff; }
.main .stButton > button[kind="secondary"], .main .stButton > button[data-testid="stBaseButton-secondary"] {
    background: __PANEL2__; color: __TEXT__; border: 1px solid __INPUT__; }
.main .stButton > button[kind="secondary"]:hover, .main .stButton > button[data-testid="stBaseButton-secondary"]:hover {
    background: __HOVER__; border-color: __BLUE__; color: __TEXT__; filter: none; }
.main .stButton > button[kind="primary"], .main .stButton > button[data-testid="stBaseButton-primary"] {
    background: __BLUE__; color: #fff; border: 1px solid __BLUE__; box-shadow: 0 0 0 3px rgba(79,124,255,.22); }
.main .stButton > button[kind="primary"] *, div[role="dialog"] button[kind="primary"] * { color: #fff !important; }
.main .stButton > button:disabled { opacity: .45; }
button:focus-visible, input:focus-visible { outline: 2px solid __BLUE__ !important; outline-offset: 2px; }
[data-testid="stAlert"] { border-radius: 10px; }

/* ---------- ML page ---------- */
.scn { background: __PANEL__; border: 1px solid __LINE__; border-left: 5px solid var(--tone); border-radius: 12px; padding: 14px 18px; margin: 10px 0 4px 0; display: flex; gap: 14px; align-items: center; flex-wrap: wrap; }
.scn-chip { background: var(--tone); color: #0A0F1F !important; font-weight: 800; font-size: 11.5px; padding: 3px 11px; border-radius: 999px; }
.scn-name { font-weight: 700; font-size: 15px; color: __TEXT__ !important; }
.scn-desc { color: __MUTED__ !important; font-size: 13.5px; }
.result { min-height: 192px; display: flex; flex-direction: column; justify-content: center; background: __PANEL__; border: 1px solid __LINE__;
    border-left: 6px solid var(--tone); border-radius: 14px; padding: 20px 24px; margin: 8px 0 0 0; }
.result-tag { color: var(--tone) !important; font-weight: 800; font-size: 13px; }
.result-title { font-size: 24px; font-weight: 800; color: __TEXT__ !important; margin-top: 4px; }
.result-text { color: __MUTED__ !important; font-size: 14px; margin-top: 6px; line-height: 1.55; }
.alert-banner { background: linear-gradient(90deg, rgba(240,86,106,.30), rgba(240,86,106,.08)); border: 1px solid rgba(240,86,106,.6); border-radius: 12px; padding: 14px 18px; margin: 6px 0 10px 0; font-size: 14.5px; color: __TEXT__ !important; }
.alert-banner b { margin-right: 8px; letter-spacing: .04em; color: __RED__ !important; }
.act { background: __PANEL__; border: 1px solid __LINE__; border-left: 5px solid var(--tone); border-radius: 12px; padding: 12px 16px; margin-bottom: 10px; }
.act-tag { color: var(--tone) !important; font-size: 11.5px; font-weight: 800; text-transform: uppercase; letter-spacing: .08em; }
.act-title { font-weight: 700; font-size: 15px; margin-top: 2px; color: __TEXT__ !important; }
.act-text { color: __MUTED__ !important; font-size: 13px; margin-top: 4px; line-height: 1.5; }
.gintro { color: __MUTED__ !important; font-size: 14px; line-height: 1.6; margin: 4px 0 14px 0; }
.ggrid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; }
.gcard { background: __PANEL2__; border: 1px solid __LINE__; border-radius: 12px; padding: 16px 16px 14px 16px; }
.gnum { display: inline-grid; place-items: center; width: 26px; height: 26px; border-radius: 50%; background: __BLUE__; color: #fff !important; font-weight: 800; font-size: 13px; }
.gtitle { font-weight: 700; font-size: 15px; margin: 10px 0 4px 0; color: __TEXT__ !important; }
.gtext { color: __MUTED__ !important; font-size: 13px; line-height: 1.55; }
.glegend { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin: 16px 0 4px 0; }
.glegend b { color: __TEXT__ !important; margin-right: 6px; font-size: 13.5px; }
.chip { padding: 4px 12px; border-radius: 999px; font-size: 12.5px; font-weight: 700; border: 1px solid transparent; }
.chip.good { background: rgba(61,214,140,.16); color: __GREEN__ !important; border-color: rgba(61,214,140,.4); }
.chip.warn { background: rgba(242,181,68,.16); color: __AMBER__ !important; border-color: rgba(242,181,68,.4); }
.chip.bad { background: rgba(240,86,106,.16); color: __RED__ !important; border-color: rgba(240,86,106,.4); }
.gterms { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 10px 22px; margin-top: 14px; padding-top: 14px; border-top: 1px solid __LINE__; }
.gterm b { display: block; color: __TEXT__ !important; font-size: 13px; }
.gterm span { color: __MUTED__ !important; font-size: 12.5px; line-height: 1.5; }
.card-title { font-weight: 700; font-size: 16px; color: __TEXT__ !important; }
[class*="st-key-session_card"] { background: __PANEL__; border: 1px solid __LINE__; border-radius: 14px; padding: 18px 18px 14px 18px; }
.tbl-wrap.flat { margin-top: 6px; border-radius: 10px; background: __PANEL__; }
[data-testid="stMarkdownContainer"] table.tbl th, [data-testid="stMarkdownContainer"] table.tbl td { border: none; border-bottom: 1px solid __LINE__; font-size: 13px; padding: 10px 14px; }
[data-testid="stMarkdownContainer"] table.tbl th { font-size: 12.5px; padding: 11px 14px; background: __PANEL2__; }
[data-testid="stMarkdownContainer"] table.tbl tr:last-child td { border-bottom: none; }
.tbl th:first-child, .tbl td:first-child { position: sticky; left: 0; z-index: 1; }
.tbl td:first-child { background: __PANEL__; }
.tbl th:first-child { background: __PANEL2__; }
.tbl tr:hover td:first-child { background: __HOVER__; }
[class*="st-key-pager"] button { min-height: 34px; padding: 0; font-size: 13px; border-radius: 8px; box-shadow: none !important; }
.footer { text-align: center; color: __DIM__; font-size: 12.5px; padding: 26px 0 6px 0; margin-top: 34px; border-top: 1px solid __LINE__; }
</style>
"""
_tokens = {**{k: THEMES[MODE][k] for k in PALETTE_KEYS}, "FONT": FONT}
for _name, _value in _tokens.items():
    CSS = CSS.replace(f"__{_name}__", _value)
st.markdown(CSS, unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
PAGES = {
    "exec": tr("🏢 E-Taylor's Core Metrics", "🏢 Metrik Utama E-Taylor's", "🏢 E-Taylor's 核心指标"),
    "analytics": tr("⚡ Smart Grid Analytics", "⚡ Analitik Grid Pintar", "⚡ 智能电网分析"),
    "ml": tr("🤖 EV Charge ML Engine", "🤖 Enjin ML Pengecasan EV", "🤖 电动车充电机器学习引擎"),
}
st.session_state.setdefault("page", "exec")
for _k in ("f_charger", "f_grid", "f_peak"):
    st.session_state.setdefault(_k, "All")
if HAS_DT:
    st.session_state.setdefault("f_date", (DATE_MIN, DATE_MAX))


def go_to(page_id):
    st.session_state["page"] = page_id


def reset_filters():
    for key in ("f_charger", "f_grid", "f_peak"):
        st.session_state[key] = "All"
    if HAS_DT:
        st.session_state["f_date"] = (DATE_MIN, DATE_MAX)
    st.session_state["tbl_page"] = 1


with st.sidebar:
    logo_col, tools_col = cols([1, 1.9])
    with logo_col:
        _logo = find_logo()
        if _logo:
            st.image(str(_logo), width=84)
        else:
            render('<div class="mono-logo">T</div>')
    with tools_col:
        lang_box, theme_box = st.columns(2, gap="small")
        with lang_box:
            if hasattr(st, "popover"):
                language_menu = st.popover(f"{icon('translate', '🌐')} {LANG_CODES[LANG]}", **STRETCH["popover"])
            else:
                language_menu = st.expander(f"🌐 {LANG_CODES[LANG]}")
            with language_menu:
                st.radio(tr("Language", "Bahasa", "语言"), list(LANGS), format_func=LANGS.get, key="lang")
        with theme_box:
            st.toggle(tr("Dark", "Gelap", "深色"), key="dark_mode")
    render(f"""
    <div class="uni-name">TAYLOR'S UNIVERSITY</div>
    <div class="uni-sub">{tr("Smart Campus • EV Analytics", "Kampus Pintar • Analitik EV", "智慧校园 • 电动车分析")}</div>
    <div class="rule"></div>
    """)

    st.markdown("### " + tr("Navigation", "Navigasi", "导航"))
    for page_id, label in PAGES.items():
        st.button(label, key=f"nav_{page_id}", **STRETCH["button"], on_click=go_to, args=(page_id,),
                  type="primary" if st.session_state["page"] == page_id else "secondary")

    render('<div class="rule"></div>')
    st.markdown("### " + tr("Filters", "Penapis", "筛选"))
    date_range = None
    if HAS_DT:
        date_range = st.date_input(tr("Date range", "Julat tarikh", "日期范围"), min_value=DATE_MIN, max_value=DATE_MAX, key="f_date")
    f_charger = st.selectbox(tr("Charger type", "Jenis pengecas", "充电桩类型"), options("Charger_Type"), key="f_charger", format_func=all_label)
    f_grid = st.selectbox(tr("Grid status", "Status grid", "电网状态"), options("Grid_Status_Label"), key="f_grid", format_func=all_label)
    f_peak = st.selectbox(tr("Peak / non-peak", "Puncak / bukan puncak", "高峰 / 非高峰"), options("Peak_Hour_Label"), key="f_peak",
                          format_func=lambda v: all_label(v) if v == "All" else pk(v))
    st.button(tr("↻  Reset filters", "↻  Set semula penapis", "↻  重置筛选"), on_click=reset_filters, **STRETCH["button"])

    render('<div class="rule"></div>')
    render(f"""
    <div class="green-card"><div class="leaf">🌿</div>
    <div><b>{tr("Building a Greener Campus", "Membina Kampus Lebih Hijau", "共建更绿色的校园")}</b>
    <small>{tr("Sustainable • Smart • Electric", "Mampan • Pintar • Elektrik", "可持续 • 智能 • 电动")}</small></div></div>
    """)

page = st.session_state["page"]

# ---------- apply filters ----------
fdf = df
if HAS_DT and isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    fdf = fdf[(fdf["Start_Dt"] >= pd.Timestamp(date_range[0])) & (fdf["Start_Dt"] < pd.Timestamp(date_range[1]) + pd.Timedelta(days=1))]
if f_charger != "All" and HAS_CHARGER:
    fdf = fdf[fdf["Charger_Type"].astype(str).str.strip() == f_charger]
if f_grid != "All":
    fdf = fdf[fdf["Grid_Status_Label"] == f_grid]
if f_peak != "All":
    fdf = fdf[fdf["Peak_Hour_Label"] == f_peak]


def need_data():
    if fdf.empty:
        st.warning(tr("No sessions match the current filters. Reset the filters to see data.",
                      "Tiada sesi sepadan dengan penapis semasa. Set semula penapis untuk melihat data.",
                      "当前筛选条件下没有会话。请重置筛选。"))
        st.stop()


# ============================================================
# PAGE 1 - EXECUTIVE KPI DASHBOARD
# ============================================================
if page == "exec":
    hero("exec", tr("EV CHARGING ANALYTICS", "ANALITIK PENGECASAN EV", "电动车充电分析"),
         tr("Executive KPI Dashboard", "Papan Pemuka KPI Eksekutif", "管理层 KPI 仪表板"),
         tr("Real-time insights into campus EV charging operations",
            "Pandangan masa nyata tentang operasi pengecasan EV kampus", "校园电动车充电运营的实时洞察"))
    need_data()

    energy_s, energy_d = series_and_delta(fdf, lambda x: x["Energy_kWh"].sum())
    dur_s, dur_d = series_and_delta(fdf, lambda x: x["Duration_Mins"].mean() / 60)
    rev_s, rev_d = series_and_delta(fdf, lambda x: x.loc[x["Peak_Hour_Label"] == "Peak", "Revenue_Val"].sum())
    cnt_s, cnt_d = series_and_delta(fdf, lambda x: float(len(x)))
    avg_hours = fdf["Duration_Mins"].mean() / 60
    avg_hours = 0.0 if np.isnan(avg_hours) else avg_hours
    peak_revenue = fdf.loc[fdf["Peak_Hour_Label"] == "Peak", "Revenue_Val"].sum()

    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, "green", "⚡", tr("Total Energy Delivered", "Jumlah Tenaga Dibekalkan", "总供电量"), f"{fdf['Energy_kWh'].sum():,.1f}", "kWh", energy_s, energy_d, "a")
    kpi(k2, "blue", "🕒", tr("Average Session Duration", "Purata Tempoh Sesi", "平均充电时长"), f"{avg_hours:,.2f}", tr("hours", "jam", "小时"), dur_s, dur_d, "b")
    kpi(k3, "purple", "💰", tr("Peak Revenue", "Hasil Waktu Puncak", "高峰收入"), f"RM {peak_revenue:,.2f}", "", rev_s, rev_d, "c")
    kpi(k4, "orange", "🔌", tr("Total Charging Sessions", "Jumlah Sesi Pengecasan", "充电会话总数"), f"{len(fdf):,}", "", cnt_s, cnt_d, "d")

    st.write("")
    c1, c2, c3 = st.columns([1.5, 1, 1])

    # --- energy over time
    with c1:
        head_l, head_r = cols([3, 1.3])
        with head_l:
            panel_title("⚡", tr("Energy Delivered by Day", "Tenaga Dibekalkan Mengikut Masa", "供电量趋势"),
                        tr("Total kWh over time", "Jumlah kWh dari semasa ke semasa", "随时间变化的总 kWh"))
        granularity = None
        if HAS_DT:
            names = {"Daily": tr("Daily", "Harian", "每日"), "Weekly": tr("Weekly", "Mingguan", "每周"), "Monthly": tr("Monthly", "Bulanan", "每月")}
            with head_r:
                granularity = st.selectbox("view", list(names), index=2, format_func=names.get, key="gran", label_visibility="collapsed")
            rule = {"Daily": "D", "Weekly": "W", "Monthly": "MS"}[granularity]
            series = fdf.set_index("Start_Dt")["Energy_kWh"].resample(rule).sum().reset_index()
            xs, ys, tick, x_title = series["Start_Dt"], series["Energy_kWh"], ("%b" if granularity == "Monthly" else "%d %b"), tr("Date", "Tarikh", "日期")
        else:
            xs, ys, tick, x_title = list(range(1, len(energy_s) + 1)), energy_s, None, tr("Session batch", "Kelompok sesi", "会话批次")
        fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(color=GREEN, width=2.5, shape="spline"),
                                   marker=dict(size=6, color=GREEN), fill="tozeroy", fillcolor="rgba(61,214,140,0.16)",
                                   hovertemplate="%{y:,.1f} kWh<extra></extra>"))
        fig.update_yaxes(title_text=tr("Energy (kWh)", "Tenaga (kWh)", "能量 (kWh)"), rangemode="tozero")
        fig.update_xaxes(title_text=x_title, **(dict(type="date", tickformat=tick) if tick else {}))
        show(fig, 330)

    # --- sessions by charger (donut)
    with c2:
        panel_title("🥧", tr("Charging Sessions by Charger Type", "Sesi Pengecasan Mengikut Jenis Pengecas", "按充电桩类型的会话数"),
                    tr("Share of sessions", "Peratusan sesi", "会话占比"))
        counts = (fdf["Charger_Type"] if HAS_CHARGER else fdf["Grid_Status_Label"]).astype(str).value_counts()
        if len(counts) > 4:
            counts = pd.concat([counts.iloc[:3], pd.Series({tr("Others", "Lain-lain", "其他"): counts.iloc[3:].sum()})])
        fig = go.Figure(go.Pie(labels=[f"{k}   {v:,}" for k, v in counts.items()], values=counts.values, hole=0.64, sort=False,
                               domain=dict(x=[0, 0.62]), marker=dict(colors=[BLUE, GREEN, ORANGE, PURPLE, "#38BDF8"][:len(counts)],
                                                                      line=dict(color=PANEL, width=3)),
                               textinfo="percent", textposition="inside", insidetextorientation="horizontal",
                               textfont=dict(color="#fff", size=12)))
        fig.add_annotation(x=0.31, y=0.5, showarrow=False, font=dict(color=MUTED, size=12),
                           text=f"<b style='font-size:22px;color:{TEXT}'>{len(fdf):,}</b><br>{tr('Sessions', 'Sesi', '会话')}")
        fig.update_layout(legend=dict(orientation="v", x=0.66, xanchor="left", y=0.5, yanchor="middle"))
        show(fig, 330)

    # --- peak vs non-peak revenue
    with c3:
        panel_title("📊", tr("Peak vs Non-peak Revenue", "Hasil Puncak lwn Bukan Puncak", "高峰与非高峰收入"),
                    tr("Revenue (RM)", "Hasil (RM)", "收入 (RM)"))
        if HAS_DT:
            key_col, x_title, tick = "Period", tr("Month", "Bulan", "月份"), "%b"
            grouped = fdf.assign(Period=fdf["Start_Dt"].dt.to_period("M").dt.to_timestamp())
        elif HAS_CHARGER:
            key_col, x_title, tick, grouped = "Charger_Type", tr("Charger type", "Jenis pengecas", "充电桩类型"), None, fdf
        else:
            key_col, x_title, tick, grouped = "Grid_Status_Label", tr("Grid status", "Status grid", "电网状态"), None, fdf
        grouped = grouped.groupby([key_col, "Peak_Hour_Label"], as_index=False)["Revenue_Val"].sum()
        grouped["Peak_Hour_Label"] = grouped["Peak_Hour_Label"].map(pk)
        fig = px.bar(grouped, x=key_col, y="Revenue_Val", color="Peak_Hour_Label", barmode="group",
                     color_discrete_map={pk("Peak"): BLUE, pk("Non-peak"): ORANGE, pk("Standard"): PURPLE},
                     labels={"Revenue_Val": tr("Revenue (RM)", "Hasil (RM)", "收入 (RM)"), key_col: x_title, "Peak_Hour_Label": ""})
        fig.update_traces(marker_line_width=0)
        fig.update_xaxes(title_text=x_title, **(dict(type="date", tickformat=tick) if tick else {}))
        fig.update_layout(legend=dict(orientation="h", x=0.5, xanchor="center", y=1.1), bargap=0.25,
                          margin=dict(l=60, r=20, t=46, b=64))
        show(fig, 330)

    # --- session table (card layout like the design mockup)
    HEADERS = {
        "Session ID": tr("Session ID", "ID Sesi", "会话编号"), "Start Time": tr("Start Time", "Masa Mula", "开始时间"),
        "End Time": tr("End Time", "Masa Tamat", "结束时间"), "Start Hour": tr("Start Hour", "Jam Mula", "开始时段"),
        "Duration (hrs)": tr("Duration (hrs)", "Tempoh (jam)", "时长（小时）"), "Energy (kWh)": tr("Energy (kWh)", "Tenaga (kWh)", "能量 (kWh)"),
        "Power (kW)": tr("Power (kW)", "Kuasa (kW)", "功率 (kW)"), "Charger Type": tr("Charger Type", "Jenis Pengecas", "充电桩类型"),
        "User Type": tr("User Type", "Jenis Pengguna", "用户类型"), "Overstay (min)": tr("Overstay (min)", "Tinggal Lebih (min)", "超时 (分钟)"),
        "Battery Temp (°C)": tr("Battery Temp (°C)", "Suhu Bateri (°C)", "电池温度 (°C)"), "Grid Status": tr("Grid Status", "Status Grid", "电网状态"),
        "Peak/Non-peak": tr("Peak/Non-peak", "Puncak/Bukan Puncak", "高峰/非高峰"), "Revenue (RM)": tr("Revenue (RM)", "Hasil (RM)", "收入 (RM)"),
    }
    NUMBER_FORMAT = {"Duration (hrs)": "{:.2f}", "Energy (kWh)": "{:.1f}", "Power (kW)": "{:.1f}", "Overstay (min)": "{:.0f}",
                     "Battery Temp (°C)": "{:.1f}", "Revenue (RM)": "{:,.2f}"}

    table = pd.DataFrame(index=fdf.index)
    table["Session ID"] = fdf[SESSION_ID_COL].astype(str) if SESSION_ID_COL else [f"S{int(i) + 1:06d}" for i in fdf.index]
    if HAS_DT:
        table["Start Time"] = fdf["Start_Dt"].dt.strftime("%Y-%m-%d %H:%M")
        table["End Time"] = fdf["End_Dt"].dt.strftime("%Y-%m-%d %H:%M")
    elif fdf["Hour_of_Day"].notna().any():
        table["Start Hour"] = fdf["Hour_of_Day"].map(lambda h: f"{int(h):02d}:00" if pd.notna(h) else "-")
    table["Duration (hrs)"] = fdf["Duration_Mins"] / 60
    table["Energy (kWh)"] = fdf["Energy_kWh"]
    if POWER_COL:
        table["Power (kW)"] = pd.to_numeric(fdf[POWER_COL], errors="coerce")
    if HAS_CHARGER:
        table["Charger Type"] = fdf["Charger_Type"].astype(str)
    if USER_COL:
        table["User Type"] = fdf[USER_COL].astype(str)
    if OVERSTAY_COL:
        table["Overstay (min)"] = pd.to_numeric(fdf[OVERSTAY_COL], errors="coerce")
    if HAS_TEMP:
        table["Battery Temp (°C)"] = fdf["Battery_Temp"]
    table["Grid Status"] = fdf["Grid_Status_Label"]
    table["Peak/Non-peak"] = fdf["Peak_Hour_Label"]
    table["Revenue (RM)"] = fdf["Revenue_Val"]

    def keyed_container(key):
        try:
            return st.container(key=key)
        except TypeError:  # older Streamlit: no keyed containers
            return st.container()

    def grid_cell(value):
        low = str(value).lower()
        color = (GREEN if any(w in low for w in ("normal", "stable", "ok")) else
                 AMBER if any(w in low for w in ("warn", "peak load")) else
                 RED if any(w in low for w in ("over", "err", "fail", "outage", "fault")) else BLUE)
        return f'<td><span class="dot" style="background:{color}"></span>{escape(str(value))}</td>'

    def body_cell(column, value):
        if column == "Session ID":
            return f'<td class="id">{escape(str(value))}</td>'
        if column == "Grid Status":
            return grid_cell(value)
        if column == "Peak/Non-peak":
            return f'<td><span class="badge {({"Peak": "peak", "Non-peak": "off"}).get(value, "std")}">{escape(pk(value))}</span></td>'
        if column in NUMBER_FORMAT:
            return "<td>-</td>" if pd.isna(value) else f"<td>{NUMBER_FORMAT[column].format(float(value))}</td>"
        return f"<td>{escape(str(value))}</td>"

    def page_window(current_page, total_pages):
        """Page numbers to show, with None marking a gap: 1 2 3 … 12."""
        if total_pages <= 7:
            return list(range(1, total_pages + 1))
        keep = {1, total_pages, current_page - 1, current_page, current_page + 1}
        if current_page <= 3:
            keep |= {2, 3, 4}
        if current_page >= total_pages - 2:
            keep |= {total_pages - 3, total_pages - 2, total_pages - 1}
        items, previous = [], 0
        for number in sorted(n for n in keep if 1 <= n <= total_pages):
            items += ([None] if number - previous > 1 else []) + [number]
            previous = number
        return items

    PAGE_SIZE = 8
    with keyed_container("session_card"):
        title_col, search_col, download_col = cols([2.6, 2, 0.5])
        title_col.markdown(f'<div class="card-title">🗃️ {tr("Filtered Charging Session Data", "Data Sesi Pengecasan Ditapis", "已筛选的充电会话数据")}</div>',
                           unsafe_allow_html=True)
        query = search_col.text_input("search", label_visibility="collapsed",
                                      placeholder=tr("Search session ID, charger type...", "Cari ID sesi, jenis pengecas...", "搜索会话编号、充电桩类型…"))
        if query:
            table = table[table.astype(str).apply(lambda r: r.str.contains(query, case=False, na=False).any(), axis=1)]
        download_col.download_button(icon("download", "⬇"), table.to_csv(index=False).encode("utf-8"), "ev_sessions_filtered.csv", "text/csv",
                                     help=tr("Download as CSV", "Muat turun sebagai CSV", "下载 CSV"), **STRETCH["download_button"])

        signature = (f_charger, f_grid, f_peak, str(date_range), query)
        if st.session_state.get("tbl_sig") != signature:
            st.session_state["tbl_sig"], st.session_state["tbl_page"] = signature, 1
        n_rows = len(table)
        n_pages = max(1, math.ceil(n_rows / PAGE_SIZE))
        current = min(max(1, st.session_state.get("tbl_page", 1)), n_pages)
        st.session_state["tbl_page"] = current
        view = table.iloc[(current - 1) * PAGE_SIZE: current * PAGE_SIZE]

        html_table([HEADERS[c] for c in view.columns],
                   [[body_cell(c, row[c]) for c in view.columns] for _, row in view.iterrows()], wrap_class="tbl-wrap flat")

        def go_to_page(number):
            st.session_state["tbl_page"] = number

        first_row = (current - 1) * PAGE_SIZE + 1 if n_rows else 0
        last_row = min(current * PAGE_SIZE, n_rows)
        items = page_window(current, n_pages)
        with keyed_container("pager"):
            slots = cols([5] + [0.55] * (len(items) + 2))
            slots[0].markdown(
                f'<div class="tbl-foot">{tr(f"Showing {first_row} to {last_row} of {n_rows:,} sessions", f"Menunjukkan {first_row} hingga {last_row} daripada {n_rows:,} sesi", f"显示第 {first_row} 至 {last_row} 条，共 {n_rows:,} 条会话")}</div>',
                unsafe_allow_html=True)
            slots[1].button("‹", key="pg_prev", disabled=current <= 1, on_click=go_to_page, args=(current - 1,), **STRETCH["button"])
            for slot, item in zip(slots[2:-1], items):
                if item is None:
                    slot.markdown('<div class="pager">…</div>', unsafe_allow_html=True)
                else:
                    slot.button(str(item), key=f"pg_{item}", on_click=go_to_page, args=(item,), **STRETCH["button"],
                                type="primary" if item == current else "secondary")
            slots[-1].button("›", key="pg_next", disabled=current >= n_pages, on_click=go_to_page, args=(current + 1,), **STRETCH["button"])

# ============================================================
# PAGE 2 - VISUAL ANALYTICS
# ============================================================
elif page == "analytics":
    hero("analytics", tr("EV CHARGING ANALYTICS", "ANALITIK PENGECASAN EV", "电动车充电分析"),
         tr("Visual Analytics Studio", "Studio Analitik Visual", "可视化分析工作室"),
         tr("Explore how temperature, time of day and charger type relate to energy use and session length.",
            "Terokai hubungan suhu, waktu dalam hari dan jenis pengecas dengan penggunaan tenaga dan tempoh sesi.",
            "探索温度、时段和充电桩类型与能耗及充电时长之间的关系。"))
    need_data()

    m1, m2, m3, m4 = st.columns(4)
    max_e, max_d = fdf["Energy_kWh"].max(), fdf["Duration_Mins"].max()
    m1.metric(tr("Total sessions", "Jumlah sesi", "会话总数"), f"{len(fdf):,}")
    m2.metric(tr("Charger types", "Jenis pengecas", "充电桩类型数"), fdf["Charger_Type"].nunique() if HAS_CHARGER else 0)
    m3.metric(tr("Largest session", "Sesi terbesar", "最大会话"), f"{0 if np.isnan(max_e) else max_e:,.1f} kWh")
    m4.metric(tr("Longest session", "Sesi terpanjang", "最长会话"), f"{0 if np.isnan(max_d) else max_d:,.0f} {tr('min', 'min', '分钟')}")

    st.write("")
    panel_title("🌡️", tr("Energy delivered vs temperature", "Tenaga dibekalkan lwn suhu", "供电量与温度"),
                tr("Each dot is a session; dot size shows duration", "Setiap titik ialah satu sesi; saiz titik menunjukkan tempoh", "每个点代表一次会话，点的大小表示时长"))
    scatter = fdf.dropna(subset=["Battery_Temp", "Energy_kWh"]).assign(Size=lambda d: d["Duration_Mins"].fillna(0).clip(lower=1))
    fig = px.scatter(scatter, x="Battery_Temp", y="Energy_kWh", color="Charger_Type" if HAS_CHARGER else None, size="Size",
                     opacity=0.75, size_max=18, hover_data={"Size": False}, color_discrete_sequence=SERIES,
                     labels={"Battery_Temp": tr("Temperature (°C)", "Suhu (°C)", "温度 (°C)"),
                             "Energy_kWh": tr("Energy (kWh)", "Tenaga (kWh)", "能量 (kWh)"),
                             "Charger_Type": tr("Charger", "Pengecas", "充电桩")})
    fig.update_layout(legend=dict(orientation="h", x=0, y=-0.2), margin=dict(l=64, r=26, t=24, b=84))
    show(fig, 420)

    c1, c2 = st.columns(2)
    with c1:
        panel_title("⏱️", tr("Peak vs non-peak sessions", "Sesi puncak lwn bukan puncak", "高峰与非高峰会话"),
                    tr("Share of sessions", "Peratusan sesi", "会话占比"))
        share = fdf["Peak_Hour_Label"].map(pk).value_counts().reset_index()
        share.columns = ["Status", "Count"]
        fig = px.pie(share, names="Status", values="Count", hole=0.64, color_discrete_sequence=[BLUE, ORANGE, PURPLE])
        fig.update_traces(textinfo="percent+label", marker=dict(line=dict(color=PANEL, width=3)))
        fig.update_layout(showlegend=False)
        show(fig, 340)
    with c2:
        if HAS_CHARGER:
            panel_title("🔌", tr("Charger type share", "Peratusan jenis pengecas", "充电桩类型占比"), tr("Share of sessions", "Peratusan sesi", "会话占比"))
            fig = px.pie(fdf, names="Charger_Type", hole=0.64, color_discrete_sequence=SERIES)
            fig.update_traces(textinfo="percent+label", marker=dict(line=dict(color=PANEL, width=3)))
            fig.update_layout(showlegend=False)
            show(fig, 340)

    c3, c4 = st.columns(2)
    with c3:
        if fdf["Hour_of_Day"].notna().any():
            panel_title("🕒", tr("Sessions by hour of day", "Sesi mengikut jam", "各时段会话数"), tr("Demand across the day", "Permintaan sepanjang hari", "全天需求"))
            hourly = fdf.dropna(subset=["Hour_of_Day"]).groupby("Hour_of_Day").size().reset_index(name="Sessions")
            fig = px.area(hourly, x="Hour_of_Day", y="Sessions",
                          labels={"Hour_of_Day": tr("Hour of day", "Jam dalam hari", "一天中的小时"), "Sessions": tr("Sessions", "Sesi", "会话")})
            fig.update_traces(line_color=BLUE, fillcolor="rgba(79,124,255,0.18)")
        else:
            panel_title("🕒", tr("Duration distribution", "Taburan tempoh", "时长分布"), tr("Minutes per session", "Minit setiap sesi", "每次会话分钟数"))
            fig = px.histogram(fdf, x="Duration_Mins", nbins=30,
                               labels={"Duration_Mins": tr("Session duration (min)", "Tempoh sesi (min)", "会话时长（分钟）"), "count": tr("Sessions", "Sesi", "会话")})
            fig.update_traces(marker_color=BLUE)
            fig.update_layout(bargap=0.05)
        show(fig, 340)
    with c4:
        if HAS_CHARGER:
            panel_title("📦", tr("Energy spread by charger type", "Sebaran tenaga mengikut jenis pengecas", "各类型充电桩能量分布"),
                        tr("Distribution of kWh per session", "Taburan kWh setiap sesi", "每次会话 kWh 分布"))
            fig = px.box(fdf, x="Charger_Type", y="Energy_kWh", color="Charger_Type", color_discrete_sequence=SERIES,
                         labels={"Charger_Type": tr("Charger type", "Jenis pengecas", "充电桩类型"), "Energy_kWh": tr("Energy (kWh)", "Tenaga (kWh)", "能量 (kWh)")})
            fig.update_layout(showlegend=False)
            show(fig, 340)

    numeric = fdf.select_dtypes(include=np.number).drop(columns=["Hour_of_Day"], errors="ignore")
    numeric = numeric.loc[:, numeric.nunique() > 1]
    if numeric.shape[1] >= 2:
        render(f'<div class="section">📈 {tr("Feature correlations", "Korelasi ciri", "特征相关性")}</div>'
               f'<div class="section-sub">{tr("Values near 1 or -1 indicate a strong relationship.", "Nilai hampir 1 atau -1 menunjukkan hubungan yang kuat.", "数值接近 1 或 -1 表示关系很强。")}</div>')
        corr = numeric.corr()
        names = [str(c).replace("_", " ") for c in corr.columns]
        fig = go.Figure(go.Heatmap(z=corr.values.tolist(), x=names, y=names, zmin=-1, zmax=1,
                                   colorscale=[[0, PURPLE], [0.5, PANEL2], [1, BLUE]], texttemplate="%{z:.2f}",
                                   textfont=dict(size=11, color="#FFFFFF"), xgap=2, ygap=2,
                                   hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
                                   colorbar=dict(tickfont=dict(color=MUTED), outlinewidth=0)))
        fig.update_xaxes(type="category", tickangle=-40, side="bottom", automargin=True)
        fig.update_yaxes(type="category", autorange="reversed", automargin=True)
        fig.update_layout(margin=dict(l=150, r=20, t=40, b=130))
        show(fig, max(480, 40 * len(names) + 200))

# ============================================================
# PAGE 3 - LIVE ML INFERENCE
# ============================================================
else:
    hero("ml", tr("MACHINE LEARNING", "PEMBELAJARAN MESIN", "机器学习"),
         tr("Live ML Inference", "Inferens ML Langsung", "实时机器学习推理"),
         tr("Pick a scenario, tune the session and watch the risk, actions, cost and carbon impact update instantly.",
            "Pilih senario, laras sesi dan lihat risiko, tindakan, kos serta kesan karbon dikemas kini serta-merta.",
            "选择场景、调整充电会话，即时查看风险、行动建议、成本与碳影响。"))
    ml_guide()

    charger_opts = [x for x in options("Charger_Type") if x != "All"] or ["AC", "DC"]
    DC, AC = ("dc", "fast"), ("ac", "standard")
    GRID_NAMES = {"Normal": tr("Normal", "Normal", "正常"), "Peak Load": tr("Peak Load", "Beban Puncak", "高峰负载"),
                  "Warning": tr("Warning", "Amaran", "警告")}

    def scenario(sid, icon, name, level, desc, kind, dur, energy, power, soc, over, temp, grid):
        return dict(id=sid, icon=icon, name=name, level=level, desc=desc, kind=kind, dur=dur, energy=energy,
                    power=power, soc=soc, over=over, temp=temp, grid=grid)

    SCENARIOS = [
        scenario("optimal", "✅", tr("Optimal Fast Charge", "Pengecasan Pantas Optimum", "最佳快充"), "healthy",
                 tr("DC fast charge at a comfortable temperature with no overstay.", "Pengecasan pantas DC pada suhu selesa tanpa tinggal lebih masa.", "温度适宜的直流快充，无超时占位。"),
                 DC, 45.0, 35.0, 50.0, 20, 0, 31.0, "Normal"),
        scenario("standard", "🔌", tr("Standard AC Session", "Sesi AC Standard", "标准交流充电"), "healthy",
                 tr("Typical campus AC charging while the driver is in class.", "Pengecasan AC kampus biasa semasa pemandu berada di kelas.", "司机上课期间的典型校园交流充电。"),
                 AC, 180.0, 18.0, 7.4, 40, 5, 29.0, "Normal"),
        scenario("overnight", "🌙", tr("Overnight Parking", "Letak Kereta Semalaman", "夜间停放充电"), "healthy",
                 tr("Long, slow AC charge left running overnight.", "Pengecasan AC perlahan dan panjang yang dibiarkan semalaman.", "整夜进行的长时间慢速交流充电。"),
                 AC, 480.0, 30.0, 7.4, 30, 0, 27.0, "Normal"),
        scenario("cold", "❄️", tr("Cold-Weather Charge", "Pengecasan Cuaca Sejuk", "低温充电"), "watch",
                 tr("A cold battery slows charging, so the energy rate drops.", "Bateri sejuk melambatkan pengecasan, jadi kadar tenaga menurun.", "低温电池会降低充电速度，能量速率下降。"),
                 AC, 120.0, 12.0, 11.0, 25, 0, 6.0, "Normal"),
        scenario("overstay", "🅿️", tr("Overstay Warning", "Amaran Tinggal Lebih Masa", "超时占位警告"), "watch",
                 tr("The car stays plugged in long after charging has finished.", "Kereta kekal bersambung lama selepas pengecasan selesai.", "充电结束后车辆长时间仍占用充电桩。"),
                 AC, 150.0, 20.0, 11.0, 30, 40, 38.0, "Peak Load"),
        scenario("peak", "📈", tr("Peak Grid Stress", "Tekanan Grid Waktu Puncak", "高峰电网压力"), "watch",
                 tr("Fast charging while the campus grid is at peak load.", "Pengecasan pantas ketika grid kampus berada pada beban puncak.", "校园电网处于高峰负载时进行快充。"),
                 DC, 60.0, 40.0, 60.0, 25, 10, 42.0, "Peak Load"),
        scenario("fault", "🛠️", tr("Charger Fault", "Kerosakan Pengecas", "充电桩故障"), "watch",
                 tr("Very little energy delivered for the rated power.", "Tenaga yang dibekalkan sangat sedikit berbanding kuasa terkadar.", "相对额定功率，输出能量极少。"),
                 AC, 120.0, 6.0, 22.0, 35, 10, 36.0, "Normal"),
        scenario("thermal", "🔥", tr("Thermal Risk", "Risiko Haba", "热风险"), "critical",
                 tr("Overheating battery, long overstay and a grid warning.", "Bateri terlalu panas, tinggal lebih masa dan amaran grid.", "电池过热、长时间超时占位并伴有电网警告。"),
                 DC, 60.0, 45.0, 120.0, 15, 50, 56.0, "Warning"),
    ]
    LEVELS = {"healthy": (tr("Healthy", "Sihat", "健康"), GREEN), "watch": (tr("Watch", "Pantau", "关注"), AMBER),
              "critical": (tr("Critical", "Kritikal", "严重"), RED), "custom": (tr("Custom", "Tersuai", "自定义"), BLUE),
              "real": (tr("Real data", "Data sebenar", "真实数据"), PURPLE)}
    DEFAULTS = {"ml_charger": charger_opts[0], "ml_duration": 90.0, "ml_energy": 25.0, "ml_power": 22.0, "ml_soc": 20,
                "ml_overstay": 10, "ml_battery": 32.0, "ml_grid": "Normal", "ml_scn": "custom"}
    for _k, _v in DEFAULTS.items():
        st.session_state.setdefault(_k, _v)
    if st.session_state["ml_charger"] not in charger_opts:
        st.session_state["ml_charger"] = charger_opts[0]

    def pick_charger(kinds):
        return next((o for o in charger_opts if any(k in o.lower() for k in kinds)), charger_opts[0])

    def apply_scenario(index):
        s = SCENARIOS[index]
        st.session_state.update(ml_charger=pick_charger(s["kind"]), ml_duration=s["dur"], ml_energy=s["energy"],
                                ml_power=s["power"], ml_soc=s["soc"], ml_overstay=s["over"], ml_battery=s["temp"],
                                ml_grid=s["grid"], ml_scn=s["id"])

    def reset_inputs():
        st.session_state.update(DEFAULTS)

    def mark_custom():
        st.session_state["ml_scn"] = "custom"

    def bounded(value, low, high, default):
        try:
            value = float(value)
            return default if np.isnan(value) else float(min(high, max(low, value)))
        except (TypeError, ValueError):
            return default

    def load_real_session():
        row = (fdf if len(fdf) else df).sample(1).iloc[0]
        power_col, over_col, soc_col = POWER_COL, OVERSTAY_COL, SOC_COL
        duration = bounded(row["Duration_Mins"], 1, 1440, 60.0)
        energy = bounded(row["Energy_kWh"], 0.1, 500, 20.0)
        power = bounded(row[power_col], 0.1, 350, 22.0) if power_col else bounded(energy / (duration / 60) * 1.15, 0.1, 350, 22.0)
        grid_text = str(row["Grid_Status_Label"]).lower()
        grid = ("Warning" if any(w in grid_text for w in ("warn", "over", "fault", "err"))
                else "Peak Load" if row["Peak_Hour_Label"] == "Peak" else "Normal")
        charger = str(row["Charger_Type"]) if HAS_CHARGER else ""
        st.session_state.update(
            ml_charger=charger if charger in charger_opts else charger_opts[0], ml_duration=duration, ml_energy=energy,
            ml_power=power, ml_soc=int(bounded(row[soc_col], 0, 100, 20)) if soc_col else 20,
            ml_overstay=int(bounded(row[over_col], 0, 300, 0)) if over_col else 0,
            ml_battery=bounded(row["Battery_Temp"], -10, 90, 32.0), ml_grid=grid, ml_scn="real")

    # ---------------- Step 1: scenario ----------------
    render(f'<div class="section" style="margin-top:2px"><span class="step">1</span>{tr("Pick a scenario", "Pilih senario", "选择场景")}</div>'
           f'<div class="section-sub">{tr("Start from a ready-made case, load a real session from your data, or enter your own values below.", "Mulakan dengan kes sedia ada, muat sesi sebenar daripada data anda, atau masukkan nilai sendiri di bawah.", "从现成案例开始、载入真实会话数据，或在下方输入自己的数值。")}</div>')
    for row_start in (0, 4):
        grid_cols = st.columns(4)
        for offset, s in enumerate(SCENARIOS[row_start:row_start + 4]):
            grid_cols[offset].button(f'{s["icon"]} {s["name"]}', key=f'scn_{s["id"]}', on_click=apply_scenario,
                                     args=(row_start + offset,), **STRETCH["button"],
                                     type="primary" if st.session_state["ml_scn"] == s["id"] else "secondary")
    u1, u2, _, _ = st.columns(4)
    u1.button(tr("🎲 Load a real session", "🎲 Muat sesi sebenar", "🎲 载入真实会话"), key="scn_real", on_click=load_real_session,
              **STRETCH["button"], type="primary" if st.session_state["ml_scn"] == "real" else "secondary")
    u2.button(tr("↺ Reset to defaults", "↺ Set semula lalai", "↺ 恢复默认"), key="scn_reset", on_click=reset_inputs, **STRETCH["button"])

    active = next((s for s in SCENARIOS if s["id"] == st.session_state["ml_scn"]), None)
    if active:
        scn_name, scn_level, scn_desc = f'{active["icon"]} {active["name"]}', active["level"], active["desc"]
    elif st.session_state["ml_scn"] == "real":
        scn_name, scn_level = tr("🎲 Real session from dataset", "🎲 Sesi sebenar daripada data", "🎲 数据集中的真实会话"), "real"
        scn_desc = tr("A random session taken from your dataset (filters applied). Adjust any value to explore what-ifs.",
                      "Sesi rawak daripada set data anda (penapis digunakan). Ubah mana-mana nilai untuk meneroka senario 'bagaimana jika'.",
                      "从数据集中随机抽取的会话（已应用筛选）。可修改任意数值进行假设分析。")
    else:
        scn_name, scn_level = tr("Custom values", "Nilai tersuai", "自定义数值"), "custom"
        scn_desc = tr("You are using your own values. Pick a scenario above to start from a preset.",
                      "Anda menggunakan nilai sendiri. Pilih senario di atas untuk bermula daripada pratetap.",
                      "当前使用自定义数值。请在上方选择一个场景作为起点。")
    level_name, level_color = LEVELS[scn_level]
    render(f'<div class="scn" style="--tone:{level_color}"><span class="scn-chip">{escape(level_name.upper())}</span>'
           f'<span class="scn-name">{escape(scn_name)}</span><span class="scn-desc">{escape(scn_desc)}</span></div>')

    # ---------------- Step 2: inputs + live result ----------------
    render(f'<div class="section"><span class="step">2</span>{tr("Adjust the inputs and watch the result", "Laraskan input dan lihat keputusan", "调整输入并查看结果")}</div>')
    left, right = st.columns([1, 1.55], gap="large")

    with left:
        with st.container(border=True):
            render(f'<div class="grp">{tr("Charging session", "Sesi pengecasan", "充电会话")}</div>')
            ml_charger = st.selectbox(tr("Charger type", "Jenis pengecas", "充电桩类型"), charger_opts, key="ml_charger", on_change=mark_custom)
            i1, i2 = st.columns(2)
            ml_duration = i1.number_input(tr("Duration (min)", "Tempoh (min)", "时长（分钟）"), 1.0, 1440.0, step=5.0, key="ml_duration", on_change=mark_custom)
            ml_energy = i2.number_input(tr("Energy (kWh)", "Tenaga (kWh)", "能量 (kWh)"), 0.1, 500.0, step=1.0, key="ml_energy", on_change=mark_custom)
            ml_power = st.number_input(tr("Rated power (kW)", "Kuasa terkadar (kW)", "额定功率 (kW)"), 0.1, 350.0, step=1.0, key="ml_power", on_change=mark_custom)
            render(f'<div class="grp" style="margin-top:12px">{tr("Battery and grid", "Bateri dan grid", "电池与电网")}</div>')
            ml_soc = st.slider(tr("Initial state of charge (%)", "Keadaan cas awal (%)", "初始电量 (%)"), 0, 100, key="ml_soc", on_change=mark_custom)
            ml_battery = st.slider(tr("Battery temperature (°C)", "Suhu bateri (°C)", "电池温度 (°C)"), -10.0, 90.0, step=0.5, key="ml_battery", on_change=mark_custom)
            ml_overstay = st.slider(tr("Overstay (min)", "Tinggal lebih masa (min)", "超时占位（分钟）"), 0, 300, key="ml_overstay", on_change=mark_custom)
            ml_grid = st.radio(tr("Grid status", "Status grid", "电网状态"), list(GRID_NAMES), format_func=GRID_NAMES.get,
                               horizontal=True, key="ml_grid", on_change=mark_custom)

        with st.expander(tr("💲 Cost and carbon assumptions", "💲 Andaian kos dan karbon", "💲 成本与碳排放假设")):
            for _k, _v in {"a_tariff": 0.60, "a_peakx": 1.25, "a_grace": 15, "a_pen": 0.50, "a_ef": 0.70, "a_kmkwh": 6.0, "a_petrol": 0.19}.items():
                st.session_state.setdefault(_k, _v)
            x1, x2 = st.columns(2)
            a_tariff = x1.number_input(tr("Tariff (RM/kWh)", "Tarif (RM/kWh)", "电价 (RM/kWh)"), 0.0, 5.0, step=0.05, key="a_tariff")
            a_peakx = x2.number_input(tr("Peak multiplier", "Pengganda puncak", "高峰倍数"), 1.0, 3.0, step=0.05, key="a_peakx")
            x3, x4 = st.columns(2)
            a_grace = x3.number_input(tr("Overstay grace (min)", "Tempoh bertolak ansur (min)", "超时宽限期（分钟）"), 0, 120, key="a_grace")
            a_pen = x4.number_input(tr("Penalty (RM/min)", "Penalti (RM/min)", "罚款 (RM/分钟)"), 0.0, 10.0, step=0.1, key="a_pen")
            y1, y2 = st.columns(2)
            a_ef = y1.number_input(tr("Grid CO₂ (kg/kWh)", "CO₂ grid (kg/kWh)", "电网 CO₂ (kg/kWh)"), 0.0, 2.0, step=0.01, key="a_ef")
            a_kmkwh = y2.number_input(tr("EV efficiency (km/kWh)", "Kecekapan EV (km/kWh)", "电动车效率 (km/kWh)"), 1.0, 12.0, step=0.5, key="a_kmkwh")
            a_petrol = st.number_input(tr("Petrol car CO₂ (kg/km)", "CO₂ kereta petrol (kg/km)", "燃油车 CO₂ (kg/km)"), 0.0, 1.0, step=0.01, key="a_petrol")
            st.caption(tr("Planning assumptions, not official figures. Replace with your campus tariff and the national grid factor.",
                          "Andaian perancangan, bukan angka rasmi. Gantikan dengan tarif kampus dan faktor grid kebangsaan anda.",
                          "仅为规划假设，并非官方数据。请替换为校园电价和国家电网排放因子。"))

    # ---------------- Computation ----------------
    hours = ml_duration / 60.0
    avg_kw = ml_energy / hours
    utilisation = ml_energy / (ml_power * hours)

    def unit(x):
        return float(min(1.0, max(0.0, x)))

    points = {
        "temp": 60 * unit((ml_battery - 35) / 20),
        "overstay": 30 * unit(ml_overstay / 60),
        "grid": {"Normal": 0, "Peak Load": 6, "Warning": 12}[ml_grid],
        "power": 6 * unit((ml_power - 22) / 128),
        "soc": 4 * unit(abs(ml_soc - 50) / 50) ** 2,
    }
    FACTOR_NAMES = {"temp": tr("Battery temperature", "Suhu bateri", "电池温度"), "overstay": tr("Overstay duration", "Tempoh tinggal lebih masa", "超时时长"),
                    "grid": tr("Grid status", "Status grid", "电网状态"), "power": tr("Power load", "Beban kuasa", "功率负载"),
                    "soc": tr("State of charge", "Keadaan cas", "电量状态")}
    risk_score = min(100.0, sum(points.values()))

    prediction, confidence, source = None, None, tr("Rule-based", "Peraturan", "规则模型")
    if ml_model is not None:
        payload = {"Charger_Type": ml_charger, "Duration_Mins": ml_duration, "Energy_kWh": ml_energy, "Power_kW": ml_power,
                   "Battery_Temp": ml_battery, "Overstay_Mins": ml_overstay, "SoC": ml_soc, "Grid_Status": ml_grid}
        for fn_name in ("predict", "run_inference", "predict_session"):
            fn = getattr(ml_model, fn_name, None)
            if not callable(fn):
                continue
            try:
                out = fn(payload)
                if isinstance(out, (tuple, list)) and len(out) >= 2:
                    prediction, confidence = out[0], out[1]
                elif isinstance(out, dict):
                    prediction = out.get("prediction") or out.get("label") or out.get("status")
                    confidence = out.get("confidence") or out.get("probability")
                else:
                    prediction = out
                if prediction is not None:
                    source = tr("Trained model", "Model terlatih", "已训练模型")
                    break
            except Exception:
                continue
        explain = getattr(ml_model, "explain", None)
        if callable(explain):
            try:
                explained = explain(payload)
                if isinstance(explained, dict) and explained:
                    points = {str(k): abs(float(v)) for k, v in explained.items()}
            except Exception:
                pass

    if prediction is None:
        state = "high" if risk_score >= 55 else "moderate" if risk_score >= 25 else "optimal" if avg_kw >= 10 else "normal"
        prediction = {"high": tr("High risk: thermal or overstay anomaly", "Risiko tinggi: anomali haba atau tinggal lebih masa", "高风险：过热或超时异常"),
                      "moderate": tr("Moderate: monitor session", "Sederhana: pantau sesi", "中等：请监控会话"),
                      "optimal": tr("Optimal charging session", "Sesi pengecasan optimum", "最佳充电会话"),
                      "normal": tr("Standard performance session", "Sesi prestasi standard", "标准性能会话")}[state]
    else:
        text = str(prediction).lower()
        state = ("high" if any(w in text for w in ("risk", "anomaly", "error", "critical")) else
                 "moderate" if any(w in text for w in ("moderate", "warning", "overstay")) else
                 "optimal" if any(w in text for w in ("optimal", "good", "efficient")) else "normal")
    TONES = {
        "high": (RED, tr("Action needed", "Tindakan diperlukan", "需要处理"),
                 tr("One or more safety limits are exceeded. Follow the recommended actions below.", "Satu atau lebih had keselamatan dilepasi. Ikuti tindakan yang disyorkan di bawah.", "已超出一项或多项安全限值。请按下方建议操作。")),
        "moderate": (AMBER, tr("Monitor", "Pantau", "关注"),
                     tr("Some factors are elevated. Act on the recommendations to avoid escalation.", "Beberapa faktor meningkat. Ambil tindakan yang disyorkan untuk mengelakkan keadaan merosot.", "部分因素偏高。请按建议处理以防情况恶化。")),
        "optimal": (GREEN, tr("Healthy", "Sihat", "健康"),
                    tr("The session is running efficiently within safe limits.", "Sesi berjalan dengan cekap dalam had selamat.", "会话运行高效，且在安全范围内。")),
        "normal": (BLUE, tr("Normal", "Normal", "正常"),
                   tr("The session is within standard campus charging behaviour.", "Sesi berada dalam tingkah laku pengecasan kampus yang standard.", "会话符合校园标准充电行为。")),
    }
    tone, tag, message = TONES[state]

    # cost, penalty and carbon
    tariff = a_tariff * (a_peakx if ml_grid == "Peak Load" else 1.0)
    energy_cost = ml_energy * tariff
    penalty = max(0, ml_overstay - a_grace) * a_pen
    co2_grid = ml_energy * a_ef
    co2_saved = ml_energy * a_kmkwh * a_petrol - co2_grid

    # operator actions: (urgency, colour, level label, title, text)
    LV = {"critical": tr("Critical", "Kritikal", "严重"), "high": tr("High", "Tinggi", "高"), "medium": tr("Medium", "Sederhana", "中"),
          "info": tr("Info", "Maklumat", "提示"), "ok": "OK"}
    actions = []
    t, o, p = ml_battery, ml_overstay, ml_power
    if t >= 55:
        actions.append((0, RED, LV["critical"], tr("Emergency cutoff", "Pemutusan kecemasan", "紧急断电"),
                        tr(f"Battery at {t:.1f} °C (limit 55 °C). Stop the session, isolate the charger and inspect the vehicle and cable.",
                           f"Bateri pada {t:.1f} °C (had 55 °C). Henti sesi, asingkan pengecas dan periksa kenderaan serta kabel.",
                           f"电池温度 {t:.1f} °C（上限 55 °C）。请停止会话、隔离充电桩，并检查车辆和线缆。")))
    elif t >= 45:
        kw = max(round(p * 0.5), 3)
        actions.append((1, ORANGE, LV["high"], tr(f"Lower power to {kw} kW", f"Kurangkan kuasa kepada {kw} kW", f"将功率降至 {kw} kW"),
                        tr(f"Battery at {t:.1f} °C. Halving output helps cooling and prevents a cutoff.",
                           f"Bateri pada {t:.1f} °C. Mengurangkan output separuh membantu penyejukan dan mengelakkan pemutusan.",
                           f"电池温度 {t:.1f} °C。输出减半有助散热并避免断电。")))
    elif t >= 40:
        kw = max(round(p * 0.75), 3)
        actions.append((2, AMBER, LV["medium"], tr(f"Lower power to {kw} kW", f"Kurangkan kuasa kepada {kw} kW", f"将功率降至 {kw} kW"),
                        tr(f"Battery at {t:.1f} °C is trending high. Reduce output and keep monitoring.",
                           f"Bateri pada {t:.1f} °C semakin meningkat. Kurangkan output dan teruskan pemantauan.",
                           f"电池温度 {t:.1f} °C 呈上升趋势。请降低输出并持续监控。")))
    elif t <= 10:
        actions.append((3, BLUE, LV["info"], tr("Expect a slower charge", "Jangkakan pengecasan lebih perlahan", "预计充电较慢"),
                        tr(f"Battery at {t:.1f} °C is cold. Charging power will be limited until it warms up.",
                           f"Bateri pada {t:.1f} °C sejuk. Kuasa pengecasan akan dihadkan sehingga ia panas.",
                           f"电池温度 {t:.1f} °C 偏低，升温前充电功率会受限。")))
    if o >= 45:
        actions.append((1, ORANGE, LV["high"], tr("Apply overstay penalty and notify the driver", "Kenakan penalti tinggal lebih masa dan maklumkan pemandu", "收取超时罚款并通知司机"),
                        tr(f"Vehicle idle for {o} min. Penalty RM {penalty:.2f}; free the bay for the next user.",
                           f"Kenderaan melahu selama {o} min. Penalti RM {penalty:.2f}; kosongkan petak untuk pengguna seterusnya.",
                           f"车辆已闲置 {o} 分钟，罚款 RM {penalty:.2f}；请腾出车位给下一位用户。")))
    elif o >= 20:
        actions.append((2, AMBER, LV["medium"], tr("Send a move-your-vehicle reminder", "Hantar peringatan alihkan kenderaan", "发送挪车提醒"),
                        tr(f"Overstay at {o} min. Penalty starts after {a_grace} min (RM {a_pen:.2f}/min).",
                           f"Tinggal lebih masa {o} min. Penalti bermula selepas {a_grace} min (RM {a_pen:.2f}/min).",
                           f"已超时 {o} 分钟。{a_grace} 分钟后开始罚款（RM {a_pen:.2f}/分钟）。")))
    if ml_grid == "Warning":
        actions.append((1, ORANGE, LV["high"], tr("Throttle to 11 kW and defer non-urgent sessions", "Hadkan kepada 11 kW dan tangguh sesi tidak segera", "限功率至 11 kW 并推迟非紧急会话"),
                        tr("Grid reports a warning state. Reduce campus load until it clears.", "Grid melaporkan keadaan amaran. Kurangkan beban kampus sehingga pulih.", "电网处于警告状态。请降低校园负载直至恢复。")))
    elif ml_grid == "Peak Load":
        title = (tr("Cap at 11 kW or shift to off-peak", "Hadkan pada 11 kW atau alihkan ke luar waktu puncak", "限制在 11 kW 或转至非高峰时段") if p > 11
                 else tr("Encourage off-peak charging", "Galakkan pengecasan luar waktu puncak", "鼓励非高峰充电"))
        actions.append((2, AMBER, LV["medium"], title,
                        tr(f"Peak tariff applies (RM {tariff:.2f}/kWh). Load shifting lowers cost and grid stress.",
                           f"Tarif puncak dikenakan (RM {tariff:.2f}/kWh). Anjakan beban mengurangkan kos dan tekanan grid.",
                           f"适用高峰电价（RM {tariff:.2f}/kWh）。负荷转移可降低成本与电网压力。")))
    if utilisation > 1.1:
        actions.append((2, AMBER, LV["medium"], tr("Check the entered values", "Semak nilai yang dimasukkan", "请检查输入数值"),
                        tr("Energy delivered is higher than rated power multiplied by time. Verify power, duration and energy.",
                           "Tenaga dibekalkan lebih tinggi daripada kuasa terkadar didarab masa. Sahkan kuasa, tempoh dan tenaga.",
                           "输出能量高于额定功率×时间。请核对功率、时长和能量。")))
    elif utilisation < 0.4:
        actions.append((2, AMBER, LV["medium"], tr("Inspect connector and charger", "Periksa penyambung dan pengecas", "检查接头与充电桩"),
                        tr(f"Only {utilisation * 100:.0f}% of rated output was delivered. Look for a worn connector or a derating fault.",
                           f"Hanya {utilisation * 100:.0f}% daripada output terkadar dibekalkan. Semak penyambung haus atau kerosakan penurunan kadar.",
                           f"仅输出了额定功率的 {utilisation * 100:.0f}%。请检查接头磨损或降额故障。")))
    if ml_soc >= 80:
        actions.append((3, BLUE, LV["info"], tr("Expect charge tapering", "Jangkakan pengecasan menyusut", "预计充电功率将递减"),
                        tr(f"Starting at {ml_soc}% SoC, power will taper. Consider ending the session early to free the bay.",
                           f"Bermula pada {ml_soc}% SoC, kuasa akan menyusut. Pertimbangkan menamatkan sesi lebih awal untuk mengosongkan petak.",
                           f"起始电量 {ml_soc}%，功率将逐步下降。可考虑提前结束会话以腾出车位。")))
    if not actions:
        actions.append((4, GREEN, LV["ok"], tr("No intervention needed", "Tiada campur tangan diperlukan", "无需干预"),
                        tr("All readings are within safe operating limits.", "Semua bacaan berada dalam had operasi selamat.", "所有读数均在安全运行范围内。")))
    actions.sort(key=lambda a: a[0])

    # ---------------- Live result (right column) ----------------
    with right:
        if actions[0][0] == 0:
            render(f'<div class="alert-banner"><b>{tr("⚠ SAFETY ALERT", "⚠ AMARAN KESELAMATAN", "⚠ 安全警报")}</b>'
                   f'{tr("Emergency cutoff recommended. Stop charging and inspect before restarting.", "Pemutusan kecemasan disyorkan. Henti pengecasan dan periksa sebelum memulakan semula.", "建议紧急断电。请停止充电并检查后再重启。")}</div>')
        g1, g2 = st.columns([1.35, 1])
        with g1:
            render(f'<div class="result" style="--tone:{tone}"><div class="result-tag">{tag}</div>'
                   f'<div class="result-title">{escape(str(prediction))}</div><div class="result-text">{message}</div></div>')
        with g2:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=risk_score, number=dict(suffix=" / 100", font=dict(size=28, color=TEXT)),
                title=dict(text=tr("Risk score", "Skor risiko", "风险评分"), font=dict(size=13, color=MUTED)),
                gauge=dict(axis=dict(range=[0, 100], tickcolor=MUTED, tickfont=dict(color=MUTED, size=10)),
                           bar=dict(color=tone, thickness=0.3), bgcolor="rgba(0,0,0,0)", borderwidth=0,
                           steps=[dict(range=[0, 25], color="rgba(61,214,140,.22)"), dict(range=[25, 55], color="rgba(242,181,68,.22)"),
                                  dict(range=[55, 100], color="rgba(240,86,106,.24)")])))
            gauge.update_layout(margin=dict(l=28, r=28, t=40, b=14))
            show(gauge, 192)

        m1, m2, m3 = st.columns(3)
        m1.metric(tr("Avg delivered power", "Purata kuasa dibekalkan", "平均输出功率"), f"{avg_kw:.1f} kW")
        m2.metric(tr("Battery temperature", "Suhu bateri", "电池温度"), f"{ml_battery:.1f} °C")
        m3.metric(tr("Utilisation", "Penggunaan", "利用率"), f"{utilisation * 100:.0f}%")
        conf_text = ""
        if confidence is not None:
            cf = float(confidence)
            conf_text = f" • {tr('Confidence', 'Keyakinan', '置信度')}: {cf * 100 if cf <= 1 else cf:.0f}%"
        st.caption(f"{tr('Model source', 'Sumber model', '模型来源')}: {source}{conf_text}")

        render(f'<div class="grp" style="margin:10px 0 6px 2px">{tr("Cost, penalty and carbon", "Kos, penalti dan karbon", "成本、罚款与碳排放")}</div>')
        n1, n2, n3 = st.columns(3)
        n1.metric(tr("Energy cost", "Kos tenaga", "电费"), f"RM {energy_cost:,.2f}", f"@ RM {tariff:.2f}/kWh", delta_color="off")
        n2.metric(tr("Overstay penalty", "Penalti tinggal lebih masa", "超时罚款"), f"RM {penalty:,.2f}")
        n3.metric(tr("Total bill", "Jumlah bil", "总费用"), f"RM {energy_cost + penalty:,.2f}")
        n4, n5, _ = st.columns(3)
        n4.metric(tr("CO₂ from grid", "CO₂ daripada grid", "电网 CO₂"), f"{co2_grid:,.1f} kg")
        n5.metric(tr("CO₂ avoided vs petrol", "CO₂ dielak berbanding petrol", "较燃油车减排 CO₂"), f"{co2_saved:,.1f} kg")

    # ---------------- Step 3: why + actions ----------------
    render(f'<div class="section"><span class="step">3</span>{tr("Why, and what to do next", "Mengapa, dan apa tindakan seterusnya", "原因与后续行动")}</div>')
    why_col, act_col = st.columns([1.1, 1], gap="large")
    with why_col:
        panel_title("🧠", tr("Why this prediction", "Mengapa ramalan ini", "预测原因"),
                    tr("Share of the risk score driven by each factor", "Sumbangan setiap faktor kepada skor risiko", "各因素对风险评分的贡献"))
        total = sum(points.values())
        ranked = sorted(((FACTOR_NAMES.get(k, k), (v / total * 100 if total > 0 else 0), v) for k, v in points.items()), key=lambda r: r[1])
        top = max(r[1] for r in ranked)
        fig = go.Figure(go.Bar(
            x=[r[1] for r in ranked], y=[r[0] for r in ranked], orientation="h", customdata=[r[2] for r in ranked],
            marker_color=[RED if (r[1] == top and top > 0) else BLUE for r in ranked],
            text=[f"{r[1]:.0f}%" for r in ranked], textposition="outside", cliponaxis=False, textfont=dict(color=TEXT, size=12),
            hovertemplate="%{y}: %{x:.0f}% (%{customdata:.1f} pts)<extra></extra>"))
        fig.update_xaxes(range=[0, 118], ticksuffix="%", title_text=tr("Contribution to risk (%)", "Sumbangan kepada risiko (%)", "对风险的贡献 (%)"))
        fig.update_layout(bargap=0.35, margin=dict(l=150, r=44, t=24, b=64))
        show(fig, 330)
        if total == 0:
            st.caption(tr("No risk factor is elevated, so every contribution is zero.", "Tiada faktor risiko meningkat, jadi semua sumbangan ialah sifar.", "没有风险因素偏高，因此各项贡献均为零。"))
        else:
            name, share = ranked[-1][0], ranked[-1][1]
            st.caption(tr(f"Biggest driver: **{name}** ({share:.0f}% of the risk score).",
                          f"Pemacu terbesar: **{name}** ({share:.0f}% daripada skor risiko).",
                          f"最大影响因素：**{name}**（占风险评分 {share:.0f}%）。"))
    with act_col:
        panel_title("🛠️", tr("Recommended operator actions", "Tindakan operator yang disyorkan", "建议的运维操作"),
                    tr("Most urgent first", "Paling mendesak dahulu", "按紧急程度排序"))
        cards = "".join(f'<div class="act" style="--tone:{color}"><div class="act-tag">{level}</div>'
                        f'<div class="act-title">{escape(title)}</div><div class="act-text">{escape(text)}</div></div>'
                        for _, color, level, title, text in actions)
        render(f'<div style="margin-top:10px">{cards}</div>')

    with st.expander(tr("Input summary", "Ringkasan input", "输入摘要")):
        summary = [(tr("Scenario", "Senario", "场景"), scn_name), (tr("Charger type", "Jenis pengecas", "充电桩类型"), ml_charger),
                   (tr("Duration", "Tempoh", "时长"), f"{ml_duration:.1f} min"), (tr("Energy delivered", "Tenaga dibekalkan", "供电量"), f"{ml_energy:.2f} kWh"),
                   (tr("Rated power", "Kuasa terkadar", "额定功率"), f"{ml_power:.2f} kW"), (tr("Battery temperature", "Suhu bateri", "电池温度"), f"{ml_battery:.1f} °C"),
                   (tr("Initial SoC", "SoC awal", "初始电量"), f"{ml_soc}%"), (tr("Overstay", "Tinggal lebih masa", "超时占位"), f"{ml_overstay} min"),
                   (tr("Grid status", "Status grid", "电网状态"), GRID_NAMES[ml_grid])]
        html_table([tr("Parameter", "Parameter", "参数"), tr("Value", "Nilai", "数值")],
                   [[f'<td class="id">{escape(k)}</td>', f"<td>{escape(v)}</td>"] for k, v in summary])
    st.caption(tr("The risk score is a transparent weighted model (battery temperature, overstay, grid status, power load and state of charge). If an ml_model.py with predict() is present, its label is used and the factor chart is replaced when it provides explain().",
                  "Skor risiko ialah model berpemberat yang telus (suhu bateri, tinggal lebih masa, status grid, beban kuasa dan keadaan cas). Jika ml_model.py dengan predict() wujud, labelnya digunakan dan carta faktor diganti apabila ia menyediakan explain().",
                  "风险评分为透明的加权模型（电池温度、超时占位、电网状态、功率负载和电量状态）。若存在带有 predict() 的 ml_model.py，则使用其标签；若提供 explain()，则替换因素图。"))

# ============================================================
# FOOTER
# ============================================================
render(f'<div class="footer">{tr("Taylor’s University", "Universiti Taylor’s", "泰莱大学")} &nbsp;|&nbsp; '
       f'{tr("Smart Campus EV Analytics", "Analitik EV Kampus Pintar", "智慧校园电动车分析")} &nbsp;|&nbsp; '
       f'{tr("Machine Learning and Data Analytics Dashboard", "Papan Pemuka Pembelajaran Mesin dan Analitik Data", "机器学习与数据分析仪表板")}</div>')