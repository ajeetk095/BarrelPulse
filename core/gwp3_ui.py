"""Small UI helpers for GWP3 pages (insight cards, provenance badges, cached runs)."""
from __future__ import annotations

import html
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from . import gwp3_config as G
from . import gwp3_pipeline as P
from . import ui_components as ui


@st.cache_resource(show_spinner="Fitting HMM and Bayesian networks on the training block…")
def default_run():
    return P.run_cached(G.GWP3Config())


@st.cache_resource(show_spinner="Fitting model variants…")
def variants_table():
    return P.compare_variants()


@st.cache_data(show_spinner="Running robustness checks…")
def robustness():
    return P.robustness_splits(), P.robustness_states(), P.robustness_seeds()


def stamp(res) -> str:
    last = res.prep.raw.index[-1].strftime("%b %Y")
    return f"Data to {last} · computed {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"


def insight(title: str, observation: str, evidence: str, interpretation: str, relevance: str,
            uncertainty: str, timestamp: str) -> None:
    rows = [("Observation", observation), ("Evidence", evidence), ("Interpretation", interpretation),
            ("Why it matters", relevance), ("Uncertainty", uncertainty)]
    body = "".join(f"<div><b>{k}.</b> {html.escape(v)}</div>" for k, v in rows)
    st.markdown(f'<div class="cc-card"><h4>{html.escape(title)}</h4>{body}'
                f'<div style="color:#9AA7B1;font-size:.8rem;margin-top:.4rem">{html.escape(timestamp)}</div></div>',
                unsafe_allow_html=True)


def provenance(kind: str) -> None:
    if kind == "computed":
        ui.badges([("Computed in this app", "ok"), ("FRED snapshot · 2006–2024 · 4 macro nodes", "info")])
    elif kind == "recorded":
        ui.badges([("Recorded from submitted notebook", "warn"), ("not re-run here — data unreachable", "warn")])
    else:
        ui.badges([("Verified against Alvi (2018) PDF", "info")])


def pct(x: float) -> str:
    return "n/a" if x != x else f"{100 * x:.1f}%"


def df_fmt(df: pd.DataFrame, pct_cols=(), nd: int = 3):
    f = {c: "{:.1%}" for c in pct_cols}
    return df.style.format(f, precision=nd)


# ---------------------------------------------------------------- theme -----
_THEME_CSS = """
.stApp { background: radial-gradient(1200px 600px at 8% -10%, rgba(167,139,250,.16), transparent 60%),
                     radial-gradient(900px 500px at 100% 0%, rgba(34,211,238,.13), transparent 55%),
                     radial-gradient(900px 600px at 50% 110%, rgba(244,114,182,.10), transparent 60%), #070A0F !important; }
.cc-title { background: linear-gradient(90deg,#22D3EE,#A78BFA 45%,#F472B6 75%,#FBBF24); -webkit-background-clip:text;
            background-clip:text; color:transparent; }
.cc-kicker { color:#22D3EE !important; }
.block-container { max-width: 1400px !important; }
.cc-kpi { border-top: 3px solid transparent; border-image: linear-gradient(90deg,#22D3EE,#A78BFA,#F472B6) 1; background: rgba(17,22,27,.85) !important; }
.cc-kpi .v { background: linear-gradient(90deg,#FBBF24,#FB923C); -webkit-background-clip:text; background-clip:text; color:transparent; }
.cc-kpi.bear .v { background:none; color:#F87171; } .cc-kpi.bull .v { background:none; color:#34D399; }
.cc-card { background: linear-gradient(160deg, rgba(34,211,238,.06), rgba(167,139,250,.06)) !important; border-color: rgba(167,139,250,.35) !important; }
.cc-card h4 { color:#A3E635; }
.cc-sec { border-image: linear-gradient(90deg,#22D3EE,#F472B6,transparent) 1; }
.stTabs [aria-selected="true"] { color:#22D3EE !important; border-bottom-color:#22D3EE !important; }
.stButton > button[kind="primary"] { background: linear-gradient(90deg,#22D3EE,#A78BFA) !important; color:#06080C !important; border:0 !important; }
.stButton > button:hover { border-color:#F472B6 !important; color:#F472B6 !important; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg,#0B1018,#0E1220) !important; }
div[data-testid="stExpander"] { border-color: rgba(34,211,238,.35) !important; }
.cc-badge.ok { border-color:#34D399; color:#34D399; } .cc-badge.warn { border-color:#FBBF24; color:#FBBF24; }
"""


def inject_theme() -> None:
    ui.inject_css()
    st.markdown(f"<style>{_THEME_CSS}</style>", unsafe_allow_html=True)


# ------------------------------------------------------------- controls -----
PRESETS = {"report": {"end": "2024-12"}, "latest": {"end": "latest"}}
DEFAULTS = {"start": "2006-04", "train_frac": 0.80, "val_frac": 0.10, "vol_window": 3, "n_states": 3, "n_seeds": 8,
            "max_parents": 3, "scores": ["bic", "k2", "bdeu"], "alpha": 1.0, "lag": 0, "filtered": False,
            "prev": False, "seed_edges": False, "force": False, "mode": "Bundled snapshot"}


def _defaults(preset: str) -> dict:
    return {**DEFAULTS, **PRESETS[preset]}


def _reset(prefix: str, preset: str) -> None:
    for k, v in _defaults(preset).items():
        st.session_state[f"{prefix}.{k}"] = v


def month_options(cfg_mode: str, nonce: int) -> list[str]:
    info = P.dataset_info(G.GWP3Config(data_mode=cfg_mode, refresh_nonce=nonce))
    return [d.strftime("%Y-%m") for d in pd.date_range(info["first"], info["last"], freq="MS")]


def sim_controls(prefix: str, preset: str = "report", groups=("data", "split", "hmm", "bn", "design")) -> G.GWP3Config:
    """Parameter panel with a reset button. Returns the resulting immutable config."""
    d = _defaults(preset)
    for k, v in d.items():
        st.session_state.setdefault(f"{prefix}.{k}", v)
    k = lambda n: f"{prefix}.{n}"
    nonce = st.session_state.setdefault("nonce", 0)
    with st.expander("⚙️ Parameters, filters & time range", expanded=True):
        top = st.columns([3, 1])
        top[1].button("↺ Reset all", key=k("reset"), on_click=_reset, args=(prefix, preset), width="stretch",
                      help="Restore every parameter and the time range on this page to its default.")
        mode = "snapshot"
        if "data" in groups:
            c = st.columns([2, 1])
            lab = c[0].radio("Data source", ["Bundled snapshot", "Live (fetch now)"], key=k("mode"), horizontal=True,
                             help="Live tries FRED/Yahoo first and falls back to cache, then the bundled snapshot.")
            mode = "live" if lab.startswith("Live") else "snapshot"
            if c[1].button("⟳ Refresh live data", key=k("refresh")):
                st.session_state["nonce"] = nonce + 1
                P._dataset.cache_clear(); P.run_cached.cache_clear()
                st.rerun()
            opts = month_options(mode, st.session_state["nonce"])
            ends = ["latest"] + opts
            cur_end = st.session_state[k("end")]
            if cur_end not in ends:
                st.session_state[k("end")] = "latest"
            if st.session_state[k("start")] not in opts:
                st.session_state[k("start")] = opts[0]
            a, b = st.columns(2)
            start = a.selectbox("Start month", opts[:-24], key=k("start"))
            end = b.selectbox("End month (latest = last complete month)", ends, key=k("end"))
        else:
            start, end = d["start"], d["end"]
        vals = {}
        if "split" in groups:
            a, b = st.columns(2)
            vals["train_frac"] = a.slider("Training share", 0.50, 0.90, step=0.05, key=k("train_frac"))
            vmax = round(min(0.30, 0.95 - vals["train_frac"]), 2)
            st.session_state[k("val_frac")] = min(st.session_state[k("val_frac")], vmax)
            vals["val_frac"] = b.slider("Validation share (test = remainder)", 0.05, max(vmax, 0.05), step=0.05, key=k("val_frac"))
        if "hmm" in groups:
            a, b, c = st.columns(3)
            vals["vol_window"] = a.slider("Volatility window (months)", 2, 12, key=k("vol_window"))
            vals["n_states"] = b.select_slider("HMM states", [2, 3], key=k("n_states"))
            vals["n_seeds"] = c.slider("HMM restarts (BIC choice)", 1, 16, key=k("n_seeds"))
        if "bn" in groups:
            a, b, c = st.columns(3)
            vals["max_parents"] = a.slider("Max parents per node", 1, 4, key=k("max_parents"))
            scores = b.multiselect("Scoring functions to compare", ["bic", "aic", "k2", "bdeu"], key=k("scores"))
            vals["scores"] = tuple(scores or ["bic"])
            vals["laplace_alpha"] = c.slider("Laplace smoothing α", 0.0, 5.0, step=0.5, key=k("alpha"))
        if "design" in groups:
            a, b, c, e, f = st.columns(5)
            vals["lag"] = a.select_slider("Evidence lag (months)", [0, 1, 2, 3], key=k("lag"),
                                          help="0 = same-month nowcast; ≥1 = genuine forecast.")
            vals["use_filtered_labels"] = b.toggle("Causal labels", key=k("filtered"))
            vals["add_prev_regime"] = c.toggle("Previous-regime node", key=k("prev"))
            vals["seed_edges"] = e.toggle("Expert-seeded start", key=k("seed_edges"))
            vals["force_connected"] = f.toggle("Force target connected", key=k("force"))
    return G.GWP3Config(start=start + "-01", end="" if end == "latest" else end + "-01", data_mode=mode,
                        refresh_nonce=st.session_state["nonce"], **vals)


def safe_run(cfg: G.GWP3Config):
    """Run the pipeline; show a clear message instead of a traceback."""
    try:
        with st.spinner("Fitting HMM and Bayesian networks on the training block…"):
            res = P.run_cached(cfg)
        if len(res.prep.panel) < 60:
            st.error("Fewer than 60 usable months in this window — widen the time range.")
            return None
        return res
    except Exception as exc:  # noqa: BLE001
        st.error(f"This parameter combination could not be fitted: {type(exc).__name__}: {str(exc)[:200]}. "
                 "Try widening the time range, using fewer states, or pressing Reset.")
        return None


def data_banner(cfg: G.GWP3Config) -> None:
    info = P.dataset_info(cfg)
    orig = set(info["origins"].values())
    tone = "ok" if "live" in orig else "warn"
    ui.badges([(f"Data origin: {', '.join(sorted(orig))}", tone), (f"Last complete month: {info['last']:%b %Y}", "info")])
    for w in info["warnings"][:3]:
        st.caption(f"⚠ {w}")
