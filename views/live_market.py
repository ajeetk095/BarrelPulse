import pandas as pd
import streamlit as st

from core import gwp3_charts as X
from core import gwp3_config as G
from core import gwp3_pipeline as P
from core import gwp3_ui as U
from core import live_monitor as L
from core import ui_components as ui

ui.page_header("Live market & today's regime", "Daily quotes from FRED / EIA / Yahoo Finance and the model's read-out for the latest complete month.", "Live monitor")

@st.cache_data(ttl=1800, show_spinner="Fetching daily quotes (FRED → EIA/Yahoo fallbacks)…")
def _quotes(nonce: int):
    return L.fetch_monitor()

c = st.columns([1, 1, 3])
if c[0].button("⟳ Refresh live data", type="primary"):
    st.session_state["nonce"] = st.session_state.get("nonce", 0) + 1
    _quotes.clear(); P._dataset.cache_clear(); P.run_cached.cache_clear()
    st.rerun()
nonce = st.session_state.get("nonce", 0)
q = _quotes(nonce)

st.subheader("1 · Daily market tape")
cols = st.columns(3); colors = ["#22D3EE", "#F472B6", "#A3E635"]; units = ["USD / bbl", "index pts", "index"]
got = 0
for col, (key, quote), clr, un in zip(cols, q.items(), colors, units):
    with col:
        if quote.series is not None:
            got += 1; mt = L.month_to_date(quote.series)
            st.metric(quote.name, f"{mt['last']:.2f}", f"{100*mt['mtd_pct']:+.1f}% month-to-date")
            st.caption(f"{mt['date']:%d %b %Y} · source: {quote.source} · fetched {quote.fetched_utc} · 20-day ann. vol {100*mt['vol_20d_ann']:.0f}%")
        else:
            st.metric(quote.name, "unavailable")
            st.caption("; ".join(quote.errors)[:300])
if got == 0:
    ui.callout("warn", "Live quotes not reachable from this host",
               "No provider answered. On Streamlit Cloud this usually means a network or rate-limit issue, or a missing API key. "
               "The model section below still runs on the most recent cached/bundled monthly data and says so.")
else:
    t = st.tabs([v.name for v in q.values()])
    for tab, (key, quote), clr, un in zip(t, q.items(), colors, units):
        with tab:
            if quote.series is not None:
                span = st.select_slider("Time range", ["1M", "3M", "6M", "1Y", "2Y"], value="1Y", key=f"live.span.{key}")
                n = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252, "2Y": 504}[span]
                st.plotly_chart(X.fig_daily(quote.series.tail(n), f"{quote.name} — daily ({span})", un, clr), width="stretch")
            else:
                st.info("No data for this series.")

st.subheader("2 · What the model says for the latest complete month")
cfg = U.sim_controls("live", "latest", groups=("data", "hmm", "bn"))
from dataclasses import replace
cfg = replace(cfg, lag=0)
U.data_banner(cfg)
try:
    with st.spinner("Computing the latest regime read-out…"):
        v = P.latest_view(cfg)
except Exception as exc:  # noqa: BLE001
    st.error(f"Model could not run for these settings: {type(exc).__name__}: {str(exc)[:180]}"); st.stop()
ui.kpis([("Latest complete month", f"{v['month']:%b %Y}", "monthly data", ""),
         ("Current regime (causal)", v["regime"], f"{v['months_in_regime']} month(s) in a row", {"Bear": "bear", "Bull": "bull"}.get(v["regime"], "stag")),
         ("P(stay in regime)", f"{v['persist_prob']:.0%}", "HMM self-transition", "acc"),
         ("WTI (monthly avg)", f"${v['raw_last']['WTI']:.2f}", "model input", "")])
st.plotly_chart(X.fig_next_probs(v["filtered"], v["hmm_next"], v["bn_next"]), width="stretch")
ev = pd.DataFrame({"Evidence node": list(v["evidence"].keys()), "Discretised state": list(v["evidence"].values())})
a, b = st.columns([1, 2]); a.dataframe(ev, hide_index=True, width="stretch")
b.markdown("**Reading this:** the first bar group is where the HMM places the market *now* using data up to the latest month only (no look-ahead); "
           "the next groups are model-implied probabilities for *next* month. The Bayesian-network bar is learned with one-month-lagged evidence "
           f"and the **{v['selected_score'].upper()}** score (chosen on validation). Earlier validation showed these models do not reliably beat "
           "a naive 'same as last month' rule — treat this as a descriptive dashboard.")
U.insight("Today's regime read-out", f"Latest complete month {v['month']:%b %Y}: {v['regime']} regime, {v['months_in_regime']} month(s) long.",
          f"Filtered probabilities {v['filtered'].round(2).to_dict()}; next-month HMM {v['hmm_next'].round(2).to_dict()}.",
          "Regimes are persistent, so next month is most likely the same regime.", "Context for risk monitoring only.",
          "Monthly resolution; mid-month moves are not reflected; unsupervised labels; small test samples.", U.stamp(P.run_cached(cfg)))
ui.disclaimer()
