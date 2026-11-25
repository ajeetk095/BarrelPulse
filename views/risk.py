import numpy as np
import streamlit as st

from core import gwp3_charts as X
from core import gwp3_config as G
from core import gwp3_pipeline as P
from core import gwp3_ui as U
from core import ui_components as ui

ui.page_header("Risk & robustness", "Monte-Carlo regime simulation, robustness checks and an illustrative rule. Not investment advice.", "Simulation")
cfg = U.sim_controls("risk", "latest", groups=("data", "hmm", "bn", "design"))
U.data_banner(cfg)
res = U.safe_run(cfg)
if res is None:
    st.stop()
fit = res.prep.fit; names = fit.names
st.subheader("1 · Monte-Carlo regime paths (from the fitted transition matrix)")
DEF = {"h": 12, "n": 2000, "seed": 0, "start": "Latest month (filtered)"}
for k, v in DEF.items():
    st.session_state.setdefault(f"risk.mc.{k}", v)
c = st.columns(5)
h = c[0].slider("Horizon (months)", 3, 36, key="risk.mc.h"); n = c[1].select_slider("Paths", [200, 500, 1000, 2000, 5000], key="risk.mc.n")
seed = c[2].number_input("Random seed", 0, 9999, key="risk.mc.seed")
start = c[3].selectbox("Starting state", ["Latest month (filtered)", *names], key="risk.mc.start")
c[4].button("↺ Reset simulation", on_click=lambda: [st.session_state.__setitem__(f"risk.mc.{k}", v) for k, v in DEF.items()])
sp = res.prep.probs_filtered.iloc[-1].values if start.startswith("Latest") else np.eye(len(names))[names.index(start)]
paths, prob = P.simulate_regimes(fit, sp, h, int(n), int(seed))
a, b = st.columns(2)
a.plotly_chart(X.fig_fan(prob, names, "Regime probability fan"), width="stretch")
b.plotly_chart(X.fig3d_mc(prob, names, "3D probability surface"), width="stretch")
st.plotly_chart(X.fig3d_paths(paths, names), width="stretch")
A = np.exp(fit.log_A)
st.caption("Expected spell lengths (months): " + ", ".join(f"{nm} {1/(1-A[i,i]):.1f}" if A[i, i] < 1 else f"{nm} ∞" for i, nm in enumerate(names)))
st.subheader("2 · Illustrative regime rule (long Bull / short Bear / flat Stagnant)")
d = st.columns(3)
blk = d[0].radio("Block", ["val", "test"], horizontal=True, key="risk.blk")
cost = d[1].slider("Round-trip cost (bp per position change)", 0, 100, 0, key="risk.cost")
d[2].caption("Same-month nowcasts are lagged one month before trading.")
tr = P.trading_demo(res, blk)
s = tr["series"].copy()
s["strategy"] = s["strategy"] - (s["position"].diff().abs().fillna(s["position"].abs()) * cost / 1e4)
cum = float((1 + s["strategy"]).prod() - 1)
ui.kpis([("Rule return", U.pct(cum), f"{blk} block, {cost} bp costs", ""), ("Buy & hold", U.pct(tr["buy_hold_cum"]), "same months", ""), ("Position changes", str(tr["n_trades"]), f"{tr['n_months']} months", "stag")])
st.plotly_chart(X.fig_trading(s), width="stretch")
ui.callout("warn", "Not evidence of a strategy", "Two dozen months cannot support a Sharpe ratio; slippage, margin and roll costs are ignored.")
st.subheader("3 · Robustness tables")
if st.button("Run robustness checks", key="risk.rob"):
    with st.spinner("Re-fitting across splits, restarts and state counts…"):
        st.session_state["risk.rob.out"] = (P.robustness_splits(cfg), P.robustness_states(cfg), P.robustness_seeds(cfg))
out = st.session_state.get("risk.rob.out")
if out:
    sp_, st_, sd_ = out
    st.dataframe(sp_.style.format({c: "{:.1%}" for c in ["val acc", "val majority", "test acc", "test majority"]}), width="stretch")
    st.dataframe(sd_.style.format({"BIC": "{:.1f}", "agreement with selected": "{:.1%}"}), width="stretch")
    st.dataframe(st_, width="stretch")
ui.disclaimer()
