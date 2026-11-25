import streamlit as st

from core import gwp3_config as G
from core import gwp3_pipeline as P
from core import gwp3_ui as U
from core import ui_components as ui
from core import gwp3_charts as X

cfg = G.GWP3Config(end="")                 # latest complete month of the bundled/cached data
res = U.safe_run(cfg)
ui.page_header("BarrelPulse — GWP3", "Hidden Markov regimes + Bayesian networks for crude oil: results, simulations and live monitoring.",
               "MScFE 660 · Group 14660 · Ajeet Kumar")
if res is None:
    st.stop()
U.data_banner(cfg)
v = P.latest_view(cfg)
m_t = res.metrics["test"]
ui.kpis([("Latest month", f"{v['month']:%b %Y}", "complete monthly data", ""),
         ("Current regime", v["regime"], f"{v['months_in_regime']} month(s) in a row", {"Bear": "bear", "Bull": "bull"}.get(v["regime"], "stag")),
         ("Test accuracy", U.pct(m_t["accuracy"]), f"majority baseline {U.pct(m_t['majority_acc'])}", "bear" if m_t["accuracy"] < m_t["majority_acc"] else "bull"),
         ("Test months", str(m_t["n"]), "small sample", "stag")])
st.plotly_chart(X.fig3d_feature_space(res.prep.raw, res.prep.regimes_viterbi), width="stretch")
ui.callout("warn", "Honest headline",
           "The network is not shown to beat the naive 'always predict the most common regime' rule on held-out months — in the submitted notebook (64.0% vs 68.0%) and in these recomputed runs. "
           "The app's value is a transparent pipeline you can stress-test, not a trading signal.")
st.subheader("Where to go")
ui.cards([("📡 Live Market", "Daily FRED/EIA/Yahoo quotes, today's regime and next-month probabilities."),
          ("🧪 Simulation Lab", "Change dates, splits, HMM and network settings; 3D regime space; export."),
          ("🎛 Scenario Explorer", "Set VIX/dollar/industry/energy-CPI conditions; 3D probability surface."),
          ("🎲 Risk & Robustness", "Monte-Carlo regime paths, sweeps, trading illustration."),
          ("✅ Validation · Forecasting · Improvements", "Replication of Alvi (2018) §4.3, metrics, variants."),
          ("📚 Contributions · Discussion · Methodology", "Assessment of the eight contributions and plain-language findings.")], per_row=3)
st.subheader("Insight cards")
c1, c2 = st.columns(2)
with c1:
    U.insight("Model vs trivial baselines", f"Test accuracy {U.pct(m_t['accuracy'])}; majority {U.pct(m_t['majority_acc'])}; persistence {U.pct(m_t['persistence_acc'])}.",
              f"{m_t['correct']}/{m_t['n']} correct, 95% CI {U.pct(m_t['ci_low'])}–{U.pct(m_t['ci_high'])}; McNemar vs majority p={m_t['mcnemar_p_vs_majority']:.2f}.",
              "Macro nodes add little beyond how common each regime is.", "A forecast must beat a naive rule to be useful.", "Small test block; wide interval.", U.stamp(res))
with c2:
    U.insight("Today's read-out", f"{v['regime']} regime in {v['months_in_regime']:d} consecutive month(s) to {v['month']:%b %Y}.",
              f"Next month (HMM): {v['hmm_next'].round(2).to_dict()}.", "Regimes persist; expect continuity unless conditions shift.",
              "Context for monitoring risk.", "Monthly resolution; not a price forecast.", U.stamp(res))
ui.disclaimer()
