import pandas as pd
import streamlit as st

from core import gwp3_charts as X
from core import gwp3_config as G
from core import gwp3_pipeline as P
from core import gwp3_ui as U
from core import ui_components as ui

ui.page_header("Simulation lab", "Change any parameter, filter or time range and re-run the whole HMM + Bayesian-network pipeline.", "Interactive")
cfg = U.sim_controls("lab", "latest")
U.data_banner(cfg)
res = U.safe_run(cfg)
if res is None:
    st.stop()
m_v, m_t = res.metrics["val"], res.metrics["test"]
ui.kpis([("Months used", str(len(res.prep.panel)), f"{res.prep.raw.index[0]:%b %Y} → {res.prep.raw.index[-1]:%b %Y}", ""),
         ("Selected score", res.selected.upper(), "chosen on validation", "acc"),
         ("Validation acc.", U.pct(m_v["accuracy"]), f"majority {U.pct(m_v['majority_acc'])}", ""),
         ("Test acc.", U.pct(m_t["accuracy"]), f"majority {U.pct(m_t['majority_acc'])}", "bear" if m_t["accuracy"] < m_t["majority_acc"] else "bull")])
if m_t["accuracy"] < m_t["majority_acc"]:
    ui.callout("warn", "Below the naive baseline", "With these settings the network does not beat always predicting the most common regime on the test block.")
t1, t2, t3, t4 = st.tabs(["3D regime space", "Regimes over time", "Network & scores", "Metrics & export"])
with t1:
    only = st.multiselect("Show regimes", list(G.REGIMES), default=list(G.REGIMES), key="lab.show")
    lab = res.prep.regimes_viterbi
    raw = res.prep.raw
    st.plotly_chart(X.fig3d_feature_space(raw[lab.isin(only)], lab[lab.isin(only)]), width="stretch")
    st.caption("Drag to rotate, scroll to zoom. Features and labels as defined in Methodology.")
with t2:
    st.plotly_chart(X.fig_filtered_probs(res.prep.probs_filtered, res.prep.split), width="stretch")
    blk = st.radio("Block", ["val", "test"], horizontal=True, key="lab.blk")
    y = res.prep.panel.loc[res.prep.split.reindex(res.prep.panel.index) == blk, "WTI_Regime"]
    st.plotly_chart(X.fig_actual_vs_pred(y, res.pred[blk], f"Actual vs predicted ({blk})"), width="stretch")
with t3:
    st.dataframe(res.candidates.style.format({"val_accuracy": "{:.1%}", "val_log_loss": "{:.3f}"}), width="stretch")
    st.plotly_chart(X.fig_dag(res.bn), width="stretch")
with t4:
    st.dataframe(pd.DataFrame(res.metrics).T.style.format(precision=3), width="stretch")
    out = pd.concat({"actual": res.prep.panel["WTI_Regime"], "predicted_val": res.pred["val"], "predicted_test": res.pred["test"]}, axis=1)
    st.download_button("⬇ Download labels & predictions (CSV)", out.to_csv().encode(), "gwp3_simulation.csv", "text/csv")
ui.disclaimer()
