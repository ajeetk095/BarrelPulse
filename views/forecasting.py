import streamlit as st

from core import gwp3_charts as X
from core import gwp3_pipeline as P
from core import gwp3_ui as U
from core import ui_components as ui

ui.page_header("Forecasting results", "Accuracy metrics, baselines and graphs for the selected network.", "Results")
cfg = U.sim_controls("fc", "report", groups=("data", "split", "bn", "design"))
U.data_banner(cfg)
res = U.safe_run(cfg)
if res is None:
    st.stop()
blk = st.radio("Block", ["val", "test"], format_func=lambda x: {"val": "Validation", "test": "Testing (evaluated once)"}[x], horizontal=True, key="fc.blk")
panel = res.prep.panel; sp = res.prep.split.reindex(panel.index)
y = panel.loc[sp == blk, "WTI_Regime"]; m = res.metrics[blk]
ui.kpis([("Accuracy", U.pct(m["accuracy"]), f"95% CI {U.pct(m['ci_low'])}–{U.pct(m['ci_high'])}", "acc"), ("Macro-F1", f"{m['macro_f1']:.2f}", "present classes", ""),
         ("Balanced acc.", U.pct(m["balanced_acc"]), "mean recall", ""), ("Log-loss", f"{m['log_loss']:.2f}", "lower is better", "")])
st.plotly_chart(X.fig_accuracy_vs_baselines(m, f"{blk.title()} accuracy vs baselines"), width="stretch")
c1, c2 = st.columns(2)
c1.plotly_chart(X.fig_confusion(res.confusion[blk], "Confusion matrix"), width="stretch")
c2.dataframe(res.report[blk].style.format(precision=2), width="stretch")
st.plotly_chart(X.fig_actual_vs_pred(y, res.pred[blk], "Actual vs predicted regime"), width="stretch")
st.plotly_chart(X.fig_prob_stack(res.probs[blk], "Predicted regime probabilities"), width="stretch")
st.plotly_chart(X.fig_dag(res.bn), width="stretch")
if blk == "test":
    ui.callout("warn", "Read with care", f"Binomial p vs majority = {m['binom_p_vs_baseline']:.2f}; McNemar p = {m['mcnemar_p_vs_majority']:.2f}. Regimes absent from the test block make some recalls undefined.")
ui.disclaimer()
