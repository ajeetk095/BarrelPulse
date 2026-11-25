import streamlit as st

from core import gwp3_charts as X
from core import gwp3_config as G
from core import gwp3_pipeline as P
from core import gwp3_ui as U
from core import ui_components as ui

ui.page_header("Data & splits", "Training / validation / testing definitions, filters and the validation-vs-testing comparison.", "Datasets")
cfg = U.sim_controls("data", "report", groups=("data", "split", "hmm"))
U.data_banner(cfg)
res = U.safe_run(cfg)
if res is None:
    st.stop()
d = P.describe_splits(res.prep)
st.plotly_chart(X.fig_split_allocation(d), width="stretch")
st.dataframe(d.style.format(precision=2), width="stretch")
a, b = st.columns(2)
a.plotly_chart(X.fig_regime_share(d), width="stretch")
b.plotly_chart(X.fig_acf(res.tests["acf"], res.tests["acf_ci"]), width="stretch")
st.subheader("3D regime space with filters")
f = st.columns(3)
blk = f[0].multiselect("Splits shown", ["train", "val", "test"], default=["train", "val", "test"], key="data.f1")
rg = f[1].multiselect("Regimes shown", list(G.REGIMES), default=list(G.REGIMES), key="data.f2")
raw, lab, sp = res.prep.raw, res.prep.regimes_viterbi, res.prep.split
m = sp.isin(blk) & lab.isin(rg)
st.plotly_chart(X.fig3d_feature_space(raw[m], lab[m]), width="stretch")
t = res.tests
st.subheader("Validation vs testing")
st.dataframe({"Test": ["Kolmogorov–Smirnov (returns)", "Mann–Whitney U (returns)", t["regime_test"]], "p-value": [t["ks_p"], t["mw_p"], t["regime_p"]]}, width="stretch")
st.download_button("⬇ Download modelling panel (CSV)", res.prep.raw.to_csv().encode(), "gwp3_panel.csv", "text/csv")
with st.expander("Recorded notebook values (not recomputed)"):
    U.provenance("recorded"); r = G.RECORDED_NOTEBOOK
    st.write({"split n": r["split"], "KS p": r["ks_p"], "Mann-Whitney p": r["mw_p"], "contingency": r["val_test_contingency"]})
ui.disclaimer()
