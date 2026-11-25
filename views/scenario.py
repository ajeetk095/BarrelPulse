import pandas as pd
import streamlit as st

from core import bn_engine as B
from core import gwp3_charts as X
from core import gwp3_config as G
from core import gwp3_pipeline as P
from core import gwp3_ui as U
from core import ui_components as ui

ui.page_header("Scenario explorer", "Set market conditions and read the network's regime probabilities.", "What-if")
cfg = U.sim_controls("scn", "latest", groups=("data", "split", "bn", "design"))
res = U.safe_run(cfg)
if res is None:
    st.stop()
bn = res.bn
KEY = "scn.ev"
defaults = {n: "(any)" for n in G.MACRO_NODES}
def _rst():
    for n in G.MACRO_NODES:
        st.session_state[f"{KEY}.{n}"] = "(any)"
st.subheader("Set evidence")
cols = st.columns(len(G.MACRO_NODES) + 1)
ev = {}
for col, n in zip(cols, G.MACRO_NODES):
    if n in bn.states:
        v = col.selectbox(n, ["(any)"] + list(bn.states[n]), key=f"{KEY}.{n}")
        if v != "(any)":
            ev[n] = v
cols[-1].button("↺ Reset scenario", on_click=_rst)
try:
    post = B.infer_network(bn, G.TARGET, ev)
except B.InferenceError as e:
    st.warning(str(e)); st.stop()
st.plotly_chart(X.fig_prob_stack(pd.DataFrame([post.reindex(list(G.REGIMES))]).rename(index={0: "Scenario"}), "P(regime | scenario)"), width="stretch")
emp, n = B.infer_empirical(bn, G.TARGET, ev)
st.caption(f"Learned parents of {G.TARGET}: {bn.parents_of(G.TARGET) or 'none'} · training months matching this scenario: {n}")
if not bn.parents_of(G.TARGET):
    ui.callout("note", "Flat network", "The selected network has no parents for the regime, so evidence does not change the answer. Turn on **Force target connected**, or compare with the empirical frequencies below.")
st.subheader("3D conditional surface")
a, b, c, d = st.columns(4)
nx = a.selectbox("X node", list(G.MACRO_NODES), 0, key="scn.x"); ny = b.selectbox("Y node", list(G.MACRO_NODES), 2, key="scn.y")
rg = c.selectbox("Regime", list(G.REGIMES), key="scn.r"); me = d.radio("Source", ["empirical", "network"], key="scn.m", horizontal=True)
if nx == ny:
    st.info("Pick two different nodes.")
else:
    g = P.bn_probability_grid(bn, nx, ny, rg, me)
    st.plotly_chart(X.fig3d_surface(g, f"P({rg}) by {nx} × {ny} ({me}, training data)", ny, nx, f"P({rg})"), width="stretch")
    st.caption("Empirical cells with no training months are blank. Associations, not causes.")
ui.disclaimer()
