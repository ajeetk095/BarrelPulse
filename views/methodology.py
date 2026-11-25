import streamlit as st

from core import gwp3_config as G
from core import gwp3_pipeline as P
from core import gwp3_ui as U
from core import ui_components as ui

ui.page_header("Methodology, data provenance & limitations", "How each number is produced and what could be wrong.", "Methodology")
c = G.GWP3Config()
st.json({"default window": [c.start, c.end], "split": f"{c.train_frac:.0%}/{c.val_frac:.0%}/{1-c.train_frac-c.val_frac:.0%}",
         "features": f"WTI log-return, {c.vol_window}-month rolling std", "HMM": f"{c.n_states} states, Gaussian full covariance, {c.n_seeds} seeds, BIC",
         "BN nodes": [G.TARGET, *G.MACRO_NODES], "scores": c.scores, "max parents": c.max_parents, "CPDs": f"Laplace alpha={c.laplace_alpha}"})
st.subheader("Data provenance")
mode = st.radio("Show provenance for", ["snapshot", "live"], horizontal=True, key="meth.mode")
info = P.dataset_info(G.GWP3Config(data_mode=mode, refresh_nonce=st.session_state.get("nonce", 0)))
st.write(f"Monthly panel {info['first']:%b %Y} → {info['last']:%b %Y} ({info['n']} months), fingerprint `{info['fingerprint']}`; origins: {info['origins']}")
st.dataframe(info["status"], width="stretch")
for w in info["warnings"]:
    st.caption(f"⚠ {w}")
ui.cards([("Leakage controls", "Scaler, HMM, cut-offs and network are fitted on training months only; Viterbi per block uses that block's future — causal labels avoid it."),
          ("Selection", "Scoring function chosen by validation accuracy (tie-break log-loss); test evaluated once."),
          ("Live data", "FRED (CSV or API key), EIA (API key) and Yahoo Finance; ordered fallbacks; last complete month only."),
          ("Statistics", "Wilson CIs, exact binomial, exact McNemar, KS, Mann–Whitney, Fisher exact.")], per_row=4)
ui.callout("warn", "Limitations", "Small validation/test blocks; unsupervised labels; discretisation loses information; associations are not causation; no like-for-like replication of Alvi; monthly resolution; live providers can be rate-limited.")
ui.disclaimer()
