import streamlit as st

from core import gwp3_charts as X
from core import gwp3_config as G
from core import gwp3_pipeline as P
from core import gwp3_ui as U
from core import ui_components as ui

ui.page_header("Model improvements & sweeps", "Compare justified variants and map accuracy across settings in 3D.", "Improvements")
cfg = U.sim_controls("imp", "report", groups=("data", "split", "hmm", "bn"))
rows = []
with st.spinner("Fitting variants…"):
    from dataclasses import replace
    for name, vc in P.VARIANTS.items():
        try:
            r = P.run_cached(replace(cfg, lag=vc.lag, use_filtered_labels=vc.use_filtered_labels, add_prev_regime=vc.add_prev_regime, seed_edges=vc.seed_edges))
            rows.append({"Variant": name, "Score": r.selected, "Val acc": r.metrics["val"]["accuracy"], "Val majority": r.metrics["val"]["majority_acc"],
                         "Test acc": r.metrics["test"]["accuracy"], "Test majority": r.metrics["test"]["majority_acc"],
                         "Test macro-F1": r.metrics["test"]["macro_f1"], "Test log-loss": r.metrics["test"]["log_loss"]})
        except Exception as e:  # noqa: BLE001
            st.warning(f"{name}: {type(e).__name__}")
if not rows:
    st.stop()
import pandas as pd
df = pd.DataFrame(rows).set_index("Variant")
sel = st.multiselect("Variants shown", list(df.index), default=list(df.index), key="imp.sel")
df = df.loc[sel] if sel else df
st.plotly_chart(X.fig_accuracy_compare(df), width="stretch")
st.dataframe(df.style.format({c: "{:.1%}" for c in ["Val acc", "Val majority", "Test acc", "Test majority"]} | {"Test macro-F1": "{:.2f}", "Test log-loss": "{:.2f}"}), width="stretch")
ui.cards([("V1 lagged evidence", "Uses last month's evidence for this month's regime — a true one-step-ahead forecast."),
          ("V2 filtered labels", "Labels use only data to date (no Viterbi look-ahead)."),
          ("V3 previous-regime node", "Adds regime(t-1); gains mostly reflect persistence."),
          ("V4 expert-seeded start", "Alvi-style warm start for hill climbing.")], per_row=4)
st.subheader("3D sensitivity sweep — training share × volatility window")
a, b = st.columns(2)
tfs = a.multiselect("Training shares", [0.6, 0.65, 0.7, 0.75, 0.8, 0.85], default=[0.6, 0.7, 0.8], key="imp.tf")
vws = b.multiselect("Volatility windows (months)", [2, 3, 4, 6, 9, 12], default=[2, 3, 6, 12], key="imp.vw")
which = st.radio("Surface", ["val_acc", "test_acc"], format_func=lambda x: "Validation accuracy" if x == "val_acc" else "Test accuracy (sensitivity only — not used for selection)", horizontal=True, key="imp.which")
if st.button("Run sweep", type="primary", key="imp.run") and len(tfs) >= 2 and len(vws) >= 2:
    with st.spinner(f"Running {len(tfs)*len(vws)} fits…"):
        st.session_state["imp.sweep"] = P.sweep_accuracy(cfg, sorted(tfs), sorted(vws))
sw = st.session_state.get("imp.sweep")
if sw is not None:
    g = sw.pivot(index="train_frac", columns="vol_window", values=which)
    st.plotly_chart(X.fig3d_surface(g, "Accuracy surface", "Volatility window (months)", "Training share", "Accuracy", zrange=(0, 1)), width="stretch")
    st.button("↺ Clear sweep", on_click=lambda: st.session_state.pop("imp.sweep", None))
else:
    st.info("Choose at least two values for each axis and press **Run sweep**.")
ui.callout("warn", "Do not over-read", "The previous-regime variant approaches the persistence baseline. Sweep surfaces are sensitivity checks, not a way to choose parameters on the test set.")
ui.disclaimer()
