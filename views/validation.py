import pandas as pd
import streamlit as st

from core import gwp3_config as G
from core import gwp3_ui as U
from core import ui_components as ui

ui.page_header("Validation & replication of Alvi (2018) §4.3", "What was replicated, what was not, and the verdict — with adjustable settings.", "Replication")
ui.callout("warn", "Replication verdict", G.REPLICATION_VERDICT)
cfg = U.sim_controls("val", "report", groups=("split", "bn", "design"))
res = U.safe_run(cfg)
if res is None:
    st.stop()
a, r = G.ALVI_VERIFIED, G.RECORDED_NOTEBOOK
tab = pd.DataFrame({"Alvi (2018) — verified in PDF": [f"{a['val_accuracy_pct']:.1f}%", f"{a['test_accuracy_pct']:.1f}%", f"{a['test_n']} obs"],
                    "Group notebook — recorded": [f"{100*r['val_accuracy']:.2f}%", f"{100*r['test_accuracy']:.2f}%", "25 obs"],
                    "This app — computed (current settings)": [U.pct(res.metrics['val']['accuracy']), U.pct(res.metrics['test']['accuracy']), f"{res.metrics['test']['n']} obs"]},
                   index=["Validation accuracy", "Test accuracy", "Test size"])
st.dataframe(tab, width="stretch")
ui.callout("note", "Why they are not comparable", "Different sample, Gaussian two-feature HMM vs Alvi's per-series binary-difference HMM, tertile discretisation, no OPEC node. Alvi's error uses `np.roll(pred, 1)`.")
st.subheader("Scoring-function selection (validation only)")
st.dataframe(res.candidates.style.format({"val_accuracy": "{:.1%}", "val_log_loss": "{:.3f}"}), width="stretch")
st.caption(f"Selected: **{res.selected}**. Parents of WTI_Regime: {res.target_parents or 'none'} (Alvi p. 62: rerun if disconnected — use *Force target connected*).")
with st.expander("Verified Alvi facts and unverified items"):
    st.json({k: v for k, v in a.items() if k != "contributions"})
ui.disclaimer()
