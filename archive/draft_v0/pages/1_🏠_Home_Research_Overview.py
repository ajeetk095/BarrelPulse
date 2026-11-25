"""PAGE 1 — GWP3 Home / Results Overview."""
import sys, os
import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.pipeline_gwp3 import build_all

st.title("🧠🛢️ CrudeCortex — GWP3")
st.markdown("### *The brain behind the barrel.*")
st.markdown("**Interpretation of Results & Model Improvement** — the final stage "
            "of the mini-capstone: is the objective fulfilled, and what does it mean?")
st.caption("Group 14660 — Ajeet Kumar · Frank Osei · Nha Tran · "
           "Data: FRED · chronological 80/10/10 split, anti-leakage protocol")

@st.cache_data(show_spinner="Running the full GWP3 pipeline (HMM → BN → backtest)…")
def _res():
    return build_all()

res = _res()

st.info("**What GWP3 adds.** GWP1 collected the data, GWP2 built the methodology. "
        "GWP3 *judges it*: strict chronological splits, re-running the Bayesian "
        "network with Hill Climb Search, measuring out-of-sample forecast accuracy, "
        "simulating a regime-based trading strategy, critically assessing the 8 "
        "proposed contributions of Alvi (2018), and improving the model.")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Feature rows", str(len(res.feature_df)), "2006-04 → 2024-12")
k2.metric("Train / Val / Test", f"{len(res.splits['train'])}/{len(res.splits['val'])}/{len(res.splits['test'])}",
          "chronological")
k3.metric("Validation accuracy", f"{res.val_acc:.1f}%", "Alvi ≈ 32.1%")
k4.metric("Test accuracy", f"{res.test_acc:.1f}%", "Alvi ≈ 57.1%")
k5.metric("Strategy vs B&H", f"{res.trading['outperformance_pp']:+.1f} pp",
          f"Sharpe {res.trading['strat_sharpe']:.2f} vs {res.trading['bh_sharpe']:.2f}")

st.markdown("## WTI with the learned regimes")
fig = go.Figure()
colors = {"Bull": "#2E7D32", "Bear": "#C62828", "Stagnant": "#E8C468"}
fig.add_scatter(x=res.df.index, y=res.df["WTI"], name="WTI spot",
                line=dict(width=1.5, color="#1F6FB2"))
lab = res.labels.reindex(res.df.index)
for r, c in colors.items():
    m = lab == r
    fig.add_trace(go.Scatter(x=res.df.index[m], y=res.df["WTI"][m], mode="markers",
                             marker=dict(size=5, color=c), name=r))
tr_end = res.splits["train"][-1]
va_end = res.splits["val"][-1]
for x, txt in [(tr_end, "train | validation"), (va_end, "validation | test")]:
    fig.add_vline(x=x, line_dash="dot", line_color="#6B7280")
fig.update_layout(title="Viterbi regimes over the full study window — dotted lines mark "
                        "the anti-leakage split boundaries",
                  xaxis_title="Month", yaxis_title="USD per barrel", height=430)
st.plotly_chart(fig, use_container_width=True)

st.markdown("## The 7 official steps")
st.markdown("""
| Step | Task | Page |
|---|---|---|
| 1 | Purpose of training / validation / testing sets (Students A/B/C) | ✂️ Datasets & Splits |
| 2 | Compare validation vs testing; show the allocation | ✂️ Datasets & Splits |
| 3 | Model validation (dissertation §4.3) | ✅ Validation & Replication |
| 4 | Re-run the BN with Hill Climb; confirm replication | ✅ Validation & Replication |
| 5 | Report forecast accuracy + graphical display (+ trading sim) | 🎯 Accuracy & Trading |
| 6 | Assess the 8 proposed contributions of Alvi (2018) with page cites | 🧾 Contribution Scorecard |
| 7 | Non-technical "Discussion" — better / faster / cheaper | 💬 Discussion |
| Suppl | Model improvement — lag-augmented BN | 🚀 Model Improvement |
""")

st.markdown("### 🧭 Understand this page")
st.markdown(f"""
<div class='insight-num'><b>{len(res.feature_df)} feature rows (80/10/10):</b> — every month is one observation; the earliest 80% teaches the models, the next 10% tunes, the final 10% grades. Nothing leaks backwards in time.</div>
<div class='insight-num'><b>Validation accuracy {res.val_acc:.1f}%:</b> — how the tuned model scored on unseen 2021-22 months; Alvi (2018) managed ≈32% on his window.</div>
<div class='insight-num'><b>Test accuracy {res.test_acc:.1f}%:</b> — the final exam on 2023-24 months; the coin-flip bar is 33.3%, Alvi's benchmark ≈57%.</div>
<div class='insight-num'><b>Trading out-performance {res.trading['outperformance_pp']:+.1f} pp:</b> — going long only in predicted-Bull months and flat otherwise beat buy-and-hold on the test window with a far smaller drawdown.</div>
""", unsafe_allow_html=True)
st.markdown("<div class='insight-plain'>🟢 <b>For non-technical readers:</b><br>"
            "This is the final exam report. The model was taught on old data, "
            "quizzed on middle data, and graded on the newest data it had never "
            "seen. It beat both a coin flip and the original paper's benchmark.</div>",
            unsafe_allow_html=True)
with st.expander("🔬 Technical detail"):
    st.caption("Chronological split (no shuffling); StandardScaler, HMM tertile "
               "thresholds and BN CPDs fitted on the training slice only. HMM: 3-state "
               "Gaussian, 10 restarts, best log-likelihood. BN: Hill Climb Search under "
               "BDeu (ESS=1) vs BIC comparison. Accuracy = exact regime-class match on "
               "each unseen month; trading = long-in-Bull/flat-otherwise on WTI "
               "monthly returns, costs excluded (disclosed).")
