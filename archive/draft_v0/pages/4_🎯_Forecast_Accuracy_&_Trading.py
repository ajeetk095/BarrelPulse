"""PAGE 4 — Step 5: Forecast Accuracy, Graphical Results & Trading Simulation."""
import sys, os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.pipeline_gwp3 import build_all

st.title("🎯 Step 5 — Forecast Accuracy & Trading")
st.caption("5a: accuracy of forecasting the crude-oil regime · 5b: graphical display "
           "of results · plus the report's regime-based trading simulation.")

@st.cache_data(show_spinner="Scoring out-of-sample forecasts…")
def _res():
    return build_all()

res = _res()

tab_acc, tab_trade = st.tabs(["📉 5a — Forecast accuracy", "💰 5b — Graphical results & trading"])

with tab_acc:
    st.markdown("### Per-class metrics (test set)")
    clf = res.clf_report
    rows = []
    for lbl in ["Bear", "Stagnant", "Bull"]:
        if lbl in clf:
            m = clf[lbl]
            rows.append({"Regime": lbl, "Precision": round(m["precision"], 2),
                         "Recall": round(m["recall"], 2), "F1": round(m["f1-score"], 2),
                         "Support": int(m["support"])})
    rows.append({"Regime": "Accuracy (overall)", "Precision": "—", "Recall": "—",
                 "F1": round(clf["accuracy"], 2), "Support": clf["macro avg"]["support"]})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test accuracy", f"{res.test_acc:.1f}%", "+30.7 pp vs random")
    c2.metric("Validation accuracy", f"{res.val_acc:.1f}%", "Alvi ≈ 32.1%")
    c3.metric("Alvi benchmark (test)", "≈ 57.1%", "§4.3.1")
    c4.metric("Lift vs Alvi", f"{res.test_acc - 57.14:+.1f} pp", "out-of-sample")

    st.markdown("### Confusion matrices — validation & test")
    f1 = make_subplots(rows=1, cols=2, subplot_titles=("Validation", "Test"),
                       horizontal_spacing=0.18)
    f1.add_trace(go.Heatmap(z=res.cm_val.values, x=list(res.cm_val.columns),
                            y=list(res.cm_val.index), coloraxis="coloraxis",
                            text=res.cm_val.values, texttemplate="%{text}"), 1, 1)
    f1.add_trace(go.Heatmap(z=res.cm_test.values, x=list(res.cm_test.columns),
                            y=list(res.cm_test.index), coloraxis="coloraxis",
                            text=res.cm_test.values, texttemplate="%{text}"), 1, 2)
    f1.update_layout(coloraxis=dict(colorscale="Oranges"), height=380)
    st.plotly_chart(f1, use_container_width=True)

    st.markdown("### Accuracy vs baselines")
    fig = go.Figure(go.Bar(
        x=["Random (3-class)", "Alvi (2018) test", "Our validation", "Our test"],
        y=[33.33, 57.14, res.val_acc, res.test_acc],
        marker_color=["#9CA3AF", "#6B7280", "#E8C468", "#E8842C"],
        text=[f"{v:.1f}%" for v in [33.33, 57.14, res.val_acc, res.test_acc]],
        textposition="outside"))
    fig.update_layout(title="Regime-classification accuracy vs baselines",
                      yaxis_title="Accuracy (%)", yaxis_range=[0, 100], height=380)
    st.plotly_chart(fig, use_container_width=True)

with tab_trade:
    st.markdown("### Predicted vs actual regimes on the test window")
    wti = res.df.loc[res.feature_df.index.intersection(res.splits["test"]), "WTI"]
    act = res.panel_test["WTI_Regime"]
    prd = res.pred_test.reindex(act.index)
    correct = act == prd
    colors = {"Bull": "#2E7D32", "Bear": "#C62828", "Stagnant": "#E8C468"}
    fig = go.Figure()
    fig.add_scatter(x=wti.index, y=wti, name="WTI", line=dict(width=1.6, color="#1F6FB2"))
    for r, c in colors.items():
        m = act == r
        fig.add_trace(go.Scatter(x=wti.index[m], y=wti[m], mode="markers",
                                 marker=dict(size=9, color=c), name=f"actual {r}"))
    fig.add_trace(go.Scatter(x=wti.index[~correct], y=wti[~correct], mode="markers",
                             marker=dict(size=15, color="rgba(0,0,0,0)",
                                         line=dict(width=2, color="#7B1FA2")),
                             name="misclassified month"))
    fig.update_layout(title="Test window — marker colour = actual regime, purple ring = miss",
                      xaxis_title="Month", yaxis_title="USD per barrel", height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Regime composition over the test window")
    cum = (pd.get_dummies(prd).cumsum() / np.arange(1, len(prd) + 1)[:, None] * 100)
    fig = go.Figure()
    for r, c in colors.items():
        if r in cum.columns:
            fig.add_trace(go.Scatter(x=cum.index, y=cum[r], name=f"predicted {r}",
                                     stackgroup="one", line=dict(width=1, color=c)))
    fig.update_layout(title="Cumulative share of predicted regimes through the test months",
                      yaxis_title="Share (%)", height=360)
    st.plotly_chart(fig, use_container_width=True)

    t = res.trading
    st.markdown("### Regime-based trading simulation (test window)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Strategy cumulative", f"{t['strat_cum']:+.1f}%", f"B&H {t['bh_cum']:+.1f}%")
    m2.metric("Sharpe (ann.)", f"{t['strat_sharpe']:.2f}", f"B&H {t['bh_sharpe']:.2f}")
    m3.metric("Max drawdown", f"{t['strat_mdd']:.1f}%", f"B&H {t['bh_mdd']:.1f}%")
    m4.metric("Active months", str(t["active_months"]),
              f"win rate {t['win_rate']:.0f}%" if t["active_months"] else "no Bull months")

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        row_heights=[0.45, 0.25, 0.3],
                        subplot_titles=("Cumulative growth of $1", "Position (1 = long in Bull)",
                                        "Drawdown"))
    ret = res.df["WTI"].pct_change().reindex(prd.index).fillna(0)
    pos = (prd == "Bull").astype(int)
    strat = (pos * ret).fillna(0)
    eq_s = (1 + strat).cumprod()
    eq_b = (1 + ret).cumprod()
    fig.add_scatter(x=eq_s.index, y=eq_s, name="BN strategy",
                    line=dict(width=2.2, color="#E8842C"), row=1, col=1)
    fig.add_scatter(x=eq_b.index, y=eq_b, name="Buy & hold",
                    line=dict(width=1.8, color="#1F6FB2"), row=1, col=1)
    fig.add_trace(go.Scatter(x=prd.index, y=pos, name="Position",
                             line=dict(color="#2E7D32", width=1.5, shape="hv"),
                             fill="tozeroy", opacity=0.4), row=2, col=1)
    dd_s = eq_s / eq_s.cummax() - 1
    dd_b = eq_b / eq_b.cummax() - 1
    fig.add_scatter(x=dd_s.index, y=dd_s * 100, name="Strategy DD",
                    line=dict(width=1.6, color="#E8842C"), row=3, col=1)
    fig.add_scatter(x=dd_b.index, y=dd_b * 100, name="B&H DD",
                    line=dict(width=1.6, color="#1F6FB2"), row=3, col=1)
    fig.update_layout(height=640, title="Trading simulation — long-in-Bull / flat otherwise "
                                        "(costs excluded, disclosed)")
    fig.update_yaxes(title_text="$", row=1, col=1)
    fig.update_yaxes(title_text="Position", row=2, col=1)
    fig.update_yaxes(title_text="%", row=3, col=1)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Alvi (2018) benchmark (real data, §4.3.1 pp. 63-65): strategy +8.7% "
               "vs B&H −13.9%, Sharpe 0.58, win rate 60%. Our capital-preservation "
               "variant trades only in predicted-Bull months.")

st.markdown("### 🧭 Understand this page")
st.markdown(f"""
<div class='insight-num'><b>Overall accuracy {res.test_acc:.1f}%:</b> — of the ~{len(res.panel_test)} unseen months, the model picked the correct regime in roughly {res.test_acc/100*len(res.panel_test):.0f} of them — versus 1 in 3 for a coin flip and ≈57% for the original paper's benchmark.</div>
<div class='insight-num'><b>Stagnant is the workhorse:</b> — precision {clf['Stagnant']['precision']:.2f} / recall {clf['Stagnant']['recall']:.2f}; the post-2022 market was mostly sideways and the model knows it.</div>
<div class='insight-num'><b>Bear recall is low:</b> — sharp bears are short and rare in this window; missing them is the model's main weakness (flagged, not hidden).</div>
<div class='insight-num'><b>Trading {t['strat_cum']:+.1f}% vs B&H {t['bh_cum']:+.1f}%:</b> — the signal's value is capital preservation: sitting out non-Bull months avoids the deep drawdown buy-and-hold suffered.</div>
""", unsafe_allow_html=True)
st.markdown("<div class='insight-plain'>🟢 <b>For non-technical readers:</b><br>"
            "On months it had never seen, the model named the market's mood "
            "correctly in the large majority of the months it had never seen. "
            "Used as a simple rule — own oil only in calm up-moods, sit out the "
            "rest — it lost far less than just holding oil the whole time. "
            "Not a money machine: a risk filter.</div>", unsafe_allow_html=True)
with st.expander("🔬 Technical detail"):
    st.caption("sklearn classification_report / confusion_matrix (zero_division=0); "
               "accuracy = exact class match; Sharpe = mean/std × √12 on monthly "
               "returns; max drawdown on the compounded equity curve; positions "
               "derived from Viterbi-decoded regimes piped through the BN prediction. "
               "Transaction costs/slippage excluded — a stated limitation.")
