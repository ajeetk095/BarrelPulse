"""Plotly charts for GWP3. Every chart states units, sample period and source in its title/axes."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import config as C
from . import gwp3_config as G
from .charts import _layout, color_map, fig_dag  # noqa: F401  (re-exported)

T = C.THEME
CM = color_map(list(G.REGIMES))
SRC = "Source: FRED monthly snapshot (WTISPLC); computed in-app"


def fig_split_allocation(desc: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    cols = {"Training": T["info"], "Validation": T["accent"], "Testing": T["bull"]}
    for nm, r in desc.iterrows():
        fig.add_trace(go.Bar(y=["Sample"], x=[r["N"]], orientation="h", name=f"{nm} ({r['Start']} → {r['End']})",
                             marker_color=cols[nm], text=[f"{nm}: {int(r['N'])}"], textposition="inside"))
    fig.update_layout(barmode="stack")
    return _layout(fig, "Chronological split — number of monthly observations", "Months (n)", None, 230)


def fig_regime_share(desc: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for r in G.REGIMES:
        fig.add_trace(go.Bar(x=desc.index, y=desc[f"{r} %"], name=r, marker_color=CM[r]))
    fig.update_layout(barmode="stack")
    return _layout(fig, "HMM regime share by split (% of months)", "Split", "Share (%)", 340)


def fig_actual_vs_pred(y: pd.Series, pred: pd.Series, title: str) -> go.Figure:
    fig = go.Figure()
    lv = {r: i for i, r in enumerate(G.REGIMES)}
    fig.add_trace(go.Scatter(x=y.index, y=y.map(lv), mode="lines+markers", name="Actual (HMM label)",
                             line=dict(shape="hv", color=T["text"]), marker=dict(size=7)))
    fig.add_trace(go.Scatter(x=pred.index, y=pred.map(lv) + 0.06, mode="lines+markers", name="Predicted (BN)",
                             line=dict(shape="hv", color=T["accent"], dash="dot"), marker=dict(symbol="diamond", size=7)))
    fig.update_yaxes(tickvals=list(lv.values()), ticktext=list(lv.keys()), range=[-0.4, 2.4])
    return _layout(fig, title, "Month", "Regime", 340)


def fig_prob_stack(P: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    for r in G.REGIMES:
        if r in P.columns:
            fig.add_trace(go.Bar(x=P.index, y=P[r], name=r, marker_color=CM[r]))
    fig.update_layout(barmode="stack")
    return _layout(fig, title, "Month", "Predicted probability", 320)


def fig_confusion(cm: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=cm.values, x=list(cm.columns), y=list(cm.index), colorscale="YlOrBr",
                               text=cm.values, texttemplate="%{text}", showscale=False))
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, title, "Predicted", "Actual", 340, legend=False)


def fig_accuracy_vs_baselines(m: dict, title: str) -> go.Figure:
    names = ["Bayesian network", "Majority class", "Persistence (prev. month)", "Random (expected)"]
    vals = [m["accuracy"], m["majority_acc"], m["persistence_acc"], m["random_acc_expected"]]
    err = [[m["accuracy"] - m["ci_low"], 0, 0, 0], [m["ci_high"] - m["accuracy"], 0, 0, 0]]
    fig = go.Figure(go.Bar(x=names, y=[100 * v for v in vals],
                           marker_color=[T["accent"], T["info"], T["stagnant"], T["muted"]],
                           error_y=dict(type="data", symmetric=False, array=[100 * e for e in err[1]],
                                        arrayminus=[100 * e for e in err[0]]),
                           text=[f"{100*v:.1f}%" for v in vals], textposition="outside"))
    fig.update_yaxes(range=[0, 110])
    return _layout(fig, title + " (whisker: 95% Wilson CI of the model)", "Predictor", "Accuracy (%)", 360, legend=False)


def fig_accuracy_compare(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    short = [v.split(" · ")[0] for v in df.index]
    fig.add_trace(go.Bar(x=short, y=100 * df["Val acc"], name="Validation accuracy", marker_color=T["info"]))
    fig.add_trace(go.Bar(x=short, y=100 * df["Test acc"], name="Test accuracy", marker_color=T["accent"]))
    fig.add_trace(go.Scatter(x=short, y=100 * df["Test majority"], mode="markers", name="Test majority baseline",
                             marker=dict(symbol="line-ew", size=26, line=dict(width=3, color=T["bear"]))))
    fig.update_layout(barmode="group")
    return _layout(fig, "Model variants — accuracy vs majority baseline", "Variant", "Accuracy (%)", 380)


def fig_acf(acf: list, ci: float) -> go.Figure:
    fig = go.Figure(go.Bar(x=list(range(len(acf))), y=acf, marker_color=T["info"]))
    fig.add_hline(y=ci, line_dash="dot", line_color=T["muted"]); fig.add_hline(y=-ci, line_dash="dot", line_color=T["muted"])
    return _layout(fig, "ACF of WTI monthly log-returns (dotted = ±1.96/√n)", "Lag (months)", "Autocorrelation", 300, legend=False)


def fig_trading(series: pd.DataFrame) -> go.Figure:
    s = (1 + series["strategy"]).cumprod(); b = (1 + series["ret"]).cumprod()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s, name="Illustrative regime rule", line=dict(color=T["accent"])))
    fig.add_trace(go.Scatter(x=b.index, y=b, name="Buy & hold WTI", line=dict(color=T["muted"])))
    return _layout(fig, "Illustrative rule vs buy & hold — growth of 1 (simple monthly returns, no costs)", "Month", "Index (start = 1)", 340)


def fig_filtered_probs(P: pd.DataFrame, split: pd.Series) -> go.Figure:
    fig = go.Figure()
    for r in P.columns:
        fig.add_trace(go.Scatter(x=P.index, y=P[r], stackgroup="one", name=r, line=dict(width=0.5, color=CM.get(r, T["info"]))))
    for part, col in (("val", T["accent"]), ("test", T["bull"])):
        idx = split[split == part].index
        fig.add_vrect(x0=idx[0], x1=idx[-1], fillcolor=col, opacity=0.10, line_width=0,
                      annotation_text=part, annotation_position="top left")
    return _layout(fig, "Causal (forward-filtered) HMM state probabilities, 2006–2024", "Month", "P(state | data to date)", 340)


# ----------------------------------------------------------------- 3D -------
def _scene(fig: go.Figure, title: str, x: str, y: str, z: str, height: int = 520) -> go.Figure:
    ax = dict(backgroundcolor="rgba(0,0,0,0)", gridcolor=T["border"], showbackground=True, zerolinecolor=T["border"])
    fig.update_layout(title=dict(text=title, x=0, font=dict(size=16, color=T["text"])), height=height,
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(color=T["text"]), margin=dict(l=0, r=0, t=56, b=0),
                      scene=dict(xaxis=dict(title=x, **ax), yaxis=dict(title=y, **ax), zaxis=dict(title=z, **ax),
                                 bgcolor="rgba(0,0,0,0)", camera=dict(eye=dict(x=1.6, y=1.6, z=0.9))),
                      legend=dict(orientation="h", y=1.02, x=1, xanchor="right"))
    return fig


def fig3d_feature_space(raw: pd.DataFrame, labels: pd.Series) -> go.Figure:
    fig = go.Figure()
    for r in G.REGIMES:
        m = labels.reindex(raw.index) == r
        if m.any():
            d = raw[m]
            fig.add_trace(go.Scatter3d(x=100 * d["ret"], y=100 * d["vol"], z=d["VIX"], mode="markers", name=r,
                                       marker=dict(size=5, color=CM[r], opacity=0.85, line=dict(width=0)),
                                       text=[f"{i:%b %Y}" for i in d.index],
                                       hovertemplate="%{text}<br>return %{x:.1f}%<br>3m vol %{y:.1f}%<br>VIX %{z:.1f}<extra>" + r + "</extra>"))
    return _scene(fig, "3D regime space — return × volatility × VIX (colour = HMM regime)",
                  "WTI monthly log-return (%)", "3-month rolling vol (%)", "VIX (index pts)")


def fig3d_surface(grid: pd.DataFrame, title: str, xt: str, yt: str, zt: str, cs: str = "Turbo", zrange=(0, 1)) -> go.Figure:
    fig = go.Figure(go.Surface(z=grid.values, x=list(grid.columns), y=list(grid.index), colorscale=cs,
                               cmin=zrange[0], cmax=zrange[1], contours=dict(z=dict(show=True, usecolormap=True, project_z=True)),
                               colorbar=dict(title=zt, thickness=14)))
    return _scene(fig, title, xt, yt, zt)


def fig3d_mc(prob: np.ndarray, names, title: str) -> go.Figure:
    h = np.arange(1, prob.shape[0] + 1)
    fig = go.Figure(go.Surface(z=prob.T, x=h, y=list(names), colorscale="Plasma", cmin=0, cmax=1,
                               contours=dict(z=dict(show=True, usecolormap=True, project_z=True)),
                               colorbar=dict(title="Probability", thickness=14)))
    return _scene(fig, title, "Months ahead", "Regime", "P(regime)")


def fig3d_paths(paths: np.ndarray, names, n_show: int = 40) -> go.Figure:
    fig = go.Figure()
    h = np.arange(1, paths.shape[1] + 1)
    rng = np.random.default_rng(0)
    for i in rng.choice(len(paths), size=min(n_show, len(paths)), replace=False):
        fig.add_trace(go.Scatter3d(x=h, y=[i] * len(h), z=paths[i] + rng.normal(0, .03, len(h)), mode="lines",
                                   line=dict(width=3, color=h, colorscale="Viridis"), showlegend=False,
                                   hovertemplate="month %{x}<extra></extra>"))
    fig.update_layout(scene=dict(zaxis=dict(tickvals=list(range(len(names))), ticktext=list(names))))
    return _scene(fig, "Simulated regime paths (each line = one Monte-Carlo path)", "Months ahead", "Path #", "Regime")


def fig_fan(prob: np.ndarray, names, title: str) -> go.Figure:
    h = list(range(1, prob.shape[0] + 1)); fig = go.Figure()
    for k, n in enumerate(names):
        fig.add_trace(go.Scatter(x=h, y=prob[:, k], stackgroup="one", name=n, line=dict(width=0.6, color=CM.get(n, T["info"]))))
    return _layout(fig, title, "Months ahead", "Probability", 340)


def fig_next_probs(cur: pd.Series, hmm_next: pd.Series, bn_next) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(G.REGIMES), y=cur.reindex(G.REGIMES).values, name="Now (HMM filtered)", marker_color=T["info"]))
    fig.add_trace(go.Bar(x=list(G.REGIMES), y=hmm_next.reindex(G.REGIMES).values, name="Next month (HMM transition)", marker_color=T["accent"]))
    if bn_next is not None:
        fig.add_trace(go.Bar(x=list(G.REGIMES), y=bn_next.reindex(G.REGIMES).values, name="Next month (Bayesian network)", marker_color="#F472B6"))
    fig.update_layout(barmode="group"); fig.update_yaxes(range=[0, 1])
    return _layout(fig, "Regime probabilities — current month and next month", "Regime", "Probability", 360)


def fig_daily(s: pd.Series, title: str, ytitle: str, color: str) -> go.Figure:
    fig = go.Figure(go.Scatter(x=s.index, y=s.values, mode="lines", line=dict(color=color, width=2), fill="tozeroy",
                               fillcolor="rgba(255,255,255,0.04)"))
    return _layout(fig, title, "Date", ytitle, 300, legend=False)
