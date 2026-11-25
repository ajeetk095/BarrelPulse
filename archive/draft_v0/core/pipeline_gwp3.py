"""GWP3 data pipeline — CrudeCortex.
25-year monthly study (2005-2024) extending GWP2: WTI log-return + 3-month
rolling volatility as HMM emissions; macro log-returns discretised for the
Bayesian network; strict chronological 80/10/10 split with anti-leakage
(scaler, HMM, tertile thresholds and BN CPDs all fitted on the training slice only).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as sstats
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

from core.data_gwp2 import FRED_DIR, END, START, _load_fred_csv, load_vix_monthly, to_monthly

REGIME_ORDER = ["Bear", "Stagnant", "Bull"]


def build_panel() -> pd.DataFrame:
    wti = to_monthly(_load_fred_csv("WTISPLC")).dropna()
    wti = wti[(wti.index >= START) & (wti.index <= END)]
    brent = to_monthly(_load_fred_csv("POILBREUSDM"))
    brent = brent[(brent.index >= START) & (brent.index <= END)]
    indpro = to_monthly(_load_fred_csv("INDPRO"))
    cpi = to_monthly(_load_fred_csv("CPIENGSL"))
    dxy = to_monthly(_load_fred_csv("DTWEXBGS"))
    vix = load_vix_monthly()
    df = pd.DataFrame({"WTI": wti, "Brent": brent, "IndPro": indpro,
                       "CPI_Nrg": cpi, "DXY": dxy, "VIX": vix}).dropna()
    df.index.name = "date"
    return df


def build_features(df: pd.DataFrame):
    """GWP3 emission spec: WTI log-return + 3-month rolling std (annualised)."""
    out = df.copy()
    out["WTI_ret"] = np.log(out["WTI"] / out["WTI"].shift(1))
    out["WTI_vol3"] = out["WTI_ret"].rolling(3).std() * np.sqrt(12)
    feats = out[["WTI_ret", "WTI_vol3"]].dropna()
    return out, feats


def chronological_split(index: pd.DatetimeIndex):
    n = len(index)
    tr_end = int(n * 0.8)
    val_end = tr_end + int(n * 0.1)
    return index[:tr_end], index[tr_end:val_end], index[val_end:]


def train_tertile_classes(series: pd.Series, train_idx: pd.DatetimeIndex) -> pd.Series:
    """Quantile classes Low/Medium/High with thresholds from the TRAIN slice only."""
    tr = series.loc[series.index.intersection(train_idx)]
    q1, q2 = tr.quantile(1 / 3), tr.quantile(2 / 3)
    cls = np.select([series <= q1, series <= q2], ["Low", "Medium"], default="High")
    return pd.Series(cls, index=series.index)


@dataclass
class GWP3Results:
    df: pd.DataFrame
    feature_df: pd.DataFrame
    splits: dict
    hmm_ll: float
    transmat: np.ndarray
    labels: pd.Series
    state_stats: pd.DataFrame
    panel_train: pd.DataFrame
    panel_val: pd.DataFrame
    panel_test: pd.DataFrame
    edges_bdeu: set
    edges_bic: set
    hc_history: list
    bn: object
    pred_val: pd.Series
    pred_test: pd.Series
    val_acc: float
    test_acc: float
    clf_report: dict
    cm_val: pd.DataFrame
    cm_test: pd.DataFrame
    ks: dict
    mw: dict
    desc: pd.DataFrame
    trading: dict
    lag: dict
    acf: dict


def _predict_regime(bn, rows: pd.DataFrame, target: str = "WTI_Regime") -> pd.Series:
    parents = bn.parents_of(target)
    marginal = bn.data[target].value_counts(normalize=True).to_dict()
    preds = []
    for _, r in rows.iterrows():
        mask = pd.Series(True, index=bn.data.index)
        for p in parents:
            mask &= bn.data[p] == r[p]
        sub = bn.data.loc[mask, target]
        if len(sub):
            preds.append(sub.value_counts(normalize=True).idxmax())
        else:
            preds.append(max(marginal, key=marginal.get))
    return pd.Series(preds, index=rows.index)


def _max_drawdown(returns: pd.Series) -> float:
    eq = (1 + returns.fillna(0)).cumprod()
    return float(((eq - eq.cummax()) / eq.cummax()).min())


def _trading(preds: pd.Series, df: pd.DataFrame) -> dict:
    """Long WTI in Bull months, flat otherwise (capital-preservation rule)."""
    ret = df["WTI"].pct_change().reindex(preds.index)
    pos = (preds == "Bull").astype(int)
    strat = pos * ret

    def _sharpe(x):
        return float(x.mean() / x.std() * np.sqrt(12)) if x.std() > 1e-12 else 0.0
    active = strat[pos == 1]
    out = {
        "strat_cum": float((1 + strat.fillna(0)).prod() - 1) * 100,
        "bh_cum": float((1 + ret.fillna(0)).prod() - 1) * 100,
        "strat_sharpe": _sharpe(strat.dropna()),
        "bh_sharpe": _sharpe(ret.dropna()),
        "strat_mdd": _max_drawdown(strat) * 100,
        "bh_mdd": _max_drawdown(ret) * 100,
        "win_rate": float((active > 0).mean() * 100) if len(active) else 0.0,
        "active_months": int((pos == 1).sum()),
        "n": int(len(preds)),
    }
    out["outperformance_pp"] = out["strat_cum"] - out["bh_cum"]
    return out


def _bdeu(data):
    from core.bn_bdeu import bdeu_score
    return lambda node, parents: bdeu_score(data, node, parents, ess=1.0)


def _lag_augmented(df, feature_df, labels, splits):
    """Supplementary improvement: 1-period lagged VIX/DXY classes as extra nodes."""
    panel = pd.DataFrame({"WTI_Regime": labels})
    panel["VIX_Level"] = train_tertile_classes(df["VIX"].pct_change(), splits["train"])
    panel["DXY_Level"] = train_tertile_classes(df["DXY"].pct_change(), splits["train"])
    panel["VIX_Level_lag1"] = panel["VIX_Level"].shift(1)
    panel["DXY_Level_lag1"] = panel["DXY_Level"].shift(1)
    panel = panel.dropna()
    tr_end = int(len(panel) * 0.8)
    val_end = tr_end + int(len(panel) * 0.1)
    ptr = panel.iloc[:tr_end]
    pva = panel.iloc[tr_end:val_end]
    pte = panel.iloc[val_end:]
    from core import bn_engine
    from core.bn_bdeu import hill_climb_search_scored
    nodes = list(panel.columns)
    edges, hist = hill_climb_search_scored(ptr, nodes, _bdeu(ptr), max_parents=2)
    bn = bn_engine.BNModel(ptr, nodes, edges, hist)
    bn = bn_engine.fit_cpds(bn)
    return {"edges": sorted(edges),
            "val_acc": float((_predict_regime(bn, pva) == pva["WTI_Regime"]).mean() * 100),
            "test_acc": float((_predict_regime(bn, pte) == pte["WTI_Regime"]).mean() * 100),
            "n": len(panel)}


def build_all() -> GWP3Results:
    import hmmlearn.hmm as hh
    from core import bn_engine
    from core.bn_bdeu import bdeu_score, hill_climb_search_scored
    from core.bn_engine import bic_score

    df = build_panel()
    df, feature_df = build_features(df)
    tr_idx, va_idx, te_idx = chronological_split(feature_df.index)
    splits = {"train": tr_idx, "val": va_idx, "test": te_idx}

    # ---- HMM fitted on TRAIN only (anti-leakage) ----
    X = feature_df[["WTI_ret", "WTI_vol3"]].values
    scaler = StandardScaler().fit(X[feature_df.index.isin(tr_idx)])
    X_std = scaler.transform(X)
    X_tr = X_std[feature_df.index.isin(tr_idx)]
    best, best_ll = None, -np.inf
    for seed in range(10):
        m = hh.GaussianHMM(n_components=3, covariance_type="full",
                           n_iter=200, tol=1e-4, random_state=seed)
        try:
            m.fit(X_tr)
        except Exception:
            continue
        if m.monitor_.converged and m.score(X_tr) > best_ll:
            best_ll, best = m.score(X_tr), m
    order = np.argsort(best.means_[:, 0])
    name_map = {order[0]: "Bear", order[1]: "Stagnant", order[2]: "Bull"}
    labels = pd.Series([name_map[h] for h in best.predict(X_std)], index=feature_df.index)

    tmp = feature_df.copy()
    tmp["Regime"] = labels
    stats = tmp.groupby("Regime").agg(
        Months=("WTI_ret", "size"),
        MeanReturn=("WTI_ret", lambda x: float(np.mean(x)) * 100),
        MeanVol=("WTI_vol3", lambda x: float(np.mean(x)) * 100),
    ).reindex(REGIME_ORDER)
    stats["PctSample"] = stats["Months"] / stats["Months"].sum() * 100

    # ---- discrete BN panel (train tertiles only) ----
    panel = pd.DataFrame({"WTI_Regime": labels})
    panel["VIX_Level"] = train_tertile_classes(df["VIX"].pct_change(), tr_idx)
    panel["DXY_Level"] = train_tertile_classes(df["DXY"].pct_change(), tr_idx)
    panel["IndPro_Level"] = train_tertile_classes(df["IndPro"].pct_change(), tr_idx)
    panel["CPI_Level"] = train_tertile_classes(df["CPI_Nrg"].pct_change(), tr_idx)
    panel["Brent_Level"] = train_tertile_classes(df["Brent"].pct_change(), tr_idx)
    panel = panel.dropna()
    ptr = panel.loc[panel.index.intersection(tr_idx)]
    pva = panel.loc[panel.index.intersection(va_idx)]
    pte = panel.loc[panel.index.intersection(te_idx)]
    nodes = list(panel.columns)

    edges_bdeu, hist_bdeu = hill_climb_search_scored(
        ptr, nodes, lambda n, p: bdeu_score(ptr, n, p, ess=1.0), max_parents=2)
    edges_bic, _ = hill_climb_search_scored(ptr, nodes, lambda n, p: bic_score(ptr, n, p), max_parents=2)

    bn = bn_engine.BNModel(ptr, nodes, edges_bdeu, hist_bdeu)
    bn = bn_engine.fit_cpds(bn)

    pred_val = _predict_regime(bn, pva)
    pred_test = _predict_regime(bn, pte)
    val_acc = float((pred_val == pva["WTI_Regime"]).mean() * 100)
    test_acc = float((pred_test == pte["WTI_Regime"]).mean() * 100)
    clf = classification_report(pte["WTI_Regime"], pred_test, zero_division=0, output_dict=True)

    present = [r for r in REGIME_ORDER if r in set(pte["WTI_Regime"]) | set(pred_test)]
    cm_test = pd.DataFrame(confusion_matrix(pte["WTI_Regime"], pred_test, labels=present),
                           index=[f"True {l}" for l in present],
                           columns=[f"Pred {l}" for l in present])
    pres_v = [r for r in REGIME_ORDER if r in set(pva["WTI_Regime"]) | set(pred_val)]
    cm_val = pd.DataFrame(confusion_matrix(pva["WTI_Regime"], pred_val, labels=pres_v),
                          index=[f"True {l}" for l in pres_v],
                          columns=[f"Pred {l}" for l in pres_v])

    # ---- Step 2 split diagnostics ----
    def _desc(idx, name):
        w = df.loc[df.index.intersection(idx), "WTI"]
        r = feature_df.loc[feature_df.index.intersection(idx), "WTI_ret"]
        v = feature_df.loc[feature_df.index.intersection(idx), "WTI_vol3"]
        return {"Split": name, "Months": int(len(w)), "MeanWTI": round(float(w.mean()), 2),
                "StdWTI": round(float(w.std()), 2), "MeanRet%": round(float(r.mean() * 100), 3),
                "Skew": round(float(r.skew()), 3), "Kurtosis": round(float(r.kurt()), 3),
                "MeanVol%": round(float(v.mean() * 100), 1)}
    desc = pd.DataFrame([_desc(tr_idx, "Train"), _desc(va_idx, "Validation"),
                         _desc(te_idx, "Test")]).set_index("Split")
    rv = feature_df.loc[feature_df.index.intersection(va_idx), "WTI_ret"]
    rt = feature_df.loc[feature_df.index.intersection(te_idx), "WTI_ret"]
    ks_s, ks_p = sstats.ks_2samp(rv, rt)
    mw_s, mw_p = sstats.mannwhitneyu(rv, rt, alternative="two-sided")
    ks = {"stat": float(ks_s), "p": float(ks_p)}
    mw = {"stat": float(mw_s), "p": float(mw_p)}

    def _acf(x, k):
        xm = x - x.mean()
        return float(np.sum(xm[k:] * xm[:-k]) / np.sum(xm * xm))
    allr = feature_df["WTI_ret"].dropna()
    acf = {"lags": list(range(1, 13)),
           "ret": [_acf(allr.values, k) for k in range(1, 13)],
           "sq": [_acf((allr ** 2).values, k) for k in range(1, 13)]}

    trading = _trading(pred_test, df)
    lag = _lag_augmented(df, feature_df, labels, splits)

    return GWP3Results(df, feature_df, splits, float(best_ll), best.transmat_.copy(),
                       labels, stats, ptr, pva, pte, edges_bdeu, edges_bic, hist_bdeu,
                       bn, pred_val, pred_test, val_acc, test_acc, clf, cm_val, cm_test,
                       ks, mw, desc, trading, lag, acf)
