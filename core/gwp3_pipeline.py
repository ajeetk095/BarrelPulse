"""GWP3 computed pipeline (offline, on the bundled FRED snapshot).

Design rules (enforced by tests):
* chronological 80/10/10 split; scaler, HMM and discretisation thresholds are
  fitted on the TRAINING block only;
* model/score selection uses VALIDATION accuracy (tie-break: log-loss) — the test
  block is evaluated once, for the selected configuration only;
* nothing here is hard-coded: every number is computed from the data.
"""
from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logsumexp

from . import bn_engine as B
from . import gwp3_config as G
from .config import DataConfig
from .data_gwp2 import get_dataset
from .hmm_engine import forward_log, gaussian_log_emissions

T = G.TARGET


# ------------------------------------------------------------------ data ---
@lru_cache(maxsize=8)
def _dataset(mode: str, nonce: int):
    """Monthly master panel. ``live`` tries providers first and falls back to cache/snapshot."""
    from .secrets import fred_api_fetcher
    kw = {}
    f = fred_api_fetcher()
    if f is not None:
        kw["fetcher"] = f
    return get_dataset(DataConfig(data_mode=mode, feature_start="2007-01-01", feature_end="2024-12-01",
                                  refresh_nonce=nonce), **kw)


def dataset_info(cfg: G.GWP3Config) -> dict:
    ds = _dataset(cfg.data_mode, cfg.refresh_nonce)
    fm = ds.full_master
    return {"first": fm.index[0], "last": fm.index[-1], "n": len(fm), "status": ds.status,
            "warnings": list(ds.warnings), "fingerprint": ds.fingerprint,
            "origins": {k: v.origin for k, v in ds.loads.items()}}


def load_panel(cfg: G.GWP3Config) -> pd.DataFrame:
    ds = _dataset(cfg.data_mode, cfg.refresh_nonce)
    end = cfg.end or None
    m = ds.full_master.loc[cfg.start:end].copy()
    m["ret"] = np.log(m["WTI"]).diff()
    m["vol"] = m["ret"].rolling(cfg.vol_window).std()
    m["simple_ret"] = m["WTI"].pct_change()
    m["VIX_x"] = m["VIX"]
    m["DXY_x"] = np.log(m["DXY"]).diff()
    m["IndPro_x"] = np.log(m["IndPro"]).diff()
    m["CPI_x"] = m["CPI_Nrg"].pct_change()
    return m.dropna()


def split_bounds(n: int, cfg: G.GWP3Config) -> tuple[int, int]:
    a = int(round(n * cfg.train_frac))
    b = int(round(n * (cfg.train_frac + cfg.val_frac)))
    return a, b


def split_frame(idx: pd.Index, cfg: G.GWP3Config) -> pd.Series:
    a, b = split_bounds(len(idx), cfg)
    s = np.array(["train"] * a + ["val"] * (b - a) + ["test"] * (len(idx) - b))
    return pd.Series(s, index=idx, name="split")


# ------------------------------------------------------------------- HMM ---
@dataclass
class HMMFit:
    model: object
    order: np.ndarray            # raw index of Bear, Stagnant, Bull (ascending mean return)
    mean: np.ndarray
    std: np.ndarray
    bic: float
    seed: int
    table: pd.DataFrame          # per-seed log-lik / BIC
    log_A: np.ndarray = None
    log_pi: np.ndarray = None
    names: list = field(default_factory=list)


def _bic(ll: float, n_states: int, d: int, n: int) -> float:
    k = (n_states - 1) + n_states * (n_states - 1) + n_states * d + n_states * d * (d + 1) // 2
    return -2 * ll + k * np.log(n)


def fit_hmm(Xtr: np.ndarray, cfg: G.GWP3Config) -> HMMFit:
    from hmmlearn.hmm import GaussianHMM
    mean, std = Xtr.mean(0), Xtr.std(0, ddof=0)
    Z = (Xtr - mean) / std
    rows, best = [], None
    for seed in range(cfg.n_seeds):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = GaussianHMM(cfg.n_states, covariance_type="full", n_iter=cfg.hmm_n_iter,
                                tol=cfg.hmm_tol, random_state=seed).fit(Z)
            ll = float(m.score(Z))
            bic = _bic(ll, cfg.n_states, Z.shape[1], len(Z))
            rows.append({"seed": seed, "loglik": ll, "bic": bic, "converged": bool(m.monitor_.converged)})
            if best is None or bic < best[0]:
                best = (bic, seed, m)
        except Exception as e:  # pragma: no cover - degenerate fit
            rows.append({"seed": seed, "loglik": np.nan, "bic": np.nan, "converged": False})
    if best is None:
        raise RuntimeError("All HMM restarts failed.")
    bic, seed, m = best
    order = np.argsort(m.means_[:, 0])               # ascending mean return
    names = list(G.REGIMES) if cfg.n_states == 3 else (["Bear", "Bull"] if cfg.n_states == 2
                                                          else [f"S{i}" for i in range(cfg.n_states)])
    A = m.transmat_[np.ix_(order, order)]
    pi = m.startprob_[order]
    with np.errstate(divide="ignore"):
        fit = HMMFit(m, order, mean, std, bic, seed, pd.DataFrame(rows),
                     np.log(A), np.log(np.maximum(pi, 1e-300)), names)
    return fit


def _log_emis(fit: HMMFit, X: np.ndarray) -> np.ndarray:
    Z = (X - fit.mean) / fit.std
    means = fit.model.means_[fit.order]
    covs = fit.model._covars_[fit.order] if hasattr(fit.model, "_covars_") else fit.model.covars_[fit.order]
    return gaussian_log_emissions(Z, means, covs)


def decode_labels(fit: HMMFit, X: np.ndarray, how: str = "viterbi") -> np.ndarray:
    """``viterbi``: smoothed full-path decode of the supplied block.
    ``filtered``: causal argmax P(state_t | x_1..x_t) over the supplied block."""
    lb = _log_emis(fit, X)
    if how == "filtered":
        la, _ = forward_log(fit.log_pi, fit.log_A, lb)
        return la.argmax(1)
    from .hmm_engine import viterbi_log
    return viterbi_log(fit.log_pi, fit.log_A, lb)[0]


def filtered_probs(fit: HMMFit, X: np.ndarray) -> np.ndarray:
    la, _ = forward_log(fit.log_pi, fit.log_A, _log_emis(fit, X))
    return np.exp(la - logsumexp(la, axis=1, keepdims=True))


# --------------------------------------------------------------- metrics ---
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def prf(y, p, labels=G.REGIMES) -> pd.DataFrame:
    rows = []
    for c in labels:
        tp = int(((y == c) & (p == c)).sum()); fp = int(((y != c) & (p == c)).sum())
        fn = int(((y == c) & (p != c)).sum())
        pr = tp / (tp + fp) if tp + fp else np.nan
        rc = tp / (tp + fn) if tp + fn else np.nan
        f1 = 2 * pr * rc / (pr + rc) if pr and rc and not np.isnan(pr + rc) else (0.0 if tp + fp + fn else np.nan)
        rows.append({"class": c, "precision": pr, "recall": rc, "f1": f1, "support": int((y == c).sum())})
    return pd.DataFrame(rows).set_index("class")


def confusion(y, p, labels=G.REGIMES) -> pd.DataFrame:
    return pd.DataFrame([[int(((y == a) & (p == b)).sum()) for b in labels] for a in labels],
                        index=[f"actual {l}" for l in labels], columns=[f"pred {l}" for l in labels])


def log_loss(y, P: pd.DataFrame, eps: float = 1e-6) -> float:
    pr = np.array([P.loc[i, c] if c in P.columns else eps for i, c in zip(P.index, y)])
    return float(-np.mean(np.log(np.clip(pr, eps, 1))))


def metrics(y: pd.Series, pred: pd.Series, P: Optional[pd.DataFrame] = None, base_acc: float = np.nan) -> dict:
    n = len(y); k = int((y.values == pred.values).sum())
    t = prf(y.values, pred.values)
    present = t[t.support > 0]
    out = {"n": n, "correct": k, "accuracy": k / n, "ci_low": wilson(k, n)[0], "ci_high": wilson(k, n)[1],
           "macro_f1": float(present.f1.mean()), "balanced_acc": float(present.recall.mean()),
           "log_loss": log_loss(y, P) if P is not None else np.nan}
    if not np.isnan(base_acc):
        out["binom_p_vs_baseline"] = float(stats.binomtest(k, n, base_acc, alternative="greater").pvalue)
    return out


def mcnemar_exact(y, a, b) -> float:
    """Exact two-sided McNemar on paired correctness of predictors a and b."""
    ca, cb = (np.asarray(a) == np.asarray(y)), (np.asarray(b) == np.asarray(y))
    n01, n10 = int((~ca & cb).sum()), int((ca & ~cb).sum())
    if n01 + n10 == 0:
        return 1.0
    return float(stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue)


def alvi_style_error(actual, pred) -> float:
    """Error rate computed the way Alvi's code does it (np.roll(pred, 1)) — shown only for audit."""
    return float((np.asarray(actual) != np.roll(np.asarray(pred), 1)).mean())


# ----------------------------------------------------------------- panel ---
def tertiles(s: pd.Series) -> tuple[float, float]:
    return float(s.quantile(1 / 3)), float(s.quantile(2 / 3))


def discretise(s: pd.Series, lo_hi: tuple[float, float], labels) -> pd.Series:
    lo, hi = lo_hi
    return pd.Series(np.where(s <= lo, labels[0], np.where(s <= hi, labels[1], labels[2])),
                     index=s.index)


@dataclass
class Prepared:
    cfg: G.GWP3Config
    raw: pd.DataFrame
    split: pd.Series
    fit: HMMFit
    regimes: pd.Series           # labels used by the BN (per cfg)
    regimes_viterbi: pd.Series
    regimes_filtered: pd.Series
    ev_all: pd.DataFrame         # un-lagged discretised evidence for every month (incl. latest)
    panel: pd.DataFrame          # discrete dataset (aligned to evidence lag)
    thresholds: dict
    probs_filtered: pd.DataFrame


def prepare(cfg: G.GWP3Config) -> Prepared:
    raw = load_panel(cfg)
    sp = split_frame(raw.index, cfg)
    X = raw[["ret", "vol"]].values
    tr = (sp == "train").values
    fit = fit_hmm(X[tr], cfg)
    names = fit.names
    # Viterbi decoded per split block (notebook / Alvi style) vs. causal filtered over the full path
    vit = np.empty(len(raw), dtype=int)
    for part in ("train", "val", "test"):
        msk = (sp == part).values
        vit[msk] = decode_labels(fit, X[msk], "viterbi")
    fil = decode_labels(fit, X, "filtered")
    reg_v = pd.Series([names[i] for i in vit], index=raw.index, name=T)
    reg_f = pd.Series([names[i] for i in fil], index=raw.index, name=T)
    pf = pd.DataFrame(filtered_probs(fit, X), index=raw.index, columns=names)
    reg = reg_f if cfg.use_filtered_labels else reg_v

    th = {"VIX_Level": tertiles(raw.loc[tr, "VIX_x"]), "DXY_Trend": tertiles(raw.loc[tr, "DXY_x"]),
          "IndPro_Trend": tertiles(raw.loc[tr, "IndPro_x"]), "CPI_Trend": tertiles(raw.loc[tr, "CPI_x"])}
    ev = pd.DataFrame({
        "VIX_Level": discretise(raw["VIX_x"], th["VIX_Level"], ("Low", "Medium", "High")),
        "DXY_Trend": discretise(raw["DXY_x"], th["DXY_Trend"], ("Down", "Flat", "Up")),
        "IndPro_Trend": discretise(raw["IndPro_x"], th["IndPro_Trend"], ("Falling", "Flat", "Rising")),
        "CPI_Trend": discretise(raw["CPI_x"], th["CPI_Trend"], ("Falling", "Flat", "Rising")),
    })
    ev_all = ev.copy()
    if cfg.lag:
        ev = ev.shift(cfg.lag)
    panel = pd.concat([reg, ev], axis=1)
    if cfg.add_prev_regime:
        panel["Prev_Regime"] = reg.shift(1)
    panel = panel.dropna()
    return Prepared(cfg, raw, sp, fit, reg, reg_v, reg_f, ev_all, panel, th, pf)


# ------------------------------------------------------------------- BN ----
def _fit_bn(train: pd.DataFrame, score: str, cfg: G.GWP3Config):
    start = None
    if cfg.seed_edges:
        start = {e for e in G.EXPERT_EDGES if e[0] in train.columns}
    nodes = list(train.columns)
    bn = B.build_bn(train[nodes], max_parents=cfg.max_parents, score=score, target=T,
                    start_edges=start, alpha=cfg.laplace_alpha)
    if cfg.force_connected and not bn.parents_of(T):
        cands = [c for c in nodes if c != T]
        best = max(cands, key=lambda c: B.local_score(train, T, (c,), score))
        bn.edges.add((best, T)); bn.cpds = {}
        bn = B.fit_cpds(bn, cfg.laplace_alpha)
    return bn


def predict(bn, block: pd.DataFrame, fallback: pd.Series) -> tuple[pd.Series, pd.DataFrame, int]:
    """MAP prediction with exact inference; unseen evidence falls back to the train marginal."""
    preds, probs, nfb = [], [], 0
    evidence_nodes = [c for c in block.columns if c != T]
    for idx, row in block.iterrows():
        ev = {c: row[c] for c in evidence_nodes}
        try:
            p = B.infer_network(bn, T, ev)
        except B.InferenceError:
            p, nfb = fallback.copy(), nfb + 1
        p = p.reindex(list(G.REGIMES)).fillna(0.0)
        probs.append(p); preds.append(p.idxmax())
    return pd.Series(preds, index=block.index), pd.DataFrame(probs, index=block.index), nfb


@dataclass
class RunResult:
    cfg: G.GWP3Config
    prep: Prepared
    candidates: pd.DataFrame      # validation metrics per scoring function
    selected: str
    bn: object
    pred: dict                    # {"train":..,"val":..,"test":..}
    probs: dict
    metrics: dict                 # {"val":.., "test":..}
    baselines: dict
    tests: dict
    confusion: dict
    report: dict
    edges: list
    target_parents: list
    fallbacks: dict


def _persistence(reg: pd.Series, block_idx: pd.Index) -> pd.Series:
    return reg.shift(1).reindex(block_idx)


def run(cfg: G.GWP3Config, prep: Optional[Prepared] = None) -> RunResult:
    prep = prep or prepare(cfg)
    panel = prep.panel
    sp = prep.split.reindex(panel.index)
    tr, va, te = (panel[sp == s] for s in ("train", "val", "test"))
    marginal = tr[T].value_counts(normalize=True).reindex(list(G.REGIMES)).fillna(0.0)
    majority = marginal.idxmax()

    # model selection on VALIDATION only
    rows, models = [], {}
    for sc in cfg.scores:
        bn = _fit_bn(tr, sc, cfg)
        pv, Pv, nfb = predict(bn, va.drop(columns=[T]), marginal)
        m = metrics(va[T], pv, Pv)
        rows.append({"score": sc, "n_edges": len(bn.edges), "target_parents": len(bn.parents_of(T)),
                     "val_accuracy": m["accuracy"], "val_log_loss": m["log_loss"], "fallbacks": nfb})
        models[sc] = bn
    cand = pd.DataFrame(rows).set_index("score")
    sel = cand.sort_values(["val_accuracy", "val_log_loss"], ascending=[False, True]).index[0]
    bn = models[sel]

    pred, probs, fbk, met = {}, {}, {}, {}
    for name, blk in (("train", tr), ("val", va), ("test", te)):
        pred[name], probs[name], fbk[name] = predict(bn, blk.drop(columns=[T]), marginal)
    base = {}
    rng = np.random.default_rng(0)
    for name, blk in (("val", va), ("test", te)):
        y = blk[T]
        maj = pd.Series(majority, index=blk.index)
        per = _persistence(prep.regimes, blk.index).fillna(majority)
        rnd = pd.Series(rng.choice(list(G.REGIMES), size=len(blk), p=marginal.values), index=blk.index)
        base[name] = {"majority": maj, "persistence": per, "random": rnd}
        ba = float((y == majority).mean())
        met[name] = metrics(y, pred[name], probs[name], base_acc=ba)
        met[name]["majority_acc"] = ba
        met[name]["persistence_acc"] = float((per == y).mean())
        met[name]["random_acc_expected"] = float((marginal ** 2).sum())
        met[name]["mcnemar_p_vs_majority"] = mcnemar_exact(y, pred[name], maj)
    conf = {k: confusion(blk[T], pred[k]) for k, blk in (("val", va), ("test", te))}
    rep = {k: prf(blk[T].values, pred[k].values) for k, blk in (("val", va), ("test", te))}
    tests = split_tests(prep)
    return RunResult(cfg, prep, cand, sel, bn, pred, probs, met, base, tests, conf, rep,
                     sorted(bn.edges), bn.parents_of(T), fbk)


# --------------------------------------------------------- diagnostics ----
def split_tests(prep: Prepared) -> dict:
    r, sp = prep.raw, prep.split
    v, t = r.loc[sp == "val", "ret"], r.loc[sp == "test", "ret"]
    ks = stats.ks_2samp(v, t); mw = stats.mannwhitneyu(v, t, alternative="two-sided")
    lab = prep.regimes_viterbi
    cv, ct = lab[sp == "val"], lab[sp == "test"]
    tab = [[int((cv == c).sum()) for c in G.REGIMES], [int((ct == c).sum()) for c in G.REGIMES]]
    nz = [j for j in range(3) if tab[0][j] + tab[1][j] > 0]
    if len(nz) == 2:
        sub = [[tab[i][j] for j in nz] for i in range(2)]
        fe = float(stats.fisher_exact(sub)[1]); test = "Fisher exact (2 occupied regimes)"
    else:
        fe, test = float("nan"), "not run (fewer than 2 occupied regimes)"
    from statsmodels.tsa.stattools import acf
    a = acf(r["ret"], nlags=12, fft=False)
    ci = 1.96 / np.sqrt(len(r))
    return {"ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue), "mw_stat": float(mw.statistic),
            "mw_p": float(mw.pvalue), "contingency": tab, "regime_test": test, "regime_p": fe,
            "acf": a.tolist(), "acf_ci": float(ci)}


def describe_splits(prep: Prepared) -> pd.DataFrame:
    r, sp, lab = prep.raw, prep.split, prep.regimes_viterbi
    rows = []
    for s, nm in (("train", "Training"), ("val", "Validation"), ("test", "Testing")):
        m = sp == s; x = r.loc[m]
        sh = lab[m].value_counts(normalize=True).reindex(list(G.REGIMES)).fillna(0) * 100
        rows.append({"Split": nm, "N": int(m.sum()), "Share %": 100 * m.mean(),
                     "Start": x.index[0].date(), "End": x.index[-1].date(),
                     "WTI mean $": x.WTI.mean(), "Ret mean %": 100 * x.ret.mean(),
                     "Ret std %": 100 * x.ret.std(), "Skew": stats.skew(x.ret), "Kurt": stats.kurtosis(x.ret),
                     "Ann. vol %": 100 * x.ret.std() * np.sqrt(12),
                     "Bear %": sh["Bear"], "Stagnant %": sh["Stagnant"], "Bull %": sh["Bull"]})
    return pd.DataFrame(rows).set_index("Split")


# ------------------------------------------------------------- variants ---
VARIANTS = {
    "V0 · Same-month nowcast (notebook replica)": G.GWP3Config(),
    "V1 · One-month-lagged evidence (true 1-step-ahead)": G.GWP3Config(lag=1),
    "V2 · V0 + causal filtered labels": G.GWP3Config(use_filtered_labels=True),
    "V3 · V1 + previous-regime node": G.GWP3Config(lag=1, add_prev_regime=True),
    "V4 · V0 + expert-seeded hill climb (Alvi style)": G.GWP3Config(seed_edges=True),
}


@lru_cache(maxsize=32)
def run_cached(cfg: G.GWP3Config) -> RunResult:
    return run(cfg)


def compare_variants() -> pd.DataFrame:
    rows = []
    for name, cfg in VARIANTS.items():
        r = run_cached(cfg)
        v, t = r.metrics["val"], r.metrics["test"]
        rows.append({"Variant": name, "Score (val-selected)": r.selected,
                     "Val acc": v["accuracy"], "Val majority": v["majority_acc"],
                     "Test acc": t["accuracy"], "Test majority": t["majority_acc"],
                     "Test macro-F1": t["macro_f1"], "Test log-loss": t["log_loss"],
                     "Target parents": len(r.target_parents)})
    return pd.DataFrame(rows).set_index("Variant")


# ------------------------------------------------------------ robustness ---
def robustness_splits(base: G.GWP3Config = G.GWP3Config()) -> pd.DataFrame:
    rows = []
    for tf, vf in [(0.6, 0.2), (0.7, 0.15), (0.8, 0.1)]:
        r = run(replace(base, train_frac=tf, val_frac=vf))
        rows.append({"train/val/test": f"{int(tf*100)}/{int(vf*100)}/{int(round((1-tf-vf)*100))}",
                     "val n": r.metrics["val"]["n"], "test n": r.metrics["test"]["n"],
                     "val acc": r.metrics["val"]["accuracy"], "val majority": r.metrics["val"]["majority_acc"],
                     "test acc": r.metrics["test"]["accuracy"], "test majority": r.metrics["test"]["majority_acc"]})
    return pd.DataFrame(rows).set_index("train/val/test")


def robustness_states(base: G.GWP3Config = G.GWP3Config()) -> pd.DataFrame:
    """2-state HMM: Bear/Bull only (no Stagnant). Reported as a sensitivity check, not selected."""
    rows = []
    for k in (2, 3):
        cfg = replace(base, n_states=k)
        try:
            prep = prepare(cfg)
            share = prep.regimes_viterbi[prep.split == "train"].value_counts(normalize=True)
            rows.append({"states": k, "BIC": prep.fit.bic, **{f"train share {s}": float(share.get(s, 0)) for s in prep.fit.names}})
        except Exception as e:  # pragma: no cover
            rows.append({"states": k, "BIC": np.nan})
    return pd.DataFrame(rows).set_index("states")


def robustness_seeds(base: G.GWP3Config = G.GWP3Config()) -> pd.DataFrame:
    prep = prepare(base)
    ref = prep.regimes_viterbi
    X = prep.raw[["ret", "vol"]].values
    tr = (prep.split == "train").values
    rows = []
    for s in range(base.n_seeds):
        f = fit_hmm_single(X[tr], base, s)
        lab = pd.Series([f.names[i] for i in decode_labels(f, X, "viterbi")], index=prep.raw.index)
        rows.append({"seed": s, "BIC": f.bic, "agreement with selected": float((lab == ref).mean())})
    return pd.DataFrame(rows).set_index("seed")


def fit_hmm_single(Xtr, cfg, seed):
    from hmmlearn.hmm import GaussianHMM
    mean, std = Xtr.mean(0), Xtr.std(0)
    Z = (Xtr - mean) / std
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = GaussianHMM(cfg.n_states, covariance_type="full", n_iter=cfg.hmm_n_iter, tol=cfg.hmm_tol,
                        random_state=seed).fit(Z)
    order = np.argsort(m.means_[:, 0])
    A = m.transmat_[np.ix_(order, order)]
    names = list(G.REGIMES)
    with np.errstate(divide="ignore"):
        return HMMFit(m, order, mean, std, _bic(float(m.score(Z)), cfg.n_states, Z.shape[1], len(Z)), seed,
                      pd.DataFrame(), np.log(A), np.log(np.maximum(m.startprob_[order], 1e-300)), names)


# -------------------------------------------------------------- trading ---
def trading_demo(res: RunResult, block: str = "test") -> dict:
    """ILLUSTRATIVE rule: long Bull / short Bear / flat Stagnant, simple returns.

    Same-month nowcast predictions use information from the month being traded,
    so they are lagged one month before use; lagged-evidence variants are used as is.
    """
    pred = res.pred[block]
    prep = res.prep
    r = prep.raw["simple_ret"].reindex(pred.index)
    sig = pred.map({"Bull": 1, "Stagnant": 0, "Bear": -1})
    pos = sig.shift(1).fillna(0) if res.cfg.lag == 0 else sig
    strat = pos * r
    n_trades = int((pos.diff().abs().fillna(pos.abs()) > 0).sum())
    sd = strat.std()
    sharpe = float(strat.mean() / sd * np.sqrt(12)) if sd and sd > 0 else float("nan")
    return {"strategy_cum": float((1 + strat).prod() - 1), "buy_hold_cum": float((1 + r).prod() - 1),
            "ann_sharpe": sharpe, "n_trades": n_trades, "n_months": int(len(r)),
            "series": pd.DataFrame({"position": pos, "ret": r, "strategy": strat})}


# ------------------------------------------------------- "today" view -----
def latest_view(cfg: G.GWP3Config) -> dict:
    """Model read-out for the most recent complete month in ``cfg``'s data.

    * current regime: argmax of the causal filtered state probability at the last month;
    * HMM next-month distribution: filtered probabilities times the transition matrix;
    * BN next-month distribution: network trained with one-month-lagged evidence, queried
      with the latest month's evidence.
    Descriptive model output on monthly data — not a price forecast or advice.
    """
    prep = prepare(replace(cfg, lag=0))
    pf = prep.probs_filtered.iloc[-1]
    A = np.exp(prep.fit.log_A)
    nxt = pd.Series(pf.values @ A, index=pf.index)
    r1 = run_cached(replace(cfg, lag=1))
    ev = {c: r1.prep.ev_all.iloc[-1][c] for c in G.MACRO_NODES}
    try:
        bn_next = B.infer_network(r1.bn, T, ev).reindex(list(G.REGIMES)).fillna(0.0)
    except B.InferenceError:
        bn_next = None
    spell = 1
    lab = prep.regimes_filtered
    while spell < len(lab) and lab.iloc[-1 - spell] == lab.iloc[-1]:
        spell += 1
    return {"month": prep.raw.index[-1], "regime": pf.idxmax(), "filtered": pf, "hmm_next": nxt,
            "bn_next": bn_next, "evidence": ev, "months_in_regime": spell,
            "raw_last": prep.raw.iloc[-1], "selected_score": r1.selected,
            "persist_prob": float(A[list(pf.index).index(pf.idxmax())][list(pf.index).index(pf.idxmax())])}


# ------------------------------------------------------ simulations --------
def simulate_regimes(fit: HMMFit, start_probs: np.ndarray, horizon: int, n_paths: int, seed: int = 0):
    """Monte-Carlo regime paths from the fitted transition matrix. Returns (paths[n,h], prob[h,K])."""
    rng = np.random.default_rng(seed)
    A = np.exp(fit.log_A); K = A.shape[0]
    state = rng.choice(K, size=n_paths, p=np.asarray(start_probs) / np.sum(start_probs))
    paths = np.empty((n_paths, horizon), dtype=int)
    cum = A.cumsum(1)
    for h in range(horizon):
        u = rng.random(n_paths)
        state = (u[:, None] > cum[state]).sum(1).clip(max=K - 1)
        paths[:, h] = state
    prob = np.stack([(paths == k).mean(0) for k in range(K)], axis=1)
    return paths, prob


def bn_probability_grid(bn, node_x: str, node_y: str, regime: str, method: str = "network") -> pd.DataFrame:
    """P(regime | node_x=a, node_y=b), other nodes marginalised.

    ``network``: exact inference in the learned DAG (flat if the DAG ignores those nodes).
    ``empirical``: relative frequency in the training data (may be sparse; cells with no data are NaN)."""
    rows = {}
    for a in bn.states[node_x]:
        rows[a] = {}
        for b in bn.states[node_y]:
            try:
                if method == "empirical":
                    d, n = B.infer_empirical(bn, T, {node_x: a, node_y: b})
                    rows[a][b] = float(d.get(regime, np.nan)) if n else np.nan
                    continue
                rows[a][b] = float(B.infer_network(bn, T, {node_x: a, node_y: b}).get(regime, np.nan))
            except B.InferenceError:
                rows[a][b] = np.nan
    return pd.DataFrame(rows).T


def sweep_accuracy(base: G.GWP3Config, train_fracs, vol_windows) -> pd.DataFrame:
    """Validation/test accuracy grid over (train fraction, volatility window). Sensitivity only."""
    rows = []
    for tf in train_fracs:
        for vw in vol_windows:
            try:
                r = run(replace(base, train_frac=tf, val_frac=(1 - tf) / 2, vol_window=vw, n_seeds=min(base.n_seeds, 3)))
                rows.append({"train_frac": tf, "vol_window": vw, "val_acc": r.metrics["val"]["accuracy"],
                             "test_acc": r.metrics["test"]["accuracy"], "test_majority": r.metrics["test"]["majority_acc"]})
            except Exception:
                rows.append({"train_frac": tf, "vol_window": vw, "val_acc": np.nan, "test_acc": np.nan, "test_majority": np.nan})
    return pd.DataFrame(rows)
