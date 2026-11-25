"""Hidden Markov Model engine — CrudeCortex GWP2.
Part 1: from-scratch discrete HMM algorithms (Forward, Backward, Viterbi,
Baum-Welch) implementing the report's pseudocode, run on the 3-state weather
toy example.
Part 2: the 3-state Gaussian HMM regime model (hmmlearn) reproducing the
report's configuration: full covariance, 200 EM iterations, tol 1e-4,
10 random restarts, best log-likelihood retained.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TOY_STATES = ["Sunny", "Cloudy", "Rainy"]
TOY_OBS = ["Dry", "Damp", "Wet"]
TOY_OBS_SEQ = [0, 1, 2]                      # Dry, Damp, Wet
TOY_PI = np.array([0.6, 0.3, 0.1])
TOY_A = np.array([[0.7, 0.2, 0.1],
                  [0.3, 0.4, 0.3],
                  [0.2, 0.3, 0.5]])
TOY_B = np.array([[0.5, 0.3, 0.2],
                  [0.2, 0.5, 0.3],
                  [0.2, 0.2, 0.6]])


def forward(A, B, pi, obs):
    """alpha_t(j) = P(o1..ot, q_t=j). Returns (alpha T x N, P(O|lambda))."""
    T, N = len(obs), A.shape[0]
    alpha = np.zeros((T, N))
    alpha[0] = pi * B[:, obs[0]]
    for t in range(1, T):
        for j in range(N):
            alpha[t, j] = B[j, obs[t]] * np.sum(alpha[t - 1] * A[:, j])
    return alpha, float(alpha[-1].sum())


def backward(A, B, pi, obs):
    """beta_t(i) = P(o_{t+1}..oT | q_t=i). Returns (beta T x N, P(O|lambda))."""
    T, N = len(obs), A.shape[0]
    beta = np.zeros((T, N))
    beta[T - 1] = 1.0
    for t in range(T - 2, -1, -1):
        for i in range(N):
            beta[t, i] = np.sum(A[i] * B[:, obs[t + 1]] * beta[t + 1])
    p = float(np.sum(pi * B[:, obs[0]] * beta[0]))
    return beta, p


def viterbi(A, B, pi, obs):
    """Most likely hidden path via max + backpointer. Returns (path, delta, logP)."""
    T, N = len(obs), A.shape[0]
    delta = np.zeros((T, N))
    psi = np.zeros((T, N), dtype=int)
    delta[0] = pi * B[:, obs[0]]
    for t in range(1, T):
        for j in range(N):
            cand = delta[t - 1] * A[:, j]
            psi[t, j] = int(np.argmax(cand))
            delta[t, j] = B[j, obs[t]] * cand.max()
    q_last = int(np.argmax(delta[-1]))
    path = [q_last]
    for t in range(T - 1, 0, -1):
        path.append(int(psi[t, path[-1]]))
    path.reverse()
    return path, delta, float(np.log(delta[-1].max()))


def baum_welch(A, B, pi, obs, n_iter=100, tol=1e-10):
    """EM re-estimation of (pi, A, B). Returns (A,B,pi,ll_history,gamma)."""
    T, N, M = len(obs), A.shape[0], B.shape[1]
    A, B, pi = A.copy(), B.copy(), pi.copy()
    ll_hist = []
    gamma_last = None
    for _ in range(n_iter):
        alpha, p_obs = forward(A, B, pi, obs)
        beta, _ = backward(A, B, pi, obs)
        if not np.isfinite(p_obs) or p_obs <= 0:
            break
        ll_hist.append(p_obs)
        # E-step (eps-guarded against numerical underflow)
        ab = alpha * beta
        gamma = ab / (ab.sum(axis=1, keepdims=True) + 1e-300)
        xi = np.zeros((T - 1, N, N))
        for t in range(T - 1):
            denom = np.sum(alpha[t] * A * (B[:, obs[t + 1]] * beta[t + 1])) + 1e-300
            xi[t] = (alpha[t][:, None] * A * (B[:, obs[t + 1]] * beta[t + 1])[None, :]) / denom
        # M-step (eps-guarded empty-state divisions)
        pi = gamma[0].copy()
        gsum = gamma[:-1].sum(axis=0)[:, None]
        A = xi.sum(axis=0) / (gsum + 1e-300)
        for k in range(M):
            mask = np.array(obs) == k
            B[:, k] = gamma[mask].sum(axis=0) / (gamma.sum(axis=0) + 1e-300)
        gamma_last = gamma
        if len(ll_hist) > 1 and abs(ll_hist[-1] - ll_hist[-2]) < tol:
            break
    return A, B, pi, ll_hist, gamma_last


def run_toy_example() -> dict:
    """Full toy run — the report's Sunny/Cloudy/Rainy weather model."""
    A, B, pi, obs = TOY_A, TOY_B, TOY_PI, TOY_OBS_SEQ
    alpha, p_fwd = forward(A, B, pi, obs)
    beta, p_bwd = backward(A, B, pi, obs)
    path, delta, logp = viterbi(A, B, pi, obs)
    Aw, Bw, piw, ll_hist, gamma = baum_welch(A, B, pi, obs)
    return {
        "states": TOY_STATES, "obs_names": TOY_OBS,
        "alpha": alpha, "beta": beta, "delta": delta,
        "p_forward": p_fwd, "p_backward": p_bwd,
        "path": [TOY_STATES[i] for i in path], "logp_viterbi": logp,
        "gamma": gamma, "ll_hist": ll_hist,
        "A_learned": Aw, "B_learned": Bw, "pi_learned": piw,
    }


@dataclass
class RegimeHMM:
    model: object
    labels: pd.Series
    ll: float
    transmat: np.ndarray
    means_std: np.ndarray
    state_stats: pd.DataFrame


def fit_regime_hmm(feature_df: pd.DataFrame, X_std: np.ndarray,
                   n_states: int = 3, n_restarts: int = 10,
                   n_iter: int = 200, tol: float = 1e-4) -> RegimeHMM:
    from hmmlearn.hmm import GaussianHMM
    best_model, best_ll = None, -np.inf
    for seed in range(n_restarts):
        m = GaussianHMM(n_components=n_states, covariance_type="full",
                        n_iter=n_iter, tol=tol, random_state=seed)
        try:
            m.fit(X_std)
        except Exception:
            continue
        if m.monitor_.converged and m.score(X_std) > best_ll:
            best_ll, best_model = m.score(X_std), m
    if best_model is None:
        raise RuntimeError("All HMM restarts failed to converge")
    hidden = best_model.predict(X_std)
    order = np.argsort(best_model.means_[:, 0])
    name_map = {order[0]: "Bear", order[1]: "Stagnant", order[2]: "Bull"}
    labels = pd.Series([name_map[h] for h in hidden], index=feature_df.index)

    raw = feature_df.copy()
    raw["Regime"] = labels
    stats = raw.groupby("Regime").agg(
        Months=("WTI_ret", "size"),
        MeanReturn=("WTI_ret", lambda x: float(np.mean(x)) * 100),
        MeanVol=("WTI_vol12", lambda x: float(np.mean(x)) * 100),
    ).reindex(["Bear", "Bull", "Stagnant"])
    stats["PctSample"] = stats["Months"] / stats["Months"].sum() * 100
    return RegimeHMM(best_model, labels, float(best_ll),
                     best_model.transmat_.copy(), best_model.means_.copy(), stats)


def rule_based_regimes(df: pd.DataFrame) -> pd.Series:
    """Report Step 2 benchmark: 12M trailing return +/-10% and volatility gates."""
    ret12 = df["WTI"].pct_change(12)
    vol12 = np.log(df["WTI"] / df["WTI"].shift(1)).rolling(12).std() * np.sqrt(12)
    regimes = pd.Series("Stagnant", index=df.index)
    regimes[(ret12 > 0.10) & (vol12 < 0.40)] = "Bull"
    regimes[(ret12 < -0.10) & (vol12 > 0.30)] = "Bear"
    return regimes


def regime_summary(df: pd.DataFrame, regimes: pd.Series,
                   vol12: pd.Series | None = None) -> pd.DataFrame:
    """Per-regime summary. Masks are reindexed to df.index so a 216-month
    regime Series can be summarised against the 228-month master panel."""
    reg = regimes.reindex(df.index)
    if vol12 is None:
        vol12 = np.log(df["WTI"] / df["WTI"].shift(1)).rolling(12).std() * np.sqrt(12)
    else:
        vol12 = vol12.reindex(df.index)
    ret12 = df["WTI"].pct_change(12)          # report's trailing 12M return
    rows = []
    for reg_name in ["Bull", "Stagnant", "Bear"]:
        m = (reg == reg_name).fillna(False)
        if m.sum() == 0:
            continue
        rows.append({
            "Regime": reg_name, "Months": int(m.sum()),
            "PctSample": round(float(m.mean() * 100), 1),
            "MeanWTI": round(float(df.loc[m, "WTI"].mean()), 2),
            "MeanMonthlyReturn": round(float(ret12[m].mean() * 100 / 12), 2),
            "MeanAnnVol": round(float(vol12[m].mean() * 100), 1),
        })
    return pd.DataFrame(rows).set_index("Regime")
