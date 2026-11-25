"""Bayesian network engine — CrudeCortex GWP2.
Score-based structure learning (Hill Climb Search with BIC), MLE CPD fitting
and exact inference by enumeration on the discretized 5-node macro panel —
reproducing the report's Step 8-9 working example.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

REGIME_ORDER = ["Bear", "Stagnant", "Bull"]


def discretize(s: pd.Series, labels=("Low", "Medium", "High")) -> pd.Series:
    """Quantile-based discretization into equal-frequency classes (plain str)."""
    return pd.qcut(s, q=3, labels=labels).astype(str)


def build_discrete_panel(feature_df: pd.DataFrame, labels: pd.Series,
                         df: pd.DataFrame) -> pd.DataFrame:
    """216-row discrete dataset: regime + 4 macro variables (Low/Med/High)."""
    panel = pd.DataFrame({"WTI_Regime": labels})
    for col, name in [("VIX", "VIX_Level"), ("IndPro", "IndPro_Level"),
                      ("CPI_Nrg", "CPI_Level"), ("DXY", "DXY_Level")]:
        panel[name] = discretize(df.loc[feature_df.index, col])
    return panel.dropna().reset_index(drop=True)


def _nz(series: pd.Series) -> pd.Series:
    """Non-zero value counts (guards empty categories)."""
    vc = series.value_counts()
    return vc[vc > 0]


def bic_score(data: pd.DataFrame, node: str, parents: tuple) -> float:
    """BIC = log-likelihood - (ln N / 2) * #free parameters. Higher is better."""
    n = len(data)
    ll = 0.0
    if not parents:
        counts = _nz(data[node])
        for c in counts:
            ll += c * math.log(c / n)
        k = (len(counts) - 1)
        return ll - 0.5 * math.log(n) * k
    g = data.groupby(list(parents), observed=True)[node]
    for _, grp in g:
        nj = len(grp)
        for c in _nz(grp):
            ll += c * math.log(c / nj)
    parent_card = 1
    for p in parents:
        parent_card *= data[p].nunique()
    n_params = (data[node].nunique() - 1) * parent_card
    return ll - 0.5 * math.log(n) * n_params


def is_dag(edges: set, nodes: list) -> bool:
    """Kahn's algorithm acyclicity check."""
    adj = {n: set() for n in nodes}
    indeg = {n: 0 for n in nodes}
    for a, b in edges:
        adj[a].add(b)
        indeg[b] += 1
    queue = [n for n in nodes if indeg[n] == 0]
    seen = 0
    while queue:
        n = queue.pop()
        seen += 1
        for m in adj[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                queue.append(m)
    return seen == len(nodes)


def hill_climb_search(data: pd.DataFrame, nodes: list, max_parents: int = 2):
    """Greedy search over add/delete/reverse operations maximising total BIC."""
    nodes = list(nodes)
    local = {n: bic_score(data, n, ()) for n in nodes}
    edges: set = set()
    history = [{"step": 0, "operation": "init (empty graph)",
                "edge": "-", "total_bic": round(sum(local.values()), 3)}]
    step = 0
    while True:
        best = None
        for x in nodes:
            for y in nodes:
                if x == y:
                    continue
                if (x, y) not in edges and len([e for e in edges if e[1] == y]) < max_parents:
                    gain = bic_score(data, y, tuple(p for a, p in edges if p == y) + (x,)) - local[y]
                    if is_dag(edges | {(x, y)}, nodes) and (best is None or gain > best[0]):
                        best = (gain, "add", (x, y))
                if (x, y) in edges:
                    new_edges = edges - {(x, y)}
                    gain = (bic_score(data, y, tuple(p for a, p in new_edges if p == y)) - local[y])
                    if best is None or gain > best[0]:
                        best = (gain, "delete", (x, y))
                    rev_edges = (edges - {(x, y)}) | {(y, x)}
                    if is_dag(rev_edges, nodes) and \
                       len([e for e in rev_edges if e[1] == x]) <= max_parents:
                        g2 = (bic_score(data, y, tuple(p for a, p in rev_edges if p == y)) - local[y]
                              + bic_score(data, x, tuple(p for a, p in rev_edges if p == x)) - local[x])
                        if best is None or g2 > best[0]:
                            best = (g2, "reverse", (x, y))
        if best is None or best[0] <= 1e-9:
            break
        gain, op, edge = best
        if op == "add":
            edges.add(edge)
            local[edge[1]] += gain
        elif op == "delete":
            edges.discard(edge)
            local[edge[1]] += gain
        elif op == "reverse":
            edges.discard(edge)
            edges.add((edge[1], edge[0]))
            local[edge[1]] += bic_score(data, edge[1],
                                        tuple(p for a, p in edges if p == edge[1])) - local[edge[1]]
            local[edge[0]] = bic_score(data, edge[0],
                                       tuple(p for a, p in edges if p == edge[0]))
        step += 1
        history.append({"step": step, "operation": op, "edge": f"{edge[0]} -> {edge[1]}",
                        "total_bic": round(sum(local.values()), 3)})
    return edges, history


@dataclass
class BNModel:
    data: pd.DataFrame
    nodes: list
    edges: set
    history: list
    cpds: dict = field(default_factory=dict)

    def parents_of(self, node: str) -> list:
        return sorted([a for a, b in self.edges if b == node])


def fit_cpds(bn: BNModel) -> BNModel:
    """MLE conditional probability tables for every node given its parents."""
    for node in bn.nodes:
        parents = bn.parents_of(node)
        if not parents:
            probs = _nz(bn.data[node])
            probs = probs / probs.sum()
            bn.cpds[node] = {"parents": [], "table": {(): probs.to_dict()}}
        else:
            table = {}
            for keys, grp in bn.data.groupby(parents, observed=True):
                key = keys if isinstance(keys, tuple) else (keys,)
                pr = _nz(grp)
                pr = pr / pr.sum()
                table[key] = pr.to_dict()
            bn.cpds[node] = {"parents": parents, "table": table}
    return bn


def infer_node(bn: BNModel, node: str, evidence: dict) -> pd.Series:
    """Exact inference by joint enumeration. P(node | evidence)."""
    mask = pd.Series(True, index=bn.data.index)
    for k, v in evidence.items():
        mask &= bn.data[k] == v
    sub = bn.data.loc[mask, node]
    if len(sub) == 0:
        return pd.Series(dtype=float)
    return sub.value_counts(normalize=True)


def build_bn(panel: pd.DataFrame) -> BNModel:
    nodes = list(panel.columns)
    edges, history = hill_climb_search(panel, nodes)
    bn = BNModel(panel, nodes, edges, history)
    return fit_cpds(bn)


def conditional_regime_table(panel: pd.DataFrame, target: str = "WTI_Regime",
                             cond: str = "VIX_Level") -> pd.DataFrame:
    """Report Table 14 — empirical P(regime | VIX level) from MLE frequencies."""
    rows = []
    for lvl in ["Low", "Medium", "High"]:
        sub = panel[panel[cond] == lvl][target]
        n = len(sub)
        if n == 0:
            continue
        vc = sub.value_counts(normalize=True)
        rows.append({cond: lvl, "Months": int(n),
                     "P(Bear)": round(float(vc.get("Bear", 0)), 3),
                     "P(Stagnant)": round(float(vc.get("Stagnant", 0)), 3),
                     "P(Bull)": round(float(vc.get("Bull", 0)), 3),
                     "Dominant regime": vc.idxmax()})
    return pd.DataFrame(rows)
