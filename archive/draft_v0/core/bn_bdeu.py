"""BDeu-scored Hill Climb Search — CrudeCortex GWP3.
BDeu marginal likelihood with equivalent sample size (ESS), the scoring family
Alvi (2018) used (K2/BDeu, §3.2.2 p.41). Greedy add/delete/reverse over DAGs.
"""
from __future__ import annotations

import math

from core.bn_engine import is_dag


def bdeu_score(data, node: str, parents: tuple, ess: float = 1.0) -> float:
    """BDeu = sum_j [lg a_j - lg(a_j+N_j)] + sum_{j,k} [lg(a_jk+N_jk) - lg a_jk]."""
    r = data[node].nunique()
    q = 1
    for p in parents:
        q *= data[p].nunique()
    a_jk = ess / (q * r)
    a_j = ess / q
    score = 0.0

    def _term(counts):
        s = math.lgamma(a_j) - math.lgamma(a_j + counts.sum())
        for nk in counts:
            s += math.lgamma(a_jk + nk) - math.lgamma(a_jk)
        return s

    if not parents:
        score += _term(data[node].value_counts())
    else:
        for _, grp in data.groupby(list(parents), observed=True)[node]:
            score += _term(grp.value_counts())
    return score


def hill_climb_search_scored(data, nodes, score_fn, max_parents: int = 2):
    """Greedy structure search with any node-parent score function."""
    nodes = list(nodes)
    local = {n: score_fn(n, ()) for n in nodes}
    edges: set = set()
    history = [{"step": 0, "operation": "init (empty graph)", "edge": "-",
                "total_score": round(sum(local.values()), 3)}]
    step = 0
    while True:
        best = None
        for x in nodes:
            for y in nodes:
                if x == y:
                    continue
                if (x, y) not in edges and len([e for e in edges if e[1] == y]) < max_parents:
                    par = tuple(p for a, p in edges if p == y) + (x,)
                    gain = score_fn(y, par) - local[y]
                    if is_dag(edges | {(x, y)}, nodes) and (best is None or gain > best[0]):
                        best = (gain, "add", (x, y))
                if (x, y) in edges:
                    new_edges = edges - {(x, y)}
                    par = tuple(p for a, p in new_edges if p == y)
                    gain = score_fn(y, par) - local[y]
                    if best is None or gain > best[0]:
                        best = (gain, "delete", (x, y))
                    rev_edges = (edges - {(x, y)}) | {(y, x)}
                    if is_dag(rev_edges, nodes) and \
                       len([e for e in rev_edges if e[1] == x]) <= max_parents:
                        par_y = tuple(p for a, p in rev_edges if p == y)
                        par_x = tuple(p for a, p in rev_edges if p == x)
                        g2 = (score_fn(y, par_y) - local[y]) + (score_fn(x, par_x) - local[x])
                        if best is None or g2 > best[0]:
                            best = (g2, "reverse", (x, y))
        if best is None or best[0] <= 1e-9:
            break
        gain, op, edge = best
        if op == "add":
            edges.add(edge)
        elif op == "delete":
            edges.discard(edge)
        elif op == "reverse":
            edges.discard(edge)
            edges.add((edge[1], edge[0]))
        for n in edge:
            local[n] = score_fn(n, tuple(p for a, p in edges if p == n))
        step += 1
        history.append({"step": step, "operation": op, "edge": f"{edge[0]} -> {edge[1]}",
                        "total_score": round(sum(local.values()), 3)})
    return edges, history
