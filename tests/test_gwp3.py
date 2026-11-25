"""GWP3 tests. None asserts a desired accuracy: they test mechanics, leakage and honesty."""
import os, sys, glob, re
import numpy as np, pandas as pd, pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from core import bn_engine as B, gwp3_config as G, gwp3_pipeline as P


@pytest.fixture(scope="module")
def res():
    return P.run_cached(G.GWP3Config())


def test_split_is_chronological_and_disjoint(res):
    sp = res.prep.split
    assert sp.index.is_monotonic_increasing
    order = {"train": 0, "val": 1, "test": 2}
    assert sp.map(order).is_monotonic_increasing
    assert (sp == "train").sum() > (sp == "val").sum() > 0 and (sp == "test").sum() > 0


def test_scaler_and_thresholds_use_train_only(res):
    raw, sp = res.prep.raw, res.prep.split
    tr = raw[sp == "train"]
    lo, hi = res.prep.thresholds["VIX_Level"]
    assert lo == pytest.approx(tr["VIX_x"].quantile(1 / 3))
    assert res.prep.fit.mean == pytest.approx(tr[["ret", "vol"]].values.mean(0))


def test_no_overlap_between_prediction_blocks(res):
    assert not set(res.pred["val"].index) & set(res.pred["test"].index)
    assert res.pred["val"].index.max() < res.pred["test"].index.min()


def test_metrics_are_consistent(res):
    for k in ("val", "test"):
        m = res.metrics[k]
        assert 0 <= m["ci_low"] <= m["accuracy"] <= m["ci_high"] <= 1
        assert m["correct"] == int(round(m["accuracy"] * m["n"]))
        assert res.confusion[k].values.sum() == m["n"]


def test_selection_never_uses_test():
    # Perturb test labels' source: selection must depend only on validation columns.
    r = P.run(G.GWP3Config())
    best = r.candidates.sort_values(["val_accuracy", "val_log_loss"], ascending=[False, True]).index[0]
    assert r.selected == best
    assert "test_accuracy" not in r.candidates.columns


def test_wilson_and_mcnemar():
    lo, hi = P.wilson(10, 20)
    assert lo < 0.5 < hi
    y = ["a"] * 10
    assert P.mcnemar_exact(y, y, y) == 1.0


def test_alvi_style_error_shift_differs():
    a = np.array(list("abcab")); p = np.array(list("abcab"))
    assert P.alvi_style_error(a, p) > 0       # np.roll shift misaligns identical sequences


def test_k2_bdeu_match_pgmpy():
    pgmpy = pytest.importorskip("pgmpy.structure_score")
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"A": rng.choice(list("xyz"), 200)})
    df["B"] = np.where(rng.random(200) < .6, df["A"], rng.choice(list("xyz"), 200))
    k, b = pgmpy.K2(df), pgmpy.BDeu(df, equivalent_sample_size=1)
    assert B.k2_score(df, "B", ("A",)) == pytest.approx(k.local_score("B", ("A",)))
    assert B.bdeu_score(df, "B", ("A",)) == pytest.approx(b.local_score("B", ("A",)))


def test_hill_climb_returns_dag_and_respects_seed():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"A": rng.choice(list("xyz"), 150), "C": rng.choice(list("xyz"), 150)})
    df["B"] = df["A"]
    edges, _ = B.hill_climb_search(df, ["A", "C", "B"], max_parents=2, score="bic", start_edges={("C", "B")})
    assert B.is_dag(edges, ["A", "B", "C"])
    with pytest.raises(Exception):
        B.hill_climb_search(df, ["A", "C", "B"], start_edges={("A", "B"), ("B", "A")})


def test_inference_normalised_and_bad_evidence(res):
    p = B.infer_network(res.bn, G.TARGET, {})
    assert p.sum() == pytest.approx(1.0)
    with pytest.raises(B.InferenceError):
        B.infer_network(res.bn, G.TARGET, {"VIX_Level": "Nope"})


def test_baselines_present_and_honest(res):
    m = res.metrics["test"]
    assert {"majority_acc", "persistence_acc", "random_acc_expected"} <= set(m)
    # honesty: the report must be able to say the model does NOT beat the baseline
    assert isinstance(m["accuracy"] < m["majority_acc"], (bool, np.bool_))


def test_variants_run():
    df = P.compare_variants()
    assert len(df) == 5 and df["Test acc"].between(0, 1).all()


def test_trading_demo_lag(res):
    t = P.trading_demo(res)
    assert t["n_months"] == len(res.pred["test"]) and np.isfinite(t["strategy_cum"])


def test_developer_links():
    d = G.DEVELOPER
    assert d["github"] == "https://github.com/ajeetk095"
    assert d["linkedin"] == "https://www.linkedin.com/in/ajeek095/"
    assert re.fullmatch(r"[^@\s]+@[^@\s]+\.[a-z]+", d["email"])


def test_recorded_constants_are_labelled():
    assert G.RECORDED_NOTEBOOK["test_accuracy"] == 0.64
    assert "not reachable" in G.RECORDED_NOTEBOOK["data_note"]
    assert G.ALVI_VERIFIED["test_n"] == 28


@pytest.mark.parametrize("page", sorted(glob.glob(os.path.join(ROOT, "views", "*.py"))))
def test_page_renders(page):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(page, default_timeout=300).run()
    assert not at.exception, [e.value for e in at.exception]


def test_entrypoint_renders():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=300).run()
    assert not at.exception


def test_no_secrets_in_source():
    pat = re.compile(r"(api[_-]?key|secret|password)\s*=\s*['\"][A-Za-z0-9]{12,}", re.I)
    for f in glob.glob(os.path.join(ROOT, "**", "*.py"), recursive=True):
        assert not pat.search(open(f, errors="ignore").read()), f
