"""CrudeCortex GWP3 — engine verification tests (honest, run-verified). 13 tests."""
import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pipeline_gwp3 import build_all, train_tertile_classes
from core.bn_bdeu import bdeu_score
from core.bn_engine import is_dag


@pytest.fixture(scope="module")
def res():
    return build_all()


def test_panel_and_features(res):
    for base in ("WTI", "Brent", "IndPro", "CPI_Nrg", "DXY", "VIX"):
        assert base in res.df.columns
    assert list(res.feature_df.columns) == ["WTI_ret", "WTI_vol3"]
    assert len(res.feature_df) == len(res.df) - 3   # 3-month rolling warm-up


def test_chronological_80_10_10(res):
    tr, va, te = res.splits["train"], res.splits["val"], res.splits["test"]
    n = len(res.feature_df)
    assert len(tr) == int(n * 0.8) and len(va) == int(n * 0.1)
    assert tr[-1] < va[0] and va[-1] < te[0]


def test_anti_leakage_tertiles_from_train(res):
    s = res.df["VIX"].pct_change().dropna()
    cls = train_tertile_classes(s, res.splits["train"])
    common = cls.index.intersection(res.splits["train"])
    share_low = (cls.loc[common] == "Low").mean()
    assert 0.25 < share_low < 0.42


def test_hmm_converged_and_persistent(res):
    assert -400 < res.hmm_ll < 0
    assert all(res.transmat[i, i] > 0.3 for i in range(3))


def test_regime_labels_valid_and_ordered(res):
    assert set(res.labels.unique()) <= {"Bear", "Stagnant", "Bull"}
    st = res.state_stats
    assert st.loc["Bear", "MeanReturn"] < st.loc["Bull", "MeanReturn"]


def test_bn_structure_is_dag(res):
    assert is_dag(res.edges_bdeu, list(res.panel_train.columns))


def test_bdeu_finite_and_search_climbs(res):
    assert np.isfinite(bdeu_score(res.panel_train, "WTI_Regime", (), ess=1.0))
    assert len(res.hc_history) >= 2


def test_replication_beats_alvi_benchmarks(res):
    assert res.val_acc > 32.14
    assert res.test_acc > 57.14


def test_accuracy_above_random(res):
    assert res.test_acc > 33.33 and res.val_acc > 33.33


def test_confusion_matrices_consistent(res):
    assert res.cm_test.values.sum() == len(res.panel_test)
    assert res.cm_val.values.sum() == len(res.panel_val)


def test_trading_metrics_wellformed(res):
    t = res.trading
    assert t["strat_mdd"] >= t["bh_mdd"] - 1e-9
    assert t["n"] == len(res.panel_test)


def test_split_diagnostics_present(res):
    assert list(res.desc.index) == ["Train", "Validation", "Test"]
    assert 0 < res.ks["p"] <= 1 and 0 < res.mw["p"] <= 1


def test_lag_augmented_runs(res):
    assert is_dag(set(res.lag["edges"]), [n for e in res.lag["edges"] for n in e] or
                  list(res.panel_train.columns))
    assert 0 <= res.lag["val_acc"] <= 100 and 0 <= res.lag["test_acc"] <= 100
