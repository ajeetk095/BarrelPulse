"""Live-data layer (mocked HTTP — real providers are not reachable from CI), simulations and widget interactions."""
import os, sys
import numpy as np, pandas as pd, pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from core import gwp3_config as G, gwp3_pipeline as P, live_monitor as L, secrets as S


class R:
    def __init__(self, text="", status=200, js=None):
        self.text, self.status_code, self._js = text, status, js
    def json(self): return self._js


def _csv(sid, n=40):
    d = pd.bdate_range("2026-09-01", periods=n)
    return f"observation_date,{sid}\n" + "\n".join(f"{x:%Y-%m-%d},{70 + i * .1:.2f}" for i, x in enumerate(d))


def test_fred_daily_csv_and_api(monkeypatch):
    s = L.fred_daily("DCOILWTICO", lambda *a, **k: R(_csv("DCOILWTICO")))
    assert len(s) == 40 and s.index.is_monotonic_increasing
    js = {"observations": [{"date": "2026-10-01", "value": "80"}, {"date": "2026-10-02", "value": "."}]}
    s2 = L.fred_daily("X", lambda *a, **k: R(js=js), key="k")
    assert len(s2) == 1


def test_http_errors_raise():
    with pytest.raises(Exception):
        L.fred_daily("X", lambda *a, **k: R(status=500))


def test_eia_parser():
    js = {"response": {"data": [{"period": "2026-10-02", "value": "81.5"}, {"period": "2026-10-01", "value": "80.0"}]}}
    s = L.eia_daily_wti("k", lambda *a, **k: R(js=js))
    assert s.iloc[-1] == 81.5 and s.index[0] < s.index[-1]


def test_monitor_fallback_to_yahoo_and_total_failure(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False); monkeypatch.delenv("EIA_API_KEY", raising=False)
    idx = pd.bdate_range("2026-09-01", periods=30)
    yahoo = lambda t, period="2y": pd.Series(np.linspace(60, 70, 30), index=idx)
    bad = lambda *a, **k: R(status=503)
    q = L.fetch_monitor(bad, yahoo)
    assert q["WTI"].series is not None and q["WTI"].source.startswith("Yahoo") and q["WTI"].errors
    def boom(t, period="2y"): raise RuntimeError("down")
    q2 = L.fetch_monitor(bad, boom)          # never raises
    assert all(v.series is None and v.errors for v in q2.values())


def test_month_to_date():
    s = pd.Series(np.linspace(60, 66, 30), index=pd.bdate_range("2026-10-01", periods=30))
    m = L.month_to_date(s)
    assert m["last"] == s.iloc[-1] and np.isfinite(m["vol_20d_ann"])


def test_secrets_absent_and_present(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    assert S.fred_api_fetcher() is None
    monkeypatch.setenv("FRED_API_KEY", "abc")
    assert callable(S.fred_api_fetcher())


def test_live_mode_falls_back_when_offline():
    cfg = G.GWP3Config(data_mode="live", refresh_nonce=991)
    info = P.dataset_info(cfg)
    assert info["last"] >= pd.Timestamp("2024-12-01")
    assert set(info["origins"].values()) <= {"live", "cache", "snapshot", "yahoo_fallback"}


def test_latest_view_probabilities():
    v = P.latest_view(G.GWP3Config(end=""))
    assert v["filtered"].sum() == pytest.approx(1) and v["hmm_next"].sum() == pytest.approx(1)
    if v["bn_next"] is not None:
        assert v["bn_next"].sum() == pytest.approx(1)


def test_simulation_is_seeded_and_normalised():
    r = P.run_cached(G.GWP3Config())
    p0 = np.array([1, 0, 0])
    a1, pr1 = P.simulate_regimes(r.prep.fit, p0, 12, 500, 1)
    a2, pr2 = P.simulate_regimes(r.prep.fit, p0, 12, 500, 1)
    assert (a1 == a2).all() and pr1.sum(1) == pytest.approx(np.ones(12))


def test_force_connected_gives_target_parent():
    r = P.run(G.GWP3Config(force_connected=True, scores=("bic",)))
    assert r.target_parents


def test_two_state_model_runs():
    r = P.run(G.GWP3Config(n_states=2))
    assert set(r.prep.fit.names) == {"Bear", "Bull"}


PAGES = {"simulation_lab": "lab", "forecasting": "fc", "scenario": "scn", "risk": "risk",
         "validation": "val", "data_splits": "data", "discussion": "dis"}


@pytest.mark.parametrize("page,pre", PAGES.items())
def test_reset_restores_defaults_and_pages_survive_changes(page, pre):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(ROOT, "views", f"{page}.py"), default_timeout=600).run()
    assert not at.exception
    key = f"{pre}.vol_window"
    if key in [s.key for s in at.slider]:
        d = at.slider(key=key).value
        at.slider(key=key).set_value(7).run(); assert not at.exception and not at.error
        at.button(key=f"{pre}.reset").click().run()
        assert at.slider(key=key).value == d
    at.radio(key=f"{pre}.mode").set_value("Live (fetch now)").run() if f"{pre}.mode" in [r.key for r in at.radio] else None
    assert not at.exception


def test_live_page_degrades_gracefully():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(ROOT, "views", "live_market.py"), default_timeout=600).run()
    assert not at.exception
