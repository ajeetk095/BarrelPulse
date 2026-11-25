# BarrelPulse — GWP3

MScFE 660 Risk Management · Group 14660 · Developer: Ajeet Kumar. Academic research; **not investment advice**.

*BarrelPulse* (the GWP3 evolution of the CrudeCortex GWP2 project) — HMM regimes + Bayesian networks for crude oil: results, interactive simulations, 3D charts and a live monitor. Dark, colourful Streamlit app.

## Quick start
```bash
pip install -r requirements.txt
streamlit run app.py
python -m pytest tests -q        # dev: pip install -r requirements-dev.txt
```
## Deploy (GitHub + Streamlit Community Cloud)
See `docs/DEPLOYMENT.md`. Optional secrets (`FRED_API_KEY`, `EIA_API_KEY`) go in Streamlit *App settings → Secrets* — see `.streamlit/secrets.toml.example`.

## Data & honesty
* Bundled FRED snapshot (to Oct 2026 for the latest bundled months) works offline; **Live** mode fetches at runtime and falls back with a visible banner.
* Computed vs recorded vs verified-Alvi values are labelled separately (`core/gwp3_config.py`).
* The network does **not** beat the majority-class baseline on held-out months — in the submitted notebook (64.0% vs 68.0%) or in recomputed runs.
* Live fetching could not be exercised from the build sandbox (outbound network blocked); it is covered by mocked tests and graceful-degradation tests.

## Layout
`app.py` · `views/` (14 pages) · `core/` (pipeline, engines, charts, live monitor) · `tests/` · `notebooks/` · `docs/` · `documents/` · `archive/draft_v0/` · `data/fred/` · `.github/workflows/ci.yml`
