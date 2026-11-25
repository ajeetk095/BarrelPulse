# Requirement → evidence matrix
| GWP3 requirement | Where addressed | Status |
|---|---|---|
| Training/validation/testing definitions | app *Data & Splits*; notebook §1 | Done |
| Validation vs testing comparison | app *Data & Splits*; notebook §2 | Done |
| Replicate Alvi §4.3 | app *Validation & Replication*; notebook §3 | Partial/conceptual (see verdict) |
| Forecast accuracy + graphs | app *Forecasting Results*; notebook §4 | Done |
| Assess the 8 contributions | app *Contributions*; `core/gwp3_contrib.py` | Done (conservative) |
| Non-technical discussion | app *Discussion* | Done |
| Justified improvements | app *Model Improvements*; notebook §5 | Done (V1–V4) |
| Robustness | app *Risk & Robustness*; notebook §6 | Done |
| Executable notebook + HTML | `notebooks/` | Done; PDF export **not performed** |
| MLA academic report | **Not produced** in this package; the submitted PDF is in `documents/` | Not done |
| Security audit | `docs/SECURITY_AUDIT.md` | Done |
| Deployment | `docs/DEPLOYMENT.md` | Guide only; **not deployed** |

## Added in the stand-alone/interactive revision
| Request | Where | Status |
|---|---|---|
| Missing GitHub/Streamlit files | `.github/workflows/ci.yml`, `.streamlit/config.toml` + `secrets.toml.example`, `CONTRIBUTING.md`, `CHANGELOG.md`, `data/README.md`, `docs/USER_GUIDE.md`, `docs/DEPLOYMENT.md` | Done |
| Interactive pages with parameters/filters | `core/gwp3_ui.py::sim_controls` on 9 analytical pages | Done |
| Reset parameters and time range | "↺ Reset all" per page (tested on 7 pages) | Done |
| 3D graphs & simulations | regime space, probability surfaces, Monte-Carlo surface/paths, accuracy sweep | Done |
| Colourful dark theme | `inject_theme()`, `.streamlit/config.toml` | Done |
| Live data from several platforms | FRED, EIA, Yahoo (`core/live_monitor.py`, `core/secrets.py`) | Built; **not exercised against real providers here** |
| Run model on today's market | Live Market page (latest complete month, next-month probabilities) | Works on live/cache/bundled data; shows origin |
