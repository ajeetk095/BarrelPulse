# Security audit (run 2026-11-25, GWP3 stand-alone)
* **bandit 1.9.4** over `core views app.py scripts` (~3,940 LOC): 0 high / 0 medium / 0 low after replacing a bare `except: pass` in `core/secrets.py` with `contextlib.suppress`, and an `assert` in `core/bn_engine.py` with an explicit exception.
* **pip-audit** over `requirements.txt`: "No known vulnerabilities found" (against the advisory data reachable at run time; rerun before release — CI does it on every push).
* **Secrets:** `FRED_API_KEY` / `EIA_API_KEY` are read only from Streamlit secrets or environment variables, never logged, never written to disk; `.env` and `.streamlit/secrets.toml` are git-ignored; a test scans source for hard-coded credentials. API keys travel only in HTTPS requests to fred/eia endpoints.
* **Network:** fixed provider URLs only (no user-supplied URLs), timeouts and bounded retries, HTML/malformed responses rejected by the parser; all provider failures are caught and shown as messages.
* **UI:** user text is not rendered as HTML except through escaping helpers; parameters are bounded by widgets and re-validated by the pipeline (errors become messages, not tracebacks).
* **Limits:** automated static scan + dependency audit, not a penetration test. Streamlit Community Cloud's own security was not assessed. No claim of being "hack-proof".
