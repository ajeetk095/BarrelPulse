# Testing report (2026-11-25)
**Performed — `pytest tests`: 50 passed, 0 failed.**
* Pipeline/leakage/statistics (28): chronological split, train-only fits, selection on validation only, K2/BDeu = pgmpy 1.1.2, hill-climb DAG + seeded start, inference, metrics, baselines, variants, trading lag, developer links, no-secrets scan.
* Live layer (mocked HTTP): FRED CSV and API parsing, EIA parsing, HTTP-error handling, ordered fallback to Yahoo, total failure never raises, month-to-date maths, secrets present/absent.
* Simulation: seeded Monte-Carlo reproducible and normalised; latest-month view probabilities sum to 1; force-connected target; 2-state HMM.
* Interaction (Streamlit AppTest): every page renders; changing a parameter re-runs without errors; **↺ Reset all** restores the default on 7 interactive pages; switching to Live mode degrades gracefully offline; 2-state model, Live mode, scenario evidence and the 3D sweep were exercised.
* Notebook executed top-to-bottom with nbconvert (no errors); HTML exported.
**Found and fixed during testing:** charts crashed for 2-state models (fixed, regression-tested).
**Not performed:** real provider calls (FRED/EIA/Yahoo are blocked from the build sandbox — live fetching is verified only against mocks and offline-degradation), browser/mobile visual QA, performance under concurrent users, Streamlit Cloud deployment, notebook PDF export, MLA report.
Tests assert mechanics and honesty, never a desired accuracy.
