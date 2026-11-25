# Changelog
## 3.0.0 — GWP3
* Stand-alone GWP3 app: 14 pages, colourful dark theme, 3D charts (regime space, probability surfaces, Monte-Carlo surface/paths, accuracy sweep).
* Live layer: FRED (CSV or API key), EIA (API key), Yahoo Finance with ordered fallbacks and visible provenance; offline fallback to the bundled snapshot.
* Parameter panels with per-page reset on every analytical page; Simulation Lab, Scenario Explorer, Risk & Robustness.
* Pipeline: chronological 80/10/10, train-only fits, validation-only selection, baselines, K2/BDeu (checked against pgmpy), seeded hill climbing, causal labels, lagged evidence.
* Docs: errata, requirement matrix, security audit, testing report, deployment guide.
