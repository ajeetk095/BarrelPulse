"""GWP3 configuration and *provenance-tagged reference values*.

Two kinds of constants live here and they must never be mixed up:

* ``RECORDED_NOTEBOOK`` — numbers printed in the group's executed Colab notebook
  (``GWP3_14660.ipynb``) / submitted report. They were produced on data the
  sandbox could not access (Google Drive CSV, EIA, Yahoo Finance). They are shown
  as *recorded*, never as results computed by this app.
* ``ALVI_VERIFIED`` — figures checked line-by-line against the dissertation PDF
  (Alvi, 2018, "Application of Probabilistic Graphical Models in Forecasting
  Crude Oil Price"), arXiv:1804.10869.

Everything the app *computes* comes from ``core.gwp3_pipeline`` on the bundled
FRED snapshot and is labelled "computed".
"""
from __future__ import annotations

from dataclasses import dataclass

DEVELOPER = {
    "name": "Ajeet Kumar",
    "headline": "Bridging code & capital.",
    "intro": ("Quantitative finance specialist with an M.Sc. in Financial Engineering and a "
              "B.Tech in IT. I turn complex data into alpha using Python, C++, SQL, and ML — "
              "from Markowitz MVO and Black-Litterman to Monte Carlo and deep learning."),
    "achievements": [
        "1st Place — IIT Kanpur ML Hackathon",
        "Felicitated by the President of India",
        "Built AI systems at Algo8.ai and Amul",
    ],
    "objective": ("Seeking full-time Quant Research / Financial Engineering roles to solve "
                  "high-stakes problems where finance meets engineering."),
    "github": "https://github.com/ajeetk095",
    "linkedin": "https://www.linkedin.com/in/ajeek095/",
    "email": "ajeetk095@gmail.com",
}


@dataclass(frozen=True)
class GWP3Config:
    start: str = "2006-01-01"          # DTWEXBGS (broad USD index) begins 2006-01
    end: str = "2024-12-01"          # "" = latest complete month available
    train_frac: float = 0.80
    val_frac: float = 0.10             # test = remainder (0.10)
    vol_window: int = 3                # months, un-annualised rolling std of log-returns
    n_states: int = 3
    n_seeds: int = 8
    hmm_n_iter: int = 300
    hmm_tol: float = 1e-5
    max_parents: int = 3
    scores: tuple = ("bic", "k2", "bdeu")
    laplace_alpha: float = 1.0
    lag: int = 0                       # evidence lag in months (0 = same-month nowcast)
    use_filtered_labels: bool = False  # True: causal forward-filtered labels
    add_prev_regime: bool = False      # True: regime_{t-1} persistence node
    seed_edges: bool = False           # True: expert-seeded hill-climb start graph
    force_connected: bool = False      # Alvi p.62 rule: ensure the target has >=1 parent
    data_mode: str = "snapshot"        # "snapshot" (bundled CSVs) or "live" (fetch now, falls back to cache/snapshot)
    refresh_nonce: int = 0             # bumped by "Refresh data" to defeat caches


TARGET = "WTI_Regime"
MACRO_NODES = ("VIX_Level", "DXY_Trend", "IndPro_Trend", "CPI_Trend")
REGIMES = ("Bear", "Stagnant", "Bull")
EXPERT_EDGES = (("VIX_Level", TARGET), ("DXY_Trend", TARGET), ("IndPro_Trend", TARGET))

# --------------------------------------------------------------------------
# Recorded in the group's executed notebook / submitted report (NOT recomputed)
# --------------------------------------------------------------------------
RECORDED_NOTEBOOK = {
    "source": "GWP3_14660.ipynb (executed Colab run) and GWP3_Report_14660.pdf",
    "data_note": ("Spot/futures WTI, OPEC production, EIA and Yahoo series loaded from a private "
                  "Google Drive CSV; not reachable from this environment, so these numbers could "
                  "not be re-run."),
    "split": {"train": 189, "val": 23, "test": 25,
              "train_range": ("2005-04-01", "2020-12-01"),
              "val_range": ("2021-01-01", "2022-11-01"),
              "test_range": ("2022-12-01", "2024-12-01")},
    "val_accuracy": 0.4783,
    "test_accuracy": 0.6400,
    "regime_share_pct": {"Training": (39.2, 59.8, 1.1), "Validation": (39.1, 60.9, 0.0),
                         "Testing": (32.0, 68.0, 0.0)},
    "test_report": {"Bear": (0.33, 0.12, 0.18, 8), "Stagnant": (0.68, 0.88, 0.77, 17),
                    "macro_avg": (0.51, 0.50, 0.48, 25), "weighted_avg": (0.57, 0.64, 0.58, 25)},
    "ks_p": 0.0455, "ks_stat": 0.3791, "mw_p": 0.0867, "mw_stat": 371.0,
    "val_test_contingency": [[9, 14, 0], [8, 17, 0]],
    "train_descriptives": {"WTI_mean": 70.69, "ret_std_pct": 10.855, "ann_vol_pct": 37.60},
    "hmm_train_summary": {"Bear": (74, 39.2), "Stagnant": (113, 59.8), "Bull": (2, 1.1)},
    "hmm_best_seed": 5, "hmm_best_bic": 821.30,
    "hmm_transition": [[0.823, 0.163, 0.014], [0.100, 0.900, 0.000], [0.500, 0.000, 0.500]],
    "majority_baseline_val": 0.609, "majority_baseline_test": 0.68,
    "notes": ("Recorded test accuracy (64.0%) is BELOW the Stagnant-majority share of the test "
              "window (68.0%); validation accuracy (47.83%) is below 60.9%."),
}

# --------------------------------------------------------------------------
# Verified directly against the Alvi (2018) PDF
# --------------------------------------------------------------------------
ALVI_VERIFIED = {
    "citation": ("Alvi, Aziz. Application of Probabilistic Graphical Models in Forecasting "
                 "Crude Oil Price. 2018. arXiv:1804.10869. MSc dissertation, "
                 "University College London."),
    "split": "80:10:10 train/validation/test",
    "scoring_choice": "Scoring function chosen on the validation set, then the test set used once.",
    "val_error_pct": 67.857, "val_accuracy_pct": 32.143,
    "test_error_pct": 42.857, "test_accuracy_pct": 57.143,
    "test_n": 28,
    "error_computation": "error compared actual vs np.roll(predicted, 1) (one-step shift)",
    "structure_learning": "Hill climbing, K2 score, expert-seeded edges (Section 4.2.6)",
    "disconnected_rule": "If hill climbing returns disconnected trees, rerun (p. 62).",
    "contributions": [
        "Chapter 5 lists eight contributions (pp. 66-67); see docs/CONTRIBUTIONS_ASSESSMENT.md "
        "for the row-by-row mapping. The notebook's own 'C8' does not match Alvi's eighth.",
    ],
    "unverified": ("The '+8.7%'-style trading figures quoted in the group's draft could not be "
                   "located in the dissertation and are NOT used by this app."),
}

REPLICATION_VERDICT = (
    "Partial / conceptual replication, not like-for-like. Alvi (2018) uses a binary-difference "
    "discrete HMM per series, expert-seeded K2 hill climbing and a 28-observation test set; the "
    "group (and this app) use a Gaussian two-feature HMM on WTI plus tertile-discretised macro "
    "variables on a different sample. Accuracy figures are therefore not directly comparable."
)
