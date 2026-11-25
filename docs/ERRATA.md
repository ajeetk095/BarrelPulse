# Errata — issues found in the draft app, notebook and submitted report
Status: **Fixed** = corrected in this package; **Flagged** = in the submitted PDF/notebook, which were left unchanged.

| # | Item | Status |
|---|------|--------|
| 1 | Draft hill-climb mishandled parent counts | Fixed (engine rewritten/validated, tests) |
| 2 | Draft tests hard-coded to pass ("beats Alvi") | Fixed (tests assert mechanics, not outcomes) |
| 3 | Notebook picks scoring function by number of edges | Fixed (validation accuracy, tie-break log-loss) |
| 4 | K2 scoring failed in the draft | Fixed (K2/BDeu verified equal to pgmpy) |
| 5 | Trading figures such as "+8.7%" attributed to Alvi were not found in the PDF | Flagged; not used |
| 6 | Report Table 1 sign error | Flagged |
| 7 | Report introduction date typo | Flagged |
| 8 | Recorded model accuracy is below the majority-class share (64.0% vs 68.0% test; 47.83% vs 60.9% validation) | Flagged; stated prominently |
| 9 | Sharpe computed from very few observations | Flagged; app labels it meaningless |
| 10 | Log-returns compounded as if simple | Fixed (simple returns in trading demo) |
| 11 | Notebook "C8" (trend forecasting) does not match Alvi's eighth contribution | Fixed in app; flagged |
| 12 | Same-month nowcast presented as forecasting; Viterbi smoothing uses future data | Fixed: V1/V2/V3 variants; stated in Methodology |
| 13 | Full-sample preprocessing in notebook (winsorising/interpolation) risked look-ahead | Flagged; app fits all transforms on training only |
