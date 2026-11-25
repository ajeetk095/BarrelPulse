"""
CRUDECORTEX — GWP3 · The brain behind the barrel.
Results, Interpretation & Model Improvement — MScFE 660 GWP3.
Run:  streamlit run app.py
"""
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.ui_components import inject_css

st.set_page_config(
    page_title="CrudeCortex GWP3 — Results, Interpretation & Model Improvement",
    page_icon="🧠🛢️", layout="wide", initial_sidebar_state="expanded")
inject_css()

HOME = st.Page("pages/1_🏠_Home_Research_Overview.py", title="Home — Results Overview",
               icon="🏠", url_path="home", default=True)
SPLITS = st.Page("pages/2_✂️_Datasets_&_Splits.py", title="Steps 1–2 — Datasets & Split Comparison",
                 icon="✂️", url_path="datasets-splits")
VALID = st.Page("pages/3_✅_Validation_&_Replication.py", title="Steps 3–4 — Model Validation & Replication",
                icon="✅", url_path="validation-replication")
ACC = st.Page("pages/4_🎯_Forecast_Accuracy_&_Trading.py", title="Step 5 — Forecast Accuracy & Trading",
              icon="🎯", url_path="accuracy-trading")
CONTRIB = st.Page("pages/5_🧾_Alvi_2018_Contribution_Scorecard.py", title="Step 6 — Contribution Scorecard (8 results)",
                  icon="🧾", url_path="contributions")
DISC = st.Page("pages/6_💬_Non_Technical_Discussion.py", title="Step 7 — Non-Technical Discussion",
               icon="💬", url_path="discussion")
IMPROVE = st.Page("pages/7_🚀_Model_Improvement.py", title="Supplementary — Model Improvement",
                  icon="🚀", url_path="improvement")
ABOUT = st.Page("pages/8_👤_Methodology_&_About.py", title="Methodology, Docs & About",
                icon="👤", url_path="methodology-about")

pg = st.navigation([HOME, SPLITS, VALID, ACC, CONTRIB, DISC, IMPROVE, ABOUT])
with st.sidebar:
    st.markdown(
        "<div style='padding:4px 4px 0 4px'>"
        "<span style='font-size:1.6em'>🧠🛢️</span> "
        "<span style='font-size:1.15em;font-weight:900;color:#C2571B'>CRUDECORTEX</span><br>"
        "<span style='font-size:.8em;font-style:italic;color:#6B7280'>"
        "The brain behind the barrel.</span></div>",
        unsafe_allow_html=True)
    st.caption("HMM Regime Detection · Bayesian Networks · GWP2 — Methodology & Model Development")
    st.divider()
    st.caption("**MScFE 660 — Risk Management**  \nGroup Work Project 2  \n"
               "Group 14660 — Ajeet Kumar · Frank Osei · Nha Tran  \n"
               "Data: FRED · Yahoo Finance · 2006-01 → 2024-12 (228 months)")
    with st.expander("📚 Quick help"):
        st.markdown("""
**New here?** Follow the pages top-to-bottom — they mirror the official
GWP2 steps 1→9.

- Every analytical page ends with an **🧭 Understand this page** panel.
- The toy weather example is fully interactive in the **Algorithms Lab**.
- All models recompute live from the bundled FRED data — no API keys needed.

Full walkthrough: `USER_GUIDE.md`.
""")

pg.run()
