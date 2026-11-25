"""BarrelPulse — GWP3 · entry point.  Run: streamlit run app.py"""
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import ui_components as ui  # noqa: E402

st.set_page_config(page_title="BarrelPulse — GWP3", page_icon=":material/oil_barrel:", layout="wide")
from core import gwp3_ui as U  # noqa: E402
U.inject_theme()

pages = {
    "Live": [
        st.Page("views/home.py", title="Home", icon=":material/home:", default=True),
        st.Page("views/live_market.py", title="Live Market", icon=":material/sensors:"),
        st.Page("views/simulation_lab.py", title="Simulation Lab", icon=":material/science:"),
        st.Page("views/scenario.py", title="Scenario Explorer", icon=":material/tune:"),
    ],
    "Results": [
        st.Page("views/data_splits.py", title="Data & Splits", icon=":material/database:"),
        st.Page("views/validation.py", title="Validation & Replication", icon=":material/fact_check:"),
        st.Page("views/forecasting.py", title="Forecasting Results", icon=":material/query_stats:"),
        st.Page("views/improvements.py", title="Model Improvements", icon=":material/build:"),
        st.Page("views/risk.py", title="Risk & Robustness", icon=":material/shield:"),
    ],
    "Assessment": [
        st.Page("views/contributions.py", title="Contributions", icon=":material/checklist:"),
        st.Page("views/discussion.py", title="Discussion", icon=":material/forum:"),
        st.Page("views/methodology.py", title="Methodology & Data", icon=":material/rule:"),
        st.Page("views/about.py", title="About", icon=":material/info:"),
        st.Page("views/developer.py", title="Developer Profile", icon=":material/person:"),
    ],
}
with st.sidebar:
    st.markdown("### Barrel**Pulse**")
    st.caption("MScFE 660 · GWP3 · Group 14660")
    st.caption("Academic research — not investment advice.")
st.navigation(pages).run()
