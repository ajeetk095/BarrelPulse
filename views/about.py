import streamlit as st

from core import gwp3_config as G
from core import gwp3_ui as U
from core import ui_components as ui

ui.page_header("About this project", "What GWP3 asks and what this app does.")
ui.cards([("GWP1 → GWP2 → GWP3 · BarrelPulse", "GWP1 framed the problem, GWP2 built the HMM + Bayesian network, GWP3 interprets the results, compares validation with testing, assesses contributions and proposes justified improvements."),
          ("Source paper", "Alvi (2018), *Application of Probabilistic Graphical Models in Forecasting Crude Oil Price* (UCL MSc dissertation, arXiv:1804.10869)."),
          ("Integrity rules", "Recorded results are labelled recorded; computed results are labelled computed. No test-set tuning, no investment advice, nothing hard-coded.")])
st.markdown(G.ALVI_VERIFIED["citation"])
ui.disclaimer()
