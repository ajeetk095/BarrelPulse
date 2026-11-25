import streamlit as st

from core import gwp3_ui as U
from core import ui_components as ui

ui.page_header("Discussion — in plain language", "What the results mean for non-specialists (updates with your settings).", "Discussion")
cfg = U.sim_controls("dis", "report", groups=("data", "split", "design"))
res = U.safe_run(cfg)
if res is None:
    st.stop()
m = res.metrics["test"]
ui.callout("plain", "What we did", "We labelled each past month as falling (Bear), flat (Stagnant) or rising (Bull) oil, then asked whether volatility, the dollar, industrial output and energy prices could tell us which label applies.")
beat = m["accuracy"] > m["majority_acc"]
ui.callout("ok" if beat else "warn", "What we found", f"On the last {m['n']} months the approach was right {U.pct(m['accuracy'])} of the time. Always guessing the most common regime would have been right {U.pct(m['majority_acc'])}. "
           + ("It edged ahead of that guess here, but with so few months the difference may be luck." if beat else "So the model did not earn its keep in this test."))
ui.callout("note", "Why that can happen", "Oil regimes last for months, so 'same as last month' is hard to beat; the test period can be calm; and a couple of dozen months is a very small test.")
ui.callout("plain", "What we do not claim", "We do not claim the method is better, faster or cheaper than alternatives, that it earns money, or that it predicts prices.")
ui.disclaimer()
