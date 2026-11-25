import pandas as pd
import streamlit as st

from core import gwp3_contrib as K
from core import ui_components as ui

ui.page_header("Assessment of the eight contributions", "Alvi (2018), Chapter 5, pp. 66–67 — judged against GWP3 evidence.", "Assessment")
df = pd.DataFrame(K.CONTRIBUTIONS, columns=["ID", "Claim", "Status", "Evidence", "Caveat"]).set_index("ID")
DEF = list(df["Status"].unique())
st.session_state.setdefault("con.sel", DEF)
c = st.columns([4, 1])
sel = c[0].multiselect("Filter by status", DEF, key="con.sel")
c[1].button("↺ Reset", on_click=lambda: st.session_state.__setitem__("con.sel", DEF))
view = df[df["Status"].isin(sel)]
st.dataframe(view, width="stretch")
st.bar_chart(df["Status"].value_counts(), color="#22D3EE")
pick = st.selectbox("Inspect one contribution", list(df.index), key="con.pick")
r = df.loc[pick]
ui.callout("plain", f"{pick} — {r['Status']}", f"**Claim:** {r['Claim']}\n\n**Evidence:** {r['Evidence']}\n\n**Caveat:** {r['Caveat']}")
ui.callout("note", "About the list", K.NOTE)
ui.disclaimer()
