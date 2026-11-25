import time

import streamlit as st

from core import contact
from core import gwp3_config as G
from core import ui_components as ui

d = G.DEVELOPER
ui.page_header(d["name"], d["headline"], "About the developer")
st.write(d["intro"])
a, b = st.columns(2)
with a:
    st.subheader("Highlights")
    for x in d["achievements"]:
        st.markdown(f"- {x}")
with b:
    st.subheader("Objective")
    st.info(d["objective"])

mail = f"mailto:{d['email']}?subject=BarrelPulse%20%E2%80%94%20Hello"
c = st.columns(5)
c[0].link_button("Hire Me", mail + "%20(Hire)", width="stretch")
c[1].link_button("Contact Developer", "#send-a-message", width="stretch")
c[2].link_button("Connect on LinkedIn", d["linkedin"], width="stretch")
c[3].link_button("Explore GitHub", d["github"], width="stretch")
c[4].link_button("Send an Email", mail, width="stretch")

st.markdown("<a id='send-a-message'></a>", unsafe_allow_html=True)
st.subheader("Send a message")


@st.dialog("Message sent")
def _thanks():
    st.success("Thank you! Your message has been sent. **The developer will contact you soon.**")
    if st.button("Close", key="contact.close"):
        st.rerun()


FIELDS = ("name", "email", "subject", "message")

for f in FIELDS:
    st.session_state.setdefault(f"contact.{f}", "")
if st.session_state.pop("contact.sent", False):
    for f in FIELDS:  # clear BEFORE the widgets are created
        st.session_state[f"contact.{f}"] = ""
    _thanks()

is_mail_ready = contact.configured()
if not is_mail_ready:
    st.warning(
        "The in-page mail service is not configured on this deployment (the owner must add Gmail secrets). "
        "You can still use the Email or LinkedIn buttons above."
    )

with st.form("contact_form", clear_on_submit=False):
    n, e = st.columns(2)
    n.text_input("Your name", max_chars=contact.MAX["name"], key="contact.name")
    e.text_input("Your email", max_chars=contact.MAX["email"], key="contact.email")
    st.text_input("Subject", max_chars=contact.MAX["subject"], key="contact.subject")
    st.text_area("Message", height=160, max_chars=contact.MAX["message"], key="contact.message")
    send = st.form_submit_button("Send message", type="primary", disabled=not is_mail_ready)

if send:
    now = time.time()
    last = st.session_state.get("contact.last", 0.0)
    count = st.session_state.get("contact.count", 0)
    if now - last < 60:
        st.error(f"Please wait {int(60 - (now - last))} s before sending another message.")
    elif count >= 5:
        st.error("Message limit for this session reached. Please use the email button.")
    else:
        try:
            contact.send_message(*(st.session_state[f"contact.{f}"] for f in FIELDS))
        except contact.ContactError as exc:
            st.error(str(exc))
        else:
            st.session_state["contact.last"], st.session_state["contact.count"] = now, count + 1
            st.session_state["contact.sent"] = True
            st.rerun()

st.caption(f"Direct: {d['email']} · [GitHub]({d['github']}) · [LinkedIn]({d['linkedin']})")
ui.disclaimer()
