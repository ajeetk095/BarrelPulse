"""PAGE 2 — Steps 1-2: Dataset definitions & split comparison."""
import sys, os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.pipeline_gwp3 import build_all

st.title("✂️ Steps 1–2 — Datasets & Split Comparison")
st.caption("Step 1 (Students A/B/C): purpose of the training, validation and testing "
           "sets · Step 2: comparison of validation vs testing and the allocation view.")

@st.cache_data(show_spinner="Building split diagnostics…")
def _res():
    return build_all()

res = _res()

tab1, tab2 = st.tabs(["📚 Step 1 — Purpose of each set", "⚖️ Step 2 — Validation vs Testing"])

with tab1:
    st.markdown("### 🅰️ Student A — The training set (80%)")
    st.markdown(f"""
**Purpose.** The training set is the model's only source of knowledge. Every
learned quantity in this project is estimated **exclusively** on it: the
StandardScaler's means and variances, the HMM's initial distribution π, its
transition matrix A and Gaussian emissions B (via Baum-Welch), the tertile
boundaries that discretise the macro signals into Low/Medium/High, and the
Bayesian network's structure (Hill Climb + BDeu) and conditional probability
tables.

**Why it must dominate.** Learning algorithms need enough months to see each
regime repeat; with {len(res.splits['train'])} training months (~80%), every
state is well represented, and BIC/BDeu have data to price complexity against
fit. Anything the training set lacks the model can never learn.

**Anti-leakage rule.** No statistic computed on validation or test data may
influence any training-set choice, including how many states the HMM has.
""")
    st.markdown("### 🅱️ Student B — The validation set (10%)")
    st.markdown(f"""
**Purpose.** The validation set is the *model-selection* referee. It is the
first unseen data the tuned pipeline meets ({len(res.splits['val'])} months),
and it answers design questions the training set cannot: does the BDeu-scored
network generalise better than the BIC variant? Is a 3-state HMM a sensible
structure? Validation scores guide these choices **without** touching the test
set, which must stay sealed.

**Why it exists at all.** Selecting models on the training set rewards
memorisation; selecting on the test set burns the test set. The validation
slice absorbs that selection bias so the test set stays an honest final exam.
""")
    st.markdown("### 🅲 Student C — The testing set (10%)")
    st.markdown(f"""
**Purpose.** The test set is the *final exam*, used **exactly once**. It
provides the headline out-of-sample numbers reported to stakeholders:
accuracy ({res.test_acc:.1f}% here), per-class precision/recall, the confusion
matrix, and the trading simulation. No parameter, threshold or structure was
chosen with reference to these months.

**Why it must stay sealed.** The moment test results influence design, the
estimate becomes optimistic and the "final exam" reading is lost. This
one-shot discipline is what makes the comparison against Alvi (2018)'s
≈57% benchmark meaningful.
""")

with tab2:
    st.markdown("### The allocation")
    alloc = pd.DataFrame({
        "Split": ["Train", "Validation", "Test"],
        "Months": [len(res.splits["train"]), len(res.splits["val"]), len(res.splits["test"])],
        "Share": ["80%", "10%", "10%"],
        "From": [str(res.splits["train"][0].date()), str(res.splits["val"][0].date()),
                 str(res.splits["test"][0].date())],
        "To": [str(res.splits["train"][-1].date()), str(res.splits["val"][-1].date()),
               str(res.splits["test"][-1].date())],
        "Role": ["Fit scaler, HMM, tertiles, BN structure + CPDs",
                 "Model selection & tuning checks",
                 "One-shot final evaluation"],
    })
    st.dataframe(alloc, use_container_width=True, hide_index=True)
    fig = go.Figure(go.Pie(labels=alloc["Split"], values=alloc["Months"], hole=0.45,
                           marker=dict(colors=["#1F6FB2", "#E8842C", "#2E7D32"])))
    fig.update_layout(title="Chronological allocation — 80 / 10 / 10 (no shuffling)",
                      height=340)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Descriptive statistics by split (report Table 6)")
    st.dataframe(res.desc, use_container_width=True)
    st.caption("WTI level mean/std; log-return mean, skewness, kurtosis; mean "
               "3-month annualised volatility per split.")

    st.markdown("### Are validation and testing statistically alike?")
    c1, c2 = st.columns(2)
    c1.metric("Kolmogorov–Smirnov stat", f"{res.ks['stat']:.4f}", f"p = {res.ks['p']:.4f}")
    c2.metric("Mann–Whitney U", f"{res.mw['stat']:.1f}", f"p = {res.mw['p']:.4f}")
    st.markdown("""
**Reading.** KS compares the full return distributions of the two unseen
slices; Mann–Whitney compares their medians. With p-values around the 5–10%
band on ~22 vs ~23 monthly observations, we cannot rule out mild distributional
drift between the windows — exactly why the report stresses that validation
similarity to test is desirable but never guaranteed in financial series.
Consequence: test accuracy is the only number we treat as the headline.
""")
    st.markdown("### Autocorrelation diagnostics (allocation sanity)")
    fig = go.Figure()
    ci = 1.96 / np.sqrt(len(res.feature_df))
    fig.add_trace(go.Bar(x=res.acf["lags"], y=res.acf["ret"], name="ACF of returns",
                         marker_color="#1F6FB2"))
    fig.add_trace(go.Bar(x=res.acf["lags"], y=res.acf["sq"], name="ACF of squared returns",
                         marker_color="#C62828"))
    fig.add_hline(y=ci, line_dash="dot", line_color="#6B7280")
    fig.add_hline(y=-ci, line_dash="dot", line_color="#6B7280")
    fig.update_layout(title="ACF of monthly log-returns and squared returns "
                            "(volatility clustering signature)",
                      xaxis_title="Lag (months)", yaxis_title="ACF", height=380)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("### 🧭 Understand this page")
st.markdown("""
<div class='insight-num'><b>80/10/10 chronological split:</b> — time only flows forward in finance; shuffling would let the model peek at the future (leakage).</div>
<div class='insight-num'><b>Train kurtosis is the highest:</b> — the training window contains the 2008 and 2020 crashes; the heavy tails the model learns are real, not noise.</div>
<div class='insight-num'><b>KS / Mann–Whitney p-values:</b> — the formal check that validation and test are statistically comparable; borderline p-values on small samples are disclosed, not hidden.</div>
""", unsafe_allow_html=True)
st.markdown("<div class='insight-plain'>🟢 <b>For non-technical readers:</b><br>"
            "We split the calendar like a teacher splits a course: 80% is the "
            "textbook (training), 10% is the mock exam (validation), and the last "
            "10% is the real exam the model sees once. The mock exam and the real "
            "exam are checked to be of similar difficulty — fairly.</div>",
            unsafe_allow_html=True)
with st.expander("🔬 Technical detail"):
    st.caption("int(n*0.8)/int(n*0.1) chronological cut points; KS_2samp and "
               "Mann-Whitney U from scipy.stats on split log-returns; ACF computed "
               "manually with 1.96/sqrt(n) bands. Anti-leakage: scaler, HMM, tertiles "
               "and CPDs are fit on train only (see core/pipeline_gwp3.py).")
