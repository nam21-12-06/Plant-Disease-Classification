import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import streamlit as st
from styles import inject_css, field_label, stat_block, field_divider


st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="wide",
)

inject_css()

# ─────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────

field_label("FIELD JOURNAL · COMPUTER VISION")

st.markdown(
    """
    <h1 style="font-size:3.2rem; line-height:1.1; border-bottom:none; margin-bottom:0.4rem;">
        Reading the leaf<br>before the harvest fails
    </h1>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <p style="font-size:1.1rem; color:#2D3A2E; max-width:640px; opacity:0.85; margin-top:0.6rem;">
    A deep learning system trained to recognize 38 plant disease conditions
    from a single leaf photograph — built to compare a lightweight custom CNN
    against ImageNet-pretrained transfer learning, with Grad-CAM transparency
    into what each model actually looks at.
    </p>
    """,
    unsafe_allow_html=True,
)

col_cta, _ = st.columns([1, 3])
with col_cta:
    if st.button("Try the demo →", use_container_width=True):
        st.switch_page("pages/2_Demo.py")

field_divider()

# ─────────────────────────────────────────────
# Key stats row
# ─────────────────────────────────────────────

field_label("AT A GLANCE")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(stat_block("38", "Disease classes"), unsafe_allow_html=True)
with c2:
    st.markdown(stat_block("70,295", "Training images"), unsafe_allow_html=True)
with c3:
    st.markdown(stat_block("99.78%", "Best val. accuracy"), unsafe_allow_html=True)
with c4:
    st.markdown(stat_block("2", "Models compared"), unsafe_allow_html=True)

field_divider()

# ─────────────────────────────────────────────
# Two model cards
# ─────────────────────────────────────────────

field_label("THE TWO APPROACHES")
st.markdown("### Lightweight vs. pretrained")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown(
        """
        <div class="field-card healthy">
            <h3 style="margin-top:0;">Baseline CNN</h3>
            <p style="font-family:'JetBrains Mono', monospace; font-size:0.8rem; opacity:0.7; margin-bottom:0.8rem;">
                466,918 params · trained from scratch
            </p>
            <p style="margin-bottom:0;">
                A compact four-block convolutional network, designed and trained
                entirely on this dataset with no external pretraining. Reached
                99.78% validation accuracy — fewer parameters, faster inference.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_b:
    st.markdown(
        """
        <div class="field-card">
            <h3 style="margin-top:0;">MobileNetV2</h3>
            <p style="font-family:'JetBrains Mono', monospace; font-size:0.8rem; opacity:0.7; margin-bottom:0.8rem;">
                2,596,710 params · ImageNet pretrained
            </p>
            <p style="margin-bottom:0;">
                Fine-tuned in two phases atop ImageNet weights. Slightly lower
                validation accuracy (99.60%), but Grad-CAM shows it attends more
                consistently to actual disease lesions rather than background.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

field_divider()

# ─────────────────────────────────────────────
# Footer nav hint
# ─────────────────────────────────────────────

st.markdown(
    """
    <p style="opacity:0.6; font-size:0.85rem;">
    Use the sidebar to explore the dataset and methodology in
    <strong>About</strong>, or jump straight into the
    <strong>Demo</strong> to upload a leaf and see predictions with
    Grad-CAM visual explanations.
    </p>
    """,
    unsafe_allow_html=True,
)