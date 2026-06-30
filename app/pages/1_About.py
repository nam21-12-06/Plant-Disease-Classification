import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import streamlit as st
from styles import inject_css, field_label, stat_block, field_divider


st.set_page_config(
    page_title="About · Plant Disease Classifier",
    page_icon="🌿",
    layout="wide",
)

inject_css()

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────

field_label("DOCUMENTATION")
st.markdown("# About this project")

st.markdown(
    """
    <p style="font-size:1.05rem; max-width:680px; opacity:0.85;">
    This page documents the dataset, methodology, and findings behind the
    classifier — including a limitation surfaced through Grad-CAM that
    complicates a simple "higher accuracy wins" reading of the results.
    </p>
    """,
    unsafe_allow_html=True,
)

field_divider()

# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

field_label("DATASET")
st.markdown("## New Plant Diseases Dataset")

st.markdown(
    """
    Leaf images of healthy and diseased plants collected under controlled
    imaging conditions, spanning 14 plant species. The dataset is pre-augmented
    and split into training and validation sets.
    """
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(stat_block("38", "Classes"), unsafe_allow_html=True)
with c2:
    st.markdown(stat_block("70,295", "Training images"), unsafe_allow_html=True)
with c3:
    st.markdown(stat_block("17,572", "Validation images"), unsafe_allow_html=True)
with c4:
    st.markdown(stat_block("256×256", "Image resolution"), unsafe_allow_html=True)

st.markdown(
    """
    <div class="field-card">
        <strong>Class balance.</strong> Largest class (Soybean — healthy, 2,022 images)
        to smallest (Corn — Cercospora leaf spot, 1,642 images) — a ratio of
        roughly 1.23, indicating low imbalance. No class weighting or
        resampling was required.
    </div>
    """,
    unsafe_allow_html=True,
)

field_divider()

# ─────────────────────────────────────────────
# Methodology
# ─────────────────────────────────────────────

field_label("METHODOLOGY")
st.markdown("## How the models were built")

tab1, tab2, tab3 = st.tabs(["Data Pipeline", "Baseline CNN", "MobileNetV2"])

with tab1:
    st.markdown(
        """
        Images were loaded with `image_dataset_from_directory`, one-hot
        encoded across 38 classes, and batched at size 32. Augmentation was
        intentionally light — random horizontal flip and ±3% rotation —
        since the dataset is already augmented and aggressive transforms
        risk distorting the lesion patterns that define each disease.
        """
    )
    st.markdown(
        """
        <div class="field-card">
        <code>train_ds → augment → batch(32) → prefetch(AUTOTUNE)</code>
        </div>
        """,
        unsafe_allow_html=True,
    )

with tab2:
    st.markdown(
        """
        Four convolutional blocks (Conv2D → BatchNorm → ReLU → MaxPool),
        followed by global average pooling and a dense classification head.
        Trained from scratch with Adam (lr=1e-3), early stopping, and
        learning rate reduction on plateau.
        """
    )
    st.markdown(stat_block("466,918", "Parameters"), unsafe_allow_html=True)

with tab3:
    st.markdown(
        """
        ImageNet-pretrained MobileNetV2 as a frozen feature extractor,
        trained in two phases: first the classification head alone
        (lr=1e-3), then the top 40 layers unfrozen and fine-tuned jointly
        with the head at a reduced learning rate (lr=1e-4).
        """
    )
    st.markdown(stat_block("2,596,710", "Parameters"), unsafe_allow_html=True)

field_divider()

# ─────────────────────────────────────────────
# Results table
# ─────────────────────────────────────────────

field_label("RESULTS")
st.markdown("## Model comparison")

st.markdown(
    """
    <table style="width:100%; border-collapse:collapse; font-size:0.92rem;">
        <thead>
            <tr style="border-bottom:2px solid #C9A876;">
                <th style="text-align:left; padding:0.6rem 0.4rem;">Model</th>
                <th style="text-align:right; padding:0.6rem 0.4rem;">Params</th>
                <th style="text-align:right; padding:0.6rem 0.4rem;">Val Accuracy</th>
                <th style="text-align:right; padding:0.6rem 0.4rem;">Val Loss</th>
                <th style="text-align:right; padding:0.6rem 0.4rem;">F1 (Macro)</th>
            </tr>
        </thead>
        <tbody>
            <tr style="border-bottom:1px solid #EDE6D6;">
                <td style="padding:0.6rem 0.4rem;">Baseline CNN</td>
                <td style="text-align:right; padding:0.6rem 0.4rem;">466,918</td>
                <td style="text-align:right; padding:0.6rem 0.4rem; color:#8B5E3C; font-weight:600;">99.78%</td>
                <td style="text-align:right; padding:0.6rem 0.4rem;">0.0075</td>
                <td style="text-align:right; padding:0.6rem 0.4rem;">99.78%</td>
            </tr>
            <tr>
                <td style="padding:0.6rem 0.4rem;">MobileNetV2</td>
                <td style="text-align:right; padding:0.6rem 0.4rem;">2,596,710</td>
                <td style="text-align:right; padding:0.6rem 0.4rem;">99.60%</td>
                <td style="text-align:right; padding:0.6rem 0.4rem;">0.0155</td>
                <td style="text-align:right; padding:0.6rem 0.4rem;">99.58%</td>
            </tr>
        </tbody>
    </table>
    """,
    unsafe_allow_html=True,
)

field_divider()

# ─────────────────────────────────────────────
# Grad-CAM finding — the nuance
# ─────────────────────────────────────────────

field_label("LIMITATION · GRAD-CAM FINDING")
st.markdown("## Accuracy isn't the whole story")

st.markdown(
    """
    <div class="field-card disease">
    On the validation set, Baseline CNN edges out MobileNetV2 on every
    headline metric. But Grad-CAM visualizations on real-world test images
    reveal a different picture: the Baseline CNN's attention frequently
    falls on <strong>background and leaf-edge artifacts</strong> rather than
    disease lesions, while MobileNetV2 — leveraging ImageNet-pretrained
    texture features — more consistently attends to the
    <strong>actual symptomatic regions</strong> on the leaf.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    This suggests the Baseline CNN's higher validation accuracy is partly
    inflated by the validation set sharing the same controlled imaging
    pipeline as training — a form of shortcut learning that validation
    accuracy alone does not expose. MobileNetV2's pretrained features may
    generalize better to leaf photos taken outside this dataset's
    conditions, despite its marginally lower benchmark score.
    """
)

field_divider()

st.markdown(
    """
    <p style="opacity:0.6; font-size:0.85rem;">
    Built as a portfolio project. Source code, training notebooks, and
    full evaluation reports are available in the project repository.
    </p>
    """,
    unsafe_allow_html=True,
)