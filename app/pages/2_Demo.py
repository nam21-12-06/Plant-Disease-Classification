import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

import streamlit as st
import numpy as np
from PIL import Image

from styles import inject_css, field_label, field_divider
from predictor import load_selected_model, predict_image
from src.config import CLASS_NAMES
from gradcam import prepare_image, generate_gradcam, overlay_heatmap




st.set_page_config(
    page_title="Demo · Plant Disease Classifier",
    page_icon="🌿",
    layout="wide",
)

inject_css()

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────

field_label("LIVE DEMO")
st.markdown("# Identify a leaf")
st.markdown(
    """
    <p style="opacity:0.8; max-width:600px;">
    Upload a photo of a single leaf. The model returns its top prediction
    across 38 disease classes, along with a Grad-CAM overlay showing which
    regions of the leaf influenced the decision.
    </p>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# Sidebar controls
# ─────────────────────────────────────────────

st.sidebar.markdown("### Model")
model_option = st.sidebar.selectbox(
    "Choose model",
    ["Baseline CNN", "MobileNetV2"],
    label_visibility="collapsed",
)

show_gradcam = st.sidebar.checkbox("Show Grad-CAM", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <p style="font-size:0.75rem; opacity:0.75; line-height:1.5;">
    <strong>Baseline CNN</strong><br>466K params · 99.78% val acc<br><br>
    <strong>MobileNetV2</strong><br>2.6M params · 99.60% val acc
    </p>
    """,
    unsafe_allow_html=True,
)

model_type = "baseline" if model_option == "Baseline CNN" else "mobilenet"

model = load_selected_model(model_type)

field_divider()

# ─────────────────────────────────────────────
# Upload
# ─────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file:
    image = Image.open(uploaded_file)

    idx, conf, probs = predict_image(model, image)
    predicted_class = CLASS_NAMES[idx]
    is_healthy = "healthy" in predicted_class.lower()
    label = predicted_class.replace("___", " — ").replace("_", " ")

    # ─────────────────────────────────────
    # Image + Grad-CAM
    # ─────────────────────────────────────

    if show_gradcam:
        col_img, col_cam = st.columns(2)

        with col_img:
            field_label("INPUT")
            st.image(image, use_container_width=True)

        with col_cam:
            field_label("MODEL ATTENTION")
            with st.spinner("Generating Grad-CAM..."):
                try:
                    img_array, original_rgb = prepare_image(image)
                    heatmap, gradcam_idx, gradcam_conf = generate_gradcam(
                        model, img_array, model_type=model_type
                    )
                    result = overlay_heatmap(original_rgb, heatmap)
                    st.image(result, use_container_width=True)
                except Exception as e:
                    st.error(f"Grad-CAM generation failed: {e}")
    else:
        field_label("INPUT")
        st.image(image, width=420)

    field_divider()

    # ─────────────────────────────────────
    # Prediction card
    # ─────────────────────────────────────

    card_class = "healthy" if is_healthy else "disease"
    status_word = "Healthy" if is_healthy else "Disease detected"

    st.markdown(
        f"""
        <div class="field-card {card_class}">
            <p style="font-family:'JetBrains Mono', monospace; font-size:0.72rem;
                      text-transform:uppercase; letter-spacing:0.08em; opacity:0.7; margin-bottom:0.3rem;">
                {status_word}
            </p>
            <h2 style="margin:0 0 0.6rem 0;">{label}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_conf, col_model = st.columns(2)
    with col_conf:
        st.metric("Confidence", f"{conf * 100:.2f}%")
    with col_model:
        st.metric("Model used", model_option)

    st.progress(float(conf))

    field_divider()

    # ─────────────────────────────────────
    # Top 5
    # ─────────────────────────────────────

    field_label("TOP 5 PREDICTIONS")
    top5_idx = probs.argsort()[-5:][::-1]

    for i in top5_idx:
        label_i = CLASS_NAMES[i].replace("___", " — ").replace("_", " ")
        col_name, col_bar = st.columns([2, 3])
        with col_name:
            st.write(label_i)
        with col_bar:
            st.progress(float(probs[i]))
            st.caption(f"{probs[i] * 100:.2f}%")

else:
    st.markdown(
        """
        <div class="field-card">
        Upload a leaf image above to get started.
        </div>
        """,
        unsafe_allow_html=True,
    )