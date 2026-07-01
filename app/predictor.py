import urllib.request
from pathlib import Path

import numpy as np
import tensorflow as tf
import streamlit as st

from src.config import MODEL_PATHS, MODEL_URLS, preprocess_pil_image

def _ensure_model_downloaded(model_name: str) -> Path:
    model_path = MODEL_PATHS.get(model_name)
    if model_path is None:
        raise ValueError(f"Unknown model: '{model_name}'. Choose from: {list(MODEL_PATHS.keys())}")

    if model_path.exists():
        return model_path

    url = MODEL_URLS.get(model_name)
    if url is None:
        raise FileNotFoundError(f"No download URL configured for {model_name}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with st.spinner(f"Downloading model weights ({model_path.name})..."):
        urllib.request.urlretrieve(url, model_path)

    return model_path


@st.cache_resource(show_spinner=False)
def load_selected_model(model_name: str):
    model_path = _ensure_model_downloaded(model_name)
    return tf.keras.models.load_model(model_path)


def predict_image(model, image):
    img_array = preprocess_pil_image(image)
    preds = model.predict(img_array, verbose=0)

    probs = preds[0]

    predicted_idx = np.argmax(probs)

    confidence = probs[predicted_idx]

    return predicted_idx, confidence, probs