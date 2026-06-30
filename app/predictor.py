import os
import urllib.request
import tensorflow as tf
import streamlit as st

MODEL_URLS = {
    "models/baseline_cnn.keras": "https://github.com/nam21-12-06/Plant-Diease/releases/download/v1.0/baseline_cnn.keras",
    "models/mobilenetv2.keras"  : "https://github.com/nam21-12-06/Plant-Diease/releases/download/v1.0/mobilenetv2.keras",
}

def _ensure_model_downloaded(model_path: str) -> str:
    if os.path.exists(model_path):
        return model_path

    url = MODEL_URLS.get(model_path)
    if url is None:
        raise FileNotFoundError(f"No download URL configured for {model_path}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with st.spinner(f"Downloading model weights ({os.path.basename(model_path)})..."):
        urllib.request.urlretrieve(url, model_path)

    return model_path


@st.cache_resource(show_spinner=False)
def load_selected_model(model_path: str):
    model_path = _ensure_model_downloaded(model_path)
    return tf.keras.models.load_model(model_path)
def predict_image(model, image):
    
    image = image.resize(IMG_SIZE)

    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)

    probs = preds[0]

    predicted_idx = np.argmax(probs)

    confidence = probs[predicted_idx]

    return predicted_idx, confidence, probs