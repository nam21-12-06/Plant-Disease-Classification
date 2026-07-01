"""
gradcam.py
Grad-CAM visualization for Baseline CNN and MobileNetV2.
"""

import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

from config import IMG_SIZE


# ─────────────────────────────────────────────
# Config — last conv layer per model
# ─────────────────────────────────────────────

LAST_CONV_LAYER = {
    "baseline" : "conv2d_3",
    "mobilenet": "out_relu",
}


# ─────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────

def prepare_image(image: Image.Image, img_size: tuple = IMG_SIZE) -> tuple:
    image = image.convert("RGB").resize(img_size)
    original_rgb = np.array(image, dtype=np.uint8)
    img_array = np.expand_dims(original_rgb.astype(np.float32), axis=0)
    return img_array, original_rgb


# ─────────────────────────────────────────────
# Shared heatmap computation
# ─────────────────────────────────────────────

def _compute_heatmap(grad_model: tf.keras.Model, img_array: np.ndarray) -> tuple:
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)
        pred_idx = int(tf.argmax(preds[0]))
        confidence = float(preds[0][pred_idx])
        class_channel = preds[:, pred_idx]

    grads = tape.gradient(class_channel, conv_outputs)

    if grads is None:
        raise RuntimeError("Gradient computation returned None.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy(), pred_idx, confidence


# ─────────────────────────────────────────────
# Baseline CNN — rebuild Sequential as Functional graph
# ─────────────────────────────────────────────

def _gradcam_baseline(model: tf.keras.Model, img_array: np.ndarray) -> tuple:
    """
    The Sequential model's internal layers have no standalone .output
    attribute until run through a Functional graph. We manually replay
    the forward pass through tf.keras.Input to expose the intermediate
    conv output.
    """
    last_conv_layer = LAST_CONV_LAYER["baseline"]

    inputs = tf.keras.Input(shape=(256, 256, 3))
    x = inputs
    conv_output = None

    for layer in model.layers:
        x = layer(x)
        if layer.name == last_conv_layer:
            conv_output = x

    if conv_output is None:
        raise ValueError(f"Layer '{last_conv_layer}' not found in Baseline CNN.")

    grad_model = tf.keras.Model(inputs=inputs, outputs=[conv_output, x])

    return _compute_heatmap(grad_model, img_array)


# ─────────────────────────────────────────────
# MobileNetV2 — auto-detect nested backbone
# ─────────────────────────────────────────────

def _find_mobilenet_backbone(model: tf.keras.Model) -> tf.keras.Model:
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "mobilenet" in layer.name.lower():
            return layer
    raise ValueError(
        "Could not find MobileNetV2 backbone. "
        "Top-level layers: " + ", ".join(l.name for l in model.layers)
    )


def _find_gap_index(model: tf.keras.Model) -> int:
    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
            return i
    raise ValueError("No GlobalAveragePooling2D layer found in model.")


def _gradcam_mobilenet(model: tf.keras.Model, img_array: np.ndarray) -> tuple:
    backbone = _find_mobilenet_backbone(model)
    last_conv_output = backbone.get_layer(LAST_CONV_LAYER["mobilenet"]).output

    gap_index = _find_gap_index(model)

    # Rebuild classifier head (everything after backbone) as standalone model
    classifier_input = tf.keras.Input(shape=last_conv_output.shape[1:])
    x = classifier_input
    for layer in model.layers[gap_index:]:
        x = layer(x)
    classifier_model = tf.keras.Model(classifier_input, x)

    backbone_extractor = tf.keras.Model(backbone.input, last_conv_output)

    with tf.GradientTape() as tape:
        conv_outputs = backbone_extractor(img_array)
        tape.watch(conv_outputs)
        preds = classifier_model(conv_outputs)
        pred_idx = int(tf.argmax(preds[0]))
        confidence = float(preds[0][pred_idx])
        class_channel = preds[:, pred_idx]

    grads = tape.gradient(class_channel, conv_outputs)

    if grads is None:
        raise RuntimeError("Gradient computation returned None for MobileNetV2.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy(), pred_idx, confidence


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def generate_gradcam(model: tf.keras.Model, img_array: np.ndarray, model_type: str) -> tuple:
    if model_type == "baseline":
        return _gradcam_baseline(model, img_array)
    elif model_type == "mobilenet":
        return _gradcam_mobilenet(model, img_array)
    else:
        raise ValueError(f"Unknown model_type: '{model_type}'. Use 'baseline' or 'mobilenet'.")


# ─────────────────────────────────────────────
# Overlay
# ─────────────────────────────────────────────

def overlay_heatmap(original_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    heatmap_resized = cv2.resize(heatmap, (original_rgb.shape[1], original_rgb.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    result_rgb = cv2.addWeighted(original_rgb, 1 - alpha, heatmap_color_rgb, alpha, 0)
    return result_rgb