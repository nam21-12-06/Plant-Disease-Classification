"""
Phase 7: Inference Pipeline
Predict plant disease from a leaf image.

Usage:
    python src/inference.py --image path/to/leaf.jpg --model baseline
    python src/inference.py --image path/to/leaf.jpg --model mobilenet
"""

import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path


# Config

IMG_SIZE = (256, 256)

MODEL_PATHS = {
    "baseline" : "models/baseline_cnn.keras",
    "mobilenet": "models/mobilenet_v2.keras",
}

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


# Load model

def load_model(model_name: str) -> tf.keras.Model:
    path = MODEL_PATHS.get(model_name)
    if path is None:
        raise ValueError(f"Unknown model: '{model_name}'. Choose from: {list(MODEL_PATHS.keys())}")
    if not Path(path).exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    print(f"Loading model: {path}")
    model = tf.keras.models.load_model(path)
    print("Model loaded.\n")
    return model



# Preprocess image


def preprocess_image(image_path: str) -> tf.Tensor:
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.expand_dims(img, axis=0)   # (1, 256, 256, 3)
    return img


# Predict

def predict(model: tf.keras.Model, img: tf.Tensor, top_k: int = 3) -> list[dict]:
    preds = model.predict(img, verbose=0)[0]   # (38,)

    top_indices = np.argsort(preds)[::-1][:top_k]
    results = [
        {
            "rank"      : i + 1,
            "class"     : CLASS_NAMES[idx],
            "confidence": float(preds[idx]) * 100,
        }
        for i, idx in enumerate(top_indices)
    ]
    return results


# Print results

def print_results(results: list[dict], model_name: str) -> None:
    print("=" * 55)
    print(f"  PREDICTION  ({model_name})")
    print("=" * 55)
    for r in results:
        bar = "█" * int(r["confidence"] / 2)
        print(f"  #{r['rank']}  {r['class']}")
        print(f"       {bar} {r['confidence']:.2f}%\n")
    print("=" * 55)


# Save result image

def save_result(image_path: str, results: list[dict], model_name: str) -> None:
    img = plt.imread(image_path)
    top = results[0]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    ax.axis("off")

    title = f"{top['class']}\n{top['confidence']:.2f}%"
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    # Top-3 annotation
    annotation = "\n".join(
        [f"#{r['rank']} {r['class']} ({r['confidence']:.2f}%)" for r in results]
    )
    fig.text(
        0.5, 0.01, annotation,
        ha="center", fontsize=8,
        color="gray", wrap=True
    )

    output_path = Path("figures") / f"inference_{Path(image_path).stem}_{model_name}.png"
    Path("figures").mkdir(exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"\nResult image saved: {output_path}")


# Main

def main():
    parser = argparse.ArgumentParser(description="Plant Disease Inference")
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to input leaf image"
    )
    parser.add_argument(
        "--model", type=str, default="baseline",
        choices=list(MODEL_PATHS.keys()),
        help="Model to use for inference (default: baseline)"
    )
    parser.add_argument(
        "--top_k", type=int, default=3,
        help="Number of top predictions to show (default: 3)"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save result image to figures/"
    )
    args = parser.parse_args()

    model   = load_model(args.model)
    img     = preprocess_image(args.image)
    results = predict(model, img, top_k=args.top_k)

    print_results(results, args.model)

    if args.save:
        save_result(args.image, results, args.model)


if __name__ == "__main__":
    main()