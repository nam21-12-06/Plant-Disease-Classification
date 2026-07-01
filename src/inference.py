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

from config import CLASS_NAMES, MODEL_PATHS, preprocess_image_file

# Load model

def load_model(model_name: str) -> tf.keras.Model:
    path = MODEL_PATHS.get(model_name)
    if path is None:
        raise ValueError(f"Unknown model: '{model_name}'. Choose from: {list(MODEL_PATHS.keys())}")
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    print(f"Loading model: {path}")
    model = tf.keras.models.load_model(path)
    print("Model loaded.\n")
    return model



# Preprocess image


def preprocess_image(image_path: str) -> np.ndarray:
    return preprocess_image_file(image_path)


# Predict

def predict(model: tf.keras.Model, img: np.ndarray, top_k: int = 3) -> list[dict]:
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