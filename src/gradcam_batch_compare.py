"""
gradcam_batch_compare.py
Run Grad-CAM on all test images for both models and save a comparison grid.

Usage:
    python src/gradcam_batch_compare.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

from gradcam import prepare_image, generate_gradcam, overlay_heatmap


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

TEST_IMAGES_DIR = "data/test_images"
OUTPUT_DIR = "figures/gradcam_comparison"

MODEL_PATHS = {
    "baseline" : "models/baseline_cnn.keras",
    "mobilenet": "models/mobilenetv2.keras",
}


# ─────────────────────────────────────────────
# Load models once
# ─────────────────────────────────────────────

def load_models() -> dict:
    print("Loading models...")
    models = {
        "baseline" : tf.keras.models.load_model(MODEL_PATHS["baseline"]),
        "mobilenet": tf.keras.models.load_model(MODEL_PATHS["mobilenet"]),
    }
    print("Models loaded.\n")
    return models


# ─────────────────────────────────────────────
# Process a single image: original + both Grad-CAMs
# ─────────────────────────────────────────────

def process_image(image_path: Path, models: dict) -> dict:
    image = Image.open(image_path)
    img_array, original_rgb = prepare_image(image)

    result = {"filename": image_path.name, "original": original_rgb}

    for model_type in ["baseline", "mobilenet"]:
        try:
            heatmap, pred_idx, conf = generate_gradcam(
                models[model_type], img_array, model_type=model_type
            )
            overlay = overlay_heatmap(original_rgb, heatmap)
            result[model_type] = {
                "overlay"   : overlay,
                "pred_idx"  : pred_idx,
                "confidence": conf,
            }
        except Exception as e:
            print(f"  [!] Grad-CAM failed for {model_type} on {image_path.name}: {e}")
            result[model_type] = None

    return result


# ─────────────────────────────────────────────
# Save individual 3-panel comparison (original | baseline | mobilenet)
# ─────────────────────────────────────────────

def save_single_comparison(result: dict, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(result["original"])
    axes[0].set_title("Original")
    axes[0].axis("off")

    for ax, model_type, label in zip(axes[1:], ["baseline", "mobilenet"], ["Baseline CNN", "MobileNetV2"]):
        r = result[model_type]
        if r is not None:
            ax.imshow(r["overlay"])
            ax.set_title(f"{label}\nconf: {r['confidence']*100:.1f}%")
        else:
            ax.text(0.5, 0.5, "Failed", ha="center", va="center")
            ax.set_title(label)
        ax.axis("off")

    plt.tight_layout()
    out_path = output_dir / f"compare_{Path(result['filename']).stem}.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────
# Save big grid: all images, 3 columns (original | baseline | mobilenet)
# ─────────────────────────────────────────────

def save_grid(results: list, output_dir: Path) -> None:
    n = len(results)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))

    if n == 1:
        axes = axes.reshape(1, 3)

    for row, result in enumerate(results):
        axes[row, 0].imshow(result["original"])
        axes[row, 0].set_ylabel(result["filename"], fontsize=8, rotation=0, ha="right", va="center")
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

        for col, model_type in enumerate(["baseline", "mobilenet"], start=1):
            r = result[model_type]
            if r is not None:
                axes[row, col].imshow(r["overlay"])
            axes[row, col].axis("off")

        if row == 0:
            axes[row, 0].set_title("Original", fontsize=10)
            axes[row, 1].set_title("Baseline CNN", fontsize=10)
            axes[row, 2].set_title("MobileNetV2", fontsize=10)

    plt.tight_layout()
    out_path = output_dir / "gradcam_full_grid.png"
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Saved full grid: {out_path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    test_dir = Path(TEST_IMAGES_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        list(test_dir.glob("*.jpg")) +
        list(test_dir.glob("*.jpeg")) +
        list(test_dir.glob("*.JPG")) +
        list(test_dir.glob("*.png"))
    )

    if not image_paths:
        print(f"No images found in {test_dir}")
        return

    print(f"Found {len(image_paths)} images.\n")

    models = load_models()

    results = []
    for i, image_path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] Processing {image_path.name}...")
        result = process_image(image_path, models)
        results.append(result)
        save_single_comparison(result, output_dir)

    print("\nGenerating full grid...")
    save_grid(results, output_dir)

    print(f"\nDone. {len(results)} comparisons saved to {output_dir}/")


if __name__ == "__main__":
    main()