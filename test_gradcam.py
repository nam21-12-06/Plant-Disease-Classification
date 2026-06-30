"""
test.py
Quick diagnostic script to inspect model architecture
before debugging Grad-CAM layer names.

Usage:
    python test.py
"""

import sys
from pathlib import Path

import tensorflow as tf

# Adjust this if running from a different location
MODELS_DIR = Path(__file__).parent / "models"


def inspect_model(model_path: str, label: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {label}  —  {model_path}")
    print("=" * 70)

    if not Path(model_path).exists():
        print(f"  [!] File not found: {model_path}")
        return

    model = tf.keras.models.load_model(model_path)

    print(f"\nModel type      : {type(model).__name__}")
    print(f"Model built     : {model.built}")
    print(f"Input shape     : {model.input_shape if hasattr(model, 'input_shape') else 'N/A'}")

    print("\n--- Top-level layers ---")
    for i, layer in enumerate(model.layers):
        print(f"  [{i:>2}] {layer.name:<35} ({layer.__class__.__name__})")

    # If there's a nested model (e.g. MobileNetV2 backbone), inspect it too
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            print(f"\n--- Nested model: '{layer.name}' layers (first 10 + last 10) ---")
            nested_layers = layer.layers
            if len(nested_layers) > 20:
                shown = nested_layers[:10] + ["..."] + nested_layers[-10:]
            else:
                shown = nested_layers

            for item in shown:
                if item == "...":
                    print("  ...")
                else:
                    print(f"      {item.name:<35} ({item.__class__.__name__})")

    # Try to identify likely "last conv layer" candidates
    print("\n--- Conv2D / DepthwiseConv2D layers found (top-level) ---")
    conv_layers = [
        layer.name for layer in model.layers
        if "conv" in layer.__class__.__name__.lower()
    ]
    if conv_layers:
        for name in conv_layers:
            print(f"  - {name}")
    else:
        print("  (none found at top level — check nested model above)")

    print()


def main():
    inspect_model(MODELS_DIR / "baseline_cnn.keras", "Baseline CNN")
    inspect_model(MODELS_DIR / "mobilenetv2.keras", "MobileNetV2")


if __name__ == "__main__":
    main()