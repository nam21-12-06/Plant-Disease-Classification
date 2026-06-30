"""
Phase 6: Model Comparison
Loads metrics from all trained models and generates comparison charts and summary.
 
Usage:
    python src/compare_models.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPORTS_DIR = "reports"
FIGURES_DIR = "figures"

METRICS_FILES = {
    "Baseline CNN": "reports/baseline_metrics.json",
    "MobileNetV2" : "reports/mobilenet_metrics.json",
}

# Load metrics
def load_metrics(metrics_files: dict) -> pd.DataFrame:
    records = []
    for name, path in metrics_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Metrics file not found: {path}")
        with open(path) as f:
            data = json.load(f)
        records.append(data)
    return pd.DataFrame(records)

# Print comparison table

def print_comparison_table(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)
 
    compare_df = pd.DataFrame({
        "Model"             : df["model_name"],
        "Parameters"        : df["params"].apply(lambda x: f"{x:,}"),
        "Best Epoch"        : df["best_epoch"],
        "Train Acc"         : df["train_accuracy"].apply(lambda x: f"{x*100:.2f}%"),
        "Val Acc"           : df["val_accuracy"].apply(lambda x: f"{x*100:.2f}%"),
        "Val Loss"          : df["val_loss"].apply(lambda x: f"{x:.4f}"),
        "F1 (Macro)"        : df["f1_macro"].apply(lambda x: f"{x*100:.2f}%"),
    }).set_index("Model")
 
    print(compare_df.to_string())
    print("=" * 70 + "\n")

# Plot: Accuracy & F1
def plot_accuracy_f1(df: pd.DataFrame, output_dir: str) -> None:
    models  = df["model_name"].tolist()
    val_acc = (df["val_accuracy"] * 100).tolist()
    f1      = (df["f1_macro"] * 100).tolist()

    x     = np.arange(len(models))
    width = 0.35
    colors = ["steelblue", "coral"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, val_acc, width, label="Val Accuracy", color=colors)
    bars2 = ax.bar(x + width / 2, f1,      width, label="F1-Score (Macro)",
                   color=[c + "99" for c in ["#4682B4", "#FF7F50"]])

    ax.set_ylabel("Score (%)")
    ax.set_title("Model Comparison — Accuracy & F1-Score")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(99, 100.2)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in list(bars1) + list(bars2):
        ax.annotate(
            f"{bar.get_height():.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3), textcoords="offset points",
            ha="center", fontsize=9
        )

    plt.tight_layout()
    path = os.path.join(output_dir, "comparison_accuracy_f1.png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: {path}")

# Plot: Parameter Count
def plot_params(df: pd.DataFrame, output_dir: str) -> None:
    models = df["model_name"].tolist()
    params = df["params"].tolist()

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(models, params, color=["steelblue", "coral"])

    ax.set_ylabel("Number of Parameters")
    ax.set_title("Model Comparison — Parameter Count")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        ax.annotate(
            f"{bar.get_height():,}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3), textcoords="offset points",
            ha="center", fontsize=9
        )

    plt.tight_layout()
    path = os.path.join(output_dir, "comparison_params.png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: {path}")

# Plot: Validation Loss

def plot_val_loss(df: pd.DataFrame, output_dir: str) -> None:
    models   = df["model_name"].tolist()
    val_loss = df["val_loss"].tolist()

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(models, val_loss, color=["steelblue", "coral"])

    ax.set_ylabel("Validation Loss")
    ax.set_title("Model Comparison — Validation Loss")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        ax.annotate(
            f"{bar.get_height():.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3), textcoords="offset points",
            ha="center", fontsize=9
        )

    plt.tight_layout()
    path = os.path.join(output_dir, "comparison_val_loss.png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: {path}")

# Save summary JSON
def save_summary(df: pd.DataFrame, output_dir: str) -> None:
    path = os.path.join(output_dir, "comparison_summary.json")
    df.to_json(path, orient="records", indent=4)
    print(f"Saved: {path}")


# Main
def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("Loading metrics...")
    df = load_metrics(METRICS_FILES)

    print_comparison_table(df)

    print("Generating charts...")
    plot_accuracy_f1(df, FIGURES_DIR)
    plot_params(df, FIGURES_DIR)
    plot_val_loss(df, FIGURES_DIR)

    save_summary(df, REPORTS_DIR)

if __name__ == "__main__":
    main()
