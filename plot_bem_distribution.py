"""
BEM Score Distribution Histogram — flexible version.
Accepts 1 or more CSV files as arguments, plots side-by-side.

Usage:
  python plot_bem_distribution.py merged_base_qwen.csv
  python plot_bem_distribution.py merged_base_qwen.csv merged_base_llama31.csv
  python plot_bem_distribution.py merged_base_qwen.csv merged_mc_phi35.csv merged_openqa_llama31.csv
"""

import csv
import sys
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_bem_scores(csv_file):
    with open(csv_file, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    col = "BEM" if "BEM" in rows[0] else "bem"
    return [float(r[col]) for r in rows]


def label_from_filename(fname):
    base = os.path.basename(fname).replace(".csv", "").replace("merged_", "")
    return base


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_bem_distribution.py file1.csv [file2.csv ...]")
        sys.exit(1)

    files = sys.argv[1:]
    n_plots = len(files)

    all_data = []
    for f in files:
        scores = load_bem_scores(f)
        label = label_from_filename(f)
        all_data.append((label, scores))

    # Print stats
    for label, scores in all_data:
        n = len(scores)
        pct_09 = sum(1 for s in scores if s >= 0.9) / n * 100
        pct_08 = sum(1 for s in scores if s >= 0.8) / n * 100
        pct_03 = sum(1 for s in scores if s < 0.3) / n * 100
        print(f"{label} (n={n}):")
        print(f"  Mean={np.mean(scores):.4f}  Median={np.median(scores):.4f}  Std={np.std(scores):.4f}")
        print(f"  >=0.9: {pct_09:.1f}%  >=0.8: {pct_08:.1f}%  <0.3: {pct_03:.1f}%")
        print()

    # Plot
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c"]
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5), sharey=True, squeeze=False)
    axes = axes[0]
    bins = np.arange(0, 1.05, 0.05)

    for i, (label, scores) in enumerate(all_data):
        ax = axes[i]
        color = colors[i % len(colors)]
        counts, _, _ = ax.hist(scores, bins=bins, color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel("BEM Score", fontsize=11)
        if i == 0:
            ax.set_ylabel("Number of Samples", fontsize=11)
        ax.axvline(x=0.9, color="black", linestyle="--", linewidth=1)
        pct = sum(1 for s in scores if s >= 0.9) / len(scores) * 100
        ax.text(0.92, max(counts) * 0.85, f"{pct:.0f}%\n≥0.9", fontsize=10, fontweight="bold", color=color)
        mean_val = np.mean(scores)
        ax.axvline(x=mean_val, color="#2c3e50", linestyle="-", linewidth=1.5, alpha=0.7)
        ax.text(max(0.01, mean_val - 0.15), max(counts) * 0.95, f"μ={mean_val:.2f}", fontsize=9, color="#2c3e50")
        ax.set_xlim(0, 1.05)

    fig.suptitle("BEM Score Distribution", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    suffix = "_".join(label_from_filename(f) for f in files)
    out_png = f"bem_distribution_{suffix}.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_png}")

    out_pdf = out_png.replace(".png", ".pdf")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
