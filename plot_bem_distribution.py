"""
BEM Score Distribution Histogram for two models (strongest vs weakest).
Generates a publication-ready figure showing BEM score distributions.

Usage:
  python plot_bem_distribution.py merged_base_qwen.csv merged_base_llama31.csv
"""

import csv
import sys
import matplotlib.pyplot as plt
import numpy as np


def load_bem_scores(csv_file):
    with open(csv_file, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    scores = [float(r["BEM"]) for r in rows]
    return scores


def main():
    file_qwen = sys.argv[1] if len(sys.argv) > 1 else "merged_base_qwen.csv"
    file_llama = sys.argv[2] if len(sys.argv) > 2 else "merged_base_llama31.csv"

    scores_qwen = load_bem_scores(file_qwen)
    scores_llama = load_bem_scores(file_llama)

    # Stats
    for label, scores in [("Qwen2.5-1.5B", scores_qwen), ("LLaMA-3.1-8B", scores_llama)]:
        n = len(scores)
        pct_above_09 = sum(1 for s in scores if s >= 0.9) / n * 100
        pct_above_08 = sum(1 for s in scores if s >= 0.8) / n * 100
        pct_below_03 = sum(1 for s in scores if s < 0.3) / n * 100
        print(f"{label} (n={n}):")
        print(f"  Mean: {np.mean(scores):.4f}  Median: {np.median(scores):.4f}  Std: {np.std(scores):.4f}")
        print(f"  >= 0.9: {pct_above_09:.1f}%   >= 0.8: {pct_above_08:.1f}%   < 0.3: {pct_below_03:.1f}%")
        print()

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    bins = np.arange(0, 1.05, 0.05)

    # Qwen (weakest)
    ax = axes[0]
    counts_q, _, patches_q = ax.hist(scores_qwen, bins=bins, color="#e74c3c", alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_title("Qwen2.5-1.5B (Base → Open QA)", fontsize=13, fontweight="bold")
    ax.set_xlabel("BEM Score", fontsize=11)
    ax.set_ylabel("Number of Samples", fontsize=11)
    ax.axvline(x=0.9, color="black", linestyle="--", linewidth=1, label="0.9 threshold")
    pct_q = sum(1 for s in scores_qwen if s >= 0.9) / len(scores_qwen) * 100
    ax.text(0.92, max(counts_q) * 0.85, f"{pct_q:.0f}%\n≥0.9", fontsize=10, fontweight="bold", color="#c0392b")
    mean_q = np.mean(scores_qwen)
    ax.axvline(x=mean_q, color="#2c3e50", linestyle="-", linewidth=1.5, alpha=0.7)
    ax.text(mean_q + 0.02, max(counts_q) * 0.95, f"mean={mean_q:.2f}", fontsize=9, color="#2c3e50")
    ax.set_xlim(0, 1.05)
    ax.legend(fontsize=9)

    # LLaMA 8B (strongest)
    ax = axes[1]
    counts_l, _, patches_l = ax.hist(scores_llama, bins=bins, color="#3498db", alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_title("LLaMA-3.1-8B (Base → Open QA)", fontsize=13, fontweight="bold")
    ax.set_xlabel("BEM Score", fontsize=11)
    ax.axvline(x=0.9, color="black", linestyle="--", linewidth=1, label="0.9 threshold")
    pct_l = sum(1 for s in scores_llama if s >= 0.9) / len(scores_llama) * 100
    ax.text(0.92, max(counts_l) * 0.85, f"{pct_l:.0f}%\n≥0.9", fontsize=10, fontweight="bold", color="#2980b9")
    mean_l = np.mean(scores_llama)
    ax.axvline(x=mean_l, color="#2c3e50", linestyle="-", linewidth=1.5, alpha=0.7)
    ax.text(mean_l + 0.02, max(counts_l) * 0.95, f"mean={mean_l:.2f}", fontsize=9, color="#2c3e50")
    ax.set_xlim(0, 1.05)
    ax.legend(fontsize=9)

    fig.suptitle("BEM Score Distribution on Open QA (247 samples)\n"
                 "High concentration near 1.0 indicates discriminative collapse",
                 fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()
    out_png = "bem_score_distribution.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_png}")

    out_pdf = "bem_score_distribution.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
