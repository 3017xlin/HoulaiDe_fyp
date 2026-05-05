"""
BEM Sanity Check: shuffled pairs + discrimination test.

1. Shuffle Test: randomly pair each question with a WRONG gold answer
   from the same dataset, compute BEM. If BEM is still high (~0.7-0.9),
   it proves BEM scores are driven by topic relevance, not answer equivalence.

2. Discrimination Test: for each sample, compute BEM on the real pred
   (positive) and on a random pred from another question (negative).
   Report AUC and mean gap. AUC near 0.5 = BEM has no discriminative power.

Usage:
  python bem_sanity_check.py merged_base_llama31.csv
"""

import csv
import json
import random
import sys
import time
from collections import Counter

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def semsim_bem(bem_tok, bem_mdl, pred, gold, question):
    pred, gold = str(pred).strip(), str(gold).strip()
    if not pred and not gold: return 1.0
    if not pred or not gold: return 0.0
    text = f"[CLS] {pred} [SEP]"
    text_pair = f"{gold} [SEP] {question} [SEP]"
    inputs = bem_tok(text=text, text_pair=text_pair, add_special_tokens=False,
                     padding="max_length", truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = bem_mdl(**inputs).logits
    return float(torch.nn.functional.softmax(logits, dim=-1)[0, 1].item())


def compute_auc(pos_scores, neg_scores):
    labels = [1] * len(pos_scores) + [0] * len(neg_scores)
    scores = pos_scores + neg_scores
    pairs = sorted(zip(scores, labels), reverse=True)
    tp, fp, auc = 0, 0, 0.0
    total_pos = sum(labels)
    total_neg = len(labels) - total_pos
    if total_pos == 0 or total_neg == 0:
        return 0.5
    prev_fp = 0
    for score, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
            auc += tp
    return auc / (total_pos * total_neg)


def main():
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "merged_base_llama31.csv"
    seed = 42

    log(f"Loading data from {csv_file}")
    with open(csv_file, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    log(f"Loaded {n} samples")

    log("Loading BEM model...")
    bem_tok = AutoTokenizer.from_pretrained("kortukov/answer-equivalence-bem")
    bem_mdl = AutoModelForSequenceClassification.from_pretrained("kortukov/answer-equivalence-bem")
    bem_mdl.eval()
    log("BEM loaded")

    questions = [r["Q"] for r in rows]
    golds = [r["GOLD"] for r in rows]
    preds = [r["PRED"] for r in rows]
    original_bems = [float(r.get("BEM", 0)) for r in rows]

    # ==========================================
    # TEST 1: Shuffled Gold Pairs
    # ==========================================
    log("=" * 70)
    log("TEST 1: Shuffled Gold Pairs")
    log("Each question is paired with a WRONG gold answer from another question.")
    log("=" * 70)

    rng = random.Random(seed)
    shuffled_golds = golds.copy()
    rng.shuffle(shuffled_golds)
    # Ensure no gold matches its original position
    for i in range(n):
        if shuffled_golds[i] == golds[i]:
            swap_idx = (i + 1) % n
            shuffled_golds[i], shuffled_golds[swap_idx] = shuffled_golds[swap_idx], shuffled_golds[i]

    shuffled_scores = []
    for i in range(n):
        score = semsim_bem(bem_tok, bem_mdl, preds[i], shuffled_golds[i], questions[i])
        shuffled_scores.append(score)
        if (i + 1) <= 5 or (i + 1) % 50 == 0 or (i + 1) == n:
            log(f"  [{i+1}/{n}] BEM_shuffled={score:.4f}")
            log(f"    Q: {questions[i][:70]}")
            log(f"    GOLD (wrong): {shuffled_golds[i][:70]}")
            log(f"    PRED: {preds[i][:70]}")

    avg_shuffled = sum(shuffled_scores) / n
    avg_original = sum(original_bems) / n

    log("")
    log("TEST 1 RESULTS:")
    log(f"  BEM on original pairs (pred vs correct gold): {avg_original:.4f}")
    log(f"  BEM on shuffled pairs (pred vs WRONG gold):   {avg_shuffled:.4f}")
    log(f"  Gap (original - shuffled):                    {avg_original - avg_shuffled:.4f}")
    log(f"  If gap is small, BEM is driven by topic, not answer content.")
    log("")

    # ==========================================
    # TEST 2: Discrimination Test (AUC)
    # ==========================================
    log("=" * 70)
    log("TEST 2: Discrimination Test")
    log("Positive: BEM(pred, correct_gold, question)")
    log("Negative: BEM(random_pred_from_another_q, correct_gold, question)")
    log("=" * 70)

    rng2 = random.Random(seed + 1)
    shuffled_preds = preds.copy()
    rng2.shuffle(shuffled_preds)
    for i in range(n):
        if shuffled_preds[i] == preds[i]:
            swap_idx = (i + 1) % n
            shuffled_preds[i], shuffled_preds[swap_idx] = shuffled_preds[swap_idx], shuffled_preds[i]

    pos_scores = []
    neg_scores = []
    for i in range(n):
        pos = semsim_bem(bem_tok, bem_mdl, preds[i], golds[i], questions[i])
        neg = semsim_bem(bem_tok, bem_mdl, shuffled_preds[i], golds[i], questions[i])
        pos_scores.append(pos)
        neg_scores.append(neg)

        if (i + 1) <= 3 or (i + 1) % 50 == 0 or (i + 1) == n:
            log(f"  [{i+1}/{n}] pos_BEM={pos:.4f} neg_BEM={neg:.4f} gap={pos-neg:.4f}")

    auc = compute_auc(pos_scores, neg_scores)
    mean_pos = sum(pos_scores) / n
    mean_neg = sum(neg_scores) / n

    log("")
    log("TEST 2 RESULTS:")
    log(f"  Mean BEM (positive / correct pred):  {mean_pos:.4f}")
    log(f"  Mean BEM (negative / random pred):   {mean_neg:.4f}")
    log(f"  Mean gap:                            {mean_pos - mean_neg:.4f}")
    log(f"  AUC:                                 {auc:.4f}")
    log(f"  AUC interpretation:")
    log(f"    1.0  = perfect discrimination")
    log(f"    0.5  = random (no discrimination)")
    log(f"    <0.6 = BEM effectively useless for this task")
    log("")

    # ==========================================
    # Save results
    # ==========================================
    out_file = csv_file.replace(".csv", "_bem_sanity.txt")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("BEM SANITY CHECK RESULTS\n")
        f.write(f"Source: {csv_file}\n")
        f.write(f"N = {n}\n\n")

        f.write("=== TEST 1: Shuffled Gold Pairs ===\n")
        f.write(f"BEM original (pred vs correct gold): {avg_original:.6f}\n")
        f.write(f"BEM shuffled (pred vs WRONG gold):   {avg_shuffled:.6f}\n")
        f.write(f"Gap: {avg_original - avg_shuffled:.6f}\n\n")

        f.write("=== TEST 2: Discrimination (AUC) ===\n")
        f.write(f"Mean BEM positive: {mean_pos:.6f}\n")
        f.write(f"Mean BEM negative: {mean_neg:.6f}\n")
        f.write(f"Mean gap: {mean_pos - mean_neg:.6f}\n")
        f.write(f"AUC: {auc:.6f}\n\n")

        f.write("=== Per-sample shuffled BEM scores ===\n")
        for i in range(n):
            f.write(f"[{i+1}] BEM_original={original_bems[i]:.4f} BEM_shuffled={shuffled_scores[i]:.4f} "
                    f"BEM_pos={pos_scores[i]:.4f} BEM_neg={neg_scores[i]:.4f}\n")

    log(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
