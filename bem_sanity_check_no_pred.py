"""
BEM Sanity Check for merged CSV files WITHOUT a PRED column.
Uses only Q and GOLD columns. Generates synthetic pairs for testing.

Test 1 — Shuffled Gold: pair each Q with a WRONG gold from another Q,
  compute BEM(wrong_gold_as_pred, correct_gold, question).
  If BEM is still high, it proves topic-driven scoring.

Test 2 — Discrimination:
  Positive: BEM(correct_gold, correct_gold, question) — perfect match baseline
  Negative: BEM(wrong_gold, correct_gold, question) — mismatched answer
  AUC near 0.5 = BEM cannot distinguish correct from incorrect.

Usage:
  python bem_sanity_check_llama32.py merged_base_llama32.csv
"""

import csv
import json
import random
import sys
import time

import torch
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
    for score, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
            auc += tp
    return auc / (total_pos * total_neg)


def main():
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "merged_base_llama32.csv"
    seed = 42

    log(f"Loading data from {csv_file}")
    with open(csv_file, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    log(f"Loaded {n} samples")

    # Detect column names
    cols = list(rows[0].keys())
    q_col = "Q" if "Q" in cols else "question"
    gold_col = "GOLD" if "GOLD" in cols else "gold"
    log(f"Using columns: question='{q_col}', gold='{gold_col}'")

    log("Loading BEM model...")
    bem_tok = AutoTokenizer.from_pretrained("kortukov/answer-equivalence-bem")
    bem_mdl = AutoModelForSequenceClassification.from_pretrained("kortukov/answer-equivalence-bem")
    bem_mdl.eval()
    log("BEM loaded")

    questions = [r[q_col] for r in rows]
    golds = [r[gold_col] for r in rows]
    original_bems = [float(r.get("BEM", 0)) for r in rows]

    # ==========================================
    # TEST 1: Shuffled Gold Pairs
    # ==========================================
    log("=" * 70)
    log("TEST 1: Shuffled Gold Pairs")
    log("BEM(wrong_gold_as_candidate, correct_gold, question)")
    log("If BEM is high, scoring is driven by topic, not answer content.")
    log("=" * 70)

    rng = random.Random(seed)
    shuffled_idx = list(range(n))
    rng.shuffle(shuffled_idx)
    for i in range(n):
        if shuffled_idx[i] == i:
            swap = (i + 1) % n
            shuffled_idx[i], shuffled_idx[swap] = shuffled_idx[swap], shuffled_idx[i]

    shuffled_scores = []
    for i in range(n):
        wrong_gold = golds[shuffled_idx[i]]
        score = semsim_bem(bem_tok, bem_mdl, wrong_gold, golds[i], questions[i])
        shuffled_scores.append(score)
        if (i + 1) <= 5 or (i + 1) % 50 == 0 or (i + 1) == n:
            log(f"  [{i+1}/{n}] BEM_shuffled={score:.4f}")
            log(f"    Q: {questions[i][:70]}")
            log(f"    GOLD (correct):  {golds[i][:50]}")
            log(f"    GOLD (wrong):    {wrong_gold[:50]}")

    avg_shuffled = sum(shuffled_scores) / n
    avg_original = sum(original_bems) / n

    log("")
    log("TEST 1 RESULTS:")
    log(f"  BEM on original model predictions:       {avg_original:.4f}")
    log(f"  BEM on shuffled gold (wrong answers):    {avg_shuffled:.4f}")
    log(f"  Gap (original - shuffled):               {avg_original - avg_shuffled:.4f}")
    log("")

    # ==========================================
    # TEST 2: Discrimination Test
    # ==========================================
    log("=" * 70)
    log("TEST 2: Discrimination Test")
    log("Positive: BEM(correct_gold, correct_gold, question) — identity baseline")
    log("Negative: BEM(wrong_gold, correct_gold, question) — mismatched answer")
    log("=" * 70)

    pos_scores = []
    neg_scores = []
    for i in range(n):
        pos = semsim_bem(bem_tok, bem_mdl, golds[i], golds[i], questions[i])
        wrong_gold = golds[shuffled_idx[i]]
        neg = semsim_bem(bem_tok, bem_mdl, wrong_gold, golds[i], questions[i])
        pos_scores.append(pos)
        neg_scores.append(neg)

        if (i + 1) <= 3 or (i + 1) % 50 == 0 or (i + 1) == n:
            log(f"  [{i+1}/{n}] pos={pos:.4f} neg={neg:.4f} gap={pos-neg:.4f}")
            log(f"    Q: {questions[i][:70]}")
            log(f"    GOLD: {golds[i][:50]}")
            log(f"    WRONG: {wrong_gold[:50]}")

    auc = compute_auc(pos_scores, neg_scores)
    mean_pos = sum(pos_scores) / n
    mean_neg = sum(neg_scores) / n

    log("")
    log("TEST 2 RESULTS:")
    log(f"  Mean BEM (positive / gold=gold):   {mean_pos:.4f}")
    log(f"  Mean BEM (negative / wrong gold):  {mean_neg:.4f}")
    log(f"  Mean gap:                          {mean_pos - mean_neg:.4f}")
    log(f"  AUC:                               {auc:.4f}")
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
        f.write("BEM SANITY CHECK RESULTS (no PRED column version)\n")
        f.write(f"Source: {csv_file}\n")
        f.write(f"N = {n}\n\n")

        f.write("=== TEST 1: Shuffled Gold Pairs ===\n")
        f.write(f"BEM original (model predictions): {avg_original:.6f}\n")
        f.write(f"BEM shuffled (wrong gold):        {avg_shuffled:.6f}\n")
        f.write(f"Gap: {avg_original - avg_shuffled:.6f}\n\n")

        f.write("=== TEST 2: Discrimination (AUC) ===\n")
        f.write(f"Mean BEM positive (gold=gold): {mean_pos:.6f}\n")
        f.write(f"Mean BEM negative (wrong gold): {mean_neg:.6f}\n")
        f.write(f"Mean gap: {mean_pos - mean_neg:.6f}\n")
        f.write(f"AUC: {auc:.6f}\n\n")

        f.write("=== Per-sample scores ===\n")
        for i in range(n):
            f.write(f"[{i+1}] BEM_orig={original_bems[i]:.4f} "
                    f"BEM_shuffled={shuffled_scores[i]:.4f} "
                    f"BEM_pos={pos_scores[i]:.4f} "
                    f"BEM_neg={neg_scores[i]:.4f}\n")

    log(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
