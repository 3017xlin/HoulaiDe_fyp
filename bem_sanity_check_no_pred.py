"""
BEM Sanity Check — works with CSV files that may lack Q/GOLD/PRED columns.
If the target CSV has no Q/GOLD, reads them from a reference CSV (e.g. llama31).

Usage:
  python bem_sanity_check_no_pred.py merged_base_llama32.csv merged_base_llama31.csv
  python bem_sanity_check_no_pred.py merged_base_llama31.csv
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


def find_col(row, candidates):
    for c in candidates:
        if c in row:
            return c
    return None


def main():
    target_file = sys.argv[1] if len(sys.argv) > 1 else "merged_base_llama32.csv"
    ref_file = sys.argv[2] if len(sys.argv) > 2 else None
    seed = 42

    log(f"Loading target: {target_file}")
    with open(target_file, "r", encoding="utf-8") as f:
        target_rows = list(csv.DictReader(f))
    n = len(target_rows)
    log(f"  {n} rows, cols: {list(target_rows[0].keys())}")

    # Try to find Q/GOLD in target file
    q_col = find_col(target_rows[0], ["Q", "question", "q"])
    gold_col = find_col(target_rows[0], ["GOLD", "gold", "gold_answer"])

    if q_col and gold_col:
        log(f"  Found Q='{q_col}', GOLD='{gold_col}' in target file")
        questions = [r[q_col] for r in target_rows]
        golds = [r[gold_col] for r in target_rows]
    elif ref_file:
        log(f"  Q/GOLD not in target file. Loading from reference: {ref_file}")
        with open(ref_file, "r", encoding="utf-8") as f:
            ref_rows = list(csv.DictReader(f))
        assert len(ref_rows) == n, f"Row count mismatch: target={n}, ref={len(ref_rows)}"
        rq = find_col(ref_rows[0], ["Q", "question", "q"])
        rg = find_col(ref_rows[0], ["GOLD", "gold", "gold_answer"])
        assert rq and rg, f"Reference file also missing Q/GOLD. Cols: {list(ref_rows[0].keys())}"
        questions = [r[rq] for r in ref_rows]
        golds = [r[rg] for r in ref_rows]
        log(f"  Got Q='{rq}', GOLD='{rg}' from reference ({n} rows)")
    else:
        print(f"ERROR: target file has no Q/GOLD columns and no reference file provided.")
        print(f"Usage: python {sys.argv[0]} target.csv reference_with_Q_GOLD.csv")
        sys.exit(1)

    original_bems = [float(target_rows[i].get("BEM", target_rows[i].get("bem", 0))) for i in range(n)]

    log("Loading BEM model...")
    bem_tok = AutoTokenizer.from_pretrained("kortukov/answer-equivalence-bem")
    bem_mdl = AutoModelForSequenceClassification.from_pretrained("kortukov/answer-equivalence-bem")
    bem_mdl.eval()
    log("BEM loaded")

    # ==========================================
    # TEST 1: Shuffled Gold Pairs
    # ==========================================
    log("=" * 70)
    log("TEST 1: Shuffled Gold Pairs")
    log("BEM(wrong_gold_as_candidate, correct_gold, question)")
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
    log("Positive: BEM(correct_gold, correct_gold, question)")
    log("Negative: BEM(wrong_gold, correct_gold, question)")
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

    auc = compute_auc(pos_scores, neg_scores)
    mean_pos = sum(pos_scores) / n
    mean_neg = sum(neg_scores) / n

    log("")
    log("TEST 2 RESULTS:")
    log(f"  Mean BEM (positive / gold=gold):   {mean_pos:.4f}")
    log(f"  Mean BEM (negative / wrong gold):  {mean_neg:.4f}")
    log(f"  Mean gap:                          {mean_pos - mean_neg:.4f}")
    log(f"  AUC:                               {auc:.4f}")
    log("")

    # ==========================================
    # Save
    # ==========================================
    out_file = target_file.replace(".csv", "_bem_sanity.txt")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f"BEM SANITY CHECK\nSource: {target_file}\nN = {n}\n\n")
        f.write(f"=== TEST 1: Shuffled Gold ===\n")
        f.write(f"BEM original: {avg_original:.6f}\nBEM shuffled: {avg_shuffled:.6f}\nGap: {avg_original - avg_shuffled:.6f}\n\n")
        f.write(f"=== TEST 2: Discrimination ===\n")
        f.write(f"Mean pos: {mean_pos:.6f}\nMean neg: {mean_neg:.6f}\nGap: {mean_pos - mean_neg:.6f}\nAUC: {auc:.6f}\n\n")
        f.write(f"=== Per-sample ===\n")
        for i in range(n):
            f.write(f"[{i+1}] BEM_orig={original_bems[i]:.4f} shuf={shuffled_scores[i]:.4f} pos={pos_scores[i]:.4f} neg={neg_scores[i]:.4f}\n")
    log(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
