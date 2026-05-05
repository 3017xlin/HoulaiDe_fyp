"""
Post-hoc computation of BiA, BiQ, CrQ from saved evaluation records.
Reads the per-example JSONL file and adds the three missing metrics.
"""

import json
import sys
from collections import Counter
from sentence_transformers import SentenceTransformer, CrossEncoder, util


def normalize_text(s):
    import re
    return re.sub(r"\s+", " ", str(s).strip().lower())


def main():
    input_file = sys.argv[1] if len(sys.argv) > 1 else "openqa_eval_base_llama32_records.jsonl"
    output_file = input_file.replace("_records.jsonl", "_records_with_bi_cr.jsonl")

    print(f"Reading: {input_file}")
    records = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records)} records")

    print("Loading bi-encoder (all-MiniLM-L6-v2)...")
    st_model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Loading cross-encoder (stsb-roberta-large)...")
    cross_model = CrossEncoder("cross-encoder/stsb-roberta-large")

    bia_list, biq_list, crq_list = [], [], []

    for i, r in enumerate(records):
        pred = str(r.get("pred", "")).strip()
        gold = str(r.get("gold", "")).strip()
        question = str(r.get("question", "")).strip()

        # BiA: bi-encoder, answer only
        if not pred or not gold:
            bia = 0.0
        else:
            emb = st_model.encode([pred, gold], convert_to_tensor=True)
            bia = float(util.cos_sim(emb[0], emb[1]).item())

        # BiQ: bi-encoder, question + answer
        if not pred or not gold:
            biq = 0.0
        else:
            emb = st_model.encode([f"{question} {pred}", f"{question} {gold}"], convert_to_tensor=True)
            biq = float(util.cos_sim(emb[0], emb[1]).item())

        # CrQ: cross-encoder, question + answer
        if not pred or not gold:
            crq = 0.0
        else:
            score = cross_model.predict([(f"{question} {pred}", f"{question} {gold}")])[0]
            crq = float(max(0.0, min(score / 5.0, 1.0)))

        bia_list.append(bia)
        biq_list.append(biq)
        crq_list.append(crq)

        r["bia"] = round(bia, 4)
        r["biq"] = round(biq, 4)
        r["crq"] = round(crq, 4)

        print(f"  [{i+1}/{len(records)}] BiA={bia:.3f} BiQ={biq:.3f} CrQ={crq:.3f}")
        print(f"    Q: {question[:70]}")
        print(f"    GOLD: {gold[:70]}")
        print(f"    PRED: {pred[:70]}")
        print("-" * 80)

    n = len(records)
    avg_bia = sum(bia_list) / n
    avg_biq = sum(biq_list) / n
    avg_crq = sum(crq_list) / n

    print("=" * 60)
    print(f"AGGREGATE (n={n}):")
    print(f"  BiA (bi-encoder, answer only)    : {avg_bia:.4f}")
    print(f"  BiQ (bi-encoder, + question)     : {avg_biq:.4f}")
    print(f"  CrQ (cross-encoder, + question)  : {avg_crq:.4f}")
    print("=" * 60)

    with open(output_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved: {output_file}")

    summary_file = input_file.replace("_records.jsonl", "_bi_cr_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(f"source = {input_file}\n")
        f.write(f"n = {n}\n")
        f.write(f"bia = {avg_bia:.6f}\n")
        f.write(f"biq = {avg_biq:.6f}\n")
        f.write(f"crq = {avg_crq:.6f}\n")
    print(f"Saved: {summary_file}")


if __name__ == "__main__":
    main()
