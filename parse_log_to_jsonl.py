"""Parse CMD log output into JSONL records for compute_bi_cr_posthoc.py"""
import re, json, sys

def parse_log(text):
    pattern = re.compile(
        r'\[(\d+)/(\d+)\].*?\n'
        r'.*?Q: (.*?)\n'
        r'.*?GOLD: (.*?)\n'
        r'.*?PRED: (.*?)(?:\n|$)',
        re.MULTILINE
    )
    records = []
    for m in pattern.finditer(text):
        idx = int(m.group(1)) - 1
        question = m.group(3).strip()
        gold = m.group(4).strip()
        pred = m.group(5).strip()
        records.append({
            "idx": idx,
            "question": question,
            "gold": gold,
            "pred": pred,
        })
    return records

if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else "log_base_llama32.txt"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "openqa_eval_base_llama32_records.jsonl"

    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    records = parse_log(text)
    print(f"Parsed {len(records)} records")

    with open(output_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved: {output_file}")
