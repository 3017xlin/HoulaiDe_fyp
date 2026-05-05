"""Merge CMD log metrics (EM/F1/CrA/BEM/PA/LJ) with post-hoc BiA/BiQ/CrQ into one CSV."""
import re, csv, sys


def parse_main_log(text):
    pattern = re.compile(
        r'\[(\d+)/\d+\]\s+EM=(\d+)\s+F1=([0-9.]+)\s+CrA=([0-9.]+)\s+BEM=([0-9.]+)\s+PA=([0-9.]+)\s+LJ=([0-9.]+)\((\w+)\).*?\n'
        r'.*?Q: (.*?)\n'
        r'.*?GOLD: (.*?)\n'
        r'.*?PRED: (.*?)(?:\n|$)',
        re.MULTILINE
    )
    records = {}
    for m in pattern.finditer(text):
        idx = int(m.group(1))
        records[idx] = {
            "idx": idx,
            "Q": m.group(9).strip(),
            "GOLD": m.group(10).strip(),
            "PRED": m.group(11).strip(),
            "EM": int(m.group(2)),
            "F1": float(m.group(3)),
            "CrA": float(m.group(4)),
            "BEM": float(m.group(5)),
            "PA": float(m.group(6)),
            "LJ": float(m.group(7)),
            "LJ_label": m.group(8),
        }
    return records


def parse_posthoc_log(text):
    pattern = re.compile(
        r'\[(\d+)/\d+\]\s+BiA=([0-9.]+)\s+BiQ=([0-9.]+)\s+CrQ=([0-9.]+)',
    )
    records = {}
    for m in pattern.finditer(text):
        idx = int(m.group(1))
        records[idx] = {
            "BiA": float(m.group(2)),
            "BiQ": float(m.group(3)),
            "CrQ": float(m.group(4)),
        }
    return records


def main():
    main_log_file = sys.argv[1] if len(sys.argv) > 1 else "log_main.txt"
    posthoc_log_file = sys.argv[2] if len(sys.argv) > 2 else "log_posthoc.txt"
    output_csv = sys.argv[3] if len(sys.argv) > 3 else "merged_base_llama32.csv"

    with open(main_log_file, "r", encoding="utf-8") as f:
        main_records = parse_main_log(f.read())

    with open(posthoc_log_file, "r", encoding="utf-8") as f:
        posthoc_records = parse_posthoc_log(f.read())

    fieldnames = ["idx", "Q", "GOLD", "PRED", "EM", "F1", "BiA", "BiQ", "CrA", "CrQ", "BEM", "PA", "LJ", "LJ_label"]

    rows = []
    for idx in sorted(main_records.keys()):
        row = main_records[idx]
        ph = posthoc_records.get(idx, {"BiA": 0.0, "BiQ": 0.0, "CrQ": 0.0})
        row["BiA"] = float(ph["BiA"]) if ph["BiA"] != "" else 0.0
        row["BiQ"] = float(ph["BiQ"]) if ph["BiQ"] != "" else 0.0
        row["CrQ"] = float(ph["CrQ"]) if ph["CrQ"] != "" else 0.0
        rows.append(row)

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Print aggregate
    n = len(rows)
    print(f"Merged {n} records → {output_csv}")
    print(f"  EM   = {sum(r['EM'] for r in rows)/n:.4f}")
    print(f"  F1   = {sum(r['F1'] for r in rows)/n:.4f}")
    print(f"  BiA  = {sum(r['BiA'] for r in rows)/n:.4f}")
    print(f"  BiQ  = {sum(r['BiQ'] for r in rows)/n:.4f}")
    print(f"  CrA  = {sum(r['CrA'] for r in rows)/n:.4f}")
    print(f"  CrQ  = {sum(r['CrQ'] for r in rows)/n:.4f}")
    print(f"  BEM  = {sum(r['BEM'] for r in rows)/n:.4f}")
    print(f"  PA   = {sum(r['PA'] for r in rows)/n:.4f}")
    print(f"  LJ   = {sum(r['LJ'] for r in rows)/n:.4f}")


if __name__ == "__main__":
    main()
