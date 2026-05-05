import json
import re
import time
import math
import random
import csv
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter, defaultdict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer, util


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# =========================
# Config
# =========================
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
RAW_OPENQA_PATH = "../combined_QA_generation2.json"
OUTPUT_DIR = "sft_qwen2p5_1p5b_openqa_qlora"   # 这里会自动找最新 checkpoint

EVAL_RATIO = 0.2
SPLIT_SEED = 42

USE_CHAT_TEMPLATE = True

# self-consistency generation config
N_SAMPLES_PER_QUESTION = 5
MAX_NEW_TOKENS = 32
TEMPERATURE = 0.7
TOP_P = 0.9
TOP_K = 50

# semantic correctness threshold for binary accuracy / calibration
SEMANTIC_CORRECT_THRESHOLD = 0.80

# output files
PER_SAMPLE_CSV = "semantic_self_consistency_per_sample.csv"
SUMMARY_TXT = "semantic_self_consistency_summary.txt"


# =========================
# Utils
# =========================
def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def normalize_text(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def f1_word(pred: str, gold: str) -> float:
    pred_tokens = normalize_text(pred).split()
    gold_tokens = normalize_text(gold).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)
    overlap = sum((pred_counter & gold_counter).values())

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def semantic_similarity_score(st_model, a: str, b: str) -> float:
    a = str(a).strip()
    b = str(b).strip()

    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    emb = st_model.encode([a, b], convert_to_tensor=True)
    return float(util.cos_sim(emb[0], emb[1]).item())


def pearson_corr(x: List[float], y: List[float]) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)
    if np.std(x_arr) == 0 or np.std(y_arr) == 0:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def rankdata_average(a: List[float]) -> np.ndarray:
    """
    scipy 不一定装了，这里自己做 average rank
    """
    arr = np.asarray(a)
    sorter = np.argsort(arr)
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(len(arr))
    arr_sorted = arr[sorter]

    ranks = np.zeros(len(arr), dtype=float)
    i = 0
    while i < len(arr_sorted):
        j = i
        while j + 1 < len(arr_sorted) and arr_sorted[j + 1] == arr_sorted[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[i:j+1] = avg_rank
        i = j + 1

    return ranks[inv]


def spearman_corr(x: List[float], y: List[float]) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    rx = rankdata_average(x)
    ry = rankdata_average(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


# =========================
# Parsing data
# =========================
def try_json_loads(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None


def parse_json_object_stream(text: str) -> List[dict]:
    objs = []
    decoder = json.JSONDecoder()
    idx = 0
    n = len(text)

    while idx < n:
        while idx < n and text[idx].isspace():
            idx += 1
        if idx >= n:
            break
        try:
            obj, end = decoder.raw_decode(text, idx)
            idx = end

            if isinstance(obj, str):
                inner = try_json_loads(obj)
                if isinstance(inner, dict):
                    obj = inner

            if isinstance(obj, dict):
                objs.append(obj)
            elif isinstance(obj, list):
                for x in obj:
                    if isinstance(x, dict):
                        objs.append(x)
        except Exception:
            idx += 1

    return objs


def load_openqa_file(path: str) -> List[dict]:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Empty file: {path}")

    obj = try_json_loads(text)
    if obj is not None:
        if isinstance(obj, dict):
            return [obj]
        if isinstance(obj, list):
            out = []
            for x in obj:
                if isinstance(x, dict):
                    out.append(x)
                elif isinstance(x, str):
                    inner = try_json_loads(x)
                    if isinstance(inner, dict):
                        out.append(inner)
            if out:
                return out

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    per_line = []
    ok = True
    for ln in lines:
        item = try_json_loads(ln)
        if item is None:
            ok = False
            break
        if isinstance(item, dict):
            per_line.append(item)
        elif isinstance(item, str):
            inner = try_json_loads(item)
            if isinstance(inner, dict):
                per_line.append(inner)
            else:
                ok = False
                break
        else:
            ok = False
            break

    if ok and per_line:
        return per_line

    objs = parse_json_object_stream(text)
    if objs:
        return objs

    raise ValueError(f"Could not parse file: {path}")


SCENARIO_RE = re.compile(r"Scenario:\s*(.*?)\nQuestion:", re.DOTALL | re.IGNORECASE)
QUESTION_RE = re.compile(r"Question:\s*(.*?)(?:\n\n|\Z)", re.DOTALL | re.IGNORECASE)


def extract_scenario_from_prompt(prompt: str) -> str:
    m = SCENARIO_RE.search(prompt)
    return m.group(1).strip() if m else ""


def extract_question_from_prompt(prompt: str) -> str:
    m = QUESTION_RE.search(prompt)
    return m.group(1).strip() if m else ""


def convert_records_to_prompt_answer(records: List[dict]) -> List[dict]:
    out = []

    for rec in records:
        if not isinstance(rec, dict):
            continue

        # format: {"prompt": "...", "answer": "..."}
        if "prompt" in rec and "answer" in rec:
            prompt = str(rec["prompt"]).strip()
            answer = str(rec["answer"]).strip()
            out.append({
                "scenario": extract_scenario_from_prompt(prompt),
                "question": extract_question_from_prompt(prompt),
                "prompt": prompt,
                "answer": answer,
            })
            continue

        # format: {"scenario": "...", "qa_pairs": [...]}
        if "scenario" in rec and "qa_pairs" in rec:
            scenario = str(rec["scenario"]).strip()
            qa_pairs = rec.get("qa_pairs", [])
            if not scenario or not isinstance(qa_pairs, list):
                continue

            for qa in qa_pairs:
                question = str(qa.get("Question", "")).strip()
                reason = str(qa.get("Reason", "")).strip()
                answer = str(qa.get("Answer", "")).strip()

                if not question or not answer:
                    continue

                prompt = (
                    f"Scenario: {scenario}\n"
                    f"Question: {question}\n\n"
                    "Please reason briefly, then give the final answer in this format:\n"
                    "<reason>...</reason>\n"
                    "<answer>...</answer>"
                )
                target = f"<reason>{reason}</reason>\n<answer>{answer}</answer>"

                out.append({
                    "scenario": scenario,
                    "question": question,
                    "prompt": prompt,
                    "answer": target,
                })

    return out


def split_by_scenario(examples: List[dict], eval_ratio: float, seed: int) -> Tuple[List[dict], List[dict]]:
    grouped = defaultdict(list)
    for ex in examples:
        grouped[ex["scenario"]].append(ex)

    scenarios = list(grouped.keys())
    rng = random.Random(seed)
    rng.shuffle(scenarios)

    n_eval = max(1, int(len(scenarios) * eval_ratio))
    eval_scenarios = set(scenarios[:n_eval])

    train_examples, eval_examples = [], []
    for scen, rows in grouped.items():
        if scen in eval_scenarios:
            eval_examples.extend(rows)
        else:
            train_examples.extend(rows)

    return train_examples, eval_examples


# =========================
# Prompt + extraction
# =========================
def build_user_prompt_constrained(ex: Dict) -> str:
    return (
        f"Scenario: {ex['scenario']}\n"
        f"Question: {ex['question']}\n\n"
        "You MUST follow the format exactly.\n"
        "ONLY output:\n"
        "<answer>...</answer>\n"
        "Do not include explanation.\n"
        "Do not include extra text."
    )


def build_chat_prompt(tokenizer, user_prompt: str) -> str:
    if not USE_CHAT_TEMPLATE:
        return user_prompt

    messages = [
        {"role": "system", "content": "You are a careful reasoning assistant."},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def normalize_extracted(text: str) -> str:
    text = str(text).strip()
    if not text:
        return ""

    text = re.sub(r"</?answer>", "", text, flags=re.I)
    text = re.sub(r"</?reason>", "", text, flags=re.I)
    text = re.sub(r"^(final answer|answer)\s*:\s*", "", text, flags=re.I)
    text = re.sub(r"^the correct answer is\s*", "", text, flags=re.I)
    text = re.split(r"\b(?:reason|explanation)\b\s*[:：]?", text, maxsplit=1, flags=re.I)[0]
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return ""

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[0] if lines else text


def extract_answer_robust(text: str) -> str:
    text = str(text).strip()
    if not text:
        return ""

    matches = re.findall(r"<answer>(.*?)</answer>", text, flags=re.I | re.S)
    if matches:
        ans = normalize_extracted(matches[-1])
        if ans:
            return ans

    m = re.search(r"<answer>\s*(.*)", text, flags=re.I | re.S)
    if m:
        ans = normalize_extracted(m.group(1))
        if ans:
            return ans

    m = re.search(r"(?:^|\n)\s*answer\s*:\s*(.*)", text, flags=re.I)
    if m:
        ans = normalize_extracted(m.group(1))
        if ans:
            return ans

    m = re.search(r"the correct answer is\s*(.*)", text, flags=re.I)
    if m:
        ans = normalize_extracted(m.group(1))
        if ans:
            return ans

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        ans = normalize_extracted(lines[-1])
        if ans:
            return ans

    return normalize_extracted(text)


# =========================
# Checkpoint
# =========================
def find_latest_checkpoint(output_dir: str):
    out = Path(output_dir)
    if not out.exists():
        return output_dir

    ckpts = []
    for p in out.iterdir():
        if p.is_dir():
            m = re.match(r"checkpoint-(\d+)$", p.name)
            if m:
                ckpts.append((int(m.group(1)), str(p)))

    if not ckpts:
        return output_dir

    ckpts.sort(key=lambda x: x[0])
    return ckpts[-1][1]


# =========================
# Self-consistency analysis
# =========================
def pairwise_mean_cosine(st_model, answers: List[str]) -> float:
    valid_answers = [a.strip() for a in answers if a.strip()]
    n = len(valid_answers)

    if n == 0:
        return 0.0
    if n == 1:
        return 1.0

    embs = st_model.encode(valid_answers, convert_to_tensor=True)
    sim_matrix = util.cos_sim(embs, embs).cpu().numpy()

    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            vals.append(float(sim_matrix[i, j]))

    return float(sum(vals) / len(vals)) if vals else 1.0


def select_consensus_answer(st_model, answers: List[str]) -> str:
    valid_answers = [a.strip() for a in answers if a.strip()]
    if not valid_answers:
        return ""
    if len(valid_answers) == 1:
        return valid_answers[0]

    embs = st_model.encode(valid_answers, convert_to_tensor=True)
    sim_matrix = util.cos_sim(embs, embs).cpu().numpy()

    avg_sims = []
    for i in range(len(valid_answers)):
        avg = (sim_matrix[i].sum() - 1.0) / (len(valid_answers) - 1)
        avg_sims.append(float(avg))

    best_idx = int(np.argmax(avg_sims))
    return valid_answers[best_idx]


def build_calibration_bins(confidences: List[float], correctness: List[int], n_bins: int = 10):
    rows = []
    if not confidences:
        return rows

    confs = np.array(confidences, dtype=float)
    cors = np.array(correctness, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    for i in range(n_bins):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]

        if i < n_bins - 1:
            mask = (confs >= lo) & (confs < hi)
        else:
            mask = (confs >= lo) & (confs <= hi)

        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue

        bin_conf = float(confs[idx].mean())
        bin_acc = float(cors[idx].mean())
        rows.append({
            "bin_left": float(lo),
            "bin_right": float(hi),
            "count": int(len(idx)),
            "mean_confidence": bin_conf,
            "empirical_accuracy": bin_acc,
            "gap": abs(bin_conf - bin_acc),
        })

    return rows


def expected_calibration_error(bin_rows: List[dict], total_n: int) -> float:
    if total_n == 0 or not bin_rows:
        return 0.0
    ece = 0.0
    for row in bin_rows:
        ece += (row["count"] / total_n) * row["gap"]
    return float(ece)


@torch.no_grad()
def run_self_consistency_analysis(model, tokenizer, eval_examples: List[Dict], st_model):
    device = next(model.parameters()).device
    model.eval()
    model.config.use_cache = True

    per_sample_rows = []

    for idx, ex in enumerate(eval_examples):
        prompt = build_chat_prompt(tokenizer, build_user_prompt_constrained(ex))
        gold = extract_answer_robust(ex["answer"])

        sampled_answers = []

        for _ in range(N_SAMPLES_PER_QUESTION):
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
            )

            gen_ids = out[0][inputs["input_ids"].shape[1]:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            pred = extract_answer_robust(gen_text)
            sampled_answers.append(pred)

            del inputs, out, gen_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        consistency = pairwise_mean_cosine(st_model, sampled_answers)
        consensus_answer = select_consensus_answer(st_model, sampled_answers)

        semantic_correctness = semantic_similarity_score(st_model, consensus_answer, gold)
        token_f1 = f1_word(consensus_answer, gold)
        exact_match = int(normalize_text(consensus_answer) == normalize_text(gold))
        semantic_correct_binary = int(semantic_correctness >= SEMANTIC_CORRECT_THRESHOLD)

        row = {
            "sample_id": idx,
            "scenario": ex["scenario"],
            "question": ex["question"],
            "gold_answer": gold,
            "consensus_answer": consensus_answer,
            "sampled_answers": sampled_answers,
            "consistency": consistency,
            "semantic_correctness": semantic_correctness,
            "semantic_correct_binary": semantic_correct_binary,
            "token_f1": token_f1,
            "exact_match": exact_match,
        }
        per_sample_rows.append(row)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(eval_examples):
            log(f"progress: {idx+1}/{len(eval_examples)}")

    return per_sample_rows


def summarize_results(per_sample_rows: List[dict]) -> Dict[str, float]:
    consistencies = [r["consistency"] for r in per_sample_rows]
    sem_correct = [r["semantic_correctness"] for r in per_sample_rows]
    sem_binary = [r["semantic_correct_binary"] for r in per_sample_rows]
    token_f1s = [r["token_f1"] for r in per_sample_rows]
    ems = [r["exact_match"] for r in per_sample_rows]

    pearson = pearson_corr(consistencies, sem_correct)
    spearman = spearman_corr(consistencies, sem_correct)

    bin_rows = build_calibration_bins(consistencies, sem_binary, n_bins=10)
    ece = expected_calibration_error(bin_rows, len(per_sample_rows))

    summary = {
        "n_eval": len(per_sample_rows),
        "mean_consistency": float(np.mean(consistencies)) if consistencies else 0.0,
        "mean_semantic_correctness": float(np.mean(sem_correct)) if sem_correct else 0.0,
        "mean_semantic_accuracy": float(np.mean(sem_binary)) if sem_binary else 0.0,
        "mean_token_f1": float(np.mean(token_f1s)) if token_f1s else 0.0,
        "mean_exact_match": float(np.mean(ems)) if ems else 0.0,
        "pearson_consistency_vs_semantic_correctness": pearson,
        "spearman_consistency_vs_semantic_correctness": spearman,
        "ece_semantic_accuracy": ece,
    }

    return summary, bin_rows


def save_per_sample_csv(rows: List[dict], path: str):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id",
            "scenario",
            "question",
            "gold_answer",
            "consensus_answer",
            "consistency",
            "semantic_correctness",
            "semantic_correct_binary",
            "token_f1",
            "exact_match",
            "sampled_answers_json",
        ])
        for r in rows:
            writer.writerow([
                r["sample_id"],
                r["scenario"],
                r["question"],
                r["gold_answer"],
                r["consensus_answer"],
                f"{r['consistency']:.6f}",
                f"{r['semantic_correctness']:.6f}",
                r["semantic_correct_binary"],
                f"{r['token_f1']:.6f}",
                r["exact_match"],
                json.dumps(r["sampled_answers"], ensure_ascii=False),
            ])


def save_summary_txt(summary: dict, bin_rows: List[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write("[SEMANTIC SELF-CONSISTENCY / CALIBRATION SUMMARY]\n")
        f.write(f"N = {summary['n_eval']}\n")
        f.write(f"Mean Consistency = {summary['mean_consistency']:.4f}\n")
        f.write(f"Mean Semantic Correctness = {summary['mean_semantic_correctness']:.4f}\n")
        f.write(f"Mean Semantic Accuracy (threshold={SEMANTIC_CORRECT_THRESHOLD:.2f}) = {summary['mean_semantic_accuracy']:.4f}\n")
        f.write(f"Mean Token F1 = {summary['mean_token_f1']:.4f}\n")
        f.write(f"Mean Exact Match = {summary['mean_exact_match']:.4f}\n")
        f.write(f"Pearson(consistency, semantic_correctness) = {summary['pearson_consistency_vs_semantic_correctness']:.4f}\n")
        f.write(f"Spearman(consistency, semantic_correctness) = {summary['spearman_consistency_vs_semantic_correctness']:.4f}\n")
        f.write(f"ECE (semantic accuracy) = {summary['ece_semantic_accuracy']:.4f}\n\n")

        f.write("[Calibration bins]\n")
        f.write("bin_left,bin_right,count,mean_confidence,empirical_accuracy,gap\n")
        for row in bin_rows:
            f.write(
                f"{row['bin_left']:.2f},"
                f"{row['bin_right']:.2f},"
                f"{row['count']},"
                f"{row['mean_confidence']:.4f},"
                f"{row['empirical_accuracy']:.4f},"
                f"{row['gap']:.4f}\n"
            )


def main():
    log("loading raw Open QA data...")
    records = load_openqa_file(RAW_OPENQA_PATH)
    examples = convert_records_to_prompt_answer(records)
    _, eval_examples = split_by_scenario(examples, EVAL_RATIO, SPLIT_SEED)
    log(f"eval examples: {len(eval_examples)}")

    log("loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log("loading base model...")
    use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if use_bf16 else (torch.float16 if torch.cuda.is_available() else torch.float32),
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )

    ckpt_path = find_latest_checkpoint(OUTPUT_DIR)
    log(f"loading LoRA checkpoint from: {ckpt_path}")
    model = PeftModel.from_pretrained(model, ckpt_path)

    log("loading sentence-transformer...")
    st_model = SentenceTransformer("all-MiniLM-L6-v2")

    log("running semantic self-consistency / calibration analysis...")
    per_sample_rows = run_self_consistency_analysis(model, tokenizer, eval_examples, st_model)

    summary, bin_rows = summarize_results(per_sample_rows)

    log("[FINAL SUMMARY]")
    log(f"N = {summary['n_eval']}")
    log(f"Mean Consistency = {summary['mean_consistency']:.4f}")
    log(f"Mean Semantic Correctness = {summary['mean_semantic_correctness']:.4f}")
    log(f"Mean Semantic Accuracy = {summary['mean_semantic_accuracy']:.4f}")
    log(f"Mean Token F1 = {summary['mean_token_f1']:.4f}")
    log(f"Mean Exact Match = {summary['mean_exact_match']:.4f}")
    log(f"Pearson(consistency, semantic_correctness) = {summary['pearson_consistency_vs_semantic_correctness']:.4f}")
    log(f"Spearman(consistency, semantic_correctness) = {summary['spearman_consistency_vs_semantic_correctness']:.4f}")
    log(f"ECE = {summary['ece_semantic_accuracy']:.4f}")

    save_per_sample_csv(per_sample_rows, PER_SAMPLE_CSV)
    save_summary_txt(summary, bin_rows, SUMMARY_TXT)

    log(f"saved per-sample csv -> {PER_SAMPLE_CSV}")
    log(f"saved summary txt   -> {SUMMARY_TXT}")


if __name__ == "__main__":
    main()