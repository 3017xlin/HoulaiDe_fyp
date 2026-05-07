"""
Calibration Analysis via Self-Consistency for all four models.

For each question, samples K outputs with do_sample=True, then computes:
  - Consistency: avg LLM-Judge similarity among sampled outputs
  - Consensus answer: the sample with highest avg similarity to others
  - Correctness: LLM-Judge similarity between consensus answer and gold

Reports: Pearson r, Spearman rho, ECE (Expected Calibration Error).

Usage:
  python calibration_analysis.py --model qwen --task mc
  python calibration_analysis.py --model llama31 --task mc
  python calibration_analysis.py --model phi35 --task openqa
  python calibration_analysis.py --model llama32 --task base
"""

# DynamicCache shims
import transformers
try:
    from transformers.cache_utils import DynamicCache
    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(lambda self: self.get_seq_length() if hasattr(self, "get_seq_length") else 0)
    if not hasattr(DynamicCache, "get_max_length"):
        DynamicCache.get_max_length = lambda self: None
    if not hasattr(DynamicCache, "get_max_cache_shape"):
        DynamicCache.get_max_cache_shape = lambda self: None
    if not hasattr(DynamicCache, "get_usable_length"):
        DynamicCache.get_usable_length = lambda self, *a, **kw: self.get_seq_length() if hasattr(self, "get_seq_length") else 0
except Exception:
    pass

import json
import csv
import re
import math
import time
import random
import argparse
import os
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

import torch
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    AutoModelForSequenceClassification, BitsAndBytesConfig,
)
from peft import PeftModel
from openai import OpenAI

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ── Model registry (same as eval_openqa_unified.py) ──
MODELS = {
    "qwen": {
        "name": "Qwen/Qwen2.5-1.5B-Instruct",
        "mc_adapter": "sft_qwen2p5_1p5b_qlora_safe_resume2",
        "openqa_adapter": "sft_qwen2p5_1p5b_openqa_qlora",
        "needs_4bit_base": False,
        "label": "Qwen2.5-1.5B",
    },
    "llama32": {
        "name": "meta-llama/Llama-3.2-3B-Instruct",
        "mc_adapter": "sft_llama3p2_3b_qlora_safe_resume2",
        "openqa_adapter": "sft_llama3p2_3b_openqa_qlora",
        "needs_4bit_base": False,
        "label": "LLaMA-3.2-3B",
    },
    "phi35": {
        "name": "microsoft/Phi-3.5-mini-instruct",
        "mc_adapter": "sft_phi3p5_mini_qlora_safe_resume2",
        "openqa_adapter": "sft_phi3p5_mini_openqa_qlora",
        "needs_4bit_base": True,
        "label": "Phi-3.5-mini",
    },
    "llama31": {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "mc_adapter": "sft_llama3p1_8b_qlora_safe_resume2",
        "openqa_adapter": "sft_llama3p1_8b_openqa_qlora",
        "needs_4bit_base": True,
        "label": "LLaMA-3.1-8B",
    },
}

TASK_LABELS = {"base": "Base→OpenQA", "mc": "MC-finetuned→OpenQA", "openqa": "OpenQA-finetuned→OpenQA"}

OPENQA_PATH = "combined_QA_generation2.json"
MAX_NEW_TOKENS = 256
USE_CHAT_TEMPLATE = True
EVAL_RATIO = 0.2
SPLIT_SEED = 42
K_SAMPLES = 5
LLM_JUDGE_MODEL = "gpt-4o"
LLM_JUDGE_MAX_RETRIES = 3


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Answer extraction ──
def _clean(text):
    text = re.sub(r"</?answer>|</?reason>", "", str(text), flags=re.I)
    text = re.sub(r"^(final answer|answer)\s*:\s*", "", text, flags=re.I)
    text = re.split(r"\b(reason|explanation)\b\s*[:：]?", text, maxsplit=1, flags=re.I)[0]
    text = re.sub(r"\s+", " ", text).strip()
    return text.splitlines()[0].strip() if text else ""

def extract_answer(text):
    text = str(text).strip()
    if not text: return ""
    m = re.findall(r"<answer>(.*?)</answer>", text, re.I | re.S)
    if m: return _clean(m[-1])
    m = re.search(r"<answer>\s*(.*)", text, re.I | re.S)
    if m: return _clean(m.group(1))
    m = re.search(r"(?:^|\n)\s*answer\s*:\s*(.*)", text, re.I)
    if m: return _clean(m.group(1))
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return _clean(lines[-1]) if lines else _clean(text)


# ── LLM-Judge for pairwise similarity ──
LLM_SIM_PROMPT = """You are evaluating semantic similarity between two answers to a psychological question.

Question: {question}
Answer A: {a}
Answer B: {b}

Rate the similarity on this scale:
  EQUIVALENT — Same meaning (topic, direction, intensity all match)
  PARTIAL — Same direction but different intensity or specificity
  NOT_EQUIVALENT — Different meaning, opposite direction, or one is empty/garbled

Respond in exactly this format:
Label: <EQUIVALENT | PARTIAL | NOT_EQUIVALENT>"""

LABEL_TO_SCORE = {"EQUIVALENT": 1.0, "PARTIAL": 0.5, "NOT_EQUIVALENT": 0.0}


def llm_sim(client, a, b, question):
    if client is None: return -1.0
    a, b = str(a).strip(), str(b).strip()
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    prompt = LLM_SIM_PROMPT.format(question=question, a=a, b=b)
    for attempt in range(LLM_JUDGE_MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=LLM_JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=100,
            )
            text = resp.choices[0].message.content.strip()
            for label in ["NOT_EQUIVALENT", "PARTIAL", "EQUIVALENT"]:
                if f"Label: {label}" in text:
                    return LABEL_TO_SCORE[label]
            continue
        except Exception:
            continue
    return -1.0


# ── Data loading ──
def load_openqa(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = []
        for line in content.splitlines():
            line = line.strip()
            if not line: continue
            obj = json.loads(line)
            if isinstance(obj, str): obj = json.loads(obj)
            data.append(obj)
    if isinstance(data, dict): data = [data]
    examples = []
    for rec in data:
        scenario = str(rec.get("scenario", "")).strip()
        for qa in rec.get("qa_pairs", []):
            question = str(qa.get("Question", "")).strip()
            answer = str(qa.get("Answer", "")).strip()
            if not question or not answer: continue
            prompt = f"Scenario: {scenario}\nQuestion: {question}"
            examples.append({"scenario": scenario, "question": question, "gold_answer": answer, "prompt": prompt})
    return examples

def split_by_scenario(examples, eval_ratio, seed):
    grouped = defaultdict(list)
    for ex in examples:
        grouped[ex["scenario"]].append(ex)
    scenarios = list(grouped.keys())
    random.Random(seed).shuffle(scenarios)
    n_eval = max(1, int(len(scenarios) * eval_ratio))
    eval_set = set(scenarios[:n_eval])
    train, evl = [], []
    for s, rows in grouped.items():
        (evl if s in eval_set else train).extend(rows)
    return train, evl


# ── Chat prompt ──
def build_chat_prompt(tokenizer, user_prompt):
    if not USE_CHAT_TEMPLATE: return user_prompt
    messages = [
        {"role": "system", "content": "You are a careful reasoning assistant. Answer the question based on the given scenario."},
        {"role": "user", "content": f"{user_prompt}\n\nPlease reason briefly, then give the final answer in this format:\n<reason>...</reason>\n<answer>...</answer>"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ── Model loading (reused from unified) ──
def load_model(model_key, task):
    mcfg = MODELS[model_key]
    model_name = mcfg["name"]
    log(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    if task == "base" and not mcfg["needs_4bit_base"]:
        log("Loading base model (full precision)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=compute_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True, trust_remote_code=False)
    else:
        log("Loading model with 4-bit NF4...")
        bnb_kwargs = dict(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                          bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute_dtype)
        if mcfg["needs_4bit_base"]:
            bnb_kwargs["llm_int8_enable_fp32_cpu_offload"] = True
        bnb_config = BitsAndBytesConfig(**bnb_kwargs)
        load_kwargs = dict(quantization_config=bnb_config, device_map="auto",
                           low_cpu_mem_usage=True, trust_remote_code=False)
        if mcfg["needs_4bit_base"]:
            load_kwargs["max_memory"] = {0: "7GiB", "cpu": "24GiB"}
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    if task != "base":
        adapter = mcfg["mc_adapter"] if task == "mc" else mcfg["openqa_adapter"]
        log(f"Loading LoRA adapter: {adapter}")
        model = PeftModel.from_pretrained(model, adapter)

    model.eval()
    log(f"Model device: {next(model.parameters()).device}")
    return model, tokenizer


# ── Calibration metrics ──
def pearson_r(x, y):
    n = len(x)
    if n < 3: return 0.0
    mx, my = np.mean(x), np.mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = math.sqrt(sum((a - mx)**2 for a in x) * sum((b - my)**2 for b in y))
    return num / den if den > 0 else 0.0

def spearman_rho(x, y):
    def rank(arr):
        sorted_idx = sorted(range(len(arr)), key=lambda i: arr[i])
        ranks = [0.0] * len(arr)
        for r, i in enumerate(sorted_idx):
            ranks[i] = r + 1
        return ranks
    rx, ry = rank(x), rank(y)
    return pearson_r(rx, ry)

def compute_ece(consistencies, correctnesses, n_bins=10):
    bins = [[] for _ in range(n_bins)]
    for c, corr in zip(consistencies, correctnesses):
        b = min(int(c * n_bins), n_bins - 1)
        bins[b].append((c, corr))
    ece = 0.0
    n = len(consistencies)
    for bin_items in bins:
        if not bin_items: continue
        avg_conf = np.mean([c for c, _ in bin_items])
        avg_acc = np.mean([corr for _, corr in bin_items])
        ece += len(bin_items) / n * abs(avg_conf - avg_acc)
    return ece


# ── Main ──
def main():
    parser = argparse.ArgumentParser(description="Calibration Analysis via Self-Consistency")
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--task", required=True, choices=["base", "mc", "openqa"])
    parser.add_argument("--k", type=int, default=K_SAMPLES, help="Number of samples per question")
    args = parser.parse_args()

    mcfg = MODELS[args.model]
    task_label = TASK_LABELS[args.task]
    K = args.k

    log(f"=== Calibration: {mcfg['label']} | {task_label} | K={K} ===")

    # Data
    all_examples = load_openqa(OPENQA_PATH)
    if args.task == "openqa":
        _, examples = split_by_scenario(all_examples, EVAL_RATIO, SPLIT_SEED)
    else:
        examples = all_examples
    log(f"Examples: {len(examples)}")

    # Model
    model, tokenizer = load_model(args.model, args.task)
    device = next(model.parameters()).device

    # OpenAI client
    api_key = os.environ.get("OPENAI_API_KEY", "")
    oai_client = OpenAI(api_key=api_key) if api_key else None
    if not api_key:
        log("[WARN] No OPENAI_API_KEY — LLM-Judge will be skipped")

    # ── Sample K outputs per question ──
    log(f"Sampling {K} outputs per question (do_sample=True, temp=0.7)...")
    all_samples = []

    for i, ex in enumerate(examples, 1):
        prompt = build_chat_prompt(tokenizer, ex["prompt"])
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        samples = []
        for s in range(K):
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True, temperature=0.7, top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                )
            gen_text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            pred = extract_answer(gen_text)
            samples.append(pred)

        all_samples.append({
            "idx": i - 1,
            "question": ex["question"],
            "gold": ex["gold_answer"],
            "samples": samples,
        })

        if i <= 3 or i % 20 == 0 or i == len(examples):
            log(f"  [{i}/{len(examples)}] sampled {K} outputs")
            log(f"    Q: {ex['question'][:60]}")
            log(f"    GOLD: {ex['gold_answer'][:40]}")
            for s_idx, s_val in enumerate(samples):
                log(f"    S{s_idx+1}: {s_val[:40]}")

        del inputs, out
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ── Compute consistency & correctness via LLM-Judge ──
    log("Computing pairwise similarities via LLM-Judge...")
    consistencies = []
    correctnesses = []
    consensus_answers = []
    records = []

    for item in all_samples:
        question = item["question"]
        gold = item["gold"]
        samples = item["samples"]
        K_actual = len(samples)

        # Pairwise similarity among samples → consistency
        pair_sims = []
        for a in range(K_actual):
            for b in range(a + 1, K_actual):
                sim = llm_sim(oai_client, samples[a], samples[b], question)
                if sim >= 0:
                    pair_sims.append(sim)

        consistency = np.mean(pair_sims) if pair_sims else 0.0

        # Consensus answer: sample with highest avg similarity to others
        avg_sims = []
        for a in range(K_actual):
            sims_to_others = []
            for b in range(K_actual):
                if a == b: continue
                sim = llm_sim(oai_client, samples[a], samples[b], question)
                if sim >= 0:
                    sims_to_others.append(sim)
            avg_sims.append(np.mean(sims_to_others) if sims_to_others else 0.0)

        consensus_idx = int(np.argmax(avg_sims))
        consensus = samples[consensus_idx]

        # Correctness: similarity between consensus and gold
        correctness = llm_sim(oai_client, consensus, gold, question)
        if correctness < 0:
            correctness = 0.0

        consistencies.append(consistency)
        correctnesses.append(correctness)
        consensus_answers.append(consensus)

        records.append({
            "idx": item["idx"],
            "question": question,
            "gold": gold,
            "consensus": consensus,
            "consistency": round(consistency, 4),
            "correctness": round(correctness, 4),
            "samples": samples,
        })

        log(f"  [{item['idx']+1}/{len(all_samples)}] cons={consistency:.3f} corr={correctness:.3f} "
            f"consensus='{consensus[:40]}'")

    # ── Compute calibration metrics ──
    pr = pearson_r(consistencies, correctnesses)
    sr = spearman_rho(consistencies, correctnesses)
    ece = compute_ece(consistencies, correctnesses)

    log("=" * 70)
    log(f"CALIBRATION RESULTS — {mcfg['label']} | {task_label}")
    log(f"  N                : {len(examples)}")
    log(f"  K (samples)      : {K}")
    log(f"  Mean consistency : {np.mean(consistencies):.4f}")
    log(f"  Mean correctness : {np.mean(correctnesses):.4f}")
    log(f"  Pearson r        : {pr:.4f}")
    log(f"  Spearman rho     : {sr:.4f}")
    log(f"  ECE              : {ece:.4f}")
    log("=" * 70)

    # ── Save ──
    out_file = f"calibration_{args.task}_{args.model}.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f"Calibration Analysis\n")
        f.write(f"Model: {mcfg['name']}\nTask: {task_label}\nK: {K}\nN: {len(examples)}\n\n")
        f.write(f"mean_consistency = {np.mean(consistencies):.6f}\n")
        f.write(f"mean_correctness = {np.mean(correctnesses):.6f}\n")
        f.write(f"pearson_r = {pr:.6f}\n")
        f.write(f"spearman_rho = {sr:.6f}\n")
        f.write(f"ece = {ece:.6f}\n")
    log(f"Saved: {out_file}")

    csv_file = f"calibration_{args.task}_{args.model}_records.csv"
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "question", "gold", "consensus", "consistency", "correctness"])
        writer.writeheader()
        for r in records:
            writer.writerow({k: r[k] for k in ["idx", "question", "gold", "consensus", "consistency", "correctness"]})
    log(f"Saved: {csv_file}")

    jsonl_file = f"calibration_{args.task}_{args.model}_full.jsonl"
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"Saved: {jsonl_file}")


if __name__ == "__main__":
    main()
