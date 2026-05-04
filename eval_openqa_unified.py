"""
Unified Open QA evaluation script for all models and tasks.

Supports three tasks (target domain = Open QA):
  1. base       — Base model evaluated on Open QA (baseline capability)
  2. mc         — MC-finetuned model evaluated on Open QA (cross-format generalisation)
  3. openqa     — OpenQA-finetuned model evaluated on Open QA (format-matched upper bound)

Supports four models:
  qwen    — Qwen2.5-1.5B-Instruct
  llama32 — LLaMA-3.2-3B-Instruct
  phi35   — Phi-3.5-mini-Instruct (3.8B)
  llama31 — LLaMA-3.1-8B-Instruct

Metrics: Exact Match, Token F1, CrA (cross-encoder answer-only),
         BEM (question-aware), PA-BEM (polarity-aware adaptive).

Usage:
  python eval_openqa_unified.py --model qwen --task base
  python eval_openqa_unified.py --model llama32 --task mc
  python eval_openqa_unified.py --model phi35 --task openqa
  python eval_openqa_unified.py --model llama31 --task base
  ...
"""

# DynamicCache shims (defensive)
import transformers  # noqa: F401
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
import re
import csv
import time
import random
import argparse
from pathlib import Path
from typing import Dict, List
from collections import Counter, defaultdict

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    AutoModelForSequenceClassification, BitsAndBytesConfig,
)
from peft import PeftModel
from sentence_transformers import CrossEncoder

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# =========================
# Model registry
# =========================
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
        "needs_4bit_base": False,
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

TASK_LABELS = {
    "base": "Base→OpenQA",
    "mc": "MC-finetuned→OpenQA",
    "openqa": "OpenQA-finetuned→OpenQA",
}

OPENQA_PATH = "combined_QA_generation2.json"
MAX_NEW_TOKENS = 64
USE_CHAT_TEMPLATE = True
EVAL_RATIO = 0.2
SPLIT_SEED = 42


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =========================
# Text utilities
# =========================
def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def f1_word(pred: str, gold: str) -> float:
    p, g = normalize_text(pred).split(), normalize_text(gold).split()
    if not p and not g: return 1.0
    if not p or not g: return 0.0
    overlap = sum((Counter(p) & Counter(g)).values())
    if overlap == 0: return 0.0
    prec, rec = overlap / len(p), overlap / len(g)
    return 2 * prec * rec / (prec + rec)


def _clean(text: str) -> str:
    text = re.sub(r"</?answer>|</?reason>", "", str(text), flags=re.I)
    text = re.sub(r"^(final answer|answer)\s*:\s*", "", text, flags=re.I)
    text = re.split(r"\b(reason|explanation)\b\s*[:：]?", text, maxsplit=1, flags=re.I)[0]
    text = re.sub(r"\s+", " ", text).strip()
    return text.splitlines()[0].strip() if text else ""


def extract_answer(text: str) -> str:
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


# =========================
# Semantic similarity: CrA, BEM, PA-BEM
# =========================
def semsim_cra(cross_model, pred, gold):
    pred, gold = str(pred).strip(), str(gold).strip()
    if not pred and not gold: return 1.0
    if not pred or not gold: return 0.0
    score = cross_model.predict([(pred, gold)])[0]
    return float(max(0.0, min(score / 5.0, 1.0)))


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


POSITIVE_POLAR = {"yes"}
NEGATIVE_POLAR = {"no", "not", "didn't", "doesn't", "don't", "isn't", "wasn't",
                  "wouldn't", "couldn't", "shouldn't", "haven't", "hasn't"}


def _extract_polarity_gold(text):
    before_comma = normalize_text(text).split(",")[0]
    tokens = set(before_comma.split())
    if tokens & NEGATIVE_POLAR: return "negative"
    if tokens & POSITIVE_POLAR: return "positive"
    return None


def _extract_polarity_pred(text):
    tokens = set(normalize_text(text).split()[:3])
    if tokens & NEGATIVE_POLAR: return "negative"
    if tokens & POSITIVE_POLAR: return "positive"
    return None


def semsim_pa_bem(bem_score, cra_score, pred_answer, gold_answer):
    gold_pol = _extract_polarity_gold(gold_answer)
    if gold_pol is None:
        return bem_score
    pred_pol = _extract_polarity_pred(pred_answer)
    if pred_pol is not None:
        return bem_score if pred_pol == gold_pol else 0.0
    return 0.5 * bem_score + 0.5 * cra_score


# =========================
# Data loading
# =========================
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
            reason = str(qa.get("Reason", "")).strip()
            if not question or not answer: continue
            prompt = f"Scenario: {scenario}\nQuestion: {question}"
            target = f"<reason>{reason}</reason>\n<answer>{answer}</answer>"
            examples.append({
                "scenario": scenario, "question": question,
                "gold_answer": answer, "prompt": prompt, "answer": target,
            })
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


# =========================
# Prompt
# =========================
def build_chat_prompt(tokenizer, user_prompt):
    if not USE_CHAT_TEMPLATE:
        return user_prompt
    messages = [
        {"role": "system", "content": "You are a careful reasoning assistant. Answer the question based on the given scenario."},
        {"role": "user", "content": (
            f"{user_prompt}\n\n"
            "Please reason briefly, then give the final answer in this format:\n"
            "<reason>...</reason>\n<answer>...</answer>"
        )},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# =========================
# Model loading
# =========================
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
        log("Loading base model (full precision, no adapter)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=compute_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True, trust_remote_code=False,
        )
    else:
        log("Loading model with 4-bit NF4 quantisation...")
        bnb_kwargs = dict(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute_dtype,
        )
        if mcfg["needs_4bit_base"]:
            bnb_kwargs["llm_int8_enable_fp32_cpu_offload"] = True
        bnb_config = BitsAndBytesConfig(**bnb_kwargs)

        load_kwargs = dict(
            quantization_config=bnb_config, device_map="auto",
            low_cpu_mem_usage=True, trust_remote_code=False,
        )
        if mcfg["needs_4bit_base"]:
            load_kwargs["max_memory"] = {0: "7GiB", "cpu": "24GiB"}

        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    if task != "base":
        adapter_path = mcfg["mc_adapter"] if task == "mc" else mcfg["openqa_adapter"]
        log(f"Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    log(f"Model device: {next(model.parameters()).device}")
    return model, tokenizer


# =========================
# Evaluation
# =========================
@torch.no_grad()
def evaluate(model, tokenizer, examples, cross_model, bem_tok, bem_mdl):
    device = next(model.parameters()).device
    em_hits, f1s, cras, bems, pa_bems = 0, [], [], [], []

    for i, ex in enumerate(examples, 1):
        prompt = build_chat_prompt(tokenizer, ex["prompt"])
        gold = ex["gold_answer"]
        question = ex["question"]

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        out = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False, use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        gen_text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = extract_answer(gen_text)

        em = int(normalize_text(pred) == normalize_text(gold))
        f1 = f1_word(pred, gold)
        cra = semsim_cra(cross_model, pred, gold)
        bem = semsim_bem(bem_tok, bem_mdl, pred, gold, question)
        pa = semsim_pa_bem(bem, cra, pred, gold)

        em_hits += em
        f1s.append(f1)
        cras.append(cra)
        bems.append(bem)
        pa_bems.append(pa)

        if i <= 5 or i % 50 == 0 or i == len(examples):
            log(f"  [{i}/{len(examples)}] EM={em} F1={f1:.3f} CrA={cra:.3f} BEM={bem:.3f} PA={pa:.3f}")
            log(f"    Q: {question[:70]}")
            log(f"    GOLD: {gold[:70]}")
            log(f"    PRED: {pred[:70]}")

        del inputs, out
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    n = len(examples)
    return {
        "n_eval": n,
        "exact_match": em_hits / n,
        "token_f1": sum(f1s) / n,
        "cra": sum(cras) / n,
        "bem": sum(bems) / n,
        "pa_bem": sum(pa_bems) / n,
    }


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(description="Unified Open QA evaluation")
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()),
                        help="Model to evaluate")
    parser.add_argument("--task", required=True, choices=["base", "mc", "openqa"],
                        help="base=no finetune, mc=MC-finetuned, openqa=OpenQA-finetuned")
    args = parser.parse_args()

    mcfg = MODELS[args.model]
    task_label = TASK_LABELS[args.task]

    log(f"=== {mcfg['label']} | {task_label} ===")
    log(f"torch={torch.__version__} | cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"gpu={torch.cuda.get_device_name(0)}")

    log("Loading Open QA data...")
    all_examples = load_openqa(OPENQA_PATH)

    if args.task == "openqa":
        _, eval_examples = split_by_scenario(all_examples, EVAL_RATIO, SPLIT_SEED)
        examples = eval_examples
        log(f"Using EVAL split only: {len(examples)} examples (model was trained on train split)")
    else:
        examples = all_examples
        log(f"Using ALL examples: {len(examples)}")

    model, tokenizer = load_model(args.model, args.task)

    log("Loading similarity models...")
    cross_model = CrossEncoder("cross-encoder/stsb-roberta-large")
    bem_tok = AutoTokenizer.from_pretrained("kortukov/answer-equivalence-bem")
    bem_mdl = AutoModelForSequenceClassification.from_pretrained("kortukov/answer-equivalence-bem")
    bem_mdl.eval()
    log("Similarity models loaded")

    log("Running evaluation...")
    metrics = evaluate(model, tokenizer, examples, cross_model, bem_tok, bem_mdl)

    log("=" * 80)
    log(f"[{mcfg['label']} | {task_label}]")
    log(f"  N              : {metrics['n_eval']}")
    log(f"  Exact Match    : {metrics['exact_match']:.4f}")
    log(f"  Token F1       : {metrics['token_f1']:.4f}")
    log(f"  CrA            : {metrics['cra']:.4f}")
    log(f"  BEM            : {metrics['bem']:.4f}")
    log(f"  PA-BEM         : {metrics['pa_bem']:.4f}")
    log("=" * 80)

    out_file = f"openqa_eval_{args.task}_{args.model}.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f"model = {mcfg['name']}\n")
        f.write(f"task = {task_label}\n")
        adapter = "None" if args.task == "base" else (mcfg["mc_adapter"] if args.task == "mc" else mcfg["openqa_adapter"])
        f.write(f"adapter = {adapter}\n")
        f.write(f"n_eval = {metrics['n_eval']}\n\n")
        f.write(f"exact_match = {metrics['exact_match']:.6f}\n")
        f.write(f"token_f1 = {metrics['token_f1']:.6f}\n")
        f.write(f"cra = {metrics['cra']:.6f}\n")
        f.write(f"bem = {metrics['bem']:.6f}\n")
        f.write(f"pa_bem = {metrics['pa_bem']:.6f}\n")
    log(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
