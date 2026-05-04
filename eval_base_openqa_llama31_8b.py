"""
Base model (no fine-tuning) → Open QA evaluation for LLaMA-3.1-8B-Instruct.
Evaluates the vanilla model's capability on 247 open-ended psychological QA
samples.  This establishes the baseline before any MC fine-tuning.
Metrics: Exact Match, Token F1, BEM semantic similarity.

Note: 8B in bf16 needs ~16GB VRAM.  On an 8GB GPU, we load with 4-bit NF4
quantisation.  This slightly affects output quality but is necessary for the
hardware constraint.
"""

# DynamicCache back-compat shims
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
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    AutoModelForSequenceClassification, BitsAndBytesConfig,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
OPENQA_PATH = "combined_QA_generation2.json"
OUTPUT_FILE = "base_openqa_results_llama3p1_8b.txt"
MAX_NEW_TOKENS = 64
USE_CHAT_TEMPLATE = True
PRINT_EXAMPLES = 20


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def normalize_text(s: str) -> str:
    s = str(s).strip().lower()
    return re.sub(r"\s+", " ", s)


def f1_word(pred: str, gold: str) -> float:
    p = normalize_text(pred).split()
    g = normalize_text(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    prec = overlap / len(p)
    rec = overlap / len(g)
    return 2 * prec * rec / (prec + rec)


def semsim_bem(bem_tok, bem_mdl, pred: str, gold: str, question: str) -> float:
    pred, gold = str(pred).strip(), str(gold).strip()
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    text = f"[CLS] {pred} [SEP]"
    text_pair = f"{gold} [SEP] {question} [SEP]"
    inputs = bem_tok(
        text=text, text_pair=text_pair,
        add_special_tokens=False, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    with torch.no_grad():
        logits = bem_mdl(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return float(probs[0, 1].item())


def extract_answer(text: str) -> str:
    text = str(text).strip()
    matches = re.findall(r"<answer>(.*?)</answer>", text, re.I | re.S)
    if matches:
        return _clean(matches[-1])
    m = re.search(r"<answer>\s*(.*)", text, re.I | re.S)
    if m:
        return _clean(m.group(1))
    m = re.search(r"(?:^|\n)\s*answer\s*:\s*(.*)", text, re.I)
    if m:
        return _clean(m.group(1))
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        return _clean(lines[-1])
    return _clean(text)


def _clean(text: str) -> str:
    text = re.sub(r"</?answer>|</?reason>", "", str(text), flags=re.I)
    text = re.sub(r"^(final answer|answer)\s*:\s*", "", text, flags=re.I)
    text = re.split(r"\b(reason|explanation)\b\s*[:：]?", text, maxsplit=1, flags=re.I)[0]
    text = re.sub(r"\s+", " ", text).strip()
    return text.splitlines()[0].strip() if text else ""


def load_openqa(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, str):
                obj = json.loads(obj)
            data.append(obj)

    if isinstance(data, dict):
        data = [data]

    flat = []
    for rec in data:
        scenario = str(rec.get("scenario", "")).strip()
        for qa in rec.get("qa_pairs", []):
            question = str(qa.get("Question", "")).strip()
            answer = str(qa.get("Answer", "")).strip()
            variable = str(qa.get("Variable", "")).strip()
            if question and answer:
                flat.append({
                    "scenario": scenario,
                    "variable": variable,
                    "question": question,
                    "gold_answer": answer,
                })
    return flat


def build_chat_prompt(tokenizer, ex: dict) -> str:
    user_prompt = (
        f"Scenario: {ex['scenario']}\n"
        f"Question: {ex['question']}\n\n"
        "Answer the question based on the scenario.\n"
        "Please reason briefly, then give the final answer in this format:\n"
        "<reason>...</reason>\n"
        "<answer>...</answer>"
    )
    if not USE_CHAT_TEMPLATE:
        return user_prompt
    messages = [
        {"role": "system", "content": "You are a careful reasoning assistant. Answer the question based on the given scenario."},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def evaluate(model, tokenizer, examples, bem_tok, bem_mdl):
    device = next(model.parameters()).device
    model.eval()

    em_hits, f1s, bems = 0, [], []

    for i, ex in enumerate(examples, 1):
        prompt = build_chat_prompt(tokenizer, ex)
        gold = ex["gold_answer"]

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        out = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False, use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )

        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        pred = extract_answer(gen_text)

        em = int(normalize_text(pred) == normalize_text(gold))
        f1 = f1_word(pred, gold)
        bem = semsim_bem(bem_tok, bem_mdl, pred, gold, ex["question"])

        em_hits += em
        f1s.append(f1)
        bems.append(bem)

        if i <= PRINT_EXAMPLES:
            log(f"[{i}/{len(examples)}] EM={em} F1={f1:.4f} BEM={bem:.4f}")
            log(f"  Q: {ex['question'][:80]}")
            log(f"  GOLD: {gold[:80]}")
            log(f"  PRED: {pred[:80]}")
            log("-" * 80)
        elif i % 50 == 0 or i == len(examples):
            log(f"[{i}/{len(examples)}] running...")

        del inputs, out, gen_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    n = len(examples)
    return {
        "n_eval": n,
        "exact_match": em_hits / n,
        "token_f1": sum(f1s) / n,
        "bem": sum(bems) / n,
    }


def main():
    log(f"torch={torch.__version__} | cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"gpu={torch.cuda.get_device_name(0)}")

    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    log("Loading base model (no adapter, 4-bit NF4 — 8B won't fit in bf16 on 8GB)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )

    log("Loading BEM...")
    bem_tok = AutoTokenizer.from_pretrained("kortukov/answer-equivalence-bem")
    bem_mdl = AutoModelForSequenceClassification.from_pretrained("kortukov/answer-equivalence-bem")
    bem_mdl.eval()

    log("Loading Open QA data...")
    examples = load_openqa(OPENQA_PATH)
    log(f"Total examples: {len(examples)}")

    metrics = evaluate(model, tokenizer, examples, bem_tok, bem_mdl)

    log("=" * 80)
    log("[BASE MODEL — LLaMA-3.1-8B → Open QA]")
    log(f"  N            : {metrics['n_eval']}")
    log(f"  Exact Match  : {metrics['exact_match']:.4f}")
    log(f"  Token F1     : {metrics['token_f1']:.4f}")
    log(f"  BEM          : {metrics['bem']:.4f}")
    log("=" * 80)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"model = {MODEL_NAME}\n")
        f.write(f"adapter = None (base model)\n")
        f.write(f"n_eval = {metrics['n_eval']}\n")
        f.write(f"exact_match = {metrics['exact_match']:.6f}\n")
        f.write(f"token_f1 = {metrics['token_f1']:.6f}\n")
        f.write(f"bem = {metrics['bem']:.6f}\n")
    log(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
