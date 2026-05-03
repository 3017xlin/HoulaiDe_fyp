"""
MC-finetuned → Open QA generalisation evaluation for LLaMA-3.1-8B-Instruct.

Loads the MC-trained QLoRA adapter from sft_llama3p1_8b_qlora_safe_resume2,
then evaluates the finetuned model on 247 open-ended psychological QA samples
(combined_QA_generation2.json).  Metrics: Exact-Match accuracy and token-level
F1 (same as the Qwen baseline in generalization_pipeline.py).

Critical fixes vs. the original generalization_pipeline.py:
  1. The base model MUST be loaded with the same 4-bit NF4 quantisation config
     that was used during training.  Loading in full precision (bf16/fp16)
     creates a distribution mismatch with the adapter weights → 0.00 accuracy.
  2. DynamicCache shim installed defensively at import time.
  3. max_new_tokens=64 (not 12).

Note:
  * meta-llama/Llama-3.1-8B-Instruct is a gated model. Accept the license on
    the Hugging Face Hub and set HF_TOKEN (or run `huggingface-cli login`)
    before launching.  If model files are already cached locally, set
    HF_HUB_OFFLINE=1 to skip network checks.
"""

# ---------------------------------------------------------------
# DynamicCache back-compat shims (defensive — normally inert)
# ---------------------------------------------------------------
import transformers  # noqa: F401
try:
    from transformers.cache_utils import DynamicCache

    if not hasattr(DynamicCache, "seen_tokens"):
        def _seen_tokens(self):
            try:
                return self.get_seq_length()
            except Exception:
                return 0
        DynamicCache.seen_tokens = property(_seen_tokens)

    if not hasattr(DynamicCache, "get_max_length"):
        DynamicCache.get_max_length = lambda self: None

    if not hasattr(DynamicCache, "get_max_cache_shape"):
        DynamicCache.get_max_cache_shape = lambda self: None

    if not hasattr(DynamicCache, "get_usable_length"):
        def _get_usable_length(self, new_seq_length, layer_idx=0):
            try:
                return self.get_seq_length(layer_idx)
            except TypeError:
                try:
                    return self.get_seq_length()
                except Exception:
                    return 0
            except Exception:
                return 0
        DynamicCache.get_usable_length = _get_usable_length
except Exception as _e:
    print(f"[warn] DynamicCache shim skipped: {_e}", flush=True)


import json
import re
import csv
import time
import random
from pathlib import Path
from typing import Dict, List
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


# =========================
# Config
# =========================
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_PATH = "sft_llama3p1_8b_qlora_safe_resume2"
INPUT_JSON = "combined_QA_generation2.json"
OUTPUT_DIR = "generalization_results_llama3p1_8b"

USE_CHAT_TEMPLATE = True
MAX_NEW_TOKENS = 64
MAX_SAMPLES = 0  # 0 = evaluate all 247 samples
SEED = 42


ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
REASON_RE = re.compile(r"<reason>(.*?)</reason>", re.DOTALL)


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =========================
# Utilities
# =========================
def extract_answer(text: str) -> str:
    m = ANSWER_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def extract_reason(text: str) -> str:
    m = REASON_RE.search(text)
    return m.group(1).strip() if m else ""


def normalize_text(s: str) -> str:
    s = s.strip().lower()
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


def sample_indices(n: int, k: int, seed: int) -> List[int]:
    if k <= 0 or k >= n:
        return list(range(n))
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    return idx[:k]


# =========================
# Dataset loading
# =========================
def load_json_maybe_string(path: str):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    try:
        data = json.loads(content)
        return data
    except json.JSONDecodeError:
        rows = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, str):
                obj = json.loads(obj)
            rows.append(obj)
        return rows


def build_open_ended_eval_rows(data) -> List[Dict]:
    rows = []
    if isinstance(data, dict):
        data = [data]

    row_id = 0
    for ex in data:
        scenario = str(ex.get("scenario", "")).strip()
        qa_pairs = ex.get("qa_pairs", [])

        for qa in qa_pairs:
            question = str(qa.get("Question", "")).strip()
            reason = str(qa.get("Reason", "")).strip()
            answer = str(qa.get("Answer", "")).strip()
            variable = str(qa.get("Variable", "")).strip()

            prompt = (
                f"Scenario: {scenario}\n"
                f"Question: {question}\n\n"
                "Please reason briefly, then give the final answer in this format:\n"
                "<reason>...</reason>\n"
                "<answer>...</answer>"
            )

            target = f"<reason>{reason}</reason>\n<answer>{answer}</answer>"

            rows.append({
                "id": row_id,
                "scenario": scenario,
                "variable": variable,
                "question": question,
                "gold_reason": reason,
                "gold_answer": answer,
                "prompt": prompt,
                "answer": target,
            })
            row_id += 1

    return rows


# =========================
# Prompt construction
# =========================
def build_chat_prompt(tokenizer, user_prompt: str) -> str:
    if not USE_CHAT_TEMPLATE:
        return user_prompt

    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful reasoning assistant. "
                "Follow the required format exactly."
            ),
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# =========================
# Model loading (with QLoRA quantisation — must match training)
# =========================
def load_model_and_tokenizer():
    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    log("Loading base model with 4-bit NF4 quantisation (matching training config)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )

    log(f"Loading trained LoRA adapter from {ADAPTER_PATH} ...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    log(f"Model device: {next(model.parameters()).device}")
    total_lora = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
    log(f"LoRA parameters loaded: {total_lora:,}")
    return model, tokenizer


# =========================
# Evaluation
# =========================
@torch.no_grad()
def evaluate_rows(model, tokenizer, rows: List[Dict], max_new_tokens: int) -> Dict:
    device = next(model.parameters()).device

    results = []
    acc_hits = 0
    f1s = []

    for i, row in enumerate(rows, start=1):
        prompt = build_chat_prompt(tokenizer, row["prompt"])
        gold_answer = row["gold_answer"]

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )

        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        pred_answer = extract_answer(gen_text)
        pred_reason = extract_reason(gen_text)

        acc = int(normalize_text(pred_answer) == normalize_text(gold_answer))
        f1 = f1_word(pred_answer, gold_answer)

        acc_hits += acc
        f1s.append(f1)

        result = {
            "id": row["id"],
            "scenario": row["scenario"],
            "variable": row["variable"],
            "question": row["question"],
            "prompt": row["prompt"],
            "gold_reason": row["gold_reason"],
            "gold_answer": row["gold_answer"],
            "raw_generation": gen_text,
            "pred_reason": pred_reason,
            "pred_answer": pred_answer,
            "answer_accuracy": acc,
            "answer_f1": f1,
        }
        results.append(result)

        log(f"[{i}/{len(rows)}] ACC={acc} F1={f1:.4f}")
        if i <= 5 or i % 50 == 0:
            log(f"  Q: {row['question'][:80]}")
            log(f"  GOLD: {gold_answer[:80]}")
            log(f"  PRED: {pred_answer[:80]}")
            log("-" * 80)

        del inputs, out, gen_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics = {
        "n_eval": len(rows),
        "answer_accuracy": acc_hits / len(rows) if rows else 0.0,
        "answer_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "results": results,
    }
    return metrics


# =========================
# Save results
# =========================
def save_jsonl(rows: List[Dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def save_csv(rows: List[Dict], path: str):
    if not rows:
        return
    fieldnames = [
        "id", "variable", "question", "gold_answer", "pred_answer",
        "answer_accuracy", "answer_f1", "gold_reason", "pred_reason",
        "raw_generation", "scenario", "prompt",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_summary(metrics: Dict, path: str):
    summary = {
        "model": MODEL_NAME,
        "adapter": ADAPTER_PATH,
        "n_eval": metrics["n_eval"],
        "answer_accuracy": metrics["answer_accuracy"],
        "answer_f1": metrics["answer_f1"],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


# =========================
# Main
# =========================
def main():
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    log(f"torch={torch.__version__} | cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"gpu={torch.cuda.get_device_name(0)}")

    log("Loading Open QA dataset...")
    data = load_json_maybe_string(INPUT_JSON)

    log("Converting to evaluation rows...")
    rows = build_open_ended_eval_rows(data)
    log(f"Total rows: {len(rows)}")

    if MAX_SAMPLES > 0:
        idxs = sample_indices(len(rows), MAX_SAMPLES, SEED)
        rows = [rows[i] for i in idxs]
        log(f"Sampled rows: {len(rows)}")

    model, tokenizer = load_model_and_tokenizer()

    log("Running MC-finetuned → Open QA generalisation evaluation...")
    metrics = evaluate_rows(
        model=model,
        tokenizer=tokenizer,
        rows=rows,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    log("=" * 80)
    log("[FINAL GENERALISATION METRICS — LLaMA-3.1-8B MC→OpenQA]")
    log(f"  Samples evaluated : {metrics['n_eval']}")
    log(f"  Answer Accuracy   : {metrics['answer_accuracy']:.4f}")
    log(f"  Answer F1         : {metrics['answer_f1']:.4f}")
    log("=" * 80)

    jsonl_path = str(Path(OUTPUT_DIR) / "predictions.jsonl")
    csv_path = str(Path(OUTPUT_DIR) / "predictions.csv")
    summary_path = str(Path(OUTPUT_DIR) / "summary.json")

    save_jsonl(metrics["results"], jsonl_path)
    save_csv(metrics["results"], csv_path)
    save_summary(metrics, summary_path)

    log(f"Saved: {jsonl_path}")
    log(f"Saved: {csv_path}")
    log(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
