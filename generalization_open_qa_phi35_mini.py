"""
MC-finetuned → Open QA generalisation evaluation for Phi-3.5-mini-Instruct (3.8B).

Loads the MC-trained QLoRA adapter from sft_phi3p5_mini_qlora_safe_resume2,
then evaluates the finetuned model on 247 open-ended psychological QA samples
(combined_QA_generation2.json).  Metrics: Exact-Match accuracy and token-level
F1 (same as the Qwen baseline in generalization_pipeline.py).

Critical fixes vs. the original generalization_pipeline.py:
  1. The base model MUST be loaded with the same 4-bit NF4 quantisation config
     that was used during training.  Loading in full precision (bf16/fp16)
     creates a distribution mismatch with the adapter weights → 0.00 accuracy.
  2. trust_remote_code=False — forces native transformers Phi3 class.
     The Hub-side modeling_phi3.py is broken on transformers 5.5+.
  3. DynamicCache shim installed defensively at import time.
  4. max_new_tokens=64 (not 12).  Phi emits reasoning before <answer>; 12
     tokens truncates the answer tag away.
"""

# ---------------------------------------------------------------
# DynamicCache back-compat shims (defensive — normally inert for
# trust_remote_code=False, but kept as insurance)
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
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer, CrossEncoder, util


# =========================
# Config
# =========================
MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"
ADAPTER_PATH = "sft_phi3p5_mini_qlora_safe_resume2"
INPUT_JSON = "combined_QA_generation2.json"
OUTPUT_DIR = "generalization_results_phi3p5_mini"

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


# =========================
# Semantic similarity ablation variants (5 methods)
# =========================
def _empty_check(pred: str, gold: str):
    pred, gold = str(pred).strip(), str(gold).strip()
    if not pred and not gold:
        return pred, gold, 1.0
    if not pred or not gold:
        return pred, gold, 0.0
    return pred, gold, None


def semsim_bienc_answer(st_model, pred, gold):
    pred, gold, short = _empty_check(pred, gold)
    if short is not None:
        return short
    emb = st_model.encode([pred, gold], convert_to_tensor=True)
    return float(util.cos_sim(emb[0], emb[1]).item())


def semsim_bienc_question(st_model, pred, gold, question):
    pred, gold, short = _empty_check(pred, gold)
    if short is not None:
        return short
    pred_ctx = f"{question} {pred}"
    gold_ctx = f"{question} {gold}"
    emb = st_model.encode([pred_ctx, gold_ctx], convert_to_tensor=True)
    return float(util.cos_sim(emb[0], emb[1]).item())


def semsim_crossenc_answer(cross_model, pred, gold):
    pred, gold, short = _empty_check(pred, gold)
    if short is not None:
        return short
    score = cross_model.predict([(pred, gold)])[0]
    return float(max(0.0, min(score / 5.0, 1.0)))


def semsim_crossenc_question(cross_model, pred, gold, question):
    pred, gold, short = _empty_check(pred, gold)
    if short is not None:
        return short
    pred_ctx = f"{question} {pred}"
    gold_ctx = f"{question} {gold}"
    score = cross_model.predict([(pred_ctx, gold_ctx)])[0]
    return float(max(0.0, min(score / 5.0, 1.0)))


def semsim_bem(bem_tok, bem_mdl, pred, gold, question):
    pred, gold, short = _empty_check(pred, gold)
    if short is not None:
        return short
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



POSITIVE_POLAR = {"yes"}
NEGATIVE_POLAR = {"no", "not", "didn't", "doesn't", "don't", "isn't", "wasn't",
                  "wouldn't", "couldn't", "shouldn't", "haven't", "hasn't"}


def _extract_polarity_gold(text: str):
    before_comma = normalize_text(text).split(",")[0]
    tokens = set(before_comma.split())
    if tokens & NEGATIVE_POLAR:
        return "negative"
    if tokens & POSITIVE_POLAR:
        return "positive"
    return None


def _extract_polarity_pred(text: str):
    tokens = set(normalize_text(text).split()[:3])
    if tokens & NEGATIVE_POLAR:
        return "negative"
    if tokens & POSITIVE_POLAR:
        return "positive"
    return None


def semsim_pa_bem(bem_score: float, cra_score: float,
                  pred_answer: str, gold_answer: str) -> float:
    gold_pol = _extract_polarity_gold(gold_answer)

    if gold_pol is None:
        return bem_score

    pred_pol = _extract_polarity_pred(pred_answer)

    if pred_pol is not None:
        if pred_pol == gold_pol:
            return bem_score
        else:
            return 0.0
    else:
        return 0.5 * bem_score + 0.5 * cra_score


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
SIM_KEYS = [
    "bienc_answer", "bienc_question",
    "crossenc_answer", "crossenc_question",
    "bem",
    "pa_bem",
]


@torch.no_grad()
def evaluate_rows(model, tokenizer, rows: List[Dict], max_new_tokens: int,
                   st_model, cross_model, bem_tok, bem_mdl) -> Dict:
    device = next(model.parameters()).device

    results = []
    acc_hits = 0
    f1s = []
    sims = {k: [] for k in SIM_KEYS}

    for i, row in enumerate(rows, start=1):
        prompt = build_chat_prompt(tokenizer, row["prompt"])
        gold_answer = row["gold_answer"]
        question = row["question"]

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

        s = {
            "bienc_answer":     semsim_bienc_answer(st_model, pred_answer, gold_answer),
            "bienc_question":   semsim_bienc_question(st_model, pred_answer, gold_answer, question),
            "crossenc_answer":  semsim_crossenc_answer(cross_model, pred_answer, gold_answer),
            "crossenc_question": semsim_crossenc_question(cross_model, pred_answer, gold_answer, question),
            "bem":              semsim_bem(bem_tok, bem_mdl, pred_answer, gold_answer, question),
        }
        s["pa_bem"] = semsim_pa_bem(s["bem"], s["crossenc_answer"], pred_answer, gold_answer)

        acc_hits += acc
        f1s.append(f1)
        for k in SIM_KEYS:
            sims[k].append(s[k])

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
        result.update(s)
        results.append(result)

        log(f"[{i}/{len(rows)}] ACC={acc} F1={f1:.4f} "
            f"BiA={s['bienc_answer']:.3f} BiQ={s['bienc_question']:.3f} "
            f"CrA={s['crossenc_answer']:.3f} CrQ={s['crossenc_question']:.3f} "
            f"BEM={s['bem']:.3f} PA={s['pa_bem']:.3f}")
        if i <= 5 or i % 50 == 0:
            log(f"  Q: {question[:80]}")
            log(f"  GOLD: {gold_answer[:80]}")
            log(f"  PRED: {pred_answer[:80]}")
            log("-" * 80)

        del inputs, out, gen_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    n = len(rows) if rows else 1
    metrics = {
        "n_eval": len(rows),
        "answer_accuracy": acc_hits / n,
        "answer_f1": sum(f1s) / n if f1s else 0.0,
    }
    for k in SIM_KEYS:
        metrics[k] = sum(sims[k]) / n if sims[k] else 0.0
    metrics["results"] = results
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
        "answer_accuracy", "answer_f1",
        "bienc_answer", "bienc_question",
        "crossenc_answer", "crossenc_question", "bem", "pa_bem",
        "gold_reason", "pred_reason", "raw_generation", "scenario", "prompt",
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
    for k in SIM_KEYS:
        summary[k] = metrics[k]
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

    log("Loading similarity models for ablation study...")
    st_model = SentenceTransformer("all-MiniLM-L6-v2")
    log("  bi-encoder loaded (all-MiniLM-L6-v2)")
    cross_model = CrossEncoder("cross-encoder/stsb-roberta-large")
    log("  cross-encoder loaded (stsb-roberta-large)")
    bem_tok = AutoTokenizer.from_pretrained("kortukov/answer-equivalence-bem")
    bem_mdl = AutoModelForSequenceClassification.from_pretrained("kortukov/answer-equivalence-bem")
    bem_mdl.eval()
    log("  BEM loaded (kortukov/answer-equivalence-bem)")

    log("Running MC-finetuned → Open QA generalisation evaluation...")
    metrics = evaluate_rows(
        model=model, tokenizer=tokenizer, rows=rows,
        max_new_tokens=MAX_NEW_TOKENS,
        st_model=st_model, cross_model=cross_model,
        bem_tok=bem_tok, bem_mdl=bem_mdl,
    )

    log("=" * 80)
    log("[FINAL GENERALISATION METRICS — Phi-3.5-mini MC→OpenQA]")
    log(f"  Samples evaluated         : {metrics['n_eval']}")
    log(f"  Exact Match (Accuracy)    : {metrics['answer_accuracy']:.4f}")
    log(f"  Token F1                  : {metrics['answer_f1']:.4f}")
    log(f"  --- Semantic Similarity Ablation ---")
    log(f"  BiEnc  (answer only)      : {metrics['bienc_answer']:.4f}")
    log(f"  BiEnc  (+ question)       : {metrics['bienc_question']:.4f}")
    log(f"  CrossEnc (answer only)    : {metrics['crossenc_answer']:.4f}")
    log(f"  CrossEnc (+ question)     : {metrics['crossenc_question']:.4f}")
    log(f"  BEM    (question-aware)   : {metrics['bem']:.4f}")
    log(f"  PA-BEM (polarity-aware)  : {metrics['pa_bem']:.4f}")

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
