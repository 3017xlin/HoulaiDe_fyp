"""
OpenQA QLoRA fine-tuning + evaluation for Qwen2.5-1.5B-Instruct.
Trains on Open QA data then evaluates on the held-out eval split.
This establishes the "task-matched upper bound" for the generalisation study.
"""

import json
import re
import time
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import Counter, defaultdict
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

USE_QLORA = True
USE_CHAT_TEMPLATE = True


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


def semsim_bem(bem_tok, bem_mdl, pred: str, gold: str, question: str) -> float:
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


# =========================
# Data loading & splitting
# =========================
def load_openqa(path: str) -> List[dict]:
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
            reason = str(qa.get("Reason", "")).strip()
            answer = str(qa.get("Answer", "")).strip()
            if not question or not answer: continue
            prompt = f"Scenario: {scenario}\nQuestion: {question}"
            target = f"<reason>{reason}</reason>\n<answer>{answer}</answer>"
            examples.append({
                "scenario": scenario, "question": question,
                "prompt": prompt, "answer": target,
                "gold_answer": answer,
            })
    return examples


def split_by_scenario(examples, eval_ratio=0.2, seed=42):
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
# Prompt & tokenization
# =========================
def build_chat_prompt(tokenizer, user_prompt: str) -> str:
    if not USE_CHAT_TEMPLATE:
        return user_prompt
    messages = [
        {"role": "system", "content": "You are a careful open-ended reasoning assistant."},
        {"role": "user", "content": (
            f"{user_prompt}\n\n"
            "Please reason briefly, then give the final answer in this format:\n"
            "<reason>...</reason>\n<answer>...</answer>"
        )},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def tokenize_sft(tokenizer, prompt, target, max_length):
    prompt_text = build_chat_prompt(tokenizer, prompt)
    full = prompt_text + target
    tok_full = tokenizer(full, truncation=True, max_length=max_length, padding=False)
    tok_prompt = tokenizer(prompt_text, truncation=True, max_length=max_length, padding=False)
    labels = tok_full["input_ids"].copy()
    labels[:len(tok_prompt["input_ids"])] = [-100] * len(tok_prompt["input_ids"])
    return {"input_ids": tok_full["input_ids"], "attention_mask": tok_full["attention_mask"], "labels": labels}


def collate_pad(tokenizer, features):
    def pad_1d(seqs, pad_id):
        maxlen = max(len(s) for s in seqs)
        out = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        return out
    return {
        "input_ids": pad_1d([f["input_ids"] for f in features], tokenizer.pad_token_id),
        "attention_mask": pad_1d([f["attention_mask"] for f in features], 0),
        "labels": pad_1d([f["labels"] for f in features], -100),
    }


# =========================
# Config
# =========================
@dataclass
class CFG:
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    raw_openqa_path: str = "combined_QA_generation2.json"
    output_dir: str = "sft_qwen2p5_1p5b_openqa_qlora"
    eval_ratio: float = 0.2
    split_seed: int = 42
    max_length: int = 384
    lr: float = 2e-4
    epochs: float = 2.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_steps: int = 200
    max_new_tokens: int = 64
    target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")


# =========================
# Model
# =========================
def make_model_and_tokenizer(cfg):
    log("loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())

    log("loading base model with 4-bit NF4...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, quantization_config=bnb_config,
        device_map="auto", low_cpu_mem_usage=True, trust_remote_code=False,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8, lora_alpha=16, target_modules=list(cfg.target_modules),
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    model.print_trainable_parameters()
    return model, tokenizer, use_bf16


# =========================
# Evaluation
# =========================
@torch.no_grad()
def run_eval(model, tokenizer, examples, max_new_tokens, bem_tok, bem_mdl):
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    em_hits, f1s, bems = 0, [], []
    for i, ex in enumerate(examples, 1):
        prompt = build_chat_prompt(tokenizer, ex["prompt"])
        gold = extract_answer(ex["answer"])

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        gen_text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = extract_answer(gen_text)

        em_hits += int(normalize_text(pred) == normalize_text(gold))
        f1s.append(f1_word(pred, gold))
        bems.append(semsim_bem(bem_tok, bem_mdl, pred, gold, ex.get("question", "")))

        if i <= 5 or i % 20 == 0 or i == len(examples):
            log(f"  eval {i}/{len(examples)} | pred={pred[:50]} | gold={gold[:50]}")

        del inputs, out
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    if was_training: model.train()
    n = len(examples)
    return {
        "n_eval": n,
        "exact_match": em_hits / n, "token_f1": sum(f1s) / n,
        "bem": sum(bems) / n,
    }


# =========================
# Main
# =========================
def main():
    cfg = CFG()

    log(f"torch={torch.__version__} | cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"gpu={torch.cuda.get_device_name(0)}")

    log("Loading Open QA data...")
    all_examples = load_openqa(cfg.raw_openqa_path)
    train_raw, eval_raw = split_by_scenario(all_examples, cfg.eval_ratio, cfg.split_seed)
    log(f"train={len(train_raw)} | eval={len(eval_raw)}")

    model, tokenizer, use_bf16 = make_model_and_tokenizer(cfg)

    def map_fn(ex):
        return tokenize_sft(tokenizer, ex["prompt"], ex["answer"], cfg.max_length)

    log("Tokenizing...")
    train_ds = Dataset.from_list(train_raw)
    eval_ds = Dataset.from_list(eval_raw)
    train_tok = train_ds.map(map_fn, remove_columns=train_ds.column_names)
    eval_tok = eval_ds.map(map_fn, remove_columns=eval_ds.column_names)

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.lr, num_train_epochs=cfg.epochs,
        warmup_ratio=cfg.warmup_ratio, weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps, save_steps=cfg.save_steps, save_total_limit=2,
        remove_unused_columns=False, report_to=[], dataloader_num_workers=0,
        max_grad_norm=1.0,
        bf16=use_bf16, fp16=(torch.cuda.is_available() and not use_bf16),
        gradient_checkpointing=True, optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
    )

    trainer = Trainer(
        model=model, args=args, train_dataset=train_tok,
        data_collator=lambda feats: collate_pad(tokenizer, feats),
    )

    if torch.cuda.is_available(): torch.cuda.empty_cache()
    log("Training OpenQA QLoRA...")
    trainer.train()

    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    log(f"Model saved -> {cfg.output_dir}")

    log("Loading BEM for evaluation...")
    bem_tok = AutoTokenizer.from_pretrained("kortukov/answer-equivalence-bem")
    bem_mdl = AutoModelForSequenceClassification.from_pretrained("kortukov/answer-equivalence-bem")
    bem_mdl.eval()

    log("Evaluating on train split...")
    m_train = run_eval(trainer.model, tokenizer, train_raw, cfg.max_new_tokens, bem_tok, bem_mdl)
    log("Evaluating on eval split...")
    m_eval = run_eval(trainer.model, tokenizer, eval_raw, cfg.max_new_tokens, bem_tok, bem_mdl)

    log("=" * 80)
    log(f"[OpenQA-finetuned — {cfg.model_name}]")
    log(f"  TRAIN (n={m_train['n_eval']}): EM={m_train['exact_match']:.4f} F1={m_train['token_f1']:.4f} BEM={m_train['bem']:.4f}")
    log(f"  EVAL  (n={m_eval['n_eval']}): EM={m_eval['exact_match']:.4f} F1={m_eval['token_f1']:.4f} BEM={m_eval['bem']:.4f}")
    log("=" * 80)

    result_path = "openqa_finetune_results_qwen25_1p5b.txt"
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"model = {cfg.model_name}\n")
        f.write(f"output_dir = {cfg.output_dir}\n")
        f.write(f"train_n = {m_train['n_eval']}\n")
        f.write(f"eval_n = {m_eval['n_eval']}\n\n")
        for label, m in [("TRAIN", m_train), ("EVAL", m_eval)]:
            f.write(f"{label}: EM={m['exact_match']:.6f} F1={m['token_f1']:.6f} BEM={m['bem']:.6f}\n")
    log(f"Results saved -> {result_path}")


if __name__ == "__main__":
    main()
