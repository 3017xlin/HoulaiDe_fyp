"""
MC QLoRA fine-tuning + post-training full-eval script for Phi-3.5-mini-Instruct (3.8B).

Sibling of train_and_full_eval_mc.py (Qwen2.5-1.5B-Instruct).  The pipeline is
deliberately kept identical so that the four-model comparison (Qwen2.5-1.5B /
Llama-3.2-3B / Phi-3.5-mini / Llama-3.1-8B) only differs in the base model.

Phi-3/3.5 specific differences vs. the Qwen baseline:
  * Attention uses a fused qkv projection: the LoRA target modules are
    ["qkv_proj", "o_proj"] (instead of the four separate q/k/v/o on Qwen/Llama).
  * We load the base model with trust_remote_code=False so that transformers'
    *native* Phi3 class is used.  The Hub-side modeling_phi3.py (commit 2fe1924)
    is incompatible with modern transformers and crashes in generate().
  * max_new_tokens is 64 (not 12) at eval time.  Phi-3.5 follows the
    "reason then answer" prompt literally and needs enough budget to emit
    <answer>X</answer>; 12 tokens is too small and produces 0 accuracy.

This file has a "smoke-test" mode gated by `train_subsample_size` below.
Set it to a small integer (default: 500) for a ~10-minute end-to-end run
to verify the pipeline before committing to the full ~8h training.  Set
it to None to train on the full 23,999-sample set.
"""

# ---------------------------------------------------------------
# DynamicCache back-compat shims — installed defensively before any
# model loading / generation happens.  We use the native Phi3 class
# (trust_remote_code=False), so these shims are normally inert, but
# they are kept in case anything in the generation stack ever reaches
# for the removed DynamicCache methods on transformers 5.5+.
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


import re
import time
import random
from dataclasses import dataclass
from typing import Dict, List, Optional
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

USE_QLORA = True
USE_CHAT_TEMPLATE = True

ANSWER_TAG_RE = re.compile(r"<answer>\s*([ABCD])\s*</answer>", re.I | re.S)
LETTER_RE = re.compile(r"\b([ABCD])\b", re.I)


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =========================
# Robust MC answer extraction
# =========================
def extract_mc_answer(text: str) -> str:
    text = str(text).strip()

    # 1) strict tag match
    m = ANSWER_TAG_RE.search(text)
    if m:
        return m.group(1).upper()

    # 2) loose <answer> ... without closing tag
    m = re.search(r"<answer>\s*([ABCD])", text, re.I)
    if m:
        return m.group(1).upper()

    # 3) common verbal patterns
    patterns = [
        r"the answer is\s*([ABCD])\b",
        r"final answer\s*[:：]?\s*([ABCD])\b",
        r"answer\s*[:：]?\s*([ABCD])\b",
        r"option\s*([ABCD])\b",
        r"choose\s*([ABCD])\b",
    ]
    lowered = text.lower()
    for p in patterns:
        m = re.search(p, lowered, re.I)
        if m:
            return m.group(1).upper()

    # 4) last standalone A/B/C/D in the output
    matches = LETTER_RE.findall(text)
    if matches:
        return matches[-1].upper()

    return ""


def normalize_choice(s: str) -> str:
    s = str(s).strip().upper()
    m = LETTER_RE.search(s)
    return m.group(1).upper() if m else s


def f1_word(pred: str, gold: str) -> float:
    pred_tokens = pred.strip().lower().split()
    gold_tokens = gold.strip().lower().split()

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
# Dataset preparation
# =========================
def ensure_prompt_answer_columns(ds, name: str):
    cols = set(ds.column_names)
    log(f"{name} original columns: {sorted(cols)}")

    if "prompt" not in cols:
        required_cols = ["context", "question", "answer0", "answer1", "answer2", "answer3"]
        missing = [c for c in required_cols if c not in cols]
        if missing:
            raise ValueError(
                f"{name} does not have 'prompt', and is missing columns needed to build it: {missing}"
            )

        def build_prompt(ex):
            context = str(ex["context"]).strip()
            question = str(ex["question"]).strip()
            a = str(ex["answer0"]).strip()
            b = str(ex["answer1"]).strip()
            c = str(ex["answer2"]).strip()
            d = str(ex["answer3"]).strip()

            prompt_text = (
                f"Context: {context}\n"
                f"Question: {question}\n"
                f"A. {a}\n"
                f"B. {b}\n"
                f"C. {c}\n"
                f"D. {d}"
            )
            return {"prompt": prompt_text}

        ds = ds.map(build_prompt)
        log(f"{name}: built 'prompt' column")

    if "answer" not in set(ds.column_names):
        if "label" not in ds.column_names:
            raise ValueError(
                f"{name} does not have 'answer', and also does not have 'label' to build it."
            )

        def build_answer(ex):
            label = int(ex["label"])
            answer_map = {0: "A", 1: "B", 2: "C", 3: "D"}
            if label not in answer_map:
                raise ValueError(f"Invalid label {label}")
            return {"answer": f"<answer>{answer_map[label]}</answer>"}

        ds = ds.map(build_answer)
        log(f"{name}: built 'answer' column from label")

    log(f"{name} final columns: {sorted(ds.column_names)}")
    return ds


def build_chat_prompt(tokenizer, user_prompt: str) -> str:
    if not USE_CHAT_TEMPLATE:
        return user_prompt

    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful multiple-choice reasoning assistant. "
                "Think briefly and answer in the required format."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{user_prompt}\n\n"
                "Please reason briefly, then give the final answer in this format:\n"
                "<reason>...</reason>\n"
                "<answer>...</answer>"
            ),
        },
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def tokenize_sft(tokenizer, prompt: str, target: str, max_length: int):
    prompt_text = build_chat_prompt(tokenizer, prompt)
    full = prompt_text + target

    tok_full = tokenizer(
        full,
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    tok_prompt = tokenizer(
        prompt_text,
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    input_ids = tok_full["input_ids"]
    attn = tok_full["attention_mask"]

    labels = input_ids.copy()
    prompt_len = len(tok_prompt["input_ids"])
    labels[:prompt_len] = [-100] * prompt_len

    return {
        "input_ids": input_ids,
        "attention_mask": attn,
        "labels": labels,
    }


def collate_pad(tokenizer, features: List[Dict]):
    def pad_1d(seqs, pad_id):
        maxlen = max(len(s) for s in seqs)
        out = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        return out

    input_ids = pad_1d([f["input_ids"] for f in features], tokenizer.pad_token_id)
    attn = pad_1d([f["attention_mask"] for f in features], 0)
    labels = pad_1d([f["labels"] for f in features], -100)

    return {
        "input_ids": input_ids,
        "attention_mask": attn,
        "labels": labels,
    }


# =========================
# Eval helpers
# =========================
def sample_indices(n: int, k: int, seed: int) -> List[int]:
    if k <= 0 or k >= n:
        return list(range(n))
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    return idx[:k]


@torch.no_grad()
def run_generation_metrics(
    model,
    tokenizer,
    raw_ds,
    max_new_tokens: int,
    max_samples: int,
    seed: int,
    print_examples: int = 5,
) -> Dict[str, float]:
    n = len(raw_ds)
    idxs = sample_indices(n, min(max_samples, n), seed)
    if not idxs:
        return {"answer_accuracy": 0.0, "answer_f1": 0.0, "n_eval": 0.0}

    device = next(model.parameters()).device
    model_was_training = model.training
    model.eval()

    acc_hits = 0
    f1s = []

    for j, i in enumerate(idxs, start=1):
        ex = raw_ds[i]
        prompt = build_chat_prompt(tokenizer, ex["prompt"])
        gold = normalize_choice(extract_mc_answer(ex["answer"]))

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
        pred = normalize_choice(extract_mc_answer(gen_text))

        acc_hits += int(pred == gold)
        f1s.append(f1_word(pred, gold))

        if j <= print_examples:
            log("=" * 90)
            log(f"[sample {j}/{len(idxs)}]")
            log(f"GOLD: {gold}")
            log(f"PRED: {pred}")
            log(f"RAW GEN: {gen_text}")

        if j % 100 == 0 or j == len(idxs):
            log(f"evaluated {j}/{len(idxs)} samples")

        del inputs, out, gen_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if model_was_training:
        model.train()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "answer_accuracy": acc_hits / len(idxs),
        "answer_f1": sum(f1s) / len(f1s),
        "n_eval": float(len(idxs)),
    }


# =========================
# Config (Phi-3.5-mini-Instruct, 3.8B)
# =========================
@dataclass
class CFG:
    # ---- Model ----
    model_name: str = "microsoft/Phi-3.5-mini-instruct"

    train_csv: str = "prepared_data/train_split.csv"
    eval_csv: str = "prepared_data/eval_split.csv"

    # NOTE: distinct output_dir so the SMOKE run does NOT overwrite any
    # previous full-training adapter in sft_phi3p5_mini_qlora_safe_resume2.
    output_dir: str = "sft_phi3p5_mini_qlora_SMOKE"

    max_length: int = 384
    lr: float = 2e-4
    # 1 epoch for smoke test; set back to 2.0 for the production run.
    epochs: float = 1.0

    # 3.8B QLoRA, micro-batch 1, grad-accum 8 fits comfortably on 8-24GB.
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8

    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    logging_steps: int = 5
    save_steps: int = 500  # won't trigger with the small smoke-test run; fine.

    # max_new_tokens bumped 12 -> 64: Phi-3.5 follows the
    # "reason then answer" prompt literally and needs enough budget to emit
    # <answer>X</answer>.  Keep this at >=32 for any Phi-style model.
    max_new_tokens: int = 64
    eval_seed: int = 1234

    # --------- SMOKE TEST knobs ---------
    # train_subsample_size: integer -> randomly take that many training
    # examples.  None -> train on the full 23,999-sample set.  For a
    # ~10 minute pipeline smoke test on an RTX 4060 Laptop, 500 works well.
    train_subsample_size: Optional[int] = 500
    train_subsample_seed: int = 42

    # Small post-training eval for the smoke test: 100 samples on train side
    # and 100 on eval side.  For a full run, set train_eval_samples=1000 and
    # eval_use_all=True to evaluate the entire 1263-sample eval set.
    train_eval_samples: int = 100
    eval_use_all: bool = False


# =========================
# Training args
# =========================
def build_training_args(cfg: CFG, use_bf16: bool) -> TrainingArguments:
    common = dict(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.lr,
        num_train_epochs=cfg.epochs,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to=[],
        dataloader_num_workers=0,
        max_grad_norm=1.0,
        bf16=use_bf16,
        fp16=(torch.cuda.is_available() and not use_bf16),
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if USE_QLORA else "adamw_torch",
        lr_scheduler_type="cosine",
    )

    try:
        return TrainingArguments(
            **common,
            eval_strategy="no",
        )
    except TypeError:
        return TrainingArguments(
            **common,
            evaluation_strategy="no",
        )


# =========================
# Model (Phi-3.5-mini)
# =========================
def make_model_and_tokenizer(cfg: CFG):
    log("loading tokenizer...")
    # trust_remote_code=False: the Hub-side modeling_phi3.py still references
    # `DynamicCache.seen_tokens` which has been removed from modern
    # transformers, which breaks model.generate() during post-training eval.
    # transformers >= 4.45 ships a native Phi3 class that uses the modern Cache
    # API correctly, so we force the native path here.
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        use_fast=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    log("tokenizer loaded")

    use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    log(f"bf16_supported={use_bf16} (we will use {'bf16' if use_bf16 else 'fp16'})")

    log("loading model...")

    if USE_QLORA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
        if torch.cuda.is_available():
            model.to("cuda")

    # Phi-3/3.5 self-attn uses fused qkv_proj + o_proj (no separate q/k/v_proj).
    # We therefore target qkv_proj and o_proj to stay analogous to the Qwen
    # attention-only LoRA configuration used in the baseline.
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["qkv_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    log(f"model device: {next(model.parameters()).device}")
    model.print_trainable_parameters()

    return model, tokenizer, use_bf16


# =========================
# Main
# =========================
def main():
    cfg = CFG()

    log("start")
    log(f"torch={torch.__version__} | cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"gpu={torch.cuda.get_device_name(0)}")

    log("loading datasets...")
    train_raw = load_dataset("csv", data_files=cfg.train_csv, split="train")
    eval_raw = load_dataset("csv", data_files=cfg.eval_csv, split="train")

    log("checking and preparing dataset columns...")
    train_raw = ensure_prompt_answer_columns(train_raw, "train_raw")
    eval_raw = ensure_prompt_answer_columns(eval_raw, "eval_raw")
    log(f"datasets loaded: train={len(train_raw)}, eval={len(eval_raw)}")

    # ---- Optional training-set subsampling for smoke tests ----
    if cfg.train_subsample_size is not None and 0 < cfg.train_subsample_size < len(train_raw):
        n_before = len(train_raw)
        train_raw = (
            train_raw.shuffle(seed=cfg.train_subsample_seed)
                     .select(range(cfg.train_subsample_size))
        )
        log(
            f"[SMOKE] subsampled train_raw: {n_before} -> {len(train_raw)} "
            f"(seed={cfg.train_subsample_seed})"
        )
    else:
        log("training on full train_raw (no subsampling)")

    model, tokenizer, use_bf16 = make_model_and_tokenizer(cfg)

    def map_fn(ex):
        return tokenize_sft(tokenizer, ex["prompt"], ex["answer"], cfg.max_length)

    log("tokenizing train...")
    train_tok = train_raw.map(map_fn, remove_columns=train_raw.column_names)
    log("tokenizing eval...")
    eval_tok = eval_raw.map(map_fn, remove_columns=eval_raw.column_names)
    log("tokenization done")

    args = build_training_args(cfg, use_bf16=use_bf16)
    log("TrainingArguments built")

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        data_collator=lambda feats: collate_pad(tokenizer, feats),
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    log("training from scratch...")
    trainer.train()

    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    log(f"[TRAINED MODEL SAVED] -> {cfg.output_dir}")

    # ========= post-training large-scale eval =========
    log("preparing large-scale evaluation...")

    train_eval_count = min(cfg.train_eval_samples, len(train_raw))
    eval_eval_count = len(eval_raw) if cfg.eval_use_all else min(cfg.train_eval_samples, len(eval_raw))

    log(f"final_train total available samples = {len(train_raw)}")
    log(f"final_eval total available samples  = {len(eval_raw)}")
    log(f"final_train will evaluate {train_eval_count} samples")
    log(f"final_eval will evaluate {eval_eval_count} samples")

    log("evaluating final_train...")
    final_train = run_generation_metrics(
        model=trainer.model,
        tokenizer=tokenizer,
        raw_ds=train_raw,
        max_new_tokens=cfg.max_new_tokens,
        max_samples=train_eval_count,
        seed=cfg.eval_seed + 10,
        print_examples=5,
    )

    log("evaluating final_eval...")
    final_eval = run_generation_metrics(
        model=trainer.model,
        tokenizer=tokenizer,
        raw_ds=eval_raw,
        max_new_tokens=cfg.max_new_tokens,
        max_samples=eval_eval_count,
        seed=cfg.eval_seed + 11,
        print_examples=5,
    )

    log("[FINAL GEN METRICS]")
    log(
        f"  TRAIN (n={int(final_train['n_eval'])}): "
        f"Acc={final_train['answer_accuracy']:.4f} | "
        f"F1={final_train['answer_f1']:.4f}"
    )
    log(
        f"  EVAL  (n={int(final_eval['n_eval'])}): "
        f"Acc={final_eval['answer_accuracy']:.4f} | "
        f"F1={final_eval['answer_f1']:.4f}"
    )

    # Distinct result file for smoke-test runs so they don't overwrite
    # the production result file.
    if cfg.train_subsample_size is not None:
        result_path = "mc_train_and_large_eval_results_phi3p5_mini_SMOKE.txt"
    else:
        result_path = "mc_train_and_large_eval_results_phi3p5_mini.txt"
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"model = {cfg.model_name}\n")
        f.write(f"output_dir = {cfg.output_dir}\n")
        f.write(f"final_train total available samples = {len(train_raw)}\n")
        f.write(f"final_eval total available samples  = {len(eval_raw)}\n")
        f.write(f"final_train evaluated samples = {int(final_train['n_eval'])}\n")
        f.write(f"final_eval evaluated samples  = {int(final_eval['n_eval'])}\n\n")
        f.write(
            f"TRAIN: Acc={final_train['answer_accuracy']:.6f}, "
            f"F1={final_train['answer_f1']:.6f}\n"
        )
        f.write(
            f"EVAL:  Acc={final_eval['answer_accuracy']:.6f}, "
            f"F1={final_eval['answer_f1']:.6f}\n"
        )

    log(f"results saved to {result_path}")


if __name__ == "__main__":
    main()
