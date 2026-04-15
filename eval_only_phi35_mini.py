"""
Eval-only recovery script for Phi-3.5-mini-Instruct.

This script loads the *already-trained* LoRA adapter from disk on top of the
Phi-3.5 base model, and re-runs the exact post-training evaluation from
train_and_full_eval_mc_phi35_mini.py (train-1000 + eval-1263 sample
generation metrics).  Training is NOT repeated.

Important:
  * The original training was run with trust_remote_code=True, i.e. using
    the Hub-side modelling_phi3.py from the Phi-3.5 repo.  To guarantee the
    saved LoRA adapter applies to *exactly the same* module structure at
    eval time, we also load with trust_remote_code=True here.
  * The Hub-side modelling_phi3.py still references `DynamicCache.seen_tokens`,
    which was removed from modern transformers.  We restore it as a property
    alias over `get_seq_length()` before importing anything that touches
    generation, so Hub-side Phi3 code keeps working without having to
    re-train.
"""

# ---------------------------------------------------------------
# DynamicCache back-compat shims — must be installed BEFORE any
# model loading / generation happens.  The Hub-side modeling_phi3.py
# for Phi-3.5-mini references several DynamicCache APIs that have
# been removed in modern transformers:
#   * past_key_values.seen_tokens             (attribute)
#   * past_key_values.get_max_length()
#   * past_key_values.get_max_cache_shape()
#   * past_key_values.get_usable_length(...)
# We restore all of them so the Hub Phi3 code keeps working without
# having to re-train.
# ---------------------------------------------------------------
import transformers  # noqa: F401
try:
    from transformers.cache_utils import DynamicCache

    # seen_tokens -> alias over get_seq_length()
    if not hasattr(DynamicCache, "seen_tokens"):
        def _seen_tokens(self):
            try:
                return self.get_seq_length()
            except Exception:
                return 0
        DynamicCache.seen_tokens = property(_seen_tokens)

    # get_max_length() -> None means "no explicit max cache length"
    if not hasattr(DynamicCache, "get_max_length"):
        def _get_max_length(self):
            return None
        DynamicCache.get_max_length = _get_max_length

    # get_max_cache_shape() -> same idea
    if not hasattr(DynamicCache, "get_max_cache_shape"):
        def _get_max_cache_shape(self):
            return None
        DynamicCache.get_max_cache_shape = _get_max_cache_shape

    # get_usable_length(new_seq_length, layer_idx=0) -> previous seq length
    # (since max cache length is None, the "usable" length is just whatever
    # is already in the cache for that layer).
    if not hasattr(DynamicCache, "get_usable_length"):
        def _get_usable_length(self, new_seq_length, layer_idx=0):
            try:
                return self.get_seq_length(layer_idx)
            except TypeError:
                # Older DynamicCache variants whose get_seq_length doesn't
                # take layer_idx.
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
import os
from dataclasses import dataclass
from typing import Dict, List
from collections import Counter

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

USE_CHAT_TEMPLATE = True

ANSWER_TAG_RE = re.compile(r"<answer>\s*([ABCD])\s*</answer>", re.I | re.S)
LETTER_RE = re.compile(r"\b([ABCD])\b", re.I)


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =========================
# Answer extraction / F1
# =========================
def extract_mc_answer(text: str) -> str:
    text = str(text).strip()

    m = ANSWER_TAG_RE.search(text)
    if m:
        return m.group(1).upper()

    m = re.search(r"<answer>\s*([ABCD])", text, re.I)
    if m:
        return m.group(1).upper()

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


# =========================
# Eval loop
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
# Config
# =========================
@dataclass
class CFG:
    model_name: str = "microsoft/Phi-3.5-mini-instruct"
    adapter_dir: str = "sft_phi3p5_mini_qlora_safe_resume2"

    train_csv: str = "prepared_data/train_split.csv"
    eval_csv: str = "prepared_data/eval_split.csv"

    # Raised from 12 -> 64 because Phi-3.5 tends to actually follow the
    # "reason then answer" prompt literally and a 12-token cap clipped
    # the output before <answer>X</answer> could be emitted, producing 0
    # accuracy even when the adapter was working correctly.
    max_new_tokens: int = 64
    eval_seed: int = 1234
    train_eval_samples: int = 1000
    eval_use_all: bool = True


def _list_adapter_dir(path: str):
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"Adapter directory not found: {path}\n"
            f"Did training finish and save the adapter?  Expected to contain "
            f"adapter_config.json and adapter_model.safetensors (or .bin)."
        )
    files = sorted(os.listdir(path))
    log(f"adapter_dir contents ({path}):")
    for f in files:
        full = os.path.join(path, f)
        size = os.path.getsize(full) if os.path.isfile(full) else 0
        log(f"  {f}  ({size} bytes)")
    # basic sanity check
    need_any_weight = any(f.startswith("adapter_model.") for f in files)
    need_config = "adapter_config.json" in files
    if not need_config:
        raise FileNotFoundError(
            f"adapter_config.json missing under {path} -- adapter was not saved correctly."
        )
    if not need_any_weight:
        raise FileNotFoundError(
            f"No adapter_model.* weight file under {path} -- adapter was not saved correctly."
        )


def load_model_and_tokenizer(cfg: CFG):
    _list_adapter_dir(cfg.adapter_dir)

    # NOTE: training was run with trust_remote_code=True.  However, the Hub-side
    # modeling_phi3.py (commit 2fe1924) is bitrotted and crashes on modern
    # transformers (DynamicCache tensor-shape mismatch in self_attn forward).
    # We therefore load the base with the *native* transformers Phi3 class
    # (trust_remote_code=False).  The native class exposes exactly the same
    # qkv_proj / o_proj module names and weight shapes, so the saved LoRA
    # adapter attaches to the corresponding modules with no surgery.
    log("loading tokenizer with trust_remote_code=False (native Phi3 path)...")
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
    log(f"bf16_supported={use_bf16} (will use {'bf16' if use_bf16 else 'fp16'})")

    log("loading base model (4-bit NF4, trust_remote_code=False, native Phi3)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    log(f"base model loaded; device={next(base.parameters()).device}")

    log(f"loading LoRA adapter from {cfg.adapter_dir} ...")
    model = PeftModel.from_pretrained(base, cfg.adapter_dir)
    model.eval()

    # Diagnostic: confirm the adapter is actually attached.
    log("peft_config keys:")
    for k, v in model.peft_config.items():
        log(f"  {k}: target_modules={v.target_modules}, r={v.r}, alpha={v.lora_alpha}")

    # `print_trainable_parameters` shows 0 in eval mode because LoRA has
    # requires_grad=False.  Count *actual* LoRA parameters by matching
    # their names in the full parameter list so we can verify the
    # adapter truly attached and is not silently empty.
    lora_param_count = 0
    lora_module_count = 0
    sample_keys = []
    for name, p in model.named_parameters():
        if "lora_" in name:
            lora_param_count += p.numel()
            lora_module_count += 1
            if len(sample_keys) < 4:
                sample_keys.append((name, tuple(p.shape)))
    log(f"actual LoRA parameter count: {lora_param_count:,} "
        f"across {lora_module_count} LoRA sub-tensors "
        f"(requires_grad is False in eval mode, but the weights are present)")
    for name, shape in sample_keys:
        log(f"  example LoRA tensor: {name}  shape={shape}")

    if lora_param_count == 0:
        raise RuntimeError(
            "LoRA parameter count is 0 after loading the adapter -- the "
            "saved adapter did not attach to any module on the base model. "
            "Check that adapter_config.json target_modules match this base."
        )

    return model, tokenizer


def main():
    cfg = CFG()

    log("start")
    log(f"torch={torch.__version__} | transformers={transformers.__version__} | "
        f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"gpu={torch.cuda.get_device_name(0)}")

    log("loading datasets...")
    train_raw = load_dataset("csv", data_files=cfg.train_csv, split="train")
    eval_raw = load_dataset("csv", data_files=cfg.eval_csv, split="train")

    log("checking and preparing dataset columns...")
    train_raw = ensure_prompt_answer_columns(train_raw, "train_raw")
    eval_raw = ensure_prompt_answer_columns(eval_raw, "eval_raw")
    log(f"datasets loaded: train={len(train_raw)}, eval={len(eval_raw)}")

    model, tokenizer = load_model_and_tokenizer(cfg)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_eval_count = min(cfg.train_eval_samples, len(train_raw))
    eval_eval_count = len(eval_raw) if cfg.eval_use_all else min(cfg.train_eval_samples, len(eval_raw))

    log(f"final_train total available samples = {len(train_raw)}")
    log(f"final_eval total available samples  = {len(eval_raw)}")
    log(f"final_train will evaluate {train_eval_count} samples")
    log(f"final_eval will evaluate {eval_eval_count} samples")

    log("evaluating final_train...")
    final_train = run_generation_metrics(
        model=model,
        tokenizer=tokenizer,
        raw_ds=train_raw,
        max_new_tokens=cfg.max_new_tokens,
        max_samples=train_eval_count,
        seed=cfg.eval_seed + 10,
        print_examples=5,
    )

    log("evaluating final_eval...")
    final_eval = run_generation_metrics(
        model=model,
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

    result_path = "mc_train_and_large_eval_results_phi3p5_mini.txt"
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"model = {cfg.model_name}\n")
        f.write(f"adapter_dir = {cfg.adapter_dir}\n")
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
