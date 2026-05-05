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
         BEM (question-aware), PA-BEM (polarity-aware adaptive),
         LLM-Judge (GPT-4o, 3-class with retry on malformed output).

Note: PA-BEM polarity heuristic is limited (yes/no detection) and largely
degrades to BEM on subjective open-ended answers. Reported as a baseline.

Usage:
  python eval_openqa_unified.py --model qwen --task base
  python eval_openqa_unified.py --model llama32 --task mc
  python eval_openqa_unified.py --model phi35 --task openqa
  python eval_openqa_unified.py --model llama31 --task base --use_eval_split_for_all
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
import os
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter, defaultdict

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    AutoModelForSequenceClassification, BitsAndBytesConfig,
)
from peft import PeftModel
from sentence_transformers import CrossEncoder
from openai import OpenAI

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

TASK_LABELS = {
    "base": "Base→OpenQA",
    "mc": "MC-finetuned→OpenQA",
    "openqa": "OpenQA-finetuned→OpenQA",
}

OPENQA_PATH = "combined_QA_generation2.json"
MAX_NEW_TOKENS = 256
USE_CHAT_TEMPLATE = True
EVAL_RATIO = 0.2
SPLIT_SEED = 42

# LLM-Judge config
LLM_JUDGE_MODEL = "gpt-4o"
LLM_JUDGE_MAX_TOKENS = 600
LLM_JUDGE_TEMPERATURE = 0.0
LLM_JUDGE_MAX_RETRIES = 5


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
# Generation degeneration detection
# =========================
def is_degenerate(pred: str) -> str:
    pred = str(pred).strip()
    if len(pred) < 3:
        return "too_short"

    words = pred.split()
    if len(words) >= 6:
        bigrams = [(words[i], words[i+1]) for i in range(len(words) - 1)]
        if len(bigrams) > 0:
            bigram_unique_ratio = len(set(bigrams)) / len(bigrams)
            if bigram_unique_ratio < 0.4:
                return f"bigram_loop(uniq={bigram_unique_ratio:.2f})"

        unigram_unique_ratio = len(set(words)) / len(words)
        if unigram_unique_ratio < 0.3:
            return f"unigram_repetition(uniq={unigram_unique_ratio:.2f})"

    weird_pattern = re.findall(r"\b(?:[A-Z][a-z]?){2,}\b", pred)
    common_caps = {"I", "USA", "UK", "OK", "TV", "AI", "API", "GPU", "CEO", "MC"}
    weird_tokens = [w for w in weird_pattern if w not in common_caps and len(w) <= 6]
    if len(weird_tokens) >= 2:
        return f"garbled_tokens({','.join(weird_tokens[:3])})"

    return "ok"


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
    tokens = set(normalize_text(text).split()[:5])
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
# LLM-as-Judge (GPT-4o)
# =========================
LLM_JUDGE_PROMPT = """You are an expert evaluator for a psychological reasoning evaluation task. \
Your role is to judge whether a candidate answer is semantically equivalent \
to a reference answer, given the question's context.

## Task context

The questions are open-ended and probe subjective psychological states \
(emotions, beliefs, attitudes, self-perception). Reference answers are \
short, natural-language expressions of these states. Candidate answers \
come from language models and may vary in length, wording, and quality, \
including degraded or broken outputs.

Semantic equivalence requires matching THREE dimensions:
  1. Topic — addresses what the question asks about.
  2. Direction — same valence/stance as the reference (negative/neutral/positive).
  3. Intensity — comparable strength of the state.

## Context-aware interpretation (for short candidates only)

Use the question's context to interpret short, well-formed answers:
  - Q: "How disappointed are you?" + Candidate: "High" → "very disappointed"
  - Q: "How confident?" + Candidate: "Low" → "not very confident"

This applies ONLY to short but COHERENT candidates. It does NOT apply to \
candidates that are truncated, garbled, or evasive (see disqualifying \
conditions below).

## DISQUALIFYING CONDITIONS (check these BEFORE judging equivalence)

If ANY of D1–D5 applies, the label MUST be NOT_EQUIVALENT, regardless \
of any topical relevance the candidate may have.

  D1. TRUNCATION/INCOMPLETENESS
      Candidate ends mid-sentence, mid-word, or trails off without \
      delivering a substantive response.
      Examples: ends with "it can be", "due to the", "concerns m".

  D2. OPPOSITE DIRECTION
      Candidate expresses the OPPOSITE valence/stance from the reference.
      Examples:
        - Reference: "I felt angry and defensive" (negative, strong)
          Candidate: "I remained calm and composed" (neutral/positive)
          → OPPOSITE direction. D2 applies.
        - Words signaling opposite direction from a NEGATIVE reference:
          "calm", "composed", "patient", "understanding", "happy", "fine".
        - Words signaling opposite direction from a POSITIVE reference:
          "upset", "disappointed", "angry", "sad".

  D3. CONTEXTUAL EVASION
      Candidate describes the scenario, generic consequences, or \
      third-person context INSTEAD of expressing the psychological state \
      the question asks about.
      Examples:
        - Q asks "How did this affect your self-esteem?" but candidate \
          says "When someone is accused, it creates a difficult \
          situation" without expressing how self-esteem changed.

  D4. FORMAT COLLAPSE
      Single letter ("A", "B"), MC option, empty string, off-task content, \
      or output that is not a natural-language answer.

  D5. GENERATION DEGENERATION
      Candidate contains repeated phrases, meaningless token sequences, \
      garbled text, broken word fragments, or other signs that the \
      language model failed to produce coherent output.
      Examples:
        - "I am sorry I am Ipm I am sorry Id. I am IdId? I am sorry Idd"
          → repetition + garbled tokens (Ipm, IdId, Idd). D5 applies.
        - "the the the the the the the the" → token loop. D5 applies.
      Surface phrases like "I am sorry" in degenerate output do NOT count \
      as legitimate sentiment.

## Rating scale

EQUIVALENT — No disqualifying condition AND matches reference in topic, \
  direction, AND intensity. Wording may differ.

PARTIAL — No disqualifying condition AND matches in topic and direction, \
  BUT differs in intensity, completeness, or specificity.

NOT_EQUIVALENT — Any disqualifying condition (D1–D5) applies, OR the \
  candidate has substantively different meaning from the reference.

## What to ignore (only for COHERENT candidates)

For candidates that pass D1–D5, the following do NOT affect the score:
  - Length differences alone.
  - Stylistic differences (formal vs. casual, first vs. third person).
  - Different but semantically equivalent wordings.

## Counter-examples (study carefully)

Example A — D2 Opposite direction → NOT_EQUIVALENT:
  Q: "What was your reaction?"
  Reference: "I felt angry and defensive."
  Candidate: "I would remain calm and composed, seeking to understand."
  Disqualifying: D2 (opposite direction: calm vs. angry)
  Label: NOT_EQUIVALENT

Example B — D3 Contextual evasion → NOT_EQUIVALENT:
  Q: "How did the accusation affect your self-esteem?"
  Reference: "It lowered my self-esteem."
  Candidate: "When someone accuses us of not working hard enough, it \
              creates a difficult workplace situation."
  Disqualifying: D3 (describes generic situation, never says how \
                 self-esteem was affected)
  Label: NOT_EQUIVALENT

Example C — D1 Truncation → NOT_EQUIVALENT:
  Q: "How did you feel?"
  Reference: "I felt deeply betrayed."
  Candidate: "When someone you trust does that to you, it can be"
  Disqualifying: D1 (truncated mid-sentence)
  Label: NOT_EQUIVALENT

Example D — D5 Degeneration → NOT_EQUIVALENT:
  Q: "What was your initial reaction?"
  Reference: "I felt angry and defensive."
  Candidate: "I am sorry I am Ipm I am sorry Id. I am IdId? I am sorry Idd"
  Disqualifying: D5 (repetitive, garbled output)
  Label: NOT_EQUIVALENT

Example E — Coherent short answer → EQUIVALENT:
  Q: "How disappointed are you?"
  Reference: "Very disappointed."
  Candidate: "High."
  Disqualifying: none
  Label: EQUIVALENT

Example F — Direction matches but intensity differs → PARTIAL:
  Q: "How upset were you?"
  Reference: "Extremely upset."
  Candidate: "A bit annoyed."
  Disqualifying: none
  Label: PARTIAL (direction matches; intensity much weaker)

## Procedure (follow EVERY step in order)

STEP 1: Analyze the REFERENCE answer ALONE — direction & intensity.
STEP 2: Analyze the CANDIDATE answer ALONE — direction & intensity \
        (use question context only if candidate is coherent).
STEP 3: Check ALL FIVE disqualifying conditions D1–D5 explicitly.
STEP 4: Apply the labeling rule:
        - If ANY of D1–D5 applies → NOT_EQUIVALENT
        - Else if direction AND intensity both match → EQUIVALENT
        - Else (direction matches but intensity differs) → PARTIAL
        - Else → NOT_EQUIVALENT

## Self-consistency rule (IMPORTANT)

Your final label MUST be consistent with your disqualifying conditions \
check. If you marked any of D1–D5 as applying, your label MUST be \
NOT_EQUIVALENT. There are NO exceptions. Output EXACTLY ONE Label line \
with EXACTLY ONE of: EQUIVALENT, PARTIAL, NOT_EQUIVALENT.

---

Question: {question}
Reference answer: {gold}
Candidate answer: {pred}

Respond in EXACTLY this format, with no additional text:

Reference direction & intensity: <e.g., "negative, strong">
Candidate direction & intensity: <e.g., "neutral, mild" or "N/A (degenerate)">
D1 truncation: <yes/no — brief reason>
D2 opposite direction: <yes/no — brief reason>
D3 contextual evasion: <yes/no — brief reason>
D4 format collapse: <yes/no — brief reason>
D5 generation degeneration: <yes/no — brief reason>
Comparison: <1-2 sentences if no D1-D5 applies, otherwise "skipped due to D#">
Label: <EQUIVALENT | PARTIAL | NOT_EQUIVALENT>
"""

LABEL_TO_SCORE = {"EQUIVALENT": 1.0, "PARTIAL": 0.5, "NOT_EQUIVALENT": 0.0}


def _validate_judge_response(text: str) -> Tuple[bool, str]:
    matches = re.findall(
        r"Label\s*:\s*(NOT_EQUIVALENT|EQUIVALENT|PARTIAL)\b",
        text,
        re.IGNORECASE,
    )

    if len(matches) == 0:
        return False, "no_label_found"
    if len(matches) > 1:
        unique = set(m.upper() for m in matches)
        if len(unique) > 1:
            return False, f"multiple_conflicting_labels:{','.join(unique)}"
        return True, "ok_duplicate_same_label"

    return True, "ok"


def _parse_judge_response(text: str) -> Tuple[str, dict]:
    label_match = re.search(
        r"Label\s*:\s*(NOT_EQUIVALENT|EQUIVALENT|PARTIAL)\b",
        text,
        re.IGNORECASE,
    )

    d_flags = {}
    for d_key in ["D1", "D2", "D3", "D4", "D5"]:
        m = re.search(
            rf"{d_key}\s*[a-z\s]*\s*:\s*(yes|no)\b",
            text,
            re.IGNORECASE,
        )
        d_flags[d_key.lower()] = m.group(1).lower() if m else "unknown"

    if not label_match:
        return "PARSE_FAIL", {"raw": text, "d_flags": d_flags}

    label = label_match.group(1).upper()

    any_disqualifying = any(v == "yes" for v in d_flags.values())
    overridden = False
    if any_disqualifying and label != "NOT_EQUIVALENT":
        label = "NOT_EQUIVALENT"
        overridden = True

    return label, {
        "raw": text,
        "d_flags": d_flags,
        "overridden": overridden,
    }


def llm_judge(client, pred: str, gold: str, question: str, degen_status: str = "ok",
              example_idx: int = -1):
    if client is None:
        return -1.0, "SKIPPED", {"reason": "no client", "retries": 0}

    pred, gold = str(pred).strip(), str(gold).strip()
    if not pred:
        return 0.0, "EMPTY", {"reason": "empty candidate", "retries": 0}

    if degen_status != "ok":
        return 0.0, "DEGENERATE_AUTO", {"reason": f"pre_filter:{degen_status}", "retries": 0}

    prompt = LLM_JUDGE_PROMPT.format(question=question, gold=gold, pred=pred)

    last_text = ""
    last_reason = ""

    for attempt in range(LLM_JUDGE_MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=LLM_JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=LLM_JUDGE_TEMPERATURE,
                max_tokens=LLM_JUDGE_MAX_TOKENS,
            )
            text = resp.choices[0].message.content.strip()
            last_text = text

            is_valid, reason = _validate_judge_response(text)

            if not is_valid:
                last_reason = reason
                idx_str = f" [idx={example_idx}]" if example_idx >= 0 else ""
                print(f"Model judgement failed ({reason}){idx_str}, "
                      f"restart another judgement. (attempt {attempt + 1}/{LLM_JUDGE_MAX_RETRIES})",
                      flush=True)
                continue

            label, debug = _parse_judge_response(text)

            if label == "PARSE_FAIL":
                last_reason = "parse_fail_after_validation"
                print(f"Model judgement failed (parse_fail_after_validation), "
                      f"restart another judgement. (attempt {attempt + 1}/{LLM_JUDGE_MAX_RETRIES})",
                      flush=True)
                continue

            debug["retries"] = attempt
            return LABEL_TO_SCORE[label], label, debug

        except Exception as e:
            print(f"[warn] LLM-Judge API error (attempt {attempt + 1}): {e}", flush=True)
            last_reason = f"api_error:{e}"
            continue

    print(f"[warn] LLM-Judge exhausted {LLM_JUDGE_MAX_RETRIES} retries "
          f"(last reason: {last_reason}). Skipping this item.", flush=True)
    return -1.0, "RETRY_EXHAUSTED", {
        "reason": last_reason,
        "raw": last_text,
        "retries": LLM_JUDGE_MAX_RETRIES,
    }


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
def evaluate(model, tokenizer, examples, cross_model, bem_tok, bem_mdl, oai_client):
    device = next(model.parameters()).device

    records = []
    em_hits = 0
    f1s, cras, bems, pa_bems, llm_scores = [], [], [], [], []
    lj_labels, lj_debugs = [], []
    degen_count = 0
    parse_fail_count = 0
    api_error_count = 0
    safeguard_override_count = 0
    retry_exhausted_count = 0
    total_retries = 0

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

        degen_status = is_degenerate(pred)
        if degen_status != "ok":
            degen_count += 1

        em = int(normalize_text(pred) == normalize_text(gold))
        f1 = f1_word(pred, gold)
        cra = semsim_cra(cross_model, pred, gold)
        bem = semsim_bem(bem_tok, bem_mdl, pred, gold, question)
        pa = semsim_pa_bem(bem, cra, pred, gold)
        lj_score, lj_label, lj_debug = llm_judge(
            oai_client, pred, gold, question, degen_status, example_idx=i - 1
        )

        if lj_label == "PARSE_FAIL":
            parse_fail_count += 1
        if lj_label == "ERROR":
            api_error_count += 1
        if lj_label == "RETRY_EXHAUSTED":
            retry_exhausted_count += 1
        if isinstance(lj_debug, dict):
            if lj_debug.get("overridden"):
                safeguard_override_count += 1
            total_retries += lj_debug.get("retries", 0)

        em_hits += em
        f1s.append(f1)
        cras.append(cra)
        bems.append(bem)
        pa_bems.append(pa)
        llm_scores.append(lj_score)
        lj_labels.append(lj_label)
        lj_debugs.append(lj_debug)

        records.append({
            "idx": i - 1,
            "scenario": ex["scenario"],
            "question": question,
            "gold": gold,
            "raw_generation": gen_text,
            "pred": pred,
            "degen_status": degen_status,
            "em": em,
            "f1": round(f1, 4),
            "cra": round(cra, 4),
            "bem": round(bem, 4),
            "pa_bem": round(pa, 4),
            "lj_score": lj_score,
            "lj_label": lj_label,
            "lj_overridden": lj_debug.get("overridden", False) if isinstance(lj_debug, dict) else False,
            "lj_retries": lj_debug.get("retries", 0) if isinstance(lj_debug, dict) else 0,
            "lj_d_flags": lj_debug.get("d_flags", {}) if isinstance(lj_debug, dict) else {},
            "lj_raw": lj_debug.get("raw", "") if isinstance(lj_debug, dict) else "",
        })

        log(f"  [{i}/{len(examples)}] EM={em} F1={f1:.3f} CrA={cra:.3f} BEM={bem:.3f} PA={pa:.3f} LJ={lj_score:.2f}({lj_label}){'[degen]' if degen_status != 'ok' else ''}")
        log(f"    Q: {question[:70]}")
        log(f"    GOLD: {gold[:70]}")
        log(f"    PRED: {pred[:70]}")

        del inputs, out
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    valid_labels = [l for l in lj_labels
                    if l in ("EQUIVALENT", "PARTIAL", "NOT_EQUIVALENT", "DEGENERATE_AUTO", "EMPTY")]
    collapsed_labels = ["NOT_EQUIVALENT" if l in ("DEGENERATE_AUTO", "EMPTY") else l for l in valid_labels]
    n_valid = len(collapsed_labels) if collapsed_labels else 1
    lj_dist = {
        "EQUIVALENT": sum(1 for l in collapsed_labels if l == "EQUIVALENT") / n_valid,
        "PARTIAL": sum(1 for l in collapsed_labels if l == "PARTIAL") / n_valid,
        "NOT_EQUIVALENT": sum(1 for l in collapsed_labels if l == "NOT_EQUIVALENT") / n_valid,
    }

    valid_lj_scores = [s for s in llm_scores if s >= 0]
    n = len(examples)

    return {
        "n_eval": n,
        "exact_match": em_hits / n,
        "token_f1": sum(f1s) / n,
        "cra": sum(cras) / n,
        "bem": sum(bems) / n,
        "pa_bem": sum(pa_bems) / n,
        "llm_judge": sum(valid_lj_scores) / len(valid_lj_scores) if valid_lj_scores else 0.0,
        "llm_judge_n": len(valid_lj_scores),
        "llm_judge_dist": lj_dist,
        "degen_count": degen_count,
        "degen_rate": degen_count / n,
        "parse_fail_count": parse_fail_count,
        "api_error_count": api_error_count,
        "safeguard_override_count": safeguard_override_count,
        "retry_exhausted_count": retry_exhausted_count,
        "total_retries": total_retries,
        "records": records,
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
    parser.add_argument("--use_eval_split_for_all", action="store_true",
                        help="If set, base/mc tasks also use only the eval split (matches openqa N).")
    args = parser.parse_args()

    mcfg = MODELS[args.model]
    task_label = TASK_LABELS[args.task]

    log(f"=== {mcfg['label']} | {task_label} ===")
    log(f"torch={torch.__version__} | cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"gpu={torch.cuda.get_device_name(0)}")

    log("Loading Open QA data...")
    all_examples = load_openqa(OPENQA_PATH)

    if args.task == "openqa" or args.use_eval_split_for_all:
        _, eval_examples = split_by_scenario(all_examples, EVAL_RATIO, SPLIT_SEED)
        examples = eval_examples
        log(f"Using EVAL split only: {len(examples)} examples")
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

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        log("[WARN] OPENAI_API_KEY not set — LLM-Judge scores will be -1")
    oai_client = OpenAI(api_key=api_key) if api_key else None

    log("Running evaluation...")
    metrics = evaluate(model, tokenizer, examples, cross_model, bem_tok, bem_mdl, oai_client)

    log("=" * 80)
    log(f"[{mcfg['label']} | {task_label}]")
    log(f"  N              : {metrics['n_eval']}")
    log(f"  Exact Match    : {metrics['exact_match']:.4f}")
    log(f"  Token F1       : {metrics['token_f1']:.4f}")
    log(f"  CrA            : {metrics['cra']:.4f}")
    log(f"  BEM            : {metrics['bem']:.4f}")
    log(f"  PA-BEM         : {metrics['pa_bem']:.4f}")
    log(f"  LLM-Judge      : {metrics['llm_judge']:.4f} (n={metrics['llm_judge_n']})")
    dist = metrics['llm_judge_dist']
    log(f"  LLM-Judge dist : EQ={dist['EQUIVALENT']:.2%} PAR={dist['PARTIAL']:.2%} NEQ={dist['NOT_EQUIVALENT']:.2%}")
    log(f"  Degen rate     : {metrics['degen_rate']:.2%} ({metrics['degen_count']}/{metrics['n_eval']})")
    log(f"  LJ retries     : total_retries={metrics['total_retries']} retry_exhausted={metrics['retry_exhausted_count']}")
    log(f"  LJ robustness  : api_err={metrics['api_error_count']} parse_fail={metrics['parse_fail_count']} overrides={metrics['safeguard_override_count']}")
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
        f.write(f"llm_judge = {metrics['llm_judge']:.6f}\n")
        f.write(f"llm_judge_n = {metrics['llm_judge_n']}\n")
        dist = metrics['llm_judge_dist']
        f.write(f"llm_judge_EQUIVALENT = {dist['EQUIVALENT']:.4f}\n")
        f.write(f"llm_judge_PARTIAL = {dist['PARTIAL']:.4f}\n")
        f.write(f"llm_judge_NOT_EQUIVALENT = {dist['NOT_EQUIVALENT']:.4f}\n")
        f.write(f"degen_rate = {metrics['degen_rate']:.6f}\n")
        f.write(f"degen_count = {metrics['degen_count']}\n")
        f.write(f"lj_total_retries = {metrics['total_retries']}\n")
        f.write(f"lj_retry_exhausted = {metrics['retry_exhausted_count']}\n")
        f.write(f"lj_api_error = {metrics['api_error_count']}\n")
        f.write(f"lj_parse_fail = {metrics['parse_fail_count']}\n")
        f.write(f"safeguard_overrides = {metrics['safeguard_override_count']}\n")
    log(f"Saved summary: {out_file}")

    records_file = f"openqa_eval_{args.task}_{args.model}_records.jsonl"
    with open(records_file, "w", encoding="utf-8") as f:
        for r in metrics["records"]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"Saved per-example records: {records_file}")


if __name__ == "__main__":
    main()
