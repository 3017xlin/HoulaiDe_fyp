import json
import re
import time
import random
import os
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel
from sentence_transformers import CrossEncoder
from openai import OpenAI


# =========================
# CONFIG
# =========================
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
CKPT_DIR = "sft_qwen2p5_1p5b_openqa_qlora"
DATA_PATH = "../combined_QA_generation2.json"

MAX_NEW_TOKENS = 256
EVAL_RATIO = 0.2
SPLIT_SEED = 42
USE_CHAT_TEMPLATE = True


# =========================
# Logging
# =========================
def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =========================
# Robust file loading
# =========================
# =========================
# LLM-as-Judge (GPT-4o-mini)
# =========================
LLM_JUDGE_PROMPT = """You are an expert evaluator for a psychological reasoning evaluation task. \
Your role is to judge whether a candidate answer is semantically equivalent \
to a reference answer, given the question's context.

## Task context

The questions are open-ended and probe subjective psychological states \
(emotions, beliefs, attitudes, self-perception). Reference answers are \
short, natural-language expressions of these states. Candidate answers \
come from language models and may vary in length, wording, and quality.

Semantic equivalence requires matching THREE dimensions:
  1. Topic — addresses what the question asks about.
  2. Direction — same valence/stance as the reference (negative/positive/neutral).
  3. Intensity — comparable strength of the state.

## Context-aware interpretation (for short candidates)

Use the question's context to interpret short answers:
  - Q: "How disappointed are you?" + Candidate: "High" → "very disappointed"
  - Q: "How confident?" + Candidate: "Low" → "not very confident"

Do NOT penalize short answers if the question's context makes meaning clear.

## DISQUALIFYING CONDITIONS (check these FIRST)

If ANY of the following applies, the label MUST be NOT_EQUIVALENT, \
regardless of topical relevance:

  D1. Truncation/Incompleteness: Candidate ends mid-sentence, mid-word, \
      or trails off without delivering a substantive response \
      (e.g., ends with "it can be", "the situation was", "concerns m").
  D2. Opposite direction: Candidate expresses the opposite valence from \
      the reference (e.g., reference: "angry/defensive"; candidate: \
      "calm/composed").
  D3. Contextual evasion: Candidate describes the scenario, generic \
      consequences, or third-person context INSTEAD of expressing the \
      psychological state the question asks about.
  D4. Format collapse: Single letter, MC option ("A"/"B"), empty string, \
      off-task content.

## Rating scale

EQUIVALENT — No disqualifying condition AND matches reference in topic, \
  direction, and intensity. Wording may differ.

PARTIAL — No disqualifying condition AND matches in topic and direction, \
  BUT differs notably in intensity, completeness, or specificity.

NOT_EQUIVALENT — Any disqualifying condition (D1–D4), OR substantively \
  different meaning despite surface similarity.

## Counter-examples (study carefully)

Example A — Opposite direction → NOT_EQUIVALENT:
  Q: "What was your reaction?"
  Reference: "I felt angry and defensive."
  Candidate: "I remained calm and asked for clarification."
  Why: "calm" is opposite to "angry/defensive" (D2).

Example B — Contextual evasion → NOT_EQUIVALENT:
  Q: "How did the accusation affect your self-esteem?"
  Reference: "It lowered my self-esteem."
  Candidate: "When someone accuses us of not working hard enough, \
              it creates a difficult situation in the workplace."
  Why: Describes generic consequences instead of the speaker's self-esteem (D3).

Example C — Truncated → NOT_EQUIVALENT:
  Q: "How did you feel?"
  Reference: "I felt deeply betrayed."
  Candidate: "When someone you trust does that to you, it can be"
  Why: Truncated mid-sentence, no substantive response delivered (D1).

## Procedure

1. Identify the psychological state dimension the question asks about.
2. Determine reference's direction and intensity.
3. Determine candidate's direction and intensity.
4. CHECK D1–D4 disqualifying conditions. If ANY applies → NOT_EQUIVALENT, skip step 5.
5. If no disqualifying condition: compare for full vs partial equivalence.
6. Output label.

## What to ignore

Length differences alone, stylistic differences (formal/casual, \
first/third person), and equivalent wordings do NOT affect the score.

---

Question: {question}
Reference answer: {gold}
Candidate answer: {pred}

Respond in EXACTLY this format, no additional text:

Reference direction & intensity: <e.g., "negative, strong">
Candidate direction & intensity: <e.g., "neutral, mild" or "N/A (truncated)">
Disqualifying conditions: <list which of D1/D2/D3/D4 apply, or "none">
Comparison: <1-2 sentences if no disqualifying condition, else "skipped">
Label: <EQUIVALENT | PARTIAL | NOT_EQUIVALENT>"""

LABEL_TO_SCORE = {"EQUIVALENT": 1.0, "PARTIAL": 0.5, "NOT_EQUIVALENT": 0.0}


def llm_judge(client, pred: str, gold: str, question: str) -> float:
    if client is None:
        return -1.0, "SKIPPED", ""
    pred, gold = str(pred).strip(), str(gold).strip()
    if not pred:
        return 0.0, "NOT_EQUIVALENT", "empty candidate"
    prompt = LLM_JUDGE_PROMPT.format(question=question, gold=gold, pred=pred)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=200,
        )
        text = resp.choices[0].message.content.strip()
        for label in ["NOT_EQUIVALENT", "PARTIAL", "EQUIVALENT"]:
            if f"Label: {label}" in text:
                return LABEL_TO_SCORE[label], label, text
        return -1.0, "PARSE_FAIL", text
    except Exception as e:
        print(f"[warn] LLM-Judge API error: {e}", flush=True)
        return -1.0, "ERROR", str(e)


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


# =========================
# Convert records -> examples
# =========================
SCENARIO_RE = re.compile(r"Scenario:\s*(.*?)\nQuestion:", re.DOTALL | re.IGNORECASE)
QUESTION_RE = re.compile(r"Question:\s*(.*?)(?:\n\n|\Z)", re.DOTALL | re.IGNORECASE)


def extract_scenario_from_prompt(prompt: str) -> str:
    m = SCENARIO_RE.search(prompt)
    return m.group(1).strip() if m else ""


def extract_question_from_prompt(prompt: str) -> str:
    m = QUESTION_RE.search(prompt)
    return m.group(1).strip() if m else ""


def convert_records_to_examples(records: List[dict]) -> List[dict]:
    examples = []

    for item in records:
        if not isinstance(item, dict):
            continue

        # 格式1：scenario + qa_pairs
        if "scenario" in item and "qa_pairs" in item:
            scenario = str(item["scenario"]).strip()
            for qa in item["qa_pairs"]:
                if not isinstance(qa, dict):
                    continue
                question = str(qa.get("Question", "")).strip()
                answer = str(qa.get("Answer", "")).strip()
                if question and answer:
                    examples.append({
                        "scenario": scenario,
                        "question": question,
                        "answer": answer,
                    })
            continue

        # 格式2：prompt + answer
        if "prompt" in item and "answer" in item:
            prompt = str(item["prompt"]).strip()
            answer_text = str(item["answer"]).strip()
            scenario = extract_scenario_from_prompt(prompt)
            question = extract_question_from_prompt(prompt)
            gold = extract_answer(answer_text)

            if scenario and question and gold:
                examples.append({
                    "scenario": scenario,
                    "question": question,
                    "answer": gold,
                })
            continue

    return examples


def load_data(path: str) -> List[dict]:
    records = load_openqa_file(path)
    log(f"Loaded raw items: {len(records)}")

    examples = convert_records_to_examples(records)
    log(f"Parsed QA pairs: {len(examples)}")

    if len(examples) == 0:
        raise ValueError("❌ 数据解析失败：没有任何 QA pair")

    return examples


# =========================
# Split by scenario
# =========================
def split_by_scenario(data: List[dict], eval_ratio: float, seed: int) -> List[dict]:
    grouped = defaultdict(list)
    for ex in data:
        grouped[ex["scenario"]].append(ex)

    scenarios = list(grouped.keys())
    random.Random(seed).shuffle(scenarios)

    n_eval = max(1, int(len(scenarios) * eval_ratio))
    eval_scenarios = set(scenarios[:n_eval])

    eval_data = []
    for scen, rows in grouped.items():
        if scen in eval_scenarios:
            eval_data.extend(rows)

    log(f"Unique scenarios: {len(scenarios)}")
    log(f"Eval scenarios: {len(eval_scenarios)}")
    log(f"Eval size: {len(eval_data)}")

    if len(eval_data) == 0:
        raise ValueError("❌ eval_data 为空，请检查切分逻辑")

    return eval_data


# =========================
# Metrics
# =========================
def normalize(x: str) -> str:
    return re.sub(r"\s+", " ", str(x).lower().strip())


def f1_score(pred: str, gold: str) -> float:
    p = normalize(pred).split()
    g = normalize(gold).split()

    if not p or not g:
        return 0.0

    common = Counter(p) & Counter(g)
    overlap = sum(common.values())

    if overlap == 0:
        return 0.0

    precision = overlap / len(p)
    recall = overlap / len(g)

    return 2 * precision * recall / (precision + recall)


def semsim_bem(bem_tok, bem_mdl, pred: str, gold: str, question: str) -> float:
    pred = str(pred).strip()
    gold = str(gold).strip()
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


def semsim_cra(cross_model, pred: str, gold: str) -> float:
    pred, gold = str(pred).strip(), str(gold).strip()
    if not pred and not gold: return 1.0
    if not pred or not gold: return 0.0
    score = cross_model.predict([(pred, gold)])[0]
    return float(max(0.0, min(score / 5.0, 1.0)))


POSITIVE_POLAR = {"yes"}
NEGATIVE_POLAR = {"no", "not", "didn't", "doesn't", "don't", "isn't", "wasn't",
                  "wouldn't", "couldn't", "shouldn't", "haven't", "hasn't"}


def _extract_polarity_gold(text):
    before_comma = normalize(text).split(",")[0]
    tokens = set(before_comma.split())
    if tokens & NEGATIVE_POLAR: return "negative"
    if tokens & POSITIVE_POLAR: return "positive"
    return None


def _extract_polarity_pred(text):
    tokens = set(normalize(text).split()[:3])
    if tokens & NEGATIVE_POLAR: return "negative"
    if tokens & POSITIVE_POLAR: return "positive"
    return None


def semsim_pa_bem(bem_score, cra_score, pred_answer, gold_answer):
    gold_pol = _extract_polarity_gold(gold_answer)
    if gold_pol is None: return bem_score
    pred_pol = _extract_polarity_pred(pred_answer)
    if pred_pol is not None:
        return bem_score if pred_pol == gold_pol else 0.0
    return 0.5 * bem_score + 0.5 * cra_score


# =========================
# Robust extraction
# =========================
def clean_text(x: str) -> str:
    x = str(x).strip()
    if not x:
        return ""

    x = re.sub(r"</?answer>", "", x, flags=re.I)
    x = re.sub(r"</?reason>", "", x, flags=re.I)
    x = re.sub(r"^(final answer|answer)\s*:\s*", "", x, flags=re.I)
    x = re.sub(r"^the correct answer is\s*", "", x, flags=re.I)
    x = re.sub(r"\s+", " ", x).strip()

    lines = [l.strip() for l in x.splitlines() if l.strip()]
    return lines[0] if lines else x


def extract_answer(text: str) -> str:
    text = str(text).strip()
    if not text:
        return ""

    matches = re.findall(r"<answer>(.*?)</answer>", text, re.S | re.I)
    if matches:
        return clean_text(matches[-1])

    m = re.search(r"<answer>\s*(.*)", text, re.S | re.I)
    if m:
        return clean_text(m.group(1))

    m = re.search(r"(?:^|\n)\s*answer\s*:\s*(.*)", text, re.I)
    if m:
        return clean_text(m.group(1))

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        return clean_text(lines[-1])

    return clean_text(text)


# =========================
# Prompt builders
# =========================
def prompt_reasoning(ex: Dict) -> str:
    return (
        f"Scenario: {ex['scenario']}\n"
        f"Question: {ex['question']}\n\n"
        "Please reason briefly, then give the final answer in this format:\n"
        "<reason>...</reason>\n"
        "<answer>...</answer>"
    )


def prompt_no_reasoning(ex: Dict) -> str:
    return (
        f"Scenario: {ex['scenario']}\n"
        f"Question: {ex['question']}\n\n"
        "ONLY output:\n"
        "<answer>...</answer>"
    )


def wrap_chat(tokenizer, text: str) -> str:
    if not USE_CHAT_TEMPLATE:
        return text

    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a helpful reasoning assistant."},
            {"role": "user", "content": text},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


# =========================
# Model loading
# =========================
def find_latest_ckpt(directory: str) -> str:
    p = Path(directory)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {directory}")

    ckpts = []
    for d in p.iterdir():
        if d.is_dir() and "checkpoint-" in d.name:
            try:
                step = int(d.name.split("-")[-1])
                ckpts.append((step, str(d)))
            except Exception:
                pass

    if ckpts:
        ckpts.sort()
        return ckpts[-1][1]

    return directory


def load_model():
    log("loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, trust_remote_code=False)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    log("loading base model with 4-bit NF4 quantisation (matching training)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )

    ckpt = find_latest_ckpt(CKPT_DIR)
    log(f"loading LoRA from: {ckpt}")
    model = PeftModel.from_pretrained(model, ckpt)
    model.eval()
    model.config.use_cache = True

    return model, tokenizer


# =========================
# Evaluation
# =========================
@torch.no_grad()
def evaluate(model, tokenizer, data: List[dict], mode: str,
             bem_tok, bem_mdl, cross_model, oai_client):
    builder = prompt_reasoning if mode == "reasoning" else prompt_no_reasoning

    em = 0
    f1s, cras, bems, pa_bems, ljs, lj_labels, lj_reasonings = [], [], [], [], [], [], [], []

    for i, ex in enumerate(data):
        prompt = wrap_chat(tokenizer, builder(ex))
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
        )

        gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = extract_answer(gen)
        gold = ex["answer"]

        em += int(normalize(pred) == normalize(gold))
        f1s.append(f1_score(pred, gold))
        cra = semsim_cra(cross_model, pred, gold)
        bem = semsim_bem(bem_tok, bem_mdl, pred, gold, ex["question"])
        pa = semsim_pa_bem(bem, cra, pred, gold)
        lj_score, lj_label, lj_reasoning = llm_judge(oai_client, pred, gold, ex["question"])
        cras.append(cra)
        bems.append(bem)
        pa_bems.append(pa)
        ljs.append(lj_score)
        lj_labels.append(lj_label)
        lj_reasonings.append(lj_reasoning)

        if (i + 1) % 20 == 0 or (i + 1) == len(data):
            log(f"{mode}: {i+1}/{len(data)}")

    valid_lj = [s for s in ljs if s >= 0]
    valid_labels = [l for l in lj_labels if l not in ("SKIPPED", "ERROR", "PARSE_FAIL")]
    n_valid = len(valid_labels) if valid_labels else 1
    lj_dist = {
        "EQUIVALENT": sum(1 for l in valid_labels if l == "EQUIVALENT") / n_valid,
        "PARTIAL": sum(1 for l in valid_labels if l == "PARTIAL") / n_valid,
        "NOT_EQUIVALENT": sum(1 for l in valid_labels if l == "NOT_EQUIVALENT") / n_valid,
    }
    n = len(data)
    return {
        "N": n,
        "EM": em / n,
        "F1": sum(f1s) / n,
        "CrA": sum(cras) / n,
        "BEM": sum(bems) / n,
        "PA_BEM": sum(pa_bems) / n,
        "LLM_Judge": sum(valid_lj) / len(valid_lj) if valid_lj else 0.0,
        "LLM_Judge_dist": lj_dist,
        "lj_reasonings": lj_reasonings,
    }


# =========================
# Main
# =========================
def main():
    log("loading data...")
    data = load_data(DATA_PATH)
    eval_data = split_by_scenario(data, EVAL_RATIO, SPLIT_SEED)

    model, tokenizer = load_model()

    log("loading similarity models...")
    cross_model = CrossEncoder("cross-encoder/stsb-roberta-large")
    bem_tok = AutoTokenizer.from_pretrained("kortukov/answer-equivalence-bem")
    bem_mdl = AutoModelForSequenceClassification.from_pretrained("kortukov/answer-equivalence-bem")
    bem_mdl.eval()
    api_key = os.environ.get("OPENAI_API_KEY", "")
    oai_client = OpenAI(api_key=api_key) if api_key else None
    if not api_key:
        log("[WARN] OPENAI_API_KEY not set — LLM-Judge will be skipped")
    log("All eval models loaded")

    log("Running mode: reasoning")
    res_reasoning = evaluate(model, tokenizer, eval_data, "reasoning", bem_tok, bem_mdl, cross_model, oai_client)

    log("Running mode: no_reasoning")
    res_no_reasoning = evaluate(model, tokenizer, eval_data, "no_reasoning", bem_tok, bem_mdl, cross_model, oai_client)

    print("\n====== FINAL RESULTS ======")
    print("\n[WITH REASONING]")
    print(res_reasoning)

    print("\n[NO REASONING]")
    print(res_no_reasoning)

    with open("reasoning_vs_no_reasoning_results.txt", "w", encoding="utf-8") as f:
        f.write("====== FINAL RESULTS ======\n\n")
        for label, res in [("WITH REASONING", res_reasoning), ("NO REASONING", res_no_reasoning)]:
            f.write(f"[{label}]\n")
            for k, v in res.items():
                if k == "lj_reasonings":
                    continue
                f.write(f"  {k} = {v}\n")
            f.write("\n")

    for mode, res in [("reasoning", res_reasoning), ("no_reasoning", res_no_reasoning)]:
        rfile = f"reasoning_{mode}_llm_judge.jsonl"
        with open(rfile, "w", encoding="utf-8") as f:
            for i, r in enumerate(res.get("lj_reasonings", [])):
                f.write(json.dumps({"idx": i, "mode": mode, "reasoning": r}, ensure_ascii=False) + "\n")
        log(f"Saved: {rfile}")


if __name__ == "__main__":
    main()