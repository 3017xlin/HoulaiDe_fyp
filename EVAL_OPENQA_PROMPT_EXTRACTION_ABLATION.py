import json
import re
import time
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter, defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, StoppingCriteria, StoppingCriteriaList, BitsAndBytesConfig
from peft import PeftModel
from sentence_transformers import CrossEncoder


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
RAW_OPENQA_PATH = "../combined_QA_generation2.json"
OUTPUT_DIR = "sft_qwen2p5_1p5b_openqa_qlora"

EVAL_RATIO = 0.2
SPLIT_SEED = 42
MAX_NEW_TOKENS = 32
USE_CHAT_TEMPLATE = True


class StopOnSubstrings(StoppingCriteria):
    def __init__(self, tokenizer, prompt_len: int, stop_strings: List[str]):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.stop_strings = stop_strings

    def __call__(self, input_ids, scores, **kwargs):
        gen_ids = input_ids[0][self.prompt_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return any(s in text for s in self.stop_strings)


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def normalize_text(s: str) -> str:
    s = str(s).strip().lower()
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
    if gold_pol is None: return bem_score
    pred_pol = _extract_polarity_pred(pred_answer)
    if pred_pol is not None:
        return bem_score if pred_pol == gold_pol else 0.0
    return 0.5 * bem_score + 0.5 * cra_score


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


SCENARIO_RE = re.compile(r"Scenario:\s*(.*?)\nQuestion:", re.DOTALL | re.IGNORECASE)
QUESTION_RE = re.compile(r"Question:\s*(.*?)(?:\n\n|\Z)", re.DOTALL | re.IGNORECASE)


def extract_scenario_from_prompt(prompt: str) -> str:
    m = SCENARIO_RE.search(prompt)
    return m.group(1).strip() if m else ""


def extract_question_from_prompt(prompt: str) -> str:
    m = QUESTION_RE.search(prompt)
    return m.group(1).strip() if m else ""


def convert_records_to_prompt_answer(records: List[dict]) -> List[dict]:
    out = []
    for rec in records:
        if not isinstance(rec, dict):
            continue

        if "prompt" in rec and "answer" in rec:
            prompt = str(rec["prompt"]).strip()
            answer = str(rec["answer"]).strip()
            out.append({
                "scenario": extract_scenario_from_prompt(prompt),
                "question": extract_question_from_prompt(prompt),
                "prompt": prompt,
                "answer": answer,
            })
            continue

        if "scenario" in rec and "qa_pairs" in rec:
            scenario = str(rec["scenario"]).strip()
            qa_pairs = rec.get("qa_pairs", [])
            if not scenario or not isinstance(qa_pairs, list):
                continue

            for qa in qa_pairs:
                question = str(qa.get("Question", "")).strip()
                reason = str(qa.get("Reason", "")).strip()
                answer = str(qa.get("Answer", "")).strip()
                if not question or not answer:
                    continue

                prompt = (
                    f"Scenario: {scenario}\n"
                    f"Question: {question}\n\n"
                    "Please reason briefly, then give the final answer in this format:\n"
                    "<reason>...</reason>\n"
                    "<answer>...</answer>"
                )
                target = f"<reason>{reason}</reason>\n<answer>{answer}</answer>"
                out.append({
                    "scenario": scenario,
                    "question": question,
                    "prompt": prompt,
                    "answer": target,
                })
    return out


def split_by_scenario(examples: List[dict], eval_ratio: float, seed: int) -> Tuple[List[dict], List[dict]]:
    grouped = defaultdict(list)
    for ex in examples:
        grouped[ex["scenario"]].append(ex)

    scenarios = list(grouped.keys())
    rng = random.Random(seed)
    rng.shuffle(scenarios)

    n_eval = max(1, int(len(scenarios) * eval_ratio))
    eval_scenarios = set(scenarios[:n_eval])

    train_examples, eval_examples = [], []
    for scen, rows in grouped.items():
        if scen in eval_scenarios:
            eval_examples.extend(rows)
        else:
            train_examples.extend(rows)
    return train_examples, eval_examples


def find_latest_checkpoint(output_dir: str):
    out = Path(output_dir)
    if not out.exists():
        return output_dir

    ckpts = []
    for p in out.iterdir():
        if p.is_dir():
            m = re.match(r"checkpoint-(\d+)$", p.name)
            if m:
                ckpts.append((int(m.group(1)), str(p)))

    if not ckpts:
        return output_dir

    ckpts.sort(key=lambda x: x[0])
    return ckpts[-1][1]


# =========================
# Prompt builders
# =========================
def build_user_prompt_normal(ex: Dict) -> str:
    return (
        f"Scenario: {ex['scenario']}\n"
        f"Question: {ex['question']}\n\n"
        "Answer the question based on the scenario."
    )


def build_user_prompt_constrained(ex: Dict) -> str:
    return (
        f"Scenario: {ex['scenario']}\n"
        f"Question: {ex['question']}\n\n"
        "You MUST follow the format exactly.\n"
        "ONLY output:\n"
        "<answer>...</answer>\n"
        "Do not include explanation.\n"
        "Do not include extra text."
    )


def build_chat_prompt(tokenizer, user_prompt: str) -> str:
    if not USE_CHAT_TEMPLATE:
        return user_prompt

    messages = [
        {"role": "system", "content": "You are a careful reasoning assistant."},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# =========================
# Extraction strategies
# =========================
def extract_answer_strict(text: str) -> str:
    text = str(text).strip()
    matches = re.findall(r"<answer>(.*?)</answer>", text, flags=re.I | re.S)
    if matches:
        return normalize_extracted(matches[-1])
    return ""


def extract_answer_relaxed(text: str) -> str:
    text = str(text).strip()
    if not text:
        return ""

    matches = re.findall(r"<answer>(.*?)</answer>", text, flags=re.I | re.S)
    if matches:
        return normalize_extracted(matches[-1])

    m = re.search(r"<answer>\s*(.*)", text, flags=re.I | re.S)
    if m:
        return normalize_extracted(m.group(1))

    m = re.search(r"(?:^|\n)\s*answer\s*:\s*(.*)", text, flags=re.I)
    if m:
        return normalize_extracted(m.group(1))

    return ""


def extract_answer_robust(text: str) -> str:
    text = str(text).strip()
    if not text:
        return ""

    matches = re.findall(r"<answer>(.*?)</answer>", text, flags=re.I | re.S)
    if matches:
        ans = normalize_extracted(matches[-1])
        if ans:
            return ans

    m = re.search(r"<answer>\s*(.*)", text, flags=re.I | re.S)
    if m:
        ans = normalize_extracted(m.group(1))
        if ans:
            return ans

    m = re.search(r"(?:^|\n)\s*answer\s*:\s*(.*)", text, flags=re.I)
    if m:
        ans = normalize_extracted(m.group(1))
        if ans:
            return ans

    m = re.search(r"the correct answer is\s*(.*)", text, flags=re.I)
    if m:
        ans = normalize_extracted(m.group(1))
        if ans:
            return ans

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        ans = normalize_extracted(lines[-1])
        if ans:
            return ans

    return normalize_extracted(text)


def normalize_extracted(text: str) -> str:
    text = str(text).strip()
    if not text:
        return ""

    text = re.sub(r"</?answer>", "", text, flags=re.I)
    text = re.sub(r"</?reason>", "", text, flags=re.I)
    text = re.sub(r"^(final answer|answer)\s*:\s*", "", text, flags=re.I)
    text = re.sub(r"^the correct answer is\s*", "", text, flags=re.I)

    # 去掉后续 explanation/reason 残留
    text = re.split(r"\b(?:reason|explanation)\b\s*[:：]?", text, maxsplit=1, flags=re.I)[0]
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return ""

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[0] if lines else text


def get_extractor(mode: str):
    if mode == "strict":
        return extract_answer_strict
    if mode == "relaxed":
        return extract_answer_relaxed
    if mode == "robust":
        return extract_answer_robust
    raise ValueError(f"Unknown extraction mode: {mode}")


def get_prompt_builder(mode: str):
    if mode == "normal":
        return build_user_prompt_normal
    if mode in {"constrained", "constrained_stop"}:
        return build_user_prompt_constrained
    raise ValueError(f"Unknown prompt mode: {mode}")


@torch.no_grad()
def run_eval(model, tokenizer, examples: List[Dict], max_new_tokens: int,
             bem_tok, bem_mdl, cross_model, prompt_mode: str, extraction_mode: str):
    device = next(model.parameters()).device
    model.eval()
    model.config.use_cache = True

    extractor = get_extractor(extraction_mode)
    prompt_builder = get_prompt_builder(prompt_mode)

    em_hits = 0
    f1s, cras, bems, pa_bems = [], [], [], []

    for idx, ex in enumerate(examples):
        prompt = build_chat_prompt(tokenizer, prompt_builder(ex))
        gold = extract_answer_robust(ex["answer"])
        question = ex.get("question", "")

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
        )

        if prompt_mode == "constrained_stop":
            stopping = StoppingCriteriaList([
                StopOnSubstrings(
                    tokenizer=tokenizer,
                    prompt_len=inputs["input_ids"].shape[1],
                    stop_strings=["</answer>", "\n\n"],
                )
            ])
            gen_kwargs["stopping_criteria"] = stopping

        out = model.generate(**inputs, **gen_kwargs)

        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        pred = extractor(gen_text)

        em_hits += int(normalize_text(pred) == normalize_text(gold))
        f1s.append(f1_word(pred, gold))
        cra = semsim_cra(cross_model, pred, gold)
        bem = semsim_bem(bem_tok, bem_mdl, pred, gold, question)
        cras.append(cra)
        bems.append(bem)
        pa_bems.append(semsim_pa_bem(bem, cra, pred, gold))

        if (idx + 1) % 20 == 0 or (idx + 1) == len(examples):
            log(f"progress: {idx+1}/{len(examples)}")

    n = len(examples) if examples else 1
    return {
        "n_eval": len(examples),
        "exact_match": em_hits / n,
        "token_f1": sum(f1s) / n,
        "CrA": sum(cras) / n,
        "BEM": sum(bems) / n,
        "PA_BEM": sum(pa_bems) / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_mode", type=str, required=True,
                        choices=["normal", "constrained", "constrained_stop"])
    parser.add_argument("--extraction_mode", type=str, required=True,
                        choices=["strict", "relaxed", "robust"])
    args = parser.parse_args()

    log(f"prompt_mode={args.prompt_mode} | extraction_mode={args.extraction_mode}")

    log("loading raw data...")
    records = load_openqa_file(RAW_OPENQA_PATH)
    examples = convert_records_to_prompt_answer(records)
    _, eval_examples = split_by_scenario(examples, EVAL_RATIO, SPLIT_SEED)
    log(f"eval examples: {len(eval_examples)}")

    log("loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log("loading base model with 4-bit NF4 quantisation (matching training)...")
    use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
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

    ckpt_path = find_latest_checkpoint(OUTPUT_DIR)
    log(f"loading LoRA checkpoint from: {ckpt_path}")
    model = PeftModel.from_pretrained(model, ckpt_path)

    log("loading similarity models...")
    cross_model = CrossEncoder("cross-encoder/stsb-roberta-large")
    bem_tok = AutoTokenizer.from_pretrained("kortukov/answer-equivalence-bem")
    bem_mdl = AutoModelForSequenceClassification.from_pretrained("kortukov/answer-equivalence-bem")
    bem_mdl.eval()
    log("CrA + BEM loaded")

    metrics = run_eval(
        model=model,
        tokenizer=tokenizer,
        examples=eval_examples,
        max_new_tokens=MAX_NEW_TOKENS,
        bem_tok=bem_tok,
        bem_mdl=bem_mdl,
        cross_model=cross_model,
        prompt_mode=args.prompt_mode,
        extraction_mode=args.extraction_mode,
    )

    title = f"[PROMPT={args.prompt_mode} | EXTRACTION={args.extraction_mode}]"
    log(title)
    log(f"N = {metrics['n_eval']}")
    log(f"Exact Match  = {metrics['exact_match']:.4f}")
    log(f"Token F1     = {metrics['token_f1']:.4f}")
    log(f"CrA          = {metrics['CrA']:.4f}")
    log(f"BEM          = {metrics['BEM']:.4f}")
    log(f"PA-BEM       = {metrics['PA_BEM']:.4f}")

    out_name = f"openqa_eval_{args.prompt_mode}_{args.extraction_mode}.txt"
    with open(out_name, "w", encoding="utf-8") as f:
        f.write(title + "\n")
        f.write(f"N = {metrics['n_eval']}\n")
        f.write(f"Exact Match = {metrics['exact_match']:.4f}\n")
        f.write(f"Token F1 = {metrics['token_f1']:.4f}\n")
        f.write(f"CrA = {metrics['CrA']:.4f}\n")
        f.write(f"BEM = {metrics['BEM']:.4f}\n")
        f.write(f"PA-BEM = {metrics['PA_BEM']:.4f}\n")


if __name__ == "__main__":
    main()