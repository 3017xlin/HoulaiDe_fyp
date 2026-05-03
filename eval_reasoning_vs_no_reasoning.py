import json
import re
import time
import random
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer, util


# =========================
# CONFIG
# =========================
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
CKPT_DIR = "sft_qwen2p5_1p5b_openqa_qlora"
DATA_PATH = "../combined_QA_generation2.json"

MAX_NEW_TOKENS = 32
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


def semantic_similarity(st_model, a: str, b: str) -> float:
    emb = st_model.encode([a, b], convert_to_tensor=True)
    return float(util.cos_sim(emb[0], emb[1]))


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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log("loading base model...")
    use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if use_bf16 else (torch.float16 if torch.cuda.is_available() else torch.float32),
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
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
def evaluate(model, tokenizer, data: List[dict], mode: str, st_model):
    builder = prompt_reasoning if mode == "reasoning" else prompt_no_reasoning

    em = 0
    f1s = []
    sims = []

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
        sims.append(semantic_similarity(st_model, pred, gold))

        if (i + 1) % 20 == 0 or (i + 1) == len(data):
            log(f"{mode}: {i+1}/{len(data)}")

    return {
        "EM": em / len(data),
        "F1": sum(f1s) / len(f1s),
        "Semantic": sum(sims) / len(sims),
        "N": len(data),
    }


# =========================
# Main
# =========================
def main():
    log("loading data...")
    data = load_data(DATA_PATH)
    eval_data = split_by_scenario(data, EVAL_RATIO, SPLIT_SEED)

    model, tokenizer = load_model()

    log("loading sentence-transformer...")
    st_model = SentenceTransformer("all-MiniLM-L6-v2")

    log("Running mode: reasoning")
    res_reasoning = evaluate(model, tokenizer, eval_data, "reasoning", st_model)

    log("Running mode: no_reasoning")
    res_no_reasoning = evaluate(model, tokenizer, eval_data, "no_reasoning", st_model)

    print("\n====== FINAL RESULTS ======")
    print("\n[WITH REASONING]")
    print(res_reasoning)

    print("\n[NO REASONING]")
    print(res_no_reasoning)

    with open("reasoning_vs_no_reasoning_results.txt", "w", encoding="utf-8") as f:
        f.write("====== FINAL RESULTS ======\n\n")
        f.write("[WITH REASONING]\n")
        f.write(json.dumps(res_reasoning, ensure_ascii=False, indent=2))
        f.write("\n\n[NO REASONING]\n")
        f.write(json.dumps(res_no_reasoning, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()