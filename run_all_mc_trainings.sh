#!/usr/bin/env bash
#
# Run the three new QLoRA fine-tuning + post-training evaluation scripts
# (LLaMA-3.2-3B-Instruct, Phi-3.5-mini-Instruct, LLaMA-3.1-8B-Instruct)
# sequentially on a single GPU.  The original Qwen2.5-1.5B-Instruct script
# (train_and_full_eval_mc.py) is left untouched so the 4-model comparison
# can be reproduced end-to-end from this repo.
#
# Expects:
#   * A CUDA GPU (ideally >= 24 GB for 3B / 3.8B, >= 40 GB preferred for 8B).
#   * Python env with: torch, transformers, peft, bitsandbytes, datasets,
#     accelerate.  See TRAIN_NEW_MODELS_README.md for the exact versions.
#   * HF_TOKEN exported (or `huggingface-cli login` already run) with
#     Llama-3.2 / Llama-3.1 licence accepted on the Hub.

set -euo pipefail

mkdir -p logs

echo "==========================================================="
echo "[$(date '+%F %T')] LLaMA-3.2-3B-Instruct ..."
echo "==========================================================="
python -u train_and_full_eval_mc_llama32_3b.py 2>&1 | tee logs/llama32_3b.log

echo "==========================================================="
echo "[$(date '+%F %T')] Phi-3.5-mini-Instruct ..."
echo "==========================================================="
python -u train_and_full_eval_mc_phi35_mini.py 2>&1 | tee logs/phi35_mini.log

echo "==========================================================="
echo "[$(date '+%F %T')] LLaMA-3.1-8B-Instruct ..."
echo "==========================================================="
python -u train_and_full_eval_mc_llama31_8b.py 2>&1 | tee logs/llama31_8b.log

echo "==========================================================="
echo "[$(date '+%F %T')] ALL DONE"
echo "==========================================================="
echo "Result files:"
ls -la mc_train_and_large_eval_results_*.txt || true
