@echo off
REM =====================================================================
REM Windows cmd launcher for the three new QLoRA MC training scripts.
REM Runs them sequentially and tees logs to .\logs\.
REM
REM Usage (from the repo root, inside an activated venv):
REM     run_all_mc_trainings.bat
REM
REM Requirements:
REM   * A CUDA GPU.
REM   * An activated Python venv with torch / transformers / peft /
REM     bitsandbytes / datasets / accelerate installed.
REM   * HF_TOKEN set (or `hf auth login` already done) for the two
REM     gated Llama models.
REM =====================================================================

setlocal

if not exist logs mkdir logs

REM Prefer the venv's python.exe if we're inside a venv, otherwise fall back
REM to whatever `python` resolves to on PATH.
if defined VIRTUAL_ENV (
    set "PY=%VIRTUAL_ENV%\Scripts\python.exe"
) else (
    set "PY=python"
)

echo ===========================================================
echo [%date% %time%] LLaMA-3.2-3B-Instruct ...
echo ===========================================================
"%PY%" -u train_and_full_eval_mc_llama32_3b.py 1> logs\llama32_3b.log 2>&1
if errorlevel 1 (
    echo [ERROR] Llama-3.2-3B training failed. See logs\llama32_3b.log
    exit /b 1
)

echo ===========================================================
echo [%date% %time%] Phi-3.5-mini-Instruct ...
echo ===========================================================
"%PY%" -u train_and_full_eval_mc_phi35_mini.py 1> logs\phi35_mini.log 2>&1
if errorlevel 1 (
    echo [ERROR] Phi-3.5-mini training failed. See logs\phi35_mini.log
    exit /b 1
)

echo ===========================================================
echo [%date% %time%] LLaMA-3.1-8B-Instruct ...
echo ===========================================================
"%PY%" -u train_and_full_eval_mc_llama31_8b.py 1> logs\llama31_8b.log 2>&1
if errorlevel 1 (
    echo [ERROR] Llama-3.1-8B training failed. See logs\llama31_8b.log
    exit /b 1
)

echo ===========================================================
echo [%date% %time%] ALL DONE
echo ===========================================================
echo Result files:
dir /b mc_train_and_large_eval_results_*.txt

endlocal
