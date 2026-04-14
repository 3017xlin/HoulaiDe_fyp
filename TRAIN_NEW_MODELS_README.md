# 扩展实验：四模型 MC QLoRA 训练脚本说明

本仓库里 `train_and_full_eval_mc.py` 是 **Qwen2.5-1.5B-Instruct** 的原始训练 + 全量评估脚本（与论文 v1 对应的基线）。为了把论文的泛化性分析扩展到 **四个不同规模 / 家族 / 训练策略的模型**，本次新增了三份独立的训练脚本：

| 脚本 | Base model | 参数量 | 家族 | 训练策略 | 定位 |
|---|---|---|---|---|---|
| `train_and_full_eval_mc.py` | Qwen2.5-1.5B-Instruct | 1.5B | Qwen | 通用 instruction | 原始基线 |
| `train_and_full_eval_mc_llama32_3b.py` | Llama-3.2-3B-Instruct | 3B | LLaMA | 通用 instruction | 跨架构 · 参数量 ×2 |
| `train_and_full_eval_mc_phi35_mini.py` | Phi-3.5-mini-Instruct | 3.8B | Phi | 推理专项优化 | 训练策略差异 |
| `train_and_full_eval_mc_llama31_8b.py` | Llama-3.1-8B-Instruct | 8B | LLaMA | 通用 instruction | 更强基线 · 参数量上限 |

所有四个脚本**完全共用同一套 pipeline**：
- 同样的数据（`prepared_data/train_split.csv` / `prepared_data/eval_split.csv`，25,262 train / 247 eval 的心理学 MC 题）
- 同样的 prompt / chat template / `<answer>X</answer>` 目标格式
- 同样的 LoRA 超参（r=8, alpha=16, dropout=0.05, bias="none"）
- 同样的训练超参（lr=2e-4, epochs=2, warmup 0.03, cosine schedule, weight decay 0.01）
- 同样的评估协议（`max_new_tokens=12`, `do_sample=False`, 训练集采样 1000 条、验证集全量评估）
- 同样的指标（Exact-Match Accuracy + token-level F1）

这样做的目的是：**只让 base model 变化**，其它变量保持严格一致，才能支撑"跨模型泛化失败是否具有普遍性"这一结论。

## 各脚本之间仅有的差异

以下是为了正确运行各模型而**必须**的、最小化的调整：

1. **Phi-3.5-mini**：注意力层使用 **fused `qkv_proj`**（不是拆开的 q/k/v_proj），因此 LoRA `target_modules` 改为 `["qkv_proj", "o_proj"]`；加载时设 `trust_remote_code=True`。
2. **Llama-3.2-3B / Llama-3.1-8B**：LoRA `target_modules` 与 Qwen 相同 `["q_proj","k_proj","v_proj","o_proj"]`；都需要 Hugging Face gated-model 授权。
3. **Llama-3.1-8B**：`gradient_accumulation_steps=16`（其他脚本=8）。这是为了在显存较紧时维持有效 batch 规模，不影响 LoRA 设计或评估协议。

所有其他行——代码结构、函数签名、日志打印、保存路径结构、eval 函数——**一字未改**。reviewer 从 diff 就能看出这是受控变量的扩展。

## 运行前置条件

### 1. GPU

| 模型 | 推荐 VRAM（QLoRA NF4） |
|---|---|
| Llama-3.2-3B | ≥ 12 GB（24 GB 舒适） |
| Phi-3.5-mini (3.8B) | ≥ 16 GB |
| Llama-3.1-8B | ≥ 24 GB（40 GB 舒适） |

### 2. Python 依赖

与已经跑过 Qwen 基线的环境完全一致即可。关键包：
```
torch >= 2.1
transformers >= 4.45    # 已原生支持 Phi-3.5 / Llama-3.1 / Llama-3.2
peft >= 0.11
bitsandbytes >= 0.43
datasets >= 2.19
accelerate >= 0.30
```

### 3. Hugging Face 授权（**重要**）

`meta-llama/Llama-3.2-3B-Instruct` 与 `meta-llama/Llama-3.1-8B-Instruct` 都是 **gated model**。在任何一台新机器上跑之前：

1. 登录 Hugging Face，打开这两个模型页面，点击 "Request access" 并接受 Meta Llama Community License。
2. 在要跑训练的机器上：
   ```bash
   export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   # 或者
   huggingface-cli login
   ```
否则脚本会在 `AutoModelForCausalLM.from_pretrained(...)` 处报 401。

## 运行方式

**分别跑单个模型：**
```bash
python train_and_full_eval_mc_llama32_3b.py
python train_and_full_eval_mc_phi35_mini.py
python train_and_full_eval_mc_llama31_8b.py
```

**一次跑完三个（推荐）：**
```bash
bash run_all_mc_trainings.sh
```
日志写入 `logs/`，最终各模型的评估结果写入：
```
mc_train_and_large_eval_results_llama3p2_3b.txt
mc_train_and_large_eval_results_phi3p5_mini.txt
mc_train_and_large_eval_results_llama3p1_8b.txt
```

## 训练后你会拿到什么

每个脚本完成后会：

1. **保存 LoRA adapter 到本地**：
   - `sft_llama3p2_3b_qlora_safe_resume2/`
   - `sft_phi3p5_mini_qlora_safe_resume2/`
   - `sft_llama3p1_8b_qlora_safe_resume2/`

2. **在终端和日志里打印 trainable parameters**（`model.print_trainable_parameters()` 的输出），例如：
   ```
   trainable params: 1,720,320 || all params: 3,212,749,824 || trainable%: 0.0535
   ```
   这就是"训练后的模型参数"里 reviewer 关心的核心数字——**可训练参数量、总参数量、可训练比例**。

3. **在终端和日志里打印 post-training 评估指标**（和 Qwen 基线同格式）：
   ```
   [FINAL GEN METRICS]
     TRAIN (n=1000): Acc=0.xxxx | F1=0.xxxx
     EVAL  (n=247):  Acc=0.xxxx | F1=0.xxxx
   ```

4. **写出 `mc_train_and_large_eval_results_*.txt`**，可以直接粘进论文 Table。

## 下一步（可选）

你论文里还有 Prompt Constraint / Extraction / Reasoning / Calibration / Error Analysis 五组消融。因为它们**只依赖训练好的 LoRA checkpoint**，把上面三个 `output_dir` 里的 adapter 文件分别喂给你现有的 ablation 脚本即可，实验 2–6 的代码**不需要任何改动**。
