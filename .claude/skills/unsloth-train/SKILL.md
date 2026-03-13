---
name: unsloth-train
description: Combined Unsloth training hub — SFT, GRPO, HPO, DPO, CPT, QAT, and export. Use when the user wants to train, optimize, or export a model and you need to choose or chain the right training approach.
allowed-tools: Bash(uv run *), Bash(torchrun *), Bash(accelerate *), Bash(python *), Bash(make *), Bash(docker *), Bash(ls *), Bash(cat *), Bash(tail *), Read, Grep, Glob
argument-hint: "[sft|grpo|hpo|dpo|cpt|qat|export|compare] [--recipe <yaml>] [--model <id>]"
---

# Unsloth Training Hub

Combined skill for all Unsloth training modes. Choose the right approach based on the argument.

## Arguments: $ARGUMENTS

## Decision Guide

| Goal | Command | When to use |
|------|---------|-------------|
| **SFT** | `/unsloth-train sft` | You have trajectory data and want supervised fine-tuning |
| **GRPO** | `/unsloth-train grpo` | You have gym tasks and want RL-based learning |
| **HPO** | `/unsloth-train hpo` | You want to find optimal hyperparameters before full training |
| **DPO** | `/unsloth-train dpo` | You have preference pairs (chosen/rejected) after SFT |
| **CPT** | `/unsloth-train cpt` | You want domain adaptation on raw text |
| **QAT** | `/unsloth-train qat` | You want quantization-aware training for deployment |
| **Export** | `/unsloth-train export` | You want to convert/export a trained model |
| **Compare** | `/unsloth-train compare` | You want to compare training runs |

---

## SFT — Supervised Fine-Tuning

The standard path. Train on expert trajectories.

```bash
# Quick start
uv run python scripts/08_train_sft_unsloth.py --recipe configs/unsloth/qwen3_8b_lora.yaml

# With 4-bit quantization
uv run python scripts/08_train_sft_unsloth.py --recipe configs/unsloth/qwen3_8b_lora.yaml --four-bit

# Full fine-tune (no LoRA)
uv run python scripts/08_train_sft_unsloth.py --recipe configs/unsloth/qwen3_8b_lora.yaml --full
```

**Data**: Requires `ft_*.jsonl` files in `logs/sft_data/`. Generate with:
```bash
make gen-trajectories && make prepare-sft
```

---

## GRPO — Reinforcement Learning

The "gym" path. Model learns by generating patches and getting rewarded.

```bash
# Default: hybrid rewards (30% format + 70% test)
uv run python scripts/09_train_grpo.py --recipe configs/unsloth/grpo_qwen3_8b.yaml

# From SFT checkpoint (recommended)
uv run python scripts/09_train_grpo.py --from-sft ./sft_output/merged

# Fast iteration with format-only rewards
uv run python scripts/09_train_grpo.py --reward-mode format
```

**Data**: Requires validated tasks in `logs/validated_tasks/`. Generate with:
```bash
make generate-tasks && make validate-tasks
```

**Loss types**: `grpo` (default), `bnpo`, `dr_grpo`, `dapo`, `gspo`

---

## HPO — Hyperparameter Optimization

Find optimal LoRA params before committing to full training.

```bash
# Quick search (10 trials, ~30 min)
uv run python scripts/08b_hpo.py --recipe configs/unsloth/qwen3_8b_lora.yaml

# Thorough search (30 trials, ~2 hrs)
uv run python scripts/08b_hpo.py --recipe configs/unsloth/qwen3_8b_lora.yaml --n-trials 30 --steps-per-trial 100

# Then train with best recipe
uv run python scripts/08_train_sft_unsloth.py --recipe hpo_output/best_recipe.yaml
```

Searches: rank, alpha, LR, weight decay, warmup, scheduler, RSLoRA, grad accum.

---

## DPO — Direct Preference Optimization

Align model to prefer good patches over bad ones. Run after SFT.

```bash
uv run python scripts/08_train_sft_unsloth.py --dpo --sft-checkpoint ./sft_output/
```

---

## CPT — Continued Pretraining

Domain-adapt the model on your repo's codebase before SFT.

```bash
uv run python scripts/08_train_sft_unsloth.py --cpt --data-dir ./corpus/
```

---

## QAT — Quantization-Aware Training

Train with quantization simulation to recover accuracy lost during quantization.

```bash
# INT4 QAT
uv run python scripts/08_train_sft_unsloth.py --recipe configs/unsloth/qwen3_8b_lora.yaml --qat-scheme int4

# FP8-INT4 mixed QAT
uv run python scripts/08_train_sft_unsloth.py --recipe configs/unsloth/qwen3_8b_lora.yaml --qat-scheme fp8-int4 --save-torchao

# GRPO with QAT
uv run python scripts/09_train_grpo.py --recipe configs/unsloth/grpo_qwen3_8b.yaml --qat-scheme int4 --save-torchao
```

Schemes: `int4`, `int8-int4`, `fp8-int4`, `fp8-fp8`

---

## Export — Model Conversion

```bash
# GGUF (for llama.cpp / Ollama)
uv run python scripts/08_train_sft_unsloth.py --recipe ... --save-gguf q4_k_m
uv run python scripts/08_train_sft_unsloth.py --recipe ... --save-gguf Q4_K_XL  # Extended layout

# Push to HuggingFace Hub
uv run python scripts/08_train_sft_unsloth.py --recipe ... --push-to-hub user/model
uv run python scripts/08_train_sft_unsloth.py --recipe ... --push-to-hub-gguf user/model-gguf

# LoRA adapter only (~100MB)
uv run python scripts/08_train_sft_unsloth.py --recipe ... --save-lora-only

# MXFP4 (75% less disk)
uv run python scripts/08_train_sft_unsloth.py --recipe ... --save-mxfp4

# TorchAO (for QAT models)
uv run python scripts/08_train_sft_unsloth.py --recipe ... --save-torchao
```

---

## Compare — Inspect Training Runs

```bash
# Check latest SFT output
ls -la sft_output/

# Check latest GRPO output
ls -la grpo_output/

# Check HPO results
cat hpo_output/hpo_results.json | python -m json.tool

# Compare WandB runs
# Visit your WandB project dashboard
```

---

## Recommended Training Pipelines

### Pipeline A: SFT Only (fastest)
```bash
make gen-trajectories
make prepare-sft
uv run python scripts/08_train_sft_unsloth.py --recipe configs/unsloth/qwen3_8b_lora.yaml --four-bit
```

### Pipeline B: HPO → SFT (best quality)
```bash
make gen-trajectories && make prepare-sft
uv run python scripts/08b_hpo.py --recipe configs/unsloth/qwen3_8b_lora.yaml --n-trials 20
uv run python scripts/08_train_sft_unsloth.py --recipe hpo_output/best_recipe.yaml
```

### Pipeline C: SFT → DPO (alignment)
```bash
# SFT first
uv run python scripts/08_train_sft_unsloth.py --recipe configs/unsloth/qwen3_8b_lora.yaml
# Then DPO
uv run python scripts/08_train_sft_unsloth.py --dpo --sft-checkpoint ./sft_output/
```

### Pipeline D: SFT → GRPO (RL fine-tuning)
```bash
# SFT for initial policy
uv run python scripts/08_train_sft_unsloth.py --recipe configs/unsloth/qwen3_8b_lora.yaml
# GRPO from SFT checkpoint
uv run python scripts/09_train_grpo.py --from-sft ./sft_output/merged --reward-mode hybrid
```

### Pipeline E: GRPO from scratch (pure RL)
```bash
make generate-tasks && make validate-tasks
uv run python scripts/09_train_grpo.py --recipe configs/unsloth/grpo_qwen3_8b.yaml
```

### Pipeline F: CPT → SFT → GRPO (full stack)
```bash
# Domain adapt
uv run python scripts/08_train_sft_unsloth.py --cpt --data-dir ./corpus/
# SFT on top
uv run python scripts/08_train_sft_unsloth.py --recipe configs/unsloth/qwen3_8b_lora.yaml --model ./sft_output/merged
# RL polish
uv run python scripts/09_train_grpo.py --from-sft ./sft_output/merged
```

---

## Available Recipes

### SFT Recipes (`configs/unsloth/`)
| Recipe | Model | Notes |
|--------|-------|-------|
| `qwen3_8b_lora.yaml` | Qwen3 8B | Default, recommended |
| `qwen3_4b_lora.yaml` | Qwen3 4B | Lighter |
| `qwen3_32b_lora.yaml` | Qwen3 32B | Needs multi-GPU or 4-bit |
| `qwen3_coder_next_lora.yaml` | Qwen3 Coder Next | Large, needs device-map |
| `llama3_8b_lora.yaml` | Llama 3 8B | Needs --token |
| `nemotron3_nano_lora.yaml` | Nemotron3 Nano | Small |
| `nemotron3_super_lora.yaml` | Nemotron3 Super | Medium |
| `gpt_oss_20b_lora.yaml` | GPT-OSS 20B | Large |
| `gpt_oss_120b_lora.yaml` | GPT-OSS 120B | Multi-GPU required |

### GRPO Recipes
| Recipe | Model | Notes |
|--------|-------|-------|
| `grpo_qwen3_8b.yaml` | Qwen3 8B | Default |
| `grpo_qwen3_4b.yaml` | Qwen3 4B | Lighter |
| `grpo_qwen3_coder_next.yaml` | Qwen3 Coder Next | Multi-GPU |
| `grpo_nemotron3_nano.yaml` | Nemotron3 Nano | Small |
| `grpo_nemotron3_super.yaml` | Nemotron3 Super | Medium |
| `grpo_gpt_oss_120b.yaml` | GPT-OSS 120B | Multi-GPU |

---

## VRAM Guide

| Model Size | 4-bit | FP8 | 16-bit | Recommended |
|------------|-------|-----|--------|-------------|
| 4B | ~4GB | ~6GB | ~10GB | Single GPU, any |
| 8B | ~8GB | ~12GB | ~20GB | Single GPU, RTX 3090+ |
| 20B | ~16GB | ~24GB | ~48GB | Single GPU 4-bit, or multi-GPU |
| 32B | ~20GB | ~32GB | ~64GB | Multi-GPU or 4-bit on 24GB |
| 120B | ~48GB | ~80GB | ~240GB | Multi-GPU required |

Add `--offload-embedding` for ~1GB extra savings. Add `--vllm-standby` for ~9GB savings (GRPO only).
