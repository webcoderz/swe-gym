---
name: unsloth-sft
description: Run Unsloth SFT/DPO/CPT fine-tuning. Use when the user wants to train, fine-tune, or adapt a model with LoRA, full fine-tune, DPO, continued pretraining, or QAT using Unsloth.
allowed-tools: Bash(uv run *), Bash(torchrun *), Bash(accelerate *), Bash(python *), Bash(ls *), Bash(cat *), Bash(tail *), Read, Grep, Glob
argument-hint: "[--recipe <yaml>] [--model <id>] [--four-bit|--fp8] [--dpo|--cpt|--full] [--qat-scheme int4]"
---

# Unsloth SFT/DPO/CPT Training

Fine-tune any HuggingFace model using Unsloth's optimized training.
Script: `scripts/08_train_sft_unsloth.py`

## Arguments: $ARGUMENTS

## Quick Start

```bash
# LoRA SFT (default, recommended)
uv run python scripts/08_train_sft_unsloth.py --recipe configs/unsloth/qwen3_8b_lora.yaml

# With 4-bit quantization (minimal VRAM)
uv run python scripts/08_train_sft_unsloth.py --recipe configs/unsloth/qwen3_8b_lora.yaml --four-bit

# FP8 quantization (RTX 40/50, H100+)
uv run python scripts/08_train_sft_unsloth.py --recipe configs/unsloth/qwen3_8b_lora.yaml --fp8
```

## Training Modes

### LoRA SFT (default)
```bash
uv run python scripts/08_train_sft_unsloth.py --recipe configs/unsloth/qwen3_8b_lora.yaml
```

### Full Fine-Tune
```bash
uv run python scripts/08_train_sft_unsloth.py --recipe configs/unsloth/qwen3_8b_lora.yaml --full
```

### DPO (after SFT)
```bash
uv run python scripts/08_train_sft_unsloth.py --dpo --sft-checkpoint ./sft_output/
```

### Continued Pretraining (domain adaptation)
```bash
uv run python scripts/08_train_sft_unsloth.py --cpt --data-dir ./corpus/
```

## Multi-GPU

```bash
# DDP (one model copy per GPU, distinct samples)
torchrun --nproc_per_node=2 scripts/08_train_sft_unsloth.py --recipe configs/unsloth/qwen3_8b_lora.yaml

# Model splitting (for 70B+ models that don't fit on one GPU)
uv run python scripts/08_train_sft_unsloth.py --recipe configs/unsloth/qwen3_coder_next_lora.yaml --device-map balanced
```

## Quantization Options

| Flag | VRAM Savings | Hardware | Notes |
|------|-------------|----------|-------|
| `--four-bit` | ~75% | Any GPU | QLoRA, minimal VRAM |
| `--eight-bit` | ~50% | Any GPU | Better quality than 4-bit |
| `--fp8` | ~60% | RTX 40/50, H100+ | 1.4x faster training |

## QAT (Quantization-Aware Training)

Recovers ~70% of accuracy lost during quantization via TorchAO:

```bash
# INT4 QAT (most common)
uv run python scripts/08_train_sft_unsloth.py --recipe configs/unsloth/qwen3_8b_lora.yaml --qat-scheme int4

# FP8-INT4 mixed QAT
uv run python scripts/08_train_sft_unsloth.py --recipe configs/unsloth/qwen3_8b_lora.yaml --qat-scheme fp8-int4 --save-torchao
```

Schemes: `int4`, `int8-int4`, `fp8-int4`, `fp8-fp8`

## Save & Export Options

```bash
# GGUF export (for llama.cpp / Ollama)
--save-gguf q4_k_m          # Standard 4-bit GGUF
--save-gguf q8_0             # 8-bit GGUF
--save-gguf Q4_K_XL          # Extended layout (better quality)

# Other formats
--save-lora-only             # Save only LoRA adapter (~100MB)
--save-mxfp4                 # MXFP4 (75% less disk)
--save-torchao               # TorchAO format (for QAT models)

# Push to HuggingFace Hub
--push-to-hub user/model-name
--push-to-hub-gguf user/model-gguf
```

## All CLI Flags

### Model & Data
| Flag | Description |
|------|-------------|
| `--recipe <yaml>` | YAML recipe config (e.g. `configs/unsloth/qwen3_8b_lora.yaml`) |
| `--model <id>` | HuggingFace model ID (overrides recipe) |
| `--data-dir <path>` | Directory with `ft_*.jsonl` files (default: auto-detect) |
| `--output-dir <path>` | Output directory (default: `./sft_output`) |
| `--token <token>` | HF token for gated models (Llama, Gemma); or set `HF_TOKEN` env var |
| `--trust-remote-code` | Trust remote code (needed for some architectures) |
| `--revision <rev>` | Model revision (commit hash, branch, tag) |

### Training Mode
| Flag | Description |
|------|-------------|
| `--full` | Full fine-tune instead of LoRA |
| `--dpo` | DPO training (requires `--sft-checkpoint`) |
| `--cpt` | Continued pretraining (domain adaptation) |
| `--sft-checkpoint <path>` | Path to SFT checkpoint for DPO |

### Quantization
| Flag | Description |
|------|-------------|
| `--four-bit` | QLoRA 4-bit quantization |
| `--eight-bit` | 8-bit quantization |
| `--fp8` | FP8 quantization (RTX 40/50, H100+) |
| `--qat-scheme <scheme>` | QAT: `int4`, `int8-int4`, `fp8-int4`, `fp8-fp8` |

### Hyperparameters
| Flag | Description |
|------|-------------|
| `--lora-rank <int>` | LoRA rank (8, 16, 32, 64, 128, 256) |
| `--lr <float>` | Learning rate |
| `--epochs <int>` | Training epochs |
| `--batch-size <int>` | Per-device batch size |
| `--grad-accum <int>` | Gradient accumulation steps |
| `--max-seq-len <int>` | Maximum sequence length |
| `--max-steps <int>` | Max steps (overrides epochs) |

### Features
| Flag | Description |
|------|-------------|
| `--train-on-completions` | Loss on assistant responses only (~1% boost) |
| `--packing` | Pack multiple examples per sequence |
| `--use-rslora` | Rank-Stabilized LoRA (recommended for rank >= 64) |
| `--offload-embedding` | Offload embeddings (save ~1GB VRAM) |
| `--tiled-mlp` | Tiled MLP for 500k+ context (60% less VRAM) |
| `--device-map <map>` | Model splitting: `balanced`, `auto` |

### Save & Export
| Flag | Description |
|------|-------------|
| `--save-gguf <quant>` | GGUF quantization (`q4_k_m`, `q8_0`, `f16`, `Q4_K_XL`) |
| `--save-lora-only` | Save only LoRA adapter (~100MB) |
| `--save-mxfp4` | Save as MXFP4 (75% less disk) |
| `--save-torchao` | Save via TorchAO (for QAT models) |
| `--push-to-hub <repo>` | Push merged model to HuggingFace Hub |
| `--push-to-hub-gguf <repo>` | Push GGUF to HuggingFace Hub |

### Eval & Logging
| Flag | Description |
|------|-------------|
| `--early-stopping <n>` | Patience for early stopping |
| `--eval-split <frac>` | Eval holdout fraction |
| `--resume [path]` | Resume from checkpoint (latest or specific) |
| `--wandb-project <name>` | WandB project name |
| `--no-wandb` | Disable WandB logging |

## YAML Recipe

All flags can also be set in a YAML recipe file. See `configs/unsloth/` for examples.

Key recipe fields beyond CLI flags:
- `lora_targets` — list of modules to apply LoRA to
- `lora_dropout` — dropout rate (0 recommended with Unsloth)
- `bias` — "none" (faster) or "all"
- `loftq_config` — LoftQ initialization
- `init_lora_weights` — True, False, or "gaussian"
- `layers_to_transform` — list of layer indices for selective LoRA
- `modules_to_save` — extra modules to save (e.g. `["lm_head", "embed_tokens"]` for CPT)
- `gradient_checkpointing` — "unsloth" (30% extra savings), true, false
- `warmup_ratio` — alternative to warmup_steps
- `lr_scheduler` — "cosine", "linear", "constant"
- `optim` — "adamw_8bit", "adamw_torch", "sgd"
- `resize_model_vocab` — resize vocab for custom special tokens
- `reasoning_effort` — for GPT-OSS models: "low", "medium", "high"

## Available Recipes

SFT recipes in `configs/unsloth/`:
- `qwen3_8b_lora.yaml` — Qwen3 8B LoRA (default)
- `qwen3_4b_lora.yaml` — Qwen3 4B LoRA (lighter)
- `qwen3_32b_lora.yaml` — Qwen3 32B LoRA (needs multi-GPU or 4-bit)
- `qwen3_coder_next_lora.yaml` — Qwen3 Coder Next (large, needs device-map)
- `llama3_8b_lora.yaml` — Llama 3 8B LoRA (needs --token)
- `nemotron3_nano_lora.yaml` — Nemotron3 Nano LoRA
- `nemotron3_super_lora.yaml` — Nemotron3 Super LoRA
- `gpt_oss_20b_lora.yaml` — GPT-OSS 20B LoRA
- `gpt_oss_120b_lora.yaml` — GPT-OSS 120B LoRA (multi-GPU required)

## Troubleshooting

- **OOM**: Try `--four-bit`, reduce `--batch-size 1`, reduce `--max-seq-len`, or add `--offload-embedding`
- **Slow training**: Try `--fp8` on RTX 40/50+, increase `--grad-accum` to compensate for smaller batch
- **Gated model 403**: Set `--token` or `HF_TOKEN` env var, accept model license on HuggingFace
- **QAT export**: Use `--save-torchao` to save QAT models (standard save loses quantization info)
