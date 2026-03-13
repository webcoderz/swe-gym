---
name: unsloth-grpo
description: Run GRPO reinforcement learning training on gym tasks. Use when the user wants to train a model with RL rewards from test suites, format checks, or hybrid scoring.
allowed-tools: Bash(uv run *), Bash(torchrun *), Bash(accelerate *), Bash(python *), Bash(docker *), Bash(ls *), Bash(cat *), Bash(tail *), Read, Grep, Glob
argument-hint: "[--recipe <yaml>] [--model <id>] [--reward-mode test|format|hybrid] [--four-bit] [--from-sft <path>]"
---

# Unsloth GRPO (RL Training)

Train a model with reinforcement learning — the model generates patches, gets rewarded for passing tests.
Script: `scripts/09_train_grpo.py`

## Arguments: $ARGUMENTS

## Prerequisites

- Step 3 completed (validated task instances in `logs/validated_tasks/`)
- Docker running (`docker ps`)
- GPU with enough VRAM (4-bit: ~8GB for 4B, ~16GB for 8B, ~24GB for 30B MoE)

## Quick Start

```bash
# Default: Qwen3-8B with hybrid rewards
uv run python scripts/09_train_grpo.py --recipe configs/unsloth/grpo_qwen3_8b.yaml

# Lighter model
uv run python scripts/09_train_grpo.py --model Qwen/Qwen3-4B

# From an SFT checkpoint
uv run python scripts/09_train_grpo.py --from-sft ./sft_output/merged

# Disable 4-bit (use full precision)
uv run python scripts/09_train_grpo.py --recipe configs/unsloth/grpo_qwen3_8b.yaml --no-four-bit
```

## Reward Modes

| Mode | Speed | Quality | Description |
|------|-------|---------|-------------|
| `test` | Slow | Best | Full Docker test suite evaluation |
| `format` | Fast | Lower | Quick format/syntax check |
| `hybrid` | Medium | Good | 30% format + 70% test (default) |

```bash
uv run python scripts/09_train_grpo.py --reward-mode test     # Most accurate
uv run python scripts/09_train_grpo.py --reward-mode format   # Fastest iteration
uv run python scripts/09_train_grpo.py --reward-mode hybrid   # Default balance
```

## Loss Types

| Type | Description |
|------|-------------|
| `grpo` | Standard GRPO (default) |
| `bnpo` | Bounded NPO |
| `dr_grpo` | Dynamic-ratio GRPO |
| `dapo` | DAPO (Dynamic Advantage Policy Optimization) |
| `gspo` | GSPO (Group Sequence Policy Optimization) |

```bash
uv run python scripts/09_train_grpo.py --loss-type dr_grpo
```

## Multi-GPU

```bash
# DDP (one model copy per GPU)
torchrun --nproc_per_node=2 scripts/09_train_grpo.py --recipe configs/unsloth/grpo_qwen3_8b.yaml

# Model splitting (for large models)
uv run python scripts/09_train_grpo.py --recipe configs/unsloth/grpo_qwen3_coder_next.yaml --device-map balanced
```

## Memory Optimization

```bash
# Stack optimizations for minimal VRAM
uv run python scripts/09_train_grpo.py \
  --recipe configs/unsloth/grpo_qwen3_8b.yaml \
  --four-bit \
  --offload-embedding \
  --vllm-standby \
  --float8-kv-cache
```

| Flag | Savings | Notes |
|------|---------|-------|
| `--four-bit` | ~75% VRAM | QLoRA 4-bit |
| `--fp8` | ~60% VRAM | RTX 40/50, H100+ |
| `--offload-embedding` | ~1GB | Moves embeddings to CPU |
| `--vllm-standby` | ~9GB | Shares weight memory between train/inference |
| `--float8-kv-cache` | 2x KV cache | RTX 3090+/A100+ |
| `--tiled-mlp` | 60% less VRAM | Enables 500k+ context on 80GB |

## QAT (Quantization-Aware Training)

```bash
uv run python scripts/09_train_grpo.py --recipe configs/unsloth/grpo_qwen3_8b.yaml \
  --qat-scheme int4 --save-torchao
```

Schemes: `int4`, `int8-int4`, `fp8-int4`, `fp8-fp8`

## All CLI Flags

### Model & Data
| Flag | Description |
|------|-------------|
| `--recipe <yaml>` | YAML recipe config |
| `--model <id>` | HuggingFace model ID or local path |
| `--from-sft <path>` | Start from SFT checkpoint |
| `--output-dir <path>` | Output directory (default: `./grpo_output`) |
| `--task-dir <path>` | Validated task instances (default: auto-detect) |
| `--token <token>` | HF token for gated models |
| `--trust-remote-code` | Trust remote code |
| `--revision <rev>` | Model revision |

### Quantization
| Flag | Description |
|------|-------------|
| `--four-bit` | 4-bit quantization (default for GRPO) |
| `--no-four-bit` | Disable 4-bit, use full precision |
| `--eight-bit` | 8-bit quantization |
| `--fp8` | FP8 quantization |
| `--qat-scheme <scheme>` | QAT: `int4`, `int8-int4`, `fp8-int4`, `fp8-fp8` |

### Training
| Flag | Description |
|------|-------------|
| `--lora-rank <int>` | LoRA rank |
| `--lr <float>` | Learning rate |
| `--steps <int>` | Training steps |
| `--batch-size <int>` | Problems per batch |
| `--num-generations <int>` | Completions per problem |
| `--max-seq-len <int>` | Maximum sequence length |
| `--max-completion-len <int>` | Max tokens for completion |

### Reward & Loss
| Flag | Description |
|------|-------------|
| `--reward-mode <mode>` | `test`, `format`, `hybrid` |
| `--loss-type <type>` | `grpo`, `bnpo`, `dr_grpo`, `dapo`, `gspo` |

### Memory & Performance
| Flag | Description |
|------|-------------|
| `--device-map <map>` | Model splitting: `balanced`, `auto` |
| `--use-rslora` | Rank-Stabilized LoRA |
| `--offload-embedding` | Offload embeddings (~1GB savings) |
| `--vllm-standby` | vLLM standby mode (~9GB savings) |
| `--float8-kv-cache` | 2x KV cache reduction |
| `--tiled-mlp` | Tiled MLP for 500k+ context |

### Save & Export
| Flag | Description |
|------|-------------|
| `--save-gguf <quant>` | GGUF quantization (`q4_k_m`, `q8_0`, `Q4_K_XL`) |
| `--save-lora-only` | Save only LoRA adapter |
| `--save-mxfp4` | Save as MXFP4 |
| `--save-torchao` | Save via TorchAO (for QAT) |
| `--push-to-hub <repo>` | Push to HuggingFace Hub |
| `--push-to-hub-gguf <repo>` | Push GGUF to Hub |

### Logging & Checkpoints
| Flag | Description |
|------|-------------|
| `--resume [path]` | Resume from checkpoint |
| `--wandb-project <name>` | WandB project name |
| `--no-wandb` | Disable WandB |

## YAML Recipe — GRPO-Specific Fields

Beyond the standard LoRA/model fields, GRPO recipes support:

```yaml
# ── GRPO loss & clipping ──
loss_type: grpo             # grpo, bnpo, dr_grpo, dapo, gspo
beta: 0.0                   # KL penalty (0 = no reference model, saves memory)
num_iterations: 1           # PPO epochs per batch
epsilon: 0.2                # PPO clipping epsilon
epsilon_high: null           # One-sided high clip (DAPO uses 0.28)
delta: null                  # Two-sided clip (recommended > 1+epsilon)
mask_truncated_completions: true
importance_sampling_level: token  # "token" (GRPO) or "sequence" (GSPO)
scale_rewards: group         # "group", "batch", or "none"

# ── vLLM inference ──
temperature: 1.5
min_p: 0.1
gpu_memory_utilization: 0.6  # Set 0.95 with vllm_standby
vllm_standby: false
float8_kv_cache: false

# ── Off-policy corrections ──
vllm_importance_sampling_correction: true
vllm_importance_sampling_cap: 2.0

# ── Reward hacking countermeasures ──
reward_stdlib_only: false    # Penalize non-stdlib imports
reward_strip_globals: false  # Strip global-scope mutations
reward_max_cache_mb: null    # Cache thrashing limit
reward_timeout: null         # Per-evaluation timeout (seconds)

# ── Long-context memory optimization ──
unsloth_grpo_mini_batch: null    # Batch dim chunking
unsloth_logit_chunk_multiplier: null  # Seq dim chunking

# ── MoE backend ──
moe_backend: null            # "grouped_mm", "unsloth_triton", "native_torch"
```

## Available GRPO Recipes

- `grpo_qwen3_8b.yaml` — Qwen3 8B (default)
- `grpo_qwen3_4b.yaml` — Qwen3 4B (lighter)
- `grpo_qwen3_coder_next.yaml` — Qwen3 Coder Next (multi-GPU)
- `grpo_nemotron3_nano.yaml` — Nemotron3 Nano
- `grpo_nemotron3_super.yaml` — Nemotron3 Super
- `grpo_gpt_oss_120b.yaml` — GPT-OSS 120B (multi-GPU)

## Training Tips

- **Expect 300+ steps** before rewards start increasing — GRPO is slow to warm up
- **Monitor rewards**: `--wandb-project swe-gym` to track reward curves
- **Hybrid mode** is recommended for balancing speed and signal quality
- **vLLM standby** is critical for VRAM savings when using large models
- **From SFT**: Starting from an SFT checkpoint gives better initial policy than base model
