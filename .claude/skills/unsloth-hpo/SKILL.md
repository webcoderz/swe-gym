---
name: unsloth-hpo
description: Run hyperparameter optimization for LoRA fine-tuning using Optuna. Use when the user wants to find optimal LoRA hyperparameters before full training.
allowed-tools: Bash(uv run *), Bash(python *), Bash(pip *), Bash(ls *), Bash(cat *), Bash(tail *), Read, Grep, Glob
argument-hint: "[--recipe <yaml>] [--n-trials 10|30] [--four-bit] [--pruning]"
---

# Unsloth HPO (Hyperparameter Optimization)

Finds optimal LoRA hyperparameters via Optuna before committing to a full training run.
Script: `scripts/08b_hpo.py`

## Arguments: $ARGUMENTS

## Quick Start

```bash
# Quick HPO (10 trials, ~30 min on single GPU)
uv run python scripts/08b_hpo.py --recipe configs/unsloth/qwen3_8b_lora.yaml

# Thorough search (30 trials, ~2 hrs)
uv run python scripts/08b_hpo.py --recipe configs/unsloth/qwen3_8b_lora.yaml --n-trials 30 --steps-per-trial 100

# With 4-bit quantization
uv run python scripts/08b_hpo.py --recipe configs/unsloth/qwen3_8b_lora.yaml --four-bit
```

## What It Searches

The HPO script optimizes these hyperparameters:

| Parameter | Default Search Space | Description |
|-----------|---------------------|-------------|
| `lora_rank` | 16, 32, 64, 128 | LoRA rank (capacity) |
| `alpha_strategy` | equal, double | Alpha = rank or rank*2 |
| `lr` | 5e-6 to 5e-4 (log) | Learning rate |
| `weight_decay` | 0.0 to 0.1 | Regularization |
| `warmup_ratio` | 0.0 to 0.15 | Warmup fraction |
| `lr_scheduler` | cosine, linear | LR schedule |
| `use_rslora` | True, False | Rank-Stabilized LoRA |
| `grad_accum` | 4, 8, 16 | Gradient accumulation |

## Custom Search Space

Override the defaults with comma-separated values:

```bash
# Custom ranks
uv run python scripts/08b_hpo.py --recipe configs/unsloth/qwen3_8b_lora.yaml --ranks 16,32,64

# Custom learning rates
uv run python scripts/08b_hpo.py --recipe configs/unsloth/qwen3_8b_lora.yaml --lrs 1e-4,2e-4,5e-4

# Custom alpha (fixed values instead of strategy)
uv run python scripts/08b_hpo.py --recipe configs/unsloth/qwen3_8b_lora.yaml --alphas 32,64,128

# Alpha strategies
uv run python scripts/08b_hpo.py --recipe configs/unsloth/qwen3_8b_lora.yaml --alphas equal
uv run python scripts/08b_hpo.py --recipe configs/unsloth/qwen3_8b_lora.yaml --alphas double
```

## All CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--recipe <yaml>` | *required* | Base YAML recipe to optimize |
| `--data-dir <path>` | auto-detect | Training data directory |
| `--output-dir <path>` | `./hpo_output` | Output for trials + best recipe |
| `--n-trials <int>` | 10 | Number of Optuna trials |
| `--steps-per-trial <int>` | 50 | Training steps per trial |
| `--eval-split <float>` | 0.15 | Fraction held out for eval |
| `--ranks <list>` | 16,32,64,128 | LoRA ranks to search |
| `--lrs <list>` | 5e-6..5e-4 | Learning rates to search |
| `--alphas <val>` | equal/double | Alpha strategy or fixed values |
| `--four-bit` | off | Use 4-bit quantization |
| `--fp8` | off | Use FP8 quantization |
| `--device-map <map>` | None | Device map for model splitting |
| `--pruning` | off | Enable median pruning (stop bad trials early) |
| `--seed <int>` | 42 | Random seed |
| `--no-wandb` | off | Disable WandB |

## How It Works

1. Loads your base recipe YAML as the starting configuration
2. For each trial, Optuna samples a combination of hyperparameters
3. Runs a short training session (default: 50 steps) with an eval split
4. Measures eval loss as the objective to minimize
5. Reports best params and saves `hpo_output/best_recipe.yaml`

With `--pruning`, Optuna uses median pruning to stop unpromising trials early, saving GPU time.

## Output Files

```
hpo_output/
├── best_recipe.yaml      # Best hyperparameters as a ready-to-use recipe
├── hpo_results.json      # All trial results with params and eval_loss
└── trial_000/            # Per-trial training artifacts
    trial_001/
    ...
```

## Using Results

After HPO completes, train with the best recipe:

```bash
# Use the optimized recipe for full training
uv run python scripts/08_train_sft_unsloth.py --recipe hpo_output/best_recipe.yaml

# Or with additional overrides
uv run python scripts/08_train_sft_unsloth.py --recipe hpo_output/best_recipe.yaml --four-bit --epochs 3
```

## Recommended Workflow

1. **Quick pass** — 10 trials, 50 steps each (~30 min):
   ```bash
   uv run python scripts/08b_hpo.py --recipe configs/unsloth/qwen3_8b_lora.yaml
   ```

2. **Narrow search** — fix ranks from step 1, explore LR more:
   ```bash
   uv run python scripts/08b_hpo.py --recipe configs/unsloth/qwen3_8b_lora.yaml \
     --ranks 64 --n-trials 15 --steps-per-trial 100
   ```

3. **Full train** with best recipe:
   ```bash
   uv run python scripts/08_train_sft_unsloth.py --recipe hpo_output/best_recipe.yaml
   ```

## Troubleshooting

- **OOM during trials**: Use `--four-bit` or reduce `--steps-per-trial`
- **Trials all failing**: Check that training data exists in `logs/sft_data/` — run `make prepare-sft` first
- **Need Optuna**: Included in `[unsloth]` extras — `uv pip install -e ".[unsloth]"`
