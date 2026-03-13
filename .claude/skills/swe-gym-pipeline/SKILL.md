---
name: swe-gym-pipeline
description: Run the SWE-Gym training pipeline — skill learning, SFT, GRPO, or the full pipeline. Use when the user wants to train skills, fine-tune models, generate tasks, or run the gym pipeline.
allowed-tools: Bash(make *), Bash(./scripts/*), Bash(uv run *), Bash(torchrun *), Bash(docker *), Bash(ls *), Bash(cat *), Bash(grep *), Bash(test *), Read, Grep, Glob
argument-hint: "[smoke-test|build-env|generate-tasks|validate-tasks|train|sft|grpo|evaluate|deploy-skills|resume|status|full]"
---

# SWE-Gym Training Pipeline

Orchestrates the full SWE-Gym pipeline: task generation, skill optimization, SFT/GRPO fine-tuning, and deployment.

## Arguments: $ARGUMENTS

## Pre-Flight Checks

Before any step, verify:

1. Docker is running: `docker ps`
2. `.env` exists with `OPENAI_API_KEY` (for GEPA skill training)
3. `repo.conf` has `REPO_OWNER` and `REPO_NAME` set

## Pipeline Steps

Based on the argument (`$ARGUMENTS`), run the appropriate step:

### `smoke-test` — Quick validation (~5 min, ~$0.50)
```bash
make smoke-test
```
Runs 3 tasks to verify the pipeline works end-to-end.

### `build-env` — Step 1: Docker environment (~10-30 min)
```bash
make build-env
```
Builds Docker image with target repo + deps. Verify: `docker images | grep swesmith`

### `generate-tasks` — Step 2: Synthesize bugs (~20-60 min)
```bash
make generate-tasks
```
Creates bug tasks from commit history. Check: `ls logs/tasks/`

### `validate-tasks` — Step 3: Filter tasks (~30-60 min)
```bash
make validate-tasks
```
Runs test harness to keep only valid tasks. Check: `ls logs/validated_tasks/`

### `train` — Step 4: GEPA skill optimization (~2-12 hrs, ~$30-80)
```bash
make train
```
Runs the GEPA gskill optimization loop. Monitor: `tail -f gepa_results/logs/run_*/terminal.log`

### `evaluate` — Step 5: Compare baseline vs skills
```bash
make evaluate
```

### `deploy-skills` — Step 6: Deploy learned skills
```bash
make deploy-skills
```
Copies `best_skills.txt` to `.claude/skills/REPO_NAME/SKILL.md`

### `sft` — Fine-tune with SFT (after steps 6-7)
Requires trajectory data in `logs/sft_data/`. Run steps 6-7 first if missing:
```bash
make gen-trajectories
make prepare-sft
```
Then train (default Qwen3-8B LoRA):
```bash
make train-sft-unsloth
```
Or with a specific recipe:
```bash
uv run python scripts/08_train_sft_unsloth.py --recipe configs/unsloth/qwen3_8b_lora.yaml --four-bit
```
See `configs/unsloth/` for all available recipes. Check `make help` for targets.

### `grpo` — GRPO RL training (after step 3)
Requires validated tasks. The model learns by generating patches and getting rewarded:
```bash
make train-grpo
```
Or with specific recipe/options:
```bash
uv run python scripts/09_train_grpo.py --recipe configs/unsloth/grpo_qwen3_8b.yaml --reward-mode hybrid
```
Monitor rewards — expect 300+ steps before they start increasing.

### `hpo` — Hyperparameter optimization
```bash
make hpo                    # 10 trials, ~30 min
make hpo-thorough           # 30 trials, ~2 hrs
```
Outputs `hpo_output/best_recipe.yaml` for use with SFT training.

### `resume` — Continue interrupted training
```bash
make resume
```

### `status` — Check current run
Show the latest run directory, logs, and results:
```bash
RUN_DIR=$(ls -td gepa_results/logs/run_* 2>/dev/null | head -1)
echo "Latest run: $RUN_DIR"
[ -n "$RUN_DIR" ] && tail -20 "$RUN_DIR/terminal.log"
```

### `full` or empty — Run complete pipeline
Run steps 1-6 sequentially:
```bash
make build-env && make generate-tasks && make validate-tasks && make train && make evaluate && make deploy-skills
```
This takes 3-15 hours depending on repo size.

## After Each Step

Always verify the step succeeded by checking for expected outputs before proceeding to the next step. If a step fails, check logs and report the error.

## Key Files

- `repo.conf` — Target repo configuration
- `.env` — API keys
- `configs/unsloth/` — Training recipe YAML files
- `logs/` — Pipeline logs (tasks, trajectories, SFT data)
- `gepa_results/` — GEPA training results
- `sft_output/` — SFT model output
- `grpo_output/` — GRPO model output
