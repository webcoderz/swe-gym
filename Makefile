# SWE-Smith + GEPA gskill gym targets
# Run from swe-gym/ directory or via root Makefile (make swe-gym-*)

.PHONY: help setup build-env generate-tasks validate-tasks train evaluate \
	smoke-test deploy-skills gen-trajectories prepare-sft train-sft \
	train-sft-unsloth train-dpo-unsloth train-grpo train-grpo-gptoss-120b \
	train-sft-unsloth-gptoss-120b train-sft-fp8 train-grpo-fp8 \
	train-sft-nemotron-super train-grpo-nemotron-super hpo hpo-thorough clean

# Load repo config (REPO_OWNER, REPO_NAME, REPO_KEY, etc.)
include repo.conf
export REPO_OWNER REPO_NAME REPO_KEY

# Override HF_HOME so the project .env (HF_HOME=/models/hf) doesn't interfere
export HF_HOME := $(HOME)/.cache/huggingface

help:
	@echo "SWE-Smith RL Gym — targeting $(REPO_OWNER)/$(REPO_NAME)"
	@echo ""
	@echo "Setup:"
	@echo "  make setup            Install swesmith + gepa[gskill] + deps"
	@echo ""
	@echo "Pipeline (run in order):"
	@echo "  make build-env        Step 1: Build Docker execution environment"
	@echo "  make generate-tasks   Step 2: Synthesize bug tasks from commits"
	@echo "  make validate-tasks   Step 3: Filter tasks via test harness"
	@echo "  make train            Step 4: GEPA gskill optimization loop"
	@echo "  make evaluate         Step 5: Evaluate with/without learned skills"
	@echo ""
	@echo "SFT Pipeline (after step 3):"
	@echo "  make gen-trajectories    Step 6: Generate expert solve traces"
	@echo "  make prepare-sft         Step 7: Evaluate & convert to JSONL"
	@echo "  make train-sft           Step 8: torchtune fine-tune (legacy)"
	@echo "  make train-sft-unsloth   Step 8: Unsloth LoRA fine-tune (recommended)"
	@echo "  make train-dpo-unsloth   DPO alignment (after SFT)"
	@echo "  make train-cpt-unsloth   Continued pretraining (domain adaptation)"
	@echo ""
	@echo "GRPO RL (after step 3):"
	@echo "  make train-grpo          Step 9: GRPO RL with gym tasks as reward"
	@echo "  make train-grpo-test     Full Docker test suite reward"
	@echo "  make train-grpo-fast     Format-only reward (fast iteration)"
	@echo ""
	@echo "Multi-GPU:"
	@echo "  make train-sft-multigpu  DDP LoRA (2 GPUs)"
	@echo "  make train-grpo-multigpu DDP GRPO (2 GPUs)"
	@echo ""
	@echo "Shortcuts:"
	@echo "  make smoke-test       Quick 3-task validation run"
	@echo "  make deploy-skills    Copy best_skills.txt to .claude/skills/"
	@echo "  make train-recipe RECIPE=configs/unsloth/xxx.yaml"
	@echo "  make clean            Remove generated logs and results"

setup:
	uv pip install -e ".[dev]"
	@echo ""
	@echo "Verify Docker: docker ps"
	@echo "Set OPENAI_API_KEY before running pipeline."

build-env:
	./scripts/01_build_env.sh

build-env-force:
	./scripts/01_build_env.sh --force

generate-tasks:
	./scripts/02_generate_tasks.sh

generate-tasks-from:
	@test -n "$(FROM)" || (echo "Usage: make generate-tasks-from FROM=2d"; exit 1)
	./scripts/02_generate_tasks.sh --from $(FROM)


validate-tasks:
	./scripts/03_validate_tasks.sh

train:
	./scripts/04_train_skills.sh

smoke-test:
	./scripts/04_train_skills.sh --smoke-test

# Smoke test using a repo already in SWE-smith (no local task gen needed)
smoke-test-demo:
	uv run python -m gepa.gskill.gskill.train_optimize_anything \
		--repo pygments__pygments --smoke-test --model gpt-4o-mini \
		--reflection-model gpt-4o --workers 2

resume:
	@RUN_DIR=$$(ls -td gepa_results/logs/run_* 2>/dev/null | head -1); \
	if [ -z "$$RUN_DIR" ]; then echo "No run found to resume."; exit 1; fi; \
	echo "Resuming from $$RUN_DIR"; \
	./scripts/04_train_skills.sh --resume "$$RUN_DIR"

evaluate:
	@RUN_DIR=$$(ls -td gepa_results/logs/run_* 2>/dev/null | head -1); \
	if [ -z "$$RUN_DIR" ]; then echo "No training run found. Run 'make train' first."; exit 1; fi; \
	./scripts/05_evaluate.sh "$$RUN_DIR"

evaluate-claude:
	@RUN_DIR=$$(ls -td gepa_results/logs/run_* 2>/dev/null | head -1); \
	if [ -z "$$RUN_DIR" ]; then echo "No training run found. Run 'make train' first."; exit 1; fi; \
	./scripts/05_evaluate.sh "$$RUN_DIR" claude haiku 4

deploy-skills:
	@RUN_DIR=$$(ls -td gepa_results/logs/run_* 2>/dev/null | head -1); \
	if [ -z "$$RUN_DIR" ]; then echo "No training run found."; exit 1; fi; \
	if [ ! -f "$$RUN_DIR/prompts/best_skills.txt" ]; then echo "No best_skills.txt found in $$RUN_DIR"; exit 1; fi; \
	mkdir -p ../.claude/skills/$(REPO_NAME); \
	cp "$$RUN_DIR/prompts/best_skills.txt" ../.claude/skills/$(REPO_NAME)/SKILL.md; \
	echo "Deployed skills to .claude/skills/$(REPO_NAME)/SKILL.md"

gen-trajectories:
	./scripts/06_gen_trajectories.sh

prepare-sft:
	./scripts/07_prepare_sft.sh

# ── Legacy torchtune training (configs in configs/torchtune/) ─
train-sft:
	./scripts/08_train_sft.sh

train-sft-full:
	./scripts/08_train_sft.sh --full

train-sft-32b:
	./scripts/08_train_sft.sh --32b --gpus 2

train-dpo:
	./scripts/08_train_sft.sh --dpo

# ── Unsloth training (recommended) ──────────────────────────
# Default: Qwen3-8B LoRA from repo.conf SFT_BASE_MODEL
train-sft-unsloth:
	uv run python scripts/08_train_sft_unsloth.py --recipe configs/unsloth/qwen3_8b_lora.yaml

train-sft-unsloth-4bit:
	uv run python scripts/08_train_sft_unsloth.py --recipe configs/unsloth/qwen3_8b_lora.yaml --four-bit

train-sft-unsloth-full:
	uv run python scripts/08_train_sft_unsloth.py --full

train-sft-unsloth-32b:
	uv run python scripts/08_train_sft_unsloth.py --recipe configs/unsloth/qwen3_32b_lora.yaml

train-sft-unsloth-gptoss:
	uv run python scripts/08_train_sft_unsloth.py --recipe configs/unsloth/gpt_oss_20b_lora.yaml

train-sft-unsloth-gptoss-120b:
	uv run python scripts/08_train_sft_unsloth.py --recipe configs/unsloth/gpt_oss_120b_lora.yaml --device-map balanced

# New model recipes
train-sft-nemotron:
	uv run python scripts/08_train_sft_unsloth.py --recipe configs/unsloth/nemotron3_nano_lora.yaml

train-sft-qwen3-coder:
	uv run python scripts/08_train_sft_unsloth.py --recipe configs/unsloth/qwen3_coder_next_lora.yaml --device-map balanced

train-sft-nemotron-super:
	uv run python scripts/08_train_sft_unsloth.py --recipe configs/unsloth/nemotron3_super_lora.yaml --device-map balanced

# DPO and CPT
train-dpo-unsloth:
	uv run python scripts/08_train_sft_unsloth.py --dpo --sft-checkpoint ./sft_output/

train-cpt-unsloth:
	uv run python scripts/08_train_sft_unsloth.py --cpt --data-dir ./corpus/

# Custom recipe: make train-recipe RECIPE=configs/unsloth/my_recipe.yaml
train-recipe:
	@test -n "$(RECIPE)" || (echo "Usage: make train-recipe RECIPE=configs/unsloth/xxx.yaml"; exit 1)
	uv run python scripts/08_train_sft_unsloth.py --recipe $(RECIPE)

# ── Multi-GPU training (DDP via torchrun) ─────────────────────
# Scales linearly with GPU count. Each GPU gets distinct samples.
NGPUS ?= 2

train-sft-multigpu:
	torchrun --nproc_per_node=$(NGPUS) scripts/08_train_sft_unsloth.py --recipe configs/unsloth/qwen3_8b_lora.yaml

train-sft-multigpu-4bit:
	torchrun --nproc_per_node=$(NGPUS) scripts/08_train_sft_unsloth.py --recipe configs/unsloth/qwen3_8b_lora.yaml --four-bit

train-grpo-multigpu:
	torchrun --nproc_per_node=$(NGPUS) scripts/09_train_grpo.py --recipe configs/unsloth/grpo_qwen3_8b.yaml

# Model splitting (pipeline parallelism — for models that don't fit on one GPU)
train-sft-split:
	uv run python scripts/08_train_sft_unsloth.py --recipe configs/unsloth/qwen3_coder_next_lora.yaml --device-map balanced

train-grpo-split:
	uv run python scripts/09_train_grpo.py --recipe configs/unsloth/grpo_qwen3_coder_next.yaml --device-map balanced

# ── GRPO RL training (step 9) ───────────────────────────────
train-grpo:
	uv run python scripts/09_train_grpo.py --recipe configs/unsloth/grpo_qwen3_8b.yaml

train-grpo-test:
	uv run python scripts/09_train_grpo.py --recipe configs/unsloth/grpo_qwen3_8b.yaml --reward-mode test

train-grpo-fast:
	uv run python scripts/09_train_grpo.py --recipe configs/unsloth/grpo_qwen3_4b.yaml --reward-mode format --steps 50

train-grpo-nemotron:
	uv run python scripts/09_train_grpo.py --recipe configs/unsloth/grpo_nemotron3_nano.yaml

train-grpo-qwen3-coder:
	uv run python scripts/09_train_grpo.py --recipe configs/unsloth/grpo_qwen3_coder_next.yaml --device-map balanced

train-grpo-nemotron-super:
	uv run python scripts/09_train_grpo.py --recipe configs/unsloth/grpo_nemotron3_super.yaml --device-map balanced

train-grpo-gptoss-120b:
	uv run python scripts/09_train_grpo.py --recipe configs/unsloth/grpo_gpt_oss_120b.yaml --device-map balanced

# ── Hyperparameter optimization ───────────────────────────────
hpo:
	uv run python scripts/08b_hpo.py --recipe configs/unsloth/qwen3_8b_lora.yaml --four-bit

hpo-thorough:
	uv run python scripts/08b_hpo.py --recipe configs/unsloth/qwen3_8b_lora.yaml --four-bit --n-trials 30 --steps-per-trial 100

# ── FP8 training (RTX 40/50, H100+) ──────────────────────────
train-sft-fp8:
	uv run python scripts/08_train_sft_unsloth.py --recipe configs/unsloth/qwen3_8b_lora.yaml --fp8

train-grpo-fp8:
	uv run python scripts/09_train_grpo.py --recipe configs/unsloth/grpo_qwen3_8b.yaml --fp8 --vllm-standby

# ── Export ────────────────────────────────────────────────────
# Save to GGUF for llama.cpp / Ollama deployment
export-gguf:
	@test -d "./sft_output/final" || (echo "No model found. Run training first."; exit 1)
	uv run python scripts/08_train_sft_unsloth.py --recipe configs/unsloth/qwen3_8b_lora.yaml --save-gguf q4_k_m

clean:
	rm -rf logs/ gepa_results/ sft_output/ grpo_output/ cpt_output/
	@echo "Cleaned generated logs and results."
