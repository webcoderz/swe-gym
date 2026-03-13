# SWE-Smith + GEPA gskill gym targets
# Run from swe-gym/ directory or via root Makefile (make swe-gym-*)

.PHONY: help setup build-env generate-tasks validate-tasks train evaluate \
	smoke-test deploy-skills gen-trajectories prepare-sft train-sft clean

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
	@echo "  make gen-trajectories Step 6: Generate expert solve traces"
	@echo "  make prepare-sft     Step 7: Evaluate & convert to JSONL"
	@echo "  make train-sft       Step 8: Local torchtune fine-tune"
	@echo ""
	@echo "Shortcuts:"
	@echo "  make smoke-test       Quick 3-task validation run"
	@echo "  make deploy-skills    Copy best_skills.txt to .claude/skills/"
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

train-sft:
	./scripts/08_train_sft.sh

train-sft-full:
	./scripts/08_train_sft.sh --full

train-sft-32b:
	./scripts/08_train_sft.sh --32b --gpus 2

train-dpo:
	./scripts/08_train_sft.sh --dpo

clean:
	rm -rf logs/ gepa_results/ sft_output/
	@echo "Cleaned generated logs and results."
