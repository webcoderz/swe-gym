#!/usr/bin/env bash
# Step 4: Run GEPA gskill optimization loop.
#
# This is the core RL training step. It:
#   1. Loads validated SWE-smith tasks for the repo
#   2. Generates an initial skill via static analysis of the repo
#   3. Iteratively refines the skill through evolutionary search
#   4. Each candidate skill is evaluated by running mini-SWE-agent in Docker
#   5. A reflection model analyzes pass/fail traces and proposes improvements
#
# Output: gepa_results/logs/run_*/prompts/best_skills.txt
#
# Prerequisites:
#   - Step 3 completed (validated task instances)
#   - OPENAI_API_KEY set
#   - Docker running
#
# Usage:
#   ./scripts/04_train_skills.sh [--smoke-test] [--resume RUN_DIR]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$SCRIPT_DIR/../configs"

# Parse args
EXTRA_ARGS=""
if [[ "${1:-}" == "--smoke-test" ]]; then
    echo "=== Running smoke test (3 tasks, minimal budget) ==="
    EXTRA_ARGS="--smoke-test"
    shift
elif [[ "${1:-}" == "--resume" ]]; then
    if [[ -z "${2:-}" ]]; then
        echo "ERROR: --resume requires a run directory path"
        echo "Usage: ./scripts/04_train_skills.sh --resume gepa_results/logs/run_XXXXXXXX"
        exit 1
    fi
    echo "=== Resuming from $2 ==="
    uv run python "$SCRIPT_DIR/local_data_patch.py" --resume "$2"
    exit $?
fi

echo "=== Step 4: GEPA gskill optimization ==="
echo ""

# Load repo config (provides REPO_KEY, TRAIN_* defaults, etc.)
source "$SCRIPT_DIR/../repo.conf"

REPO="$REPO_KEY"
TRAIN_SIZE="${TRAIN_SIZE:-200}"
VAL_SIZE="${VAL_SIZE:-50}"
TEST_SIZE="${TEST_SIZE:-100}"
MODEL="${TRAIN_MODEL:-gpt-5-mini}"
REFLECTION_MODEL="${REFLECTION_MODEL:-gpt-5.2-pro}"
WORKERS="${TRAIN_WORKERS:-6}"
MAX_METRIC_CALLS="${MAX_METRIC_CALLS:-600}"
PROPOSER="loop"
SEED=42

echo "Config:"
echo "  repo:             $REPO"
echo "  model:            $MODEL"
echo "  reflection_model: $REFLECTION_MODEL"
echo "  workers:          $WORKERS"
echo "  max_metric_calls: $MAX_METRIC_CALLS"
echo "  proposer:         $PROPOSER"
echo "  train/val/test:   $TRAIN_SIZE/$VAL_SIZE/$TEST_SIZE"
echo ""

# Use --wandb only if WANDB_API_KEY is set
WANDB_FLAG=""
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    WANDB_FLAG="--wandb"
fi

# Use local_data_patch.py if local tasks exist, otherwise fall through to HF
if [[ -f "logs/valid_instances/$REPO.json" ]]; then
    TRAIN_CMD="uv run python $SCRIPT_DIR/local_data_patch.py"
else
    TRAIN_CMD="uv run python -m gepa.gskill.gskill.train_optimize_anything"
fi

$TRAIN_CMD \
    --repo "$REPO" \
    --train-size "$TRAIN_SIZE" \
    --val-size "$VAL_SIZE" \
    --test-size "$TEST_SIZE" \
    --model "$MODEL" \
    --reflection-model "$REFLECTION_MODEL" \
    --workers "$WORKERS" \
    --max-metric-calls "$MAX_METRIC_CALLS" \
    --proposer "$PROPOSER" \
    --seed "$SEED" \
    $WANDB_FLAG \
    $EXTRA_ARGS

echo ""
echo "=== Training complete ==="
echo ""

# Find the latest run directory
LATEST_RUN=$(ls -td gepa_results/logs/run_* 2>/dev/null | head -1)
if [[ -n "$LATEST_RUN" ]]; then
    echo "Results: $LATEST_RUN"
    echo ""
    if [[ -f "$LATEST_RUN/prompts/best_skills.txt" ]]; then
        echo "Best skills file: $LATEST_RUN/prompts/best_skills.txt"
        echo ""
        echo "To deploy as a Claude Code skill:"
        echo "  mkdir -p .claude/skills/$REPO_NAME"
        echo "  cp $LATEST_RUN/prompts/best_skills.txt .claude/skills/$REPO_NAME/SKILL.md"
    fi
    if [[ -f "$LATEST_RUN/cost_summary.txt" ]]; then
        echo ""
        echo "Cost summary:"
        cat "$LATEST_RUN/cost_summary.txt"
    fi
fi

echo ""
echo "Next: run ./scripts/05_evaluate.sh to test skills on held-out instances."
