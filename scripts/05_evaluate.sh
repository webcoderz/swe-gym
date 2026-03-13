#!/usr/bin/env bash
# Step 5: Evaluate learned skills on held-out test instances.
#
# Runs both baseline (no skills) and skills-enhanced conditions on the test
# split, comparing resolve rates. Supports both mini-SWE-agent and Claude Code.
#
# Prerequisites:
#   - Step 4 completed (training run with best_skills.txt)
#   - OPENAI_API_KEY and/or ANTHROPIC_API_KEY set
#   - Docker running
#
# Usage:
#   ./scripts/05_evaluate.sh <run_dir> [--agent mini|claude] [--model MODEL] [--workers N]

set -euo pipefail

# Parse args
RUN_DIR="${1:-}"
AGENT="${2:-mini}"
MODEL="${3:-gpt-5-mini}"
WORKERS="${4:-8}"

if [[ -z "$RUN_DIR" ]]; then
    # Auto-detect latest run
    RUN_DIR=$(ls -td gepa_results/logs/run_* 2>/dev/null | head -1)
    if [[ -z "$RUN_DIR" ]]; then
        echo "ERROR: No run directory found. Provide path or run training first."
        echo "Usage: ./scripts/05_evaluate.sh <run_dir>"
        exit 1
    fi
    echo "Auto-detected run: $RUN_DIR"
fi

CONFIG="$RUN_DIR/config.json"
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: Config not found at $CONFIG"
    exit 1
fi

echo "=== Step 5: Evaluation ==="
echo "Run dir: $RUN_DIR"
echo "Agent:   $AGENT"
echo "Model:   $MODEL"
echo "Workers: $WORKERS"
echo ""

case "$AGENT" in
    mini)
        # mini-SWE-agent evaluation (runs both baseline and with-skills automatically)
        echo "--- Evaluating with mini-SWE-agent ---"
        uv run python -m gepa.gskill.gskill.evaluate.mini_swe_agent \
            --config "$CONFIG" \
            --workers "$WORKERS"
        ;;
    claude)
        # Claude Code evaluation
        echo "--- [Baseline] Claude Code without skills ---"
        uv run python -m gepa.gskill.gskill.evaluate.claude_code \
            --config "$CONFIG" \
            --model "${MODEL:-haiku}" \
            --workers "$WORKERS"

        echo ""
        echo "--- [With Skills] Claude Code with learned skills ---"
        uv run python -m gepa.gskill.gskill.evaluate.claude_code \
            --config "$CONFIG" \
            --model "${MODEL:-haiku}" \
            --workers "$WORKERS" \
            --use-skills
        ;;
    *)
        echo "ERROR: Unknown agent type '$AGENT'. Use 'mini' or 'claude'."
        exit 1
        ;;
esac

echo ""
echo "=== Evaluation complete ==="
echo "Results saved in: $RUN_DIR"
