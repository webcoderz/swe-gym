#!/usr/bin/env bash
# Step 6: Generate expert trajectories using SWE-agent.
#
# Runs SWE-agent on validated task instances to produce solve traces.
# These trajectories are the raw material for supervised fine-tuning.
#
# Prerequisites:
#   - Step 3 completed (validated task instances)
#   - SWE-agent installed (pip install sweagent)
#   - OPENAI_API_KEY set (or the model provider key for --model)
#   - Docker running
#
# Usage:
#   ./scripts/06_gen_trajectories.sh [--model MODEL] [--workers N]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load repo config
source "$SCRIPT_DIR/../repo.conf"

MODEL="${SFT_AGENT_MODEL:-gpt-5-mini}"
WORKERS="${SFT_WORKERS:-4}"
SUBSET_NAME="${REPO_KEY}"
TASKS_FILE="logs/valid_instances/${REPO_KEY}.json"
TRAJ_DIR="logs/trajectories/${REPO_KEY}"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "=== Step 6: Generating expert trajectories ==="
echo "  Tasks:   $TASKS_FILE"
echo "  Model:   $MODEL"
echo "  Workers: $WORKERS"
echo ""

if [ ! -f "$TASKS_FILE" ]; then
    echo "ERROR: No validated tasks at $TASKS_FILE"
    echo "Run steps 1-3 first."
    exit 1
fi

# Count tasks
TASK_COUNT=$(uv run python -c "
import json
data = json.load(open('$TASKS_FILE'))
if isinstance(data, dict): data = list(data.values())
print(len(data))
" 2>/dev/null || echo "?")
echo "Found $TASK_COUNT validated task instances"
echo ""

# Ensure SWE-agent is available
if ! uv run python -c "import sweagent" 2>/dev/null; then
    echo "Installing SWE-agent..."
    uv pip install sweagent
    echo ""
fi

mkdir -p "$TRAJ_DIR"

# Run SWE-agent on each task instance
# Uses sweagent run-batch for parallel execution
echo "--- Running SWE-agent on tasks ---"
uv run sweagent run-batch \
    --instances_path "$TASKS_FILE" \
    --model "$MODEL" \
    --output_dir "$TRAJ_DIR" \
    --num_workers "$WORKERS"

# Merge predictions into a single file
echo ""
echo "--- Merging predictions ---"
uv run sweagent merge-preds "$TRAJ_DIR/"

echo ""
echo "=== Trajectory generation complete ==="
echo "Trajectories: $TRAJ_DIR/"
echo ""
echo "Next: run ./scripts/07_prepare_sft.sh to convert to SFT format."
