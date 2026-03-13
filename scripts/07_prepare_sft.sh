#!/usr/bin/env bash
# Step 7: Evaluate trajectories and convert to SFT training format.
#
# 1. Evaluates which trajectories actually resolved the task (via Docker test harness)
# 2. Converts successful trajectories to chat-format JSONL for fine-tuning
# 3. Optionally combines/shuffles multiple trajectory sets
#
# Output: logs/sft_data/<REPO_KEY>/ft_xml_<run_id>.jsonl
#
# Prerequisites:
#   - Step 6 completed (trajectories generated)
#   - Docker running
#
# Usage:
#   ./scripts/07_prepare_sft.sh [--style xml|ticks|tool] [--workers N]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load repo config
source "$SCRIPT_DIR/../repo.conf"

# All commands go through swesmith_run.py which registers the private repo profile
RUN="uv run python $SCRIPT_DIR/swesmith_run.py"

STYLE="${1:-xml}"
WORKERS="${SFT_WORKERS:-4}"
TASKS_FILE="logs/valid_instances/${REPO_KEY}.json"
TRAJ_DIR="logs/trajectories/${REPO_KEY}"
SFT_DIR="logs/sft_data/${REPO_KEY}"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --style) STYLE="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        *) shift ;;
    esac
done

echo "=== Step 7: Preparing SFT training data ==="
echo "  Trajectories: $TRAJ_DIR"
echo "  Style:        $STYLE"
echo "  Workers:      $WORKERS"
echo ""

if [ ! -d "$TRAJ_DIR" ]; then
    echo "ERROR: No trajectories found at $TRAJ_DIR"
    echo "Run ./scripts/06_gen_trajectories.sh first."
    exit 1
fi

# Find the predictions file
PREDS_FILE=$(find "$TRAJ_DIR" -name "preds.json" -o -name "all_preds.jsonl" | head -1)
if [ -z "$PREDS_FILE" ]; then
    echo "ERROR: No predictions file found in $TRAJ_DIR"
    echo "Run: uv run sweagent merge-preds $TRAJ_DIR/"
    exit 1
fi
echo "Predictions: $PREDS_FILE"

# --- Step 7a: Evaluate trajectories against test harness ---
echo ""
echo "--- [7a] Evaluating trajectories ---"

# Detect run_id from trajectory directory
RUN_ID=$(basename "$TRAJ_DIR")

$RUN swesmith.harness.eval \
    --dataset_path "$TASKS_FILE" \
    --predictions_path "$PREDS_FILE" \
    --run_id "$RUN_ID" \
    --workers "$WORKERS" \
    --timeout 240

EVAL_DIR="logs/run_evaluation/${RUN_ID}"
echo "Evaluation results: $EVAL_DIR"

# Count resolved
RESOLVED=$(uv run python -c "
import json, pathlib
eval_dir = pathlib.Path('$EVAL_DIR')
resolved = 0
total = 0
for f in eval_dir.glob('*/report.json'):
    total += 1
    report = json.loads(f.read_text())
    if report.get('resolved', False):
        resolved += 1
print(f'{resolved}/{total}')
" 2>/dev/null || echo "?/?")
echo "Resolved: $RESOLVED"

# --- Step 7b: Convert to SFT format ---
echo ""
echo "--- [7b] Converting to SFT JSONL ---"

mkdir -p "$SFT_DIR"

uv run python -m swesmith.train.traj_mgr.collect_trajs \
    --traj_dir "$TRAJ_DIR" \
    --eval_dir "$EVAL_DIR" \
    --style "$STYLE" \
    --out_dir "$SFT_DIR"

# Count output
SFT_COUNT=$(uv run python -c "
import pathlib
files = list(pathlib.Path('$SFT_DIR').glob('ft_*.jsonl'))
total = 0
for f in files:
    total += sum(1 for _ in open(f))
print(f'{total} examples in {len(files)} file(s)')
" 2>/dev/null || echo "?")

echo ""
echo "=== SFT data preparation complete ==="
echo "Output:   $SFT_DIR/"
echo "Examples: $SFT_COUNT"
echo ""
echo "Next: run ./scripts/08_train_sft.sh to fine-tune with torchtune."
