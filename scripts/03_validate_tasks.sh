#!/usr/bin/env bash
# Step 3: Validate and filter task instances via test harness.
#
# Runs each candidate bug patch against the test suite inside Docker.
# Keeps only tasks that:
#   - Break at least 1 existing test (FAIL_TO_PASS >= 1)
#   - Leave at least 1 test still passing (PASS_TO_PASS >= 1)
#
# Prerequisites:
#   - Step 2 completed (task instances generated)
#   - Docker running
#
# Usage:
#   ./scripts/03_validate_tasks.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load repo config
source "$SCRIPT_DIR/../repo.conf"
BUG_DIR="logs/bug_gen/$REPO_KEY"

# All commands go through swesmith_run.py which registers the private repo profile
RUN="uv run python $SCRIPT_DIR/swesmith_run.py"

echo "=== Step 3: Validating task instances ==="
echo ""

if [ ! -d "$BUG_DIR" ]; then
    echo "ERROR: No generated tasks found at $BUG_DIR"
    echo "Run ./scripts/02_generate_tasks.sh first."
    exit 1
fi

# Collect all candidate instances into a single JSON
echo "--- [3a] Collecting candidates ---"
$RUN swesmith.harness.gather "$BUG_DIR"

# Run validation harness (tests baseline then tests with each patch applied)
echo ""
echo "--- [3b] Running validation harness ---"
$RUN swesmith.harness.valid "$REPO_KEY"

# Collect valid instances (those with FAIL_TO_PASS and PASS_TO_PASS)
echo ""
echo "--- [3c] Collecting valid instances ---"
$RUN swesmith.harness.eval "$REPO_KEY"

# Count results
VALID_COUNT=$(uv run python -c "
import json, pathlib
p = pathlib.Path('logs/valid_instances/$REPO_KEY.json')
if p.exists():
    data = json.loads(p.read_text())
    print(len(data))
else:
    print(0)
" 2>/dev/null || echo "?")

echo ""
echo "=== Validation complete ==="
echo "Valid task instances: $VALID_COUNT"
echo "Output: logs/valid_instances/$REPO_KEY.json"
echo ""
echo "Next: run ./scripts/04_train_skills.sh to start GEPA optimization."
